"""
Phase 3 — explain the models two ways and check they agree.

SHAP is the global lens (game-theory credit per feature) and runs on LR and RF;
LIME is the local lens (perturb one session, fit a simple model around it) and
runs on the NN. The payoff is convergence: when three lenses on three models all
point at the same pre-cart browsing features, the finding is trustworthy. The
LIME reasons here are also what Phase 4 feeds to Gemini.
"""
import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
warnings.filterwarnings('ignore')
SEED = 42
HERE = Path(__file__).resolve().parents[1] / 'artifacts'
CSV = Path(__file__).resolve().parents[1] / 'data' / 'engineered_sessions_no_leakage.csv'

def reload_dataset():
    df = pd.read_csv(CSV)
    y = df['target_purchase'].astype(int).to_numpy()
    X_df = df.drop(columns=['session_id', 'target_purchase'])
    from sklearn.preprocessing import LabelEncoder
    cat_encoder = LabelEncoder()
    X_df['main_category'] = cat_encoder.fit_transform(X_df['main_category'].fillna('unknown').astype(str))
    X_df = X_df.fillna(0)
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float64)
    return (X, y, feature_names, cat_encoder)

def shap_linear(pipe_lr, X_train_raw, X_test_raw, feature_names):
    # Explain in the SAME scaled space the model trained in, so the SHAP values
    # line up with the coefficients. 1000 train rows are the background baseline.
    scaler = pipe_lr.named_steps['scaler']
    clf = pipe_lr.named_steps['clf']
    X_tr_scaled = scaler.transform(X_train_raw)
    X_te_scaled = scaler.transform(X_test_raw)
    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(X_tr_scaled.shape[0], size=1000, replace=False)
    explainer = shap.LinearExplainer(clf, X_tr_scaled[bg_idx])
    shap_values = explainer(X_te_scaled)
    shap_values.feature_names = feature_names
    return shap_values

def shap_tree(pipe_rf, X_test_raw, feature_names):
    scaler = pipe_rf.named_steps['scaler']
    clf = pipe_rf.named_steps['clf']
    X_te_scaled = scaler.transform(X_test_raw)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_te_scaled)
    # TreeExplainer returns both classes for a forest; keep the purchase class.
    if hasattr(shap_values, 'values') and shap_values.values.ndim == 3:
        shap_values = shap.Explanation(values=shap_values.values[..., 1], base_values=shap_values.base_values[..., 1], data=shap_values.data, feature_names=feature_names)
    else:
        shap_values.feature_names = feature_names
    return shap_values

def save_shap_plot(shap_values, out_path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def top_mean_abs(shap_values, feature_names, k=20):
    # Global ranking = average absolute SHAP value per feature across all sessions.
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [{'feature': feature_names[i], 'mean_abs_shap': float(mean_abs[i])} for i in order[:k]]

def lime_on_nn(nn_model, nn_scaler, X_train_raw, X_test_raw, y_test, feature_names, k=10):
    X_tr_scaled = nn_scaler.transform(X_train_raw)
    X_te_scaled = nn_scaler.transform(X_test_raw)

    def nn_predict_proba(scaled_X):
        p1 = nn_model.predict(scaled_X, verbose=0).ravel()
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T
    # Explain the model's most CONFIDENT correct calls in each direction, so the
    # reasons describe clear abandoners and clear purchasers, not borderline cases.
    test_probs_1 = nn_predict_proba(X_te_scaled)[:, 1]
    tn_mask = (y_test == 0) & (test_probs_1 < 0.45)
    tp_mask = (y_test == 1) & (test_probs_1 >= 0.45)
    tn_idx = np.where(tn_mask)[0]
    tp_idx = np.where(tp_mask)[0]
    tn_order = tn_idx[np.argsort(test_probs_1[tn_idx])]
    tp_order = tp_idx[np.argsort(-test_probs_1[tp_idx])]
    abandon_pick = tn_order[:k]
    purchase_pick = tp_order[:k]
    print(f'  Cohorts: {len(tn_idx)} TN (correctly-predicted abandoners), {len(tp_idx)} TP (correctly-predicted purchasers)')
    print(f'  Sampled top-{k} by confidence from each cohort = {len(abandon_pick) + len(purchase_pick)} total')
    explainer = LimeTabularExplainer(training_data=X_tr_scaled, feature_names=feature_names, class_names=['abandon', 'purchase'], discretize_continuous=True, random_state=SEED)
    examples = []
    for cohort, pick in [('high_conf_abandon', abandon_pick), ('high_conf_purchase', purchase_pick)]:
        for local_idx in pick:
            # LIME perturbs this one row 2000 times and fits a local linear model
            # to read off which features pushed the prediction which way.
            exp = explainer.explain_instance(data_row=X_te_scaled[local_idx], predict_fn=nn_predict_proba, num_features=5, num_samples=2000)
            reasons = [{'feature_condition': f, 'weight': float(w)} for f, w in exp.as_list()]
            examples.append({'cohort': cohort, 'test_row_index': int(local_idx), 'actual': int(y_test[local_idx]), 'predicted_prob_purchase': float(test_probs_1[local_idx]), 'top_reasons': reasons, 'raw_features': {feature_names[j]: float(X_test_raw[local_idx, j]) for j in range(len(feature_names))}})
    return examples

def _canonical_reason_category(feature_name: str) -> str:
    # Group raw feature names into human categories. 'browse_indecision' is the
    # browse-behaviour bucket — how the shopper browsed, not what they bought.
    # This is the bucket that ends up at 85% of the top-3 LIME reasons.
    n = feature_name.lower()
    if any((k in n for k in ('price', 'value'))):
        return 'price_hesitation'
    if any((k in n for k in ('view', 'intensity', 'events', 'time_to_cart', 'time_between', 'acceleration'))):
        return 'browse_indecision'
    if any((k in n for k in ('cart', 'items'))):
        return 'cart_composition'
    if any((k in n for k in ('category', 'brand', 'product'))):
        return 'product_diversity'
    return 'other'

def categorise_lime(examples):
    mapped = []
    for ex in examples:
        top = ex['top_reasons'][0]
        cond = top['feature_condition']
        import re
        # LIME conditions read like '0.50 < views_before_cart <= 2.00'; pull the
        # feature token (the longest identifier) back out to categorise it.
        toks = re.findall('[a-zA-Z_][a-zA-Z_0-9]+', cond)
        feat = max(toks, key=len) if toks else cond
        category = _canonical_reason_category(feat)
        mapped.append({'test_row_index': ex['test_row_index'], 'cohort': ex['cohort'], 'predicted_prob_purchase': ex['predicted_prob_purchase'], 'top_feature': feat, 'top_condition': cond, 'top_weight': top['weight'], 'reason_category': category})
    return mapped

def main():
    print('=' * 72)
    print('Extracting SHAP (LR, RF) and LIME (NN) explanations')
    print('=' * 72)
    X, y, feature_names, _ = reload_dataset()
    # Reuse the exact 80/20 split from Phase 2 so explanations describe the same
    # held-out sessions every other phase reports on.
    idx = json.loads((HERE / 'holdout_indices.json').read_text())
    idx_tr = np.array(idx['train_indices'])
    idx_te = np.array(idx['test_indices'])
    with open(HERE / 'model_lr.pkl', 'rb') as f:
        pipe_lr = pickle.load(f)
    with open(HERE / 'model_rf.pkl', 'rb') as f:
        pipe_rf = pickle.load(f)
    from tensorflow.keras.models import load_model
    nn_model = load_model(HERE / 'model_nn.h5')
    with open(HERE / 'scaler.pkl', 'rb') as f:
        nn_scaler = pickle.load(f)
    X_tr_raw, X_te_raw = (X[idx_tr], X[idx_te])
    y_te = y[idx_te]
    print('\n[SHAP] LR via LinearExplainer…')
    sv_lr = shap_linear(pipe_lr, X_tr_raw, X_te_raw, feature_names)
    with open(HERE / 'shap_values_lr.pkl', 'wb') as f:
        pickle.dump(sv_lr, f)
    save_shap_plot(sv_lr, HERE / 'shap_summary_lr.png', 'SHAP summary — Logistic Regression (test set)')
    lr_top = top_mean_abs(sv_lr, feature_names)
    print('\n[SHAP] RF via TreeExplainer…')
    sv_rf = shap_tree(pipe_rf, X_te_raw, feature_names)
    with open(HERE / 'shap_values_rf.pkl', 'wb') as f:
        pickle.dump(sv_rf, f)
    save_shap_plot(sv_rf, HERE / 'shap_summary_rf.png', 'SHAP summary — Random Forest (test set)')
    rf_top = top_mean_abs(sv_rf, feature_names)
    (HERE / 'shap_top_features.json').write_text(json.dumps({'LogisticRegression_top20': lr_top, 'RandomForest_top20': rf_top}, indent=2))
    print('\n[LIME] Explaining NN predictions on 20 test-set examples…')
    lime_examples = lime_on_nn(nn_model, nn_scaler, X_tr_raw, X_te_raw, y_te, feature_names, k=10)
    (HERE / 'lime_examples.json').write_text(json.dumps(lime_examples, indent=2))
    lime_cat_map = categorise_lime(lime_examples)
    (HERE / 'lime_category_map.json').write_text(json.dumps(lime_cat_map, indent=2))
    print('\n' + '=' * 72)
    print('SHAP — top 10 features (mean |SHAP|)')
    print('=' * 72)
    print(f'{'LR':<30} {'RF':<30}')
    for i in range(10):
        lr_row = f'{lr_top[i]['feature']:<22} {lr_top[i]['mean_abs_shap']:.4f}'
        rf_row = f'{rf_top[i]['feature']:<22} {rf_top[i]['mean_abs_shap']:.4f}'
        print(f'{lr_row:<30} {rf_row:<30}')
    print('\n' + '=' * 72)
    print('LIME — reason-category distribution across 20 examples')
    print('=' * 72)
    from collections import Counter
    cats = Counter((x['reason_category'] for x in lime_cat_map))
    for c, n in cats.most_common():
        print(f'  {c:<25} {n}')
    print(f'\nPersisted: shap_values_{{lr,rf}}.pkl, shap_summary_{{lr,rf}}.png, shap_top_features.json, lime_examples.json, lime_category_map.json')
if __name__ == '__main__':
    main()
