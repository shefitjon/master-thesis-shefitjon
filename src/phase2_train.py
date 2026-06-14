"""
Phase 2 — train and honestly evaluate LR, RF, and a small neural net.

Two leaks were closed here. The temporal one lives in Phase 1; the second is
subtler and lives in the cross-validation below: SMOTE must be re-fit INSIDE
each fold, never once over the whole training set, or the validation fold sees
synthetic points made from itself. Doing it right shrank the CV-vs-test gap
from ~0.28 to under 0.01.
"""
import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
warnings.filterwarnings('ignore')
SEED = 42
HERE = Path(__file__).resolve().parents[1] / 'artifacts'
CSV = Path(__file__).resolve().parents[1] / 'data' / 'engineered_sessions_no_leakage.csv'

def load_and_prepare():
    df = pd.read_csv(CSV)
    y = df['target_purchase'].astype(int).to_numpy()
    X_df = df.drop(columns=['session_id', 'target_purchase'])
    cat_encoder = LabelEncoder()
    X_df['main_category'] = cat_encoder.fit_transform(X_df['main_category'].fillna('unknown').astype(str))
    X_df = X_df.fillna(0)
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float64)
    return (X, y, feature_names, cat_encoder)

def score_predictions(y_true, y_pred, y_prob):
    return {'accuracy': float(accuracy_score(y_true, y_pred)), 'precision': float(precision_score(y_true, y_pred, zero_division=0)), 'recall': float(recall_score(y_true, y_pred, zero_division=0)), 'f1': float(f1_score(y_true, y_pred, zero_division=0)), 'roc_auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None}

def cv_sklearn(model_ctor, X, y, label):
    print(f'\n[{label}] 5-fold CV with SMOTE inside each fold…')
    # 5-fold CV: train on 4 parts, test on the held-out 5th, five times over.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        # SMOTE is a pipeline STEP, so .fit() below re-balances using only this
        # fold's training rows (tr). The validation rows (va) are never seen by
        # SMOTE or the scaler — that is what makes the CV score honest.
        pipe = ImbPipeline(steps=[('scaler', StandardScaler()), ('smote', SMOTE(random_state=SEED, k_neighbors=3)), ('clf', model_ctor())])
        pipe.fit(X[tr], y[tr])
        y_pred = pipe.predict(X[va])
        y_prob = pipe.predict_proba(X[va])[:, 1]
        s = score_predictions(y[va], y_pred, y_prob)
        fold_scores.append(s)
        print(f'  fold {fold + 1}: F1={s['f1']:.4f}  AUC={s['roc_auc']:.4f}  Acc={s['accuracy']:.4f}  P={s['precision']:.4f}  R={s['recall']:.4f}')
    agg = {metric: {'per_fold': [sc[metric] for sc in fold_scores], 'mean': float(np.mean([sc[metric] for sc in fold_scores if sc[metric] is not None])), 'std': float(np.std([sc[metric] for sc in fold_scores if sc[metric] is not None]))} for metric in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')}
    return (agg, fold_scores)

def cv_nn(X, y, label):
    print(f'\n[{label}] 5-fold CV with class_weight (no SMOTE)…')
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    tf.random.set_seed(SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        tf.keras.backend.clear_session()
        # The net handles imbalance with class weights instead of SMOTE, but the
        # same discipline holds: the scaler is fit on the training fold only.
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = (scaler.transform(X[tr]), scaler.transform(X[va]))
        model = Sequential([Dense(16, activation='relu', input_dim=Xtr.shape[1]), BatchNormalization(), Dropout(0.3), Dense(8, activation='relu'), BatchNormalization(), Dropout(0.3), Dense(1, activation='sigmoid')])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        cw = {0: 1.0, 1: (len(y[tr]) - y[tr].sum()) / max(y[tr].sum(), 1)}
        model.fit(Xtr, y[tr], epochs=100, batch_size=64, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)], class_weight=cw, verbose=0)
        y_prob = model.predict(Xva, verbose=0).ravel()
        # Threshold 0.45, not 0.5: we tilt toward recall because missing a real
        # abandoner is the costly error, a needless pop-up is cheap.
        y_pred = (y_prob > 0.45).astype(int)
        s = score_predictions(y[va], y_pred, y_prob)
        fold_scores.append(s)
        print(f'  fold {fold + 1}: F1={s['f1']:.4f}  AUC={s['roc_auc']:.4f}  Acc={s['accuracy']:.4f}  P={s['precision']:.4f}  R={s['recall']:.4f}')
    agg = {metric: {'per_fold': [sc[metric] for sc in fold_scores], 'mean': float(np.mean([sc[metric] for sc in fold_scores if sc[metric] is not None])), 'std': float(np.std([sc[metric] for sc in fold_scores if sc[metric] is not None]))} for metric in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')}
    return (agg, fold_scores)

def fit_final_sklearn(model_ctor, X_tr, y_tr, X_te, y_te, label):
    print(f'\n[{label}] final fit on 80% train + eval on 20% test…')
    # Same pipeline as in CV, now fit on the full 80% train and judged once on
    # the untouched 20% test set — the number that goes in the thesis.
    pipe = ImbPipeline(steps=[('scaler', StandardScaler()), ('smote', SMOTE(random_state=SEED, k_neighbors=3)), ('clf', model_ctor())])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    s = score_predictions(y_te, y_pred, y_prob)
    cm = confusion_matrix(y_te, y_pred).tolist()
    print(f'  test: F1={s['f1']:.4f}  AUC={s['roc_auc']:.4f}')
    return (pipe, s, cm)

def fit_final_nn(X_tr, y_tr, X_te, y_te):
    print('\n[Neural Network] final fit…')
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    tf.keras.backend.clear_session()
    scaler = StandardScaler().fit(X_tr)
    Xtr, Xte = (scaler.transform(X_tr), scaler.transform(X_te))
    model = Sequential([Dense(16, activation='relu', input_dim=Xtr.shape[1]), BatchNormalization(), Dropout(0.3), Dense(8, activation='relu'), BatchNormalization(), Dropout(0.3), Dense(1, activation='sigmoid')])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    cw = {0: 1.0, 1: (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)}
    model.fit(Xtr, y_tr, epochs=100, batch_size=64, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)], class_weight=cw, verbose=0)
    y_prob = model.predict(Xte, verbose=0).ravel()
    y_pred = (y_prob > 0.45).astype(int)
    s = score_predictions(y_te, y_pred, y_prob)
    cm = confusion_matrix(y_te, y_pred).tolist()
    print(f'  test: F1={s['f1']:.4f}  AUC={s['roc_auc']:.4f}')
    return (model, scaler, s, cm)

def main():
    print('=' * 72)
    print('CV-corrected training — REES46 leakage-free dataset')
    print('=' * 72)
    X, y, feature_names, _ = load_and_prepare()
    print(f'\nDataset: {X.shape[0]} rows, {X.shape[1]} features')
    print(f'Class balance: 0={(y == 0).mean():.4f} / 1={(y == 1).mean():.4f}')
    # One stratified 80/20 split, fixed by SEED, reused by every later phase via
    # holdout_indices.json so explanations and stats see the exact same test set.
    idx = np.arange(len(y))
    idx_tr, idx_te, _, _ = train_test_split(idx, y, test_size=0.2, random_state=SEED, stratify=y)
    X_tr, X_te = (X[idx_tr], X[idx_te])
    y_tr, y_te = (y[idx_tr], y[idx_te])
    print(f'\nSplit: {len(y_tr)} train / {len(y_te)} test  (stratified, seed={SEED})')
    lr_ctor = lambda: LogisticRegression(random_state=SEED, max_iter=1000, C=0.1)
    rf_ctor = lambda: RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=50, min_samples_leaf=20, random_state=SEED, n_jobs=-1)
    cv_lr, _ = cv_sklearn(lr_ctor, X_tr, y_tr, 'Logistic Regression')
    cv_rf, _ = cv_sklearn(rf_ctor, X_tr, y_tr, 'Random Forest')
    cv_nn_agg, _ = cv_nn(X_tr, y_tr, 'Neural Network')
    pipe_lr, test_lr, cm_lr = fit_final_sklearn(lr_ctor, X_tr, y_tr, X_te, y_te, 'Logistic Regression')
    pipe_rf, test_rf, cm_rf = fit_final_sklearn(rf_ctor, X_tr, y_tr, X_te, y_te, 'Random Forest')
    nn_model, nn_scaler, test_nn, cm_nn = fit_final_nn(X_tr, y_tr, X_te, y_te)
    print('\nExtracting feature importance…')
    # Pull the trained coefficients straight out of the pipeline. These numbers
    # are later copied by hand into the live-demo widget and the Magento module's
    # RiskScorer — there is no runtime link, so retraining means re-copying them.
    lr_clf = pipe_lr.named_steps['clf']
    lr_coefs = lr_clf.coef_.ravel().tolist()
    rf_clf = pipe_rf.named_steps['clf']
    rf_gini = rf_clf.feature_importances_.tolist()
    print('  permutation importance for NN (can take ~1 min)…')
    from sklearn.metrics import f1_score as _f1

    def nn_predict(scaled_X):
        return (nn_model.predict(scaled_X, verbose=0).ravel() > 0.45).astype(int)

    class _NNWrapper:

        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler

        def predict(self, Xin):
            return nn_predict(self.scaler.transform(Xin))

        def fit(self, *a, **kw):
            return self

        def score(self, Xin, yin):
            return _f1(yin, self.predict(Xin), zero_division=0)
    nn_wrap = _NNWrapper(nn_model, nn_scaler)
    perm = permutation_importance(nn_wrap, X_te, y_te, n_repeats=5, random_state=SEED, scoring=None, n_jobs=1)
    nn_perm = perm.importances_mean.tolist()
    feature_importance = {'feature_names': feature_names, 'lr_coefficients': lr_coefs, 'rf_gini': rf_gini, 'nn_permutation_mean_delta_f1': nn_perm}
    results = {'seed': SEED, 'n_rows': int(len(y)), 'n_features': int(X.shape[1]), 'class_balance': {'0': float((y == 0).mean()), '1': float((y == 1).mean())}, 'split': {'train_n': int(len(y_tr)), 'test_n': int(len(y_te))}, 'cv': {'LogisticRegression': cv_lr, 'RandomForest': cv_rf, 'NeuralNetwork': cv_nn_agg}, 'test': {'LogisticRegression': test_lr, 'RandomForest': test_rf, 'NeuralNetwork': test_nn}, 'cv_vs_test_gap_f1': {'LogisticRegression': test_lr['f1'] - cv_lr['f1']['mean'], 'RandomForest': test_rf['f1'] - cv_rf['f1']['mean'], 'NeuralNetwork': test_nn['f1'] - cv_nn_agg['f1']['mean']}}
    (HERE / 'cv_corrected_results.json').write_text(json.dumps(results, indent=2))
    (HERE / 'confusion_matrices.json').write_text(json.dumps({'labels': ['abandon', 'purchase'], 'note': 'rows = actual, cols = predicted (sklearn order)', 'LogisticRegression': cm_lr, 'RandomForest': cm_rf, 'NeuralNetwork': cm_nn}, indent=2))
    (HERE / 'feature_importance.json').write_text(json.dumps(feature_importance, indent=2))
    (HERE / 'holdout_indices.json').write_text(json.dumps({'train_indices': idx_tr.tolist(), 'test_indices': idx_te.tolist(), 'seed': SEED}))
    (HERE / 'feature_names.json').write_text(json.dumps(feature_names, indent=2))
    with open(HERE / 'model_lr.pkl', 'wb') as f:
        pickle.dump(pipe_lr, f)
    with open(HERE / 'model_rf.pkl', 'wb') as f:
        pickle.dump(pipe_rf, f)
    nn_model.save(HERE / 'model_nn.h5')
    with open(HERE / 'scaler.pkl', 'wb') as f:
        pickle.dump(nn_scaler, f)
    print('\n' + '=' * 72)
    print('SUMMARY')
    print('=' * 72)
    print(f'{'Model':<20} {'CV F1':>10} {'± std':>8} {'Test F1':>10} {'Gap':>8}')
    # The Gap column is the headline integrity check: |CV - Test| under 0.01.
    for name, cv, te in [('LogisticRegression', cv_lr, test_lr), ('RandomForest', cv_rf, test_rf), ('NeuralNetwork', cv_nn_agg, test_nn)]:
        gap = te['f1'] - cv['f1']['mean']
        print(f'{name:<20} {cv['f1']['mean']:>10.4f} {cv['f1']['std']:>8.4f} {te['f1']:>10.4f} {gap:>+8.4f}')
    print(f'\nPersisted: cv_corrected_results.json, confusion_matrices.json, feature_importance.json, holdout_indices.json, feature_names.json, model_{{lr,rf}}.pkl, model_nn.h5, scaler.pkl')
if __name__ == '__main__':
    main()
