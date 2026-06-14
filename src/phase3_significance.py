"""
Phase 2/3 — is the NN really better than LR, or is it luck?

Two tests, on purpose. McNemar looks at one held-out split; the Dietterich
5x2cv test repeats the comparison over ten splits. They disagree here (McNemar
p=0.002, 5x2cv p=0.96), and the rule is simple: a single split can get lucky,
so when the split-robust test says 'no difference', that is the honest verdict
— which lets us keep LR for interpretability at no real cost.
"""
import json
import math
import pickle
import sys
import warnings
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, t as t_dist
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parents[1] / 'artifacts'
# Make the sibling Phase-2 module importable so we reuse the exact same data
# loading and fold structure rather than re-implementing them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_train import load_and_prepare, fit_final_sklearn, fit_final_nn, SEED
from sklearn.linear_model import LogisticRegression
SIGNIFICANCE_ALPHA = 0.05
N_CV_ITERATIONS = 5

def mcnemar_exact_test(lr_correct: np.ndarray, nn_correct: np.ndarray):
    assert lr_correct.shape == nn_correct.shape
    # Only the disagreements matter: b = LR right while NN wrong, c = the reverse.
    # If the models were equal, b and c should be ~50/50 — an exact binomial test.
    b = int(((lr_correct == 1) & (nn_correct == 0)).sum())
    c = int(((lr_correct == 0) & (nn_correct == 1)).sum())
    n = b + c
    if n == 0:
        return {'b': 0, 'c': 0, 'n_discordant': 0, 'p_value': 1.0, 'test': 'mcnemar_exact', 'note': 'no discordant pairs'}
    k = min(b, c)
    p = binomtest(k, n=n, p=0.5, alternative='two-sided').pvalue
    return {'b': b, 'c': c, 'n_discordant': n, 'p_value': float(p), 'test': 'mcnemar_exact', 'note': ''}

def mcnemar_on_heldout():
    print('[McNemar] loading held-out test predictions…')
    idx = json.loads((HERE / 'holdout_indices.json').read_text())
    idx_te = np.array(idx['test_indices'])
    X, y, feature_names, _ = load_and_prepare()
    X_te, y_te = (X[idx_te], y[idx_te])
    with open(HERE / 'model_lr.pkl', 'rb') as f:
        pipe_lr = pickle.load(f)
    y_pred_lr = pipe_lr.predict(X_te)
    from tensorflow.keras.models import load_model
    nn_model = load_model(HERE / 'model_nn.h5')
    with open(HERE / 'scaler.pkl', 'rb') as f:
        nn_scaler = pickle.load(f)
    y_prob_nn = nn_model.predict(nn_scaler.transform(X_te), verbose=0).ravel()
    y_pred_nn = (y_prob_nn > 0.45).astype(int)
    lr_correct = (y_pred_lr == y_te).astype(int)
    nn_correct = (y_pred_nn == y_te).astype(int)
    result = mcnemar_exact_test(lr_correct, nn_correct)
    result.update({'n_test_rows': int(len(y_te)), 'lr_accuracy': float(lr_correct.mean()), 'nn_accuracy': float(nn_correct.mean()), 'lr_f1': float(f1_score(y_te, y_pred_lr, zero_division=0)), 'nn_f1': float(f1_score(y_te, y_pred_nn, zero_division=0))})
    print(f'[McNemar] b={result['b']} (LR right / NN wrong), c={result['c']} (LR wrong / NN right), n_discordant={result['n_discordant']}, p={result['p_value']:.4g}')
    return result

def lr_ctor():
    return LogisticRegression(random_state=SEED, max_iter=1000, C=0.1)

def dietterich_5x2cv():
    import tensorflow as tf
    print(f'[5x2cv] loading training split ({N_CV_ITERATIONS} iterations)…')
    idx = json.loads((HERE / 'holdout_indices.json').read_text())
    idx_tr = np.array(idx['train_indices'])
    X_all, y_all, _, _ = load_and_prepare()
    X_tr, y_tr = (X_all[idx_tr], y_all[idx_tr])
    n = len(X_tr)
    print(f'[5x2cv] {n} rows in training split')
    deltas = []
    # Five times: split in half, train each model on A and score on B, then swap.
    # Each pass yields an LR-minus-NN F1 difference; we want their spread.
    for i in range(N_CV_ITERATIONS):
        iter_seed = SEED + i * 7
        rng = np.random.default_rng(iter_seed)
        perm = rng.permutation(n)
        half = n // 2
        A_idx, B_idx = (perm[:half], perm[half:2 * half])
        tf.random.set_seed(iter_seed)
        _, f1_lr_B, _ = _sklearn_f1(X_tr[A_idx], y_tr[A_idx], X_tr[B_idx], y_tr[B_idx])
        f1_nn_B = _nn_f1(X_tr[A_idx], y_tr[A_idx], X_tr[B_idx], y_tr[B_idx])
        p_1 = float(f1_lr_B - f1_nn_B)
        tf.random.set_seed(iter_seed + 1)
        _, f1_lr_A, _ = _sklearn_f1(X_tr[B_idx], y_tr[B_idx], X_tr[A_idx], y_tr[A_idx])
        f1_nn_A = _nn_f1(X_tr[B_idx], y_tr[B_idx], X_tr[A_idx], y_tr[A_idx])
        p_2 = float(f1_lr_A - f1_nn_A)
        print(f'[5x2cv] iter {i + 1}/{N_CV_ITERATIONS}: LR F1 B={f1_lr_B:.4f} A={f1_lr_A:.4f} | NN F1 B={f1_nn_B:.4f} A={f1_nn_A:.4f} | p1={p_1:+.4f} p2={p_2:+.4f}')
        deltas.append((p_1, p_2))
    # Dietterich's combined 5x2cv t-statistic: first-fold difference over the
    # pooled within-pair variance. Small |t| / large p => no reliable difference.
    variances = []
    for p_1, p_2 in deltas:
        pbar = (p_1 + p_2) / 2.0
        s2 = (p_1 - pbar) ** 2 + (p_2 - pbar) ** 2
        variances.append(s2)
    denom = math.sqrt(1.0 / N_CV_ITERATIONS * sum(variances))
    p_1_first = deltas[0][0]
    if denom == 0:
        t_stat, p_value = (0.0, 1.0)
    else:
        t_stat = p_1_first / denom
        p_value = 2.0 * float(t_dist.sf(abs(t_stat), df=N_CV_ITERATIONS))
    print(f'[5x2cv] t = {t_stat:+.4f},  df = {N_CV_ITERATIONS},  p = {p_value:.4g}')
    return {'test': 'dietterich_5x2cv_paired_t', 'iterations': N_CV_ITERATIONS, 'per_iter_deltas_lr_minus_nn': [{'p_1': p_1, 'p_2': p_2} for p_1, p_2 in deltas], 'mean_delta_lr_minus_nn': float(np.mean([p for row in deltas for p in row])), 't_statistic': float(t_stat), 'df': N_CV_ITERATIONS, 'p_value': float(p_value), 'note': 'Positive mean delta means LR > NN on F1 across folds; the p-value is two-sided.'}

def _sklearn_f1(X_tr, y_tr, X_te, y_te):
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    pipe = ImbPipeline(steps=[('scaler', StandardScaler()), ('smote', SMOTE(random_state=SEED, k_neighbors=3)), ('clf', lr_ctor())])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    return (pipe, float(f1), pipe.predict_proba(X_te)[:, 1])

def _nn_f1(X_tr, y_tr, X_te, y_te):
    from sklearn.preprocessing import StandardScaler
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
    return float(f1_score(y_te, y_pred, zero_division=0))

def make_verdict(mcnemar, t5x2):
    mc_sig = mcnemar['p_value'] < SIGNIFICANCE_ALPHA
    t_sig = t5x2['p_value'] < SIGNIFICANCE_ALPHA
    mc_fragment = f'McNemar p={mcnemar['p_value']:.4g} ({('significant' if mc_sig else 'not significant')} at α={SIGNIFICANCE_ALPHA})'
    t_fragment = f'5x2cv paired t-test p={t5x2['p_value']:.4g} ({('significant' if t_sig else 'not significant')} at α={SIGNIFICANCE_ALPHA})'
    # The honest tie-break: a claim only counts if BOTH the single-split and the
    # split-robust test agree. When they don't, default to 'no reliable difference'.
    if not mc_sig and (not t_sig):
        narrative = 'Under both tests, the NN does not statistically outperform LR on F1. Honest thesis framing: on leakage-free features, a heavily regularised logistic regression is statistically indistinguishable from the MLP; the added model complexity is not justified by these data.'
    elif mc_sig and t_sig:
        direction = 'NN better' if mcnemar['b'] < mcnemar['c'] and t5x2['mean_delta_lr_minus_nn'] < 0 else 'LR better'
        narrative = f'Both tests agree on significance (α={SIGNIFICANCE_ALPHA}); direction: {direction}.'
    else:
        narrative = "Tests disagree. McNemar operates on the single held-out test set; 5x2cv averages over 10 different splits. A split-robust claim requires both to agree; treat as 'no reliable difference' for the thesis."
    return {'summary': f'{mc_fragment}; {t_fragment}.', 'narrative': narrative}

def main():
    print('=' * 72)
    print('Statistical tests: NN vs LR')
    print('=' * 72)
    mcnemar = mcnemar_on_heldout()
    print()
    t5x2 = dietterich_5x2cv()
    print()
    verdict = make_verdict(mcnemar, t5x2)
    payload = {'seed': SEED, 'alpha': SIGNIFICANCE_ALPHA, 'mcnemar': mcnemar, 'dietterich_5x2cv': t5x2, 'verdict': verdict}
    (HERE / 'stat_tests_nn_vs_lr.json').write_text(json.dumps(payload, indent=2))
    print('=' * 72)
    print('VERDICT')
    print('=' * 72)
    print(verdict['summary'])
    print()
    print(verdict['narrative'])
    print()
    print(f'Saved {HERE / 'stat_tests_nn_vs_lr.json'}')
if __name__ == '__main__':
    main()
