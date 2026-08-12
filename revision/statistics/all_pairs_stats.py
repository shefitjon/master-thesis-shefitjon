#!/usr/bin/env python3
"""
all_pairs_stats.py — statistics cluster for the thesis revision.

Answers the jury's quantitative asks, all on the saved seed-42 split:
  Q13  all-pairs significance: McNemar (test set) + Dietterich 5x2cv paired t
       for EVERY model pair (LR-NN, LR-RF, NN-RF) — RF was never tested before.
  Q12  proper EQUIVALENCE testing (TOST): the old "5x2cv p=0.96 => equivalent"
       is invalid (failing to reject H0 != equivalence). We use the bootstrap
       90% CI of the F1 difference within +/- margin (TOST), plus a 5x2cv TOST.
  Q14  bootstrap 95% CIs on F1 (and P/R/AUC) per model.
  Q4   confusion-matrix heatmaps + per-class precision/recall, and the
       reconciliation of the mislabeled "66.5% of true abandoners" claim.

Self-contained (correct final-form paths; the old train script's paths are
stale). Hyperparameters / NN architecture copied verbatim from train_cv_corrected.py.
"""
import json, math, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest, t as t_dist
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

SEED = 42
MODELS = Path("/home/user/PyCharmMiscProject/final-form/latest-corrected")
CSV = Path("/home/user/PyCharmMiscProject/final-form/latest-results/engineered_sessions_no_leakage.csv")
OUT = Path("/home/user/Downloads/thesis_revision/new/statistics")
OUT.mkdir(parents=True, exist_ok=True)
NN_THRESH = 0.45
B_BOOT = 2000
N_CV = 5
MARGINS = [0.01, 0.02, 0.05]
PAIRS = [("LogisticRegression", "NeuralNetwork"), ("LogisticRegression", "RandomForest"),
         ("NeuralNetwork", "RandomForest")]
SHORT = {"LogisticRegression": "LR", "RandomForest": "RF", "NeuralNetwork": "NN"}


def load_and_prepare():
    df = pd.read_csv(CSV)
    y = df["target_purchase"].astype(int).to_numpy()
    Xd = df.drop(columns=["session_id", "target_purchase"])
    Xd["main_category"] = LabelEncoder().fit_transform(Xd["main_category"].fillna("unknown").astype(str))
    Xd = Xd.fillna(0)
    return Xd.to_numpy(np.float64), y


def lr_ctor():
    return LogisticRegression(random_state=SEED, max_iter=1000, C=0.1)


def rf_ctor():
    return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=50,
                                  min_samples_leaf=20, random_state=SEED, n_jobs=-1)


def build_nn(input_dim):
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    tf.keras.backend.clear_session()
    m = Sequential([Dense(16, activation="relu", input_dim=input_dim),
                    BatchNormalization(), Dropout(0.3),
                    Dense(8, activation="relu"), BatchNormalization(), Dropout(0.3),
                    Dense(1, activation="sigmoid")])
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def nn_fit_predict(X_tr, y_tr, X_te):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    sc = StandardScaler().fit(X_tr)
    m = build_nn(X_tr.shape[1])
    cw = {0: 1.0, 1: (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)}
    m.fit(sc.transform(X_tr), y_tr, epochs=100, batch_size=64, validation_split=0.2,
          callbacks=[EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
          class_weight=cw, verbose=0)
    return (m.predict(sc.transform(X_te), verbose=0).ravel() > NN_THRESH).astype(int)


def sk_fit_predict(ctor, X_tr, y_tr, X_te):
    pipe = ImbPipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=SEED, k_neighbors=3)), ("clf", ctor())])
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


# ===================== test-set predictions (saved models) =====================
def test_predictions(X, y, te):
    Xte, yte = X[te], y[te]
    preds, probs = {}, {}
    with open(MODELS / "model_lr.pkl", "rb") as f: lr = pickle.load(f)
    with open(MODELS / "model_rf.pkl", "rb") as f: rf = pickle.load(f)
    preds["LogisticRegression"] = lr.predict(Xte); probs["LogisticRegression"] = lr.predict_proba(Xte)[:, 1]
    preds["RandomForest"] = rf.predict(Xte); probs["RandomForest"] = rf.predict_proba(Xte)[:, 1]
    from tensorflow.keras.models import load_model
    nn = load_model(MODELS / "model_nn.h5")
    with open(MODELS / "scaler.pkl", "rb") as f: sc = pickle.load(f)
    pr = nn.predict(sc.transform(Xte), verbose=0).ravel()
    preds["NeuralNetwork"] = (pr > NN_THRESH).astype(int); probs["NeuralNetwork"] = pr
    return yte, preds, probs


# ===================== Q13: McNemar all pairs =====================
def mcnemar(a_correct, b_correct):
    b = int(((a_correct == 1) & (b_correct == 0)).sum())
    c = int(((a_correct == 0) & (b_correct == 1)).sum())
    n = b + c
    p = 1.0 if n == 0 else binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
    return {"b": b, "c": c, "n_discordant": n, "p_value": float(p)}


# ===================== Q14: bootstrap CIs =====================
def bootstrap_metrics(yte, preds, probs):
    rng = np.random.default_rng(SEED)
    n = len(yte)
    idxs = [rng.integers(0, n, n) for _ in range(B_BOOT)]
    out = {}
    for m in preds:
        f1s, ps, rs, aucs = [], [], [], []
        yp, pr = preds[m], probs[m]
        for bi in idxs:
            yt = yte[bi]
            if yt.sum() == 0 or yt.sum() == len(yt):
                continue
            f1s.append(f1_score(yt, yp[bi], zero_division=0))
            ps.append(precision_score(yt, yp[bi], zero_division=0))
            rs.append(recall_score(yt, yp[bi], zero_division=0))
            aucs.append(roc_auc_score(yt, pr[bi]))
        def ci(v): return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
        out[m] = {"f1": {"point": round(f1_score(yte, yp, zero_division=0), 4), "ci95": ci(f1s)},
                  "precision": {"point": round(precision_score(yte, yp, zero_division=0), 4), "ci95": ci(ps)},
                  "recall": {"point": round(recall_score(yte, yp, zero_division=0), 4), "ci95": ci(rs)},
                  "roc_auc": {"point": round(roc_auc_score(yte, pr), 4), "ci95": ci(aucs)}}
    return out, idxs


# ===================== Q12: equivalence (bootstrap TOST) =====================
def equivalence_bootstrap(yte, preds, idxs):
    res = {}
    for a, b in PAIRS:
        diffs = []
        for bi in idxs:
            yt = yte[bi]
            if yt.sum() == 0 or yt.sum() == len(yt):
                continue
            diffs.append(f1_score(yt, preds[a][bi], zero_division=0) - f1_score(yt, preds[b][bi], zero_division=0))
        diffs = np.array(diffs)
        point = f1_score(yte, preds[a], zero_division=0) - f1_score(yte, preds[b], zero_division=0)
        ci90 = [float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))]
        ci95 = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        eq = {f"margin_{m}": bool(ci90[0] > -m and ci90[1] < m) for m in MARGINS}
        res[f"{SHORT[a]}_vs_{SHORT[b]}"] = {
            "f1_diff_point": round(float(point), 4),
            "ci90_diff": [round(ci90[0], 4), round(ci90[1], 4)],
            "ci95_diff": [round(ci95[0], 4), round(ci95[1], 4)],
            "equivalent_within_margin_TOST_alpha0.05": eq}
    return res


# ===================== Q13/Q12: 5x2cv all pairs + 5x2cv TOST =====================
def five_by_two_cv(X, y, tr):
    Xtr, ytr = X[tr], y[tr]
    n = len(Xtr)
    import tensorflow as tf
    # per iteration, per half: F1 of each model on the opposite half
    fold_f1 = {m: [] for m in SHORT}   # each entry: (f1_B, f1_A) for an iteration
    for i in range(N_CV):
        seed = SEED + i * 7
        perm = np.random.default_rng(seed).permutation(n)
        half = n // 2
        A, Bx = perm[:half], perm[half:2 * half]
        tf.random.set_seed(seed)
        f1B = {"LogisticRegression": f1_score(ytr[Bx], sk_fit_predict(lr_ctor, Xtr[A], ytr[A], Xtr[Bx]), zero_division=0),
               "RandomForest": f1_score(ytr[Bx], sk_fit_predict(rf_ctor, Xtr[A], ytr[A], Xtr[Bx]), zero_division=0),
               "NeuralNetwork": f1_score(ytr[Bx], nn_fit_predict(Xtr[A], ytr[A], Xtr[Bx]), zero_division=0)}
        tf.random.set_seed(seed + 1)
        f1A = {"LogisticRegression": f1_score(ytr[A], sk_fit_predict(lr_ctor, Xtr[Bx], ytr[Bx], Xtr[A]), zero_division=0),
               "RandomForest": f1_score(ytr[A], sk_fit_predict(rf_ctor, Xtr[Bx], ytr[Bx], Xtr[A]), zero_division=0),
               "NeuralNetwork": f1_score(ytr[A], nn_fit_predict(Xtr[Bx], ytr[Bx], Xtr[A]), zero_division=0)}
        for m in SHORT:
            fold_f1[m].append((f1B[m], f1A[m]))
        print(f"[5x2cv] iter {i+1}/{N_CV}: " + " ".join(f"{SHORT[m]} B={f1B[m]:.3f}/A={f1A[m]:.3f}" for m in SHORT))
    # pairwise Dietterich t + 5x2cv TOST
    res = {}
    for a, b in PAIRS:
        deltas = [(fold_f1[a][i][0] - fold_f1[b][i][0], fold_f1[a][i][1] - fold_f1[b][i][1]) for i in range(N_CV)]
        var = sum((p1 - (p1+p2)/2)**2 + (p2 - (p1+p2)/2)**2 for p1, p2 in deltas)
        denom = math.sqrt((1.0/N_CV) * var)
        mean_d = float(np.mean([d for row in deltas for d in row]))
        if denom == 0:
            t_stat, p_val = 0.0, 1.0
        else:
            t_stat = deltas[0][0] / denom
            p_val = 2.0 * float(t_dist.sf(abs(t_stat), df=N_CV))
        # 5x2cv TOST: two one-sided tests of mean_d vs +/-margin (df=5)
        tost = {}
        for m in MARGINS:
            if denom == 0:
                tost[f"margin_{m}"] = bool(abs(mean_d) < m)
                continue
            t_lower = (mean_d + m) / denom   # H0: d <= -m
            t_upper = (mean_d - m) / denom   # H0: d >= +m
            p_lower = float(t_dist.sf(t_lower, df=N_CV))     # one-sided
            p_upper = float(t_dist.cdf(t_upper, df=N_CV))    # one-sided
            tost[f"margin_{m}"] = bool(p_lower < 0.05 and p_upper < 0.05)
        res[f"{SHORT[a]}_vs_{SHORT[b]}"] = {
            "mean_delta_a_minus_b": round(mean_d, 4), "t_statistic": round(float(t_stat), 4),
            "df": N_CV, "p_value": round(float(p_val), 4),
            "significant_alpha0.05": bool(p_val < 0.05),
            "equivalent_TOST_alpha0.05": tost,
            "per_iter_deltas": [{"p1": round(p1, 4), "p2": round(p2, 4)} for p1, p2 in deltas]}
    fold_means = {SHORT[m]: round(float(np.mean([v for row in fold_f1[m] for v in row])), 4) for m in SHORT}
    return res, fold_means


# ===================== Q4: confusion matrices + per-class + 66.5% =====================
def confusion_analysis(yte, preds):
    out, per_class = {}, {}
    for m in preds:
        cm = confusion_matrix(yte, preds[m])  # [[TN,FP],[FN,TP]]
        tn, fp, fn, tp = cm.ravel()
        out[m] = cm.tolist()
        per_class[m] = {
            "abandon_recall": round(tn / (tn + fp), 4), "abandon_precision": round(tn / (tn + fn), 4),
            "purchase_recall": round(tp / (fn + tp), 4), "purchase_precision": round(tp / (fp + tp), 4),
            "accuracy": round((tn + tp) / (tn + fp + fn + tp), 4),
            "counts": {"TN_abandon_correct": int(tn), "FP": int(fp), "FN": int(fn), "TP_purchase_correct": int(tp)}}
        # heatmap
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()*0.5 else "black", fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["abandon", "purchase"]); ax.set_yticklabels(["abandon", "purchase"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{SHORT[m]} confusion matrix (test, n={len(yte):,})")
        fig.colorbar(im, fraction=0.046); fig.tight_layout()
        fig.savefig(OUT / f"cm_{SHORT[m].lower()}.png", dpi=150); plt.close(fig)
    reconciliation = {
        "claim_in_thesis": "66.5% of true abandoners (Table 4.2 caption / §4.2)",
        "actual_meaning": f"66.5% is the Neural Network PURCHASE-class recall (TP/(FN+TP) = {per_class['NeuralNetwork']['purchase_recall']:.4f}), i.e. the share of true purchasers the NN catches at threshold 0.45 — NOT abandoners.",
        "correct_per_class_recall": {SHORT[m]: {"abandon_recall": per_class[m]["abandon_recall"],
                                                "purchase_recall": per_class[m]["purchase_recall"]} for m in preds},
        "fix": "Restate as 'the NN recovers 66.5% of would-be purchasers (purchase recall)'. Abandon-class recall is LR 0.52 / RF 0.71 / NN 0.52."}
    return out, per_class, reconciliation


def main():
    print("Loading data + saved split…")
    X, y = load_and_prepare()
    idx = json.loads((MODELS / "holdout_indices.json").read_text())
    tr, te = np.array(idx["train_indices"]), np.array(idx["test_indices"])
    yte, preds, probs = test_predictions(X, y, te)
    correct = {m: (preds[m] == yte).astype(int) for m in preds}

    print("Q13: McNemar all pairs…")
    mc = {f"{SHORT[a]}_vs_{SHORT[b]}": mcnemar(correct[a], correct[b]) for a, b in PAIRS}

    print("Q14: bootstrap CIs…")
    boot, idxs = bootstrap_metrics(yte, preds, probs)

    print("Q12: equivalence (bootstrap TOST)…")
    eq_boot = equivalence_bootstrap(yte, preds, idxs)

    print("Q4: confusion matrices + per-class + 66.5% reconciliation…")
    cms, per_class, recon = confusion_analysis(yte, preds)

    print("Q13/Q12: 5x2cv all pairs (retraining; slowest)…")
    cv5, fold_means = five_by_two_cv(X, y, tr)

    payload = {"seed": SEED, "n_test": int(len(yte)), "nn_threshold": NN_THRESH,
               "bootstrap_B": B_BOOT, "tost_margins": MARGINS,
               "mcnemar_all_pairs": mc, "five_by_two_cv_all_pairs": cv5,
               "five_by_two_cv_fold_mean_f1": fold_means,
               "bootstrap_metric_ci95": boot, "equivalence_bootstrap_TOST": eq_boot,
               "confusion_matrices": cms, "per_class_metrics": per_class,
               "claim_reconciliation_66pct": recon}
    (OUT / "all_pairs_significance.json").write_text(json.dumps(payload, indent=2))
    print("\n=== McNemar (test) ===")
    for k, v in mc.items():
        print(f"  {k}: p={v['p_value']:.4g} (b={v['b']} c={v['c']})")
    print("=== 5x2cv ===")
    for k, v in cv5.items():
        print(f"  {k}: t={v['t_statistic']} p={v['p_value']} sig={v['significant_alpha0.05']} TOST={v['equivalent_TOST_alpha0.05']}")
    print("=== bootstrap F1 95% CI ===")
    for m, v in boot.items():
        print(f"  {SHORT[m]}: F1 {v['f1']['point']} CI {v['f1']['ci95']}")
    print("=== equivalence (bootstrap 90% CI within margin) ===")
    for k, v in eq_boot.items():
        print(f"  {k}: diff {v['f1_diff_point']} CI90 {v['ci90_diff']} -> {v['equivalent_within_margin_TOST_alpha0.05']}")
    print(f"\nWrote {OUT/'all_pairs_significance.json'} + cm_*.png")


if __name__ == "__main__":
    main()
