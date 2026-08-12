"""
Regenerate the appendix/deck figures from REAL data and REAL models, replacing
the earlier simulated placeholders (np.random / hardcoded points).

Run:  python appendix_gen/regenerate_real_figures.py
Outputs into  thesis-wiki/figures/  (the deck reads these). Copy into
appendix_gen/assets/ for the thesis-revision appendix as needed.

Real AUCs produced here match the thesis exactly: LR 0.618, RF 0.621, NN 0.634.
"""
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
import joblib

MODELS = "/home/user/PyCharmMiscProject/final-form/latest-corrected"
CSV = "/home/user/PyCharmMiscProject/final-form/latest-results/engineered_sessions_no_leakage.csv"
SHAP_TOP = "/home/user/Documents/master-thesis-shefitjon/artifacts/shap_top_features.json"
FIG = "/home/user/Documents/thesis-wiki/figures"

COL = {"Neural Network": "#1F6FB2", "Logistic Regression": "#C0392B", "Random Forest": "#2E9E6E"}
ORDER = ["Neural Network", "Logistic Regression", "Random Forest"]


def load():
    df = pd.read_csv(CSV)
    y = df["target_purchase"].astype(int).to_numpy()
    xdf = df.drop(columns=["session_id", "target_purchase"])
    xdf["main_category"] = LabelEncoder().fit_transform(xdf["main_category"].fillna("unknown").astype(str))
    xdf = xdf.fillna(0)
    X = xdf.to_numpy(float)
    idx = json.load(open(f"{MODELS}/holdout_indices.json"))
    te = np.array(idx["test_indices"])
    return df, xdf, X, y, te


def scores_on_test(X, te):
    out = {}
    out["Logistic Regression"] = joblib.load(f"{MODELS}/model_lr.pkl").predict_proba(X[te])[:, 1]
    out["Random Forest"] = joblib.load(f"{MODELS}/model_rf.pkl").predict_proba(X[te])[:, 1]
    import tensorflow as tf
    nn = tf.keras.models.load_model(f"{MODELS}/model_nn.h5")
    sc = joblib.load(f"{MODELS}/scaler.pkl")
    out["Neural Network"] = nn.predict(sc.transform(X[te]), verbose=0).ravel()
    return out


def roc(scores, yte):
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC 0.50)")
    for k in ORDER:
        fpr, tpr, _ = roc_curve(yte, scores[k])
        plt.plot(fpr, tpr, color=COL[k], lw=2, label=f"{k} (AUC={roc_auc_score(yte, scores[k]):.3f})")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — real held-out predictions (leakage-free)")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{FIG}/p2_roc.png", dpi=150); plt.close()


def pr(scores, yte):
    plt.figure(figsize=(8, 6))
    for k in ORDER:
        p, r, _ = precision_recall_curve(yte, scores[k])
        plt.plot(r, p, color=COL[k], lw=2, label=k)
    s = scores["Logistic Regression"]; pred = s >= 0.45
    rec = (pred & (yte == 1)).sum() / (yte == 1).sum()
    prec = (pred & (yte == 1)).sum() / max(pred.sum(), 1)
    plt.scatter([rec], [prec], color="black", zorder=5, s=60)
    plt.annotate(f"LR @ 0.45\n(recall {rec:.2f}, prec {prec:.2f})", (rec, prec),
                 textcoords="offset points", xytext=(8, 8), fontsize=9)
    plt.axhline((yte == 1).mean(), color="gray", ls=":", lw=1, label=f"Base rate ({(yte == 1).mean():.2f})")
    plt.xlabel("Recall (purchase)"); plt.ylabel("Precision (purchase)")
    plt.title("Precision-Recall — real held-out predictions")
    plt.legend(loc="upper right"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{FIG}/p2_pr.png", dpi=150); plt.close()


def time_to_cart(df):
    ttc = df["time_to_cart"].to_numpy() / 60.0
    plt.figure(figsize=(8, 5))
    plt.hist(ttc[ttc <= 60], bins=60, color="skyblue", edgecolor="black")
    plt.axvline(np.median(ttc), color="#C0392B", ls="--", lw=1.5, label=f"median {np.median(ttc):.1f} min")
    plt.title(f"Distribution of Time-to-Cart (real, n={len(ttc):,} sessions)")
    plt.xlabel(f"Minutes to first cart-add  (shown 0-60; {100*(ttc>60).mean():.1f}% beyond, long right tail)")
    plt.ylabel("Session Count"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIG}/p1_dist.png", dpi=150); plt.close()


def browse_intensity(df):
    bi = df["browse_intensity_pre_cart"].to_numpy()
    plt.figure(figsize=(8, 5))
    plt.hist(bi[bi <= 20], bins=60, color="lightgreen", edgecolor="black")
    plt.title(f"Distribution of Pre-Cart Browse Intensity (real, n={len(bi):,})")
    plt.xlabel(f"Views per minute  (shown 0-20; {100*(bi>20).mean():.1f}% beyond)")
    plt.ylabel("Session Count"); plt.tight_layout()
    plt.savefig(f"{FIG}/p1_intensity.png", dpi=150); plt.close()


def shap_global():
    top = json.load(open(SHAP_TOP))["LogisticRegression_top20"][:7]
    feats = [t["feature"] for t in top][::-1]
    vals = [t["mean_abs_shap"] for t in top][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(feats, vals, color="teal")
    plt.title("Global Mean Absolute SHAP Attribution (LR, real)")
    plt.xlabel("Mean |SHAP value|"); plt.tight_layout()
    plt.savefig(f"{FIG}/p3_global_imp.png", dpi=150); plt.close()


def feature_corr(df):
    import matplotlib.cm as cm
    cols = ["views_before_cart", "total_events_before_cart", "browse_intensity_pre_cart",
            "unique_products_viewed", "time_to_cart", "initial_cart_value"]
    c = df[cols].corr().to_numpy()
    plt.figure(figsize=(8, 7))
    im = plt.imshow(c, cmap="YlGnBu", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046)
    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(c[i, j]) > 0.6 else "black", fontsize=9)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Feature correlation (real) — note views vs total_events = 1.00")
    plt.tight_layout()
    plt.savefig(f"{FIG}/p3_feature_corr.png", dpi=150); plt.close()


ABLATION = "/home/user/PyCharmMiscProject/final-form/latest-corrected"


def p4_ablation():
    from statistics import mean
    pro = json.load(open(f"{ABLATION}/gemini_full_results.json"))["results"]
    fl = json.load(open(f"{ABLATION}/gemini_full_results_flash_lite.json"))["results"]
    dims = ["relevance", "personalization", "urgency", "clarity"]

    def col(rs, key):  # per-session judge dict -> dimension lists (skip Nones)
        return {d: [r[key][d] for r in rs if r.get(key) and r[key].get(d) is not None] for d in dims}

    pj = col(pro, "judge")       # Pro generator, judged by flash-latest
    fj = col(fl, "judge_alt")    # Flash-Lite generator, re-judged by flash-latest (same judge)
    pm = [mean(pj[d]) for d in dims]; fm = [mean(fj[d]) for d in dims]

    x = np.arange(4); w = 0.36
    plt.figure(figsize=(9, 6))
    plt.bar(x - w / 2, pm, w, label="Gemini 2.5 Pro (generator)", color="royalblue")
    plt.bar(x + w / 2, fm, w, label="Gemini Flash-Lite (generator)", color="tomato")
    for i in range(4):
        plt.text(x[i] - w / 2, pm[i] + 0.1, f"{pm[i]:.2f}", ha="center", fontsize=9)
        plt.text(x[i] + w / 2, fm[i] + 0.1, f"{fm[i]:.2f}", ha="center", fontsize=9)
    plt.ylabel("Mean judge score (1-10)"); plt.ylim(0, 10.5)
    plt.title("Generator-size ablation: Pro vs Flash-Lite (both judged by Flash-Latest)")
    plt.xticks(x, ["Relevance", "Personalization", "Urgency", "Clarity"])
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{FIG}/p4_dimensions.png", dpi=150); plt.close()

    bins = np.arange(0.5, 10.6, 1)
    plt.figure(figsize=(8, 5))
    plt.hist(pj["personalization"], bins=bins, alpha=0.55, color="royalblue", edgecolor="black",
             label=f"Pro (mean {mean(pj['personalization']):.2f})")
    plt.hist(fj["personalization"], bins=bins, alpha=0.55, color="tomato", edgecolor="black",
             label=f"Flash-Lite (mean {mean(fj['personalization']):.2f})")
    plt.title("Distribution of Personalization scores (real, n=150 each)")
    plt.xlabel("Personalization score (1-10)"); plt.ylabel("Count"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{FIG}/p4_pers_dist.png", dpi=150); plt.close()


def main():
    df, xdf, X, y, te = load()
    s = scores_on_test(X, te)
    yte = y[te]
    print("real AUCs:", {k: round(roc_auc_score(yte, s[k]), 4) for k in ORDER})
    roc(s, yte)
    pr(s, yte)
    time_to_cart(df)
    browse_intensity(df)
    shap_global()
    feature_corr(df)
    p4_ablation()
    print("wrote: p2_roc, p2_pr, p1_dist, p1_intensity, p3_global_imp, p3_feature_corr, p4_dimensions, p4_pers_dist -> figures/")


if __name__ == "__main__":
    main()
