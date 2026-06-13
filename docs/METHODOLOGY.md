# Methodology

The pipeline has four phases. Each `src/` file implements one of them.

## The leakage correction (why the headline number changed)

The first implementation reported an F1‑score of 99.7%. That is implausible for predicting
human purchase intent and was traced to **temporal data leakage**: several features were
computed from events that happen *after* the prediction moment (the first cart addition) —
for example, a count of events that occurred after the cart was created. The model was, in
effect, reading the future.

Two fixes removed it:

1. **Temporal shield.** No feature may use any event after the first cart addition. Feature
   engineering finds the timestamp of the first cart event and keeps only earlier events.
   `phase1_feature_engineering.py` raises an error if any feature is found to depend on
   post‑cart information.
2. **SMOTE inside each cross‑validation fold.** Oversampling the whole training set *before*
   cross‑validation lets validation folds see synthetic points derived from themselves.
   `phase2_train.py` puts SMOTE inside an `imblearn` pipeline so it is fitted within each
   fold only. This closed a suspicious 0.28 gap between cross‑validation and test F1.

After both fixes, F1 fell to a realistic ~0.47 with a cross‑validation/test gap below 0.01.

## Phase 1 — Data engineering (`phase1_feature_engineering.py`)

Aggregates 42.4M REES46 events (October 2019) into 99,941 sessions with 20 leakage‑free
features describing pre‑cart browsing, prices, and cart composition. Target: whether the
session ended in a purchase. Cart abandonment rate: 69.6%.

## Phase 2 — Prediction (`phase2_train.py`)

Trains three models spanning the interpretability–complexity spectrum: Logistic Regression,
Random Forest, and a Neural Network. Stratified 80/20 split, 5‑fold cross‑validation with
SMOTE‑in‑fold. Persists the fitted models, the scaler, feature names, the exact train/test
indices, per‑fold metrics, confusion matrices, and feature importances.

Held‑out test results: LR F1 0.467, NN F1 0.481, RF F1 0.438; ROC‑AUC ≈ 0.62–0.63.

## Phase 3 — Explainability (`phase3_explainability.py`) and significance (`phase3_significance.py`)

SHAP is applied to Logistic Regression and Random Forest; LIME to the Neural Network. The
three independent methods converge on the same top features — `views_before_cart`,
`browse_intensity_pre_cart`, `total_events_before_cart`. Pre‑cart browse behaviour, not price
and not cart contents, is the dominant signal; 85% of LIME reasons fall under "browse
indecision".

`phase3_significance.py` tests whether the Neural Network really beats Logistic Regression.
McNemar's test on the single held‑out split favours the NN (p = 0.002), but the more
conservative Dietterich 5×2cv test finds no difference (p = 0.96). The split‑robust conclusion
is that NN ≈ LR, so Logistic Regression is chosen as the primary model for its interpretability
with no real loss of predictive power.

## Phase 4 — Generation and evaluation (`phase4_generate_and_judge.py`)

For each at‑risk session, a real Google Gemini call generates a personalized recovery message,
prompted with that session's LIME reason. A second Gemini model (a different family from the
generator) acts as an **LLM‑as‑a‑judge**, scoring each message on clarity, relevance, urgency,
and personalization. Using a different model family for the judge mitigates self‑preference
bias (Zheng et al., 2023). Average judge score ≈ 7.0–7.3 / 10; clarity is near ceiling (~9.8)
and personalization is the weakest dimension (~7.3).
