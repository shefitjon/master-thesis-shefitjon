# Analysis of Artificial Intelligence Models for Optimizing E‑Commerce Performance

A four‑phase AI pipeline for shopping‑cart abandonment. A **data‑engineering** layer
(Phase 1) builds leakage‑free session features; the pipeline then **predicts** abandonment
(Phase 2), **explains** it with SHAP and LIME (Phase 3), and **generates** a personalized
recovery message with a large language model, scored by a second model acting as judge (Phase 4).

M.Sc. thesis — Eskişehir Osmangazi University, Graduate School of Natural and Applied
Sciences, Department of Computer Engineering. Author: Shefitjon Bregu. Advisor:
Dr. Öğr. Üyesi Savaş Okyay.

## Honest results (read this first)

An early version of this work reported an F1‑score of **99.7%**. That number was the
result of **temporal data leakage** — some features were computed from events that occur
*after* the prediction moment (the first cart addition). After removing every such feature
and re‑engineering with a strict temporal cutoff, the realistic, leakage‑free result is:

| Model | Test F1 | Test ROC‑AUC |
|---|---|---|
| Logistic Regression | **0.467** | 0.618 |
| Neural Network | 0.481 | 0.634 |
| Random Forest | 0.438 | 0.621 |

The cross‑validation‑to‑test gap is below 0.01, so the models generalize. Predicting
abandonment from pre‑cart behaviour alone is genuinely hard; an honest ~0.47 is the result,
and the contribution is the integrated, leakage‑free, explainable, generative pipeline — not
a headline accuracy number. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full story.

## Repository layout

```
master-thesis-shefitjon/
├── src/
│   ├── phase1_data_engineering.py      # full ETL: raw events -> engineered CSV (chunked, 70/30, shield)
│   ├── phase1_feature_engineering.py   # focused, runnable demo of the temporal-shield logic
│   ├── phase2_train.py                 # LR / RF / NN with SMOTE inside each CV fold
│   ├── phase3_explainability.py        # SHAP (LR, RF) + LIME (NN)
│   ├── phase3_significance.py          # McNemar + Dietterich 5x2cv (NN vs LR)
│   └── phase4_generate_and_judge.py    # Gemini intervention + LLM-as-a-judge
├── demo/
│   ├── live_demo.py                    # runs the real model in ~2 s (English / Turkish)
│   └── README.md
├── data/
│   ├── engineered_sessions_no_leakage.csv   # 99,941 sessions, 20 leakage-free features
│   └── 2019-Oct.csv                          # raw REES46 events — NOT in git, download separately
├── artifacts/                          # fitted LR model + the JSON results (so you can
│                                       #   inspect outputs and run the demo without re-training)
└── docs/METHODOLOGY.md                 # the four phases and the leakage correction in prose
```

The code carries short, human-written comments on the tricky lines; the broader
rationale lives here and in `docs/METHODOLOGY.md`.

## Setup

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Live demo (no API key needed)

Runs the real fitted Logistic Regression on the held‑out test set and prints the honest
F1, a browse‑behaviour example, and a real generated intervention:

```bash
python demo/live_demo.py          # English
python demo/live_demo.py --tr     # Turkish
```

## Running the pipeline, phase by phase

Run from the repository root. Each phase reads `data/` and reads/writes `artifacts/`.

```bash
python src/phase1_data_engineering.py      # data engineering: raw 2019-Oct.csv -> engineered CSV (optional; CSV is shipped)
python src/phase2_train.py                 # writes models + cv_corrected_results.json + indices
python src/phase3_explainability.py        # writes SHAP / LIME outputs   (needs phase 2 first)
python src/phase3_significance.py          # writes stat_tests_nn_vs_lr.json (needs phase 2 first)
python src/phase4_generate_and_judge.py    # needs a Gemini API key — see below
```

`artifacts/` already ships the Logistic Regression model and the result JSONs, so you can read
the outputs and run the demo immediately. Re‑running phase 2 regenerates the Random Forest,
Neural Network and scaler (these large binaries are not shipped).

## Phase 4 — Gemini API key

Phase 4 calls the Google Gemini API, so it needs a free key. **Phases 1–3 and the demo do not.**

1. **Create** a key at Google AI Studio: <https://aistudio.google.com/apikey> (sign in →
   "Create API key").
2. **Set** it in your shell:
   ```bash
   export GEMINI_API_KEY="your-key-here"      # macOS / Linux
   setx GEMINI_API_KEY "your-key-here"        # Windows (open a new terminal afterwards)
   ```
3. **Run** it — the script reads `GEMINI_API_KEY` from the environment automatically:
   ```bash
   python src/phase4_generate_and_judge.py
   ```

Free‑tier quota is limited (a few requests per minute); the script paces itself between calls.

## Data

The engineered, leakage‑free dataset (`data/engineered_sessions_no_leakage.csv`) is derived
from the **REES46 eCommerce behavior** dataset, October 2019
(<https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store>):
42.4M raw events aggregated into 99,941 shopping sessions. October 2019 was chosen as a
"normal" month — pre‑pandemic, and free of the November–December Black Friday / Christmas spikes.

## Citation

Bregu, S., & Kartal, Y. (2025). *Comparative analysis of decision trees, support vector
machines, and neural networks for online purchase prediction.* ICADA 2025 — 5th International
Conference on Applied Data Science.
