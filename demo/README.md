# Live demo

Runs the **real fitted Logistic Regression** on the held‑out test set and prints, in colour,
in about two seconds (no TensorFlow, no internet, no API key):

```bash
python demo/live_demo.py          # English
python demo/live_demo.py --tr     # Turkish
```

It shows:

1. **The honest F1 = 0.4674** and ROC‑AUC = 0.6177, recomputed live on the 19,989 held‑out
   sessions the model never trained on.
2. **Browse behaviour, not price** — a real abandoner (≈$511 cart, P(purchase) ≈ 0%) versus a
   real purchaser (≈$90 cart, P(purchase) ≈ 97%). The verdict tracks the browsing pattern, not
   the cart value.
3. **A real Gemini intervention** and the judge's average scores per dimension.

It reads the shipped files in `../artifacts/` (the fitted model and result JSONs) and
`../data/engineered_sessions_no_leakage.csv`, so it works out of the box after
`pip install -r requirements.txt`.
