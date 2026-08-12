# Statistics cluster (revision round)

Answers the jury's quantitative asks on the saved seed-42 split (held-out test
n=19,989). All numbers from real models/data; reproducible via the two scripts.

## Files
- `all_pairs_stats.py` — computes everything below (self-contained, correct paths).
- `build_tables.py` — renders the markdown/CSV tables in `../tables/`.
- `all_pairs_significance.json` — full machine-readable results.
- `cm_lr.png`, `cm_rf.png`, `cm_nn.png` — confusion-matrix heatmaps (Q4).

## Headline results

**Bootstrap 95% CIs on F1 (Q14)** — LR 0.467 [0.458, 0.477] · RF 0.438 [0.427, 0.448]
· NN 0.481 [0.471, 0.490].

**All-pairs significance (Q13)** — McNemar (single split) + Dietterich 5×2cv (split-robust):
- **LR vs NN:** 5×2cv p=0.77 (no reliable difference). McNemar p=0.0019 on the single
  split, but that is not split-robust.
- **LR vs RF:** 5×2cv p=0.002 — **RF significantly worse on F1.**
- **NN vs RF:** 5×2cv p=0.018 — **RF significantly worse on F1.**

**Proper equivalence (Q12, bootstrap TOST — 90% CI of the F1 difference within ±margin):**
- **LR ≡ NN within ±0.02 F1** (CI90 [-0.018, -0.009]) — equivalence *confirmed*, not
  merely "failed to reject." This is the correct replacement for the earlier invalid
  "5×2cv p=0.96 ⇒ equivalent."
- LR vs RF and NN vs RF are **not** equivalent within ±0.02 (RF is reliably weaker on F1).

**Take-away for model choice:** LR and NN are statistically equivalent on F1, so the
thesis selects **LR** on interpretability + speed (0.13 ms vs NN's 37.6 ms/session,
~290×). RF is significantly weaker on F1/purchase-recall — it buys higher *accuracy*
and *abandon-recall* only by leaning to the majority class.

**Confusion matrices / per-class (Q4) + the "66.5%" fix:** the thesis's "66.5% of true
abandoners" is actually the **NN purchase-class recall** (0.665). True abandon-class
recall is LR 0.517 / RF 0.707 / NN 0.520. See `../tables/table_per_class_and_66pct.md`.
