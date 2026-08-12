# Table — All-pairs model comparison: significance & equivalence

Held-out McNemar (single test set) + Dietterich 5×2cv paired t (split-robust) + equivalence by bootstrap TOST (90% CI of the F1 difference within ±margin).

| Pair | McNemar p | 5×2cv t | 5×2cv p | Diff sig.? | F1 diff [90% CI] | Equivalent ±0.02? | Equivalent ±0.05? |
|---|---|---|---|---|---|---|---|
| LR_vs_NN | 0.001924 | +0.303 | 0.774 | no | [-0.018, -0.009] | YES | YES |
| LR_vs_RF | 6.222e-108 | +5.848 | 0.002 | yes | [+0.023, +0.038] | no | YES |
| NN_vs_RF | 1.351e-101 | +3.491 | 0.018 | yes | [+0.037, +0.051] | no | no |

**Reading it:** McNemar tests one fixed test split (sensitive to it); 5×2cv averages over 10 splits (robust). *Failing to reject* a difference (high 5×2cv p) is **not** proof of equivalence — that needs TOST: equivalence holds only when the whole 90% CI of the F1 difference sits inside the margin. Margins are in F1 points (0.02 ≈ practically negligible).

**Multiple comparisons (Holm):** conclusions unchanged after Holm step-down over the three pairs — LR-RF holm-p = 0.0063, NN-RF = 0.035 (both still significant), LR-NN = 0.774.

**Margin justification (Δ = 0.02 F1):** chosen a priori as ≤ each model's own bootstrap 95% CI width (single-test-set sampling noise) and smaller than the F1 shift from a 0.05 decision-threshold nudge. Note TOST fails at Δ = 0.01, so the equivalence claim always states its margin. **Framing:** the LR-NN CI excludes zero — the NN is statistically better; the correct statement is "statistically detectable but bounded below 0.02 F1 — practically negligible," never "no difference."
