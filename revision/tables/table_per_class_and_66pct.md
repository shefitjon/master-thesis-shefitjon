# Per-class performance & the “66.5%” reconciliation (integrity)

| Model | Abandon recall | Abandon prec. | Purchase recall | Purchase prec. | Accuracy |
|---|---|---|---|---|---|
| LR | 0.517 | 0.768 | 0.643 | 0.367 | 0.555 |
| RF | 0.707 | 0.753 | 0.468 | 0.411 | 0.634 |
| NN | 0.520 | 0.781 | 0.665 | 0.377 | 0.564 |

## The 66.5% claim
- **Thesis said:** 66.5% of true abandoners (Table 4.2 caption / §4.2)
- **Actually:** 66.5% is the Neural Network PURCHASE-class recall (TP/(FN+TP) = 0.6652), i.e. the share of true purchasers the NN catches at threshold 0.45 — NOT abandoners.
- **Fix:** Restate as 'the NN recovers 66.5% of would-be purchasers (purchase recall)'. Abandon-class recall is LR 0.52 / RF 0.71 / NN 0.52.

Confusion-matrix heatmaps: `../statistics/cm_lr.png`, `cm_rf.png`, `cm_nn.png`.
