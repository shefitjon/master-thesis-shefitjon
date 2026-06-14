# Future work — sequence model over raw events

This is the **future-work** experiment referenced in the thesis (Appendix D) and
in the defense: instead of the 20 aggregated, hand-engineered features, read each
session's **raw pre-cart event sequence in order** and let a small Transformer
learn the temporal pattern itself.

It is **not part of the defended pipeline** — it is the most promising next step,
and the strongest single argument that event *order* carries signal the aggregate
features average away.

## Result

| Model | Test F1 @0.45 | Test ROC-AUC |
|---|---|---|
| Logistic Regression (thesis) | 0.4674 | 0.618 |
| Neural Network (thesis) | 0.4811 | 0.634 |
| Random Forest (thesis) | 0.4375 | 0.621 |
| XGBoost (explored) | 0.4349 | 0.630 |
| **Transformer (this dir)** | **0.5024** | **0.642** |

The Transformer is the only model that clears 0.50 — modest, but a real step above
every aggregate-feature model, and it does it by reading the sequence rather than a
summary of it. Full training history and confusion matrices are in
`transformer_results.json`.

## What it does

`transformer_seq.py` builds a small encoder: each event becomes four embeddings
(event type, time-since-start bucket, category, price bucket), concatenated to a
64-dim token, plus sinusoidal positional encoding. Two Transformer-encoder layers
(4 heads, FFN 128, dropout 0.2) attend across the session; the unmasked positions
are mean-pooled and a small head outputs the purchase probability. About 70k
parameters — deliberately small so it trains on CPU.

Class imbalance is handled with a weighted loss (`pos_weight = n_neg / n_pos`),
the sequence-model equivalent of the SMOTE / class-weight tricks in Phase 2. It
trains on the **exact same 80/20 split** as every other model
(`artifacts/holdout_indices.json`) so the F1 is directly comparable.

The Temporal Shield still holds: the sequences are truncated at the first cart-add
during the ETL, so no event after the prediction moment ever reaches the model.

## Files

| File | What |
|---|---|
| `transformer_seq.py` | The model, training loop, and evaluation. |
| `transformer_results.json` | Training history, best validation F1, final test metrics + confusion matrices. |
| `sequence_meta.json` | Vocabulary sizes and sequence-length stats from the ETL. |

## Running it

The model reads `event_sequences.npz` — the per-session integer sequences. That
file is large and **gitignored**, so it is not in the repo. To reproduce:

1. Regenerate `event_sequences.npz` from the raw REES46 file with the sequence
   ETL (the same one that enforces the Temporal Shield at the event level).
2. `pip install torch` (CPU build is fine).
3. `python future-work/transformer_seq.py` — about 30–90 minutes on CPU.

It writes a fresh `transformer_results.json`; the committed one is the run quoted
above and in the thesis.
