"""
transformer_seq.py
==================

Small Transformer encoder over per-session pre-cart event sequences,
trained on the same 80/20 split used by every model in the thesis. Goal:
one number — held-out test F1 — to compare against LR (0.4674), NN
(0.4811), RF (0.4375), XGBoost (~0.43-0.47).

Architecture: deliberately small so it fits on CPU.
  embeddings (event_type, time_bucket, category, price) -> 64-dim
  + sinusoidal positional encoding
  2-layer TransformerEncoder, 4 heads, hidden 64, ffn 128, dropout 0.2
  mean-pool over unmasked positions
  dense(64->32) + ReLU + dropout(0.3) + dense(32->1) sigmoid

Loss: BCEWithLogitsLoss with pos_weight = n_neg / n_pos for class
imbalance (no SMOTE on sequences). Optimiser: Adam, lr=1e-3,
weight_decay=1e-4. Training: 30 epochs, batch 256, early stop on
validation F1 (patience=5).

This is thesis future-work (Appendix D), not part of the defended pipeline.
The event_sequences.npz it reads is built by extract_event_sequences.py and
is gitignored (large binary); regenerate it from the raw REES46 file to run.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)
from torch.utils.data import DataLoader, TensorDataset

# Paths are relative to this repo: future-work/ holds the sequence data and
# results; the engineered CSV and the shared 80/20 split live one level up.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NPZ = HERE / "event_sequences.npz"
META = HERE / "sequence_meta.json"
ENG_CSV = REPO / "data" / "engineered_sessions_no_leakage.csv"
HOLDOUT = REPO / "artifacts" / "holdout_indices.json"
OUT = HERE / "transformer_results.json"

SEED = 42
MAX_LEN = 64
EMB_DIM = 64
N_LAYERS = 2
N_HEADS = 4
FFN_DIM = 128
DROPOUT = 0.2
BATCH = 256
LR = 1e-3
WD = 1e-4
EPOCHS = 30
PATIENCE = 5
THRESHOLD = 0.45

torch.manual_seed(SEED)
np.random.seed(SEED)


class EventSeqTransformer(nn.Module):
    def __init__(self, n_events, n_time, n_cat, n_price, dim=EMB_DIM):
        super().__init__()
        # Each component embedded into dim/4, then concatenated -> dim.
        d4 = dim // 4
        self.event_emb = nn.Embedding(n_events, d4, padding_idx=0)
        self.time_emb = nn.Embedding(n_time, d4)
        self.cat_emb = nn.Embedding(n_cat, d4)
        self.price_emb = nn.Embedding(n_price, d4)
        self.pos = self._build_positional(MAX_LEN, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=N_HEADS,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.head = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    @staticmethod
    def _build_positional(L, D):
        pe = torch.zeros(L, D)
        pos = torch.arange(0, L, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * -(math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, ev, tm, cat, pr, mask):
        x = torch.cat([self.event_emb(ev), self.time_emb(tm), self.cat_emb(cat), self.price_emb(pr)], dim=-1)
        x = x + self.pos[:, : x.size(1), :]
        # PyTorch transformer expects True for "ignore"
        attn_pad = ~mask
        x = self.encoder(x, src_key_padding_mask=attn_pad)
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


def main():
    t0 = time.time()
    print("Loading sequences...")
    npz = np.load(NPZ)
    meta = json.loads(META.read_text())
    sid_seq = npz["session_ids"]
    ev = npz["event_type"].astype(np.int64)
    tm = npz["time_bucket"].astype(np.int64)
    cat = npz["category_id"].astype(np.int64)
    pr = npz["price_bucket"].astype(np.int64)
    mask = npz["mask"].astype(bool)
    label = npz["labels"].astype(np.int64)
    print(f"  shapes: ev={ev.shape}  mask={mask.shape}  label_mean={label.mean():.4f}")

    print("Aligning sequence rows to engineered-CSV row order so holdout_indices matches...")
    eng = pd.read_csv(ENG_CSV, usecols=["session_id"])
    csv_to_seq = {sid: i for i, sid in enumerate(sid_seq)}
    order = np.array([csv_to_seq[s] for s in eng["session_id"]])
    ev = ev[order]; tm = tm[order]; cat = cat[order]; pr = pr[order]; mask = mask[order]; label = label[order]
    print(f"  aligned, {len(order)} rows")

    holdout = json.loads(HOLDOUT.read_text())
    train_idx = np.asarray(holdout["train_indices"])
    test_idx = np.asarray(holdout["test_indices"])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train_idx))
    val_size = max(1, int(0.1 * len(train_idx)))
    val_idx = train_idx[perm[:val_size]]
    fit_idx = train_idx[perm[val_size:]]
    print(f"  train={len(fit_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    n_neg = (label[fit_idx] == 0).sum()
    n_pos = (label[fit_idx] == 1).sum()
    pos_weight = float(n_neg / max(1, n_pos))
    print(f"  pos_weight = {pos_weight:.4f}")

    def to_loader(idx, shuffle):
        ds = TensorDataset(
            torch.from_numpy(ev[idx]),
            torch.from_numpy(tm[idx]),
            torch.from_numpy(cat[idx]),
            torch.from_numpy(pr[idx]),
            torch.from_numpy(mask[idx]),
            torch.from_numpy(label[idx].astype(np.float32)),
        )
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0, drop_last=False)

    train_loader = to_loader(fit_idx, True)
    val_loader = to_loader(val_idx, False)
    test_loader = to_loader(test_idx, False)

    n_events = max(ev.max() + 1, len(meta["event_type_map"]) + 1)
    n_time = max(tm.max() + 1, meta["n_time_buckets"])
    n_cat = max(cat.max() + 1, meta["n_categories"] + 1)
    n_price = max(pr.max() + 1, meta["n_price_buckets"])
    print(f"  vocab: events={n_events} time={n_time} cat={n_cat} price={n_price}")

    model = EventSeqTransformer(int(n_events), int(n_time), int(n_cat), int(n_price))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: {n_params:,}")
    # pos_weight tells the loss that the minority (purchase) class costs more,
    # the sequence-model equivalent of the SMOTE / class_weight tricks elsewhere.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        all_y = []; all_p = []
        for batch in loader:
            ev_b, tm_b, cat_b, pr_b, mask_b, y_b = batch
            logits = model(ev_b, tm_b, cat_b, pr_b, mask_b)
            all_y.append(y_b.numpy()); all_p.append(torch.sigmoid(logits).numpy())
        y = np.concatenate(all_y); p = np.concatenate(all_p)
        pred = (p >= THRESHOLD).astype(int)
        return {
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            "y": y, "p": p, "pred": pred,
        }

    history = []
    best_val_f1 = -1.0
    best_state = None
    epochs_since_best = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_e = time.time()
        running = 0.0; n = 0
        for ev_b, tm_b, cat_b, pr_b, mask_b, y_b in train_loader:
            opt.zero_grad()
            logits = model(ev_b, tm_b, cat_b, pr_b, mask_b)
            loss = loss_fn(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * y_b.size(0); n += y_b.size(0)
        train_loss = running / max(1, n)
        val_metrics = evaluate(val_loader)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_auc": round(val_metrics["roc_auc"], 4) if val_metrics["roc_auc"] is not None else None,
            "secs": round(time.time() - t_e, 1),
        })
        auc_str = f"{val_metrics['roc_auc']:.4f}" if val_metrics['roc_auc'] is not None else "N/A "
        print(f"epoch {epoch:2d}  train_loss={train_loss:.4f}  val_f1={val_metrics['f1']:.4f}  val_auc={auc_str}  ({history[-1]['secs']}s)")
        # Keep the best-on-validation weights and stop if they stop improving,
        # so the reported test number is not from an over-trained final epoch.
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                print(f"  early stop after {epoch} epochs (no val_f1 improvement for {PATIENCE} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(test_loader)
    cm = confusion_matrix(test_metrics["y"], test_metrics["pred"]).tolist()
    cm_default = confusion_matrix(test_metrics["y"], (test_metrics["p"] >= 0.5).astype(int)).tolist()

    out = {
        "seed": SEED,
        "model": {
            "name": "Transformer",
            "params": int(n_params),
            "max_len": MAX_LEN,
            "emb_dim": EMB_DIM,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "ffn_dim": FFN_DIM,
            "dropout": DROPOUT,
        },
        "train": {
            "fit_n": int(len(fit_idx)),
            "val_n": int(len(val_idx)),
            "test_n": int(len(test_idx)),
            "pos_weight": pos_weight,
            "epochs_run": len(history),
            "best_val_f1": best_val_f1,
            "history": history,
        },
        "test": {
            "threshold_0p45": {
                "metrics": {k: float(v) for k, v in test_metrics.items() if k in ("f1","precision","recall","roc_auc")},
                "confusion_matrix": cm,
            },
            "threshold_0p50": {
                "metrics": {
                    "f1": float(f1_score(test_metrics["y"], (test_metrics["p"] >= 0.5).astype(int), zero_division=0)),
                    "roc_auc": float(test_metrics["roc_auc"]),
                },
                "confusion_matrix": cm_default,
            },
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT}")
    print(f"\nFINAL TEST @ 0.45: F1={test_metrics['f1']:.4f}  AUC={test_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
