#!/usr/bin/env python3
"""build_tables.py — turn the stats JSONs into jury-facing markdown tables.

Reads cv_corrected_results.json, inference_latency.json (from final-form) and
all_pairs_significance.json (from new/statistics), emits markdown + CSV tables
into new/tables/. The vs-literature numbers are hard-coded from the verified
bibliography (thesis-wiki/raw/thesis/literature.md) with citations.
"""
import json, csv
from pathlib import Path

MODELS = Path("/home/user/PyCharmMiscProject/final-form/latest-corrected")
STATS = Path("/home/user/Downloads/thesis_revision/new/statistics")
OUT = Path("/home/user/Downloads/thesis_revision/new/tables")
OUT.mkdir(parents=True, exist_ok=True)
SHORT = {"LogisticRegression": "LR", "RandomForest": "RF", "NeuralNetwork": "NN"}
ORDER = ["LogisticRegression", "RandomForest", "NeuralNetwork"]

cv = json.loads((MODELS / "cv_corrected_results.json").read_text())
lat = json.loads((MODELS / "inference_latency.json").read_text())
sig = json.loads((STATS / "all_pairs_significance.json").read_text())
boot = sig["bootstrap_metric_ci95"]
pc = sig["per_class_metrics"]


def w(name, text):
    (OUT / name).write_text(text)
    print("wrote", OUT / name)


# ---- Table 1: model comparison ----
rows = []
hdr = ["Model", "Test F1 (95% CI)", "Precision", "Recall", "ROC-AUC", "Accuracy",
       "CV F1 (mean±std)", "Abandon recall", "Purchase recall", "Latency/session (ms)"]
for m in ORDER:
    t = cv["test"][m]; b = boot[m]; cvm = cv["cv"][m]
    rows.append([
        SHORT[m],
        f"{t['f1']:.3f} [{b['f1']['ci95'][0]:.3f}, {b['f1']['ci95'][1]:.3f}]",
        f"{t['precision']:.3f}", f"{t['recall']:.3f}", f"{t['roc_auc']:.3f}", f"{t['accuracy']:.3f}",
        f"{cvm['f1']['mean']:.3f}±{cvm['f1']['std']:.3f}",
        f"{pc[m]['abandon_recall']:.3f}", f"{pc[m]['purchase_recall']:.3f}",
        f"{lat[m]['single_session_ms']['median']:.2f}"])
md = ["# Table — Model comparison (REES46, leakage-free, held-out test n=19,989)\n",
      "| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
md += ["| " + " | ".join(r) + " |" for r in rows]
md += ["\n*Positive class = purchase. NN threshold 0.45; LR/RF threshold 0.50. "
       "95% CI from 2,000-sample bootstrap of the test set. Latency = single-session "
       "median, single-thread CPU.*\n"]
w("table_model_comparison.md", "\n".join(md))
with open(OUT / "table_model_comparison.csv", "w", newline="") as f:
    cw = csv.writer(f); cw.writerow(hdr); cw.writerows(rows)
print("wrote", OUT / "table_model_comparison.csv")

# ---- Table 2: all-pairs significance + equivalence ----
md = ["# Table — All-pairs model comparison: significance & equivalence\n",
      "Held-out McNemar (single test set) + Dietterich 5×2cv paired t (split-robust) "
      "+ equivalence by bootstrap TOST (90% CI of the F1 difference within ±margin).\n",
      "| Pair | McNemar p | 5×2cv t | 5×2cv p | Diff sig.? | F1 diff [90% CI] | Equivalent ±0.02? | Equivalent ±0.05? |",
      "|---|---|---|---|---|---|---|---|"]
for pair in sig["mcnemar_all_pairs"]:
    mc = sig["mcnemar_all_pairs"][pair]; cv5 = sig["five_by_two_cv_all_pairs"][pair]
    eq = sig["equivalence_bootstrap_TOST"][pair]
    md.append(f"| {pair} | {mc['p_value']:.4g} | {cv5['t_statistic']:+.3f} | {cv5['p_value']:.3f} | "
              f"{'yes' if cv5['significant_alpha0.05'] else 'no'} | "
              f"[{eq['ci90_diff'][0]:+.3f}, {eq['ci90_diff'][1]:+.3f}] | "
              f"{'YES' if eq['equivalent_within_margin_TOST_alpha0.05']['margin_0.02'] else 'no'} | "
              f"{'YES' if eq['equivalent_within_margin_TOST_alpha0.05']['margin_0.05'] else 'no'} |")
md += ["\n**Reading it:** McNemar tests one fixed test split (sensitive to it); 5×2cv averages "
       "over 10 splits (robust). *Failing to reject* a difference (high 5×2cv p) is **not** proof of "
       "equivalence — that needs TOST: equivalence holds only when the whole 90% CI of the F1 "
       "difference sits inside the margin. Margins are in F1 points (0.02 ≈ practically negligible).\n"]
w("table_significance.md", "\n".join(md))

# ---- Table 3: vs-literature (verified citations) ----
lit = [
    ("Abdullah-All-Tanvir et al. (2023)", "UCI Online Shoppers", "full incl. PageValues", "0.898", "leaky"),
    ("Sakar et al. (2019)", "UCI Online Shoppers", "full incl. GA fields", "0.58 (MLP) / 0.84 (LSTM)", "leaky"),
    ("Baati & Mohsil (2020)", "UCI Online Shoppers", "metadata only (GA removed)", "0.60", "leaky cols removed"),
    ("Esmeli & Gökçe (2025)", "retail clickstream", "session-level", "0.89", "leaky (contrast)"),
    ("Esmeli et al. (2021)", "retail clickstream", "event-bounded (incremental)", "0.55 → 0.78–0.83", "leakage-aware"),
    ("Requena et al. (2020)", "Coveo fashion", "truncated window", "0.86–0.91", "leakage-aware"),
    ("**This work — LR**", "**REES46**", "**strictly pre-cart (Temporal Shield)**", "**0.467**", "**leakage-free**"),
    ("**This work — NN**", "**REES46**", "**strictly pre-cart (Temporal Shield)**", "**0.481**", "**leakage-free**"),
]
md = ["# Table — This work vs. literature (purchase / cart-abandonment prediction)\n",
      "| Study | Dataset | Feature regime | Reported F1 | Leakage status |",
      "|---|---|---|---|---|"]
md += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in lit]
md += ["\n*The **0.898 → 0.60** drop on UCI when the Google-Analytics fields (PageValues, "
       "ExitRates, BounceRates) are removed (Abdullah-All-Tanvir 2023 → Baati & Mohsil 2020) "
       "**is the size of the leakage**. Our ~0.47 on REES46 is the leakage-free measurement — "
       "in the regime of the leakage-aware antecedents' early-event F1 (Esmeli 2021: 0.55 at "
       "event 1), not the leaky 0.85+ figures.*\n",
       "*Footnote: Asfe et al. (2025) report F1 0.825 on REES46, but for a **churn** task "
       "(different target), so it is not a direct cart-abandonment comparator. To our knowledge "
       "no prior **leakage-free** cart-abandonment F1 on REES46 has been published — this is the first.*\n"]
w("table_vs_literature.md", "\n".join(md))

# ---- Table 4: per-class + 66.5% reconciliation ----
recon = sig["claim_reconciliation_66pct"]
md = ["# Per-class performance & the “66.5%” reconciliation (integrity)\n",
      "| Model | Abandon recall | Abandon prec. | Purchase recall | Purchase prec. | Accuracy |",
      "|---|---|---|---|---|---|"]
for m in ORDER:
    p = pc[m]
    md.append(f"| {SHORT[m]} | {p['abandon_recall']:.3f} | {p['abandon_precision']:.3f} | "
              f"{p['purchase_recall']:.3f} | {p['purchase_precision']:.3f} | {p['accuracy']:.3f} |")
md += ["\n## The 66.5% claim",
       f"- **Thesis said:** {recon['claim_in_thesis']}",
       f"- **Actually:** {recon['actual_meaning']}",
       f"- **Fix:** {recon['fix']}",
       "\nConfusion-matrix heatmaps: `../statistics/cm_lr.png`, `cm_rf.png`, `cm_nn.png`.\n"]
w("table_per_class_and_66pct.md", "\n".join(md))
print("\nAll tables written to", OUT)
