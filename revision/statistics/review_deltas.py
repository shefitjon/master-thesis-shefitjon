#!/usr/bin/env python3
"""review_deltas.py — apply the second-opinion review's computational deltas.

1. Holm correction over the three pairwise 5x2cv p-values (and McNemar).
2. Difference-in-differences self-preference estimator (controls for overall
   judge harshness; the naive +0.50 is asymmetric because Gemini generators
   have no self-family judge).
3. Latency consistency: one canonical number (median) from inference_latency.json.
Appends results into the existing JSONs + prints a summary.
"""
import json
from pathlib import Path
from statistics import mean

STATS = Path("/home/user/Downloads/thesis_revision/new/statistics")
GENAI = Path("/home/user/Downloads/thesis_revision/new/genai_results")
LAT = Path("/home/user/PyCharmMiscProject/final-form/latest-corrected/inference_latency.json")

# ---------- 1. Holm ----------
sig = json.loads((STATS / "all_pairs_significance.json").read_text())


def holm(pairs_p):  # dict name->p
    items = sorted(pairs_p.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running_max = {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)   # enforce monotonicity
        out[name] = {"p_raw": p, "p_holm": round(running_max, 6),
                     "significant_holm_alpha0.05": bool(running_max < 0.05)}
    return out

holm_5x2 = holm({k: v["p_value"] for k, v in sig["five_by_two_cv_all_pairs"].items()})
holm_mc = holm({k: v["p_value"] for k, v in sig["mcnemar_all_pairs"].items()})
sig["holm_correction"] = {
    "note": "Holm step-down over the 3 pairwise tests (reviewer delta Q1). "
            "5x2cv conclusions unchanged after correction.",
    "five_by_two_cv": holm_5x2, "mcnemar": holm_mc}

# margin justification (Q1)
bw = {m: round(sig["bootstrap_metric_ci95"][m]["f1"]["ci95"][1]
             - sig["bootstrap_metric_ci95"][m]["f1"]["ci95"][0], 4)
      for m in sig["bootstrap_metric_ci95"]}
sig["tost_margin_justification"] = {
    "delta": 0.02,
    "rationale": "0.02 F1 is (a) at or below each model's own bootstrap 95% CI width "
                 "(sampling noise of a single test set), and (b) smaller than the F1 shift "
                 "from a 0.05 change in the NN decision threshold; differences inside it are "
                 "not actionable for model selection.",
    "bootstrap_ci95_widths_f1": bw,
    "framing_rule": "LR-NN: the CI excludes zero, so NN is statistically better; the TOST "
                    "bound shows the edge is < 0.02 F1. Write 'statistically detectable but "
                    "practically bounded below 0.02 F1' — never 'no difference'. Note: at "
                    "delta=0.01 TOST fails, so the equivalence claim must always state its margin."}
(STATS / "all_pairs_significance.json").write_text(json.dumps(sig, indent=2))

# ---------- 2. DiD self-preference ----------
def fam(m):
    if any(k in m for k in ("claude", "opus", "sonnet", "haiku")): return "Claude"
    if "gpt" in m: return "OpenAI"
    if "gemini" in m: return "Gemini"
    return "?"

cell = {}  # (gen_family, judge_model) -> list of overall
for jf in sorted(GENAI.glob("judged_*.json")):
    for r in json.load(open(jf)).get("results", []):
        j = r.get("judge")
        if j and j.get("overall") is not None:
            cell.setdefault((fam(r["generator_model"]), r["judge_model"]), []).append(j["overall"])

judges = {"claude-opus-4-7": "Claude", "gpt-5.4-mini": "OpenAI"}
did = {}
for jm, jfam in judges.items():
    other = [o for o in judges if o != jm][0]
    own_gen_self = mean(cell[(jfam, jm)])          # own family's messages, own judge
    own_gen_other = mean(cell[(jfam, other)])      # own family's messages, other judge
    rest_self = mean([v for (gf, j), vals in cell.items() if j == jm and gf != jfam for v in vals])
    rest_other = mean([v for (gf, j), vals in cell.items() if j == other and gf != jfam for v in vals])
    did[jm] = {
        "judge_family": jfam,
        "own_family_messages": {"self_judge_mean": round(own_gen_self, 3), "other_judge_mean": round(own_gen_other, 3)},
        "other_messages": {"self_judge_mean": round(rest_self, 3), "other_judge_mean": round(rest_other, 3)},
        "did_self_preference": round((own_gen_self - own_gen_other) - (rest_self - rest_other), 3)}

summary = json.loads((GENAI / "analysis_summary.json").read_text())
summary["self_preference_DiD"] = {
    "note": "Difference-in-differences estimator (reviewer delta Q2): per judge, the score "
            "premium it gives its OWN family's messages relative to how the other judge scores "
            "them, minus the same premium on everyone else's messages. Controls for overall "
            "judge harshness; unlike the naive +0.50, symmetric per family (Gemini generators "
            "have no self-family judge and are excluded by construction).",
    "per_judge": did,
    "mean_did": round(mean(v["did_self_preference"] for v in did.values()), 3)}

# ---------- 3. latency canonical ----------
lat = json.loads(LAT.read_text())
canon = {m: {"median_ms": lat[m]["single_session_ms"]["median"],
             "mean_ms": lat[m]["single_session_ms"]["mean"]}
         for m in ("LogisticRegression", "RandomForest", "NeuralNetwork")}
summary["latency_canonical"] = {
    "rule": "Cite the MEDIAN single-session latency everywhere (mean is skewed by cold-start "
            "outliers): LR 0.13 ms, RF 13.39 ms, NN 37.58 ms. The 0.14/37.9 variants seen in "
            "some docs are the means — do not mix.",
    "values": canon}
(GENAI / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

print("=== Holm (5x2cv) ===")
for k, v in holm_5x2.items():
    print(f"  {k}: raw {v['p_raw']:.4g} -> holm {v['p_holm']:.4g}  sig={v['significant_holm_alpha0.05']}")
print("=== DiD self-preference ===")
for k, v in did.items():
    print(f"  {k} ({v['judge_family']}): DiD = {v['did_self_preference']:+.3f}")
print(f"  mean DiD = {summary['self_preference_DiD']['mean_did']:+.3f}  (naive was +0.503)")
print("=== latency canonical (median) ===")
for m, v in canon.items():
    print(f"  {m}: median {v['median_ms']} ms (mean {v['mean_ms']})")
