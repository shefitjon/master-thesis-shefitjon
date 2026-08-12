#!/usr/bin/env python3
"""effort_sensitivity.py — reviewer delta Q2.

Re-judge a stratified subsample (~17 per generator = 102) of the interventions
already scored by claude-opus-4-7 at effort=low, using the SAME judge model at
effort=high. Report Pearson r + mean |diff| between low- and high-effort scores.
High agreement defends the low-effort judging design.
"""
import json, os, sys, time
from pathlib import Path
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprovider_phase4 as M
import judge_matrix as J

GENAI = Path("/home/user/Downloads/thesis_revision/new/genai_results")
OUT = GENAI / "effort_sensitivity_check.json"  # v2: full context
JUDGE = "claude-opus-4-7"
PER_GEN = 17

low = json.load(open(GENAI / "judged_anthropic_claude-opus-4-7.json"))["results"]

# join the full session context back from the generator files —
# judged rows store only (generator, index, intervention), and judging
# without context invalidates the comparison (found the hard way).
ctx = {}
for gf in GENAI.glob("gen_*.json"):
    g = json.load(open(gf))
    gm = g["meta"]["generator_model"]
    for row in g["results"]:
        ctx[(gm, row["test_row_index"])] = row
low = [dict(r, **{k: ctx[(r["generator_model"], r["test_row_index"])].get(k)
                  for k in ("cart_value", "views_before_cart", "time_to_cart",
                            "top_3_reasons", "category")})
       for r in low if (r["generator_model"], r["test_row_index"]) in ctx]
by_gen = {}
for r in low:
    if r.get("judge") and r["judge"].get("overall") is not None:
        by_gen.setdefault(r["generator_model"], []).append(r)

sample = []
for g, rows in sorted(by_gen.items()):
    rows = sorted(rows, key=lambda r: r["test_row_index"])
    step = max(1, len(rows) // PER_GEN)
    sample.extend(rows[::step][:PER_GEN])
print(f"subsample: {len(sample)} rows across {len(by_gen)} generators")

led = M.ledger_load()
pairs, results = [], []
done_idx = set()
if OUT.exists():
    prior = json.loads(OUT.read_text())
    results = prior.get("results", [])
    done_idx = {(r["generator_model"], r["test_row_index"]) for r in results}
    print(f"resume: {len(done_idx)} already done")

for i, r in enumerate(sample, 1):
    key = (r["generator_model"], r["test_row_index"])
    if key in done_idx:
        continue
    if led["anthropic"] >= M.CAP["anthropic"]:
        print(f"CAP hit at ${led['anthropic']:.3f} — stopping")
        break
    prompt = M.build_judge_prompt(r["intervention"], J.row_to_ex(r))
    time.sleep(M.PACE)
    text, ti, to, stop = M.call_with_retry(
        "anthropic", M.claude_call, JUDGE, prompt, 2000, "high",
        label=f"high-effort {i}/{len(sample)}")
    if text is not None:
        M.ledger_add(led, "anthropic", JUDGE, "judge_high_effort", ti, to)
    parsed = M.parse_judge_json(text)
    results.append({"generator_model": r["generator_model"],
                    "test_row_index": r["test_row_index"],
                    "overall_low": r["judge"]["overall"],
                    "judge_high": parsed,
                    "overall_high": parsed.get("overall") if parsed else None})
    OUT.write_text(json.dumps({"meta": {"judge_model": JUDGE, "design": "same judge, effort low vs high",
                                        "n_target": len(sample)}, "results": results}, indent=2))
    if i % 10 == 0:
        print(f"  {i}/{len(sample)} | spend ${led['anthropic']:.3f}")

xs = [r["overall_low"] for r in results if r.get("overall_high") is not None]
ys = [r["overall_high"] for r in results if r.get("overall_high") is not None]
def pearson(a, b):
    ma, mb = mean(a), mean(b)
    cov = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    va = sum((x-ma)**2 for x in a) ** .5; vb = sum((y-mb)**2 for y in b) ** .5
    return cov/(va*vb) if va and vb else None
stats = {"n": len(xs), "pearson_r": round(pearson(xs, ys), 3) if len(xs) > 2 else None,
         "mean_abs_diff": round(mean(abs(x-y) for x, y in zip(xs, ys)), 3) if xs else None,
         "mean_low": round(mean(xs), 3) if xs else None,
         "mean_high": round(mean(ys), 3) if ys else None}
data = json.loads(OUT.read_text()); data["statistics"] = stats
OUT.write_text(json.dumps(data, indent=2))
print(f"\nEFFORT SENSITIVITY: n={stats['n']}  r={stats['pearson_r']}  "
      f"mean|diff|={stats['mean_abs_diff']}  low-mean {stats['mean_low']} vs high-mean {stats['mean_high']}")
print(f"spend now: ${led['anthropic']:.3f}")
