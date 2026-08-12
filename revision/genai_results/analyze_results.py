#!/usr/bin/env python3
"""Comprehensive analysis of the cross-provider Phase-4 results."""
import json, re
from pathlib import Path
from statistics import mean, pstdev

D = Path("/home/user/Downloads/thesis_revision/new/genai_results")
DIMS = ["relevance", "personalization", "urgency", "clarity", "overall"]


def fam(m):
    if any(k in m for k in ("claude", "opus", "sonnet", "haiku")): return "Claude"
    if "gpt" in m: return "OpenAI"
    if "gemini" in m: return "Gemini"
    return "?"


# ---- generators: opener diversity + n ----
gen_stats = {}
for f in sorted(D.glob("gen_*.json")):
    d = json.load(open(f)); m = d["meta"]["generator_model"]
    iv = [x["intervention"] for x in d["results"] if x.get("intervention")]
    op = [" ".join(s.split()[:2]).lower().strip(".,!?'\"") for s in iv]
    dm = re.compile(r"don(?:'|’)?t\s+(miss|let)", re.I)
    gen_stats[m] = {"n": len(iv),
                    "unique_opener_rate": round(len(set(op)) / len(op), 3) if op else None,
                    "dont_miss_rate": round(sum(1 for s in iv if dm.search(s)) / len(iv), 3) if iv else None,
                    "unique_full": round(len(set(iv)) / len(iv), 3) if iv else None}

# ---- judges: per (gen,judge) dimension lists + per-item overall for agreement ----
cells, item_scores = {}, {}   # item_scores[(gen,idx)][judge]=overall
for jf in sorted(D.glob("judged_*.json")):
    for r in json.load(open(jf)).get("results", []):
        j = r.get("judge")
        if not j: continue
        g, jm = r["generator_model"], r["judge_model"]
        c = cells.setdefault((g, jm), {d: [] for d in DIMS})
        for d in DIMS:
            if j.get(d) is not None: c[d].append(j[d])
        if j.get("overall") is not None:
            item_scores.setdefault((g, r["test_row_index"]), {})[jm] = j["overall"]

judges = sorted({jm for _, jm in cells})
gens = sorted({g for g, _ in cells})

# generator ranking: mean overall across judges
gen_overall = {g: round(mean([mean(cells[(g, jm)]["overall"]) for jm in judges if (g, jm) in cells]), 3)
               for g in gens}
# per-dim per generator (avg across judges)
gen_dims = {g: {d: round(mean([mean(cells[(g, jm)][d]) for jm in judges
                               if (g, jm) in cells and cells[(g, jm)][d]]), 3) for d in DIMS}
            for g in gens}

# cross-provider inter-judge agreement (Pearson r + mean abs diff) on shared items
xs, ys = [], []
for k, jd in item_scores.items():
    if len(jd) == 2:
        a, b = [jd[jm] for jm in judges]
        xs.append(a); ys.append(b)
def pearson(a, b):
    n = len(a); ma, mb = mean(a), mean(b)
    cov = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    va = sum((x-ma)**2 for x in a) ** .5; vb = sum((y-mb)**2 for y in b) ** .5
    return cov/(va*vb) if va and vb else None
inter = {"judges": judges, "n_shared_items": len(xs),
         "pearson_r": round(pearson(xs, ys), 3) if xs else None,
         "mean_abs_diff": round(mean(abs(x-y) for x, y in zip(xs, ys)), 3) if xs else None}

# self vs cross family (overall)
same, cross = [], []
for (g, jm), c in cells.items():
    (same if fam(g) == fam(jm) else cross).extend(c["overall"])
family = {"self_family_mean": round(mean(same), 3), "cross_family_mean": round(mean(cross), 3),
          "self_minus_cross": round(mean(same)-mean(cross), 3),
          "n_self": len(same), "n_cross": len(cross)}

out = {"generators": gen_stats, "generator_overall_rank": dict(sorted(gen_overall.items(), key=lambda kv: -kv[1])),
       "generator_dimensions": gen_dims, "inter_judge_agreement": inter, "family_bias": family,
       "overall_grid": {g: {jm: (round(mean(cells[(g, jm)]["overall"]), 3) if (g, jm) in cells else None)
                            for jm in judges} for g in gens}}
(D / "analysis_summary.json").write_text(json.dumps(out, indent=2))

print("GENERATOR RANK (mean overall across judges):")
for g, v in out["generator_overall_rank"].items():
    s = gen_stats[g]
    print(f"  {g:<20} {v:.2f} | opener-div {s['unique_opener_rate']} | urgency {gen_dims[g]['urgency']}")
print(f"\ncross-provider inter-judge: Pearson r = {inter['pearson_r']}, mean|diff| = {inter['mean_abs_diff']} (n={inter['n_shared_items']})")
print(f"family bias self-cross = +{family['self_minus_cross']}  (self {family['self_family_mean']} vs cross {family['cross_family_mean']})")
print(f"\nwrote {D/'analysis_summary.json'}")
