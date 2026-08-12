#!/usr/bin/env python3
"""
judge_matrix.py — cross-provider LLM-as-a-judge over all generators.

Each judge scores every generator's 150 interventions (generator != judge
always; cross-family is the point). Reuses the verbatim thesis judge rubric.
Cost-tracked against the same per-provider ledger/cap as generation. Resumable.

  --mode judge --judge-provider anthropic --judge-model claude-opus-4-7 ...
  --mode matrix   # aggregate all judged_*.json into a generator x judge grid
"""
import argparse, json, os, sys, time
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprovider_phase4 as M

OUT = M.OUT_DIR
GEMINI_PRICE = (0.30, 2.50)  # gemini-flash-latest approx $/1M (Google credit; not in $10 caps)

# ---- google adapter (IPv4 force + timeout, like the thesis scripts) ----
import socket as _s
_o = _s.getaddrinfo
_s.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _o(h, p, _s.AF_INET, t, pr, fl)
_s.setdefaulttimeout(60.0)
_gc = None


def gemini_call(model, prompt, max_out, effort=None):
    global _gc
    from google import genai
    from google.genai import types
    if _gc is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY in the environment.")
        _gc = genai.Client(api_key=key)
    r = _gc.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1, max_output_tokens=max_out,
            thinking_config=types.ThinkingConfig(thinking_budget=0)))
    text = (r.text or "").strip() or None
    um = r.usage_metadata
    return text, getattr(um, "prompt_token_count", 0) or 0, \
           getattr(um, "candidates_token_count", 0) or 0, "stop"


def row_to_ex(row):
    return {"top_3_reasons": row.get("top_3_reasons", []),
            "category_name": row.get("category", "general"),
            "raw_features": {"initial_cart_value": row.get("cart_value", 0.0),
                             "views_before_cart": row.get("views_before_cart", 0),
                             "time_to_cart": row.get("time_to_cart", 0.0)}}


def _save(out_file, results, prov, judge_model, led, final=False):
    sp = led.get(prov, 0.0)
    out_file.write_text(json.dumps({
        "meta": {"judge_provider": prov, "judge_model": judge_model, "role": "judge",
                 "n_judged": sum(1 for r in results if r.get("judge")),
                 "provider_spend_usd": round(sp, 4), "final": final},
        "results": results}, indent=2))


def run_judge(prov, judge_model, effort, max_tokens):
    led = M.ledger_load()
    led.setdefault("google", 0.0)
    out_file = OUT / f"judged_{prov}_{judge_model.replace('.', '-')}.json"
    done, results = set(), []
    if out_file.exists():
        for r in json.load(open(out_file)).get("results", []):
            if r.get("judge"):
                done.add((r["generator_model"], r["test_row_index"])); results.append(r)
    fn = {"anthropic": M.claude_call, "openai": M.gpt_call, "google": gemini_call}[prov]
    cap = M.CAP.get(prov, 9.5)
    print(f"[judge {judge_model}] resume {len(done)} done; spent ${led.get(prov,0):.3f}")
    n = 0
    for gf in sorted(OUT.glob("gen_*.json")):
        d = json.load(open(gf)); gmodel = d["meta"]["generator_model"]
        if gmodel == judge_model:           # never the exact same model
            continue
        for row in d["results"]:
            iv = row.get("intervention")
            if not iv or (gmodel, row["test_row_index"]) in done:
                continue
            if prov in ("anthropic", "openai") and led[prov] >= cap:
                print(f"[judge {judge_model}] CAP ${led[prov]:.3f}>=${cap}. stop.")
                _save(out_file, results, prov, judge_model, led, True); return
            prompt = M.build_judge_prompt(iv, row_to_ex(row))
            time.sleep(M.PACE)
            text, ti, to, stop = M.call_with_retry(
                prov, fn, judge_model, prompt, max_tokens, effort,
                label=f"{judge_model}<{gmodel}#{row['test_row_index']}")
            if text is not None:
                if prov == "google":
                    led["google"] += ti / 1e6 * GEMINI_PRICE[0] + to / 1e6 * GEMINI_PRICE[1]
                    M.LEDGER.write_text(json.dumps(led, indent=2))
                else:
                    M.ledger_add(led, prov, judge_model, "judge", ti, to)
            results.append({"generator_model": gmodel, "judge_model": judge_model,
                            "test_row_index": row["test_row_index"], "cohort": row["cohort"],
                            "intervention": iv, "judge": M.parse_judge_json(text),
                            "judge_in": ti, "judge_out": to})
            n += 1
            if n % 15 == 0:
                _save(out_file, results, prov, judge_model, led)
                print(f"  {judge_model}: judged {n} new | {gmodel} | spend ${led.get(prov,0):.3f}")
    _save(out_file, results, prov, judge_model, led, True)
    ok = sum(1 for r in results if r.get("judge"))
    print(f"[judge {judge_model}] DONE {ok} scored rows -> {out_file}; spent ${led.get(prov,0):.3f}")


def matrix():
    """Aggregate judged_*.json into generator x judge grids + family analysis."""
    dims = ["relevance", "personalization", "urgency", "clarity", "overall"]
    cells = {}  # (gen, judge) -> {dim: [vals]}
    for jf in sorted(OUT.glob("judged_*.json")):
        for r in json.load(open(jf)).get("results", []):
            j = r.get("judge")
            if not j:
                continue
            key = (r["generator_model"], r["judge_model"])
            c = cells.setdefault(key, {d: [] for d in dims})
            for d in dims:
                if j.get(d) is not None:
                    c[d].append(j[d])
    gens = sorted({g for g, _ in cells}); judges = sorted({j for _, j in cells})

    def fam(m):
        if "claude" in m or "opus" in m or "sonnet" in m or "haiku" in m: return "Claude"
        if "gpt" in m: return "OpenAI"
        if "gemini" in m: return "Gemini"
        return "?"

    grid = {f"{g} || {j}": {d: (round(mean(cells[(g, j)][d]), 3) if cells[(g, j)][d] else None) for d in dims}
            for g in gens for j in judges if (g, j) in cells}
    # self vs cross family on overall
    same, cross = [], []
    for (g, j), c in cells.items():
        (same if fam(g) == fam(j) else cross).extend(c["overall"])
    summary = {
        "generators": gens, "judges": judges,
        "overall_grid": {g: {j: (round(mean(cells[(g, j)]["overall"]), 3) if (g, j) in cells and cells[(g, j)]["overall"] else None)
                             for j in judges} for g in gens},
        "self_family_overall_mean": round(mean(same), 3) if same else None,
        "cross_family_overall_mean": round(mean(cross), 3) if cross else None,
        "self_vs_cross_gap": round((mean(same) - mean(cross)), 3) if same and cross else None,
        "per_cell_all_dims": grid,
    }
    (OUT / "judge_matrix_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overall_grid"], indent=2))
    print(f"\nself-family overall mean : {summary['self_family_overall_mean']}")
    print(f"cross-family overall mean: {summary['cross_family_overall_mean']}")
    print(f"self - cross (self-pref) : {summary['self_vs_cross_gap']}")
    print(f"\nwrote {OUT/'judge_matrix_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["judge", "matrix"], required=True)
    ap.add_argument("--judge-provider"); ap.add_argument("--judge-model")
    ap.add_argument("--effort", default="low"); ap.add_argument("--max-tokens", type=int, default=600)
    a = ap.parse_args()
    if a.mode == "matrix":
        matrix()
    else:
        run_judge(a.judge_provider, a.judge_model, a.effort, a.max_tokens)


if __name__ == "__main__":
    main()
