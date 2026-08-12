# Phase 4 — cross-provider GenAI results (revision round)

Real LLM generation + LLM-as-a-judge scoring across **three model families**
(Gemini, Claude, OpenAI), built to answer the jury's "does generator/judge being
the *same vs different family* change the results?" question (Ek-1 Q17) and to
give the thesis a genuine cross-provider basis instead of Gemini-only.

Run 2026-06-28. All generation + judging uses **real paid API calls**, not
templates. Methodology (prompts, rubric, 150 stratified sessions) is **identical**
to the thesis Gemini Phase-4 run — the prompt builders are copied verbatim — so
the comparison is fair.

## Hard rule enforced
**Generator and judge are never the same model** (cross-family preferred), to
avoid self-preference bias (Zheng et al. 2023 / Wataoka 2024 "CALM").

## Models

| Role | Family | Models |
|---|---|---|
| Generator | Gemini | `gemini-2.5-pro`, `gemini-flash-lite` *(from the original thesis run)* |
| Generator | Claude | `claude-opus-4-8`, `claude-sonnet-4-6` |
| Generator | OpenAI | `gpt-5.5`, `gpt-5.4` |
| Judge | Claude | `claude-opus-4-7` |
| Judge | OpenAI | `gpt-5.4-mini` |
| Judge | Gemini | `gemini-flash-latest` — **PENDING** (Google project spend cap hit; add when raised) |

6 generators × 2 judges = **1,800 scores** (3rd judge pending → would make it 2,700).
Each judge scored all 150 interventions from all 6 generators.

## Headline results

**Generator quality (mean overall 1–10, averaged across the 2 judges):**

| Rank | Generator | Overall | Opener diversity | Urgency |
|---|---|---|---|---|
| 1 | GPT-5.5 | 6.99 | 0.17 | 3.07 |
| 2 | GPT-5.4 | 6.93 | 0.09 | 3.19 |
| 3 | Claude Sonnet 4.6 | 6.89 | 0.44 | 3.59 |
| 4 | Claude Opus 4.8 | 6.82 | 0.42 | 3.11 |
| 5 | Gemini Flash-Lite | 6.10 | 0.39 | 2.39 |
| 6 | Gemini 2.5 Pro | 5.89 | 0.50 | 2.47 |

**Self-preference / family bias:** judges score their **own family's** generators
higher — self-family mean **6.94** vs cross-family **6.44** → **+0.50 gap**. Real
but modest. (Directly answers Q17.)

**Cross-provider inter-judge agreement:** Claude vs OpenAI judge — **Pearson r =
0.64**, mean |diff| = 0.55 (n=900). Notably *higher* agreement than the thesis's
within-Gemini inter-judge r = 0.474 — two different-family judges agree *more*
than two same-family ones did.

**Three findings worth putting in the thesis:**
1. **GPT is highest-scoring but most formulaic** — GPT-5.4/5.5 top the rubric yet
   have the lowest opener diversity (0.09–0.17 vs Claude's 0.42–0.44). Rubric
   quality ≠ message variety.
2. **The thesis's own Gemini generations rank lowest** under external cross-family
   judges (5.89–6.10), despite Gemini having the highest opener diversity.
3. **Urgency is the weakest dimension for every generator** (2.4–3.6), corroborating
   the thesis's existing urgency finding (~1.7–3.2). "Don't miss" opener rate = 0.00
   for all (the anti-repetition rule held).

## Spend (real)
Claude **$6.09** (gen $2.20 + judge $3.89) · OpenAI **$2.23** (gen $1.52 + judge $0.71)
· Gemini judge $0 (pending). Well inside the $10 + $10 budget — **no top-up used.**

## Files
- `gen_<provider>_<model>.json` — 6 generators, 150 interventions each (raw text,
  reason restatement, token usage, cohort).
- `judged_<provider>_<model>.json` — 2 judges, 900 scored rows each
  (per-dimension scores + comment per intervention).
- `judge_matrix_summary.json` — generator×judge grid + family analysis.
- `analysis_summary.json` — generator ranking, per-dimension means, inter-judge
  agreement, diversity, family bias (the numbers above).
- `figures/` *(sibling dir)* — the verified-real thesis figures.
- Scripts (reproducible; read API keys from env — **no keys embedded**):
  `multiprovider_phase4.py` (generation), `judge_matrix.py` (judging + matrix),
  `analyze_results.py` (analysis).

## Reproduce
```bash
export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  GEMINI_API_KEY=...
python multiprovider_phase4.py --mode generate --provider anthropic --model claude-opus-4-8 --effort low --max-tokens 600
python judge_matrix.py --mode judge --judge-provider openai --judge-model gpt-5.4-mini --effort low --max-tokens 1000
python judge_matrix.py --mode matrix
python analyze_results.py
```
Source data: 150 stratified sessions in
`PyCharmMiscProject/final-form/latest-corrected/lime_examples_stratified_150.json`.

## Caveats
- **Gemini judge pending** — Google project monthly spend cap was hit (429); add
  `gemini-flash-latest` as the 3rd judge once the cap is raised to complete the 3×6 grid.
- Generation effort: Claude `effort=low` (thinking off), OpenAI `reasoning_effort=none`
  — appropriate for short copywriting, keeps cost/latency down.
- `gpt-5.4`/`gpt-5.5` reject `reasoning_effort=minimal`; `none` used. `claude-fable-5`
  is not available on this account (API: "use Opus 4.8").

## Post-review additions (2026-07-07)

- **DiD self-preference (replaces the naive number as the headline):** the naive
  own-family gap (+0.50) is confounded by genuine quality differences and is
  asymmetric (no Gemini judge yet). Difference-in-differences — each judge's premium
  on its own family's messages vs the other judge, minus the same premium on everyone
  else's — gives **+0.08 (Claude judge) / +0.10 (GPT judge), mean +0.09**. Cite both:
  naive +0.50, controlled ≈ +0.09 (small). In `analysis_summary.json` → `self_preference_DiD`.
- **Judge effort-sensitivity check** (`effort_sensitivity_check.json`): 102-message
  stratified subsample re-judged by claude-opus-4-7 at effort=high vs the matrix's
  effort=low: **Pearson r = 0.96, mean |diff| = 0.17** (means 6.47 vs 6.48; 48/102
  identical). Low-effort judging is validated.
- Method caution: a first attempt sampled from the slim `judged_*` rows and fed the
  judge an empty session context → r = 0.61 and a −2.6 calibration collapse (the judge
  punished "fabricated" details that were actually real). **Judge context completeness
  is load-bearing**; the main 1,800-score matrix always used full context (rows from
  `gen_*` files) and is unaffected.
