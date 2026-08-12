#!/usr/bin/env python3
"""
multiprovider_phase4.py
=======================
Cross-provider Phase-4 runner for the thesis revision.

Replicates the thesis Phase-4 methodology — LIME-grounded cart-recovery
*generation* + LLM-as-a-judge rubric *scoring* — across Claude (Anthropic) and
GPT (OpenAI), to complement the existing Gemini runs.

Rules (thesis methodology):
  * Generator and judge are ALWAYS different models; cross-family preferred
    (avoids self-preference bias, Zheng et al. 2023 / Wataoka 2024).
  * Generator + judge prompts are copied VERBATIM from the Gemini v2 runner so
    the cross-provider comparison is fair.

Cost safety:
  * Every call's real token usage is priced and added to a persistent ledger.
  * A per-provider USD cap stops the run (with a checkpoint) before overspend.
  * Resumable: re-running skips sessions already present in the output file.

Modes:
  smoke     1 session per generator candidate; prints real tokens + projected
            150-session cost; validates model access. Tiny spend (pennies).
  generate  full 150-session generation for each (provider, model) gen job.
  judge     score an existing generation file with a cross-family judge model.
"""
import argparse, json, os, re, sys, time
from pathlib import Path
from statistics import mean

HERE = Path("/home/user/PyCharmMiscProject/final-form/latest-corrected")
LIME_FILE = HERE / "lime_examples_stratified_150.json"
OUT_DIR = Path("/home/user/Downloads/thesis_revision/new/genai_results")
LEDGER = Path(os.environ.get("SPEND_LEDGER",
              "/tmp/claude-1000/-home-user-Documents-thesis-wiki/"
              "07615438-7914-4c24-abf1-95a7e33b55eb/scratchpad/spend_ledger.json"))

# ---- Prices, USD per 1M tokens (input, output). Output includes thinking/
#      reasoning tokens. Unknown models -> conservative (10,50). -----------
PRICES = {
    "claude-fable-5": (10.0, 50.0), "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0), "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5.5": (5.0, 30.0), "gpt-5.4": (2.5, 15.0),
    "gpt-5.4-mini": (0.75, 4.5), "gpt-5.4-nano": (0.2, 1.25),
    "gpt-5-mini": (0.25, 2.0),
}
# per-provider USD cap (generators first; reassess judges from leftover)
CAP = {"anthropic": 9.50, "openai": 9.50}
PACE = 1.0  # seconds between calls (polite for new/low-tier accounts)


# ======================= prompts (verbatim from v2) =======================
FEWSHOT_EXEMPLARS = """Example 1 — browse_indecision tone (empathetic, unhurried):
Risk: high time to cart, many views, low browse intensity.
Context: electronics, $312, 9 product views, 7 minutes deliberating.
Intervention: "Ready to pull the trigger on those headphones? We'll hold your cart with free shipping until tomorrow."

Example 2 — price_hesitation tone (value-framing):
Risk: high price variance, high-value cart.
Context: computers, $1,249, 14 products viewed across several price points.
Intervention: "The Dell XPS 15 you chose pairs with 12-month 0% financing, cutting the monthly cost to $104 — apply at checkout."

Example 3 — cart_composition tone (scarcity, time-pressure):
Risk: small cart, fast decision, high conversion signal.
Context: appliances, $89, single item added within 90 seconds.
Intervention: "Only 4 of the Instant Pot 6-quart left in stock at this price — check out in the next 15 minutes to secure yours."

Example 4 — browse_indecision tone (research-confirming, social proof):
Risk: 14 views across 3 categories, user is comparing.
Context: appliances, $240, 14 views, 6 categories explored.
Intervention: "Great pick — the Ninja blender ranks top-three in the last hundred reviews from buyers who compared similar cart sizes."

Example 5 — browse_indecision tone (confirmation + value):
Risk: long time to cart, steady browse intensity.
Context: electronics, $180, 5 views, 4-minute session.
Intervention: "Your electronics are ready to check out — and you have free returns for 30 days if it isn't quite right."
"""


def build_generator_prompt(ex: dict) -> str:
    reasons = ex["top_3_reasons"]
    reason_block = "\n".join(
        f"- ({r['category']}) {r['feature_condition']}  [LIME weight {r['weight']:+.3f}]"
        for r in reasons)
    raw = ex["raw_features"]
    category = ex.get("category_name", "general")
    return f"""You are an e-commerce cart-recovery copywriter. Write a SINGLE short message that will recover this specific user who is about to abandon their cart.

## ABANDONMENT RISK SIGNALS (from LIME on the predictive model)
The top three reasons the model is flagging this user, strongest first:
{reason_block}

## USER CONTEXT
- Category: {category}
- Cart value: ${raw['initial_cart_value']:.2f}
- Cart items: {int(raw['initial_cart_items'])}
- Views before cart: {int(raw['views_before_cart'])}
- Unique products viewed: {int(raw['unique_products_viewed'])}
- Unique categories viewed: {int(raw['unique_categories_viewed'])}
- Time to cart: {raw['time_to_cart']:.0f} seconds
- Browse intensity: {raw['browse_intensity_pre_cart']:.2f} clicks/min
- Avg viewed price: ${raw['avg_viewed_price']:.2f}
- Model predicted P(purchase): {ex['predicted_prob_purchase']:.3f}

## STYLE GUIDE — match the tone to the risk category
Different risk categories call for different tones. Three examples:

{FEWSHOT_EXEMPLARS}

## RULES
- ≤ 25 words, ONE sentence.
- Reference the primary risk signal AND at least one concrete user detail (category, cart value, view count, time).
- DO NOT open with "Don't miss out" or any paraphrase ("don't let this slip", "don't let it go", "don't leave this behind", etc.).
- DO NOT open with "Still deciding / still thinking / still considering / still pondering / taking your time" or any close paraphrase. Earlier runs over-used these and we need variety.
- Your opener must vary across sessions. Rotate among: a direct question ("Ready to..."), a product-specific observation ("Your chosen [category] pairs well with..."), a scarcity note ("Only N of..."), a benefit-led statement ("Free shipping on your $X cart..."), a finance offer ("12-month 0% on..."), a brief compliment on the user's research ("Great pick — the [item]..."), a confirmation ("Your [category] is ready to check out..."), or a simple value phrasing. Pick the opener that fits THIS user's risk signal best; do not default to the same pattern.
- No emoji, no hashtags, no ALL CAPS, no quoted text inside the message.
- Sound like a human marketing copywriter, not an AI.

## OUTPUT FORMAT
Return STRICTLY a single JSON object, no code fences, no surrounding text:
{{"reason_restatement": "<one natural-language sentence that paraphrases the primary LIME risk signal for this user>", "intervention": "<the single-sentence recovery message>"}}
"""


JUDGE_RUBRIC = """Score the following cart-recovery message strictly on the 1–10 scale.

Dimensions:
- RELEVANCE: does it address the stated abandonment risk signal?
- PERSONALIZATION: does it reference concrete user context (category, price, cart)?
- URGENCY: does it include a time-bounded or scarcity element without being aggressive?
- CLARITY: is it concise, grammatical, and actionable?

Return STRICTLY a JSON object with fields
  relevance, personalization, urgency, clarity (all integers 1–10)
  overall (float, the mean)
  comment (one sentence explaining the score)

No code fences, no surrounding prose — just the JSON."""


def build_judge_prompt(intervention: str, ex: dict) -> str:
    reasons = ex["top_3_reasons"]
    primary = reasons[0]["feature_condition"] if reasons else "(n/a)"
    raw = ex["raw_features"]
    return f"""{JUDGE_RUBRIC}

USER CONTEXT:
- Category: {ex.get('category_name', 'general')}
- Cart value: ${raw['initial_cart_value']:.2f}
- Views before cart: {int(raw['views_before_cart'])}
- Time to cart: {raw['time_to_cart']:.0f} seconds

PRIMARY ABANDONMENT RISK SIGNAL:
{primary}

MESSAGE TO SCORE:
{intervention}
"""


# ======================= parsing (verbatim from v2) =======================
def parse_generator_json(text):
    if text is None:
        return None, None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t); t = re.sub(r"\n?```$", "", t)
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None, t.strip() or None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, t.strip() or None
    rs, iv = obj.get("reason_restatement"), obj.get("intervention")
    if isinstance(rs, str): rs = rs.strip() or None
    if isinstance(iv, str): iv = iv.strip() or None
    return rs, iv


def parse_judge_json(text):
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t); t = re.sub(r"\n?```$", "", t)
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in ("relevance", "personalization", "urgency", "clarity"):
        try: out[k] = float(obj.get(k))
        except (TypeError, ValueError): out[k] = None
    try:
        out["overall"] = float(obj["overall"]) if obj.get("overall") is not None else \
            (mean([v for v in out.values() if v is not None]) or None)
    except (TypeError, ValueError):
        out["overall"] = None
    out["comment"] = str(obj.get("comment", "")).strip()
    return out


# ======================= ledger =======================
def ledger_load():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"anthropic": 0.0, "openai": 0.0, "calls": []}


def ledger_add(led, provider, model, role, usage_in, usage_out):
    pin, pout = PRICES.get(model, (10.0, 50.0))
    cost = usage_in / 1e6 * pin + usage_out / 1e6 * pout
    led[provider] = led.get(provider, 0.0) + cost
    led.setdefault("calls", []).append(
        {"provider": provider, "model": model, "role": role,
         "in": usage_in, "out": usage_out, "cost": round(cost, 6)})
    LEDGER.write_text(json.dumps(led, indent=2))
    return cost


# ======================= provider adapters =======================
_ac = None
_oc = None


def claude_call(model, prompt, max_tokens, effort="low"):
    """Returns (text, in_tok, out_tok, stop_reason)."""
    global _ac
    if _ac is None:
        import anthropic
        _ac = anthropic.Anthropic()
    kwargs = dict(model=model, max_tokens=max_tokens,
                  messages=[{"role": "user", "content": prompt}])
    if effort:
        kwargs["output_config"] = {"effort": effort}
    r = _ac.messages.create(**kwargs)
    text = "".join(b.text for b in r.content if b.type == "text").strip() or None
    return text, r.usage.input_tokens, r.usage.output_tokens, r.stop_reason


def gpt_call(model, prompt, max_out, effort="minimal"):
    """Returns (text, in_tok, out_tok, finish_reason). Tries chat.completions
    with reasoning_effort; falls back to the Responses API if rejected."""
    global _oc
    if _oc is None:
        import openai
        _oc = openai.OpenAI()
    try:
        r = _oc.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=effort,
            max_completion_tokens=max_out,
        )
        text = (r.choices[0].message.content or "").strip() or None
        return text, r.usage.prompt_tokens, r.usage.completion_tokens, r.choices[0].finish_reason
    except Exception as e:
        # Fallback: Responses API (newer GPT-5 surface)
        if "reasoning_effort" in str(e) or "max_completion_tokens" in str(e) or "responses" in str(e).lower():
            r = _oc.responses.create(
                model=model, input=prompt,
                reasoning={"effort": effort}, max_output_tokens=max_out)
            text = (getattr(r, "output_text", None) or "").strip() or None
            return text, r.usage.input_tokens, r.usage.output_tokens, "stop"
        raise


_NON_RETRY = ("NotFound", "BadRequest", "Authentication", "PermissionDenied",
              "Unauthorized", "Forbidden", "UnprocessableEntity")
_NON_RETRY_MSG = ("not_found_error", "invalid_request_error", " 404", " 401", " 403", "404", "401", "403")


def call_with_retry(provider, fn, *args, label="", retries=6, **kw):
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kw)
        except Exception as e:
            name, msg = type(e).__name__, str(e)
            if any(k in name for k in _NON_RETRY) or any(c in msg for c in _NON_RETRY_MSG):
                print(f"  [{label}] NON-RETRYABLE {name}: {msg[:180]}", file=sys.stderr)
                return (None, 0, 0, "error")
            is_rate = "429" in msg or "rate" in msg.lower() or "overloaded" in msg.lower()
            if attempt == retries:
                print(f"  [{label}] give up: {name} {msg[:160]}", file=sys.stderr)
                return (None, 0, 0, "error")
            sleep = 6.0 * (2 ** (attempt - 1))
            print(f"  [{label}] attempt {attempt} {'rate-limited' if is_rate else name}; "
                  f"sleep {sleep:.0f}s", file=sys.stderr)
            time.sleep(sleep)


# ======================= sessions =======================
def load_sessions():
    return json.loads(LIME_FILE.read_text())


def smoke():
    led = ledger_load()
    sessions = load_sessions()
    ex = sessions[0]
    prompt = build_generator_prompt(ex)
    print(f"Smoke = generate 1 session ({ex['cohort']}, ${ex['raw_features']['initial_cart_value']:.0f}) per model.\n")
    cands = [
        ("anthropic", "claude-opus-4-8", "low", 600),
        ("anthropic", "claude-sonnet-4-6", "low", 600),
        ("openai", "gpt-5.5", "none", 2000),
        ("openai", "gpt-5.4", "none", 2000),
    ]
    print(f"{'model':<20}{'ok':<4}{'in':>6}{'out':>7}{'$/call':>9}{'$/150':>8}  intervention")
    print("-" * 110)
    for prov, model, eff, mx in cands:
        t0 = time.time()
        if prov == "anthropic":
            text, ti, to, stop = call_with_retry(prov, claude_call, model, prompt, mx, eff, label=model)
        else:
            text, ti, to, stop = call_with_retry(prov, gpt_call, model, prompt, mx, eff, label=model)
        cost = ledger_add(led, prov, model, "smoke_gen", ti, to) if text is not None else 0.0
        _, iv = parse_generator_json(text)
        ok = "Y" if iv else ("R" if stop == "refusal" else "n")
        per150 = cost * 150
        ivs = (iv or (text or "")[:50] or f"<{stop}>")[:55]
        print(f"{model:<20}{ok:<4}{ti:>6}{to:>7}{cost:>9.4f}{per150:>8.2f}  {ivs}  ({time.time()-t0:.1f}s)")
    print(f"\nLedger so far: anthropic ${led['anthropic']:.4f}, openai ${led['openai']:.4f}")


def run_generation(prov, model, effort, max_tokens, n=150):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"gen_{prov}_{model.replace('.', '-')}.json"
    led = ledger_load()
    sessions = load_sessions()[:n]
    done = {}
    if out_file.exists():
        prior = json.loads(out_file.read_text())
        for r in prior.get("results", []):
            if r.get("intervention"):
                done[r["test_row_index"]] = r
    results = list(done.values())
    print(f"[gen {model}] {len(done)} already done; cap ${CAP[prov]:.2f}; "
          f"spent ${led[prov]:.3f}")
    for i, ex in enumerate(sessions, 1):
        if ex["test_row_index"] in done:
            continue
        if led[prov] >= CAP[prov]:
            print(f"[gen {model}] BUDGET CAP hit (${led[prov]:.3f} >= ${CAP[prov]}). Stopping.")
            break
        prompt = build_generator_prompt(ex)
        time.sleep(PACE)
        fn = claude_call if prov == "anthropic" else gpt_call
        text, ti, to, stop = call_with_retry(prov, fn, model, prompt, max_tokens, effort,
                                             label=f"{model} {i}/{len(sessions)}")
        if text is not None:
            ledger_add(led, prov, model, "gen", ti, to)
        rs, iv = parse_generator_json(text)
        row = {"test_row_index": ex["test_row_index"], "cohort": ex["cohort"],
               "actual": ex["actual"], "predicted_prob_purchase": ex["predicted_prob_purchase"],
               "correct": ex.get("correct"), "category": ex.get("category_name"),
               "cart_value": ex["raw_features"]["initial_cart_value"],
               "views_before_cart": ex["raw_features"]["views_before_cart"],
               "time_to_cart": ex["raw_features"]["time_to_cart"],
               "top_3_reasons": ex["top_3_reasons"],
               "reason_restatement": rs, "intervention": iv,
               "gen_stop": stop, "gen_in": ti, "gen_out": to,
               "gen_raw": None if iv else text}
        results.append(row)
        _save_gen(out_file, results, prov, model, led)
        print(f"  {i:03d}/{len(sessions)} [{ex['cohort']:<16}] ${led[prov]:>6.3f} "
              f"| {(iv or '<'+str(stop)+'>')[:60]!r}")
    _save_gen(out_file, results, prov, model, led, final=True)
    ok = sum(1 for r in results if r.get("intervention"))
    print(f"[gen {model}] done: {ok}/{len(sessions)} interventions; spent ${led[prov]:.3f}; -> {out_file}")


def _save_gen(out_file, results, prov, model, led, final=False):
    out_file.write_text(json.dumps({
        "meta": {"provider": prov, "generator_model": model, "role": "generator",
                 "n_complete": sum(1 for r in results if r.get("intervention")),
                 "provider_spend_usd": round(led[prov], 4), "final": final},
        "results": results}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "generate"], required=True)
    ap.add_argument("--provider"); ap.add_argument("--model")
    ap.add_argument("--effort", default="low"); ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    if a.mode == "smoke":
        smoke()
    else:
        run_generation(a.provider, a.model, a.effort, a.max_tokens, a.n)


if __name__ == "__main__":
    main()
