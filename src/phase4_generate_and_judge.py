import socket as _socket
_orig_getaddrinfo = _socket.getaddrinfo

def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, _socket.AF_INET, type, proto, flags)
_socket.getaddrinfo = _ipv4_getaddrinfo
_socket.setdefaulttimeout(60.0)
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from google import genai
from google.genai import types
HERE = Path(__file__).resolve().parents[1] / 'artifacts'
LIME_FILE = HERE / 'lime_examples.json'
CAT_FILE = HERE / 'category_mapping.json'
OUT_FILE = HERE / 'gemini_live_results.json'
MODEL_GEN = 'gemini-2.5-flash-lite'
MODEL_JUDGE = 'gemini-flash-latest'
MAX_RETRIES = 6
BACKOFF_BASE = 8.0
PACE_BETWEEN_CALLS = 6.5

def get_client():
    key = os.environ['GEMINI_API_KEY']
    return genai.Client(api_key=key)

def call_with_retry(client, model, prompt, *, temperature, max_tokens, call_label):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens, thinking_config=types.ThinkingConfig(thinking_budget=0)))
            text = (resp.text or '').strip()
            if not text:
                raise RuntimeError('empty response')
            return text
        except Exception as e:
            msg = str(e)
            is_429 = '429' in msg or 'RESOURCE_EXHAUSTED' in msg
            if attempt == MAX_RETRIES:
                print(f'  [{call_label}] giving up after {attempt} attempts: {type(e).__name__} {msg[:150]}', file=sys.stderr)
                return None
            sleep = BACKOFF_BASE * 2 ** (attempt - 1)
            reason = 'rate-limited' if is_429 else type(e).__name__
            print(f'  [{call_label}] attempt {attempt} {reason}; sleeping {sleep}s', file=sys.stderr)
            time.sleep(sleep)
    return None

def build_generator_prompt(ex, category_name, lime_reason_summary):
    raw = ex['raw_features']
    return f"You are an e-commerce cart-recovery copywriter.\n\nUSER SESSION (right at the moment they added the item to cart):\n- Category: {category_name}\n- Cart value: ${raw['initial_cart_value']:.2f}\n- Cart items: {int(raw['initial_cart_items'])}\n- Views before cart: {int(raw['views_before_cart'])}\n- Unique products viewed: {int(raw['unique_products_viewed'])}\n- Unique categories viewed: {int(raw['unique_categories_viewed'])}\n- Time to cart: {raw['time_to_cart']:.0f} seconds\n- Browse intensity: {raw['browse_intensity_pre_cart']:.2f} clicks/min\n- Avg viewed price: ${raw['avg_viewed_price']:.2f}\n- Model's predicted P(purchase): {ex['predicted_prob_purchase']:.3f}\n\nABANDONMENT RISK SIGNAL (from LIME on the predictive model):\n{lime_reason_summary}\n\nTASK: write ONE persuasive single-sentence pop-up message (max 25 words) that\n- addresses this user's specific risk signal,\n- is concrete to their category and cart,\n- creates a gentle urgency,\n- reads as natural human copy — no hashtags, no emoji, no ALL CAPS, no quotes.\n\nOutput only the single sentence, nothing else."
JUDGE_RUBRIC = 'Score the following cart-recovery message strictly on the 1–10 scale.\n\nDimensions:\n- RELEVANCE: does it address the stated abandonment risk signal?\n- PERSONALIZATION: does it reference concrete user context (category, price, cart)?\n- URGENCY: does it include a time-bounded or scarcity element without being aggressive?\n- CLARITY: is it concise, grammatical, and actionable?\n\nReturn STRICTLY a JSON object with fields\n  relevance, personalization, urgency, clarity (all integers 1–10)\n  overall (float, the mean)\n  comment (one sentence explaining the score)\n\nNo code fences, no surrounding prose — just the JSON.'

def build_judge_prompt(message, ex, category_name, lime_reason_summary):
    raw = ex['raw_features']
    return f'{JUDGE_RUBRIC}\n\nUSER CONTEXT:\n- Category: {category_name}\n- Cart value: ${raw['initial_cart_value']:.2f}\n- Views before cart: {int(raw['views_before_cart'])}\n- Time to cart: {raw['time_to_cart']:.0f} seconds\n\nABANDONMENT RISK SIGNAL:\n{lime_reason_summary}\n\nMESSAGE TO SCORE:\n{message}\n'

def parse_judge_json(text):
    t = text.strip()
    if t.startswith('```'):
        t = re.sub('^```[a-zA-Z]*\\n?', '', t)
        t = re.sub('\\n?```$', '', t)
    m = re.search('\\{.*\\}', t, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in ('relevance', 'personalization', 'urgency', 'clarity'):
        try:
            out[k] = float(obj.get(k, 0))
        except (TypeError, ValueError):
            out[k] = None
    try:
        if obj.get('overall') is not None:
            out['overall'] = float(obj['overall'])
        else:
            present = [v for v in (out['relevance'], out['personalization'], out['urgency'], out['clarity']) if v is not None]
            out['overall'] = mean(present) if present else None
    except (TypeError, ValueError):
        out['overall'] = None
    out['comment'] = str(obj.get('comment', '')).strip()
    return out

def summarise_lime(ex):
    lines = []
    for r in ex['top_reasons'][:3]:
        direction = '↑ purchase' if r['weight'] > 0 else '↓ purchase'
        lines.append(f'- {r['feature_condition']}  ({direction}, weight {r['weight']:+.3f})')
    return '\n'.join(lines)

def main():
    if not LIME_FILE.exists():
        print('Missing lime_examples.json — run extract_shap_lime.py first.', file=sys.stderr)
        sys.exit(1)
    examples = json.loads(LIME_FILE.read_text())
    cat_map = json.loads(CAT_FILE.read_text())
    cat_map = {int(k): v for k, v in cat_map.items()}
    client = get_client()
    api_calls = 0
    results = []
    failures = {'generator': 0, 'judge': 0}
    print(f'Processing {len(examples)} sessions through Gemini 2.5 Flash (generator + judge)…')
    for i, ex in enumerate(examples, 1):
        cat_id = int(ex['raw_features']['main_category'])
        cat_name = cat_map.get(cat_id, 'unknown')
        lime_summary = summarise_lime(ex)
        gen_prompt = build_generator_prompt(ex, cat_name, lime_summary)
        time.sleep(PACE_BETWEEN_CALLS)
        message = call_with_retry(client, MODEL_GEN, gen_prompt, temperature=0.7, max_tokens=80, call_label=f'gen {i}/{len(examples)}')
        api_calls += 1
        if message is None:
            failures['generator'] += 1
            results.append({'test_row_index': ex['test_row_index'], 'cohort': ex['cohort'], 'category': cat_name, 'lime_summary': lime_summary, 'intervention': None, 'judge': None, 'error': 'generator_failed'})
            continue
        judge_prompt = build_judge_prompt(message, ex, cat_name, lime_summary)
        time.sleep(PACE_BETWEEN_CALLS)
        judge_text = call_with_retry(client, MODEL_JUDGE, judge_prompt, temperature=0.1, max_tokens=400, call_label=f'judge {i}/{len(examples)}')
        api_calls += 1
        judge_parsed = parse_judge_json(judge_text) if judge_text else None
        if judge_parsed is None:
            failures['judge'] += 1
        results.append({'test_row_index': ex['test_row_index'], 'cohort': ex['cohort'], 'actual': ex['actual'], 'predicted_prob_purchase': ex['predicted_prob_purchase'], 'category': cat_name, 'cart_value': ex['raw_features']['initial_cart_value'], 'views_before_cart': ex['raw_features']['views_before_cart'], 'time_to_cart': ex['raw_features']['time_to_cart'], 'lime_summary': lime_summary, 'intervention': message, 'judge_raw': judge_text, 'judge': judge_parsed})
        overall = judge_parsed['overall'] if judge_parsed else None
        overall_s = f'{overall:.2f}' if overall is not None else 'N/A'
        print(f'  {i:02d}. {cat_name:<12} ${ex['raw_features']['initial_cart_value']:>7.2f} | score {overall_s} | {message[:70]!r}')
    scores = [r['judge']['overall'] for r in results if r.get('judge') and r['judge'].get('overall') is not None]
    per_dim = {k: [r['judge'][k] for r in results if r.get('judge') and r['judge'].get(k) is not None] for k in ('relevance', 'personalization', 'urgency', 'clarity')}
    stats = {'api_calls': api_calls, 'n_sessions': len(examples), 'n_successful_interventions': sum((1 for r in results if r.get('intervention'))), 'n_successful_judge': sum((1 for r in results if r.get('judge'))), 'generator_failures': failures['generator'], 'judge_parse_failures': failures['judge'], 'overall_score': {'mean': mean(scores) if scores else None, 'std': pstdev(scores) if len(scores) >= 2 else None, 'min': min(scores) if scores else None, 'max': max(scores) if scores else None, 'n': len(scores)}, 'dimension_means': {k: mean(v) if v else None for k, v in per_dim.items()}, 'unique_interventions': len({r['intervention'] for r in results if r.get('intervention')}), 'models': {'generator': MODEL_GEN, 'judge': MODEL_JUDGE}}
    OUT_FILE.write_text(json.dumps({'statistics': stats, 'results': results}, indent=2))
    print('\n' + '=' * 60)
    print(f'Saved {OUT_FILE}')
    print(f'api_calls = {api_calls}, successful generations = {stats['n_successful_interventions']}/{stats['n_sessions']}, successful judges = {stats['n_successful_judge']}/{stats['n_sessions']}')
    if scores:
        print(f'overall: mean={stats['overall_score']['mean']:.2f} std={stats['overall_score']['std']:.2f} min={stats['overall_score']['min']:.1f} max={stats['overall_score']['max']:.1f}')
        print(f'unique interventions: {stats['unique_interventions']}/{stats['n_successful_interventions']}')
        for k, v in stats['dimension_means'].items():
            print(f'  {k:<17} mean {v:.2f}')
if __name__ == '__main__':
    main()
