# Bregu_CartRecoveryAb

A Magento 2 + Hyvä module that runs a **live A/B test of AI-generated cart-recovery
interventions** — the production bridge for the thesis's final future-work item. It scores a
live cart session's abandonment risk with the **thesis's own Logistic Regression
coefficients**, deterministically assigns the session to control or treatment, shows a
personalized recovery message (live Gemini or a pre-generated bank) to high-risk treatment
sessions, and logs outcomes so the real revenue lift can be measured with the same statistical
test as Chapter 4.

> **Status: complete (phases 1–5).** This is a research artifact, not a hardened production
> extension — it compiles, installs, and runs end-to-end on a Magento 2.4.x + Hyvä store. The
> [honest limitations](#honest-limitations) section says exactly what a real rollout would still
> need to harden.

## Why this exists
The thesis predicts, explains, and generates interventions offline, then evaluates them with an
LLM-as-a-judge. The one thing offline work cannot answer is whether the interventions actually
recover carts in a real store. This module answers that: control sees the normal store,
treatment sees the intervention, and the difference in conversion and order value is the lift.

## How it reuses the thesis (the honest tie-in)
- **Risk score** — `RiskScorer` computes `P(abandon) = 1 − sigmoid(intercept + Σ coef·z)` using
  the exact Logistic Regression coefficients exported by the thesis
  (`artifacts/feature_importance.json` + the scaler's mean/scale from `model_lr.pkl`). The model
  predicts `P(purchase)`, so abandonment is its complement. Only the three live-observable
  features carry signal — `views_before_cart`, `total_events_before_cart`,
  `browse_intensity_pre_cart` — and every other feature sits at its training mean (z = 0), exactly
  as the defense live-demo widget does. The constants in `Model/RiskScorer.php` are the numbers
  the thesis defends, not re-fit values.
- **Intervention** — `InterventionProvider` builds the same Phase-4 generator prompt, grounded in
  the session's dominant pre-cart signal, and calls Gemini (`gemini-2.5-flash-lite`,
  `thinkingBudget = 0`) just like `src/phase4_generate_and_judge.py`. If the call fails or the
  source is set to pre-generated, it falls back to a per-signal message bank.
- **Evaluation** — `bregu:ab:report` mirrors Ch.4: per-bucket conversion + revenue and a 2×2
  chi-square test (Yates-corrected) with a p-value — the same significance-test discipline used
  for the model comparison.

## Architecture
```
registration.php · composer.json · README.md
etc/module.xml · etc/acl.xml · etc/config.xml · etc/di.xml
etc/db_schema.xml · etc/db_schema_whitelist.json   event table (exposures + outcomes)
etc/adminhtml/system.xml                           admin configuration
etc/frontend/events.xml · etc/frontend/routes.xml  observers + the track endpoint

Model/
  Config.php                  typed reader for every setting (decrypts the API key)
  RiskScorer.php              LR sigmoid from the exported thesis coefficients
  AbAssigner.php              deterministic control/treatment bucket (crc32 of quote id)
  SessionSignals.php          accumulates pre-cart view/event counts in the checkout session
  InterventionProvider.php    Gemini live, or the message bank, behind the source switch
  MessageBank.php             per-signal fallback copy
  EventLogger.php             insert/update/aggregate the event table (ResourceConnection)
  ChiSquareTest.php           2×2 chi-square with Yates correction + p-value
  Config/Source/*.php         admin dropdowns (intervention source, trigger)
Service/GeminiClient.php      server-side REST call; reads the encrypted key
Observer/
  TrackProductView.php        counts a pre-cart view
  FreezeOnCartAdd.php         freezes the signals at first cart add (browse intensity)
  RecordConversion.php        attributes the order back to its bucket
  RegisterModuleForHyvaConfig.php   registers the module in hyva-themes.json
ViewModel/RecoveryWidget.php  orchestrates score → bucket → message for the template
Controller/Ab/Track.php       POST endpoint: exposure / impression / dismiss
Console/Command/AbReport.php  bregu:ab:report
view/frontend/
  layout/checkout_cart_index.xml
  templates/recovery.phtml    Hyvä + Tailwind + CSP-safe Alpine banner
  tailwind/tailwind.config.js  so the banner's classes survive build-prod
i18n/tr_TR.csv · i18n/de_DE.csv
```

## Data flow
1. `TrackProductView` counts each pre-cart product view into the checkout session (no PII).
2. `FreezeOnCartAdd` freezes the counts at the first cart add and computes browse intensity
   (`views ÷ minutes`).
3. On the cart page, `RecoveryWidget` reads the frozen signals, `RiskScorer` computes
   `P(abandon)`, and `AbAssigner` buckets the session on its quote id.
4. The template renders a tiny Alpine component for **every** eligible session (both buckets) that
   beacons an `exposure` row — this is the denominator for the experiment. Only
   **treatment + risk ≥ threshold** sessions also render the visible banner, whose copy comes from
   `InterventionProvider`.
5. The banner fires by the configured trigger (cart load / exit intent / idle), beacons an
   `impression` on first show and a `dismiss` if closed, all to `Controller/Ab/Track` via
   `navigator.sendBeacon`.
6. `RecordConversion` (`checkout_submit_all_after`) marks the bucket's row converted with the
   order id and grand total.
7. `bin/magento bregu:ab:report` prints per-bucket exposures / impressions / conversions / revenue
   and the chi-square p-value.

## Install
This module lives in a subfolder of the thesis repo. To use it in a Magento 2.4.x store, either:

1. Copy/symlink it into the store:
   ```bash
   mkdir -p app/code/Bregu/CartRecoveryAb
   cp -r magento-ab-module/* app/code/Bregu/CartRecoveryAb/
   ```
2. Or add it as a Composer path repository pointing at `magento-ab-module/`.

Then:
```bash
bin/magento module:enable Bregu_CartRecoveryAb
bin/magento setup:upgrade            # creates the bregu_cart_recovery_event table
bin/magento setup:di:compile         # production mode
bin/magento hyva:config:generate     # registers the module's Tailwind config with Hyvä
# rebuild the theme's Tailwind so the banner classes survive prod:
# cd app/design/frontend/<Vendor>/<theme>/web/tailwind && npm run build-prod
```
`hyva:config:generate` is **not optional** — without it the banner renders unstyled in a prod
build. See the design note in `patterns/hyva-tailwind-custom-module` of the Magento wiki.

## Configuration
**Stores → Configuration → Bregu → Cart Recovery A/B → General**

| Field | Meaning | Default |
|---|---|---|
| Enabled | Master on/off | No |
| Treatment Share (%) | Fraction of eligible sessions in the treatment group | 50 |
| Risk Threshold | Show the intervention when P(abandon) ≥ this (0–1) | 0.45 |
| Intervention Source | `Live Gemini` or `Pre-generated bank` | Live Gemini |
| Trigger | When to show it: on cart load / exit intent / after idle | Exit intent |
| Gemini API Key | Encrypted; used only when source is Live | (empty) |

### Where the Gemini key comes from
Create a key at <https://aistudio.google.com/app/apikey>, paste it into the **Gemini API Key**
field (only visible when Intervention Source is Live). Magento stores it with its encrypted
backend model — it is never written to code or committed. With no key, or on any API error, the
module silently falls back to the pre-generated message bank, so the A/B test still runs.

## Reporting
```bash
bin/magento bregu:ab:report
```
```
control    exposures=412 impressions=0   conversions=109 rate=26.46% revenue=18230.40
treatment  exposures=398 impressions=121 conversions=131 rate=32.91% revenue=22944.10

chi-square = 4.1287, p = 0.0422  (significant at 0.05)
```
(Illustrative numbers.) The chi-square tests conversion rate between buckets; the p-value uses a
Yates-corrected 2×2 statistic with an Abramowitz–Stegun `erfc` for the 1-dof survival function.

## Privacy
The Gemini prompt carries only **behavioural aggregates** — event counts, browse intensity, cart
value and item count. No name, email, or address ever leaves the store. The banner is dismissible,
control sees the unmodified store (non-deceptive), and the event table keys on the quote id, not on
any personal identifier. Add the experiment to the store's KVKK/GDPR notice before going live.

## Honest limitations
A research artifact, scoped to demonstrate the loop end-to-end. A production rollout would still:
- **Live signals approximate the offline features.** `views_before_cart` and
  `total_events_before_cart` are both incremented from product-view events; the offline pipeline
  distinguishes view events from all event types. The three driving features match; the long tail
  does not, and they sit at their training mean.
- **Generate asynchronously.** Live Gemini is called during cart-page render (≈1–2 s, 10 s
  timeout, bank fallback). Production should pre-generate per signal or move the call off the render
  path. `Pre-generated bank` already does this.
- **Harden bucketing across devices.** Bucketing is per quote id — stable within a cart, but a
  logged-in customer on a new device gets a fresh quote. A customer-id salt would fix attribution.
- **Add coverage.** No automated tests ship here; the thesis pipeline carries the validated model.

## Conventions
Comments follow the host project's house style: structural / PHPDoc only, no explanatory prose in
code — all reasoning lives in this README. PHP uses fully-qualified class names with promoted
readonly constructor properties (no `use` statements), per the Magento-wiki backend rule. Strings
are English at source with translations in `i18n/<locale>.csv`. The frontend follows Hyvä's
strict-CSP Alpine rules (dataset-driven config, no function arguments in directives, inline script
emitted through `$secureRenderer`) and the Tailwind-custom-module pipeline.
