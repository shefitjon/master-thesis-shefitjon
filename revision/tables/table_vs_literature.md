# Table — This work vs. literature (purchase / cart-abandonment prediction)

| Study | Dataset | Feature regime | Reported F1 | Leakage status |
|---|---|---|---|---|
| Abdullah-All-Tanvir et al. (2023) | UCI Online Shoppers | full incl. PageValues | 0.898 | leaky |
| Sakar et al. (2019) | UCI Online Shoppers | full incl. GA fields | 0.58 (MLP) / 0.84 (LSTM) | leaky |
| Baati & Mohsil (2020) | UCI Online Shoppers | metadata only (GA removed) | 0.60 | leaky cols removed |
| Esmeli & Gökçe (2025) | retail clickstream | session-level | 0.89 | leaky (contrast) |
| Esmeli et al. (2021) | retail clickstream | event-bounded (incremental) | 0.55 → 0.78–0.83 | leakage-aware |
| Requena et al. (2020) | Coveo fashion | truncated window | 0.86–0.91 | leakage-aware |
| **This work — LR** | **REES46** | **strictly pre-cart (Temporal Shield)** | **0.467** | **leakage-free** |
| **This work — NN** | **REES46** | **strictly pre-cart (Temporal Shield)** | **0.481** | **leakage-free** |

*The **0.898 → 0.60** drop on UCI when the Google-Analytics fields (PageValues, ExitRates, BounceRates) are removed (Abdullah-All-Tanvir 2023 → Baati & Mohsil 2020) **is the size of the leakage**. Our ~0.47 on REES46 is the leakage-free measurement — in the regime of the leakage-aware antecedents' early-event F1 (Esmeli 2021: 0.55 at event 1), not the leaky 0.85+ figures.*

*Footnote 1: Asfe et al. (2025) report F1 0.825 on REES46, but for a **churn** task (different target), so it is not a direct cart-abandonment comparator.*

*Footnote 2: Jain (2025, arXiv:2506.17543) is the closest same-dataset-family comparator (mkechinov Electronics Store, the sibling release to REES46; F1 0.44–0.55) but does not apply a temporal cutoff at the cart-add moment and reports internally inconsistent AUCs. **To our knowledge, this work is the first leakage-free measurement at the cart-add moment on REES46** — "leakage-free" meaning every feature is computed strictly from events before t⋆, the first cart addition.*
