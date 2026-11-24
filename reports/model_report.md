# Model report

2000 leads, base conversion rate 22.8%. Held-out test set: 500 leads (stratified 25 %). CV: 5-fold stratified.

## Model comparison

Top-10 % columns: held-out test set (50 leads) / out-of-fold on all leads (200 leads).

| Model | Test ROC-AUC | CV ROC-AUC | Top-10 % conversion | Lift vs. base | Lift vs. rule |
|---|---|---|---|---|---|
| Logistic regression | 0.696 | 0.718 ± 0.030 | 52.0% / 54.5% | 2.28× / 2.40× | 1.00× / 1.04× |
| Gradient boosting | 0.700 | 0.688 ± 0.011 | 50.0% / 49.5% | 2.19× / 2.18× | 0.96× / 0.94× |
| Rule: demo requested, then most website visits | 0.635 | | 52.0% / 52.5% | 2.28× / 2.31× | 1.00× |

Logistic regression chosen: cross-validated AUC gap gradient boosting minus logistic regression is -0.030; gradient boosting is only preferred above +0.01, because the simpler model is easier to explain and more stable.

## Leakage check

Conversion rate by `deal_stage`:

| deal_stage | leads | conversion rate |
|---|---|---|
| Contract | 345 | 100.0% |
| Negotiation | 89 | 100.0% |
| Proposal | 167 | 12.6% |
| First contact | 478 | 0.0% |
| Lost | 544 | 0.0% |
| Qualified | 377 | 0.0% |

Single-column ROC-AUC (out-of-fold):

| column | AUC |
|---|---|
| deal_stage | 0.997 |
| demo_requested | 0.623 |
| source | 0.609 |
| website_visits | 0.592 |
| company_size | 0.580 |
| distance_to_nearest_customer_km | 0.520 |
| industry | 0.507 |
| region | 0.501 |
| city | 0.501 |

Flagged (AUC > 0.85): deal_stage. Logistic regression CV AUC with `deal_stage`: 0.999, without: 0.718.

`deal_stage` is only known late in the sales cycle (`Contract`, `Lost`, `Negotiation` are outcomes or near-outcomes), so a model that uses it would look excellent offline and be useless for new leads. It is excluded from training and from the app.

Excluded columns:

- `deal_stage`: Leakage: set late in the sales process, it almost reveals the outcome.
- `region`: Some regions are tiny (Ulm: 9 leads); geography enters via distance to the nearest customer.
- `city`: Same as region.
- `lat`: Raw coordinates; geography enters via distance to the nearest customer.
- `lon`: Raw coordinates; geography enters via distance to the nearest customer.
- `lead_id`: Identifier.
- `company`: Identifier.

## Regions (flagged below 50 leads)

| region | leads | won | conversion | 95 % CI | |
|---|---|---|---|---|---|
| Stuttgart | 422 | 104 | 24.6% | 21% to 29% |  |
| Esslingen | 226 | 51 | 22.6% | 18% to 28% |  |
| Ludwigsburg | 187 | 48 | 25.7% | 20% to 32% |  |
| Böblingen | 180 | 36 | 20.0% | 15% to 26% |  |
| Karlsruhe | 172 | 31 | 18.0% | 13% to 24% |  |
| Rems-Murr | 162 | 41 | 25.3% | 19% to 32% |  |
| Heilbronn | 160 | 34 | 21.2% | 16% to 28% |  |
| Reutlingen | 141 | 30 | 21.3% | 15% to 29% |  |
| Tübingen | 126 | 27 | 21.4% | 15% to 29% |  |
| Göppingen | 124 | 28 | 22.6% | 16% to 31% |  |
| Pforzheim | 91 | 18 | 19.8% | 13% to 29% |  |
| Ulm | 9 | 7 | 77.8% | 45% to 94% | ⚠ too few leads |

Adding `region` as a feature: CV AUC 0.714 (without: 0.718). Region stays out of the model, so a lucky streak in a tiny region does not inflate its scores.

## Data checks

- missing_values: 0
- duplicate_lead_ids: 0
- duplicate_companies: 0
- distance_column_max_gap_km: 0.05
