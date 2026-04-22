# Project 3 Drift Report

- Baseline split: training window from `C:\Users\ivana\OneDrive\Everything\Pet projects\AI tools\Payments bots\Project 3\Claude files\project3_modeling_table.csv`
- Reference CSV: `C:\Users\ivana\OneDrive\Everything\Pet projects\AI tools\Payments bots\Project 3\Claude files\project3_modeling_table.csv`
- Features evaluated: `78`

## Highest Numeric Drift (PSI)
- `event_month`: PSI `0.0264`, KS `0.2712`, p-value `0`
- `mastercard_advice_code`: PSI `0.0065`, KS `0.0376`, p-value `0.8572`
- `fx_rate`: PSI `0.0026`, KS `0.0131`, p-value `1`
- `interchange_estimate_bps`: PSI `0.0005`, KS `0.0085`, p-value `0.8566`
- `scheme_ms`: PSI `0.0004`, KS `0.0059`, p-value `0.9944`
- `three_ds_eci`: PSI `0.0004`, KS `0.0094`, p-value `0.9914`
- `latency_ms`: PSI `0.0004`, KS `0.0043`, p-value `1`
- `event_hour`: PSI `0.0004`, KS `0.0062`, p-value `0.9888`
- `risk_score`: PSI `0.0003`, KS `0.0044`, p-value `1`
- `latency_auth_ms`: PSI `0.0003`, KS `0.0041`, p-value `1`

## Highest Categorical Drift (JS Divergence)
- `shipping_country`: JS divergence `0.0290`
- `merchant_country`: JS divergence `0.0288`
- `settlement_currency`: JS divergence `0.0247`
- `processor_name`: JS divergence `0.0130`
- `currency`: JS divergence `0.0001`
- `ip_country`: JS divergence `0.0001`
- `issuer_country`: JS divergence `0.0001`
- `issuer_bank_country`: JS divergence `0.0001`
- `billing_country`: JS divergence `0.0001`
- `card_country`: JS divergence `0.0001`

## Interpretation
- PSI above ~0.2 is a useful yellow flag for numeric features in this synthetic setup.
- Larger JS divergence on categorical features indicates a meaningful shift in the mix of top categories.
- This script is meant for offline monitoring and model-review workflows, not inline inference.
