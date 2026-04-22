# Project 3 SHAP Explanations

## Global Readout
- The global table shows which features most strongly influence recoverability predictions on the sampled holdout set.
- This is the bridge between model performance and explainable product behavior.

## Top Global Drivers
- `is_soft_decline`: mean |SHAP| = `0.7378`
- `risk_skip_flag`: mean |SHAP| = `0.5230`
- `response_code`: mean |SHAP| = `0.2658`
- `interchange_estimate_bps`: mean |SHAP| = `0.1411`
- `processor_name`: mean |SHAP| = `0.1388`
- `merchant_vertical`: mean |SHAP| = `0.1373`
- `scheme_ms`: mean |SHAP| = `0.0837`
- `card_country`: mean |SHAP| = `0.0696`
- `archetype`: mean |SHAP| = `0.0622`
- `shipping_country`: mean |SHAP| = `0.0487`
- `latency_3ds_ms`: mean |SHAP| = `0.0371`
- `processor_fee_bps`: mean |SHAP| = `0.0360`
- `ip_country`: mean |SHAP| = `0.0360`
- `scheme_fee_bps`: mean |SHAP| = `0.0354`
- `latency_ms`: mean |SHAP| = `0.0346`

## Local Example Explanations
- Example row `316`: `2025-07-15 16:51:53` `CA` `high-risk-or-orchestrator-b` code `51` amount `$50.72` predicted recoverability `0.589`
  - `is_soft_decline=1` increased the predicted retry recoverability.
  - Response code `51` increased the predicted retry recoverability.
  - `risk_skip_flag=0` increased the predicted retry recoverability.
  - Processor `high-risk-or-orchestrator-b` increased the predicted retry recoverability.
  - Feature `interchange_estimate_bps` with value `211.1` increased the predicted retry recoverability.
- Example row `146`: `2025-07-13 08:36:07` `US` `high-risk-or-orchestrator-b` code `51` amount `$102.40` predicted recoverability `0.573`
  - `is_soft_decline=1` increased the predicted retry recoverability.
  - Response code `51` increased the predicted retry recoverability.
  - `risk_skip_flag=0` increased the predicted retry recoverability.
  - Processor `high-risk-or-orchestrator-b` increased the predicted retry recoverability.
  - Feature `merchant_vertical` with value `ecom` increased the predicted retry recoverability.
- Example row `86`: `2025-07-12 13:47:33` `DE` `global-acquirer-b` code `51` amount `$128.11` predicted recoverability `0.546`
  - `is_soft_decline=1` increased the predicted retry recoverability.
  - Response code `51` increased the predicted retry recoverability.
  - `risk_skip_flag=0` increased the predicted retry recoverability.
  - Feature `merchant_vertical` with value `marketplace` increased the predicted retry recoverability.
  - Processor `global-acquirer-b` increased the predicted retry recoverability.
- Example row `178`: `2025-07-13 16:56:20` `DE` `global-acquirer-b` code `96` amount `$46.88` predicted recoverability `0.545`
  - Response code `96` increased the predicted retry recoverability.
  - `is_soft_decline=1` increased the predicted retry recoverability.
  - `risk_skip_flag=0` increased the predicted retry recoverability.
  - Feature `interchange_estimate_bps` with value `27.5` increased the predicted retry recoverability.
  - Feature `merchant_vertical` with value `ecom` increased the predicted retry recoverability.
- Example row `388`: `2025-07-16 18:59:56` `DE` `high-risk-or-orchestrator-a` code `51` amount `$561.84` predicted recoverability `0.537`
  - `is_soft_decline=1` increased the predicted retry recoverability.
  - `risk_skip_flag=0` increased the predicted retry recoverability.
  - Response code `51` increased the predicted retry recoverability.
  - Processor `high-risk-or-orchestrator-a` increased the predicted retry recoverability.
  - Feature `archetype` with value `high-risk-or-orchestrator` increased the predicted retry recoverability.
