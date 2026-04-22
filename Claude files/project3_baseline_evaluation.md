# Project 3 Baseline Evaluation

## Setup
- Model: custom numpy logistic regression baseline
- Reason: bundled runtime did not include `scikit-learn` or `lightgbm`, so this run establishes a credible baseline with no external installs
- Split: temporal 70/15/15 using the original decline timestamp
- Training rows: `8610`
- Validation rows: `1845`
- Test rows: `1845`
- Feature count after preprocessing and one-hot expansion: `386`
- Decision threshold selected on validation set: `0.20`

## Validation Metrics
- AUC: `0.788`
- Log loss: `0.358`
- Brier score: `0.114`
- Precision: `0.329`
- Recall: `0.677`
- F1: `0.443`
- Predicted positive rate: `0.318`

## Test Metrics
- AUC: `0.741`
- Log loss: `0.348`
- Brier score: `0.108`
- Precision: `0.238`
- Recall: `0.573`
- F1: `0.336`
- Predicted positive rate: `0.323`

## Rule Baseline on Test Set
- Baseline rule: retry soft declines by default, boost certain recoverable codes and higher-value transactions, never retry obviously hard declines
- AUC: `0.612`
- Log loss: `0.440`
- Brier score: `0.142`
- Precision: `0.163`
- Recall: `1.000`
- F1: `0.280`

## Interpretation
- This baseline is meant to validate that the richer original-decline feature set carries predictive signal before we invest in a more advanced model stack.
- If the learned model beats the soft-decline rule baseline cleanly, that supports staying with the current dataset for the next stage.
- If the learned model only marginally improves on the rule baseline, the next lever may be better feature engineering or a stronger tree-based model rather than immediately generating more data.

## Strongest Positive Signals
- `is_soft_decline`: `0.406`
- `recurring_type_MISSING`: `0.308`
- `archetype_high-risk-or-orchestrator`: `0.219`
- `response_code_51`: `0.211`
- `response_message_Insufficient funds`: `0.211`
- `scheme_response_code_51`: `0.211`
- `psp_raw_response_51`: `0.211`
- `decline_bucket_issuer_soft`: `0.144`
- `mastercard_advice_code`: `0.138`
- `decline_bucket_processor`: `0.133`
- `processor_name_high-risk-or-orchestrator-a`: `0.120`
- `response_code_91`: `0.113`
- `response_message_Issuer unavailable`: `0.113`
- `scheme_response_code_91`: `0.113`
- `psp_raw_response_91`: `0.113`

## Strongest Negative Signals
- `risk_skip_flag`: `-0.566`
- `decline_bucket_issuer_hard`: `-0.522`
- `is_recurring`: `-0.424`
- `is_mit`: `-0.424`
- `recurring_type_subscription`: `-0.424`
- `archetype_regional-card-specialist`: `-0.201`
- `risk_score`: `-0.166`
- `is_cross_border`: `-0.161`
- `cvv_result_M`: `-0.153`
- `merchant_vertical_saas`: `-0.150`
- `mcc_category_saas`: `-0.150`
- `response_code_54`: `-0.146`
- `response_message_Expired card`: `-0.146`
- `scheme_response_code_54`: `-0.146`
- `psp_raw_response_54`: `-0.146`

## Recommendation
- Proceed to a stronger training implementation next.
- Preferred next model: LightGBM or another tree-based tabular learner once the runtime supports it.
- Keep the current modeling table as the training foundation.
- Revisit data generation only after comparing this baseline with a stronger model and checking segment stability.
