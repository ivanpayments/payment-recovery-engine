# Project 3 Baseline Evaluation V2

## Setup
- Model: custom numpy logistic regression baseline with target encoding for categorical features
- Why this version: less noisy than wide one-hot encoding, better fit for a self-contained runtime, and easier to inspect for segment stability
- Split: temporal 70/15/15 using original decline timestamp
- Training rows: `8610`
- Validation rows: `1845`
- Test rows: `1845`
- Feature count after preprocessing: `101`
- Decision threshold selected on validation set: `0.19`

## Validation Metrics
- AUC: `0.774`
- Log loss: `0.375`
- Brier score: `0.117`
- Precision: `0.292`
- Recall: `0.761`
- F1: `0.423`

## Test Metrics
- AUC: `0.729`
- Log loss: `0.358`
- Brier score: `0.109`
- Precision: `0.221`
- Recall: `0.657`
- F1: `0.331`

## Rule Baseline on Test Set
- AUC: `0.612`
- Log loss: `0.440`
- Brier score: `0.142`
- Precision: `0.163`
- Recall: `1.000`
- F1: `0.280`

## Readout
- This V2 baseline is meant to be more realistic and less overfit than the first wide one-hot pass.
- If it still beats the rules baseline clearly, that is a good sign that the dataset is sufficient for the next modeling stage.

## Strongest Positive Features
- `is_soft_decline`: `0.524`
- `response_code__target_encoded`: `0.216`
- `response_message__target_encoded`: `0.216`
- `scheme_response_code__target_encoded`: `0.216`
- `psp_raw_response__target_encoded`: `0.216`
- `latency_3ds_ms`: `0.159`
- `mastercard_advice_code`: `0.129`
- `decline_bucket__target_encoded`: `0.103`
- `routing_optimized`: `0.054`
- `account_updater_used`: `0.040`
- `amount`: `0.040`
- `event_dayofweek`: `0.039`
- `merchant_mcc`: `0.038`
- `latency_ms`: `0.037`
- `interchange_estimate_bps`: `0.036`

## Strongest Negative Features
- `is_recurring`: `-0.755`
- `is_mit`: `-0.755`
- `risk_skip_flag`: `-0.736`
- `is_cross_border`: `-0.437`
- `three_ds_requested`: `-0.189`
- `is_weekend`: `-0.159`
- `wallet_type__target_encoded`: `-0.141`
- `routing_reason__target_encoded`: `-0.141`
- `risk_model_version__target_encoded`: `-0.141`
- `payment_method_details__target_encoded`: `-0.141`
- `card_category__target_encoded`: `-0.141`
- `payment_method__target_encoded`: `-0.141`
- `transaction_type__target_encoded`: `-0.141`
- `card_commercial_type__target_encoded`: `-0.141`
- `user_agent_family__target_encoded`: `-0.141`

## Segment Stability Snapshot
- `merchant_country=US`: rows `364`, positive rate `12.36%`, AUC `0.702`, F1 `0.279`
- `merchant_country=DE`: rows `181`, positive rate `12.15%`, AUC `0.592`, F1 `0.213`
- `merchant_country=GB`: rows `160`, positive rate `11.88%`, AUC `0.748`, F1 `0.257`
- `merchant_country=BR`: rows `126`, positive rate `10.32%`, AUC `0.715`, F1 `0.270`
- `merchant_country=IN`: rows `110`, positive rate `18.18%`, AUC `0.706`, F1 `0.418`
- `merchant_country=FR`: rows `110`, positive rate `25.45%`, AUC `0.753`, F1 `0.488`
- `merchant_country=JP`: rows `69`, positive rate `10.14%`, AUC `0.696`, F1 `0.267`
- `merchant_country=AE`: rows `55`, positive rate `7.27%`, AUC `0.745`, F1 `0.174`
- `merchant_vertical=ecom`: rows `488`, positive rate `12.91%`, AUC `0.743`, F1 `0.322`
- `merchant_vertical=saas`: rows `340`, positive rate `6.18%`, AUC `0.872`, F1 `0.333`
- `merchant_vertical=marketplace`: rows `304`, positive rate `14.14%`, AUC `0.698`, F1 `0.307`
- `merchant_vertical=travel`: rows `264`, positive rate `19.32%`, AUC `0.601`, F1 `0.327`
- `merchant_vertical=high_risk`: rows `253`, positive rate `17.39%`, AUC `0.682`, F1 `0.360`
- `merchant_vertical=digital_goods`: rows `196`, positive rate `13.27%`, AUC `0.751`, F1 `0.350`
- `processor_name=global-acquirer-b`: rows `360`, positive rate `14.44%`, AUC `0.696`, F1 `0.324`
- `processor_name=global-acquirer-a`: rows `276`, positive rate `14.13%`, AUC `0.744`, F1 `0.361`

## Recommendation
- Keep the current modeling table and proceed to a proper tree-based model when the runtime permits it.
- Use this V2 run as the defensible baseline for the case study.
- Only expand the synthetic dataset if a stronger model still shows weak segment stability or obvious undercoverage.
