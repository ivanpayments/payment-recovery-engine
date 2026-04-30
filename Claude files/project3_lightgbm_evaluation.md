# Project 3 LightGBM Evaluation

## Setup
- Model: LightGBM binary classifier
- Training objective: predict whether an original declined transaction is eventually recoverable by retry
- Split: temporal 70/15/15
- Allowed features from policy: `74`
- Numeric features: `19`
- Boolean features: `16`
- Categorical features: `39`
- Decision threshold chosen on validation set: `0.11`

## Validation Metrics
- AUC: `0.763`
- Log loss: `0.334`
- Brier score: `0.095`
- Precision: `0.231`
- Recall: `0.492`
- F1: `0.315`

## Test Metrics
- AUC: `0.743`
- Log loss: `0.327`
- Brier score: `0.092`
- Precision: `0.188`
- Recall: `0.411`
- F1: `0.258`

## Rule Baseline on Test Set
- AUC: `0.619`
- Log loss: `0.421`
- Brier score: `0.133`
- Precision: `0.126`
- Recall: `1.000`
- F1: `0.224`

## Top Feature Importances
- `response_code`: `452.97899627685547`
- `risk_skip_flag`: `294.14500427246094`
- `interchange_estimate_bps`: `243.62078762054443`
- `merchant_country`: `142.0019989013672`
- `shipping_country`: `129.35899353027344`
- `risk_score`: `74.54617881774902`
- `ip_country`: `63.030701637268066`
- `latency_auth_ms`: `57.3019905090332`
- `latency_ms`: `44.26601982116699`
- `amount_usd`: `43.901719093322754`
- `scheme_ms`: `35.78898906707764`
- `processor_fee_bps`: `27.106379508972168`
- `is_soft_decline`: `21.329350471496582`
- `archetype`: `17.888900756835938`
- `scheme_fee_bps`: `14.733279705047607`
- `mastercard_advice_code`: `14.48900032043457`
- `amount`: `14.121600151062012`
- `card_country`: `13.439200401306152`
- `three_ds_eci`: `11.749899864196777`
- `merchant_mcc`: `10.714200019836426`

## Segment Stability Snapshot
- `merchant_country=US`: rows `364`, positive rate `13.46%`, AUC `0.668`, F1 `0.274`
- `merchant_country=DE`: rows `181`, positive rate `17.13%`, AUC `0.748`, F1 `0.365`
- `merchant_country=GB`: rows `160`, positive rate `7.50%`, AUC `0.773`, F1 `0.259`
- `merchant_country=BR`: rows `126`, positive rate `12.70%`, AUC `0.634`, F1 `0.000`
- `merchant_country=IN`: rows `110`, positive rate `1.82%`, AUC `0.676`, F1 `0.000`
- `merchant_country=FR`: rows `110`, positive rate `16.36%`, AUC `0.688`, F1 `0.213`
- `merchant_country=JP`: rows `69`, positive rate `7.25%`, AUC `0.509`, F1 `0.069`
- `merchant_country=AE`: rows `55`, positive rate `7.27%`, AUC `0.863`, F1 `0.400`
- `merchant_vertical=ecom`: rows `488`, positive rate `9.02%`, AUC `0.756`, F1 `0.252`
- `merchant_vertical=saas`: rows `340`, positive rate `4.71%`, AUC `0.765`, F1 `0.156`
- `merchant_vertical=marketplace`: rows `304`, positive rate `15.13%`, AUC `0.752`, F1 `0.327`
- `merchant_vertical=travel`: rows `264`, positive rate `14.39%`, AUC `0.692`, F1 `0.259`
- `merchant_vertical=high_risk`: rows `253`, positive rate `11.86%`, AUC `0.742`, F1 `0.278`
- `merchant_vertical=digital_goods`: rows `196`, positive rate `9.18%`, AUC `0.719`, F1 `0.241`
- `processor_name=global-acquirer-b`: rows `360`, positive rate `13.06%`, AUC `0.746`, F1 `0.304`
- `processor_name=global-acquirer-a`: rows `276`, positive rate `12.68%`, AUC `0.752`, F1 `0.283`

## Recommendation
- Compare this LightGBM run directly against the V2 logistic baseline.
- If the gain is meaningful, promote LightGBM as the canonical model in the project story.
- Next after that: SHAP explanations and a business decision layer.
