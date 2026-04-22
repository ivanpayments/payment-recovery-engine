# Project 3 LightGBM Evaluation

## Setup
- Model: LightGBM binary classifier
- Training objective: predict whether an original declined transaction is eventually recoverable by retry
- Split: temporal 70/15/15
- Allowed features from policy: `78`
- Numeric features: `19`
- Boolean features: `16`
- Categorical features: `43`
- Decision threshold chosen on validation set: `0.19`

## Validation Metrics
- AUC: `0.766`
- Log loss: `0.362`
- Brier score: `0.117`
- Precision: `0.298`
- Recall: `0.660`
- F1: `0.410`

## Test Metrics
- AUC: `0.732`
- Log loss: `0.350`
- Brier score: `0.111`
- Precision: `0.230`
- Recall: `0.593`
- F1: `0.331`

## Rule Baseline on Test Set
- AUC: `0.612`
- Log loss: `0.440`
- Brier score: `0.142`
- Precision: `0.163`
- Recall: `1.000`
- F1: `0.280`

## Top Feature Importances
- `is_soft_decline`: `3610.3051319122314`
- `risk_skip_flag`: `3236.7366766929626`
- `response_code`: `2065.8711733818054`
- `interchange_estimate_bps`: `1836.7523217201233`
- `processor_name`: `1036.4289875030518`
- `scheme_fee_bps`: `935.5959877967834`
- `scheme_ms`: `909.8697466850281`
- `processor_fee_bps`: `902.9350123405457`
- `card_country`: `887.5990762710571`
- `merchant_vertical`: `868.64657330513`
- `amount`: `675.944221496582`
- `latency_ms`: `668.3577919006348`
- `risk_score`: `648.7326221466064`
- `latency_auth_ms`: `626.2867383956909`
- `amount_usd`: `623.7389407157898`
- `event_hour`: `571.2797181606293`
- `ip_country`: `519.500569820404`
- `latency_3ds_ms`: `495.8111023902893`
- `archetype`: `488.58840322494507`
- `shipping_country`: `453.70357179641724`

## Segment Stability Snapshot
- `merchant_country=US`: rows `364`, positive rate `12.36%`, AUC `0.704`, F1 `0.301`
- `merchant_country=DE`: rows `181`, positive rate `12.15%`, AUC `0.761`, F1 `0.326`
- `merchant_country=GB`: rows `160`, positive rate `11.88%`, AUC `0.725`, F1 `0.274`
- `merchant_country=BR`: rows `126`, positive rate `10.32%`, AUC `0.726`, F1 `0.271`
- `merchant_country=IN`: rows `110`, positive rate `18.18%`, AUC `0.767`, F1 `0.441`
- `merchant_country=FR`: rows `110`, positive rate `25.45%`, AUC `0.729`, F1 `0.411`
- `merchant_country=JP`: rows `69`, positive rate `10.14%`, AUC `0.641`, F1 `0.211`
- `merchant_country=AE`: rows `55`, positive rate `7.27%`, AUC `0.740`, F1 `0.250`
- `merchant_vertical=ecom`: rows `488`, positive rate `12.91%`, AUC `0.746`, F1 `0.315`
- `merchant_vertical=saas`: rows `340`, positive rate `6.18%`, AUC `0.797`, F1 `0.250`
- `merchant_vertical=marketplace`: rows `304`, positive rate `14.14%`, AUC `0.716`, F1 `0.338`
- `merchant_vertical=travel`: rows `264`, positive rate `19.32%`, AUC `0.671`, F1 `0.395`
- `merchant_vertical=high_risk`: rows `253`, positive rate `17.39%`, AUC `0.626`, F1 `0.359`
- `merchant_vertical=digital_goods`: rows `196`, positive rate `13.27%`, AUC `0.712`, F1 `0.247`
- `processor_name=global-acquirer-b`: rows `360`, positive rate `14.44%`, AUC `0.706`, F1 `0.323`
- `processor_name=global-acquirer-a`: rows `276`, positive rate `14.13%`, AUC `0.730`, F1 `0.358`

## Recommendation
- Compare this LightGBM run directly against the V2 logistic baseline.
- If the gain is meaningful, promote LightGBM as the canonical model in the project story.
- Next after that: SHAP explanations and a business decision layer.
