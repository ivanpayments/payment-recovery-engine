# Model Card: ML Payment Recovery Engine

## Intended Use

This model is a decision-support classifier for declined card transactions. It estimates the probability that an original decline is recoverable by retry and combines that probability with a simple expected-value rule to recommend either `retry` or `do_not_retry`. It is suitable for demos, architecture walkthroughs, and offline operational analysis. It is not suitable for unattended real-time production use until it has completed a human-in-the-loop validation period with live monitoring, rollback controls, and drift checks.

## Training Data

- Dataset: fully synthetic payment-decline data
- Modeling unit: one row per original declined transaction
- Volume: about 12.3K rows
- Positive class: about 14.7% `target_recovered_by_retry`
- Feature set: 78 allowed features from `project3_feature_policy.json`
- Split: temporal `70/15/15` train, validation, test

No real cardholder data is used. The training boundary is intentionally limited to fields known at original decline time so the score does not rely on retry outcomes or other post-decision leakage.

## Model Architecture

- Family: LightGBM binary classifier
- Best iteration: `66`
- Decision threshold used in serving: `0.05`
- Decision logic: recommend retry only when probability is above threshold and expected retry value is positive

The serving layer returns the score, the expected value, the action recommendation, and the top three inference-time contribution drivers. That gives the project a concrete operational narrative beyond raw AUC.

## Performance Metrics

### Validation

- AUC: `0.766`
- Log loss: `0.362`
- Brier score: `0.117`
- Precision: `0.298`
- Recall: `0.660`
- F1: `0.410`

### Test

- AUC: `0.732`
- Log loss: `0.350`
- Brier score: `0.111`
- Precision: `0.230`
- Recall: `0.593`
- F1: `0.331`

### Segment Readout

- `merchant_country=US`: AUC `0.704`, F1 `0.301`
- `merchant_country=DE`: AUC `0.761`, F1 `0.326`
- `merchant_country=FR`: AUC `0.729`, F1 `0.411`
- `merchant_country=IN`: AUC `0.767`, F1 `0.441`
- `merchant_vertical=saas`: AUC `0.797`, F1 `0.250`
- `processor_name=global-acquirer-a`: AUC `0.730`, F1 `0.358`

Performance varies by country, merchant vertical, and processor. That is acceptable for a synthetic case study, but it reinforces the need for cohort-level monitoring before any live rollout.

## Explainability And Regulatory Framing

The project uses two explanation layers:

- Offline SHAP analysis to identify global feature importance and local example drivers
- Inference-time LightGBM contribution values to return the top factors for an individual prediction

This does not make the model automatically compliant with any regulation. It does, however, support a more defensible internal control posture: analysts can inspect why the model preferred retry versus no retry, and product stakeholders can challenge features that appear unstable, biased, or operationally unsafe.

## Known Limitations

- The dataset is synthetic, so distribution shift and issuer behavior are not validated against live production traffic.
- The retry policy assumes fixed retry cost, margin rate, and friction cost values; those economics are stylized rather than contracted.
- The current threshold maximizes synthetic validation net value, not a live merchant objective with actual customer fatigue data.
- Geographic coverage is broad enough for the case study, but not validated outside the modeled countries and processors.
- The current system does not yet include calibration, drift detection, or an online feedback loop.

## Ethical Considerations

- The model should not be used as a proxy for fraud adjudication or customer-value ranking.
- Segment performance should be reviewed regularly to avoid systematically weaker outcomes in specific countries or processor cohorts.
- Any future live use should include human review, monitoring, and a conservative rollout with fallback rules.

## Security And PII Posture

- Synthetic data only. No PAN, CVV, or direct cardholder identifiers are permitted in the dataset.
- The feature policy acts as the API boundary. Requests outside that boundary are rejected.
- Structured JSON logging records request metadata and prediction outcomes, but production deployment should avoid logging raw sensitive fields.

## Recommended Next Steps

- Add MLflow-backed experiment history for all three training scripts
- Run the API in Docker for portable local demos
- Add Great Expectations and drift checks before claiming production readiness
- Keep the case-study narrative explicit that this is a synthetic ML system built to demonstrate decision quality, controls, and deployment discipline
