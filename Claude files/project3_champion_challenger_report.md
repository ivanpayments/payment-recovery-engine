# Project 3 Champion Challenger Reconciliation

## Setup
- Champion: LightGBM model used in serving
- Challenger: baseline_v2 logistic regression with target encoding
- Test rows reconciled: `1845`

## Readout
- Mean champion probability: `0.151`
- Mean challenger probability: `0.162`
- Mean absolute delta: `0.084`
- Share of rows with |delta| > 0.20: `4.77%`

## Largest Disagreements
- `2025-07-27 15:55:11` `CA` `high-risk-or-orchestrator-a` code `51`: champion `0.552` vs challenger `0.187`
- `2025-08-05 20:59:55` `DE` `high-risk-or-orchestrator-a` code `51`: champion `0.556` vs challenger `0.210`
- `2025-07-26 22:48:54` `AE` `global-acquirer-a` code `91`: champion `0.582` vs challenger `0.245`
- `2025-07-12 13:47:33` `DE` `global-acquirer-b` code `51`: champion `0.546` vs challenger `0.214`
- `2025-07-26 20:30:27` `DE` `global-acquirer-b` code `51`: champion `0.519` vs challenger `0.193`
- `2025-07-13 08:36:07` `US` `high-risk-or-orchestrator-b` code `51`: champion `0.573` vs challenger `0.253`
- `2025-07-26 13:07:44` `DE` `global-acquirer-a` code `51`: champion `0.525` vs challenger `0.209`
- `2025-08-04 20:17:39` `DE` `global-acquirer-b` code `51`: champion `0.556` vs challenger `0.242`
- `2025-07-15 16:51:53` `CA` `high-risk-or-orchestrator-b` code `51`: champion `0.589` vs challenger `0.275`
- `2025-07-16 20:58:10` `CA` `high-risk-or-orchestrator-b` code `51`: champion `0.522` vs challenger `0.210`
- `2025-08-04 19:45:21` `US` `high-risk-or-orchestrator-b` code `51`: champion `0.553` vs challenger `0.242`
- `2025-07-13 16:56:20` `DE` `global-acquirer-b` code `96`: champion `0.545` vs challenger `0.235`

## Interpretation
- Large positive deltas indicate scenarios where the tree model is finding nonlinear recovery signal the linear challenger does not capture.
- Large negative deltas are useful review candidates for feature interactions or calibration drift.
