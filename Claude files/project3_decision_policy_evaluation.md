# Project 3 Decision Policy Evaluation

## Setup
- Holdout: temporal test split from the Project 3 modeling table
- Retry threshold used for ML policy: `0.05`
- Threshold basis: `maximize validation realized_net_value_usd`
- Retry cost assumption: `$0.12` per retry
- Margin rate assumption: `35%` of amount_usd
- Friction cost assumption: `$0.03` per retry

## Policy Comparison
- ML policy retries `1263` declines and recovers `244`
- Rules policy retries `1524` declines and recovers `248`
- Net value delta (ML - rules): `$-34.32`
- Recovered decline delta (ML - rules): `-4`
- Retry volume delta (ML - rules): `-261`

## ML Policy
- Retry rate: `68.46%`
- Recovered volume: `$81,601.68`
- Gross margin recovered: `$28,560.59`
- Wasted retry cost: `$122.28`
- Friction cost: `$37.89`
- Realized net value: `$28,400.42`

## Rules Policy
- Retry rate: `82.60%`
- Recovered volume: `$81,810.22`
- Gross margin recovered: `$28,633.58`
- Wasted retry cost: `$153.12`
- Friction cost: `$45.72`
- Realized net value: `$28,434.74`

## Interpretation
- This layer turns model probability into an operational recommendation by combining recoverability with simple economic assumptions.
- These numbers are synthetic and depend on the chosen cost assumptions, but they make the product story much more concrete than model metrics alone.
- The next refinement should be configurable business thresholds and SHAP-backed explanations for why a specific decline is recommended for retry.

## Sample High-Confidence Decisions
- `2025-07-26 11:16:06` `US` `global-acquirer-a` code `51` amount `$5834.24` -> `retry` (p=`0.315`, expected value=`$643.97`)
- `2025-07-22 11:12:05` `DE` `global-acquirer-b` code `51` amount `$3741.10` -> `retry` (p=`0.406`, expected value=`$530.88`)
- `2025-07-21 07:20:42` `IN` `regional-card-specialist-a` code `51` amount `$2562.55` -> `retry` (p=`0.308`, expected value=`$276.05`)
- `2025-08-03 19:19:05` `AR` `global-acquirer-b` code `51` amount `$2892.11` -> `retry` (p=`0.271`, expected value=`$273.93`)
- `2025-07-19 10:48:29` `DE` `regional-card-specialist-b` code `51` amount `$2747.66` -> `retry` (p=`0.275`, expected value=`$264.15`)
- `2025-07-11 17:08:55` `MX` `regional-card-specialist-a` code `91` amount `$3163.72` -> `retry` (p=`0.226`, expected value=`$250.40`)
- `2025-07-11 21:48:14` `GB` `global-acquirer-a` code `51` amount `$1618.50` -> `retry` (p=`0.409`, expected value=`$231.58`)
- `2025-08-03 17:39:23` `DE` `regional-bank-processor-b` code `51` amount `$1962.54` -> `retry` (p=`0.324`, expected value=`$222.33`)
- `2025-07-13 20:39:15` `PL` `global-acquirer-b` code `05` amount `$5637.90` -> `retry` (p=`0.102`, expected value=`$201.07`)
- `2025-07-29 19:10:49` `CN` `global-acquirer-b` code `51` amount `$1574.39` -> `retry` (p=`0.335`, expected value=`$184.20`)
- `2025-07-18 13:51:12` `BR` `high-risk-or-orchestrator-a` code `05` amount `$2140.80` -> `retry` (p=`0.245`, expected value=`$183.51`)
- `2025-08-05 08:01:40` `HK` `global-acquirer-a` code `51` amount `$1554.62` -> `retry` (p=`0.326`, expected value=`$177.41`)
