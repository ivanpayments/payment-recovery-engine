# Project 3 Recovery Simulator Report

## Run Setup
- Source: temporal test split from Project 3 modeling table
- Rows scored: `1845`
- Decision threshold used: `0.05`
- Threshold basis: `maximize validation realized_net_value_usd`
- Retry cost assumption: `$0.12`
- Margin rate assumption: `35%`
- Friction cost assumption: `$0.03`

## Summary
- Recommended `retry`: `1263` declines
- Recommended `do_not_retry`: `582` declines
- Aggregate expected retry value of recommended retries: `$23,706.19`
- Actual recovered-in-hindsight within recommended retries: `244`

## Top Retry Candidates
- `2025-07-26 11:16:06` `US` `global-acquirer-a` code `51` amount `$5,834.24` -> `retry` (p=`0.315`, expected value=`$643.97`)
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - Response code `51` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
- `2025-07-22 11:12:05` `DE` `global-acquirer-b` code `51` amount `$3,741.10` -> `retry` (p=`0.406`, expected value=`$530.88`)
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - Response code `51` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
- `2025-07-21 07:20:42` `IN` `regional-card-specialist-a` code `51` amount `$2,562.55` -> `retry` (p=`0.308`, expected value=`$276.05`)
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - Response code `51` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
- `2025-08-03 19:19:05` `AR` `global-acquirer-b` code `51` amount `$2,892.11` -> `retry` (p=`0.271`, expected value=`$273.93`)
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - Response code `51` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
- `2025-07-19 10:48:29` `DE` `regional-card-specialist-b` code `51` amount `$2,747.66` -> `retry` (p=`0.275`, expected value=`$264.15`)
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - Response code `51` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
- `2025-07-11 17:08:55` `MX` `regional-card-specialist-a` code `91` amount `$3,163.72` -> `retry` (p=`0.226`, expected value=`$250.40`)
  - Response code `91` increased the predicted retry recoverability.
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
- `2025-07-11 21:48:14` `GB` `global-acquirer-a` code `51` amount `$1,618.50` -> `retry` (p=`0.409`, expected value=`$231.58`)
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
  - Response code `51` increased the predicted retry recoverability.
- `2025-08-03 17:39:23` `DE` `regional-bank-processor-b` code `51` amount `$1,962.54` -> `retry` (p=`0.324`, expected value=`$222.33`)
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
  - Response code `51` increased the predicted retry recoverability.
- `2025-07-13 20:39:15` `PL` `global-acquirer-b` code `05` amount `$5,637.90` -> `retry` (p=`0.102`, expected value=`$201.07`)
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
  - Response code `05` decreased the predicted retry recoverability.
- `2025-07-29 19:10:49` `CN` `global-acquirer-b` code `51` amount `$1,574.39` -> `retry` (p=`0.335`, expected value=`$184.20`)
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - Response code `51` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.

## Top Do-Not-Retry Examples
- `2025-07-28 18:37:29` `GB` `regional-card-specialist-a` code `92` amount `$1,448.23` -> `do_not_retry` (p=`0.041`, expected value=`$20.48`)
  - Feature `interchange_estimate_bps` with value `12.5` decreased the predicted retry recoverability.
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
- `2025-07-14 01:33:22` `US` `global-acquirer-b` code `05` amount `$1,440.99` -> `do_not_retry` (p=`0.040`, expected value=`$20.23`)
  - Feature `interchange_estimate_bps` with value `5.0` decreased the predicted retry recoverability.
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
- `2025-07-28 17:18:42` `US` `cross-border-fx-specialist-a` code `05` amount `$990.49` -> `do_not_retry` (p=`0.050`, expected value=`$17.02`)
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
  - Response code `05` decreased the predicted retry recoverability.
- `2025-07-22 15:58:56` `SG` `regional-card-specialist-a` code `05` amount `$825.05` -> `do_not_retry` (p=`0.049`, expected value=`$14.08`)
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - `risk_skip_flag=False` increased the predicted retry recoverability.
  - Response code `05` decreased the predicted retry recoverability.
- `2025-07-26 15:19:27` `AE` `regional-card-specialist-b` code `61` amount `$558.95` -> `do_not_retry` (p=`0.044`, expected value=`$8.48`)
  - Merchant vertical `saas` decreased the predicted retry recoverability.
  - `is_soft_decline=True` increased the predicted retry recoverability.
  - Response code `61` decreased the predicted retry recoverability.
