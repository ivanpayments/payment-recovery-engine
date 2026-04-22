# Project 3 Training Pipeline Design

## Purpose
This document defines the next-stage training and evaluation pipeline for the ML Payment Recovery Engine. It builds on the baseline work already completed and translates it into a more production-like, portfolio-defensible plan.

The goal is not just to train a model. The goal is to train a model that:
- uses only data available at original decline time
- is explainable
- is measured against business outcomes
- can eventually support both API inference and simulator-based demos

## Current State
Completed so far:
- reused the Project 2 synthetic transaction dataset as the Project 3 data foundation
- derived a Project 3 modeling table with one row per original declined transaction
- defined the primary label as `target_recovered_by_retry`
- ran two baseline models
- confirmed the current dataset is sufficient for MVP model development

Key artifacts already available:
- `project3_modeling_table.csv`
- `project3_modeling_profile.md`
- `project3_feature_coverage.md`
- `project3_baseline_evaluation.md`
- `project3_baseline_v2_evaluation.md`

## Modeling Objective
The MVP model should predict:

`probability_that_this_original_decline_is_recoverable_by_retry`

This is the right objective because it keeps prediction and policy separate:
- the model estimates recoverability
- the decision engine later converts that into `retry` or `do not retry`

This is more flexible and easier to explain than directly training the model to output a business action.

## Primary Label
Primary target:
- `target_recovered_by_retry`

Definition:
- `1` if any retry linked to the original declined transaction was eventually approved
- `0` otherwise

Secondary analysis label:
- `target_first_retry_approved`

Use of the secondary label:
- evaluation only for now
- possible later model variant if we want to tighten the product from "eventual recoverability" to "next retry success"

## Recommended Next Model
Recommended model class:
- gradient-boosted trees, preferably LightGBM

Why:
- strong fit for tabular payments data
- handles non-linear interactions well
- naturally supports mixed numerical and categorical patterns
- easier to pair with SHAP than a more complex ensemble
- likely to outperform logistic-style baselines on this type of data

Do not do yet:
- stacking ensemble
- neural networks
- multiple-model orchestration

Those are unnecessary for the current stage and would weaken the story by adding complexity before need.

## Training Pipeline Architecture
The next-stage pipeline should have six parts.

### 1. Input
Source:
- `project3_modeling_table.csv`

Unit:
- one row per original declined transaction

### 2. Feature Policy
Before training, split columns into three categories:

- allowed training features
- excluded identifier or leakage fields
- analysis-only fields retained for reporting and traceability

This is critical. The modeling table should remain wide, but the training feature set should be narrower and intentional.

### 3. Temporal Split
Use time-based train/validation/test splits only.

Recommended default:
- train: first 70%
- validation: next 15%
- test: final 15%

Why:
- prevents future leakage
- matches the real deployment pattern, where decisions are made on future declines using past data

### 4. Model Training
Initial production-style model:
- LightGBM binary classifier

Suggested first settings:
- objective: binary classification
- metric tracking: AUC, log loss, PR-style metrics, calibration checks
- class weighting: consider only if needed after first clean run
- early stopping on validation set

### 5. Evaluation Layer
Evaluation should happen in two dimensions:

Technical metrics:
- AUC
- log loss
- Brier score
- precision
- recall
- F1

Business metrics:
- recovered decline count under a decision threshold
- estimated recovery value
- retry volume
- false-positive retry burden
- comparison against a rules baseline

### 6. Explanation Layer
Primary explanation method:
- SHAP

Why:
- strongest fit for tree-based tabular modeling
- supports the core product narrative of explainable decline recovery
- gives both per-prediction explanations and global importance views

## Feature Policy
The feature policy should be explicit and versioned.

### Allowed Feature Families
These should be eligible for training if they are known at original decline time.

Transaction and amount:
- amount
- amount_usd
- currency
- processor_fee_bps
- interchange_estimate_bps
- scheme_fee_bps
- fx_applied
- fx_rate
- settlement_currency

Merchant and routing context:
- merchant_vertical
- merchant_mcc
- merchant_country
- archetype
- processor_name
- routing_reason

Card and payment method context:
- card_brand
- card_type
- card_country
- card_funding_source
- is_token
- token_type
- present_mode
- payment_method
- wallet_type
- network_token_present
- card_product_type
- card_category
- card_commercial_type

Decline and auth context:
- response_code
- response_message
- decline_bucket
- is_soft_decline
- scheme_response_code
- timeout_flag
- mastercard_advice_code

Authentication context:
- three_ds_requested
- three_ds_outcome
- three_ds_version
- three_ds_flow
- three_ds_eci
- sca_exemption

Geography and identity-of-context:
- billing_country
- shipping_country
- ip_country
- issuer_country
- is_cross_border
- acquirer_country
- issuer_size

Risk and operational context:
- risk_score
- fraud_flag
- risk_skip_flag
- account_updater_used
- latency_ms
- latency_auth_ms
- latency_3ds_ms
- latency_bucket

Channel and device context:
- device_os
- user_agent_family
- entry_mode
- pan_entry_mode
- cardholder_verification_method
- cvv_result
- avs_result
- avs_zip_match
- avs_street_match
- contactless
- nfc_used
- apple_pay
- google_pay
- samsung_pay
- click_to_pay

Derived time features from timestamp:
- hour of day
- day of week
- month
- weekend flag

### Excluded from Training
These fields should remain in the modeling table but must not be model inputs.

Identifiers:
- transaction_id
- merchant_id
- psp_transaction_id
- psp_reference
- network_transaction_id
- stan
- rrn
- arn
- session_id
- correlation_id
- trace_id
- device_fingerprint

Retry-chain leakage:
- is_retry
- original_transaction_id
- retry_attempt_num
- retry_reason
- hours_since_original
- has_any_retry
- retry_count
- first_retry_status
- first_retry_attempt_num
- hours_to_first_retry
- recovery_attempt_num
- hours_to_recovery

Post-decision fields:
- approved_amount
- auth_code
- captured_amount
- refunded_amount
- captured_at
- refunded_at
- voided_at
- settled_at

### Fields to Review Before Final Inclusion
These are not obviously wrong, but they need judgment:
- is_recurring
- recurring_type
- is_mit
- mit_flag_revoked
- chargeback-related fields
- some routing optimization flags

Reason:
- some of these may be valid at decision time
- some may simply re-encode generator logic too directly
- some may create a synthetic-world shortcut instead of a realistic feature

## Evaluation Framework
The next real model should be judged against three baselines:

### Baseline 1: Soft decline retry rule
Simple policy:
- retry soft declines
- avoid obvious hard declines
- modest boosts for certain codes and higher-value transactions

### Baseline 2: Logistic baseline V2
Current strong baseline artifact:
- `project3_baseline_v2_evaluation.md`

### Baseline 3: Business threshold policy
After the LightGBM model outputs probabilities, apply a first business rule such as:
- retry if predicted recoverability exceeds threshold and expected value is positive

## Decision Layer Design
The decision layer should remain separate from the model.

Suggested first formula:

`expected_retry_value = p_recovery * expected_recovered_value - retry_cost - friction_cost`

Where:
- `p_recovery` comes from the model
- `expected_recovered_value` can initially be approximated using amount or amount_usd
- `retry_cost` is a configurable constant or band
- `friction_cost` is a configurable penalty

First MVP decision rule:
- if expected value > 0 and predicted recoverability exceeds threshold -> `retry`
- else -> `do_not_retry`

This is simple enough for an MVP and easy to explain to business users.

## Explainability Plan
The next model should support two explanation views.

### Global explanation
Questions it answers:
- what generally drives recoverability predictions?
- are the model's strongest factors sensible?

Outputs:
- global SHAP importance
- top positive and negative drivers

### Local explanation
Questions it answers:
- why did the model recommend retrying this specific decline?
- what were the top contributing factors?

Outputs:
- top feature contributions for a given scored decline
- business-language translation layer on top

## Simulator Integration
The model should also feed a simple simulator or replay layer.

Input:
- a batch of declined transactions

Output:
- ranked retry candidates
- recommended decisions
- recovery-rate comparison vs rules
- estimated business impact

This matters because:
- real merchants integrate this in flow
- but demos and stakeholder understanding benefit from a replay view

## Recommended File Layout for Next Stage
Suggested next-stage files:

- `features/project3_feature_policy.md`
- `features/project3_feature_list.json`
- `models/train_lightgbm.py`
- `models/evaluate_lightgbm.py`
- `models/business_decision.py`
- `explanations/shap_project3.py`
- `reports/project3_model_report.md`
- `simulation/project3_replay.py`

For now, we do not need to build all of these at once. The correct order is:

1. feature policy
2. stronger model training script
3. evaluation report
4. decision layer
5. explainability
6. simulator

## Risks to Watch
### 1. Generator-shaped learning
Because the data is synthetic, the model may partly learn the assumptions of the Project 2 generator.

Response:
- keep the caveat explicit
- prefer understandable features and explainability
- avoid overclaiming external realism

### 2. Segment instability
Some slices may still be thin.

Response:
- track segment-level metrics
- only expand the dataset if a stronger model still shows clear instability

### 3. Leakage through hidden operational fields
Some columns may quietly encode future outcomes or generator shortcuts.

Response:
- maintain a strict feature policy
- review suspicious high-importance features

## Recommendation
The next build step should be:

1. formalize the feature policy as a project artifact
2. build a LightGBM-ready training script design
3. add the first business decision layer

The current baseline results justify moving forward without generating more data yet.

## Final Position
Project 3 is now at the point where it should transition from:

"Can we learn anything useful from this synthetic retry universe?"

to:

"Can we train an explainable, business-oriented recovery model using a disciplined feature policy and evaluate it in a way that looks credible to real stakeholders?"

That is the correct next stage.
