# Project 3 Data Fit and Label Strategy

## Purpose
This note evaluates whether the synthetic transaction dataset from Project 2 is suitable for Project 3 and recommends the supervised target definition for the ML Payment Recovery Engine.

## Dataset Reuse Decision
Recommendation: reuse the Project 2 dataset as the foundation for Project 3.

Source dataset:
- `C:\Users\ivana\OneDrive\Everything\Pet projects\AI tools\Payments bots\Project 2\Claude files\routing_transactions.csv`

Why it is useful:
- it includes original transaction outcomes
- it includes decline-specific fields such as `auth_status`, `response_code`, `decline_bucket`, and `is_soft_decline`
- it includes retry-chain fields such as `is_retry`, `original_transaction_id`, `retry_attempt_num`, and `hours_since_original`
- it includes rich transaction context that can become model features
- retry attempts are represented as separate rows tied to the original decline

This is much stronger than a one-shot payments table because the retry world already exists inside the synthetic dataset.

## Key Profiling Results
Using the Project 2 dataset:

- total rows: `106,739`
- original declines: `12,300`
- retry rows: `6,739`
- original declines with at least one retry: `4,044`
- original declines with no retry: `8,256`
- approved retry rows: `1,806`
- declined retry rows: `4,933`
- max retry attempt number: `3`

At the original-decline level:

- eventual recovery count: `1,806`
- first-retry recovery count: `1,143`
- eventual recovery rate across all original declines: `14.68%`
- eventual recovery rate among declines that were retried: `44.66%`
- first-retry recovery rate among declines that were retried: `28.26%`
- soft-decline eventual recovery rate: `17.79%`
- hard-decline eventual recovery rate: `0.00%`

Top decline-code signals observed in the dataset:

| Response code | Original declines | Retried rate | Eventual recovery rate |
|---|---:|---:|---:|
| `05` | 4,856 | 42.79% | 15.44% |
| `51` | 3,352 | 36.58% | 22.70% |
| `54` | 708 | 0.00% | 0.00% |
| `57` | 592 | 38.01% | 9.97% |
| `62` | 554 | 0.00% | 0.00% |
| `14` | 407 | 0.00% | 0.00% |
| `91` | 228 | 36.84% | 30.70% |
| `61` | 202 | 38.12% | 10.89% |
| `65` | 195 | 37.44% | 10.77% |
| `RC` | 187 | 45.99% | 17.65% |

## What This Means for Project 3
The dataset is suitable for Project 3 because it gives us both:

1. the original declined payment context
2. the downstream retry outcomes needed to derive labels

This means Project 3 does not have to invent retry labels from scratch. Instead, it can derive labels from the retry chains already encoded in Project 2.

## Recommended Modeling Unit
Recommendation: build the Project 3 training dataset at the level of the original declined transaction.

Why:
- the product receives a newly declined transaction as input
- the system should predict what to do at the moment of decline
- training on retry rows directly would blur the product's decision point

So each training row for Project 3 should represent:
- one original decline
- its context at the time of the decline
- a derived label based on what happened in its later retry chain

## Candidate Label Definitions
There are three realistic label options.

### Option A: Eventual retry recovery
Definition:
- label = `1` if any later retry tied to the original decline eventually succeeded
- label = `0` otherwise

Pros:
- aligns with the broad business question: "is this decline recoverable through retry?"
- uses the full retry chain already present in the dataset
- gives the richest positive class

Cons:
- combines first retry and later retries into one label
- may slightly overstate what is knowable at the moment of the first decision

### Option B: First-retry success
Definition:
- label = `1` if the first retry succeeded
- label = `0` otherwise

Pros:
- tighter operational framing
- closer to immediate retry decisioning
- cleaner if the product is positioned as next-action guidance

Cons:
- narrower positive class
- ignores value recovered on second or third retry

### Option C: Business-decision label
Definition:
- label = `1` if retrying produced positive expected business value
- label = `0` otherwise

Pros:
- closest to the eventual product decision
- aligns with business economics rather than pure success/failure

Cons:
- requires more assumptions up front
- makes the training target less transparent
- harder to explain early in the project

## Recommended Target
Recommendation: use **Option A, eventual retry recovery**, as the primary supervised target for MVP.

Why this is the best starting point:
- it is simple to explain
- it is directly derivable from the existing dataset
- it matches the broad product question of recoverability
- it allows the model to estimate recovery likelihood first
- a separate decision engine can then convert that probability into `retry` or `do not retry`

This keeps the architecture clean:
- model predicts probability of recovery
- business layer decides whether retrying is worth it

## Recommended Product Framing
Project 3 should be framed as:

"Given a newly declined card transaction, estimate the probability that it can be recovered through retry, then convert that into an explainable retry recommendation."

This is cleaner than directly saying the model predicts `retry` versus `abandon`.

## Training Dataset Construction Plan
Recommended construction steps:

1. Filter Project 2 data to original declined transactions where:
   - `auth_status = DECLINED`
   - `is_retry = False`

2. Group retry rows by `original_transaction_id`.

3. For each original decline, derive:
   - `has_any_retry`
   - `retry_count`
   - `any_retry_approved`
   - `first_retry_approved`
   - `max_retry_attempt_num`
   - `time_to_first_retry`
   - `time_to_recovery` if available

4. Use only features available at original decline time for model input.

5. Set the initial target as:
   - `target_recovered_by_retry = any_retry_approved`

## Feature Guardrails
To avoid leakage, Project 3 should not use any information that only becomes known after the original decline.

Allowed examples:
- response code
- decline bucket
- soft vs hard decline
- amount
- country
- card brand
- provider
- time of day
- risk score at decline time
- 3DS requested at original attempt
- tokenization status at original attempt

Not allowed as model inputs:
- retry attempt count
- hours since original on retry rows
- final approved retry outcome
- any feature derived from later retry events

## Important Caveat
The Project 2 retry logic already encodes certain retry heuristics. That means Project 3 may learn patterns that partly reflect the Project 2 generator's built-in assumptions.

This is acceptable for a portfolio project, but it must be stated honestly:
- Project 3 is trained on a synthetic retry universe
- the model is learning recoverability patterns from that synthetic world
- the objective is product realism and explainable architecture, not production-grade external validity

## Final Recommendation
Use the Project 2 dataset as the Project 3 foundation.

Specifically:
- keep the unit of prediction as the original decline
- derive labels from retry chains
- use `eventual retry recovery` as the MVP supervised target
- keep business decisioning in a separate expected-value layer

## Suggested Next Step
Create a derived Project 3 modeling table with one row per original decline and columns for:
- original decline context
- retry-chain summary fields
- target label

That derived table should become the main training input for the ML Payment Recovery Engine.
