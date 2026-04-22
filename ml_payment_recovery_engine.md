# ML Payment Recovery Engine — Decline Recovery Predictor (+ Chatbot Tool)

## Product Summary

ML Payment Recovery Engine is the **decline recovery predictor** module of the unified AI chatbot portfolio. It powers the Payment Data Chatbot's `predict_decline_recovery` tool — so merchants can ask "which of today's declined transactions should I retry?" and get back a ranked list with SHAP explanations in plain business language. Under the hood: LightGBM with Optuna-tuned hyperparameters, trained on synthetic transaction retry outcomes. Every prediction includes SHAP and LIME explanations translated to business language. The engine also stands alone as a REST API, consumes real-time `payment.declined` events from Kafka, and has full MLOps (MLflow, CI/CD with model validation, drift detection, champion/challenger). This is the most technically complex product in the portfolio — the deepest engineering showcase with ~23 technologies spanning ML, event-driven architecture, infrastructure-as-code, and observability.

**Platform role**: Third module of the unified chatbot portfolio (see `plan.md`). The ML engine exposes `POST /predict-recovery` returning `{decision, confidence, net_retry_value, explanation}`; the chatbot calls this whenever users ask decline / retry / recovery questions.

**Complexity**: COMPLEX (engineering showcase)
**Ship date**: June 26 (Phase 3, weeks 8-11 code + deck)
**Build effort**: 27 hours code across 14 sessions (13 × 2h + 1h QA) + deck sessions
**Primary role target**: Engineering-focused Pre-sales (reusable for AI Engineer, TPM)
**Secondary role target**: Solutions Engineer, Product Manager

## The Problem

When a payment declines, someone has to decide: retry the same provider, try a different one, or give up. This decision happens millions of times per day across the industry, and almost everyone gets it wrong.

**The current approach is static rules:**
- "Retry all soft declines twice, then cascade to the backup provider."
- "Never retry hard declines."
- "Wait 30 minutes between retries."

These rules ignore context. A $500 Brazilian Visa transaction declining at 3AM with code 05 (Do Not Honor) has very different retry odds than a $20 US Mastercard declining at noon with the same code. The Brazilian transaction might succeed at 34% on retry; the US one at 12%. But static rules treat them identically.

**The hidden cost: most companies don't measure the ROI of each retry attempt.** Every retry has a cost — the processor charges $0.05-0.30 per attempt, and the customer waits longer (increasing abandonment risk). A retry that costs $0.15 and has 5% success odds on a $8 transaction is destroying value. But the same retry on a $500 transaction is worth it. Static rules can't do this math.

**Who has this problem**: Heads of payments, payment optimization teams, revenue recovery managers at any company processing significant transaction volume. Also: PSPs that offer retry/cascade as a value-add service to their merchants.

**How often**: Every declined transaction — typically 5-15% of all transactions. On $1B GMV, that's 50-150M declined transactions per year. Each one is a retry/cascade/abandon decision.

**What it costs**: A 2% improvement in retry success rate on $1B GMV recovers $20M annually. Most companies leave 30-50% of this on the table because static rules can't optimize for transaction-level context.

## What the User Sees

**API call**: Send a declined transaction's context (decline code, provider, country, card brand, amount, time of day, attempt number) → get back: "Retry with Stripe, 72% confidence, expected recovery value $3.40" with a plain-English explanation: "Decline code 05 in Brazil has 34% historical retry success. Transaction amount $450 makes retry worthwhile despite 66% failure odds — expected net value is positive."

**Real-time pipeline**: The ML engine consumes declined payment events from Kafka as they happen. Within 2 seconds of a decline, it publishes a retry/cascade/abandon decision with confidence and explanation to a decisions topic. No batch processing — the recommendation is available before the customer's checkout page times out.

**Simulation mode**: Replay 10,000 historical transactions under 5 different strategies and see the comparison: "ML model recovers 23% more revenue than rules-based at 15% lower retry cost." Show this to stakeholders to justify the investment.

## Why Any Team Would Build This

- **Revenue recovery is the highest-ROI optimization in payments**: 2% improvement on $1B GMV = $20M/year. No other single change has this leverage.
- **Static rules leave money on the table**: They can't account for 30+ contextual features that affect retry success. ML can. The gap between rules and ML is typically 15-30% more recovered revenue.
- **Cost-aware decisions**: The model optimizes net value (revenue recovered minus retry costs minus customer friction), not just success rate. It won't waste money retrying low-value transactions with slim odds.
- **Explainability builds trust**: Every prediction comes with plain-English reasoning from two independent methods (SHAP + LIME). Payment teams can verify the logic instead of trusting a black box. When explanations from both methods agree, confidence is high.
- **Measurable lift**: The simulation harness compares ML decisions against rules on your actual transaction data. The business case writes itself — "here's how much more we'd recover."

## Tech Stack

- Python 3.12
- LightGBM (gradient-boosted decision trees — primary model)
- scikit-learn (preprocessing, train/test split, calibration, evaluation metrics, pipeline utilities)
- SHAP (TreeExplainer for feature-level explanations)
- LIME (local surrogate model explanations — comparison method)
- Optuna (hyperparameter tuning — TPE sampler, 20 trials)
- MLflow (experiment tracking, model registry, artifact storage)
- DuckDB (fast SQL-based analytics for exploration and feature engineering)
- FastAPI (decision API: /predict, /batch, /model-card, /health)
- Kafka (consume payment.declined events from Payment Routing Simulator, publish retry decisions)
- Great Expectations (data validation — check input data quality before pipeline processes it)
- Docker + docker-compose (containerized API + MLflow server + Redis + Grafana)
- GitHub Actions (CI/CD with model validation gates — new model must beat champion)
- Prometheus (metrics: prediction_count, prediction_latency, confidence_distribution, drift_score)
- OpenTelemetry (tracing: request → feature extraction → model inference → explanation → response)
- Structured logging (JSON format)
- pandas + numpy (data manipulation, numerical operations)
- Pydantic (API request/response validation)
- pytest (unit + integration tests)
- Jupyter notebooks (EDA, analysis, visualization)
- Isotonic regression (probability calibration — from scikit-learn)
- Redis (feature cache for repeated predictions, Kafka consumer deduplication)
- Terraform (infrastructure-as-code for DigitalOcean deployment)
- Grafana (model monitoring dashboards: prediction volume, confidence, drift scores)

## Technical Solution

### What you're building

A Python package that takes a failed transaction's context (decline code, provider, country, card brand, amount, time of day, attempt number) and returns a decision: retry with the same provider, cascade to a different provider, or abandon. The decision comes with a confidence score, dollar-value estimate, and two sets of plain-English explanations — one from SHAP (exact feature contributions) and one from LIME (local surrogate model) — so you can see if both methods agree. The system has 4 parts: (1) a LightGBM model tuned with Optuna over 20 trials, optimizing a custom business metric (net_retry_value) instead of generic accuracy, (2) a FastAPI API serving predictions with explanations, (3) a Kafka consumer that processes declined payment events from the Payment Routing Simulator in real time, and (4) a simulation harness that replays 10K transactions comparing 5 strategies. Full MLOps: every training run tracked in MLflow, CI/CD validates new models before deployment, drift detection alerts when data distribution shifts. Redis caches computed feature vectors and deduplicates Kafka events. Terraform manages infrastructure-as-code deployment. Grafana dashboards provide real-time model monitoring.

### Architecture

```
Data Pipeline:
  synthetic_transactions.csv (10K rows, 72 cols)
    → DuckDB: SQL queries for exploration + feature engineering (faster than pandas for aggregations)
    → Target Variable Construction:
        Filter to declined transactions
        Simulate retry success: soft declines (codes 05, 51, 61) → 20-40% success
                                hard declines (codes 14, 62, 54) → 0-5% success
        Modified by country (Brazil soft declines: 34%) and provider
    → Feature Pipeline (72 raw cols → 35 engineered features):
        Provider (4): name, segment approval rate, decline rate, retry success rate
        Geographic (4): country, currency, cross-border flag, region
        Card (3): brand, type, BIN prefix
        Amount (3): raw amount, bucket, deviation from merchant average
        Decline (5): code, category, soft/hard, code retry rate, code retry rate by country
        Temporal (4): hour, day of week, business hours, minutes since last attempt
        Merchant (4): approval rate, retry success rate, chargeback rate, avg amount
        Sequence (3): attempt number, time since first, decline code changed
    → Temporal Train/Val/Test Split (70/15/15 by date — no random split)

Training:
    → Great Expectations: validate feature distributions before training
    → LightGBM with Optuna (20 trials, TPE sampler):
        Objective: maximize net_retry_value on validation set
        Search: num_leaves, learning_rate, min_child_samples, subsample, colsample, regularization
        5-fold temporal cross-validation within training set
    → Isotonic Regression Calibration (well-calibrated probabilities for business decisions)
    → MLflow: log every trial (params, metrics, model artifact, SHAP plots)

Prediction:
    → Input: failed transaction context
    → Redis feature cache check (hit? return cached features)
    → Feature extraction (35 features) → cache in Redis (TTL 1hr)
    → LightGBM inference → calibrated probability
    → Decision Engine:
        net_retry_value = P(success) × amount × margin - P(failure) × retry_cost - friction_cost
        > $0.50 → "retry" | < -$0.10 → "abandon" | between + cascade available → "cascade"
    → Explanations (both methods):
        SHAP TreeExplainer → top 3 feature contributions
        LIME local surrogate → independent explanation
        Business translator → plain English for both
    → FastAPI response: {decision, confidence, net_retry_value, shap_explanations[], lime_explanations[]}

Event-Driven:
    → Kafka consumer: subscribe to payment.declined events from Payment Routing Simulator
    → Redis dedup check (skip if already processed)
    → For each event: extract features → predict → publish to payment.decisions topic
    → Champion/challenger: route traffic between current model and candidate model

Monitoring:
    → Drift detection: PSI (Population Stability Index) on feature distributions
    → KS test: per-feature distribution shift detection
    → Alert if PSI > 0.2 (moderate drift) or > 0.25 (significant drift)
    → Prometheus: prediction_count, latency_histogram, confidence_distribution, drift_score_gauge, cache_hit_rate
    → Grafana dashboards: prediction volume, confidence distribution, drift PSI gauge, champion vs challenger, latency percentiles
```

### Key files to create

| File | What it does |
|------|-------------|
| `notebooks/01_exploration.ipynb` | EDA with DuckDB: column profiling, distributions, target variable analysis |
| `data/target.py` | `create_target(df)`: filters to declined txns, simulates retry outcomes |
| `data/split.py` | Temporal train/val/test split (70/15/15 by timestamp) |
| `data/validation.py` | Great Expectations suite: validate feature ranges, distributions, completeness |
| `features/standard.py` | 30 feature functions organized by group (provider, geographic, card, amount, decline, temporal, merchant, sequence) |
| `features/pipeline.py` | `build_features(df)` — orchestrates all feature functions → 35-column matrix |
| `models/lightgbm_model.py` | `train_lightgbm(X, y, X_val, y_val, params)` with early stopping, MLflow logging |
| `tuning/optuna_search.py` | `tune_lightgbm(n_trials=20)` — TPE sampler, maximizes net_retry_value |
| `models/calibration.py` | Isotonic regression calibration, reliability diagram, ECE calculation |
| `explanations/shap_engine.py` | SHAP TreeExplainer → top 3 contributions per prediction |
| `explanations/lime_engine.py` | LIME local surrogate → independent explanation per prediction |
| `explanations/business_translator.py` | Translate SHAP/LIME values to plain English (15-20 templates) |
| `explanations/decision.py` | `make_decision(probability, transaction)` → retry/cascade/abandon + net_retry_value |
| `simulation/harness.py` | Replay 10K transactions under 5 strategies, output comparison |
| `monitoring/drift.py` | PSI + KS test on feature distributions, alert thresholds |
| `monitoring/champion_challenger.py` | A/B model comparison: route traffic, compare metrics |
| `kafka/consumer.py` | Consume `payment.declined` events from Kafka |
| `kafka/producer.py` | Publish `payment.decisions` (retry/cascade/abandon) to Kafka |
| `api.py` | FastAPI: POST /predict, POST /batch, GET /model-card, GET /feature-importance, GET /health, GET /metrics |
| `Dockerfile` | Multi-stage build (training image + slim inference image) |
| `docker-compose.yml` | API + MLflow server + Redis + Grafana |
| `.github/workflows/ci.yml` | CI/CD: lint → test → train → validate → deploy if better |
| `cache/feature_cache.py` | Redis-backed feature cache — avoid recomputing features for same transaction |
| `cache/dedup.py` | Kafka consumer deduplication — skip already-processed events |
| `terraform/` | Terraform configs for DigitalOcean deployment |
| `grafana/dashboards/` | Pre-built Grafana dashboard JSON for model monitoring |
| `tests/` | pytest: feature pipeline, model prediction, API endpoints, drift detection |

### Data layer

Target variable does NOT exist in the CSV. You engineer it:
1. Filter to declined transactions (where `transaction_status` = "DECLINED")
2. For each declined transaction, determine if a retry would succeed:
   - Look at `iso8583_response_code`: soft decline codes (05, 51, 61, 65) get 20-40% success probability, hard decline codes (14, 62, 54, 04) get 0-5%
   - Modify by country: Brazil soft declines succeed at 34%, Germany at 12%
   - Modify by provider: some providers have better retry acceptance
3. Random draw per transaction → `retry_success` = 0 or 1

This gives a realistic but synthetic retry outcomes dataset. Target rate should land around 20-40% positive.

### Key decisions

- **LightGBM only, not stacking ensemble**: Stacking (LightGBM + XGBoost + MLP + meta-learner) was descoped. On 10K rows, a single well-tuned LightGBM matches ensemble performance within 1-2% AUC while being far simpler to deploy, explain, and maintain. In interviews, discuss stacking as "what I'd add at scale" — shows you know the technique without over-engineering.
- **DuckDB for feature engineering, not just pandas**: DuckDB runs SQL directly on CSV/DataFrames and is 10-100x faster for aggregations. Feature engineering involves many GROUP BY operations — DuckDB handles these efficiently. Also demonstrates modern data tooling knowledge.
- **Both SHAP and LIME**: Two independent explanation methods. SHAP gives exact game-theoretic feature contributions; LIME builds a local linear model around each prediction. When they agree, confidence is high. When they disagree, it's a signal to investigate. Interviewers always ask "SHAP vs LIME — when do you use each?"
- **Custom metric (net_retry_value), not log-loss**: The key differentiator. The model optimizes business outcomes — a $500 transaction with 40% retry success is worth retrying, but a $5 transaction with the same probability isn't. Log-loss treats all errors equally.
- **Optuna 20 trials, not 100**: 10K rows train fast. 20 trials with TPE (Tree-structured Parzen Estimator) is efficient — it learns which hyperparameter regions are promising. 100 trials is overkill at this dataset size.
- **Temporal split, not random**: Random split leaks future information into training. Real-world deployment sees only past data. Temporal split matches production conditions.
- **MLflow for experiment tracking**: Every training run logged — hyperparameters, metrics, model artifacts, SHAP plots. Can compare runs side-by-side. Model registry manages staging → production promotion.
- **Kafka consumer for real-time predictions**: The ML model subscribes to `payment.declined` events from the Payment Routing Simulator. This creates a genuine event-driven ML system, not just a batch model — a strong differentiator in ML interviews.
- **Great Expectations for data validation**: Check data quality BEFORE the pipeline processes it. Catches schema changes, missing values, distribution shifts early — before they corrupt the model.
- **Champion/challenger for model comparison**: New models don't replace the current one automatically. Traffic is split — compare performance on live predictions before promoting. Same pattern Netflix and Stripe use.
- **Isotonic calibration, not Platt scaling**: Isotonic regression handles non-linear calibration curves better than Platt's sigmoid. Payment retry probabilities are not sigmoid-shaped.
- **Redis for feature caching**: Computing 35 features involves multiple lookups (merchant stats, provider stats, historical rates). Cache the computed feature vector for each transaction — if the same transaction is retried or re-evaluated, features are ready instantly. Also used by the Kafka consumer to deduplicate events.
- **Terraform for deployment**: Same infrastructure-as-code approach as the Simulator. The ML engine's deployment is more complex (API + MLflow + Kafka consumer as separate services), making Terraform even more valuable.
- **Grafana for model monitoring**: Custom dashboards showing prediction confidence distribution over time, drift scores (PSI), model latency, champion vs challenger performance. Connects to the Prometheus metrics the API already exports.

## Key Features

### 1. Feature Engineering (35 features)
Extracted from transaction context:

**Standard features (30)**:
- **Provider features**: provider name, provider historical approval rate for segment
- **Geographic**: country, currency, cross-border flag, region
- **Card features**: brand, type (credit/debit), BIN-level features
- **Amount features**: amount bucket, deviation from merchant average
- **Decline context**: decline code, decline category (issuer/network/processor), soft vs hard
- **Temporal**: hour of day, day of week, minutes since last attempt
- **Merchant history**: merchant approval rate, merchant retry success rate, merchant chargeback rate
- **Sequence features**: attempt number, time since first attempt, decline code pattern

### 2. LightGBM Model
Single gradient-boosted tree model, tuned via Optuna:
- Objective: maximize net_retry_value (custom business metric), not log-loss
- Hyperparameters tuned: num_leaves (15-63), learning_rate (0.01-0.3), min_child_samples (5-100), subsample (0.6-1.0), colsample_bytree (0.6-1.0), reg_alpha (0-5), reg_lambda (0-5)
- 20 Optuna trials with TPE sampler
- 5-fold temporal cross-validation within training set
- Early stopping (patience=50) on validation net_retry_value
- Probability calibration via isotonic regression

### 3. Custom Business Metric: net_retry_value
```
net_retry_value = P(success) * transaction_amount * margin
                - P(failure) * retry_cost
                - P(failure) * customer_friction_cost
```
- `retry_cost`: processor fee per attempt (~$0.05-0.30 depending on provider)
- `customer_friction_cost`: estimated revenue loss from customer abandonment during retry delay
- Decision threshold tuned on Pareto frontier (recovery rate vs retry cost)
- Calibrated probabilities ensure threshold decisions are economically optimal

### 4. Dual Explanations: SHAP + LIME
Every prediction includes explanations from two independent methods:

**SHAP (TreeExplainer)**:
- Exact game-theoretic feature contributions based on Shapley values
- Fast for tree models (polynomial, not exponential)
- Global summary: beeswarm plot showing all features ranked by importance
- Local: top 3 contributing features per prediction with direction and magnitude

**LIME (Local Interpretable Model-agnostic Explanations)**:
- Builds a simple linear model around each prediction point
- Perturbs input features, observes prediction changes, fits local linear approximation
- Independent from SHAP — uses a completely different methodology
- Useful cross-check: when SHAP and LIME agree, explanation is robust

**Business Language Translation**:
Both SHAP and LIME outputs are translated to plain English via template strings:
- "Decline code 05 (Do Not Honor) in Brazil has 34% historical retry success — above the 25% threshold"
- "Transaction at 3:00 AM — retry success peaks at 10:00 AM (28%), consider delaying retry"
- "Amount $450 is 3x above merchant average — high-value retries succeed at 15% vs 28% overall"

### 5. MLflow Experiment Tracking
- Every training run logged: hyperparameters, all metrics (AUC, precision, recall, net_retry_value, calibration error), model artifact, SHAP plots
- Model registry: staging → production lifecycle
- Compare runs: tuned vs default, feature ablation studies
- Artifact storage: trained models, calibration curves, feature distributions

### 6. Decision API (FastAPI)
- `POST /predict` — input transaction context → `{decision, confidence, net_retry_value, shap_explanations[], lime_explanations[], cascade_target}`
- `POST /batch` — batch predictions for simulation
- `GET /model-card` — model version, training date, performance metrics, feature list
- `GET /feature-importance` — global SHAP summary
- `GET /health` — API health + model loaded check
- `GET /metrics` — Prometheus format metrics

### 7. Kafka Integration
- **Consumer**: subscribes to `payment.declined` events from the Payment Routing Simulator's Kafka topic
- **Processing**: for each declined event → extract features → predict → generate explanation
- **Producer**: publishes `{transaction_id, decision, confidence, retry_value, explanation}` to `payment.decisions` topic
- Graceful degradation: if Kafka is unavailable, API still works for direct requests

### 8. Simulation Harness
Replays all 10K transactions comparing 5 strategies:
- **Always-retry**: Retry every soft decline up to 3 times
- **Never-retry**: Accept first decline as final
- **Rules-based**: Industry-standard rules (retry soft declines, cascade on timeout, abandon hard declines)
- **LightGBM (this model)**: ML-driven decisions using calibrated probabilities
- **Random baseline**: Random retry/abandon with same retry rate as model

Output: total recovered revenue, total retry cost, net value, lift over baseline per strategy.

### 9. Data Validation (Great Expectations)
- Validate input data before pipeline processes it
- Check: no missing values in critical columns, feature ranges within expected bounds, decline code distribution hasn't shifted dramatically, country codes are valid ISO 3166-1
- Run automatically before each training run and on each Kafka event batch
- Failures: log warning + skip invalid records (don't crash the pipeline)

### 10. Drift Detection
- **PSI (Population Stability Index)**: compare feature distributions between training data and recent production data
- **KS test**: per-feature distribution shift detection
- Alert thresholds: PSI > 0.2 (moderate drift, investigate) → PSI > 0.25 (significant, consider retraining)
- Runs on a sliding window of recent predictions
- Results logged to MLflow + Prometheus drift_score gauge

### 11. Champion/Challenger A/B Testing
- Current deployed model = "champion"
- Newly trained model = "challenger"
- Traffic split: 80% champion / 20% challenger
- Compare: net_retry_value, precision, recall, calibration error on live predictions
- Promote challenger to champion only if it outperforms on net_retry_value with statistical significance
- All comparison results logged to MLflow

### 12. CI/CD Pipeline (GitHub Actions)
1. **On every push**: ruff lint → pytest (unit tests + model smoke test on 100 samples)
2. **On merge to main**: full training run → evaluate on test set → compare to current champion
3. **Promotion gate**: new model must beat champion on net_retry_value by >1% AND calibration error < 0.05
4. **On promotion**: build Docker image → push to registry → deploy to DigitalOcean
- Model versioning: Git SHA + MLflow run ID

## Build Plan

### Session 1 (2h): EDA + target variable + DuckDB + MLflow setup
**Build**:
1. Create repo `ivanpayments/ml-payment-recovery`, init venv, install all dependencies
2. DuckDB setup: load CSV, run SQL queries for column profiling (types, nulls, distributions, cardinality)
3. `notebooks/01_exploration.ipynb`: approval/decline split by provider, response code frequency, country distribution, amount distribution
4. `data/target.py`: `create_target(df)` — filter to declined txns, assign `retry_success` based on decline code (soft 20-40%, hard 0-5%) × country modifier × provider modifier, random draw
5. `data/split.py`: sort by `transaction_created_at`, train 70% / val 15% / test 15% — no random split, no date overlap
6. MLflow setup: create "payment-recovery" experiment, log first run with dataset stats + target distribution

**Done when**: Target variable created (~25-35% positive rate). Temporal split verified. DuckDB queries work. MLflow UI shows first logged run.

---

### Session 1.5 (2h): dbt analytics layer — **Ivan drives, Claude holds back**

**Why this session exists**: The ML pipeline uses Python feature engineering (Session 2) because that's the right tool for model input. But the **chatbot / dashboard / merchant-facing reporting layer** needs analytics SQL over the same data — "which merchants had the worst decline rates last week", "approval rate by BIN country", "decline reason distribution by PSP". That's what dbt is for.

This session builds a **mini modern-data-stack** (DuckDB + dbt) so Ivan has hands-on experience with the tool every fintech data team uses. Orthogonal to the ML pipeline — doesn't interfere with feature engineering.

**Scope discipline**: 3 models, 3 tests, docs. Do NOT convert Session 2's feature pipeline to dbt.

**Ivan's task list (Ivan drives, Claude answers questions)**:

1. **Install dbt**: `pip install dbt-duckdb`. Run `dbt init ml_recovery_analytics` to scaffold a project. Point it at the same DuckDB file Session 1 created.

2. **Model 1 — staging (`models/staging/stg_transactions.sql`)**: one `SELECT` that reads the raw synthetic_transactions table and does basic cleanup — cast types, rename columns to `snake_case`, filter invalid rows. Staging = "clean but not transformed".

3. **Model 2 — intermediate (`models/intermediate/int_decline_enriched.sql`)**: filter to declined transactions only, join against a decline-code reference table (build this as a seed file — `seeds/decline_codes.csv`, ~20 rows). Result: one row per declined transaction enriched with decline-reason text + severity.

4. **Model 3 — mart (`models/mart/mart_merchant_decline_rates.sql`)**: aggregate by merchant × week → approval rate, decline rate, top-3 decline reasons. This is the table the chatbot's "reporting" tools query.

5. **Add 3 tests** in a schema YAML file:
   - `stg_transactions.tx_id` is `unique` and `not_null`
   - `int_decline_enriched.decline_code` has `accepted_values` = list of valid codes
   - `mart_merchant_decline_rates.approval_rate` has a custom test: `approval_rate BETWEEN 0 AND 1`

6. **Generate docs**: `dbt docs generate && dbt docs serve`. Open the lineage graph in the browser — confirm the DAG shows: seed + raw → staging → intermediate → mart. Screenshot the lineage graph for the interview deck.

7. **Wire to chatbot**: add one tool to the chatbot: `get_merchant_decline_rates(week_start)` → SELECTs from `mart_merchant_decline_rates`. The chatbot now answers merchant-facing reporting questions via dbt marts + ML questions via the LightGBM API.

8. **Write a 1-page comparison note** (`docs/dbt_vs_python_transforms.md`): when you'd use dbt vs Python for transforms. Rule of thumb: dbt for shared analytics marts consumed by multiple tools (BI, chatbot, dashboard); Python for ML feature pipelines where vectorized pandas/numpy is faster and the consumers are model code, not SQL.

**Claude's role in this session**:
- Do NOT write dbt models for Ivan.
- DO explain the staging/intermediate/mart convention when asked.
- DO answer "what does `{{ ref() }}` actually do under the hood" (it compiles to a fully-qualified table name in the right schema for the current target).
- DO review Ivan's SQL and flag mistakes.
- DO point at dbt docs URLs, not summaries.

**Done when**:
- `dbt run` executes all 3 models in dependency order; warehouse shows 3 tables + 1 seed.
- `dbt test` passes all 3 tests.
- `dbt docs serve` shows a lineage graph; screenshot saved.
- Chatbot's `get_merchant_decline_rates(week_start)` tool returns real data from the mart.
- `docs/dbt_vs_python_transforms.md` exists.

**Interview leverage**: "Yes, I've built dbt models end-to-end with tests and lineage" is a strong PMT / TPM answer at any fintech with a data team. The lineage screenshot goes in the project deck.

**Stop-points (do NOT expand scope)**:
- No Fivetran / Airbyte / extract layer. CSV → DuckDB is the load.
- No incremental models. Full refresh is fine at this scale.
- No custom macros. Standard dbt is enough.
- Do NOT convert Python feature pipeline to dbt. That's wrong tool for wrong job.

---

### Session 2 (2h): Feature engineering pipeline
**Build**:
1. `features/standard.py`: all 30 feature functions organized by group (provider 4, geographic 4, card 3, amount 3, decline 5, temporal 4, merchant 4, sequence 3)
2. Each function: takes DataFrame → returns DataFrame with new column(s). Pure functions, independently testable.
3. `features/pipeline.py`: `build_features(df)` — applies all functions in order, drops raw columns, returns 35-column feature matrix
4. Log feature statistics to MLflow: means, medians, correlations with target, cardinality
5. pytest: test each feature function on a small fixture DataFrame

**Done when**: `build_features(train_df)` returns DataFrame with 35 columns. No NaN. All types correct. Feature stats logged in MLflow.

---

### Session 3 (2h): LightGBM baseline + evaluation + MLflow
**Build**:
1. `models/lightgbm_model.py`: `train_lightgbm(X_train, y_train, X_val, y_val, params)` with early stopping (patience=50), logs params + metrics + model to MLflow
2. `evaluation/metrics.py`: `evaluate(y_true, y_pred_proba, amounts)` returns: AUC-ROC, precision at various thresholds, recall, F1, net_retry_value (summed across all decisions), Brier score, 10-bin calibration curve
3. Train with default params: `objective="binary"`, `num_leaves=31`, `learning_rate=0.1`
4. Compare to baselines: always-retry (retry all soft declines), never-retry, random
5. Per-segment evaluation: metrics by country, provider, decline code (check for geographic bias)
6. Log everything to MLflow

**Done when**: LightGBM AUC > 0.70. net_retry_value is positive (model adds value over baselines). MLflow run shows all metrics.

---

### Session 4 (2h): Optuna tuning + calibration
**Build**:
1. `tuning/optuna_search.py`: search space for num_leaves, learning_rate, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda
2. Objective: maximize net_retry_value on validation set
3. TPE sampler, 20 trials, each with 5-fold temporal cross-validation
4. Retrain with best parameters on full training set
5. `models/calibration.py`: isotonic regression calibration, before/after reliability diagram, expected calibration error (ECE)
6. Log all 20 trials + final tuned model + calibration artifacts to MLflow

**Done when**: Tuned model AUC improves over default. Calibration ECE < 0.05. MLflow shows all 20 trial runs.

---

### Session 5 (2h): SHAP + LIME explanations + business translator
**Build**:
1. `explanations/shap_engine.py`: `explain_prediction(model, X_single)` → SHAP TreeExplainer → top 3 contributions. Output: `[{feature, value, shap_value, direction}]`
2. `explanations/lime_engine.py`: `explain_prediction_lime(model, X_single, X_train)` → LIME explainer → top 3 features. Output: `[{feature, value, lime_weight, direction}]`
3. `explanations/business_translator.py`: `translate(explanations, method)` → plain English strings. 15-20 templates per feature type (decline_code, hour_of_day, amount, provider, attempt_number, etc.)
4. Comparison function: `compare_explanations(shap_result, lime_result)` → agreement score (do top features match?), flagging when methods disagree
5. `explanations/decision.py`: `make_decision(probability, transaction)` → retry/cascade/abandon + net_retry_value + cascade_target + both explanation sets
6. Global SHAP beeswarm plot (PNG for portfolio), log to MLflow

**Done when**: 10 sample predictions each get explanations from BOTH SHAP and LIME. Business translator produces readable English. Agreement score computed.

---

### Session 6 (2h): Decision engine + FastAPI API
**Build**:
1. Decision logic: net_retry_value calculation with configurable thresholds (> $0.50 retry, < -$0.10 abandon, between → cascade if available)
2. Cascade target selection: pick provider with highest predicted approval rate for this transaction's characteristics
3. `api.py`: `POST /predict` → `{decision, confidence, net_retry_value, shap_explanations[], lime_explanations[], cascade_target}`
4. `POST /batch` → list of predictions
5. `GET /model-card` → model version, metrics summary, feature list, known limitations
6. `GET /feature-importance` → global SHAP summary (top 15 features)
7. `GET /health`, `GET /metrics`
8. Pydantic request/response models with validation
9. Edge cases: hard decline codes → always "abandon" (skip model), unknown provider → use average stats
10. **Ship a thin Python SDK wrapping the API** — `recovery_sdk/` package with a `RecoveryClient` class exposing `.predict(...)`, `.batch(...)`, `.feature_importance()`, `.model_card()`; typed Pydantic response objects; retry + timeout + bearer-token auth; minimal README with "install + 5-line example". Publish to TestPyPI (not real PyPI) so the `pip install` story is real. Ivan's task: define the public SDK surface and write the README before any code — forces you to think like an API consumer, not a builder. This is the artifact you'll point to in SE / PMT interviews when asked "have you designed a developer-facing API".

**Done when**:
- `curl -X POST /predict -d '{"decline_code":"05", "provider":"stripe", "country":"BR", "amount":150}'` returns decision with both SHAP and LIME explanations.
- `pip install -i https://test.pypi.org/simple/ recovery-sdk` works, and a 5-line script (`RecoveryClient(api_key=...).predict(...)`) returns the same response. README renders cleanly on TestPyPI.

---

### Session 7 (2h): Kafka consumer + event-driven predictions
**Build**:
1. `kafka/consumer.py`: subscribe to `payment.declined` events from Payment Routing Simulator
2. For each event: parse payload → extract features → run prediction → generate explanations
3. `kafka/producer.py`: publish prediction result to `payment.decisions` topic: `{transaction_id, decision, confidence, net_retry_value, explanations, cascade_target}`
4. Redis dedup check: store processed event IDs in Redis (TTL 24hr), skip already-processed events
5. Graceful degradation: if Kafka unavailable, log warning and continue (API still works for direct requests)
6. Consumer runs as separate process/thread alongside FastAPI
7. Batch processing option: accumulate events, predict in batches for efficiency

**Done when**: Start Kafka consumer → start Payment Routing Simulator → simulate a declined payment → ML engine consumes event → prediction appears in payment.decisions topic within 2 seconds. Duplicate events are skipped.

---

### Session 7.5 (2h): Webhook ingestion — **Ivan drives, Claude holds back**

**Why this session exists**: A second ingestion path alongside Kafka. Kafka is internal-service-to-service; webhooks are how **external PSPs** (Stripe, Adyen) notify you of events. Fintechs expect you to have handled both. This is a learning session for Ivan, not a build session for Claude.

**Ivan's task list (do these yourself, ask Claude only when stuck)**:
1. **Design the endpoint**: pick the URL path (e.g. `POST /webhooks/stripe`), decide what events to accept (`charge.failed`, `payment_intent.payment_failed`), write a short design note before touching code.
2. **Write down — in plain English — the 5 threats a webhook endpoint faces** (spoofing, replay, flood, duplicate, slow handler). Then design a mitigation for each. This is the piece interviewers probe on.
3. **Implement signature verification yourself**: read Stripe's docs on `Stripe-Signature`, extract `t=` and `v1=`, compute the HMAC, constant-time compare. No Claude code — use Stripe's docs + stdlib `hmac` + `hashlib`.
4. **Implement idempotency**: use the event ID from Stripe's payload as the dedup key, store in Redis with 48hr TTL. Reject duplicates with `200 OK` (NOT an error — retries must be idempotent).
5. **Implement fast ack + async processing**: the HTTP handler must return `200` within 5s. Heavier work (feature extraction + prediction + publish to `payment.decisions`) happens in a background task. Why: Stripe retries if you don't ack fast.
6. **Test retries**: use the Stripe CLI (`stripe listen`, `stripe trigger charge.failed`) against your local endpoint. Deliberately return 500 once and confirm Stripe retries with backoff.
7. **Write a 1-page runbook**: what to do when webhook processing falls behind, when signature verification fails in bulk, when Stripe retries pile up.

**Claude's role in this session**:
- Do NOT implement the endpoint, signature verification, or dedup logic.
- DO answer conceptual questions: "what's HMAC doing here?", "why constant-time compare?", "why return 200 on a duplicate?"
- DO review Ivan's code once he's written it and flag mistakes.
- DO point at Stripe docs URLs rather than paste Python.

**Done when**:
- `stripe trigger charge.failed` hits `POST /webhooks/stripe`, signature validates, event is deduped in Redis, background task runs a prediction, result appears in `payment.decisions` topic.
- Ivan can whiteboard the request lifecycle and the 5 threats cold — because he built it, not because he read it.
- 1-page runbook exists in `docs/webhook_runbook.md`.

**Interview leverage**: this is one of the top-5 things an SE / PMT will be asked to walk through at a payments fintech. *"Walk me through how you'd set up webhook ingestion from a PSP."* After this session you have a real, honest answer.

---

### Session 8 (2h): Simulation harness
**Build**:
1. `simulation/harness.py`: replay all 10K transactions under 5 strategies (always-retry, never-retry, rules-based, LightGBM model, random baseline)
2. For each strategy: total recovered revenue, total retry cost, net value, customer friction estimate, false positive rate (retried but failed)
3. Output: comparison table (markdown + CSV) + bar chart (net value per strategy)
4. Calculate lift: "ML model recovers X% more revenue at Y% lower retry cost vs rules-based"
5. Per-segment breakdown: which strategy wins for which country/provider/decline code
6. Log simulation results + charts to MLflow

**Done when**: Simulation runs on full 10K dataset. ML model shows positive lift over rules-based. Comparison table + chart saved. Results in MLflow.

---

### Session 9 (2h): Redis feature cache + data validation (Great Expectations)
**Build**:
1. Redis connection setup (async redis-py)
2. `cache/feature_cache.py`: cache computed feature vectors by transaction_id (TTL 1hr)
3. Cache-first pattern in prediction pipeline: check Redis → compute if miss → cache result
4. `cache/dedup.py`: Kafka consumer deduplication — store processed event IDs in Redis (TTL 24hr)
5. `data/validation.py`: Great Expectations suite — validate feature completeness, ranges, distributions
6. Run validation before training and on Kafka event batches
7. Failures: log warning + skip invalid records

**Done when**: Feature cache hit rate measurable. Great Expectations validates training data. Dedup skips duplicate events.

---

### Session 10 (2h): Drift detection + champion/challenger
**Build**:
1. `monitoring/drift.py`: PSI + KS test on feature distributions. Alert thresholds.
2. `monitoring/champion_challenger.py`: load two models, 80/20 traffic split, compare metrics
3. Promotion logic: challenger promoted if net_retry_value improvement > 1% AND statistically significant
4. Log all comparisons to MLflow

**Done when**: Drift detection triggers on synthetic shifted data. Champion/challenger routing works.

---

### Session 11 (2h): Docker + Terraform + Grafana
**Build**:
1. `Dockerfile`: multi-stage (training image + slim inference)
2. `docker-compose.yml`: api (FastAPI), mlflow (tracking server), redis (feature cache), grafana
3. Terraform: DigitalOcean droplet, firewall rules, DNS record, systemd service definition
4. Grafana dashboards: prediction volume over time, confidence distribution histogram, drift PSI gauge, champion vs challenger comparison, latency percentiles
5. Grafana connects to Prometheus metrics endpoint

**Done when**: `docker-compose up` starts API + MLflow + Redis + Grafana. Grafana dashboards load. Terraform plan shows valid config.

---

### Session 12 (2h): CI/CD + observability
**Build**:
1. `.github/workflows/ci.yml`: on push → lint + test + smoke. On merge → full train → evaluate → compare → promote → build Docker → deploy
2. Structured JSON logging: every prediction with request_id, features, decision, confidence, latency
3. Prometheus /metrics: prediction_total, latency_histogram, confidence_distribution, drift_psi_gauge, cache_hit_rate
4. OpenTelemetry: instrument FastAPI + Kafka consumer + model inference. Trace: API request → feature cache check → feature extraction → model predict → SHAP/LIME explain → response.

**Done when**: CI passes. Logs are JSON. /metrics returns Prometheus format. Traces visible.

---

### Session 13 (2h): Model card + packaging + deploy
**Build**:
1. Model card (Google format): intended use, training data, model, performance metrics, known limitations, geographic bias audit, card brand parity
2. Package as installable Python module
3. Deploy to DigitalOcean via Terraform: systemd service (port 8084), Caddy proxy at `/ml-recovery/*`
4. Start Kafka consumer as separate systemd service
5. README: quickstart, architecture diagram, simulation results, model card, "Try it live"
6. Push to GitHub

**Done when**: Package installs. API live. Kafka consumer running. Model card complete. README with screenshots.

---

### Session 14 (1h): QA + chatbot integration
End-to-end: prediction accuracy, SHAP + LIME agreement, Kafka event processing, Redis cache, drift detection, Great Expectations, champion/challenger, Grafana dashboards, API edge cases, Docker stack restart. Fix bugs. Screenshots.

**Chatbot integration**:
1. Add `POST /predict-recovery` endpoint (if not already exposed) — input transaction context → output `{decision, confidence, net_retry_value, shap_explanations, business_language}`
2. Add `predict_decline_recovery` tool function in Payment Data Chatbot (`tools.py`) that HTTP-calls this endpoint
3. Add batch variant: `predict_batch_recovery(date_range)` — chatbot fetches all declines from PostgreSQL, calls the predictor, ranks by expected recovery value, returns top-N as markdown table with per-row SHAP reasoning
4. Update chatbot system prompt to route retry / decline / recovery questions to these tools
5. End-to-end verification: "Which of yesterday's declines should I retry?" → chatbot returns ranked table with dollar recovery estimates and SHAP reasons

---

### Session 14.5 (2.5h): RAG layer for decline-code + troubleshooting context — **Ivan drives, Claude holds back**

**Why this session exists**: The model predicts *whether* to retry. It can't explain *why decline code 51 means this specific thing for a Brazilian issuer* or *what the merchant playbook says to do for this decline pattern*. That context lives in docs — PSP decline-code reference, internal troubleshooting notes, past incident write-ups. RAG (Retrieval-Augmented Generation) is how you expose that knowledge to the chatbot without retraining the model.

**Interview leverage**: RAG is one of the most-asked AI patterns in 2026 PMT / SE / AI-PM interviews. You should be able to whiteboard ingestion → embedding → vector store → retrieval → prompt construction, and name the gotchas. Only way to get that is to build one yourself.

**Ivan's task list (do these yourself, ask Claude only when stuck)**:
1. **Scope the corpus** — decide what goes in: PSP decline-code reference (build this as a ~100-row CSV from public docs), internal troubleshooting playbook (write 15–20 short entries yourself), historical SHAP explanation glossary. Write a 1-page corpus scope doc before any code.
2. **Choose chunking strategy**: fixed-size (e.g. 500 tokens) vs semantic (split on sentence/section). Write down your choice + why, in plain English. Gotcha: too-small chunks lose context; too-large chunks dilute retrieval.
3. **Choose the embedding model**: OpenAI `text-embedding-3-small`, Cohere `embed-english-v3`, or a local one. Pick ONE, justify in 2 sentences (cost, dimensions, quality trade-off).
4. **Choose the vector store**: `pgvector` (Postgres extension — you already have Postgres), Pinecone (managed), Chroma (local). Pick ONE, justify. Reason for `pgvector` in this project: you already run Postgres, no new infra.
5. **Implement ingestion yourself**: read corpus → chunk → embed → upsert into vector store. Script name: `rag/ingest.py`. Include a `--reingest` flag for when you change chunking.
6. **Implement retrieval yourself**: given a query, embed it, fetch top-K chunks, return with similarity scores. Script: `rag/retrieve.py`. Gotcha you must handle: metadata filters (e.g. only retrieve chunks tagged `psp=stripe` when the query is about a Stripe decline).
7. **Implement the prompt template yourself**: given a user question + top-K chunks, construct the LLM prompt. Include explicit "answer only from the provided context, cite chunk IDs, say 'I don't know' if the context doesn't cover it". This is the anti-hallucination constraint.
8. **Wire it to the chatbot**: add a `lookup_decline_context` tool to the chatbot's tool set. The tool takes a decline-related query, calls `retrieve`, returns the top-K chunks + their IDs. The chatbot's LLM then uses those as grounded context.
9. **Evaluate**: build a 20-question eval set (question → expected chunk IDs in top-K, expected answer sketch). Run it, measure retrieval hit-rate@5 and answer quality (manually score). Target: hit-rate@5 ≥ 0.85.
10. **Write a 1-page RAG runbook**: what to do when retrieval misses obvious matches, when the LLM hallucinates despite context, when the corpus changes, when you'd reach for fine-tuning instead.

**Claude's role in this session**:
- Do NOT implement ingestion, retrieval, or the prompt template.
- DO answer conceptual Qs: "why constant-time-compare… no wait, why is cosine similarity the default?", "why does too-small chunking hurt?", "when would I NOT use RAG?"
- DO review Ivan's code once written, flag mistakes.
- DO point at library docs (`pgvector`, `langchain`, `llama_index`) rather than paste code.

**Done when**:
- Ivan can whiteboard the full RAG pipeline cold (ingest → chunk → embed → store → query-embed → retrieve → rerank → prompt → generate).
- `lookup_decline_context("Brazilian card got code 51 on Stripe — what does this mean?")` returns 5 relevant chunks from the decline-code reference + troubleshooting playbook, with similarity scores.
- Chatbot answers: *"Code 51 = insufficient funds. For Brazilian cards via Stripe, the troubleshooting playbook says to retry after 24h because many users are paid monthly. Net retry value model estimates +$0.42 per retry — recommend retry."* — citing both chunks and model output.
- Eval set hit-rate@5 ≥ 0.85 on 20 questions.
- `docs/rag_runbook.md` exists.

**Stop-points (do NOT expand scope)**:
- No reranking (just top-K cosine). Rerankers are a session of their own.
- No hybrid search (BM25 + vector). Pure vector is fine for this corpus size.
- No fine-tuning. RAG is the whole point.
- No graph RAG / knowledge graph. Classic RAG only.

---

### Session 14.9 (3h): AWS sandbox — free-tier only, prerequisite to Session 15

**Why this session exists**: To write Session 15's trade-off doc credibly, Ivan needs hands-on contact with S3, Lambda, DynamoDB, and IAM — not mastery, just vocabulary-with-muscle-memory. This is scoped to ~$0 spend and deleted at the end. Not ongoing infra.

**Hard rules for this session**:
- **Free tier only.** No EC2 with default instance sizes, no RDS, no NAT Gateway, no always-on anything bigger than t3.micro.
- **Billing alarm at $1, $5, $10 — set BEFORE touching any service.** This is the first task, not the last.
- **Tear down at the end.** Sandbox, not infra.
- **No credentials committed to git.** Ever.

**Ivan's task list (Ivan drives — consistent with Project 3 struggle-to-learn rule)**:

1. **Account setup (45 min)**:
   - Create AWS account (needs credit card; Canada address fine).
   - Enable MFA on root user immediately. Log out of root after.
   - Create an IAM user `ivan-dev` with `AdministratorAccess` policy (sandbox only — don't ship this pattern in real projects).
   - Generate access key + secret for `ivan-dev`, save in `~/.aws/credentials`.
   - **Create a billing alarm at $1 via CloudWatch + SNS email.** Test it fires by temporarily lowering threshold to $0.01 if needed. This is non-negotiable.
   - Pick a region: `ca-central-1` (Montreal) — closest to you, data residency-friendly for Canadian fintech interviews.

2. **CLI setup (15 min)**:
   - Install AWS CLI (`pip install awscli` or `choco install awscli`).
   - `aws configure` — paste access key, secret, region, output format (`json`).
   - Verify with `aws sts get-caller-identity` → should return your IAM user ARN.

3. **Hands-on exercise A — S3 (30 min)**:
   - Create bucket `ivan-sandbox-2026-<random-suffix>` (must be globally unique) in `ca-central-1`.
   - Upload `synthetic_transactions.csv` via CLI: `aws s3 cp synthetic_transactions.csv s3://ivan-sandbox-2026-xxx/data/`
   - Block public access (check bucket settings).
   - Generate a presigned URL valid for 10 min: `aws s3 presign s3://.../data/synthetic_transactions.csv --expires-in 600`
   - Open the URL in incognito — confirm download works.
   - Wait 10 min, retry — should 403.
   - Note: S3 costs in this exercise = pennies at most.

4. **Hands-on exercise B — Lambda + API Gateway (45 min)**:
   - Via console: create a Lambda `hello-sandbox` using Python 3.12 runtime.
   - Paste a minimal handler that returns `{"statusCode": 200, "body": "hello from lambda"}`.
   - Add an API Gateway HTTP trigger → get a public URL.
   - `curl` the URL — confirm 200 + body.
   - Check CloudWatch Logs — see the invocation logged.
   - Experiment: return the request's source IP. Understand how `event` is structured (Claude can explain).

5. **Hands-on exercise C — DynamoDB (30 min)**:
   - Create a table `sandbox-transactions` with partition key `tx_id` (String). On-demand billing mode.
   - Via boto3 (Python), insert 5 rows, then `query` by `tx_id`, then `scan` the table.
   - Observe the cost model: first 25GB + 25 WCU/RCU are always-free.

6. **Teardown (20 min)**:
   - Delete the S3 bucket (empty it first).
   - Delete the Lambda + API Gateway route.
   - Delete the DynamoDB table.
   - Verify **Billing → Bills** shows $0 or cents. Wait 24hr, check again.
   - Keep the IAM user + billing alarm. Disable access keys if not used weekly.

7. **Capture (15 min)**: write `docs/aws_sandbox_notes.md` — 1 page: what you did, what the console felt like, the 3 AWS-specific vocab items that are now concrete (bucket policy, presigned URL, API Gateway route), and the 1 thing that surprised you. This note feeds into Session 15.

**Claude's role**:
- Do NOT click through the console for Ivan. Ivan does the clicks.
- DO answer "where is X in the console?" when Ivan is stuck.
- DO explain IAM policy JSON when Ivan reads one.
- DO review the sandbox notes doc for factual accuracy.
- **Hard rule**: if Ivan asks "should I skip the billing alarm just for today" — answer: NO. Every real AWS incident starts that way.

**Done when**:
- All three exercises ran successfully, then fully torn down.
- `docs/aws_sandbox_notes.md` exists.
- Ivan can say (and mean) "I've deployed to S3, Lambda, and DynamoDB in `ca-central-1`."
- Bill is ≤ $0.50 for the month (almost certainly $0).

**Interview leverage**: unlocks the honest "yes, I've worked with AWS hands-on" answer without needing ongoing infra. Combined with Session 15, this is the full AWS story.

---

### Session 15 (2h): AWS managed-services trade-off doc — **Ivan writes, Claude fact-checks**

**Why this session exists**: Every enterprise-fintech interviewer lives in AWS. They will ask "why didn't you use Bedrock / Kendra / SageMaker?". The interview-valuable answer isn't "I didn't think about it" — it's a crisp, numbers-backed comparison showing you evaluated and chose. This doc IS that answer.

**No code in this session. Pure writing + research.**

**Ivan's task list (Ivan writes, Claude fact-checks numbers and flags gaps)**:

1. **Scope the doc**: 1 page max (target 600–800 words). Audience = a technical interviewer at Stripe / Brex / Shopify / Anthropic who knows AWS. Goal = prove you made informed choices, not lazy ones.

2. **Build a decision matrix** covering these 5 AWS services vs your actual choice:

   | Service | Your alternative | Cost at your scale | Lock-in | Learning value | Verdict |
   |---|---|---|---|---|---|
   | Bedrock | Anthropic API direct | | | | |
   | Kendra | pgvector on Postgres | | | | |
   | SageMaker | FastAPI + Docker + MLflow | | | | |
   | OpenSearch (vector) | pgvector on Postgres | | | | |
   | Fraud Detector | LightGBM + SHAP | | | | |

   Fill in real 2026 pricing (check AWS pricing pages). For scale, use your actual corpus size (~120 RAG chunks, 10K transactions, 1 dev + 1 demo env).

3. **Write the "when I would flip" section** — for each service, the scenario where you WOULD use it. Shows you're not anti-AWS, you're context-aware. Example: *"Bedrock makes sense when the rest of the stack is AWS-native and compliance requires data never leaves AWS — e.g. if I were at Brex, I'd flip."*

4. **Write the "what I lose by not using AWS managed" section** — honest trade-offs you accepted: no managed scaling, no IAM-integrated audit logs for model access, you own the runbooks. Interviewers respect candidates who name the costs of their choices.

5. **Add a 3-line summary at the top** so interviewers who skim get the punchline: *"Evaluated 5 AWS managed services. Chose self-managed for 3 reasons: cost at my scale (Kendra entry tier $810/mo vs pgvector $0), learning value (can't speak to what AWS abstracts), portability across non-AWS fintechs. Would flip on managed if: compliance required AWS-resident data, corpus exceeded ~10K docs, or ops headcount > 1."*

**Claude's role in this session**:
- Do NOT write the doc.
- DO verify every cost number Ivan cites against AWS pricing pages (Claude can WebFetch).
- DO flag if a trade-off argument is weak ("you can't say 'lock-in is bad' without naming the concrete portability cost").
- DO point at AWS docs URLs, not summaries.

**Done when**:
- `docs/aws_tradeoffs.md` exists, 1 page, 5-service matrix filled with verified 2026 numbers.
- Ivan can deliver the 30-second oral version cold — without notes — because he wrote it.
- Linked from the main README and the model card's "limitations" section.

**Interview leverage**: turns the hardest architecture question ("why not AWS?") into your strongest answer. Reusable across ~80% of the target-company interview list.

### Deck Sessions
After code ships, deck sessions build company-specific presentations with SHAP waterfall as the hero visual, simulation results, and model card. See schedule.md for dates.

## Model Card (Draft Outline)

**Intended use**: Decision support for payment retry/cascade routing. Not intended for real-time production use without human-in-the-loop validation period.

**Training data**: 10K synthetic transactions, 72 features, 12 merchants, 15 countries, 10 providers. Fully synthetic — no real cardholder data.

**Model architecture**: LightGBM (gradient-boosted decision trees), Optuna-tuned hyperparameters, calibrated via isotonic regression.

**Limitations**:
- Trained on synthetic data; real-world distributions differ
- No issuer-side features (BIN-level only, no issuer response time history)
- Country coverage: 15 countries; extrapolation to uncovered countries not validated

**Fairness analysis**: Geographic bias audit — model performance by country, checking for systematic under-performance in specific regions. Card brand parity check. Provider fairness: ensure no provider is systematically disadvantaged by cascade decisions.

**Performance metrics**: [To be filled after training — target: precision >0.80 at recall >0.65, calibration error <0.03]

## Deliverables

- [ ] GitHub repo with full ML pipeline: data → features → training → evaluation → serving
- [ ] FastAPI endpoint at `ivanantonov.com/ml-recovery/api`
- [ ] LightGBM model with Optuna tuning, SHAP + LIME dual explanations
- [ ] MLflow experiment tracking with model registry
- [ ] Kafka integration: consume declined events, produce retry decisions
- [ ] Simulation results: 5-strategy comparison on 10K transactions
- [ ] Champion/challenger A/B model comparison framework
- [ ] Drift detection (PSI + KS test) with alert thresholds
- [ ] Great Expectations data validation suite
- [ ] Docker-compose: API + MLflow server
- [ ] GitHub Actions CI/CD with model validation gates
- [ ] Model card (Google format): performance, limitations, bias audit
- [ ] Prometheus metrics + structured JSON logging
- [ ] Redis feature cache for prediction performance
- [ ] Terraform infrastructure-as-code for deployment
- [ ] Grafana model monitoring dashboards

## Interview Talking Points

**For SE / AI Engineer roles**:
- "The model optimizes net_retry_value — a custom business metric — not generic accuracy. A $500 transaction with 40% retry success is worth retrying; a $5 transaction with the same probability isn't. That distinction is the entire value of ML here."
- "Every prediction includes dual explanations: SHAP gives exact feature contributions, LIME builds a local linear model. When they agree, the explanation is robust. When they disagree, it's a flag to investigate."
- "The model consumes Kafka events from the Payment Routing Simulator — when a payment is declined, it predicts retry/cascade/abandon within 2 seconds. Real-time ML, not batch."
- "MLflow tracks every experiment — I can show you the 20 Optuna trials, which hyperparameters mattered most, and why the tuned model beats the default by X% on net_retry_value."
- "Redis caches computed feature vectors — the 35-feature extraction involves multiple aggregation lookups, and caching avoids redundant computation on retried transactions."

**For PM roles**:
- "Retry/cascade is the highest-ROI optimization in payments — 2% improvement on $1B GMV recovers $20M annually. This model demonstrates I can think about ML in business-outcome terms, not just accuracy."
- "The simulation harness compares 5 strategies on 10K transactions — I can show you exactly how much more revenue the ML model recovers vs rules-based, and at what cost."
- "Champion/challenger A/B: new models don't auto-deploy. They prove themselves on live traffic first. Same approach Stripe uses for Radar updates."

**For TPM roles**:
- "Full CI/CD for ML: lint + test on every push, full training + evaluation on merge, promotion gate (must beat champion by >1%), Docker build + deploy. Zero manual steps."
- "Drift detection monitors feature distributions against training data. If PSI exceeds 0.2, it flags for investigation before the model degrades."
- "Data validation with Great Expectations catches bad data before it reaches the model — schema changes, missing values, distribution shifts."
- "Infrastructure-as-code with Terraform — the ML engine deployment involves 4 services (API, MLflow, Redis, Kafka consumer), and Terraform manages the entire stack reproducibly."
- "Grafana dashboards monitor model health in real time — prediction confidence drift, latency degradation, champion vs challenger performance. The same monitoring setup production ML teams use."

## Coverage Matrix

| Role | Relevance | Signal |
|------|-----------|--------|
| Engineering-focused Pre-sales | PRIMARY | Full ML pipeline demo + production deployment + customer-facing explanations |
| AI Engineer | PRIMARY | LightGBM + Optuna, MLflow, Kafka, monitoring, CI/CD — production ML engineering |
| TPM | Strong | System architecture, CI/CD pipeline, cross-component orchestration |
| Solutions Engineer | Strong | Retry/cascade is core SE domain; demonstrates analytical depth beyond dashboards |
| Product Manager | Moderate | Custom business metric shows product judgment; simulation harness shows data-driven thinking |

## Data Source

All features and targets derived from `C:\Users\ivana\synthetic_transactions.csv`:
- 10,000 transactions, 72 columns
- 12 fake merchants, 15 countries, 10 fake providers
- Retry outcomes simulated from decline code patterns and provider behavior
- Safe for public repos (fully synthetic)
