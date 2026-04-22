# Project 3 ML Payment Recovery Engine

This folder contains the shipped modeling artifacts plus the serving layer for the retry-recovery model. The current production-style entrypoint is `app.py`, which loads the serialized LightGBM model, enforces the feature-policy allowlist, applies the shipped retry decision threshold of `0.05`, and returns inference-time contribution drivers.

## Live deployment

The API is deployed at `https://ivanantonov.com/recovery/`:

- `GET /recovery/health` — liveness, model version, decision threshold, feature count
- `GET /recovery/model-card` — full model card JSON (training data, performance, segment breakdowns, limitations, ethics, PII posture)
- `POST /recovery/predict` — single-transaction scoring; optional `?include_explanation=true&explanation_depth=N` for SHAP-backed top-N drivers in plain language

```bash
curl https://ivanantonov.com/recovery/health
curl https://ivanantonov.com/recovery/model-card | jq .performance.test
```

## Quickstart

1. Create a Python 3.12 environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the API with `uvicorn app:app --host 0.0.0.0 --port 8000`.
4. Check liveness with `curl http://localhost:8000/health`.
5. Score a payload with the fast path:

```bash
curl -X POST "http://localhost:8000/predict?include_explanation=false" \
  -H "Content-Type: application/json" \
  -d "{\"amount\": 5834.24, \"amount_usd\": 5834.24, \"currency\": \"USD\", \"merchant_vertical\": \"ecom\", \"merchant_mcc\": 5734, \"merchant_country\": \"US\", \"archetype\": \"mid_market_saas\", \"processor_name\": \"global-acquirer-a\", \"routing_reason\": \"highest_auth\", \"card_brand\": \"visa\", \"card_type\": \"credit\", \"card_country\": \"US\", \"card_funding_source\": \"consumer\", \"is_cross_border\": false, \"is_token\": true, \"token_type\": \"network_token\", \"present_mode\": \"cnp\", \"response_code\": \"51\", \"response_message\": \"Insufficient funds\", \"decline_bucket\": \"soft\", \"is_soft_decline\": true, \"scheme_response_code\": \"51\", \"three_ds_requested\": true, \"three_ds_outcome\": \"authenticated\", \"three_ds_version\": 2.2, \"three_ds_flow\": \"frictionless\", \"three_ds_eci\": 5, \"sca_exemption\": \"none\", \"latency_ms\": 317, \"latency_auth_ms\": 317, \"latency_3ds_ms\": 0, \"processor_fee_bps\": 240, \"interchange_estimate_bps\": 180, \"scheme_fee_bps\": 15, \"fx_applied\": false, \"fx_rate\": 1.0, \"settlement_currency\": \"USD\", \"risk_score\": 42, \"fraud_flag\": false, \"risk_model_version\": \"risk_v3\", \"billing_country\": \"US\", \"shipping_country\": \"US\", \"ip_country\": \"US\", \"issuer_country\": \"US\", \"payment_method\": \"card\", \"wallet_type\": \"none\", \"network_token_present\": true, \"entry_mode\": \"ecommerce\", \"pan_entry_mode\": \"manual\", \"cardholder_verification_method\": \"cvv\", \"cvv_result\": \"M\", \"avs_result\": \"Y\", \"avs_zip_match\": true, \"avs_street_match\": true, \"payment_method_details\": \"visa_credit\", \"issuer_bank_country\": \"US\", \"card_product_type\": \"consumer\", \"card_category\": \"consumer\", \"card_commercial_type\": \"no\", \"user_agent_family\": \"mobile_web\", \"timeout_flag\": false, \"device_os\": \"ios\", \"issuer_size\": \"large\", \"account_updater_used\": false, \"mastercard_advice_code\": 0, \"routing_optimized\": true, \"mcc_routing_optimized\": false, \"smart_routed\": true, \"scheme_ms\": 110, \"transaction_type\": \"subscription\", \"fx_bps\": 0, \"routed_network\": \"visa\", \"risk_skip_flag\": false, \"event_hour\": 11, \"event_dayofweek\": 5, \"event_month\": 7, \"is_weekend\": true}"
```

If the caller wants a richer explanation, set `include_explanation=true`. That path is slower by design because it computes a higher-quality SHAP explanation instead of using the latency-optimized fast path.

## Docker

Run the API and Redis locally with:

```bash
docker-compose up --build
```

For a droplet deployment, use:

- `docker-compose.yml` — base service definitions (api + redis)
- `docker-compose.prod.yml` — overlay binding api to `127.0.0.1:8091` so a reverse proxy can sit in front
- `deploy_project3.sh` — tar + scp + `docker compose up --build` on the droplet

The droplet runs Caddy. After deploy, add this snippet to the active server block in `/etc/caddy/Caddyfile` and run `systemctl reload caddy`:

```caddy
handle /recovery {
    redir /recovery/ permanent
}
handle_path /recovery/* {
    reverse_proxy localhost:8091
}
```

The public routes become `/recovery/health`, `/recovery/model-card`, and `/recovery/predict`.

## Architecture

```text
payment event / analyst tool
          |
          v
    FastAPI /predict
          |
          +--> feature-policy allowlist validation
          +--> LightGBM booster + decision policy
          +--> contribution-based explanation
          |
          +--> JSON logs / model-card endpoint
          |
          v
        response

redis
  |
  +--> reserved for future request dedup / feature cache
```

## Repo references

- Product spec: `../ml_payment_recovery_engine.md`
- Model card: `./model_card.md`
- Decision policy evaluation: `./project3_decision_policy_evaluation.md`
- Calibration output: `./project3_lightgbm_calibration_report.md`
- Drift monitoring: `./project3_drift_report.md`
- Champion/challenger reconciliation: `./project3_champion_challenger_report.md`
