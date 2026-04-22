"""Project 3 retry-recovery tool for the payment chatbot.

The chatbot exposes `predict_decline_recovery` as a custom tool. It looks up a
transaction from the local Project 1 CSV, maps overlapping fields into the
Project 3 feature schema, calls the Project 3 FastAPI `/predict` endpoint, and
returns a compact explanation payload that the model can narrate.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import pandas as pd


PROJECT3_API_URL = os.getenv("PROJECT3_API_URL", "http://localhost:8000/predict")
PROJECT3_TIMEOUT_SEC = float(os.getenv("PROJECT3_TIMEOUT_SEC", "15"))
PROJECT3_EXPLANATION_DEPTH = int(os.getenv("PROJECT3_EXPLANATION_DEPTH", "5"))
PROJECT1_TRANSACTIONS_CSV = os.getenv(
    "PROJECT1_TRANSACTIONS_CSV",
    str(Path(__file__).resolve().parent / "Claude files" / "transactions.csv"),
)

PROJECT3_TOOL_SCHEMA: dict[str, Any] = {
    "name": "predict_decline_recovery",
    "description": (
        "Predict whether a declined transaction should be retried using the Project 3 "
        "recovery model. Use this when the user asks questions like 'should I retry "
        "transaction X?', 'is txn_123 recoverable?', or 'what does the retry model say "
        "about this decline?'. Input must be a transaction_id from the Project 1 CSV."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "transaction_id": {
                "type": "string",
                "description": "Transaction identifier from the local payment-attempt dataset.",
            }
        },
        "required": ["transaction_id"],
    },
}


def _err(kind: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"error": True, "error_kind": kind, "error_message": message}
    payload.update(extra)
    return payload


@lru_cache(maxsize=1)
def load_transactions() -> pd.DataFrame:
    path = Path(PROJECT1_TRANSACTIONS_CSV)
    if not path.exists():
        raise FileNotFoundError(f"transactions CSV not found at {path}")
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _boolish(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def _derive_event_fields(created_at: Any) -> dict[str, Any]:
    ts = pd.to_datetime(created_at, errors="coerce", utc=True)
    if pd.isna(ts):
        return {}
    day = int(ts.dayofweek)
    return {
        "event_hour": int(ts.hour),
        "event_dayofweek": day,
        "event_month": int(ts.month),
        "is_weekend": day in (5, 6),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def project3_payload_from_transaction(row: pd.Series) -> dict[str, Any]:
    processing_fee_bps = None
    interchange_bps = None
    scheme_fee_bps = None
    amount_usd = pd.to_numeric(pd.Series([row.get("amount_usd")]), errors="coerce").iloc[0]
    if not pd.isna(amount_usd) and float(amount_usd) > 0:
        processing_fee = pd.to_numeric(pd.Series([row.get("processing_fee_usd")]), errors="coerce").iloc[0]
        interchange_fee = pd.to_numeric(pd.Series([row.get("interchange_fee_usd")]), errors="coerce").iloc[0]
        scheme_fee = pd.to_numeric(pd.Series([row.get("scheme_fee_usd")]), errors="coerce").iloc[0]
        if not pd.isna(processing_fee):
            processing_fee_bps = float(processing_fee) / float(amount_usd) * 10000
        if not pd.isna(interchange_fee):
            interchange_bps = float(interchange_fee) / float(amount_usd) * 10000
        if not pd.isna(scheme_fee):
            scheme_fee_bps = float(scheme_fee) / float(amount_usd) * 10000

    three_ds_requested = bool(str(row.get("three_ds_version") or "").strip())
    three_ds_outcome = row.get("three_ds_status") or ("authenticated" if _boolish(row.get("three_ds_frictionless")) else "not_requested")
    decline_bucket = str(row.get("decline_category") or "unknown").lower()

    payload = {
        "amount": row.get("amount"),
        "amount_usd": row.get("amount_usd"),
        "currency": row.get("currency"),
        "merchant_vertical": row.get("merchant_vertical"),
        "merchant_mcc": row.get("merchant_mcc"),
        "merchant_country": row.get("merchant_country"),
        "archetype": row.get("merchant_tier") or row.get("industry"),
        "processor_name": row.get("processor"),
        "routing_reason": row.get("routing_strategy"),
        "card_brand": row.get("card_brand"),
        "card_type": row.get("card_funding_type"),
        "card_country": row.get("card_country"),
        "card_funding_source": row.get("card_funding_type"),
        "is_cross_border": _boolish(row.get("is_cross_border")),
        "is_token": _boolish(row.get("is_tokenized")),
        "token_type": row.get("token_type"),
        "present_mode": row.get("presence_mode"),
        "response_code": row.get("response_code"),
        "response_message": row.get("response_message"),
        "decline_bucket": decline_bucket,
        "is_soft_decline": decline_bucket == "soft",
        "scheme_response_code": row.get("provider_response_code") or row.get("response_code"),
        "three_ds_requested": three_ds_requested,
        "three_ds_outcome": three_ds_outcome,
        "three_ds_version": row.get("three_ds_version"),
        "three_ds_flow": row.get("authentication_flow"),
        "three_ds_eci": row.get("eci"),
        "sca_exemption": row.get("sca_exemption"),
        "latency_ms": row.get("processing_time_ms"),
        "latency_auth_ms": row.get("provider_latency_ms"),
        "latency_3ds_ms": 0,
        "processor_fee_bps": processing_fee_bps,
        "interchange_estimate_bps": interchange_bps,
        "scheme_fee_bps": scheme_fee_bps,
        "fx_applied": (pd.to_numeric(pd.Series([row.get("fx_rate")]), errors="coerce").fillna(1.0).iloc[0] != 1.0),
        "fx_rate": row.get("fx_rate"),
        "settlement_currency": row.get("settlement_currency"),
        "risk_score": row.get("risk_score"),
        "fraud_flag": _boolish(row.get("is_fraud")),
        "risk_model_version": row.get("risk_provider"),
        "billing_country": row.get("customer_country"),
        "shipping_country": row.get("customer_country"),
        "ip_country": row.get("customer_ip_country"),
        "issuer_country": row.get("card_country"),
        "payment_method": row.get("payment_method_type"),
        "wallet_type": row.get("wallet_provider"),
        "network_token_present": str(row.get("token_type") or "").strip().lower() == "network_token",
        "entry_mode": row.get("channel"),
        "pan_entry_mode": row.get("presence_mode"),
        "cardholder_verification_method": row.get("authentication_flow"),
        "cvv_result": row.get("cvv_result"),
        "avs_result": row.get("avs_result"),
        "avs_zip_match": str(row.get("avs_result") or "").strip().upper() in {"Y", "Z"},
        "avs_street_match": str(row.get("avs_result") or "").strip().upper() in {"Y", "A"},
        "payment_method_details": row.get("payment_method_subtype"),
        "issuer_bank_country": row.get("card_country"),
        "card_product_type": row.get("card_category"),
        "card_category": row.get("card_category"),
        "card_commercial_type": row.get("card_category"),
        "user_agent_family": row.get("device_type"),
        "timeout_flag": _boolish(row.get("is_outage")),
        "device_os": row.get("channel"),
        "issuer_size": row.get("issuer_parent_group"),
        "account_updater_used": _boolish(row.get("account_updater_triggered")),
        "mastercard_advice_code": row.get("merchant_advice_code"),
        "routing_optimized": _boolish(row.get("smart_routing")),
        "mcc_routing_optimized": _boolish(row.get("smart_routing")),
        "smart_routed": _boolish(row.get("smart_routing")),
        "scheme_ms": row.get("provider_latency_ms"),
        "transaction_type": row.get("transaction_type"),
        "fx_bps": row.get("fx_spread_pct"),
        "routed_network": row.get("card_brand"),
        "risk_skip_flag": str(row.get("risk_decision") or "").strip().lower() == "approve",
    }
    payload.update(_derive_event_fields(row.get("created_at")))
    return {
        k: _json_ready(v)
        for k, v in payload.items()
        if v is not None and not (isinstance(v, float) and pd.isna(v))
    }


def call_project3_api(args: dict[str, Any]) -> dict[str, Any]:
    transaction_id = str(args.get("transaction_id", "")).strip()
    if not transaction_id:
        return _err("validation", "transaction_id is required")

    try:
        df = load_transactions()
    except FileNotFoundError as exc:
        return _err("config", str(exc))

    match = df.loc[df["transaction_id"].astype(str) == transaction_id]
    if match.empty:
        return _err("not_found", f"Transaction `{transaction_id}` was not found in the local CSV.")

    payload = project3_payload_from_transaction(match.iloc[0])
    try:
        response = httpx.post(
            PROJECT3_API_URL,
            params={"include_explanation": "true", "explanation_depth": PROJECT3_EXPLANATION_DEPTH},
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=PROJECT3_TIMEOUT_SEC,
        )
    except Exception as exc:
        timeout_exc = getattr(httpx, "TimeoutException", None)
        request_exc = getattr(httpx, "RequestError", None)
        if timeout_exc is not None and isinstance(exc, timeout_exc):
            return _err("timeout", f"Project 3 API did not respond within {PROJECT3_TIMEOUT_SEC:.0f}s.")
        if request_exc is not None and isinstance(exc, request_exc):
            return _err("network", f"Could not reach Project 3 API: {exc.__class__.__name__}.")
        raise

    if response.status_code >= 400:
        detail = response.text
        try:
            detail_json = response.json()
            detail = detail_json.get("detail", detail_json)
        except Exception:
            pass
        return _err("api", f"Project 3 API rejected the request: {detail}", status=response.status_code)

    data = response.json()
    explanation = "; ".join(
        item.get("business_explanation") or f"{item['feature']}={item['feature_value']} ({item['direction']})"
        for item in data.get("top_explanation_features", [])
    )
    return {
        "transaction_id": transaction_id,
        "recovery_probability": data.get("recovery_probability"),
        "recommended_action": data.get("recommended_action"),
        "expected_value": data.get("expected_value"),
        "confidence": data.get("confidence"),
        "explanation": explanation,
        "source": "Project 3 retry-recovery API",
    }
