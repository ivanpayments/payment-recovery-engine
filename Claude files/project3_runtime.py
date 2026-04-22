from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_TXT = ROOT / "project3_lightgbm_model.txt"
MODEL_META = ROOT / "project3_lightgbm_model_metadata.json"
FEATURE_POLICY = ROOT / "project3_feature_policy.json"
MODELING_TABLE = ROOT / "project3_modeling_table.csv"
DECISION_POLICY_REPORT = ROOT / "project3_decision_policy_evaluation.md"
LIGHTGBM_EVAL = ROOT / "project3_lightgbm_evaluation.md"
SEGMENT_METRICS = ROOT / "project3_lightgbm_segment_metrics.csv"
GLOBAL_IMPORTANCE = ROOT / "project3_shap_global_importance.csv"

DEFAULT_RETRY_COST_USD = 0.12
DEFAULT_MARGIN_RATE = 0.35
DEFAULT_FRICTION_COST_USD = 0.03
DEFAULT_THRESHOLD = 0.05


def expected_retry_value(prob: np.ndarray, amount_usd: np.ndarray) -> np.ndarray:
    recovered_margin = prob * amount_usd * DEFAULT_MARGIN_RATE
    return recovered_margin - DEFAULT_RETRY_COST_USD - DEFAULT_FRICTION_COST_USD


def decision_from_prob_and_value(prob: np.ndarray, exp_value: np.ndarray, threshold: float) -> np.ndarray:
    return np.where((prob >= threshold) & (exp_value > 0), "retry", "do_not_retry")


def business_phrase(feature: str, value, shap_value: float) -> str:
    direction = "increased" if shap_value > 0 else "decreased"
    try:
        amount_value = float(value)
    except (TypeError, ValueError):
        amount_value = None
    templates = {
        "response_code": f"Response code `{value}` {direction} the predicted retry recoverability.",
        "is_soft_decline": f"`is_soft_decline={value}` {direction} the predicted retry recoverability.",
        "risk_skip_flag": f"`risk_skip_flag={value}` {direction} the predicted retry recoverability.",
        "processor_name": f"Processor `{value}` {direction} the predicted retry recoverability.",
        "merchant_country": f"Merchant country `{value}` {direction} the predicted retry recoverability.",
        "merchant_vertical": f"Merchant vertical `{value}` {direction} the predicted retry recoverability.",
        "amount_usd": (
            f"Transaction amount `${amount_value:,.2f}` {direction} the predicted retry recoverability."
            if amount_value is not None
            else f"Feature `amount_usd` with value `{value}` {direction} the predicted retry recoverability."
        ),
        "latency_ms": f"Authorization latency of `{value}` ms {direction} the predicted retry recoverability.",
        "event_hour": f"Event hour `{value}` {direction} the predicted retry recoverability.",
    }
    return templates.get(feature, f"Feature `{feature}` with value `{value}` {direction} the predicted retry recoverability.")


@lru_cache(maxsize=1)
def load_feature_policy() -> dict[str, Any]:
    return json.loads(FEATURE_POLICY.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_model_metadata() -> dict[str, Any]:
    return json.loads(MODEL_META.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_model():
    import lightgbm as lgb

    return lgb.Booster(model_file=str(MODEL_TXT))


@lru_cache(maxsize=1)
def load_reference_frame() -> pd.DataFrame:
    return pd.read_csv(MODELING_TABLE, engine="python", on_bad_lines="skip")


@lru_cache(maxsize=1)
def categorical_category_maps() -> dict[str, list[str]]:
    metadata = load_model_metadata()
    df = load_reference_frame()
    category_maps: dict[str, list[str]] = {}
    for col in metadata.get("categorical_columns", []):
        if col not in df.columns:
            category_maps[col] = ["MISSING"]
            continue
        values = df[col].fillna("MISSING").astype(str)
        category_maps[col] = sorted(values.unique().tolist())
    return category_maps


@lru_cache(maxsize=1)
def top_global_features() -> list[dict[str, Any]]:
    if not GLOBAL_IMPORTANCE.exists():
        return []
    df = pd.read_csv(GLOBAL_IMPORTANCE)
    if "mean_abs_shap" in df.columns:
        return df.sort_values("mean_abs_shap", ascending=False).head(10).to_dict(orient="records")
    return df.head(10).to_dict(orient="records")


@lru_cache(maxsize=1)
def decision_threshold() -> float:
    if DECISION_POLICY_REPORT.exists():
        text = DECISION_POLICY_REPORT.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"Retry threshold used for ML policy:\s*`([0-9.]+)`", text)
        if match:
            return float(match.group(1))
    metadata = load_model_metadata()
    return float(metadata.get("decision_threshold", DEFAULT_THRESHOLD))


def _to_float_or_nan(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _derive_latency_bucket(latency_ms: float | None) -> str | None:
    if latency_ms is None or pd.isna(latency_ms):
        return None
    if latency_ms < 250:
        return "fast"
    if latency_ms < 700:
        return "medium"
    return "slow"


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def enrich_payload(payload: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(payload)

    latency_ms = _to_float_or_nan(enriched.get("latency_ms"))
    if enriched.get("latency_bucket") in (None, ""):
        latency_bucket = _derive_latency_bucket(None if np.isnan(latency_ms) else latency_ms)
        if latency_bucket is not None:
            enriched["latency_bucket"] = latency_bucket

    amount_usd = _to_float_or_nan(enriched.get("amount_usd"))
    amount = _to_float_or_nan(enriched.get("amount"))
    if np.isnan(amount_usd) and not np.isnan(amount):
        enriched["amount_usd"] = amount
    if np.isnan(amount) and not np.isnan(amount_usd):
        enriched["amount"] = amount_usd

    if enriched.get("fx_applied") in (None, "") and enriched.get("fx_rate") not in (None, ""):
        fx_rate = _to_float_or_nan(enriched.get("fx_rate"))
        if not np.isnan(fx_rate):
            enriched["fx_applied"] = bool(abs(fx_rate - 1.0) > 1e-9)

    return enriched


def _coerce_bool(value: Any) -> bool | None:
    if value is None or value == "":
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


def payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    metadata = load_model_metadata()
    feature_columns = metadata["feature_columns"]
    numeric_columns = set(metadata.get("numeric_columns", []))
    boolean_columns = set(metadata.get("boolean_columns", []))
    categorical_columns = set(metadata.get("categorical_columns", []))
    category_maps = categorical_category_maps()

    enriched = enrich_payload(payload)
    row: dict[str, Any] = {}
    for col in feature_columns:
        value = enriched.get(col)
        if col in boolean_columns:
            bool_value = _coerce_bool(value)
            row[col] = 0 if bool_value is None else int(bool_value)
        elif col in numeric_columns:
            row[col] = _to_float_or_nan(value)
        elif col in categorical_columns:
            row[col] = "MISSING" if value in (None, "") else str(value)
        else:
            row[col] = value

    df = pd.DataFrame([row], columns=feature_columns)
    for col in metadata.get("categorical_columns", []):
        categories = category_maps.get(col, ["MISSING"])
        value = str(df.at[0, col]) if pd.notna(df.at[0, col]) else "MISSING"
        if value not in categories:
            value = "MISSING"
        df[col] = pd.Categorical([value], categories=categories)
    return df


def top_contributors(payload: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    booster = load_model()
    frame = payload_to_frame(payload)
    contributions = booster.predict(frame, pred_contrib=True)
    contribution_row = np.array(contributions[0])
    feature_names = list(frame.columns)
    contribution_row = contribution_row[: len(feature_names)]
    ranked = np.argsort(np.abs(contribution_row))[::-1][:limit]

    rows = []
    for idx in ranked:
        feature = feature_names[int(idx)]
        rows.append(
            {
                "feature": feature,
                "feature_value": None if pd.isna(frame.iloc[0][feature]) else _json_safe(frame.iloc[0][feature]),
                "contribution": float(contribution_row[int(idx)]),
                "direction": "increase" if contribution_row[int(idx)] >= 0 else "decrease",
            }
        )
    return rows


@lru_cache(maxsize=1)
def load_shap_explainer():
    import shap

    return shap.TreeExplainer(load_model())


def rich_explanation(payload: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    metadata = load_model_metadata()
    frame = payload_to_frame(payload)
    explainer = load_shap_explainer()
    shap_values = explainer.shap_values(frame)

    if isinstance(shap_values, list):
        shap_row = np.array(shap_values[1] if len(shap_values) > 1 else shap_values[0])[0]
    else:
        shap_row = np.array(shap_values)[0]

    if shap_row.shape[0] == len(frame.columns) + 1:
        shap_row = shap_row[:-1]

    ranked = np.argsort(np.abs(shap_row))[::-1][:limit]
    rows = []
    for idx in ranked:
        feature = frame.columns[int(idx)]
        feature_value = None if pd.isna(frame.iloc[0][feature]) else _json_safe(frame.iloc[0][feature])
        contribution = float(shap_row[int(idx)])
        rows.append(
            {
                "feature": feature,
                "feature_value": feature_value,
                "contribution": contribution,
                "direction": "increase" if contribution >= 0 else "decrease",
                "business_explanation": (
                    f"{feature}={feature_value} {'increased' if contribution >= 0 else 'decreased'} "
                    "the predicted retry recoverability."
                ),
            }
        )
    return rows


def confidence_from_probability(probability: float, threshold: float) -> float:
    distance = abs(probability - threshold)
    return round(min(1.0, distance / 0.5), 4)


def predict_one(payload: dict[str, Any], include_explanation: bool = False, explanation_depth: int = 3) -> dict[str, Any]:
    frame = payload_to_frame(payload)
    booster = load_model()
    metadata = load_model_metadata()
    threshold = decision_threshold()
    probability = float(booster.predict(frame, num_iteration=metadata.get("best_iteration"))[0])
    amount_usd = _to_float_or_nan(payload.get("amount_usd"))
    if np.isnan(amount_usd):
        amount_usd = 0.0
    expected_value = float(expected_retry_value(np.array([probability]), np.array([amount_usd]))[0])
    action = str(decision_from_prob_and_value(np.array([probability]), np.array([expected_value]), threshold)[0])
    result = {
        "recovery_probability": round(probability, 6),
        "recommended_action": action,
        "expected_value": round(expected_value, 4),
        "confidence": confidence_from_probability(probability, threshold),
        "decision_threshold": threshold,
    }
    if include_explanation:
        result["top_explanation_features"] = rich_explanation(payload, limit=explanation_depth)
    return result


def build_model_card() -> dict[str, Any]:
    metadata = load_model_metadata()
    policy = load_feature_policy()
    segment_preview = []
    if SEGMENT_METRICS.exists():
        segment_df = pd.read_csv(SEGMENT_METRICS)
        segment_preview = segment_df.head(12).to_dict(orient="records")

    return {
        "model_name": "project3_lightgbm_recovery_model",
        "model_version": metadata.get("feature_policy_version", "v1"),
        "decision_threshold": decision_threshold(),
        "intended_use": (
            "Decision support for retrying a declined card transaction based on predicted "
            "recoverability and expected net value. Not for unattended production use without "
            "human validation and live monitoring."
        ),
        "training_data": {
            "description": "Synthetic original-decline payment transactions",
            "rows": 12300,
            "positive_class_rate": 0.147,
            "feature_count": len(metadata.get("feature_columns", [])),
            "target": metadata.get("target"),
        },
        "model_architecture": {
            "family": "LightGBM binary classifier",
            "best_iteration": metadata.get("best_iteration"),
            "feature_policy_version": policy.get("version", "v1"),
        },
        "performance": {
            "validation": metadata.get("validation_metrics", {}),
            "test": metadata.get("test_metrics", {}),
            "segment_preview": segment_preview,
        },
        "limitations": [
            "Training and evaluation are fully synthetic; no live issuer or merchant drift validation exists.",
            "The current decision policy is calibrated to synthetic retry cost and margin assumptions.",
            "Coverage outside the modeled countries, processors, and decline patterns is not validated.",
        ],
        "ethical_considerations": [
            "Do not use the score as a proxy for fraud adjudication or customer-worth decisions.",
            "Monitor segment performance by country, processor, and card cohort before expanding usage.",
            "Keep the feature boundary limited to fields available at original decline time.",
        ],
        "security_and_pii_posture": [
            "Synthetic data only; no PAN, CVV, or direct cardholder identifiers are permitted in the training set.",
            "Inference payloads should stay within the feature-policy allowlist and avoid unnecessary identifiers.",
        ],
        "explainability": {
            "method": "LightGBM contribution scores at inference plus offline SHAP analyses",
            "global_top_features": top_global_features(),
        },
    }
