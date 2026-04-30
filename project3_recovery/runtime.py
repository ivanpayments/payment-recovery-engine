"""Inference runtime — loads the LightGBM model, runs predict, and applies
the business decision policy.

The decision policy is parameterised by four values that a merchant can
override per request:

    margin_rate         — % of transaction amount retained as margin
    retry_cost_usd      — cost of one retry attempt (processor/gateway fee)
    friction_cost_usd   — soft cost of annoying the customer with a retry
    decision_threshold  — minimum predicted recovery probability to attempt

If the caller omits any field, we fall back to the platform defaults below.
Defaults are tuned to the synthetic training distribution; merchants with
different unit economics should pass their own values.
"""
from __future__ import annotations

import json
import os
from project3_recovery.ood_guard import (
    categorical_out_of_scope,
    categorical_out_of_scope_response,
    is_supported_response_code,
    out_of_scope_response,
)
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Artifact paths — the Dockerfile copies these into the package dir.
_ARTIFACT_DIR_ENV = os.environ.get("PROJECT3_ARTIFACTS_DIR", "")
if _ARTIFACT_DIR_ENV:
    ROOT = Path(_ARTIFACT_DIR_ENV).resolve()
else:
    ROOT = Path(__file__).resolve().parent / "artifacts"

MODEL_TXT = ROOT / "project3_lightgbm_model.txt"
MODEL_META = ROOT / "project3_lightgbm_model_metadata.json"
TIMING_MODEL_TXT = ROOT / "project3_timing_model.txt"
TIMING_MODEL_META = ROOT / "project3_timing_model_metadata.json"
FEATURE_POLICY = ROOT / "project3_feature_policy.json"
MODELING_TABLE = ROOT / "project3_modeling_table.csv"
DECISION_POLICY_REPORT = ROOT / "project3_decision_policy_evaluation.md"
SEGMENT_METRICS = ROOT / "project3_lightgbm_segment_metrics.csv"
GLOBAL_IMPORTANCE = ROOT / "project3_shap_global_importance.csv"

# Delay grid used by the timing model at inference time. Log-spaced from
# 15 minutes to 30 days — wide enough to cover all retry-cadence benchmarks.
DELAY_GRID_HOURS = [0.25, 1.0, 4.0, 12.0, 24.0, 48.0, 72.0, 168.0, 336.0, 720.0]


# Platform defaults — used when the caller omits an override.
DEFAULT_RETRY_COST_USD = 0.12
DEFAULT_MARGIN_RATE = 0.35
DEFAULT_FRICTION_COST_USD = 0.03

# Per-vertical default margin rates. Used when the caller passes
# merchant_vertical but omits an explicit margin_rate override. Values are
# representative gross-margin benchmarks used to compute recoverable margin
# on a successful retry — not exact merchant economics.
VERTICAL_MARGIN_RATES: dict[str, float] = {
    "saas": 0.75,
    "digital_goods": 0.70,
    "high_risk": 0.40,
    "ecom": 0.30,
    "marketplace": 0.15,
    "travel": 0.10,
}
# v2 (2026-04-24): raised default from 0.05 to 0.10 after adversarial
# threshold-probe Run-2. Operators override per-deployment via
# RECOVERY_DECISION_THRESHOLD; merchants override per-request via
# config.decision_threshold in the /predict body.
DEFAULT_THRESHOLD = float(os.environ.get("RECOVERY_DECISION_THRESHOLD", "0.10"))
DEFAULT_MIN_RETRY_EV_USD = 1.00

# Amount ceiling — requests above this skip the model entirely and return
# action=out_of_scope. Rationale: the model was trained on a ~$5 – $10K
# distribution; on $100K+ transactions the recovery-probability × amount
# math produces absurd EVs (~$12K on a $1M txn) and no real merchant would
# route large tickets through an unattended ML retry loop anyway.
AMOUNT_CAP_USD = 25000.0

# Hard-decline response codes. Retrying any of these is structurally futile
# because the card is flagged/invalid at the cardholder level — no delay
# recovers them. We short-circuit before the model runs.
HARD_DECLINE_CODES = {"04", "07", "14", "15", "41", "43", "54", "57", "59", "62", "R0", "R1"}
HARD_DECLINE_REASONS = {
    "04": "Pick up card — card flagged by issuer",
    "07": "Pick up card, special conditions",
    "14": "Invalid card number",
    "15": "No such issuer",
    "41": "Lost card — do not retry",
    "43": "Stolen card — do not retry",
    "54": "Expired card — needs a new PAN (Account Updater may help)",
    "57": "Transaction not permitted to cardholder",
    "59": "Suspected fraud",
    "62": "Restricted card",
    "R0": "Customer requested stop",
    "R1": "Customer requested stop of all recurring",
}


@dataclass(frozen=True)
class DecisionConfig:
    """Per-request economic assumptions for the decision layer."""

    margin_rate: float
    retry_cost_usd: float
    friction_cost_usd: float
    decision_threshold: float
    min_retry_ev_usd: float

    def to_dict(self) -> dict[str, float]:
        return {
            "margin_rate": self.margin_rate,
            "retry_cost_usd": self.retry_cost_usd,
            "friction_cost_usd": self.friction_cost_usd,
            "decision_threshold": self.decision_threshold,
            "min_retry_ev_usd": self.min_retry_ev_usd,
        }


def resolve_config(
    margin_rate: float | None = None,
    retry_cost_usd: float | None = None,
    friction_cost_usd: float | None = None,
    decision_threshold: float | None = None,
    min_retry_ev_usd: float | None = None,
    merchant_vertical: str | None = None,
) -> DecisionConfig:
    if margin_rate is None:
        resolved_margin = VERTICAL_MARGIN_RATES.get(merchant_vertical, DEFAULT_MARGIN_RATE)
    else:
        resolved_margin = float(margin_rate)
    return DecisionConfig(
        margin_rate=resolved_margin,
        retry_cost_usd=DEFAULT_RETRY_COST_USD if retry_cost_usd is None else float(retry_cost_usd),
        friction_cost_usd=DEFAULT_FRICTION_COST_USD if friction_cost_usd is None else float(friction_cost_usd),
        decision_threshold=platform_default_threshold() if decision_threshold is None else float(decision_threshold),
        min_retry_ev_usd=DEFAULT_MIN_RETRY_EV_USD if min_retry_ev_usd is None else float(min_retry_ev_usd),
    )


def expected_retry_value(
    prob: np.ndarray,
    amount_usd: np.ndarray,
    config: DecisionConfig,
) -> np.ndarray:
    recovered_margin = prob * amount_usd * config.margin_rate
    return recovered_margin - config.retry_cost_usd - config.friction_cost_usd


def decision_from_prob_and_value(
    prob: np.ndarray,
    exp_value: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return np.where((prob >= threshold) & (exp_value > 0), "retry", "do_not_retry")


def business_phrase(feature: str, value, shap_value: float) -> str:
    # Neutral phrasing for features the caller did not provide. We do not let
    # a missing input read as "the value decreased recoverability" — the
    # absence of a signal is not a negative signal.
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return (
            f"Feature `{feature}` was not provided in the request; "
            "its contribution is treated as neutral."
        )
    # Neutral phrasing when contribution is exactly zero (e.g. redacted by
    # the explanation layer for a feature the caller omitted).
    if shap_value == 0:
        return (
            f"Feature `{feature}` had no measurable contribution to the "
            "predicted retry recoverability."
        )
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
def load_timing_model():
    import lightgbm as lgb
    return lgb.Booster(model_file=str(TIMING_MODEL_TXT))


@lru_cache(maxsize=1)
def load_timing_metadata() -> dict[str, Any]:
    return json.loads(TIMING_MODEL_META.read_text(encoding="utf-8"))


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
def platform_default_threshold() -> float:
    """Threshold resolution (first match wins):
        1. RECOVERY_DECISION_THRESHOLD env var
        2. decision_policy_evaluation.md
        3. model_metadata.decision_threshold
        4. DEFAULT_THRESHOLD constant (0.10 after v2)
    """
    env_val = os.environ.get("RECOVERY_DECISION_THRESHOLD")
    if env_val:
        try:
            return float(env_val)
        except (TypeError, ValueError):
            pass
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


@lru_cache(maxsize=1)
def load_shap_explainer():
    import shap
    return shap.TreeExplainer(load_model())


def _feature_was_provided(payload: dict[str, Any], feature: str) -> bool:
    """A feature counts as 'provided' only if the caller sent a non-empty value.

    This is used by the explanation layer to suppress SHAP contributions for
    features the caller did not populate: we do not let a missing input read
    as "this value decreased recoverability." The model itself still handles
    the missing value natively — we only adjust what we surface back.
    """
    if feature not in payload:
        return False
    raw = payload[feature]
    if raw is None:
        return False
    if isinstance(raw, str) and raw.strip() == "":
        return False
    if isinstance(raw, float) and np.isnan(raw):
        return False
    return True


def _contribution_rows(
    frame: pd.DataFrame,
    contributions: np.ndarray,
    limit: int,
    payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    contributions = np.asarray(contributions, dtype=float).copy()
    # Zero out contributions for features the caller did not populate, so
    # nulls never surface as top SHAP drivers. Model prediction itself is
    # unchanged — we only redact the explanation surface.
    if payload is not None:
        for col_idx, feature in enumerate(frame.columns):
            if not _feature_was_provided(payload, feature):
                contributions[col_idx] = 0.0

    ranked = np.argsort(np.abs(contributions))[::-1][:limit]
    rows = []
    for idx in ranked:
        feature = frame.columns[int(idx)]
        raw_value = frame.iloc[0][feature]
        feature_value = None if pd.isna(raw_value) else _json_safe(raw_value)
        contribution = float(contributions[int(idx)])
        if contribution == 0.0:
            direction = "neutral"
        else:
            direction = "increase" if contribution > 0 else "decrease"
        rows.append(
            {
                "feature": feature,
                "feature_value": feature_value,
                "contribution": contribution,
                "direction": direction,
                "business_explanation": business_phrase(feature, feature_value, contribution),
            }
        )
    return rows


def fast_explanation(payload: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    # LightGBM's pred_contrib=True returns per-feature contributions from the
    # same tree traversal used for the prediction itself — sub-millisecond.
    # The trailing column is the bias (base value) and is dropped.
    frame = payload_to_frame(payload)
    booster = load_model()
    metadata = load_model_metadata()
    contribs = booster.predict(
        frame,
        num_iteration=metadata.get("best_iteration"),
        pred_contrib=True,
    )
    row = np.asarray(contribs)[0][:-1]
    return _contribution_rows(frame, row, limit, payload=payload)


def rich_explanation(payload: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    # Audit-grade path: full TreeSHAP. ~100x slower than the fast path but
    # produces the exact game-theoretic Shapley values regulators expect.
    frame = payload_to_frame(payload)
    explainer = load_shap_explainer()
    shap_values = explainer.shap_values(frame)

    if isinstance(shap_values, list):
        shap_row = np.array(shap_values[1] if len(shap_values) > 1 else shap_values[0])[0]
    else:
        shap_row = np.array(shap_values)[0]

    if shap_row.shape[0] == len(frame.columns) + 1:
        shap_row = shap_row[:-1]

    return _contribution_rows(frame, shap_row, limit, payload=payload)


def explanation_for(
    payload: dict[str, Any],
    mode: str,
    limit: int,
) -> list[dict[str, Any]]:
    # NOTE (D5 / N7): for tree ensembles (LightGBM included), the LightGBM
    # `pred_contrib=True` path already returns the exact TreeSHAP values that
    # the SHAP TreeExplainer would compute — `fast_explanation` and
    # `rich_explanation` are mathematically identical to ~1e-9. We retain the
    # `mode` arg only for back-compat with existing callers and always route
    # to the fast path. The `explanation_mode` query parameter has been
    # removed from the public API.
    return fast_explanation(payload, limit=limit)


def confidence_from_probability(probability: float, threshold: float) -> float:
    distance = abs(probability - threshold)
    return round(min(1.0, distance / 0.5), 4)


def _timing_frame(payload: dict[str, Any], delay_h: float, attempt_num: int) -> pd.DataFrame:
    """Build a single-row frame matching the timing model's feature layout."""
    meta = load_timing_metadata()
    feature_columns = meta["feature_columns"]
    numeric_columns = set(meta.get("numeric_columns", []))
    boolean_columns = set(meta.get("boolean_columns", []))
    categorical_columns = set(meta.get("categorical_columns", []))
    category_maps = meta.get("category_maps", {})

    enriched = enrich_payload(payload)
    enriched["delay_hours"] = delay_h
    enriched["attempt_num"] = attempt_num

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
    # Re-apply the same pandas Categorical dtypes used at training time.
    for col in categorical_columns:
        categories = category_maps.get(col, [])
        if categories:
            value = str(df.at[0, col]) if pd.notna(df.at[0, col]) else "MISSING"
            if value not in categories:
                value = "MISSING" if "MISSING" in categories else categories[0]
            df[col] = pd.Categorical([value], categories=categories)
    return df


def predict_delay_curve(
    payload: dict[str, Any],
    config: DecisionConfig,
    amount_usd: float,
    attempt_num: int = 1,
) -> list[dict[str, Any]]:
    """Evaluate the timing model at each delay in the grid and compute EV."""
    booster = load_timing_model()
    meta = load_timing_metadata()
    curve = []
    for d in DELAY_GRID_HOURS:
        frame = _timing_frame(payload, d, attempt_num)
        prob = float(booster.predict(frame, num_iteration=meta.get("best_iteration"))[0])
        ev = prob * amount_usd * config.margin_rate - config.retry_cost_usd - config.friction_cost_usd
        curve.append({
            "delay_hours": d,
            "success_probability": round(prob, 6),
            "expected_value": round(ev, 4),
        })
    return curve


def predict_one(
    payload: dict[str, Any],
    config: DecisionConfig,
    include_explanation: bool = False,
    explanation_depth: int = 3,
    explanation_mode: str = "fast",
    attempt_num: int = 1,
) -> dict[str, Any]:
    # Business-rule short-circuit 1: hard declines never recover regardless of
    # what the model would predict. Cardholder needs a new payment method.
    # v2 OOD refusal short-circuit (adversarial P1.2)
    response_code_raw = payload.get("response_code")
    if not is_supported_response_code(response_code_raw):
        explanation = (
            f"Response code `{response_code_raw}` not in trained vocabulary "
            "— model shortcircuited; no per-feature SHAP available"
        ) if include_explanation else None
        return out_of_scope_response(
            response_code_raw,
            config.to_dict() if hasattr(config, "to_dict") else {},
            explanation=explanation,
        )

    response_code = str(payload.get("response_code", "") or "").strip().upper()
    if response_code in HARD_DECLINE_CODES:
        # D4 (N6): when caller asked for explanation, return a string saying
        # WHY no per-feature SHAP is available.
        explanation = (
            f"Hard decline (code {response_code}, "
            f"{HARD_DECLINE_REASONS.get(response_code, 'see model card')}) "
            "— model shortcircuited; no per-feature SHAP available"
        ) if include_explanation else None
        body: dict[str, Any] = {
            # v4 (N3): `recoverability_score` is the new canonical name —
            # this is the recoverability model's unconditional score for
            # the original decline; it does NOT condition on attempt_num
            # (only the timing model does, via `recommended_delay_*` and
            # `delay_curve`). `recovery_probability` is kept as a back-
            # compat alias and will be removed in a future major.
            "recoverability_score": 0.0,
            "recovery_probability": 0.0,  # deprecated alias of recoverability_score
            "recommended_action": "do_not_retry",
            "expected_value": 0.0,
            # D6 (N8): confidence is not meaningful when the model never ran.
            "confidence": None,
            "config_used": config.to_dict(),
            # D3 (N5): when action is do_not_retry there is no recommended
            # retry to delay. Null all the recommended_delay_* fields.
            "recommended_delay_hours": None,
            "recommended_delay_probability": None,
            "recommended_delay_expected_value": None,
            "delay_curve": [],
            "hard_decline": True,
            "hard_decline_reason": HARD_DECLINE_REASONS.get(response_code, "Hard decline — no delay recovers this code"),
        }
        if explanation is not None:
            body["explanation"] = explanation
            body["top_explanation_features"] = []
        return body

    # D1 (N3): categorical out-of-scope refusal — country / vertical /
    # processor must be in the trained allowlist. Without this, unknown
    # values silently flow through the Pandas Categorical "MISSING" fallback
    # and the model produces a number despite the model card disclaimer.
    cat_off = categorical_out_of_scope(payload)
    if cat_off is not None:
        cat_field, cat_value = cat_off
        cat_explanation = (
            f"{cat_field} `{cat_value}` not in trained allowlist "
            "— model shortcircuited; no per-feature SHAP available"
        ) if include_explanation else None
        return categorical_out_of_scope_response(
            cat_field,
            cat_value,
            config.to_dict() if hasattr(config, "to_dict") else {},
            explanation=cat_explanation,
        )

    # Business-rule short-circuit 2: very large amounts fall outside the
    # model's training envelope. Refuse to score rather than produce
    # absurd-looking EVs that no merchant should auto-execute.
    amount_raw = _to_float_or_nan(payload.get("amount_usd"))
    if not np.isnan(amount_raw) and amount_raw > AMOUNT_CAP_USD:
        # v5 (N24): use 2-decimal formatting in the operator-facing
        # message so a payload like $25,000.01 doesn't render as
        # "Amount $25,000 exceeds $25,000" (which reads as "exceeds
        # itself" — operator confusion). With 2 decimals it reads
        # "Amount $25,000.01 exceeds $25,000.00".
        amount_explanation = (
            f"Amount ${amount_raw:,.2f} exceeds training envelope "
            f"(${AMOUNT_CAP_USD:,.2f}) — model shortcircuited; "
            "no per-feature SHAP available"
        ) if include_explanation else None
        body = {
            # v4 (N3): see hard-decline branch above for naming rationale.
            "recoverability_score": None,
            "recovery_probability": None,  # deprecated alias of recoverability_score
            "recommended_action": "out_of_scope",
            "expected_value": None,
            # D6 (N8): confidence is not meaningful when the model never ran.
            "confidence": None,
            "config_used": config.to_dict(),
            "recommended_delay_hours": None,
            "recommended_delay_probability": None,
            "recommended_delay_expected_value": None,
            "delay_curve": [],
            "out_of_scope": True,
            "out_of_scope_reason": f"Amount ${amount_raw:,.2f} exceeds the model's training envelope (${AMOUNT_CAP_USD:,.2f}). Use a manual review or merchant's high-ticket workflow rather than automated retry.",
        }
        if amount_explanation is not None:
            body["explanation"] = amount_explanation
            body["top_explanation_features"] = []
        return body

    frame = payload_to_frame(payload)
    booster = load_model()
    metadata = load_model_metadata()
    probability = float(booster.predict(frame, num_iteration=metadata.get("best_iteration"))[0])

    amount_usd = _to_float_or_nan(payload.get("amount_usd"))
    if np.isnan(amount_usd):
        amount_usd = 0.0

    # Aggregate EV derived from the recoverability model — kept for reporting.
    expected_value = float(
        expected_retry_value(np.array([probability]), np.array([amount_usd]), config)[0]
    )

    # Timing model: sweep delay grid and pick argmax-EV. The timing model is
    # the authoritative signal for whether to retry, because it conditions on
    # the specific delay we would attempt. Tie-break on shorter delay when
    # multiple points share the top EV.
    delay_curve = predict_delay_curve(payload, config, amount_usd, attempt_num)
    best = min(
        (r for r in delay_curve if r["expected_value"] == max(x["expected_value"] for x in delay_curve)),
        key=lambda r: r["delay_hours"],
    )
    # Action gate: the EV floor (min_retry_ev_usd) prevents tiny-amount waste
    # where retry_cost + friction eats whatever recovery the model predicts.
    ev_ok = best["expected_value"] >= config.min_retry_ev_usd
    prob_ok = best["success_probability"] >= config.decision_threshold
    if ev_ok and prob_ok:
        action = "retry"
        attempt_decision_reason: str | None = None
    else:
        action = "do_not_retry"
        # Build a human-readable explanation of WHY this attempt was skipped.
        # Populated only when the action is do_not_retry AND no hard_decline
        # short-circuit fired — the buyer's "why did you stop at attempt N?"
        # question is answered by exposing the EV / threshold gates that
        # closed.
        reasons: list[str] = []
        if not prob_ok:
            reasons.append(
                f"best-delay success probability {best['success_probability']:.3f} "
                f"is below decision_threshold {config.decision_threshold:.2f}"
            )
        if not ev_ok:
            reasons.append(
                f"best-delay expected value ${best['expected_value']:.2f} "
                f"is below min_retry_ev_usd ${config.min_retry_ev_usd:.2f}"
            )
        attempt_decision_reason = (
            f"Attempt {attempt_num}: " + "; ".join(reasons) +
            ". Timing model conditions on attempt_num, so success probability "
            "decays as retries accumulate; once it crosses either the "
            "probability or EV floor the engine stops."
        )

    # D3 (N5): when the action is do_not_retry, there is no recommended retry
    # to delay. Null the recommended_delay_* fields rather than leaving the
    # winning-EV-curve point in place, which reads as "we recommend retrying
    # at this delay" even though the action says do_not_retry. The delay_curve
    # itself is preserved for transparency / debugging.
    if action == "do_not_retry":
        recommended_delay_hours: Any = None
        recommended_delay_probability: Any = None
        recommended_delay_expected_value: Any = None
    else:
        recommended_delay_hours = best["delay_hours"]
        recommended_delay_probability = best["success_probability"]
        recommended_delay_expected_value = best["expected_value"]

    # v4 (N3): `recoverability_score` is the new canonical name for the
    # recoverability-model output. It is the model's unconditional score
    # for the *original* decline and does NOT condition on attempt_num —
    # the timing model is what adjusts success probability per attempt
    # (see `recommended_delay_probability` and `delay_curve`). The
    # `recovery_probability` field is preserved as a back-compat alias
    # and will be removed in a future major version.
    score = round(probability, 6)
    # v5 (N23): when the action is do_not_retry, the unconditional EV
    # headline reads as a contradiction next to it ("do_not_retry" with
    # "+$16.60 EV" implies the engine left money on the table). Mirror the
    # Wave 2D N5 fix that nulled `recommended_delay_*` on do_not_retry: null
    # `expected_value` too on this soft-decline path. Hard-decline and
    # out-of-scope branches already null EV (hard-decline returns 0.0 by
    # design — the model never ran), so this only affects the model-ran
    # path. The `delay_curve` is preserved for transparency.
    if action == "do_not_retry":
        expected_value_out: Any = None
    else:
        expected_value_out = round(expected_value, 4)

    result = {
        "recoverability_score": score,
        "recovery_probability": score,  # deprecated alias of recoverability_score
        "recommended_action": action,
        "expected_value": expected_value_out,
        "confidence": confidence_from_probability(probability, config.decision_threshold),
        "config_used": config.to_dict(),
        "recommended_delay_hours": recommended_delay_hours,
        "recommended_delay_probability": recommended_delay_probability,
        "recommended_delay_expected_value": recommended_delay_expected_value,
        "delay_curve": delay_curve,
        "attempt_decision_reason": attempt_decision_reason,
    }
    if include_explanation:
        # D5 (N7): `explanation_mode` arg retained for back-compat but no
        # longer surfaced on the response; LightGBM `pred_contrib=True`
        # already returns exact TreeSHAP values for tree models so there is
        # no slow-path to switch to.
        result["top_explanation_features"] = explanation_for(payload, "fast", explanation_depth)
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
        "platform_default_threshold": platform_default_threshold(),
        "platform_defaults": {
            "margin_rate": DEFAULT_MARGIN_RATE,
            "retry_cost_usd": DEFAULT_RETRY_COST_USD,
            "friction_cost_usd": DEFAULT_FRICTION_COST_USD,
            "decision_threshold": platform_default_threshold(),
            "decision_threshold_tunable": True,
            "decision_threshold_override_keys": {
                "per_request": "config.decision_threshold",
                "per_deployment_env": "RECOVERY_DECISION_THRESHOLD",
            },
        },
        "intended_use": (
            "Decision support for retrying a declined card transaction based on predicted "
            "recoverability and expected net value. Merchants with different margin/cost "
            "structures should override the five config fields per request "
            "(margin_rate, retry_cost_usd, friction_cost_usd, decision_threshold, "
            "min_retry_ev_usd). Not for unattended production use without "
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
            "The default decision policy is calibrated to synthetic retry cost and margin assumptions — pass merchant-specific values for real decisions.",
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
        # v4 (N3) doc note surfaced via /model-card so consumers see it
        # without reading the OpenAPI body description.
        "response_field_notes": {
            "recoverability_score": (
                "Unconditional recoverability score from the recoverability "
                "model — the probability the original decline can be recovered "
                "by some retry. Does NOT condition on attempt_num; it is "
                "bit-identical across attempt 1..N for a given input. The "
                "timing model (`recommended_delay_*` and `delay_curve`) is "
                "what conditions on attempt_num and decays per attempt."
            ),
            "recovery_probability": (
                "Deprecated alias of `recoverability_score`. Same value, "
                "kept for back-compat with existing consumers; will be "
                "removed in a future major version."
            ),
            "recommended_delay_probability": (
                "Success probability from the timing model at the chosen "
                "delay AND attempt_num. This is the per-attempt probability."
            ),
        },
    }
