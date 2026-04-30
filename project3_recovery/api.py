"""FastAPI REST API for the decline-recovery decision engine.

Endpoints:
    POST /predict          — score a decline, return action + explanation
    GET  /health           — liveness + model version + platform defaults
    GET  /model-card       — training/eval snapshot

All POST endpoints require  Authorization: Bearer sk_test_...
Rate limits: 100 req/min per API key, 60 req/min per client IP.
Optional Idempotency-Key header caches responses for 24h.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import redis
import secrets
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse as _BaseJSONResponse


# v4 (N8): emit `Content-Type: application/json; charset=utf-8` on every
# JSON response so Windows clients render em-dashes / unicode without
# mojibake. Starlette's default JSONResponse omits the charset parameter,
# which causes some clients (notably Windows curl, Excel Power Query, and
# default .NET HttpClient) to fall back to cp1252 / latin-1 and corrupt
# every non-ASCII byte. We override `media_type` on a subclass and wire it
# in as both the FastAPI app's `default_response_class` AND the local
# `JSONResponse` symbol so all explicit `JSONResponse(...)` constructions
# inside this module pick up the same charset declaration.
class JSONResponse(_BaseJSONResponse):
    media_type = "application/json; charset=utf-8"
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from project3_recovery import __version__
from project3_recovery.api_keys import seed_test_key, validate_secret_key
from project3_recovery.db import create_tables, engine, get_db
from project3_recovery.idempotency import get_cached, store as idem_store
from project3_recovery.rate_limit import is_rate_limited
from project3_recovery.runtime import (
    DEFAULT_FRICTION_COST_USD,
    DEFAULT_MARGIN_RATE,
    DEFAULT_MIN_RETRY_EV_USD,
    DEFAULT_RETRY_COST_USD,
    build_model_card,
    categorical_category_maps,
    load_feature_policy,
    load_model,
    load_model_metadata,
    load_reference_frame,
    load_shap_explainer,
    platform_default_threshold,
    predict_one,
    resolve_config,
    top_global_features,
)
from project3_recovery.ood_guard import (
    categorical_out_of_scope,
    categorical_out_of_scope_response,
    is_supported_response_code,
    out_of_scope_response,
    unsupported_code_http422_detail,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        for field in ("request_id", "latency_ms", "prediction", "model_version", "path", "api_key_id"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload)


_handler = logging.StreamHandler()
_handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
log = logging.getLogger("project3.api")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DecisionConfigPayload(BaseModel):
    """Optional per-request overrides. Any omitted field falls back to
    the platform default returned by /health.

    v5 (N16, N25): each numeric field below is run through
    `_validate_finite_or_omit` to reject NaN / +Inf / -Inf (which would
    otherwise crash the EV math at HTTP 500) AND to reject explicit
    null values (which would otherwise silently fall back to the
    platform default — inconsistent with `amount_usd: null` returning
    422). Omitting a field still means "use the platform default";
    sending the field with `null` is now a 422 with a helpful message
    pointing the caller at omission instead.
    """

    model_config = ConfigDict(extra="forbid")

    margin_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Share of amount retained as margin (0–1). Default 0.35.",
    )
    retry_cost_usd: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Processor + gateway cost of one retry attempt, in USD. Default 0.12.",
    )
    friction_cost_usd: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Soft customer-annoyance cost per retry, in USD. Default 0.03.",
    )
    decision_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Minimum predicted recovery probability for retry. Default 0.10 (raised from 0.05 in v2 after adversarial threshold-probe tuning). Operators override per-deployment via RECOVERY_DECISION_THRESHOLD env var.",
    )
    min_retry_ev_usd: Optional[float] = Field(
        default=None, ge=0.0, le=10000.0,
        description="Minimum expected value (USD) a retry must clear to be recommended. Default 1.00. Prevents tiny-amount waste where retry cost + friction eats the recovery.",
    )
    attempt_num: Optional[int] = Field(
        default=1, ge=1, le=10,
        description="Which retry attempt this would be (1 = first retry, 2 = second, etc.). Affects the timing model's success-rate estimate.",
    )

    @field_validator(
        "margin_rate",
        "retry_cost_usd",
        "friction_cost_usd",
        "decision_threshold",
        "min_retry_ev_usd",
        mode="before",
    )
    @classmethod
    def _validate_finite_or_omit(cls, value, info):
        # v5 (N16, N25): reject explicit null AND NaN / ±Infinity at the
        # validator layer.
        #   - Explicit `null` (e.g. `"margin_rate": null`) used to silently
        #     fall back to the platform default, which was inconsistent
        #     with `amount_usd: null` returning 422. Strict rejection
        #     forces the caller to OMIT the field instead, matching the
        #     amount_usd contract.
        #   - NaN / ±Infinity used to crash the EV math at HTTP 500 plain
        #     text. We now reject them with a 422 JSON body.
        # The `Optional[float] = None` default still applies when the
        # field is OMITTED (Pydantic does not call this validator when
        # the key is absent from the payload).
        field_name = info.field_name if info is not None else "config field"
        if value is None:
            raise ValueError(
                f"config.{field_name} must be a finite number when present; "
                "got null. Omit the field (do not send the key at all) if "
                "you want the platform default."
            )
        if isinstance(value, bool):
            raise ValueError(
                f"config.{field_name} must be a finite number; got boolean "
                f"({value})."
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"config.{field_name} must be a finite number; got {value!r}."
            )
        if not math.isfinite(numeric):
            raise ValueError(
                f"config.{field_name} must be a finite number; got {value}."
            )
        return numeric


class FeaturePayloadBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("amount_usd", "amount", mode="before", check_fields=False)
    @classmethod
    def _validate_amount_positive(cls, value, info):
        # v4 (N1): tighten amount validation — the previous `>0` check
        # accepted `None`, booleans, and non-numeric strings (which then
        # silently coerced to NaN/0 downstream and produced bogus EVs at
        # HTTP 200). We now reject all four shapes at validation time:
        #   - `None`               → 422 (use omission with default-NaN)
        #   - bool (`True`/`False`) → 422 (bool is a subclass of int in
        #                              Python, but conceptually not an
        #                              amount — accept-by-default would
        #                              evaluate True as $1.00)
        #   - non-numeric strings  → 422 ("not-a-number")
        #   - empty string         → 422
        #   - <= 0                 → 422 (refunds/chargebacks belong on a
        #                              separate endpoint)
        field_name = info.field_name if info is not None else "amount_usd"

        if value is None:
            raise ValueError(
                f"{field_name} must be a positive number; got null. "
                "Omit the field if you want the model to treat it as missing, "
                "or pass a number > 0."
            )
        if isinstance(value, bool):
            raise ValueError(
                f"{field_name} must be a positive number; got boolean ({value}). "
                "Pass a numeric value in USD (e.g. 342.50)."
            )
        if isinstance(value, str) and value.strip() == "":
            raise ValueError(
                f"{field_name} must be a positive number; got an empty string. "
                "Pass a numeric value in USD."
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{field_name} must be a positive number; got {value!r}. "
                "Pass a numeric value in USD (e.g. 342.50)."
            )
        # v5 (N16): tighten the NaN check and also reject ±Infinity. The
        # previous `numeric == numeric` line caught NaN but let `inf` /
        # `-inf` through, which then crashed downstream with HTTP 500
        # plain-text. `math.isfinite()` rejects both NaN and ±Infinity in
        # one call and produces a helpful 422 JSON body instead.
        if not math.isfinite(numeric):
            raise ValueError(
                f"{field_name} must be a finite positive number; got {value}."
            )
        if numeric <= 0:
            raise ValueError(
                f"{field_name} must be > 0; got {value}. "
                "For partial-refund or chargeback flows, use a separate endpoint."
            )
        return value


# Build the dynamic feature payload. All allow-listed feature columns are
# Optional[Any] EXCEPT `response_code`, which the decision engine cannot
# reason about when missing. A missing `response_code` is a request-shape
# error (FastAPI/Pydantic returns 422), not an OOD code (which is a valid
# shape with an unknown vocabulary value — that path returns 200 with
# `out_of_scope`). See audit v2 finding N2.
_feature_columns = load_model_metadata()["feature_columns"]
_optional_features = {
    name: (Any | None, None)
    for name in _feature_columns
    if name != "response_code"
}
FeaturePayload = create_model(
    "FeaturePayload",
    __base__=FeaturePayloadBase,
    response_code=(
        str,
        Field(
            ...,
            min_length=1,
            description=(
                "Card-network response code from the original decline (e.g. "
                "'05', '51', '91'). Required. If the code is not in the model's "
                "trained vocabulary the response is HTTP 200 with "
                "`recommended_action: 'out_of_scope'`; if the field is missing "
                "or empty the response is HTTP 422."
            ),
        ),
    ),
    **_optional_features,
    config=(DecisionConfigPayload | None, None),
)


# Realistic payload covering every feature in the v2 allowlist. Exposed on
# /docs so a user can one-click pre-fill the request body instead of hand-
# typing 70+ fields. Values are drawn from the training distribution so the
# example produces a plausible prediction rather than a degenerate one.
FULL_FEATURE_EXAMPLE: dict[str, Any] = {
    "amount": 342.50,
    "amount_usd": 342.50,
    "currency": "USD",
    "merchant_vertical": "ecom",
    "merchant_mcc": 5734,
    "merchant_country": "US",
    "archetype": "mid_market_saas",
    "processor_name": "global-acquirer-a",
    "routing_reason": "highest_auth",
    "card_brand": "visa",
    "card_type": "credit",
    "card_country": "US",
    "card_funding_source": "consumer",
    "is_cross_border": False,
    "is_token": True,
    "token_type": "network_token",
    "present_mode": "cnp",
    "response_code": "51",
    "response_message": "Insufficient funds",
    "decline_bucket": "soft",
    "is_soft_decline": True,
    "scheme_response_code": "51",
    "three_ds_requested": True,
    "three_ds_outcome": "authenticated",
    "three_ds_version": 2.2,
    "three_ds_flow": "frictionless",
    "three_ds_eci": 5,
    "sca_exemption": "none",
    "latency_ms": 317,
    "latency_auth_ms": 317,
    "latency_3ds_ms": 0,
    "latency_bucket": "fast",
    "processor_fee_bps": 240,
    "interchange_estimate_bps": 180,
    "scheme_fee_bps": 15,
    "fx_applied": False,
    "fx_rate": 1.0,
    "settlement_currency": "USD",
    "risk_score": 42,
    "fraud_flag": False,
    "risk_model_version": "risk_v3",
    "shipping_country": "US",
    "ip_country": "US",
    "issuer_country": "US",
    "payment_method": "card",
    "wallet_type": "none",
    "network_token_present": True,
    "entry_mode": "ecommerce",
    "cardholder_verification_method": "cvv",
    "cvv_result": "M",
    "avs_result": "Y",
    "avs_zip_match": True,
    "avs_street_match": True,
    "card_product_type": "consumer",
    "card_category": "consumer",
    "card_commercial_type": "no",
    "user_agent_family": "mobile_web",
    "timeout_flag": False,
    "device_os": "ios",
    "issuer_size": "large",
    "account_updater_used": False,
    "mastercard_advice_code": 0,
    "routing_optimized": True,
    "mcc_routing_optimized": False,
    "smart_routed": True,
    "scheme_ms": 110,
    "transaction_type": "subscription",
    "fx_bps": 0,
    "routed_network": "visa",
    "risk_skip_flag": False,
    "event_hour": 11,
    "event_dayofweek": 5,
    "event_month": 7,
    "is_weekend": True,
    "config": {"attempt_num": 1},
}


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class _LimitBodySize(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 1_048_576:
            return JSONResponse({"detail": "Request body too large (max 1 MB)"}, status_code=413)
        return await call_next(request)


class _JsonCharsetMiddleware(BaseHTTPMiddleware):
    """v4 (N8): force `charset=utf-8` on every JSON response.

    The custom `JSONResponse` subclass + `default_response_class` should
    already cover this, but Starlette's response stream sometimes lands
    `application/json` without a charset on dict-returning endpoints
    (FastAPI re-encodes the body with its own JSONResponse if the
    handler returns a raw dict and ignores the app-level default for
    the header value). This middleware adds the charset declaration as
    a defence-in-depth so Windows clients never see mojibake.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        ctype = response.headers.get("content-type", "")
        if ctype.startswith("application/json") and "charset=" not in ctype.lower():
            response.headers["content-type"] = "application/json; charset=utf-8"
        return response


# ---------------------------------------------------------------------------
# Lifespan — model warmup + DB + Redis
# ---------------------------------------------------------------------------

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")


@asynccontextmanager
async def lifespan(application: FastAPI):
    create_tables()
    if os.environ.get("ENV", "local") == "local":
        from sqlalchemy.orm import Session as DBSession
        with DBSession(engine) as db:
            seed_test_key(db)

    try:
        application.state.redis = redis.from_url(_REDIS_URL, decode_responses=True)
        application.state.redis.ping()
        application.state.redis_available = True
    except Exception:
        application.state.redis_available = False
        application.state.redis = None

    load_model()
    load_reference_frame()
    categorical_category_maps()
    top_global_features()
    build_model_card()
    try:
        load_shap_explainer()
    except Exception as exc:
        log.warning(f"SHAP explainer failed to warm: {exc}")

    log.info(
        "model loaded",
        extra={
            "model_version": load_model_metadata().get("feature_policy_version", "v1"),
            "prediction": "startup",
            "path": "/startup",
        },
    )
    yield
    if application.state.redis:
        application.state.redis.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Project 3 — ML Payment Recovery Engine",
    description=(
        "Scores a declined card payment and recommends whether to retry it. "
        "Returns probability, expected retry value, action, and optional SHAP-backed "
        "explanation. Five economic parameters (margin_rate, retry_cost_usd, "
        "friction_cost_usd, decision_threshold, min_retry_ev_usd) can be "
        "overridden per request."
    ),
    version=__version__,
    lifespan=lifespan,
    root_path=os.environ.get("ROOT_PATH", ""),
    openapi_tags=[
        {"name": "predict", "description": "Score a single decline with optional per-request config overrides."},
        {"name": "ops", "description": "Health, model card, and observability endpoints."},
    ],
    # v4 (N8): default_response_class also sets the OpenAPI-advertised
    # content-type for endpoints that return raw dict / pydantic objects
    # (i.e. the ones that DON'T explicitly construct a JSONResponse).
    default_response_class=JSONResponse,
)
app.state.redis = None
app.state.redis_available = False
app.add_middleware(_LimitBodySize)
# v4 (N8): JSON charset must run AFTER the body-size guard so it sees
# the response from downstream handlers. Starlette executes middleware
# in reverse-add order, so adding the charset middleware second means
# it runs first on the response path.
app.add_middleware(_JsonCharsetMiddleware)


# ---------------------------------------------------------------------------
# Auth + rate-limit dependency
# ---------------------------------------------------------------------------

bearer_scheme = HTTPBearer(
    bearerFormat="sk_test_*",
    description="Paste a secret API key (sk_test_...). Use sk_test_fUhRdnZj3lEedZJeAkzqT1gJa6OZwdRm for the shared public demo key.",
    auto_error=False,
)


def get_current_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
):
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    api_key = validate_secret_key(db, credentials.credentials)
    if api_key is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    rc = request.app.state.redis
    client_ip = request.client.host if request.client else None
    if rc and is_rate_limited(rc, api_key.id, client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return api_key


# ---------------------------------------------------------------------------
# Idempotency helpers
# ---------------------------------------------------------------------------

def _idem_check(request: Request, api_key_id: str, idem_key: str | None) -> dict | None:
    if not idem_key:
        return None
    rc = request.app.state.redis
    if rc is None:
        return None
    return get_cached(rc, api_key_id, idem_key)


def _idem_store(request: Request, api_key_id: str, idem_key: str | None, body: dict) -> None:
    if not idem_key:
        return
    rc = request.app.state.redis
    if rc is None:
        return
    idem_store(rc, api_key_id, idem_key, body)


# ---------------------------------------------------------------------------
# Health + model card (no auth)
# ---------------------------------------------------------------------------

# D8 (N11): /health is reachable by anonymous load balancers and external
# uptime probes — keep it minimal and public. Economic + rate-limit settings
# (which describe billing-relevant defaults and let an attacker map our
# throttle policy) move behind HTTP Basic on /admin/health.
@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    metadata = load_model_metadata()
    return {
        "ok": True,
        "model_loaded": True,
        "model_version": metadata.get("feature_policy_version", "v1"),
    }


_admin_basic = HTTPBasic(auto_error=False)


def _require_admin(credentials: HTTPBasicCredentials | None = Depends(_admin_basic)) -> None:
    expected_user = os.environ.get("RECOVERY_HEALTH_USER")
    expected_pass = os.environ.get("RECOVERY_HEALTH_PASSWORD")
    if not expected_user or not expected_pass:
        # Fail closed: if creds aren't configured, the admin endpoint is
        # disabled rather than silently exposed.
        raise HTTPException(
            status_code=503,
            detail="Admin endpoint not configured (RECOVERY_HEALTH_USER / RECOVERY_HEALTH_PASSWORD env vars unset)",
        )
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    user_ok = secrets.compare_digest(credentials.username, expected_user)
    pass_ok = secrets.compare_digest(credentials.password, expected_pass)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


@app.get("/admin/health", tags=["ops"])
def admin_health(_: None = Depends(_require_admin)) -> dict[str, Any]:
    metadata = load_model_metadata()
    threshold = platform_default_threshold()
    return {
        "ok": True,
        "model_loaded": True,
        "model_version": metadata.get("feature_policy_version", "v1"),
        "feature_count": len(metadata.get("feature_columns", [])),
        "platform_defaults": {
            "margin_rate": DEFAULT_MARGIN_RATE,
            "retry_cost_usd": DEFAULT_RETRY_COST_USD,
            "friction_cost_usd": DEFAULT_FRICTION_COST_USD,
            "decision_threshold": threshold,
            # D7 (N10): document min_retry_ev_usd that already shows up in
            # /predict responses under config_used; previously undocumented.
            "min_retry_ev_usd": DEFAULT_MIN_RETRY_EV_USD,
        },
        "rate_limit": {"per_key_per_minute": 100, "per_ip_per_minute": 60},
    }


@app.get("/model-card", tags=["ops"])
def model_card() -> dict[str, Any]:
    return build_model_card()


# ---------------------------------------------------------------------------
# Predict — with optional per-request config override
# ---------------------------------------------------------------------------

@app.post("/predict", tags=["predict"])
def predict(
    request: Request,
    payload: FeaturePayload = Body(
        ...,
        openapi_examples={
            "prefilled_full": {
                "summary": "Populate with pre-filled data (all features)",
                "description": (
                    "Complete realistic payload covering every field in the v2 "
                    "feature allowlist. Click this example to auto-populate the "
                    "request body with training-distribution values, then edit "
                    "anything you want before sending."
                ),
                "value": FULL_FEATURE_EXAMPLE,
            },
            "minimal": {
                "summary": "Minimal payload (core fields only)",
                "description": (
                    "Smallest useful payload. Missing features default to "
                    "NaN / MISSING on the server — LightGBM handles missing "
                    "values natively, so this still produces a valid prediction "
                    "with less signal."
                ),
                "value": {
                    "amount_usd": 342.50,
                    "response_code": "51",
                    "is_soft_decline": True,
                    "processor_name": "global-acquirer-a",
                    "merchant_country": "US",
                    "merchant_vertical": "ecom",
                    "config": {"attempt_num": 1},
                },
            },
        },
    ),
    include_explanation: bool = Query(default=False),
    explanation_depth: int = Query(default=5, ge=1, le=10),
    api_key=Depends(get_current_api_key),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    """Score one declined transaction.

    Supports an optional `config` object in the body with any of:
      margin_rate, retry_cost_usd, friction_cost_usd, decision_threshold.
    Omitted fields use the platform defaults. The response echoes back
    which config values drove the decision under `config_used`.
    """
    cached = _idem_check(request, api_key.id, idempotency_key)
    if cached is not None:
        return JSONResponse(cached, headers={"X-Idempotency-Replay": "true"})

    started = time.perf_counter()
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]

    body = payload.model_dump(exclude_none=True)
    config_dict = body.pop("config", None) or {}
    attempt_num = int(config_dict.pop("attempt_num", 1) or 1)
    config = resolve_config(
        margin_rate=config_dict.get("margin_rate"),
        retry_cost_usd=config_dict.get("retry_cost_usd"),
        friction_cost_usd=config_dict.get("friction_cost_usd"),
        decision_threshold=config_dict.get("decision_threshold"),
        min_retry_ev_usd=config_dict.get("min_retry_ev_usd"),
        merchant_vertical=body.get("merchant_vertical"),
    )

    allowlist = set(load_feature_policy()["allowlist"])
    unexpected = sorted(set(body) - allowlist)
    if unexpected:
        raise HTTPException(status_code=422, detail=f"Unexpected features: {unexpected}")


    # v2 OOD refusal at body level (defence-in-depth; runtime.py has parallel check)
    if "response_code" in body and not is_supported_response_code(body["response_code"]):
        rc_explanation = (
            f"Response code `{body['response_code']}` not in trained vocabulary "
            "— model shortcircuited; no per-feature SHAP available"
        ) if include_explanation else None
        result = out_of_scope_response(
            body["response_code"],
            config.to_dict() if hasattr(config, "to_dict") else {},
            explanation=rc_explanation,
        )
    else:
        # D1 (N3) defence-in-depth at body level: refuse unknown
        # country / vertical / processor before predict_one runs.
        cat_off = categorical_out_of_scope(body)
        if cat_off is not None:
            cat_field, cat_value = cat_off
            cat_explanation = (
                f"{cat_field} `{cat_value}` not in trained allowlist "
                "— model shortcircuited; no per-feature SHAP available"
            ) if include_explanation else None
            result = categorical_out_of_scope_response(
                cat_field,
                cat_value,
                config.to_dict() if hasattr(config, "to_dict") else {},
                explanation=cat_explanation,
            )
        else:
            result = predict_one(
                body,
                config=config,
                include_explanation=include_explanation,
                explanation_depth=explanation_depth,
                attempt_num=attempt_num,
            )
    latency_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "prediction served",
        extra={
            "request_id": request_id,
            "latency_ms": latency_ms,
            "prediction": f"{result.get('recommended_action', 'unknown')}|explain={include_explanation}",
            "model_version": load_model_metadata().get("feature_policy_version", "v1"),
            "path": "/predict",
            "api_key_id": api_key.id,
        },
    )
    # Echo version + policy provenance on every response so audit / dispute
    # logs answer "which model + threshold made this call?" without joining
    # against /health snapshots taken at a different point in time.
    metadata = load_model_metadata()
    feature_policy = load_feature_policy()
    body_out = {
        "request_id": request_id,
        "latency_ms": latency_ms,
        "model_version": metadata.get("feature_policy_version", "v1"),
        "policy_version": feature_policy.get("version", "v1"),
        "decision_threshold_used": config.decision_threshold,
        # D9 (N9): renamed from `engine_commit_sha` — the value is a SemVer
        # release identifier, not a git SHA. The ENGINE_COMMIT_SHA env var
        # name is preserved for back-compat with deploy scripts; if a real
        # git SHA is wanted later, populate that env var at build time.
        "engine_version": os.environ.get("ENGINE_COMMIT_SHA", __version__),
        **result,
    }
    _idem_store(request, api_key.id, idempotency_key, body_out)
    return JSONResponse(body_out)
