"""Validator-reviewed server patch — OOD response_code refusal (FINAL).

Reviewed the builder's draft `server_patch_ood_v2.py` against the live droplet
at /opt/project3-recovery/project3_recovery/{runtime,api}.py on 2026-04-24.

Status: builder's patch is correct and integrates cleanly. This file is a
near-identical copy with:
  1. Corrected INTEGRATION POINT 1 line-anchor (runtime.py line 530, the
     response_code normalisation line — NOT BEFORE the hard-decline check
     but the raw payload.get must use `payload.get("response_code")` rather
     than the already-normalised `response_code` var; normaliser handles None).
  2. Corrected INTEGRATION POINT 2 line-anchor (api.py line 443, after the
     Unexpected features raise; confirmed authoritative line location).
  3. Confirmed HARD_DECLINE_CODES in live runtime.py (line 69) exactly matches
     the builder's constant.
  4. Added an explicit note that the allowlist check must be adjacent-to
     (not replacing) existing schema validators.

Everything else in the builder's module (normaliser, out_of_scope_response
body, 422 detail, self-test) is byte-identical. This file re-exports those
helpers so the import path downstream remains stable.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Allow-list — confirmed against live /opt/project3-recovery/project3_recovery/
# runtime.py line 69 HARD_DECLINE_CODES set (validator read on 2026-04-24).
# ---------------------------------------------------------------------------

HARD_DECLINE_CODES: frozenset[str] = frozenset({
    "04", "07", "14", "15", "41", "43", "54", "57", "59", "62", "R0", "R1",
})

# Soft codes the v2 (and v2.1) model trained on, per training_prior_sources_v2.md.
SOFT_DECLINE_CODES: frozenset[str] = frozenset({
    "05", "51", "61", "65", "91", "92", "96", "NW", "PR", "RC",
})

SUPPORTED_RESPONSE_CODES: frozenset[str] = HARD_DECLINE_CODES | SOFT_DECLINE_CODES


# ---------------------------------------------------------------------------
# Categorical allow-lists (D1 / N3) — derived 2026-04-26 from
# `categorical_category_maps()` (live model artifact, runtime.py). Without
# these guards the model silently scores unknown country / vertical / processor
# values via the Pandas Categorical "MISSING" fallback, which contradicts the
# model card's explicit "coverage outside the modeled categories is not
# validated" disclaimer. Any value not in the corresponding allow-list now
# triggers a 200-with-out_of_scope response (matches existing pattern for
# hard-decline shortcircuit and amount > $25K).
# ---------------------------------------------------------------------------

SUPPORTED_MERCHANT_COUNTRIES: frozenset[str] = frozenset({
    "AE", "AR", "AU", "BR", "CA", "CL", "CN", "CO", "DE", "EG", "ES", "FR",
    "GB", "HK", "ID", "IE", "IN", "IT", "JP", "MX", "MY", "NL", "PE", "PH",
    "PL", "SA", "SE", "SG", "TH", "US",
})

SUPPORTED_MERCHANT_VERTICALS: frozenset[str] = frozenset({
    "digital_goods", "ecom", "high_risk", "marketplace", "saas", "travel",
})

SUPPORTED_PROCESSORS: frozenset[str] = frozenset({
    "cross-border-fx-specialist-a", "cross-border-fx-specialist-b",
    "global-acquirer-a", "global-acquirer-b",
    "high-risk-or-orchestrator-a", "high-risk-or-orchestrator-b",
    "regional-bank-processor-a", "regional-bank-processor-b",
    "regional-bank-processor-c",
    "regional-card-specialist-a", "regional-card-specialist-b",
})


def _normalize_code(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s:
        return None
    return s


def is_supported_response_code(raw: Any) -> bool:
    code = _normalize_code(raw)
    return code is not None and code in SUPPORTED_RESPONSE_CODES


def _check_categorical(raw: Any, allowlist: frozenset[str], case_sensitive: bool) -> str | None:
    """Return the offending value (verbatim) if it is provided AND not in
    the allowlist. Returns None when value is missing/empty (the model
    handles MISSING natively) or supported."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    canonical = s if case_sensitive else s.upper()
    pool = allowlist if case_sensitive else frozenset(v.upper() for v in allowlist)
    if canonical in pool:
        return None
    return s


def categorical_out_of_scope(payload: dict[str, Any]) -> tuple[str, str] | None:
    """Return (field_name, value) for the first categorical field that is
    populated with an out-of-allowlist value, else None.

    Country is upper-case canonical (ISO-2). Vertical and processor are
    case-sensitive (training values are lower-case identifiers)."""
    country_offending = _check_categorical(
        payload.get("merchant_country"), SUPPORTED_MERCHANT_COUNTRIES, case_sensitive=False
    )
    if country_offending:
        return ("merchant_country", country_offending)
    vertical_offending = _check_categorical(
        payload.get("merchant_vertical"), SUPPORTED_MERCHANT_VERTICALS, case_sensitive=True
    )
    if vertical_offending:
        return ("merchant_vertical", vertical_offending)
    processor_offending = _check_categorical(
        payload.get("processor_name"), SUPPORTED_PROCESSORS, case_sensitive=True
    )
    if processor_offending:
        return ("processor_name", processor_offending)
    return None


def categorical_out_of_scope_response(
    field: str,
    value: str,
    config_dict: dict[str, Any] | None = None,
    explanation: str | None = None,
) -> dict[str, Any]:
    allowlist_map = {
        "merchant_country": sorted(SUPPORTED_MERCHANT_COUNTRIES),
        "merchant_vertical": sorted(SUPPORTED_MERCHANT_VERTICALS),
        "processor_name": sorted(SUPPORTED_PROCESSORS),
    }
    body = {
        # v4 (N3): emit `recoverability_score` as the canonical name and
        # keep `recovery_probability` as a back-compat alias.
        "recoverability_score": None,
        "recovery_probability": None,  # deprecated alias of recoverability_score
        "recommended_action": "out_of_scope",
        "expected_value": None,
        # D6 (N8): confidence is not meaningful when the model never ran.
        "confidence": None,
        "config_used": config_dict or {},
        "recommended_delay_hours": None,
        "recommended_delay_probability": None,
        "recommended_delay_expected_value": None,
        "delay_curve": [],
        "out_of_scope": True,
        "out_of_scope_reason": (
            f"{field} `{value}` not in trained allowlist "
            f"{allowlist_map.get(field, [])}"
        ),
        f"supported_{field}_values": allowlist_map.get(field, []),
    }
    if explanation is not None:
        body["explanation"] = explanation
        body["top_explanation_features"] = []
    return body


def out_of_scope_response(
    raw: Any,
    config_dict: dict[str, Any] | None = None,
    explanation: str | None = None,
) -> dict[str, Any]:
    code_display = _normalize_code(raw) or "<empty>"
    body = {
        # v4 (N3): see categorical_out_of_scope_response above for naming.
        "recoverability_score": None,
        "recovery_probability": None,  # deprecated alias of recoverability_score
        "recommended_action": "out_of_scope",
        "expected_value": None,
        # D6 (N8): confidence is not meaningful when the model never ran.
        "confidence": None,
        "config_used": config_dict or {},
        "recommended_delay_hours": None,
        "recommended_delay_probability": None,
        "recommended_delay_expected_value": None,
        "delay_curve": [],
        "out_of_scope": True,
        "out_of_scope_reason": (
            f"Response code `{code_display}` is not in the model's trained "
            f"vocabulary. Use a manual review path rather than automated retry."
        ),
        "supported_response_codes": sorted(SUPPORTED_RESPONSE_CODES),
    }
    # D4 (N6): when caller asked for an explanation, return a string saying
    # WHY no per-feature SHAP is available, plus an empty top-features list,
    # rather than letting the field be silently absent.
    if explanation is not None:
        body["explanation"] = explanation
        body["top_explanation_features"] = []
    return body


def unsupported_code_http422_detail(raw: Any) -> dict[str, Any]:
    code_display = _normalize_code(raw) or "<empty>"
    return {
        "error": "response_code out of supported vocabulary",
        "received": code_display,
        "supported_codes": sorted(SUPPORTED_RESPONSE_CODES),
    }


# ---------------------------------------------------------------------------
# CORRECTED — Integration point 1 (runtime.py predict_one, ~line 530)
# ---------------------------------------------------------------------------
# Live runtime.py confirmed (2026-04-24):
#   line 520: def predict_one(
#   line 530: response_code = str(payload.get("response_code", "") or "").strip().upper()
#   line 531: if response_code in HARD_DECLINE_CODES:
#
# Patch: insert BEFORE line 530, immediately after the function header /
# docstring (validator confirms payload.get("response_code") is still the raw
# pre-normalised value at that point).
#
#     # v2.1 — OOD refusal short-circuit (adversarial finding P1.2).
#     response_code_raw = payload.get("response_code")
#     if not is_supported_response_code(response_code_raw):
#         return out_of_scope_response(response_code_raw, config.to_dict() if hasattr(config, "to_dict") else {})
#
#     # existing:
#     response_code = str(payload.get("response_code", "") or "").strip().upper()
#     if response_code in HARD_DECLINE_CODES:
#         ...
#
# NOTE on config.to_dict: verify `config` binding exists in predict_one scope
# at the insertion point (it does — `config` is a param of predict_one, line
# 520+). The `to_dict()` method exists on the Config dataclass at line 93-105.
#
# ---------------------------------------------------------------------------
# CORRECTED — Integration point 2 (api.py /predict handler, line 441-445)
# ---------------------------------------------------------------------------
# Live api.py confirmed (2026-04-24):
#   line 373: @app.post("/predict", tags=["predict"])
#   line 374: def predict(
#   line 440: allowlist = set(load_feature_policy()["allowlist"])
#   line 441: unexpected = sorted(set(body) - allowlist)
#   line 443:     raise HTTPException(status_code=422, detail=f"Unexpected features: {unexpected}")
#
# Patch: insert AFTER line 443 closes, BEFORE line 445 `result = predict_one(...)`.
#
#     # NEW — v2.1 OOD refusal (body-level; runtime.py has a parallel
#     # defence-in-depth check).
#     if "response_code" in body and not is_supported_response_code(body["response_code"]):
#         if strict:  # new FastAPI Query param (see below)
#             raise HTTPException(
#                 status_code=422,
#                 detail=unsupported_code_http422_detail(body["response_code"]),
#             )
#         # Default — return 200 with out_of_scope (matches amount-cap behavior).
#         return {
#             "request_id": request_id,
#             "latency_ms": int((time.perf_counter() - started) * 1000),
#             **out_of_scope_response(body["response_code"], config.to_dict() if hasattr(config, "to_dict") else {}),
#         }
#
# Query param — in the `def predict(` signature add:
#     strict: bool = Query(default=False, description="Return 4xx for unsupported codes")
#
# Imports — api.py already imports `Query` via FastAPI (verified on droplet).
#
# ---------------------------------------------------------------------------
# CORRECTED — Feature policy JSON edit (optional, validator says SKIP)
# ---------------------------------------------------------------------------
# Builder's draft proposes adding a `response_code_enum` key to
# project3_feature_policy.json. The validator notes the HARD_DECLINE_CODES
# live in runtime.py (line 69) and SUPPORTED_RESPONSE_CODES lives in this
# module. Adding a third location of truth invites drift. RECOMMEND: keep
# the enum in THIS module only; do not edit the JSON. Ivan can revisit if
# he wants a single source-of-truth for external consumers.


def _self_test() -> None:
    for code in ["05", "51", "91", "NW", "04", "57"]:
        assert is_supported_response_code(code), f"{code} should be supported"
    for code in ["X99", "ZZZ", "999", "00", "", None, " ", "abc"]:
        assert not is_supported_response_code(code), f"{code} should be refused"
    body = out_of_scope_response("X99", {"margin_rate": 0.35})
    assert body["recommended_action"] == "out_of_scope"
    assert body["recovery_probability"] is None
    assert body["out_of_scope"] is True
    assert body["confidence"] is None, "D6: confidence must be null for OOD code"
    assert "X99" in body["out_of_scope_reason"]
    assert "05" in body["supported_response_codes"]
    body_explained = out_of_scope_response("X99", {}, explanation="model shortcircuited; no SHAP available")
    assert "explanation" in body_explained, "D4: explanation must be returned when requested"
    assert body_explained["top_explanation_features"] == []
    # D1 categorical allowlist
    assert categorical_out_of_scope({"merchant_country": "XX"}) == ("merchant_country", "XX")
    assert categorical_out_of_scope({"merchant_country": "us"}) is None  # case-insensitive country
    assert categorical_out_of_scope({"merchant_vertical": "unicorn-vertical"}) == ("merchant_vertical", "unicorn-vertical")
    assert categorical_out_of_scope({"merchant_vertical": "ecom"}) is None
    assert categorical_out_of_scope({"processor_name": "zzz-fake"}) == ("processor_name", "zzz-fake")
    assert categorical_out_of_scope({"processor_name": "global-acquirer-a"}) is None
    assert categorical_out_of_scope({}) is None  # empty payload — model handles MISSING natively
    cat_body = categorical_out_of_scope_response("merchant_country", "XX", {"margin_rate": 0.35})
    assert cat_body["recommended_action"] == "out_of_scope"
    assert cat_body["confidence"] is None
    assert "XX" in cat_body["out_of_scope_reason"]
    assert "BR" in cat_body["supported_merchant_country_values"]
    print("[server_patch_ood_v2_final] self-test OK")


if __name__ == "__main__":
    _self_test()
