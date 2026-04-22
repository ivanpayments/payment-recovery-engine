from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, create_model

from project3_runtime import (
    build_model_card,
    categorical_category_maps,
    decision_threshold,
    load_feature_policy,
    load_model,
    load_model_metadata,
    load_reference_frame,
    load_shap_explainer,
    predict_one,
    top_global_features,
)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        for field in ("request_id", "latency_ms", "prediction", "model_version", "path"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload)


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
log = logging.getLogger("project3.api")


class FeaturePayloadBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


FeaturePayload = create_model(
    "FeaturePayload",
    __base__=FeaturePayloadBase,
    **{name: (Any | None, None) for name in load_model_metadata()["feature_columns"]},
)


app = FastAPI(title="Project 3 ML Payment Recovery Engine")
PORT = int(os.getenv("PROJECT3_PORT", "8000"))


@app.on_event("startup")
def startup_event() -> None:
    load_model()
    load_reference_frame()
    categorical_category_maps()
    top_global_features()
    build_model_card()
    load_shap_explainer()
    log.info(
        "model loaded",
        extra={
            "model_version": load_model_metadata().get("feature_policy_version", "v1"),
            "prediction": "startup",
            "path": "/startup",
        },
    )


@app.get("/health")
def health() -> dict[str, Any]:
    metadata = load_model_metadata()
    return {
        "ok": True,
        "model_loaded": True,
        "model_version": metadata.get("feature_policy_version", "v1"),
        "decision_threshold": decision_threshold(),
        "feature_count": len(metadata.get("feature_columns", [])),
    }


@app.get("/model-card")
def model_card() -> dict[str, Any]:
    return build_model_card()


@app.post("/predict")
def predict(
    payload: FeaturePayload,
    request: Request,
    include_explanation: bool = Query(default=False),
    explanation_depth: int = Query(default=5, ge=1, le=10),
) -> dict[str, Any]:
    started = time.perf_counter()
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    body = payload.model_dump(exclude_none=True)

    allowlist = set(load_feature_policy()["allowlist"])
    unexpected = sorted(set(body) - allowlist)
    if unexpected:
        raise HTTPException(status_code=422, detail=f"Unexpected features: {unexpected}")

    result = predict_one(
        body,
        include_explanation=include_explanation,
        explanation_depth=explanation_depth,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "prediction served",
        extra={
            "request_id": request_id,
            "latency_ms": latency_ms,
            "prediction": f"{result['recommended_action']}|explain={include_explanation}",
            "model_version": load_model_metadata().get("feature_policy_version", "v1"),
            "path": "/predict",
        },
    )
    return {
        "request_id": request_id,
        "latency_ms": latency_ms,
        **result,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
