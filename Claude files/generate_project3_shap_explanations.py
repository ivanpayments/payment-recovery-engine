from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from train_project3_lightgbm import MODEL_META, MODEL_TXT, load_data, temporal_split


ROOT = Path(__file__).resolve().parent
GLOBAL_CSV = ROOT / "project3_shap_global_importance.csv"
LOCAL_CSV = ROOT / "project3_shap_local_explanations.csv"
REPORT_MD = ROOT / "project3_shap_explanations.md"

TARGET = "target_recovered_by_retry"


def load_model_and_metadata():
    import lightgbm as lgb

    if not MODEL_TXT.exists() or not MODEL_META.exists():
        raise SystemExit("Model artifacts missing. Run train_project3_lightgbm.py first.")

    booster = lgb.Booster(model_file=str(MODEL_TXT))
    metadata = json.loads(MODEL_META.read_text(encoding="utf-8"))
    return booster, metadata


def prepare_features(df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
    feature_cols = metadata["feature_columns"]
    categorical_cols = set(metadata["categorical_columns"])
    work = df[feature_cols].copy()

    for col in feature_cols:
        if col in categorical_cols:
            work[col] = work[col].fillna("MISSING").astype(str)
        elif pd.api.types.is_bool_dtype(work[col]):
            work[col] = work[col].fillna(False).astype(int)
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    return work


from project3_runtime import business_phrase  # noqa: E402  (re-exported for callers)


def main() -> None:
    import shap

    booster, metadata = load_model_and_metadata()
    df = load_data()
    train_df, _, test_df = temporal_split(df)
    X_test = prepare_features(test_df, metadata)
    X_train = prepare_features(train_df, metadata)

    for col in metadata["categorical_columns"]:
        cats = sorted(X_train[col].fillna("MISSING").astype(str).unique().tolist())
        X_test[col] = pd.Categorical(X_test[col].fillna("MISSING").astype(str), categories=cats)

    sample_size = min(500, len(X_test))
    sample_df = X_test.head(sample_size).copy()
    sample_meta = test_df.head(sample_size).copy()

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(sample_df)

    if isinstance(shap_values, list):
        shap_matrix = np.array(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        shap_matrix = np.array(shap_values)

    if shap_matrix.shape[1] == sample_df.shape[1] + 1:
        shap_matrix = shap_matrix[:, :-1]

    global_importance = pd.DataFrame(
        {
            "feature": sample_df.columns,
            "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    global_importance.to_csv(GLOBAL_CSV, index=False)

    predictions = booster.predict(sample_df, num_iteration=metadata.get("best_iteration"))
    local_rows: List[Dict] = []

    top_indices = np.argsort(predictions)[-20:][::-1]
    for idx in top_indices:
        row_features = sample_df.iloc[idx]
        row_meta = sample_meta.iloc[idx]
        row_shap = shap_matrix[idx]
        top_feature_idx = np.argsort(np.abs(row_shap))[-5:][::-1]
        for rank, feat_idx in enumerate(top_feature_idx, start=1):
            feature = sample_df.columns[feat_idx]
            shap_value = float(row_shap[feat_idx])
            value = row_features[feature]
            local_rows.append(
                {
                    "row_index": int(idx),
                    "rank": rank,
                    "timestamp": row_meta["timestamp"],
                    "merchant_country": row_meta["merchant_country"],
                    "processor_name": row_meta["processor_name"],
                    "response_code": row_meta["response_code"],
                    "amount_usd": row_meta["amount_usd"],
                    "target_recovered_by_retry": row_meta[TARGET],
                    "predicted_recovery_probability": float(predictions[idx]),
                    "feature": feature,
                    "feature_value": value,
                    "shap_value": shap_value,
                    "business_explanation": business_phrase(feature, value, shap_value),
                }
            )

    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(LOCAL_CSV, index=False)

    lines = [
        "# Project 3 SHAP Explanations",
        "",
        "## Global Readout",
        "- The global table shows which features most strongly influence recoverability predictions on the sampled holdout set.",
        "- This is the bridge between model performance and explainable product behavior.",
        "",
        "## Top Global Drivers",
    ]
    for _, row in global_importance.head(15).iterrows():
        lines.append(f"- `{row['feature']}`: mean |SHAP| = `{row['mean_abs_shap']:.4f}`")

    lines.extend(["", "## Local Example Explanations"])
    grouped = local_df.groupby("row_index", sort=False)
    shown = 0
    for row_index, group in grouped:
        first = group.iloc[0]
        lines.append(
            f"- Example row `{row_index}`: `{first['timestamp']}` `{first['merchant_country']}` `{first['processor_name']}` code `{first['response_code']}` amount `${float(first['amount_usd']):,.2f}` predicted recoverability `{first['predicted_recovery_probability']:.3f}`"
        )
        for _, item in group.iterrows():
            lines.append(f"  - {item['business_explanation']}")
        shown += 1
        if shown >= 5:
            break

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(GLOBAL_CSV)
    print(LOCAL_CSV)
    print(REPORT_MD)


if __name__ == "__main__":
    main()
