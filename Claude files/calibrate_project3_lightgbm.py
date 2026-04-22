from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from project3_runtime import categorical_category_maps
from train_project3_lightgbm import (
    MODEL_META,
    MODEL_TXT,
    TARGET,
    brier_score,
    coerce_feature_types,
    load_data,
    load_feature_policy,
    log_loss,
    metrics,
    roc_auc_score_manual,
    select_features,
    temporal_split,
)


ROOT = Path(__file__).resolve().parent
CALIBRATOR_JSON = ROOT / "project3_isotonic_calibrator.json"
CALIBRATION_REPORT = ROOT / "project3_lightgbm_calibration_report.md"
CALIBRATED_PREDICTIONS = ROOT / "project3_lightgbm_calibrated_predictions_sample.csv"


def load_model_and_metadata():
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(MODEL_TXT))
    metadata = json.loads(MODEL_META.read_text(encoding="utf-8"))
    return booster, metadata


def prepare_features_for_scoring(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    feature_cols = metadata["feature_columns"]
    X_df, _, _, categorical_cols = coerce_feature_types(df, feature_cols)
    category_maps = categorical_category_maps()
    for col in categorical_cols:
        cats = category_maps.get(col, ["MISSING"])
        X_df[col] = pd.Categorical(X_df[col].fillna("MISSING").astype(str), categories=cats)
    return X_df


def calibration_curve_points(y_true: np.ndarray, y_prob: np.ndarray, buckets: int = 10) -> list[dict[str, float]]:
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values("y_prob")
    df["bucket"] = pd.qcut(df["y_prob"], q=min(buckets, len(df)), duplicates="drop")
    grouped = df.groupby("bucket", observed=False).agg(mean_pred=("y_prob", "mean"), event_rate=("y_true", "mean"))
    return grouped.reset_index(drop=True).to_dict(orient="records")


def main() -> None:
    booster, metadata = load_model_and_metadata()
    policy = load_feature_policy()
    df = load_data()
    _, val_df, test_df = temporal_split(df)
    feature_cols = select_features(df, policy)
    _ = feature_cols

    X_val_df = prepare_features_for_scoring(val_df, metadata)
    X_test_df = prepare_features_for_scoring(test_df, metadata)
    y_val = val_df[TARGET].astype(int).to_numpy()
    y_test = test_df[TARGET].astype(int).to_numpy()

    val_probs = booster.predict(X_val_df, num_iteration=metadata.get("best_iteration"))
    test_probs = booster.predict(X_test_df, num_iteration=metadata.get("best_iteration"))

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs, y_val)
    calibrated_test_probs = calibrator.transform(test_probs)
    threshold = float(metadata.get("decision_threshold", 0.19))

    uncalibrated = metrics(y_test, test_probs, threshold)
    calibrated = metrics(y_test, calibrated_test_probs, threshold)

    calibrator_payload = {
        "x_thresholds": [float(x) for x in calibrator.X_thresholds_],
        "y_thresholds": [float(y) for y in calibrator.y_thresholds_],
        "fit_split": "validation",
        "base_model": MODEL_TXT.name,
    }
    CALIBRATOR_JSON.write_text(json.dumps(calibrator_payload, indent=2), encoding="utf-8")

    sample = test_df[["timestamp", "merchant_country", "processor_name", "response_code", "amount_usd", TARGET]].copy()
    sample["raw_probability"] = test_probs
    sample["calibrated_probability"] = calibrated_test_probs
    sample.sort_values("calibrated_probability", ascending=False).head(200).to_csv(CALIBRATED_PREDICTIONS, index=False)

    curve_raw = calibration_curve_points(y_test, test_probs)
    curve_cal = calibration_curve_points(y_test, calibrated_test_probs)
    lines = [
        "# Project 3 Isotonic Calibration Report",
        "",
        "## Setup",
        "- Base model: LightGBM",
        "- Calibration method: IsotonicRegression fitted on validation predictions",
        f"- Threshold reused for downstream metrics: `{threshold:.2f}`",
        "",
        "## Test Metrics Before Calibration",
        f"- AUC: `{uncalibrated['auc']:.3f}`",
        f"- Log loss: `{uncalibrated['log_loss']:.3f}`",
        f"- Brier score: `{uncalibrated['brier']:.3f}`",
        "",
        "## Test Metrics After Calibration",
        f"- AUC: `{calibrated['auc']:.3f}`",
        f"- Log loss: `{calibrated['log_loss']:.3f}`",
        f"- Brier score: `{calibrated['brier']:.3f}`",
        "",
        "## Readout",
        f"- Brier delta (after - before): `{calibrated['brier'] - uncalibrated['brier']:.4f}`",
        f"- Log-loss delta (after - before): `{calibrated['log_loss'] - uncalibrated['log_loss']:.4f}`",
        "- Use the calibrated score when merchant-facing probability quality matters more than raw ranking purity.",
        "",
        "## Calibration Curve Snapshot",
    ]
    for idx, (raw_row, cal_row) in enumerate(zip(curve_raw[:6], curve_cal[:6]), start=1):
        lines.append(
            f"- Bucket `{idx}`: raw mean `{raw_row['mean_pred']:.3f}` vs event `{raw_row['event_rate']:.3f}`; calibrated mean `{cal_row['mean_pred']:.3f}` vs event `{cal_row['event_rate']:.3f}`"
        )
    CALIBRATION_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(CALIBRATOR_JSON)
    print(CALIBRATION_REPORT)
    print(CALIBRATED_PREDICTIONS)


if __name__ == "__main__":
    main()
