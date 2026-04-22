from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from project3_runtime import categorical_category_maps
from train_project3_baseline_v2 import (
    TARGET,
    build_matrix,
    feature_lists,
    fit_numeric_scaler,
    fit_target_encoding,
    load_data,
    sigmoid,
    temporal_split,
    train_logistic_regression,
)
from train_project3_lightgbm import MODEL_META, MODEL_TXT, coerce_feature_types


ROOT = Path(__file__).resolve().parent
RECON_CSV = ROOT / "project3_champion_challenger_reconciliation.csv"
RECON_REPORT = ROOT / "project3_champion_challenger_report.md"


def load_lightgbm():
    import json
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(MODEL_TXT))
    meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
    return booster, meta


def prepare_lightgbm(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    X_df, _, _, categorical_cols = coerce_feature_types(df, meta["feature_columns"])
    category_maps = categorical_category_maps()
    for col in categorical_cols:
        cats = category_maps.get(col, ["MISSING"])
        X_df[col] = pd.Categorical(X_df[col].fillna("MISSING").astype(str), categories=cats)
    return X_df


def main() -> None:
    df = load_data()
    train_df, val_df, test_df = temporal_split(df)
    numeric_cols, bool_cols, categorical_cols = feature_lists(df)

    numeric_stats = fit_numeric_scaler(train_df, numeric_cols)
    encoders, global_mean = fit_target_encoding(train_df, categorical_cols, TARGET)

    X_train, _ = build_matrix(train_df, numeric_cols, bool_cols, categorical_cols, numeric_stats, encoders, global_mean)
    X_val, _ = build_matrix(val_df, numeric_cols, bool_cols, categorical_cols, numeric_stats, encoders, global_mean)
    X_test, feature_names = build_matrix(test_df, numeric_cols, bool_cols, categorical_cols, numeric_stats, encoders, global_mean)
    _ = feature_names
    y_train = train_df[TARGET].astype(int).to_numpy()
    y_val = val_df[TARGET].astype(int).to_numpy()
    challenger_weights, challenger_bias = train_logistic_regression(X_train, y_train, X_val, y_val)
    challenger_probs = sigmoid(X_test @ challenger_weights + challenger_bias)

    champion_model, champion_meta = load_lightgbm()
    champion_X = prepare_lightgbm(test_df, champion_meta)
    champion_probs = champion_model.predict(champion_X, num_iteration=champion_meta.get("best_iteration"))

    out = test_df[["timestamp", "merchant_country", "processor_name", "response_code", "amount_usd", TARGET]].copy()
    out["champion_probability"] = champion_probs
    out["challenger_probability"] = challenger_probs
    out["probability_delta"] = out["champion_probability"] - out["challenger_probability"]
    out["absolute_delta"] = out["probability_delta"].abs()
    out.sort_values("absolute_delta", ascending=False).to_csv(RECON_CSV, index=False)

    disagreement = float((out["absolute_delta"] > 0.20).mean())
    lines = [
        "# Project 3 Champion Challenger Reconciliation",
        "",
        "## Setup",
        "- Champion: LightGBM model used in serving",
        "- Challenger: baseline_v2 logistic regression with target encoding",
        f"- Test rows reconciled: `{len(out)}`",
        "",
        "## Readout",
        f"- Mean champion probability: `{out['champion_probability'].mean():.3f}`",
        f"- Mean challenger probability: `{out['challenger_probability'].mean():.3f}`",
        f"- Mean absolute delta: `{out['absolute_delta'].mean():.3f}`",
        f"- Share of rows with |delta| > 0.20: `{disagreement:.2%}`",
        "",
        "## Largest Disagreements",
    ]
    for _, row in out.sort_values("absolute_delta", ascending=False).head(12).iterrows():
        lines.append(
            f"- `{row['timestamp']}` `{row['merchant_country']}` `{row['processor_name']}` code `{row['response_code']}`: champion `{row['champion_probability']:.3f}` vs challenger `{row['challenger_probability']:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Large positive deltas indicate scenarios where the tree model is finding nonlinear recovery signal the linear challenger does not capture.",
            "- Large negative deltas are useful review candidates for feature interactions or calibration drift.",
        ]
    )
    RECON_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(RECON_REPORT)
    print(RECON_CSV)


if __name__ == "__main__":
    main()
