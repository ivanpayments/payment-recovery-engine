from pathlib import Path

import pandas as pd


PROJECT2_CSV = Path(
    r"C:\Users\ivana\OneDrive\Everything\Pet projects\AI tools\Payments bots\Project 2\Claude files\routing_transactions.csv"
)
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "project3_modeling_table.csv"
OUTPUT_MD = ROOT / "project3_modeling_profile.md"
OUTPUT_FEATURE_GUIDE = ROOT / "project3_feature_coverage.md"


def load_base_dataframe() -> pd.DataFrame:
    # Keep the original row as complete as possible so feature selection can happen later.
    df = pd.read_csv(PROJECT2_CSV, engine="python", on_bad_lines="skip")
    bool_cols = [
        "is_cross_border",
        "is_token",
        "is_soft_decline",
        "three_ds_requested",
        "is_retry",
        "is_chargeback",
        "fraud_flag",
        "fx_applied",
        "network_token_present",
        "timeout_flag",
        "is_mit",
        "account_updater_used",
        "mit_flag_revoked",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(str).str.lower().map(
                {"true": True, "false": False}
            ).fillna(df[col])
            df[col] = df[col].astype(bool)
    df["retry_attempt_num"] = pd.to_numeric(df["retry_attempt_num"], errors="coerce").fillna(0).astype(int)
    df["hours_since_original"] = pd.to_numeric(df["hours_since_original"], errors="coerce")
    df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    return df


def build_modeling_table(df: pd.DataFrame) -> pd.DataFrame:
    original_declines = df[(df["auth_status"] == "DECLINED") & (~df["is_retry"])].copy()
    retries = df[df["is_retry"]].copy()
    retries["original_transaction_id"] = retries["original_transaction_id"].fillna("")

    retry_group = retries.groupby("original_transaction_id", dropna=False)

    retry_summary = pd.DataFrame(index=original_declines["transaction_id"])
    retry_summary["has_any_retry"] = retry_summary.index.isin(retry_group.groups.keys())
    retry_summary["retry_count"] = retry_summary.index.map(retry_group.size()).fillna(0).astype(int)
    retry_summary["target_recovered_by_retry"] = (
        retry_summary.index.map(retry_group["auth_status"].apply(lambda s: (s == "APPROVED").any()))
        .fillna(False)
        .astype(bool)
    )
    retry_summary["target_first_retry_approved"] = False

    first_retry = (
        retries.sort_values(["original_transaction_id", "retry_attempt_num"])
        .drop_duplicates("original_transaction_id", keep="first")
        .loc[:, ["original_transaction_id", "auth_status", "retry_attempt_num", "hours_since_original"]]
        .rename(
            columns={
                "auth_status": "first_retry_status",
                "retry_attempt_num": "first_retry_attempt_num",
                "hours_since_original": "hours_to_first_retry",
            }
        )
    )
    retry_summary = retry_summary.reset_index().rename(columns={"index": "transaction_id"})
    retry_summary = retry_summary.merge(
        first_retry, how="left", left_on="transaction_id", right_on="original_transaction_id"
    )
    retry_summary["target_first_retry_approved"] = retry_summary["first_retry_status"].eq("APPROVED")

    eventual_recovery = (
        retries[retries["auth_status"] == "APPROVED"]
        .sort_values(["original_transaction_id", "retry_attempt_num"])
        .drop_duplicates("original_transaction_id", keep="first")
        .loc[:, ["original_transaction_id", "retry_attempt_num", "hours_since_original"]]
        .rename(
            columns={
                "retry_attempt_num": "recovery_attempt_num",
                "hours_since_original": "hours_to_recovery",
            }
        )
    )
    retry_summary = retry_summary.merge(
        eventual_recovery, how="left", left_on="transaction_id", right_on="original_transaction_id"
    )

    modeling_table = original_declines.merge(retry_summary, how="left", on="transaction_id")
    modeling_table["has_any_retry"] = modeling_table["has_any_retry"].fillna(False).astype(bool)
    modeling_table["retry_count"] = modeling_table["retry_count"].fillna(0).astype(int)
    modeling_table["target_recovered_by_retry"] = (
        modeling_table["target_recovered_by_retry"].fillna(False).astype(bool)
    )
    modeling_table["target_first_retry_approved"] = (
        modeling_table["target_first_retry_approved"].fillna(False).astype(bool)
    )
    return modeling_table


def profile_modeling_table(df: pd.DataFrame) -> str:
    total = len(df)
    positive = int(df["target_recovered_by_retry"].sum())
    first_positive = int(df["target_first_retry_approved"].sum())
    retried = int(df["has_any_retry"].sum())
    base_cols = len([c for c in df.columns if not c.startswith("target_") and c not in {
        "has_any_retry",
        "retry_count",
        "first_retry_status",
        "first_retry_attempt_num",
        "hours_to_first_retry",
        "recovery_attempt_num",
        "hours_to_recovery",
        "original_transaction_id_y",
    }])
    lines = [
        "# Project 3 Modeling Table Profile",
        "",
        "## Overview",
        f"- rows: `{total}`",
        f"- columns carried into the modeling table: `{len(df.columns)}`",
        f"- original-row context columns preserved: `{base_cols}`",
        f"- positive label (`target_recovered_by_retry`): `{positive}` ({positive / total:.2%})",
        f"- first-retry positive label: `{first_positive}` ({first_positive / total:.2%})",
        f"- rows with any retry chain: `{retried}` ({retried / total:.2%})",
        "",
        "## Recovery by Soft/Hard Decline",
    ]

    soft_hard = (
        df.groupby("is_soft_decline")
        .agg(
            rows=("transaction_id", "count"),
            recovered=("target_recovered_by_retry", "sum"),
        )
        .reset_index()
    )
    for _, row in soft_hard.iterrows():
        label = "soft decline" if row["is_soft_decline"] else "hard decline"
        rate = (row["recovered"] / row["rows"]) if row["rows"] else 0.0
        lines.append(f"- {label}: `{int(row['rows'])}` rows, `{int(row['recovered'])}` recovered ({rate:.2%})")

    lines.extend(["", "## Largest Segments", "### By response code"])
    response_code = (
        df.groupby("response_code")
        .agg(
            rows=("transaction_id", "count"),
            recovered=("target_recovered_by_retry", "sum"),
        )
        .sort_values("rows", ascending=False)
        .head(10)
    )
    for code, row in response_code.iterrows():
        rate = (row["recovered"] / row["rows"]) if row["rows"] else 0.0
        lines.append(f"- `{code}`: `{int(row['rows'])}` rows, recovery `{rate:.2%}`")

    lines.extend(["", "### By merchant country"])
    country = (
        df.groupby("merchant_country")
        .agg(
            rows=("transaction_id", "count"),
            recovered=("target_recovered_by_retry", "sum"),
        )
        .sort_values("rows", ascending=False)
        .head(10)
    )
    for code, row in country.iterrows():
        rate = (row["recovered"] / row["rows"]) if row["rows"] else 0.0
        lines.append(f"- `{code}`: `{int(row['rows'])}` rows, recovery `{rate:.2%}`")

    lines.extend(["", "### By processor"])
    processor = (
        df.groupby("processor_name")
        .agg(
            rows=("transaction_id", "count"),
            recovered=("target_recovered_by_retry", "sum"),
        )
        .sort_values("rows", ascending=False)
        .head(10)
    )
    for name, row in processor.iterrows():
        rate = (row["recovered"] / row["rows"]) if row["rows"] else 0.0
        lines.append(f"- `{name}`: `{int(row['rows'])}` rows, recovery `{rate:.2%}`")

    lines.extend(
        [
            "",
            "## Sufficiency Read",
            "- The dataset is large enough for a baseline tabular model.",
            "- The positive class is healthy enough for MVP modeling, but some segment-level slices will still be sparse.",
            "- This supports training a first recovery-probability model before generating more data.",
            "- We should revisit data expansion only after checking model stability across major segments.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_feature_guide(df: pd.DataFrame) -> str:
    derived_cols = {
        "has_any_retry",
        "retry_count",
        "first_retry_status",
        "first_retry_attempt_num",
        "hours_to_first_retry",
        "recovery_attempt_num",
        "hours_to_recovery",
        "target_recovered_by_retry",
        "target_first_retry_approved",
        "original_transaction_id_y",
    }
    generally_exclude_from_training = {
        "transaction_id",
        "original_transaction_id",
        "original_transaction_id_y",
        "psp_transaction_id",
        "psp_reference",
        "network_transaction_id",
        "stan",
        "rrn",
        "arn",
        "session_id",
        "correlation_id",
        "trace_id",
        "device_fingerprint",
        "auth_code",
        "captured_at",
        "refunded_at",
        "voided_at",
        "settled_at",
        "refunded_amount",
        "captured_amount",
        "approved_amount",
    }
    candidate_cols = [
        c for c in df.columns
        if c not in derived_cols and c not in generally_exclude_from_training
    ]
    lines = [
        "# Project 3 Feature Coverage",
        "",
        "## Summary",
        f"- total modeling-table columns: `{len(df.columns)}`",
        f"- candidate original-decline columns available for feature engineering: `{len(candidate_cols)}`",
        f"- derived label or retry-summary columns: `{len(derived_cols)}`",
        f"- columns generally excluded from training due to identifiers or post-decision data: `{len(generally_exclude_from_training)}`",
        "",
        "## Rich Feature Areas Available",
        "- transaction economics: amount, amount_usd, currency, processor fees, interchange estimate, scheme fee, fx flags, fx rate, settlement currency",
        "- payment context: card brand, card type, funding source, tokenization, wallet type, payment method, BIN, present mode",
        "- decline context: response_code, response_message, decline_bucket, is_soft_decline, scheme_response_code, timeout flag, mastercard advice code",
        "- geography: merchant country, card country, issuer country, billing country, shipping country, IP country, cross-border status",
        "- authentication: three_ds_requested, three_ds_outcome, three_ds_version, three_ds_flow, three_ds_eci, sca_exemption",
        "- latency and ops: latency_ms, latency_auth_ms, latency_3ds_ms, latency_bucket, routing metadata",
        "- risk context: risk_score, fraud_flag, chargeback flag, issuer size, MIT flag, account updater used",
        "- device and channel context: device_os, user agent family, contactless / wallet-related fields, recurring / stored credential context",
        "",
        "## Modeling Rule",
        "- We should keep as many original-decline columns as possible in the modeling table.",
        "- During actual model training, we should only use fields available at the moment the original decline happened.",
        "- Identifier columns and post-decision settlement fields can remain in the table for traceability but should not be used as model features.",
        "",
        "## Candidate Columns",
    ]
    for col in candidate_cols:
        lines.append(f"- `{col}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load_base_dataframe()
    modeling_table = build_modeling_table(df)
    modeling_table.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_MD.write_text(profile_modeling_table(modeling_table), encoding="utf-8")
    OUTPUT_FEATURE_GUIDE.write_text(build_feature_guide(modeling_table), encoding="utf-8")
    print(OUTPUT_CSV)
    print(OUTPUT_MD)
    print(OUTPUT_FEATURE_GUIDE)


if __name__ == "__main__":
    main()
