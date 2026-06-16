from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["run_id", "broad_no"]

STATE_COLS = [
    "minute_idx",
    "log_viewer", "log_chat", "log_unique",
    "viewer_chat_gap", "viewer_unique_gap",
    "zero_chat", "zero_run_len", "log_zero_run_len",
    "viewer_bin", "expected_log_chat_bin", "expected_log_unique_bin",
    "chat_deficit", "unique_deficit",
    "rolling_chat_deficit_5m", "rolling_chat_deficit_10m",
    "rolling_unique_deficit_5m", "rolling_unique_deficit_10m",
    "rolling_zero_rate_5m", "rolling_zero_rate_10m",
    "rolling_chat_sum_5m", "rolling_unique_sum_5m",
]

RAW_MINUTE_COLS = [
    "source_file",
    "run_id", "broad_no", "session_key", "user_id", "category_id", "minute_ts",
    "viewer_count_last", "chat_count", "unique_chatters", "avg_msg_len",
    "repeat_msg_ratio", "new_chatter_ratio", "chat_per_viewer",
    "delta_viewer_1m", "delta_chat_1m",
]

FEATURE_MINUTE_COLS = RAW_MINUTE_COLS + STATE_COLS

# Backward-compatible alias for older callers.
MINUTE_COLS = FEATURE_MINUTE_COLS

NUM_COLS = [
    "viewer_count_last", "chat_count", "unique_chatters", "avg_msg_len",
    "repeat_msg_ratio", "new_chatter_ratio", "chat_per_viewer",
    "delta_viewer_1m", "delta_chat_1m"
]


def raw_minute_cols(df):
    return [c for c in RAW_MINUTE_COLS if c in df.columns]


def feat_minute_cols(df):
    return [c for c in FEATURE_MINUTE_COLS if c in df.columns]


def minute_cols(df):
    return feat_minute_cols(df)


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def id_str(s):
    """Keep numeric and string broadcast IDs stable without creating 1101.0 text."""
    x = s.astype("string").str.strip().replace("", pd.NA)
    as_num = pd.to_numeric(x, errors="coerce")
    int_like = as_num.notna() & np.isclose(as_num, np.floor(as_num))
    x = x.mask(int_like, as_num.round().astype("Int64").astype("string"))
    return x


def fill_id(df, col, value):
    if col not in df.columns:
        df[col] = np.nan
    df[col] = df.groupby(KEY)[col].transform(lambda x: x.ffill().bfill())
    df[col] = df[col].fillna(value)
    return df


def fill_metric(df, col, flag):
    bad = df[col].isna() & df["chat_count"].gt(0)
    df[flag] = bad.astype(int)
    med = df.groupby(KEY)[col].transform("median")
    df[col] = df[col].fillna(med).fillna(0)
    return df


def clean_viewer(g, cfg):
    g = g.sort_values("minute_ts").copy()
    v = to_num(g["viewer_count_last"]).mask(lambda x: x.lt(0))
    if cfg["prep"].get("zero_viewer_na", True):
        v = v.mask(v.eq(0))

    miss = v.isna()
    g["v_miss_r"] = float(miss.mean()) if len(g) else np.nan
    g["v_edge"] = int(bool(len(g) and (miss.iloc[0] or miss.iloc[-1])))
    g["v_qc"] = int(miss.all())
    g["viewer_count_last"] = v.interpolate(method="linear", limit_direction="both").clip(lower=0)
    return g


def prep_minute(df, cfg):
    df = df.copy()
    df["minute_ts"] = pd.to_datetime(df["minute_ts"], errors="coerce")

    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = to_num(df[c])

    df["run_id"] = to_num(df["run_id"])
    df["broad_no"] = id_str(df["broad_no"])
    df = df.dropna(subset=["run_id", "broad_no", "minute_ts"]).copy()

    df["run_id"] = df["run_id"].astype(int)
    df = df.sort_values(KEY + ["minute_ts"]).reset_index(drop=True)

    df = fill_id(df, "user_id", "UNKNOWN_USER")
    df = fill_id(df, "category_id", "UNKNOWN_CAT")
    df = df.groupby(KEY, group_keys=False).apply(lambda g: clean_viewer(g, cfg)).reset_index(drop=True)

    df["chat_count"] = df["chat_count"].fillna(0).clip(lower=0)
    df["unique_chatters"] = df["unique_chatters"].fillna(0).clip(lower=0)

    no_chat = df["chat_count"].eq(0)
    for c in ["avg_msg_len", "repeat_msg_ratio", "new_chatter_ratio"]:
        df[c] = df[c].mask(no_chat & df[c].isna(), 0)

    df = fill_metric(df, "avg_msg_len", "avg_qc")
    df = fill_metric(df, "repeat_msg_ratio", "rep_qc")
    df = fill_metric(df, "new_chatter_ratio", "new_qc")

    df["chat_per_viewer"] = np.where(df["viewer_count_last"].gt(0), df["chat_count"] / df["viewer_count_last"], np.nan)

    df = df.sort_values(KEY + ["minute_ts"]).reset_index(drop=True)
    df["delta_viewer_1m"] = df.groupby(KEY)["viewer_count_last"].diff()
    df["delta_chat_1m"] = df.groupby(KEY)["chat_count"].diff()
    df["session_key"] = df["run_id"].astype(str) + "_" + df["broad_no"].astype(str)
    return df


def split_minute(df):
    all_zero = df.groupby(KEY)["chat_count"].transform(lambda x: x.eq(0).all())

    minute_all = df.copy()
    minute_model = df.loc[~all_zero].copy()
    qc_zero = df.loc[all_zero].copy()

    qcols = [
        "source_file", "session_key", "run_id", "broad_no", "minute_ts",
        "v_qc", "v_edge", "v_miss_r", "avg_qc", "rep_qc", "new_qc"
    ]
    qcols = [c for c in qcols if c in df.columns]
    row_qc = df.loc[
        df["v_qc"].eq(1) | df["v_edge"].eq(1) | df["avg_qc"].eq(1) | df["rep_qc"].eq(1) | df["new_qc"].eq(1),
        qcols
    ].copy()

    return minute_all, minute_model, qc_zero, row_qc


QC_ZERO_SESSION_REVIEW_COLS = [
    "run_id",
    "broad_no",
    "session_key",
    "user_id",
    "category_id",
    "n_minutes",
    "start_ts",
    "end_ts",
    "median_viewer",
    "max_viewer",
    "min_viewer",
    "mean_viewer",
    "total_chat_count",
    "total_unique_chatters",
    "zero_chat_rate",
    "all_zero_chat",
    "v_qc",
    "v_edge",
    "v_miss_r",
    "qc_reason",
    "note",
]


def write_qc_zero_session_review(qc_zero_raw, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = qc_zero_raw.copy()
    if df.empty:
        review = pd.DataFrame(columns=QC_ZERO_SESSION_REVIEW_COLS)
        review.to_csv(out / "qc_zero_session_review.csv", index=False, encoding="utf-8-sig")
        return review

    if "session_key" not in df.columns and {"run_id", "broad_no"}.issubset(df.columns):
        df["session_key"] = df["run_id"].astype(str) + "_" + df["broad_no"].astype(str)
    df["minute_ts"] = pd.to_datetime(df.get("minute_ts"), errors="coerce")
    for col in ["viewer_count_last", "chat_count", "unique_chatters", "v_qc", "v_edge", "v_miss_r"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["user_id", "category_id"]:
        if col not in df.columns:
            df[col] = pd.NA

    def first_non_null(s):
        s = s.dropna()
        return s.iloc[0] if len(s) else pd.NA

    grouped = df.groupby("session_key", dropna=False, sort=True)
    review = grouped.agg(
        run_id=("run_id", "first"),
        broad_no=("broad_no", "first"),
        user_id=("user_id", first_non_null),
        category_id=("category_id", first_non_null),
        n_minutes=("minute_ts", "size"),
        start_ts=("minute_ts", "min"),
        end_ts=("minute_ts", "max"),
        median_viewer=("viewer_count_last", "median"),
        max_viewer=("viewer_count_last", "max"),
        min_viewer=("viewer_count_last", "min"),
        mean_viewer=("viewer_count_last", "mean"),
        total_chat_count=("chat_count", "sum"),
        total_unique_chatters=("unique_chatters", "sum"),
        zero_chat_rate=("chat_count", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).eq(0).mean()),
        v_qc=("v_qc", "max"),
        v_edge=("v_edge", "max"),
        v_miss_r=("v_miss_r", "max"),
    ).reset_index()
    review["all_zero_chat"] = pd.to_numeric(review["total_chat_count"], errors="coerce").fillna(0).eq(0)
    review["qc_reason"] = "websocket_qc_or_extreme_mismatch_candidate"
    review["note"] = "Preserved for manual QC appendix; excluded from behavior modeling."

    for col in QC_ZERO_SESSION_REVIEW_COLS:
        if col not in review.columns:
            review[col] = pd.NA
    review = review[QC_ZERO_SESSION_REVIEW_COLS]
    review.to_csv(out / "qc_zero_session_review.csv", index=False, encoding="utf-8-sig")
    return review
