import numpy as np
import pandas as pd

from src.session import clock_gap_reset_min


KEY = ["run_id", "broad_no"]


def zero_run_clock(g, gap_reset_min):
    z = g["zero_chat"].fillna(False).astype(bool)
    gap = (
        pd.to_datetime(g["minute_ts"], errors="coerce")
        .diff()
        .dt.total_seconds()
        .div(60.0)
        .gt(float(gap_reset_min))
        .fillna(False)
    )
    block = (z.ne(z.shift()) | gap).cumsum()
    return z.groupby(block).cumcount().add(1).where(z, 0).astype(int)


def _viewer_bins(log_viewer, n_bin):
    x = pd.to_numeric(log_viewer, errors="coerce")
    valid = x.dropna()
    if valid.nunique() < 2:
        return pd.Series(0, index=log_viewer.index, dtype="Int64")
    try:
        b = pd.qcut(valid, q=min(n_bin, valid.nunique()), labels=False, duplicates="drop")
    except ValueError:
        b = pd.qcut(valid.rank(method="first"), q=min(n_bin, len(valid)), labels=False, duplicates="drop")
    out = pd.Series(pd.NA, index=log_viewer.index, dtype="Int64")
    out.loc[b.index] = b.astype("Int64")
    return out


def _rolling_by_session(df, col, window, reducer="mean"):
    parts = []
    for _, g in df.groupby(KEY, sort=False):
        s = pd.to_numeric(g[col], errors="coerce")
        r = s.rolling(window=window, min_periods=1)
        vals = r.sum() if reducer == "sum" else r.mean()
        parts.append(pd.Series(vals.to_numpy(), index=g.index))
    return pd.concat(parts).sort_index()


def add_minute_state_features(df, cfg):
    """Add minute-level viewer-chat mismatch state features.

    viewer_bin avoids absolute viewer thresholds. Minutes are compared against
    other minutes with a similar log_viewer scale, so chat deficit is relative
    to observed scale rather than a hard-coded viewer count cutoff.
    """
    if df is None or df.empty:
        return df.copy()

    out = df.copy()
    gap_reset_min = clock_gap_reset_min(cfg)
    out["minute_ts"] = pd.to_datetime(out["minute_ts"], errors="coerce")
    out = out.sort_values(KEY + ["minute_ts"]).reset_index(drop=True)

    for col in ["viewer_count_last", "chat_count", "unique_chatters"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "session_key" not in out.columns:
        out["session_key"] = out["run_id"].astype(str) + "_" + out["broad_no"].astype(str)

    out["minute_idx"] = out.groupby(KEY).cumcount() + 1
    out["log_viewer"] = np.log1p(out["viewer_count_last"].clip(lower=0))
    out["log_chat"] = np.log1p(out["chat_count"].clip(lower=0))
    out["log_unique"] = np.log1p(out["unique_chatters"].clip(lower=0))
    out["viewer_chat_gap"] = out["log_viewer"] - out["log_chat"]
    out["viewer_unique_gap"] = out["log_viewer"] - out["log_unique"]
    out["zero_chat"] = out["chat_count"].eq(0)
    out["zero_run_len"] = 0
    for _, idx in out.groupby(KEY, sort=False).groups.items():
        out.loc[idx, "zero_run_len"] = zero_run_clock(out.loc[idx], gap_reset_min=gap_reset_min).to_numpy()
    out["log_zero_run_len"] = np.log1p(out["zero_run_len"])

    all_zero = out.groupby(KEY)["chat_count"].transform(lambda s: s.eq(0).all())
    behavior = out.loc[~all_zero].copy()
    n_bin = int(cfg.get("minute_state", {}).get("viewer_bin_n", 10))
    behavior_bins = _viewer_bins(behavior["log_viewer"], n_bin)
    out["viewer_bin"] = pd.NA
    out.loc[behavior.index, "viewer_bin"] = behavior_bins
    out["viewer_bin"] = out["viewer_bin"].astype("Int64")

    expected = (
        out.loc[out["viewer_bin"].notna()]
        .groupby("viewer_bin")
        .agg(
            expected_log_chat_bin=("log_chat", "median"),
            expected_log_unique_bin=("log_unique", "median"),
        )
    )
    global_chat = out.loc[~all_zero, "log_chat"].median()
    global_unique = out.loc[~all_zero, "log_unique"].median()
    out["expected_log_chat_bin"] = out["viewer_bin"].map(expected["expected_log_chat_bin"])
    out["expected_log_unique_bin"] = out["viewer_bin"].map(expected["expected_log_unique_bin"])
    out["expected_log_chat_bin"] = out["expected_log_chat_bin"].fillna(global_chat)
    out["expected_log_unique_bin"] = out["expected_log_unique_bin"].fillna(global_unique)

    out["chat_deficit"] = out["expected_log_chat_bin"] - out["log_chat"]
    out["unique_deficit"] = out["expected_log_unique_bin"] - out["log_unique"]

    windows = cfg.get("minute_state", {}).get("rolling_windows", [5, 10])
    windows = sorted({int(w) for w in windows if int(w) > 0})
    for window in windows:
        out[f"rolling_chat_deficit_{window}m"] = _rolling_by_session(out, "chat_deficit", window)
        out[f"rolling_unique_deficit_{window}m"] = _rolling_by_session(out, "unique_deficit", window)
        out[f"rolling_zero_rate_{window}m"] = _rolling_by_session(out, "zero_chat", window)
        out[f"rolling_chat_sum_{window}m"] = _rolling_by_session(out, "chat_count", window, reducer="sum")
        out[f"rolling_unique_sum_{window}m"] = _rolling_by_session(out, "unique_chatters", window, reducer="sum")

    return out
