import numpy as np
import pandas as pd


KEY = ["run_id", "broad_no"]
DEFAULT_CLOCK_GAP_RESET_MIN = 1.1


def clock_gap_reset_min(cfg=None):
    cfg = cfg or {}
    return float(cfg.get("prep", {}).get("clock_gap_reset_min", DEFAULT_CLOCK_GAP_RESET_MIN))


def zero_run(s):
    z = s.fillna(False).astype(bool)
    block = z.ne(z.shift()).cumsum()
    return z.groupby(block).cumcount().add(1).where(z, 0).astype(int)


def zero_run_clock(g, gap_reset_min=DEFAULT_CLOCK_GAP_RESET_MIN):
    z = g["zero"].fillna(False).astype(bool)
    gap = (
        pd.to_datetime(g["minute_ts"], errors="coerce")
        .diff()
        .dt.total_seconds()
        .div(60)
        .gt(float(gap_reset_min))
        .fillna(False)
    )
    block = (z.ne(z.shift()) | gap).cumsum()
    return z.groupby(block).cumcount().add(1).where(z, 0).astype(int)


def add_plot_cols(df, cfg=None):
    p = df.copy()
    gap_reset_min = clock_gap_reset_min(cfg)
    p["minute_ts"] = pd.to_datetime(p["minute_ts"], errors="coerce")
    p = p.sort_values(KEY + ["minute_ts"]).reset_index(drop=True)

    for c in ["v_qc", "v_edge", "v_miss_r"]:
        if c not in p.columns:
            p[c] = 0

    p["vlog"] = np.log1p(p["viewer_count_last"].clip(lower=0))
    p["clog"] = np.log1p(p["chat_count"].clip(lower=0))
    p["ulog"] = np.log1p(p["unique_chatters"].clip(lower=0))
    p["gap"] = p["vlog"] - p["clog"]
    p["zero"] = p["chat_count"].eq(0)
    p["zrun"] = 0
    for _, idx in p.groupby(KEY, sort=False).groups.items():
        p.loc[idx, "zrun"] = zero_run_clock(p.loc[idx], gap_reset_min=gap_reset_min).to_numpy()
    return p


def make_session(minute_all, minute_model, cfg):
    p = add_plot_cols(minute_all, cfg)

    sess = (
        p.groupby(KEY)
        .agg(
            user_id=("user_id", "first"),
            category_id=("category_id", "first"),
            n=("minute_ts", "size"),
            start=("minute_ts", "min"),
            end=("minute_ts", "max"),
            viewer_med=("viewer_count_last", "median"),
            viewer_max=("viewer_count_last", "max"),
            chat_mean=("chat_count", "mean"),
            unique_mean=("unique_chatters", "mean"),
            zero_rate=("zero", "mean"),
            zrun_max=("zrun", "max"),
            gap_med=("gap", "median"),
            gap_max=("gap", "max"),
            v_qc=("v_qc", "max"),
            v_edge=("v_edge", "max"),
            v_miss_r=("v_miss_r", "max"),
            n_all=("minute_ts", "size"),
            chat_all=("chat_count", "sum"),
            all_zero=("zero", "all"),
        )
        .reset_index()
    )

    sess["session_key"] = sess["run_id"].astype(str) + "_" + sess["broad_no"].astype(str)
    sess["ok"] = (
        sess["n"].ge(int(cfg["prep"]["min_n"]))
        & sess["v_qc"].fillna(0).eq(0)
        & ~sess["all_zero"].fillna(False)
    )

    sess["log_viewer"] = np.log1p(sess["viewer_med"])
    sess["log_chat"] = np.log1p(sess["chat_mean"])
    sess["log_unique"] = np.log1p(sess["unique_mean"])
    sess["log_zrun"] = np.log1p(sess["zrun_max"])

    return (
        sess.copy(),
        sess.loc[sess["ok"]].copy().reset_index(drop=True),
        sess.loc[~sess["ok"]].copy().reset_index(drop=True),
    )
