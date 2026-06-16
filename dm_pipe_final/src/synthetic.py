from pathlib import Path
import numpy as np
import pandas as pd
from .session import make_session


def recalc(g):
    g = g.sort_values("minute_ts").copy()
    g["chat_count"] = g["chat_count"].clip(lower=0)
    g["unique_chatters"] = g["unique_chatters"].clip(lower=0)
    for c in ["avg_msg_len", "repeat_msg_ratio", "new_chatter_ratio"]:
        g[c] = np.where(g["chat_count"].eq(0), 0, g[c].fillna(0))
    g["chat_per_viewer"] = np.where(g["viewer_count_last"].gt(0), g["chat_count"] / g["viewer_count_last"], np.nan)
    g["delta_viewer_1m"] = g["viewer_count_last"].diff()
    g["delta_chat_1m"] = g["chat_count"].diff()
    return g


def choose_seg(n, rng, lo, hi):
    min_len = max(3, int(np.floor(lo * n)))
    max_len = max(min_len, int(np.ceil(hi * n)))
    length = int(rng.integers(min_len, max_len + 1))
    length = min(length, n)
    start = int(rng.integers(0, max(1, n - length + 1)))
    return start, start + length


def _mark_interval(g, ix, kind, idx):
    interval_id = f"{kind}_{idx}"
    start_ts = g.loc[ix, "minute_ts"].min()
    end_ts = g.loc[ix, "minute_ts"].max()
    g.loc[ix, "is_injected_minute"] = 1
    g.loc[ix, "injected_type"] = kind
    g.loc[ix, "injected_interval_id"] = interval_id
    g.loc[ix, "injected_start_ts"] = start_ts
    g.loc[ix, "injected_end_ts"] = end_ts
    return g


def inject(base, kind, idx, rng):
    g = base.copy().sort_values("minute_ts").reset_index(drop=True)
    for c in ["viewer_count_last", "chat_count", "unique_chatters"]:
        if c not in g.columns:
            g[c] = 0.0
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0).astype(float)
    n = len(g)

    g["run_id"] = -100000 - idx
    g["broad_no"] = f"syn_{kind}_{idx}"
    g["session_key"] = g["run_id"].astype(str) + "_" + g["broad_no"].astype(str)
    g["syn_type"] = kind
    g["y_syn"] = 1
    g["is_injected_minute"] = 0
    g["injected_type"] = None
    g["injected_interval_id"] = None
    g["injected_start_ts"] = pd.NaT
    g["injected_end_ts"] = pd.NaT

    if kind == "hi_view_low_chat":
        a, b = choose_seg(n, rng, 0.30, 0.60)
        ix = g.index[a:b]
        g.loc[ix, "viewer_count_last"] *= rng.uniform(1.05, 1.50)
        g.loc[ix, "chat_count"] = np.floor(g.loc[ix, "chat_count"] * rng.uniform(0.02, 0.25))
        g.loc[ix, "unique_chatters"] = np.floor(g.loc[ix, "unique_chatters"] * rng.uniform(0.02, 0.30))
        g = _mark_interval(g, ix, kind, idx)

    if kind == "silent_run":
        a, b = choose_seg(n, rng, 0.15, 0.35)
        ix = g.index[a:b]
        g.loc[ix, ["chat_count", "unique_chatters", "avg_msg_len", "repeat_msg_ratio", "new_chatter_ratio"]] = 0
        g = _mark_interval(g, ix, kind, idx)

    if kind == "view_spike_no_chat":
        a, b = choose_seg(n, rng, 0.05, 0.15)
        ix = g.index[a:b]
        g.loc[ix, "viewer_count_last"] *= rng.uniform(2.0, 4.0)
        g.loc[ix, "chat_count"] = np.floor(g.loc[ix, "chat_count"] * rng.uniform(0.3, 1.0))
        g.loc[ix, "unique_chatters"] = np.floor(g.loc[ix, "unique_chatters"] * rng.uniform(0.3, 1.0))
        g = _mark_interval(g, ix, kind, idx)

    return recalc(g)


def _counts_text(df, col):
    if df is None or df.empty or col not in df.columns:
        return "none"
    vc = df[col].value_counts(dropna=False).sort_index()
    return ", ".join(f"{k}:{v}" for k, v in vc.items())


def _save_intervals(syn_min, out):
    cols = [
        "injected_interval_id", "injected_type", "session_key", "run_id", "broad_no",
        "injected_start_ts", "injected_end_ts", "is_injected_minute",
    ]
    if syn_min is None or syn_min.empty or "is_injected_minute" not in syn_min.columns:
        pd.DataFrame(columns=cols).to_csv(out / "synthetic_intervals.csv", index=False, encoding="utf-8-sig")
        return
    injected = syn_min.loc[syn_min["is_injected_minute"].eq(1)].copy()
    if injected.empty:
        pd.DataFrame(columns=cols).to_csv(out / "synthetic_intervals.csv", index=False, encoding="utf-8-sig")
        return
    intervals = (
        injected.groupby("injected_interval_id")
        .agg(
            injected_type=("injected_type", "first"),
            session_key=("session_key", "first"),
            run_id=("run_id", "first"),
            broad_no=("broad_no", "first"),
            injected_start_ts=("minute_ts", "min"),
            injected_end_ts=("minute_ts", "max"),
            injected_minute_count=("is_injected_minute", "sum"),
        )
        .reset_index()
    )
    intervals.to_csv(out / "synthetic_intervals.csv", index=False, encoding="utf-8-sig")


def write_injection_doc(out, cfg, base_n=0, syn_min=None, train=None, note="ok"):
    out = Path(out)
    seed = int(cfg["synthetic"]["seed"])
    n_per = int(cfg["synthetic"]["n_per_type"])
    min_n = int(cfg["prep"]["min_n"])
    syn_min_n = 0 if syn_min is None else len(syn_min)
    train_n = 0 if train is None else len(train)
    syn_sess_n = int(train["y_syn"].eq(1).sum()) if train is not None and "y_syn" in train.columns else 0
    real_sess_n = int(train["y_syn"].eq(0).sum()) if train is not None and "y_syn" in train.columns else 0
    injected_n = int(syn_min["is_injected_minute"].sum()) if syn_min is not None and "is_injected_minute" in syn_min.columns else 0

    lines = [
        "Synthetic Injection Summary",
        "===========================",
        "Purpose: artificial viewer-chat mismatch injection for sanity checks and legacy supervised auxiliary scores.",
        "This is not ground-truth label. y_syn must not be documented as ground truth.",
        f"base eligible sessions: {base_n}",
        f"seed: {seed}, n_per_type: {n_per}, min_n: {min_n}",
        f"status: {note}",
        f"syn_minute rows: {syn_min_n}",
        f"injected minute rows: {injected_n}",
        f"syn_minute scenario distribution: {_counts_text(syn_min, 'syn_type')}",
        f"syn_train rows: {train_n}",
        f"syn_train real sessions(y_syn=0): {real_sess_n}",
        f"syn_train synthetic sessions(y_syn=1): {syn_sess_n}",
        "Saved files: out/syn_minute.csv, out/syn_train.csv, out/synthetic_intervals.csv, out/synthetic_injection.txt",
    ]
    (out / "synthetic_injection.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_synthetic(minute_model, session_model, out, cfg):
    out = Path(out)
    rng = np.random.default_rng(int(cfg["synthetic"]["seed"]))

    if minute_model.empty or session_model.empty:
        _save_intervals(pd.DataFrame(), out)
        write_injection_doc(out, cfg, base_n=0, note="minute_model or session_model is empty; synthetic generation skipped")
        return pd.DataFrame()

    base_sessions = session_model
    keys = base_sessions["session_key"].dropna().unique()
    n_per = min(int(cfg["synthetic"]["n_per_type"]), len(keys))
    kinds = ["hi_view_low_chat", "silent_run", "view_spike_no_chat"]

    rows = []
    idx = 0
    for kind in kinds:
        for key in rng.choice(keys, size=n_per, replace=len(keys) < n_per):
            base = minute_model[minute_model["session_key"].eq(key)]
            if len(base) >= int(cfg["prep"]["min_n"]):
                rows.append(inject(base, kind, idx, rng))
                idx += 1

    if not rows:
        _save_intervals(pd.DataFrame(), out)
        write_injection_doc(out, cfg, base_n=len(base_sessions), note="no base sessions satisfy min_n; synthetic generation skipped")
        return pd.DataFrame()

    syn_min = pd.concat(rows, ignore_index=True)
    syn_min.to_csv(out / "syn_minute.csv", index=False, encoding="utf-8-sig")
    _save_intervals(syn_min, out)

    _, syn_sess, _ = make_session(syn_min, syn_min, cfg)
    syn_sess["y_syn"] = 1

    real_norm = base_sessions.copy()
    real_norm["y_syn"] = 0

    train = pd.concat([real_norm, syn_sess], ignore_index=True)
    train.to_csv(out / "syn_train.csv", index=False, encoding="utf-8-sig")
    write_injection_doc(out, cfg, base_n=len(base_sessions), syn_min=syn_min, train=train, note="ok")
    return train
