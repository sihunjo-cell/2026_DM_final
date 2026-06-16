from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["run_id", "broad_no"]
ID_COLS = ["session_key", "run_id", "broad_no", "minute_ts"]
SCAN_COLS = [
    "session_key",
    "run_id",
    "broad_no",
    "top_interval_start_idx",
    "top_interval_end_idx",
    "top_interval_start_ts",
    "top_interval_end_ts",
    "top_interval_duration",
    "scan_block_id",
    "observed_row_duration",
    "clock_duration_min",
    "clock_gap_count",
    "max_clock_gap_min",
    "interval_start_ts",
    "interval_end_ts",
    "observed_scan_z",
    "null_scan_z_mean",
    "null_scan_z_p95",
    "empirical_p",
    "scan_rank",
    "interval_mean_rank",
    "interval_p95_rank",
    "interval_chat_deficit_mean",
    "interval_unique_deficit_mean",
    "interval_model_chat_deficit_mean",
    "interval_model_unique_deficit_mean",
    "interval_zero_rate",
    "interval_max_zero_run",
    "interval_mismatch_state_rate",
    "dominant_reason",
    "note",
]


def _write_csv(df, path, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if columns is not None:
        for col in columns:
            if col not in out.columns:
                out[col] = np.nan
        out = out[columns]
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def _read(out, name):
    path = Path(out) / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def _num(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _ensure_keys(df):
    out = df.copy()
    if out.empty:
        return out
    if "session_key" not in out.columns:
        out["session_key"] = out["run_id"].astype(str) + "_" + out["broad_no"].astype(str)
    out["minute_ts"] = pd.to_datetime(out.get("minute_ts"), errors="coerce")
    if "minute_idx" not in out.columns:
        out = out.sort_values(KEY + ["minute_ts"]).reset_index(drop=True)
        out["minute_idx"] = out.groupby(KEY).cumcount() + 1
    return out.sort_values(KEY + ["minute_idx", "minute_ts"]).reset_index(drop=True)


def _merge_base_pred(score_df, out):
    base = _read(out, "base_pred.csv")
    if base.empty:
        return score_df
    keep = [c for c in ID_COLS + [
        "base_log_chat_q50",
        "base_log_unique_q50",
        "model_chat_deficit",
        "model_unique_deficit",
        "baseline_agree_chat",
        "baseline_agree_unique",
    ] if c in base.columns]
    if len(keep) <= len(ID_COLS):
        return score_df
    left = score_df.copy()
    right = base[keep].copy()
    for key in ["session_key", "run_id", "broad_no"]:
        if key in left.columns and key in right.columns:
            left[key] = left[key].astype("string")
            right[key] = right[key].astype("string")
    left["minute_ts"] = pd.to_datetime(left["minute_ts"], errors="coerce")
    right["minute_ts"] = pd.to_datetime(right["minute_ts"], errors="coerce")
    return left.merge(right, on=ID_COLS, how="left")


def _mismatch_cluster(out):
    prof = _read(out, "mc_profile.csv")
    if prof.empty or "cluster_id" not in prof.columns or "cluster_mismatch_rank" not in prof.columns:
        return None
    p = prof.copy()
    p["cluster_mismatch_rank"] = pd.to_numeric(p["cluster_mismatch_rank"], errors="coerce")
    p = p.dropna(subset=["cluster_mismatch_rank"])
    if p.empty:
        return None
    return pd.to_numeric(pd.Series([p.sort_values("cluster_mismatch_rank", ascending=False).iloc[0]["cluster_id"]]), errors="coerce").iloc[0]


def _scan_cache(n, cache):
    if n not in cache:
        starts, ends = np.triu_indices(n)
        denom = np.sqrt(ends - starts + 1.0)
        cache[n] = (starts, ends, denom)
    return cache[n]


def _best_scan(z, cache):
    z = np.asarray(z, dtype=float)
    n = len(z)
    if n == 0:
        return np.nan, 0, 0
    starts, ends, denom = _scan_cache(n, cache)
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(z, nan=0.0))])
    scores = (csum[ends + 1] - csum[starts]) / denom
    if len(scores) == 0 or not np.isfinite(scores).any():
        return np.nan, 0, max(n - 1, 0)
    best_pos = int(np.nanargmax(scores))
    return float(scores[best_pos]), int(starts[best_pos]), int(ends[best_pos])


def _eps(total_minutes, cfg):
    policy = str(cfg.get("m2_scan", {}).get("eps_policy", "1/(n_minutes_total+1)")) if cfg else "1/(n_minutes_total+1)"
    if policy == "1/(n_minutes_total+1)":
        return 1.0 / (max(int(total_minutes), 1) + 1.0)
    try:
        return float(policy)
    except Exception:
        return 1.0 / (max(int(total_minutes), 1) + 1.0)


def _z_transform(rank, eps):
    p = np.clip(1.0 - np.asarray(rank, dtype=float) + eps, eps, 1.0)
    try:
        from scipy.stats import norm

        z = norm.ppf(1.0 - p)
        return np.where(np.isfinite(z), z, 0.0), "sum_z_over_sqrt_len"
    except Exception:
        z = -np.log(p)
        return np.where(np.isfinite(z), z, 0.0), "sum_neglogp_over_sqrt_len fallback because scipy.stats.norm was unavailable"


def _interval_reason(row):
    vals = {
        "high_chat_deficit": row.get("interval_chat_deficit_mean_rank"),
        "high_unique_deficit": row.get("interval_unique_deficit_mean_rank"),
        "high_rolling_zero_rate": row.get("interval_zero_rate_rank"),
        "long_zero_run": row.get("interval_max_zero_run_rank"),
        "mismatch_cluster_state": row.get("interval_mismatch_state_rate_rank"),
    }
    valid = {k: v for k, v in vals.items() if pd.notna(v)}
    if not valid:
        return "insufficient_signal"
    return max(valid.items(), key=lambda item: item[1])[0]


def _add_reason_ranks(scan_df):
    out = scan_df.copy()
    rank_map = {
        "interval_chat_deficit_mean": "interval_chat_deficit_mean_rank",
        "interval_unique_deficit_mean": "interval_unique_deficit_mean_rank",
        "interval_zero_rate": "interval_zero_rate_rank",
        "interval_max_zero_run": "interval_max_zero_run_rank",
        "interval_mismatch_state_rate": "interval_mismatch_state_rate_rank",
    }
    for col, rank_col in rank_map.items():
        x = pd.to_numeric(out.get(col), errors="coerce")
        out[rank_col] = x.rank(method="average", pct=True) if x.notna().any() else np.nan
    out["dominant_reason"] = out.apply(_interval_reason, axis=1)
    return out.drop(columns=list(rank_map.values()), errors="ignore")


def _interval_features(g, start_pos, end_pos, rank_col="minute_mismatch_rank"):
    part = g.iloc[start_pos:end_pos + 1]
    return {
        "interval_mean_rank": _num(part, rank_col).mean(),
        "interval_p95_rank": _num(part, rank_col).quantile(0.95),
        "interval_chat_deficit_mean": _num(part, "chat_deficit").mean(),
        "interval_unique_deficit_mean": _num(part, "unique_deficit").mean(),
        "interval_model_chat_deficit_mean": _num(part, "model_chat_deficit").mean(),
        "interval_model_unique_deficit_mean": _num(part, "model_unique_deficit").mean(),
        "interval_zero_rate": _num(part, "_zero_chat").mean(),
        "interval_max_zero_run": _num(part, "zero_run_len").max(),
        "interval_mismatch_state_rate": _num(part, "_mismatch_state").mean(),
    }


def _clock_duration(start_ts, end_ts):
    if pd.isna(start_ts) or pd.isna(end_ts):
        return np.nan
    return float((end_ts - start_ts).total_seconds() / 60.0 + 1.0)


def _clock_gap_stats(part, break_gap_min):
    ts = pd.to_datetime(part.get("minute_ts"), errors="coerce")
    gaps = ts.diff().dt.total_seconds().div(60.0)
    valid = gaps.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return 0, 0.0
    return int(valid.gt(float(break_gap_min)).sum()), float(valid.max())


def _scan_block_candidate(session_key, block_id, g, eps, n_perm, max_scan_n, rng, cache, break_gap_min, note_prefix, rank_col):
    g = g.sort_values(["minute_ts", "minute_idx"]).reset_index(drop=True)
    n = len(g)
    rank = _num(g, rank_col).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    z, stat_note = _z_transform(rank, eps)
    note_parts = [
        note_prefix,
        f"scan_block_id={block_id}",
        f"score_rank_col={rank_col}",
        f"eps={eps:.6g}",
        stat_note,
        "top_interval_duration is observed_row_duration; clock_duration_min is also retained",
    ]
    if n > max_scan_n:
        keep_pos = np.sort(np.unique(np.r_[0, n - 1, np.argsort(z)[-max_scan_n:]]))
        work_z = z[keep_pos]
        obs, start_work, end_work = _best_scan(work_z, cache)
        start_pos = int(keep_pos[start_work])
        end_pos = int(keep_pos[end_work])
        note_parts.append(f"top candidate pruning applied because block n={n} > max_scan_n={max_scan_n}")
    else:
        keep_pos = None
        obs, start_pos, end_pos = _best_scan(z, cache)

    null_max = []
    if n_perm > 0 and n > 1:
        for _ in range(n_perm):
            perm_z = rng.permutation(z)
            if keep_pos is not None:
                perm_z = perm_z[keep_pos]
            null_max.append(_best_scan(perm_z, cache)[0])
    null = pd.Series(null_max, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if null.empty:
        null_mean = np.nan
        null_p95 = np.nan
        empirical_p = np.nan
    else:
        null_mean = float(null.mean())
        null_p95 = float(null.quantile(0.95))
        empirical_p = float((1.0 + null.ge(obs).sum()) / (1.0 + len(null)))

    start_pos = max(0, min(int(start_pos), n - 1))
    end_pos = max(start_pos, min(int(end_pos), n - 1))
    part = g.iloc[start_pos:end_pos + 1]
    start_ts = pd.to_datetime(g["minute_ts"].iloc[start_pos], errors="coerce")
    end_ts = pd.to_datetime(g["minute_ts"].iloc[end_pos], errors="coerce")
    observed_row_duration = int(len(part))
    clock_gap_count, max_clock_gap_min = _clock_gap_stats(part, break_gap_min)
    feats = _interval_features(g, start_pos, end_pos, rank_col=rank_col)
    return {
        "session_key": session_key,
        "run_id": g["run_id"].iloc[0] if "run_id" in g.columns else np.nan,
        "broad_no": g["broad_no"].iloc[0] if "broad_no" in g.columns else np.nan,
        "top_interval_start_idx": g["minute_idx"].iloc[start_pos],
        "top_interval_end_idx": g["minute_idx"].iloc[end_pos],
        "top_interval_start_ts": start_ts,
        "top_interval_end_ts": end_ts,
        "top_interval_duration": observed_row_duration,
        "scan_block_id": int(block_id),
        "observed_row_duration": observed_row_duration,
        "clock_duration_min": _clock_duration(start_ts, end_ts),
        "clock_gap_count": clock_gap_count,
        "max_clock_gap_min": max_clock_gap_min,
        "interval_start_ts": start_ts,
        "interval_end_ts": end_ts,
        "observed_scan_z": obs,
        "null_scan_z_mean": null_mean,
        "null_scan_z_p95": null_p95,
        "empirical_p": empirical_p,
        **feats,
        "note": "; ".join(note_parts),
    }


def build_m2_scan(m2_scores, out, cfg=None, score_rank_col="minute_mismatch_rank", output_path=None, merge_base_pred=True):
    out = Path(out)
    cfg = cfg or {}
    score_df = _ensure_keys(m2_scores if m2_scores is not None else _read(out, "m2_scores.csv"))
    if score_df.empty:
        target = Path(output_path) if output_path is not None else out / "m2_scan.csv"
        return _write_csv(pd.DataFrame(columns=SCAN_COLS), target, SCAN_COLS)

    if score_rank_col not in score_df.columns:
        raise ValueError(f"score_rank_col not found in m2_scores: {score_rank_col}")
    if merge_base_pred:
        score_df = _merge_base_pred(score_df, out)
    total_minutes = len(score_df)
    eps = _eps(total_minutes, cfg)
    mismatch_cluster = _mismatch_cluster(out)
    if "zero_chat" in score_df.columns:
        score_df["_zero_chat"] = _num(score_df, "zero_chat")
    else:
        score_df["_zero_chat"] = _num(score_df, "chat_count").fillna(0).eq(0).astype(float)
    if mismatch_cluster is None or "minute_cluster" not in score_df.columns:
        score_df["_mismatch_state"] = np.nan
    else:
        score_df["_mismatch_state"] = _num(score_df, "minute_cluster").eq(mismatch_cluster).astype(float)

    scan_cfg = cfg.get("m2_scan", {})
    n_perm = int(scan_cfg.get("n_perm", 200))
    max_scan_n = int(cfg.get("m2_scan", {}).get("max_scan_n", 500))
    break_gap_min = float(scan_cfg.get("break_on_clock_gap_min", 1))
    rng = np.random.default_rng(42)
    cache = {}
    rows = []
    for session_key, g in score_df.groupby("session_key", sort=False):
        g = g.sort_values(["minute_ts", "minute_idx"]).reset_index(drop=True)
        gap_min = pd.to_datetime(g["minute_ts"], errors="coerce").diff().dt.total_seconds().div(60.0)
        g["scan_block_id"] = gap_min.gt(break_gap_min).fillna(False).cumsum().astype(int)
        candidates = []
        note_prefix = "adaptive contiguous interval scan by clock-contiguous block; empirical_p is within-block shuffled-null evidence, not probability"
        for block_id, block in g.groupby("scan_block_id", sort=True):
            if block.empty:
                continue
            candidates.append(_scan_block_candidate(
                session_key,
                block_id,
                block,
                eps,
                n_perm,
                max_scan_n,
                rng,
                cache,
                break_gap_min,
                note_prefix,
                score_rank_col,
            ))
        if candidates:
            candidates = sorted(
                candidates,
                key=lambda row: (
                    pd.isna(row["empirical_p"]),
                    row["empirical_p"] if pd.notna(row["empirical_p"]) else 1.0,
                    -(row["observed_scan_z"] if pd.notna(row["observed_scan_z"]) else -np.inf),
                    row["scan_block_id"],
                ),
            )
            rows.append(candidates[0])

    scan = pd.DataFrame(rows)
    if not scan.empty:
        scan = scan.sort_values(["empirical_p", "observed_scan_z", "session_key"], ascending=[True, False, True]).reset_index(drop=True)
        scan["scan_rank"] = np.arange(1, len(scan) + 1, dtype=float)
        scan = _add_reason_ranks(scan)
    target = Path(output_path) if output_path is not None else out / "m2_scan.csv"
    return _write_csv(scan, target, SCAN_COLS)
