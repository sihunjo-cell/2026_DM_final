from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler


KEY = ["run_id", "broad_no"]
ID_COLS = ["session_key", "run_id", "broad_no", "minute_ts"]
GRID_NOTE = "grid only, no ground-truth threshold selected"
DPI = 200

_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = "Malgun Gothic" if "Malgun Gothic" in _available_fonts else "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

M2_SCORE_COLS = [
    "session_key",
    "run_id",
    "broad_no",
    "minute_ts",
    "minute_idx",
    "viewer_count_last",
    "chat_count",
    "unique_chatters",
    "log_viewer",
    "log_chat",
    "log_unique",
    "viewer_bin",
    "chat_deficit",
    "unique_deficit",
    "rolling_chat_deficit_5m",
    "zero_run_len",
    "rolling_zero_rate_5m",
    "minute_cluster",
    "cluster_mismatch_rank",
    "minute_mismatch_score",
    "minute_mismatch_rank",
    "dominant_reason",
]

EP_COLS = [
    "score_source",
    "threshold_q",
    "score_cutoff",
    "min_duration",
    "session_key",
    "run_id",
    "broad_no",
    "episode_id",
    "start_ts",
    "end_ts",
    "duration_min",
    "mean_score",
    "max_score",
    "mean_viewer",
    "mean_chat",
    "mean_unique",
    "max_chat_deficit",
    "max_unique_deficit",
    "max_zero_run",
    "dominant_reason",
]

SENS_COLS = [
    "score_source",
    "threshold_q",
    "score_cutoff",
    "min_duration",
    "episode_count",
    "candidate_session_count",
    "candidate_session_rate",
    "median_duration",
    "p75_duration",
    "median_episode_score",
    "max_episode_score",
    "note",
]

CANDIDATE_COLS = [
    "score_source",
    "threshold_q",
    "score_cutoff",
    "min_duration",
    "session_key",
    "run_id",
    "broad_no",
    "episode_count",
    "episode_total_duration_min",
    "episode_duration_ratio",
    "max_episode_score",
    "p95_minute_score",
    "candidate_rank",
]

NULL_COLS = [
    "threshold_q",
    "min_duration",
    "observed_episode_count",
    "null_episode_count_mean",
    "null_episode_count_p95",
    "observed_candidate_session_rate",
    "null_candidate_session_rate_mean",
    "null_candidate_session_rate_p95",
    "observed_max_duration",
    "null_max_duration_mean",
    "null_max_duration_p95",
    "enrichment_episode_count",
    "enrichment_candidate_rate",
    "enrichment_max_duration",
    "empirical_p_max_duration",
    "note",
]

STATE_COLS = [
    "session_key",
    "run_id",
    "broad_no",
    "n_minutes",
    "mismatch_minute_count",
    "mismatch_minute_rate",
    "mismatch_run_count",
    "mismatch_max_run",
    "mismatch_total_run_min",
    "mismatch_mean_run",
    "active_max_run",
    "transition_count",
]

TRANS_COLS = ["from_state", "to_state", "count", "transition_share"]

TRANSITION_AUDIT_COLS = [
    "session_key",
    "old_review_order",
    "new_review_order",
    "old_state_dwell_rank",
    "new_state_dwell_rank",
    "old_transition_count",
    "new_transition_count",
    "order_changed",
    "note",
]

STAB_COLS = [
    "seed",
    "sample_size",
    "ari_vs_base",
    "selected_k",
    "mismatch_cluster_id",
    "mismatch_cluster_share",
    "chat_deficit_med_mismatch",
    "zero_chat_rate_mismatch",
    "note",
]

SYNTH_COLS = [
    "status",
    "reason",
    "source",
    "injected_interval_count",
    "stale_injected_interval_count",
    "scored_synthetic_session_count",
    "detected_synthetic_session_count",
    "recovered_interval_count",
    "recovered_rate",
    "mean_iou",
    "median_review_order",
    "top10_recall",
    "top50_recall",
    "top100_recall",
    "note",
]

SYNTH_MATCH_COLS = [
    "injected_interval_id",
    "injected_type",
    "session_key",
    "injected_start_ts",
    "injected_end_ts",
    "detected_start_ts",
    "detected_end_ts",
    "overlap_min",
    "union_min",
    "iou",
    "recovered",
    "review_order",
    "note",
]


def _write_csv(df, path, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        out = df.copy()
        for col in columns:
            if col not in out.columns:
                out[col] = np.nan
        out = out[columns]
    else:
        out = df
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _save_fig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _blank_plot(path, msg):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
    ax.axis("off")
    _save_fig(fig, path)


def _safe_div(num, den):
    num = float(num) if pd.notna(num) else np.nan
    den = float(den) if pd.notna(den) else np.nan
    if not np.isfinite(den) or den <= 0:
        return np.nan
    return num / den


def _pct_rank(s):
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    return x.rank(method="average", pct=True)


def _grid(cfg):
    grid = cfg.get("episode_grid", {})
    quantiles = [float(q) for q in grid.get("score_quantiles", [0.90, 0.95, 0.97, 0.99])]
    durations = [int(x) for x in grid.get("min_duration_grid", [1, 2, 3, 5, 10])]
    merge_gap = int(grid.get("merge_gap_min", 1))
    return quantiles, durations, merge_gap


def _ensure_session_key(df):
    out = df.copy()
    if "session_key" not in out.columns:
        out["session_key"] = out["run_id"].astype(str) + "_" + out["broad_no"].astype(str)
    if "minute_idx" not in out.columns:
        out = out.sort_values(KEY + ["minute_ts"]).reset_index(drop=True)
        out["minute_idx"] = out.groupby(KEY).cumcount() + 1
    return out


def _dominant_reason(row, rank_cols):
    vals = row[rank_cols].dropna()
    if vals.empty:
        return "insufficient_signal"
    return str(vals.idxmax()).replace("_rank_signal", "")


def build_m2_scores(minute_df, out, cfg):
    """Build method 2 rule-rank minute mismatch scores.

    The output is a review index. It is not a probability and is not a label.
    """
    out = Path(out)
    s = _ensure_session_key(minute_df)
    signals = cfg.get("minute_score", {}).get("signals", [
        "chat_deficit",
        "unique_deficit",
        "rolling_chat_deficit_5m",
        "zero_run_len",
        "rolling_zero_rate_5m",
        "cluster_mismatch_rank",
    ])

    rank_cols = []
    for col in signals:
        if col not in s.columns:
            s[col] = np.nan
        rcol = f"{col}_rank_signal"
        s[rcol] = _pct_rank(s[col])
        rank_cols.append(rcol)

    s["minute_mismatch_score"] = s[rank_cols].mean(axis=1, skipna=True)
    s["minute_mismatch_rank"] = _pct_rank(s["minute_mismatch_score"])
    s["dominant_reason"] = s.apply(lambda row: _dominant_reason(row, rank_cols), axis=1)

    for col in M2_SCORE_COLS:
        if col not in s.columns:
            s[col] = np.nan
    return _write_csv(s[M2_SCORE_COLS], out / "m2_scores.csv", M2_SCORE_COLS)


def _score_frame(m2_scores, out, ml_scores=None):
    base = _ensure_session_key(m2_scores)
    score_df = base.copy()
    sources = {"rule_rank": "minute_mismatch_rank"}
    return score_df, sources


def _episode_groups(flagged, merge_gap_min):
    groups = []
    if flagged.empty:
        return groups
    flagged = flagged.sort_values(KEY + ["minute_idx", "minute_ts"])
    for _, g in flagged.groupby(KEY, sort=False):
        current = []
        prev_idx = None
        for _, row in g.iterrows():
            idx = int(row["minute_idx"])
            if prev_idx is None or idx - prev_idx <= merge_gap_min + 1:
                current.append(row)
            else:
                groups.append(pd.DataFrame(current))
                current = [row]
            prev_idx = idx
        if current:
            groups.append(pd.DataFrame(current))
    return groups


def _episode_base_rows(score_df, score_source, score_col, q, cutoff, merge_gap_min):
    score_df = score_df.copy()
    score_df["_source_score"] = pd.to_numeric(score_df[score_col], errors="coerce")
    flagged = score_df.loc[score_df["_source_score"].ge(cutoff)].copy()
    rows = []
    for ep_no, ep in enumerate(_episode_groups(flagged, merge_gap_min), start=1):
        duration = int(ep["minute_idx"].max() - ep["minute_idx"].min() + 1)
        reason = "insufficient_signal"
        if "dominant_reason" in ep.columns and ep["dominant_reason"].notna().any():
            reason = str(ep["dominant_reason"].value_counts().idxmax())
        rows.append({
            "score_source": score_source,
            "threshold_q": q,
            "score_cutoff": cutoff,
            "session_key": ep["session_key"].iloc[0],
            "run_id": ep["run_id"].iloc[0],
            "broad_no": ep["broad_no"].iloc[0],
            "start_ts": ep["minute_ts"].min(),
            "end_ts": ep["minute_ts"].max(),
            "duration_min": duration,
            "mean_score": ep["_source_score"].mean(),
            "max_score": ep["_source_score"].max(),
            "mean_viewer": ep.get("viewer_count_last", pd.Series(np.nan, index=ep.index)).mean(),
            "mean_chat": ep.get("chat_count", pd.Series(np.nan, index=ep.index)).mean(),
            "mean_unique": ep.get("unique_chatters", pd.Series(np.nan, index=ep.index)).mean(),
            "max_chat_deficit": ep.get("chat_deficit", pd.Series(np.nan, index=ep.index)).max(),
            "max_unique_deficit": ep.get("unique_deficit", pd.Series(np.nan, index=ep.index)).max(),
            "max_zero_run": ep.get("zero_run_len", pd.Series(np.nan, index=ep.index)).max(),
            "dominant_reason": reason,
        })
    return rows


def build_m2_episodes(m2_scores, out, cfg, ml_scores=None):
    out = Path(out)
    score_df, sources = _score_frame(m2_scores, out, ml_scores)
    quantiles, durations, merge_gap = _grid(cfg)

    rows = []
    for source, col in sources.items():
        scores = pd.to_numeric(score_df[col], errors="coerce")
        if scores.notna().sum() == 0:
            continue
        for q in quantiles:
            cutoff = float(scores.quantile(q))
            base_rows = _episode_base_rows(score_df, source, col, q, cutoff, merge_gap)
            for min_duration in durations:
                ep_seq = 0
                for row in base_rows:
                    if int(row["duration_min"]) < min_duration:
                        continue
                    ep_seq += 1
                    out_row = dict(row)
                    out_row["min_duration"] = min_duration
                    out_row["episode_id"] = (
                        f"{source}_q{int(round(q * 100)):02d}_d{min_duration}_{ep_seq}"
                    )
                    rows.append(out_row)

    episodes = pd.DataFrame(rows)
    return _write_csv(episodes, out / "m2_ep.csv", EP_COLS)


def _threshold_lookup(score_df, sources, quantiles):
    rows = []
    for source, col in sources.items():
        scores = pd.to_numeric(score_df[col], errors="coerce")
        if scores.notna().sum() == 0:
            continue
        for q in quantiles:
            rows.append({"score_source": source, "threshold_q": q, "score_cutoff": float(scores.quantile(q))})
    return pd.DataFrame(rows)


def build_m2_sensitivity(episodes, m2_scores, out, cfg, ml_scores=None):
    out = Path(out)
    score_df, sources = _score_frame(m2_scores, out, ml_scores)
    quantiles, durations, _ = _grid(cfg)
    thresholds = _threshold_lookup(score_df, sources, quantiles)
    total_sessions = int(score_df["session_key"].nunique())
    rows = []
    for source, q, min_duration in itertools.product(sources.keys(), quantiles, durations):
        t = thresholds.loc[thresholds["score_source"].eq(source) & thresholds["threshold_q"].eq(q)]
        cutoff = float(t["score_cutoff"].iloc[0]) if not t.empty else np.nan
        ep = episodes.loc[
            episodes["score_source"].eq(source)
            & episodes["threshold_q"].eq(q)
            & episodes["min_duration"].eq(min_duration)
        ].copy()
        rows.append({
            "score_source": source,
            "threshold_q": q,
            "score_cutoff": cutoff,
            "min_duration": min_duration,
            "episode_count": int(len(ep)),
            "candidate_session_count": int(ep["session_key"].nunique()) if not ep.empty else 0,
            "candidate_session_rate": float(ep["session_key"].nunique() / max(total_sessions, 1)) if not ep.empty else 0.0,
            "median_duration": ep["duration_min"].median() if not ep.empty else np.nan,
            "p75_duration": ep["duration_min"].quantile(0.75) if not ep.empty else np.nan,
            "median_episode_score": ep["mean_score"].median() if not ep.empty else np.nan,
            "max_episode_score": ep["max_score"].max() if not ep.empty else np.nan,
            "note": GRID_NOTE,
        })
    sens = pd.DataFrame(rows)
    return _write_csv(sens, out / "m2_sens.csv", SENS_COLS)


def build_m2_candidates(episodes, m2_scores, out, cfg, ml_scores=None):
    out = Path(out)
    score_df, sources = _score_frame(m2_scores, out, ml_scores)
    quantiles, durations, _ = _grid(cfg)
    thresholds = _threshold_lookup(score_df, sources, quantiles)
    n_minutes = score_df.groupby("session_key").size().rename("n_minutes").reset_index()
    meta = score_df.groupby("session_key").agg(run_id=("run_id", "first"), broad_no=("broad_no", "first")).reset_index()

    frames = []
    for source, col in sources.items():
        score_df["_source_score"] = pd.to_numeric(score_df[col], errors="coerce")
        p95 = score_df.groupby("session_key")["_source_score"].quantile(0.95).rename("p95_minute_score").reset_index()
        for q, min_duration in itertools.product(quantiles, durations):
            t = thresholds.loc[thresholds["score_source"].eq(source) & thresholds["threshold_q"].eq(q)]
            cutoff = float(t["score_cutoff"].iloc[0]) if not t.empty else np.nan
            ep = episodes.loc[
                episodes["score_source"].eq(source)
                & episodes["threshold_q"].eq(q)
                & episodes["min_duration"].eq(min_duration)
            ]
            if ep.empty:
                continue
            agg = ep.groupby("session_key").agg(
                episode_count=("episode_id", "nunique"),
                episode_total_duration_min=("duration_min", "sum"),
                max_episode_score=("max_score", "max"),
            ).reset_index()
            cand = meta.merge(n_minutes, on="session_key", how="left").merge(p95, on="session_key", how="left").merge(agg, on="session_key", how="inner")
            cand["episode_duration_ratio"] = cand["episode_total_duration_min"] / cand["n_minutes"].replace(0, np.nan)
            cand["score_source"] = source
            cand["threshold_q"] = q
            cand["score_cutoff"] = cutoff
            cand["min_duration"] = min_duration
            cand = cand.sort_values(
                ["episode_duration_ratio", "max_episode_score", "p95_minute_score", "episode_count", "session_key"],
                ascending=[False, False, False, False, True],
            ).reset_index(drop=True)
            cand["candidate_rank"] = np.arange(1, len(cand) + 1)
            frames.append(cand[CANDIDATE_COLS])

    candidates = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANDIDATE_COLS)
    return _write_csv(candidates, out / "m2_candidates.csv", CANDIDATE_COLS)


def build_m2_eval(episodes, candidates, m2_scores, sens, out, cfg, ml_scores=None):
    out = Path(out)
    eval_rows = []
    if sens is not None and not sens.empty:
        for row in sens.itertuples(index=False):
            eval_rows.append({
                "threshold_q": row.threshold_q,
                "min_duration": row.min_duration,
                "candidate_session_count": row.candidate_session_count,
                "candidate_session_rate": row.candidate_session_rate,
                "episode_count": row.episode_count,
                "median_duration": row.median_duration,
                "p75_duration": row.p75_duration,
                "median_episode_score": row.median_episode_score,
                "max_episode_score": row.max_episode_score,
                "note": "rule_rank only; no source agreement; no actual label; AUC-ROC, accuracy, and F1 are not computed as true performance",
            })
    eval_df = pd.DataFrame(eval_rows)
    _write_csv(eval_df, out / "m2_eval.csv")
    return eval_df


def _run_bins(score_df):
    sess = (
        score_df.groupby("session_key")
        .agg(run_id=("run_id", "first"), broad_no=("broad_no", "first"), start_ts=("minute_ts", "min"))
        .reset_index()
        .sort_values(["run_id", "start_ts", "session_key"])
    )
    if sess.empty:
        sess["run_bin"] = pd.Series(dtype="object")
        return sess[["session_key", "run_bin"]]
    n_bins = min(4, len(sess))
    labels = [f"bin_{i + 1}" for i in range(n_bins)]
    if n_bins == 1:
        sess["run_bin"] = labels[0]
    else:
        sess["run_bin"] = pd.qcut(np.arange(len(sess)), q=n_bins, labels=labels)
    return sess[["session_key", "run_bin"]]


def build_m2_stability(episodes, candidates, m2_scores, out, cfg, ml_scores=None):
    out = Path(out)
    score_df, sources = _score_frame(m2_scores, out, ml_scores)
    bins = _run_bins(score_df)
    score_df = score_df.merge(bins, on="session_key", how="left")
    candidates = candidates.merge(bins, on="session_key", how="left") if candidates is not None and not candidates.empty else pd.DataFrame()
    episodes = episodes.merge(bins, on="session_key", how="left") if episodes is not None and not episodes.empty else pd.DataFrame()
    quantiles, durations, _ = _grid(cfg)
    rows = []
    run_bins = sorted(score_df["run_bin"].dropna().unique())
    for run_bin, source, q, min_duration in itertools.product(run_bins, sources.keys(), quantiles, durations):
        col = sources[source]
        score_part = score_df.loc[score_df["run_bin"].eq(run_bin)].copy()
        cand_part = candidates.loc[
            candidates.get("run_bin", pd.Series(dtype=object)).eq(run_bin)
            & candidates.get("score_source", pd.Series(dtype=object)).eq(source)
            & candidates.get("threshold_q", pd.Series(dtype=float)).eq(q)
            & candidates.get("min_duration", pd.Series(dtype=int)).eq(min_duration)
        ] if not candidates.empty else pd.DataFrame()
        ep_part = episodes.loc[
            episodes.get("run_bin", pd.Series(dtype=object)).eq(run_bin)
            & episodes.get("score_source", pd.Series(dtype=object)).eq(source)
            & episodes.get("threshold_q", pd.Series(dtype=float)).eq(q)
            & episodes.get("min_duration", pd.Series(dtype=int)).eq(min_duration)
        ] if not episodes.empty else pd.DataFrame()
        n_sessions = int(score_part["session_key"].nunique())
        rows.append({
            "run_bin": run_bin,
            "score_source": source,
            "threshold_q": q,
            "min_duration": min_duration,
            "n_sessions": n_sessions,
            "candidate_session_count": int(cand_part["session_key"].nunique()) if not cand_part.empty else 0,
            "candidate_session_rate": float(cand_part["session_key"].nunique() / max(n_sessions, 1)) if not cand_part.empty else 0.0,
            "median_score": pd.to_numeric(score_part[col], errors="coerce").median() if col in score_part else np.nan,
            "p95_score": pd.to_numeric(score_part[col], errors="coerce").quantile(0.95) if col in score_part else np.nan,
            "median_episode_duration": ep_part["duration_min"].median() if not ep_part.empty else np.nan,
        })
    stability = pd.DataFrame(rows)
    return _write_csv(stability, out / "m2_stability.csv")


def _durations_from_flags(minute_idx, flags, merge_gap_min):
    idx = np.asarray(minute_idx)
    flag = np.asarray(flags, dtype=bool)
    picked = idx[flag]
    picked = picked[np.isfinite(picked)]
    if len(picked) == 0:
        return []
    picked = np.sort(picked.astype(int))
    durations = []
    start = int(picked[0])
    prev = int(picked[0])
    for raw_idx in picked[1:]:
        cur = int(raw_idx)
        if cur - prev <= merge_gap_min + 1:
            prev = cur
        else:
            durations.append(int(prev - start + 1))
            start = cur
            prev = cur
    durations.append(int(prev - start + 1))
    return durations


def _stats_by_duration(durations_by_session, duration_grid, total_sessions):
    rows = {}
    total_sessions = max(int(total_sessions), 1)
    for min_duration in duration_grid:
        durs = []
        candidate_sessions = 0
        for session_durs in durations_by_session:
            keep = [int(d) for d in session_durs if int(d) >= int(min_duration)]
            if keep:
                candidate_sessions += 1
                durs.extend(keep)
        rows[int(min_duration)] = {
            "episode_count": int(len(durs)),
            "candidate_session_rate": float(candidate_sessions / total_sessions),
            "max_duration": int(max(durs)) if durs else 0,
        }
    return rows


def _null_session_parts(score_df, flags):
    temp = score_df.copy()
    temp["_high_flag"] = np.asarray(flags, dtype=bool)
    parts = []
    for session_key, g in temp.groupby("session_key", sort=False):
        g = g.sort_values("minute_idx")
        idx = pd.to_numeric(g["minute_idx"], errors="coerce").to_numpy()
        flag = g["_high_flag"].to_numpy(dtype=bool)
        parts.append((session_key, idx, flag))
    return parts


def _plot_null(null_df, path):
    if null_df.empty:
        _blank_plot(path, "m2_null.csv not available")
        return
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
    specs = [
        ("observed_episode_count", "null_episode_count_p95", "episodes: observed / null p95"),
        ("observed_candidate_session_rate", "null_candidate_session_rate_p95", "candidate rate: observed / null p95"),
        ("observed_max_duration", "null_max_duration_p95", "max duration: observed / null p95"),
    ]
    for a, (obs_col, null_col, title) in zip(ax, specs):
        obs = null_df.pivot(index="min_duration", columns="threshold_q", values=obs_col).sort_index()
        ref = null_df.pivot(index="min_duration", columns="threshold_q", values=null_col).sort_index()
        ratio = obs / ref.replace(0, np.nan)
        im = a.imshow(ratio.values, aspect="auto", cmap="viridis")
        a.set_xticks(range(len(ratio.columns)), [str(c) for c in ratio.columns])
        a.set_yticks(range(len(ratio.index)), [str(i) for i in ratio.index])
        a.set_xlabel("threshold_q")
        a.set_ylabel("min_duration")
        a.set_title(title)
        finite = ratio.to_numpy(dtype=float)
        max_val = np.nanmax(finite) if np.isfinite(finite).any() else 1.0
        for i in range(ratio.shape[0]):
            for j in range(ratio.shape[1]):
                obs_val = obs.iloc[i, j]
                ref_val = ref.iloc[i, j]
                if "rate" in obs_col:
                    txt = f"{obs_val:.2f}/{ref_val:.2f}"
                else:
                    txt = f"{obs_val:.0f}/{ref_val:.0f}"
                color = "white" if pd.notna(ratio.iloc[i, j]) and ratio.iloc[i, j] > max_val / 2 else "black"
                a.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
        fig.colorbar(im, ax=a, shrink=0.75, label="observed / null p95")
    fig.suptitle("15 Null persistence check - not supervised performance", fontsize=14)
    _save_fig(fig, path)


def build_m2_null(m2_scores, out, cfg):
    out = Path(out)
    score_df, sources = _score_frame(m2_scores, out)
    quantiles, durations, merge_gap = _grid(cfg)
    score_col = sources["rule_rank"]
    score_df = score_df.sort_values(KEY + ["minute_idx", "minute_ts"]).reset_index(drop=True)
    scores = pd.to_numeric(score_df[score_col], errors="coerce")
    total_sessions = int(score_df["session_key"].nunique())
    n_perm = int(cfg.get("m2_eval", {}).get("null_perm", 100))
    rng = np.random.default_rng(42)
    rows = []

    if score_df.empty or scores.notna().sum() == 0:
        null_df = pd.DataFrame(columns=NULL_COLS)
        _write_csv(null_df, out / "m2_null.csv", NULL_COLS)
        _blank_plot(out / "plots" / "15_null.png", "m2_scores.csv not available")
        return null_df

    for q in quantiles:
        cutoff = float(scores.quantile(q))
        flags = scores.ge(cutoff).fillna(False).to_numpy(dtype=bool)
        parts = _null_session_parts(score_df, flags)
        observed_durations = [
            _durations_from_flags(idx, flag, merge_gap)
            for _, idx, flag in parts
        ]
        observed = _stats_by_duration(observed_durations, durations, total_sessions)
        null_stats = {
            d: {"episode_count": [], "candidate_session_rate": [], "max_duration": []}
            for d in durations
        }

        for _ in range(max(n_perm, 0)):
            shuffled_durations = []
            for _, idx, flag in parts:
                n_true = int(flag.sum())
                if n_true <= 0:
                    shuffled_durations.append([])
                    continue
                if n_true >= len(flag):
                    perm_flag = np.ones(len(flag), dtype=bool)
                else:
                    perm_flag = np.zeros(len(flag), dtype=bool)
                    perm_flag[rng.choice(len(flag), size=n_true, replace=False)] = True
                shuffled_durations.append(_durations_from_flags(idx, perm_flag, merge_gap))
            perm_stats = _stats_by_duration(shuffled_durations, durations, total_sessions)
            for min_duration in durations:
                for metric in ["episode_count", "candidate_session_rate", "max_duration"]:
                    null_stats[min_duration][metric].append(perm_stats[min_duration][metric])

        for min_duration in durations:
            obs = observed[min_duration]
            null_ep = np.asarray(null_stats[min_duration]["episode_count"], dtype=float)
            null_rate = np.asarray(null_stats[min_duration]["candidate_session_rate"], dtype=float)
            null_max = np.asarray(null_stats[min_duration]["max_duration"], dtype=float)
            ep_mean = float(np.mean(null_ep)) if len(null_ep) else np.nan
            rate_mean = float(np.mean(null_rate)) if len(null_rate) else np.nan
            max_mean = float(np.mean(null_max)) if len(null_max) else np.nan
            rows.append({
                "threshold_q": q,
                "min_duration": int(min_duration),
                "observed_episode_count": obs["episode_count"],
                "null_episode_count_mean": ep_mean,
                "null_episode_count_p95": float(np.quantile(null_ep, 0.95)) if len(null_ep) else np.nan,
                "observed_candidate_session_rate": obs["candidate_session_rate"],
                "null_candidate_session_rate_mean": rate_mean,
                "null_candidate_session_rate_p95": float(np.quantile(null_rate, 0.95)) if len(null_rate) else np.nan,
                "observed_max_duration": obs["max_duration"],
                "null_max_duration_mean": max_mean,
                "null_max_duration_p95": float(np.quantile(null_max, 0.95)) if len(null_max) else np.nan,
                "enrichment_episode_count": _safe_div(obs["episode_count"], ep_mean),
                "enrichment_candidate_rate": _safe_div(obs["candidate_session_rate"], rate_mean),
                "enrichment_max_duration": _safe_div(obs["max_duration"], max_mean),
                "empirical_p_max_duration": float((1 + np.sum(null_max >= obs["max_duration"])) / (1 + len(null_max))) if len(null_max) else np.nan,
                "note": "null persistence check, not supervised performance",
            })

    null_df = pd.DataFrame(rows)
    null_df = _write_csv(null_df, out / "m2_null.csv", NULL_COLS)
    _plot_null(null_df, out / "plots" / "15_null.png")
    return null_df


def _run_lengths(states):
    states = list(states)
    if not states:
        return []
    runs = []
    current = states[0]
    length = 1
    for state in states[1:]:
        if state == current:
            length += 1
        else:
            runs.append((current, length))
            current = state
            length = 1
    runs.append((current, length))
    return runs


def _state_change_count(values):
    s = pd.Series(values).reset_index(drop=True)
    if len(s) <= 1:
        return 0
    return int(s.iloc[1:].ne(s.shift().iloc[1:]).fillna(False).sum())


def _prepare_transition_audit_seed(out, new_state_df):
    out = Path(out)
    old_state = _read_csv(out / "m2_state.csv")
    old_review = _read_csv(out / "m2_review.csv")
    sessions = set(new_state_df.get("session_key", pd.Series(dtype=object)).dropna().astype(str))
    if not old_state.empty and "session_key" in old_state.columns:
        sessions.update(old_state["session_key"].dropna().astype(str))
    if not old_review.empty and "session_key" in old_review.columns:
        sessions.update(old_review["session_key"].dropna().astype(str))
    audit = pd.DataFrame({"session_key": sorted(sessions)})
    if old_review.empty:
        audit["old_review_order"] = np.nan
        audit["old_state_dwell_rank"] = np.nan
    else:
        cols = [c for c in ["session_key", "review_order", "state_dwell_rank"] if c in old_review.columns]
        audit = audit.merge(
            old_review[cols].rename(columns={"review_order": "old_review_order", "state_dwell_rank": "old_state_dwell_rank"}),
            on="session_key",
            how="left",
        )
    if old_state.empty:
        audit["old_transition_count"] = np.nan
    else:
        cols = [c for c in ["session_key", "transition_count"] if c in old_state.columns]
        audit = audit.merge(
            old_state[cols].rename(columns={"transition_count": "old_transition_count"}),
            on="session_key",
            how="left",
        )
    audit = audit.merge(
        new_state_df[["session_key", "transition_count"]].rename(columns={"transition_count": "new_transition_count"}),
        on="session_key",
        how="left",
    )
    audit["new_review_order"] = np.nan
    audit["new_state_dwell_rank"] = np.nan
    audit["order_changed"] = False
    audit["note"] = "pending review rebuild; transition_count is actual minute_cluster state changes"
    return _write_csv(audit, out / "m2_review_transition_fix_audit.csv", TRANSITION_AUDIT_COLS)


def _plot_state(state_df, trans_df, path):
    if state_df.empty or trans_df.empty:
        _blank_plot(path, "m2_state.csv not available")
        return
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    states = ["active", "mismatch"]
    mat = (
        trans_df.pivot(index="from_state", columns="to_state", values="transition_share")
        .reindex(index=states, columns=states)
        .fillna(0)
    )
    im = ax[0, 0].imshow(mat.values, cmap="Blues", vmin=0, vmax=1)
    ax[0, 0].set_xticks(range(len(states)), states)
    ax[0, 0].set_yticks(range(len(states)), states)
    ax[0, 0].set_xlabel("to_state")
    ax[0, 0].set_ylabel("from_state")
    ax[0, 0].set_title("상태 전이 비중")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax[0, 0].text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax[0, 0], shrink=0.75, label="transition share")

    ax[0, 1].hist(pd.to_numeric(state_df["mismatch_max_run"], errors="coerce").dropna(), bins="sturges", edgecolor="black")
    ax[0, 1].set_title("Mismatch max run distribution")
    ax[0, 1].set_xlabel("minutes")
    ax[0, 1].set_ylabel("sessions")

    ax[1, 0].hist(pd.to_numeric(state_df["mismatch_minute_rate"], errors="coerce").dropna(), bins="sturges", edgecolor="black", color="tab:green")
    ax[1, 0].set_title("Mismatch minute rate distribution")
    ax[1, 0].set_xlabel("rate")
    ax[1, 0].set_ylabel("sessions")

    ax[1, 1].scatter(
        pd.to_numeric(state_df["mismatch_run_count"], errors="coerce"),
        pd.to_numeric(state_df["mismatch_max_run"], errors="coerce"),
        s=18,
        alpha=0.45,
        edgecolor="none",
    )
    ax[1, 1].set_title("Mismatch run count vs max run")
    ax[1, 1].set_xlabel("mismatch_run_count")
    ax[1, 1].set_ylabel("mismatch_max_run")
    for a in ax.ravel():
        a.grid(alpha=0.25)
    fig.suptitle("16 Minute state persistence - cluster state, not label", fontsize=14)
    _save_fig(fig, path)


def build_m2_state(m2_scores, out, cfg):
    out = Path(out)
    score_df = _ensure_session_key(m2_scores)
    score_df = score_df.sort_values(KEY + ["minute_idx", "minute_ts"]).reset_index(drop=True)
    profile_path = out / "mc_profile.csv"
    if profile_path.exists():
        profile = pd.read_csv(profile_path, encoding="utf-8-sig")
    else:
        profile = pd.DataFrame()

    if score_df.empty or profile.empty or "cluster_mismatch_rank" not in profile.columns:
        state_df = _write_csv(pd.DataFrame(columns=STATE_COLS), out / "m2_state.csv", STATE_COLS)
        trans_df = _write_csv(pd.DataFrame(columns=TRANS_COLS), out / "m2_trans.csv", TRANS_COLS)
        _blank_plot(out / "plots" / "16_state.png", "minute cluster profile not available")
        return state_df, trans_df

    prof = profile.copy()
    prof["cluster_mismatch_rank"] = pd.to_numeric(prof["cluster_mismatch_rank"], errors="coerce")
    mismatch_row = prof.sort_values("cluster_mismatch_rank", ascending=False).head(1)
    mismatch_cluster = pd.to_numeric(mismatch_row["cluster_id"], errors="coerce").iloc[0]
    clusters = pd.to_numeric(score_df["minute_cluster"], errors="coerce")
    score_df["_state"] = np.where(clusters.eq(mismatch_cluster), "mismatch", "active")

    rows = []
    trans_counts = {(a, b): 0 for a in ["active", "mismatch"] for b in ["active", "mismatch"]}
    for session_key, g in score_df.groupby("session_key", sort=False):
        states = g["_state"].tolist()
        runs = _run_lengths(states)
        mismatch_runs = [length for state, length in runs if state == "mismatch"]
        active_runs = [length for state, length in runs if state == "active"]
        clusters_in_session = pd.to_numeric(g["minute_cluster"], errors="coerce").tolist()
        for left, right in zip(states[:-1], states[1:]):
            trans_counts[(left, right)] += 1
        n_minutes = int(len(g))
        mismatch_count = int(np.sum(np.asarray(states) == "mismatch"))
        rows.append({
            "session_key": session_key,
            "run_id": g["run_id"].iloc[0] if "run_id" in g.columns else np.nan,
            "broad_no": g["broad_no"].iloc[0] if "broad_no" in g.columns else np.nan,
            "n_minutes": n_minutes,
            "mismatch_minute_count": mismatch_count,
            "mismatch_minute_rate": float(mismatch_count / max(n_minutes, 1)),
            "mismatch_run_count": int(len(mismatch_runs)),
            "mismatch_max_run": int(max(mismatch_runs)) if mismatch_runs else 0,
            "mismatch_total_run_min": int(sum(mismatch_runs)),
            "mismatch_mean_run": float(np.mean(mismatch_runs)) if mismatch_runs else 0.0,
            "active_max_run": int(max(active_runs)) if active_runs else 0,
            "transition_count": _state_change_count(clusters_in_session),
        })

    trans_rows = []
    for from_state in ["active", "mismatch"]:
        denom = sum(trans_counts[(from_state, to_state)] for to_state in ["active", "mismatch"])
        for to_state in ["active", "mismatch"]:
            count = int(trans_counts[(from_state, to_state)])
            trans_rows.append({
                "from_state": from_state,
                "to_state": to_state,
                "count": count,
                "transition_share": float(count / denom) if denom else 0.0,
            })

    new_state_df = pd.DataFrame(rows)
    _prepare_transition_audit_seed(out, new_state_df)
    state_df = _write_csv(new_state_df, out / "m2_state.csv", STATE_COLS)
    trans_df = _write_csv(pd.DataFrame(trans_rows), out / "m2_trans.csv", TRANS_COLS)
    _plot_state(state_df, trans_df, out / "plots" / "16_state.png")
    return state_df, trans_df


def _selected_k(out):
    path = Path(out) / "mc_select.csv"
    if not path.exists():
        return 2
    try:
        select = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return 2
    if select.empty or "param" not in select.columns:
        return 2
    if "selected" in select.columns:
        selected = select[select["selected"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if not selected.empty:
            select = selected
    param = str(select.iloc[0]["param"])
    if "=" in param:
        param = param.split("=", 1)[1]
    try:
        return max(2, int(float(param)))
    except Exception:
        return 2


def _stability_features(df, cfg):
    cfg_features = cfg.get("minute_cluster", {}).get("features", [])
    default = [
        "log_viewer",
        "chat_deficit",
        "unique_deficit",
        "rolling_chat_deficit_5m",
        "log_zero_run_len",
        "rolling_zero_rate_5m",
    ]
    features = cfg_features or default
    return [col for col in features if col in df.columns]


def _make_stability_x(df, features):
    x = df[features].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True)).fillna(0)


def _sample_mismatch_profile(sample_df, labels):
    tmp = sample_df.copy()
    tmp["_stab_cluster"] = labels
    if "zero_chat" not in tmp.columns:
        chat = tmp["chat_count"] if "chat_count" in tmp.columns else pd.Series(0, index=tmp.index)
        tmp["zero_chat"] = pd.to_numeric(chat, errors="coerce").fillna(0).eq(0)
    for col in [
        "chat_deficit",
        "unique_deficit",
        "rolling_chat_deficit_5m",
        "zero_run_len",
        "rolling_zero_rate_5m",
    ]:
        if col not in tmp.columns:
            tmp[col] = np.nan
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    prof = (
        tmp.groupby("_stab_cluster")
        .agg(
            n=("_stab_cluster", "size"),
            chat_deficit_med=("chat_deficit", "median"),
            unique_deficit_med=("unique_deficit", "median"),
            rolling_chat_deficit_5m_med=("rolling_chat_deficit_5m", "median"),
            zero_chat_rate=("zero_chat", "mean"),
            zero_run_len_p75=("zero_run_len", lambda x: x.quantile(0.75)),
            rolling_zero_rate_5m_p90=("rolling_zero_rate_5m", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )
    rank_cols = [
        "chat_deficit_med",
        "unique_deficit_med",
        "rolling_chat_deficit_5m_med",
        "zero_chat_rate",
        "zero_run_len_p75",
        "rolling_zero_rate_5m_p90",
    ]
    prof["_mismatch_rank"] = prof[rank_cols].rank(pct=True).mean(axis=1, skipna=True)
    if prof["_mismatch_rank"].notna().any():
        row = prof.sort_values("_mismatch_rank", ascending=False).iloc[0]
    else:
        row = prof.sort_values("n", ascending=False).iloc[0]
    return {
        "mismatch_cluster_id": int(row["_stab_cluster"]),
        "mismatch_cluster_share": float(row["n"] / max(len(tmp), 1)),
        "chat_deficit_med_mismatch": float(row["chat_deficit_med"]) if pd.notna(row["chat_deficit_med"]) else np.nan,
        "zero_chat_rate_mismatch": float(row["zero_chat_rate"]) if pd.notna(row["zero_chat_rate"]) else np.nan,
    }


def _plot_stability(stab_df, path):
    if stab_df.empty:
        _blank_plot(path, "mc_stab.csv not available")
        return
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = stab_df["seed"].astype(str)
    ax[0, 0].bar(x, pd.to_numeric(stab_df["ari_vs_base"], errors="coerce"), edgecolor="black")
    ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_title("ARI vs seed")
    ax[0, 0].set_xlabel("seed")
    ax[0, 0].set_ylabel("ARI vs base")

    ax[0, 1].plot(stab_df["seed"], pd.to_numeric(stab_df["mismatch_cluster_share"], errors="coerce"), marker="o")
    ax[0, 1].set_title("Mismatch cluster share")
    ax[0, 1].set_xlabel("seed")
    ax[0, 1].set_ylabel("share")

    ax[1, 0].plot(stab_df["seed"], pd.to_numeric(stab_df["chat_deficit_med_mismatch"], errors="coerce"), marker="o", color="tab:orange")
    ax[1, 0].set_title("Mismatch chat deficit median")
    ax[1, 0].set_xlabel("seed")
    ax[1, 0].set_ylabel("median")

    ax[1, 1].plot(stab_df["seed"], pd.to_numeric(stab_df["zero_chat_rate_mismatch"], errors="coerce"), marker="o", color="tab:green")
    ax[1, 1].set_title("Mismatch zero-chat rate")
    ax[1, 1].set_xlabel("seed")
    ax[1, 1].set_ylabel("rate")
    for a in ax.ravel():
        a.grid(alpha=0.25)
    fig.suptitle("17 KMeans stability - not label performance", fontsize=14)
    _save_fig(fig, path)


def build_mc_stability(minute_df, out, cfg):
    out = Path(out)
    df = minute_df.copy()
    features = _stability_features(df, cfg)
    selected_k = _selected_k(out)
    if df.empty or "minute_cluster" not in df.columns or len(features) < 2:
        stab_df = _write_csv(pd.DataFrame(columns=STAB_COLS), out / "mc_stab.csv", STAB_COLS)
        _blank_plot(out / "plots" / "17_mc_stab.png", "minute cluster features not available")
        return stab_df

    base_labels = pd.to_numeric(df["minute_cluster"], errors="coerce")
    keep = base_labels.notna()
    df = df.loc[keep].reset_index(drop=True)
    base_labels = base_labels.loc[keep].reset_index(drop=True).astype(int)
    if len(df) < selected_k:
        stab_df = _write_csv(pd.DataFrame(columns=STAB_COLS), out / "mc_stab.csv", STAB_COLS)
        _blank_plot(out / "plots" / "17_mc_stab.png", "too few rows for KMeans stability")
        return stab_df

    n_seeds = int(cfg.get("m2_eval", {}).get("stability_seeds", 10))
    sample_size_cfg = int(cfg.get("m2_eval", {}).get("stability_sample_size", 50000))
    sample_size = int(min(sample_size_cfg, len(df)))
    rows = []
    for seed in range(max(n_seeds, 0)):
        rng = np.random.default_rng(seed)
        if sample_size < len(df):
            sample_idx = rng.choice(np.arange(len(df)), size=sample_size, replace=False)
        else:
            sample_idx = np.arange(len(df))
        sample = df.iloc[sample_idx].copy()
        x = _make_stability_x(sample, features)
        xs = RobustScaler().fit_transform(x)
        labels = KMeans(n_clusters=selected_k, random_state=seed, n_init=10).fit_predict(xs)
        ari = adjusted_rand_score(base_labels.iloc[sample_idx].to_numpy(), labels)
        profile = _sample_mismatch_profile(sample, labels)
        rows.append({
            "seed": int(seed),
            "sample_size": int(sample_size),
            "ari_vs_base": float(ari),
            "selected_k": int(selected_k),
            **profile,
            "note": "KMeans stability diagnostic, not label performance",
        })

    stab_df = pd.DataFrame(rows)
    stab_df = _write_csv(stab_df, out / "mc_stab.csv", STAB_COLS)
    _plot_stability(stab_df, out / "plots" / "17_mc_stab.png")
    return stab_df


def _remove_synth_matches(out):
    matches = Path(out) / "m2_synth_matches.csv"
    if matches.exists():
        matches.unlink()


def _synth_summary_row(status, reason, source, injected=0, stale_injected=0, scored=0, detected=0, **values):
    row = {
        "status": status,
        "reason": reason,
        "source": source,
        "injected_interval_count": injected,
        "stale_injected_interval_count": stale_injected,
        "scored_synthetic_session_count": scored,
        "detected_synthetic_session_count": detected,
        "recovered_interval_count": np.nan,
        "recovered_rate": np.nan,
        "mean_iou": np.nan,
        "median_review_order": np.nan,
        "top10_recall": np.nan,
        "top50_recall": np.nan,
        "top100_recall": np.nan,
        "note": "Synthetic sanity was not executed; this row is not a detection performance result.",
    }
    row.update(values)
    return row


def _interval_iou(injected_start, injected_end, detected_start, detected_end):
    if any(pd.isna(x) for x in [injected_start, injected_end, detected_start, detected_end]):
        return np.nan, np.nan, np.nan
    overlap_start = max(injected_start, detected_start)
    overlap_end = min(injected_end, detected_end)
    union_start = min(injected_start, detected_start)
    union_end = max(injected_end, detected_end)
    overlap_min = max(0.0, (overlap_end - overlap_start).total_seconds() / 60.0 + 1.0)
    union_min = max(1.0, (union_end - union_start).total_seconds() / 60.0 + 1.0)
    return overlap_min, union_min, overlap_min / union_min


def build_m2_synth(out, cfg=None):
    out = Path(out)
    cfg = cfg or {}
    syn_path = out / "synthetic_intervals.csv"
    syn = _read_csv(syn_path)
    if not syn.empty and "session_key" in syn.columns:
        syn["session_key"] = syn["session_key"].astype(str)
    injected = int(syn.get("injected_interval_id", pd.Series(dtype=object)).nunique()) if not syn.empty else 0
    syn_keys = set(syn.get("session_key", pd.Series(dtype=object)).dropna().astype(str)) if not syn.empty else set()
    scores = _read_csv(out / "m2_scores.csv")
    scan = _read_csv(out / "m2_scan.csv")
    review = _read_csv(out / "m2_review.csv")
    scored_keys = set(scores.get("session_key", pd.Series(dtype=object)).dropna().astype(str)).intersection(syn_keys)
    detected_keys = set(scan.get("session_key", pd.Series(dtype=object)).dropna().astype(str)).intersection(syn_keys)
    scored_count = len(scored_keys)
    detected_count = len(detected_keys)

    if syn_path.exists() and injected > 0 and (scored_count == 0 or detected_count == 0):
        _remove_synth_matches(out)
        row = _synth_summary_row(
            "not_run_stale_input",
            "stale_synthetic_intervals_without_current_scored_or_scanned_sessions",
            "stale synthetic_intervals.csv ignored",
            injected=0,
            stale_injected=injected,
            scored=scored_count,
            detected=detected_count,
        )
        return _write_csv(pd.DataFrame([row]), out / "m2_synth.csv", SYNTH_COLS)

    synth_enabled = bool(cfg.get("synthetic_sanity", {}).get("enabled", False))
    episode_enabled = bool(cfg.get("episode_grid", {}).get("enabled", False))
    if not synth_enabled or not episode_enabled or injected == 0:
        _remove_synth_matches(out)
        row = _synth_summary_row(
            "not_run",
            "synthetic_sanity_disabled_or_not_scored",
            "no current-run synthetic scoring",
            injected=0,
            stale_injected=injected if syn_path.exists() else 0,
            scored=scored_count,
            detected=detected_count,
        )
        return _write_csv(pd.DataFrame([row]), out / "m2_synth.csv", SYNTH_COLS)

    required_syn = {"injected_interval_id", "session_key", "injected_start_ts", "injected_end_ts"}
    required_scan = {"session_key", "top_interval_start_ts", "top_interval_end_ts"}
    if not required_syn.issubset(syn.columns) or not required_scan.issubset(scan.columns):
        _remove_synth_matches(out)
        row = _synth_summary_row(
            "not_run",
            "synthetic_sanity_missing_required_columns",
            "synthetic input or scan table missing required columns",
            injected=0,
            stale_injected=injected,
            scored=scored_count,
            detected=detected_count,
        )
        return _write_csv(pd.DataFrame([row]), out / "m2_synth.csv", SYNTH_COLS)

    threshold = float(cfg.get("synthetic_sanity", {}).get("match_iou_threshold", 0.5))
    injected_intervals = syn.drop_duplicates("injected_interval_id").copy()
    injected_intervals["injected_start_ts"] = pd.to_datetime(injected_intervals["injected_start_ts"], errors="coerce")
    injected_intervals["injected_end_ts"] = pd.to_datetime(injected_intervals["injected_end_ts"], errors="coerce")
    scan_work = scan.copy()
    scan_work["session_key"] = scan_work["session_key"].astype(str)
    scan_work["detected_start_ts"] = pd.to_datetime(scan_work["top_interval_start_ts"], errors="coerce")
    scan_work["detected_end_ts"] = pd.to_datetime(scan_work["top_interval_end_ts"], errors="coerce")
    review_order = pd.Series(dtype=float)
    if not review.empty and {"session_key", "review_order"}.issubset(review.columns):
        review_keyed = review.drop_duplicates("session_key").copy()
        review_keyed["session_key"] = review_keyed["session_key"].astype(str)
        review_order = pd.to_numeric(review_keyed.set_index("session_key")["review_order"], errors="coerce")

    matches = []
    for _, row in injected_intervals.iterrows():
        session_key = str(row.get("session_key"))
        cand = scan_work.loc[scan_work["session_key"].eq(session_key)].copy()
        best = None
        for _, detected in cand.iterrows():
            overlap_min, union_min, iou = _interval_iou(
                row["injected_start_ts"],
                row["injected_end_ts"],
                detected["detected_start_ts"],
                detected["detected_end_ts"],
            )
            if best is None or (pd.notna(iou) and (pd.isna(best["iou"]) or iou > best["iou"])):
                best = {
                    "detected_start_ts": detected["detected_start_ts"],
                    "detected_end_ts": detected["detected_end_ts"],
                    "overlap_min": overlap_min,
                    "union_min": union_min,
                    "iou": iou,
                }
        if best is None:
            best = {
                "detected_start_ts": pd.NaT,
                "detected_end_ts": pd.NaT,
                "overlap_min": np.nan,
                "union_min": np.nan,
                "iou": np.nan,
            }
        recovered = bool(pd.notna(best["iou"]) and best["iou"] >= threshold)
        order = review_order.get(session_key, np.nan) if not review_order.empty else np.nan
        matches.append({
            "injected_interval_id": row.get("injected_interval_id"),
            "injected_type": row.get("injected_type", np.nan),
            "session_key": session_key,
            "injected_start_ts": row["injected_start_ts"],
            "injected_end_ts": row["injected_end_ts"],
            "detected_start_ts": best["detected_start_ts"],
            "detected_end_ts": best["detected_end_ts"],
            "overlap_min": best["overlap_min"],
            "union_min": best["union_min"],
            "iou": best["iou"],
            "recovered": recovered,
            "review_order": order,
            "note": "same session_key interval IoU match; synthetic sanity only",
        })

    match_df = _write_csv(pd.DataFrame(matches), out / "m2_synth_matches.csv", SYNTH_MATCH_COLS)
    recovered_mask = match_df["recovered"].astype(bool) if "recovered" in match_df.columns else pd.Series(dtype=bool)
    recovered = int(recovered_mask.sum())
    iou_values = pd.to_numeric(match_df.get("iou", pd.Series(dtype=float)), errors="coerce")
    review_values = pd.to_numeric(match_df.get("review_order", pd.Series(dtype=float)), errors="coerce")
    topk_values = {}
    for k in [10, 50, 100]:
        topk_values[f"top{k}_recall"] = float((recovered_mask & review_values.le(k)).sum() / max(injected, 1))
    row = _synth_summary_row(
        "ok",
        "synthetic_sanity_executed",
        "current-run synthetic scoring and adaptive scan",
        injected=injected,
        stale_injected=0,
        scored=scored_count,
        detected=detected_count,
        recovered_interval_count=recovered,
        recovered_rate=float(recovered / max(injected, 1)),
        mean_iou=float(iou_values.mean()) if iou_values.notna().any() else np.nan,
        median_review_order=float(review_values[recovered_mask].median()) if recovered_mask.any() and review_values[recovered_mask].notna().any() else np.nan,
        top10_recall=topk_values["top10_recall"],
        top50_recall=topk_values["top50_recall"],
        top100_recall=topk_values["top100_recall"],
        note="Synthetic sanity executed as an internal injected-session check; y_syn is not a real viewbot label.",
    )
    return _write_csv(pd.DataFrame([row]), out / "m2_synth.csv", SYNTH_COLS)


def write_m2_docs(out, cfg, synthetic_available=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2_pipeline.md").write_text(
        "\n".join([
            "# Method 2 Pipeline",
            "",
            "Method 2 is not threshold selection.",
            "q grid and duration grid are not final criteria.",
            "Fixed window_size 5/10/20 grids are not final criteria.",
            "The final review sessions are ordered by equal-weight family consensus plus family RRA.",
            "Method 2 ranks reviewable sessions using adaptive minute-level viewer-chat mismatch interval evidence.",
            "`m2_review.csv` is the main Method 2 output.",
            "`m2_scan.csv` is the adaptive interval evidence table used by final review ordering.",
            "Raw scan evidence ranks are preserved, but scan_interval_rank, empirical_p_rank, and scan_strength_rank are not each entered as independent final RRA evidence.",
            "The final family list is scan, persistence, expected-response, minute-state, interval-anomaly, and reason-support.",
            "The goal is to explain which sessions are reviewable and why.",
            "Method 2 does not train a viewbot classifier and does not create actual labels.",
            "",
            "## Input Unit",
            "`run_id + broad_no + minute_ts` is the minute input unit. Sessions are identified by `run_id + broad_no`.",
            "",
            "## Minute State Features",
            "`viewer_bin` is built from behavior-minute `log_viewer` deciles as a descriptive fallback.",
            "`expected_log_chat_bin` and `expected_log_unique_bin` are bin medians used for transparent fallback deficits.",
            "`chat_deficit = expected_log_chat_bin - log_chat`; `unique_deficit = expected_log_unique_bin - log_unique`.",
            "`zero_run_len` counts current consecutive zero-chat minutes inside the same session.",
            "Rolling features use only the current and past minutes after sorting inside the same session.",
            "",
            "## Expected Response",
            "`base_pred.csv` estimates normal q50 chat and unique response from log viewer scale, viewer bin, normalized minute position, category if available, and hour if available.",
            "Out-of-fold prediction by `run_id` is preferred; fallback rows are documented in `base_pred_info.txt` as descriptive baseline rows.",
            "Expected-response deficits are evidence only and are not supervised labels.",
            "The expected-response model is not a viewbot classifier.",
            "",
            "## Minute Clustering",
            "KMeans discovers the minute behavior state used for mismatch-state dwell evidence.",
            "GMM and HDBSCAN diagnostics are written only as structural diagnostics.",
            "Cluster ids are state summaries, not labels and not probabilities.",
            "KMeans K search is state diagnostic only and is not a final review criterion.",
            "",
            "## Minute Scores And Legacy Grid Diagnostics",
            "`rule_rank` uses equal-weight percentile ranks of transparent mismatch signals.",
            "`rule_rank` is `minute_mismatch_rank`, the percentile rank of `minute_mismatch_score`.",
            "Therefore `threshold_q=0.95` for `score_source=rule_rank` means `minute_mismatch_rank >= 0.95`.",
            "It does not mean raw `minute_mismatch_score >= 0.95`.",
            "`m2_sens.csv` is legacy sensitivity diagnostic.",
            "`m2_ep.csv` is legacy threshold-grid episode diagnostic.",
            "`m2_candidates.csv` is not used for final review ordering.",
            "No `threshold_q`, `min_duration`, or q=0.95/duration=10 setting is selected as final truth cutoff.",
            "",
            "## Adaptive Interval Scan",
            "`m2_scan.csv` searches each session for the strongest contiguous mismatch interval using a scan statistic.",
            "The scan statistic is based on minute-level evidence transformed within the full minute population, then calibrated with within-session shuffled null permutations.",
            "`empirical_p` is no-label interval evidence against the shuffled null, not probability.",
            f"`empirical_p` uses {cfg.get('m2_scan', {}).get('n_perm', 200)} permutations; minimum resolution is 1/(n_perm+1).",
            f"Clock gaps larger than {cfg.get('m2_scan', {}).get('break_on_clock_gap_min', 1)} minute start a new scan block, so top intervals do not cross real timestamp gaps.",
            "`top_interval_duration` is observed_row_duration from rows in the selected block; `clock_duration_min` is retained separately.",
            "`int_scores.csv` computes interval-level anomaly evidence from the adaptive top intervals, not from fixed windows.",
            "",
            "## Review Priority",
            "`m2_review.csv` combines scan interval, expected-response, state dwell, interval anomaly, and reason evidence ranking lists using Robust Rank Aggregation.",
            "Robust Rank Aggregation is used to avoid arbitrary weighting of evidence sources.",
            "`rra_p` and `rra_q` are rank-aggregation statistics.",
            "`rra_q` is not probability.",
            "`m2_review.csv` is a review priority table, not a label table.",
            "reason threshold is for explanation only, not for final ranking.",
            "",
            "## Supporting Diagnostics",
            "`m2_sens` reports threshold-grid candidate volume and episode behavior.",
            "`m2_null` compares observed persistence with shuffled within-session null persistence.",
            "`m2_state` reports minute cluster state transition and dwell-time behavior.",
            "`mc_stab` reports KMeans cluster stability across seeds and subsamples.",
            "No final threshold is selected.",
            "No association-rule scoring is used.",
            "No ground-truth label is generated.",
            "",
            "## Limits",
            "Actual label 1 is not available, so this pipeline does not claim true performance.",
            "AUC-ROC, accuracy, and F1 are not reported as true performance.",
            "These outputs are no-label evidence diagnostics.",
            "`minute_mismatch_score` is not probability. Episodes are high mismatch interval candidates, not activation labels.",
        ])
        + "\n",
        encoding="utf-8",
    )
    synth_line = "\n".join([
        "Synthetic intervals, if present, are optional internal sanity-check artifacts.",
        "They are not actual labels and are not final Method 2 performance evidence.",
        "Method 2 reports no-label evidence diagnostics only.",
    ])
    (out / "m2_eval_plan.md").write_text(
        "\n".join([
            "# Method 2 Evaluation Plan",
            "",
            "AUC-ROC is not used because there is no independent actual label 1.",
            "Accuracy and F1 are not computed for the same reason.",
            "",
            "Without labels, method 2 reports:",
            "- adaptive interval empirical p",
            "- expected-response residual evidence",
            "- state dwell evidence",
            "- interval anomaly evidence",
            "- reason evidence",
            "- family-level equal-weight consensus score",
            "- family RRA consensus rank",
            "",
            "Adaptive interval empirical p checks whether observed scan evidence is larger than shuffled within-session null evidence.",
            "Persistence family evidence combines top interval duration and state dwell evidence using ranks/percentiles, not a hard duration cutoff.",
            "Interval anomaly evidence checks whether the adaptive top interval is unusual in interval feature space.",
            "Reason evidence explains the top adaptive interval using deficit, zero-run, state, and anomaly support.",
            "m2_sens is only legacy sensitivity diagnostic.",
            "These are no-label diagnostics, not supervised performance metrics.",
            "",
            synth_line,
            "",
            "If actual label becomes available later, `m2_review.review_order` can be evaluated using PR-AUC, ROC-AUC, and Precision@K.",
            "Until then, no AUC-ROC, accuracy, or F1 value is reported as true performance.",
        ])
        + "\n",
        encoding="utf-8",
    )
    (out / "m2_model_notes.md").write_text(
        "\n".join([
            "# Method 2 Model Notes",
            "",
            "Expected-response model estimates normal chat/unique response.",
            "viewer_bin decile is fallback/descriptive, not final evidence.",
            "KMeans discovers minute state.",
            "KMeans K search is state diagnostic only.",
            "GMM and HDBSCAN are structural diagnostics.",
            "fixed window_size grid is not used for final review selection.",
            "q/duration grid is not used for final review selection.",
            "Adaptive interval scan creates final contiguous interval evidence units.",
            f"Interval anomaly features use median imputation and {cfg.get('m2_interval_anomaly', {}).get('scaler', 'RobustScaler')} before IsolationForest, LOF, and ECOD when available.",
            "Family consensus averages family strengths without manual weights.",
            "Family RRA avoids counting scan_interval_rank, empirical_p_rank, and scan_strength_rank as separate final evidence.",
            "raw_rra_p/raw_rra_q preserve the old raw-evidence audit values.",
            "rra_q is family_rra_q and is not probability.",
            "Reason patterns are descriptive support summaries only; no confidence and no lift are computed.",
            "reason threshold is for explanation only, not for final ranking.",
            f"Load policy: valid_windows={cfg.get('time', {}).get('valid_windows')}, tolerance_min={cfg.get('time', {}).get('tolerance_min')}, off_window_max_rate={cfg.get('time', {}).get('off_window_max_rate')}, trim_off_window_rows={cfg.get('time', {}).get('trim_off_window_rows')}.",
            "Actual label 1 is unavailable, so detector/model superiority cannot be judged.",
            "None of these outputs are ground-truth labels.",
        ])
        + "\n",
        encoding="utf-8",
    )
