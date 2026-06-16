from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import font_manager


DPI = 200
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = "Malgun Gothic" if "Malgun Gothic" in _available_fonts else "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

REASON_COLS = [
    "session_key",
    "run_id",
    "broad_no",
    "reason",
    "support_source",
    "support_value",
    "support_rank",
    "text_reason",
    "text_reason_included",
    "top_interval_start",
    "top_interval_end",
    "top_interval_duration",
    "note",
]

PATTERN_COLS = ["reason_pattern", "support_count", "support_rate", "example_sessions", "note"]

REVIEW_COLS = [
    "session_key",
    "run_id",
    "broad_no",
    "review_order",
    "rra_p",
    "rra_q",
    "raw_rra_p",
    "raw_rra_q",
    "scan_interval_rank",
    "empirical_p_rank",
    "interval_duration_rank",
    "scan_strength_rank",
    "expected_response_rank",
    "state_dwell_rank",
    "interval_anomaly_rank",
    "reason_rank",
    "reason_support_rank",
    "scan_family_rank",
    "persistence_family_rank",
    "expected_response_family_rank",
    "minute_state_family_rank",
    "interval_anomaly_family_rank",
    "reason_support_family_rank",
    "scan_family_strength",
    "persistence_family_strength",
    "expected_response_family_strength",
    "minute_state_family_strength",
    "interval_anomaly_family_strength",
    "reason_support_family_strength",
    "family_consensus_score",
    "family_rra_p",
    "family_rra_q",
    "evidence_family_count",
    "ranking_method",
    "evidence_repetition_score",
    "evidence_count",
    "n_session_minutes",
    "eligible_review",
    "review_qc_reason",
    "all_zero_session",
    "v_qc",
    "top_interval_start",
    "top_interval_end",
    "top_interval_duration",
    "observed_scan_z",
    "empirical_p",
    "dominant_reason",
    "reason_set",
    "review_note",
]

RANKING_METHOD = "equal_weight_family_consensus_plus_family_rra"

RAW_EVIDENCE_COLS = [
    "scan_interval_rank",
    "empirical_p_rank",
    "interval_duration_rank",
    "scan_strength_rank",
    "expected_response_rank",
    "state_dwell_rank",
    "interval_anomaly_rank",
    "reason_support_rank",
]

FAMILY_RANK_COLS = [
    "scan_family_rank",
    "persistence_family_rank",
    "expected_response_family_rank",
    "minute_state_family_rank",
    "interval_anomaly_family_rank",
    "reason_support_family_rank",
]

FAMILY_STRENGTH_COLS = [col.replace("_rank", "_strength") for col in FAMILY_RANK_COLS]

# reason-support just re-counts the other families' signals, so it is explanation,
# not evidence: still written to output, but excluded from the consensus/RRA aggregate.
CONSENSUS_FAMILY_RANK_COLS = [c for c in FAMILY_RANK_COLS if c != "reason_support_family_rank"]
CONSENSUS_FAMILY_STRENGTH_COLS = [c.replace("_rank", "_strength") for c in CONSENSUS_FAMILY_RANK_COLS]

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

RANK_AUDIT_COLS = [
    "session_key",
    "old_review_order",
    "new_review_order",
    "old_rra_q",
    "new_family_rra_q",
    "old_top_interval_duration",
    "new_top_interval_duration",
    "family_consensus_score",
    "persistence_family_rank",
    "scan_family_rank",
    "expected_response_family_rank",
    "interval_anomaly_family_rank",
    "reason_support_family_rank",
    "order_changed",
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


def _pct_rank(s):
    if s is None:
        return pd.Series(dtype=float)
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=getattr(s, "index", None), dtype=float)
    return x.rank(method="average", pct=True)


def _session_meta(scan):
    if scan.empty:
        return pd.DataFrame(columns=["session_key", "run_id", "broad_no"])
    return scan[["session_key", "run_id", "broad_no"]].drop_duplicates("session_key")


def _add_reason(rows, scan, meta, reason, support_col, support_source):
    if support_col not in scan.columns:
        return
    m = scan[["session_key", support_col, "top_interval_start_ts", "top_interval_end_ts", "top_interval_duration"]].copy()
    m["support_value"] = pd.to_numeric(m[support_col], errors="coerce")
    if m["support_value"].notna().sum() == 0:
        return
    m["support_rank"] = _pct_rank(m["support_value"])
    picked = m.loc[m["support_value"].notna()].copy()
    if picked.empty:
        return
    picked = picked.merge(meta, on="session_key", how="left")
    for row in picked.itertuples(index=False):
        rows.append({
            "session_key": row.session_key,
            "run_id": getattr(row, "run_id", np.nan),
            "broad_no": getattr(row, "broad_no", np.nan),
            "reason": reason,
            "support_source": support_source,
            "support_value": float(row.support_value) if pd.notna(row.support_value) else np.nan,
            "support_rank": float(row.support_rank) if pd.notna(row.support_rank) else np.nan,
            "text_reason": "",
            "text_reason_included": False,
            "top_interval_start": getattr(row, "top_interval_start_ts", np.nan),
            "top_interval_end": getattr(row, "top_interval_end_ts", np.nan),
            "top_interval_duration": getattr(row, "top_interval_duration", np.nan),
            "note": "순위 기반 설명 근거다. 특정 숫자 cutoff로 설명 포함 여부를 정하지 않는다.",
        })


def _add_interval_score_metrics(scan, scores):
    if scan.empty or scores.empty:
        return scan
    work = scan.copy()
    scores = scores.copy()
    scores["minute_idx"] = pd.to_numeric(scores.get("minute_idx"), errors="coerce")
    score_groups = {str(key): g.copy() for key, g in scores.groupby(scores["session_key"].astype(str), sort=False)}
    metric_map = {
        "rolling_chat_deficit_5m": "interval_rolling_chat_deficit_mean",
        "rolling_zero_rate_5m": "interval_rolling_zero_rate_mean",
    }
    rows = []
    for row in work.itertuples(index=False):
        part = score_groups.get(str(row.session_key), pd.DataFrame())
        start = pd.to_numeric(pd.Series([getattr(row, "top_interval_start_idx", np.nan)]), errors="coerce").iloc[0]
        end = pd.to_numeric(pd.Series([getattr(row, "top_interval_end_idx", np.nan)]), errors="coerce").iloc[0]
        if pd.notna(start) and pd.notna(end):
            part = part.loc[part["minute_idx"].between(start, end)]
        out_row = {"session_key": row.session_key}
        for src, dst in metric_map.items():
            out_row[dst] = pd.to_numeric(part.get(src), errors="coerce").mean() if src in part.columns else np.nan
        rows.append(out_row)
    metric_df = pd.DataFrame(rows)
    return work.merge(metric_df, on="session_key", how="left")


def build_m2_reason(out, cfg=None):
    out = Path(out)
    scan = _read(out, "m2_scan.csv")
    if scan.empty:
        empty = _write_csv(pd.DataFrame(columns=REASON_COLS), out / "m2_reason.csv", REASON_COLS)
        _write_csv(pd.DataFrame(columns=PATTERN_COLS), out / "m2_patterns.csv", PATTERN_COLS)
        return empty

    scan = _add_interval_score_metrics(scan, _read(out, "m2_scores.csv"))
    meta = _session_meta(scan)
    rows = []
    for reason, col in [
        ("high_chat_deficit", "interval_chat_deficit_mean"),
        ("high_unique_deficit", "interval_unique_deficit_mean"),
        ("high_rolling_chat_deficit", "interval_rolling_chat_deficit_mean"),
        ("long_zero_run", "interval_max_zero_run"),
        ("high_rolling_zero_rate", "interval_rolling_zero_rate_mean"),
        ("mismatch_cluster_state", "interval_mismatch_state_rate"),
    ]:
        _add_reason(rows, scan, meta, reason, col, col)

    int_scores = _read(out, "int_scores.csv")
    if not int_scores.empty:
        work = scan[["session_key", "top_interval_start_ts", "top_interval_end_ts", "top_interval_duration"]].merge(
            int_scores,
            on="session_key",
            how="left",
        )
        dir_cols = [c for c in ["int_ecod_dir_score", "int_if_dir_score", "int_lof_dir_score"] if c in work.columns]
        if dir_cols:
            work["support_value"] = work[dir_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).max(axis=1, skipna=True)
            work = work.rename(columns={"support_value": "high_interval_anomaly_support"})
            _add_reason(
                rows,
                work,
                meta,
                "high_interval_anomaly",
                "high_interval_anomaly_support",
                "int_scores directional anomaly",
            )

    reason_df = pd.DataFrame(rows)
    if not reason_df.empty:
        reason_df["support_rank"] = pd.to_numeric(reason_df["support_rank"], errors="coerce")
        reason_df["support_value"] = pd.to_numeric(reason_df["support_value"], errors="coerce")
        reason_df = reason_df.sort_values(
            ["session_key", "support_rank", "support_value", "reason"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
        top_idx = reason_df.groupby("session_key").cumcount().lt(3)
        reason_df["text_reason_included"] = top_idx
        reason_df["text_reason"] = np.where(top_idx, reason_df["reason"], "")
    reason_df = _write_csv(reason_df, out / "m2_reason.csv", REASON_COLS)

    if reason_df.empty:
        patterns = pd.DataFrame(columns=PATTERN_COLS)
    else:
        text_reason = reason_df.loc[reason_df.get("text_reason_included", False).astype(bool)].copy()
        pattern_source = text_reason if not text_reason.empty else reason_df
        by_session = pattern_source.groupby("session_key")["reason"].apply(lambda s: ";".join(sorted(set(s)))).reset_index(name="reason_pattern")
        total = max(int(meta["session_key"].nunique()), 1)
        patterns = (
            by_session.groupby("reason_pattern")
            .agg(
                support_count=("session_key", "nunique"),
                example_sessions=("session_key", lambda s: ";".join(s.astype(str).head(5))),
            )
            .reset_index()
            .sort_values(["support_count", "reason_pattern"], ascending=[False, True])
        )
        patterns["support_rate"] = patterns["support_count"] / total
        patterns["note"] = "설명 근거 동시 등장 요약이다. 특정 숫자 cutoff, confidence, lift, prediction rule이 아니다."
        patterns = patterns[PATTERN_COLS]
    _write_csv(patterns, out / "m2_patterns.csv", PATTERN_COLS)
    return reason_df


def _rank_from_sort(meta, metric, rank_col, sort_cols, ascending):
    out = meta[["session_key"]].copy()
    out[rank_col] = np.nan
    if metric.empty:
        return out
    m = metric.copy()
    for col in sort_cols:
        if col not in m.columns:
            m[col] = np.nan
        m[col] = pd.to_numeric(m[col], errors="coerce")
    m = m.dropna(subset=sort_cols, how="all")
    if m.empty:
        return out
    m = m.sort_values(sort_cols + ["session_key"], ascending=ascending + [True]).reset_index(drop=True)
    m[rank_col] = np.arange(1, len(m) + 1, dtype=float)
    return out.drop(columns=[rank_col]).merge(m[["session_key", rank_col]], on="session_key", how="left")


def _build_evidence_ranks(out, meta, scan):
    ranks = {}
    ranks["scan_interval_rank"] = _rank_from_sort(
        meta,
        scan[["session_key", "empirical_p", "observed_scan_z"]],
        "scan_interval_rank",
        ["empirical_p", "observed_scan_z"],
        [True, False],
    )
    ranks["empirical_p_rank"] = _rank_from_sort(
        meta,
        scan[["session_key", "empirical_p"]],
        "empirical_p_rank",
        ["empirical_p"],
        [True],
    )
    ranks["interval_duration_rank"] = _rank_from_sort(
        meta,
        scan[["session_key", "top_interval_duration"]],
        "interval_duration_rank",
        ["top_interval_duration"],
        [False],
    )
    ranks["scan_strength_rank"] = _rank_from_sort(
        meta,
        scan[["session_key", "observed_scan_z"]],
        "scan_strength_rank",
        ["observed_scan_z"],
        [False],
    )

    expected = scan[["session_key", "interval_model_chat_deficit_mean", "interval_model_unique_deficit_mean"]].copy()
    expected["expected_response_metric"] = expected[[
        "interval_model_chat_deficit_mean",
        "interval_model_unique_deficit_mean",
    ]].apply(lambda s: pd.to_numeric(s, errors="coerce")).max(axis=1, skipna=True)
    ranks["expected_response_rank"] = _rank_from_sort(
        meta,
        expected,
        "expected_response_rank",
        ["expected_response_metric"],
        [False],
    )

    state = _read(out, "m2_state.csv")
    ranks["state_dwell_rank"] = _rank_from_sort(
        meta,
        state,
        "state_dwell_rank",
        ["mismatch_minute_rate", "mismatch_max_run"],
        [False, False],
    )

    int_scores = _read(out, "int_scores.csv")
    if not int_scores.empty:
        dir_cols = [c for c in ["int_ecod_dir_score", "int_if_dir_score", "int_lof_dir_score"] if c in int_scores.columns]
        int_metric = int_scores[["session_key"]].copy()
        int_metric["interval_anomaly_metric"] = int_scores[dir_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).max(axis=1, skipna=True) if dir_cols else np.nan
        ranks["interval_anomaly_rank"] = _rank_from_sort(
            meta,
            int_metric,
            "interval_anomaly_rank",
            ["interval_anomaly_metric"],
            [False],
        )
    else:
        ranks["interval_anomaly_rank"] = _rank_from_sort(meta, pd.DataFrame(), "interval_anomaly_rank", ["metric"], [False])

    reason = _read(out, "m2_reason.csv")
    if not reason.empty:
        reason["support_rank"] = pd.to_numeric(reason.get("support_rank"), errors="coerce")
        reason["support_value"] = pd.to_numeric(reason.get("support_value"), errors="coerce")
        r = reason.groupby("session_key").agg(
            reason_support_rank_metric=("support_rank", "max"),
            support_value_max=("support_value", "max"),
        ).reset_index()
        ranks["reason_support_rank"] = _rank_from_sort(
            meta,
            r,
            "reason_support_rank",
            ["reason_support_rank_metric", "support_value_max"],
            [False, False],
        )
        ranks["reason_rank"] = ranks["reason_support_rank"].rename(columns={"reason_support_rank": "reason_rank"})
    else:
        ranks["reason_support_rank"] = _rank_from_sort(meta, pd.DataFrame(), "reason_support_rank", ["metric"], [False])
        ranks["reason_rank"] = _rank_from_sort(meta, pd.DataFrame(), "reason_rank", ["metric"], [False])
    return ranks


def _rank_family_from_metric(meta, metric, rank_col, metric_col):
    return _rank_from_sort(meta, metric, rank_col, [metric_col], [False])


def _strength_from_rank(rank, n_sessions):
    x = pd.to_numeric(rank, errors="coerce")
    return (1.0 - (x / (float(n_sessions) + 1.0))).clip(lower=0.0, upper=1.0)


def _build_family_ranks(out, meta, scan, raw_ranks):
    state = _read(out, "m2_state.csv")
    family = {}

    family["scan_family_rank"] = _rank_from_sort(
        meta,
        scan[["session_key", "empirical_p", "observed_scan_z"]],
        "scan_family_rank",
        ["empirical_p", "observed_scan_z"],
        [True, False],
    )

    persistence = meta[["session_key"]].merge(
        scan[["session_key", "top_interval_duration"]],
        on="session_key",
        how="left",
    )
    if not state.empty:
        keep = [c for c in ["session_key", "mismatch_max_run", "mismatch_total_run_min"] if c in state.columns]
        persistence = persistence.merge(state[keep], on="session_key", how="left")
    for col in ["top_interval_duration", "mismatch_max_run", "mismatch_total_run_min"]:
        if col not in persistence.columns:
            persistence[col] = np.nan
        persistence[col] = pd.to_numeric(persistence[col], errors="coerce")
        persistence[f"{col}_strength"] = _pct_rank(persistence[col])
    persistence["persistence_metric"] = persistence[
        ["top_interval_duration_strength", "mismatch_max_run_strength", "mismatch_total_run_min_strength"]
    ].mean(axis=1, skipna=True)
    family["persistence_family_rank"] = _rank_family_from_metric(
        meta,
        persistence[["session_key", "persistence_metric"]],
        "persistence_family_rank",
        "persistence_metric",
    )

    expected = scan[["session_key", "interval_model_chat_deficit_mean", "interval_model_unique_deficit_mean"]].copy()
    expected["expected_response_metric"] = expected[[
        "interval_model_chat_deficit_mean",
        "interval_model_unique_deficit_mean",
    ]].apply(lambda s: pd.to_numeric(s, errors="coerce")).max(axis=1, skipna=True)
    family["expected_response_family_rank"] = _rank_family_from_metric(
        meta,
        expected[["session_key", "expected_response_metric"]],
        "expected_response_family_rank",
        "expected_response_metric",
    )

    minute_state = meta[["session_key"]].merge(
        scan[["session_key", "interval_mismatch_state_rate"]],
        on="session_key",
        how="left",
    )
    if not state.empty:
        keep = [c for c in ["session_key", "mismatch_minute_rate"] if c in state.columns]
        minute_state = minute_state.merge(state[keep], on="session_key", how="left")
    for col in ["interval_mismatch_state_rate", "mismatch_minute_rate"]:
        if col not in minute_state.columns:
            minute_state[col] = np.nan
        minute_state[col] = pd.to_numeric(minute_state[col], errors="coerce")
        minute_state[f"{col}_strength"] = _pct_rank(minute_state[col])
    minute_state["minute_state_metric"] = minute_state[
        ["interval_mismatch_state_rate_strength", "mismatch_minute_rate_strength"]
    ].mean(axis=1, skipna=True)
    family["minute_state_family_rank"] = _rank_family_from_metric(
        meta,
        minute_state[["session_key", "minute_state_metric"]],
        "minute_state_family_rank",
        "minute_state_metric",
    )

    family["interval_anomaly_family_rank"] = raw_ranks["interval_anomaly_rank"].rename(
        columns={"interval_anomaly_rank": "interval_anomaly_family_rank"}
    )
    family["reason_support_family_rank"] = raw_ranks["reason_support_rank"].rename(
        columns={"reason_support_rank": "reason_support_family_rank"}
    )
    return family


def _write_transition_fix_audit(out, old_review, new_review):
    out = Path(out)
    audit_path = out / "m2_review_transition_fix_audit.csv"
    seed = _read(out, "m2_review_transition_fix_audit.csv")
    state = _read(out, "m2_state.csv")
    sessions = set()
    for frame in [seed, old_review, new_review, state]:
        if not frame.empty and "session_key" in frame.columns:
            sessions.update(frame["session_key"].dropna().astype(str))
    audit = pd.DataFrame({"session_key": sorted(sessions)})
    if not seed.empty:
        seed_cols = [c for c in ["session_key", "old_review_order", "old_state_dwell_rank", "old_transition_count", "new_transition_count"] if c in seed.columns]
        audit = audit.merge(seed[seed_cols], on="session_key", how="left")
    if "old_review_order" not in audit.columns:
        audit["old_review_order"] = np.nan
    if "old_state_dwell_rank" not in audit.columns:
        audit["old_state_dwell_rank"] = np.nan
    if "old_transition_count" not in audit.columns:
        audit["old_transition_count"] = np.nan
    if "new_transition_count" not in audit.columns:
        audit["new_transition_count"] = np.nan
    if not old_review.empty and "session_key" in old_review.columns:
        old_cols = [c for c in ["session_key", "review_order", "state_dwell_rank"] if c in old_review.columns]
        old = old_review[old_cols].rename(columns={"review_order": "old_review_order_fallback", "state_dwell_rank": "old_state_dwell_rank_fallback"})
        audit = audit.merge(old, on="session_key", how="left")
        audit["old_review_order"] = audit["old_review_order"].combine_first(audit.pop("old_review_order_fallback"))
        audit["old_state_dwell_rank"] = audit["old_state_dwell_rank"].combine_first(audit.pop("old_state_dwell_rank_fallback"))
    if not state.empty and {"session_key", "transition_count"}.issubset(state.columns):
        new_state = state[["session_key", "transition_count"]].rename(columns={"transition_count": "new_transition_count_fallback"})
        audit = audit.merge(new_state, on="session_key", how="left")
        audit["new_transition_count"] = audit["new_transition_count"].combine_first(audit.pop("new_transition_count_fallback"))
    if not new_review.empty and "session_key" in new_review.columns:
        new_cols = [c for c in ["session_key", "review_order", "state_dwell_rank"] if c in new_review.columns]
        new = new_review[new_cols].rename(columns={"review_order": "new_review_order", "state_dwell_rank": "new_state_dwell_rank"})
        audit = audit.merge(new, on="session_key", how="left")
    else:
        audit["new_review_order"] = np.nan
        audit["new_state_dwell_rank"] = np.nan
    old_order = pd.to_numeric(audit["old_review_order"], errors="coerce")
    new_order = pd.to_numeric(audit["new_review_order"], errors="coerce")
    audit["order_changed"] = old_order.ne(new_order) & ~(old_order.isna() & new_order.isna())
    old_rank = pd.to_numeric(audit["old_state_dwell_rank"], errors="coerce")
    new_rank = pd.to_numeric(audit["new_state_dwell_rank"], errors="coerce")
    rank_changed = old_rank.ne(new_rank) & ~(old_rank.isna() & new_rank.isna())
    old_transition = pd.to_numeric(audit["old_transition_count"], errors="coerce")
    new_transition = pd.to_numeric(audit["new_transition_count"], errors="coerce")
    transition_changed = old_transition.ne(new_transition) & ~(old_transition.isna() & new_transition.isna())
    audit["note"] = np.select(
        [
            audit["order_changed"].fillna(False),
            rank_changed.fillna(False),
            transition_changed.fillna(False),
        ],
        [
            "review_order changed due to corrected state transition feature",
            "state_dwell_rank changed; review_order unchanged",
            "transition_count corrected; review_order unchanged",
        ],
        default="transition_count already consistent; review_order unchanged",
    )
    return _write_csv(audit, audit_path, TRANSITION_AUDIT_COLS)


def _write_rank_audit(out, old_review, new_review):
    out = Path(out)
    sessions = set()
    for frame in [old_review, new_review]:
        if not frame.empty and "session_key" in frame.columns:
            sessions.update(frame["session_key"].dropna().astype(str))
    audit = pd.DataFrame({"session_key": sorted(sessions)})
    if audit.empty:
        return _write_csv(audit, out / "m2_review_rank_audit.csv", RANK_AUDIT_COLS)

    if not old_review.empty and "session_key" in old_review.columns:
        old_cols = [c for c in ["session_key", "review_order", "rra_q", "top_interval_duration"] if c in old_review.columns]
        old = old_review[old_cols].copy().rename(columns={
            "review_order": "old_review_order",
            "rra_q": "old_rra_q",
            "top_interval_duration": "old_top_interval_duration",
        })
        audit = audit.merge(old, on="session_key", how="left")
    for col in ["old_review_order", "old_rra_q", "old_top_interval_duration"]:
        if col not in audit.columns:
            audit[col] = np.nan

    new_cols = [
        "session_key",
        "review_order",
        "family_rra_q",
        "top_interval_duration",
        "family_consensus_score",
        "persistence_family_rank",
        "scan_family_rank",
        "expected_response_family_rank",
        "interval_anomaly_family_rank",
        "reason_support_family_rank",
    ]
    if not new_review.empty and "session_key" in new_review.columns:
        new = new_review[[c for c in new_cols if c in new_review.columns]].copy().rename(columns={
            "review_order": "new_review_order",
            "family_rra_q": "new_family_rra_q",
            "top_interval_duration": "new_top_interval_duration",
        })
        audit = audit.merge(new, on="session_key", how="left")
    for col in RANK_AUDIT_COLS:
        if col not in audit.columns:
            audit[col] = np.nan

    old_order = pd.to_numeric(audit["old_review_order"], errors="coerce")
    new_order = pd.to_numeric(audit["new_review_order"], errors="coerce")
    audit["order_changed"] = old_order.ne(new_order) & ~(old_order.isna() & new_order.isna())
    audit["note"] = np.select(
        [
            old_order.isna() & new_order.notna(),
            audit["order_changed"].fillna(False),
        ],
        [
            "new family-consensus review row; no previous order",
            "review_order changed under equal-weight family consensus",
        ],
        default="review_order unchanged under equal-weight family consensus",
    )
    return _write_csv(audit, out / "m2_review_rank_audit.csv", RANK_AUDIT_COLS)


def _rra_p_value(row, evidence_cols, n_sessions):
    vals = []
    for col in evidence_cols:
        rank = row.get(col)
        if pd.notna(rank):
            vals.append(min(max(float(rank) / (n_sessions + 1.0), 1.0 / (n_sessions + 1.0)), 1.0))
        else:
            vals.append(1.0)
    vals = sorted(vals)
    m = len(vals)
    if m == 0:
        return 1.0
    try:
        from scipy.stats import beta

        probs = [float(beta.cdf(r, i, m - i + 1)) for i, r in enumerate(vals, start=1)]
    except Exception:
        probs = [float(r ** i) for i, r in enumerate(vals, start=1)]
    return float(min(max(min(probs) * m, 0.0), 1.0))


def _bh_q_values(p_values):
    p = pd.to_numeric(pd.Series(p_values), errors="coerce").fillna(1.0).clip(0, 1).to_numpy()
    try:
        from statsmodels.stats.multitest import multipletests

        return multipletests(p, method="fdr_bh")[1]
    except Exception:
        n = len(p)
        order = np.argsort(p)
        ranked = p[order]
        q = ranked * n / np.arange(1, n + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0, 1)
        out = np.empty(n, dtype=float)
        out[order] = q
        return out


def _reason_summary(out, meta):
    reason = _read(out, "m2_reason.csv")
    if reason.empty:
        return meta[["session_key"]].assign(dominant_reason=np.nan, reason_set=np.nan)
    if "text_reason_included" in reason.columns:
        text_mask = reason["text_reason_included"].astype(str).str.lower().isin(["true", "1", "yes"])
        text_reason = reason.loc[text_mask].copy()
        if not text_reason.empty:
            reason = text_reason
    reason["support_value"] = pd.to_numeric(reason["support_value"], errors="coerce")
    dom = (
        reason.sort_values(["session_key", "support_value", "support_rank"], ascending=[True, False, False])
        .drop_duplicates("session_key")
        [["session_key", "reason"]]
        .rename(columns={"reason": "dominant_reason"})
    )
    sets = reason.groupby("session_key")["reason"].apply(lambda s: ";".join(sorted(set(s)))).reset_index(name="reason_set")
    return meta[["session_key"]].merge(dom, on="session_key", how="left").merge(sets, on="session_key", how="left")


def _review_note(row):
    bits = []
    if pd.notna(row.get("family_consensus_score")):
        bits.append(f"family consensus score={float(row['family_consensus_score']):.4g}")
    if pd.notna(row.get("persistence_family_rank")):
        bits.append(f"persistence evidence rank={int(round(float(row['persistence_family_rank'])))}")
    if pd.notna(row.get("scan_family_rank")):
        bits.append(f"adaptive scan family rank={int(round(float(row['scan_family_rank'])))}")
    if pd.notna(row.get("expected_response_family_rank")):
        bits.append(f"expected-response deficit rank={int(round(float(row['expected_response_family_rank'])))}")
    if pd.notna(row.get("minute_state_family_rank")):
        bits.append(f"minute-state mismatch rank={int(round(float(row['minute_state_family_rank'])))}")
    duration = row.get("top_interval_duration")
    if pd.notna(duration):
        bits.append(f"top interval duration={int(round(float(duration)))}min")
        if float(duration) <= 1:
            bits.append("short interval evidence; review requires manual confirmation")
    if isinstance(row.get("reason_set"), str) and row["reason_set"]:
        bits.append(row["reason_set"].replace("_", " ").replace(";", ", "))
    if pd.notna(row.get("interval_anomaly_family_rank")):
        bits.append("interval profile 보조 근거 포함")
    return "수동 검토 우선순위 근거: " + ", ".join(bits or ["여러 비지도 근거의 순위 결합"]) + "."


def _session_review_eligibility(out, meta, cfg):
    cfg = cfg or {}
    min_n = int(cfg.get("prep", {}).get("min_n", 10))
    session_all = _read(out, "session_all.csv")
    session_model = _read(out, "session_summary_processed.csv")
    source = session_all if not session_all.empty else session_model
    cols = ["session_key", "n", "v_qc", "all_zero", "ok"]
    if source.empty:
        base = meta[["session_key"]].copy()
        base["n_session_minutes"] = np.nan
        base["v_qc"] = np.nan
        base["all_zero_session"] = np.nan
        base["eligible_review"] = False
        base["review_qc_reason"] = "missing_session_qc"
        return base
    keep = [c for c in cols if c in source.columns]
    s = source[keep].drop_duplicates("session_key").copy()
    if "n" not in s.columns:
        s["n"] = np.nan
    if "v_qc" not in s.columns:
        s["v_qc"] = 0
    if "all_zero" not in s.columns:
        s["all_zero"] = False
    s["n_session_minutes"] = pd.to_numeric(s["n"], errors="coerce")
    s["v_qc"] = pd.to_numeric(s["v_qc"], errors="coerce").fillna(0)
    s["all_zero_session"] = s["all_zero"].astype(str).str.lower().isin(["true", "1", "yes"])
    if "ok" in s.columns:
        ok = s["ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        ok = s["n_session_minutes"].ge(min_n) & s["v_qc"].eq(0) & ~s["all_zero_session"]
    s["eligible_review"] = ok.fillna(False)

    def reason(row):
        bits = []
        if pd.isna(row.get("n_session_minutes")):
            bits.append("missing_n")
        elif float(row.get("n_session_minutes")) < min_n:
            bits.append(f"n_below_min_n_{min_n}")
        if float(row.get("v_qc", 0) or 0) != 0:
            bits.append("v_qc_nonzero")
        if bool(row.get("all_zero_session", False)):
            bits.append("all_zero_session")
        return "eligible" if not bits else ";".join(bits)

    s["review_qc_reason"] = s.apply(reason, axis=1)
    s = s[["session_key", "n_session_minutes", "eligible_review", "review_qc_reason", "all_zero_session", "v_qc"]]
    out_df = meta[["session_key"]].merge(s, on="session_key", how="left")
    out_df["eligible_review"] = out_df["eligible_review"].fillna(False)
    out_df["review_qc_reason"] = out_df["review_qc_reason"].fillna("missing_session_qc")
    return out_df


def build_m2_review(out, cfg=None):
    out = Path(out)
    cfg = cfg or {}
    old_review = _read(out, "m2_review.csv")
    scan = _read(out, "m2_scan.csv")
    meta = _session_meta(scan)
    if meta.empty:
        empty = _write_csv(pd.DataFrame(columns=REVIEW_COLS), out / "m2_review.csv", REVIEW_COLS)
        _write_transition_fix_audit(out, old_review, empty)
        return empty

    review = meta.copy()
    review = review.merge(_session_review_eligibility(out, meta, cfg), on="session_key", how="left")
    ranks = _build_evidence_ranks(out, meta, scan)
    for _, df in ranks.items():
        review = review.merge(df, on="session_key", how="left")

    n_sessions = max(len(meta), 1)
    review["raw_rra_p"] = review.apply(lambda row: _rra_p_value(row, RAW_EVIDENCE_COLS, n_sessions), axis=1)
    review["raw_rra_q"] = _bh_q_values(review["raw_rra_p"])
    rank_matrix = review[RAW_EVIDENCE_COLS].apply(pd.to_numeric, errors="coerce")
    rank_strength = 1.0 - (rank_matrix / (n_sessions + 1.0))
    review["evidence_repetition_score"] = rank_strength.clip(lower=0, upper=1).mean(axis=1, skipna=True)
    review["evidence_count"] = rank_matrix.notna().sum(axis=1)

    family_ranks = _build_family_ranks(out, meta, scan, ranks)
    for _, df in family_ranks.items():
        review = review.merge(df, on="session_key", how="left")
    for rank_col, strength_col in zip(FAMILY_RANK_COLS, FAMILY_STRENGTH_COLS):
        review[strength_col] = _strength_from_rank(review[rank_col], n_sessions)
    # missing family = no evidence in that view -> 0, not skipna, so a session
    # strong in one family but absent in the rest can't free-ride. Matches RRA's
    # worst-fill below for the same incomplete-evidence sessions.
    family_strength = review[CONSENSUS_FAMILY_STRENGTH_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    review["family_consensus_score"] = family_strength.mean(axis=1)
    review["evidence_family_count"] = review[CONSENSUS_FAMILY_RANK_COLS].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
    review["family_rra_p"] = review.apply(lambda row: _rra_p_value(row, CONSENSUS_FAMILY_RANK_COLS, n_sessions), axis=1)
    review["family_rra_q"] = _bh_q_values(review["family_rra_p"])
    review["rra_p"] = review["family_rra_p"]
    review["rra_q"] = review["family_rra_q"]
    review["ranking_method"] = RANKING_METHOD

    keep = scan[[
        "session_key",
        "top_interval_start_ts",
        "top_interval_end_ts",
        "top_interval_duration",
        "observed_scan_z",
        "empirical_p",
    ]].rename(columns={
        "top_interval_start_ts": "top_interval_start",
        "top_interval_end_ts": "top_interval_end",
    })
    review = review.merge(keep, on="session_key", how="left").merge(_reason_summary(out, meta), on="session_key", how="left")
    review["top_interval_duration"] = pd.to_numeric(review["top_interval_duration"], errors="coerce")
    review["empirical_p"] = pd.to_numeric(review["empirical_p"], errors="coerce")
    review["observed_scan_z"] = pd.to_numeric(review["observed_scan_z"], errors="coerce")
    review["review_qc_reason"] = review["review_qc_reason"].fillna("missing_session_qc")
    review_all = review.copy()
    if bool(cfg.get("m2_review", {}).get("write_all_review_file", True)):
        review_all_sorted = review_all.sort_values(
            [
                "eligible_review",
                "family_consensus_score",
                "family_rra_q",
                "family_rra_p",
                "persistence_family_rank",
                "scan_family_rank",
                "session_key",
            ],
            ascending=[False, False, True, True, True, True, True],
            na_position="last",
        ).reset_index(drop=True)
        review_all_sorted["review_order"] = np.arange(1, len(review_all_sorted) + 1)
        review_all_sorted["review_note"] = review_all_sorted.apply(_review_note, axis=1)
        _write_csv(review_all_sorted, out / "m2_review_all.csv", REVIEW_COLS)
    if not bool(cfg.get("m2_review", {}).get("include_ineligible_in_final_review", False)):
        review = review.loc[review["eligible_review"].fillna(False).astype(bool)].copy()
    review = review.sort_values(
        [
            "family_consensus_score",
            "family_rra_q",
            "family_rra_p",
            "persistence_family_rank",
            "scan_family_rank",
            "session_key",
        ],
        ascending=[False, True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    review["review_order"] = np.arange(1, len(review) + 1)
    review["review_note"] = review.apply(_review_note, axis=1)
    final_review = _write_csv(review, out / "m2_review.csv", REVIEW_COLS)
    _write_transition_fix_audit(out, old_review, final_review)
    _write_rank_audit(out, old_review, final_review)
    return final_review


def _plot_review(out, plots):
    review = _read(out, "m2_review.csv")
    if review.empty:
        _blank_plot(plots / "18_review.png", "m2_review.csv not available")
        return
    top = review.sort_values("review_order").head(10).copy().iloc[::-1]
    labels = (
        top["session_key"].astype(str)
        + " | "
        + pd.to_numeric(top["top_interval_duration"], errors="coerce").fillna(0).round(0).astype(int).astype(str)
        + "m | "
        + top["dominant_reason"].fillna("reason_pending").astype(str)
    )
    score = -np.log10(pd.to_numeric(top["rra_q"], errors="coerce").clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.barh(labels, score, edgecolor="black", color="tab:blue", alpha=0.82)
    ax.set_xlabel("-log10(family_rra_q)")
    ax.set_title("18 수동 검토 우선순위: family consensus evidence - 정답 라벨/확률 아님")
    ax.grid(axis="x", alpha=0.25)
    _save_fig(fig, plots / "18_review.png")


def _plot_reason(out, plots):
    reason = _read(out, "m2_reason.csv")
    patterns = _read(out, "m2_patterns.csv")
    if reason.empty and patterns.empty:
        _blank_plot(plots / "19_reason.png", "m2_reason.csv not available")
        return
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    if not reason.empty:
        if "text_reason_included" in reason.columns:
            mask = reason["text_reason_included"].astype(str).str.lower().isin(["true", "1", "yes"])
            reason_plot = reason.loc[mask].copy()
        else:
            reason_plot = reason.iloc[0:0].copy()
        counts = reason_plot["reason"].value_counts().head(12).iloc[::-1] if not reason_plot.empty else pd.Series(dtype=int)
        ax[0].barh(counts.index, counts.values, edgecolor="black", color="tab:green")
    ax[0].set_title("top text reason count")
    ax[0].set_xlabel("sessions where reason appears in top-3 explanation")
    if not patterns.empty:
        p = patterns.sort_values("support_count", ascending=False).head(8).iloc[::-1]
        ax[1].barh(p["reason_pattern"].astype(str), pd.to_numeric(p["support_count"], errors="coerce"), edgecolor="black", color="tab:orange")
    ax[1].set_title("reason pattern")
    ax[1].set_xlabel("sessions")
    for a in ax:
        a.grid(axis="x", alpha=0.25)
    fig.suptitle("19 reason support 분해 - confidence/lift/정답 라벨 아님", fontsize=14)
    _save_fig(fig, plots / "19_reason.png")


def _plot_rra(out, plots):
    review = _read(out, "m2_review.csv")
    if review.empty:
        _blank_plot(plots / "20_rra.png", "m2_review.csv not available")
        return
    review = review.sort_values("review_order")
    x = pd.to_numeric(review["review_order"], errors="coerce")
    y = -np.log10(pd.to_numeric(review["rra_q"], errors="coerce").clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(x, y, s=24, alpha=0.75, edgecolor="white", linewidth=0.3)
    for i, (_, row) in enumerate(review.head(5).iterrows()):
        q = pd.to_numeric(pd.Series([row["rra_q"]]), errors="coerce").fillna(1.0).iloc[0]
        offset = (6, 8 + (i % 3) * 9)
        ax.annotate(
            str(row["session_key"]),
            xy=(row["review_order"], -np.log10(max(float(q), 1e-300))),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.4, alpha=0.55),
        )
    ax.set_xlabel("review_order")
    ax.set_ylabel("-log10(family_rra_q)")
    ax.set_title("20 family RRA rank 구성: rra_q는 family_rra_q이며 확률이나 정답 라벨이 아님; labels show top review candidates only")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots / "20_rra.png")


def _plot_interval(out, plots):
    review = _read(out, "m2_review.csv")
    if review.empty:
        _blank_plot(plots / "21_interval.png", "adaptive interval inputs not available")
        return
    top = review.sort_values("review_order").head(100).copy()
    top["review_order"] = pd.to_numeric(top["review_order"], errors="coerce")
    top["top_interval_duration"] = pd.to_numeric(top["top_interval_duration"], errors="coerce")
    top["empirical_p"] = pd.to_numeric(top["empirical_p"], errors="coerce")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    strength = pd.to_numeric(top["family_consensus_score"], errors="coerce") if "family_consensus_score" in top.columns else pd.Series(np.nan, index=top.index)
    colors = strength.fillna(strength.median() if strength.notna().any() else 0.0)
    norm = Normalize(vmin=float(colors.min()), vmax=float(colors.max())) if len(colors) and float(colors.min()) != float(colors.max()) else Normalize(vmin=0.0, vmax=1.0)
    cmap = "viridis"
    sc0 = ax[0].scatter(top["review_order"], top["top_interval_duration"], c=colors, cmap=cmap, norm=norm, s=35, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax[0].set_xlabel("검토 순서")
    ax[0].set_ylabel("상위 scan 구간 길이(분)")
    ax[0].set_title("검토 순서와 구간 지속성")
    sc1 = ax[1].scatter(top["review_order"], top["empirical_p"], c=colors, cmap=cmap, norm=norm, s=35, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax[1].set_xlabel("검토 순서")
    ax[1].set_ylabel("empirical_p from shuffled-null; not detection probability")
    ax[1].set_title("검토 순서와 shuffled-null 근거")
    fig.colorbar(sc1, ax=ax, label="family consensus score, not probability")
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle("21 adaptive interval 근거: 숫자 cutoff flag 없이 연속값으로 확인", fontsize=14)
    _save_fig(fig, plots / "21_interval.png")


def build_m2_review_plots(out, cfg=None):
    out = Path(out)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _plot_review(out, plots)
    _plot_reason(out, plots)
    _plot_rra(out, plots)
    _plot_interval(out, plots)
