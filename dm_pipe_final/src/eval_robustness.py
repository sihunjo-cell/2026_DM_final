from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.m2_scan import build_m2_scan


DPI = 200
NOTE_NOT_REAL = "synthetic mismatch recovery only; not real viewbot performance"
REPORT_REQUIRED_SENTENCE = "본 평가는 실제 viewbot label에 대한 성능평가가 아니라, label-free review ranking pipeline의 robustness와 synthetic mismatch recovery를 확인하기 위한 것이다."

DEFAULT_FAMILIES = [
    "scan_family_rank",
    "persistence_family_rank",
    "expected_response_family_rank",
    "minute_state_family_rank",
    "interval_anomaly_family_rank",
    "reason_support_family_rank",
]

DEFAULT_SIGNALS = [
    "chat_deficit",
    "unique_deficit",
    "rolling_chat_deficit_5m",
    "zero_run_len",
    "rolling_zero_rate_5m",
    "cluster_mismatch_rank",
]

FAMILY_ABLATION_SUMMARY_COLS = [
    "removed_family",
    "top20_overlap",
    "top50_overlap",
    "top100_overlap",
    "top200_overlap",
    "top20_jaccard",
    "top50_jaccard",
    "top100_jaccard",
    "top200_jaccard",
    "spearman_eligible",
    "median_abs_rank_shift_top100",
    "interpretation",
]

AGGREGATION_SUMMARY_COLS = [
    "variant",
    "top20_overlap",
    "top50_overlap",
    "top100_overlap",
    "top200_overlap",
    "spearman_eligible",
    "median_abs_rank_shift_top100",
    "interpretation",
]

EVIDENCE_AUDIT_COLS = [
    "session_key",
    "review_order",
    "eligible_review",
    "strong_family_count",
    "family_strength_mean",
    "family_strength_median",
    "family_strength_std",
    "family_strength_entropy",
    "top_family_name",
    "top_to_second_gap",
    "single_family_dominant",
]

TIE_AUDIT_COLS = [
    "family",
    "n_sessions_all",
    "n_unique_values",
    "n_tied_sessions",
    "max_tie_block_size",
    "pct_sessions_in_ties",
    "note",
]

SIGNAL_ABLATION_COLS = [
    "removed_signal",
    "top1pct_minute_overlap",
    "top5pct_minute_overlap",
    "top50_session_overlap",
    "top100_session_overlap",
    "median_top_interval_iou_base_top100",
    "median_duration_shift_base_top100",
    "interpretation",
]

SIGNAL_WEIGHT_COLS = [
    "variant",
    "weight_chat_deficit",
    "weight_unique_deficit",
    "weight_rolling_chat_deficit_5m",
    "weight_zero_run_len",
    "weight_rolling_zero_rate_5m",
    "weight_cluster_mismatch_rank",
    "spearman_minute_score",
    "top1pct_minute_overlap",
    "top5pct_minute_overlap",
    "top50_session_overlap",
    "top100_session_overlap",
    "median_top_interval_iou_base_top100",
    "interpretation",
]

SYNTHETIC_RECOVERY_COLS = [
    "synthetic_session_key",
    "source_session_key",
    "injection_type",
    "injected_start_idx",
    "injected_end_idx",
    "detected_start_idx",
    "detected_end_idx",
    "iou",
    "recovered_iou_0.3",
    "recovered_iou_0.5",
    "recovered_iou_0.7",
    "note",
]

SCORECARD_COLS = [
    "metric",
    "value",
    "scale",
    "what_it_checks",
    "note",
]

PROFILE_COLS = [
    "feature",
    "top_n",
    "rest_n",
    "top_median",
    "rest_median",
    "top_mean",
    "rest_mean",
    "std_mean_diff",
    "direction",
    "note",
]

# (m2_scores column, aggregation) used to profile why top review sessions rank high.
PROFILE_FEATURES = [
    ("log_viewer", "median"),
    ("chat_deficit", "median"),
    ("unique_deficit", "median"),
    ("rolling_chat_deficit_5m", "median"),
    ("zero_run_len", "max"),
    ("rolling_zero_rate_5m", "median"),
    ("minute_mismatch_score", "median"),
]


_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = "Malgun Gothic" if "Malgun Gothic" in _available_fonts else "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    for enc in ["utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


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


def _write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _remove_if_exists(path):
    path = Path(path)
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _resolve_child_path(out, configured, default_rel):
    out = Path(out)
    raw = configured or default_rel
    path = Path(raw)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == out.name:
        return out.parent / path
    if parts and parts[0] == "out":
        return out.parent / path
    return out / path


def _resolve_project_path(out, configured):
    path = Path(configured)
    if path.is_absolute():
        return path
    return Path(out).parent / path


def _families(eval_cfg):
    return list(eval_cfg.get("families") or DEFAULT_FAMILIES)


def _signals(eval_cfg):
    return list(eval_cfg.get("minute_score_signals") or DEFAULT_SIGNALS)


def _topk(eval_cfg):
    vals = eval_cfg.get("topk", [20, 50, 100, 200])
    return [int(v) for v in vals]


def _to_bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _pct_rank(s):
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return x.rank(method="average", pct=True)


def _strength_col(rank_col):
    return rank_col.replace("_rank", "_strength")


def _family_strengths(review_all, families):
    out = pd.DataFrame({"session_key": review_all.get("session_key", pd.Series(dtype=str)).astype(str)})
    n_all = max(int(len(review_all)), 1)
    for family in families:
        rank = pd.to_numeric(review_all.get(family), errors="coerce")
        out[_strength_col(family)] = (1.0 - rank / (n_all + 1.0)).clip(lower=0.0, upper=1.0)
    return out


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
        if n == 0:
            return np.array([], dtype=float)
        order = np.argsort(p)
        ranked = p[order]
        q = ranked * n / np.arange(1, n + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty(n, dtype=float)
        out[order] = np.clip(q, 0, 1)
        return out


def _review_frames(out, eval_cfg):
    out = Path(out)
    review_all = _read_csv(out / "m2_review_all.csv")
    review = _read_csv(out / "m2_review.csv")
    families = _families(eval_cfg)
    if review_all.empty:
        return review_all, review, families, []
    review_all = review_all.copy()
    review_all["session_key"] = review_all["session_key"].astype(str)
    strengths = _family_strengths(review_all, families)
    for col in [_strength_col(f) for f in families]:
        review_all[col] = strengths[col]
    if "eligible_review" not in review_all.columns:
        review_all["eligible_review"] = True
    review_all["eligible_review"] = _to_bool(review_all["eligible_review"]) if review_all["eligible_review"].dtype == object else review_all["eligible_review"].fillna(False).astype(bool)
    if not review.empty and "session_key" in review.columns:
        review = review.copy()
        review["session_key"] = review["session_key"].astype(str)
        review["review_order"] = pd.to_numeric(review.get("review_order"), errors="coerce")
    return review_all, review, families, [_strength_col(f) for f in families]


def _base_order(review, review_all):
    if review is not None and not review.empty and "review_order" in review.columns:
        base = review[["session_key", "review_order"]].copy()
        base["review_order"] = pd.to_numeric(base["review_order"], errors="coerce")
        base = base.dropna(subset=["review_order"]).sort_values(["review_order", "session_key"])
        return base
    if review_all is None or review_all.empty:
        return pd.DataFrame(columns=["session_key", "review_order"])
    eligible = review_all.loc[review_all["eligible_review"].fillna(False).astype(bool)].copy()
    eligible["review_order"] = pd.to_numeric(eligible.get("review_order"), errors="coerce")
    return eligible.dropna(subset=["review_order"])[["session_key", "review_order"]].sort_values(["review_order", "session_key"])


def _assign_variant_ranks(df, sort_cols, ascending):
    work = df.copy()
    work = work.sort_values(sort_cols + ["session_key"], ascending=ascending + [True], na_position="last").reset_index(drop=True)
    work["variant_rank_all"] = np.arange(1, len(work) + 1, dtype=float)
    eligible = work.loc[work.get("eligible_review", False).fillna(False).astype(bool)].copy()
    eligible["variant_rank_eligible"] = np.arange(1, len(eligible) + 1, dtype=float)
    work = work.merge(eligible[["session_key", "variant_rank_eligible"]], on="session_key", how="left")
    return work


def _overlap_stats(base, variant, topk, include_jaccard=True):
    stats = {}
    base_order = base.sort_values(["review_order", "session_key"])
    variant_order = variant.dropna(subset=["variant_rank_eligible"]).sort_values(["variant_rank_eligible", "session_key"])
    for k in topk:
        bset = set(base_order.head(k)["session_key"].astype(str))
        vset = set(variant_order.head(k)["session_key"].astype(str))
        inter = len(bset & vset)
        stats[f"top{k}_overlap"] = inter / len(bset) if bset else np.nan
        if include_jaccard:
            union = len(bset | vset)
            stats[f"top{k}_jaccard"] = inter / union if union else np.nan
    return stats


def _spearman_eligible(base, variant):
    merged = base.merge(variant[["session_key", "variant_rank_eligible"]], on="session_key", how="inner")
    merged["review_order"] = pd.to_numeric(merged["review_order"], errors="coerce")
    merged["variant_rank_eligible"] = pd.to_numeric(merged["variant_rank_eligible"], errors="coerce")
    merged = merged.dropna(subset=["review_order", "variant_rank_eligible"])
    if len(merged) < 2:
        return np.nan
    return float(merged["review_order"].corr(merged["variant_rank_eligible"], method="spearman"))


def _median_shift_top100(base, variant):
    top = base.sort_values(["review_order", "session_key"]).head(100)
    merged = top.merge(variant[["session_key", "variant_rank_eligible"]], on="session_key", how="left")
    shift = (pd.to_numeric(merged["variant_rank_eligible"], errors="coerce") - pd.to_numeric(merged["review_order"], errors="coerce")).abs()
    return float(shift.median()) if shift.notna().any() else np.nan


def _family_interpretation(overlap):
    if pd.isna(overlap):
        return "Insufficient eligible review rows for ranking robustness comparison."
    if overlap >= 0.8:
        return "High overlap: review ranking robustness is not driven by this single family."
    if overlap < 0.5:
        return "Lower overlap: ranking sensitivity indicates this family is core evidence."
    return "Moderate overlap: ranking sensitivity is present but not a complete reordering."


def run_family_ablation(out, eval_dir, cfg):
    eval_cfg = cfg.get("eval_robustness", {})
    review_all, review, families, strength_cols = _review_frames(out, eval_cfg)
    base = _base_order(review, review_all)
    if review_all.empty:
        _write_csv(pd.DataFrame(columns=FAMILY_ABLATION_SUMMARY_COLS), eval_dir / "family_ablation_summary.csv", FAMILY_ABLATION_SUMMARY_COLS)
        _write_csv(pd.DataFrame(), eval_dir / "family_ablation_rankings.csv")
        return pd.DataFrame(columns=FAMILY_ABLATION_SUMMARY_COLS)

    summaries = []
    ranking_parts = []
    topk = _topk(eval_cfg)
    for removed in families:
        remaining = [_strength_col(f) for f in families if f != removed]
        variant = review_all[["session_key", "eligible_review"]].copy()
        variant["removed_family"] = removed
        variant["variant_score"] = review_all[remaining].apply(lambda col: pd.to_numeric(col, errors="coerce")).mean(axis=1, skipna=True) if remaining else np.nan
        ranked = _assign_variant_ranks(variant, ["variant_score"], [False])
        stats = _overlap_stats(base, ranked, topk, include_jaccard=True)
        row = {"removed_family": removed, **stats}
        row["spearman_eligible"] = _spearman_eligible(base, ranked)
        row["median_abs_rank_shift_top100"] = _median_shift_top100(base, ranked)
        row["interpretation"] = _family_interpretation(row.get("top100_overlap"))
        summaries.append(row)
        ranking_parts.append(ranked)

    summary = pd.DataFrame(summaries)
    rankings = pd.concat(ranking_parts, ignore_index=True) if ranking_parts else pd.DataFrame()
    _write_csv(summary, eval_dir / "family_ablation_summary.csv", FAMILY_ABLATION_SUMMARY_COLS)
    _write_csv(rankings, eval_dir / "family_ablation_rankings.csv")
    return summary


def _trimmed_mean(row):
    vals = pd.to_numeric(row, errors="coerce").dropna().sort_values().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan
    if len(vals) <= 2:
        return float(np.mean(vals))
    return float(np.mean(vals[1:-1]))


def run_aggregation_sensitivity(out, eval_dir, cfg):
    eval_cfg = cfg.get("eval_robustness", {})
    review_all, review, families, strength_cols = _review_frames(out, eval_cfg)
    base = _base_order(review, review_all)
    if review_all.empty:
        _write_csv(pd.DataFrame(columns=AGGREGATION_SUMMARY_COLS), eval_dir / "aggregation_sensitivity_summary.csv", AGGREGATION_SUMMARY_COLS)
        _write_csv(pd.DataFrame(), eval_dir / "aggregation_sensitivity_rankings.csv")
        return pd.DataFrame(columns=AGGREGATION_SUMMARY_COLS)

    topk = _topk(eval_cfg)
    variants = []
    strengths = review_all[strength_cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))
    n_all = max(len(review_all), 1)

    def score_variant(name, score):
        frame = review_all[["session_key", "eligible_review"]].copy()
        frame["variant"] = name
        frame["variant_score"] = score
        return _assign_variant_ranks(frame, ["variant_score"], [False])

    variants.append(score_variant("family_consensus_only", strengths.mean(axis=1, skipna=True)))

    rra = review_all[["session_key", "eligible_review"] + families].copy()
    rra["variant"] = "family_rra_only"
    rra["variant_rra_p"] = rra.apply(lambda row: _rra_p_value(row, families, n_all), axis=1)
    rra["variant_rra_q"] = _bh_q_values(rra["variant_rra_p"])
    rra["variant_score"] = 1.0 - pd.to_numeric(rra["variant_rra_q"], errors="coerce")
    variants.append(_assign_variant_ranks(rra, ["variant_rra_q", "variant_rra_p"], [True, True]))

    combo = review_all[["session_key", "eligible_review"] + families].copy()
    combo["variant"] = "family_consensus_plus_rra"
    combo["variant_score"] = strengths.mean(axis=1, skipna=True)
    combo["variant_rra_p"] = combo.apply(lambda row: _rra_p_value(row, families, n_all), axis=1)
    combo["variant_rra_q"] = _bh_q_values(combo["variant_rra_p"])
    variants.append(_assign_variant_ranks(combo, ["variant_score", "variant_rra_q", "variant_rra_p"], [False, True, True]))

    variants.append(score_variant("median_family_strength", strengths.median(axis=1, skipna=True)))
    variants.append(score_variant("trimmed_mean_family_strength", strengths.apply(_trimmed_mean, axis=1)))

    for family in ["interval_anomaly_family_rank", "reason_support_family_rank"]:
        keep = [_strength_col(f) for f in families if f != family]
        variants.append(score_variant(f"no_{family.replace('_rank', '')}", strengths[keep].mean(axis=1, skipna=True) if keep else np.nan))

    summaries = []
    for ranked in variants:
        name = str(ranked["variant"].iloc[0]) if not ranked.empty else "unknown"
        stats = _overlap_stats(base, ranked, topk, include_jaccard=False)
        row = {"variant": name, **stats}
        row["spearman_eligible"] = _spearman_eligible(base, ranked)
        row["median_abs_rank_shift_top100"] = _median_shift_top100(base, ranked)
        row["interpretation"] = _family_interpretation(row.get("top100_overlap"))
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    rankings = pd.concat(variants, ignore_index=True) if variants else pd.DataFrame()
    _write_csv(summary, eval_dir / "aggregation_sensitivity_summary.csv", AGGREGATION_SUMMARY_COLS)
    _write_csv(rankings, eval_dir / "aggregation_sensitivity_rankings.csv")
    return summary


def _entropy(vals):
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().clip(lower=0).to_numpy(dtype=float)
    if len(arr) == 0 or arr.sum() <= 0:
        return np.nan
    p = arr / arr.sum()
    den = math.log(len(p)) if len(p) > 1 else 1.0
    return float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))) / den)


def _bucket(order):
    if pd.isna(order):
        return "rest"
    order = float(order)
    if order <= 20:
        return "top20"
    if order <= 50:
        return "top50"
    if order <= 100:
        return "top100"
    if order <= 200:
        return "top200"
    return "rest"


def run_evidence_balance(out, eval_dir, cfg):
    eval_cfg = cfg.get("eval_robustness", {})
    review_all, review, families, strength_cols = _review_frames(out, eval_cfg)
    threshold = float(eval_cfg.get("strong_family_strength_threshold", 0.90))
    if review_all.empty:
        _write_csv(pd.DataFrame(columns=EVIDENCE_AUDIT_COLS), eval_dir / "evidence_balance_audit.csv", EVIDENCE_AUDIT_COLS)
        _write_csv(pd.DataFrame(), eval_dir / "evidence_balance_summary.csv")
        return pd.DataFrame()

    strengths = review_all[strength_cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))
    audit = review_all[["session_key", "review_order", "eligible_review"]].copy()
    for col in strength_cols:
        audit[col] = strengths[col]
    audit["strong_family_count"] = strengths.ge(threshold).sum(axis=1)
    audit["family_strength_mean"] = strengths.mean(axis=1, skipna=True)
    audit["family_strength_median"] = strengths.median(axis=1, skipna=True)
    audit["family_strength_std"] = strengths.std(axis=1, skipna=True)
    audit["family_strength_entropy"] = strengths.apply(_entropy, axis=1)

    top_names = []
    gaps = []
    dominant = []
    for _, row in strengths.iterrows():
        vals = row.dropna().sort_values(ascending=False)
        if vals.empty:
            top_names.append(np.nan)
            gaps.append(np.nan)
            dominant.append(False)
            continue
        top_names.append(vals.index[0].replace("_strength", "_rank"))
        second = vals.iloc[1] if len(vals) > 1 else 0.0
        gap = float(vals.iloc[0] - second)
        gaps.append(gap)
        dominant.append(bool(vals.iloc[0] >= threshold and gap >= 0.15))
    audit["top_family_name"] = top_names
    audit["top_to_second_gap"] = gaps
    audit["single_family_dominant"] = dominant
    audit["review_order_bucket"] = pd.to_numeric(audit.get("review_order"), errors="coerce").map(_bucket)

    rows = []
    for bucket in ["top20", "top50", "top100", "top200", "rest"]:
        part = audit.loc[audit["review_order_bucket"].eq(bucket)]
        rows.append({
            "review_order_bucket": bucket,
            "n_sessions": int(len(part)),
            "mean_strong_family_count": float(part["strong_family_count"].mean()) if not part.empty else np.nan,
            "median_strong_family_count": float(part["strong_family_count"].median()) if not part.empty else np.nan,
            "mean_family_strength_entropy": float(part["family_strength_entropy"].mean()) if not part.empty else np.nan,
            "single_family_dominant_rate": float(part["single_family_dominant"].mean()) if not part.empty else np.nan,
            "median_top_to_second_gap": float(part["top_to_second_gap"].median()) if not part.empty else np.nan,
            "interpretation": "Multiple families support this review tier when strong_family_count is high; single-family dominance is reported as a limitation.",
        })
    summary = pd.DataFrame(rows)
    _write_csv(audit, eval_dir / "evidence_balance_audit.csv")
    _write_csv(summary, eval_dir / "evidence_balance_summary.csv")
    return summary


def run_tie_audit(out, eval_dir, cfg):
    eval_cfg = cfg.get("eval_robustness", {})
    review_all = _read_csv(Path(out) / "m2_review_all.csv")
    rows = []
    for family in _families(eval_cfg):
        vals = pd.to_numeric(review_all.get(family), errors="coerce").dropna() if not review_all.empty else pd.Series(dtype=float)
        counts = vals.value_counts(dropna=True)
        tied = counts[counts.gt(1)]
        n_all = int(len(review_all))
        rows.append({
            "family": family,
            "n_sessions_all": n_all,
            "n_unique_values": int(vals.nunique(dropna=True)),
            "n_tied_sessions": int(tied.sum()) if not tied.empty else 0,
            "max_tie_block_size": int(tied.max()) if not tied.empty else 0,
            "pct_sessions_in_ties": float(tied.sum() / n_all) if n_all else np.nan,
            "note": "Exact tie diagnostic for review ranking robustness; main outputs are not changed.",
        })
    audit = pd.DataFrame(rows)
    _write_csv(audit, eval_dir / "tie_audit.csv", TIE_AUDIT_COLS)
    return audit


def _top_minute_set(scores, rank_col, pct):
    rank = pd.to_numeric(scores.get(rank_col), errors="coerce")
    n = int(math.ceil(rank.notna().sum() * pct))
    n = max(n, 1) if rank.notna().any() else 0
    if n == 0:
        return set()
    work = scores.loc[rank.notna(), ["session_key", "minute_idx"]].copy()
    work["_rank"] = rank.loc[work.index]
    work = work.sort_values(["_rank", "session_key", "minute_idx"], ascending=[False, True, True]).head(n)
    return set(zip(work["session_key"].astype(str), pd.to_numeric(work["minute_idx"], errors="coerce").astype("Int64").astype(str)))


def _overlap_rate(left, right):
    return len(left & right) / len(left) if left else np.nan


def _session_rank(scores, rank_col):
    if scores.empty or "session_key" not in scores.columns:
        return pd.DataFrame(columns=["session_key", "session_rank_score", "session_rank"])
    work = scores[["session_key"]].copy()
    work["session_key"] = work["session_key"].astype(str)
    work["_rank"] = pd.to_numeric(scores.get(rank_col), errors="coerce")
    ranked = (
        work.groupby("session_key", as_index=False)
        .agg(session_rank_score=("_rank", "max"))
        .sort_values(["session_rank_score", "session_key"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    ranked["session_rank"] = np.arange(1, len(ranked) + 1, dtype=float)
    return ranked


def _session_overlap(base_rank, variant_rank, k):
    bset = set(base_rank.head(k)["session_key"].astype(str))
    vset = set(variant_rank.head(k)["session_key"].astype(str))
    return len(bset & vset) / len(bset) if bset else np.nan


def _idx_iou(a_start, a_end, b_start, b_end):
    vals = pd.to_numeric(pd.Series([a_start, a_end, b_start, b_end]), errors="coerce")
    if vals.isna().any():
        return np.nan
    a0, a1, b0, b1 = [int(v) for v in vals]
    if a1 < a0 or b1 < b0:
        return np.nan
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = max(a1, b1) - min(a0, b0) + 1
    return float(inter / union) if union > 0 else np.nan


def _scan_cfg_fast(cfg):
    out = dict(cfg or {})
    scan = dict(out.get("m2_scan", {}))
    scan["n_perm"] = 0
    out["m2_scan"] = scan
    return out


def _variant_scan(scores, out, cfg, eval_subdir, rank_col, sessions, merge_base_pred=True):
    eval_subdir = Path(eval_subdir)
    eval_subdir.mkdir(parents=True, exist_ok=True)
    if sessions:
        subset = scores.loc[scores["session_key"].astype(str).isin(set(map(str, sessions)))].copy()
    else:
        subset = scores.iloc[0:0].copy()
    return build_m2_scan(
        subset,
        out,
        _scan_cfg_fast(cfg),
        score_rank_col=rank_col,
        output_path=eval_subdir / "m2_scan.csv",
        merge_base_pred=merge_base_pred,
    )


def _interval_compare(base_scan, variant_scan, base_sessions):
    if base_scan.empty or variant_scan.empty:
        return np.nan, np.nan
    base = base_scan.copy()
    var = variant_scan.copy()
    base["session_key"] = base["session_key"].astype(str)
    var["session_key"] = var["session_key"].astype(str)
    keep = set(map(str, base_sessions))
    base = base.loc[base["session_key"].isin(keep)]
    merged = base.merge(
        var[["session_key", "top_interval_start_idx", "top_interval_end_idx", "top_interval_duration"]],
        on="session_key",
        how="left",
        suffixes=("_base", "_variant"),
    )
    if merged.empty:
        return np.nan, np.nan
    iou = merged.apply(
        lambda row: _idx_iou(
            row.get("top_interval_start_idx_base"),
            row.get("top_interval_end_idx_base"),
            row.get("top_interval_start_idx_variant"),
            row.get("top_interval_end_idx_variant"),
        ),
        axis=1,
    )
    shift = (
        pd.to_numeric(merged.get("top_interval_duration_variant"), errors="coerce")
        - pd.to_numeric(merged.get("top_interval_duration_base"), errors="coerce")
    ).abs()
    return (float(iou.median()) if iou.notna().any() else np.nan, float(shift.median()) if shift.notna().any() else np.nan)


def _signal_interpretation(overlap):
    if pd.isna(overlap):
        return "Insufficient minutes for signal sensitivity comparison."
    if overlap >= 0.8:
        return "Stable review ranking robustness: the minute score is not driven by this setting."
    if overlap < 0.5:
        return "Ranking sensitivity is visible; this signal or weight setting is important evidence."
    return "Moderate ranking sensitivity; inspect interval IoU before interpreting review changes."


def _rank_signal_columns(scores, signals):
    out = scores.copy()
    rank_cols = []
    for sig in signals:
        if sig not in out.columns:
            out[sig] = np.nan
        rcol = f"{sig}_rank_signal_eval"
        out[rcol] = _pct_rank(out[sig])
        rank_cols.append(rcol)
    return out, dict(zip(signals, rank_cols))


def run_minute_signal_sensitivity(out, eval_dir, cfg):
    eval_cfg = cfg.get("eval_robustness", {})
    minute_dir = Path(eval_dir) / "minute_signal"
    minute_dir.mkdir(parents=True, exist_ok=True)
    scores = _read_csv(Path(out) / "m2_scores.csv")
    base_scan = _read_csv(Path(out) / "m2_scan.csv")
    if scores.empty:
        _write_csv(pd.DataFrame(columns=SIGNAL_ABLATION_COLS), minute_dir / "signal_ablation_summary.csv", SIGNAL_ABLATION_COLS)
        _write_csv(pd.DataFrame(columns=SIGNAL_WEIGHT_COLS), minute_dir / "signal_weight_sensitivity_summary.csv", SIGNAL_WEIGHT_COLS)
        return pd.DataFrame(), pd.DataFrame()

    scores["session_key"] = scores["session_key"].astype(str)
    signals = _signals(eval_cfg)
    scores, rank_map = _rank_signal_columns(scores, signals)
    base_rank_col = "minute_mismatch_rank"
    base_min_top1 = _top_minute_set(scores, base_rank_col, 0.01)
    base_min_top5 = _top_minute_set(scores, base_rank_col, 0.05)
    base_sessions = _session_rank(scores, base_rank_col)
    base_top100_sessions = list(base_sessions.head(100)["session_key"].astype(str))

    ablation_rows = []
    for removed in signals:
        keep = [rank_map[s] for s in signals if s != removed]
        variant_scores = scores.copy()
        variant_scores["minute_mismatch_score_variant"] = variant_scores[keep].mean(axis=1, skipna=True) if keep else np.nan
        variant_scores["minute_mismatch_rank_variant"] = _pct_rank(variant_scores["minute_mismatch_score_variant"])
        variant_sessions = _session_rank(variant_scores, "minute_mismatch_rank_variant")
        scan_sessions = set(base_top100_sessions) | set(variant_sessions.head(100)["session_key"].astype(str))
        variant_scan = _variant_scan(
            variant_scores,
            out,
            cfg,
            minute_dir / f"signal_minus_{removed}",
            "minute_mismatch_rank_variant",
            scan_sessions,
        )
        med_iou, med_shift = _interval_compare(base_scan, variant_scan, base_top100_sessions)
        top100_overlap = _session_overlap(base_sessions, variant_sessions, 100)
        ablation_rows.append({
            "removed_signal": removed,
            "top1pct_minute_overlap": _overlap_rate(base_min_top1, _top_minute_set(variant_scores, "minute_mismatch_rank_variant", 0.01)),
            "top5pct_minute_overlap": _overlap_rate(base_min_top5, _top_minute_set(variant_scores, "minute_mismatch_rank_variant", 0.05)),
            "top50_session_overlap": _session_overlap(base_sessions, variant_sessions, 50),
            "top100_session_overlap": top100_overlap,
            "median_top_interval_iou_base_top100": med_iou,
            "median_duration_shift_base_top100": med_shift,
            "interpretation": _signal_interpretation(top100_overlap),
        })
    ablation = pd.DataFrame(ablation_rows)
    _write_csv(ablation, minute_dir / "signal_ablation_summary.csv", SIGNAL_ABLATION_COLS)

    weight_rows = []
    variants = eval_cfg.get("signal_weight_variants", {}) or {}
    for name, weights in variants.items():
        weights = weights or {}
        raw = {sig: float(weights.get(sig, 0) or 0) for sig in signals}
        total = sum(raw.values())
        norm = {sig: (raw[sig] / total if total > 0 else np.nan) for sig in signals}
        variant_scores = scores.copy()
        if total > 0:
            value = pd.Series(0.0, index=variant_scores.index)
            for sig in signals:
                value = value + variant_scores[rank_map[sig]].fillna(0.0) * norm[sig]
            variant_scores["minute_mismatch_score_variant"] = value
        else:
            variant_scores["minute_mismatch_score_variant"] = np.nan
        variant_scores["minute_mismatch_rank_variant"] = _pct_rank(variant_scores["minute_mismatch_score_variant"])
        variant_sessions = _session_rank(variant_scores, "minute_mismatch_rank_variant")
        scan_sessions = set(base_top100_sessions) | set(variant_sessions.head(100)["session_key"].astype(str))
        variant_scan = _variant_scan(
            variant_scores,
            out,
            cfg,
            minute_dir / f"weight_{name}",
            "minute_mismatch_rank_variant",
            scan_sessions,
        )
        med_iou, _ = _interval_compare(base_scan, variant_scan, base_top100_sessions)
        top100_overlap = _session_overlap(base_sessions, variant_sessions, 100)
        row = {
            "variant": name,
            "spearman_minute_score": float(pd.to_numeric(scores.get("minute_mismatch_score"), errors="coerce").corr(pd.to_numeric(variant_scores["minute_mismatch_score_variant"], errors="coerce"), method="spearman")),
            "top1pct_minute_overlap": _overlap_rate(base_min_top1, _top_minute_set(variant_scores, "minute_mismatch_rank_variant", 0.01)),
            "top5pct_minute_overlap": _overlap_rate(base_min_top5, _top_minute_set(variant_scores, "minute_mismatch_rank_variant", 0.05)),
            "top50_session_overlap": _session_overlap(base_sessions, variant_sessions, 50),
            "top100_session_overlap": top100_overlap,
            "median_top_interval_iou_base_top100": med_iou,
            "interpretation": _signal_interpretation(top100_overlap),
        }
        for sig in DEFAULT_SIGNALS:
            row[f"weight_{sig}"] = norm.get(sig, 0.0)
        weight_rows.append(row)
    weights = pd.DataFrame(weight_rows)
    _write_csv(weights, minute_dir / "signal_weight_sensitivity_summary.csv", SIGNAL_WEIGHT_COLS)
    return ablation, weights


def _average_percentile(reference, values):
    ref = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    if len(ref) == 0:
        return pd.Series(out, index=getattr(values, "index", None))
    ref = np.sort(ref)
    valid = np.isfinite(vals)
    left = np.searchsorted(ref, vals[valid], side="left") / len(ref)
    right = np.searchsorted(ref, vals[valid], side="right") / len(ref)
    out[valid] = 0.5 * (left + right)
    return pd.Series(out, index=getattr(values, "index", None))


def _rolling_by_session(df, col, window, reducer="mean"):
    parts = []
    for _, g in df.groupby("session_key", sort=False):
        s = pd.to_numeric(g[col], errors="coerce")
        r = s.rolling(window=window, min_periods=1)
        vals = r.sum() if reducer == "sum" else r.mean()
        parts.append(pd.Series(vals.to_numpy(), index=g.index))
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


def _zero_run_by_session(df):
    parts = []
    for _, g in df.groupby("session_key", sort=False):
        z = g["zero_chat"].fillna(False).astype(bool)
        block = z.ne(z.shift()).cumsum()
        parts.append(z.groupby(block).cumcount().add(1).where(z, 0).astype(int).rename(None).set_axis(g.index))
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


def _standardize_synthetic(minutes, intervals):
    minutes = minutes.copy()
    intervals = intervals.copy()
    if "synthetic_session_key" not in minutes.columns:
        if "session_key" in minutes.columns:
            minutes["synthetic_session_key"] = minutes["session_key"].astype(str)
        else:
            minutes["synthetic_session_key"] = minutes.get("run_id", "").astype(str) + "_" + minutes.get("broad_no", "").astype(str)
    minutes["session_key"] = minutes["synthetic_session_key"].astype(str)
    if "source_session_key" not in minutes.columns:
        minutes["source_session_key"] = np.nan
    if "injection_type" not in minutes.columns:
        minutes["injection_type"] = np.nan

    if not intervals.empty:
        if "synthetic_session_key" not in intervals.columns:
            intervals["synthetic_session_key"] = intervals.get("session_key", pd.Series(dtype=str)).astype(str)
        if "injected_start_idx" not in intervals.columns and "planned_start_idx" in intervals.columns:
            intervals["injected_start_idx"] = intervals["planned_start_idx"]
        if "injected_end_idx" not in intervals.columns and "planned_end_idx" in intervals.columns:
            intervals["injected_end_idx"] = intervals["planned_end_idx"]
        for col in ["source_session_key", "injection_type", "injected_start_idx", "injected_end_idx", "injected_start_ts", "injected_end_ts"]:
            if col not in intervals.columns:
                intervals[col] = np.nan
    return minutes, intervals


def _assign_real_viewer_bins(syn, real_scores):
    out = syn.copy()
    real_log = pd.to_numeric(real_scores.get("log_viewer"), errors="coerce").dropna()
    real_bins = pd.to_numeric(real_scores.get("viewer_bin"), errors="coerce")
    n_bins = int(real_bins.nunique(dropna=True)) if real_bins.notna().any() else 0
    if n_bins < 2 or real_log.nunique() < 2:
        out["viewer_bin"] = 0
        return out
    try:
        _, edges = pd.qcut(real_log, q=min(n_bins, real_log.nunique()), retbins=True, duplicates="drop")
        labels = list(range(len(edges) - 1))
        out["viewer_bin"] = pd.cut(pd.to_numeric(out["log_viewer"], errors="coerce"), bins=edges, labels=labels, include_lowest=True)
        out["viewer_bin"] = pd.to_numeric(out["viewer_bin"], errors="coerce").ffill().bfill().fillna(0).astype(int)
    except Exception:
        out["viewer_bin"] = 0
    return out


def _cluster_synthetic(syn, out, cfg):
    out_dir = Path(out)
    syn = syn.copy()
    status = "not_available"
    try:
        import joblib

        bundle_path = out_dir / "mc_kmeans.joblib"
        profile = _read_csv(out_dir / "mc_profile.csv")
        if not bundle_path.exists() or profile.empty:
            syn["minute_cluster"] = np.nan
            syn["cluster_mismatch_rank"] = np.nan
            return syn, status
        bundle = joblib.load(bundle_path)
        feats = [f for f in bundle.get("features", []) if f in syn.columns]
        if not feats:
            syn["minute_cluster"] = np.nan
            syn["cluster_mismatch_rank"] = np.nan
            return syn, "missing_features"
        x = syn[feats].replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.median(numeric_only=True)).fillna(0)
        xs = bundle["scaler"].transform(x)
        syn["minute_cluster"] = bundle["kmeans"].predict(xs)
        prof = profile[["cluster_id", "cluster_mismatch_rank"]].rename(columns={"cluster_id": "minute_cluster"})
        syn = syn.merge(prof, on="minute_cluster", how="left")
        status = "ok"
    except Exception as exc:
        syn["minute_cluster"] = np.nan
        syn["cluster_mismatch_rank"] = np.nan
        status = f"failed:{type(exc).__name__}"
    return syn, status


def _synthetic_features(minutes, real_scores, out, cfg):
    syn = minutes.copy()
    syn["minute_ts"] = pd.to_datetime(syn.get("minute_ts"), errors="coerce")
    for col in ["viewer_count_last", "chat_count", "unique_chatters"]:
        if col not in syn.columns:
            syn[col] = np.nan
        syn[col] = pd.to_numeric(syn[col], errors="coerce")
    syn = syn.sort_values(["session_key", "minute_ts"]).reset_index(drop=True)
    if "minute_idx" not in syn.columns:
        syn["minute_idx"] = syn.groupby("session_key").cumcount() + 1
    syn["log_viewer"] = np.log1p(syn["viewer_count_last"].clip(lower=0))
    syn["log_chat"] = np.log1p(syn["chat_count"].clip(lower=0))
    syn["log_unique"] = np.log1p(syn["unique_chatters"].clip(lower=0))
    syn["zero_chat"] = syn["chat_count"].eq(0)
    syn["zero_run_len"] = _zero_run_by_session(syn)
    syn["log_zero_run_len"] = np.log1p(pd.to_numeric(syn["zero_run_len"], errors="coerce").fillna(0))
    syn = _assign_real_viewer_bins(syn, real_scores)

    real_expected = real_scores.copy()
    real_expected["viewer_bin"] = pd.to_numeric(real_expected.get("viewer_bin"), errors="coerce")
    expected = real_expected.groupby("viewer_bin").agg(
        expected_log_chat_bin=("log_chat", "median"),
        expected_log_unique_bin=("log_unique", "median"),
    )
    global_chat = pd.to_numeric(real_scores.get("log_chat"), errors="coerce").median()
    global_unique = pd.to_numeric(real_scores.get("log_unique"), errors="coerce").median()
    syn["expected_log_chat_bin"] = pd.to_numeric(syn["viewer_bin"], errors="coerce").map(expected["expected_log_chat_bin"]).fillna(global_chat)
    syn["expected_log_unique_bin"] = pd.to_numeric(syn["viewer_bin"], errors="coerce").map(expected["expected_log_unique_bin"]).fillna(global_unique)
    syn["chat_deficit"] = syn["expected_log_chat_bin"] - syn["log_chat"]
    syn["unique_deficit"] = syn["expected_log_unique_bin"] - syn["log_unique"]
    syn["rolling_chat_deficit_5m"] = _rolling_by_session(syn, "chat_deficit", 5)
    syn["rolling_zero_rate_5m"] = _rolling_by_session(syn, "zero_chat", 5)
    syn, cluster_status = _cluster_synthetic(syn, out, cfg)
    return syn, cluster_status


def run_synthetic_interval_recovery(out, eval_dir, cfg):
    eval_cfg = cfg.get("eval_robustness", {})
    syn_cfg = eval_cfg.get("synthetic", {})
    syn_dir = Path(eval_dir) / "synthetic"
    syn_dir.mkdir(parents=True, exist_ok=True)
    keep_intermediate = bool(syn_cfg.get("keep_intermediate", False))
    minutes_path = _resolve_project_path(out, syn_cfg.get("external_minutes_csv", "data/synthetic/synthetic_injection_example_minutes.csv"))
    intervals_path = _resolve_project_path(out, syn_cfg.get("external_intervals_csv", "data/synthetic/synthetic_injection_example_intervals.csv"))
    thresholds = [float(x) for x in syn_cfg.get("iou_thresholds", [0.3, 0.5, 0.7])]
    real_scores = _read_csv(Path(out) / "m2_scores.csv")
    minutes = _read_csv(minutes_path)
    intervals = _read_csv(intervals_path)

    if minutes.empty or real_scores.empty:
        audit = pd.DataFrame([{
            "external_minutes_csv": str(minutes_path),
            "external_intervals_csv": str(intervals_path),
            "source_location_ok": "data/synthetic" in str(minutes_path).replace("\\", "/"),
            "n_synthetic_minutes": int(len(minutes)),
            "n_synthetic_sessions": 0,
            "n_synthetic_intervals": int(len(intervals)),
            "cluster_transform_status": "not_run",
            "note": "Synthetic mismatch recovery sanity check was not run because required inputs were missing.",
        }])
        _write_csv(audit, syn_dir / "synthetic_key_audit.csv")
        if keep_intermediate:
            _write_csv(pd.DataFrame(), syn_dir / "synthetic_m2_scores.csv")
            _write_csv(pd.DataFrame(), syn_dir / "synthetic_m2_scan.csv")
        else:
            _remove_if_exists(syn_dir / "synthetic_m2_scores.csv")
            _remove_if_exists(syn_dir / "synthetic_m2_scan.csv")
        _write_csv(pd.DataFrame(columns=SYNTHETIC_RECOVERY_COLS), syn_dir / "synthetic_interval_recovery.csv", SYNTHETIC_RECOVERY_COLS)
        _write_csv(pd.DataFrame(), syn_dir / "synthetic_recovery_by_scenario.csv")
        return pd.DataFrame(columns=SYNTHETIC_RECOVERY_COLS)

    minutes, intervals = _standardize_synthetic(minutes, intervals)
    syn_scores, cluster_status = _synthetic_features(minutes, real_scores, out, cfg)
    signals = _signals(eval_cfg)
    percentile_cols = []
    for sig in signals:
        if sig not in syn_scores.columns:
            syn_scores[sig] = np.nan
        pcol = f"{sig}_reference_percentile"
        syn_scores[pcol] = _average_percentile(real_scores.get(sig), syn_scores[sig])
        percentile_cols.append(pcol)
    syn_scores["minute_mismatch_score"] = syn_scores[percentile_cols].mean(axis=1, skipna=True)
    syn_scores["minute_mismatch_rank"] = _average_percentile(real_scores.get("minute_mismatch_score"), syn_scores["minute_mismatch_score"])
    syn_scores["dominant_reason"] = syn_scores[percentile_cols].idxmax(axis=1).str.replace("_reference_percentile", "", regex=False)

    audit = pd.DataFrame([{
        "external_minutes_csv": str(minutes_path),
        "external_intervals_csv": str(intervals_path),
        "source_location_ok": "data/synthetic" in str(minutes_path).replace("\\", "/") and "data/synthetic" in str(intervals_path).replace("\\", "/"),
        "n_synthetic_minutes": int(len(syn_scores)),
        "n_synthetic_sessions": int(syn_scores["session_key"].nunique()),
        "n_synthetic_intervals": int(len(intervals)),
        "cluster_transform_status": cluster_status,
        "note": "Synthetic rows are scored against real m2_scores distributions; they are not used for model fitting.",
    }])
    _write_csv(audit, syn_dir / "synthetic_key_audit.csv")
    if keep_intermediate:
        _write_csv(syn_scores, syn_dir / "synthetic_m2_scores.csv")
    else:
        _remove_if_exists(syn_dir / "synthetic_m2_scores.csv")

    scan_path = syn_dir / "synthetic_m2_scan.csv" if keep_intermediate else syn_dir / "_synthetic_m2_scan.tmp.csv"
    syn_scan = build_m2_scan(
        syn_scores,
        out,
        _scan_cfg_fast(cfg),
        score_rank_col="minute_mismatch_rank",
        output_path=scan_path,
        merge_base_pred=False,
    )
    if not keep_intermediate:
        _remove_if_exists(scan_path)
        _remove_if_exists(syn_dir / "synthetic_m2_scan.csv")

    rows = []
    if not intervals.empty:
        syn_scan = syn_scan.copy()
        syn_scan["synthetic_session_key"] = syn_scan["session_key"].astype(str)
        merged = intervals.merge(
            syn_scan[["synthetic_session_key", "top_interval_start_idx", "top_interval_end_idx"]],
            on="synthetic_session_key",
            how="left",
        )
        for _, row in merged.iterrows():
            iou = _idx_iou(row.get("injected_start_idx"), row.get("injected_end_idx"), row.get("top_interval_start_idx"), row.get("top_interval_end_idx"))
            out_row = {
                "synthetic_session_key": row.get("synthetic_session_key"),
                "source_session_key": row.get("source_session_key"),
                "injection_type": row.get("injection_type"),
                "injected_start_idx": row.get("injected_start_idx"),
                "injected_end_idx": row.get("injected_end_idx"),
                "detected_start_idx": row.get("top_interval_start_idx"),
                "detected_end_idx": row.get("top_interval_end_idx"),
                "iou": iou,
                "note": NOTE_NOT_REAL,
            }
            for threshold in thresholds:
                out_row[f"recovered_iou_{threshold:g}"] = bool(pd.notna(iou) and iou >= threshold)
            rows.append(out_row)
    recovery = pd.DataFrame(rows)
    for threshold in [0.3, 0.5, 0.7]:
        col = f"recovered_iou_{threshold:g}"
        if col not in recovery.columns:
            recovery[col] = False
    recovery = _write_csv(recovery, syn_dir / "synthetic_interval_recovery.csv", SYNTHETIC_RECOVERY_COLS)

    scenario_rows = []
    if not recovery.empty:
        recovery["is_positive_denominator"] = ~recovery["injection_type"].astype(str).str.contains("intermittent_zero_control", case=False, na=False)
        for scenario, part in recovery.groupby("injection_type", dropna=False):
            positive = part.loc[part["is_positive_denominator"]]
            row = {
                "injection_type": scenario,
                "n_examples": int(len(part)),
                "n_positive_denominator": int(len(positive)),
                "median_iou": float(pd.to_numeric(part["iou"], errors="coerce").median()) if part["iou"].notna().any() else np.nan,
                "note": "negative-control-like diagnostic" if int(len(positive)) == 0 else NOTE_NOT_REAL,
            }
            for threshold in thresholds:
                col = f"recovered_iou_{threshold:g}"
                row[f"recovery_rate_iou_{threshold:g}"] = float(positive[col].mean()) if len(positive) and col in positive.columns else np.nan
            scenario_rows.append(row)
    _write_csv(pd.DataFrame(scenario_rows), syn_dir / "synthetic_recovery_by_scenario.csv")
    return recovery


def _save_fig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _blank_plot(path, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, "No evaluation rows available", ha="center", va="center")
    ax.set_title(title)
    ax.axis("off")
    _save_fig(fig, path)


def write_eval_plots(out, eval_dir):
    plots = Path(out) / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    fam = _read_csv(Path(eval_dir) / "family_ablation_summary.csv")
    if fam.empty:
        _blank_plot(plots / "22_family_ablation_heatmap.png", "review ranking robustness - not ground-truth label")
    else:
        cols = [c for c in ["top20_overlap", "top50_overlap", "top100_overlap", "top200_overlap"] if c in fam.columns]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(fam) + 2)))
        vals = fam[cols].apply(lambda col: pd.to_numeric(col, errors="coerce")).to_numpy(dtype=float)
        im = ax.imshow(vals, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(np.arange(len(cols)), labels=cols, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(fam)), labels=fam["removed_family"].astype(str))
        ax.set_title("Family ablation heatmap - review ranking robustness")
        fig.colorbar(im, ax=ax, label="topK overlap")
        _save_fig(fig, plots / "22_family_ablation_heatmap.png")

    agg = _read_csv(Path(eval_dir) / "aggregation_sensitivity_summary.csv")
    if agg.empty:
        _blank_plot(plots / "23_aggregation_sensitivity.png", "review ranking robustness - not ground-truth label")
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        vals = pd.to_numeric(agg.get("top100_overlap"), errors="coerce")
        ax.barh(agg["variant"].astype(str), vals, color="tab:blue", edgecolor="black", alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("top100 overlap")
        ax.set_title("Aggregation sensitivity - review ranking robustness")
        ax.grid(axis="x", alpha=0.25)
        _save_fig(fig, plots / "23_aggregation_sensitivity.png")

    bal = _read_csv(Path(eval_dir) / "evidence_balance_summary.csv")
    if bal.empty:
        _blank_plot(plots / "24_evidence_balance.png", "review ranking robustness - not ground-truth label")
    else:
        fig, ax1 = plt.subplots(figsize=(9, 5.2))
        x = np.arange(len(bal))
        ax1.bar(x - 0.18, pd.to_numeric(bal.get("mean_strong_family_count"), errors="coerce"), width=0.36, label="mean strong family count", color="tab:green", edgecolor="black")
        ax2 = ax1.twinx()
        ax2.bar(x + 0.18, pd.to_numeric(bal.get("single_family_dominant_rate"), errors="coerce"), width=0.36, label="single family dominant rate", color="tab:orange", edgecolor="black", alpha=0.75)
        ax1.set_xticks(x, labels=bal["review_order_bucket"].astype(str))
        ax1.set_ylabel("strong family count")
        ax2.set_ylabel("dominant rate")
        ax1.set_title("Evidence balance - review ranking robustness")
        ax1.grid(axis="y", alpha=0.25)
        _save_fig(fig, plots / "24_evidence_balance.png")

    sig = _read_csv(Path(eval_dir) / "minute_signal" / "signal_ablation_summary.csv")
    weights = _read_csv(Path(eval_dir) / "minute_signal" / "signal_weight_sensitivity_summary.csv")
    if sig.empty and weights.empty:
        _blank_plot(plots / "25_minute_signal_sensitivity.png", "review ranking robustness - not ground-truth label")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
        if not sig.empty:
            ax[0].barh(sig["removed_signal"].astype(str), pd.to_numeric(sig.get("top100_session_overlap"), errors="coerce"), color="tab:purple", edgecolor="black", alpha=0.8)
            ax[0].set_title("Signal removal")
        if not weights.empty:
            ax[1].barh(weights["variant"].astype(str), pd.to_numeric(weights.get("top100_session_overlap"), errors="coerce"), color="tab:cyan", edgecolor="black", alpha=0.8)
            ax[1].set_title("Weight variants")
        for a in ax:
            a.set_xlim(0, 1)
            a.set_xlabel("top100 session overlap")
            a.grid(axis="x", alpha=0.25)
        fig.suptitle("Minute signal sensitivity - review ranking robustness", fontsize=13)
        _save_fig(fig, plots / "25_minute_signal_sensitivity.png")

    rec = _read_csv(Path(eval_dir) / "synthetic" / "synthetic_interval_recovery.csv")
    scen = _read_csv(Path(eval_dir) / "synthetic" / "synthetic_recovery_by_scenario.csv")
    rec = rec.copy()
    if not rec.empty:
        rec["iou_num"] = pd.to_numeric(rec.get("iou"), errors="coerce")
        rec = rec.dropna(subset=["iou_num"])
    if rec.empty:
        _blank_plot(plots / "26_synthetic_interval_recovery.png", "synthetic mismatch recovery, not real viewbot performance")
    else:
        # negative-control scenarios use the same flag as the scorecard (no positive denominator).
        control_types = set()
        if not scen.empty and "n_positive_denominator" in scen.columns:
            control_types = set(scen.loc[pd.to_numeric(scen["n_positive_denominator"], errors="coerce").fillna(1).eq(0), "injection_type"].astype(str))
        grouped = rec.groupby("injection_type")["iou_num"]
        order = grouped.median().sort_values().index.tolist()
        data = [grouped.get_group(t).to_numpy() for t in order]
        n = max(len(order), 1)
        fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * n + 1.2)))
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6, showfliers=False)
        for patch, t in zip(bp["boxes"], order):
            patch.set_facecolor("tab:gray" if t in control_types else "tab:red")
            patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_yticklabels([f"{t} (n={len(grouped.get_group(t))})" for t in order])
        for i, arr in enumerate(data, start=1):
            m = float(np.median(arr))
            ax.text(min(m + 0.02, 0.98), i, f"{m:.2f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("injected-interval localization IoU (per synthetic session)")
        ax.set_title("Synthetic mismatch recovery by scenario, not real viewbot performance")
        ax.grid(axis="x", alpha=0.25)
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor="tab:red", alpha=0.6, label="injected positive scenario"),
                Patch(facecolor="tab:gray", alpha=0.6, label="negative-control-like"),
            ],
            loc="lower right",
            fontsize=8,
        )
        _save_fig(fig, plots / "26_synthetic_interval_recovery.png")


def _fmt(value):
    if pd.isna(value):
        return "not available"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def write_evaluation_report(out, eval_dir, cfg):
    eval_cfg = cfg.get("eval_robustness", {}) if cfg else {}
    syn_cfg = eval_cfg.get("synthetic", {})
    topk = _topk(eval_cfg)
    families = _families(eval_cfg)
    signals = _signals(eval_cfg)
    fam = _read_csv(Path(eval_dir) / "family_ablation_summary.csv")
    agg = _read_csv(Path(eval_dir) / "aggregation_sensitivity_summary.csv")
    bal = _read_csv(Path(eval_dir) / "evidence_balance_summary.csv")
    tie = _read_csv(Path(eval_dir) / "tie_audit.csv")
    sig = _read_csv(Path(eval_dir) / "minute_signal" / "signal_ablation_summary.csv")
    weights = _read_csv(Path(eval_dir) / "minute_signal" / "signal_weight_sensitivity_summary.csv")
    rec = _read_csv(Path(eval_dir) / "synthetic" / "synthetic_interval_recovery.csv")
    profile = _read_csv(Path(eval_dir) / "top_session_profile.csv")
    scorecard = _read_csv(Path(eval_dir) / "eval_scorecard.csv")

    fam_min = pd.to_numeric(fam.get("top100_overlap"), errors="coerce").min() if not fam.empty else np.nan
    agg_min = pd.to_numeric(agg.get("top100_overlap"), errors="coerce").min() if not agg.empty else np.nan
    sig_min = pd.to_numeric(sig.get("top100_session_overlap"), errors="coerce").min() if not sig.empty else np.nan
    weight_min = pd.to_numeric(weights.get("top100_session_overlap"), errors="coerce").min() if not weights.empty else np.nan
    rec_med = pd.to_numeric(rec.get("iou"), errors="coerce").median() if not rec.empty else np.nan

    top_feature = "not available"
    if not profile.empty and "std_mean_diff" in profile.columns:
        prof = profile.copy()
        prof["_abs_d"] = pd.to_numeric(prof["std_mean_diff"], errors="coerce").abs()
        prof = prof.dropna(subset=["_abs_d"]).sort_values("_abs_d", ascending=False)
        if not prof.empty:
            row = prof.iloc[0]
            top_feature = f"{row['feature']}({row['direction']}, d={_fmt(row['std_mean_diff'])})"

    def _score_value(metric):
        if scorecard.empty or "metric" not in scorecard.columns:
            return np.nan
        hit = scorecard.loc[scorecard["metric"].astype(str).eq(metric), "value"]
        return pd.to_numeric(hit, errors="coerce").iloc[0] if not hit.empty else np.nan

    lines = [
        "# 평가 안정성 리포트",
        "",
        REPORT_REQUIRED_SENTENCE,
        "",
        "## 1. 감독학습 성능지표를 쓰지 않는 이유",
        "현재 프로젝트에는 실제 최종 판정 라벨이 없으므로 accuracy, precision, recall, ROC-AUC, PR-AUC를 실제 성능처럼 계산하지 않는다.",
        "이 폴더의 산출물은 label-free review ranking의 안정성과 synthetic mismatch 회수 여부를 확인하는 진단 자료다.",
        "요약 문구의 not real viewbot performance는 synthetic 결과가 실제 viewbot 성능평가가 아니라는 제한을 명시하기 위한 표현이다.",
        "",
        "## 2. 설정 스냅샷",
        f"- topk 기준: {topk}",
        f"- family 목록: {', '.join(map(str, families))}",
        f"- minute signal 목록: {', '.join(map(str, signals))}",
        f"- synthetic 입력: minutes={syn_cfg.get('external_minutes_csv')}, intervals={syn_cfg.get('external_intervals_csv')}",
        f"- synthetic 중간 CSV 보존 여부: {bool(syn_cfg.get('keep_intermediate', False))}",
        "",
        "## 3. Family 제거 민감도",
        f"각 family를 하나씩 제거한 뒤 남은 family strength로 후보 순서를 다시 계산한다. 최소 top100 overlap은 {_fmt(fam_min)}이다.",
        "overlap이 높으면 최종 review_order가 특정 family 하나에만 임의로 의존하지 않는다는 근거가 된다. 낮은 값은 핵심 근거 family를 식별하는 민감도 신호로만 해석한다.",
        "",
        "## 4. 집계 방식 민감도",
        f"consensus-only, RRA-only, consensus-plus-RRA, median, trimmed mean, family-exclusion 변형을 비교한다. 최소 top100 overlap은 {_fmt(agg_min)}이다.",
        "RRA는 순위 집계 근거이며, 주 해석은 consensus-first이고 RRA는 보조 근거로 사용한다.",
        "",
        "## 5. 근거 균형과 동점 점검",
        f"Evidence balance는 review_order 구간별로 요약한다. tie audit 행 수는 {len(tie)}개다.",
        "상위 구간에서 strong-family 수가 높으면 여러 근거 family가 함께 지지한다는 의미다. 단일 family 지배가 있으면 제한사항으로 보고한다.",
        "",
        "## 6. Minute signal 민감도",
        f"signal 제거 실험의 최소 top100 session overlap은 {_fmt(sig_min)}이고, weight 변형의 최소 top100 session overlap은 {_fmt(weight_min)}이다.",
        "equal weight는 라벨 없는 상태에서 학습 가중치를 임의로 만들지 않기 위한 보수적 설계다. overlap 안정성은 minute score가 하나의 signal 또는 하나의 가중치 설정에 과도하게 의존하지 않는지 확인한다.",
        "",
        "## 7. Synthetic mismatch interval 회수",
        f"Synthetic interval recovery의 median IoU는 {_fmt(rec_med)}이다. 이 값은 synthetic mismatch recovery이며 not real viewbot performance이다.",
        "연속 mismatch scenario는 interval localization sanity check로 해석한다. intermittent zero control은 별도 negative-control-like diagnostic으로 보고한다.",
        "",
        "## 8. 해석 제한",
        "이 진단은 supervised class correctness를 증명하지 않는다. evidence-family 제거, 집계 방식 변경, minute-signal 제거, synthetic interval localization에 대해 review 우선순위가 얼마나 안정적인지 확인한다.",
        "",
        "## 9. 권장 해석",
        "최종 review_order는 label-free mismatch pipeline이 만든 수동 검토 우선순위로 사용한다. 이 eval 폴더는 robustness 근거와 민감도 한계를 기록하는 appendix로 해석한다.",
        "",
        "## 10. 상위 후보 vs 나머지 프로파일 대비",
        f"top_session_profile.csv는 상위 review 세션과 나머지 세션의 minute-signal 프로파일을 표준화 평균차(Cohen d)로 비교한다. 표준화 평균차 절댓값이 가장 큰 신호는 {top_feature}이다.",
        "이는 상위 후보가 어떤 신호 때문에 위로 올라갔는지 설명하는 자료이며 확률이나 판정 라벨이 아니다.",
        "",
        "## 11. label-free 평가 스코어카드",
        f"eval_scorecard.csv는 robustness 최소 overlap과 synthetic positive/negative-control localization을 한 표로 모은다. synthetic positive median IoU는 {_fmt(_score_value('synthetic_positive_median_iou'))}, negative-control median IoU는 {_fmt(_score_value('synthetic_negative_control_median_iou'))}이다.",
        "모든 값은 robustness 또는 synthetic sanity 진단이며 supervised 성능지표가 아니다.",
    ]
    _write_text(Path(eval_dir) / "evaluation_report.md", "\n".join(lines))


def _std_mean_diff(top_vals, rest_vals):
    """Standardized mean difference (Cohen d, pooled SD) of top vs rest sessions."""
    t = pd.to_numeric(pd.Series(top_vals), errors="coerce").dropna().to_numpy(dtype=float)
    r = pd.to_numeric(pd.Series(rest_vals), errors="coerce").dropna().to_numpy(dtype=float)
    if len(t) < 2 or len(r) < 2:
        return np.nan
    sp = math.sqrt(((len(t) - 1) * np.var(t, ddof=1) + (len(r) - 1) * np.var(r, ddof=1)) / (len(t) + len(r) - 2))
    if sp <= 0:
        return np.nan
    return float((np.mean(t) - np.mean(r)) / sp)


def run_top_session_profile(out, eval_dir, cfg):
    """Contrast minute-signal profiles of top review sessions vs the rest, so the
    final ranking is explainable (which signals push a session to the top)."""
    out = Path(out)
    eval_dir = Path(eval_dir)
    eval_cfg = cfg.get("eval_robustness", {})
    review_all, review, families, _ = _review_frames(out, eval_cfg)
    base = _base_order(review, review_all)
    scores = _read_csv(out / "m2_scores.csv")
    if base.empty or scores.empty or "session_key" not in scores.columns:
        _write_csv(pd.DataFrame(columns=PROFILE_COLS), eval_dir / "top_session_profile.csv", PROFILE_COLS)
        return pd.DataFrame(columns=PROFILE_COLS)

    scores = scores.copy()
    scores["session_key"] = scores["session_key"].astype(str)
    agg_map = {col: how for col, how in PROFILE_FEATURES if col in scores.columns}
    if not agg_map:
        _write_csv(pd.DataFrame(columns=PROFILE_COLS), eval_dir / "top_session_profile.csv", PROFILE_COLS)
        return pd.DataFrame(columns=PROFILE_COLS)

    per_session = scores.groupby("session_key").agg(agg_map).reset_index()
    base = base.copy()
    base["session_key"] = base["session_key"].astype(str)
    base["review_order"] = pd.to_numeric(base["review_order"], errors="coerce")
    merged = base.merge(per_session, on="session_key", how="left")

    topk_list = _topk(eval_cfg)
    default_k = topk_list[1] if len(topk_list) > 1 else (topk_list[0] if topk_list else 50)
    k = int(eval_cfg.get("profile_top_k", default_k))
    top_mask = merged["review_order"].le(k)
    top = merged.loc[top_mask]
    rest = merged.loc[~top_mask]

    rows = []
    for col, how in agg_map.items():
        top_col = pd.to_numeric(top[col], errors="coerce")
        rest_col = pd.to_numeric(rest[col], errors="coerce")
        top_mean = float(top_col.mean()) if top_col.notna().any() else np.nan
        rest_mean = float(rest_col.mean()) if rest_col.notna().any() else np.nan
        direction = "higher_in_top" if (pd.notna(top_mean) and pd.notna(rest_mean) and top_mean >= rest_mean) else "lower_in_top"
        rows.append({
            "feature": f"{col}_{how}",
            "top_n": int(top_col.notna().sum()),
            "rest_n": int(rest_col.notna().sum()),
            "top_median": float(top_col.median()) if top_col.notna().any() else np.nan,
            "rest_median": float(rest_col.median()) if rest_col.notna().any() else np.nan,
            "top_mean": top_mean,
            "rest_mean": rest_mean,
            "std_mean_diff": _std_mean_diff(top[col], rest[col]),
            "direction": direction,
            "note": f"top=review_order<={k}; standardized mean difference is review evidence, not probability",
        })
    profile = _write_csv(pd.DataFrame(rows), eval_dir / "top_session_profile.csv", PROFILE_COLS)

    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    pdat = profile.loc[pd.to_numeric(profile["std_mean_diff"], errors="coerce").notna()].copy()
    if pdat.empty:
        _blank_plot(plots / "28_top_session_profile.png", "top vs rest session profile - review evidence, not label")
    else:
        pdat["d"] = pd.to_numeric(pdat["std_mean_diff"], errors="coerce")
        pdat = pdat.sort_values("d")
        colors = ["tab:red" if v >= 0 else "tab:blue" for v in pdat["d"]]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.barh(pdat["feature"].astype(str), pdat["d"], color=colors, edgecolor="black", alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"standardized mean difference (top{k} minus rest)")
        ax.set_title(f"Top{k} vs rest session profile - review evidence, not ground-truth label")
        ax.grid(axis="x", alpha=0.25)
        _save_fig(fig, plots / "28_top_session_profile.png")
    return profile


def write_eval_scorecard(out, eval_dir, cfg):
    """One-page roll-up of label-free diagnostics: ranking robustness, synthetic
    positive vs negative-control localization, and the top expected-response feature."""
    out = Path(out)
    eval_dir = Path(eval_dir)
    fam = _read_csv(eval_dir / "family_ablation_summary.csv")
    agg = _read_csv(eval_dir / "aggregation_sensitivity_summary.csv")
    sig = _read_csv(eval_dir / "minute_signal" / "signal_ablation_summary.csv")
    weights = _read_csv(eval_dir / "minute_signal" / "signal_weight_sensitivity_summary.csv")
    scenario = _read_csv(eval_dir / "synthetic" / "synthetic_recovery_by_scenario.csv")
    importance = _read_csv(out / "base_importance.csv")
    fit = _read_csv(out / "base_fit.csv")

    def _min(df, col):
        return float(pd.to_numeric(df.get(col), errors="coerce").min()) if not df.empty else np.nan

    rows = [
        {"metric": "family_ablation_min_top100_overlap", "value": _min(fam, "top100_overlap"), "scale": "0-1 higher is more robust", "what_it_checks": "review_order가 단일 evidence family에 의존하지 않는가", "note": "ranking robustness diagnostic, not accuracy"},
        {"metric": "aggregation_min_top100_overlap", "value": _min(agg, "top100_overlap"), "scale": "0-1 higher is more robust", "what_it_checks": "집계 방식 변경에 review 순서가 안정적인가", "note": "ranking robustness diagnostic, not accuracy"},
        {"metric": "signal_ablation_min_top100_session_overlap", "value": _min(sig, "top100_session_overlap"), "scale": "0-1 higher is more robust", "what_it_checks": "minute signal 하나 제거에 안정적인가", "note": "ranking robustness diagnostic, not accuracy"},
        {"metric": "signal_weight_min_top100_session_overlap", "value": _min(weights, "top100_session_overlap"), "scale": "0-1 higher is more robust", "what_it_checks": "신호 가중치 변형에 안정적인가", "note": "ranking robustness diagnostic, not accuracy"},
    ]

    pos_iou = neg_iou = np.nan
    if not scenario.empty:
        is_control = scenario["injection_type"].astype(str).str.contains("intermittent_zero_control", case=False, na=False)
        positive = scenario.loc[(pd.to_numeric(scenario.get("n_positive_denominator"), errors="coerce").fillna(0) > 0) & ~is_control]
        control = scenario.loc[is_control]
        pos_iou = float(pd.to_numeric(positive.get("median_iou"), errors="coerce").median()) if not positive.empty else np.nan
        neg_iou = float(pd.to_numeric(control.get("median_iou"), errors="coerce").median()) if not control.empty else np.nan
    rows.append({"metric": "synthetic_positive_median_iou", "value": pos_iou, "scale": "0-1 higher localizes injected interval", "what_it_checks": "주입한 연속 mismatch 구간을 회수하는가", "note": NOTE_NOT_REAL})
    rows.append({"metric": "synthetic_negative_control_median_iou", "value": neg_iou, "scale": "0-1 lower means fewer false intervals", "what_it_checks": "간헐적 zero 대조군에서 거짓 구간을 덜 잡는가", "note": "negative-control-like diagnostic"})

    if not fit.empty and "target" in fit.columns:
        for tgt in ["log_chat", "log_unique"]:
            hit = fit.loc[fit["target"].astype(str).eq(tgt)]
            if not hit.empty:
                rows.append({"metric": f"expected_response_{tgt}_mae_skill_vs_median", "value": float(pd.to_numeric(hit["mae_skill_vs_median"], errors="coerce").iloc[0]), "scale": "0-1 higher beats unconditional median", "what_it_checks": f"{tgt} 조건부 baseline의 held-out 오차가 무조건부 median보다 얼마나 낮은가", "note": "OOF fit skill, not a classifier"})

    if not importance.empty and "target" in importance.columns:
        chat_imp = importance.loc[importance["target"].astype(str).eq("log_chat")].copy()
        chat_imp = chat_imp.sort_values("importance_mae_increase", ascending=False)
        if not chat_imp.empty:
            rows.append({"metric": "expected_response_top_feature_log_chat", "value": str(chat_imp.iloc[0]["feature"]), "scale": "feature name", "what_it_checks": "조건부 기대 chat 추정에서 기여가 가장 큰 변수", "note": "OOF permutation importance, not probability"})

    scorecard = _write_csv(pd.DataFrame(rows), eval_dir / "eval_scorecard.csv", SCORECARD_COLS)

    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    numeric = scorecard.copy()
    numeric["value_num"] = pd.to_numeric(numeric["value"], errors="coerce")
    numeric = numeric.loc[numeric["value_num"].notna()]
    if numeric.empty:
        _blank_plot(plots / "27_eval_scorecard.png", "label-free evaluation scorecard - not a supervised score")
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.barh(numeric["metric"].astype(str), numeric["value_num"], color="tab:purple", edgecolor="black", alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("score (0-1)")
        ax.set_title("Label-free evaluation scorecard - robustness and synthetic sanity, not a supervised score")
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
        _save_fig(fig, plots / "27_eval_scorecard.png")
    return scorecard


def run_eval_robustness(out, cfg):
    out = Path(out)
    cfg = cfg or {}
    eval_cfg = cfg.get("eval_robustness", {})
    eval_dir = _resolve_child_path(out, eval_cfg.get("out_dir"), "eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    if eval_cfg.get("run_family_ablation", True):
        run_family_ablation(out, eval_dir, cfg)
    if eval_cfg.get("run_aggregation_sensitivity", True):
        run_aggregation_sensitivity(out, eval_dir, cfg)
    if eval_cfg.get("run_evidence_balance", True):
        run_evidence_balance(out, eval_dir, cfg)
    if eval_cfg.get("run_tie_audit", True):
        run_tie_audit(out, eval_dir, cfg)
    if eval_cfg.get("run_minute_signal_sensitivity", True):
        run_minute_signal_sensitivity(out, eval_dir, cfg)
    if eval_cfg.get("run_synthetic_interval_recovery", True):
        run_synthetic_interval_recovery(out, eval_dir, cfg)
    if eval_cfg.get("run_top_session_profile", True):
        run_top_session_profile(out, eval_dir, cfg)
    write_eval_scorecard(out, eval_dir, cfg)
    write_eval_plots(out, eval_dir)
    write_evaluation_report(out, eval_dir, cfg)
    return {"eval_dir": str(eval_dir)}
