from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

M2_PATH = ROOT / "m2_review.csv"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    group_col: str
    low_priority_value: str


MODEL_SPECS = [
    ModelSpec("03", ROOT / "outputs" / "03_review_candidates.csv", "review_group", "low_priority"),
    ModelSpec("07a", ROOT / "outputs" / "07a_cluster_aware_review_candidates.csv", "cluster_aware_review_group", "low_priority"),
    ModelSpec("07b", ROOT / "outputs" / "07b_advanced_binary_review_candidates.csv", "advanced_binary_review_group", "low_priority"),
    ModelSpec("07c", ROOT / "outputs" / "07c_hidden_candidate_review_candidates.csv", "discovery_group", "normal_low_priority"),
    ModelSpec("07d", ROOT / "outputs" / "07d_pu_hidden_review_candidates.csv", "pu_discovery_group", "pu_likely_normal"),
]

TOP_KS = [50, 100, 200, 500]


def load_m2() -> tuple[pd.DataFrame, pd.Series]:
    m2 = pd.read_csv(M2_PATH).copy()
    m2 = m2.sort_values("review_order", kind="stable").reset_index(drop=True)
    m2["minute_rank"] = range(1, len(m2) + 1)
    minute_rank = m2.set_index("session_key")["minute_rank"]
    return m2, minute_rank


def load_model(spec: ModelSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.path).copy()
    df["model_rank"] = range(1, len(df) + 1)
    df["flagged_by_model"] = df[spec.group_col].astype("string").ne(spec.low_priority_value)
    df["model_name"] = spec.name
    return df


def ranking_summary(m2: pd.DataFrame, model_df: pd.DataFrame) -> dict[str, float]:
    merged = m2[["session_key", "minute_rank"]].merge(
        model_df[["session_key", "model_rank"]],
        on="session_key",
        how="inner",
    )
    row: dict[str, float] = {
        "model": model_df["model_name"].iat[0],
        "spearman_rank_corr": float(merged["minute_rank"].corr(merged["model_rank"], method="spearman")),
        "mean_abs_rank_gap": float((merged["minute_rank"] - merged["model_rank"]).abs().mean()),
        "median_abs_rank_gap": float((merged["minute_rank"] - merged["model_rank"]).abs().median()),
    }
    for k in TOP_KS:
        minute_top = set(m2.nsmallest(k, "minute_rank")["session_key"])
        model_top = set(model_df.nsmallest(k, "model_rank")["session_key"])
        inter = len(minute_top & model_top)
        union = len(minute_top | model_top)
        row[f"top{k}_overlap"] = inter
        row[f"top{k}_jaccard"] = inter / union if union else 0.0
        row[f"top{k}_minute_recall"] = inter / k if k else 0.0
    return row


def flagged_summary(m2: pd.DataFrame, minute_rank: pd.Series, model_df: pd.DataFrame, spec: ModelSpec) -> dict[str, float]:
    flagged = model_df.loc[model_df["flagged_by_model"]].copy()
    flagged_n = int(len(flagged))
    minute_top_same_n = set(m2.nsmallest(flagged_n, "minute_rank")["session_key"])
    overlap = int(flagged["session_key"].isin(minute_top_same_n).sum())
    row = {
        "model": spec.name,
        "group_col": spec.group_col,
        "flagged_n": flagged_n,
        "overlap_with_minute_top_same_n": overlap,
        "precision_vs_minute_top_same_n": overlap / flagged_n if flagged_n else 0.0,
        "mean_minute_rank_of_flagged": float(flagged["session_key"].map(minute_rank).mean()),
        "median_minute_rank_of_flagged": float(flagged["session_key"].map(minute_rank).median()),
        "flagged_outside_minute_top_same_n": flagged_n - overlap,
    }
    return row


def mismatch_table(m2: pd.DataFrame, minute_rank: pd.Series, model_df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    flagged = model_df.loc[model_df["flagged_by_model"]].copy()
    flagged["minute_rank"] = flagged["session_key"].map(minute_rank)
    flagged_n = len(flagged)
    flagged["minute_top_same_n"] = flagged["minute_rank"].le(flagged_n)
    flagged["rank_gap"] = flagged["minute_rank"] - flagged["model_rank"]
    keep_cols = [
        "session_key",
        "run_id",
        "broad_no",
        spec.group_col,
        "model_rank",
        "minute_rank",
        "rank_gap",
        "minute_top_same_n",
    ]
    optional_cols = [
        "review_group",
        "cluster_aware_review_group",
        "advanced_binary_review_group",
        "discovery_group",
        "hidden_candidate_priority",
        "pu_discovery_group",
        "pu_priority",
        "score_consensus",
        "score_cluster_aware_consensus",
        "score_advanced_binary_consensus",
        "unsupervised_hidden_score",
        "pu_positive_score",
    ]
    keep_cols.extend([col for col in optional_cols if col in flagged.columns and col not in keep_cols])
    keep_cols.extend([col for col in ["family_consensus_score", "review_order"] if col in flagged.columns and col not in keep_cols])
    out = flagged[keep_cols].sort_values(
        ["minute_top_same_n", "minute_rank", "model_rank"],
        ascending=[True, False, True],
        kind="stable",
    )
    out.insert(0, "model", spec.name)
    return out


def minute_miss_table(m2: pd.DataFrame, model_df: pd.DataFrame, spec: ModelSpec, top_n: int = 100) -> pd.DataFrame:
    minute_top = m2.nsmallest(top_n, "minute_rank")[["session_key", "run_id", "broad_no", "minute_rank", "family_consensus_score", "reason_set"]].copy()
    flagged_keys = set(model_df.loc[model_df["flagged_by_model"], "session_key"])
    minute_top["flagged_by_model"] = minute_top["session_key"].isin(flagged_keys)
    missed = minute_top.loc[~minute_top["flagged_by_model"]].copy()
    missed.insert(0, "model", spec.name)
    return missed


def main() -> None:
    m2, minute_rank = load_m2()
    ranking_rows = []
    flagged_rows = []
    mismatch_frames = []
    missed_frames = []

    for spec in MODEL_SPECS:
        model_df = load_model(spec)
        ranking_rows.append(ranking_summary(m2, model_df))
        flagged_rows.append(flagged_summary(m2, minute_rank, model_df, spec))
        mismatch_frames.append(mismatch_table(m2, minute_rank, model_df, spec).head(50))
        missed_frames.append(minute_miss_table(m2, model_df, spec, top_n=100))

    ranking_df = pd.DataFrame(ranking_rows).sort_values(["top100_jaccard", "spearman_rank_corr"], ascending=[True, True])
    flagged_df = pd.DataFrame(flagged_rows).sort_values(
        ["precision_vs_minute_top_same_n", "mean_minute_rank_of_flagged"],
        ascending=[True, False],
    )
    mismatch_df = pd.concat(mismatch_frames, ignore_index=True)
    missed_df = pd.concat(missed_frames, ignore_index=True)

    ranking_df.to_csv(OUT_DIR / "minute_vs_session_ranking_summary.csv", index=False, encoding="utf-8-sig")
    flagged_df.to_csv(OUT_DIR / "minute_vs_session_flagged_set_summary.csv", index=False, encoding="utf-8-sig")
    mismatch_df.to_csv(OUT_DIR / "minute_vs_session_flagged_mismatch_examples.csv", index=False, encoding="utf-8-sig")
    missed_df.to_csv(OUT_DIR / "minute_top100_missed_by_session_models.csv", index=False, encoding="utf-8-sig")

    print("Saved:")
    print(OUT_DIR / "minute_vs_session_ranking_summary.csv")
    print(OUT_DIR / "minute_vs_session_flagged_set_summary.csv")
    print(OUT_DIR / "minute_vs_session_flagged_mismatch_examples.csv")
    print(OUT_DIR / "minute_top100_missed_by_session_models.csv")
    print()
    print("Flagged-set summary")
    print(flagged_df.to_string(index=False))


if __name__ == "__main__":
    main()
