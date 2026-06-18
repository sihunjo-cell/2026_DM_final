from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from session_common import add_rule_scores, find_project_root, pct_rank
from session_hidden_candidate_discovery import build_normal_seed


RANDOM_STATE = 42


def make_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "pu_logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "pu_random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "pu_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=120,
                        learning_rate=0.05,
                        max_depth=2,
                        subsample=0.85,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def make_reliable_negative_score(session: pd.DataFrame) -> pd.Series:
    """Higher means more likely to be normal/reliable negative."""
    normal_components = pd.concat(
        [
            pct_rank(-session["score_rule_directional"], True),
            pct_rank(-session["zero_rate"], True),
            pct_rank(-session["gap_med"], True),
            pct_rank(session["chat_mean"], True),
            pct_rank(session["unique_mean"], True),
            pct_rank(-session["zrun_max"], True),
        ],
        axis=1,
    )
    return normal_components.mean(axis=1)


def pu_bagging_scores(
    session: pd.DataFrame,
    X: pd.DataFrame,
    positive_mask: pd.Series,
    reliable_negative_mask: pd.Series,
    n_rounds: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    positive_idx = np.flatnonzero(positive_mask.to_numpy())
    unlabeled_idx = np.flatnonzero(~positive_mask.to_numpy())
    reliable_negative_idx = np.flatnonzero(reliable_negative_mask.to_numpy())
    if len(reliable_negative_idx) < len(positive_idx):
        reliable_negative_idx = unlabeled_idx

    score_frames = []
    diagnostics = []
    sample_size = min(len(reliable_negative_idx), max(len(positive_idx) * 3, len(positive_idx)))

    for round_id in range(n_rounds):
        sampled_neg = rng.choice(reliable_negative_idx, size=sample_size, replace=False)
        train_idx = np.concatenate([positive_idx, sampled_neg])
        y_train = np.concatenate([np.ones(len(positive_idx), dtype=int), np.zeros(len(sampled_neg), dtype=int)])

        for model_name, model in make_models(RANDOM_STATE + round_id).items():
            model.fit(X.iloc[train_idx], y_train)
            score = model.predict_proba(X)[:, 1]
            score_frames.append(
                pd.DataFrame(
                    {
                        "session_key": session["session_key"].to_numpy(),
                        "round_id": round_id,
                        "model": model_name,
                        "pu_score": score,
                    }
                )
            )
            diagnostics.append(
                {
                    "round_id": round_id,
                    "model": model_name,
                    "n_positive": len(positive_idx),
                    "n_sampled_reliable_negative": len(sampled_neg),
                    "positive_score_mean": float(score[positive_idx].mean()),
                    "unlabeled_score_mean": float(score[unlabeled_idx].mean()),
                    "reliable_negative_score_mean": float(score[sampled_neg].mean()),
                }
            )

    long_scores = pd.concat(score_frames, ignore_index=True)
    diagnostics_df = pd.DataFrame(diagnostics)
    return long_scores, diagnostics_df


def add_reason_codes(row: pd.Series) -> str:
    reasons = []
    if row["pu_positive_score"] >= 0.80:
        reasons.append("high_pu_similarity")
    if row["pu_score_stability"] >= 0.70:
        reasons.append("stable_across_bags")
    if row["score_rule_directional"] >= row["_rule_q90"]:
        reasons.append("high_rule_mismatch")
    if row["zero_rate"] >= row["_zero_q90"]:
        reasons.append("very_high_zero_rate")
    if row["gap_med"] >= row["_gap_q90"]:
        reasons.append("very_high_gap")
    if row["chat_mean"] <= row["_chat_q10"]:
        reasons.append("very_low_chat")
    return "|".join(reasons) if reasons else "none"


def main() -> None:
    root = find_project_root()
    output_dir = root / "outputs"

    session = pd.read_csv(root / "csv" / "session_summary_processed.csv")
    session["known_positive"] = session["cluster_number"].eq(1)
    session["silver_suspicious"] = session["known_positive"].astype(int)
    session = add_rule_scores(session)

    feature_cols = [
        "n",
        "viewer_med",
        "viewer_max",
        "chat_mean",
        "unique_mean",
        "zero_rate",
        "zrun_max",
        "gap_med",
        "gap_max",
        "log_viewer",
        "log_chat",
        "log_unique",
        "log_zrun",
        "score_rule_directional",
    ]
    X = session[feature_cols].copy()

    session["normal_seed"] = build_normal_seed(session)
    session["reliable_negative_score"] = make_reliable_negative_score(session)
    reliable_negative_threshold = session.loc[~session["known_positive"], "reliable_negative_score"].quantile(0.75)
    session["reliable_negative_seed"] = (
        ~session["known_positive"]
        & session["normal_seed"]
        & session["reliable_negative_score"].ge(reliable_negative_threshold)
    )

    long_scores, diagnostics = pu_bagging_scores(
        session=session,
        X=X,
        positive_mask=session["known_positive"],
        reliable_negative_mask=session["reliable_negative_seed"],
        n_rounds=30,
    )

    score_summary = (
        long_scores.groupby("session_key")
        .agg(
            pu_positive_score=("pu_score", "mean"),
            pu_score_std=("pu_score", "std"),
            pu_score_min=("pu_score", "min"),
            pu_score_max=("pu_score", "max"),
            pu_score_stability=("pu_score", lambda s: float((s >= 0.50).mean())),
        )
        .reset_index()
    )
    session = session.merge(score_summary, on="session_key", how="left")

    hidden_threshold = session.loc[~session["known_positive"], "pu_positive_score"].quantile(0.95)
    watch_threshold = session.loc[~session["known_positive"], "pu_positive_score"].quantile(0.90)
    stable_threshold = 0.60

    session["pu_discovery_group"] = "pu_likely_normal"
    session.loc[session["known_positive"], "pu_discovery_group"] = "known_positive"
    session.loc[
        ~session["known_positive"]
        & session["pu_positive_score"].ge(watch_threshold)
        & session["pu_score_stability"].ge(stable_threshold),
        "pu_discovery_group",
    ] = "pu_watchlist"
    session.loc[
        ~session["known_positive"]
        & session["pu_positive_score"].ge(hidden_threshold)
        & session["pu_score_stability"].ge(stable_threshold),
        "pu_discovery_group",
    ] = "pu_hidden_candidate"

    session["pu_priority"] = "low"
    session.loc[session["pu_discovery_group"].eq("pu_watchlist"), "pu_priority"] = "medium"
    session.loc[session["pu_discovery_group"].eq("pu_hidden_candidate"), "pu_priority"] = "high"
    session.loc[session["pu_discovery_group"].eq("known_positive"), "pu_priority"] = "known_positive"

    session["_rule_q90"] = session["score_rule_directional"].quantile(0.90)
    session["_zero_q90"] = session["zero_rate"].quantile(0.90)
    session["_gap_q90"] = session["gap_med"].quantile(0.90)
    session["_chat_q10"] = session["chat_mean"].quantile(0.10)
    session["reason_codes"] = session.apply(add_reason_codes, axis=1)

    group_summary = (
        session.groupby("pu_discovery_group")
        .agg(
            n_sessions=("session_key", "count"),
            mean_pu_score=("pu_positive_score", "mean"),
            median_pu_score=("pu_positive_score", "median"),
            mean_stability=("pu_score_stability", "mean"),
            known_positive_rate=("known_positive", "mean"),
            viewer_med_median=("viewer_med", "median"),
            chat_mean_median=("chat_mean", "median"),
            zero_rate_median=("zero_rate", "median"),
            gap_med_median=("gap_med", "median"),
        )
        .reset_index()
        .sort_values("mean_pu_score", ascending=False)
    )

    evidence_summary = pd.DataFrame(
        [
            {"item": "known_positive_kmeans_cluster_1", "n_sessions": int(session["known_positive"].sum())},
            {"item": "unlabeled_kmeans_cluster_0", "n_sessions": int((~session["known_positive"]).sum())},
            {"item": "normal_seed", "n_sessions": int(session["normal_seed"].sum())},
            {"item": "reliable_negative_seed", "n_sessions": int(session["reliable_negative_seed"].sum())},
            {"item": "pu_hidden_candidate", "n_sessions": int(session["pu_discovery_group"].eq("pu_hidden_candidate").sum())},
            {"item": "pu_watchlist", "n_sessions": int(session["pu_discovery_group"].eq("pu_watchlist").sum())},
        ]
    )

    review_cols = [
        "session_key",
        "run_id",
        "broad_no",
        "user_id",
        "category_id",
        "pu_discovery_group",
        "pu_priority",
        "known_positive",
        "cluster_number",
        "normal_seed",
        "reliable_negative_seed",
        "reason_codes",
        "pu_positive_score",
        "pu_score_stability",
        "pu_score_std",
        "score_rule_directional",
        "reliable_negative_score",
        "viewer_med",
        "viewer_max",
        "chat_mean",
        "unique_mean",
        "zero_rate",
        "zrun_max",
        "gap_med",
        "gap_max",
        "n",
        "start",
        "end",
    ]
    review_candidates = session.sort_values("pu_positive_score", ascending=False)[review_cols]

    # Diagnostic only: how well the PU score recovers the known positives. This is not a true viewbot metric.
    y_known = session["known_positive"].astype(int)
    diagnostic_metrics = pd.DataFrame(
        [
            {
                "metric_scope": "known_positive_recovery_not_ground_truth",
                "pr_auc_vs_known_positive": average_precision_score(y_known, session["pu_positive_score"]),
                "roc_auc_vs_known_positive": roc_auc_score(y_known, session["pu_positive_score"]),
                "hidden_threshold_unlabeled_q95": hidden_threshold,
                "watch_threshold_unlabeled_q90": watch_threshold,
                "stability_threshold": stable_threshold,
            }
        ]
    )

    review_candidates.to_csv(output_dir / "07d_pu_hidden_review_candidates.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(output_dir / "07d_pu_group_summary.csv", index=False, encoding="utf-8-sig")
    evidence_summary.to_csv(output_dir / "07d_pu_evidence_summary.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(output_dir / "07d_pu_bagging_diagnostics.csv", index=False, encoding="utf-8-sig")
    diagnostic_metrics.to_csv(output_dir / "07d_pu_known_positive_recovery_metrics.csv", index=False, encoding="utf-8-sig")

    print("PU hidden discovery outputs saved")
    print(group_summary.to_string(index=False))
    print()
    print(evidence_summary.to_string(index=False))
    print()
    print(diagnostic_metrics.to_string(index=False))
    print()
    print(review_candidates.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
