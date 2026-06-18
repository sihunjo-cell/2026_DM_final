from __future__ import annotations

import pandas as pd

from session_common import (
    add_rule_scores,
    evaluate_supervised_models,
    evaluate_unsupervised_scores,
    find_project_root,
    run_ablation,
)


def main() -> None:
    root = find_project_root()
    output_dir = root / "outputs"
    session = pd.read_csv(root / "csv" / "session_summary_processed.csv")
    session["silver_suspicious"] = session["cluster_number"].eq(1).astype(int)
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
    ]
    X = session[feature_cols].copy()
    y = session["silver_suspicious"].astype(int)

    supervised_metrics, tuning_summary = evaluate_supervised_models(session, X, y)
    unsupervised_metrics = evaluate_unsupervised_scores(session, X, y)
    model_metrics = pd.concat([supervised_metrics, unsupervised_metrics], ignore_index=True)
    model_metrics = model_metrics.sort_values(["pr_auc", "f2_at_prior_rate"], ascending=False)

    ablation = run_ablation(X, y)

    consensus_cols = [
        "score_rule_directional",
        "score_logistic_balanced_pct",
        "score_gradient_boosting_pct",
        "score_tuned_gradient_boosting_pct",
        "score_iforest_normal_fit_pct",
        "score_lof_normal_fit_pct",
        "score_one_class_svm_normal_fit_pct",
        "score_pca_reconstruction_5_pct",
    ]
    consensus_cols = [c for c in consensus_cols if c in session.columns]
    session["score_advanced_binary_consensus"] = session[consensus_cols].mean(axis=1)

    q90 = session["score_advanced_binary_consensus"].quantile(0.90)
    q95 = session["score_advanced_binary_consensus"].quantile(0.95)
    session["advanced_binary_review_group"] = "low_priority"
    session.loc[session["score_advanced_binary_consensus"].ge(q90), "advanced_binary_review_group"] = "medium_review"
    session.loc[session["score_advanced_binary_consensus"].ge(q95), "advanced_binary_review_group"] = "high_review"

    score_cols = [c for c in session.columns if c.startswith("score_") and (c.endswith("_pct") or c in ["score_rule_directional"])]
    review_cols = [
        "session_key",
        "run_id",
        "broad_no",
        "user_id",
        "category_id",
        "advanced_binary_review_group",
        "score_advanced_binary_consensus",
        "cluster_number",
        "silver_suspicious",
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
        *score_cols,
    ]
    review_candidates = session.sort_values("score_advanced_binary_consensus", ascending=False)[review_cols]

    model_metrics.to_csv(output_dir / "07b_advanced_binary_model_cv_metrics.csv", index=False, encoding="utf-8-sig")
    tuning_summary.to_csv(output_dir / "07b_hyperparameter_tuning_summary.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(output_dir / "07b_session_ablation_study.csv", index=False, encoding="utf-8-sig")
    review_candidates.to_csv(output_dir / "07b_advanced_binary_review_candidates.csv", index=False, encoding="utf-8-sig")

    print("advanced binary outputs saved")
    print(model_metrics.head(12).to_string(index=False))
    print(review_candidates["advanced_binary_review_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
