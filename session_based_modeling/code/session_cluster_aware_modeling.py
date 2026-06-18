from __future__ import annotations

import pandas as pd

from session_common import (
    RANDOM_STATE,
    add_cluster_aware_rule_score,
    add_rule_scores,
    binary_metrics_from_score,
    evaluate_gmm,
    evaluate_hdbscan,
    find_project_root,
    make_preprocess,
    pct_rank,
    profile_clusters,
)


def main() -> None:
    root = find_project_root()
    output_dir = root / "outputs"
    session = pd.read_csv(root / "csv" / "session_summary_processed.csv")
    session["silver_suspicious"] = session["cluster_number"].eq(1).astype(int)
    session = add_rule_scores(session)

    cluster_features = [
        "viewer_med",
        "chat_mean",
        "unique_mean",
        "zero_rate",
        "zrun_max",
        "gap_med",
        "gap_max",
    ]
    X_cluster_scaled = make_preprocess().fit_transform(session[cluster_features])

    gmm_selection, gmm = evaluate_gmm(session, cluster_features)
    session["gmm_cluster"] = gmm.predict(X_cluster_scaled)
    session["gmm_max_prob"] = gmm.predict_proba(X_cluster_scaled).max(axis=1)
    session["score_gmm_cluster_aware_rule"] = add_cluster_aware_rule_score(session, "gmm_cluster")
    session["score_gmm_cluster_aware_rule_pct"] = pct_rank(session["score_gmm_cluster_aware_rule"], True)

    hdbscan_selection, hdbscan = evaluate_hdbscan(session, cluster_features)
    session["hdbscan_cluster"] = hdbscan.fit_predict(X_cluster_scaled)
    session["score_hdbscan_cluster_aware_rule"] = add_cluster_aware_rule_score(session, "hdbscan_cluster")
    session["score_hdbscan_cluster_aware_rule_pct"] = pct_rank(session["score_hdbscan_cluster_aware_rule"], True)

    gmm_profile = profile_clusters(session, "gmm_cluster")
    hdbscan_profile = profile_clusters(session, "hdbscan_cluster")

    positive_count = int(session["silver_suspicious"].sum())
    cluster_metrics = pd.DataFrame(
        [
            {
                "score": "score_rule_directional",
                "score_family": "global_rule",
                **binary_metrics_from_score(
                    session["silver_suspicious"], session["score_rule_directional"].to_numpy(), positive_count
                ),
            },
            {
                "score": "score_gmm_cluster_aware_rule",
                "score_family": "cluster_aware_rule",
                **binary_metrics_from_score(
                    session["silver_suspicious"], session["score_gmm_cluster_aware_rule"].to_numpy(), positive_count
                ),
            },
            {
                "score": "score_hdbscan_cluster_aware_rule",
                "score_family": "cluster_aware_rule",
                **binary_metrics_from_score(
                    session["silver_suspicious"], session["score_hdbscan_cluster_aware_rule"].to_numpy(), positive_count
                ),
            },
        ]
    ).sort_values("pr_auc", ascending=False)

    session["score_cluster_aware_consensus"] = session[
        [
            "score_rule_directional",
            "score_gmm_cluster_aware_rule_pct",
            "score_hdbscan_cluster_aware_rule_pct",
        ]
    ].mean(axis=1)

    q90 = session["score_cluster_aware_consensus"].quantile(0.90)
    q95 = session["score_cluster_aware_consensus"].quantile(0.95)
    session["cluster_aware_review_group"] = "low_priority"
    session.loc[session["score_cluster_aware_consensus"].ge(q90), "cluster_aware_review_group"] = "medium_review"
    session.loc[session["score_cluster_aware_consensus"].ge(q95), "cluster_aware_review_group"] = "high_review"

    review_cols = [
        "session_key",
        "run_id",
        "broad_no",
        "user_id",
        "category_id",
        "cluster_aware_review_group",
        "score_cluster_aware_consensus",
        "cluster_number",
        "silver_suspicious",
        "gmm_cluster",
        "gmm_max_prob",
        "hdbscan_cluster",
        "score_rule_directional",
        "score_gmm_cluster_aware_rule",
        "score_hdbscan_cluster_aware_rule",
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
    review_candidates = session.sort_values("score_cluster_aware_consensus", ascending=False)[review_cols]

    gmm_selection.to_csv(output_dir / "07a_gmm_model_selection.csv", index=False, encoding="utf-8-sig")
    gmm_profile.to_csv(output_dir / "07a_gmm_cluster_profile.csv", index=False, encoding="utf-8-sig")
    hdbscan_selection.to_csv(output_dir / "07a_hdbscan_model_selection.csv", index=False, encoding="utf-8-sig")
    hdbscan_profile.to_csv(output_dir / "07a_hdbscan_cluster_profile.csv", index=False, encoding="utf-8-sig")
    cluster_metrics.to_csv(output_dir / "07a_cluster_aware_score_metrics.csv", index=False, encoding="utf-8-sig")
    review_candidates.to_csv(output_dir / "07a_cluster_aware_review_candidates.csv", index=False, encoding="utf-8-sig")

    print("cluster-aware outputs saved")
    print(cluster_metrics.to_string(index=False))
    print(review_candidates["cluster_aware_review_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
