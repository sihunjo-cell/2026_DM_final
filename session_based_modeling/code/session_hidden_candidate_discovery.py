from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from session_common import add_rule_scores, evaluate_hdbscan, find_project_root, make_preprocess, pct_rank, profile_clusters


RANDOM_STATE = 42


def safe_pct(s: pd.Series, ascending: bool = True) -> pd.Series:
    return pct_rank(pd.Series(s), ascending=ascending).fillna(0.5)


def build_normal_seed(session: pd.DataFrame) -> pd.Series:
    """Conservative normal seed, used only for normal-only anomaly models."""
    return (
        session["cluster_number"].eq(0)
        & session["zero_rate"].le(session["zero_rate"].quantile(0.45))
        & session["gap_med"].le(session["gap_med"].quantile(0.55))
        & session["chat_mean"].ge(session["chat_mean"].quantile(0.35))
        & session["unique_mean"].ge(session["unique_mean"].quantile(0.35))
        & session["zrun_max"].le(session["zrun_max"].quantile(0.60))
    )


def add_normal_only_anomaly_scores(session: pd.DataFrame, feature_cols: list[str], normal_seed: pd.Series) -> pd.DataFrame:
    out = session.copy()
    X_scaled = make_preprocess().fit_transform(out[feature_cols])
    normal_idx = normal_seed.to_numpy()

    iforest = IsolationForest(n_estimators=600, contamination=0.08, random_state=RANDOM_STATE, n_jobs=-1)
    iforest.fit(X_scaled[normal_idx])
    out["score_normal_iforest"] = -iforest.decision_function(X_scaled)

    lof = LocalOutlierFactor(n_neighbors=35, contamination=0.08, novelty=True)
    lof.fit(X_scaled[normal_idx])
    out["score_normal_lof"] = -lof.decision_function(X_scaled)

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.08)
    ocsvm.fit(X_scaled[normal_idx])
    out["score_normal_ocsvm"] = -ocsvm.decision_function(X_scaled)

    n_components = min(5, X_scaled.shape[1] - 1)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(X_scaled[normal_idx])
    reconstructed = pca.inverse_transform(pca.transform(X_scaled))
    out["score_normal_pca_reconstruction"] = np.mean((X_scaled - reconstructed) ** 2, axis=1)

    return out


def add_density_and_cluster_scores(session: pd.DataFrame, cluster_features: list[str]) -> pd.DataFrame:
    out = session.copy()
    X_scaled = make_preprocess().fit_transform(out[cluster_features])

    gmm = GaussianMixture(n_components=7, covariance_type="full", n_init=10, random_state=RANDOM_STATE)
    out["gmm_unsup_cluster"] = gmm.fit_predict(X_scaled)
    out["score_gmm_low_density"] = -gmm.score_samples(X_scaled)
    out["gmm_max_prob"] = gmm.predict_proba(X_scaled).max(axis=1)
    out["score_gmm_ambiguity"] = 1 - out["gmm_max_prob"]

    _, hdbscan = evaluate_hdbscan(out, cluster_features)
    out["hdbscan_unsup_cluster"] = hdbscan.fit_predict(X_scaled)
    out["score_hdbscan_noise"] = out["hdbscan_unsup_cluster"].eq(-1).astype(float)

    return out


def add_reason_codes(row: pd.Series) -> str:
    reasons = []
    if row["rule_mismatch_pct"] >= 0.90:
        reasons.append("high_rule_mismatch")
    if row["normal_anomaly_pct"] >= 0.90:
        reasons.append("normal_manifold_outlier")
    if row["density_anomaly_pct"] >= 0.90:
        reasons.append("low_density_or_ambiguous_cluster")
    if row["score_hdbscan_noise"] > 0:
        reasons.append("hdbscan_noise")
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
    cluster_features = ["viewer_med", "chat_mean", "unique_mean", "zero_rate", "zrun_max", "gap_med", "gap_max"]

    session["normal_seed"] = build_normal_seed(session)
    session = add_normal_only_anomaly_scores(session, feature_cols, session["normal_seed"])
    session = add_density_and_cluster_scores(session, cluster_features)

    session["rule_mismatch_pct"] = safe_pct(session["score_rule_directional"], True)
    anomaly_cols = [
        "score_normal_iforest",
        "score_normal_lof",
        "score_normal_ocsvm",
        "score_normal_pca_reconstruction",
    ]
    for col in anomaly_cols:
        session[col + "_pct"] = safe_pct(session[col], True)
    session["normal_anomaly_pct"] = session[[col + "_pct" for col in anomaly_cols]].mean(axis=1)

    session["score_gmm_low_density_pct"] = safe_pct(session["score_gmm_low_density"], True)
    session["score_gmm_ambiguity_pct"] = safe_pct(session["score_gmm_ambiguity"], True)
    session["density_anomaly_pct"] = session[["score_gmm_low_density_pct", "score_gmm_ambiguity_pct"]].mean(axis=1)

    session["unsupervised_hidden_score"] = (
        0.35 * session["rule_mismatch_pct"]
        + 0.35 * session["normal_anomaly_pct"]
        + 0.20 * session["density_anomaly_pct"]
        + 0.10 * session["score_hdbscan_noise"]
    )

    hidden_threshold = session["unsupervised_hidden_score"].quantile(0.90)
    strong_threshold = session["unsupervised_hidden_score"].quantile(0.95)
    kmeans_flag = session["cluster_number"].eq(1)
    independent_flag = session["unsupervised_hidden_score"].ge(hidden_threshold)
    strong_independent_flag = session["unsupervised_hidden_score"].ge(strong_threshold)

    session["discovery_group"] = "normal_low_priority"
    session.loc[kmeans_flag & independent_flag, "discovery_group"] = "confirmed_suspicious"
    session.loc[~kmeans_flag & independent_flag, "discovery_group"] = "hidden_candidate"
    session.loc[kmeans_flag & ~independent_flag, "discovery_group"] = "kmeans_only_candidate"

    session["hidden_candidate_priority"] = "low"
    session.loc[independent_flag, "hidden_candidate_priority"] = "medium"
    session.loc[strong_independent_flag, "hidden_candidate_priority"] = "high"

    session["_zero_q90"] = session["zero_rate"].quantile(0.90)
    session["_gap_q90"] = session["gap_med"].quantile(0.90)
    session["_chat_q10"] = session["chat_mean"].quantile(0.10)
    session["reason_codes"] = session.apply(add_reason_codes, axis=1)

    score_cols = [
        "score_rule_directional",
        "rule_mismatch_pct",
        "normal_anomaly_pct",
        "density_anomaly_pct",
        "score_hdbscan_noise",
        "unsupervised_hidden_score",
    ]
    review_cols = [
        "session_key",
        "run_id",
        "broad_no",
        "user_id",
        "category_id",
        "discovery_group",
        "hidden_candidate_priority",
        "cluster_number",
        "silver_suspicious",
        "normal_seed",
        "gmm_unsup_cluster",
        "gmm_max_prob",
        "hdbscan_unsup_cluster",
        "reason_codes",
        *score_cols,
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
    review_candidates = session.sort_values("unsupervised_hidden_score", ascending=False)[review_cols]

    group_summary = (
        session.groupby("discovery_group")
        .agg(
            n_sessions=("session_key", "count"),
            mean_hidden_score=("unsupervised_hidden_score", "mean"),
            median_hidden_score=("unsupervised_hidden_score", "median"),
            silver_rate=("silver_suspicious", "mean"),
            viewer_med_median=("viewer_med", "median"),
            chat_mean_median=("chat_mean", "median"),
            zero_rate_median=("zero_rate", "median"),
            gap_med_median=("gap_med", "median"),
            hdbscan_noise_rate=("score_hdbscan_noise", "mean"),
        )
        .reset_index()
        .sort_values("mean_hidden_score", ascending=False)
    )

    evidence_agreement = pd.DataFrame(
        {
            "evidence": [
                "rule_mismatch_top10",
                "normal_anomaly_top10",
                "density_anomaly_top10",
                "hdbscan_noise",
                "kmeans_cluster_1",
                "hidden_candidate_cluster_0_top10",
            ],
            "n_sessions": [
                int((session["rule_mismatch_pct"] >= 0.90).sum()),
                int((session["normal_anomaly_pct"] >= 0.90).sum()),
                int((session["density_anomaly_pct"] >= 0.90).sum()),
                int(session["score_hdbscan_noise"].sum()),
                int(kmeans_flag.sum()),
                int((~kmeans_flag & independent_flag).sum()),
            ],
        }
    )

    hdbscan_profile = profile_clusters(session, "hdbscan_unsup_cluster")
    gmm_profile = profile_clusters(session, "gmm_unsup_cluster")

    review_candidates.to_csv(output_dir / "07c_hidden_candidate_review_candidates.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(output_dir / "07c_discovery_group_summary.csv", index=False, encoding="utf-8-sig")
    evidence_agreement.to_csv(output_dir / "07c_evidence_agreement_summary.csv", index=False, encoding="utf-8-sig")
    gmm_profile.to_csv(output_dir / "07c_gmm_unsupervised_profile.csv", index=False, encoding="utf-8-sig")
    hdbscan_profile.to_csv(output_dir / "07c_hdbscan_unsupervised_profile.csv", index=False, encoding="utf-8-sig")

    print("hidden candidate discovery outputs saved")
    print(group_summary.to_string(index=False))
    print()
    print(evidence_agreement.to_string(index=False))
    print()
    print(review_candidates.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
