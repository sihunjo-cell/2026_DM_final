from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import HDBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC, OneClassSVM, SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"
OUTPUT_DIR = ROOT / "outputs"
IMG_DIR = ROOT / "img"

OUTPUT_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def pct_rank(s: pd.Series, ascending: bool = True) -> pd.Series:
    return s.rank(pct=True, ascending=ascending, method="average")


def find_project_root(start: Path | None = None) -> Path:
    here = Path.cwd() if start is None else Path(start)
    for p in [here, *here.parents]:
        if (p / "csv" / "session_summary_processed.csv").exists():
            return p
    return ROOT


def make_preprocess() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )


def precision_recall_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score)
    k = int(min(max(k, 1), len(score)))
    idx = np.argsort(score)[::-1][:k]
    positives = y_true.sum()
    precision = float(y_true[idx].mean())
    recall = float(y_true[idx].sum() / positives) if positives else np.nan
    return precision, recall


def binary_metrics_from_score(y_true: pd.Series, score: np.ndarray, positive_count: int) -> dict[str, float]:
    score = np.asarray(score, dtype=float)
    y_arr = np.asarray(y_true).astype(int)
    threshold = np.quantile(score, 1 - y_arr.mean())
    pred = (score >= threshold).astype(int)
    precision, recall, f2, _ = precision_recall_fscore_support(
        y_arr, pred, beta=2, average="binary", zero_division=0
    )
    p_at_k, r_at_k = precision_recall_at_k(y_arr, score, positive_count)
    return {
        "pr_auc": float(average_precision_score(y_arr, score)),
        "roc_auc": float(roc_auc_score(y_arr, score)),
        "precision_at_pos_k": p_at_k,
        "recall_at_pos_k": r_at_k,
        "precision_at_prior_rate": float(precision),
        "recall_at_prior_rate": float(recall),
        "f2_at_prior_rate": float(f2),
    }


def model_score_oof(name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> np.ndarray:
    method = "predict_proba"
    scores = cross_val_predict(model, X, y, cv=cv, method=method, n_jobs=-1)[:, 1]
    return scores


def tune_model(name: str, model: Pipeline, param_grid: dict, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        model,
        param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    best = search.best_estimator_
    best._midterm_best_params = search.best_params_
    best._midterm_best_score = search.best_score_
    return best


def profile_clusters(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for label, g in df.groupby(label_col, dropna=False):
        rows.append(
            {
                label_col: label,
                "n_sessions": len(g),
                "silver_rate": g["silver_suspicious"].mean(),
                "viewer_med_median": g["viewer_med"].median(),
                "chat_mean_median": g["chat_mean"].median(),
                "unique_mean_median": g["unique_mean"].median(),
                "zero_rate_median": g["zero_rate"].median(),
                "zrun_max_median": g["zrun_max"].median(),
                "gap_med_median": g["gap_med"].median(),
                "gap_max_median": g["gap_max"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values(["silver_rate", "gap_med_median"], ascending=[False, False])


def add_rule_scores(session: pd.DataFrame) -> pd.DataFrame:
    out = session.copy()
    out["rule_zero_rate"] = pct_rank(out["zero_rate"], True)
    out["rule_gap_med"] = pct_rank(out["gap_med"], True)
    out["rule_gap_max"] = pct_rank(out["gap_max"], True)
    out["rule_zrun_max"] = pct_rank(out["zrun_max"], True)
    out["rule_low_chat"] = pct_rank(-out["chat_mean"], True)
    out["rule_low_unique"] = pct_rank(-out["unique_mean"], True)
    out["score_rule_directional"] = out[
        ["rule_zero_rate", "rule_gap_med", "rule_zrun_max", "rule_low_chat", "rule_low_unique"]
    ].mean(axis=1)
    return out


def add_cluster_aware_rule_score(session: pd.DataFrame, label_col: str) -> pd.Series:
    parts = []
    for _, idx in session.groupby(label_col).groups.items():
        g = session.loc[idx]
        if len(g) < 30:
            local = session.loc[idx, "score_rule_directional"]
        else:
            local = pd.concat(
                [
                    pct_rank(g["zero_rate"], True),
                    pct_rank(g["gap_med"], True),
                    pct_rank(g["zrun_max"], True),
                    pct_rank(-g["chat_mean"], True),
                    pct_rank(-g["unique_mean"], True),
                ],
                axis=1,
            ).mean(axis=1)
        parts.append(local)
    return pd.concat(parts).sort_index()


def evaluate_gmm(session: pd.DataFrame, cluster_features: list[str]) -> tuple[pd.DataFrame, GaussianMixture]:
    X_scaled = make_preprocess().fit_transform(session[cluster_features])
    rows = []
    best_model = None
    best_bic = np.inf
    for n in range(2, 9):
        gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=RANDOM_STATE, n_init=10)
        labels = gmm.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
        bic = gmm.bic(X_scaled)
        aic = gmm.aic(X_scaled)
        rows.append({"n_components": n, "bic": bic, "aic": aic, "silhouette": sil})
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
    return pd.DataFrame(rows), best_model


def evaluate_hdbscan(session: pd.DataFrame, cluster_features: list[str]) -> tuple[pd.DataFrame, HDBSCAN]:
    X_scaled = make_preprocess().fit_transform(session[cluster_features])
    rows = []
    best_model = None
    best_key = (-np.inf, -np.inf)
    for min_cluster_size in [5, 8, 10, 15, 20, 30, 50]:
        for min_samples in [None, 5, 10]:
            model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
            non_noise = labels != -1
            n_clusters = len(set(labels[non_noise]))
            coverage = float(non_noise.mean())
            if n_clusters >= 2 and non_noise.sum() > n_clusters:
                sil = float(silhouette_score(X_scaled[non_noise], labels[non_noise]))
            else:
                sil = np.nan
            rows.append(
                {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": -1 if min_samples is None else min_samples,
                    "n_clusters": n_clusters,
                    "n_noise": int((labels == -1).sum()),
                    "coverage": coverage,
                    "silhouette_non_noise": sil,
                }
            )
            key = (0 if np.isnan(sil) else sil, coverage)
            if n_clusters >= 2 and coverage > 0.5 and key > best_key:
                best_key = key
                best_model = model
    if best_model is None:
        best_model = HDBSCAN(min_cluster_size=10).fit(X_scaled)
    return pd.DataFrame(rows), best_model


def evaluate_unsupervised_scores(session: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X_scaled = make_preprocess().fit_transform(X)
    normal_mask = y.eq(0).to_numpy()
    contamination = float(np.clip(y.mean(), 0.01, 0.2))

    scores = {}

    iforest = IsolationForest(
        n_estimators=600,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X_scaled[normal_mask])
    scores["iforest_normal_fit"] = -iforest.decision_function(X_scaled)

    lof = LocalOutlierFactor(n_neighbors=35, contamination=contamination, novelty=True)
    lof.fit(X_scaled[normal_mask])
    scores["lof_normal_fit"] = -lof.decision_function(X_scaled)

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=contamination)
    ocsvm.fit(X_scaled[normal_mask])
    scores["one_class_svm_normal_fit"] = -ocsvm.decision_function(X_scaled)

    pca_rows = []
    for n_components in [2, 3, 5, 8]:
        n_components = min(n_components, X_scaled.shape[1] - 1)
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        pca.fit(X_scaled[normal_mask])
        reconstructed = pca.inverse_transform(pca.transform(X_scaled))
        err = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        scores[f"pca_reconstruction_{n_components}"] = err
        pca_rows.append((n_components, pca.explained_variance_ratio_.sum()))

    positive_count = int(y.sum())
    rows = []
    for name, score in scores.items():
        rows.append({"model": name, "model_family": "unsupervised_or_autoencoder_proxy", **binary_metrics_from_score(y, score, positive_count)})
        session[f"score_{name}_pct"] = pct_rank(pd.Series(score, index=session.index), True)
    return pd.DataFrame(rows)


def evaluate_supervised_models(session: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    positive_count = int(y.sum())

    models: dict[str, Pipeline] = {
        "logistic_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "decision_tree_depth3": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "gaussian_nb": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        max_depth=6,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=160,
                        learning_rate=0.05,
                        max_depth=2,
                        subsample=0.85,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_lightgbm_proxy": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=150,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "linear_svc_calibrated": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE, max_iter=20000),
                        cv=3,
                    ),
                ),
            ]
        ),
        "rbf_svc_calibrated": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=SVC(C=2.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=RANDOM_STATE),
                        cv=3,
                    ),
                ),
            ]
        ),
        "mlp_neural_network": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(16, 8),
                        activation="relu",
                        alpha=0.001,
                        max_iter=1200,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "pu_style_weighted_logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight={0: 0.30, 1: 1.0}, random_state=RANDOM_STATE)),
            ]
        ),
    }

    tuned_specs = {
        "tuned_gradient_boosting": (
            models["gradient_boosting"],
            {
                "model__n_estimators": [80, 160],
                "model__learning_rate": [0.03, 0.07],
                "model__max_depth": [1, 2],
            },
        ),
        "tuned_rbf_svc": (
            models["rbf_svc_calibrated"],
            {
                "model__estimator__C": [0.5, 2.0, 8.0],
                "model__estimator__gamma": ["scale", 0.1],
            },
        ),
        "tuned_mlp": (
            models["mlp_neural_network"],
            {
                "model__hidden_layer_sizes": [(12,), (16, 8), (32, 16)],
                "model__alpha": [0.0001, 0.001, 0.01],
            },
        ),
    }

    tuning_rows = []
    for tuned_name, (base_model, grid) in tuned_specs.items():
        tuned = tune_model(tuned_name, base_model, grid, X, y)
        models[tuned_name] = tuned
        tuning_rows.append(
            {
                "model": tuned_name,
                "best_cv_pr_auc": getattr(tuned, "_midterm_best_score", np.nan),
                "best_params": str(getattr(tuned, "_midterm_best_params", {})),
            }
        )

    rows = []
    for name, model in models.items():
        scores = model_score_oof(name, model, X, y, cv)
        session[f"score_{name}"] = scores
        session[f"score_{name}_pct"] = pct_rank(pd.Series(scores, index=session.index), True)
        rows.append({"model": name, "model_family": "supervised_silver_label", **binary_metrics_from_score(y, scores, positive_count)})

    return pd.DataFrame(rows), pd.DataFrame(tuning_rows)


def run_ablation(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    feature_groups = {
        "all_features": list(X.columns),
        "no_zero_rate": [c for c in X.columns if c != "zero_rate"],
        "no_gap_features": [c for c in X.columns if not c.startswith("gap")],
        "no_chat_features": [c for c in X.columns if c not in ["chat_mean", "unique_mean", "log_chat", "log_unique"]],
        "no_viewer_features": [c for c in X.columns if c not in ["viewer_med", "viewer_max", "log_viewer"]],
        "mismatch_core_only": ["zero_rate", "zrun_max", "gap_med", "gap_max", "chat_mean", "unique_mean"],
        "viewer_only": ["viewer_med", "viewer_max", "log_viewer"],
        "chat_only": ["chat_mean", "unique_mean", "log_chat", "log_unique"],
    }
    model_specs = {
        "logistic_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=160,
                        learning_rate=0.05,
                        max_depth=2,
                        subsample=0.85,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    positive_count = int(y.sum())
    rows = []
    for model_name, model in model_specs.items():
        for group_name, cols in feature_groups.items():
            scores = model_score_oof(model_name, clone(model), X[cols], y, cv)
            rows.append(
                {
                    "model": model_name,
                    "feature_group": group_name,
                    "n_features": len(cols),
                    **binary_metrics_from_score(y, scores, positive_count),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "pr_auc"], ascending=[True, False])


def main() -> None:
    root = find_project_root()
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

    X = session[feature_cols].copy()
    y = session["silver_suspicious"].astype(int)

    gmm_selection, gmm = evaluate_gmm(session, cluster_features)
    X_cluster_scaled = make_preprocess().fit_transform(session[cluster_features])
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

    supervised_metrics, tuning_summary = evaluate_supervised_models(session, X, y)
    unsupervised_metrics = evaluate_unsupervised_scores(session, X, y)

    positive_count = int(y.sum())
    cluster_rows = []
    for score_col in ["score_gmm_cluster_aware_rule", "score_hdbscan_cluster_aware_rule", "score_rule_directional"]:
        cluster_rows.append(
            {
                "model": score_col,
                "model_family": "cluster_aware_or_rule",
                **binary_metrics_from_score(y, session[score_col].to_numpy(), positive_count),
            }
        )
    model_metrics = pd.concat([supervised_metrics, unsupervised_metrics, pd.DataFrame(cluster_rows)], ignore_index=True)
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
        "score_gmm_cluster_aware_rule_pct",
        "score_hdbscan_cluster_aware_rule_pct",
        "score_pca_reconstruction_5_pct",
    ]
    consensus_cols = [c for c in consensus_cols if c in session.columns]
    session["score_midterm_completion_consensus"] = session[consensus_cols].mean(axis=1)

    q90 = session["score_midterm_completion_consensus"].quantile(0.90)
    q95 = session["score_midterm_completion_consensus"].quantile(0.95)
    session["midterm_completion_review_group"] = "low_priority"
    session.loc[session["score_midterm_completion_consensus"].ge(q90), "midterm_completion_review_group"] = "medium_review"
    session.loc[session["score_midterm_completion_consensus"].ge(q95), "midterm_completion_review_group"] = "high_review"

    review_cols = [
        "session_key",
        "run_id",
        "broad_no",
        "user_id",
        "category_id",
        "midterm_completion_review_group",
        "score_midterm_completion_consensus",
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
    review_candidates = session.sort_values("score_midterm_completion_consensus", ascending=False)[review_cols]

    completion_summary = pd.DataFrame(
        [
            {"midterm_plan": "Increase data size", "implementation": f"{len(session)} sessions loaded", "status": "done"},
            {"midterm_plan": "SVM baseline", "implementation": "linear SVC, calibrated RBF SVC, One-Class SVM", "status": "done"},
            {"midterm_plan": "LOF baseline", "implementation": "LOF novelty detector trained on silver-normal sessions", "status": "done"},
            {"midterm_plan": "XGBoost/LightGBM", "implementation": "sklearn GradientBoosting and HistGradientBoosting fallback", "status": "partial_fallback"},
            {"midterm_plan": "PU-learning", "implementation": "PU-style weighted logistic using silver positives vs weak negatives", "status": "proxy_done"},
            {"midterm_plan": "AutoEncoder", "implementation": "PCA reconstruction anomaly score as non-deep autoencoder proxy", "status": "proxy_done"},
            {"midterm_plan": "Cluster-aware modeling", "implementation": "GMM model selection, sklearn HDBSCAN, cluster-aware rule score", "status": "done"},
            {"midterm_plan": "PR-AUC / F-beta metrics", "implementation": "PR-AUC, ROC-AUC, precision/recall@K, F2 at prior-rate threshold", "status": "done"},
            {"midterm_plan": "Cross-validation", "implementation": "5-fold stratified out-of-fold scoring", "status": "done"},
            {"midterm_plan": "Hyperparameter tuning", "implementation": "GridSearchCV for GBM, RBF SVC, MLP", "status": "done"},
            {"midterm_plan": "Ablation study", "implementation": "feature-group ablation for logistic and GBM", "status": "done"},
            {"midterm_plan": "Ground truth issue", "implementation": "explicitly retained as silver-label screening limitation", "status": "documented_limitation"},
        ]
    )

    gmm_selection.to_csv(root / "outputs" / "07_gmm_model_selection.csv", index=False, encoding="utf-8-sig")
    gmm_profile.to_csv(root / "outputs" / "07_gmm_cluster_profile.csv", index=False, encoding="utf-8-sig")
    hdbscan_selection.to_csv(root / "outputs" / "07_hdbscan_model_selection.csv", index=False, encoding="utf-8-sig")
    hdbscan_profile.to_csv(root / "outputs" / "07_hdbscan_cluster_profile.csv", index=False, encoding="utf-8-sig")
    model_metrics.to_csv(root / "outputs" / "07_model_cv_metrics.csv", index=False, encoding="utf-8-sig")
    tuning_summary.to_csv(root / "outputs" / "07_hyperparameter_tuning_summary.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(root / "outputs" / "07_session_ablation_study.csv", index=False, encoding="utf-8-sig")
    review_candidates.to_csv(root / "outputs" / "07_cluster_aware_review_candidates.csv", index=False, encoding="utf-8-sig")
    completion_summary.to_csv(root / "outputs" / "07_midterm_plan_completion_summary.csv", index=False, encoding="utf-8-sig")

    print("saved:")
    for name in [
        "07_gmm_model_selection.csv",
        "07_gmm_cluster_profile.csv",
        "07_hdbscan_model_selection.csv",
        "07_hdbscan_cluster_profile.csv",
        "07_model_cv_metrics.csv",
        "07_hyperparameter_tuning_summary.csv",
        "07_session_ablation_study.csv",
        "07_cluster_aware_review_candidates.csv",
        "07_midterm_plan_completion_summary.csv",
    ]:
        print(" -", root / "outputs" / name)
    print()
    print("top model metrics:")
    print(model_metrics.head(10).to_string(index=False))
    print()
    print("review groups:")
    print(review_candidates["midterm_completion_review_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
