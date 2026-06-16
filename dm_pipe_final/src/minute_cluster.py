from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler
import joblib

from src.k_selection import good_rank


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

DEFAULT_FEATURES = [
    "log_viewer",
    "chat_deficit",
    "unique_deficit",
    "rolling_chat_deficit_5m",
    "log_zero_run_len",
    "rolling_zero_rate_5m",
]

ASSIGN_COLS = ["session_key", "run_id", "broad_no", "minute_ts", "minute_cluster"]
PROFILE_COLS = [
    "cluster_id",
    "n",
    "share",
    "viewer_med",
    "chat_med",
    "unique_med",
    "chat_deficit_med",
    "unique_deficit_med",
    "rolling_chat_deficit_5m_med",
    "zero_chat_rate",
    "zero_run_len_p75",
    "rolling_zero_rate_5m_mean",
    "rolling_zero_rate_5m_p90",
    "rolling_zero_rate_5m_p95",
    "cluster_mismatch_rank",
    "interpretation",
]
SELECT_COLS = [
    "algorithm",
    "param",
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "cluster_size_balance",
    "cluster_profile_separation",
    "random_seed_stability",
    "selection_score",
    "selected_k_num",
    "bic",
    "aic",
    "selected",
    "note",
]
GMM_COLS = ["algorithm", "param", "bic", "aic", "sample_size", "selected", "note"]
HDBSCAN_COLS = ["algorithm", "param", "n_clusters", "noise_share", "sample_size", "selected", "note"]
MODEL_SELECTION_COLS = [
    "algorithm",
    "parameter_setting",
    "features",
    "scaler",
    "selection_metric_1",
    "selection_metric_2",
    "selection_metric_3",
    "n_clusters",
    "noise_share",
    "selected_as_final_state",
    "selection_reason",
    "why_not_selected",
]


def _enabled(value):
    return str(value).lower() not in {"false", "0", "no", "none", "off", "disabled"}


def _cluster_cfg(cfg):
    mc = cfg.get("minute_cluster", {})
    km = mc.get("kmeans", {})
    if not isinstance(km, dict):
        km = {}
    return mc, km


def _features(cfg):
    mc = cfg.get("minute_cluster", {})
    return mc.get("features") or cfg.get("minute_state", {}).get("cluster_features") or DEFAULT_FEATURES


def _make_x(df, feats):
    x = df[feats].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True)).fillna(0)


def _silhouette(xs, labels, sample_size, seed):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return np.nan
    if len(labels) <= sample_size:
        return silhouette_score(xs, labels)
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(len(labels)), size=sample_size, replace=False)
    if len(np.unique(labels[idx])) < 2:
        return np.nan
    return silhouette_score(xs[idx], labels[idx])


def _cluster_size_balance(labels):
    counts = pd.Series(labels).value_counts()
    if counts.empty or counts.max() == 0:
        return np.nan
    return float(counts.min() / counts.max())


def _profile_separation(xs, labels):
    work = pd.DataFrame(xs)
    work["_cluster"] = labels
    centers = work.groupby("_cluster").median(numeric_only=True)
    if len(centers) < 2:
        return np.nan
    vals = centers.to_numpy(dtype=float)
    dists = []
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            dists.append(float(np.linalg.norm(vals[i] - vals[j])))
    return float(np.mean(dists)) if dists else np.nan


def _seed_stability(xs, k, base_labels, seed, n_init):
    if k < 2 or len(xs) < k:
        return np.nan
    try:
        alt = KMeans(n_clusters=k, random_state=int(seed) + 1, n_init=n_init).fit_predict(xs)
        return float(adjusted_rand_score(base_labels, alt))
    except Exception:
        return np.nan


def _param_k_number(param):
    text = str(param)
    if "=" in text:
        text = text.split("=", 1)[1]
    return pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]


def _candidate_selection_score(select):
    if select.empty:
        return select
    work = select.copy()
    metrics = {
        "silhouette": True,
        "calinski_harabasz": True,
        "davies_bouldin": False,
        "cluster_size_balance": True,
        "cluster_profile_separation": True,
        "random_seed_stability": True,
    }
    ranks = []
    for col, higher_is_better in metrics.items():
        vals = pd.to_numeric(work.get(col), errors="coerce")
        if vals.notna().sum() == 0:
            continue
        rcol = f"_{col}_rank"
        work[rcol] = good_rank(vals, higher_is_better=higher_is_better)
        ranks.append(rcol)
    work["selection_score"] = work[ranks].mean(axis=1, skipna=True) if ranks else np.nan
    return work.drop(columns=ranks, errors="ignore")


def _write_model_selection(out, features, scaler_name, kmeans_select, gmm_diag, hdb_diag, best_param):
    rows = []
    feature_text = ", ".join(features)
    if kmeans_select is not None and not kmeans_select.empty:
        for _, row in kmeans_select.iterrows():
            selected = str(row.get("param")) == str(best_param)
            rows.append({
                "algorithm": "KMeans",
                "parameter_setting": row.get("param"),
                "features": feature_text,
                "scaler": scaler_name,
                "selection_metric_1": f"silhouette={row.get('silhouette')}; Calinski-Harabasz={row.get('calinski_harabasz')}; Davies-Bouldin={row.get('davies_bouldin')}",
                "selection_metric_2": f"cluster_size_balance={row.get('cluster_size_balance')}; cluster_profile_separation={row.get('cluster_profile_separation')}",
                "selection_metric_3": f"random_seed_stability={row.get('random_seed_stability')}; selection_score={row.get('selection_score')}",
                "n_clusters": str(row.get("param")).split("=", 1)[1] if "=" in str(row.get("param")) else row.get("param"),
                "noise_share": 0.0,
                "selected_as_final_state": bool(selected),
                "selection_reason": "동일 feature와 RobustScaler 조건에서 해석 가능한 전체 분 단위 상태 번호가 필요하여 KMeans를 최종 후보로 비교했다." if selected else "",
                "why_not_selected": "" if selected else "선택된 KMeans 후보보다 종합 선택 점수가 낮다.",
            })
    if gmm_diag is not None and not gmm_diag.empty:
        best_bic = pd.to_numeric(gmm_diag.get("bic"), errors="coerce").min()
        for _, row in gmm_diag.iterrows():
            rows.append({
                "algorithm": "GMM",
                "parameter_setting": row.get("param"),
                "features": feature_text,
                "scaler": scaler_name,
                "selection_metric_1": f"BIC={row.get('bic')}",
                "selection_metric_2": f"AIC={row.get('aic')}",
                "selection_metric_3": f"best_BIC={best_bic}",
                "n_clusters": str(row.get("param")).split("=", 1)[1] if "=" in str(row.get("param")) else row.get("param"),
                "noise_share": np.nan,
                "selected_as_final_state": False,
                "selection_reason": "",
                "why_not_selected": "component 기반 하위 구조 진단에는 유용하지만 최종 handoff용 단일 분 단위 행동 상태 번호로는 KMeans보다 설명과 재현이 어렵다.",
            })
    if hdb_diag is not None and not hdb_diag.empty:
        for _, row in hdb_diag.iterrows():
            n_clusters = row.get("n_clusters")
            noise_share = row.get("noise_share")
            rows.append({
                "algorithm": "HDBSCAN",
                "parameter_setting": row.get("param"),
                "features": feature_text,
                "scaler": scaler_name,
                "selection_metric_1": f"n_clusters={n_clusters}",
                "selection_metric_2": f"noise_share={noise_share}",
                "selection_metric_3": f"coverage={1 - float(noise_share) if pd.notna(noise_share) else np.nan}",
                "n_clusters": n_clusters,
                "noise_share": noise_share,
                "selected_as_final_state": False,
                "selection_reason": "",
                "why_not_selected": "밀도 기반 안정 군집과 noise 비중 진단에는 유용하지만 noise 처리 때문에 모든 분 row에 일관된 handoff 상태 번호를 붙이기에는 부적절하다.",
            })
    table = pd.DataFrame(rows)
    for col in MODEL_SELECTION_COLS:
        if col not in table.columns:
            table[col] = np.nan
    table[MODEL_SELECTION_COLS].to_csv(Path(out) / "minute_cluster_model_selection.csv", index=False, encoding="utf-8-sig")
    return table[MODEL_SELECTION_COLS]


def _empty_outputs(out, df, note):
    select = pd.DataFrame(columns=SELECT_COLS)
    assign = df.copy()
    for col in ASSIGN_COLS:
        if col not in assign.columns:
            assign[col] = np.nan
    assign["minute_cluster"] = np.nan
    profile = pd.DataFrame(columns=PROFILE_COLS)
    select.to_csv(out / "mc_select.csv", index=False, encoding="utf-8-sig")
    assign[ASSIGN_COLS].to_csv(out / "mc_assign.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(out / "mc_profile.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=GMM_COLS).to_csv(out / "gmm_diag.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=HDBSCAN_COLS).to_csv(out / "hdbscan_diag.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=MODEL_SELECTION_COLS).to_csv(out / "minute_cluster_model_selection.csv", index=False, encoding="utf-8-sig")
    (out / "mc_info.txt").write_text(note.rstrip() + "\n", encoding="utf-8")
    (out / "minute_cluster.txt").write_text(note.rstrip() + "\n", encoding="utf-8")
    _write_minute_cluster_doc(out, {})
    assign["cluster_mismatch_rank"] = np.nan
    return assign


def _interpret_profile(profile):
    if profile.empty:
        return profile
    out = profile.copy()
    rank_cols = [
        "chat_deficit_med",
        "unique_deficit_med",
        "rolling_chat_deficit_5m_med",
        "zero_chat_rate",
        "zero_run_len_p75",
        "rolling_zero_rate_5m_p90",
    ]
    out["cluster_mismatch_rank"] = out[rank_cols].rank(pct=True).mean(axis=1)
    viewer_global = out["viewer_med"].median()
    chat_global = out["chat_med"].median()
    unique_global = out["unique_med"].median()
    top_rank = out["cluster_mismatch_rank"].max()

    interpretations = []
    for _, row in out.iterrows():
        if pd.notna(top_rank) and row["cluster_mismatch_rank"] == top_rank:
            interpretations.append("mismatch-like state")
        elif (
            row["chat_med"] >= chat_global
            and row["unique_med"] >= unique_global
            and row["chat_deficit_med"] <= 0
        ):
            interpretations.append("active/high-chat state")
        elif row["viewer_med"] <= viewer_global and row["chat_med"] <= chat_global:
            interpretations.append("low-scale quiet state")
        else:
            interpretations.append("mixed state")
    out["interpretation"] = interpretations
    return out


def _selected_k_from_select(select):
    if select is None or select.empty:
        return "not available"
    work = select.copy()
    if "algorithm" in work.columns:
        work = work[work["algorithm"].astype(str).str.lower().eq("kmeans")]
    if work.empty:
        return "not available"
    if "selected" in work.columns:
        selected = work[work["selected"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if len(selected) != 1:
            return "invalid_selected_count"
        work = selected
    else:
        return "invalid_selected_count"
    param = str(work.iloc[0].get("param", "not available"))
    return param.split("=", 1)[1] if "=" in param else param


def _write_minute_cluster_doc(out, cfg):
    out = Path(out)
    mc_cfg = cfg.get("minute_cluster", {})
    km_cfg = mc_cfg.get("kmeans", {})
    features = mc_cfg.get("features") or DEFAULT_FEATURES
    k_min = int(km_cfg.get("k_min", 2))
    k_max = int(km_cfg.get("k_max", 8))
    seed = km_cfg.get("seed", 42)
    n_init = km_cfg.get("n_init", 50)
    sample_size = km_cfg.get("sample_size_for_silhouette")
    scaler_name = mc_cfg.get("scaler", "RobustScaler")
    try:
        select = pd.read_csv(out / "mc_select.csv", encoding="utf-8-sig")
    except Exception:
        select = pd.DataFrame()
    try:
        profile = pd.read_csv(out / "mc_profile.csv", encoding="utf-8-sig")
    except Exception:
        profile = pd.DataFrame()
    try:
        model_selection = pd.read_csv(out / "minute_cluster_model_selection.csv", encoding="utf-8-sig")
    except Exception:
        model_selection = pd.DataFrame()

    gmm_cfg = mc_cfg.get("gmm", {})
    hdb_cfg = mc_cfg.get("hdbscan", {})
    selected_rows = model_selection.loc[model_selection.get("selected_as_final_state", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1", "yes"])] if not model_selection.empty else pd.DataFrame()
    final_algorithm = selected_rows["algorithm"].iloc[0] if not selected_rows.empty else "KMeans"
    final_param = selected_rows["parameter_setting"].iloc[0] if not selected_rows.empty else f"k={_selected_k_from_select(select)}"

    lines = [
        "분 단위 행동 상태 군집 문서",
        "========================",
        "",
        "분석 단위: 1분 row.",
        "세션 정의: run_id + broad_no.",
        "목적: viewer 규모 대비 chat/unique 반응 부족과 zero-chat 지속성을 비지도 방식으로 요약한 분 단위 행동 상태 번호를 만든다.",
        "주의: minute_cluster는 정답 라벨이나 확률이 아니라 행동 상태 번호다.",
        "",
        "사용 feature 목록:",
        *[f"- {feature}" for feature in features],
        "",
        "feature 생성 방식:",
        "- log_viewer: viewer_count_last에 log1p를 적용한 값으로 방송 규모 차이를 통제한다.",
        "- chat_deficit: 같은 viewer 규모 구간에서 기대되는 log_chat 중앙값보다 실제 log_chat이 부족한 정도다.",
        "- unique_deficit: 같은 viewer 규모 구간에서 기대되는 log_unique 중앙값보다 실제 log_unique가 부족한 정도다.",
        "- rolling_chat_deficit_5m: 최근 5분 chat_deficit 평균으로 짧은 구간의 지속성을 반영한다.",
        "- log_zero_run_len: clock gap reset을 반영한 zero-chat 연속 길이에 log1p를 적용한 값이다.",
        "- rolling_zero_rate_5m: 최근 5분 zero-chat 비율이다.",
        "",
        "스케일링:",
        "- 적용: 적용",
        f"- 사용 방식: {scaler_name}",
        "- 선택 이유: viewer/chat 계열 변수의 heavy-tail과 극단값 영향을 줄이기 위해 중앙값/IQR 기반 스케일링을 사용한다.",
        "",
        "사용한 clustering 알고리즘:",
        "- KMeans: 전체 1분 row에 재현 가능한 행동 상태 번호를 붙이는 최종 후보.",
        "- GMM: component 기반 하위 구조와 BIC/AIC 진단.",
        "- HDBSCAN: 밀도 기반 안정 군집과 noise 비중 진단. noise는 이상치 확정이 아니라 안정 군집에 속하지 않은 분 단위 상태다.",
        "",
        "후보 하이퍼파라미터:",
        f"- KMeans K 후보: {k_min}..{k_max}",
        f"- random_state: {seed}",
        f"- n_init: {n_init}",
        f"- GMM n_components 후보: {gmm_cfg.get('n_min', 2)}..{gmm_cfg.get('n_max', 8)}",
        f"- HDBSCAN min_cluster_size 후보: {hdb_cfg.get('min_sizes', cfg.get('hdbscan', {}).get('min_sizes', [5, 8, 10]))}",
    ]
    if sample_size is not None:
        lines.append(f"- sample_size_for_silhouette: {sample_size}")

    lines.extend([
        "",
        "최종 선택:",
        f"- 최종 선택 알고리즘: {final_algorithm}",
        f"- 최종 선택 하이퍼파라미터: {final_param}",
        "- 최종 선택 근거: 동일 feature와 동일 스케일링 조건에서 silhouette, Calinski-Harabasz, Davies-Bouldin, 군집 크기 균형, profile 분리도, seed 안정성을 함께 비교했다.",
        "- GMM은 하위 구조 진단에는 유용하지만 component 해석과 handoff 상태 번호 재현성이 KMeans보다 낮아 최종 상태 번호로 쓰지 않았다.",
        "- HDBSCAN은 안정 군집과 noise 비중 확인에는 유용하지만 noise 처리 때문에 모든 1분 row에 일관된 상태 번호를 붙이기 어려워 최종 상태 번호로 쓰지 않았다.",
        "",
        "모델 선택 표 요약:",
    ])
    if model_selection.empty:
        lines.append("- minute_cluster_model_selection.csv가 비어 있다.")
    else:
        for _, row in model_selection.head(30).iterrows():
            selected = str(row.get("selected_as_final_state")).lower() in {"true", "1", "yes"}
            lines.append(
                f"- {row.get('algorithm')} {row.get('parameter_setting')}: selected_as_final_state={selected}, "
                f"{row.get('selection_metric_1')}, {row.get('selection_metric_2')}, {row.get('selection_metric_3')}"
            )

    lines.extend(["", "cluster 번호별 profile 요약:"])
    if profile.empty:
        lines.append("- profile을 만들 수 없었다.")
    else:
        for _, row in profile.iterrows():
            cid = row.get("cluster_id")
            interp = row.get("interpretation")
            if interp == "mismatch-like state":
                interp_ko = "viewer 대비 채팅 반응 약한 상태"
            elif interp == "active/high-chat state":
                interp_ko = "채팅 반응 활발 상태"
            elif interp == "low-scale quiet state":
                interp_ko = "소규모 조용한 상태"
            else:
                interp_ko = "혼합 행동 상태"
            parts = [f"cluster_id={cid}", f"해석={interp_ko}"]
            for col in ["n", "share", "viewer_med", "chat_med", "unique_med", "chat_deficit_med", "unique_deficit_med", "zero_chat_rate", "rolling_zero_rate_5m_p90", "cluster_mismatch_rank"]:
                if col in profile.columns:
                    val = row.get(col)
                    if isinstance(val, float):
                        val = f"{val:.4g}"
                    parts.append(f"{col}={val}")
            lines.append("- " + ", ".join(parts))
    lines.extend([
        "",
        "해석 제한:",
        "- viewer 대비 채팅 반응 약한 상태는 확정 판정이 아니라 비지도 군집 profile 해석이다.",
        "- cluster_mismatch_rank는 분 단위 mismatch 신호 profile을 설명하기 위한 보조 순위 값이다.",
    ])
    text = "\n".join(lines) + "\n"
    (out / "minute_cluster.txt").write_text(text, encoding="utf-8")
    (out / "mc_info.txt").write_text(text, encoding="utf-8")


def _profile(s):
    profile = (
        s.groupby("minute_cluster")
        .agg(
            n=("session_key", "size"),
            viewer_med=("viewer_count_last", "median"),
            chat_med=("chat_count", "median"),
            unique_med=("unique_chatters", "median"),
            chat_deficit_med=("chat_deficit", "median"),
            unique_deficit_med=("unique_deficit", "median"),
            rolling_chat_deficit_5m_med=("rolling_chat_deficit_5m", "median"),
            zero_chat_rate=("zero_chat", "mean"),
            zero_run_len_p75=("zero_run_len", lambda x: x.quantile(0.75)),
            rolling_zero_rate_5m_mean=("rolling_zero_rate_5m", "mean"),
            rolling_zero_rate_5m_p90=("rolling_zero_rate_5m", lambda x: x.quantile(0.90)),
            rolling_zero_rate_5m_p95=("rolling_zero_rate_5m", lambda x: x.quantile(0.95)),
        )
        .reset_index()
        .rename(columns={"minute_cluster": "cluster_id"})
    )
    profile["share"] = profile["n"] / max(len(s), 1)
    profile = _interpret_profile(profile)
    return profile[PROFILE_COLS]


def _run_kmeans(s, xs, km_cfg):
    k_min = int(km_cfg.get("k_min", 2))
    k_max = min(int(km_cfg.get("k_max", 8)), len(s) - 1)
    seed = int(km_cfg.get("seed", 42))
    n_init = int(km_cfg.get("n_init", 50))
    sample_size = int(km_cfg.get("sample_size_for_silhouette", 10000))
    rows, models, labels_by_k = [], {}, {}
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = model.fit_predict(xs)
        sil = _silhouette(xs, labels, sample_size, seed)
        ch = calinski_harabasz_score(xs, labels) if len(np.unique(labels)) >= 2 else np.nan
        db = davies_bouldin_score(xs, labels) if len(np.unique(labels)) >= 2 else np.nan
        rows.append({
            "algorithm": "kmeans",
            "param": f"k={k}",
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "cluster_size_balance": _cluster_size_balance(labels),
            "cluster_profile_separation": _profile_separation(xs, labels),
            "random_seed_stability": _seed_stability(xs, k, labels, seed, n_init),
            "selection_score": np.nan,
            "bic": np.nan,
            "aic": np.nan,
            "selected": False,
            "note": "KMeans final minute behavior state candidate; behavior state number only",
        })
        models[k] = model
        labels_by_k[k] = labels
    select = _candidate_selection_score(pd.DataFrame(rows))
    if select.empty:
        return select, None, None, None
    select["selected_k_num"] = select["param"].map(_param_k_number)
    valid = select.dropna(subset=["selection_score"])
    pick = valid if not valid.empty else select
    best_param = pick.sort_values(["selection_score", "silhouette", "selected_k_num"], ascending=[False, False, True]).iloc[0]["param"]
    best_k = int(str(best_param).split("=")[1])
    select.loc[select["param"].eq(best_param), "selected"] = True
    return select, best_k, labels_by_k[best_k], models[best_k]


def _diagnostic_sample(xs, cfg):
    sample_size = int(cfg.get("minute_cluster", {}).get("diagnostic_sample_size", 50000))
    if len(xs) <= sample_size:
        return xs
    rng = np.random.default_rng(int(cfg.get("minute_cluster", {}).get("diagnostic_seed", 42)))
    idx = rng.choice(np.arange(len(xs)), size=sample_size, replace=False)
    return xs[idx]


def _write_gmm_diag(xs, out, cfg):
    gmm_cfg = cfg.get("minute_cluster", {}).get("gmm", {})
    n_min = int(gmm_cfg.get("n_min", 2))
    n_max = min(int(gmm_cfg.get("n_max", 8)), len(xs) - 1)
    seed = int(gmm_cfg.get("seed", 42))
    covariance_type = str(gmm_cfg.get("covariance_type", "full"))
    sample = _diagnostic_sample(xs, cfg)
    rows = []
    if len(sample) < max(3, n_min):
        rows.append({
            "algorithm": "gmm",
            "param": "not_run",
            "bic": np.nan,
            "aic": np.nan,
            "sample_size": int(len(sample)),
            "selected": False,
            "note": "GMM diagnostic skipped because there were too few rows; not a label",
        })
    else:
        for n_components in range(n_min, n_max + 1):
            try:
                model = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    random_state=seed,
                )
                model.fit(sample)
                rows.append({
                    "algorithm": "gmm",
                    "param": f"n_components={n_components}",
                    "bic": float(model.bic(sample)),
                    "aic": float(model.aic(sample)),
                    "sample_size": int(len(sample)),
                    "selected": False,
                    "note": "GMM structural diagnostic only; not a cluster label source",
                })
            except Exception as exc:
                rows.append({
                    "algorithm": "gmm",
                    "param": f"n_components={n_components}",
                    "bic": np.nan,
                    "aic": np.nan,
                    "sample_size": int(len(sample)),
                    "selected": False,
                    "note": f"GMM diagnostic failed: {type(exc).__name__}; not a label",
                })
    diag = pd.DataFrame(rows)
    if not diag.empty and diag["bic"].notna().any():
        best_idx = diag["bic"].astype(float).idxmin()
        diag.loc[best_idx, "selected"] = True
    diag[GMM_COLS].to_csv(out / "gmm_diag.csv", index=False, encoding="utf-8-sig")
    return diag


def _hdbscan_class():
    try:
        from sklearn.cluster import HDBSCAN

        return HDBSCAN, "sklearn.cluster.HDBSCAN"
    except Exception:
        try:
            from hdbscan import HDBSCAN

            return HDBSCAN, "hdbscan.HDBSCAN"
        except Exception:
            return None, None


def _write_hdbscan_diag(xs, out, cfg):
    cls, module_name = _hdbscan_class()
    hdb_cfg = cfg.get("minute_cluster", {}).get("hdbscan", {})
    sample_size = int(hdb_cfg.get("diagnostic_sample_size", 10000))
    if len(xs) <= sample_size:
        sample = xs
    else:
        rng = np.random.default_rng(int(cfg.get("minute_cluster", {}).get("diagnostic_seed", 42)))
        sample = xs[rng.choice(np.arange(len(xs)), size=sample_size, replace=False)]
    sizes = hdb_cfg.get("min_sizes", cfg.get("hdbscan", {}).get("min_sizes", [5, 8, 10]))
    rows = []
    if cls is None:
        rows.append({
            "algorithm": "hdbscan",
            "param": "not_available",
            "n_clusters": np.nan,
            "noise_share": np.nan,
            "sample_size": int(len(sample)),
            "selected": False,
            "note": "HDBSCAN package not available; diagnostic skipped",
        })
    elif len(sample) < 5:
        rows.append({
            "algorithm": "hdbscan",
            "param": "not_run",
            "n_clusters": np.nan,
            "noise_share": np.nan,
            "sample_size": int(len(sample)),
            "selected": False,
            "note": "HDBSCAN diagnostic skipped because there were too few rows; not a label",
        })
    else:
        for min_size in [int(x) for x in sizes if int(x) > 1]:
            try:
                model = cls(min_cluster_size=min_size)
                labels = model.fit_predict(sample)
                labels = np.asarray(labels)
                non_noise = labels[labels >= 0]
                rows.append({
                    "algorithm": "hdbscan",
                    "param": f"min_cluster_size={min_size}",
                    "n_clusters": int(len(np.unique(non_noise))),
                    "noise_share": float(np.mean(labels < 0)),
                    "sample_size": int(len(sample)),
                    "selected": False,
                    "note": f"{module_name} structural diagnostic only; not a label source",
                })
            except Exception as exc:
                rows.append({
                    "algorithm": "hdbscan",
                    "param": f"min_cluster_size={min_size}",
                    "n_clusters": np.nan,
                    "noise_share": np.nan,
                    "sample_size": int(len(sample)),
                    "selected": False,
                    "note": f"HDBSCAN diagnostic failed: {type(exc).__name__}; not a label",
                })
    diag = pd.DataFrame(rows)
    if not diag.empty and diag["n_clusters"].notna().any():
        eligible = diag.loc[pd.to_numeric(diag["n_clusters"], errors="coerce").gt(1)].copy()
        if not eligible.empty:
            best_idx = eligible.sort_values(["noise_share", "n_clusters"], ascending=[True, False]).index[0]
            diag.loc[best_idx, "selected"] = True
    diag[HDBSCAN_COLS].to_csv(out / "hdbscan_diag.csv", index=False, encoding="utf-8-sig")
    return diag


def add_minute_clusters(minute_df, out, cfg):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    s = minute_df.copy()
    mc_cfg, km_cfg = _cluster_cfg(cfg)
    if s.empty or not _enabled(km_cfg.get("enabled", True)):
        return _empty_outputs(out, s, "Minute behavior clustering skipped because KMeans is disabled or data is empty.")
    if "session_key" not in s.columns:
        s["session_key"] = s["run_id"].astype(str) + "_" + s["broad_no"].astype(str)

    feats = [c for c in _features(cfg) if c in s.columns]
    if len(feats) < 2 or len(s) < 3:
        return _empty_outputs(out, s, "Minute behavior clustering skipped because there were too few eligible features or rows.")

    x = _make_x(s, feats)
    scaler_name = str(mc_cfg.get("scaler", "RobustScaler"))
    scaler = RobustScaler()
    xs = scaler.fit_transform(x)
    gmm_diag = _write_gmm_diag(xs, out, cfg)
    hdb_diag = _write_hdbscan_diag(xs, out, cfg)
    select, best_k, labels, model = _run_kmeans(s, xs, km_cfg)
    if labels is None:
        return _empty_outputs(out, s, "KMeans did not produce a valid minute behavior clustering.")
    best_param = f"k={best_k}"
    _write_model_selection(out, feats, scaler_name, select, gmm_diag, hdb_diag, best_param)

    s["minute_cluster"] = labels
    profile = _profile(s)
    s = s.merge(
        profile[["cluster_id", "cluster_mismatch_rank"]].rename(columns={"cluster_id": "minute_cluster"}),
        on="minute_cluster",
        how="left",
    )

    select = select[SELECT_COLS]

    select.to_csv(out / "mc_select.csv", index=False, encoding="utf-8-sig")
    s[ASSIGN_COLS].to_csv(out / "mc_assign.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(out / "mc_profile.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"features": feats, "scaler": scaler, "kmeans": model, "best_k": best_k}, out / "mc_kmeans.joblib")

    lines = [
        "Minute behavior cluster outputs written.",
    ]
    text = "\n".join(lines) + "\n"
    (out / "mc_info.txt").write_text(text, encoding="utf-8")
    (out / "minute_cluster.txt").write_text(text, encoding="utf-8")
    _write_minute_cluster_doc(out, cfg)
    return s
