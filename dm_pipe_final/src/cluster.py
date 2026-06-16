import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
try:
    from sklearn.cluster import HDBSCAN
except Exception:
    HDBSCAN = None
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
import joblib

from src.k_selection import good_rank


FEATS = ["log_viewer", "log_chat", "log_unique", "zero_rate", "log_zrun", "gap_med", "gap_max"]
ASSIGN_COLS = [
    "session_key",
    "run_id",
    "broad_no",
    "cluster_number",
    "gmm",
    "hdbscan",
    "hdbscan_noise",
]


def make_x(df, feats=FEATS):
    x = df[feats].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True)).fillna(0)


def _sparse_cluster_id(profile):
    required = {"cluster_number", "zero_rate", "gap_med", "zrun_max", "chat_mean", "unique_mean"}
    if profile.empty or not required.issubset(profile.columns):
        return None
    prof = profile.set_index("cluster_number")
    sparse_score = (
        prof["zero_rate"].rank(pct=True)
        + prof["gap_med"].rank(pct=True)
        + prof["zrun_max"].rank(pct=True)
        + (-prof["chat_mean"]).rank(pct=True)
        + (-prof["unique_mean"]).rank(pct=True)
    ) / 5
    if sparse_score.dropna().empty:
        return None
    return sparse_score.idxmax()


def _cluster_size_balance(labels):
    labels = pd.Series(labels)
    counts = labels.value_counts()
    if counts.empty or counts.max() == 0:
        return np.nan
    return float(counts.min() / counts.max())


def _profile_separation(xs, labels):
    work = pd.DataFrame(xs)
    work["_cluster"] = labels
    centers = work.groupby("_cluster").median(numeric_only=True)
    if len(centers) < 2:
        return np.nan
    dists = []
    vals = centers.to_numpy(dtype=float)
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            dists.append(float(np.linalg.norm(vals[i] - vals[j])))
    return float(np.mean(dists)) if dists else np.nan


def _kmeans_selection_score(sel):
    if sel.empty:
        return sel
    work = sel.copy()
    metrics = {
        "silhouette": True,
        "calinski_harabasz": True,
        "davies_bouldin": False,
        "cluster_size_balance": True,
        "cluster_profile_separation": True,
    }
    rank_cols = []
    for col, higher_is_better in metrics.items():
        if col not in work.columns or pd.to_numeric(work[col], errors="coerce").notna().sum() == 0:
            continue
        rcol = f"_{col}_rank"
        work[rcol] = good_rank(work[col], higher_is_better=higher_is_better)
        rank_cols.append(rcol)
    work["selection_score"] = work[rank_cols].mean(axis=1, skipna=True) if rank_cols else np.nan
    return work.drop(columns=rank_cols, errors="ignore")


def write_doc(out, cfg, k_min, k_max, best_k, gmm_note="", hdb_note="", note=""):
    cluster_cfg = cfg.get("cluster", {})
    scaler_name = cluster_cfg.get("scaler", "RobustScaler")
    try:
        select = pd.read_csv(Path(out) / "cluster_select.csv", encoding="utf-8-sig")
    except Exception:
        select = pd.DataFrame()
    try:
        profile = pd.read_csv(Path(out) / "cluster_profile.csv", encoding="utf-8-sig")
    except Exception:
        profile = pd.DataFrame()

    select_lines = []
    if select.empty:
        select_lines.append("- 후보 K 평가표가 비어 있다.")
    else:
        cols = [c for c in ["k", "silhouette", "calinski_harabasz", "davies_bouldin", "cluster_size_balance", "cluster_profile_separation", "selection_score", "selected"] if c in select.columns]
        for _, row in select.sort_values("k").iterrows():
            parts = []
            for col in cols:
                val = row.get(col)
                if isinstance(val, float):
                    val = f"{val:.4g}"
                parts.append(f"{col}={val}")
            select_lines.append("- " + ", ".join(parts))

    profile_lines = []
    if profile.empty:
        profile_lines.append("- 군집 profile을 만들 수 없었다.")
    else:
        sparse_cluster = _sparse_cluster_id(profile)
        for _, row in profile.sort_values("cluster_number").iterrows():
            cid = row.get("cluster_number")
            meaning = "viewer 대비 채팅 반응 약한 세션 행동 군집" if sparse_cluster is not None and cid == sparse_cluster else "상대적으로 채팅 반응이 활발하거나 혼합된 세션 행동 군집"
            metrics = []
            for col in ["n", "viewer_med", "chat_mean", "unique_mean", "zero_rate", "gap_med", "zrun_max"]:
                if col in row.index:
                    val = row.get(col)
                    if isinstance(val, float):
                        val = f"{val:.4g}"
                    metrics.append(f"{col}={val}")
            profile_lines.append(f"- cluster_number={cid}: {meaning}; " + ", ".join(metrics))

    lines = [
        "세션 행동 군집 문서",
        "==================",
        "",
        "분석 단위: 방송 세션.",
        "세션 정의: run_id + broad_no.",
        "session_summary_processed.csv 생성 방식: minute_model.csv의 1분 row를 세션 단위로 집계하고, 설정 파일의 최소 관측 길이 기준을 통과한 세션에 대해 비지도 군집 번호를 부여한다.",
        "cluster_number의 의미: 세션 요약 feature로 계산한 세션 행동 군집 번호이며, 정답 라벨이나 확률이 아니다.",
        "",
        "사용 feature:",
        *[f"- {feature}" for feature in FEATS],
        "",
        "스케일링:",
        f"- 적용: 적용",
        f"- 사용 방식: {scaler_name}",
        "- 선택 이유: viewer/chat 계열 변수의 heavy-tail과 극단값 영향을 줄이기 위해 중앙값/IQR 기반 스케일링을 사용한다.",
        "",
        "사용한 clustering 알고리즘:",
        "- KMeans: 최종 cluster_number 후보.",
        "- GMM: component 기반 하위 구조 진단.",
        "- HDBSCAN: 밀도 기반 안정 군집과 noise 비중 진단. 여기서 noise는 이상치 확정이 아니라 안정 군집에 속하지 않은 상태를 뜻한다.",
        "",
        "후보 하이퍼파라미터:",
        f"- KMeans K 후보: {k_min}..{k_max}",
        f"- KMeans random_state: {cfg['cluster']['seed']}",
        f"- KMeans n_init: {cfg['cluster']['n_init']}",
        f"- GMM n_components 후보: {gmm_note}",
        f"- HDBSCAN min_cluster_size 후보: {hdb_note}",
        "",
        "최종 선택:",
        f"- 선택된 알고리즘: KMeans",
        f"- 선택된 K: {best_k}",
        "- 선택 근거: silhouette, Calinski-Harabasz, Davies-Bouldin, 군집 크기 균형, 군집 profile 분리도를 함께 비교하여 해석 안정성이 높은 K를 선택했다.",
        "",
        "K 후보별 평가:",
        *select_lines,
        "",
        "군집 번호별 profile 요약:",
        *profile_lines,
        "",
        "주의:",
        "- cluster_number는 정답 라벨이나 확률이 아니라 세션 행동 군집 번호다.",
        "- viewer 대비 채팅 반응이 약한 profile은 수동 검토 근거 중 하나일 뿐이며 확정 판정이 아니다.",
    ]
    if note:
        lines.append(f"비고: {note}")
    Path(out, "cluster.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_gmm(s, xs, cfg):
    if len(s) < 4:
        s["gmm"] = np.nan
        return s, pd.DataFrame(columns=["n", "bic", "aic"]), "not available"
    rows, labels = [], {}
    for n in range(2, min(6, len(s) - 1) + 1):
        gm = GaussianMixture(n_components=n, random_state=int(cfg["cluster"]["seed"]), n_init=10)
        lab = gm.fit_predict(xs)
        rows.append({"n": n, "bic": gm.bic(xs), "aic": gm.aic(xs)})
        labels[n] = lab
    sel = pd.DataFrame(rows)
    best = int(sel.sort_values(["bic", "n"]).iloc[0]["n"])
    s["gmm"] = labels[best]
    return s, sel, str(best)


def add_hdbscan(s, xs, cfg):
    if HDBSCAN is None or len(s) < 5:
        s["hdbscan"] = np.nan
        s["hdbscan_noise"] = np.nan
        return s, pd.DataFrame(), "not available"
    sizes = cfg.get("hdbscan", {}).get("min_sizes", [5, 8, 10])
    sizes = sorted({int(v) for v in sizes if 2 <= int(v) < len(s)})
    if not sizes:
        sizes = [max(2, min(5, len(s) - 1))]
    rows, labels = [], {}
    for min_size in sizes:
        min_samples = max(2, min_size // 2)
        model = HDBSCAN(min_cluster_size=min_size, min_samples=min_samples, metric="euclidean", copy=True)
        lab = model.fit_predict(xs)
        labels[min_size] = lab
        non_noise = lab != -1
        n_noise = int((lab == -1).sum())
        n_cluster = len([x for x in np.unique(lab) if x != -1])
        sil = np.nan
        if n_cluster >= 2 and non_noise.sum() > n_cluster:
            sil = silhouette_score(xs[non_noise], lab[non_noise])
        noise_ratio = n_noise / len(lab)
        rows.append({
            "min_cluster_size": min_size,
            "min_samples": min_samples,
            "n_clusters": n_cluster,
            "n_noise": n_noise,
            "noise_ratio": noise_ratio,
            "coverage": 1 - noise_ratio,
            "silhouette": sil,
            "balanced_score": sil * (1 - noise_ratio) if np.isfinite(sil) else np.nan,
        })
    sel = pd.DataFrame(rows)
    valid = sel.dropna(subset=["balanced_score"])
    valid = valid[valid["n_clusters"].ge(2)]
    if valid.empty:
        best = int(sel.sort_values(["noise_ratio", "min_cluster_size"]).iloc[0]["min_cluster_size"])
    else:
        best = int(valid.sort_values(["balanced_score", "silhouette", "coverage"], ascending=[False, False, False]).iloc[0]["min_cluster_size"])
    s["hdbscan"] = labels[best]
    s["hdbscan_noise"] = s["hdbscan"].eq(-1).astype(int)
    return s, sel, str(best)


def add_cluster(df, out, cfg):
    out = Path(out)
    s = df.copy()
    k_min = int(cfg["cluster"]["k_min"])
    k_max_cfg = int(cfg["cluster"]["k_max"])

    if len(s) < 3:
        s["cluster_number"] = 0
        s["gmm"] = np.nan
        s["hdbscan"] = np.nan
        s["hdbscan_noise"] = np.nan
        pd.DataFrame(columns=["k", "silhouette", "calinski_harabasz", "davies_bouldin", "cluster_size_balance", "cluster_profile_separation", "inertia", "selection_score", "selected"]).to_csv(out / "cluster_select.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["n", "bic", "aic"]).to_csv(out / "gmm_select.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(out / "hdbscan_select.csv", index=False, encoding="utf-8-sig")
        s[ASSIGN_COLS].to_csv(out / "cluster_assignments.csv", index=False, encoding="utf-8-sig")
        write_doc(out, cfg, k_min, k_max_cfg, 1, note="fewer than 3 eligible sessions")
        return s.drop(columns=["gmm", "hdbscan", "hdbscan_noise"], errors="ignore")

    scaler = RobustScaler()
    xs = scaler.fit_transform(make_x(s))
    k_max = min(k_max_cfg, len(s) - 1)

    rows, models, labels = [], {}, {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=int(cfg["cluster"]["seed"]), n_init=int(cfg["cluster"]["n_init"]))
        lab = km.fit_predict(xs)
        if len(np.unique(lab)) < 2:
            continue
        rows.append({
            "k": k,
            "silhouette": silhouette_score(xs, lab),
            "calinski_harabasz": calinski_harabasz_score(xs, lab),
            "davies_bouldin": davies_bouldin_score(xs, lab),
            "cluster_size_balance": _cluster_size_balance(lab),
            "cluster_profile_separation": _profile_separation(xs, lab),
            "inertia": km.inertia_,
            "selected": False,
        })
        models[k] = km
        labels[k] = lab

    if not rows:
        s["cluster_number"] = 0
        best_k = 1
        sel = pd.DataFrame(columns=["k", "silhouette", "calinski_harabasz", "davies_bouldin", "cluster_size_balance", "cluster_profile_separation", "inertia", "selection_score", "selected"])
        note = "session features are nearly identical; single behavior cluster used"
    else:
        sel = _kmeans_selection_score(pd.DataFrame(rows))
        best_k = int(sel.sort_values(["selection_score", "silhouette", "k"], ascending=[False, False, True]).iloc[0]["k"])
        sel.loc[sel["k"].eq(best_k), "selected"] = True
        s["cluster_number"] = labels[best_k]
        note = ""

    s, g_sel, g_note = add_gmm(s, xs, cfg)
    s, h_sel, h_note = add_hdbscan(s, xs, cfg)

    profile = s.groupby("cluster_number").agg(
        n=("session_key", "size"),
        viewer_med=("viewer_med", "median"),
        chat_mean=("chat_mean", "mean"),
        unique_mean=("unique_mean", "mean"),
        zero_rate=("zero_rate", "mean"),
        gap_med=("gap_med", "median"),
        zrun_max=("zrun_max", "median"),
    ).reset_index()
    sparse_cluster = _sparse_cluster_id(profile) if best_k > 1 else None
    profile["cluster_is_sparse_silent_like"] = profile["cluster_number"].eq(sparse_cluster) if sparse_cluster is not None else False

    sel.to_csv(out / "cluster_select.csv", index=False, encoding="utf-8-sig")
    g_sel.to_csv(out / "gmm_select.csv", index=False, encoding="utf-8-sig")
    h_sel.to_csv(out / "hdbscan_select.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(out / "cluster_profile.csv", index=False, encoding="utf-8-sig")
    s[ASSIGN_COLS].to_csv(out / "cluster_assignments.csv", index=False, encoding="utf-8-sig")

    if rows:
        joblib.dump(models[best_k], out / "kmeans.joblib")
        joblib.dump({"features": FEATS, "scaler": scaler, "kmeans": models[best_k], "best_k": best_k}, out / "kmeans_bundle.joblib")

    write_doc(out, cfg, k_min, k_max, best_k, gmm_note=g_note, hdb_note=h_note, note=note)
    return s.drop(columns=["gmm", "hdbscan", "hdbscan_noise"], errors="ignore")
