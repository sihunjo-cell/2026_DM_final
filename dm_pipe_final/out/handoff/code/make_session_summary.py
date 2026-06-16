import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler


KEY = ["run_id", "broad_no"]
CONFIG = {'features': ['log_viewer', 'log_chat', 'log_unique', 'zero_rate', 'log_zrun', 'gap_med', 'gap_max'], 'scaler': 'RobustScaler', 'k_min': 2, 'k_max': 6, 'random_state': 42, 'n_init': 50, 'min_n': 10, 'clock_gap_reset_min': 1.1, 'cluster_number_semantics': '세션 행동 군집 번호이며 정답 라벨이나 확률이 아님'}
OUT_COLS = ['run_id', 'broad_no', 'session_key', 'user_id', 'category_id', 'n', 'start', 'end', 'viewer_med', 'viewer_max', 'chat_mean', 'unique_mean', 'zero_rate', 'zrun_max', 'gap_med', 'gap_max', 'log_viewer', 'log_chat', 'log_unique', 'log_zrun', 'cluster_number']


def id_str(s):
    x = s.astype("string").str.strip().replace("", pd.NA)
    as_num = pd.to_numeric(x, errors="coerce")
    int_like = as_num.notna() & np.isclose(as_num, np.floor(as_num))
    x = x.mask(int_like, as_num.round().astype("Int64").astype("string"))
    return x


def zero_run_clock(g):
    z = g["zero"].fillna(False).astype(bool)
    gap = (
        pd.to_datetime(g["minute_ts"], errors="coerce")
        .diff()
        .dt.total_seconds()
        .div(60)
        .gt(float(CONFIG.get("clock_gap_reset_min", 1.1)))
        .fillna(False)
    )
    block = (z.ne(z.shift()) | gap).cumsum()
    return z.groupby(block).cumcount().add(1).where(z, 0).astype(int)


def read_minute(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["minute_ts"] = pd.to_datetime(df["minute_ts"], errors="coerce")
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df["broad_no"] = id_str(df["broad_no"])
    df = df.dropna(subset=["run_id", "broad_no", "minute_ts"]).copy()
    df["run_id"] = df["run_id"].astype(int)
    df = df.sort_values(KEY + ["minute_ts"]).reset_index(drop=True)
    audit = {"path": str(path), "rows_after_key_filter": int(len(df))}
    for col in ["viewer_count_last", "chat_count", "unique_chatters"]:
        if col not in df.columns:
            df[col] = np.nan
        values = pd.to_numeric(df[col], errors="coerce")
        audit[f"missing_{col}_rows"] = int(values.isna().sum())
        audit[f"negative_{col}_rows"] = int(values.lt(0).sum())
        values = values.mask(values.lt(0))
        if col == "viewer_count_last":
            values = values.groupby([df["run_id"], df["broad_no"]]).transform(lambda s: s.ffill().bfill())
            audit["viewer_count_last_missing_after_ffill_bfill_rows"] = int(values.isna().sum())
            df[col] = values.fillna(0).clip(lower=0)
        else:
            df[col] = values.fillna(0).clip(lower=0)
    for col, default in [("user_id", "UNKNOWN_USER"), ("category_id", "UNKNOWN_CAT")]:
        if col not in df.columns:
            df[col] = default
        df[col] = df.groupby(KEY)[col].transform(lambda s: s.ffill().bfill()).fillna(default)
    if "session_key" not in df.columns:
        df["session_key"] = df["run_id"].astype(str) + "_" + df["broad_no"].astype(str)
    return df, audit


def build_session(minute):
    df = minute.copy()
    df["log_viewer_min"] = np.log1p(df["viewer_count_last"])
    df["log_chat_min"] = np.log1p(df["chat_count"])
    df["gap"] = df["log_viewer_min"] - df["log_chat_min"]
    df["zero"] = df["chat_count"].eq(0)
    df["zrun"] = 0
    for _, idx in df.groupby(KEY, sort=False).groups.items():
        df.loc[idx, "zrun"] = zero_run_clock(df.loc[idx]).to_numpy()
    sess = df.groupby(KEY).agg(
        session_key=("session_key", "first"),
        user_id=("user_id", "first"),
        category_id=("category_id", "first"),
        n=("minute_ts", "size"),
        start=("minute_ts", "min"),
        end=("minute_ts", "max"),
        viewer_med=("viewer_count_last", "median"),
        viewer_max=("viewer_count_last", "max"),
        chat_mean=("chat_count", "mean"),
        unique_mean=("unique_chatters", "mean"),
        zero_rate=("zero", "mean"),
        zrun_max=("zrun", "max"),
        gap_med=("gap", "median"),
        gap_max=("gap", "max"),
    ).reset_index()
    sess["log_viewer"] = np.log1p(sess["viewer_med"])
    sess["log_chat"] = np.log1p(sess["chat_mean"])
    sess["log_unique"] = np.log1p(sess["unique_mean"])
    sess["log_zrun"] = np.log1p(sess["zrun_max"])
    return sess


def make_x(sess, features):
    work = sess.copy()
    for col in features:
        if col not in work.columns:
            work[col] = np.nan
    x = work[features].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True)).fillna(0)


def cluster_size_balance(labels):
    counts = pd.Series(labels).value_counts()
    if counts.empty or counts.max() == 0:
        return np.nan
    return float(counts.min() / counts.max())


def profile_separation(xs, labels):
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


def good_rank(s, higher_is_better=True):
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if vals.notna().sum() == 0:
        return pd.Series(np.nan, index=vals.index, dtype=float)
    return vals.rank(ascending=True if higher_is_better else False, method="average", pct=True)


def add_selection_score(table):
    if table.empty:
        return table
    work = table.copy()
    metrics = {
        "silhouette": True,
        "calinski_harabasz": True,
        "davies_bouldin": False,
        "cluster_size_balance": True,
        "cluster_profile_separation": True,
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


def choose_k(xs, cfg, fixed_k=None):
    if len(xs) < 3:
        return 1, pd.DataFrame(columns=["k", "silhouette", "inertia", "selected"])
    if fixed_k is not None:
        k = max(1, min(int(fixed_k), len(xs) - 1))
        return k, pd.DataFrame([{"k": k, "silhouette": np.nan, "inertia": np.nan, "selected": True}])
    k_min = int(cfg.get("k_min", 2))
    k_max = min(int(cfg.get("k_max", 6)), len(xs) - 1)
    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(
            n_clusters=k,
            random_state=int(cfg.get("random_state", 42)),
            n_init=int(cfg.get("n_init", 50)),
        )
        labels = model.fit_predict(xs)
        if len(np.unique(labels)) >= 2:
            rows.append({
                "k": k,
                "silhouette": float(silhouette_score(xs, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(xs, labels)),
                "davies_bouldin": float(davies_bouldin_score(xs, labels)),
                "cluster_size_balance": cluster_size_balance(labels),
                "cluster_profile_separation": profile_separation(xs, labels),
                "inertia": float(model.inertia_),
                "selected": False,
            })
    if not rows:
        return 1, pd.DataFrame(columns=["k", "silhouette", "inertia", "selected"])
    table = add_selection_score(pd.DataFrame(rows))
    selected_k = int(table.sort_values(["selection_score", "silhouette", "k"], ascending=[False, False, True]).iloc[0]["k"])
    table.loc[table["k"].eq(selected_k), "selected"] = True
    return selected_k, table


def sparse_cluster_id(profile, selected_k):
    if selected_k <= 1 or profile.empty:
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


def behavior_alias(cluster_number, is_sparse):
    if pd.isna(cluster_number):
        return "cluster_unknown"
    cid = int(cluster_number)
    return f"cluster_{cid}_sparse_silent_like" if bool(is_sparse) else f"cluster_{cid}_behavior_state"


def add_cluster(sess, cfg, fixed_k=None):
    out = sess.copy()
    if len(out) < 2:
        out["cluster_number"] = 0
        return out, 1, pd.DataFrame(columns=["k", "silhouette", "inertia", "selected"])
    xs = RobustScaler().fit_transform(make_x(out, cfg.get("features", [])))
    k, silhouette_table = choose_k(xs, cfg, fixed_k=fixed_k)
    if k < 2:
        raw = np.zeros(len(out), dtype=int)
    else:
        raw = KMeans(
            n_clusters=k,
            random_state=int(cfg.get("random_state", 42)),
            n_init=int(cfg.get("n_init", 50)),
        ).fit_predict(xs)
    out["cluster_number"] = raw
    prof = out.groupby("cluster_number").agg(
        zero_rate=("zero_rate", "mean"),
        gap_med=("gap_med", "median"),
        zrun_max=("zrun_max", "median"),
        chat_mean=("chat_mean", "mean"),
        unique_mean=("unique_mean", "mean"),
    ).reset_index()
    return out, k, silhouette_table


def _audit_report_lines(label, audit):
    return [
        f"{label} key/time filter 후 row 수: {audit.get('rows_after_key_filter')}",
        f"{label} viewer_count_last 원본 결측 row 수: {audit.get('missing_viewer_count_last_rows')}",
        f"{label} viewer_count_last ffill/bfill 후 잔여 결측 row 수: {audit.get('viewer_count_last_missing_after_ffill_bfill_rows')}",
        f"{label} chat_count 결측 row 수: {audit.get('missing_chat_count_rows')}",
        f"{label} unique_chatters 결측 row 수: {audit.get('missing_unique_chatters_rows')}",
        f"{label} viewer_count_last 음수 row 수: {audit.get('negative_viewer_count_last_rows')}",
        f"{label} chat_count 음수 row 수: {audit.get('negative_chat_count_rows')}",
        f"{label} unique_chatters 음수 row 수: {audit.get('negative_unique_chatters_rows')}",
    ]


def write_report(output_path, cfg, selected_k, silhouette_table, sess_before, sess_after, fixed_k, audit_all, audit_model):
    report_path = Path(output_path).with_name("session_summary_cluster_report.txt")
    lines = [
        "세션 요약 군집 재생성 보고서",
        "==========================",
        "",
        "minute_all.csv는 결측/음수 audit 확인에만 사용하고, 세션 요약은 minute_model.csv에서 생성한다.",
        f"사용 feature: {', '.join(cfg.get('features', []))}",
        f"스케일링: {cfg.get('scaler', 'RobustScaler')}",
        f"K 후보: {cfg.get('k_min')}..{cfg.get('k_max')}",
        f"선택된 K: {selected_k}",
        "K 선택 규칙: 고정 K 옵션 사용" if fixed_k is not None else "K 선택 규칙: 방향 보정 percentile rank composite selection_score 내림차순, 동률이면 silhouette 내림차순, 그 다음 작은 K",
        f"random_state: {cfg.get('random_state')}",
        f"n_init: {cfg.get('n_init')}",
        f"min_n: {cfg.get('min_n')}",
        f"min_n filter 전 session 수: {sess_before}",
        f"min_n filter 후 session 수: {sess_after}",
        "cluster_number 의미: KMeans 기반 세션 행동 군집 번호이며 정답 라벨이나 확률이 아님",
        f"zero-run clock gap reset: run_id + broad_no 내부 minute_ts.diff() > {cfg.get('clock_gap_reset_min', 1.1)}분이면 reset",
        "zrun_max 의미: clock-contiguous zero-chat 구간의 최대 길이",
        "",
        "결측 처리 정책:",
        "- viewer_count_last: minute_ts 정렬 후 run_id + broad_no 내부 ffill/bfill을 먼저 적용하고 남은 결측은 0으로 채운다.",
        "- chat_count: 결측은 0으로 채운다.",
        "- unique_chatters: 결측은 0으로 채운다.",
        "",
        "결측/음수 audit:",
        *_audit_report_lines("minute_all", audit_all),
        *_audit_report_lines("minute_model", audit_model),
        "",
        "K 후보별 selection_score table:",
    ]
    if silhouette_table.empty:
        lines.append("- not available")
    else:
        for _, row in silhouette_table.sort_values("k").iterrows():
            sil = row.get("silhouette")
            ch = row.get("calinski_harabasz")
            db = row.get("davies_bouldin")
            score = row.get("selection_score")
            inertia = row.get("inertia")
            sil_text = "nan" if pd.isna(sil) else f"{float(sil):.8g}"
            ch_text = "nan" if pd.isna(ch) else f"{float(ch):.8g}"
            db_text = "nan" if pd.isna(db) else f"{float(db):.8g}"
            score_text = "nan" if pd.isna(score) else f"{float(score):.8g}"
            inertia_text = "nan" if pd.isna(inertia) else f"{float(inertia):.8g}"
            lines.append(f"- k={int(row['k'])}, silhouette={sil_text}, Calinski-Harabasz={ch_text}, Davies-Bouldin={db_text}, selection_score={score_text}, inertia={inertia_text}, selected={bool(row.get('selected', False))}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-all", required=True)
    parser.add_argument("--minute-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-n", type=int, default=None)
    parser.add_argument("--fixed-k", type=int, default=None)
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.min_n is not None:
        cfg["min_n"] = int(args.min_n)

    minute_all, audit_all = read_minute(args.minute_all)
    minute_model, audit_model = read_minute(args.minute_model)
    sess = build_session(minute_model)
    sess_before = len(sess)
    sess = sess[sess["n"] >= int(cfg["min_n"])].copy()
    sess_after = len(sess)
    sess, selected_k, silhouette_table = add_cluster(sess, cfg, fixed_k=args.fixed_k)
    for col in OUT_COLS:
        if col not in sess.columns:
            sess[col] = pd.NA
    sess[OUT_COLS].to_csv(args.out, index=False, encoding="utf-8-sig")
    write_report(args.out, cfg, selected_k, silhouette_table, sess_before, sess_after, args.fixed_k, audit_all, audit_model)
    print("분 단위 입력 파일을 읽었습니다.")
    print("minute_all.csv는 audit 확인에만 사용합니다.")
    print(f"minute_all 입력 크기: {minute_all.shape}")
    print(f"minute_model 입력 크기: {minute_model.shape}")
    print("결측 row 수:")
    for label, audit in [("minute_all", audit_all), ("minute_model", audit_model)]:
        print(f"{label} viewer_count_last 결측 row 수: {audit.get('missing_viewer_count_last_rows')}")
        print(f"{label} chat_count 결측 row 수: {audit.get('missing_chat_count_rows')}")
        print(f"{label} unique_chatters 결측 row 수: {audit.get('missing_unique_chatters_rows')}")
    print(f"min_n 값: {cfg['min_n']}")
    print(f"min_n 적용 전 세션 수: {sess_before}")
    print(f"min_n 적용 후 세션 수: {sess_after}")
    print(f"선택된 K: {selected_k}")
    print(f"session_summary_processed.csv를 생성했습니다: {sess[OUT_COLS].shape}")
    print("cluster_number 빈도:")
    print(sess["cluster_number"].value_counts(dropna=False).sort_index().to_string())
    print(f"출력 경로: {args.out}")


if __name__ == "__main__":
    main()
