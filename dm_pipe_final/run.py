import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile

import numpy as np
import pandas as pd
import yaml

from src.load import load_features
from src.prep import prep_minute, split_minute, raw_minute_cols, feat_minute_cols, write_qc_zero_session_review
from src.minute_state import add_minute_state_features
from src.minute_cluster import add_minute_clusters
from src.m2 import (
    build_m2_scores,
    build_m2_episodes,
    build_m2_sensitivity,
    build_m2_candidates,
    build_m2_eval,
    build_m2_stability,
    build_m2_null,
    build_m2_state,
    build_mc_stability,
    build_m2_synth,
    write_m2_docs,
)
from src.m2_baseline import build_expected_response
from src.m2_scan import build_m2_scan
from src.m2_interval import build_interval_anomaly
from src.m2_review import (
    build_m2_reason,
    build_m2_review,
    build_m2_review_plots,
)
from src.session import clock_gap_reset_min, make_session
from src.cluster import FEATS as SESSION_CLUSTER_FEATS, add_cluster
from src.plots import make_plots, write_plot_doc


SESSION_KEY_COLS = ["run_id", "broad_no"]

STRICT_MINUTE_CORE_COLS = [
    "source_file",
    "run_id",
    "broad_no",
    "session_key",
    "user_id",
    "category_id",
    "minute_ts",
    "viewer_count_last",
    "chat_count",
    "unique_chatters",
]


X_CORE_COLS = [
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
    "cluster_number",
]

X_NO_LEAK_COLS = [
    "review_stability",
    "max_episode_duration_ratio",
    "max_episode_score",
    "max_p95_minute_mismatch_score",
    "max_episode_count",
    "overall_session_review_rank_score",
    "overall_session_review_rank",
    "session_review_rank_score",
    "session_review_rank",
    "session_bucket",
    "minute_cluster",
    "cluster_mismatch_rank",
    "minute_mismatch_score",
    "minute_mismatch_rank",
    "m2_scores",
    "m2_ep",
    "m2_sens",
    "m2_candidates",
    "base_pred",
    "m2_scan",
    "int_scores",
    "m2_reason",
    "m2_patterns",
    "m2_review",
    "candidate_rank",
    "episode_count",
    "episode_total_duration_min",
    "episode_duration_ratio",
    "max_episode_score",
    "p95_minute_score",
    "threshold_q",
    "score_cutoff",
    "min_duration",
    "if_score",
    "if_rank",
    "if_dir_score",
    "lof_score",
    "lof_rank",
    "lof_dir_score",
    "ocsvm_score",
    "ocsvm_rank",
    "ocsvm_dir_score",
    "directional_weight",
    "rra_p",
    "rra_q",
    "review_order",
    "scan_interval_rank",
    "interval_anomaly_rank",
    "y_syn",
]

HANDOFF_SESSION_COLS = [
    "run_id",
    "broad_no",
    "session_key",
    "user_id",
    "category_id",
    "n",
    "start",
    "end",
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
    "cluster_number",
]


def save(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장했습니다: {path} {df.shape}")


def resolve_cfg_path(cfg_path="cfg.yml"):
    p = Path(cfg_path)
    if p.is_absolute():
        if p.exists():
            return p
        raise FileNotFoundError(f"cfg file not found: {p}")
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    script_candidate = (Path(__file__).resolve().parent / p).resolve()
    if script_candidate.exists():
        return script_candidate
    raise FileNotFoundError(f"cfg file not found: tried {cwd_candidate} and {script_candidate}")


def load_runtime_config(cfg_path="cfg.yml"):
    cfg_file = resolve_cfg_path(cfg_path)
    project_dir = cfg_file.parent
    with open(cfg_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    path_cfg = cfg.setdefault("path", {})
    for key in ["in_dir", "out_dir", "label_file"]:
        val = path_cfg.get(key)
        if val and not Path(val).is_absolute():
            path_cfg[key] = str((project_dir / val).resolve())
    return cfg, project_dir, cfg_file


def _numeric_sum(df, col):
    if col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def write_load_audit_summary(file_audit, out):
    out = Path(out)
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    fa = file_audit.copy()
    if fa.empty:
        summary = {
            "total_files": 0,
            "keep_all_files": 0,
            "trim_files": 0,
            "drop_major_files": 0,
            "no_valid_files": 0,
            "total_raw_rows": 0,
            "rows_after_basic_clean": 0,
            "rows_used_after_policy": 0,
            "rows_trimmed_off_window": 0,
            "rows_dropped_major_off_window": 0,
            "recovered_rows_by_trim_vs_old_policy": 0,
        }
    else:
        policy = fa.get("off_window_policy", pd.Series("", index=fa.index)).fillna("")
        reason = fa.get("reason", pd.Series("", index=fa.index)).fillna("")
        use = pd.to_numeric(fa.get("use", pd.Series(0, index=fa.index)), errors="coerce").fillna(0).astype(int)
        drop_major = reason.eq("drop_off_window_major")
        trim_rows = policy.eq("trim_rows")
        summary = {
            "total_files": int(len(fa)),
            "keep_all_files": int((policy.eq("keep_all") & use.eq(1)).sum()),
            "trim_files": int(trim_rows.sum()),
            "drop_major_files": int(drop_major.sum()),
            "no_valid_files": int(reason.eq("no_valid_rows").sum()),
            "total_raw_rows": _numeric_sum(fa, "rows_raw"),
            "rows_after_basic_clean": _numeric_sum(fa, "rows_after_basic_clean"),
            "rows_used_after_policy": _numeric_sum(fa, "rows_used_after_trim"),
            "rows_trimmed_off_window": _numeric_sum(fa, "rows_trimmed_off_window"),
            "rows_dropped_major_off_window": _numeric_sum(fa.loc[drop_major], "rows_after_basic_clean"),
            "recovered_rows_by_trim_vs_old_policy": _numeric_sum(fa.loc[trim_rows], "rows_used_after_trim"),
        }
    save(pd.DataFrame([summary]), table_dir / "load_audit_sum.csv")


def write_load_policy(out, cfg, file_audit):
    out = Path(out)
    time_cfg = cfg.get("time", {})
    fa = file_audit.copy() if file_audit is not None else pd.DataFrame()
    if fa.empty:
        counts = {"keep_all": 0, "trim_rows": 0, "drop_file": 0}
        active_targets = 0
    else:
        policy = fa.get("off_window_policy", pd.Series("", index=fa.index)).fillna("")
        counts = {
            "keep_all": int(policy.eq("keep_all").sum()),
            "trim_rows": int(policy.eq("trim_rows").sum()),
            "drop_file": int(policy.eq("drop_file").sum()),
        }
        active_targets = int(pd.to_numeric(fa.get("use", pd.Series(0, index=fa.index)), errors="coerce").fillna(0).astype(int).sum())
    audit = pd.DataFrame([{
        "active_target_count": active_targets,
        "source": "file_audit.csv use=1",
        **{f"{k}_files": v for k, v in counts.items()},
    }])
    save(audit, out / "active_target_audit.csv")
    lines = [
        "# 데이터 로딩 및 시간 window 필터링 정책",
        "",
        "목적: 원천 feature 파일을 읽고 분석 대상 시간 window 밖의 row를 일관된 규칙으로 정리한다.",
        f"허용 window: {time_cfg.get('valid_windows')}",
        f"허용 오차(분): {time_cfg.get('tolerance_min')}",
        f"window 밖 row 허용 비율: {time_cfg.get('off_window_max_rate')}",
        f"window 밖 row trim 여부: {time_cfg.get('trim_off_window_rows')}",
        f"window 밖 파일 drop 여부: {time_cfg.get('drop_off_window')}",
        "",
        "file_audit.csv 기준 파일 처리 수:",
        f"- 전체 유지 파일: {counts['keep_all']}",
        f"- row trim 파일: {counts['trim_rows']}",
        f"- 제외 파일: {counts['drop_file']}",
        f"- active target 파일: {active_targets}",
        "",
        "주의: 이 단계는 데이터 품질 정리 절차이며 정답 라벨이나 확률을 만들지 않는다.",
    ]
    (out / "load_policy.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_obsolete_outputs(out):
    """Remove generated artifacts from the old anchor/session-modeling path."""
    out = Path(out)
    obsolete_files = [
        "minute_scores.csv",
        "episode_candidates_all_thresholds.csv",
        "threshold_calibration.csv",
        "session_review_candidates.csv",
        "session_review_summary.csv",
        "episode_definition.txt",
        "final_presentation_slide_plan.md",
        "final_task_slide_plan.md",
        "task_definition.md",
        "data_dictionary.md",
        "modeling_handoff.md",
        "pipeline_definition.md",
        "score.txt",
        "model_metrics.csv",
        "scores_model.csv",
        "scores_unsup.csv",
        "synthetic_intervals.csv",
        "m2_synth.csv",
        "m2_synth_matches.csv",
        "syn_minute.csv",
        "syn_train.csv",
        "synthetic_injection.txt",
        "minute_cluster_select.csv",
        "minute_cluster_profile.csv",
        "minute_cluster_assignments.csv",
        "minute_kmeans_bundle.joblib",
        "ml_scores.csv",
        "ml_info.txt",
        "m2_src_overlap.csv",
        "m2_src_corr.csv",
        "mc_gmm.csv",
        "mc_hdbscan.csv",
        "m2_rules.csv",
        "m2_window.csv",
        "win_scores.csv",
        "win_ep.csv",
        "mp_disc.csv",
        "discord_info.txt",
    ]
    obsolete_tables = [
        "cal_sum.csv",
        "sc_rank.csv",
        "top_sess.csv",
        "ep_ex.csv",
        "ms_sum.csv",
        "mc_prof.csv",
        "eda_tables.xlsx",
        "cluster_profile.csv",
        "corr_table.csv",
        "lag_corr.csv",
        "metric_table.csv",
        "quality_table.csv",
    ]
    obsolete_plots = [
        "05_detectors.png",
        "06_models.png",
        "14_m2_ml.png",
        "17_window_cluster.png",
        "21_window.png",
        "22_discord.png",
    ]
    for name in obsolete_files:
        path = out / name
        if path.exists():
            path.unlink()
    for name in obsolete_tables:
        path = out / "tables" / name
        if path.exists():
            path.unlink()
    for name in obsolete_plots:
        path = out / "plots" / name
        if path.exists():
            path.unlink()


def remove_legacy_episode_outputs(out):
    out = Path(out)
    for name in [
        "m2_ep.csv",
        "m2_sens.csv",
        "m2_candidates.csv",
        "m2_eval.csv",
        "m2_stability.csv",
    ]:
        path = out / name
        if path.exists():
            path.unlink()
    for name in ["10_sens.png", "11_ep.png", "12_m2_rank.png"]:
        path = out / "plots" / name
        if path.exists():
            path.unlink()


def _only_value(path, col, value):
    path = Path(path)
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, usecols=[col], encoding="utf-8-sig")
    except Exception:
        return False
    vals = set(df[col].dropna().astype(str).unique())
    return vals == {str(value)}


def remove_existing_archives(project_dir):
    project_dir = Path(project_dir)
    for name in ARCHIVE_NAMES:
        path = project_dir / name
        if path.exists():
            path.unlink()


def _zip_names(zip_path):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return set()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return {name.replace("\\", "/") for name in zf.namelist()}
    except Exception:
        return set()


def _zip_entry_bytes(zip_path, arcname):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return zf.read(arcname)
    except Exception:
        return None


def _zip_entry_text(zip_path, arcname):
    data = _zip_entry_bytes(zip_path, arcname)
    if data is None:
        return ""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return ""


def _required_handoff_manifest_text():
    manifest_lines = [
        "최종 전달용 필수자료 manifest",
        "",
        *sorted(REQUIRED_HANDOFF_FILES),
        "",
        "Method 2 folder는 포함하지 않는다.",
        "m2_*.csv, base_pred.csv, int_scores.csv, rebuilt CSV는 포함하지 않는다.",
    ]
    return "\n".join(manifest_lines) + "\n"


def _required_handoff_zip_matches_current(zip_path, handoff):
    if not Path(zip_path).exists():
        return False
    for arcname, rel_parts in REQUIRED_HANDOFF_FILES.items():
        data = _zip_entry_bytes(zip_path, arcname)
        if data is None:
            return False
        if rel_parts is None:
            if data != _required_handoff_manifest_text().encode("utf-8"):
                return False
            continue
        src = Path(handoff).joinpath(*rel_parts)
        if not src.exists() or data != src.read_bytes():
            return False
    return True


def _zip_has_no_windows_paths(zip_path):
    names = _zip_names(zip_path)
    if any("\\" in name or ":\\" in name or ":/" in name for name in names):
        return False
    text_entries = [
        "df_required_handoff/txt/session_cluster.txt",
        "df_required_handoff/code/make_session_summary.py",
    ]
    return all(":\\" not in _zip_entry_text(zip_path, name) and ":/" not in _zip_entry_text(zip_path, name) for name in text_entries)


def _forbidden_archive_member(arcname):
    normalized = str(arcname).replace("\\", "/")
    lower = normalized.lower()
    parts = lower.split("/")
    filename = parts[-1] if parts else lower
    return (
        "__pycache__" in parts
        or lower.endswith(".pyc")
        or lower.endswith(".xlsx")
        or "session_summary_processed_rebuilt" in filename
        or "_rebuilt" in filename
        or "chzzk-crawler" in parts
        or "my_job" in parts
    )


def _write_archive_file(zf, src, arcname, written):
    src = Path(src)
    arcname = str(arcname).replace("\\", "/")
    if not src.exists() or _forbidden_archive_member(arcname) or arcname in written:
        return
    zf.write(src, arcname)
    written.add(arcname)


def write_x_col_files(out):
    out = Path(out)
    core_lines = [
        "# 세션 단위 모델 학습용 기본 feature 목록",
        "# cluster_number는 선택적으로 사용할 수 있는 범주형 행동 군집 feature이다.",
        "# cluster_number는 정답 라벨이나 확률이 아니다.",
        "# downstream 목표가 같은 KMeans cluster_number에서 파생되면 cluster_number를 feature로 쓰는 것은 목표 누수다.",
        "# supervised train/test 평가에서는 전체 데이터에 RobustScaler/KMeans를 fit하면 안 된다.",
        "# fold별 train split 안에서 RobustScaler와 KMeans를 fit하고, test fold에는 transform/predict로 cluster를 부여해야 한다.",
        *X_CORE_COLS,
    ]
    no_leak_lines = [
        "# 금지 또는 누수 위험 column 목록",
        "# 아래 항목은 output, diagnostic, synthetic field, Method 2 근거 또는 목표 누수 위험이 있으므로 session model input으로 쓰지 않는다.",
        "# cluster_number는 여기서 제외되어 있지만 선택 feature일 뿐이며, 위 X_core_cols.txt의 목표 누수 주의문을 반드시 따른다.",
        *X_NO_LEAK_COLS,
    ]
    (out / "X_core_cols.txt").write_text("\n".join(core_lines) + "\n", encoding="utf-8")
    (out / "X_no_leak_cols.txt").write_text("\n".join(no_leak_lines) + "\n", encoding="utf-8")
    print("X_core_cols.txt와 X_no_leak_cols.txt를 저장했습니다.")


def write_session_cluster_alias(out):
    out = Path(out)
    cfg_doc = {}
    cluster_cfg = {}
    # cfg.yml is already loaded by main, but this writer intentionally reads
    # generated artifacts plus cfg-like values passed through the output files.
    # The defaults below only guard missing artifacts.
    try:
        with open(resolve_cfg_path("cfg.yml"), encoding="utf-8") as f:
            cfg_doc = yaml.safe_load(f) or {}
            cluster_cfg = cfg_doc.get("cluster", {})
    except Exception:
        cfg_doc = {}
        cluster_cfg = {}
    gap_reset_min = clock_gap_reset_min(cfg_doc)
    k_min = int(cluster_cfg.get("k_min", 2))
    k_max = int(cluster_cfg.get("k_max", 6))
    seed = cluster_cfg.get("seed", 42)
    n_init = cluster_cfg.get("n_init", 50)
    min_n = int(cfg_doc.get("prep", {}).get("min_n", 10))
    try:
        select = pd.read_csv(out / "cluster_select.csv", encoding="utf-8-sig")
    except Exception:
        select = pd.DataFrame()
    selected_k = _selected_session_k(select)
    try:
        session_df = pd.read_csv(out / "session_summary_processed.csv", encoding="utf-8-sig")
        actual_cluster_count = _unique_cluster_count(session_df, "cluster_number")
    except Exception:
        actual_cluster_count = "not available"
    try:
        profile = pd.read_csv(out / "cluster_profile.csv", encoding="utf-8-sig")
    except Exception:
        profile = pd.DataFrame()
    try:
        minute_model = pd.read_csv(out / "minute_model.csv", usecols=SESSION_KEY_COLS + ["minute_ts"], encoding="utf-8-sig")
        session_sizes = minute_model.groupby(SESSION_KEY_COLS).size()
        sessions_before_min_n = len(session_sizes)
        sessions_after_min_n = int(session_sizes.ge(min_n).sum())
    except Exception:
        sessions_before_min_n = "not available"
        sessions_after_min_n = "not available"

    silhouette_lines = []
    if select.empty:
        silhouette_lines.append("- not available")
    else:
        work = select.copy()
        if "k" in work.columns:
            work["k"] = pd.to_numeric(work["k"], errors="coerce")
            work = work.sort_values("k")
        for _, row in work.iterrows():
            k_value = row.get("k", np.nan)
            sil = row.get("silhouette", np.nan)
            ch = row.get("calinski_harabasz", np.nan)
            db = row.get("davies_bouldin", np.nan)
            balance = row.get("cluster_size_balance", np.nan)
            sep = row.get("cluster_profile_separation", np.nan)
            score = row.get("selection_score", np.nan)
            selected = bool(_selected_flag_mask(pd.DataFrame([row])).iloc[0]) if "selected" in row.index else False
            k_text = "nan" if pd.isna(k_value) else str(int(k_value))
            sil_text = "nan" if pd.isna(sil) else f"{float(sil):.8g}"
            ch_text = "nan" if pd.isna(ch) else f"{float(ch):.8g}"
            db_text = "nan" if pd.isna(db) else f"{float(db):.8g}"
            bal_text = "nan" if pd.isna(balance) else f"{float(balance):.8g}"
            sep_text = "nan" if pd.isna(sep) else f"{float(sep):.8g}"
            score_text = "nan" if pd.isna(score) else f"{float(score):.8g}"
            silhouette_lines.append(f"- k={k_text}, silhouette={sil_text}, Calinski-Harabasz={ch_text}, Davies-Bouldin={db_text}, size_balance={bal_text}, profile_separation={sep_text}, selection_score={score_text}, selected={selected}")

    profile_lines = []
    if profile.empty:
        profile_lines.append("- profile not available")
    else:
        sparse_cluster = _rank_sparse_profile(profile.rename(columns={"cluster_number": "cluster_number"}))
        for _, row in profile.iterrows():
            cluster_number = row.get("cluster_number")
            meaning = "viewer 대비 채팅 반응 약한 세션 행동 군집" if sparse_cluster is not None and cluster_number == sparse_cluster else "상대적으로 채팅 반응이 활발하거나 혼합된 세션 행동 군집"
            parts = [f"cluster_number={cluster_number}", f"해석={meaning}"]
            for col in ["n", "viewer_med", "chat_mean", "unique_mean", "zero_rate", "gap_med", "zrun_max"]:
                if col in profile.columns:
                    val = row.get(col)
                    if isinstance(val, float):
                        val = f"{val:.4g}"
                    parts.append(f"{col}={val}")
            profile_lines.append("- " + ", ".join(parts))

    text = "\n".join([
        "세션 행동 군집 문서",
        "==================",
        "",
        "분석 단위: 방송 세션.",
        "세션 정의: run_id + broad_no.",
        "목적: session_summary_processed.csv에 세션 행동 군집 번호를 붙이기 위한 비지도 clustering이다.",
        "주의: cluster_number는 정답 라벨이나 확률이 아니라 세션 행동 군집 번호다.",
        "",
        "session_summary_processed.csv 생성 방식:",
        "- minute_model.csv의 1분 row를 run_id + broad_no 단위로 집계한다.",
        "- min_n은 의심 기준이 아니라 세션 요약 통계 안정성을 위한 최소 관측 길이다.",
        "- all-zero-chat session은 QC bucket으로 보존하되 세션 행동 군집 학습에서는 제외한다.",
        "",
        "사용 feature와 근거:",
        "- log_viewer: viewer 규모가 heavy-tail이므로 log1p로 요약한다.",
        "- log_chat: chat 규모를 log1p로 요약한다.",
        "- log_unique: unique chatter 규모를 log1p로 요약한다.",
        "- zero_rate: session 내 chat=0 minute 비율이다.",
        "- gap_med: log1p(viewer_count_last) - log1p(chat_count)의 session median으로 viewer 대비 chat 반응 약화를 요약한다.",
        "- log_zrun: clock-contiguous zero-chat persistence를 log1p로 요약한다.",
        "viewer와 chat은 규모 수준에서는 양의 관계가 있지만 minute-level delta response는 약하므로 단순 delta보다 session-level gap과 zero-chat persistence를 사용한다.",
        "",
        "실제 feature 목록:",
        *[f"- {feature}" for feature in SESSION_CLUSTER_FEATS],
        "",
        "스케일링:",
        "- 적용: 적용",
        "- 사용 방식: RobustScaler",
        "- 선택 이유: viewer/chat 계열 변수의 heavy-tail과 극단값 영향을 줄이기 위해 중앙값/IQR 기반 스케일링을 사용한다.",
        "",
        "사용한 clustering 알고리즘:",
        "- KMeans: 최종 cluster_number 후보.",
        "- GMM: component 기반 하위 구조 진단.",
        "- HDBSCAN: 밀도 기반 안정 군집과 noise 비중 진단. noise는 이상치 확정이 아니라 안정 군집에 속하지 않은 세션 상태다.",
        "",
        "후보 하이퍼파라미터:",
        f"- KMeans K 후보: {k_min}..{k_max}",
        f"- random_state={seed}",
        f"- n_init={n_init}",
        f"- GMM n_components 후보: 2..{min(6, max(2, sessions_after_min_n if isinstance(sessions_after_min_n, int) else 6))}",
        f"- HDBSCAN min_cluster_size 후보: {cfg_doc.get('hdbscan', {}).get('min_sizes', [5, 8, 10])}",
        f"- min_n={min_n}",
        f"- clock_gap_reset_min={gap_reset_min}",
        "",
        "최종 선택:",
        "- 최종 선택 알고리즘: KMeans",
        f"- 최종 선택 K: {selected_k}",
        f"- 실제 cluster_number unique count: {actual_cluster_count}",
        "- 선택 근거: silhouette, Calinski-Harabasz, Davies-Bouldin, 군집 크기 균형, 군집 profile 분리도를 방향에 맞춘 percentile rank로 변환한 composite selection_score를 사용했다.",
        "- GMM은 하위 구조 진단에는 유용하지만 최종 세션 행동 군집 번호로는 KMeans보다 설명과 재현이 어렵다.",
        "- HDBSCAN은 밀도 기반 안정 군집 확인에는 유용하지만 noise 처리 때문에 handoff용 전체 세션 번호로 쓰기 어렵다.",
        "",
        "결측 처리 정책:",
        "- viewer_count_last: standalone rebuild에서 minute_ts 정렬 후 run_id + broad_no 내부 ffill/bfill을 먼저 적용하고 남은 결측은 0으로 채운다.",
        "- chat_count / unique_chatters: chat event가 관측되지 않은 minute은 0으로 볼 수 있으므로 0 fill한다.",
        "- 원본 결측 row 수와 ffill/bfill 이후 잔여 결측 row 수는 validation_report와 standalone report에 남긴다.",
        "",
        "Zero-run clock gap 정책:",
        f"- 같은 run_id + broad_no 내부에서도 minute_ts.diff() > {gap_reset_min}분이면 zero-chat run을 reset한다.",
        "- zrun_max는 clock-contiguous zero-chat segment의 최대 길이이다.",
        "",
        "min_n 전/후 session 수:",
        f"- min_n filter 전: {sessions_before_min_n}",
        f"- min_n filter 후: {sessions_after_min_n}",
        "",
        "K 후보별 평가:",
        *silhouette_lines,
        "",
        "cluster_number별 profile 요약:",
        *profile_lines,
        "",
        "주의 문구:",
        "- viewer 대비 채팅 반응이 약한 profile 해석은 TXT에만 적고 CSV 컬럼으로 넣지 않는다.",
        "- all-zero-chat session은 QC bucket 후보이며 KMeans cluster_number를 부여하지 않는다.",
        "- 운영 판단은 raw WebSocket/chat QC와 수동 검토 없이는 불가하다.",
    ])
    (out / "session_cluster.txt").write_text(text + "\n", encoding="utf-8")


def _selected_minute_k_text(select):
    return _selected_minute_k(select)


def write_minute_cluster_doc(out, cfg):
    out = Path(out)
    select = _read_csv_safe(out / "mc_select.csv")
    profile = _read_csv_safe(out / "mc_profile.csv")
    model_selection = _read_csv_safe(out / "minute_cluster_model_selection.csv")
    mc_cfg = cfg.get("minute_cluster", {})
    km_cfg = mc_cfg.get("kmeans", {})
    gmm_cfg = mc_cfg.get("gmm", {})
    hdb_cfg = mc_cfg.get("hdbscan", {})
    features = mc_cfg.get("features", [])
    scaler_name = mc_cfg.get("scaler", "RobustScaler")
    selected_minute_k = _selected_minute_k(select)
    final_algorithm = "KMeans"
    final_param = f"k={selected_minute_k}" if isinstance(selected_minute_k, int) else str(selected_minute_k)
    try:
        minute_assign = _read_csv_safe(out / "mc_assign.csv")
        actual_minute_cluster_count = _unique_cluster_count(minute_assign, "minute_cluster")
    except Exception:
        actual_minute_cluster_count = "not available"
    selection_lines = []
    if model_selection.empty:
        selection_lines.append("- minute_cluster_model_selection.csv가 비어 있다.")
    else:
        for _, row in model_selection.head(30).iterrows():
            selection_lines.append(
                f"- {row.get('algorithm')} {row.get('parameter_setting')}: final={row.get('selected_as_final_state')}, "
                f"{row.get('selection_metric_1')}, {row.get('selection_metric_2')}, {row.get('selection_metric_3')}"
            )
    profile_lines = []
    if profile.empty:
        profile_lines.append("- profile을 만들 수 없었다.")
    else:
        for _, row in profile.iterrows():
            interp = row.get("interpretation")
            if interp == "mismatch-like state":
                interp = "viewer 대비 채팅 반응 약한 상태"
            elif interp == "active/high-chat state":
                interp = "채팅 반응 활발 상태"
            elif interp == "low-scale quiet state":
                interp = "소규모 조용한 상태"
            else:
                interp = "혼합 행동 상태"
            cols = [c for c in ["n", "share", "viewer_med", "chat_med", "unique_med", "chat_deficit_med", "unique_deficit_med", "zero_chat_rate", "rolling_zero_rate_5m_p90", "cluster_mismatch_rank"] if c in profile.columns]
            profile_lines.append("- " + ", ".join([f"cluster_id={row.get('cluster_id')}", f"해석={interp}"] + [f"{c}={row.get(c)}" for c in cols]))
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
        "- log_viewer = log1p(viewer_count_last): viewer 규모 요약.",
        "- chat_deficit = expected_log_chat_bin - log_chat: viewer scale 대비 chat 부족.",
        "- unique_deficit = expected_log_unique_bin - log_unique: viewer scale 대비 unique chatter 부족.",
        "- rolling_chat_deficit_5m: 최근 5분 chat_deficit 평균.",
        "- log_zero_run_len = log1p(clock-gap-aware zero_run_len): zero-chat 지속성.",
        "- rolling_zero_rate_5m: 최근 5분 zero_chat 비율.",
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
        f"- KMeans K 후보: {km_cfg.get('k_min', 2)}..{km_cfg.get('k_max', 8)}",
        f"- KMeans random_state: {km_cfg.get('seed', mc_cfg.get('seed', 42))}",
        f"- KMeans n_init: {km_cfg.get('n_init', mc_cfg.get('n_init', 50))}",
        f"- GMM n_components 후보: {gmm_cfg.get('n_min', 2)}..{gmm_cfg.get('n_max', 8)}",
        f"- HDBSCAN min_cluster_size 후보: {hdb_cfg.get('min_sizes', cfg.get('hdbscan', {}).get('min_sizes', [5, 8, 10]))}",
        "",
        "최종 선택:",
        f"- 최종 선택 알고리즘: {final_algorithm}",
        f"- 최종 선택 하이퍼파라미터: {final_param}",
        f"- 실제 minute_cluster unique count: {actual_minute_cluster_count}",
        "- 최종 선택 근거: 동일 feature와 동일 스케일링 조건에서 silhouette, Calinski-Harabasz, Davies-Bouldin, 군집 크기 균형, profile 분리도, seed 안정성을 방향에 맞춘 percentile rank로 변환한 composite selection_score를 사용했다.",
        "- GMM은 하위 구조 진단에는 유용하지만 component 해석과 handoff 상태 번호 재현성이 KMeans보다 낮아 최종 상태 번호로 쓰지 않았다.",
        "- HDBSCAN은 안정 군집과 noise 비중 확인에는 유용하지만 noise 처리 때문에 모든 1분 row에 일관된 상태 번호를 붙이기 어려워 최종 상태 번호로 쓰지 않았다.",
        "",
        "모델 선택 표 요약:",
        *selection_lines,
        "",
        "cluster 번호별 profile 요약:",
        *profile_lines,
        "",
        "해석 제한:",
        "- viewer 대비 채팅 반응 약한 상태는 확정 판정이 아니라 비지도 군집 profile 해석이다.",
        "- cluster_mismatch_rank는 분 단위 mismatch 신호 profile을 설명하기 위한 보조 순위 값이다.",
    ]
    (out / "minute_cluster.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method_registry(out, cfg):
    out = Path(out)
    gap_reset_min = clock_gap_reset_min(cfg)
    rows = [
        ("데이터 로딩 및 시간 window 필터링", "final", "raw feature files", "file_audit.csv, used_windows.csv, load_policy.txt", "분석 가능한 시간 window를 일관되게 유지", "모든 downstream 입력 row 결정", "minute_ts, file metadata", "없음", "policy filter", str(cfg.get("time", {})), "cfg.yml time policy", "row/file counts", "원천 수집 품질에 의존", "라벨/확률 아님"),
        ("minute core table 생성", "final", "cleaned raw rows", "minute_all.csv, minute_model.csv", "strict raw minute core 제공", "handoff와 session summary 재생성", ", ".join(STRICT_MINUTE_CORE_COLS), "없음", "deterministic aggregation", "없음", "전달 요구 schema", "strict column validation", "분 단위 집계에 의존", "라벨/확률 아님"),
        ("all-zero-chat QC bucket 분리", "final", "minute_all", "qc_zero.csv, qc_zero_session_review.csv", "WebSocket 미수집 가능성 또는 극단 후보 보존", "minute_model/session KMeans 제외 및 수동 QC appendix 제공", "chat_count", "없음", "session group rule", "chat sum == 0", "QC 정책", "session/minute counts와 all-zero session summary", "원인 확정 불가", "라벨/확률 아님"),
        ("viewer_count 결측 처리", "final", "minute core", "imputed viewer_count_last", "viewer 기반 feature 안정화", "session/minute feature 계산", "viewer_count_last", "없음", "ffill/bfill then 0", "group=run_id+broad_no", "시간 순서 보존", "missing audit", "장기 결측은 품질 위험", "라벨/확률 아님"),
        ("zero-run 계산", "final", "minute core", "zero_run_len, zrun_max", "침묵 지속성 측정", "session feature와 Method2 interval support", "chat_count, minute_ts", "없음", "clock-gap-aware run length", f"clock_gap_reset_min={gap_reset_min}", "clock gap 단절 반영", "gap reset mismatch count", "수집 gap 영향", "라벨/확률 아님"),
        ("session_summary_processed 생성", "final", "strict minute_model", "session_summary_processed.csv", "session-level modeling table", "teammate handoff", ", ".join(HANDOFF_SESSION_COLS), "log1p", "group aggregation", f"min_n={cfg.get('prep', {}).get('min_n')}", "session 단위 요약", "value-level rebuild", "session 내부 분포 요약 손실", "라벨/확률 아님"),
        ("session-level KMeans clustering", "final", "session_summary_processed", "cluster_number", "behavior state id 부여", "optional feature/profile 해석", ", ".join(SESSION_CLUSTER_FEATS), "RobustScaler", "KMeans", str(cfg.get("cluster", {})), "silhouette, Calinski-Harabasz, Davies-Bouldin, cluster size balance, profile separation을 방향에 맞춘 percentile rank로 변환한 composite selection_score", "selection_score/profile", "구형 군집 가정", "cluster_number는 라벨/확률 아님"),
        ("session K 선택", "final", "scaled session features", "cluster_select.csv", "K 근거 제시", "selected K 결정", ", ".join(SESSION_CLUSTER_FEATS), "RobustScaler", "composite K search", "K=2..6", "selection_score 최대, silhouette 동률, 작은 K", "selection_score/profile", "정답 K 아님", "라벨/확률 아님"),
        ("minute-state KMeans stability diagnostic", "diagnostic", "mc_stab.csv", "07_session_cluster_stability.png", "minute KMeans behavior-state seed/subsample 안정성 확인", "final cluster를 바꾸지 않음", "ARI, selected_k, mismatch share", "없음", "ARI diagnostic", "seeds/sample_size", "재현성 점검", "ARI/subsample stability; supervised 성능지표 아님", "라벨/확률 아님"),
        ("minute-state feature 생성", "final", "minute_model_feat", "m2_scores inputs", "viewer-chat mismatch minute signal 생성", "Method2 ranking evidence", "deficit, zero_run, rolling features", "부분 log1p", "deterministic features", str(cfg.get("minute_state", {})), "viewer scale 대비 반응 약화", "feature integrity", "no-label feature", "라벨/확률 아님"),
        ("minute-level KMeans clustering", "final", "minute-state features", "minute_cluster, mc_profile.csv", "minute behavior state 요약", "mismatch state rate evidence", ", ".join(cfg.get("minute_cluster", {}).get("features", [])), "RobustScaler", "KMeans", str(cfg.get("minute_cluster", {})), "silhouette, Calinski-Harabasz, Davies-Bouldin, cluster size balance, profile separation, stability를 방향에 맞춘 percentile rank로 변환한 composite selection_score", "selection_score/profile/stability", "behavior state일 뿐", "라벨/확률 아님"),
        ("expected-response baseline", "final", "minute-state features", "base_pred.csv", "viewer scale 대비 기대 chat/unique 추정", "expected_response_rank", "log_viewer, category, hour, minute_idx", "model internal", "HistGradientBoostingRegressor quantile", "GroupKFold by run_id", "leakage 완화", "OOF/fallback note", "classifier 아님", "라벨/확률 아님"),
        ("adaptive interval scan", "final", "m2_scores", "m2_scan.csv", "지속적인 mismatch interval 탐색", "scan_family_rank의 내부 근거", "minute_mismatch_rank", "없음", "scan statistic + shuffled null", str(cfg.get("m2_scan", {})), "지속 interval 목적", "empirical_p, duration, z", "empirical_p는 확률 아님", "라벨/확률 아님"),
        ("interval anomaly support", "diagnostic", "m2_scan.csv", "int_scores.csv", "top interval profile 이질성 보조 evidence", "interval_anomaly_rank 보조", "interval features", "RobustScaler", "IsolationForest/LOF/ECOD", str(cfg.get("m2_interval_anomaly", {})), "보조 support", "directional score/rank", "short spike 과대 가능", "라벨/확률 아님"),
        ("reason support", "final", "m2_scan, int_scores", "m2_reason.csv", "연속 support 근거 설명", "reason_support_rank", "interval support values", "없음", "rank aggregation support", str(cfg.get("m2_review", {})), "세션별 상위 근거를 순위 기반으로 선택", "support rank", "confidence/lift 아님", "라벨/확률 아님"),
        ("family consensus review priority", "final", "family ranks", "m2_review.csv", "수동 검토 우선순위 생성", "review_order", "scan/persistence/expected-response/minute-state/anomaly/reason family ranks", "없음", "equal-weight family consensus + family RRA", "BH q on family ranks", "raw scan evidence duplication 방지", "family_consensus_score, family_rra_q", "rra_q와 family_consensus_score는 확률 아님", "라벨/확률 아님"),
        ("Method 2 null/stability diagnostic", "diagnostic", "m2_scores/mc_assign", "m2_null.csv, mc_stab.csv", "no-label sanity check", "final order를 직접 바꾸지 않음", "episode/state/stability stats", "없음", "shuffle/ARI", "configured seeds/permutations", "진단 목적", "ARI/null p", "성능지표 아님", "라벨/확률 아님"),
        ("synthetic sanity", "appendix", "synthetic_intervals, m2_scan, m2_review", "m2_synth.csv, m2_synth_matches.csv", "합성 주입 세션이 같은 Method2 scoring/scan/review 경로를 통과했을 때만 내부 sanity 확인", "status=ok일 때만 recovery summary를 appendix로 표시", "session_key, injected/detected interval time", "없음", "same-session interval IoU", str(cfg.get("synthetic_sanity", {})), "cfg synthetic_sanity.enabled", "status와 IoU summary", "status=not_run이면 recovered_rate를 보고하지 않음; y_syn은 실제 viewbot label이 아님", "라벨/확률 아님"),
        ("GMM/HDBSCAN diagnostic", "appendix", "session features", "gmm_select.csv, hdbscan_select.csv", "구조 진단", "final cluster_number를 바꾸지 않음", ", ".join(SESSION_CLUSTER_FEATS), "RobustScaler", "GMM/HDBSCAN", "legacy diagnostics", "appendix 격리", "BIC/coverage", "final label source 아님", "라벨/확률 아님"),
        ("DecisionTree/Naive Bayes/Neural Network/supervised metrics", "removed", "없음", "없음", "정답 라벨 부재로 final pipeline에서 제거", "사용하지 않음", "없음", "없음", "removed legacy", "없음", "중간발표 legacy만", "없음", "실제 성능 claim 불가", "라벨/확률 아님"),
    ]
    header = [
        "절차명", "사용 여부", "입력 데이터", "출력 산출물", "왜 필요한가", "어떤 결과 판단에 실제로 쓰이는가",
        "사용 feature", "스케일링 여부", "알고리즘", "하이퍼파라미터", "하이퍼파라미터 선택 근거", "검증 지표", "한계", "라벨/확률 아님 여부",
    ]
    lines = [
        "# Method Registry",
        "",
        "최종 표현은 확정 탐지가 아니라 viewer-chat mismatch 기반 수동 검토 우선순위 생성으로 제한한다.",
        "final ranking이나 handoff에 직접 쓰이지 않는 절차는 diagnostic/appendix/removed로 격리한다.",
        "final review_order는 raw evidence RRA가 아니라 family-level equal-weight consensus + family RRA로 정렬한다.",
        "scan_interval_rank, empirical_p_rank, scan_strength_rank는 같은 scan family의 내부 근거이며 final RRA에 각각 독립 evidence로 들어가지 않는다.",
        "",
        "|" + "|".join(header) + "|",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        safe = [str(cell).replace("|", "/") for cell in row]
        lines.append("|" + "|".join(safe) + "|")
    (out / "method_registry.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_korean_m2_docs(out, cfg):
    out = Path(out)
    review_cfg = cfg.get("m2_review", {})
    eval_cfg = cfg.get("eval_robustness", {})
    families = eval_cfg.get("families") or [
        "scan_family_rank",
        "persistence_family_rank",
        "expected_response_family_rank",
        "minute_state_family_rank",
        "interval_anomaly_family_rank",
        "reason_support_family_rank",
    ]
    aggregation = review_cfg.get("aggregation", "equal_weight_family_consensus_plus_family_rra")
    lines = [
        "# Method 2 최종 파이프라인",
        "",
        "목표: 확정 탐지가 아니라 viewer-chat mismatch 기반 수동 검토 우선순위를 만든다.",
        "`cluster_number`, `minute_cluster`, `rra_q`, `empirical_p`, `review_order`는 정답 라벨이나 확률이 아니다.",
        "",
        f"final review_order는 raw evidence RRA가 아니라 `{aggregation}` 설정에 따라 정렬한다.",
        "legacy threshold grid 산출물은 appendix diagnostic이며 final ranking을 직접 바꾸지 않는다.",
        f"Family evidence list: {', '.join(map(str, families))}.",
        "scan_interval_rank, empirical_p_rank, scan_strength_rank는 같은 scan family의 내부 근거이며 final RRA에 각각 독립 evidence로 들어가지 않는다.",
        "persistence_family는 top interval duration과 state dwell evidence를 hard threshold 없이 rank/percentile 기반으로 반영한다.",
        "reason_support_rank는 continuous support 기반이며 설명 문구는 세션별 상위 근거를 순위 기반으로 선택한다.",
        "empirical_p는 shuffled-null 대비 scan statistic 근거이며 확률이 아니다.",
        "rra_q, family_rra_q, family_consensus_score는 확률이 아니다.",
        "",
        "all-zero chat sessions are preserved in qc_zero_session_review.csv for manual QC.",
        "They are not used as positive labels or confirmed cases.",
        "They are excluded from behavior modeling to avoid mixing WebSocket collection failures with behavioral mismatch.",
        "",
        "synthetic sanity는 실제 ground-truth 성능 평가가 아니다.",
        "status=not_run이면 recovered_rate를 보고하지 않으며 0% recovery로 해석하지 않는다.",
        "status=ok일 때만 recovery summary를 내부 sanity check로 표시한다.",
        "",
        "모든 operational conclusion은 raw WebSocket/chat QC와 수동 검토가 필요하다.",
    ]
    (out / "m2_pipeline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    eval_lines = [
        "# Method 2 평가/진단 계획",
        "",
        "현재 실제 ground-truth label이 없으므로 accuracy, precision, recall, F1, AUC를 실제 성능처럼 보고하지 않는다.",
        "사용 가능한 검증은 no-label sanity check, shuffled-null diagnostic, stability diagnostic, handoff 재현성 검증이다.",
        "m2_review.csv의 review_order는 family-level equal-weight consensus + family RRA 기반 수동 검토 우선순위이지 정답 라벨이 아니다.",
        "top10 short interval count는 WARNING으로 보고하며 duration <= 1 같은 hard cutoff로 세션을 자동 제외하지 않는다.",
        "synthetic sanity는 합성 세션이 같은 Method2 scoring/scan/review pipeline을 통과하고 status=ok일 때만 recovery summary를 sanity check로 제시한다.",
        "status=not_run 또는 not_run_stale_input이면 recovered_rate는 보고하지 않는다.",
        "y_syn은 실제 viewbot label이 아니다.",
        "all-zero chat sessions are preserved in qc_zero_session_review.csv for manual QC.",
        "They are not used as positive labels or confirmed cases.",
        "They are excluded from behavior modeling to avoid mixing WebSocket collection failures with behavioral mismatch.",
    ]
    (out / "m2_eval_plan.md").write_text("\n".join(eval_lines) + "\n", encoding="utf-8")
    notes = [
        "# Method 2 모델 노트",
        "",
        "expected-response baseline은 classifier가 아니며 GroupKFold by run_id 기반 no-label baseline이다.",
        "minute KMeans와 session KMeans는 behavior state 요약이며 final label source가 아니다.",
        "interval anomaly support는 보조 evidence이고 short spike를 과도하게 상위로 올리면 diagnostic으로 격리한다.",
        "최종 review_order는 family-level equal-weight consensus + family RRA이며 rra_q는 family_rra_q와 같다.",
        "raw evidence 기반 값은 raw_rra_p/raw_rra_q로 보존한다.",
        "scan_interval_rank, empirical_p_rank, scan_strength_rank는 같은 scan family의 내부 근거이며 final RRA에 각각 독립 evidence로 들어가지 않는다.",
        "hard threshold나 확정 label을 쓰지 않고 rank consensus로 처리한다.",
        "synthetic_sanity.enabled=false이면 m2_synth.csv는 status=not_run으로 남기고 recovered_rate를 비워 둔다.",
        "stale synthetic_intervals.csv만 있고 현재 m2_scores/m2_scan에 synthetic session_key가 없으면 status=not_run_stale_input으로 기록한다.",
        "qc_zero_session_review.csv는 manual QC appendix이며 Method2 ranking 입력이 아니다.",
    ]
    (out / "m2_model_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def write_parameter_rationale(out, cfg):
    out = Path(out)
    prep = cfg.get("prep", {})
    minute_state = cfg.get("minute_state", {})
    minute_cluster = cfg.get("minute_cluster", {})
    cluster = cfg.get("cluster", {})
    m2_scan = cfg.get("m2_scan", {})
    m2_expected = cfg.get("m2_expected_response", {})
    m2_interval = cfg.get("m2_interval_anomaly", {})
    synthetic = cfg.get("synthetic_sanity", {})
    minute_features = minute_cluster.get("features", [])
    lines = [
        "# Parameter Rationale",
        "",
        "This document explains why the main configuration values exist. They are engineering and diagnostic choices, not certainty labels or probability outputs.",
        "주요 파라미터는 재현성과 진단 안정성을 위한 설정이며 확정 판정 기준이 아니다.",
        "",
        f"## prep.min_n={prep.get('min_n', 10)}",
        "- Minimum observation length for stable session summary statistics.",
        "- It is not a suspicious-session criterion.",
        "",
        f"## prep.clock_gap_reset_min={prep.get('clock_gap_reset_min', 1.1)}",
        "- Clock-gap reset reflecting the roughly 60-second viewer snapshot cadence.",
        "- Prevents collection gaps from being misread as one long zero-chat run.",
        "",
        f"## minute_state.viewer_bin_n={minute_state.get('viewer_bin_n', 10)}",
        "- Decile bins make expected chat and unique chatter comparable across viewer scale.",
        "- The bin is descriptive and not a labeling rule.",
        "",
        f"## minute_state.rolling_windows={minute_state.get('rolling_windows', [5, 10])}",
        "- Rolling windows capture short persistence in the 0-10 minute range seen during EDA.",
        "- Final ranking uses rolling evidence as continuous rank evidence, not a hard threshold.",
        "",
        "## minute_cluster.features",
        "- `log_viewer`: viewer scale after log transform.",
        "- `chat_deficit`: chat response deficit relative to viewer scale.",
        "- `unique_deficit`: unique chatter deficit relative to viewer scale.",
        "- `rolling_chat_deficit_5m`: short persistence of chat deficit.",
        "- `log_zero_run_len`: clock-gap-aware zero-chat persistence.",
        "- `rolling_zero_rate_5m`: recent zero-chat concentration.",
        f"- Configured features: {', '.join(map(str, minute_features))}",
        "",
        "## RobustScaler",
        f"- Session scaler: {cluster.get('scaler', 'RobustScaler')}. Minute scaler: {minute_cluster.get('scaler', 'RobustScaler')}.",
        "- Median/IQR scaling reduces the influence of heavy-tailed viewer and chat values.",
        "",
        "## KMeans K candidates",
        f"- Session K candidates: {cluster.get('k_min', 2)}..{cluster.get('k_max', 6)}.",
        f"- Minute K candidates: {minute_cluster.get('kmeans', {}).get('k_min', 2)}..{minute_cluster.get('kmeans', {}).get('k_max', 8)}.",
        "- K is selected by selection_score, not fixed by assumption.",
        "- selection_score combines silhouette, Calinski-Harabasz, Davies-Bouldin, size balance, profile separation, and stability in the correct direction.",
        "",
        f"## m2_scan.n_perm={m2_scan.get('n_perm', 200)}",
        "- Shuffled-null diagnostic count; with n_perm=200, the minimum empirical resolution is 1/(200+1).",
        "- The resulting empirical_p is diagnostic evidence, not a calibrated probability.",
        "- If a tighter null estimate is required, increase n_perm and document the sensitivity plan.",
        "",
        f"## m2_scan.max_scan_n={m2_scan.get('max_scan_n', 500)}",
        "- Computational cap for long-session scan search.",
        "- Any pruning behavior is documented in m2_scan.csv note fields.",
        "",
        "## expected-response baseline",
        "- GroupKFold by run_id reduces leakage across broadcasts from the same run.",
        f"- max_train_rows_per_fold={m2_expected.get('max_train_rows_per_fold', 200000)} keeps training cost bounded.",
        "- This estimates expected chat and unique chatter response; it is not a classifier.",
        "",
        "## interval anomaly",
        f"- RobustScaler setting: {m2_interval.get('scaler', 'RobustScaler')}.",
        "- IsolationForest, ECOD, and LOF are auxiliary directional evidence sources.",
        "- They are not final label sources.",
        "",
        f"## synthetic_sanity.enabled={synthetic.get('enabled', False)}",
        "- When disabled, m2_synth.csv reports status=not_run and leaves recovered_rate blank.",
        "- Blank recovered_rate must not be interpreted as 0% recovery.",
        "",
        "## Interpretation limits",
        "- cluster_number, minute_cluster, rra_q, empirical_p, family_consensus_score, and review_order are review-priority or diagnostic values.",
        "- No output column is a final decision field.",
    ]
    (out / "parameter_rationale.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_if_exists(src, dst, copied, missing):
    src = Path(src)
    dst = Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst))
    else:
        missing.append(str(src))


def _rank_sparse_profile(df):
    if df.empty or "cluster_number" not in df.columns:
        return None
    prof = df.groupby("cluster_number").agg(
        zero_rate=("zero_rate", "mean"),
        gap_med=("gap_med", "median"),
        zrun_max=("zrun_max", "median"),
        chat_mean=("chat_mean", "mean"),
        unique_mean=("unique_mean", "mean"),
    )
    if prof.empty:
        return None
    sparse_score = (
        prof["zero_rate"].rank(pct=True)
        + prof["gap_med"].rank(pct=True)
        + prof["zrun_max"].rank(pct=True)
        + (-prof["chat_mean"]).rank(pct=True)
        + (-prof["unique_mean"]).rank(pct=True)
    ) / 5
    return sparse_score.idxmax()


def make_handoff_session_df(session_df):
    df = session_df.copy()
    for col in HANDOFF_SESSION_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    raw_cluster = pd.to_numeric(df["cluster_number"], errors="coerce")
    if raw_cluster.notna().any():
        df["cluster_number"] = raw_cluster.astype("Int64")
    else:
        df["cluster_number"] = 0
    return df[HANDOFF_SESSION_COLS]


def standalone_make_session_summary_script(cfg=None):
    cfg = cfg or {}
    cluster_cfg = cfg.get("cluster", {})
    handoff_cfg = cfg.get("handoff", {})
    script_config = {
        "features": list(SESSION_CLUSTER_FEATS),
        "scaler": "RobustScaler",
        "k_min": int(cluster_cfg.get("k_min", 2)),
        "k_max": int(cluster_cfg.get("k_max", 6)),
        "random_state": int(cluster_cfg.get("seed", 42)),
        "n_init": int(cluster_cfg.get("n_init", 50)),
        "min_n": int(cfg.get("prep", {}).get("min_n", 10)),
        "clock_gap_reset_min": float(cfg.get("prep", {}).get("clock_gap_reset_min", 1.1)),
        "cluster_number_semantics": handoff_cfg.get(
            "cluster_number_semantics",
            "세션 행동 군집 번호이며 정답 라벨이나 확률이 아님",
        ),
    }
    script = r'''import os
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
CONFIG = __CONFIG__
OUT_COLS = __OUT_COLS__


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
'''
    return script.replace("__CONFIG__", repr(script_config)).replace("__OUT_COLS__", repr(HANDOFF_SESSION_COLS))


def write_standalone_make_session_summary(path, cfg=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(standalone_make_session_summary_script(cfg), encoding="utf-8")


def _session_count(df):
    if df.empty or not set(SESSION_KEY_COLS).issubset(df.columns):
        return 0
    return int(df[SESSION_KEY_COLS].drop_duplicates().shape[0])


def _handoff_stats(out, handoff=None):
    out = Path(out)
    handoff = Path(handoff) if handoff is not None else out / "handoff"
    minute_all = _read_csv_safe(handoff / "csv" / "minute_all.csv")
    minute_model = _read_csv_safe(handoff / "csv" / "minute_model.csv")
    session_handoff = _read_csv_safe(handoff / "csv" / "session_summary_processed.csv")
    min_n = 0
    try:
        with open(resolve_cfg_path("cfg.yml"), encoding="utf-8") as f:
            min_n = int((yaml.safe_load(f) or {}).get("prep", {}).get("min_n", 10))
    except Exception:
        min_n = 10
    if not minute_all.empty and {"session_key", "chat_count"}.issubset(minute_all.columns):
        chat_sum = pd.to_numeric(minute_all["chat_count"], errors="coerce").fillna(0).groupby(minute_all["session_key"].astype(str)).sum()
        all_zero_sessions = int(chat_sum.eq(0).sum())
        all_zero_keys = set(chat_sum[chat_sum.eq(0)].index.astype(str))
        all_zero_minutes = int(minute_all["session_key"].astype(str).isin(all_zero_keys).sum())
    else:
        all_zero_sessions = 0
        all_zero_minutes = 0
    if not minute_model.empty and set(SESSION_KEY_COLS).issubset(minute_model.columns):
        sizes = minute_model.groupby(SESSION_KEY_COLS).size()
        sessions_before_min_n = int(len(sizes))
        sessions_after_min_n = int(sizes.ge(min_n).sum())
    else:
        sessions_before_min_n = 0
        sessions_after_min_n = 0
    return {
        "minute_all_sessions": _session_count(minute_all),
        "minute_model_sessions": _session_count(minute_model),
        "all_zero_sessions": all_zero_sessions,
        "all_zero_minutes": all_zero_minutes,
        "session_summary_sessions": int(len(session_handoff)),
        "sessions_before_min_n": sessions_before_min_n,
        "sessions_after_min_n": sessions_after_min_n,
    }


def write_handoff_package(out, project_dir, cfg=None):
    out = Path(out)
    project_dir = Path(project_dir)
    cfg = cfg or {}
    gap_reset_min = clock_gap_reset_min(cfg)
    handoff = out / "handoff"
    if handoff.exists():
        shutil.rmtree(handoff)
    copied, missing = [], []

    for name in ["minute_all.csv", "minute_model.csv"]:
        src = out / name
        dst = handoff / "csv" / name
        if src.exists():
            minute = pd.read_csv(src, encoding="utf-8-sig")
            for col in STRICT_MINUTE_CORE_COLS:
                if col not in minute.columns:
                    minute[col] = pd.NA
            dst.parent.mkdir(parents=True, exist_ok=True)
            minute[STRICT_MINUTE_CORE_COLS].to_csv(dst, index=False, encoding="utf-8-sig")
            copied.append(str(dst))
        else:
            missing.append(str(src))
    session_src = out / "session_summary_processed.csv"
    if session_src.exists():
        session_df = pd.read_csv(session_src, encoding="utf-8-sig")
        handoff_session = make_handoff_session_df(session_df)
        dst = handoff / "csv" / "session_summary_processed.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        handoff_session.to_csv(dst, index=False, encoding="utf-8-sig")
        copied.append(str(dst))
    else:
        missing.append(str(session_src))
    for name in ["session_cluster.txt"]:
        _copy_if_exists(out / name, handoff / "txt" / name, copied, missing)
    for name in [
        "04_cluster_session.png",
    ]:
        _copy_if_exists(out / "plots" / name, handoff / "img" / name, copied, missing)

    code_path = handoff / "code" / "make_session_summary.py"
    write_standalone_make_session_summary(code_path, cfg)
    copied.append(str(code_path))

    stats = _handoff_stats(out, handoff)
    manifest_entries = sorted(str(Path(name).relative_to("df_required_handoff")).replace("\\", "/") for name in REQUIRED_HANDOFF_FILES)
    lines = [
        "# 최종 전달용 필수자료 README",
        "",
        "이 압축파일은 세션 단위 모델 학습과 재현 확인에 필요한 최소 자료만 포함한다.",
        "분 단위 mismatch 신호와 행동 군집은 수동 검토 우선순위를 돕기 위한 비지도 결과이며 확정 판정이 아니다.",
        "",
        "## 데이터 단위",
        "세션 단위는 `run_id + broad_no`로 고정한다.",
        "같은 run 안에서 한 스트리머가 방송을 재시작할 수 있으므로 `user_id + run_id`보다 실제 방송 단위인 `run_id + broad_no`가 더 타당하다.",
        f"- minute_all session 수: {stats['minute_all_sessions']}",
        f"- minute_model session 수: {stats['minute_model_sessions']}",
        f"- all-zero-chat session 수: {stats['all_zero_sessions']}",
        f"- all-zero-chat minute 수: {stats['all_zero_minutes']}",
        f"- session_summary_processed session 수: {stats['session_summary_sessions']}",
        f"- min_n filter 전/후 session 수: {stats['sessions_before_min_n']} / {stats['sessions_after_min_n']}",
        "",
        "## CSV",
        "- `csv/minute_all.csv`: all-zero-chat session까지 포함하는 원천 1분 row 핵심 컬럼이다.",
        "- `csv/minute_model.csv`: all-zero-chat session을 제외한 세션 요약 재생성 입력이다.",
        "- `csv/session_summary_processed.csv`: `csv/minute_model.csv`에서 생성되는 세션 단위 요약 table이며 `cluster_number`를 포함한다.",
        f"- 분 단위 CSV 허용 컬럼: {', '.join(STRICT_MINUTE_CORE_COLS)}",
        "분 단위 CSV에는 파생 feature, 군집 번호, anomaly score, review rank, 확률 컬럼을 넣지 않았다.",
        "",
        "## QC 정책",
        "all-zero-chat session은 WebSocket 미수집 가능성 또는 극단적 mismatch 후보이므로 삭제하지 않고 QC bucket으로 보존한다.",
        "단, 세션 행동 군집 학습용 `minute_model.csv`와 `session_summary_processed.csv`에서는 제외되며 `cluster_number`를 부여하지 않는다.",
        "",
        "## TXT",
        "- `txt/minute_cluster.txt`: 분 단위 행동 상태 군집의 feature, 스케일링, 후보 알고리즘, 하이퍼파라미터, 선택 근거, cluster profile을 설명한다.",
        "- `txt/session_cluster.txt`: 세션 단위 행동 군집의 feature, 스케일링, 후보 알고리즘, 하이퍼파라미터, 선택 근거, cluster profile을 설명한다.",
        "- `txt/X_core_cols.txt`, `txt/X_no_leak_cols.txt`: 다음 단계 모델 학습에서 사용할 수 있는 column과 누수 위험 column을 구분한다.",
        "",
        "## 이미지",
        "- `img/04_cluster_session.png`: 세션 단위 행동 군집 산점도와 군집별 세션 수를 확인한다.",
        "- `img/08_cluster_minute.png`: 분 단위 행동 상태 군집 비중과 주요 신호 profile을 확인한다.",
        "",
        "## 코드",
        "`code/make_session_summary.py`는 strict `csv/minute_model.csv`만으로 `session_summary_processed.csv`를 재생성할 수 있어야 한다.",
        "stdout과 `session_summary_cluster_report.txt`에는 minute_all.csv가 audit 확인용이라는 점이 한국어로 명시된다.",
        f"zero-chat run은 같은 session 내부에서도 `minute_ts.diff() > {gap_reset_min}`분이면 reset한다.",
        "viewer_count_last 결측은 session 내부 ffill/bfill 후 남은 값만 0으로 채우며, chat_count와 unique_chatters 결측은 0으로 채운다.",
        "",
        "## 포함 파일",
        *[f"- `{entry}`" for entry in manifest_entries],
        "",
    ]
    (handoff / "README_Handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_lines = [
        "최종 전달용 필수자료 목록",
        "",
        "이 압축파일은 세션 단위 모델 학습에 필요한 최소 파일만 포함한다.",
        "Method 2 내부 점수표, review 후보 파일, rebuilt CSV, __pycache__, pyc, Excel 파일은 포함하지 않는다.",
        "",
        *manifest_entries,
    ]
    (handoff / "MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    for extra_path in [handoff / "README_Handoff.md", handoff / "MANIFEST.txt"]:
        if extra_path.exists():
            extra_path.unlink()
    print(f"6-file handoff folder를 저장했습니다: {handoff}")


def write_revision_checklist(out, session_model):
    out = Path(out)
    sens = pd.read_csv(out / "m2_sens.csv") if (out / "m2_sens.csv").exists() else pd.DataFrame()
    handoff_session = pd.read_csv(out / "handoff" / "csv" / "session_summary_processed.csv", encoding="utf-8-sig") if (out / "handoff" / "csv" / "session_summary_processed.csv").exists() else pd.DataFrame()
    handoff_minute_all = pd.read_csv(out / "handoff" / "csv" / "minute_all.csv", nrows=5, encoding="utf-8-sig") if (out / "handoff" / "csv" / "minute_all.csv").exists() else pd.DataFrame()
    handoff_minute_model = pd.read_csv(out / "handoff" / "csv" / "minute_model.csv", nrows=5, encoding="utf-8-sig") if (out / "handoff" / "csv" / "minute_model.csv").exists() else pd.DataFrame()
    handoff_readme = (out / "handoff" / "README_Handoff.md").read_text(encoding="utf-8") if (out / "handoff" / "README_Handoff.md").exists() else ""
    session_cluster = (out / "handoff" / "txt" / "session_cluster.txt").read_text(encoding="utf-8") if (out / "handoff" / "txt" / "session_cluster.txt").exists() else ""
    minute_cluster_doc = (out / "handoff" / "txt" / "minute_cluster.txt").read_text(encoding="utf-8") if (out / "handoff" / "txt" / "minute_cluster.txt").exists() else ""
    m2_pipeline = (out / "m2_pipeline.md").read_text(encoding="utf-8") if (out / "m2_pipeline.md").exists() else ""
    m2_eval_plan = (out / "m2_eval_plan.md").read_text(encoding="utf-8") if (out / "m2_eval_plan.md").exists() else ""
    m2_model_notes = (out / "m2_model_notes.md").read_text(encoding="utf-8") if (out / "m2_model_notes.md").exists() else ""
    standalone_script = (out / "handoff" / "code" / "make_session_summary.py").read_text(encoding="utf-8") if (out / "handoff" / "code" / "make_session_summary.py").exists() else ""
    m2_scores_df = pd.read_csv(out / "m2_scores.csv", encoding="utf-8-sig") if (out / "m2_scores.csv").exists() else pd.DataFrame()
    m2_ep = pd.read_csv(out / "m2_ep.csv", encoding="utf-8-sig") if (out / "m2_ep.csv").exists() else pd.DataFrame()
    m2_candidates = pd.read_csv(out / "m2_candidates.csv", encoding="utf-8-sig") if (out / "m2_candidates.csv").exists() else pd.DataFrame()
    base_pred = pd.read_csv(out / "base_pred.csv", encoding="utf-8-sig") if (out / "base_pred.csv").exists() else pd.DataFrame()
    m2_scan = pd.read_csv(out / "m2_scan.csv", encoding="utf-8-sig") if (out / "m2_scan.csv").exists() else pd.DataFrame()
    int_scores = pd.read_csv(out / "int_scores.csv", encoding="utf-8-sig") if (out / "int_scores.csv").exists() else pd.DataFrame()
    m2_reason = pd.read_csv(out / "m2_reason.csv", encoding="utf-8-sig") if (out / "m2_reason.csv").exists() else pd.DataFrame()
    m2_patterns = pd.read_csv(out / "m2_patterns.csv", encoding="utf-8-sig") if (out / "m2_patterns.csv").exists() else pd.DataFrame()
    m2_review = pd.read_csv(out / "m2_review.csv", encoding="utf-8-sig") if (out / "m2_review.csv").exists() else pd.DataFrame()
    forbidden_sens = {"recommended_by_anchor", "selected_threshold", "final_threshold"}
    forbidden_cols = {"viewbot_probability", "true_viewbot_label", "bot_detected"}
    handoff_forbidden = {"gmm", "hdbscan", "hdbscan_noise"}
    output_col_frames = [session_model, sens, m2_scores_df, m2_ep, m2_candidates, base_pred, m2_scan, int_scores, m2_reason, m2_patterns, m2_review]
    forbidden_cols_absent = all(forbidden_cols.isdisjoint(frame.columns) for frame in output_col_frames)
    forbidden_threshold_cols_absent = all(forbidden_sens.isdisjoint(frame.columns) for frame in output_col_frames)
    pattern_cols = {str(c).lower() for c in m2_patterns.columns}
    required_handoff_csv = {"minute_all.csv", "minute_model.csv", "session_summary_processed.csv"}
    handoff_csv_dir = out / "handoff" / "csv"
    handoff_csv_names = {p.name for p in handoff_csv_dir.glob("*.csv") if "rebuilt" not in p.name} if handoff_csv_dir.exists() else set()
    required_zip = out.parent / "df_required_handoff.zip"
    allowed_zip_names = set(REQUIRED_HANDOFF_FILES)
    required_zip_names = _zip_names(required_zip)
    full_zip = out.parent / "df_m2.zip"
    full_zip_names = _zip_names(full_zip)
    full_required_names = set(M2_REQUIRED_ARCHIVE_NAMES)
    required_zip_session_cluster = _zip_entry_text(required_zip, "df_required_handoff/txt/session_cluster.txt")
    required_zip_matches_current = _required_handoff_zip_matches_current(required_zip, out / "handoff")
    required_zip_exact = required_zip_names == allowed_zip_names and len(required_zip_names) == 12
    required_zip_no_method2 = all("method2" not in name.split("/") for name in required_zip_names)
    required_zip_no_m2_csv = all(not (Path(name).name.startswith("m2_") and Path(name).suffix.lower() == ".csv") for name in required_zip_names)
    required_zip_no_rebuilt = all("rebuilt" not in Path(name).name for name in required_zip_names)
    required_zip_no_readme_minimal = all(Path(name).name != "README_minimal.txt" for name in required_zip_names)
    required_zip_no_base_pred = all(Path(name).name != "base_pred.csv" for name in required_zip_names)
    required_zip_no_int_scores = all(Path(name).name != "int_scores.csv" for name in required_zip_names)
    full_zip_no_rebuilt = all("rebuilt" not in Path(name).name for name in full_zip_names)
    full_zip_no_pycache = all("__pycache__" not in name.split("/") for name in full_zip_names)
    full_zip_no_xlsx = all(not name.lower().endswith(".xlsx") for name in full_zip_names)
    full_zip_no_external_projects = all("chzzk-crawler" not in name.split("/") and "my_job" not in name.split("/") for name in full_zip_names)
    raw_forbidden = {"log_viewer", "chat_deficit", "minute_cluster", "minute_mismatch_score", "rra_p", "rra_q"}
    session_forbidden = {"m2_review", "rra_p", "rra_q", "minute_mismatch_score", "episode_count", "review_order"}
    readme_uses_relative = "C:\\" not in handoff_readme and "C:/" not in handoff_readme
    minute_cluster_required_text = all(
        text in minute_cluster_doc
        for text in ["Features:", "Scaling:", "Main algorithm:", "K candidates:", "Selected K:", "Hyperparameters:"]
    )
    session_cluster_required_text = all(
        text in session_cluster
        for text in ["Features:", "Scaling:", "Algorithm:", "K candidates:", "Selected K:", "Hyperparameters:"]
    )
    ep_examples = pd.DataFrame()
    if not m2_ep.empty:
        ep_examples = m2_ep[
            m2_ep["score_source"].eq("rule_rank")
            & m2_ep["threshold_q"].eq(0.95)
            & m2_ep["min_duration"].eq(10)
        ]
    no_auc_true_perf = "not reported as true performance" in m2_pipeline or "does not claim true performance" in m2_pipeline
    standalone_uses_config_snapshot = (
        "CONFIG =" in standalone_script
        and "--fixed-k" in standalone_script
        and "--search-k" not in standalone_script
        and "selection_score" in standalone_script
    )
    m2_rules_absent = not (out / "m2_rules.csv").exists() and not (out / "handoff" / "method2" / "csv" / "m2_rules.csv").exists()
    null_doc = "adaptive interval empirical p" in m2_eval_plan and "m2_sens is only legacy sensitivity diagnostic" in m2_eval_plan
    state_doc = "state dwell evidence" in m2_eval_plan and "KMeans K search is state diagnostic only" in m2_model_notes
    cluster_stability_doc = "RRA consensus rank" in m2_eval_plan and "RRA avoids manual evidence weights" in m2_model_notes
    review_cols = {
        "rra_p",
        "rra_q",
        "review_order",
        "scan_interval_rank",
        "expected_response_rank",
        "state_dwell_rank",
        "interval_anomaly_rank",
        "reason_rank",
        "reason_support_rank",
        "eligible_review",
        "review_qc_reason",
        "n_session_minutes",
        "observed_scan_z",
        "empirical_p",
    }
    scan_duration_matches = False
    if not m2_review.empty and not m2_scan.empty:
        merged_duration = m2_review[["session_key", "top_interval_duration"]].merge(
            m2_scan[["session_key", "top_interval_duration"]],
            on="session_key",
            how="left",
            suffixes=("_review", "_scan"),
        )
        scan_duration_matches = pd.to_numeric(merged_duration["top_interval_duration_review"], errors="coerce").equals(
            pd.to_numeric(merged_duration["top_interval_duration_scan"], errors="coerce")
        )
    eligible_top100_ok = True
    if not m2_review.empty and "eligible_review" in m2_review.columns:
        top100 = m2_review.sort_values("review_order").head(100) if "review_order" in m2_review.columns else m2_review.head(100)
        eligible_top100_ok = not top100["eligible_review"].astype(str).str.lower().isin(["false", "0", "no"]).any()
    scan_clock_gap_ok = True
    if not m2_scan.empty and {"clock_gap_count", "max_clock_gap_min"}.issubset(m2_scan.columns):
        break_gap = 1.0
        scan_clock_gap_ok = (
            pd.to_numeric(m2_scan["clock_gap_count"], errors="coerce").fillna(0).le(0).all()
            and pd.to_numeric(m2_scan["max_clock_gap_min"], errors="coerce").fillna(0).le(break_gap).all()
        )
    checks = [
        ("m2_sens has no forbidden threshold-selection columns", forbidden_sens.isdisjoint(sens.columns)),
        ("all Method 2 outputs have no forbidden threshold-selection columns", forbidden_threshold_cols_absent),
        ("session_summary_processed has cluster_number", "cluster_number" in session_model.columns),
        ("forbidden label/probability columns absent", forbidden_cols_absent),
        ("base_pred.csv exists", (out / "base_pred.csv").exists()),
        ("gmm_diag.csv exists", (out / "gmm_diag.csv").exists()),
        ("hdbscan_diag.csv exists", (out / "hdbscan_diag.csv").exists()),
        ("m2_scores.csv exists", (out / "m2_scores.csv").exists()),
        ("m2_ep.csv exists", (out / "m2_ep.csv").exists()),
        ("m2_candidates.csv exists", (out / "m2_candidates.csv").exists()),
        ("m2_null.csv exists", (out / "m2_null.csv").exists()),
        ("m2_state.csv exists", (out / "m2_state.csv").exists()),
        ("m2_trans.csv exists", (out / "m2_trans.csv").exists()),
        ("m2_scan.csv exists", (out / "m2_scan.csv").exists()),
        ("int_scores.csv exists", (out / "int_scores.csv").exists()),
        ("m2_reason.csv exists", (out / "m2_reason.csv").exists()),
        ("m2_patterns.csv exists", (out / "m2_patterns.csv").exists()),
        ("m2_review.csv exists", (out / "m2_review.csv").exists()),
        ("m2_review has adaptive RRA columns and no label columns", review_cols.issubset(m2_review.columns) and forbidden_cols.isdisjoint(m2_review.columns)),
        ("m2_review top 100 is eligible", eligible_top100_ok),
        ("m2_scan top intervals do not cross clock gaps", scan_clock_gap_ok),
        ("top_interval_duration comes from m2_scan", scan_duration_matches),
        ("m2_review no longer contains legacy candidate/window ranks", {"minute_episode_rank", "p95_minute_rank", "window_anomaly_rank", "discord_rank"}.isdisjoint(m2_review.columns)),
        ("m2_patterns has no confidence or lift columns", "confidence" not in pattern_cols and "lift" not in pattern_cols),
        ("df_required_handoff.zip exists", required_zip.exists()),
        ("df_required_handoff.zip regenerated from current out/handoff", required_zip_matches_current),
        ("df_required_handoff.zip has exactly 12 required files", required_zip_exact),
        ("df_required_handoff.zip contains only allowed required files", required_zip_names == allowed_zip_names),
        ("df_required_handoff.zip has no README_minimal.txt", required_zip_no_readme_minimal),
        ("df_required_handoff.zip has no method2 folder", required_zip_no_method2),
        ("df_required_handoff.zip has no m2 csv files", required_zip_no_m2_csv),
        ("df_required_handoff.zip has no base_pred.csv", required_zip_no_base_pred),
        ("df_required_handoff.zip has no int_scores.csv", required_zip_no_int_scores),
        ("df_required_handoff.zip has no rebuilt csv", required_zip_no_rebuilt),
        ("df_required_handoff.zip has no Windows absolute paths", _zip_has_no_windows_paths(required_zip)),
        ("df_required_handoff session_cluster.txt has no GMM/HDBSCAN legacy text", "GMM" not in required_zip_session_cluster and "HDBSCAN" not in required_zip_session_cluster),
        ("minute_all.csv has only raw minute columns", raw_forbidden.isdisjoint(handoff_minute_all.columns)),
        ("minute_model.csv has only raw minute columns", raw_forbidden.isdisjoint(handoff_minute_model.columns)),
        ("session_summary_processed.csv has cluster_number", "cluster_number" in handoff_session.columns),
        ("session_summary_processed.csv has no review/m2/rra/score columns", session_forbidden.isdisjoint(handoff_session.columns)),
        ("minute_cluster.txt contains dynamic cluster details", minute_cluster_required_text),
        ("session_cluster.txt contains dynamic cluster details", session_cluster_required_text),
        ("README uses relative zip paths", readme_uses_relative),
        ("mc_stab.csv exists", (out / "mc_stab.csv").exists()),
        ("15_null.png exists", (out / "plots" / "15_null.png").exists()),
        ("16_state.png exists", (out / "plots" / "16_state.png").exists()),
        ("17_mc_stab.png exists", (out / "plots" / "17_mc_stab.png").exists()),
        ("18_review.png exists", (out / "plots" / "18_review.png").exists()),
        ("19_reason.png exists", (out / "plots" / "19_reason.png").exists()),
        ("20_rra.png exists", (out / "plots" / "20_rra.png").exists()),
        ("21_interval.png exists", (out / "plots" / "21_interval.png").exists()),
        ("df_m2.zip exists", full_zip.exists()),
        ("df_m2.zip contains all required CSV/MD/plot files", full_required_names.issubset(full_zip_names)),
        ("df_m2.zip contains m2_scan.csv", "out/m2_scan.csv" in full_zip_names),
        ("df_m2.zip contains int_scores.csv", "out/int_scores.csv" in full_zip_names),
        ("df_m2.zip contains m2_review.csv", "out/m2_review.csv" in full_zip_names),
        ("df_m2.zip contains m2_reason.csv", "out/m2_reason.csv" in full_zip_names),
        ("df_m2.zip contains m2_patterns.csv", "out/m2_patterns.csv" in full_zip_names),
        ("df_m2.zip contains 18_review.png", "out/plots/18_review.png" in full_zip_names),
        ("df_m2.zip contains 20_rra.png", "out/plots/20_rra.png" in full_zip_names),
        ("df_m2.zip contains 21_interval.png", "out/plots/21_interval.png" in full_zip_names),
        ("df_m2.zip does not contain rebuilt csv", full_zip_no_rebuilt),
        ("df_m2.zip does not contain __pycache__", full_zip_no_pycache),
        ("df_m2.zip does not contain raw xlsx feature files", full_zip_no_xlsx),
        ("df_m2.zip does not contain external project folders", full_zip_no_external_projects),
        ("minimal handoff has no method2 folder", not (out / "handoff" / "method2").exists()),
        ("make_session_summary.py runs standalone", bool(standalone_script) and "from src" not in standalone_script and "from run" not in standalone_script),
        ("make_session_summary.py uses embedded config snapshot and searches K by default", standalone_uses_config_snapshot),
        ("handoff/csv contains exactly three required csv files", handoff_csv_names == required_handoff_csv),
        ("handoff session_summary_processed has no gmm/hdbscan columns", handoff_forbidden.isdisjoint(handoff_session.columns)),
        ("session_cluster.txt describes KMeans only", "KMeans" in session_cluster and "GMM" not in session_cluster and "HDBSCAN" not in session_cluster),
        ("11_ep.png has multiple episode examples", (out / "plots" / "11_ep.png").exists() and ep_examples["session_key"].nunique() >= 3),
        ("12_m2_rank.png shows actual episode metrics, not only rank values", (out / "plots" / "12_m2_rank.png").exists() and {"episode_total_duration_min", "episode_duration_ratio", "max_episode_score", "p95_minute_score"}.issubset(m2_candidates.columns)),
        ("m2_pipeline explains rule_rank", "`rule_rank` is `minute_mismatch_rank`" in m2_pipeline),
        ("m2_pipeline documents m2_review as main output", "`m2_review.csv` is the main Method 2 output" in m2_pipeline),
        ("m2_pipeline documents RRA q is not probability", "`rra_q` is not probability" in m2_pipeline),
        ("m2_pipeline explains q/duration grid is not final", "q grid and duration grid are not final criteria" in m2_pipeline),
        ("m2_pipeline documents adaptive scan final selection", "adaptive interval scan and robust rank aggregation" in m2_pipeline),
        ("association rule output not generated", m2_rules_absent),
        ("null test documented as no-label sanity check", null_doc),
        ("state transition documented as cluster state diagnostic", state_doc),
        ("cluster stability documented as stability diagnostic", cluster_stability_doc),
        ("no recommended_by_anchor", "recommended_by_anchor" not in sens.columns),
        ("no selected_threshold", "selected_threshold" not in sens.columns),
        ("no final_threshold", "final_threshold" not in sens.columns),
        ("no AUC-ROC true performance", no_auc_true_perf),
        ("no viewbot_probability", all("viewbot_probability" not in frame.columns for frame in output_col_frames)),
        ("no true_viewbot_label", all("true_viewbot_label" not in frame.columns for frame in output_col_frames)),
        ("fixed window final-selection outputs absent", not (out / "m2_window.csv").exists() and not (out / "win_scores.csv").exists() and not (out / "win_ep.csv").exists()),
        ("05_detectors.png not generated", not (out / "plots" / "05_detectors.png").exists()),
        ("06_models.png not generated", not (out / "plots" / "06_models.png").exists()),
        ("14_m2_ml.png not included in minimal handoff", not (out / "handoff" / "img" / "14_m2_ml.png").exists()),
        ("Method 2 uses KMeans only as final minute clustering", _only_value(out / "mc_select.csv", "algorithm", "kmeans")),
        ("Method 2 has no session-level ML methods", not (out / "ml_scores.csv").exists() and not (out / "plots" / "14_m2_ml.png").exists()),
        ("m2_sens has only rule_rank score_source", _only_value(out / "m2_sens.csv", "score_source", "rule_rank")),
        ("m2_ep has only rule_rank score_source", _only_value(out / "m2_ep.csv", "score_source", "rule_rank")),
        ("m2_candidates has only rule_rank score_source", _only_value(out / "m2_candidates.csv", "score_source", "rule_rank")),
    ]
    lines = ["# Revision Checklist", ""]
    lines.extend(f"- [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in checks)
    (out / "revision_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


REQUIRED_HANDOFF_ZIP_KO = "최종_전달용_필수자료.zip"
FULL_ANALYSIS_ZIP_KO = "전체_분석자료_Method2.zip"
REQUIRED_HANDOFF_ZIP_ALIAS = "df_required_handoff.zip"
FULL_ANALYSIS_ZIP_ALIAS = "df_m2.zip"
ARCHIVE_NAMES = (
    FULL_ANALYSIS_ZIP_KO,
    REQUIRED_HANDOFF_ZIP_KO,
    FULL_ANALYSIS_ZIP_ALIAS,
    REQUIRED_HANDOFF_ZIP_ALIAS,
)

M2_REQUIRED_ARCHIVE_NAMES = {
    "out/base_pred.csv",
    "out/base_pred_info.txt",
    "out/m2_scores.csv",
    "out/m2_synth.csv",
    "out/qc_zero_session_review.csv",
    "out/m2_null.csv",
    "out/m2_state.csv",
    "out/m2_trans.csv",
    "out/m2_scan.csv",
    "out/int_scores.csv",
    "out/m2_reason.csv",
    "out/m2_patterns.csv",
    "out/m2_review.csv",
    "out/m2_review_all.csv",
    "out/m2_review_transition_fix_audit.csv",
    "out/m2_review_rank_audit.csv",
    "out/mc_stab.csv",
    "out/mc_select.csv",
    "out/mc_profile.csv",
    "out/minute_cluster_model_selection.csv",
    "out/load_policy.txt",
    "out/active_target_audit.csv",
    "out/interval_anomaly.txt",
    "out/method_registry.md",
    "out/plot_guide.txt",
    "out/plot_manifest.csv",
    "out/plot_audit.csv",
    "out/minute_cluster.txt",
    "out/session_cluster.txt",
    "out/X_core_cols.txt",
    "out/X_no_leak_cols.txt",
    "out/m2_pipeline.md",
    "out/m2_eval_plan.md",
    "out/m2_model_notes.md",
    "out/parameter_rationale.md",
    "out/revision_checklist.md",
    "out/validation_report.txt",
    "out/plots/07_ms.png",
    "out/plots/08_mc.png",
    "out/plots/13_m2_pipe.png",
    "out/plots/15_null.png",
    "out/plots/16_state.png",
    "out/plots/18_review.png",
    "out/plots/19_reason.png",
    "out/plots/20_rra.png",
    "out/plots/21_interval.png",
}

M2_OPTIONAL_ARCHIVE_NAMES = {
    "out/m2_synth_matches.csv",
    "out/gmm_diag.csv",
    "out/hdbscan_diag.csv",
    "out/plots/01_data_quality.png",
    "out/plots/02_dist_time.png",
    "out/plots/03_view_chat.png",
    "out/plots/04_cluster_session.png",
    "out/plots/05_session_k_selection.png",
    "out/plots/06_session_cluster_profile.png",
    "out/plots/07_session_cluster_stability.png",
    "out/plots/08_cluster_minute.png",
}


def _source_for_out_arcname(out, arcname):
    arc_path = Path(arcname)
    return Path(out).joinpath(*arc_path.parts[1:])


def _replace_with_hardlink_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def create_zip(out, project_dir):
    out = Path(out)
    project_dir = Path(project_dir)
    zip_path = project_dir / FULL_ANALYSIS_ZIP_KO
    if zip_path.exists():
        zip_path.unlink()
    missing = [arcname for arcname in sorted(M2_REQUIRED_ARCHIVE_NAMES) if not _source_for_out_arcname(out, arcname).exists()]
    if missing:
        raise FileNotFoundError(f"required df_m2 archive files missing: {missing}")

    written = set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        handoff = out / "handoff"
        if handoff.exists():
            for file in sorted(handoff.rglob("*")):
                if file.is_file():
                    arcname = "out/handoff/" + file.relative_to(handoff).as_posix()
                    _write_archive_file(zf, file, arcname, written)
        for arcname in sorted(M2_REQUIRED_ARCHIVE_NAMES | M2_OPTIONAL_ARCHIVE_NAMES):
            _write_archive_file(zf, _source_for_out_arcname(out, arcname), arcname, written)
    alias = project_dir / FULL_ANALYSIS_ZIP_ALIAS
    _replace_with_hardlink_or_copy(zip_path, alias)
    print(f"압축파일을 저장했습니다: {zip_path}")
    print(f"동일 내용의 호환 파일을 연결했습니다: {alias}")
    return zip_path


REQUIRED_HANDOFF_FILES = {
    "df_required_handoff/txt/session_cluster.txt": ("txt", "session_cluster.txt"),
    "df_required_handoff/img/04_cluster_session.png": ("img", "04_cluster_session.png"),
    "df_required_handoff/csv/minute_all.csv": ("csv", "minute_all.csv"),
    "df_required_handoff/csv/minute_model.csv": ("csv", "minute_model.csv"),
    "df_required_handoff/csv/session_summary_processed.csv": ("csv", "session_summary_processed.csv"),
    "df_required_handoff/code/make_session_summary.py": ("code", "make_session_summary.py"),
}


def create_required_handoff_zip(out, project_dir):
    out = Path(out)
    project_dir = Path(project_dir)
    handoff = out / "handoff"
    zip_path = project_dir / REQUIRED_HANDOFF_ZIP_KO
    if zip_path.exists():
        zip_path.unlink()
    manifest_text = _required_handoff_manifest_text()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for arcname, rel_parts in REQUIRED_HANDOFF_FILES.items():
            if rel_parts is None:
                zf.writestr(arcname, manifest_text)
                continue
            src = handoff.joinpath(*rel_parts)
            if not src.exists():
                raise FileNotFoundError(f"required handoff file missing: {src}")
            zf.write(src, arcname)
    print(f"압축파일을 저장했습니다: {zip_path}")
    return zip_path


def _read_csv_safe(path, nrows=None):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", nrows=nrows)
    except Exception:
        try:
            return pd.read_csv(path, nrows=nrows)
        except Exception:
            return pd.DataFrame()


def _session_key_set(df):
    if df.empty or "session_key" not in df.columns:
        return set()
    return set(df["session_key"].dropna().astype(str))


def _all_zero_qc_result(out, minute_model, session_handoff):
    out = Path(out)
    qc_zero = _read_csv_safe(out / "qc_zero.csv")
    review = _read_csv_safe(out / "qc_zero_session_review.csv")
    review_path = out / "qc_zero_session_review.csv"
    source_keys = _session_key_set(qc_zero)
    review_keys = _session_key_set(review)
    model_keys = _session_key_set(minute_model)
    handoff_keys = _session_key_set(session_handoff)
    required = {
        "run_id", "broad_no", "session_key", "user_id", "category_id", "n_minutes",
        "start_ts", "end_ts", "median_viewer", "max_viewer", "min_viewer",
        "mean_viewer", "total_chat_count", "total_unique_chatters",
        "zero_chat_rate", "all_zero_chat", "v_qc", "v_edge", "v_miss_r",
        "qc_reason", "note",
    }
    missing_cols = sorted(required - set(review.columns))
    if review.empty:
        all_zero_chat_ok = len(source_keys) == 0
        total_chat_zero_ok = len(source_keys) == 0
        reason_terms_ok = True
    else:
        all_zero_chat_ok = review.get("all_zero_chat", pd.Series(False, index=review.index)).astype(str).str.lower().isin(["true", "1", "yes"]).all()
        total_chat_zero_ok = pd.to_numeric(review.get("total_chat_count", pd.Series(np.nan, index=review.index)), errors="coerce").fillna(-1).eq(0).all()
        reason_text = " ".join(review.get("qc_reason", pd.Series("", index=review.index)).fillna("").astype(str).str.lower())
        forbidden_reason_terms = ["confirmed_viewbot", "websocket_failed", "bot_detected", "true_label", "viewbot_probability"]
        reason_terms_ok = not any(term in reason_text for term in forbidden_reason_terms)
    overlap_model = sorted(review_keys.intersection(model_keys))
    overlap_handoff = sorted(review_keys.intersection(handoff_keys))
    return {
        "exists": review_path.exists(),
        "minute_rows": int(len(qc_zero)),
        "source_sessions": int(len(source_keys)),
        "review_sessions": int(len(review_keys)),
        "missing_cols": missing_cols,
        "session_count_match": review_keys == source_keys,
        "all_zero_chat_ok": bool(all_zero_chat_ok),
        "total_chat_zero_ok": bool(total_chat_zero_ok),
        "excluded_ok": not overlap_model and not overlap_handoff,
        "reason_terms_ok": bool(reason_terms_ok),
        "overlap_model": overlap_model[:10],
        "overlap_handoff": overlap_handoff[:10],
    }


def _synthetic_sanity_result(out):
    out = Path(out)
    path = out / "m2_synth.csv"
    df = _read_csv_safe(path)
    required = {
        "status", "reason", "source", "injected_interval_count",
        "scored_synthetic_session_count", "detected_synthetic_session_count",
        "recovered_interval_count", "recovered_rate", "mean_iou",
        "median_review_order", "top10_recall", "top50_recall",
        "top100_recall", "note",
    }
    if df.empty:
        return {
            "exists": path.exists(),
            "status": "missing",
            "reason": "missing_or_empty_m2_synth",
            "rate_text": "not reported",
            "missing_cols": sorted(required - set(df.columns)),
            "status_valid": False,
            "not_run_blank": False,
            "ok_rate_numeric": False,
            "ok_zero_rate_allowed": False,
            "matches_ok": False,
            "interpretation": "synthetic sanity summary is missing.",
        }
    row = df.iloc[0]
    status = str(row.get("status", "")).strip()
    reason = str(row.get("reason", "")).strip()
    recovered_rate = pd.to_numeric(pd.Series([row.get("recovered_rate")]), errors="coerce").iloc[0]
    recovered_count = pd.to_numeric(pd.Series([row.get("recovered_interval_count")]), errors="coerce").iloc[0]
    injected_count = pd.to_numeric(pd.Series([row.get("injected_interval_count")]), errors="coerce").fillna(0).iloc[0]
    status_valid = status in {"not_run", "not_run_stale_input", "ok"}
    not_run_blank = status not in {"not_run", "not_run_stale_input"} or (pd.isna(recovered_rate) and pd.isna(recovered_count))
    ok_rate_numeric = status != "ok" or pd.notna(recovered_rate)
    ok_zero_rate_allowed = status != "ok" or float(injected_count) <= 0 or (pd.notna(recovered_rate) and float(recovered_rate) != 0.0)
    matches_ok = status != "ok" or (out / "m2_synth_matches.csv").exists()
    rate_text = "not reported" if pd.isna(recovered_rate) else f"{float(recovered_rate):.6g}"
    if status in {"not_run", "not_run_stale_input"}:
        interpretation = "synthetic sanity was not executed and is not used as performance evidence."
    elif status == "ok":
        interpretation = "synthetic sanity executed as an internal injected-session check, not true viewbot performance."
    else:
        interpretation = "synthetic sanity status is invalid."
    return {
        "exists": path.exists(),
        "status": status,
        "reason": reason,
        "rate_text": rate_text,
        "missing_cols": sorted(required - set(df.columns)),
        "status_valid": status_valid,
        "not_run_blank": bool(not_run_blank),
        "ok_rate_numeric": bool(ok_rate_numeric),
        "ok_zero_rate_allowed": bool(ok_zero_rate_allowed),
        "matches_ok": bool(matches_ok),
        "interpretation": interpretation,
    }


def _selected_flag_mask(df, column="selected"):
    if df is None or df.empty or column not in df.columns:
        return pd.Series(False, index=df.index if df is not None else pd.Index([]))
    return df[column].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _coerce_int_or_none(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(numeric)


def _k_from_param(value):
    text = str(value).strip()
    if "=" in text:
        text = text.split("=", 1)[1]
    return _coerce_int_or_none(text)


def _selected_session_info(select):
    if select is None or select.empty or "k" not in select.columns:
        return {"selected_count": 0, "k": None, "status": "invalid_selected_count"}
    chosen = select.loc[_selected_flag_mask(select)].copy()
    if len(chosen) != 1:
        return {"selected_count": int(len(chosen)), "k": None, "status": "invalid_selected_count"}
    k = _coerce_int_or_none(chosen.iloc[0].get("k"))
    return {"selected_count": 1, "k": k, "status": "ok" if k is not None else "invalid_selected_k"}


def _selected_minute_info(select):
    if select is None or select.empty:
        return {"selected_count": 0, "k": None, "status": "invalid_selected_count"}
    work = select.copy()
    if "algorithm" in work.columns:
        work = work[work["algorithm"].astype(str).str.lower().eq("kmeans")]
    if work.empty:
        return {"selected_count": 0, "k": None, "status": "invalid_selected_count"}
    chosen = work.loc[_selected_flag_mask(work)].copy()
    if len(chosen) != 1:
        return {"selected_count": int(len(chosen)), "k": None, "status": "invalid_selected_count"}
    row = chosen.iloc[0]
    k = _coerce_int_or_none(row.get("selected_k_num")) if "selected_k_num" in chosen.columns else None
    if k is None:
        k = _k_from_param(row.get("param", ""))
    return {"selected_count": 1, "k": k, "status": "ok" if k is not None else "invalid_selected_k"}


def _selected_session_k(select):
    info = _selected_session_info(select)
    return info["k"] if info["status"] == "ok" else info["status"]


def _selected_minute_k(select):
    info = _selected_minute_info(select)
    return info["k"] if info["status"] == "ok" else info["status"]


def _silhouette_table_lines(select, mode):
    if select.empty:
        return ["- not available"]
    lines = []
    if mode == "session":
        cols = [c for c in ["k", "silhouette", "calinski_harabasz", "davies_bouldin", "selection_score", "selected"] if c in select.columns]
        work = select[cols].copy()
        for _, row in work.iterrows():
            lines.append("- " + ", ".join(f"{c}={row.get(c)}" for c in cols))
    else:
        work = select.copy()
        if "algorithm" in work.columns:
            work = work[work["algorithm"].astype(str).str.lower().eq("kmeans")]
        cols = [c for c in ["param", "silhouette", "calinski_harabasz", "davies_bouldin", "selection_score", "selected"] if c in work.columns]
        for _, row in work.iterrows():
            lines.append("- " + ", ".join(f"{c}={row.get(c)}" for c in cols))
    return lines or ["- not available"]


def _unique_cluster_count(df, column):
    if df is None or df.empty or column not in df.columns:
        return None
    vals = pd.to_numeric(df[column], errors="coerce").dropna()
    if vals.empty:
        vals = df[column].dropna()
    return int(vals.nunique()) if len(vals) else None


def _doc_selected_k(path):
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    patterns = [
        r"최종 선택 K:\s*(\d+)",
        r"Selected K:\s*(\d+)",
        r"최종 선택 하이퍼파라미터:\s*k\s*=\s*(\d+)",
        r"final(?: selected)? hyperparameter:\s*k\s*=\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _minute_model_selection_info(model_selection):
    if model_selection is None or model_selection.empty:
        return {"selected_count": 0, "k": None, "status": "invalid_selected_count"}
    work = model_selection.copy()
    if "algorithm" in work.columns:
        work = work[work["algorithm"].astype(str).str.lower().eq("kmeans")]
    chosen = work.loc[_selected_flag_mask(work, "selected_as_final_state")].copy()
    if len(chosen) != 1:
        return {"selected_count": int(len(chosen)), "k": None, "status": "invalid_selected_count"}
    k = _k_from_param(chosen.iloc[0].get("parameter_setting", ""))
    return {"selected_count": 1, "k": k, "status": "ok" if k is not None else "invalid_selected_k"}


def _kmeans_consistency_result(out, session_select=None, minute_select=None, session_handoff=None, minute_model_selection=None, minute_assign=None, m2_scores=None):
    out = Path(out)
    session_select = session_select if session_select is not None else _read_csv_safe(out / "cluster_select.csv")
    minute_select = minute_select if minute_select is not None else _read_csv_safe(out / "mc_select.csv")
    session_handoff = session_handoff if session_handoff is not None else _read_csv_safe(out / "handoff" / "csv" / "session_summary_processed.csv")
    minute_model_selection = minute_model_selection if minute_model_selection is not None else _read_csv_safe(out / "minute_cluster_model_selection.csv")
    minute_assign = minute_assign if minute_assign is not None else _read_csv_safe(out / "mc_assign.csv")
    m2_scores = m2_scores if m2_scores is not None else _read_csv_safe(out / "m2_scores.csv")

    session_info = _selected_session_info(session_select)
    minute_info = _selected_minute_info(minute_select)
    model_info = _minute_model_selection_info(minute_model_selection)
    session_actual = _unique_cluster_count(session_handoff, "cluster_number")
    minute_actual = _unique_cluster_count(minute_assign, "minute_cluster")
    if minute_actual is None:
        minute_actual = _unique_cluster_count(m2_scores, "minute_cluster")
    session_doc_k = _doc_selected_k(out / "session_cluster.txt")
    minute_doc_k = _doc_selected_k(out / "minute_cluster.txt")

    session_count_ok = session_info["selected_count"] == 1 and session_info["status"] == "ok"
    minute_count_ok = minute_info["selected_count"] == 1 and minute_info["status"] == "ok"
    session_k_match = bool(session_count_ok and session_actual is not None and session_info["k"] == session_actual)
    minute_k_match = bool(minute_count_ok and minute_actual is not None and minute_info["k"] == minute_actual)
    session_doc_match = bool(session_count_ok and session_doc_k is not None and session_info["k"] == session_doc_k)
    minute_doc_match = bool(minute_count_ok and minute_doc_k is not None and minute_info["k"] == minute_doc_k)
    model_k_match = bool(minute_count_ok and model_info["status"] == "ok" and minute_info["k"] == model_info["k"])

    return {
        "session_selected_count": session_info["selected_count"],
        "session_selected_k": session_info["k"],
        "session_selected_status": session_info["status"],
        "session_actual_count": session_actual,
        "session_doc_k": session_doc_k,
        "session_selected_count_ok": session_count_ok,
        "session_k_match_actual": session_k_match,
        "session_k_match_doc": session_doc_match,
        "session_consistency_ok": session_count_ok and session_k_match and session_doc_match,
        "minute_selected_count": minute_info["selected_count"],
        "minute_selected_k": minute_info["k"],
        "minute_selected_status": minute_info["status"],
        "minute_actual_count": minute_actual,
        "minute_doc_k": minute_doc_k,
        "minute_model_selected_count": model_info["selected_count"],
        "minute_model_selected_k": model_info["k"],
        "minute_model_selected_status": model_info["status"],
        "minute_selected_count_ok": minute_count_ok,
        "minute_k_match_actual": minute_k_match,
        "minute_k_match_doc": minute_doc_match,
        "minute_model_k_match": model_k_match,
        "minute_consistency_ok": minute_count_ok and minute_k_match and minute_doc_match and model_k_match,
    }


def _actual_transition_counts_from_minutes(minute_df):
    if minute_df is None or minute_df.empty or not {"session_key", "minute_cluster"}.issubset(minute_df.columns):
        return pd.DataFrame(columns=["session_key", "actual_transition_count"])
    sort_cols = ["session_key"]
    if "minute_ts" in minute_df.columns:
        sort_cols.append("minute_ts")
    elif "minute_idx" in minute_df.columns:
        sort_cols.append("minute_idx")
    work = minute_df.copy()
    if "minute_ts" in work.columns:
        work["minute_ts"] = pd.to_datetime(work["minute_ts"], errors="coerce")
    if "minute_idx" in work.columns:
        work["minute_idx"] = pd.to_numeric(work["minute_idx"], errors="coerce")
    work = work.sort_values(sort_cols).reset_index(drop=True)
    rows = []
    for session_key, group in work.groupby(work["session_key"].astype(str), sort=False):
        states = pd.to_numeric(group["minute_cluster"], errors="coerce").reset_index(drop=True)
        if len(states) <= 1:
            count = 0
        else:
            count = int(states.iloc[1:].ne(states.shift().iloc[1:]).fillna(False).sum())
        rows.append({"session_key": str(session_key), "actual_transition_count": count})
    return pd.DataFrame(rows)


def _state_transition_consistency_result(out, m2_state=None, m2_scores=None, mc_assign=None):
    out = Path(out)
    m2_state = m2_state if m2_state is not None else _read_csv_safe(out / "m2_state.csv")
    m2_scores = m2_scores if m2_scores is not None else _read_csv_safe(out / "m2_scores.csv")
    mc_assign = mc_assign if mc_assign is not None else _read_csv_safe(out / "mc_assign.csv")
    source = m2_scores if not m2_scores.empty and "minute_cluster" in m2_scores.columns else mc_assign
    actual = _actual_transition_counts_from_minutes(source)
    if m2_state.empty or actual.empty or "transition_count" not in m2_state.columns:
        return {
            "checked_sessions": 0,
            "mismatched_sessions": 0,
            "consistency_ok": False,
            "all_adjacent_pair_count": False,
            "fixed_n_minus_1_ok": False,
            "source": "missing",
        }
    observed = m2_state[["session_key", "transition_count", "n_minutes"]].copy()
    observed["session_key"] = observed["session_key"].astype(str)
    observed["transition_count"] = pd.to_numeric(observed["transition_count"], errors="coerce")
    observed["n_minutes"] = pd.to_numeric(observed["n_minutes"], errors="coerce")
    merged = observed.merge(actual, on="session_key", how="left")
    merged["actual_transition_count"] = pd.to_numeric(merged["actual_transition_count"], errors="coerce")
    mismatch = merged["transition_count"].ne(merged["actual_transition_count"]) | merged["actual_transition_count"].isna()
    all_adjacent = bool(len(merged) > 0 and merged["transition_count"].eq((merged["n_minutes"] - 1).clip(lower=0)).all())
    return {
        "checked_sessions": int(len(merged)),
        "mismatched_sessions": int(mismatch.fillna(True).sum()),
        "consistency_ok": int(mismatch.fillna(True).sum()) == 0,
        "all_adjacent_pair_count": all_adjacent,
        "fixed_n_minus_1_ok": not all_adjacent,
        "source": "m2_scores.csv" if source is m2_scores else "mc_assign.csv",
    }


def _family_ranking_result(out, m2_review=None):
    out = Path(out)
    m2_review = m2_review if m2_review is not None else _read_csv_safe(out / "m2_review.csv")
    family_rank_cols = {
        "scan_family_rank",
        "persistence_family_rank",
        "expected_response_family_rank",
        "minute_state_family_rank",
        "interval_anomaly_family_rank",
        "reason_support_family_rank",
    }
    family_strength_cols = {c.replace("_rank", "_strength") for c in family_rank_cols}
    raw_cols = {
        "raw_rra_p",
        "raw_rra_q",
        "scan_interval_rank",
        "empirical_p_rank",
        "scan_strength_rank",
    }
    required = family_rank_cols | family_strength_cols | {
        "family_consensus_score",
        "family_rra_p",
        "family_rra_q",
        "evidence_family_count",
        "ranking_method",
        "rra_p",
        "rra_q",
        "review_order",
        "top_interval_duration",
    } | raw_cols
    missing = sorted(required - set(m2_review.columns))
    method = "equal_weight_family_consensus_plus_family_rra"
    method_ok = bool(not m2_review.empty and "ranking_method" in m2_review.columns and m2_review["ranking_method"].astype(str).eq(method).all())
    if not m2_review.empty and {"rra_q", "family_rra_q", "rra_p", "family_rra_p"}.issubset(m2_review.columns):
        final_equals_family = bool(
            np.allclose(
                pd.to_numeric(m2_review["rra_q"], errors="coerce").fillna(1.0),
                pd.to_numeric(m2_review["family_rra_q"], errors="coerce").fillna(1.0),
                rtol=1e-12,
                atol=1e-12,
            )
            and np.allclose(
                pd.to_numeric(m2_review["rra_p"], errors="coerce").fillna(1.0),
                pd.to_numeric(m2_review["family_rra_p"], errors="coerce").fillna(1.0),
                rtol=1e-12,
                atol=1e-12,
            )
        )
    else:
        final_equals_family = bool(m2_review.empty)
    raw_preserved = bool(raw_cols.issubset(m2_review.columns) and (m2_review.empty or pd.to_numeric(m2_review.get("raw_rra_q"), errors="coerce").notna().any()))
    raw_scan_not_duplicated = bool(final_equals_family and family_rank_cols.issubset(m2_review.columns))
    audit_exists = (out / "m2_review_rank_audit.csv").exists()
    consensus_present = bool("family_consensus_score" in m2_review.columns)
    top10_short_count = 0
    if not m2_review.empty and {"review_order", "top_interval_duration"}.issubset(m2_review.columns):
        top10 = m2_review.sort_values("review_order").head(10)
        duration = pd.to_numeric(top10["top_interval_duration"], errors="coerce")
        top10_short_count = int(duration.le(1).sum())
    order_ok = True
    if not m2_review.empty and {
        "review_order",
        "family_consensus_score",
        "family_rra_q",
        "family_rra_p",
        "persistence_family_rank",
        "scan_family_rank",
        "session_key",
    }.issubset(m2_review.columns):
        ordered = m2_review.sort_values("review_order").reset_index(drop=True)

        def better_or_equal(left, right, eps=1e-12):
            left_score = pd.to_numeric(pd.Series([left.get("family_consensus_score")]), errors="coerce").iloc[0]
            right_score = pd.to_numeric(pd.Series([right.get("family_consensus_score")]), errors="coerce").iloc[0]
            if pd.notna(left_score) and pd.notna(right_score) and abs(float(left_score) - float(right_score)) > eps:
                return float(left_score) > float(right_score)
            return True

        order_ok = all(
            better_or_equal(ordered.iloc[i], ordered.iloc[i + 1])
            for i in range(max(len(ordered) - 1, 0))
        )
    return {
        "method": method,
        "method_ok": method_ok,
        "missing": missing,
        "family_cols_present": not missing,
        "consensus_present": consensus_present,
        "raw_rra_preserved": raw_preserved,
        "raw_scan_not_duplicated": raw_scan_not_duplicated,
        "final_equals_family": final_equals_family,
        "audit_exists": audit_exists,
        "top10_short_count": top10_short_count,
        "order_ok": order_ok,
    }


def _parameter_rationale_result(out):
    path = Path(out) / "parameter_rationale.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    required_terms = [
        "prep.min_n",
        "prep.clock_gap_reset_min",
        "minute_state.viewer_bin_n",
        "minute_state.rolling_windows",
        "minute_cluster.features",
        "RobustScaler",
        "KMeans K candidates",
        "m2_scan.n_perm",
        "m2_scan.max_scan_n",
        "GroupKFold",
        "synthetic_sanity.enabled",
    ]
    forbidden = ["viewbot probability", "true label", "confirmed bot", "confirmed decision"]
    missing = [term for term in required_terms if term not in text]
    lower = text.lower()
    forbidden_found = [term for term in forbidden if term in lower]
    return {
        "exists": path.exists(),
        "missing_terms": missing,
        "forbidden_found": forbidden_found,
        "ok": path.exists() and not missing and not forbidden_found,
    }


def _plot_semantics_result(out):
    out = Path(out)
    audit = _read_csv_safe(out / "plot_audit.csv")
    session = _read_csv_safe(out / "session_summary_processed.csv")
    actual_cluster_count = _unique_cluster_count(session, "cluster_number")

    def row(plot_file):
        if audit.empty or "plot_file" not in audit.columns:
            return pd.Series(dtype=object)
        hit = audit.loc[audit["plot_file"].astype(str).eq(plot_file)]
        return hit.iloc[0] if not hit.empty else pd.Series(dtype=object)

    r04 = row("04_cluster_session.png")
    r05 = row("05_session_k_selection.png")
    r06 = row("06_session_cluster_profile.png")
    r08 = row("08_mc.png")
    r19 = row("19_reason.png")
    r20 = row("20_rra.png")
    r21 = row("21_interval.png")
    legend_count = pd.to_numeric(pd.Series([r04.get("actual_legend_count")]), errors="coerce").iloc[0] if not r04.empty else np.nan
    expected_count = pd.to_numeric(pd.Series([r04.get("expected_category_count")]), errors="coerce").iloc[0] if not r04.empty else np.nan
    r04_colorbar_label = str(r04.get("colorbar_label", ""))
    scaled06 = "scaled profile value" in str(r06.get("colorbar_label", "")) and "not cluster id" in str(r06.get("colorbar_label", ""))
    scaled08 = "scaled profile value" in str(r08.get("colorbar_label", "")) and "not cluster id" in str(r08.get("colorbar_label", ""))
    result = {
        "exists": not audit.empty and (out / "plot_audit.csv").exists(),
        "session_cluster_discrete_legend": str(r04.get("colorbar_or_legend", "")).lower() == "legend"
        and pd.notna(legend_count)
        and pd.notna(expected_count)
        and int(legend_count) == int(expected_count)
        and (actual_cluster_count is None or int(legend_count) == int(actual_cluster_count)),
        "session_cluster_no_colorbar": "cluster_number" not in r04_colorbar_label.lower(),
        "k_selection_score": str(r05.get("primary_visual_encoding", "")).lower() == "selection_score",
        "k_selection_no_inertia_criterion": "not presented as selection criterion" in str(r05.get("note", "")).lower()
        or "not a selection criterion" in str(r05.get("note", "")).lower(),
        "heatmap_scaled_not_cluster": scaled06 and scaled08,
        "reason_top_text": "text_reason_included=True" in str(r19.get("interpretation", "")) or "top explanation reasons only" in str(r19.get("interpretation", "")),
        "rra_labels_limited": "top 5" in str(r20.get("note", "")).lower() or "top review candidates" in str(r20.get("interpretation", "")).lower(),
        "interval_shared_colorbar": str(r21.get("colorbar_or_legend", "")).lower() == "one shared colorbar",
        "cluster_count": actual_cluster_count,
        "legend_count": None if pd.isna(legend_count) else int(legend_count),
    }
    result["all_pass"] = all(
        bool(result[key])
        for key in [
            "exists",
            "session_cluster_discrete_legend",
            "session_cluster_no_colorbar",
            "k_selection_score",
            "k_selection_no_inertia_criterion",
            "heatmap_scaled_not_cluster",
            "reason_top_text",
            "rra_labels_limited",
            "interval_shared_colorbar",
        ]
    )
    return result


def _leak_cols(df):
    leak = []
    for col in df.columns:
        lower = str(col).lower()
        if (
            lower.startswith("m2_")
            or "_m2" in lower
            or "review" in lower
            or "rra" in lower
            or "score" in lower
            or "probability" in lower
            or "label" in lower
        ):
            leak.append(col)
    return leak


def _duplicate_key_count(df, keys):
    if df.empty or not set(keys).issubset(df.columns):
        return 0
    return int(df.duplicated(keys).sum())


def _metric_audit_counts(df):
    counts = {}
    for col in ["viewer_count_last", "chat_count", "unique_chatters"]:
        values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
        counts[f"missing_{col}"] = int(values.isna().sum()) if col in df.columns else len(df)
        counts[f"negative_{col}"] = int(values.lt(0).sum()) if col in df.columns else 0
    return counts


def _chat_lt_unique_count(df):
    if df.empty or not {"chat_count", "unique_chatters"}.issubset(df.columns):
        return 0
    chat = pd.to_numeric(df["chat_count"], errors="coerce")
    unique = pd.to_numeric(df["unique_chatters"], errors="coerce")
    return int(chat.lt(unique).fillna(False).sum())


def _clock_gap_count(df, threshold_min=1.0):
    if df.empty or not set(SESSION_KEY_COLS + ["minute_ts"]).issubset(df.columns):
        return 0
    work = df[SESSION_KEY_COLS + ["minute_ts"]].copy()
    work["minute_ts"] = pd.to_datetime(work["minute_ts"], errors="coerce")
    work = work.dropna(subset=SESSION_KEY_COLS + ["minute_ts"]).sort_values(SESSION_KEY_COLS + ["minute_ts"])
    gap = work.groupby(SESSION_KEY_COLS)["minute_ts"].diff().dt.total_seconds().div(60)
    return int(gap.gt(float(threshold_min)).fillna(False).sum())


def _clock_zrun_max_by_session(minute, gap_reset_min=1.1):
    if minute.empty or not set(SESSION_KEY_COLS + ["minute_ts", "chat_count"]).issubset(minute.columns):
        return pd.Series(dtype=float)
    work = minute.copy()
    work["minute_ts"] = pd.to_datetime(work["minute_ts"], errors="coerce")
    work["chat_count"] = pd.to_numeric(work["chat_count"], errors="coerce").fillna(0).clip(lower=0)
    if "session_key" not in work.columns:
        work["session_key"] = work["run_id"].astype(str) + "_" + work["broad_no"].astype(str)
    work = work.dropna(subset=SESSION_KEY_COLS + ["minute_ts"]).sort_values(SESSION_KEY_COLS + ["minute_ts"]).reset_index(drop=True)
    work["zero"] = work["chat_count"].eq(0)
    work["zrun"] = 0
    for _, idx in work.groupby(SESSION_KEY_COLS, sort=False).groups.items():
        g = work.loc[idx]
        z = g["zero"].fillna(False).astype(bool)
        gap = g["minute_ts"].diff().dt.total_seconds().div(60).gt(float(gap_reset_min)).fillna(False)
        block = (z.ne(z.shift()) | gap).cumsum()
        work.loc[idx, "zrun"] = z.groupby(block).cumcount().add(1).where(z, 0).astype(int).to_numpy()
    return work.groupby("session_key")["zrun"].max()


def _zrun_reset_check(minute_model, session_handoff, cfg=None):
    expected = _clock_zrun_max_by_session(minute_model, clock_gap_reset_min(cfg))
    if expected.empty or session_handoff.empty or "session_key" not in session_handoff.columns or "zrun_max" not in session_handoff.columns:
        return False, "missing minute/session zrun input"
    observed = pd.to_numeric(session_handoff.set_index(session_handoff["session_key"].astype(str))["zrun_max"], errors="coerce")
    expected.index = expected.index.astype(str)
    expected = expected.reindex(observed.index)
    missing = int(expected.isna().sum())
    diff = (observed - expected).abs()
    bad = int(diff.gt(1e-9).fillna(True).sum())
    ok = missing == 0 and bad == 0
    return ok, f"compared={len(observed)}, missing={missing}, mismatched={bad}"


def _required_columns_check(minute_all, minute_model, session_handoff):
    minute_required = {"run_id", "broad_no", "session_key", "minute_ts", "viewer_count_last", "chat_count", "unique_chatters"}
    session_required = set(HANDOFF_SESSION_COLS)
    missing = {
        "minute_all": sorted(minute_required - set(minute_all.columns)),
        "minute_model": sorted(minute_required - set(minute_model.columns)),
        "session_summary_processed": sorted(session_required - set(session_handoff.columns)),
    }
    ok = all(not cols for cols in missing.values())
    detail = "; ".join(f"{name} missing={cols or 'none'}" for name, cols in missing.items())
    return ok, detail


def _standalone_rebuild_check(out):
    handoff = Path(out) / "handoff"
    script = handoff / "code" / "make_session_summary.py"
    minute_all = handoff / "csv" / "minute_all.csv"
    minute_model = handoff / "csv" / "minute_model.csv"
    expected_path = handoff / "csv" / "session_summary_processed.csv"
    rebuilt_path = handoff / "csv" / "session_summary_processed_rebuilt.csv"
    if not script.exists() or not minute_all.exists() or not minute_model.exists() or not expected_path.exists():
        return False, "missing handoff rebuild input"
    if rebuilt_path.exists():
        rebuilt_path.unlink()
    cmd = [
        sys.executable,
        str(script),
        "--minute-all",
        str(minute_all),
        "--minute-model",
        str(minute_model),
        "--out",
        str(rebuilt_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"standalone failed: {proc.stderr.strip()[:500]}"
    if "minute_all.csv는 audit 확인에만 사용합니다." not in proc.stdout:
        return False, "standalone stdout missing minute_all audit statement"
    expected = _read_csv_safe(expected_path)
    rebuilt = _read_csv_safe(rebuilt_path)
    if expected.empty or rebuilt.empty:
        return False, "empty expected or rebuilt CSV"
    if set(expected["session_key"].astype(str)) != set(rebuilt["session_key"].astype(str)):
        return False, "session_key set mismatch"
    if list(expected.columns) != list(rebuilt.columns):
        return False, "column order/name mismatch"
    e = expected.sort_values("session_key").reset_index(drop=True)
    r = rebuilt.sort_values("session_key").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(e, r, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9)
    except AssertionError as exc:
        return False, f"value-level mismatch: {str(exc).splitlines()[0][:300]}"
    return True, f"value-level match for {len(expected)} sessions"


def write_validation_report(out, project_dir, cfg):
    out = Path(out)
    project_dir = Path(project_dir)
    minute_all = _read_csv_safe(out / "handoff" / "csv" / "minute_all.csv")
    minute_model = _read_csv_safe(out / "handoff" / "csv" / "minute_model.csv")
    session_handoff = _read_csv_safe(out / "handoff" / "csv" / "session_summary_processed.csv")
    session_select = _read_csv_safe(out / "cluster_select.csv")
    minute_select = _read_csv_safe(out / "mc_select.csv")
    minute_model_selection = _read_csv_safe(out / "minute_cluster_model_selection.csv")
    m2_review = _read_csv_safe(out / "m2_review.csv")
    m2_scan = _read_csv_safe(out / "m2_scan.csv")
    m2_reason = _read_csv_safe(out / "m2_reason.csv")

    rebuild_ok, rebuild_note = _standalone_rebuild_check(out)
    required_cols_ok, required_cols_note = _required_columns_check(minute_all, minute_model, session_handoff)
    dup_minute_all = _duplicate_key_count(minute_all, SESSION_KEY_COLS + ["minute_ts"])
    dup_minute_model = _duplicate_key_count(minute_model, SESSION_KEY_COLS + ["minute_ts"])
    minute_all_audit = _metric_audit_counts(minute_all)
    minute_model_audit = _metric_audit_counts(minute_model)
    negative_total = sum(v for k, v in {**minute_all_audit, **minute_model_audit}.items() if k.startswith("negative_"))
    chat_lt_unique_all = _chat_lt_unique_count(minute_all)
    chat_lt_unique_model = _chat_lt_unique_count(minute_model)
    gap_gt_1_all = _clock_gap_count(minute_all, threshold_min=1.0)
    gap_gt_1_model = _clock_gap_count(minute_model, threshold_min=1.0)
    zrun_reset_ok, zrun_reset_note = _zrun_reset_check(minute_model, session_handoff)
    raw_leaks = sorted(set(_leak_cols(minute_all) + _leak_cols(minute_model) + [c for c in minute_all.columns if "cluster" in str(c).lower()] + [c for c in minute_model.columns if "cluster" in str(c).lower()]))
    session_leaks = sorted(set(_leak_cols(session_handoff)))
    leakage_ok = not raw_leaks and not session_leaks
    all_zero_in_handoff = 0
    if not minute_all.empty and {"session_key", "chat_count"}.issubset(minute_all.columns) and "session_key" in session_handoff.columns:
        chat_by_session = pd.to_numeric(minute_all["chat_count"], errors="coerce").fillna(0).groupby(minute_all["session_key"].astype(str)).sum()
        all_zero_keys = set(chat_by_session[chat_by_session.eq(0)].index.astype(str))
        all_zero_in_handoff = len(all_zero_keys.intersection(set(session_handoff["session_key"].astype(str))))
    eligible_ok = True
    if not m2_review.empty and "eligible_review" in m2_review.columns:
        top100 = m2_review.sort_values("review_order").head(100) if "review_order" in m2_review.columns else m2_review.head(100)
        eligible_ok = not top100["eligible_review"].astype(str).str.lower().isin(["false", "0", "no"]).any()
    clock_gap_ok = True
    if not m2_scan.empty:
        gap_count = pd.to_numeric(m2_scan.get("clock_gap_count", 0), errors="coerce").fillna(0)
        max_gap = pd.to_numeric(m2_scan.get("max_clock_gap_min", 0), errors="coerce").fillna(0)
        break_gap = float(cfg.get("m2_scan", {}).get("break_on_clock_gap_min", 1))
        clock_gap_ok = bool(gap_count.le(0).all() and max_gap.le(break_gap).all())
    threshold_removed = (
        "reason_support_rank" in m2_review.columns
        and "support_rank" in m2_reason.columns
        and "text_reason_included" in m2_reason.columns
    )
    cluster_counts = (
        session_handoff["cluster_number"].astype("string").fillna("<NA>").value_counts(dropna=False).sort_index()
        if "cluster_number" in session_handoff.columns
        else pd.Series(dtype=int)
    )
    cluster_count_lines = [f"- {idx}: {int(val)}" for idx, val in cluster_counts.items()] or ["- not available"]
    missing_lines = []
    for label, audit in [("minute_all", minute_all_audit), ("minute_model", minute_model_audit)]:
        missing_lines.extend([
            f"- {label} missing viewer_count_last rows: {audit.get('missing_viewer_count_last')}",
            f"- {label} missing chat_count rows: {audit.get('missing_chat_count')}",
            f"- {label} missing unique_chatters rows: {audit.get('missing_unique_chatters')}",
        ])

    checks = [
        ("required columns", required_cols_ok, required_cols_note),
        ("duplicate minute key count", dup_minute_all == 0 and dup_minute_model == 0, f"minute_all={dup_minute_all}, minute_model={dup_minute_model}"),
        ("negative metric counts", negative_total == 0, f"negative_total={negative_total}"),
        ("chat_count < unique_chatters count", chat_lt_unique_all == 0 and chat_lt_unique_model == 0, f"minute_all={chat_lt_unique_all}, minute_model={chat_lt_unique_model}"),
        ("clock gaps >1 min count computed", True, f"minute_all={gap_gt_1_all}, minute_model={gap_gt_1_model}"),
        ("zero-run crossing gap reset check", zrun_reset_ok, zrun_reset_note),
        ("all-zero-chat sessions excluded from KMeans handoff", all_zero_in_handoff == 0, f"all_zero_in_handoff={all_zero_in_handoff}"),
        ("standalone run value-level match", rebuild_ok, rebuild_note),
        ("leakage absence", leakage_ok, f"raw={', '.join(raw_leaks) if raw_leaks else 'none'}; session={', '.join(session_leaks) if session_leaks else 'none'}"),
        ("m2_review eligibility top100", eligible_ok, ""),
        ("m2_scan clock-gap continuity", clock_gap_ok, ""),
        ("review ranking uses rank aggregation without cutoff flags", threshold_removed, ""),
    ]
    all_pass = all(ok for _, ok, _ in checks)

    lines = [
        "# Validation Report",
        "",
        "## Handoff CSV Shapes",
        f"- minute_all.csv: rows={len(minute_all)}, columns={len(minute_all.columns)}",
        f"- minute_model.csv: rows={len(minute_model)}, columns={len(minute_model.columns)}",
        f"- session_summary_processed.csv: rows={len(session_handoff)}, columns={len(session_handoff.columns)}",
        "",
        "## Required Columns",
        f"- {'PASS' if required_cols_ok else 'FAIL'} {required_cols_note}",
        "",
        "## Missing Counts",
        *missing_lines,
        "",
        "## Integrity Counts",
        f"- duplicate minute key count: minute_all={dup_minute_all}, minute_model={dup_minute_model}",
        f"- negative count total: {negative_total}",
        f"- chat_count < unique_chatters count: minute_all={chat_lt_unique_all}, minute_model={chat_lt_unique_model}",
        f"- clock gaps >1 min count: minute_all={gap_gt_1_all}, minute_model={gap_gt_1_model}",
        f"- zero-run crossing gap reset check: {'PASS' if zrun_reset_ok else 'FAIL'} ({zrun_reset_note})",
        "",
        "## Session Cluster",
        f"- features: {', '.join(SESSION_CLUSTER_FEATS)}",
        "- scaler: RobustScaler",
        f"- K candidates: {cfg.get('cluster', {}).get('k_min')}..{cfg.get('cluster', {}).get('k_max')}",
        f"- selected K: {_selected_session_k(session_select)}",
        "- selection rule: maximum composite selection_score, tie-broken by silhouette then smaller K",
        "- silhouette table:",
        *_silhouette_table_lines(session_select, "session"),
        "- cluster_number value counts:",
        *cluster_count_lines,
        "",
        "## Minute Cluster",
        f"- features: {', '.join(cfg.get('minute_cluster', {}).get('features', []))}",
        f"- scaler: {cfg.get('minute_cluster', {}).get('scaler', 'RobustScaler')}",
        f"- K candidates: {cfg.get('minute_cluster', {}).get('kmeans', {}).get('k_min')}..{cfg.get('minute_cluster', {}).get('kmeans', {}).get('k_max')}",
        f"- selected K: {_selected_minute_k(minute_select)}",
        "- selection rule: maximum composite selection_score among configured K candidates",
        "- silhouette table:",
        *_silhouette_table_lines(minute_select, "minute"),
        "",
        "## Checks",
        *[f"- {'PASS' if ok else 'FAIL'} {name}{f' ({detail})' if detail else ''}" for name, ok, detail in checks],
        f"- overall: {'PASS' if all_pass else 'FAIL'}",
        "",
        "## Limits",
        "- No ground-truth label is available; outputs are manual review priority evidence.",
        "- `cluster_number`, `minute_cluster`, and `review_order` are not probability values or ground-truth labels.",
        "- Any operational conclusion requires raw WebSocket/chat QC.",
    ]
    text = "\n".join(lines) + "\n"
    (out / "validation_report.txt").write_text(text, encoding="utf-8")
    (project_dir / "validation_report.txt").write_text(text, encoding="utf-8")
    print(f"검증 보고서를 저장했습니다: {out / 'validation_report.txt'}")
    print(f"검증 보고서를 저장했습니다: {project_dir / 'validation_report.txt'}")
    return {
        "rebuild_ok": rebuild_ok,
        "raw_leaks": raw_leaks,
        "session_leaks": session_leaks,
        "eligible_ok": eligible_ok,
        "clock_gap_ok": clock_gap_ok,
        "threshold_removed": threshold_removed,
        "required_cols_ok": required_cols_ok,
        "duplicate_key_ok": dup_minute_all == 0 and dup_minute_model == 0,
        "negative_counts_ok": negative_total == 0,
        "chat_unique_ok": chat_lt_unique_all == 0 and chat_lt_unique_model == 0,
        "zrun_reset_ok": zrun_reset_ok,
        "all_zero_excluded_ok": all_zero_in_handoff == 0,
        "all_pass": all_pass,
    }


def _sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hangul_count(text):
    return sum(1 for ch in str(text) if "\uac00" <= ch <= "\ud7a3")


def _zip_forbidden_members(names):
    bad = []
    for name in names:
        norm = str(name).replace("\\", "/")
        lower = norm.lower()
        parts = lower.split("/")
        filename = parts[-1]
        if (
            "__pycache__" in parts
            or filename.endswith(".pyc")
            or filename.endswith(".xlsx")
            or "session_summary_processed_rebuilt" in filename
            or ":\\" in norm
            or ":/" in norm
            or "\\" in str(name)
            or "chzzk-crawler" in parts
            or "my_job" in parts
        ):
            bad.append(name)
    return bad


def _compare_handoff_zip_extract(zip_path, handoff):
    zip_path = Path(zip_path)
    handoff = Path(handoff)
    if not zip_path.exists():
        return False, "zip 없음"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        rows = []
        for arcname, rel_parts in REQUIRED_HANDOFF_FILES.items():
            extracted = tmp_path / Path(arcname)
            source = handoff.joinpath(*rel_parts) if rel_parts is not None else None
            if source is None or not source.exists() or not extracted.exists():
                rows.append((arcname, False, "missing"))
                continue
            rows.append((arcname, _sha256_file(source) == _sha256_file(extracted), _sha256_file(source)))
        bad = [r for r in rows if not r[1]]
        if bad:
            return False, "; ".join(f"{name}:{detail}" for name, _, detail in bad[:5])
        return True, "; ".join(f"{name}={digest[:12]}" for name, _, digest in rows)


def _standalone_rebuild_check_tmp(out):
    handoff = Path(out) / "handoff"
    script = handoff / "code" / "make_session_summary.py"
    minute_all = handoff / "csv" / "minute_all.csv"
    minute_model = handoff / "csv" / "minute_model.csv"
    expected_path = handoff / "csv" / "session_summary_processed.csv"
    if not script.exists() or not minute_all.exists() or not minute_model.exists() or not expected_path.exists():
        return False, "standalone 입력 누락"
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt_path = Path(tmp) / "session_summary_processed_rebuilt.csv"
        cmd = [
            sys.executable,
            str(script),
            "--minute-all",
            str(minute_all),
            "--minute-model",
            str(minute_model),
            "--out",
            str(rebuilt_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, f"standalone 실패: {proc.stderr.strip()[:300]}"
        if "minute_all.csv는 audit 확인에만 사용합니다." not in proc.stdout:
            return False, "stdout audit 확인 문구 없음"
        expected = _read_csv_safe(expected_path)
        rebuilt = _read_csv_safe(rebuilt_path)
        try:
            pd.testing.assert_frame_equal(
                expected.sort_values("session_key").reset_index(drop=True),
                rebuilt.sort_values("session_key").reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-9,
                atol=1e-9,
            )
        except AssertionError as exc:
            return False, f"value mismatch: {str(exc).splitlines()[0][:200]}"
    return True, f"value-level match rows={len(expected)} cols={len(expected.columns)}"


def _compute_gap_zero_run_df(minute, cfg=None):
    gap_reset_min = clock_gap_reset_min(cfg)
    work = minute.copy()
    if work.empty:
        return work.assign(_gap_zero_run=pd.Series(dtype=int))
    work["minute_ts"] = pd.to_datetime(work["minute_ts"], errors="coerce")
    work["chat_count"] = pd.to_numeric(work.get("chat_count"), errors="coerce").fillna(0)
    work = work.sort_values(SESSION_KEY_COLS + ["minute_ts"]).reset_index(drop=False).rename(columns={"index": "_orig_index"})
    work["_zero_chat"] = work["chat_count"].eq(0)
    work["_gap_zero_run"] = 0
    for _, idx in work.groupby(SESSION_KEY_COLS, sort=False).groups.items():
        g = work.loc[idx]
        z = g["_zero_chat"].fillna(False).astype(bool)
        gap = g["minute_ts"].diff().dt.total_seconds().div(60.0).gt(gap_reset_min).fillna(False)
        block = (z.ne(z.shift()) | gap).cumsum()
        work.loc[idx, "_gap_zero_run"] = z.groupby(block).cumcount().add(1).where(z, 0).astype(int).to_numpy()
    return work.sort_values("_orig_index").drop(columns=["_orig_index"])


def _minute_state_zrun_check(out, cfg=None):
    feat = _read_csv_safe(Path(out) / "minute_model_feat.csv")
    if feat.empty or "zero_run_len" not in feat.columns:
        return False, "minute_model_feat.csv 또는 zero_run_len 없음"
    computed = _compute_gap_zero_run_df(feat, cfg)
    observed = pd.to_numeric(feat["zero_run_len"], errors="coerce").fillna(-1).astype(int)
    expected = pd.to_numeric(computed["_gap_zero_run"], errors="coerce").fillna(-2).astype(int)
    mismatch = int((observed.to_numpy() != expected.to_numpy()).sum())
    return mismatch == 0, f"mismatched_rows={mismatch}, rows={len(feat)}"


def _interval_zrun_check(out, cfg=None):
    scan = _read_csv_safe(Path(out) / "m2_scan.csv")
    scores = _read_csv_safe(Path(out) / "m2_scores.csv")
    if scan.empty or scores.empty:
        return False, "m2_scan 또는 m2_scores 없음"
    scores = _compute_gap_zero_run_df(scores, cfg)
    scores["minute_idx"] = pd.to_numeric(scores.get("minute_idx"), errors="coerce")
    bad = 0
    checked = 0
    groups = {str(k): g for k, g in scores.groupby(scores["session_key"].astype(str), sort=False)}
    for _, row in scan.iterrows():
        part = groups.get(str(row.get("session_key")), pd.DataFrame())
        if part.empty:
            continue
        start = pd.to_numeric(pd.Series([row.get("top_interval_start_idx")]), errors="coerce").iloc[0]
        end = pd.to_numeric(pd.Series([row.get("top_interval_end_idx")]), errors="coerce").iloc[0]
        if pd.isna(start) or pd.isna(end):
            continue
        sub = part[part["minute_idx"].between(start, end)]
        expected = pd.to_numeric(sub["_gap_zero_run"], errors="coerce").max()
        observed = pd.to_numeric(pd.Series([row.get("interval_max_zero_run")]), errors="coerce").iloc[0]
        checked += 1
        if pd.isna(expected) or pd.isna(observed) or abs(float(expected) - float(observed)) > 1e-9:
            bad += 1
    return bad == 0, f"checked={checked}, mismatched_intervals={bad}"


def _old_vs_gap_session_change(minute_model, session_handoff, cfg=None):
    if minute_model.empty or session_handoff.empty:
        return {"changed_sessions": 0, "changed_cluster_number": 0, "gap_zrun_max": 0, "gap_zrun_sum": 0}
    gap = _clock_zrun_max_by_session(minute_model, clock_gap_reset_min(cfg))
    work = minute_model.copy()
    work["minute_ts"] = pd.to_datetime(work["minute_ts"], errors="coerce")
    work["chat_count"] = pd.to_numeric(work["chat_count"], errors="coerce").fillna(0)
    work = work.sort_values(SESSION_KEY_COLS + ["minute_ts"])
    work["_zero"] = work["chat_count"].eq(0)
    work["_old_zrun"] = work.groupby(SESSION_KEY_COLS)["_zero"].transform(lambda s: s.astype(bool).groupby(s.astype(bool).ne(s.astype(bool).shift()).cumsum()).cumcount().add(1).where(s.astype(bool), 0))
    old = work.groupby("session_key")["_old_zrun"].max()
    gap.index = gap.index.astype(str)
    old.index = old.index.astype(str)
    common = gap.index.intersection(old.index)
    changed = int((gap.loc[common] != old.loc[common]).sum())
    changed_cluster = 0
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import RobustScaler

        old_session = session_handoff.copy()
        old_session = old_session.set_index(old_session["session_key"].astype(str))
        old_session["zrun_max"] = old.reindex(old_session.index)
        old_session["log_zrun"] = np.log1p(pd.to_numeric(old_session["zrun_max"], errors="coerce").fillna(0))
        x = old_session[SESSION_CLUSTER_FEATS].replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.median(numeric_only=True)).fillna(0)
        xs = RobustScaler().fit_transform(x)
        current_k = int(pd.to_numeric(session_handoff["cluster_number"], errors="coerce").nunique())
        if current_k >= 2:
            labels = KMeans(n_clusters=current_k, random_state=42, n_init=50).fit_predict(xs)
            current = pd.to_numeric(old_session["cluster_number"], errors="coerce").fillna(-1).astype(int).to_numpy()
            changed_cluster = int((labels != current).sum())
    except Exception:
        changed_cluster = -1
    return {
        "changed_sessions": changed,
        "changed_cluster_number": changed_cluster,
        "gap_zrun_max": int(pd.to_numeric(gap, errors="coerce").max()) if len(gap) else 0,
        "gap_zrun_sum": int(pd.to_numeric(gap, errors="coerce").sum()) if len(gap) else 0,
    }


def _collect_validation(out, project_dir, cfg, expect_zips=True):
    out = Path(out)
    project_dir = Path(project_dir)
    handoff = out / "handoff"
    minute_all = _read_csv_safe(handoff / "csv" / "minute_all.csv")
    minute_model = _read_csv_safe(handoff / "csv" / "minute_model.csv")
    session_handoff = _read_csv_safe(handoff / "csv" / "session_summary_processed.csv")
    session_select = _read_csv_safe(out / "cluster_select.csv")
    minute_select = _read_csv_safe(out / "mc_select.csv")
    minute_model_selection = _read_csv_safe(out / "minute_cluster_model_selection.csv")
    minute_assign = _read_csv_safe(out / "mc_assign.csv")
    m2_scores = _read_csv_safe(out / "m2_scores.csv")
    m2_state = _read_csv_safe(out / "m2_state.csv")
    m2_review = _read_csv_safe(out / "m2_review.csv")
    stats = _handoff_stats(out, handoff)
    qc_zero_result = _all_zero_qc_result(out, minute_model, session_handoff)
    synth_result = _synthetic_sanity_result(out)
    kmeans_consistency = _kmeans_consistency_result(
        out,
        session_select=session_select,
        minute_select=minute_select,
        session_handoff=session_handoff,
        minute_model_selection=minute_model_selection,
        minute_assign=minute_assign,
        m2_scores=m2_scores,
    )
    state_transition = _state_transition_consistency_result(out, m2_state=m2_state, m2_scores=m2_scores, mc_assign=minute_assign)
    family_ranking = _family_ranking_result(out, m2_review=m2_review)
    parameter_rationale = _parameter_rationale_result(out)
    plot_semantics = _plot_semantics_result(out)

    checks = []
    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    strict_ok = list(minute_all.columns) == STRICT_MINUTE_CORE_COLS and list(minute_model.columns) == STRICT_MINUTE_CORE_COLS
    add("strict raw minute columns", strict_ok, f"minute_all={list(minute_all.columns)}; minute_model={list(minute_model.columns)}")
    req_ok, req_note = _required_columns_check(minute_all, minute_model, session_handoff)
    add("required columns", req_ok, req_note)
    dup_all = _duplicate_key_count(minute_all, SESSION_KEY_COLS + ["minute_ts"])
    dup_model = _duplicate_key_count(minute_model, SESSION_KEY_COLS + ["minute_ts"])
    add("duplicate key count", dup_all == 0 and dup_model == 0, f"minute_all={dup_all}, minute_model={dup_model}")
    audit_all = _metric_audit_counts(minute_all)
    audit_model = _metric_audit_counts(minute_model)
    neg_total = sum(v for k, v in {**audit_all, **audit_model}.items() if k.startswith("negative_"))
    add("negative viewer/chat/unique count", neg_total == 0, f"negative_total={neg_total}")
    chat_unique_all = _chat_lt_unique_count(minute_all)
    chat_unique_model = _chat_lt_unique_count(minute_model)
    add("chat_count < unique_chatters count", chat_unique_all == 0 and chat_unique_model == 0, f"minute_all={chat_unique_all}, minute_model={chat_unique_model}")
    gap_all = _clock_gap_count(minute_all, 1.0)
    gap_model = _clock_gap_count(minute_model, 1.0)
    add("clock gaps >1 minute count", True, f"minute_all={gap_all}, minute_model={gap_model}")
    zrun_ok, zrun_note = _zrun_reset_check(minute_model, session_handoff, cfg)
    add("session zrun_max gap-aware", zrun_ok, zrun_note)
    minute_zrun_ok, minute_zrun_note = _minute_state_zrun_check(out, cfg)
    add("minute_state zero_run_len gap-aware", minute_zrun_ok, minute_zrun_note)
    interval_zrun_ok, interval_zrun_note = _interval_zrun_check(out, cfg)
    add("interval_max_zero_run gap-aware", interval_zrun_ok, interval_zrun_note)
    old_gap = _old_vs_gap_session_change(minute_model, session_handoff, cfg)
    rebuild_ok, rebuild_note = _standalone_rebuild_check_tmp(out)
    add("standalone rebuild value-level match", rebuild_ok, rebuild_note)
    raw_leaks = sorted(set(_leak_cols(minute_all) + _leak_cols(minute_model) + [c for c in minute_all.columns if "cluster" in str(c).lower()] + [c for c in minute_model.columns if "cluster" in str(c).lower()]))
    banned_session = ["m2", "review", "rra", "score", "probability", "prob", "label", "threshold", "gmm", "hdbscan", "lof", "iforest", "ecod", "anomaly"]
    session_leaks = [c for c in session_handoff.columns if any(token in str(c).lower() for token in banned_session)]
    add("leakage absence", not raw_leaks and not session_leaks, f"raw={raw_leaks or 'none'}; session={session_leaks or 'none'}")
    add("cluster_number exists", "cluster_number" in session_handoff.columns, f"columns={list(session_handoff.columns)}")
    add(
        "cluster_select selected=True row count",
        kmeans_consistency["session_selected_count_ok"],
        f"selected_count={kmeans_consistency['session_selected_count']}; selected_k={kmeans_consistency['session_selected_k']}; status={kmeans_consistency['session_selected_status']}",
    )
    add(
        "session K consistency",
        kmeans_consistency["session_consistency_ok"],
        f"selected_k={kmeans_consistency['session_selected_k']}; actual_unique={kmeans_consistency['session_actual_count']}; txt_k={kmeans_consistency['session_doc_k']}",
    )
    add(
        "mc_select KMeans selected=True row count",
        kmeans_consistency["minute_selected_count_ok"],
        f"selected_count={kmeans_consistency['minute_selected_count']}; selected_k={kmeans_consistency['minute_selected_k']}; status={kmeans_consistency['minute_selected_status']}",
    )
    add(
        "minute K consistency",
        kmeans_consistency["minute_consistency_ok"],
        f"selected_k={kmeans_consistency['minute_selected_k']}; actual_unique={kmeans_consistency['minute_actual_count']}; txt_k={kmeans_consistency['minute_doc_k']}; model_selection_k={kmeans_consistency['minute_model_selected_k']}; model_selection_count={kmeans_consistency['minute_model_selected_count']}",
    )
    model_selection_required = {
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
    }
    selected_final_count = int(minute_model_selection.get("selected_as_final_state", pd.Series(dtype=object)).astype(str).str.lower().isin(["true", "1", "yes"]).sum()) if not minute_model_selection.empty else 0
    algorithms_present = set(minute_model_selection.get("algorithm", pd.Series(dtype=object)).dropna().astype(str))
    add(
        "minute cluster model selection table",
        model_selection_required.issubset(minute_model_selection.columns) and selected_final_count == 1 and {"KMeans", "GMM", "HDBSCAN"}.issubset(algorithms_present),
        f"columns_missing={sorted(model_selection_required - set(minute_model_selection.columns))}; selected_final_count={selected_final_count}; algorithms={sorted(algorithms_present)}",
    )
    forbidden_review_flags = {"short_interval_flag", "weak_null_evidence_flag", "transient_spike_caution"}
    add("m2_review threshold caution flags absent", forbidden_review_flags.isdisjoint(set(m2_review.columns)), f"present={sorted(forbidden_review_flags.intersection(set(m2_review.columns)))}")
    order_ok = family_ranking["order_ok"]
    add("m2_review order uses family consensus", order_ok, "review_order의 1차 정렬 근거가 family_consensus_score 및 family RRA인지 확인")
    add("raw scan evidence not duplicated in final RRA", family_ranking["raw_scan_not_duplicated"], "scan_interval/empirical_p/scan_strength are represented by scan_family_rank")
    add("family rank columns present", family_ranking["family_cols_present"], f"missing={family_ranking['missing']}")
    add("family_consensus_score present", family_ranking["consensus_present"], "")
    add("raw_rra_q preserved for audit", family_ranking["raw_rra_preserved"], "")
    add("m2_review_rank_audit.csv exists", family_ranking["audit_exists"], "")
    forbidden_review = {"viewbot_probability", "true_viewbot_label", "final_threshold", "selected_threshold", "predicted_label", "bot_detected"}
    add("m2_review forbidden columns absent", forbidden_review.isdisjoint(set(m2_review.columns)), f"forbidden_present={sorted(forbidden_review.intersection(set(m2_review.columns)))}")
    add(
        "m2_state transition_count actual state changes",
        state_transition["consistency_ok"],
        f"source={state_transition['source']}; checked={state_transition['checked_sessions']}; mismatched_sessions={state_transition['mismatched_sessions']}",
    )
    add(
        "transition_count not fixed adjacent-pair count",
        state_transition["fixed_n_minus_1_ok"],
        f"transition_count_equals_n_minutes_minus_1_for_all_sessions={state_transition['all_adjacent_pair_count']}",
    )
    add("synthetic sanity summary exists/status", synth_result["exists"] and synth_result["status_valid"] and not synth_result["missing_cols"], f"status={synth_result['status']}; reason={synth_result['reason']}; missing_cols={synth_result['missing_cols']}")
    add("synthetic sanity not_run metrics blank", synth_result["not_run_blank"], f"status={synth_result['status']}; recovered_rate={synth_result['rate_text']}")
    add("synthetic sanity ok metrics valid", synth_result["ok_rate_numeric"] and synth_result["ok_zero_rate_allowed"] and synth_result["matches_ok"], f"status={synth_result['status']}; recovered_rate={synth_result['rate_text']}; matches_ok={synth_result['matches_ok']}")
    add("qc_zero_session_review exists/schema", qc_zero_result["exists"] and not qc_zero_result["missing_cols"], f"missing_cols={qc_zero_result['missing_cols']}")
    add("qc_zero_session_review session count matches qc_zero.csv", qc_zero_result["session_count_match"], f"qc_zero_sessions={qc_zero_result['source_sessions']}; review_sessions={qc_zero_result['review_sessions']}")
    add("qc_zero_session_review all-zero content", qc_zero_result["all_zero_chat_ok"] and qc_zero_result["total_chat_zero_ok"], f"all_zero_chat={qc_zero_result['all_zero_chat_ok']}; total_chat_zero={qc_zero_result['total_chat_zero_ok']}")
    add("qc_zero_session_review excluded from behavior/session handoff", qc_zero_result["excluded_ok"], f"minute_model_overlap={qc_zero_result['overlap_model']}; session_summary_overlap={qc_zero_result['overlap_handoff']}")
    add("qc_zero_session_review qc_reason avoids forbidden certainty terms", qc_zero_result["reason_terms_ok"], "qc_reason is manual-QC wording only")
    legacy_outputs = [out / name for name in ["m2_ep.csv", "m2_sens.csv", "m2_candidates.csv", "m2_eval.csv", "m2_stability.csv"]]
    legacy_disabled_ok = bool(cfg.get("episode_grid", {}).get("enabled", False)) or not any(path.exists() for path in legacy_outputs)
    add("legacy episode grid disabled outputs absent", legacy_disabled_ok, ", ".join(f"{p.name}={p.exists()}" for p in legacy_outputs))
    add(
        "07 stability plot uses mc_stab",
        (out / "mc_stab.csv").exists() and (out / "plots" / "07_session_cluster_stability.png").exists(),
        "minute KMeans behavior-state stability diagnostic; mc_stab.csv 및 backward-compatible plot filename 존재",
    )
    plot_manifest = _read_csv_safe(out / "plot_manifest.csv")
    actual_plots = sorted(f"plots/{p.name}" for p in (out / "plots").glob("*.png")) if (out / "plots").exists() else []
    manifest_plots = sorted(plot_manifest.get("file", pd.Series(dtype=object)).dropna().astype(str).tolist()) if not plot_manifest.empty else []
    plot_guide_text = (out / "plot_guide.txt").read_text(encoding="utf-8") if (out / "plot_guide.txt").exists() else ""
    stale_plot_mentions = [name for name in ["10_sens.png", "11_ep.png", "12_m2_rank.png", "17_mc_stab.png"] if name in plot_guide_text]
    add(
        "plot_manifest matches actual plots",
        actual_plots == manifest_plots and not stale_plot_mentions,
        f"actual={actual_plots}; manifest={manifest_plots}; stale_plot_mentions={stale_plot_mentions}",
    )
    add("plot_audit exists", plot_semantics["exists"], "")
    add("04_cluster_session uses discrete legend for cluster_number", plot_semantics["session_cluster_discrete_legend"], f"cluster_count={plot_semantics['cluster_count']}; legend_count={plot_semantics['legend_count']}")
    add("04_cluster_session has no cluster_number colorbar", plot_semantics["session_cluster_no_colorbar"], "")
    add("05_session_k_selection shows selection_score", plot_semantics["k_selection_score"], "")
    add("05_session_k_selection does not present inertia as selection criterion", plot_semantics["k_selection_no_inertia_criterion"], "")
    add("06/08 heatmap colorbars labeled as scaled profile values, not cluster ids", plot_semantics["heatmap_scaled_not_cluster"], "")
    add("19_reason uses text_reason_included counts", plot_semantics["reason_top_text"], "")
    add("20_rra labels limited to top candidates", plot_semantics["rra_labels_limited"], "")
    add("21_interval uses one shared colorbar", plot_semantics["interval_shared_colorbar"], "")
    add("parameter_rationale.md exists", parameter_rationale["exists"], "")
    add("cfg parameters documented in parameter_rationale.md", not parameter_rationale["missing_terms"], f"missing={parameter_rationale['missing_terms']}")
    add("parameter_rationale.md avoids certainty wording", not parameter_rationale["forbidden_found"], f"forbidden={parameter_rationale['forbidden_found']}")
    registry = (out / "method_registry.md").read_text(encoding="utf-8") if (out / "method_registry.md").exists() else ""
    add("method_registry exists/status", all(x in registry for x in ["final", "diagnostic", "appendix", "removed"]), f"hangul={_hangul_count(registry)}")
    x_core = (out / "X_core_cols.txt").read_text(encoding="utf-8") if (out / "X_core_cols.txt").exists() else ""
    x_no = (out / "X_no_leak_cols.txt").read_text(encoding="utf-8") if (out / "X_no_leak_cols.txt").exists() else ""
    add("cluster_number leakage warning", "목표 누수" in x_core and "cluster_number" in x_core, "X_core_cols.txt warning 확인")
    add("X_no_leak excludes cluster_number", not any(line.strip() == "cluster_number" for line in x_no.splitlines()), "cluster_number 안전 feature처럼 기재되지 않음")
    doc_paths = [
        handoff / "txt" / "session_cluster.txt",
        out / "method_registry.md",
        out / "m2_pipeline.md",
        out / "m2_model_notes.md",
        out / "m2_eval_plan.md",
        out / "parameter_rationale.md",
        out / "base_pred_info.txt",
        out / "interval_anomaly.txt",
        out / "load_policy.txt",
        out / "plot_guide.txt",
    ]
    korean_details = []
    korean_ok = True
    for path in doc_paths:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        count = _hangul_count(text)
        korean_details.append(f"{path.name}={count}")
        if count == 0:
            korean_ok = False
    add("Korean docs hangul count", korean_ok, "; ".join(korean_details))
    expected_rel = {str(Path(name).relative_to("df_required_handoff")).replace("\\", "/") for name in REQUIRED_HANDOFF_FILES}
    actual_rel = {str(path.relative_to(handoff)).replace("\\", "/") for path in handoff.rglob("*") if path.is_file()} if handoff.exists() else set()
    readme_manifest_text = "\n".join(
        path.read_text(encoding="utf-8") for path in [handoff / "README_Handoff.md", handoff / "MANIFEST.txt"] if path.exists()
    )
    mentioned_missing = sorted(rel for rel in expected_rel if rel not in actual_rel)
    unexpected_files = sorted(actual_rel - expected_rel)
    stale_forbidden_mentions = [token for token in ["Copied", "Missing Optional", "README_minimal.txt", "session_summary_processed_rebuilt.csv"] if token in readme_manifest_text]
    add(
        "handoff README/MANIFEST file list consistency",
        actual_rel == expected_rel and not mentioned_missing and not unexpected_files and not stale_forbidden_mentions,
        f"missing={mentioned_missing}; unexpected={unexpected_files}; stale_mentions={stale_forbidden_mentions}",
    )
    handoff_text = ""
    for path in [p for p in handoff.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".py"}]:
        try:
            handoff_text += "\n" + path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            pass
    banned_phrases = [
        "ViewBot 확률",
        "ViewBot 확정",
        "AI가 판단한 의심 세션",
        "threshold 기준으로 의심",
        "KMeans가 찾아낸 ViewBot",
        "정확도",
        "Recall",
        "Precision",
        "분류 성능",
        "Copied",
        "Missing Optional",
    ]
    found_banned = [phrase for phrase in banned_phrases if phrase in handoff_text]
    add("handoff banned claim phrases absent", not found_banned, f"found={found_banned}")

    req_zip = project_dir / REQUIRED_HANDOFF_ZIP_KO
    req_alias = project_dir / REQUIRED_HANDOFF_ZIP_ALIAS
    full_zip = project_dir / FULL_ANALYSIS_ZIP_KO
    full_alias = project_dir / FULL_ANALYSIS_ZIP_ALIAS
    zip_hashes = {}
    if expect_zips:
        for path in [req_zip]:
            if path.exists():
                zip_hashes[path.name] = _sha256_file(path)
        req_names = _zip_names(req_zip)
        full_names = _zip_names(full_zip)
        add("Korean handoff zip exists", req_zip.exists(), f"{req_zip.name}={req_zip.exists()}")
        add("English handoff alias absent", not req_alias.exists(), f"{req_alias.name}={req_alias.exists()}")
        add("final handoff zip exact 6 files", req_names == set(REQUIRED_HANDOFF_FILES) and len(req_names) == 6, f"count={len(req_names)}, extra={sorted(req_names - set(REQUIRED_HANDOFF_FILES))}, missing={sorted(set(REQUIRED_HANDOFF_FILES) - req_names)}")
        add("final handoff zip forbidden files absent", not _zip_forbidden_members(req_names) and not any(Path(n).name.startswith("m2_") or Path(n).name in {"base_pred.csv", "int_scores.csv"} for n in req_names), f"bad={_zip_forbidden_members(req_names)}")
        add("handoff-only packaging does not require full archive", True, f"{full_zip.name}={full_zip.exists()}, {full_alias.name}={full_alias.exists()}")
        extract_ok, extract_note = _compare_handoff_zip_extract(req_zip, handoff)
        add("out/handoff vs extracted final zip hash", extract_ok, extract_note)
    else:
        add("zip validation deferred", True, "pre-zip phase")

    cluster_counts = session_handoff["cluster_number"].astype("string").fillna("<NA>").value_counts(dropna=False).sort_index().to_dict() if "cluster_number" in session_handoff.columns else {}
    return {
        "checks": checks,
        "all_pass": all(c["ok"] for c in checks),
        "stats": stats,
        "missing": {"minute_all": audit_all, "minute_model": audit_model},
        "clock_gaps": {"minute_all": gap_all, "minute_model": gap_model},
        "old_gap": old_gap,
        "cluster_counts": cluster_counts,
        "session_select": session_select,
        "minute_select": minute_select,
        "kmeans_consistency": kmeans_consistency,
        "state_transition": state_transition,
        "family_ranking": family_ranking,
        "parameter_rationale": parameter_rationale,
        "plot_semantics": plot_semantics,
        "qc_zero_result": qc_zero_result,
        "synthetic_sanity": synth_result,
        "zip_hashes": zip_hashes,
        "raw_leaks": raw_leaks,
        "session_leaks": session_leaks,
        "rebuild_ok": rebuild_ok,
        "required_cols_ok": req_ok,
        "duplicate_key_ok": dup_all == 0 and dup_model == 0,
        "negative_counts_ok": neg_total == 0,
        "chat_unique_ok": chat_unique_all == 0 and chat_unique_model == 0,
        "zrun_reset_ok": zrun_ok and minute_zrun_ok and interval_zrun_ok,
        "all_zero_excluded_ok": stats["all_zero_sessions"] + stats["minute_model_sessions"] == stats["minute_all_sessions"],
        "eligible_ok": True,
        "clock_gap_ok": True,
        "threshold_removed": order_ok and forbidden_review_flags.isdisjoint(set(m2_review.columns)),
    }


def _eval_dir_from_cfg(out, cfg):
    out = Path(out)
    eval_cfg = cfg.get("eval_robustness", {}) if cfg else {}
    raw = eval_cfg.get("out_dir", "out/eval")
    path = Path(raw)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == out.name:
        return out.parent / path
    if parts and parts[0] == "out":
        return out.parent / path
    return out / path


def run_eval_robustness_safely(out, cfg):
    out = Path(out)
    if not bool(cfg.get("eval_robustness", {}).get("enabled", False)):
        return {"status": "disabled", "eval_dir": str(_eval_dir_from_cfg(out, cfg))}
    eval_dir = _eval_dir_from_cfg(out, cfg)
    eval_dir.mkdir(parents=True, exist_ok=True)
    error_path = eval_dir / "evaluation_error.txt"
    try:
        from src.eval_robustness import run_eval_robustness

        result = run_eval_robustness(out, cfg)
        if error_path.exists():
            error_path.unlink()
        lines = [
            "status: PASS",
            f"eval_dir: {result.get('eval_dir', str(eval_dir))}",
            "main_outputs_changed: no",
            "note: evaluation diagnostics are separate from main validation.",
        ]
        (eval_dir / "evaluation_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"status": "ok", "eval_dir": result.get("eval_dir", str(eval_dir))}
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        lines = [
            "status: FAIL",
            f"eval_dir: {eval_dir}",
            f"error_file: {error_path}",
            "note: main pipeline outputs were left in place; evaluation validation is separate.",
        ]
        (eval_dir / "evaluation_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Evaluation robustness failed; wrote {error_path}")
        return {"status": "failed", "eval_dir": str(eval_dir)}


def _eval_validation_lines(out, cfg):
    if not bool(cfg.get("eval_robustness", {}).get("enabled", False)):
        return [
            "## Evaluation Robustness Validation",
            "- status: disabled",
            "- interpretation: separate from main validation",
        ]
    eval_dir = _eval_dir_from_cfg(out, cfg)
    required = [
        "family_ablation_summary.csv",
        "aggregation_sensitivity_summary.csv",
        "evidence_balance_summary.csv",
        "tie_audit.csv",
        "evaluation_report.md",
    ]
    error_path = eval_dir / "evaluation_error.txt"
    status = "FAIL" if error_path.exists() else ("PASS" if (eval_dir / "evaluation_report.md").exists() else "not_run")
    lines = [
        "## Evaluation Robustness Validation",
        f"- status: {status}",
        f"- eval_dir: {eval_dir}",
        "- scope: separate label-free review ranking robustness diagnostics",
    ]
    for name in required:
        lines.append(f"- {name}: {'PASS' if (eval_dir / name).exists() else 'MISSING'}")
    lines.append(f"- evaluation_error.txt: {'present' if error_path.exists() else 'absent'}")
    return lines


def write_validation_report(out, project_dir, cfg, expect_zips=True):
    out = Path(out)
    project_dir = Path(project_dir)
    result = _collect_validation(out, project_dir, cfg, expect_zips=expect_zips)
    checks = result["checks"]
    stats = result["stats"]
    missing = result["missing"]
    qc_zero = result["qc_zero_result"]
    synthetic = result["synthetic_sanity"]
    kmeans = result["kmeans_consistency"]
    state_transition = result["state_transition"]
    family_ranking = result["family_ranking"]
    parameter_rationale = result["parameter_rationale"]
    plot_semantics = result["plot_semantics"]
    lines = [
        "# Validation Report",
        "",
        f"검증 단계: {'post-zip' if expect_zips else 'pre-zip'}",
        f"전체 결과: {'PASS' if result['all_pass'] else 'FAIL'}",
        "",
        "## 데이터/QC 수치",
        f"- minute_all session 수: {stats['minute_all_sessions']}",
        f"- minute_model session 수: {stats['minute_model_sessions']}",
        f"- all-zero-chat session 수: {stats['all_zero_sessions']}",
        f"- all-zero-chat minute 수: {stats['all_zero_minutes']}",
        f"- session_summary_processed session 수: {stats['session_summary_sessions']}",
        f"- min_n filter 전/후 session 수: {stats['sessions_before_min_n']} / {stats['sessions_after_min_n']}",
        "",
        "## 결측/무결성",
        f"- minute_all viewer_count_last original missing rows: {missing['minute_all'].get('missing_viewer_count_last')}",
        f"- minute_model viewer_count_last original missing rows: {missing['minute_model'].get('missing_viewer_count_last')}",
        f"- viewer_count_last missing after ffill/bfill: validation은 standalone report와 값 단위 재생성으로 확인",
        f"- chat_count missing rows: minute_all={missing['minute_all'].get('missing_chat_count')}, minute_model={missing['minute_model'].get('missing_chat_count')}",
        f"- unique_chatters missing rows: minute_all={missing['minute_all'].get('missing_unique_chatters')}, minute_model={missing['minute_model'].get('missing_unique_chatters')}",
        f"- clock gaps >1 min count: minute_all={result['clock_gaps']['minute_all']}, minute_model={result['clock_gaps']['minute_model']}",
        f"- old no-gap zero-run 대비 바뀐 session 수: {result['old_gap']['changed_sessions']}",
        f"- 바뀐 cluster_number 수(참고 재계산): {result['old_gap']['changed_cluster_number']}",
        f"- gap-aware zrun_max max/sum: {result['old_gap']['gap_zrun_max']} / {result['old_gap']['gap_zrun_sum']}",
        "",
        "## KMeans",
        f"- selected session K from cluster_select.csv: {kmeans['session_selected_k'] if kmeans['session_selected_status'] == 'ok' else kmeans['session_selected_status']}",
        f"- actual session cluster_number unique count: {kmeans['session_actual_count']}",
        f"- session K consistency: {'PASS' if kmeans['session_consistency_ok'] else 'FAIL'}",
        "- session selection_score table:",
        *_silhouette_table_lines(result["session_select"], "session"),
        f"- cluster_number value counts: {result['cluster_counts']}",
        f"- selected minute K from mc_select.csv: {kmeans['minute_selected_k'] if kmeans['minute_selected_status'] == 'ok' else kmeans['minute_selected_status']}",
        f"- actual minute_cluster unique count: {kmeans['minute_actual_count']}",
        f"- minute K consistency: {'PASS' if kmeans['minute_consistency_ok'] else 'FAIL'}",
        "- minute selection_score table:",
        *_silhouette_table_lines(result["minute_select"], "minute"),
        "",
        "## State Transition Consistency",
        "- m2_state transition_count definition: actual minute_cluster state changes within session",
        f"- source used for recomputation: {state_transition['source']}",
        f"- checked sessions: {state_transition['checked_sessions']}",
        f"- mismatched transition_count sessions: {state_transition['mismatched_sessions']}",
        f"- transition_count consistency: {'PASS' if state_transition['consistency_ok'] else 'FAIL'}",
        f"- transition_count equals n_minutes-1 for all sessions: {'FAIL' if state_transition['all_adjacent_pair_count'] else 'PASS'}",
        "",
        "## Method2 Family Ranking",
        f"- final ranking method: {family_ranking['method']}",
        f"- raw scan evidence duplicated in final RRA: {'FAIL' if not family_ranking['raw_scan_not_duplicated'] else 'PASS'}",
        f"- family rank columns present: {'PASS' if family_ranking['family_cols_present'] else 'FAIL'}",
        f"- family_consensus_score present: {'PASS' if family_ranking['consensus_present'] else 'FAIL'}",
        f"- raw_rra_q preserved for audit: {'PASS' if family_ranking['raw_rra_preserved'] else 'FAIL'}",
        f"- m2_review_rank_audit.csv exists: {'PASS' if family_ranking['audit_exists'] else 'FAIL'}",
        f"- top10 short interval count, duration <= 1: WARNING {family_ranking['top10_short_count']}",
        "- interpretation: review_order is manual review priority, not probability or ground-truth label",
        "",
        "## Parameter Rationale",
        f"- parameter_rationale.md exists: {'PASS' if parameter_rationale['exists'] else 'FAIL'}",
        f"- documented cfg parameters: {'PASS' if not parameter_rationale['missing_terms'] else 'FAIL'} missing={parameter_rationale['missing_terms']}",
        f"- certainty wording absent: {'PASS' if not parameter_rationale['forbidden_found'] else 'FAIL'} forbidden={parameter_rationale['forbidden_found']}",
        "",
        "## Plot Semantics",
        f"- 04_cluster_session uses discrete legend for cluster_number: {'PASS' if plot_semantics['session_cluster_discrete_legend'] else 'FAIL'}",
        f"- 04_cluster_session has no cluster_number colorbar: {'PASS' if plot_semantics['session_cluster_no_colorbar'] else 'FAIL'}",
        f"- 05_session_k_selection shows selection_score: {'PASS' if plot_semantics['k_selection_score'] else 'FAIL'}",
        f"- 05_session_k_selection does not present inertia as selection criterion: {'PASS' if plot_semantics['k_selection_no_inertia_criterion'] else 'FAIL'}",
        f"- 06/08 heatmap colorbars labeled as scaled profile values, not cluster ids: {'PASS' if plot_semantics['heatmap_scaled_not_cluster'] else 'FAIL'}",
        f"- 19_reason uses text_reason_included counts: {'PASS' if plot_semantics['reason_top_text'] else 'FAIL'}",
        f"- 20_rra labels not heavily overlapped or limited to top candidates: {'PASS' if plot_semantics['rra_labels_limited'] else 'FAIL'}",
        f"- 21_interval uses one shared colorbar: {'PASS' if plot_semantics['interval_shared_colorbar'] else 'FAIL'}",
        f"- plot_audit exists: {'PASS' if plot_semantics['exists'] else 'FAIL'}",
        "",
        "## Synthetic Sanity",
        f"- status: {synthetic['status']}",
        f"- reason: {synthetic['reason']}",
        f"- recovered_rate: {synthetic['rate_text']}",
        f"- interpretation: {synthetic['interpretation']}",
        "",
        *_eval_validation_lines(out, cfg),
        "",
        "## All-zero Chat QC Bucket",
        f"- qc_zero minute rows: {qc_zero['minute_rows']}",
        f"- qc_zero sessions: {qc_zero['source_sessions']}",
        f"- qc_zero_session_review.csv: {'PASS' if qc_zero['exists'] and not qc_zero['missing_cols'] and qc_zero['session_count_match'] else 'FAIL'}",
        f"- all_zero_chat all true: {'PASS' if qc_zero['all_zero_chat_ok'] else 'FAIL'}",
        f"- total_chat_count all zero: {'PASS' if qc_zero['total_chat_zero_ok'] else 'FAIL'}",
        f"- excluded from minute_model/session_summary_processed: {'PASS' if qc_zero['excluded_ok'] else 'FAIL'}",
        "- interpretation: preserved for manual QC; not used as behavior-model label.",
        "",
        "## Zip hash",
        *[f"- {name}: {digest}" for name, digest in sorted(result["zip_hashes"].items())],
        "",
        "## PASS/FAIL",
        *[f"- {'PASS' if c['ok'] else 'FAIL'} {c['name']}: {c['detail']}" for c in checks],
        "",
        "## 해석 제한",
        "- 최종 표현은 확정 탐지가 아니라 viewer-chat mismatch 기반 수동 검토 우선순위 생성이다.",
        "- cluster_number, minute_cluster, rra_q, empirical_p, review_order는 정답 라벨이나 확률이 아니다.",
    ]
    text = "\n".join(lines) + "\n"
    (out / "validation_report.txt").write_text(text, encoding="utf-8")
    (project_dir / "validation_report.txt").write_text(text, encoding="utf-8")
    print(f"검증 보고서를 저장했습니다: {out / 'validation_report.txt'}")
    print(f"검증 보고서를 저장했습니다: {project_dir / 'validation_report.txt'}")
    return result


def write_revision_checklist(out, session_model=None, validation_result=None):
    out = Path(out)
    if validation_result is None:
        cfg, project_dir, _ = load_runtime_config("cfg.yml")
        validation_result = _collect_validation(out, project_dir, cfg, expect_zips=True)
    lines = [
        "# Revision Checklist",
        "",
        "최종 제출 가능" if validation_result["all_pass"] else "최종 제출 불가",
        "",
    ]
    lines.extend(f"- [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in validation_result["checks"])
    (out / "revision_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(cfg_path="cfg.yml", archive_mode="full"):
    if archive_mode not in {"full", "handoff", "none"}:
        raise ValueError("archive_mode must be one of: full, handoff, none")
    cfg, project_dir, cfg_file = load_runtime_config(cfg_path)
    out = Path(cfg["path"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    remove_existing_archives(project_dir)
    clean_obsolete_outputs(out)
    print(f"설정 파일을 읽었습니다: {cfg_file}")

    raw, file_audit, used_windows = load_features(cfg)
    save(file_audit, out / "file_audit.csv")
    save(used_windows, out / "used_windows.csv")
    write_load_audit_summary(file_audit, out)
    write_load_policy(out, cfg, file_audit)

    minute_base = prep_minute(raw, cfg)
    minute_all_raw, minute_model_raw, qc_zero_raw, row_qc = split_minute(minute_base)
    save(minute_all_raw[raw_minute_cols(minute_all_raw)], out / "minute_all.csv")
    save(minute_model_raw[raw_minute_cols(minute_model_raw)], out / "minute_model.csv")
    save(qc_zero_raw[raw_minute_cols(qc_zero_raw)], out / "qc_zero.csv")
    qc_zero_review = write_qc_zero_session_review(qc_zero_raw, out)
    print(f"저장했습니다: {out / 'qc_zero_session_review.csv'} {qc_zero_review.shape}")
    save(row_qc, out / "row_qc.csv")

    minute_full = add_minute_state_features(minute_base, cfg)
    minute_all_feat, minute_model_feat, qc_zero_feat, _ = split_minute(minute_full)
    save(minute_all_feat[feat_minute_cols(minute_all_feat)], out / "minute_all_feat.csv")
    save(minute_model_feat[feat_minute_cols(minute_model_feat)], out / "minute_model_feat.csv")
    save(qc_zero_feat[feat_minute_cols(qc_zero_feat)], out / "qc_zero_feat.csv")

    minute_model_clustered = add_minute_clusters(minute_model_feat, out, cfg)
    m2_scores = build_m2_scores(minute_model_clustered, out, cfg)
    build_expected_response(minute_model_clustered, out, cfg)
    if bool(cfg.get("episode_grid", {}).get("enabled", False)):
        m2_ep = build_m2_episodes(m2_scores, out, cfg)
        m2_sens = build_m2_sensitivity(m2_ep, m2_scores, out, cfg)
        m2_candidates = build_m2_candidates(m2_ep, m2_scores, out, cfg)
        build_m2_eval(m2_ep, m2_candidates, m2_scores, m2_sens, out, cfg)
        build_m2_stability(m2_ep, m2_candidates, m2_scores, out, cfg)
    else:
        remove_legacy_episode_outputs(out)
        m2_ep = pd.DataFrame()
        m2_sens = pd.DataFrame()
        m2_candidates = pd.DataFrame()
    build_m2_scan(m2_scores, out, cfg)
    build_interval_anomaly(out, cfg)

    session_all, session_model, session_qc = make_session(minute_all_feat, minute_model_clustered, cfg)
    save(session_all, out / "session_all.csv")
    save(session_qc, out / "session_qc.csv")
    session_model = add_cluster(session_model, out, cfg)
    save(session_model, out / "session_summary_processed.csv")
    write_session_cluster_alias(out)
    write_minute_cluster_doc(out, cfg)
    write_x_col_files(out)

    build_mc_stability(minute_model_clustered, out, cfg)
    make_plots(minute_all_feat, minute_model_clustered, session_all, session_model, out)
    build_m2_null(m2_scores, out, cfg)
    build_m2_state(m2_scores, out, cfg)
    build_m2_reason(out, cfg)
    build_m2_review(out, cfg)
    m2_synth = build_m2_synth(out, cfg)
    write_m2_docs(out, cfg, synthetic_available=bool(not m2_synth.empty and "status" in m2_synth.columns and m2_synth["status"].astype(str).eq("ok").any()))
    write_korean_m2_docs(out, cfg)
    write_parameter_rationale(out, cfg)
    build_m2_review_plots(out, cfg)
    run_eval_robustness_safely(out, cfg)
    if not bool(cfg.get("episode_grid", {}).get("enabled", False)):
        remove_legacy_episode_outputs(out)
    write_plot_doc(out)
    write_method_registry(out, cfg)
    write_handoff_package(out, project_dir, cfg)
    pre_validation = write_validation_report(out, project_dir, cfg, expect_zips=False)
    write_revision_checklist(out, session_model, pre_validation)
    if not pre_validation["all_pass"]:
        remove_existing_archives(project_dir)
        raise RuntimeError("pre-zip validation failed; final zip files were not created")
    if archive_mode == "none":
        return
    create_required_handoff_zip(out, project_dir)
    if archive_mode == "full":
        create_zip(out, project_dir)
    post_validation = write_validation_report(out, project_dir, cfg, expect_zips=True)
    write_revision_checklist(out, session_model, post_validation)
    if not post_validation["all_pass"]:
        remove_existing_archives(project_dir)
        raise RuntimeError("post-zip validation failed; final zip files were removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", nargs="?", default="cfg.yml")
    parser.add_argument("--archive-mode", choices=["full", "handoff", "none"], default="full")
    args = parser.parse_args()
    main(args.cfg_path, archive_mode=args.archive_mode)
