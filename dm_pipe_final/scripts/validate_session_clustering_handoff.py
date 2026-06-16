from pathlib import Path
import sys
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run  # noqa: E402


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FORBIDDEN_ZIP_BASENAMES = {
    "README_Handoff.md",
    "MANIFEST.txt",
    "minute_cluster.txt",
    "X_core_cols.txt",
    "X_no_leak_cols.txt",
    "08_cluster_minute.png",
    "m2_review.csv",
    "m2_scores.csv",
    "m2_scan.csv",
    "base_pred.csv",
    "int_scores.csv",
    "qc_zero_session_review.csv",
    run.FULL_ANALYSIS_ZIP_KO,
    run.FULL_ANALYSIS_ZIP_ALIAS,
    run.REQUIRED_HANDOFF_ZIP_ALIAS,
    "session_clustering_handoff_FINAL.zip",
}
DERIVED_TOKENS = [
    "feature",
    "cluster",
    "review",
    "anomaly",
    "probability",
    "predicted",
    "label",
    "rra",
    "m2",
    "score",
    "bot_detected",
]


def _zip_names(zip_path):
    if not zip_path.exists():
        return set()
    with zipfile.ZipFile(zip_path) as zf:
        return set(zf.namelist())


def _check_png(path):
    return path.exists() and path.stat().st_size > 0 and path.read_bytes()[:8] == PNG_SIGNATURE


def _read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def _raw_minute_ok(df):
    if list(df.columns) != run.STRICT_MINUTE_CORE_COLS:
        return False
    bad = [col for col in df.columns if any(token in col.lower() for token in DERIVED_TOKENS)]
    return not bad


def _session_cluster_doc_ok(path):
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    required = ["feature", "RobustScaler", "KMeans", "K 후보", "최종 선택 K", "random_state", "n_init"]
    return all(term in text for term in required)


def _validate(cfg_path):
    cfg, project_dir, _ = run.load_runtime_config(cfg_path)
    out = Path(cfg["path"]["out_dir"])
    handoff = out / "handoff"
    zip_path = project_dir / run.REQUIRED_HANDOFF_ZIP_KO
    expected = set(run.REQUIRED_HANDOFF_FILES)
    names = _zip_names(zip_path)

    checks = {}
    checks["exact 6-file handoff zip"] = zip_path.exists() and names == expected and len(names) == 6
    checks["forbidden zip files absent"] = not any(Path(name).name in FORBIDDEN_ZIP_BASENAMES or Path(name).name.startswith("m2_") for name in names)
    checks["alias/full archives absent"] = not any((project_dir / name).exists() for name in [
        run.REQUIRED_HANDOFF_ZIP_ALIAS,
        run.FULL_ANALYSIS_ZIP_KO,
        run.FULL_ANALYSIS_ZIP_ALIAS,
        "session_clustering_handoff_FINAL.zip",
    ])
    minute_all = _read_csv(handoff / "csv" / "minute_all.csv")
    minute_model = _read_csv(handoff / "csv" / "minute_model.csv")
    session_summary = _read_csv(handoff / "csv" / "session_summary_processed.csv")
    checks["raw minute files contain no derived/leakage columns"] = _raw_minute_ok(minute_all) and _raw_minute_ok(minute_model)
    checks["session_summary_processed has cluster_number"] = "cluster_number" in session_summary.columns
    selected_k = run._doc_selected_k(handoff / "txt" / "session_cluster.txt")
    actual_k = int(pd.to_numeric(session_summary["cluster_number"], errors="coerce").nunique()) if "cluster_number" in session_summary.columns else None
    checks["selected K matches cluster_number unique count"] = selected_k is not None and selected_k == actual_k
    checks["session_cluster.txt has required method details"] = _session_cluster_doc_ok(handoff / "txt" / "session_cluster.txt")
    checks["04_cluster_session.png valid"] = _check_png(handoff / "img" / "04_cluster_session.png")
    rebuild_ok, _ = run._standalone_rebuild_check_tmp(out)
    checks["make_session_summary.py rebuilds session_summary_processed"] = rebuild_ok
    return checks


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "cfg.yml"
    checks = _validate(cfg_path)
    display = [
        "exact 6-file handoff zip",
        "raw minute files contain no derived/leakage columns",
        "session_summary_processed has cluster_number",
        "selected K matches cluster_number unique count",
        "make_session_summary.py rebuilds session_summary_processed",
    ]
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        for name in failed:
            print(f"FAIL {name}")
        return 1
    for name in display:
        print(f"PASS {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
