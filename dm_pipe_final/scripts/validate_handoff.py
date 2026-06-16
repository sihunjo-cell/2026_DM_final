from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run  # noqa: E402


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _is_png(path):
    path = Path(path)
    return path.exists() and path.stat().st_size > 0 and path.read_bytes()[:8] == PNG_SIGNATURE


def _zip_has_required(zip_path, required):
    if not zip_path.exists():
        return False, [f"missing zip: {zip_path}"]
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    missing = sorted(set(required) - names)
    extra = sorted(names - set(required))
    problems = []
    if missing:
        problems.append(f"missing entries: {missing}")
    if extra:
        problems.append(f"extra entries: {extra}")
    return not problems, problems


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "cfg.yml"
    cfg, project_dir, _ = run.load_runtime_config(cfg_path)
    out = Path(cfg["path"]["out_dir"])
    result = run.write_validation_report(out, project_dir, cfg)

    checks = []
    ok, problems = _zip_has_required(project_dir / run.REQUIRED_HANDOFF_ZIP_KO, set(run.REQUIRED_HANDOFF_FILES))
    checks.append((f"{run.REQUIRED_HANDOFF_ZIP_KO} exact manifest", ok, "; ".join(problems)))
    required_handoff_pngs = sorted(
        "/".join(("handoff",) + tuple(rel_parts))
        for rel_parts in run.REQUIRED_HANDOFF_FILES.values()
        if rel_parts is not None and rel_parts[-1].lower().endswith(".png")
    )
    for rel in required_handoff_pngs:
        checks.append((f"{rel} is valid PNG", _is_png(out / rel), ""))
    checks.extend([
        ("validation_report overall PASS", bool(result["all_pass"]), ""),
        ("required columns present", bool(result["required_cols_ok"]), ""),
        ("duplicate minute keys absent", bool(result["duplicate_key_ok"]), ""),
        ("negative metric counts absent", bool(result["negative_counts_ok"]), ""),
        ("chat_count < unique_chatters absent", bool(result["chat_unique_ok"]), ""),
        ("zero-run crossing gap reset", bool(result["zrun_reset_ok"]), ""),
        ("all-zero-chat sessions excluded from KMeans handoff", bool(result["all_zero_excluded_ok"]), ""),
        ("qc_zero_session_review valid", bool(result["qc_zero_result"]["exists"] and result["qc_zero_result"]["session_count_match"] and result["qc_zero_result"]["all_zero_chat_ok"] and result["qc_zero_result"]["total_chat_zero_ok"] and result["qc_zero_result"]["excluded_ok"]), ""),
        ("synthetic sanity status valid", bool(result["synthetic_sanity"]["status_valid"] and result["synthetic_sanity"]["not_run_blank"] and result["synthetic_sanity"]["ok_rate_numeric"] and result["synthetic_sanity"]["ok_zero_rate_allowed"]), ""),
        ("handoff rebuild match", bool(result["rebuild_ok"]), ""),
        ("raw minute leak columns absent", not result["raw_leaks"], ",".join(result["raw_leaks"])),
        ("session leak columns absent", not result["session_leaks"], ",".join(result["session_leaks"])),
        ("m2_review top100 eligible", bool(result["eligible_ok"]), ""),
        ("m2_scan clock continuity", bool(result["clock_gap_ok"]), ""),
        ("reason threshold removed from ranking", bool(result["threshold_removed"]), ""),
    ])

    failed = [(name, detail) for name, ok, detail in checks if not ok]
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
