from pathlib import Path
import hashlib
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run  # noqa: E402


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _remove_stale_archives(project_dir):
    stale = [
        run.REQUIRED_HANDOFF_ZIP_KO,
        run.REQUIRED_HANDOFF_ZIP_ALIAS,
        run.FULL_ANALYSIS_ZIP_KO,
        run.FULL_ANALYSIS_ZIP_ALIAS,
        "session_clustering_handoff_FINAL.zip",
    ]
    for name in stale:
        path = Path(project_dir) / name
        if path.exists():
            path.unlink()


def _verify_zip(zip_path, handoff):
    expected = set(run.REQUIRED_HANDOFF_FILES)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        if names != expected or len(names) != 6:
            raise RuntimeError(f"zip manifest mismatch: count={len(names)}, extra={sorted(names - expected)}, missing={sorted(expected - names)}")
        for arcname, rel_parts in run.REQUIRED_HANDOFF_FILES.items():
            source = Path(handoff).joinpath(*rel_parts)
            with zf.open(arcname) as f:
                data_hash = hashlib.sha256(f.read()).hexdigest()
            source_hash = _sha256(source)
            if data_hash != source_hash:
                raise RuntimeError(f"hash mismatch for {arcname}")


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "cfg.yml"
    cfg, project_dir, _ = run.load_runtime_config(cfg_path)
    out = Path(cfg["path"]["out_dir"])
    _remove_stale_archives(project_dir)
    run.write_handoff_package(out, project_dir, cfg)
    zip_path = run.create_required_handoff_zip(out, project_dir)
    _verify_zip(zip_path, out / "handoff")
    result = run.write_validation_report(out, project_dir, cfg, expect_zips=True)
    if not result["all_pass"]:
        raise RuntimeError("post-package validation failed")
    print("PASS exact 6-file handoff zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
