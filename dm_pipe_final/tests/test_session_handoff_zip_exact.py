import tempfile
import unittest
from pathlib import Path
import zipfile

import pandas as pd

import run
from scripts import validate_session_clustering_handoff as handoff_validator


class SessionHandoffZipExactTest(unittest.TestCase):
    def test_required_zip_has_exact_six_entries_and_no_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            out = project / "out"
            handoff = out / "handoff"
            for arcname, rel_parts in run.REQUIRED_HANDOFF_FILES.items():
                path = handoff.joinpath(*rel_parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix.lower() == ".png":
                    path.write_bytes(b"\x89PNG\r\n\x1a\npayload")
                else:
                    path.write_text(f"{arcname}\n", encoding="utf-8")

            zip_path = run.create_required_handoff_zip(out, project)
            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())

            self.assertEqual(set(run.REQUIRED_HANDOFF_FILES), names)
            self.assertEqual(6, len(names))
            self.assertFalse((project / run.REQUIRED_HANDOFF_ZIP_ALIAS).exists())
            self.assertFalse(any(Path(name).name.startswith("m2_") for name in names))
            self.assertFalse(any(Path(name).name in {"README_Handoff.md", "MANIFEST.txt", "08_cluster_minute.png"} for name in names))

    def test_raw_minute_and_session_columns(self):
        rows = [{
            "source_file": "a.xlsx",
            "run_id": 1,
            "broad_no": 2,
            "session_key": "1_2",
            "user_id": "u",
            "category_id": "c",
            "minute_ts": "2026-01-01 00:00:00",
            "viewer_count_last": 10,
            "chat_count": 1,
            "unique_chatters": 1,
        }]
        minute = pd.DataFrame(rows, columns=run.STRICT_MINUTE_CORE_COLS)
        session = pd.DataFrame([{"session_key": "1_2", "cluster_number": 0}])

        self.assertTrue(handoff_validator._raw_minute_ok(minute))
        self.assertIn("cluster_number", session.columns)
        minute_bad = minute.assign(minute_cluster=0)
        self.assertFalse(handoff_validator._raw_minute_ok(minute_bad))


if __name__ == "__main__":
    unittest.main()
