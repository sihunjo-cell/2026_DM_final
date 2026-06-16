import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.m2 import build_m2_synth


def _cfg(enabled=True):
    return {
        "episode_grid": {"enabled": enabled},
        "synthetic_sanity": {
            "enabled": enabled,
            "match_iou_threshold": 0.5,
            "topk": [10, 50, 100],
        },
    }


class SyntheticSanityTests(unittest.TestCase):
    def test_disabled_writes_not_run_with_blank_recovered_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = build_m2_synth(out, _cfg(enabled=False))

            self.assertTrue((out / "m2_synth.csv").exists())
            self.assertEqual(summary.loc[0, "status"], "not_run")
            self.assertTrue(pd.isna(summary.loc[0, "recovered_rate"]))
            self.assertTrue(pd.isna(summary.loc[0, "recovered_interval_count"]))

    def test_stale_synthetic_intervals_without_current_scan_is_not_run_stale_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pd.DataFrame([{
                "injected_interval_id": "silent_1",
                "injected_type": "silent_run",
                "session_key": "syn_session",
                "injected_start_ts": "2026-01-01 00:00:00",
                "injected_end_ts": "2026-01-01 00:09:00",
            }]).to_csv(out / "synthetic_intervals.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([{"session_key": "real_session"}]).to_csv(out / "m2_scores.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([{"session_key": "real_session"}]).to_csv(out / "m2_scan.csv", index=False, encoding="utf-8-sig")

            summary = build_m2_synth(out, _cfg(enabled=True))

            self.assertEqual(summary.loc[0, "status"], "not_run_stale_input")
            self.assertTrue(pd.isna(summary.loc[0, "recovered_rate"]))
            self.assertFalse((out / "m2_synth_matches.csv").exists())

    def test_ok_status_populates_numeric_recovered_rate_and_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pd.DataFrame([{
                "injected_interval_id": "silent_1",
                "injected_type": "silent_run",
                "session_key": "syn_session",
                "injected_start_ts": "2026-01-01 00:00:00",
                "injected_end_ts": "2026-01-01 00:09:00",
            }]).to_csv(out / "synthetic_intervals.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([{"session_key": "syn_session"}]).to_csv(out / "m2_scores.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([{
                "session_key": "syn_session",
                "top_interval_start_ts": "2026-01-01 00:02:00",
                "top_interval_end_ts": "2026-01-01 00:08:00",
            }]).to_csv(out / "m2_scan.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([{"session_key": "syn_session", "review_order": 3}]).to_csv(out / "m2_review.csv", index=False, encoding="utf-8-sig")

            summary = build_m2_synth(out, _cfg(enabled=True))
            matches = pd.read_csv(out / "m2_synth_matches.csv", encoding="utf-8-sig")

            self.assertEqual(summary.loc[0, "status"], "ok")
            self.assertEqual(float(summary.loc[0, "recovered_rate"]), 1.0)
            self.assertEqual(int(summary.loc[0, "recovered_interval_count"]), 1)
            self.assertEqual(bool(matches.loc[0, "recovered"]), True)


if __name__ == "__main__":
    unittest.main()
