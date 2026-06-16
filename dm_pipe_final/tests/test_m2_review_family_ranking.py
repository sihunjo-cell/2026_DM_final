import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run
from src.m2_review import build_m2_reason, build_m2_review


class M2ReviewFamilyRankingTest(unittest.TestCase):
    def _write_fixture(self, out):
        scan = pd.DataFrame([
            {
                "session_key": "s1",
                "run_id": 1,
                "broad_no": 1,
                "top_interval_start_ts": "2026-01-01 00:00:00",
                "top_interval_end_ts": "2026-01-01 00:00:00",
                "top_interval_duration": 1,
                "empirical_p": 0.01,
                "observed_scan_z": 9.0,
                "interval_model_chat_deficit_mean": 2.0,
                "interval_model_unique_deficit_mean": 2.5,
                "interval_mismatch_state_rate": 0.9,
                "interval_chat_deficit_mean": 3.0,
                "interval_unique_deficit_mean": 2.0,
                "interval_max_zero_run": 1,
            },
            {
                "session_key": "s2",
                "run_id": 1,
                "broad_no": 2,
                "top_interval_start_ts": "2026-01-01 00:00:00",
                "top_interval_end_ts": "2026-01-01 00:39:00",
                "top_interval_duration": 40,
                "empirical_p": 0.02,
                "observed_scan_z": 8.0,
                "interval_model_chat_deficit_mean": 2.2,
                "interval_model_unique_deficit_mean": 2.1,
                "interval_mismatch_state_rate": 0.8,
                "interval_chat_deficit_mean": 2.5,
                "interval_unique_deficit_mean": 2.0,
                "interval_max_zero_run": 20,
            },
            {
                "session_key": "s3",
                "run_id": 1,
                "broad_no": 3,
                "top_interval_start_ts": "2026-01-01 00:00:00",
                "top_interval_end_ts": "2026-01-01 00:09:00",
                "top_interval_duration": 10,
                "empirical_p": 0.5,
                "observed_scan_z": 1.0,
                "interval_model_chat_deficit_mean": 0.1,
                "interval_model_unique_deficit_mean": 0.1,
                "interval_mismatch_state_rate": 0.2,
                "interval_chat_deficit_mean": 0.2,
                "interval_unique_deficit_mean": 0.1,
                "interval_max_zero_run": 2,
            },
        ])
        scan.to_csv(out / "m2_scan.csv", index=False, encoding="utf-8-sig")
        state = pd.DataFrame([
            {"session_key": "s1", "mismatch_minute_rate": 0.9, "mismatch_max_run": 1, "mismatch_total_run_min": 1},
            {"session_key": "s2", "mismatch_minute_rate": 0.8, "mismatch_max_run": 30, "mismatch_total_run_min": 40},
            {"session_key": "s3", "mismatch_minute_rate": 0.2, "mismatch_max_run": 5, "mismatch_total_run_min": 8},
        ])
        state.to_csv(out / "m2_state.csv", index=False, encoding="utf-8-sig")
        int_scores = pd.DataFrame([
            {"session_key": "s1", "int_ecod_dir_score": 3.0, "int_if_dir_score": 2.0, "int_lof_dir_score": 1.0},
            {"session_key": "s2", "int_ecod_dir_score": 2.0, "int_if_dir_score": 2.0, "int_lof_dir_score": 2.0},
            {"session_key": "s3", "int_ecod_dir_score": 0.1, "int_if_dir_score": 0.1, "int_lof_dir_score": 0.1},
        ])
        int_scores.to_csv(out / "int_scores.csv", index=False, encoding="utf-8-sig")
        session = pd.DataFrame([
            {"session_key": "s1", "n": 10, "v_qc": 0, "all_zero": False, "ok": True},
            {"session_key": "s2", "n": 10, "v_qc": 0, "all_zero": False, "ok": True},
            {"session_key": "s3", "n": 10, "v_qc": 0, "all_zero": False, "ok": True},
        ])
        session.to_csv(out / "session_summary_processed.csv", index=False, encoding="utf-8-sig")

    def test_family_ranking_columns_order_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_fixture(out)
            cfg = {"prep": {"min_n": 1}, "m2_review": {"include_ineligible_in_final_review": False, "write_all_review_file": True}}
            build_m2_reason(out, cfg)
            review = build_m2_review(out, cfg)

            required = {
                "raw_rra_p",
                "raw_rra_q",
                "scan_family_rank",
                "persistence_family_rank",
                "expected_response_family_rank",
                "minute_state_family_rank",
                "interval_anomaly_family_rank",
                "reason_support_family_rank",
                "family_consensus_score",
                "family_rra_p",
                "family_rra_q",
                "evidence_family_count",
                "ranking_method",
            }
            self.assertTrue(required.issubset(review.columns))
            self.assertTrue(np.allclose(review["rra_q"], review["family_rra_q"]))
            self.assertTrue(np.allclose(review["rra_p"], review["family_rra_p"]))
            self.assertTrue((out / "m2_review_rank_audit.csv").exists())
            self.assertEqual({"equal_weight_family_consensus_plus_family_rra"}, set(review["ranking_method"]))

            result = run._family_ranking_result(out, review)
            self.assertTrue(result["raw_scan_not_duplicated"])
            self.assertTrue(result["order_ok"])
            self.assertGreaterEqual(result["top10_short_count"], 1)
            short_notes = review.loc[pd.to_numeric(review["top_interval_duration"], errors="coerce").le(1), "review_note"].astype(str)
            self.assertTrue(short_notes.str.contains("manual confirmation").any())
            forbidden = {"viewbot_probability", "true_viewbot_label", "bot_detected", "predicted_label"}
            self.assertTrue(forbidden.isdisjoint(review.columns))


if __name__ == "__main__":
    unittest.main()
