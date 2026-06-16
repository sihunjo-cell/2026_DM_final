import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval_robustness import DEFAULT_FAMILIES, DEFAULT_SIGNALS, NOTE_NOT_REAL, _family_strengths, run_eval_robustness


def _pct_rank(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.rank(method="average", pct=True)


class EvalRobustnessTests(unittest.TestCase):
    def _fixture(self, root):
        root = Path(root)
        out = root / "out"
        out.mkdir(parents=True, exist_ok=True)
        data_syn = root / "data" / "synthetic"
        data_syn.mkdir(parents=True, exist_ok=True)

        review_all = pd.DataFrame([
            {"session_key": "s1", "run_id": 1, "broad_no": 1, "review_order": 1, "eligible_review": True, "scan_family_rank": 1, "persistence_family_rank": 2, "expected_response_family_rank": 1, "minute_state_family_rank": 3, "interval_anomaly_family_rank": 1, "reason_support_family_rank": 2},
            {"session_key": "s2", "run_id": 1, "broad_no": 2, "review_order": 2, "eligible_review": True, "scan_family_rank": 2, "persistence_family_rank": 1, "expected_response_family_rank": 2, "minute_state_family_rank": 1, "interval_anomaly_family_rank": 2, "reason_support_family_rank": 1},
            {"session_key": "s3", "run_id": 1, "broad_no": 3, "review_order": 3, "eligible_review": True, "scan_family_rank": 3, "persistence_family_rank": 3, "expected_response_family_rank": 3, "minute_state_family_rank": 2, "interval_anomaly_family_rank": 3, "reason_support_family_rank": 3},
            {"session_key": "s4", "run_id": 1, "broad_no": 4, "review_order": 4, "eligible_review": False, "scan_family_rank": 4, "persistence_family_rank": 4, "expected_response_family_rank": 4, "minute_state_family_rank": 4, "interval_anomaly_family_rank": 4, "reason_support_family_rank": 4},
            {"session_key": "s5", "run_id": 1, "broad_no": 5, "review_order": 5, "eligible_review": True, "scan_family_rank": 5, "persistence_family_rank": 5, "expected_response_family_rank": 5, "minute_state_family_rank": 5, "interval_anomaly_family_rank": 5, "reason_support_family_rank": 5},
        ])
        review_all.to_csv(out / "m2_review_all.csv", index=False, encoding="utf-8-sig")
        review = review_all.loc[review_all["eligible_review"]].copy()
        review["review_order"] = np.arange(1, len(review) + 1)
        review.to_csv(out / "m2_review.csv", index=False, encoding="utf-8-sig")

        rows = []
        for sess_i, session_key in enumerate(["s1", "s2", "s3", "s5"], start=1):
            for minute in range(1, 7):
                hot = 1 if 2 <= minute <= 4 and sess_i <= 2 else 0
                rows.append({
                    "session_key": session_key,
                    "run_id": 1,
                    "broad_no": sess_i,
                    "minute_ts": f"2026-01-01 00:{minute - 1:02d}:00",
                    "minute_idx": minute,
                    "viewer_count_last": 100 + sess_i * 10,
                    "chat_count": 0 if hot else 10 + minute,
                    "unique_chatters": 0 if hot else 5 + minute,
                    "log_viewer": np.log1p(100 + sess_i * 10),
                    "log_chat": np.log1p(0 if hot else 10 + minute),
                    "log_unique": np.log1p(0 if hot else 5 + minute),
                    "viewer_bin": sess_i,
                    "chat_deficit": 3.0 if hot else 0.1 * minute,
                    "unique_deficit": 2.5 if hot else 0.1 * minute,
                    "rolling_chat_deficit_5m": 2.0 if hot else 0.1 * minute,
                    "zero_run_len": minute if hot else 0,
                    "rolling_zero_rate_5m": 0.8 if hot else 0.0,
                    "minute_cluster": 0 if hot else 1,
                    "cluster_mismatch_rank": 0.95 if hot else 0.2,
                })
        scores = pd.DataFrame(rows)
        rank_cols = []
        for sig in DEFAULT_SIGNALS:
            rcol = f"{sig}_rank_signal"
            scores[rcol] = _pct_rank(scores[sig])
            rank_cols.append(rcol)
        scores["minute_mismatch_score"] = scores[rank_cols].mean(axis=1)
        scores["minute_mismatch_rank"] = _pct_rank(scores["minute_mismatch_score"])
        scores["dominant_reason"] = "fixture"
        scores.to_csv(out / "m2_scores.csv", index=False, encoding="utf-8-sig")

        scan = pd.DataFrame([
            {"session_key": "s1", "run_id": 1, "broad_no": 1, "top_interval_start_idx": 2, "top_interval_end_idx": 4, "top_interval_start_ts": "2026-01-01 00:01:00", "top_interval_end_ts": "2026-01-01 00:03:00", "top_interval_duration": 3, "observed_scan_z": 3.0, "empirical_p": 0.01},
            {"session_key": "s2", "run_id": 1, "broad_no": 2, "top_interval_start_idx": 2, "top_interval_end_idx": 4, "top_interval_start_ts": "2026-01-01 00:01:00", "top_interval_end_ts": "2026-01-01 00:03:00", "top_interval_duration": 3, "observed_scan_z": 2.0, "empirical_p": 0.02},
            {"session_key": "s3", "run_id": 1, "broad_no": 3, "top_interval_start_idx": 1, "top_interval_end_idx": 1, "top_interval_start_ts": "2026-01-01 00:00:00", "top_interval_end_ts": "2026-01-01 00:00:00", "top_interval_duration": 1, "observed_scan_z": 1.0, "empirical_p": 0.5},
            {"session_key": "s5", "run_id": 1, "broad_no": 4, "top_interval_start_idx": 1, "top_interval_end_idx": 1, "top_interval_start_ts": "2026-01-01 00:00:00", "top_interval_end_ts": "2026-01-01 00:00:00", "top_interval_duration": 1, "observed_scan_z": 1.0, "empirical_p": 0.6},
        ])
        scan.to_csv(out / "m2_scan.csv", index=False, encoding="utf-8-sig")

        syn_rows = []
        for minute in range(1, 8):
            injected = 3 <= minute <= 5
            syn_rows.append({
                "source_file": "fixture.csv",
                "run_id": -1,
                "broad_no": "syn1",
                "session_key": "syn_s1",
                "synthetic_session_key": "syn_s1",
                "source_session_key": "s1",
                "injection_type": "silent_run",
                "minute_ts": f"2026-01-02 00:{minute - 1:02d}:00",
                "viewer_count_last": 150,
                "chat_count": 0 if injected else 15,
                "unique_chatters": 0 if injected else 8,
            })
        pd.DataFrame(syn_rows).to_csv(data_syn / "synthetic_injection_example_minutes.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{
            "synthetic_session_key": "syn_s1",
            "source_session_key": "s1",
            "injection_type": "silent_run",
            "planned_start_idx": 3,
            "planned_end_idx": 5,
            "injected_start_ts": "2026-01-02 00:02:00",
            "injected_end_ts": "2026-01-02 00:04:00",
        }]).to_csv(data_syn / "synthetic_injection_example_intervals.csv", index=False, encoding="utf-8-sig")

        cfg = {
            "eval_robustness": {
                "enabled": True,
                "out_dir": str(out / "eval"),
                "topk": [20, 50, 100, 200],
                "strong_family_strength_threshold": 0.80,
                "families": DEFAULT_FAMILIES,
                "minute_score_signals": DEFAULT_SIGNALS,
                "signal_weight_variants": {
                    "equal_weight_main": {sig: 1 for sig in DEFAULT_SIGNALS},
                    "no_kmeans_state": {**{sig: 1 for sig in DEFAULT_SIGNALS}, "cluster_mismatch_rank": 0},
                },
                "synthetic": {
                    "external_minutes_csv": str(data_syn / "synthetic_injection_example_minutes.csv"),
                    "external_intervals_csv": str(data_syn / "synthetic_injection_example_intervals.csv"),
                    "iou_thresholds": [0.3, 0.5, 0.7],
                },
            },
            "m2_scan": {"n_perm": 0, "break_on_clock_gap_min": 1.1, "max_scan_n": 100},
        }
        return out, cfg

    def _hashes(self, out):
        names = ["m2_review.csv", "m2_review_all.csv", "m2_scores.csv", "m2_scan.csv"]
        return {name: hashlib.sha256((Path(out) / name).read_bytes()).hexdigest() for name in names}

    def test_eval_outputs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cfg = self._fixture(tmp)
            run_eval_robustness(out, cfg)
            eval_dir = out / "eval"
            for name in [
                "family_ablation_summary.csv",
                "aggregation_sensitivity_summary.csv",
                "evidence_balance_summary.csv",
                "tie_audit.csv",
                "evaluation_report.md",
            ]:
                self.assertTrue((eval_dir / name).exists(), name)

    def test_main_outputs_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cfg = self._fixture(tmp)
            before = self._hashes(out)
            run_eval_robustness(out, cfg)
            after = self._hashes(out)
            self.assertEqual(before, after)

    def test_review_all_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cfg = self._fixture(tmp)
            review_all = pd.read_csv(out / "m2_review_all.csv", encoding="utf-8-sig")
            strengths = _family_strengths(review_all, DEFAULT_FAMILIES)
            self.assertAlmostEqual(strengths.loc[0, "scan_family_strength"], 1 - 1 / (len(review_all) + 1))
            self.assertNotAlmostEqual(strengths.loc[0, "scan_family_strength"], 1 - 1 / (len(pd.read_csv(out / "m2_review.csv", encoding="utf-8-sig")) + 1))

    def test_synthetic_files_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cfg = self._fixture(tmp)
            run_eval_robustness(out, cfg)
            audit = pd.read_csv(out / "eval" / "synthetic" / "synthetic_key_audit.csv", encoding="utf-8-sig")
            combined = " ".join(audit[["external_minutes_csv", "external_intervals_csv"]].iloc[0].astype(str))
            self.assertIn("data", combined)
            self.assertIn("synthetic", combined)
            self.assertNotIn("data/features", combined.replace("\\", "/"))

    def test_synthetic_not_real_performance_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cfg = self._fixture(tmp)
            run_eval_robustness(out, cfg)
            report = (out / "eval" / "evaluation_report.md").read_text(encoding="utf-8")
            recovery = pd.read_csv(out / "eval" / "synthetic" / "synthetic_interval_recovery.csv", encoding="utf-8-sig")
            self.assertIn("not real viewbot performance", report)
            self.assertTrue(recovery["note"].astype(str).str.contains(NOTE_NOT_REAL, regex=False).all())

    def test_no_probability_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cfg = self._fixture(tmp)
            run_eval_robustness(out, cfg)
            report = (out / "eval" / "evaluation_report.md").read_text(encoding="utf-8")
            for forbidden in ["viewbot probability", "real detection accuracy", "ground-truth performance"]:
                self.assertNotIn(forbidden, report)


if __name__ == "__main__":
    unittest.main()
