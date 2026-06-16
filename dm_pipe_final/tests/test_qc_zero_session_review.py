import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.prep import write_qc_zero_session_review


class QcZeroSessionReviewTests(unittest.TestCase):
    def test_qc_zero_session_review_summarizes_sessions_and_stays_excluded(self):
        qc_zero = pd.DataFrame([
            {
                "run_id": 1,
                "broad_no": "a",
                "session_key": "1_a",
                "user_id": "u1",
                "category_id": "c1",
                "minute_ts": "2026-01-01 00:00:00",
                "viewer_count_last": 10,
                "chat_count": 0,
                "unique_chatters": 0,
                "v_qc": 0,
                "v_edge": 1,
                "v_miss_r": 0.1,
            },
            {
                "run_id": 1,
                "broad_no": "a",
                "session_key": "1_a",
                "user_id": "u1",
                "category_id": "c1",
                "minute_ts": "2026-01-01 00:01:00",
                "viewer_count_last": 20,
                "chat_count": 0,
                "unique_chatters": 0,
                "v_qc": 0,
                "v_edge": 0,
                "v_miss_r": 0.0,
            },
            {
                "run_id": 2,
                "broad_no": "b",
                "session_key": "2_b",
                "user_id": "u2",
                "category_id": "c2",
                "minute_ts": "2026-01-01 00:00:00",
                "viewer_count_last": 5,
                "chat_count": 0,
                "unique_chatters": 0,
                "v_qc": 1,
                "v_edge": 1,
                "v_miss_r": 1.0,
            },
        ])
        minute_model = pd.DataFrame({"session_key": ["3_c"]})
        session_summary = pd.DataFrame({"session_key": ["4_d"]})

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            review = write_qc_zero_session_review(qc_zero, out)
            from_disk = pd.read_csv(out / "qc_zero_session_review.csv", encoding="utf-8-sig")

            self.assertTrue((out / "qc_zero_session_review.csv").exists())
            self.assertEqual(set(review["session_key"]), set(qc_zero["session_key"].unique()))
            self.assertEqual(set(from_disk["session_key"]), set(qc_zero["session_key"].unique()))
            self.assertTrue(pd.to_numeric(review["total_chat_count"], errors="coerce").eq(0).all())
            self.assertTrue(review["all_zero_chat"].astype(bool).all())
            self.assertTrue(set(review["session_key"]).isdisjoint(set(minute_model["session_key"])))
            self.assertTrue(set(review["session_key"]).isdisjoint(set(session_summary["session_key"])))

            reason_text = " ".join(review["qc_reason"].astype(str).str.lower())
            for forbidden in ["confirmed_viewbot", "websocket_failed", "bot_detected", "true_label", "viewbot_probability"]:
                self.assertNotIn(forbidden, reason_text)


if __name__ == "__main__":
    unittest.main()
