import tempfile
import unittest
from pathlib import Path

import pandas as pd

import run


class StateTransitionCountTests(unittest.TestCase):
    def test_actual_transition_count_ignores_first_row_and_same_state_pairs(self):
        minutes = pd.DataFrame([
            {"session_key": "same", "minute_ts": "2026-01-01 00:00:00", "minute_cluster": 1},
            {"session_key": "same", "minute_ts": "2026-01-01 00:01:00", "minute_cluster": 1},
            {"session_key": "same", "minute_ts": "2026-01-01 00:02:00", "minute_cluster": 1},
            {"session_key": "zigzag", "minute_ts": "2026-01-01 00:00:00", "minute_cluster": 0},
            {"session_key": "zigzag", "minute_ts": "2026-01-01 00:01:00", "minute_cluster": 1},
            {"session_key": "zigzag", "minute_ts": "2026-01-01 00:02:00", "minute_cluster": 1},
            {"session_key": "zigzag", "minute_ts": "2026-01-01 00:03:00", "minute_cluster": 0},
        ])

        actual = run._actual_transition_counts_from_minutes(minutes).set_index("session_key")["actual_transition_count"]

        self.assertEqual(int(actual.loc["same"]), 0)
        self.assertEqual(int(actual.loc["zigzag"]), 2)

    def test_state_transition_validation_detects_stale_adjacent_pair_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            minutes = pd.DataFrame([
                {"session_key": "same", "minute_ts": "2026-01-01 00:00:00", "minute_cluster": 1},
                {"session_key": "same", "minute_ts": "2026-01-01 00:01:00", "minute_cluster": 1},
                {"session_key": "same", "minute_ts": "2026-01-01 00:02:00", "minute_cluster": 1},
            ])
            state = pd.DataFrame([{
                "session_key": "same",
                "n_minutes": 3,
                "transition_count": 2,
            }])

            result = run._state_transition_consistency_result(out, m2_state=state, m2_scores=minutes)

            self.assertFalse(result["consistency_ok"])
            self.assertEqual(result["mismatched_sessions"], 1)
            self.assertTrue(result["all_adjacent_pair_count"])


if __name__ == "__main__":
    unittest.main()
