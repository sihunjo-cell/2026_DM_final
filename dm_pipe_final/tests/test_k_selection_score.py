import tempfile
import unittest
from pathlib import Path

import pandas as pd

import run
from src.k_selection import good_rank


def _write_csv(path, rows, columns=None):
    df = pd.DataFrame(rows, columns=columns)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def _session_rows(labels):
    rows = []
    for i, label in enumerate(labels, start=1):
        row = {col: 0 for col in run.HANDOFF_SESSION_COLS}
        row.update({
            "run_id": i,
            "broad_no": "b",
            "session_key": f"{i}_b",
            "user_id": f"u{i}",
            "category_id": "cat",
            "n": 2,
            "start": "2026-01-01 00:00:00",
            "end": "2026-01-01 00:01:00",
            "viewer_med": 10 + i,
            "viewer_max": 12 + i,
            "chat_mean": 1,
            "unique_mean": 1,
            "cluster_number": label,
        })
        rows.append(row)
    return rows


def _minute_rows(labels):
    rows = []
    for i, label in enumerate(labels, start=1):
        row = {
            "source_file": "fixture.csv",
            "run_id": i,
            "broad_no": "b",
            "session_key": f"{i}_b",
            "user_id": f"u{i}",
            "category_id": "cat",
            "minute_ts": f"2026-01-01 00:0{i}:00",
            "viewer_count_last": 10 + i,
            "chat_count": 1,
            "unique_chatters": 1,
            "minute_cluster": label,
        }
        rows.append(row)
    return rows


def _minimal_cfg():
    return {
        "cluster": {"k_min": 2, "k_max": 6},
        "minute_cluster": {
            "features": ["log_viewer", "chat_deficit"],
            "scaler": "RobustScaler",
            "kmeans": {"k_min": 2, "k_max": 8},
        },
        "prep": {"min_n": 1, "clock_gap_reset_min": 1.1},
    }


def _write_k_fixture(out, session_selected, session_labels, minute_selected=2, minute_labels=None):
    out = Path(out)
    minute_labels = minute_labels if minute_labels is not None else [0, 1]
    _write_csv(out / "cluster_select.csv", [
        {"k": 2, "silhouette": 0.5, "calinski_harabasz": 10, "davies_bouldin": 0.5, "selection_score": 0.9, "selected": session_selected == 2},
        {"k": 5, "silhouette": 0.4, "calinski_harabasz": 8, "davies_bouldin": 0.8, "selection_score": 0.6, "selected": session_selected == 5},
    ])
    _write_csv(out / "session_summary_processed.csv", _session_rows(session_labels), run.HANDOFF_SESSION_COLS)
    _write_csv(out / "handoff" / "csv" / "session_summary_processed.csv", _session_rows(session_labels), run.HANDOFF_SESSION_COLS)
    _write_csv(out / "mc_select.csv", [
        {"algorithm": "kmeans", "param": "k=2", "silhouette": 0.3, "calinski_harabasz": 10, "davies_bouldin": 0.7, "selection_score": 0.8, "selected_k_num": 2, "selected": minute_selected == 2},
        {"algorithm": "kmeans", "param": "k=3", "silhouette": 0.2, "calinski_harabasz": 8, "davies_bouldin": 0.9, "selection_score": 0.5, "selected_k_num": 3, "selected": minute_selected == 3},
    ])
    _write_csv(out / "mc_assign.csv", _minute_rows(minute_labels))
    _write_csv(out / "m2_scores.csv", _minute_rows(minute_labels))
    _write_csv(out / "minute_cluster_model_selection.csv", [
        {"algorithm": "KMeans", "parameter_setting": f"k={minute_selected}", "selected_as_final_state": True},
        {"algorithm": "GMM", "parameter_setting": "n_components=2", "selected_as_final_state": False},
        {"algorithm": "HDBSCAN", "parameter_setting": "min_cluster_size=5", "selected_as_final_state": False},
    ])
    (out / "session_cluster.txt").write_text(f"최종 선택 K: {session_selected}\n", encoding="utf-8")
    (out / "minute_cluster.txt").write_text(f"최종 선택 하이퍼파라미터: k={minute_selected}\n", encoding="utf-8")
    minute_core = pd.DataFrame(_minute_rows([0, 1]))[run.STRICT_MINUTE_CORE_COLS]
    _write_csv(out / "handoff" / "csv" / "minute_all.csv", minute_core.to_dict("records"), run.STRICT_MINUTE_CORE_COLS)
    _write_csv(out / "handoff" / "csv" / "minute_model.csv", minute_core.to_dict("records"), run.STRICT_MINUTE_CORE_COLS)


class KSelectionScoreTests(unittest.TestCase):
    def test_good_rank_direction(self):
        vals = pd.Series([1, 2, 3])

        higher = good_rank(vals, higher_is_better=True)
        lower = good_rank(vals, higher_is_better=False)

        self.assertGreater(higher.iloc[2], higher.iloc[0])
        self.assertGreater(lower.iloc[0], lower.iloc[2])

    def test_session_and_minute_consistency_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_k_fixture(out, session_selected=2, session_labels=[0, 1], minute_selected=2, minute_labels=[0, 1])

            result = run._kmeans_consistency_result(out)

            self.assertEqual(result["session_selected_count"], 1)
            self.assertTrue(result["session_consistency_ok"])
            self.assertEqual(result["minute_selected_count"], 1)
            self.assertTrue(result["minute_consistency_ok"])

    def test_validation_report_prints_selected_k_from_selected_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            _write_k_fixture(out, session_selected=2, session_labels=[0, 1], minute_selected=2, minute_labels=[0, 1])

            run.write_validation_report(out, Path(tmp), _minimal_cfg(), expect_zips=False)
            report = (out / "validation_report.txt").read_text(encoding="utf-8")

            self.assertIn("selected session K from cluster_select.csv: 2", report)
            self.assertIn("actual session cluster_number unique count: 2", report)
            self.assertIn("session K consistency: PASS", report)
            self.assertIn("selected minute K from mc_select.csv: 2", report)
            self.assertIn("actual minute_cluster unique count: 2", report)
            self.assertIn("minute K consistency: PASS", report)

    def test_stale_selected_session_k_fails_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_k_fixture(out, session_selected=5, session_labels=[0, 1], minute_selected=2, minute_labels=[0, 1])

            result = run._kmeans_consistency_result(out)

            self.assertFalse(result["session_consistency_ok"])
            self.assertEqual(result["session_selected_k"], 5)
            self.assertEqual(result["session_actual_count"], 2)

    def test_missing_selected_row_fails_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_k_fixture(out, session_selected=99, session_labels=[0, 1], minute_selected=2, minute_labels=[0, 1])

            result = run._kmeans_consistency_result(out)

            self.assertFalse(result["session_selected_count_ok"])
            self.assertEqual(run._selected_session_k(pd.read_csv(out / "cluster_select.csv", encoding="utf-8-sig")), "invalid_selected_count")

    def test_multiple_selected_rows_fail_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_k_fixture(out, session_selected=2, session_labels=[0, 1], minute_selected=2, minute_labels=[0, 1])
            select = pd.read_csv(out / "cluster_select.csv", encoding="utf-8-sig")
            select["selected"] = True
            select.to_csv(out / "cluster_select.csv", index=False, encoding="utf-8-sig")

            result = run._kmeans_consistency_result(out)

            self.assertFalse(result["session_selected_count_ok"])
            self.assertEqual(run._selected_session_k(select), "invalid_selected_count")


if __name__ == "__main__":
    unittest.main()
