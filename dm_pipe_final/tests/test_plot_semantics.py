import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import plots
from src import m2_review


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PlotSemanticsTest(unittest.TestCase):
    def test_session_cluster_source_uses_legend_not_cluster_colorbar(self):
        source = inspect.getsource(plots._plot_session_cluster)
        self.assertIn("legend(title=\"cluster_number\"", source)
        self.assertNotIn("label=\"cluster_number\"", source)
        self.assertNotIn("fig.colorbar(sc", source)

    def test_plot_audit_records_expected_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plot_dir = out / "plots"
            plot_dir.mkdir(parents=True)
            for name in [
                "04_cluster_session.png",
                "05_session_k_selection.png",
                "06_session_cluster_profile.png",
                "08_mc.png",
                "16_state.png",
                "19_reason.png",
                "20_rra.png",
                "21_interval.png",
            ]:
                (plot_dir / name).write_bytes(PNG_SIGNATURE + b"stub")
            pd.DataFrame([
                {"session_key": "a", "cluster_number": 0},
                {"session_key": "b", "cluster_number": 1},
                {"session_key": "c", "cluster_number": 1},
            ]).to_csv(out / "session_summary_processed.csv", index=False, encoding="utf-8-sig")

            plots.write_plot_doc(out)
            audit = pd.read_csv(out / "plot_audit.csv", encoding="utf-8-sig").set_index("plot_file")

            self.assertEqual("legend", audit.loc["04_cluster_session.png", "colorbar_or_legend"])
            self.assertEqual(2, int(audit.loc["04_cluster_session.png", "actual_legend_count"]))
            self.assertEqual("selection_score", audit.loc["05_session_k_selection.png", "primary_visual_encoding"])
            self.assertIn("not cluster id", audit.loc["06_session_cluster_profile.png", "colorbar_label"])
            self.assertIn("not cluster id", audit.loc["08_mc.png", "colorbar_label"])
            self.assertIn("text_reason_included=True", audit.loc["19_reason.png", "interpretation"])
            self.assertEqual("one shared colorbar", audit.loc["21_interval.png", "colorbar_or_legend"])

    def test_reason_and_interval_plot_sources_use_requested_inputs(self):
        reason_source = inspect.getsource(m2_review._plot_reason)
        interval_source = inspect.getsource(m2_review._plot_interval)
        self.assertIn("text_reason_included", reason_source)
        self.assertIn("top-3 explanation", reason_source)
        self.assertIn("Normalize", interval_source)
        self.assertIn("fig.colorbar(sc1, ax=ax", interval_source)


if __name__ == "__main__":
    unittest.main()
