import tempfile
import unittest
from pathlib import Path

import run


class ParameterRationaleTest(unittest.TestCase):
    def test_parameter_rationale_documents_key_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg, _, _ = run.load_runtime_config("cfg.yml")
            run.write_parameter_rationale(out, cfg)
            text = (out / "parameter_rationale.md").read_text(encoding="utf-8")

            required = [
                "prep.min_n",
                "prep.clock_gap_reset_min",
                "minute_state.viewer_bin_n",
                "minute_state.rolling_windows",
                "minute_cluster.features",
                "RobustScaler",
                "KMeans K candidates",
                "m2_scan.n_perm",
                "m2_scan.max_scan_n",
                "GroupKFold",
                "synthetic_sanity.enabled",
            ]
            for term in required:
                self.assertIn(term, text)

            lower = text.lower()
            for phrase in ["viewbot probability", "true label", "confirmed bot"]:
                self.assertNotIn(phrase, lower)

            result = run._parameter_rationale_result(out)
            self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
