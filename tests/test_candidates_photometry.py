"""Fast non-network tests for candidate ranking and direct moving-aperture photometry."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    np = None

from kbo_tess.candidates import select_bounded_candidate_windows, summarize_detector_candidates
from kbo_tess.injection import inject_source_into_cube_bundle
from kbo_tess.photometry import run_moving_aperture_photometry


@unittest.skipUnless(np is not None, "numpy is required for the candidate/photometry tests")
class CandidatePhotometryTest(unittest.TestCase):
    def _make_inputs(self):
        rng = np.random.default_rng(21)
        times = 2000.0 + np.arange(7, dtype=float)
        x = np.asarray([1500.0, 1498.0, 1496.0, 1494.0, 1492.0, 1490.0, 1488.0], dtype=float)
        y = np.asarray([900.0, 905.0, 910.0, 915.0, 920.0, 925.0, 930.0], dtype=float)
        cube = rng.normal(loc=0.0, scale=0.2, size=(len(times), 23, 23))
        corner1 = np.floor(x - 10.0)
        corner2 = np.floor(y - 11.0)
        local_x = x - corner1
        local_y = y - corner2
        ephem_df = {
            "target": ["Pluto"] * len(times),
            "sector": [92] * len(times),
            "camera": [2] * len(times),
            "ccd": [1] * len(times),
            "time_tdb_jd_minus_2457000": times,
            "x": x,
            "y": y,
            "time_reference": ["spacecraft"] * len(times),
        }
        cube_bundle = {
            "cube": cube,
            "time_tdb_jd_minus_2457000": times,
            "local_x": local_x,
            "local_y": local_y,
            "detector_x": x,
            "detector_y": y,
            "corner1": corner1,
            "corner2": corner2,
            "pos_corr1": local_x - 11.0,
            "pos_corr2": local_y - 11.0,
            "quality_mask": np.ones(len(times), dtype=bool),
            "metadata": {
                "target": "Pluto",
                "sector": 92,
                "camera": 2,
                "ccd": 1,
                "loader_bg_method": "rolling",
            },
        }
        return ephem_df, cube_bundle

    def test_candidate_summary_and_window_selection(self):
        ephem_df, _ = self._make_inputs()
        summary = summarize_detector_candidates(ephem_df, target="Pluto")
        selected = select_bounded_candidate_windows(ephem_df, target="Pluto", window_days=2.0, max_candidates=2)

        self.assertEqual(len(summary), 1)
        self.assertEqual(int(summary.iloc[0]["sector"]), 92)
        self.assertGreaterEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["candidate_label"], "primary")

    def test_moving_aperture_photometry_improves_with_injection(self):
        ephem_df, cube_bundle = self._make_inputs()
        baseline = run_moving_aperture_photometry(ephem_df, cube_bundle, background_method="none")
        injected_bundle = inject_source_into_cube_bundle(cube_bundle, amplitude=3.0, sigma=1.0)
        injected = run_moving_aperture_photometry(ephem_df, injected_bundle, background_method="none")

        self.assertGreater(
            injected["summary"]["median_aperture_snr"],
            baseline["summary"]["median_aperture_snr"],
        )
        self.assertGreater(
            injected["summary"]["max_net_flux"],
            baseline["summary"]["max_net_flux"],
        )


if __name__ == "__main__":
    unittest.main()
