"""Fast non-network tests for injection and null-track control helpers."""

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

from kbo_tess.control_studies import evaluate_null_track_controls
from kbo_tess.injection import inject_source_into_cube_bundle
from kbo_tess.recovery import run_known_path_recovery


@unittest.skipUnless(np is not None, "numpy is required for the injection/control tests")
class InjectionControlsTest(unittest.TestCase):
    def _make_synthetic_inputs(self):
        rng = np.random.default_rng(13)
        n_frames = 11
        shape = (24, 24)
        times = 1400.0 + 0.02 * np.arange(n_frames)
        x = 10.2 + 0.25 * np.arange(n_frames)
        y = 11.0 + 0.15 * np.arange(n_frames)
        corner1 = np.floor(x - 5.0)
        corner2 = np.floor(y - 6.0)
        local_x = x - corner1
        local_y = y - corner2
        cube = rng.normal(loc=0.0, scale=0.3, size=(n_frames, *shape))
        ephem_df = {
            "target": ["Sedna"] * n_frames,
            "sector": [5] * n_frames,
            "camera": [1] * n_frames,
            "ccd": [4] * n_frames,
            "time_tdb_jd_minus_2457000": times,
            "x": x,
            "y": y,
        }
        bundle = {
            "cube": cube,
            "time_tdb_jd_minus_2457000": times,
            "local_x": local_x,
            "local_y": local_y,
            "detector_x": x,
            "detector_y": y,
            "corner1": corner1,
            "corner2": corner2,
            "pos_corr1": local_x - 0.5 * (shape[1] - 1),
            "pos_corr2": local_y - 0.5 * (shape[0] - 1),
            "quality_mask": np.ones(n_frames, dtype=bool),
            "metadata": {
                "target": "Sedna",
                "sector": 5,
                "camera": 1,
                "ccd": 4,
                "cutout_shape": list(shape),
                "barycentric_mode": False,
                "loader_bg_method": "rolling",
            },
        }
        return ephem_df, bundle

    def test_injection_preserves_metadata_and_boosts_nominal_score(self):
        ephem_df, bundle = self._make_synthetic_inputs()
        baseline = run_known_path_recovery(
            ephem_df,
            bundle,
            background_method="none",
            interpolation="nearest",
            offset_grid_radius=1.0,
            offset_grid_step=0.5,
        )
        injected_bundle = inject_source_into_cube_bundle(bundle, amplitude=2.0, sigma=1.0)
        injected = run_known_path_recovery(
            ephem_df,
            injected_bundle,
            background_method="none",
            interpolation="nearest",
            offset_grid_radius=1.0,
            offset_grid_step=0.5,
        )

        self.assertIn("injections", injected_bundle["metadata"])
        self.assertGreater(
            injected["summary"]["nominal_score"],
            baseline["summary"]["nominal_score"],
        )

    def test_null_track_controls_prefer_the_nominal_or_near_nominal_track(self):
        ephem_df, bundle = self._make_synthetic_inputs()
        injected_bundle = inject_source_into_cube_bundle(bundle, amplitude=2.5, sigma=1.0)
        recovery = run_known_path_recovery(
            ephem_df,
            injected_bundle,
            background_method="none",
            interpolation="nearest",
            offset_grid_radius=1.0,
            offset_grid_step=0.5,
        )
        null_controls = evaluate_null_track_controls(recovery, control_radius=1.0, near_radius=0.5)

        self.assertTrue(null_controls["summary"]["best_offset_is_near_nominal"])
        self.assertGreater(null_controls["summary"]["best_minus_best_control"], 0.0)


if __name__ == "__main__":
    unittest.main()
