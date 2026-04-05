"""Fast non-network tests for narrow x-track calibration helpers."""

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

from kbo_tess.track_calibration import (
    build_nominal_vs_calibrated_growth_curve,
    run_centered_null_control_gate,
    run_x_offset_sweep,
)


@unittest.skipUnless(np is not None, "numpy is required for track-calibration tests")
class TrackCalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        n_frames = 24
        shape = (21, 21)
        yy, xx = np.indices(shape, dtype=float)
        self.reference = (10.0, 10.0)
        self.x = 9.5 + 0.05 * np.arange(n_frames)
        self.y = 10.2 - 0.03 * np.arange(n_frames)
        self.true_dx = -0.30

        cube = np.zeros((n_frames, *shape), dtype=float)
        for frame_index in range(n_frames):
            dx = xx - (self.x[frame_index] + self.true_dx)
            dy = yy - self.y[frame_index]
            cube[frame_index] += 10.0 * np.exp(-(dx * dx + dy * dy) / (2.0 * 0.8**2))
        self.cube = cube

    def test_x_offset_sweep_finds_known_offset(self) -> None:
        x_offsets = np.asarray([0.0, -0.1, -0.2, -0.3, -0.4, -0.5], dtype=float)
        result = run_x_offset_sweep(
            self.cube,
            self.x,
            self.y,
            x_offsets=x_offsets,
            reference=self.reference,
            interpolation="linear",
        )

        self.assertAlmostEqual(float(result["best_stack_row"]["x_offset"]), self.true_dx, places=6)
        table = result["table"].set_index("x_offset")
        self.assertGreater(
            float(table.loc[self.true_dx, "mean_aperture_snr"]),
            float(table.loc[0.0, "mean_aperture_snr"]),
        )

    def test_calibrated_track_beats_outer_controls_on_toy_case(self) -> None:
        gate = run_centered_null_control_gate(
            self.cube,
            self.x,
            self.y,
            calibrated_x_offset=self.true_dx,
            reference=self.reference,
            dx_grid=np.asarray([-0.5, 0.0, 0.5], dtype=float),
            dy_grid=np.asarray([-0.5, 0.0, 0.5], dtype=float),
            interpolation="linear",
        )
        self.assertGreater(float(gate["summary"]["calibrated_track_score"]), float(gate["summary"]["exact_nominal_score"]))
        self.assertGreater(float(gate["summary"]["calibrated_minus_best_control"]), 0.0)

    def test_calibrated_growth_curve_finishes_above_nominal(self) -> None:
        growth = build_nominal_vs_calibrated_growth_curve(
            self.cube,
            self.x,
            self.y,
            calibrated_x_offset=self.true_dx,
            reference=self.reference,
            interpolation="linear",
        )
        self.assertGreater(float(growth.iloc[-1]["calibrated_score"]), float(growth.iloc[-1]["nominal_score"]))


if __name__ == "__main__":
    unittest.main()
