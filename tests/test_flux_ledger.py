"""Fast non-network tests for flux-ledger and 2-D offset diagnostics."""

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

from kbo_tess.flux_ledger import compute_box_aperture_time_series, compute_pixel_residency_map
from kbo_tess.track_calibration import run_offset_grid_metrics


@unittest.skipUnless(np is not None, "numpy is required for flux-ledger tests")
class FluxLedgerTest(unittest.TestCase):
    def test_pixel_residency_counts_stationary_track(self) -> None:
        x = np.full(5, 4.0, dtype=float)
        y = np.full(5, 3.0, dtype=float)
        residency = compute_pixel_residency_map((9, 9), x, y, radius=0.25)
        self.assertEqual(float(residency[3, 4]), 5.0)
        self.assertEqual(float(residency.sum()), 5.0)

    def test_box_aperture_tracks_true_source_better_than_offset(self) -> None:
        n_frames = 12
        shape = (15, 15)
        yy, xx = np.indices(shape, dtype=float)
        time = np.arange(n_frames, dtype=float)
        x = 6.2 + 0.08 * time
        y = 7.1 - 0.05 * time
        cube = np.zeros((n_frames, *shape), dtype=float)
        for frame_index in range(n_frames):
            dx = xx - x[frame_index]
            dy = yy - y[frame_index]
            cube[frame_index] += 20.0 * np.exp(-(dx * dx + dy * dy) / (2.0 * 0.6**2))

        truth = compute_box_aperture_time_series(cube, time, x, y, half_size=1)
        offset = compute_box_aperture_time_series(cube, time, x + 1.0, y, half_size=1)
        self.assertGreater(float(truth["net_flux"].mean()), float(offset["net_flux"].mean()))

    def test_offset_grid_metrics_find_known_offset(self) -> None:
        n_frames = 18
        shape = (17, 17)
        yy, xx = np.indices(shape, dtype=float)
        time = np.arange(n_frames, dtype=float)
        x = 7.0 + 0.04 * time
        y = 8.0 - 0.03 * time
        true_dx = 0.5
        true_dy = -0.25
        cube = np.zeros((n_frames, *shape), dtype=float)
        for frame_index in range(n_frames):
            dx = xx - (x[frame_index] + true_dx)
            dy = yy - (y[frame_index] + true_dy)
            cube[frame_index] += 12.0 * np.exp(-(dx * dx + dy * dy) / (2.0 * 0.7**2))

        result = run_offset_grid_metrics(
            cube,
            x,
            y,
            dx_grid=np.asarray([0.0, 0.25, 0.5, 0.75], dtype=float),
            dy_grid=np.asarray([-0.5, -0.25, 0.0], dtype=float),
            reference=(float(x[0]), float(y[0])),
            interpolation="linear",
        )
        self.assertLessEqual(abs(float(result["best_direct_row"]["dx"]) - true_dx), 0.25)
        self.assertLessEqual(abs(float(result["best_direct_row"]["dy"]) - true_dy), 0.25)
        self.assertAlmostEqual(float(result["best_stack_row"]["dx"]), true_dx, places=6)
        self.assertAlmostEqual(float(result["best_stack_row"]["dy"]), true_dy, places=6)


if __name__ == "__main__":
    unittest.main()
