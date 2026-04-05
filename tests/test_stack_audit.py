"""Targeted tests for stack-alignment audit helpers."""

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

from kbo_tess.stack_audit import measure_local_centroids
from kbo_tess.shiftstack import stack_with_support


@unittest.skipUnless(np is not None, "numpy is required for stack-audit tests")
class StackAuditTest(unittest.TestCase):
    def test_stack_with_support_preserves_shift_direction_and_axis_order(self):
        cube = np.zeros((3, 7, 7), dtype=float)
        x = np.asarray([1.0, 2.0, 3.0], dtype=float)
        y = np.asarray([5.0, 4.0, 3.0], dtype=float)
        for index in range(len(x)):
            cube[index, int(round(y[index])), int(round(x[index]))] = 10.0

        reference = (4.0, 2.0)
        result = stack_with_support(
            cube,
            x,
            y,
            method="sum",
            interpolation="nearest",
            reference=reference,
            integer_shifts=True,
        )

        stack = result["stack"]
        peak_y, peak_x = np.unravel_index(np.nanargmax(stack), stack.shape)
        self.assertEqual((float(peak_x), float(peak_y)), reference)
        self.assertEqual(float(result["support_map"][int(reference[1]), int(reference[0])]), 3.0)

        first_shift = result["shift_rows"][0]
        self.assertEqual(first_shift["requested_shift_x"], 3.0)
        self.assertEqual(first_shift["requested_shift_y"], -3.0)
        self.assertEqual(first_shift["applied_axis0_shift"], -3.0)
        self.assertEqual(first_shift["applied_axis1_shift"], 3.0)

    def test_measure_local_centroids_recovers_injected_track(self):
        n_frames = 9
        shape = (21, 21)
        yy, xx = np.indices(shape, dtype=float)
        x = 9.8 + 0.15 * np.arange(n_frames)
        y = 10.3 - 0.10 * np.arange(n_frames)
        cube = np.zeros((n_frames, *shape), dtype=float)
        for frame_index in range(n_frames):
            dx = xx - x[frame_index]
            dy = yy - y[frame_index]
            cube[frame_index] += 20.0 * np.exp(-(dx * dx + dy * dy) / (2.0 * 0.8**2))

        centroids = measure_local_centroids(cube, x, y)
        self.assertTrue(bool(centroids["valid_centroid"].all()))
        self.assertLess(abs(float(np.nanmedian(centroids["residual_x"]))), 0.15)
        self.assertLess(abs(float(np.nanmedian(centroids["residual_y"]))), 0.15)


if __name__ == "__main__":
    unittest.main()
