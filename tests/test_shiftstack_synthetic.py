"""Fast synthetic tests for the detector-local known-path shift-stack core."""

from __future__ import annotations

import json
import sys
import tempfile
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

from kbo_tess.ephem import load_ephemeris_csv
from kbo_tess.recovery import recover_from_local_files, write_recovery_products
from kbo_tess.shiftstack import search_local_track_offsets, stack_along_track

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@unittest.skipUnless(np is not None, "numpy is required for the synthetic shift-stack test")
class ShiftStackSyntheticTest(unittest.TestCase):
    def _make_cube(self):
        rng = np.random.default_rng(7)
        n_frames = 13
        shape = (28, 28)
        yy, xx = np.indices(shape, dtype=float)
        x = 10.5 + 0.30 * np.arange(n_frames)
        y = 11.0 + 0.25 * np.arange(n_frames)

        cube = rng.normal(loc=0.0, scale=1.0, size=(n_frames, *shape))
        for frame_index in range(n_frames):
            dx = xx - x[frame_index]
            dy = yy - y[frame_index]
            cube[frame_index] += 3.0 * np.exp(-(dx * dx + dy * dy) / (2.0 * 1.0**2))
        return cube, x, y

    def test_stack_along_track_boosts_peak(self):
        cube, x, y = self._make_cube()
        stacked = stack_along_track(cube, x, y, method="sum", interpolation="nearest")
        reference_y = int(round(y[0]))
        reference_x = int(round(x[0]))
        stacked_peak = float(stacked[reference_y, reference_x])
        single_frame_peak = float(cube[0, reference_y, reference_x])
        self.assertGreater(stacked_peak, single_frame_peak)

    def test_local_offset_search_prefers_true_track(self):
        cube, x, y = self._make_cube()
        search = search_local_track_offsets(
            cube,
            x,
            y,
            dx_grid=(-1.0, -0.5, 0.0, 0.5, 1.0),
            dy_grid=(-1.0, -0.5, 0.0, 0.5, 1.0),
            interpolation="nearest",
            stack_method="mean",
            score_method="snr",
        )
        self.assertAlmostEqual(search["best"]["dx"], 0.0, delta=0.5)
        self.assertAlmostEqual(search["best"]["dy"], 0.0, delta=0.5)

    def test_local_file_recovery_pipeline(self):
        ephem_path = FIXTURE_DIR / "sedna_sector5_ephem_fixture.csv"
        ephem_df = load_ephemeris_csv(ephem_path, require_exact_tdb=True)
        cube_times = ephem_df["time_tdb_jd_minus_2457000"].to_numpy(dtype=float)
        x = ephem_df["x"].to_numpy(dtype=float)
        y = ephem_df["y"].to_numpy(dtype=float)

        rng = np.random.default_rng(11)
        shape = (32, 32)
        yy, xx = np.indices(shape, dtype=float)
        cube = rng.normal(loc=100.0, scale=1.0, size=(len(cube_times), *shape))
        cube += np.linspace(-0.25, 0.25, len(cube_times))[:, None, None]
        for frame_index in range(len(cube_times)):
            dx = xx - x[frame_index]
            dy = yy - y[frame_index]
            cube[frame_index] += 4.0 * np.exp(-(dx * dx + dy * dy) / (2.0 * 1.1**2))

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            cube_npz = temp_dir / "cube.npz"
            np.savez(
                cube_npz,
                cube=cube,
                time_tdb_jd_minus_2457000=cube_times,
                quality_mask=np.ones(len(cube_times), dtype=bool),
                metadata_json=json.dumps({"target": "Sedna", "sector": 5, "camera": 1, "ccd": 4}),
            )

            recovery = recover_from_local_files(
                ephem_path,
                cube_npz,
                background_method="median",
                interpolation="nearest",
                offset_grid_radius=1.0,
                offset_grid_step=0.5,
            )
            output_paths = write_recovery_products(recovery, temp_dir)

            self.assertAlmostEqual(recovery["summary"]["best_dx"], 0.0, delta=0.5)
            self.assertAlmostEqual(recovery["summary"]["best_dy"], 0.0, delta=0.5)
            self.assertGreater(
                recovery["summary"]["best_score"],
                recovery["summary"]["representative_single_frame_scores"]["aperture_snr"],
            )
            self.assertTrue(output_paths["summary_json"].exists())
            self.assertTrue(output_paths["offset_csv"].exists())
            self.assertTrue(output_paths["growth_csv"].exists())
            self.assertTrue(output_paths["diagnostic_png"].exists())


if __name__ == "__main__":
    unittest.main()
