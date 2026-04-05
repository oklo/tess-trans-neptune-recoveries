"""Fast non-network tests for the cube-bundle materialization seam."""

from __future__ import annotations

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
from kbo_tess.recovery import align_ephemeris_to_cube_bundle, load_cube_npz
from kbo_tess.tess_loader import materialize_cube_bundle_to_npz

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@unittest.skipUnless(np is not None, "numpy is required for the bundle-materialization test")
class TessLoaderBundleTest(unittest.TestCase):
    def test_npz_materialization_preserves_local_geometry(self):
        ephem_df = load_ephemeris_csv(FIXTURE_DIR / "sedna_sector5_ephem_fixture.csv", require_exact_tdb=True)
        times = ephem_df["time_tdb_jd_minus_2457000"].to_numpy(dtype=float)
        detector_x = ephem_df["x"].to_numpy(dtype=float)
        detector_y = ephem_df["y"].to_numpy(dtype=float)
        n_frames = len(times)
        shape = (17, 19)

        corner1 = np.floor(detector_x - 4.0)
        corner2 = np.floor(detector_y - 5.0)
        local_x = detector_x - corner1
        local_y = detector_y - corner2
        pos_corr1 = local_x - 0.5 * (shape[1] - 1)
        pos_corr2 = local_y - 0.5 * (shape[0] - 1)

        bundle = {
            "cube": np.zeros((n_frames, *shape), dtype=float),
            "time_tdb_jd_minus_2457000": times,
            "local_x": local_x,
            "local_y": local_y,
            "detector_x": detector_x,
            "detector_y": detector_y,
            "corner1": corner1,
            "corner2": corner2,
            "pos_corr1": pos_corr1,
            "pos_corr2": pos_corr2,
            "quality_mask": np.ones(n_frames, dtype=bool),
            "metadata": {
                "target": "Sedna",
                "sector": 5,
                "camera": 1,
                "ccd": 4,
                "cutout_shape": list(shape),
                "barycentric_mode": False,
                "loader_bg_method": "linear_model",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir_name:
            bundle_path = Path(temp_dir_name) / "bundle.npz"
            materialize_cube_bundle_to_npz(bundle, bundle_path)
            loaded = load_cube_npz(bundle_path)
            aligned = align_ephemeris_to_cube_bundle(ephem_df, loaded)

            self.assertIn("metadata", loaded)
            self.assertAlmostEqual(float(np.nanmax(np.abs(aligned["local_corner_residual_r"]))), 0.0, delta=1e-9)
            self.assertAlmostEqual(float(np.nanmax(np.abs(aligned["local_poscorr_residual_r"]))), 0.0, delta=1e-9)
            self.assertAlmostEqual(float(np.nanmax(np.abs(aligned["detector_residual_r"]))), 0.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
