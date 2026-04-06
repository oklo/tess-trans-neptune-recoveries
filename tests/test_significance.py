"""Fast non-network tests for empirical-significance helpers."""

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

from kbo_tess.significance import (
    empirical_p_value,
    generate_translated_pseudo_tracks,
    scrub_track_from_cube_bundle,
)


@unittest.skipUnless(np is not None, "numpy is required for significance tests")
class SignificanceHelpersTest(unittest.TestCase):
    def test_generate_translated_pseudo_tracks_preserves_shape_and_margin(self) -> None:
        x = 9.5 + 0.1 * np.arange(12)
        y = 10.3 - 0.05 * np.arange(12)
        tracks = generate_translated_pseudo_tracks(
            x,
            y,
            calibrated_x_offset=-0.45,
            shape=(21, 21),
            n_realizations=32,
            edge_margin=5.0,
            exclusion_radius=3.0,
            rng_seed=7,
        )

        self.assertEqual(len(tracks), 32)
        for row in tracks.itertuples(index=False):
            translated_x = x + float(row.translation_x)
            translated_y = y + float(row.translation_y)
            calibrated_x = translated_x - 0.45
            self.assertGreaterEqual(float(translated_x.min()), 5.0)
            self.assertLessEqual(float(translated_x.max()), 15.0)
            self.assertGreaterEqual(float(calibrated_x.min()), 5.0)
            self.assertLessEqual(float(calibrated_x.max()), 15.0)
            self.assertGreaterEqual(float(translated_y.min()), 5.0)
            self.assertLessEqual(float(translated_y.max()), 15.0)
            self.assertGreaterEqual(float(np.hypot(row.translation_x, row.translation_y)), 3.0)

    def test_empirical_p_value_is_conservative(self) -> None:
        p_value = empirical_p_value(3.0, [0.0, 1.0, 2.0, 4.0])
        self.assertAlmostEqual(p_value, 2.0 / 5.0)

    def test_scrub_track_from_cube_bundle_preserves_metadata_and_removes_source(self) -> None:
        n_frames = 6
        shape = (11, 11)
        cube = np.zeros((n_frames, *shape), dtype=float)
        x = np.full(n_frames, 5.0)
        y = np.full(n_frames, 5.0)
        for frame_index in range(n_frames):
            cube[frame_index, 5, 5] = 10.0 + frame_index
            cube[frame_index, 2:4, 2:4] = 1.5

        bundle = {
            "cube": cube,
            "time_tdb_jd_minus_2457000": np.linspace(1.0, 2.0, n_frames),
            "local_x": x,
            "local_y": y,
            "metadata": {"target": "Pluto"},
        }
        result = scrub_track_from_cube_bundle(
            bundle,
            x,
            y,
            mask_radius=1.0,
            fill_inner_radius=2.0,
            fill_outer_radius=3.0,
            rng_seed=5,
        )
        scrubbed_bundle = result["bundle"]
        scrubbed_cube = scrubbed_bundle["cube"]
        self.assertEqual(scrubbed_bundle["metadata"]["target"], "Pluto")
        self.assertTrue(bool(scrubbed_bundle["metadata"]["scrubbed_source"]))
        self.assertLess(float(scrubbed_cube[:, 5, 5].max()), 5.0)
        self.assertEqual(scrubbed_cube.shape, cube.shape)


if __name__ == "__main__":
    unittest.main()
