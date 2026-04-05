#!/usr/bin/env python3
"""Geometry sanity check for a real Sedna moving-cutout bundle."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from kbo_tess.ephem import load_ephemeris_csv, select_detector_track  # noqa: E402
from kbo_tess.geometry import evaluate_geometry_consistency, write_geometry_products  # noqa: E402
from kbo_tess.recovery import load_cube_npz  # noqa: E402
from kbo_tess.tess_loader import build_cube_bundle_from_tess_asteroids  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check Sedna timing and geometry consistency before any recovery claim.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Sedna", help="Target designation.")
    parser.add_argument("--sector", type=int, default=5, help="TESS sector.")
    parser.add_argument("--camera", type=int, default=1, help="TESS camera.")
    parser.add_argument("--ccd", type=int, default=4, help="TESS CCD.")
    parser.add_argument("--ephem-csv", type=Path, required=True, help="Normalized ephemeris CSV.")
    parser.add_argument("--cube-npz", type=Path, help="Optional prebuilt cube bundle NPZ.")
    parser.add_argument("--shape", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(21, 21), help="Cutout shape if building live.")
    parser.add_argument("--frame-limit", type=int, help="Optional frame cap if building live.")
    parser.add_argument("--time-start", type=float, help="Optional lower BTJD bound if building live.")
    parser.add_argument("--time-stop", type=float, help="Optional upper BTJD bound if building live.")
    parser.add_argument("--barycentric-mode", choices=("true", "false"), help="Explicit barycentric mode if building live.")
    parser.add_argument("--bg-method", choices=("linear_model", "rolling"), default="rolling", help="tess_asteroids background method if building live.")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs", help="Directory for derived outputs.")
    return parser


def _prefix(args: argparse.Namespace) -> str:
    return f"{args.target.lower()}_sector{args.sector}_cam{args.camera}_ccd{args.ccd}"


def main() -> int:
    args = build_parser().parse_args()
    ephem_df = load_ephemeris_csv(args.ephem_csv, target=args.target, require_exact_tdb=True)
    detector_track = select_detector_track(
        ephem_df,
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        target=args.target,
        require_exact_tdb=True,
    )

    if args.cube_npz is not None:
        cube_bundle = load_cube_npz(args.cube_npz)
    else:
        cube_bundle = build_cube_bundle_from_tess_asteroids(
            detector_track,
            target=args.target,
            shape=tuple(args.shape),
            frame_limit=args.frame_limit,
            time_start=args.time_start,
            time_stop=args.time_stop,
            barycentric_mode=args.barycentric_mode,
            bg_method=args.bg_method,
        )

    geometry = evaluate_geometry_consistency(detector_track, cube_bundle)
    outputs = write_geometry_products(geometry, args.outdir, prefix=_prefix(args))
    summary = geometry["summary"]
    print("Geometry summary:")
    print(f"  n_cadences: {summary['n_cadences']}")
    print(f"  time span: {summary['time_start']:.8f} -> {summary['time_stop']:.8f}")
    print(f"  corner median/max residual [px]: {summary['median_geometry_residual_corner_px']:.5f} / {summary['max_geometry_residual_corner_px']:.5f}")
    print(f"  poscorr median/max residual [px]: {summary['median_geometry_residual_poscorr_px']:.5f} / {summary['max_geometry_residual_poscorr_px']:.5f}")
    print(f"  detector median/max residual [px]: {summary['median_detector_residual_px']:.5f} / {summary['max_detector_residual_px']:.5f}")
    print(f"  barycentric_mode: {summary['barycentric_mode']}")
    print(f"  bg_method: {summary['bg_method']}")
    print(f"  geometry_check_passed: {summary['geometry_check_passed']}")
    print(f"  summary_json: {outputs['summary_json']}")
    print(f"  residual_csv: {outputs['residual_csv']}")
    print(f"  diagnostic_png: {outputs['diagnostic_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
