#!/usr/bin/env python3
"""Run a small null-track control study around the nominal Sedna path."""

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

from kbo_tess.control_studies import evaluate_null_track_controls, write_null_control_products  # noqa: E402
from kbo_tess.ephem import load_ephemeris_csv, select_detector_track  # noqa: E402
from kbo_tess.recovery import load_cube_npz, run_known_path_recovery, write_recovery_products  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the nominal Sedna track against a small set of neighboring null tracks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Sedna", help="Target designation.")
    parser.add_argument("--sector", type=int, default=5, help="TESS sector.")
    parser.add_argument("--camera", type=int, default=1, help="TESS camera.")
    parser.add_argument("--ccd", type=int, default=4, help="TESS CCD.")
    parser.add_argument("--ephem-csv", type=Path, required=True, help="Normalized detector-local ephemeris CSV.")
    parser.add_argument("--cube-npz", type=Path, required=True, help="Local cube bundle NPZ.")
    parser.add_argument("--background-method", choices=("none", "median", "savgol"), default="none", help="Temporal background handling during recovery.")
    parser.add_argument("--interpolation", choices=("nearest", "linear"), default="linear", help="Shift interpolation mode.")
    parser.add_argument("--offset-grid-radius", type=float, default=1.0, help="Offset-search half-width in pixels.")
    parser.add_argument("--offset-grid-step", type=float, default=0.5, help="Offset-search step size in pixels.")
    parser.add_argument("--control-radius", type=float, help="Ring radius to treat as null controls. Defaults to the outer search ring.")
    parser.add_argument("--near-radius", type=float, help="Radius defining the near-nominal neighborhood. Defaults to the inner non-zero search ring.")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "real_control_study", help="Directory for null-control outputs.")
    return parser


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
    cube_bundle = load_cube_npz(args.cube_npz)
    recovery = run_known_path_recovery(
        detector_track,
        cube_bundle,
        background_method=args.background_method,
        interpolation=args.interpolation,
        offset_grid_radius=args.offset_grid_radius,
        offset_grid_step=args.offset_grid_step,
    )
    write_recovery_products(recovery, args.outdir / "null_control_recovery", prefix="sedna_sector5_null_control_recovery")
    null_summary_bundle = evaluate_null_track_controls(
        recovery,
        control_radius=args.control_radius,
        near_radius=args.near_radius,
    )
    output_paths = write_null_control_products(
        null_summary_bundle,
        args.outdir,
        prefix="sedna_sector5_null_track_controls",
    )

    summary = null_summary_bundle["summary"]
    print("Null-track controls:")
    print(f"  nominal_score: {summary['nominal_score']:.3f}")
    print(f"  best_local_score: {summary['best_score']:.3f} at ({summary['best_dx']:.2f},{summary['best_dy']:.2f})")
    print(
        f"  best_near_nominal_score: {summary['best_near_nominal_score']:.3f} "
        f"at ({summary['best_near_nominal_dx']:.2f},{summary['best_near_nominal_dy']:.2f})"
    )
    print(
        f"  best_null_control_score: {summary['best_control_score']:.3f} "
        f"at ({summary['best_control_dx']:.2f},{summary['best_control_dy']:.2f})"
    )
    print(f"  best_offset_is_near_nominal: {summary['best_offset_is_near_nominal']}")
    print(f"  nominal_minus_best_control: {summary['nominal_minus_best_control']:.3f}")
    print(f"  best_minus_best_control: {summary['best_minus_best_control']:.3f}")
    print(f"  summary_json: {output_paths['summary_json']}")
    print(f"  controls_csv: {output_paths['controls_csv']}")
    print(f"  cadence_csv: {output_paths['cadence_csv']}")
    print(f"  diagnostic_png: {output_paths['diagnostic_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
