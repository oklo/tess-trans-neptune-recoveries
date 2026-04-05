#!/usr/bin/env python3
"""Compare spacecraft-vs-barycentric timing interpretations on one bounded Sedna subset."""

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

from kbo_tess.control_studies import (  # noqa: E402
    build_mode_comparison_summary,
    run_recovery_suite,
    write_mode_comparison_products,
)
from kbo_tess.ephem import load_ephemeris_csv, select_detector_track  # noqa: E402
from kbo_tess.geometry import evaluate_geometry_consistency, write_geometry_products  # noqa: E402
from kbo_tess.recovery import write_recovery_products  # noqa: E402
from kbo_tess.tess_loader import (  # noqa: E402
    build_cube_bundle_from_tess_asteroids,
    materialize_cube_bundle_to_npz,
    subset_ephemeris_for_build,
    write_cube_bundle_metadata_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a side-by-side barycentric timing control on one bounded real Sedna subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Sedna", help="Target designation.")
    parser.add_argument("--sector", type=int, default=5, help="TESS sector.")
    parser.add_argument("--camera", type=int, default=1, help="TESS camera.")
    parser.add_argument("--ccd", type=int, default=4, help="TESS CCD.")
    parser.add_argument("--ephem-csv", type=Path, required=True, help="Normalized Sedna ephemeris CSV.")
    parser.add_argument("--shape", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(21, 21), help="Moving cutout shape.")
    parser.add_argument("--time-start", type=float, default=1438.5008, help="Lower BTJD bound matching the current rehearsal subset.")
    parser.add_argument("--time-stop", type=float, default=1440.6008, help="Upper BTJD bound matching the current rehearsal subset.")
    parser.add_argument(
        "--loader-bg-method",
        choices=("rolling", "linear_model"),
        default="rolling",
        help="tess_asteroids background method used when building both bundles.",
    )
    parser.add_argument(
        "--background-methods",
        nargs="+",
        choices=("none", "median", "savgol"),
        default=("none", "median", "savgol"),
        help="Temporal background methods to compare on each bundle.",
    )
    parser.add_argument(
        "--primary-background",
        choices=("none", "median", "savgol"),
        default="none",
        help="Background method to highlight in the side-by-side summary.",
    )
    parser.add_argument("--interpolation", choices=("nearest", "linear"), default="linear", help="Shift interpolation mode.")
    parser.add_argument("--offset-grid-radius", type=float, default=1.0, help="Offset-search half-width in pixels.")
    parser.add_argument("--offset-grid-step", type=float, default=0.5, help="Offset-search step size in pixels.")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "real_control_study", help="Directory for comparison products.")
    return parser


def _mode_prefix(mode_label: str) -> str:
    return f"sedna_sector5_cam1_ccd4_{mode_label}"


def main() -> int:
    args = build_parser().parse_args()
    if args.primary_background not in args.background_methods:
        raise SystemExit("--primary-background must be included in --background-methods.")

    ephem_df = load_ephemeris_csv(args.ephem_csv, target=args.target, require_exact_tdb=True)
    detector_track = select_detector_track(
        ephem_df,
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        target=args.target,
        require_exact_tdb=True,
    )
    bounded_track = subset_ephemeris_for_build(
        detector_track,
        target=args.target,
        time_start=args.time_start,
        time_stop=args.time_stop,
    )

    mode_inputs = {"spacecraft": False, "barycentric": True}
    mode_results: dict[str, dict[str, object]] = {}
    for mode_label, barycentric_mode in mode_inputs.items():
        mode_dir = args.outdir / mode_label
        bundle = build_cube_bundle_from_tess_asteroids(
            bounded_track,
            target=args.target,
            shape=tuple(args.shape),
            barycentric_mode=barycentric_mode,
            bg_method=args.loader_bg_method,
        )
        bundle_npz = materialize_cube_bundle_to_npz(bundle, mode_dir / f"{_mode_prefix(mode_label)}_cube_bundle.npz")
        write_cube_bundle_metadata_json(bundle, mode_dir / f"{_mode_prefix(mode_label)}_cube_bundle_metadata.json")

        geometry = evaluate_geometry_consistency(bounded_track, bundle)
        write_geometry_products(geometry, mode_dir, prefix=_mode_prefix(mode_label))

        recoveries = run_recovery_suite(
            bounded_track,
            bundle,
            background_methods=tuple(args.background_methods),
            interpolation=args.interpolation,
            offset_grid_radius=args.offset_grid_radius,
            offset_grid_step=args.offset_grid_step,
        )
        for background_method, recovery in recoveries.items():
            write_recovery_products(
                recovery,
                mode_dir / f"recovery_{background_method}",
                prefix=_mode_prefix(mode_label),
            )

        mode_results[mode_label] = {
            "bundle_npz": bundle_npz,
            "geometry": geometry,
            "recoveries": recoveries,
        }

    comparison_summary = build_mode_comparison_summary(
        mode_results,
        primary_background=args.primary_background,
    )
    output_paths = write_mode_comparison_products(
        comparison_summary,
        args.outdir,
        prefix="sedna_sector5_cam1_ccd4_barycentric_compare",
    )

    print("Barycentric comparison:")
    for row in comparison_summary["csv_rows"]:
        print(
            f"  {row['mode']}: frames={row['n_frames']}, geometry_passed={row['geometry_passed']}, "
            f"corner_median={row['geometry_corner_median_px']:.5f}, best_dxdy=({row['best_dx']:.2f},{row['best_dy']:.2f}), "
            f"nominal={row['nominal_score']:.3f}, best={row['best_score']:.3f}"
        )
    print(f"  preferred_by_geometry: {comparison_summary['preferred_by_geometry']}")
    print(f"  preferred_by_nominal_score: {comparison_summary['preferred_by_nominal_score']}")
    print(f"  preferred_by_best_score: {comparison_summary['preferred_by_best_score']}")
    print(f"  summary_json: {output_paths['summary_json']}")
    print(f"  summary_csv: {output_paths['summary_csv']}")
    print(f"  diagnostic_png: {output_paths['diagnostic_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
