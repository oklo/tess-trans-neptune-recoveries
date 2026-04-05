#!/usr/bin/env python3
"""Known-path detector-local recovery driver for a target TESS track."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kbo_tess.ephem import load_ephemeris_csv, query_tess_ephem, select_detector_track  # noqa: E402
from kbo_tess.recovery import recover_from_local_files, run_known_path_recovery, write_recovery_products  # noqa: E402
from kbo_tess.tess_loader import build_cube_bundle_from_tess_asteroids  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover a target from a detector-local TESS track with known-path shift-and-stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Sedna", help="Target designation.")
    parser.add_argument("--sector", type=int, default=5, help="TESS observing sector.")
    parser.add_argument("--camera", type=int, default=1, help="TESS camera number.")
    parser.add_argument("--ccd", type=int, default=4, help="TESS CCD number.")
    parser.add_argument(
        "--ephem-csv",
        type=Path,
        help="Optional recovery-ready ephemeris CSV for the local-file path.",
    )
    parser.add_argument(
        "--cube-npz",
        type=Path,
        help=(
            "Optional detector-local NPZ cube for the local-file path. Required keys: "
            "`cube`, `time_tdb_jd_minus_2457000`; optional keys: `quality_mask`, `metadata_json`."
        ),
    )
    parser.add_argument(
        "--synthetic-smoke-test",
        action="store_true",
        help="Run the known-path recovery pipeline on a tiny synthetic cube.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned workflow without writing outputs.",
    )
    parser.add_argument(
        "--background-method",
        choices=("none", "median", "savgol"),
        default="median",
        help="Background subtraction method for the cutout cube.",
    )
    parser.add_argument(
        "--offset-grid-radius",
        type=float,
        default=1.0,
        help="Half-width of the constant dx/dy offset search grid in pixels.",
    )
    parser.add_argument(
        "--offset-grid-step",
        type=float,
        default=0.5,
        help="Step size for the constant dx/dy offset search grid in pixels.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "linear"),
        default="nearest",
        help="Interpolation mode for per-frame image shifts.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for derived outputs.",
    )
    return parser


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for the recovery driver. Install project dependencies with "
            "`python -m pip install -e .` before running recovery."
        ) from exc
    return np


def _load_detector_track(args: argparse.Namespace):
    if args.ephem_csv is not None:
        ephem_df = load_ephemeris_csv(
            args.ephem_csv,
            target=args.target,
            require_exact_tdb=True,
        )
    else:
        ephem_df = query_tess_ephem(
            target=args.target,
            sector=args.sector,
            camera=args.camera,
            ccd=args.ccd,
            require_exact_tdb=True,
        )
    return select_detector_track(
        ephem_df,
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        target=args.target,
        require_exact_tdb=True,
    )


def _build_synthetic_inputs(
    n_frames: int = 17,
    shape: tuple[int, int] = (32, 32),
    amplitude: float = 5.0,
    noise_sigma: float = 1.0,
):
    np = _require_numpy()
    rng = np.random.default_rng(5)
    yy, xx = np.indices(shape, dtype=float)
    times = 1400.0 + 0.020833333 * np.arange(n_frames)
    x = 12.0 + 0.35 * np.arange(n_frames)
    y = 14.0 + 0.20 * np.arange(n_frames)

    cube = rng.normal(loc=100.0, scale=noise_sigma, size=(n_frames, *shape))
    cube += np.linspace(-0.5, 0.5, n_frames)[:, None, None]
    for frame_index in range(n_frames):
        dx = xx - x[frame_index]
        dy = yy - y[frame_index]
        cube[frame_index] += amplitude * np.exp(-(dx * dx + dy * dy) / (2.0 * 1.2**2))

    ephem_df = {
        "target": ["Sedna"] * n_frames,
        "sector": [5] * n_frames,
        "camera": [1] * n_frames,
        "ccd": [4] * n_frames,
        "time_tdb_jd_minus_2457000": times,
        "x": x,
        "y": y,
    }
    cube_bundle = {
        "cube": cube,
        "time_tdb_jd_minus_2457000": times,
        "quality_mask": np.ones(n_frames, dtype=bool),
        "metadata": {
            "mode": "synthetic",
            "target": "Sedna",
            "sector": 5,
            "camera": 1,
            "ccd": 4,
            "shape": list(shape),
            "n_frames_total": n_frames,
        },
    }
    return ephem_df, cube_bundle


def _infer_mode(args: argparse.Namespace) -> str:
    if args.synthetic_smoke_test:
        return "synthetic-smoke-test"
    if args.cube_npz is not None:
        return "local-files"
    return "live-adapter"


def _print_plan(args: argparse.Namespace, mode: str) -> None:
    print("Recovery configuration:")
    print(f"  mode: {mode}")
    print(f"  target: {args.target}")
    print(f"  sector/camera/ccd: {args.sector}/{args.camera}/{args.ccd}")
    if args.ephem_csv is not None:
        print(f"  ephem_csv: {args.ephem_csv}")
    if args.cube_npz is not None:
        print(f"  cube_npz: {args.cube_npz}")
    print(f"  background_method: {args.background_method}")
    print(f"  offset_grid_radius: {args.offset_grid_radius}")
    print(f"  offset_grid_step: {args.offset_grid_step}")
    print(f"  interpolation: {args.interpolation}")
    print(f"  outdir: {args.outdir}")


def main() -> int:
    args = build_parser().parse_args()
    if args.cube_npz is not None and args.ephem_csv is None:
        raise SystemExit("--cube-npz requires --ephem-csv so the recovery track is explicit.")

    mode = _infer_mode(args)
    _print_plan(args, mode=mode)
    if args.dry_run:
        print("Dry run: no data loaded and no outputs written.")
        return 0

    if mode == "synthetic-smoke-test":
        ephem_df, cube_bundle = _build_synthetic_inputs()
        recovery = run_known_path_recovery(
            ephem_df,
            cube_bundle,
            background_method=args.background_method,
            interpolation=args.interpolation,
            offset_grid_radius=args.offset_grid_radius,
            offset_grid_step=args.offset_grid_step,
        )
    elif mode == "local-files":
        recovery = recover_from_local_files(
            args.ephem_csv,
            args.cube_npz,
            background_method=args.background_method,
            interpolation=args.interpolation,
            offset_grid_radius=args.offset_grid_radius,
            offset_grid_step=args.offset_grid_step,
        )
    else:
        detector_track = _load_detector_track(args)
        cube_bundle = build_cube_bundle_from_tess_asteroids(detector_track, target=args.target)
        recovery = run_known_path_recovery(
            detector_track,
            cube_bundle,
            background_method=args.background_method,
            interpolation=args.interpolation,
            offset_grid_radius=args.offset_grid_radius,
            offset_grid_step=args.offset_grid_step,
        )

    output_paths = write_recovery_products(recovery, args.outdir)
    summary = recovery["summary"]
    print("Recovery summary:")
    print(f"  n_frames: {summary['n_frames']}")
    print(f"  best_dx: {summary['best_dx']}")
    print(f"  best_dy: {summary['best_dy']}")
    print(f"  best_score: {summary['best_score']}")
    print(f"  recovered_reference_pixel: {summary['recovered_reference_pixel']}")
    print(f"  summary_json: {output_paths['summary_json']}")
    print(f"  offset_csv: {output_paths['offset_csv']}")
    print(f"  growth_csv: {output_paths['growth_csv']}")
    print(f"  diagnostic_png: {output_paths['diagnostic_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
