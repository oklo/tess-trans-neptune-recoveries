#!/usr/bin/env python3
"""Run a small real-bundle injection sweep on a bounded Sedna cutout bundle."""

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
    estimate_bundle_noise_sigma,
    run_injection_sweep,
    write_injection_products,
)
from kbo_tess.ephem import load_ephemeris_csv, select_detector_track  # noqa: E402
from kbo_tess.recovery import load_cube_npz  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inject a synthetic moving source into a real Sedna bundle and sweep amplitudes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Sedna", help="Target designation.")
    parser.add_argument("--sector", type=int, default=5, help="TESS sector.")
    parser.add_argument("--camera", type=int, default=1, help="TESS camera.")
    parser.add_argument("--ccd", type=int, default=4, help="TESS CCD.")
    parser.add_argument("--ephem-csv", type=Path, required=True, help="Normalized detector-local ephemeris CSV.")
    parser.add_argument("--cube-npz", type=Path, required=True, help="Real local bundle NPZ to inject into.")
    parser.add_argument("--amplitudes", type=float, nargs="*", help="Explicit injection amplitudes in cube flux units.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Injected Gaussian sigma in pixels.")
    parser.add_argument("--inject-dx", type=float, default=0.0, help="Constant x offset applied to the injected source track.")
    parser.add_argument("--inject-dy", type=float, default=0.0, help="Constant y offset applied to the injected source track.")
    parser.add_argument("--background-method", choices=("none", "median", "savgol"), default="none", help="Temporal background handling during recovery.")
    parser.add_argument("--interpolation", choices=("nearest", "linear"), default="linear", help="Shift interpolation mode.")
    parser.add_argument("--offset-grid-radius", type=float, default=1.0, help="Offset-search half-width in pixels.")
    parser.add_argument("--offset-grid-step", type=float, default=0.5, help="Offset-search step size in pixels.")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "real_control_study", help="Directory for injection-calibration outputs.")
    return parser


def _default_amplitudes(noise_sigma: float) -> list[float]:
    factors = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    return [float(factor * noise_sigma) for factor in factors]


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
    noise_sigma = estimate_bundle_noise_sigma(cube_bundle)
    amplitudes = list(args.amplitudes) if args.amplitudes else _default_amplitudes(noise_sigma)

    injection_summary = run_injection_sweep(
        detector_track,
        cube_bundle,
        amplitudes=amplitudes,
        sigma=args.sigma,
        dx=args.inject_dx,
        dy=args.inject_dy,
        background_method=args.background_method,
        interpolation=args.interpolation,
        offset_grid_radius=args.offset_grid_radius,
        offset_grid_step=args.offset_grid_step,
    )
    injection_summary["noise_sigma_estimate"] = float(noise_sigma)
    output_paths = write_injection_products(
        injection_summary,
        args.outdir,
        prefix="sedna_sector5_injection_calibration",
    )

    print("Injection calibration:")
    print(f"  noise_sigma_estimate: {noise_sigma:.6f}")
    print(f"  amplitudes: {', '.join(f'{value:.6f}' for value in amplitudes)}")
    for row in injection_summary["results"]:
        print(
            f"  amp={row['amplitude']:.6f}: nominal={row['nominal_score']:.3f}, "
            f"best={row['best_score']:.3f}, best_dxdy=({row['best_dx']:.2f},{row['best_dy']:.2f})"
        )
    print(f"  summary_json: {output_paths['summary_json']}")
    print(f"  summary_csv: {output_paths['summary_csv']}")
    print(f"  diagnostic_png: {output_paths['diagnostic_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
