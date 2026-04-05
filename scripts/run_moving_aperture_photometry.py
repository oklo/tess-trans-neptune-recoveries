#!/usr/bin/env python3
"""Run a direct moving-aperture sanity check on a detector-local target bundle."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from kbo_tess.ephem import load_ephemeris_csv, select_detector_track  # noqa: E402
from kbo_tess.photometry import run_moving_aperture_photometry, write_moving_aperture_products  # noqa: E402
from kbo_tess.recovery import load_cube_npz  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure direct moving-aperture photometry along a predicted detector-local track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Sedna", help="Target designation.")
    parser.add_argument("--sector", type=int, required=True, help="TESS sector.")
    parser.add_argument("--camera", type=int, required=True, help="TESS camera.")
    parser.add_argument("--ccd", type=int, required=True, help="TESS CCD.")
    parser.add_argument("--ephem-csv", type=Path, required=True, help="Normalized ephemeris CSV.")
    parser.add_argument("--cube-npz", type=Path, required=True, help="Local cube bundle NPZ.")
    parser.add_argument("--background-method", choices=("none", "median", "savgol"), default="none", help="Optional temporal background handling before aperture photometry.")
    parser.add_argument("--aperture-radius", type=float, default=1.5, help="Aperture radius in pixels.")
    parser.add_argument("--annulus-inner-radius", type=float, default=3.0, help="Annulus inner radius in pixels.")
    parser.add_argument("--annulus-outer-radius", type=float, default=5.0, help="Annulus outer radius in pixels.")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs", help="Directory for derived outputs.")
    return parser


def _prefix(target: str, sector: int, camera: int, ccd: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", target.lower()).strip("_")
    return f"{slug}_sector{sector}_cam{camera}_ccd{ccd}"


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
    photometry = run_moving_aperture_photometry(
        detector_track,
        cube_bundle,
        background_method=args.background_method,
        aperture_radius=args.aperture_radius,
        annulus_inner_radius=args.annulus_inner_radius,
        annulus_outer_radius=args.annulus_outer_radius,
    )
    output_paths = write_moving_aperture_products(
        photometry,
        args.outdir,
        prefix=_prefix(args.target, args.sector, args.camera, args.ccd),
    )
    summary = photometry["summary"]
    print("Moving-aperture photometry:")
    print(f"  n_frames: {summary['n_frames']}")
    print(f"  background_method: {summary['background_method']}")
    print(f"  median_net_flux: {summary['median_net_flux']:.3f}")
    print(f"  max_net_flux: {summary['max_net_flux']:.3f}")
    print(f"  median_aperture_snr: {summary['median_aperture_snr']:.3f}")
    print(f"  max_aperture_snr: {summary['max_aperture_snr']:.3f}")
    print(f"  summary_json: {output_paths['summary_json']}")
    print(f"  lightcurve_csv: {output_paths['lightcurve_csv']}")
    print(f"  diagnostic_png: {output_paths['diagnostic_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
