#!/usr/bin/env python3
"""Build a bounded real-data Sedna moving-cutout bundle for Sector 5 / camera 1 / CCD 4."""

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

from kbo_tess.ephem import load_ephemeris_csv, select_detector_track, summarize_track  # noqa: E402
from kbo_tess.tess_loader import (  # noqa: E402
    build_cube_bundle_from_tess_asteroids,
    materialize_cube_bundle_to_npz,
    subset_ephemeris_for_build,
    write_cube_bundle_metadata_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a real moving TESS cutout bundle for a bounded Sedna rehearsal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Sedna", help="Target designation.")
    parser.add_argument("--sector", type=int, default=5, help="TESS sector.")
    parser.add_argument("--camera", type=int, default=1, help="TESS camera.")
    parser.add_argument("--ccd", type=int, default=4, help="TESS CCD.")
    parser.add_argument("--ephem-csv", type=Path, required=True, help="Normalized ephemeris CSV input.")
    parser.add_argument("--shape", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(21, 21), help="Moving cutout shape.")
    parser.add_argument("--frame-limit", type=int, help="Optional upper limit on retained cadences.")
    parser.add_argument("--time-start", type=float, help="Optional lower BTJD bound for the ephemeris subset.")
    parser.add_argument("--time-stop", type=float, help="Optional upper BTJD bound for the ephemeris subset.")
    parser.add_argument(
        "--barycentric-mode",
        choices=("true", "false"),
        help="Explicit barycentric mode for tess_asteroids custom ephemerides.",
    )
    parser.add_argument(
        "--bg-method",
        choices=("linear_model", "rolling"),
        default="rolling",
        help="tess_asteroids background method applied during MovingTPF construction.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for derived products.",
    )
    return parser


def _prefix(args: argparse.Namespace) -> str:
    return f"{args.target.lower()}_sector{args.sector}_cam{args.camera}_ccd{args.ccd}"


def _plot_bundle_preview(bundle: dict[str, object], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cube = np.asarray(bundle["cube"], dtype=float)
    local_x = np.asarray(bundle["local_x"], dtype=float)
    local_y = np.asarray(bundle["local_y"], dtype=float)
    frame_index = int(len(cube) // 2)
    frame = cube[frame_index]
    finite = frame[np.isfinite(frame)]
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [5, 99])
    else:
        vmin, vmax = -1.0, 1.0

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(frame, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].plot(local_x, local_y, color="white", linewidth=1.0)
    axes[0].scatter([local_x[frame_index]], [local_y[frame_index]], color="red", marker="x", s=60)
    axes[0].set_title("Representative Real Cutout")
    axes[0].set_xlabel("Local x [pix]")
    axes[0].set_ylabel("Local y [pix]")

    axes[1].plot(np.asarray(bundle["time_tdb_jd_minus_2457000"], dtype=float), local_x, label="local x")
    axes[1].plot(np.asarray(bundle["time_tdb_jd_minus_2457000"], dtype=float), local_y, label="local y")
    axes[1].set_title("Predicted Local Track")
    axes[1].set_xlabel("BTJD")
    axes[1].set_ylabel("Pixel")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main() -> int:
    args = build_parser().parse_args()
    prefix = _prefix(args)
    npz_path = args.outdir / f"{prefix}_cube_bundle.npz"
    metadata_path = args.outdir / f"{prefix}_cube_bundle_metadata.json"
    preview_path = args.outdir / f"{prefix}_cube_bundle.png"

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
        frame_limit=args.frame_limit,
    )
    summary = summarize_track(bounded_track)
    print("Ephemeris subset:")
    print(f"  rows: {summary['rows']}")
    print(f"  time span: {summary['time_start']:.8f} -> {summary['time_stop']:.8f}")
    print(f"  x span: {summary['x_min']:.3f} -> {summary['x_max']:.3f}")
    print(f"  y span: {summary['y_min']:.3f} -> {summary['y_max']:.3f}")

    bundle = build_cube_bundle_from_tess_asteroids(
        bounded_track,
        target=args.target,
        shape=tuple(args.shape),
        barycentric_mode=args.barycentric_mode,
        bg_method=args.bg_method,
    )
    materialize_cube_bundle_to_npz(bundle, npz_path)
    write_cube_bundle_metadata_json(bundle, metadata_path)
    _plot_bundle_preview(bundle, preview_path)

    print("Cube bundle:")
    print(f"  frames: {len(bundle['time_tdb_jd_minus_2457000'])}")
    print(f"  barycentric_mode: {bundle['metadata']['barycentric_mode']}")
    print(f"  loader_bg_method: {bundle['metadata']['loader_bg_method']}")
    print(f"  npz: {npz_path}")
    print(f"  metadata_json: {metadata_path}")
    print(f"  preview_png: {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
