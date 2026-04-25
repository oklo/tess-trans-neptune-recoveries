#!/usr/bin/env python3
"""Run a Holman-style image-domain stack for one known target detector track."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from kbo_tess.automask import compute_mask_bundle
from kbo_tess.holman import (
    aperture_stats,
    build_differential_cube,
    holman_stack,
    snr_map_from_stack,
    standard_error_from_aligned,
    write_holman_stack_figure,
    write_stack_npz,
    write_summary_json,
)
from kbo_tess.moving_cutout import (
    build_raw_flux_cube,
    load_bundle,
    local_track_from_bundle,
    match_bundle_times_to_raw,
    select_detector_ephem,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Holman-style differential image stack for one known target detector track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--sector", type=int, required=True)
    parser.add_argument("--camera", type=int, required=True)
    parser.add_argument("--ccd", type=int, required=True)
    parser.add_argument("--ephem-csv", type=Path, required=True)
    parser.add_argument("--bundle-npz", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--cutout-size", type=int, default=21)
    parser.add_argument("--bg-method", choices=("linear_model", "rolling"), default="rolling")
    parser.add_argument("--barycentric-mode", choices=("true", "false"), default="true")
    parser.add_argument("--interpolation", choices=("nearest", "linear"), default="linear")
    parser.add_argument("--target-box-size", type=int, default=5)
    parser.add_argument("--aperture-radius", type=float, default=1.5)
    parser.add_argument("--mask-mode", choices=("simple", "common_mode_v1"), default="common_mode_v1")
    parser.add_argument("--mask-window", type=int, default=49)
    parser.add_argument("--mask-sigma", type=float, default=5.0)
    parser.add_argument("--path-box-size", type=int, default=9)
    parser.add_argument("--baseline-hours", type=float, default=24.0)
    parser.add_argument("--slope-smooth-hours", type=float, default=1.0)
    parser.add_argument("--seed-z-outer", type=float, default=4.0)
    parser.add_argument("--seed-z-path", type=float, default=4.0)
    parser.add_argument("--seed-z-q90", type=float, default=3.0)
    parser.add_argument("--seed-z-scatter", type=float, default=3.0)
    parser.add_argument("--seed-z-slope", type=float, default=4.0)
    parser.add_argument("--grow-z", type=float, default=2.0)
    parser.add_argument("--merge-gap-hours", type=float, default=10.0)
    parser.add_argument("--interval-pad-hours", type=float, default=6.0)
    parser.add_argument("--min-interval-hours", type=float, default=6.0)
    parser.add_argument("--edge-window-hours", type=float, default=12.0)
    parser.add_argument("--edge-reference-hours", type=float, default=24.0)
    parser.add_argument("--edge-release-hours", type=float, default=6.0)
    parser.add_argument("--edge-release-z", type=float, default=1.5)
    parser.add_argument("--extra-mask-times", type=float, nargs="*", default=())
    parser.add_argument("--extra-mask-ranges", type=float, nargs=2, action="append", default=[])
    return parser


def slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text).strip()).strip("_")


def main() -> int:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{slug(args.target)}_sector{int(args.sector)}_cam{int(args.camera)}_ccd{int(args.ccd)}"

    bundle = load_bundle(Path(args.bundle_npz))
    bundle_time = np.asarray(bundle["time_tdb_jd_minus_2457000"], dtype=float)
    local_x, local_y = local_track_from_bundle(bundle)

    ephem = select_detector_ephem(Path(args.ephem_csv), str(args.target), int(args.sector), int(args.camera), int(args.ccd))
    raw_cube, raw_time = build_raw_flux_cube(
        ephem,
        shape=(int(args.cutout_size), int(args.cutout_size)),
        bg_method=str(args.bg_method),
        barycentric=(str(args.barycentric_mode) == "true"),
    )
    match_idx = match_bundle_times_to_raw(raw_time, bundle_time)
    raw_matched = raw_cube[match_idx]

    keep, mask_df, interval_records, mask_summary = compute_mask_bundle(
        raw_matched,
        bundle_time=bundle_time,
        local_x=local_x,
        local_y=local_y,
        mask_mode=str(args.mask_mode),
        mask_window=int(args.mask_window),
        mask_sigma=float(args.mask_sigma),
        target_box_size=int(args.target_box_size),
        path_box_size=int(args.path_box_size),
        baseline_hours=float(args.baseline_hours),
        slope_smooth_hours=float(args.slope_smooth_hours),
        seed_z_outer=float(args.seed_z_outer),
        seed_z_path=float(args.seed_z_path),
        seed_z_q90=float(args.seed_z_q90),
        seed_z_scatter=float(args.seed_z_scatter),
        seed_z_slope=float(args.seed_z_slope),
        grow_z=float(args.grow_z),
        merge_gap_hours=float(args.merge_gap_hours),
        interval_pad_hours=float(args.interval_pad_hours),
        min_interval_hours=float(args.min_interval_hours),
        edge_window_hours=float(args.edge_window_hours),
        edge_reference_hours=float(args.edge_reference_hours),
        edge_release_hours=float(args.edge_release_hours),
        edge_release_z=float(args.edge_release_z),
        extra_mask_times=tuple(float(value) for value in args.extra_mask_times),
        extra_mask_ranges=tuple((float(start), float(stop)) for start, stop in args.extra_mask_ranges),
    )

    diff = build_differential_cube(
        raw_matched,
        local_x=local_x,
        local_y=local_y,
        keep=keep,
        target_box_size=int(args.target_box_size),
    )
    stack_result = holman_stack(
        diff["differential_cube"],
        local_x=local_x,
        local_y=local_y,
        keep=keep,
        interpolation=str(args.interpolation),
    )
    center = tuple(float(value) for value in stack_result["reference"])
    stderr_map = standard_error_from_aligned(np.asarray(stack_result["aligned_frames"], dtype=float))
    snr_map = snr_map_from_stack(np.asarray(stack_result["stack"], dtype=float), stderr_map)
    aperture_summary = aperture_stats(
        np.asarray(stack_result["stack"], dtype=float),
        stderr_map,
        center=center,
        radius=float(args.aperture_radius),
    )

    mask_csv = outdir / f"{prefix}_holman_mask.csv"
    interval_json = outdir / f"{prefix}_holman_mask_intervals.json"
    stack_png = outdir / f"{prefix}_holman_stack.png"
    stack_npz = outdir / f"{prefix}_holman_stack.npz"
    summary_json = outdir / f"{prefix}_holman_summary.json"

    mask_df.to_csv(mask_csv, index=False)
    interval_json.write_text(json.dumps(interval_records, indent=2) + "\n", encoding="utf-8")
    write_stack_npz(
        stack_npz,
        reference_image=np.asarray(diff["reference_image"], dtype=float),
        differential_stack=np.asarray(stack_result["stack"], dtype=float),
        stderr_map=stderr_map,
        snr_map=snr_map,
        support_map=np.asarray(stack_result["support_map"], dtype=float),
    )
    write_holman_stack_figure(
        np.asarray(diff["reference_image"], dtype=float),
        np.asarray(stack_result["stack"], dtype=float),
        snr_map,
        np.asarray(stack_result["support_map"], dtype=float),
        out_png=stack_png,
        target=str(args.target),
        sector=int(args.sector),
        camera=int(args.camera),
        ccd=int(args.ccd),
        keep_count=int(np.count_nonzero(keep)),
        total_count=int(len(keep)),
        center=center,
        aperture_radius=float(args.aperture_radius),
        interpolation=str(args.interpolation),
        background_mode_label="outer-median + temporal-median reference",
        aperture_summary=aperture_summary,
    )
    write_summary_json(
        {
            "target": str(args.target),
            "sector": int(args.sector),
            "camera": int(args.camera),
            "ccd": int(args.ccd),
            "ephem_csv": str(args.ephem_csv),
            "bundle_npz": str(args.bundle_npz),
            "mask_csv": str(mask_csv),
            "interval_json": str(interval_json),
            "stack_png": str(stack_png),
            "stack_npz": str(stack_npz),
            "mask_mode": str(args.mask_mode),
            "interpolation": str(args.interpolation),
            "cutout_size": int(args.cutout_size),
            "target_box_size": int(args.target_box_size),
            "aperture_radius_pix": float(args.aperture_radius),
            "n_cadences_total": int(len(keep)),
            "n_cadences_kept": int(np.count_nonzero(keep)),
            "n_intervals": int(len(interval_records)),
            "reference_center_x": float(center[0]),
            "reference_center_y": float(center[1]),
            "reference_image_median": float(np.nanmedian(diff["reference_image"])),
            "differential_stack_peak": float(np.nanmax(stack_result["stack"])),
            "differential_stack_min": float(np.nanmin(stack_result["stack"])),
            "snr_map_peak": float(np.nanmax(snr_map)),
            "support_peak": float(np.nanmax(stack_result["support_map"])),
            "mask_summary": mask_summary,
            "aperture_summary": aperture_summary,
        },
        summary_json,
    )

    print(stack_png)
    print(summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
