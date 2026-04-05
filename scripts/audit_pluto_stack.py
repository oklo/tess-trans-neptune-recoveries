#!/usr/bin/env python3
"""Audit the frozen Pluto stack-alignment path on a bounded real-data bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kbo_tess.ephem import load_ephemeris_csv  # noqa: E402
from kbo_tess.geometry import evaluate_geometry_consistency  # noqa: E402
from kbo_tess.recovery import (  # noqa: E402
    align_ephemeris_to_cube_bundle,
    filter_cube_bundle,
    load_cube_npz,
    resolve_track_coordinates,
    run_known_path_recovery,
)
from kbo_tess.stack_audit import (  # noqa: E402
    build_stack_variant,
    evaluate_frame_offset_photometry,
    measure_local_centroids,
    plot_centroid_residuals_and_offset_preferences,
    plot_stack_audit_summary,
    run_real_bundle_injection_check,
    select_brightest_frames_by_track_photometry,
    summarize_offset_preference_counts,
)
from kbo_tess.shiftstack import search_local_track_offsets  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded stack-alignment audit on a real Pluto bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Pluto", help="Target designation.")
    parser.add_argument("--sector", type=int, default=92, help="TESS sector.")
    parser.add_argument("--camera", type=int, default=2, help="TESS camera.")
    parser.add_argument("--ccd", type=int, default=1, help="TESS CCD.")
    parser.add_argument(
        "--ephem-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_calibration" / "pluto_all_detector_inventory.csv",
        help="Normalized ephemeris CSV used to derive the frozen bundle.",
    )
    parser.add_argument(
        "--cube-npz",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_calibration" / "primary" / "pluto_sector92_cam2_ccd1_cube_bundle.npz",
        help="Frozen real-data cube bundle NPZ.",
    )
    parser.add_argument("--subset-size", type=int, default=32, help="Number of bright cadences to use for centroid diagnostics.")
    parser.add_argument("--track-x-offset", type=float, default=-0.5, help="Small constant x-offset to compare against the nominal track.")
    parser.add_argument("--track-y-offset", type=float, default=0.0, help="Small constant y-offset to compare against the nominal track.")
    parser.add_argument("--offset-grid-radius", type=float, default=1.0, help="Half-width of the subset local dx/dy search.")
    parser.add_argument("--offset-grid-step", type=float, default=0.5, help="Step size of the subset local dx/dy search.")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "pluto_stack_audit", help="Directory for derived audit products.")
    return parser


def _prefix(args: argparse.Namespace) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", args.target.lower()).strip("_")
    return f"{slug}_sector{args.sector}_cam{args.camera}_ccd{args.ccd}"


def _subset_bundle(bundle: dict[str, object], indices: np.ndarray) -> dict[str, object]:
    subset: dict[str, object] = {}
    n_frames = int(np.asarray(bundle["cube"]).shape[0])
    for key, value in bundle.items():
        if key == "metadata":
            continue
        if hasattr(value, "shape"):
            array = np.asarray(value)
            if array.ndim >= 1 and array.shape[0] == n_frames:
                subset[key] = array[indices].copy()
                continue
        subset[key] = value
    subset["metadata"] = dict(bundle.get("metadata", {}))
    subset["metadata"]["subset_indices"] = [int(index) for index in indices]
    return subset


def _top_shift_examples(shift_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    if not shift_rows:
        return []
    picks = [0, len(shift_rows) // 2, len(shift_rows) - 1]
    return [shift_rows[index] for index in sorted(set(picks))]


def main() -> int:
    args = build_parser().parse_args()
    prefix = _prefix(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    ephem = load_ephemeris_csv(args.ephem_csv, target=args.target, require_exact_tdb=True)
    ephem = ephem[
        (ephem["sector"].astype(int) == int(args.sector))
        & (ephem["camera"].astype(int) == int(args.camera))
        & (ephem["ccd"].astype(int) == int(args.ccd))
    ].reset_index(drop=True)
    if ephem.empty:
        raise ValueError(
            f"No ephemeris rows found for {args.target} sector={args.sector} camera={args.camera} ccd={args.ccd}."
        )

    bundle = filter_cube_bundle(load_cube_npz(args.cube_npz))
    cube = np.asarray(bundle["cube"], dtype=float)
    aligned = align_ephemeris_to_cube_bundle(ephem, bundle)
    x, y, track_coordinate_system = resolve_track_coordinates(aligned)
    geometry = evaluate_geometry_consistency(ephem, bundle)

    subset_indices = select_brightest_frames_by_track_photometry(cube, x, y, subset_size=args.subset_size)
    subset_bundle = _subset_bundle(bundle, subset_indices)
    subset_cube = np.asarray(subset_bundle["cube"], dtype=float)
    subset_x = np.asarray(x, dtype=float)[subset_indices]
    subset_y = np.asarray(y, dtype=float)[subset_indices]
    subset_reference = (float(np.nanmedian(subset_x)), float(np.nanmedian(subset_y)))

    centroid_table = measure_local_centroids(subset_cube, subset_x, subset_y)
    centroid_good = centroid_table["valid_centroid"].astype(bool).to_numpy()
    centroid_x = centroid_table.loc[centroid_good, "measured_x"].astype(float).to_numpy()
    centroid_y = centroid_table.loc[centroid_good, "measured_y"].astype(float).to_numpy()

    dx_grid = np.arange(-args.offset_grid_radius, args.offset_grid_radius + 0.5 * args.offset_grid_step, args.offset_grid_step)
    dy_grid = np.arange(-args.offset_grid_radius, args.offset_grid_radius + 0.5 * args.offset_grid_step, args.offset_grid_step)
    subset_search = search_local_track_offsets(
        subset_cube,
        subset_x,
        subset_y,
        dx_grid=dx_grid,
        dy_grid=dy_grid,
        interpolation="linear",
        stack_method="mean",
        score_method="aperture_snr",
        reference=subset_reference,
        score_position=subset_reference,
    )

    variants = {
        "nominal": build_stack_variant(
            subset_cube,
            subset_x,
            subset_y,
            reference=subset_reference,
            interpolation="linear",
            ranking_position=subset_reference,
        ),
        "x_minus_half": build_stack_variant(
            subset_cube,
            subset_x + float(args.track_x_offset),
            subset_y + float(args.track_y_offset),
            reference=subset_reference,
            interpolation="linear",
            ranking_position=subset_reference,
        ),
        "centroid": build_stack_variant(
            subset_cube[centroid_good],
            centroid_x,
            centroid_y,
            reference=subset_reference,
            interpolation="linear",
            ranking_position=subset_reference,
        ),
        "integer_nominal": build_stack_variant(
            subset_cube,
            subset_x,
            subset_y,
            reference=(float(round(subset_reference[0])), float(round(subset_reference[1]))),
            interpolation="nearest",
            ranking_position=(float(round(subset_reference[0])), float(round(subset_reference[1]))),
            integer_shifts=True,
        ),
    }

    full_offset_photometry = evaluate_frame_offset_photometry(
        cube,
        x,
        y,
        offsets=[
            (0.0, 0.0),
            (-0.5, 0.0),
            (0.5, 0.0),
            (0.0, -0.5),
            (0.0, 0.5),
            (-0.5, -0.5),
            (-0.5, 0.5),
            (0.5, -0.5),
            (0.5, 0.5),
        ],
    )
    full_offset_summary = summarize_offset_preference_counts(full_offset_photometry)

    injection_nominal = run_real_bundle_injection_check(
        bundle,
        reference=subset_reference,
        subset_indices=subset_indices,
        amplitude=20.0,
        sigma=0.8,
        injected_dx=0.0,
        injected_dy=0.0,
    )
    injection_offset = run_real_bundle_injection_check(
        bundle,
        reference=subset_reference,
        subset_indices=subset_indices,
        amplitude=20.0,
        sigma=0.8,
        injected_dx=0.5,
        injected_dy=0.0,
    )

    full_recovery = run_known_path_recovery(
        ephem,
        bundle,
        background_method="none",
        interpolation="linear",
        offset_grid_radius=args.offset_grid_radius,
        offset_grid_step=args.offset_grid_step,
        ranking_score_method="aperture_snr",
    )
    shifted_full = build_stack_variant(
        cube,
        x + float(args.track_x_offset),
        y + float(args.track_y_offset),
        reference=(
            float(full_recovery["summary"]["recovered_reference_pixel"]["x"]),
            float(full_recovery["summary"]["recovered_reference_pixel"]["y"]),
        ),
        interpolation="linear",
        ranking_position=(
            float(full_recovery["summary"]["recovered_reference_pixel"]["x"]),
            float(full_recovery["summary"]["recovered_reference_pixel"]["y"]),
        ),
    )

    centroid_residual_x = centroid_table.loc[centroid_good, "residual_x"].astype(float)
    centroid_residual_y = centroid_table.loc[centroid_good, "residual_y"].astype(float)
    centroid_summary = {
        "n_subset_frames": int(len(subset_indices)),
        "n_valid_centroids": int(centroid_good.sum()),
        "median_residual_x": float(np.nanmedian(centroid_residual_x)),
        "median_residual_y": float(np.nanmedian(centroid_residual_y)),
        "mean_residual_x": float(np.nanmean(centroid_residual_x)),
        "mean_residual_y": float(np.nanmean(centroid_residual_y)),
        "std_residual_x": float(np.nanstd(centroid_residual_x)),
        "std_residual_y": float(np.nanstd(centroid_residual_y)),
    }

    variant_summary = []
    for name, variant in variants.items():
        variant_summary.append(
            {
                "variant": name,
                "ranking_score": float(variant["ranking_score"]),
                "aperture_snr": float(variant["scores"]["aperture_snr"]),
                "peak_x": float(variant["peak_location"]["x"]),
                "peak_y": float(variant["peak_location"]["y"]),
                "peak_value": float(variant["peak_location"]["value"]),
                "peak_aperture_snr": float(variant["peak_location"]["peak_aperture_snr"]),
                "support_at_reference": float(
                    variant["support_map"][int(round(subset_reference[1])), int(round(subset_reference[0]))]
                ),
                "max_support": float(np.nanmax(variant["support_map"])),
            }
        )

    summary = {
        "target": args.target,
        "sector": int(args.sector),
        "camera": int(args.camera),
        "ccd": int(args.ccd),
        "frozen_bundle_path": str(args.cube_npz.resolve()),
        "ephem_csv_path": str(args.ephem_csv.resolve()),
        "track_coordinate_system": track_coordinate_system,
        "geometry_check_passed": bool(geometry["summary"]["geometry_check_passed"]),
        "geometry_corner_median_px": float(geometry["summary"]["median_geometry_residual_corner_px"]),
        "geometry_corner_max_px": float(geometry["summary"]["max_geometry_residual_corner_px"]),
        "subset_indices": [int(index) for index in subset_indices],
        "subset_reference_pixel": {"x": float(subset_reference[0]), "y": float(subset_reference[1])},
        "subset_best_local_offset": {
            "dx": float(subset_search["best"]["dx"]),
            "dy": float(subset_search["best"]["dy"]),
            "score": float(subset_search["best"]["score"]),
        },
        "centroid_summary": centroid_summary,
        "full_recovery_nominal_score": float(full_recovery["summary"]["nominal_score"]),
        "full_recovery_best_dx": float(full_recovery["summary"]["best_dx"]),
        "full_recovery_best_dy": float(full_recovery["summary"]["best_dy"]),
        "full_recovery_best_score": float(full_recovery["summary"]["best_score"]),
        "full_track_offset_score": float(shifted_full["ranking_score"]),
        "track_x_offset_tested": float(args.track_x_offset),
        "track_y_offset_tested": float(args.track_y_offset),
        "variant_summary": variant_summary,
        "frame_offset_preference_summary": full_offset_summary,
        "nominal_shift_examples": _top_shift_examples(variants["nominal"]["shift_rows"]),
        "integer_shift_examples": _top_shift_examples(variants["integer_nominal"]["shift_rows"]),
        "injection_nominal_score": float(injection_nominal["ranking_score"]),
        "injection_nominal_peak": dict(injection_nominal["peak_location"]),
        "injection_offset_score": float(injection_offset["ranking_score"]),
        "injection_offset_peak": dict(injection_offset["peak_location"]),
    }

    summary_path = args.outdir / f"{prefix}_stack_audit_summary.json"
    variant_csv_path = args.outdir / f"{prefix}_stack_audit_variants.csv"
    centroid_csv_path = args.outdir / f"{prefix}_subset_centroid_residuals.csv"
    offset_csv_path = args.outdir / f"{prefix}_frame_offset_photometry.csv"
    stack_png_path = args.outdir / f"{prefix}_stack_audit.png"
    residual_png_path = args.outdir / f"{prefix}_stack_offsets.png"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    pd.DataFrame(variant_summary).to_csv(variant_csv_path, index=False)
    centroid_table.to_csv(centroid_csv_path, index=False)
    full_offset_photometry.to_csv(offset_csv_path, index=False)
    plot_stack_audit_summary(subset_cube[len(subset_cube) // 2], subset_reference, variants, stack_png_path)
    plot_centroid_residuals_and_offset_preferences(centroid_table, full_offset_summary, residual_png_path)

    print("Stack audit summary:")
    print(f"  target: {args.target}")
    print(f"  frozen_bundle: {args.cube_npz}")
    print(f"  geometry_passed: {geometry['summary']['geometry_check_passed']}")
    print(
        "  centroid median residual (x,y): "
        f"({centroid_summary['median_residual_x']:.3f}, {centroid_summary['median_residual_y']:.3f})"
    )
    print(
        "  subset scores: "
        f"nominal={variants['nominal']['ranking_score']:.3f}, "
        f"x_offset={variants['x_minus_half']['ranking_score']:.3f}, "
        f"centroid={variants['centroid']['ranking_score']:.3f}"
    )
    print(
        "  full bundle scores: "
        f"nominal={full_recovery['summary']['nominal_score']:.3f}, "
        f"best=({full_recovery['summary']['best_dx']:.1f},{full_recovery['summary']['best_dy']:.1f}) "
        f"{full_recovery['summary']['best_score']:.3f}, "
        f"x_offset={shifted_full['ranking_score']:.3f}"
    )
    print(f"  summary_json: {summary_path}")
    print(f"  variant_csv: {variant_csv_path}")
    print(f"  stack_png: {stack_png_path}")
    print(f"  offsets_png: {residual_png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
