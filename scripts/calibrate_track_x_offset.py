#!/usr/bin/env python3
"""Run a narrow constant-x track calibration on a frozen detector-local bundle."""

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

from kbo_tess.background import subtract_background  # noqa: E402
from kbo_tess.ephem import load_ephemeris_csv  # noqa: E402
from kbo_tess.recovery import (  # noqa: E402
    align_ephemeris_to_cube_bundle,
    filter_cube_bundle,
    load_cube_npz,
    resolve_track_coordinates,
)
from kbo_tess.track_calibration import (  # noqa: E402
    build_nominal_vs_calibrated_growth_curve,
    build_variant_dictionary,
    evaluate_track_photometry,
    measure_local_centroids,
    plot_calibrated_null_controls,
    plot_growth_comparison,
    plot_time_residual_diagnostic,
    plot_x_offset_comparison,
    plot_x_offset_sweep,
    run_centered_null_control_gate,
    run_x_offset_sweep,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a frozen target track with a narrow constant x-offset sweep.",
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
        help="Normalized ephemeris CSV.",
    )
    parser.add_argument(
        "--cube-npz",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_calibration" / "primary" / "pluto_sector92_cam2_ccd1_cube_bundle.npz",
        help="Frozen real-data cube bundle NPZ.",
    )
    parser.add_argument(
        "--audit-summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_stack_audit" / "pluto_sector92_cam2_ccd1_stack_audit_summary.json",
        help="Stack-audit summary JSON containing the canonical 32-frame subset.",
    )
    parser.add_argument(
        "--background-method",
        choices=("none", "median", "savgol"),
        default="none",
        help="Temporal background handling for this calibration pass.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "linear"),
        default="linear",
        help="Shift interpolation mode.",
    )
    parser.add_argument(
        "--offset-grid-radius",
        type=float,
        default=1.0,
        help="Half-width of the calibrated null-control search in pixels.",
    )
    parser.add_argument(
        "--offset-grid-step",
        type=float,
        default=0.5,
        help="Step size of the calibrated null-control search in pixels.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_x_calibration",
        help="Directory for derived calibration outputs.",
    )
    return parser


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _load_audit_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sparse_sample_indices(n_frames: int, step: int = 24) -> np.ndarray:
    indices = np.arange(0, n_frames, step, dtype=int)
    if indices[-1] != n_frames - 1:
        indices = np.unique(np.concatenate([indices, np.asarray([n_frames - 1], dtype=int)]))
    return indices


def main() -> int:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_slug(args.target)}_sector{args.sector}_cam{args.camera}_ccd{args.ccd}"

    audit_summary = _load_audit_summary(args.audit_summary_json)
    frozen_bundle_path = Path(str(audit_summary["frozen_bundle_path"]))
    if frozen_bundle_path.resolve() != args.cube_npz.resolve():
        raise ValueError(
            "The requested cube bundle does not match the stack-audit frozen bundle. "
            f"audit={frozen_bundle_path}, requested={args.cube_npz}"
        )

    x_grid = np.asarray([0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50], dtype=float)
    centroid_informed_x = float(audit_summary["centroid_summary"]["median_residual_x"])

    ephem = load_ephemeris_csv(args.ephem_csv, target=args.target, require_exact_tdb=True)
    ephem = ephem[
        (ephem["sector"].astype(int) == int(args.sector))
        & (ephem["camera"].astype(int) == int(args.camera))
        & (ephem["ccd"].astype(int) == int(args.ccd))
    ].reset_index(drop=True)
    cube_bundle = filter_cube_bundle(load_cube_npz(args.cube_npz))
    cube = np.asarray(cube_bundle["cube"], dtype=float)
    cube_times = np.asarray(cube_bundle["time_tdb_jd_minus_2457000"], dtype=float)
    _, analysis_cube = subtract_background(cube, method=args.background_method)
    aligned = align_ephemeris_to_cube_bundle(ephem, cube_bundle)
    x, y, track_coordinate_system = resolve_track_coordinates(aligned)

    subset_indices = np.asarray(audit_summary["subset_indices"], dtype=int)
    subset_reference = (
        float(audit_summary["subset_reference_pixel"]["x"]),
        float(audit_summary["subset_reference_pixel"]["y"]),
    )
    full_reference = (float(x[0]), float(y[0]))

    subset_sweep = run_x_offset_sweep(
        analysis_cube[subset_indices],
        np.asarray(x, dtype=float)[subset_indices],
        np.asarray(y, dtype=float)[subset_indices],
        x_offsets=x_grid,
        reference=subset_reference,
        interpolation=args.interpolation,
    )
    subset_best_x = float(subset_sweep["best_stack_row"]["x_offset"])
    subset_variants = build_variant_dictionary(
        analysis_cube[subset_indices],
        np.asarray(x, dtype=float)[subset_indices],
        np.asarray(y, dtype=float)[subset_indices],
        reference=subset_reference,
        best_grid_x_offset=subset_best_x,
        centroid_informed_x_offset=centroid_informed_x,
        interpolation=args.interpolation,
    )

    full_sweep = run_x_offset_sweep(
        analysis_cube,
        x,
        y,
        x_offsets=x_grid,
        reference=full_reference,
        interpolation=args.interpolation,
    )
    full_best_x = float(full_sweep["best_stack_row"]["x_offset"])

    dx_grid = np.arange(-args.offset_grid_radius, args.offset_grid_radius + 0.5 * args.offset_grid_step, args.offset_grid_step)
    dy_grid = np.arange(-args.offset_grid_radius, args.offset_grid_radius + 0.5 * args.offset_grid_step, args.offset_grid_step)
    null_gate = run_centered_null_control_gate(
        analysis_cube,
        x,
        y,
        calibrated_x_offset=full_best_x,
        reference=full_reference,
        dx_grid=dx_grid,
        dy_grid=dy_grid,
        interpolation=args.interpolation,
    )
    growth_curve = build_nominal_vs_calibrated_growth_curve(
        analysis_cube,
        x,
        y,
        calibrated_x_offset=full_best_x,
        reference=full_reference,
        interpolation=args.interpolation,
    )

    constant_x_sufficient = bool(null_gate["summary"]["calibrated_minus_best_control"] > 0.0)

    sample_indices = _sparse_sample_indices(len(analysis_cube), step=24)
    subset_centroids = measure_local_centroids(
        analysis_cube[subset_indices],
        np.asarray(x, dtype=float)[subset_indices],
        np.asarray(y, dtype=float)[subset_indices],
    )
    sample_centroids = measure_local_centroids(
        analysis_cube[sample_indices],
        np.asarray(x, dtype=float)[sample_indices],
        np.asarray(y, dtype=float)[sample_indices],
    )

    subset_json = args.outdir / f"{prefix}_subset_x_sweep.json"
    subset_csv = args.outdir / f"{prefix}_subset_x_sweep.csv"
    subset_png = args.outdir / f"{prefix}_subset_x_sweep.png"
    subset_compare_png = args.outdir / f"{prefix}_subset_x_comparison.png"
    full_json = args.outdir / f"{prefix}_full_x_sweep.json"
    full_csv = args.outdir / f"{prefix}_full_x_sweep.csv"
    full_png = args.outdir / f"{prefix}_full_x_sweep.png"
    null_json = args.outdir / f"{prefix}_calibrated_null_controls.json"
    null_csv = args.outdir / f"{prefix}_calibrated_null_controls.csv"
    null_png = args.outdir / f"{prefix}_calibrated_null_controls.png"
    growth_csv = args.outdir / f"{prefix}_nominal_vs_calibrated_growth.csv"
    growth_png = args.outdir / f"{prefix}_nominal_vs_calibrated_growth.png"
    time_png = args.outdir / f"{prefix}_centroid_residuals_vs_time.png"
    summary_json = args.outdir / f"{prefix}_x_calibration_summary.json"

    subset_summary = {
        "target": args.target,
        "frozen_bundle_path": str(args.cube_npz.resolve()),
        "audit_summary_json": str(args.audit_summary_json.resolve()),
        "background_method": args.background_method,
        "interpolation": args.interpolation,
        "subset_indices": subset_indices.tolist(),
        "subset_reference_pixel": {"x": float(subset_reference[0]), "y": float(subset_reference[1])},
        "x_grid": [float(value) for value in x_grid],
        "centroid_informed_x_offset": float(centroid_informed_x),
        "best_stack_x_offset": subset_best_x,
        "best_direct_x_offset": float(subset_sweep["best_direct_row"]["x_offset"]),
        "best_stack_row": {key: float(value) for key, value in subset_sweep["best_stack_row"].items()},
        "best_direct_row": {key: float(value) for key, value in subset_sweep["best_direct_row"].items()},
        "track_coordinate_system": track_coordinate_system,
    }
    full_summary = {
        "target": args.target,
        "frozen_bundle_path": str(args.cube_npz.resolve()),
        "background_method": args.background_method,
        "interpolation": args.interpolation,
        "full_reference_pixel": {"x": float(full_reference[0]), "y": float(full_reference[1])},
        "x_grid": [float(value) for value in x_grid],
        "best_stack_x_offset": full_best_x,
        "best_direct_x_offset": float(full_sweep["best_direct_row"]["x_offset"]),
        "best_stack_row": {key: float(value) for key, value in full_sweep["best_stack_row"].items()},
        "best_direct_row": {key: float(value) for key, value in full_sweep["best_direct_row"].items()},
        "subset_best_x_offset": subset_best_x,
        "subset_and_full_best_agree_within_0p1": bool(abs(subset_best_x - full_best_x) <= 0.10),
    }

    mean_sample_residual_x = float(sample_centroids["residual_x"].median())
    residual_classification = "approximately_constant"
    if abs(mean_sample_residual_x - full_best_x) > 0.15:
        residual_classification = "drifting_or_systematics_dominated"
    elif float(sample_centroids["residual_x"].std()) > 0.35:
        residual_classification = "scatter_dominated"

    note_summary = {
        "target": args.target,
        "sector": int(args.sector),
        "camera": int(args.camera),
        "ccd": int(args.ccd),
        "frozen_bundle_path": str(args.cube_npz.resolve()),
        "audit_summary_json": str(args.audit_summary_json.resolve()),
        "track_coordinate_system": track_coordinate_system,
        "background_method": args.background_method,
        "interpolation": args.interpolation,
        "x_grid": [float(value) for value in x_grid],
        "centroid_informed_x_offset": float(centroid_informed_x),
        "subset_best_x_offset": float(subset_best_x),
        "full_best_x_offset": float(full_best_x),
        "subset_full_best_agree_within_0p1": bool(abs(subset_best_x - full_best_x) <= 0.10),
        "nominal_score": float(null_gate["summary"]["exact_nominal_score"]),
        "calibrated_x_score": float(null_gate["summary"]["calibrated_track_score"]),
        "best_control_score": float(null_gate["summary"]["best_control_score"]),
        "calibrated_minus_best_control": float(null_gate["summary"]["calibrated_minus_best_control"]),
        "growth_final_nominal_score": float(growth_curve.iloc[-1]["nominal_score"]),
        "growth_final_calibrated_score": float(growth_curve.iloc[-1]["calibrated_score"]),
        "subset_centroid_median_residual_x": float(subset_centroids["residual_x"].median()),
        "subset_centroid_median_residual_y": float(subset_centroids["residual_y"].median()),
        "subset_centroid_std_residual_x": float(subset_centroids["residual_x"].std()),
        "subset_centroid_std_residual_y": float(subset_centroids["residual_y"].std()),
        "sample_centroid_median_residual_x": mean_sample_residual_x,
        "sample_centroid_median_residual_y": float(sample_centroids["residual_y"].median()),
        "sample_centroid_std_residual_x": float(sample_centroids["residual_x"].std()),
        "sample_centroid_std_residual_y": float(sample_centroids["residual_y"].std()),
        "constant_x_sufficient": constant_x_sufficient,
        "residual_classification": residual_classification,
    }

    write_json(subset_json, subset_summary)
    subset_sweep["table"].to_csv(subset_csv, index=False)
    plot_x_offset_sweep(subset_sweep, subset_png, title="Bright 32-frame subset")
    plot_x_offset_comparison(
        analysis_cube[subset_indices[len(subset_indices) // 2]],
        subset_reference,
        subset_variants,
        subset_compare_png,
    )

    write_json(full_json, full_summary)
    full_sweep["table"].to_csv(full_csv, index=False)
    plot_x_offset_sweep(full_sweep, full_png, title="Full frozen bundle")

    write_json(null_json, {**null_gate["summary"], "x_grid": [float(value) for value in x_grid]})
    null_gate["results"].to_csv(null_csv, index=False)
    plot_calibrated_null_controls(null_gate, null_png)

    growth_curve.to_csv(growth_csv, index=False)
    plot_growth_comparison(growth_curve, growth_png)
    plot_time_residual_diagnostic(
        cube_times[subset_indices],
        subset_centroids,
        cube_times[sample_indices],
        sample_centroids,
        calibrated_x_offset=full_best_x,
        output_path=time_png,
    )
    write_json(summary_json, note_summary)

    print("Pluto x-calibration summary:")
    print(f"  frozen_bundle: {args.cube_npz}")
    print(f"  subset_best_x_offset: {subset_best_x:.3f}")
    print(f"  full_best_x_offset: {full_best_x:.3f}")
    print(
        "  null-control comparison: "
        f"nominal={null_gate['summary']['exact_nominal_score']:.3f}, "
        f"calibrated={null_gate['summary']['calibrated_track_score']:.3f}, "
        f"best_control={null_gate['summary']['best_control_score']:.3f}"
    )
    print(
        "  cadence growth final: "
        f"nominal={growth_curve.iloc[-1]['nominal_score']:.3f}, "
        f"calibrated={growth_curve.iloc[-1]['calibrated_score']:.3f}"
    )
    print(f"  subset_json: {subset_json}")
    print(f"  full_json: {full_json}")
    print(f"  null_json: {null_json}")
    print(f"  growth_csv: {growth_csv}")
    print(f"  time_png: {time_png}")
    print(f"  summary_json: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
