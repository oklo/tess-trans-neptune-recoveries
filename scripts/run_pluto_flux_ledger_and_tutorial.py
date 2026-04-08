#!/usr/bin/env python3
"""Run a transparent Pluto flux-ledger audit on the frozen bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kbo_tess.flux_ledger import (  # noqa: E402
    compute_box_aperture_time_series,
    compute_pixel_residency_map,
    extract_box_aperture_ledger,
    extract_pixel_time_series,
    plot_aperture_ledger_waterfall,
    plot_flux_capture_comparison,
    plot_pixel_residency_maps,
    plot_selected_pixel_time_series,
    select_flux_ledger_sample_indices,
    select_representative_pixels,
    summarize_flux_capture,
    write_json,
)
from kbo_tess.photometry import run_moving_aperture_photometry  # noqa: E402
from kbo_tess.recovery import filter_cube_bundle, load_cube_npz  # noqa: E402
from kbo_tess.stack_audit import build_stack_variant  # noqa: E402
from kbo_tess.track_calibration import (  # noqa: E402
    evaluate_track_photometry,
    plot_offset_grid_heatmaps,
    run_offset_grid_metrics,
)
from kbo_tess.tutorial_benchmark import (  # noqa: E402
    attempt_docs_quickstart_benchmark,
    run_custom_ephem_high_level_path,
    write_status_plot,
)
from kbo_tess.tess_loader import subset_ephemeris_for_build  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a transparent Pluto flux-ledger and tutorial benchmark on the frozen bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Pluto", help="Target designation.")
    parser.add_argument("--sector", type=int, default=92, help="TESS sector.")
    parser.add_argument("--camera", type=int, default=2, help="TESS camera.")
    parser.add_argument("--ccd", type=int, default=1, help="TESS CCD.")
    parser.add_argument(
        "--cube-npz",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_calibration" / "primary" / "pluto_sector92_cam2_ccd1_cube_bundle.npz",
        help="Frozen Pluto cube bundle.",
    )
    parser.add_argument(
        "--ephem-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_calibration" / "pluto_all_detector_inventory.csv",
        help="Normalized ephemeris CSV used to build the Pluto bundle.",
    )
    parser.add_argument(
        "--x-calibration-summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_x_calibration" / "pluto_sector92_cam2_ccd1_x_calibration_summary.json",
        help="X-calibration summary used to read the current calibrated-x value.",
    )
    parser.add_argument(
        "--background-method",
        choices=("none", "median", "savgol"),
        default="none",
        help="Temporal background handling for the custom-path comparison.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "linear"),
        default="linear",
        help="Shift interpolation mode for stack comparisons.",
    )
    parser.add_argument(
        "--dx-min",
        type=float,
        default=-1.0,
        help="Minimum dx for the full 2-D track map.",
    )
    parser.add_argument(
        "--dx-max",
        type=float,
        default=1.0,
        help="Maximum dx for the full 2-D track map.",
    )
    parser.add_argument(
        "--dy-min",
        type=float,
        default=-1.0,
        help="Minimum dy for the full 2-D track map.",
    )
    parser.add_argument(
        "--dy-max",
        type=float,
        default=1.0,
        help="Maximum dy for the full 2-D track map.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.25,
        help="Grid step for the full 2-D track map.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_flux_ledger_tutorial",
        help="Directory for derived Pluto transparency products.",
    )
    return parser


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_local_track(bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    for x_key, y_key in (("local_x", "local_y"), ("bundle_local_x", "bundle_local_y")):
        if x_key in bundle and y_key in bundle:
            return np.asarray(bundle[x_key], dtype=float), np.asarray(bundle[y_key], dtype=float)
    raise KeyError("Cube bundle is missing local_x/local_y track coordinates.")


def _growth_checkpoints(n_frames: int) -> list[int]:
    counts = {max(2, int(round(n_frames * fraction))) for fraction in (0.25, 0.5, 0.75, 1.0)}
    return sorted(counts)


def _build_stack_growth_table(
    cube: np.ndarray,
    nominal_x: np.ndarray,
    nominal_y: np.ndarray,
    calibrated_x: np.ndarray,
    reference: tuple[float, float],
    interpolation: str,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for n_frames in _growth_checkpoints(len(cube)):
        nominal_variant = build_stack_variant(
            cube[:n_frames],
            nominal_x[:n_frames],
            nominal_y[:n_frames],
            reference=reference,
            interpolation=interpolation,
            ranking_position=reference,
        )
        calibrated_variant = build_stack_variant(
            cube[:n_frames],
            calibrated_x[:n_frames],
            nominal_y[:n_frames],
            reference=reference,
            interpolation=interpolation,
            ranking_position=reference,
        )
        rows.append(
            {
                "n_frames": float(n_frames),
                "nominal_stack_score": float(nominal_variant["ranking_score"]),
                "calibrated_stack_score": float(calibrated_variant["ranking_score"]),
            }
        )
    return pd.DataFrame(rows)


def _series_summary(table: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_net_flux": float(table["net_flux"].mean()),
        "median_net_flux": float(table["net_flux"].median()),
        "max_net_flux": float(table["net_flux"].max()),
        "mean_aperture_snr": float(table["aperture_snr"].mean()),
        "median_aperture_snr": float(table["aperture_snr"].median()),
        "max_aperture_snr": float(table["aperture_snr"].max()),
    }


def _on_axis_answer(grid_table: pd.DataFrame, score_column: str) -> dict[str, Any]:
    best_row = grid_table.loc[grid_table[score_column].idxmax()]
    on_axis_rows = grid_table[np.isclose(grid_table["dy"].astype(float), 0.0)]
    best_on_axis = on_axis_rows.loc[on_axis_rows[score_column].idxmax()]
    return {
        "best_dx": float(best_row["dx"]),
        "best_dy": float(best_row["dy"]),
        "best_score": float(best_row[score_column]),
        "best_on_axis_dx": float(best_on_axis["dx"]),
        "best_on_axis_score": float(best_on_axis[score_column]),
        "best_is_on_axis": bool(abs(float(best_row["dy"])) <= 1e-9),
        "off_axis_advantage": float(best_row[score_column] - best_on_axis[score_column]),
    }


def main() -> int:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_slug(args.target)}_sector{args.sector}_cam{args.camera}_ccd{args.ccd}"

    xcal_summary = _load_json(args.x_calibration_summary_json)
    calibrated_x_offset = float(xcal_summary["full_best_x_offset"])
    frozen_bundle_path = Path(str(xcal_summary["frozen_bundle_path"]))
    if frozen_bundle_path.resolve() != args.cube_npz.resolve():
        raise ValueError(
            "The requested bundle does not match the x-calibration frozen bundle. "
            f"xcal={frozen_bundle_path} requested={args.cube_npz}"
        )

    raw_bundle = load_cube_npz(args.cube_npz)
    filtered_bundle = filter_cube_bundle(raw_bundle)
    raw_cube = np.asarray(raw_bundle["cube"], dtype=float)
    filtered_cube = np.asarray(filtered_bundle["cube"], dtype=float)
    raw_time = np.asarray(raw_bundle["time_tdb_jd_minus_2457000"], dtype=float)
    filtered_time = np.asarray(filtered_bundle["time_tdb_jd_minus_2457000"], dtype=float)
    raw_nominal_x, raw_nominal_y = _extract_local_track(raw_bundle)
    filtered_nominal_x, filtered_nominal_y = _extract_local_track(filtered_bundle)
    raw_calibrated_x = raw_nominal_x + calibrated_x_offset
    filtered_calibrated_x = filtered_nominal_x + calibrated_x_offset
    reference = (float(filtered_nominal_x[0]), float(filtered_nominal_y[0]))

    quality_mask = np.asarray(raw_bundle.get("quality_mask", np.ones(raw_cube.shape[0], dtype=bool)), dtype=bool)

    selected_pixels = select_representative_pixels(
        raw_calibrated_x,
        raw_nominal_y,
        shape=raw_cube.shape[1:],
        occupancy_radius=0.5,
    )
    pixel_table = extract_pixel_time_series(
        raw_cube,
        raw_time,
        selected_pixels=selected_pixels,
        nominal_x=raw_nominal_x,
        nominal_y=raw_nominal_y,
        calibrated_x=raw_calibrated_x,
        calibrated_y=raw_nominal_y,
        quality_mask=quality_mask,
    )
    pixel_table.to_csv(args.outdir / f"{prefix}_selected_pixel_timeseries.csv", index=False)
    selected_pixels.to_csv(args.outdir / f"{prefix}_selected_pixels.csv", index=False)
    nominal_residency = compute_pixel_residency_map(raw_cube.shape[1:], raw_nominal_x, raw_nominal_y, radius=0.5)
    calibrated_residency = compute_pixel_residency_map(raw_cube.shape[1:], raw_calibrated_x, raw_nominal_y, radius=0.5)

    direct_nominal_table = evaluate_track_photometry(filtered_cube, filtered_nominal_x, filtered_nominal_y)
    direct_calibrated_table = evaluate_track_photometry(filtered_cube, filtered_calibrated_x, filtered_nominal_y)
    direct_nominal_table["track_label"] = "direct_nominal"
    direct_calibrated_table["track_label"] = "direct_calibrated_x"
    direct_nominal_table.to_csv(args.outdir / f"{prefix}_direct_nominal.csv", index=False)
    direct_calibrated_table.to_csv(args.outdir / f"{prefix}_direct_calibrated_x.csv", index=False)

    box3_nominal = compute_box_aperture_time_series(
        filtered_cube,
        filtered_time,
        filtered_nominal_x,
        filtered_nominal_y,
        half_size=1,
        track_label="box3_nominal",
    )
    box3_calibrated = compute_box_aperture_time_series(
        filtered_cube,
        filtered_time,
        filtered_calibrated_x,
        filtered_nominal_y,
        half_size=1,
        track_label="box3_calibrated_x",
    )
    box5_nominal = compute_box_aperture_time_series(
        filtered_cube,
        filtered_time,
        filtered_nominal_x,
        filtered_nominal_y,
        half_size=2,
        track_label="box5_nominal",
    )
    box5_calibrated = compute_box_aperture_time_series(
        filtered_cube,
        filtered_time,
        filtered_calibrated_x,
        filtered_nominal_y,
        half_size=2,
        track_label="box5_calibrated_x",
    )

    pd.concat([box3_nominal, box3_calibrated, box5_nominal, box5_calibrated], ignore_index=True).to_csv(
        args.outdir / f"{prefix}_simple_aperture_time_series.csv",
        index=False,
    )

    sample_indices = select_flux_ledger_sample_indices(direct_nominal_table, n_uniform=6, n_bright=6)
    nominal_ledger = extract_box_aperture_ledger(
        filtered_cube,
        filtered_time,
        filtered_nominal_x,
        filtered_nominal_y,
        sample_indices=sample_indices,
        half_size=1,
        track_label="nominal",
    )
    calibrated_ledger = extract_box_aperture_ledger(
        filtered_cube,
        filtered_time,
        filtered_calibrated_x,
        filtered_nominal_y,
        sample_indices=sample_indices,
        half_size=1,
        track_label="calibrated_x",
    )
    ledger_rows = pd.concat([nominal_ledger["ledger_rows"], calibrated_ledger["ledger_rows"]], ignore_index=True)
    ledger_rows.to_csv(args.outdir / f"{prefix}_aperture_ledger.csv", index=False)

    flux_capture_summary = summarize_flux_capture(
        direct_nominal_table,
        direct_calibrated_table,
        box3_nominal,
        box3_calibrated,
        box5_nominal,
        box5_calibrated,
    )

    dx_grid = np.arange(args.dx_min, args.dx_max + 0.5 * args.grid_step, args.grid_step, dtype=float)
    dy_grid = np.arange(args.dy_min, args.dy_max + 0.5 * args.grid_step, args.grid_step, dtype=float)
    offset_grid = run_offset_grid_metrics(
        filtered_cube,
        filtered_nominal_x,
        filtered_nominal_y,
        dx_grid=dx_grid,
        dy_grid=dy_grid,
        reference=reference,
        interpolation=args.interpolation,
    )
    offset_grid["table"].to_csv(args.outdir / f"{prefix}_dx_dy_grid.csv", index=False)
    direct_axis_answer = _on_axis_answer(offset_grid["table"], "direct_score")
    stack_axis_answer = _on_axis_answer(offset_grid["table"], "stack_ranking_score")

    stack_growth = _build_stack_growth_table(
        filtered_cube,
        filtered_nominal_x,
        filtered_nominal_y,
        filtered_calibrated_x,
        reference=reference,
        interpolation=args.interpolation,
    )
    stack_growth.to_csv(args.outdir / f"{prefix}_stack_growth.csv", index=False)

    nominal_variant = build_stack_variant(
        filtered_cube,
        filtered_nominal_x,
        filtered_nominal_y,
        reference=reference,
        interpolation=args.interpolation,
        ranking_position=reference,
    )
    calibrated_variant = build_stack_variant(
        filtered_cube,
        filtered_calibrated_x,
        filtered_nominal_y,
        reference=reference,
        interpolation=args.interpolation,
        ranking_position=reference,
    )

    pixel_plot = plot_selected_pixel_time_series(pixel_table, args.outdir / f"{prefix}_pixel_timeseries.png")
    residency_plot = plot_pixel_residency_maps(
        nominal_residency,
        calibrated_residency,
        raw_nominal_x,
        raw_nominal_y,
        raw_calibrated_x,
        raw_nominal_y,
        args.outdir / f"{prefix}_pixel_residency.png",
    )
    ledger_plot = plot_aperture_ledger_waterfall(ledger_rows, args.outdir / f"{prefix}_aperture_ledger_waterfall.png")
    comparison_plot = plot_flux_capture_comparison(
        direct_nominal_table,
        direct_calibrated_table,
        box3_nominal,
        box3_calibrated,
        box5_nominal,
        box5_calibrated,
        stack_growth,
        args.outdir / f"{prefix}_signal_comparison.png",
    )
    heatmap_plot = plot_offset_grid_heatmaps(
        offset_grid,
        args.outdir / f"{prefix}_dx_dy_heatmaps.png",
        nominal_point=(0.0, 0.0),
        calibrated_point=(calibrated_x_offset, 0.0),
    )

    moving_phot_result = run_moving_aperture_photometry(
        pd.DataFrame(
            {
                "time_tdb_jd_minus_2457000": filtered_time,
                "x": filtered_bundle.get("detector_x", np.full_like(filtered_time, np.nan)),
                "y": filtered_bundle.get("detector_y", np.full_like(filtered_time, np.nan)),
                "sector": np.full_like(filtered_time, args.sector),
                "camera": np.full_like(filtered_time, args.camera),
                "ccd": np.full_like(filtered_time, args.ccd),
                "target": [args.target] * len(filtered_time),
                "time_scale": ["tdb"] * len(filtered_time),
                "time_reference": ["barycentric"] * len(filtered_time),
                "bundle_local_x": filtered_nominal_x,
                "bundle_local_y": filtered_nominal_y,
            }
        ),
        filtered_bundle,
        background_method=args.background_method,
    )
    moving_phot_result["lightcurve"].to_csv(args.outdir / f"{prefix}_current_moving_aperture.csv", index=False)

    ephem_subset = subset_ephemeris_for_build(
        args.ephem_csv,
        target=args.target,
        time_start=float(raw_time.min()) - 0.1,
        time_stop=float(raw_time.max()) + 0.1,
    )

    tutorial_docs = attempt_docs_quickstart_benchmark(outdir=args.outdir)
    if tutorial_docs.get("status") != "success":
        tutorial_docs["plot_path"] = str(
            write_status_plot(
                "Official docs quickstart benchmark unavailable",
                tutorial_docs.get("error_message", "Unknown tutorial benchmark failure."),
                args.outdir / f"{prefix}_official_tutorial_benchmark_status.png",
            )
        )
    tutorial_pluto: dict[str, Any]
    try:
        tutorial_pluto = run_custom_ephem_high_level_path(
            ephem_subset,
            target=args.target,
            shape=tuple(int(value) for value in filtered_cube.shape[1:]),
            barycentric=True,
            bg_method="rolling",
            outdir=args.outdir,
        )
        for method_name, table in tutorial_pluto["lightcurve_tables"].items():
            table.to_csv(args.outdir / f"{prefix}_official_{method_name}_lightcurve.csv", index=False)
        tutorial_pluto_summary = {
            "status": "success",
            "summary": tutorial_pluto["summary"],
            "plot_path": tutorial_pluto["plot_path"],
            "lightcurve_methods": list(tutorial_pluto["lightcurve_tables"].keys()),
        }
    except Exception as exc:
        tutorial_pluto_summary = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    if tutorial_pluto_summary.get("status") != "success":
        tutorial_pluto_summary["plot_path"] = str(
            write_status_plot(
                "Pluto official/simple path unavailable",
                tutorial_pluto_summary.get("error_message", "Unknown Pluto official-path failure."),
                args.outdir / f"{prefix}_official_simple_pluto_status.png",
            )
        )

    summary = {
        "target": args.target,
        "sector": int(args.sector),
        "camera": int(args.camera),
        "ccd": int(args.ccd),
        "frozen_bundle_path": str(args.cube_npz.resolve()),
        "background_method": args.background_method,
        "interpolation": args.interpolation,
        "calibrated_x_offset": float(calibrated_x_offset),
        "selected_pixels": selected_pixels.to_dict(orient="records"),
        "selected_pixel_summary": {
            "n_selected_pixels": int(selected_pixels.shape[0]),
            "n_total_cadences": int(raw_cube.shape[0]),
            "n_quality_kept": int(quality_mask.sum()),
        },
        "simple_aperture_summary": {
            "sample_indices": [int(value) for value in sample_indices],
            "direct_nominal": _series_summary(direct_nominal_table),
            "direct_calibrated_x": _series_summary(direct_calibrated_table),
            "box3_nominal": _series_summary(box3_nominal),
            "box3_calibrated_x": _series_summary(box3_calibrated),
            "box5_nominal": _series_summary(box5_nominal),
            "box5_calibrated_x": _series_summary(box5_calibrated),
            "flux_capture": flux_capture_summary,
        },
        "stack_summary": {
            "nominal_score": float(nominal_variant["ranking_score"]),
            "calibrated_x_score": float(calibrated_variant["ranking_score"]),
            "nominal_peak": nominal_variant["peak_location"],
            "calibrated_x_peak": calibrated_variant["peak_location"],
        },
        "dx_dy_grid": {
            "dx_values": [float(value) for value in dx_grid],
            "dy_values": [float(value) for value in dy_grid],
            "best_direct_row": offset_grid["best_direct_row"],
            "best_stack_row": offset_grid["best_stack_row"],
            "direct_axis_answer": direct_axis_answer,
            "stack_axis_answer": stack_axis_answer,
        },
        "tutorial_benchmark": {
            "docs_quickstart": tutorial_docs,
            "pluto_high_level": tutorial_pluto_summary,
        },
        "plots": {
            "pixel_timeseries": str(pixel_plot),
            "pixel_residency": str(residency_plot),
            "aperture_ledger_waterfall": str(ledger_plot),
            "signal_comparison": str(comparison_plot),
            "dx_dy_heatmaps": str(heatmap_plot),
        },
    }
    write_json(summary, args.outdir / f"{prefix}_flux_ledger_summary.json")

    print(f"Frozen bundle: {args.cube_npz}")
    print(f"Calibrated x offset: {calibrated_x_offset:+.2f} px")
    print(
        "Direct best dx/dy = "
        f"({direct_axis_answer['best_dx']:+.2f}, {direct_axis_answer['best_dy']:+.2f}), "
        f"stack best dx/dy = ({stack_axis_answer['best_dx']:+.2f}, {stack_axis_answer['best_dy']:+.2f})"
    )
    print(
        "Direct mean net flux ratios: "
        f"nominal/direct-to-box5={flux_capture_summary['direct_nominal_to_box5_nominal_ratio']:.3f}, "
        f"calibrated/direct-to-box5={flux_capture_summary['direct_calibrated_to_box5_calibrated_ratio']:.3f}"
    )
    print(f"Docs quickstart status: {tutorial_docs['status']}")
    print(f"Pluto high-level official-path status: {tutorial_pluto_summary['status']}")
    print(f"Wrote products to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
