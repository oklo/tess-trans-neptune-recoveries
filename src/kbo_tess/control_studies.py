"""Lightweight control-study helpers for bounded real-data Sedna analysis."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    matplotlib = None
    plt = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    pd = None

from kbo_tess.injection import inject_source_into_cube_bundle
from kbo_tess.recovery import run_known_path_recovery


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for control studies. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return np


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for control studies. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return pd


def _require_matplotlib() -> Any:
    if plt is None:
        raise ImportError(
            "matplotlib is required for control-study diagnostics. Install project dependencies "
            "with `python -m pip install -e .`."
        )
    return plt


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def flatten_growth_curve(growth_curve: list[dict[str, float]]) -> dict[str, float]:
    """Flatten recovery growth checkpoints into CSV-friendly columns."""

    flat: dict[str, float] = {}
    for row in growth_curve:
        checkpoint = int(round(float(row["n_frames"])))
        flat[f"growth_nominal_{checkpoint}"] = float(row["nominal_ranking_score"])
        flat[f"growth_best_{checkpoint}"] = float(row["best_ranking_score"])
    return flat


def compact_growth_curve(growth_curve: list[dict[str, float]]) -> list[dict[str, float]]:
    """Return just the ranking-score checkpoints from a recovery growth curve."""

    return [
        {
            "n_frames": float(row["n_frames"]),
            "nominal_score": float(row["nominal_ranking_score"]),
            "best_score": float(row["best_ranking_score"]),
        }
        for row in growth_curve
    ]


def summarize_recovery_result(recovery_result: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact summary from a recovery result for reporting tables."""

    numpy = _require_numpy()
    summary = dict(recovery_result["summary"])
    summary["best_offset_radius"] = float(
        numpy.hypot(float(summary["best_dx"]), float(summary["best_dy"]))
    )
    summary["growth_curve"] = compact_growth_curve(recovery_result.get("growth_curve", []))
    summary.update(flatten_growth_curve(recovery_result.get("growth_curve", [])))
    return summary


def run_recovery_suite(
    ephem_df: Any,
    cube_bundle: dict[str, Any],
    background_methods: tuple[str, ...] = ("none",),
    interpolation: str = "linear",
    offset_grid_radius: float = 1.0,
    offset_grid_step: float = 0.5,
    ranking_score_method: str = "aperture_snr",
) -> dict[str, dict[str, Any]]:
    """Run the same known-path recovery configuration for a list of background methods."""

    recoveries: dict[str, dict[str, Any]] = {}
    for background_method in background_methods:
        recoveries[background_method] = run_known_path_recovery(
            ephem_df,
            cube_bundle,
            background_method=background_method,
            interpolation=interpolation,
            offset_grid_radius=offset_grid_radius,
            offset_grid_step=offset_grid_step,
            ranking_score_method=ranking_score_method,
        )
    return recoveries


def build_mode_comparison_summary(
    mode_results: dict[str, dict[str, Any]],
    primary_background: str = "none",
) -> dict[str, Any]:
    """Assemble a side-by-side barycentric/spacecraft comparison summary."""

    numpy = _require_numpy()
    mode_summaries: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for mode_label, result in mode_results.items():
        geometry_summary = dict(result["geometry"]["summary"])
        recoveries = result["recoveries"]
        if primary_background not in recoveries:
            raise KeyError(
                f"Primary background {primary_background!r} not present in recovery suite."
            )
        primary_summary = summarize_recovery_result(recoveries[primary_background])
        background_summaries = {
            name: summarize_recovery_result(recovery_result) for name, recovery_result in recoveries.items()
        }
        mode_summary = {
            "bundle_npz": str(result["bundle_npz"]),
            "geometry": geometry_summary,
            "primary_background": primary_background,
            "primary_recovery": primary_summary,
            "recoveries": background_summaries,
        }
        mode_summaries[mode_label] = mode_summary

        csv_row = {
            "mode": mode_label,
            "bundle_npz": str(result["bundle_npz"]),
            "geometry_passed": bool(geometry_summary["geometry_check_passed"]),
            "n_frames": int(primary_summary["n_frames"]),
            "geometry_corner_median_px": float(geometry_summary["median_geometry_residual_corner_px"]),
            "geometry_corner_max_px": float(geometry_summary["max_geometry_residual_corner_px"]),
            "geometry_poscorr_median_px": float(geometry_summary["median_geometry_residual_poscorr_px"]),
            "geometry_poscorr_max_px": float(geometry_summary["max_geometry_residual_poscorr_px"]),
            "best_dx": float(primary_summary["best_dx"]),
            "best_dy": float(primary_summary["best_dy"]),
            "best_offset_radius": float(primary_summary["best_offset_radius"]),
            "nominal_score": float(primary_summary["nominal_score"]),
            "best_score": float(primary_summary["best_score"]),
        }
        csv_row.update(flatten_growth_curve(result["recoveries"][primary_background]["growth_curve"]))
        csv_rows.append(csv_row)

    geometry_candidates = [
        (label, summary["geometry"]["median_geometry_residual_corner_px"])
        for label, summary in mode_summaries.items()
        if summary["geometry"]["geometry_check_passed"]
    ]
    preferred_by_geometry = (
        min(geometry_candidates, key=lambda item: float(item[1]))[0] if geometry_candidates else None
    )
    preferred_by_nominal_score = max(
        mode_summaries.items(),
        key=lambda item: float(item[1]["primary_recovery"]["nominal_score"]),
    )[0]
    preferred_by_best_score = max(
        mode_summaries.items(),
        key=lambda item: float(item[1]["primary_recovery"]["best_score"]),
    )[0]

    return {
        "primary_background": primary_background,
        "preferred_by_geometry": preferred_by_geometry,
        "preferred_by_nominal_score": preferred_by_nominal_score,
        "preferred_by_best_score": preferred_by_best_score,
        "modes": mode_summaries,
        "csv_rows": csv_rows,
    }


def _plot_mode_comparison(summary: dict[str, Any], output_path: str | Path) -> Path:
    plotting = _require_matplotlib()
    labels = list(summary["modes"])
    figure, axes = plotting.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    corner_medians = [
        summary["modes"][label]["geometry"]["median_geometry_residual_corner_px"] for label in labels
    ]
    corner_maxima = [
        summary["modes"][label]["geometry"]["max_geometry_residual_corner_px"] for label in labels
    ]
    axes[0, 0].bar(labels, corner_medians, color=["tab:blue", "tab:orange"][: len(labels)], alpha=0.8, label="median")
    axes[0, 0].scatter(labels, corner_maxima, color="black", label="max", zorder=3)
    axes[0, 0].set_title("Geometry Residuals")
    axes[0, 0].set_ylabel("Pixels")
    axes[0, 0].legend(loc="best")

    nominal_scores = [summary["modes"][label]["primary_recovery"]["nominal_score"] for label in labels]
    best_scores = [summary["modes"][label]["primary_recovery"]["best_score"] for label in labels]
    x_index = range(len(labels))
    axes[0, 1].bar([index - 0.18 for index in x_index], nominal_scores, width=0.35, label="nominal")
    axes[0, 1].bar([index + 0.18 for index in x_index], best_scores, width=0.35, label="best offset")
    axes[0, 1].set_xticks(list(x_index), labels)
    axes[0, 1].set_title(f"Recovery Scores ({summary['primary_background']})")
    axes[0, 1].set_ylabel(summary["modes"][labels[0]]["primary_recovery"]["ranking_score_method"])
    axes[0, 1].legend(loc="best")

    axes[1, 0].axhline(0.0, color="lightgray", linewidth=1.0)
    axes[1, 0].axvline(0.0, color="lightgray", linewidth=1.0)
    for label in labels:
        recovery_summary = summary["modes"][label]["primary_recovery"]
        axes[1, 0].scatter(
            [recovery_summary["best_dx"]],
            [recovery_summary["best_dy"]],
            label=label,
            s=80,
        )
    axes[1, 0].set_title("Best Local Offset")
    axes[1, 0].set_xlabel("dx [pix]")
    axes[1, 0].set_ylabel("dy [pix]")
    axes[1, 0].legend(loc="best")

    for label in labels:
        growth = summary["modes"][label]["primary_recovery"]["growth_curve"]
        axes[1, 1].plot(
            [row["n_frames"] for row in growth],
            [row["nominal_score"] for row in growth],
            marker="o",
            linestyle="--",
            label=f"{label} nominal",
        )
        axes[1, 1].plot(
            [row["n_frames"] for row in growth],
            [row["best_score"] for row in growth],
            marker="o",
            linestyle="-",
            label=f"{label} best",
        )
    axes[1, 1].set_title("Growth Checkpoints")
    axes[1, 1].set_xlabel("Cumulative frames")
    axes[1, 1].set_ylabel(summary["modes"][labels[0]]["primary_recovery"]["ranking_score_method"])
    axes[1, 1].legend(loc="best", fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plotting.close(figure)
    return destination


def write_mode_comparison_products(
    comparison_summary: dict[str, Any],
    outdir: str | Path,
    prefix: str = "sedna_sector5_barycentric_compare",
) -> dict[str, Path]:
    """Write side-by-side mode comparison JSON, CSV, and PNG products."""

    pandas = _require_pandas()
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    png_path = output_dir / f"{prefix}.png"

    serializable = dict(comparison_summary)
    serializable.pop("csv_rows", None)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)
    pandas.DataFrame(comparison_summary["csv_rows"]).to_csv(csv_path, index=False)
    _plot_mode_comparison(comparison_summary, png_path)
    return {"summary_json": json_path, "summary_csv": csv_path, "diagnostic_png": png_path}


def estimate_bundle_noise_sigma(cube_bundle: dict[str, Any]) -> float:
    """Estimate a robust per-pixel noise scale from a bundle's cube."""

    numpy = _require_numpy()
    cube = numpy.asarray(cube_bundle["cube"], dtype=float)
    finite = cube[numpy.isfinite(cube)]
    if finite.size == 0:
        raise ValueError("Cube bundle contains no finite pixels for noise estimation.")
    median = float(numpy.nanmedian(finite))
    mad = float(numpy.nanmedian(numpy.abs(finite - median)))
    return 1.4826 * mad if mad > 0 else float(numpy.nanstd(finite))


def run_injection_sweep(
    ephem_df: Any,
    cube_bundle: dict[str, Any],
    amplitudes: list[float],
    sigma: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    background_method: str = "none",
    interpolation: str = "linear",
    offset_grid_radius: float = 1.0,
    offset_grid_step: float = 0.5,
    ranking_score_method: str = "aperture_snr",
) -> dict[str, Any]:
    """Inject a source into a real bundle at a set of amplitudes and recover each case."""

    numpy = _require_numpy()
    rows: list[dict[str, Any]] = []
    recoveries: dict[str, dict[str, Any]] = {}
    for amplitude in amplitudes:
        injected_bundle = inject_source_into_cube_bundle(
            cube_bundle,
            amplitude=float(amplitude),
            sigma=float(sigma),
            dx=float(dx),
            dy=float(dy),
        )
        recovery = run_known_path_recovery(
            ephem_df,
            injected_bundle,
            background_method=background_method,
            interpolation=interpolation,
            offset_grid_radius=offset_grid_radius,
            offset_grid_step=offset_grid_step,
            ranking_score_method=ranking_score_method,
        )
        key = f"{float(amplitude):.6g}"
        recoveries[key] = recovery
        row = {
            "amplitude": float(amplitude),
            "best_dx": float(recovery["summary"]["best_dx"]),
            "best_dy": float(recovery["summary"]["best_dy"]),
            "best_offset_radius": float(
                numpy.hypot(float(recovery["summary"]["best_dx"]), float(recovery["summary"]["best_dy"]))
            ),
            "nominal_score": float(recovery["summary"]["nominal_score"]),
            "best_score": float(recovery["summary"]["best_score"]),
        }
        row.update(flatten_growth_curve(recovery["growth_curve"]))
        rows.append(row)

    return {
        "background_method": background_method,
        "interpolation": interpolation,
        "ranking_score_method": ranking_score_method,
        "sigma": float(sigma),
        "dx": float(dx),
        "dy": float(dy),
        "results": rows,
        "recoveries": recoveries,
    }


def _plot_injection_sweep(summary: dict[str, Any], output_path: str | Path) -> Path:
    plotting = _require_matplotlib()

    amplitudes = [row["amplitude"] for row in summary["results"]]
    nominal_scores = [row["nominal_score"] for row in summary["results"]]
    best_scores = [row["best_score"] for row in summary["results"]]
    offset_radius = [row["best_offset_radius"] for row in summary["results"]]

    figure, axes = plotting.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(amplitudes, nominal_scores, marker="o", label="nominal")
    axes[0].plot(amplitudes, best_scores, marker="o", label="best offset")
    axes[0].set_title("Injection Score Sweep")
    axes[0].set_xlabel("Injected amplitude")
    axes[0].set_ylabel(summary["ranking_score_method"])
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(amplitudes, offset_radius, marker="o", color="tab:purple")
    axes[1].set_title("Recovered Offset Radius")
    axes[1].set_xlabel("Injected amplitude")
    axes[1].set_ylabel("sqrt(dx^2 + dy^2) [pix]")
    axes[1].grid(alpha=0.3)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plotting.close(figure)
    return destination


def write_injection_products(
    injection_summary: dict[str, Any],
    outdir: str | Path,
    prefix: str = "sedna_sector5_injection_calibration",
) -> dict[str, Path]:
    """Write JSON, CSV, and PNG products for an injection sweep."""

    pandas = _require_pandas()
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    png_path = output_dir / f"{prefix}.png"

    serializable = dict(injection_summary)
    serializable.pop("recoveries", None)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)
    pandas.DataFrame(injection_summary["results"]).to_csv(csv_path, index=False)
    _plot_injection_sweep(injection_summary, png_path)
    return {"summary_json": json_path, "summary_csv": csv_path, "diagnostic_png": png_path}


def evaluate_null_track_controls(
    recovery_result: dict[str, Any],
    control_radius: float | None = None,
    near_radius: float | None = None,
) -> dict[str, Any]:
    """Compare the nominal Sedna track against a small ring of neighboring null tracks."""

    numpy = _require_numpy()
    results = [
        {
            "dx": float(row["dx"]),
            "dy": float(row["dy"]),
            "score": float(row["score"]),
        }
        for row in recovery_result["offset_search"]["results"]
    ]
    unique_abs = sorted(
        {
            round(abs(float(value)), 8)
            for row in results
            for value in (row["dx"], row["dy"])
            if not numpy.isclose(float(value), 0.0)
        }
    )
    if not unique_abs:
        raise ValueError("Offset search did not contain any non-zero control offsets.")
    if control_radius is None:
        control_radius = float(unique_abs[-1])
    if near_radius is None:
        near_radius = float(unique_abs[0])

    control_offsets = [
        row
        for row in results
        if numpy.isclose(max(abs(row["dx"]), abs(row["dy"])), float(control_radius))
        and not (numpy.isclose(row["dx"], 0.0) and numpy.isclose(row["dy"], 0.0))
    ]
    near_nominal = [
        row
        for row in results
        if max(abs(row["dx"]), abs(row["dy"])) <= float(near_radius) + 1e-8
    ]

    nominal_score = float(recovery_result["summary"]["nominal_score"])
    best_dx = float(recovery_result["summary"]["best_dx"])
    best_dy = float(recovery_result["summary"]["best_dy"])
    best_score = float(recovery_result["summary"]["best_score"])
    control_scores = [row["score"] for row in control_offsets]
    best_near_nominal = max(near_nominal, key=lambda row: row["score"])
    best_control = max(control_offsets, key=lambda row: row["score"])

    summary = {
        "target": recovery_result["summary"]["target"],
        "sector": recovery_result["summary"]["sector"],
        "camera": recovery_result["summary"]["camera"],
        "ccd": recovery_result["summary"]["ccd"],
        "background_method": recovery_result["summary"]["background_method"],
        "ranking_score_method": recovery_result["summary"]["ranking_score_method"],
        "control_radius": float(control_radius),
        "near_radius": float(near_radius),
        "nominal_score": nominal_score,
        "best_dx": best_dx,
        "best_dy": best_dy,
        "best_score": best_score,
        "best_offset_is_near_nominal": bool(max(abs(best_dx), abs(best_dy)) <= float(near_radius) + 1e-8),
        "best_near_nominal_dx": float(best_near_nominal["dx"]),
        "best_near_nominal_dy": float(best_near_nominal["dy"]),
        "best_near_nominal_score": float(best_near_nominal["score"]),
        "best_control_dx": float(best_control["dx"]),
        "best_control_dy": float(best_control["dy"]),
        "best_control_score": float(best_control["score"]),
        "nominal_minus_best_control": nominal_score - float(best_control["score"]),
        "best_minus_best_control": best_score - float(best_control["score"]),
        "control_score_median": float(numpy.nanmedian(control_scores)),
        "control_score_max": float(numpy.nanmax(control_scores)),
        "growth_curve": compact_growth_curve(recovery_result.get("growth_curve", [])),
    }
    return {
        "summary": summary,
        "controls": sorted(control_offsets, key=lambda row: (row["dy"], row["dx"])),
        "all_results": results,
        "recovery_result": recovery_result,
    }


def _score_grid(results: list[dict[str, float]]) -> tuple[Any, Any, Any]:
    numpy = _require_numpy()
    dx_values = numpy.asarray(sorted({float(item["dx"]) for item in results}), dtype=float)
    dy_values = numpy.asarray(sorted({float(item["dy"]) for item in results}), dtype=float)
    score_map = numpy.full((dy_values.size, dx_values.size), numpy.nan, dtype=float)
    dx_index = {float(value): index for index, value in enumerate(dx_values)}
    dy_index = {float(value): index for index, value in enumerate(dy_values)}
    for result in results:
        score_map[dy_index[float(result["dy"])], dx_index[float(result["dx"])]] = float(result["score"])
    return score_map, dx_values, dy_values


def _plot_null_track_controls(summary_bundle: dict[str, Any], output_path: str | Path) -> Path:
    plotting = _require_matplotlib()
    recovery_result = summary_bundle["recovery_result"]
    summary = summary_bundle["summary"]

    figure, axes = plotting.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    score_map, dx_values, dy_values = _score_grid(summary_bundle["all_results"])
    image = axes[0].imshow(
        score_map,
        origin="lower",
        aspect="auto",
        extent=(dx_values.min(), dx_values.max(), dy_values.min(), dy_values.max()),
        cmap="magma",
    )
    axes[0].scatter([0.0], [0.0], marker="o", color="white", s=40, linewidths=1.0)
    axes[0].scatter([summary["best_dx"]], [summary["best_dy"]], marker="x", color="cyan", s=60, linewidths=1.5)
    axes[0].set_title("Local Offset Score Map")
    axes[0].set_xlabel("dx [pix]")
    axes[0].set_ylabel("dy [pix]")
    figure.colorbar(image, ax=axes[0], shrink=0.8)

    controls = summary_bundle["controls"]
    labels = [f"({row['dx']:.1f},{row['dy']:.1f})" for row in controls]
    scores = [row["score"] for row in controls]
    axes[1].bar(range(len(scores)), scores, color="tab:orange", alpha=0.8)
    axes[1].axhline(summary["nominal_score"], color="black", linestyle="--", label="nominal")
    axes[1].axhline(summary["best_score"], color="tab:blue", linestyle="-", label="best local")
    axes[1].set_xticks(range(len(scores)), labels, rotation=45, ha="right")
    axes[1].set_title("Neighboring Null-Track Scores")
    axes[1].set_ylabel(summary["ranking_score_method"])
    axes[1].legend(loc="best")

    growth = summary["growth_curve"]
    axes[2].plot([row["n_frames"] for row in growth], [row["nominal_score"] for row in growth], marker="o", label="nominal")
    axes[2].plot([row["n_frames"] for row in growth], [row["best_score"] for row in growth], marker="o", label="best offset")
    axes[2].set_title("Cadence Ladder")
    axes[2].set_xlabel("Cumulative frames")
    axes[2].set_ylabel(summary["ranking_score_method"])
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.3)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plotting.close(figure)
    return destination


def write_null_control_products(
    null_summary_bundle: dict[str, Any],
    outdir: str | Path,
    prefix: str = "sedna_sector5_null_track_controls",
) -> dict[str, Path]:
    """Write JSON, CSV, cadence ladder CSV, and PNG products for null-track controls."""

    pandas = _require_pandas()
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    ladder_path = output_dir / f"{prefix}_cadence_ladder.csv"
    png_path = output_dir / f"{prefix}.png"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(null_summary_bundle["summary"], handle, indent=2, sort_keys=True)
    pandas.DataFrame(null_summary_bundle["controls"]).to_csv(csv_path, index=False)
    pandas.DataFrame(null_summary_bundle["summary"]["growth_curve"]).to_csv(ladder_path, index=False)
    _plot_null_track_controls(null_summary_bundle, png_path)
    return {
        "summary_json": json_path,
        "controls_csv": csv_path,
        "cadence_csv": ladder_path,
        "diagnostic_png": png_path,
    }
