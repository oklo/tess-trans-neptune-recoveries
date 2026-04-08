"""Narrow track-calibration helpers for bounded detector-local studies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

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

from kbo_tess.photometry import aperture_photometry_at_position
from kbo_tess.shiftstack import search_local_track_offsets
from kbo_tess.stack_audit import build_stack_variant, measure_local_centroids


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for track calibration. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return np


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for track calibration. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return pd


def _require_matplotlib() -> Any:
    if plt is None:
        raise ImportError(
            "matplotlib is required for track-calibration plots. Install project dependencies "
            "with `python -m pip install -e .`."
        )
    return plt


def _growth_checkpoint_counts(n_frames: int) -> list[int]:
    if n_frames <= 5:
        return list(range(2, n_frames + 1))
    counts = {2, n_frames}
    for fraction in (0.25, 0.5, 0.75):
        counts.add(max(2, min(n_frames, int(round(fraction * n_frames)))))
    return sorted(counts)


def _summarize_track_photometry(table: Any) -> dict[str, float]:
    pandas = _require_pandas()
    phot_table = pandas.DataFrame(table)
    return {
        "median_aperture_snr": float(phot_table["aperture_snr"].median()),
        "mean_aperture_snr": float(phot_table["aperture_snr"].mean()),
        "max_aperture_snr": float(phot_table["aperture_snr"].max()),
        "median_net_flux": float(phot_table["net_flux"].median()),
        "mean_net_flux": float(phot_table["net_flux"].mean()),
        "max_net_flux": float(phot_table["net_flux"].max()),
    }


def evaluate_track_photometry(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
) -> Any:
    """Measure per-frame moving-aperture photometry along a supplied local track."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    cube_array = numpy.asarray(cube, dtype=float)
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    rows: list[dict[str, float]] = []
    for frame_index, frame in enumerate(cube_array):
        phot = aperture_photometry_at_position(frame, (float(x_array[frame_index]), float(y_array[frame_index])))
        rows.append(
            {
                "frame_index": float(frame_index),
                "track_x": float(x_array[frame_index]),
                "track_y": float(y_array[frame_index]),
                "background_level": float(phot["background_level"]),
                "background_sigma": float(phot["background_sigma"]),
                "net_flux": float(phot["net_flux"]),
                "aperture_snr": float(phot["aperture_snr"]),
            }
        )
    return pandas.DataFrame(rows)


def run_x_offset_sweep(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    x_offsets: Iterable[float],
    reference: tuple[float, float],
    interpolation: str = "linear",
    ranking_method: str = "aperture_snr",
) -> dict[str, Any]:
    """Evaluate a narrow constant x-offset grid with direct and stacked metrics."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    rows: list[dict[str, float]] = []
    photometry_tables: dict[float, Any] = {}
    for x_offset in [float(value) for value in x_offsets]:
        shifted_x = numpy.asarray(x, dtype=float) + float(x_offset)
        shifted_y = numpy.asarray(y, dtype=float)
        photometry = evaluate_track_photometry(cube, shifted_x, shifted_y)
        photometry_tables[float(x_offset)] = photometry
        phot_summary = _summarize_track_photometry(photometry)
        stack_variant = build_stack_variant(
            cube,
            shifted_x,
            shifted_y,
            reference=reference,
            interpolation=interpolation,
            ranking_position=reference,
            ranking_method=ranking_method,
        )
        rows.append(
            {
                "x_offset": float(x_offset),
                **phot_summary,
                "stack_ranking_score": float(stack_variant["ranking_score"]),
                "stack_aperture_snr": float(stack_variant["scores"]["aperture_snr"]),
                "stack_peak_aperture_snr": float(stack_variant["peak_location"]["peak_aperture_snr"]),
                "stack_peak_x": float(stack_variant["peak_location"]["x"]),
                "stack_peak_y": float(stack_variant["peak_location"]["y"]),
                "stack_peak_value": float(stack_variant["peak_location"]["value"]),
                "support_at_reference": float(
                    stack_variant["support_map"][int(round(reference[1])), int(round(reference[0]))]
                ),
                "max_support": float(numpy.nanmax(stack_variant["support_map"])),
            }
        )

    table = pandas.DataFrame(rows).sort_values("x_offset", ascending=False).reset_index(drop=True)
    best_stack_row = table.loc[table["stack_ranking_score"].idxmax()].to_dict()
    best_direct_row = table.loc[table["mean_aperture_snr"].idxmax()].to_dict()
    return {
        "table": table,
        "best_stack_row": best_stack_row,
        "best_direct_row": best_direct_row,
        "photometry_tables": photometry_tables,
        "reference": {"x": float(reference[0]), "y": float(reference[1])},
        "ranking_method": ranking_method,
        "interpolation": interpolation,
    }


def run_offset_grid_metrics(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    dx_grid: Iterable[float],
    dy_grid: Iterable[float],
    reference: tuple[float, float],
    interpolation: str = "linear",
    ranking_method: str = "aperture_snr",
) -> dict[str, Any]:
    """Evaluate direct and stacked metrics on a fixed 2-D dx/dy grid."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    rows: list[dict[str, float]] = []
    for dy in [float(value) for value in dy_grid]:
        for dx in [float(value) for value in dx_grid]:
            shifted_x = x_array + float(dx)
            shifted_y = y_array + float(dy)
            direct_table = evaluate_track_photometry(cube, shifted_x, shifted_y)
            direct_summary = _summarize_track_photometry(direct_table)
            stack_variant = build_stack_variant(
                cube,
                shifted_x,
                shifted_y,
                reference=reference,
                interpolation=interpolation,
                ranking_position=reference,
                ranking_method=ranking_method,
            )
            rows.append(
                {
                    "dx": float(dx),
                    "dy": float(dy),
                    **direct_summary,
                    "direct_score": float(direct_summary["mean_aperture_snr"]),
                    "stack_ranking_score": float(stack_variant["ranking_score"]),
                    "stack_aperture_snr": float(stack_variant["scores"]["aperture_snr"]),
                    "stack_peak_x": float(stack_variant["peak_location"]["x"]),
                    "stack_peak_y": float(stack_variant["peak_location"]["y"]),
                    "stack_peak_value": float(stack_variant["peak_location"]["value"]),
                    "stack_peak_aperture_snr": float(stack_variant["peak_location"]["peak_aperture_snr"]),
                    "support_at_reference": float(
                        stack_variant["support_map"][int(round(reference[1])), int(round(reference[0]))]
                    ),
                }
            )
    table = pandas.DataFrame(rows).sort_values(["dy", "dx"]).reset_index(drop=True)
    best_direct_row = table.loc[table["direct_score"].idxmax()].to_dict()
    best_stack_row = table.loc[table["stack_ranking_score"].idxmax()].to_dict()
    return {
        "table": table,
        "best_direct_row": best_direct_row,
        "best_stack_row": best_stack_row,
        "reference": {"x": float(reference[0]), "y": float(reference[1])},
        "ranking_method": ranking_method,
        "interpolation": interpolation,
    }


def plot_offset_grid_heatmaps(
    grid_result: dict[str, Any],
    output_path: str | Path,
    nominal_point: tuple[float, float] = (0.0, 0.0),
    calibrated_point: tuple[float, float] | None = None,
) -> Path:
    """Plot direct and stack dx/dy heatmaps with key tracks marked."""

    plotting = _require_matplotlib()
    pandas = _require_pandas()
    table = pandas.DataFrame(grid_result["table"])
    dx_values = sorted(table["dx"].astype(float).unique())
    dy_values = sorted(table["dy"].astype(float).unique())
    direct_grid = (
        table.pivot_table(index="dy", columns="dx", values="direct_score", aggfunc="first")
        .reindex(index=dy_values, columns=dx_values)
    )
    stack_grid = (
        table.pivot_table(index="dy", columns="dx", values="stack_ranking_score", aggfunc="first")
        .reindex(index=dy_values, columns=dx_values)
    )

    extent = [min(dx_values), max(dx_values), min(dy_values), max(dy_values)]
    figure, axes = plotting.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, grid, title, best_row in (
        (axes[0], direct_grid, "Direct Mean Aperture S/N", grid_result["best_direct_row"]),
        (axes[1], stack_grid, "Stack Aperture S/N", grid_result["best_stack_row"]),
    ):
        artist = axis.imshow(
            grid.to_numpy(),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
        )
        axis.scatter([nominal_point[0]], [nominal_point[1]], marker="o", facecolors="none", edgecolors="white", s=70)
        if calibrated_point is not None:
            axis.scatter(
                [calibrated_point[0]],
                [calibrated_point[1]],
                marker="s",
                facecolors="none",
                edgecolors="cyan",
                s=70,
            )
        axis.scatter([float(best_row["dx"])], [float(best_row["dy"])], marker="x", color="lime", s=70)
        axis.set_title(title)
        axis.set_xlabel("dx [pix]")
        axis.set_ylabel("dy [pix]")
        figure.colorbar(artist, ax=axis, shrink=0.85)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plotting.close(figure)
    return destination


def build_variant_dictionary(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    reference: tuple[float, float],
    best_grid_x_offset: float,
    centroid_informed_x_offset: float,
    interpolation: str = "linear",
) -> dict[str, dict[str, Any]]:
    """Build highlighted stack variants for side-by-side comparison."""

    numpy = _require_numpy()
    centroid_table = measure_local_centroids(cube, x, y)
    centroid_good = centroid_table["valid_centroid"].astype(bool).to_numpy()
    centroid_x = centroid_table.loc[centroid_good, "measured_x"].astype(float).to_numpy()
    centroid_y = centroid_table.loc[centroid_good, "measured_y"].astype(float).to_numpy()
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    return {
        "nominal": build_stack_variant(
            cube,
            x_array,
            y_array,
            reference=reference,
            interpolation=interpolation,
            ranking_position=reference,
        ),
        "centroid_informed": build_stack_variant(
            cube,
            x_array + float(centroid_informed_x_offset),
            y_array,
            reference=reference,
            interpolation=interpolation,
            ranking_position=reference,
        ),
        "x_minus_half": build_stack_variant(
            cube,
            x_array - 0.5,
            y_array,
            reference=reference,
            interpolation=interpolation,
            ranking_position=reference,
        ),
        "best_fine_grid": build_stack_variant(
            cube,
            x_array + float(best_grid_x_offset),
            y_array,
            reference=reference,
            interpolation=interpolation,
            ranking_position=reference,
        ),
    }


def run_centered_null_control_gate(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    calibrated_x_offset: float,
    reference: tuple[float, float],
    dx_grid: Iterable[float],
    dy_grid: Iterable[float],
    interpolation: str = "linear",
    ranking_method: str = "aperture_snr",
    control_radius: float | None = None,
    near_radius: float | None = None,
) -> dict[str, Any]:
    """Run a local null-control comparison around a calibrated-x track."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    base_x = numpy.asarray(x, dtype=float) + float(calibrated_x_offset)
    base_y = numpy.asarray(y, dtype=float)
    nominal_variant = build_stack_variant(
        cube,
        numpy.asarray(x, dtype=float),
        numpy.asarray(y, dtype=float),
        reference=reference,
        interpolation=interpolation,
        ranking_position=reference,
        ranking_method=ranking_method,
    )
    calibrated_variant = build_stack_variant(
        cube,
        base_x,
        base_y,
        reference=reference,
        interpolation=interpolation,
        ranking_position=reference,
        ranking_method=ranking_method,
    )
    search = search_local_track_offsets(
        cube,
        base_x,
        base_y,
        dx_grid=dx_grid,
        dy_grid=dy_grid,
        interpolation=interpolation,
        stack_method="mean",
        score_method=ranking_method,
        reference=reference,
        score_position=reference,
    )
    results = pandas.DataFrame(search["results"]).sort_values(["dy", "dx"]).reset_index(drop=True)
    unique_abs = sorted(
        {
            round(abs(float(value)), 8)
            for row in search["results"]
            for value in (row["dx"], row["dy"])
            if not numpy.isclose(float(value), 0.0)
        }
    )
    if not unique_abs:
        raise ValueError("Calibrated null-control grid does not contain any non-zero offsets.")
    if control_radius is None:
        control_radius = float(unique_abs[-1])
    if near_radius is None:
        near_radius = float(unique_abs[0])

    control_rows = results[
        (results[["dx", "dy"]].abs().max(axis=1) >= float(control_radius) - 1e-8)
        & ~((results["dx"].abs() < 1e-8) & (results["dy"].abs() < 1e-8))
    ].copy()
    near_rows = results[(results[["dx", "dy"]].abs().max(axis=1) <= float(near_radius) + 1e-8)].copy()
    best_local = dict(search["best"])
    best_control_row = control_rows.loc[control_rows["score"].idxmax()].to_dict()
    best_near_row = near_rows.loc[near_rows["score"].idxmax()].to_dict()

    summary = {
        "calibrated_x_offset": float(calibrated_x_offset),
        "exact_nominal_score": float(nominal_variant["ranking_score"]),
        "calibrated_track_score": float(calibrated_variant["ranking_score"]),
        "calibrated_track_peak_x": float(calibrated_variant["peak_location"]["x"]),
        "calibrated_track_peak_y": float(calibrated_variant["peak_location"]["y"]),
        "best_local_dx_delta": float(best_local["dx"]),
        "best_local_dy_delta": float(best_local["dy"]),
        "best_local_score": float(best_local["score"]),
        "best_near_dx_delta": float(best_near_row["dx"]),
        "best_near_dy_delta": float(best_near_row["dy"]),
        "best_near_score": float(best_near_row["score"]),
        "best_control_dx_delta": float(best_control_row["dx"]),
        "best_control_dy_delta": float(best_control_row["dy"]),
        "best_control_score": float(best_control_row["score"]),
        "calibrated_minus_best_control": float(calibrated_variant["ranking_score"] - float(best_control_row["score"])),
        "exact_nominal_minus_best_control": float(nominal_variant["ranking_score"] - float(best_control_row["score"])),
        "control_radius": float(control_radius),
        "near_radius": float(near_radius),
    }
    return {
        "summary": summary,
        "results": results,
        "control_rows": control_rows,
        "calibrated_variant": calibrated_variant,
        "nominal_variant": nominal_variant,
        "best_stack": numpy.asarray(best_local["stack"], dtype=float),
    }


def build_nominal_vs_calibrated_growth_curve(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    calibrated_x_offset: float,
    reference: tuple[float, float],
    interpolation: str = "linear",
    ranking_method: str = "aperture_snr",
) -> Any:
    """Compare cadence growth for exact nominal and calibrated-x tracks."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    cube_array = numpy.asarray(cube, dtype=float)
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    rows: list[dict[str, float]] = []
    for count in _growth_checkpoint_counts(int(cube_array.shape[0])):
        nominal = build_stack_variant(
            cube_array[:count],
            x_array[:count],
            y_array[:count],
            reference=reference,
            interpolation=interpolation,
            ranking_position=reference,
            ranking_method=ranking_method,
        )
        calibrated = build_stack_variant(
            cube_array[:count],
            x_array[:count] + float(calibrated_x_offset),
            y_array[:count],
            reference=reference,
            interpolation=interpolation,
            ranking_position=reference,
            ranking_method=ranking_method,
        )
        rows.append(
            {
                "n_frames": float(count),
                "nominal_score": float(nominal["ranking_score"]),
                "calibrated_score": float(calibrated["ranking_score"]),
                "nominal_peak_aperture_snr": float(nominal["peak_location"]["peak_aperture_snr"]),
                "calibrated_peak_aperture_snr": float(calibrated["peak_location"]["peak_aperture_snr"]),
            }
        )
    return pandas.DataFrame(rows)


def plot_x_offset_sweep(
    sweep_result: dict[str, Any],
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot direct and stacked score summaries as a function of x offset."""

    plotting = _require_matplotlib()
    table = sweep_result["table"]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plotting.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(table["x_offset"], table["mean_aperture_snr"], marker="o", label="direct mean S/N")
    axes[0].plot(table["x_offset"], table["median_aperture_snr"], marker="o", linestyle="--", label="direct median S/N")
    axes[0].set_title(f"{title}: direct photometry")
    axes[0].set_xlabel("Constant x offset [pix]")
    axes[0].set_ylabel("Aperture S/N")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(table["x_offset"], table["stack_ranking_score"], marker="o", label="stack score")
    axes[1].plot(table["x_offset"], table["stack_peak_aperture_snr"], marker="o", linestyle="--", label="peak aperture S/N")
    axes[1].set_title(f"{title}: stacked signal")
    axes[1].set_xlabel("Constant x offset [pix]")
    axes[1].set_ylabel(sweep_result["ranking_method"])
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    figure.savefig(destination, dpi=170)
    plotting.close(figure)
    return destination


def plot_x_offset_comparison(
    representative_frame: Any,
    reference: tuple[float, float],
    variants: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot nominal, centroid-informed, x-0.5, and best fine-grid stacks."""

    numpy = _require_numpy()
    plotting = _require_matplotlib()
    frame = numpy.asarray(representative_frame, dtype=float)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plotting.subplots(1, 5, figsize=(18, 4.5), constrained_layout=True)
    finite = frame[numpy.isfinite(frame)]
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [5, 99])
    else:
        vmin, vmax = -1.0, 1.0
    axes[0].imshow(frame, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].scatter([reference[0]], [reference[1]], marker="x", color="white", s=50, linewidths=1.5)
    axes[0].set_title("Representative frame")
    axes[0].set_xlabel("x [pix]")
    axes[0].set_ylabel("y [pix]")

    order = [
        ("nominal", "Nominal"),
        ("centroid_informed", "Centroid-informed"),
        ("x_minus_half", "x - 0.5"),
        ("best_fine_grid", "Best fine-grid x"),
    ]
    for axis, (key, title) in zip(axes[1:], order):
        variant = variants[key]
        image = numpy.asarray(variant["stack"], dtype=float)
        finite = image[numpy.isfinite(image)]
        if finite.size:
            x_vmin, x_vmax = np.nanpercentile(finite, [5, 99.5])
        else:
            x_vmin, x_vmax = -1.0, 1.0
        axis.imshow(image, origin="lower", cmap="magma", vmin=x_vmin, vmax=x_vmax)
        axis.scatter([reference[0]], [reference[1]], marker="x", color="white", s=50, linewidths=1.5)
        axis.set_title(f"{title}\nS/N={variant['ranking_score']:.2f}")
        axis.set_xlabel("x [pix]")
        axis.set_ylabel("y [pix]")

    figure.savefig(destination, dpi=170)
    plotting.close(figure)
    return destination


def plot_calibrated_null_controls(
    null_result: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Plot a calibrated-track null-control comparison."""

    numpy = _require_numpy()
    plotting = _require_matplotlib()
    results = null_result["results"]
    summary = null_result["summary"]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    dx_values = numpy.asarray(sorted(results["dx"].astype(float).unique()), dtype=float)
    dy_values = numpy.asarray(sorted(results["dy"].astype(float).unique()), dtype=float)
    score_map = numpy.full((dy_values.size, dx_values.size), numpy.nan, dtype=float)
    for _, row in results.iterrows():
        dx_index = int(numpy.where(numpy.isclose(dx_values, float(row["dx"])))[0][0])
        dy_index = int(numpy.where(numpy.isclose(dy_values, float(row["dy"])))[0][0])
        score_map[dy_index, dx_index] = float(row["score"])

    figure, axes = plotting.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    image = axes[0].imshow(
        score_map,
        origin="lower",
        aspect="auto",
        extent=(dx_values.min(), dx_values.max(), dy_values.min(), dy_values.max()),
        cmap="magma",
    )
    axes[0].scatter([0.0], [0.0], marker="o", color="white", s=40)
    axes[0].scatter([summary["best_local_dx_delta"]], [summary["best_local_dy_delta"]], marker="x", color="cyan", s=60)
    axes[0].set_title("Offsets around calibrated x track")
    axes[0].set_xlabel("delta x [pix]")
    axes[0].set_ylabel("delta y [pix]")
    figure.colorbar(image, ax=axes[0], shrink=0.8)

    control_rows = null_result["control_rows"]
    labels = [f"({row.dx:+.1f},{row.dy:+.1f})" for row in control_rows.itertuples()]
    values = control_rows["score"].astype(float).to_list()
    positions = list(range(len(values)))
    axes[1].bar(positions, values)
    axes[1].axhline(summary["exact_nominal_score"], color="black", linestyle="--", label="exact nominal")
    axes[1].axhline(summary["calibrated_track_score"], color="tab:blue", linestyle="-", label="calibrated x")
    axes[1].set_xticks(positions, labels, rotation=45, ha="right")
    axes[1].set_title("Outer null-control ring")
    axes[1].set_ylabel("aperture_snr")
    axes[1].legend(loc="best")

    axes[2].axis("off")
    axes[2].text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Calibrated x offset: {summary['calibrated_x_offset']:.3f} px",
                f"Exact nominal score: {summary['exact_nominal_score']:.3f}",
                f"Calibrated score: {summary['calibrated_track_score']:.3f}",
                f"Best local delta: ({summary['best_local_dx_delta']:+.2f}, {summary['best_local_dy_delta']:+.2f})",
                f"Best local score: {summary['best_local_score']:.3f}",
                f"Best control score: {summary['best_control_score']:.3f}",
                f"Calibrated - best control: {summary['calibrated_minus_best_control']:.3f}",
            ]
        ),
        transform=axes[2].transAxes,
        ha="left",
        va="top",
    )

    figure.savefig(destination, dpi=170)
    plotting.close(figure)
    return destination


def plot_growth_comparison(growth_curve: Any, output_path: str | Path) -> Path:
    """Plot exact nominal vs calibrated-x cadence growth."""

    plotting = _require_matplotlib()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plotting.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(growth_curve["n_frames"], growth_curve["nominal_score"], marker="o", label="exact nominal")
    axis.plot(growth_curve["n_frames"], growth_curve["calibrated_score"], marker="o", label="calibrated x")
    axis.set_title("Cadence growth: nominal vs calibrated-x")
    axis.set_xlabel("Cumulative frames")
    axis.set_ylabel("aperture_snr")
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    figure.savefig(destination, dpi=170)
    plotting.close(figure)
    return destination


def plot_time_residual_diagnostic(
    subset_times: Any,
    subset_centroids: Any,
    sample_times: Any,
    sample_centroids: Any,
    calibrated_x_offset: float,
    output_path: str | Path,
) -> Path:
    """Plot centroid residuals versus time for subset and sparse full-bundle samples."""

    plotting = _require_matplotlib()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plotting.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex=True)
    axes[0].scatter(subset_times, subset_centroids["residual_x"], label="bright subset", s=30)
    axes[0].scatter(sample_times, sample_centroids["residual_x"], label="sparse full bundle", s=25, alpha=0.7)
    axes[0].axhline(float(calibrated_x_offset), color="tab:red", linestyle="--", label="calibrated x")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("Residual x [pix]")
    axes[0].set_title("Centroid residuals versus time")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].scatter(subset_times, subset_centroids["residual_y"], label="bright subset", s=30)
    axes[1].scatter(sample_times, sample_centroids["residual_y"], label="sparse full bundle", s=25, alpha=0.7)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].set_xlabel("BTJD")
    axes[1].set_ylabel("Residual y [pix]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")
    figure.savefig(destination, dpi=170)
    plotting.close(figure)
    return destination


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON payload with stable formatting."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return destination
