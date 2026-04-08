"""Transparent flux-ledger helpers for frozen detector-local bundle studies.

This module is intentionally simple and explicit. It is meant to answer:

- what do selected raw pixels do as the target track passes nearby?
- how do those raw pixel values become a simple aperture sum?
- how do simple box apertures compare with the existing moving-aperture path?

The simple aperture benchmark here is a nearest-pixel box aperture:

- the box is centered on the nearest detector pixel to the supplied track point
- aperture pixels have unit weights
- the background estimate is the median of a circular annulus around the same
  subpixel track center

That choice is not meant to be optimal. It is meant to be transparent.
"""

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


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for flux-ledger diagnostics. Install project dependencies "
            "with `python -m pip install -e .`."
        )
    return np


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for flux-ledger diagnostics. Install project dependencies "
            "with `python -m pip install -e .`."
        )
    return pd


def _require_matplotlib() -> Any:
    if plt is None:
        raise ImportError(
            "matplotlib is required for flux-ledger plots. Install project dependencies "
            "with `python -m pip install -e .`."
        )
    return plt


def compute_pixel_residency_map(
    shape: tuple[int, int],
    x: Iterable[float],
    y: Iterable[float],
    radius: float = 0.5,
) -> Any:
    """Count how many cadences place the target center near each pixel center."""

    numpy = _require_numpy()
    ny, nx = int(shape[0]), int(shape[1])
    yy, xx = numpy.indices((ny, nx), dtype=float)
    x_array = numpy.asarray(x, dtype=float)[:, None, None]
    y_array = numpy.asarray(y, dtype=float)[:, None, None]
    rr = numpy.hypot(xx[None, :, :] - x_array, yy[None, :, :] - y_array)
    return numpy.sum(rr <= float(radius), axis=0).astype(float)


def select_representative_pixels(
    track_x: Iterable[float],
    track_y: Iterable[float],
    shape: tuple[int, int],
    occupancy_radius: float = 0.5,
    n_track_pixels: int = 5,
) -> Any:
    """Pick a few track-adjacent pixels plus one off-track comparison pixel."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    occupancy = compute_pixel_residency_map(shape, track_x, track_y, radius=occupancy_radius)
    flat_order = numpy.argsort(occupancy.ravel())[::-1]
    rows: list[dict[str, float | str]] = []
    seen: set[tuple[int, int]] = set()
    for rank, flat_index in enumerate(flat_order, start=1):
        iy, ix = numpy.unravel_index(int(flat_index), occupancy.shape)
        count = float(occupancy[iy, ix])
        if count <= 0:
            break
        key = (int(ix), int(iy))
        if key in seen:
            continue
        rows.append(
            {
                "label": f"track_pixel_{len(rows) + 1}",
                "pixel_x": float(ix),
                "pixel_y": float(iy),
                "category": "track_residency",
                "residency_count": count,
                "residency_rank": float(rank),
            }
        )
        seen.add(key)
        if len(rows) >= int(n_track_pixels):
            break

    if rows:
        anchor_x = int(rows[0]["pixel_x"])
        anchor_y = int(rows[0]["pixel_y"])
        for dx, dy, label_suffix in ((-1, 0, "left"), (1, 0, "right"), (0, -1, "down"), (0, 1, "up")):
            ix = anchor_x + int(dx)
            iy = anchor_y + int(dy)
            if not (0 <= ix < occupancy.shape[1] and 0 <= iy < occupancy.shape[0]):
                continue
            key = (int(ix), int(iy))
            if key in seen:
                continue
            rows.append(
                {
                    "label": f"neighbor_{label_suffix}",
                    "pixel_x": float(ix),
                    "pixel_y": float(iy),
                    "category": "neighbor",
                    "residency_count": float(occupancy[iy, ix]),
                    "residency_rank": float("nan"),
                }
            )
            seen.add(key)

    yy, xx = numpy.indices(occupancy.shape, dtype=float)
    distance_from_track = numpy.full(occupancy.shape, numpy.inf, dtype=float)
    track_x_array = numpy.asarray(track_x, dtype=float)
    track_y_array = numpy.asarray(track_y, dtype=float)
    for x0, y0 in zip(track_x_array, track_y_array, strict=False):
        distance_from_track = numpy.minimum(distance_from_track, numpy.hypot(xx - float(x0), yy - float(y0)))
    off_index = int(numpy.nanargmax(distance_from_track))
    off_y, off_x = numpy.unravel_index(off_index, occupancy.shape)
    if (int(off_x), int(off_y)) not in seen:
        rows.append(
            {
                "label": "off_track_reference",
                "pixel_x": float(off_x),
                "pixel_y": float(off_y),
                "category": "off_track",
                "residency_count": float(occupancy[off_y, off_x]),
                "residency_rank": float("nan"),
            }
        )

    return pandas.DataFrame(rows)


def extract_pixel_time_series(
    cube: Any,
    time: Iterable[float],
    selected_pixels: Any,
    nominal_x: Iterable[float],
    nominal_y: Iterable[float],
    calibrated_x: Iterable[float],
    calibrated_y: Iterable[float],
    quality_mask: Iterable[bool] | None = None,
) -> Any:
    """Return a long-form table of raw selected-pixel time series."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    cube_array = numpy.asarray(cube, dtype=float)
    time_array = numpy.asarray(time, dtype=float)
    nominal_x_array = numpy.asarray(nominal_x, dtype=float)
    nominal_y_array = numpy.asarray(nominal_y, dtype=float)
    calibrated_x_array = numpy.asarray(calibrated_x, dtype=float)
    calibrated_y_array = numpy.asarray(calibrated_y, dtype=float)
    if quality_mask is None:
        keep_array = numpy.ones(cube_array.shape[0], dtype=bool)
    else:
        keep_array = numpy.asarray(quality_mask, dtype=bool)

    pixels = pandas.DataFrame(selected_pixels)
    rows: list[dict[str, float | bool | str]] = []
    for pixel in pixels.itertuples(index=False):
        ix = int(pixel.pixel_x)
        iy = int(pixel.pixel_y)
        for frame_index in range(cube_array.shape[0]):
            value = float(cube_array[frame_index, iy, ix])
            dist_nominal = float(numpy.hypot(ix - nominal_x_array[frame_index], iy - nominal_y_array[frame_index]))
            dist_calibrated = float(
                numpy.hypot(ix - calibrated_x_array[frame_index], iy - calibrated_y_array[frame_index])
            )
            rows.append(
                {
                    "label": str(pixel.label),
                    "category": str(pixel.category),
                    "pixel_x": float(ix),
                    "pixel_y": float(iy),
                    "frame_index": float(frame_index),
                    "time_tdb_jd_minus_2457000": float(time_array[frame_index]),
                    "value": value,
                    "quality_keep": bool(keep_array[frame_index]),
                    "dist_nominal": dist_nominal,
                    "dist_calibrated": dist_calibrated,
                    "nominal_within_0p25": bool(dist_nominal <= 0.25),
                    "nominal_within_0p5": bool(dist_nominal <= 0.5),
                    "nominal_within_1p0": bool(dist_nominal <= 1.0),
                    "calibrated_within_0p25": bool(dist_calibrated <= 0.25),
                    "calibrated_within_0p5": bool(dist_calibrated <= 0.5),
                    "calibrated_within_1p0": bool(dist_calibrated <= 1.0),
                }
            )
    return pandas.DataFrame(rows)


def select_flux_ledger_sample_indices(
    ranking_table: Any,
    n_uniform: int = 6,
    n_bright: int = 6,
) -> list[int]:
    """Choose a compact cadence sample for the per-pixel aperture ledger."""

    pandas = _require_pandas()
    table = pandas.DataFrame(ranking_table)
    n_frames = int(table.shape[0])
    if n_frames == 0:
        return []
    bright = table.sort_values("aperture_snr", ascending=False).head(int(n_bright)).index.astype(int).tolist()
    if n_uniform <= 1:
        uniform = [0]
    else:
        uniform = np.linspace(0, n_frames - 1, int(n_uniform), dtype=int).tolist()
    return sorted(set(int(value) for value in (bright + uniform)))


def compute_box_aperture_time_series(
    cube: Any,
    time: Iterable[float],
    x: Iterable[float],
    y: Iterable[float],
    half_size: int = 1,
    annulus_inner_radius: float = 3.0,
    annulus_outer_radius: float = 5.0,
    track_label: str = "track",
) -> Any:
    """Compute a transparent nearest-pixel box-aperture time series."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    cube_array = numpy.asarray(cube, dtype=float)
    time_array = numpy.asarray(time, dtype=float)
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    ny, nx = cube_array.shape[1:]
    yy, xx = numpy.indices((ny, nx), dtype=float)
    rows: list[dict[str, float | str]] = []
    for frame_index, frame in enumerate(cube_array):
        x0 = float(x_array[frame_index])
        y0 = float(y_array[frame_index])
        center_x = int(round(x0))
        center_y = int(round(y0))
        x_min = max(0, center_x - int(half_size))
        x_max = min(nx, center_x + int(half_size) + 1)
        y_min = max(0, center_y - int(half_size))
        y_max = min(ny, center_y + int(half_size) + 1)
        aperture = frame[y_min:y_max, x_min:x_max]
        finite_aperture = aperture[numpy.isfinite(aperture)]
        rr = numpy.hypot(xx - x0, yy - y0)
        annulus_mask = (rr >= float(annulus_inner_radius)) & (rr <= float(annulus_outer_radius))
        annulus_values = frame[annulus_mask]
        finite_annulus = annulus_values[numpy.isfinite(annulus_values)]
        background_level = float(numpy.nanmedian(finite_annulus)) if finite_annulus.size else 0.0
        background_sigma = float(numpy.nanstd(finite_annulus, ddof=1)) if finite_annulus.size >= 2 else float("nan")
        raw_sum = float(numpy.nansum(finite_aperture))
        net_flux = float(numpy.nansum(finite_aperture - background_level)) if finite_aperture.size else float("nan")
        if not numpy.isfinite(background_sigma) or background_sigma == 0.0:
            aperture_snr = net_flux
        else:
            aperture_snr = net_flux / (background_sigma * float(numpy.sqrt(max(1, finite_aperture.size))))
        rows.append(
            {
                "track_label": track_label,
                "frame_index": float(frame_index),
                "time_tdb_jd_minus_2457000": float(time_array[frame_index]),
                "track_x": x0,
                "track_y": y0,
                "center_pixel_x": float(center_x),
                "center_pixel_y": float(center_y),
                "half_size": float(half_size),
                "aperture_npix": float(finite_aperture.size),
                "raw_sum": raw_sum,
                "background_level": background_level,
                "background_sigma": background_sigma,
                "net_flux": net_flux,
                "aperture_snr": float(aperture_snr),
            }
        )
    return pandas.DataFrame(rows)


def extract_box_aperture_ledger(
    cube: Any,
    time: Iterable[float],
    x: Iterable[float],
    y: Iterable[float],
    sample_indices: Iterable[int],
    half_size: int = 1,
    annulus_inner_radius: float = 3.0,
    annulus_outer_radius: float = 5.0,
    track_label: str = "track",
) -> dict[str, Any]:
    """Return per-pixel ledger rows for a small cadence sample."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    cube_array = numpy.asarray(cube, dtype=float)
    time_array = numpy.asarray(time, dtype=float)
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    series = compute_box_aperture_time_series(
        cube_array,
        time_array,
        x_array,
        y_array,
        half_size=half_size,
        annulus_inner_radius=annulus_inner_radius,
        annulus_outer_radius=annulus_outer_radius,
        track_label=track_label,
    )
    ny, nx = cube_array.shape[1:]
    rows: list[dict[str, float | str]] = []
    for frame_index in [int(value) for value in sample_indices]:
        x0 = float(x_array[frame_index])
        y0 = float(y_array[frame_index])
        center_x = int(round(x0))
        center_y = int(round(y0))
        x_min = max(0, center_x - int(half_size))
        x_max = min(nx, center_x + int(half_size) + 1)
        y_min = max(0, center_y - int(half_size))
        y_max = min(ny, center_y + int(half_size) + 1)
        summary_row = series.iloc[frame_index]
        for pixel_y in range(y_min, y_max):
            for pixel_x in range(x_min, x_max):
                rows.append(
                    {
                        "track_label": track_label,
                        "frame_index": float(frame_index),
                        "time_tdb_jd_minus_2457000": float(time_array[frame_index]),
                        "track_x": x0,
                        "track_y": y0,
                        "center_pixel_x": float(center_x),
                        "center_pixel_y": float(center_y),
                        "rel_pixel_x": float(pixel_x - center_x),
                        "rel_pixel_y": float(pixel_y - center_y),
                        "pixel_x": float(pixel_x),
                        "pixel_y": float(pixel_y),
                        "pixel_value": float(cube_array[frame_index, pixel_y, pixel_x]),
                        "box_raw_sum": float(summary_row["raw_sum"]),
                        "background_level": float(summary_row["background_level"]),
                        "background_sigma": float(summary_row["background_sigma"]),
                        "net_flux": float(summary_row["net_flux"]),
                        "aperture_snr": float(summary_row["aperture_snr"]),
                    }
                )
    return {
        "time_series": series,
        "ledger_rows": pandas.DataFrame(rows),
    }


def summarize_flux_capture(
    direct_nominal: Any,
    direct_calibrated: Any,
    box3_nominal: Any,
    box3_calibrated: Any,
    box5_nominal: Any,
    box5_calibrated: Any,
) -> dict[str, Any]:
    """Summarize how the direct moving-aperture path compares to simple box apertures."""

    pandas = _require_pandas()
    direct_nominal_df = pandas.DataFrame(direct_nominal)
    direct_calibrated_df = pandas.DataFrame(direct_calibrated)
    box3_nominal_df = pandas.DataFrame(box3_nominal)
    box3_calibrated_df = pandas.DataFrame(box3_calibrated)
    box5_nominal_df = pandas.DataFrame(box5_nominal)
    box5_calibrated_df = pandas.DataFrame(box5_calibrated)

    def _mean_ratio(numerator: Any, denominator: Any) -> float:
        denom = float(denominator["net_flux"].mean())
        if denom == 0.0:
            return float("nan")
        return float(numerator["net_flux"].mean() / denom)

    return {
        "direct_nominal_mean_net_flux": float(direct_nominal_df["net_flux"].mean()),
        "direct_calibrated_mean_net_flux": float(direct_calibrated_df["net_flux"].mean()),
        "box3_nominal_mean_net_flux": float(box3_nominal_df["net_flux"].mean()),
        "box3_calibrated_mean_net_flux": float(box3_calibrated_df["net_flux"].mean()),
        "box5_nominal_mean_net_flux": float(box5_nominal_df["net_flux"].mean()),
        "box5_calibrated_mean_net_flux": float(box5_calibrated_df["net_flux"].mean()),
        "direct_nominal_to_box3_nominal_ratio": _mean_ratio(direct_nominal_df, box3_nominal_df),
        "direct_nominal_to_box5_nominal_ratio": _mean_ratio(direct_nominal_df, box5_nominal_df),
        "direct_calibrated_to_box3_calibrated_ratio": _mean_ratio(direct_calibrated_df, box3_calibrated_df),
        "direct_calibrated_to_box5_calibrated_ratio": _mean_ratio(direct_calibrated_df, box5_calibrated_df),
    }


def write_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return destination


def plot_selected_pixel_time_series(
    pixel_table: Any,
    output_path: str | Path,
) -> Path:
    """Plot a compact panel of selected raw-pixel time series."""

    plotting = _require_matplotlib()
    pandas = _require_pandas()
    table = pandas.DataFrame(pixel_table)
    labels = list(table["label"].drop_duplicates())
    n_panels = len(labels)
    ncols = 2
    nrows = max(1, int(np.ceil(n_panels / ncols)))
    figure, axes = plotting.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True, constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    for axis, label in zip(axes_array, labels, strict=False):
        subset = table[table["label"] == label].copy()
        axis.plot(subset["time_tdb_jd_minus_2457000"], subset["value"], color="black", linewidth=1.0)
        near_cal = subset[subset["calibrated_within_0p5"].astype(bool)]
        near_nom = subset[subset["nominal_within_0p5"].astype(bool)]
        axis.scatter(
            near_nom["time_tdb_jd_minus_2457000"],
            near_nom["value"],
            color="#1f77b4",
            s=10,
            label="nominal <= 0.5 px",
        )
        axis.scatter(
            near_cal["time_tdb_jd_minus_2457000"],
            near_cal["value"],
            color="#ff7f0e",
            s=10,
            label="calibrated <= 0.5 px",
        )
        rejected = subset[~subset["quality_keep"].astype(bool)]
        if not rejected.empty:
            axis.scatter(
                rejected["time_tdb_jd_minus_2457000"],
                rejected["value"],
                marker="x",
                color="red",
                s=12,
                label="rejected",
            )
        axis.set_title(f"{label} ({int(subset.iloc[0]['pixel_x'])}, {int(subset.iloc[0]['pixel_y'])})")
        axis.set_xlabel("BTJD")
        axis.set_ylabel("Pixel value")
        axis.grid(alpha=0.3)
    for axis in axes_array[n_panels:]:
        axis.axis("off")
    handles, labels_text = axes_array[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels_text, loc="upper right", frameon=False)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plotting.close(figure)
    return destination


def plot_pixel_residency_maps(
    nominal_map: Any,
    calibrated_map: Any,
    nominal_x: Iterable[float],
    nominal_y: Iterable[float],
    calibrated_x: Iterable[float],
    calibrated_y: Iterable[float],
    output_path: str | Path,
) -> Path:
    """Plot nominal and calibrated pixel-residency maps side by side."""

    plotting = _require_matplotlib()
    numpy = _require_numpy()
    nominal_image = numpy.asarray(nominal_map, dtype=float)
    calibrated_image = numpy.asarray(calibrated_map, dtype=float)
    figure, axes = plotting.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    vmax = float(numpy.nanmax([nominal_image.max(), calibrated_image.max(), 1.0]))
    for axis, image, title, tx, ty in (
        (axes[0], nominal_image, "Nominal Residency", nominal_x, nominal_y),
        (axes[1], calibrated_image, "Calibrated Residency", calibrated_x, calibrated_y),
    ):
        artist = axis.imshow(image, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        axis.plot(tx, ty, color="white", linewidth=1.0, alpha=0.9)
        axis.set_title(title)
        axis.set_xlabel("Local x [pix]")
        axis.set_ylabel("Local y [pix]")
        figure.colorbar(artist, ax=axis, shrink=0.85)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plotting.close(figure)
    return destination


def plot_aperture_ledger_waterfall(
    ledger_rows: Any,
    output_path: str | Path,
) -> Path:
    """Plot a compact cadence-vs-pixel waterfall for the simple box aperture."""

    plotting = _require_matplotlib()
    pandas = _require_pandas()
    table = pandas.DataFrame(ledger_rows)
    track_labels = list(table["track_label"].drop_duplicates())
    figure, axes = plotting.subplots(1, len(track_labels), figsize=(6 * len(track_labels), 5), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    for axis, track_label in zip(axes_array, track_labels, strict=False):
        subset = table[table["track_label"] == track_label].copy()
        subset["offset_label"] = subset.apply(
            lambda row: f"dx={int(row['rel_pixel_x']):+d},dy={int(row['rel_pixel_y']):+d}",
            axis=1,
        )
        pivot = subset.pivot_table(
            index="frame_index",
            columns="offset_label",
            values="pixel_value",
            aggfunc="first",
        ).sort_index()
        artist = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
        axis.set_title(f"{track_label} 3x3 Box Ledger")
        axis.set_xlabel("Relative aperture pixel")
        axis.set_ylabel("Ledger cadence")
        axis.set_xticks(range(pivot.shape[1]))
        axis.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
        figure.colorbar(artist, ax=axis, shrink=0.85)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plotting.close(figure)
    return destination


def plot_flux_capture_comparison(
    direct_nominal: Any,
    direct_calibrated: Any,
    box3_nominal: Any,
    box3_calibrated: Any,
    box5_nominal: Any,
    box5_calibrated: Any,
    stack_growth_table: Any,
    output_path: str | Path,
) -> Path:
    """Plot transparent direct/aperture comparisons against stack growth."""

    plotting = _require_matplotlib()
    pandas = _require_pandas()
    direct_nominal_df = pandas.DataFrame(direct_nominal)
    direct_calibrated_df = pandas.DataFrame(direct_calibrated)
    box3_nominal_df = pandas.DataFrame(box3_nominal)
    box3_calibrated_df = pandas.DataFrame(box3_calibrated)
    box5_nominal_df = pandas.DataFrame(box5_nominal)
    box5_calibrated_df = pandas.DataFrame(box5_calibrated)
    growth_df = pandas.DataFrame(stack_growth_table)

    figure, axes = plotting.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].plot(direct_nominal_df["frame_index"], direct_nominal_df["net_flux"].cumsum(), label="direct nominal")
    axes[0].plot(direct_calibrated_df["frame_index"], direct_calibrated_df["net_flux"].cumsum(), label="direct calibrated-x")
    axes[0].plot(box3_nominal_df["frame_index"], box3_nominal_df["net_flux"].cumsum(), label="box3 nominal")
    axes[0].plot(box3_calibrated_df["frame_index"], box3_calibrated_df["net_flux"].cumsum(), label="box3 calibrated-x")
    axes[0].plot(box5_nominal_df["frame_index"], box5_nominal_df["net_flux"].cumsum(), label="box5 nominal", alpha=0.7)
    axes[0].plot(box5_calibrated_df["frame_index"], box5_calibrated_df["net_flux"].cumsum(), label="box5 calibrated-x", alpha=0.7)
    axes[0].set_title("Cumulative Transparent Aperture Flux")
    axes[0].set_xlabel("Frame index")
    axes[0].set_ylabel("Cumulative net flux")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(growth_df["n_frames"], growth_df["nominal_stack_score"], marker="o", label="stack nominal")
    axes[1].plot(
        growth_df["n_frames"],
        growth_df["calibrated_stack_score"],
        marker="o",
        label="stack calibrated-x",
    )
    axes[1].set_title("Stack Growth Checkpoints")
    axes[1].set_xlabel("Frames included")
    axes[1].set_ylabel("Stack aperture S/N")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plotting.close(figure)
    return destination
