"""Geometry diagnostics for detector-local moving-cutout bundles."""

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

from kbo_tess.recovery import align_ephemeris_to_cube_bundle


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for geometry diagnostics. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return np


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for geometry diagnostics. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return pd


def _require_matplotlib() -> Any:
    if plt is None:
        raise ImportError(
            "matplotlib is required for geometry diagnostics. Install project dependencies "
            "with `python -m pip install -e .`."
        )
    return plt


def _residual_stats(values: Any) -> dict[str, float]:
    numpy = _require_numpy()
    array = numpy.asarray(values, dtype=float)
    finite = array[numpy.isfinite(array)]
    if finite.size == 0:
        return {"median": float("nan"), "max": float("nan")}
    return {"median": float(numpy.nanmedian(finite)), "max": float(numpy.nanmax(finite))}


def evaluate_geometry_consistency(
    ephem_df: Any,
    cube_bundle: dict[str, Any],
    median_threshold_px: float = 0.1,
    max_threshold_px: float = 0.5,
) -> dict[str, Any]:
    """Compare global and local geometry conventions for a moving cutout bundle."""

    numpy = _require_numpy()
    aligned = align_ephemeris_to_cube_bundle(ephem_df, cube_bundle)
    if "bundle_local_x" not in aligned.columns:
        raise ValueError(
            "Cube bundle is missing `local_x`/`local_y`, so the geometry check cannot compare "
            "the detector track against cutout-local predictions."
        )

    corner_stats = (
        _residual_stats(aligned["local_corner_residual_r"])
        if "local_corner_residual_r" in aligned.columns
        else {"median": float("nan"), "max": float("nan")}
    )
    poscorr_stats = (
        _residual_stats(aligned["local_poscorr_residual_r"])
        if "local_poscorr_residual_r" in aligned.columns
        else {"median": float("nan"), "max": float("nan")}
    )
    detector_stats = (
        _residual_stats(aligned["detector_residual_r"])
        if "detector_residual_r" in aligned.columns
        else {"median": float("nan"), "max": float("nan")}
    )

    geometry_passed = True
    for stats in (corner_stats, poscorr_stats):
        if not numpy.isfinite(stats["median"]) or not numpy.isfinite(stats["max"]):
            geometry_passed = False
            continue
        if stats["median"] > float(median_threshold_px) or stats["max"] > float(max_threshold_px):
            geometry_passed = False

    summary = {
        "target": str(aligned.iloc[0]["target"]) if "target" in aligned.columns else cube_bundle.get("metadata", {}).get("target", "unknown"),
        "sector": int(aligned.iloc[0]["sector"]) if "sector" in aligned.columns else cube_bundle.get("metadata", {}).get("sector"),
        "camera": int(aligned.iloc[0]["camera"]) if "camera" in aligned.columns else cube_bundle.get("metadata", {}).get("camera"),
        "ccd": int(aligned.iloc[0]["ccd"]) if "ccd" in aligned.columns else cube_bundle.get("metadata", {}).get("ccd"),
        "n_cadences": int(len(aligned)),
        "time_start": float(aligned["time_tdb_jd_minus_2457000"].min()),
        "time_stop": float(aligned["time_tdb_jd_minus_2457000"].max()),
        "barycentric_mode": cube_bundle.get("metadata", {}).get("barycentric_mode"),
        "bg_method": cube_bundle.get("metadata", {}).get("loader_bg_method"),
        "median_geometry_residual_corner_px": corner_stats["median"],
        "max_geometry_residual_corner_px": corner_stats["max"],
        "median_geometry_residual_poscorr_px": poscorr_stats["median"],
        "max_geometry_residual_poscorr_px": poscorr_stats["max"],
        "median_detector_residual_px": detector_stats["median"],
        "max_detector_residual_px": detector_stats["max"],
        "geometry_check_passed": bool(geometry_passed),
    }
    return {
        "aligned_ephemeris": aligned,
        "summary": summary,
        "cube_bundle": cube_bundle,
    }


def plot_geometry_diagnostics(geometry_result: dict[str, Any], output_path: str | Path) -> Path:
    """Write a geometry diagnostic PNG."""

    plotting = _require_matplotlib()
    numpy = _require_numpy()

    cube = numpy.asarray(geometry_result["cube_bundle"]["cube"], dtype=float)
    aligned = geometry_result["aligned_ephemeris"]
    bundle_local_x = numpy.asarray(aligned["bundle_local_x"], dtype=float)
    bundle_local_y = numpy.asarray(aligned["bundle_local_y"], dtype=float)
    frame_index = int(len(cube) // 2)
    frame = cube[frame_index]

    figure, axes = plotting.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    finite = frame[numpy.isfinite(frame)]
    if finite.size:
        vmin, vmax = numpy.nanpercentile(finite, [5, 99])
    else:
        vmin, vmax = -1.0, 1.0
    axes[0, 0].imshow(frame, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].plot(bundle_local_x, bundle_local_y, color="white", linewidth=1.0, alpha=0.8)
    axes[0, 0].scatter(
        [bundle_local_x[frame_index]],
        [bundle_local_y[frame_index]],
        color="red",
        marker="x",
        s=60,
        linewidths=1.5,
    )
    axes[0, 0].set_title("Representative Real Cutout Frame")
    axes[0, 0].set_xlabel("Local x [pix]")
    axes[0, 0].set_ylabel("Local y [pix]")

    time_axis = numpy.asarray(aligned["time_tdb_jd_minus_2457000"], dtype=float)
    if "local_corner_residual_r" in aligned.columns:
        axes[0, 1].plot(time_axis, aligned["local_corner_residual_r"], label="corner mapping")
    if "local_poscorr_residual_r" in aligned.columns:
        axes[0, 1].plot(time_axis, aligned["local_poscorr_residual_r"], label="center + POS_CORR")
    axes[0, 1].set_title("Local Geometry Residuals")
    axes[0, 1].set_xlabel("BTJD")
    axes[0, 1].set_ylabel("Residual [pix]")
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(alpha=0.3)

    if "detector_residual_r" in aligned.columns:
        axes[1, 0].plot(time_axis, aligned["detector_residual_r"], color="black")
        axes[1, 0].set_title("Detector-Track Residual")
        axes[1, 0].set_xlabel("BTJD")
        axes[1, 0].set_ylabel("Residual [pix]")
        axes[1, 0].grid(alpha=0.3)
    else:
        axes[1, 0].axis("off")

    summary = geometry_result["summary"]
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Cadences: {summary['n_cadences']}",
                f"BTJD span: {summary['time_start']:.5f} -> {summary['time_stop']:.5f}",
                f"Barycentric mode: {summary['barycentric_mode']}",
                f"Loader BG: {summary['bg_method']}",
                f"Corner median/max: {summary['median_geometry_residual_corner_px']:.4f} / {summary['max_geometry_residual_corner_px']:.4f}",
                f"POS_CORR median/max: {summary['median_geometry_residual_poscorr_px']:.4f} / {summary['max_geometry_residual_poscorr_px']:.4f}",
                f"Geometry passed: {summary['geometry_check_passed']}",
            ]
        ),
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plotting.close(figure)
    return destination


def write_geometry_products(
    geometry_result: dict[str, Any],
    outdir: str | Path,
    prefix: str | None = None,
) -> dict[str, Path]:
    """Write geometry summary, residual table, and diagnostic PNG."""

    pandas = _require_pandas()
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = dict(geometry_result["summary"])
    if prefix is None:
        target = re.sub(r"[^a-z0-9]+", "_", str(summary.get("target", "target")).lower()).strip("_")
        sector = summary.get("sector", "unknown")
        prefix = f"{target}_sector{sector}"

    summary_path = output_dir / f"{prefix}_geometry_summary.json"
    residual_csv_path = output_dir / f"{prefix}_geometry_residuals.csv"
    diagnostic_path = output_dir / f"{prefix}_geometry.png"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    pandas.DataFrame(geometry_result["aligned_ephemeris"]).to_csv(residual_csv_path, index=False)
    plot_geometry_diagnostics(geometry_result, diagnostic_path)
    return {
        "summary_json": summary_path,
        "residual_csv": residual_csv_path,
        "diagnostic_png": diagnostic_path,
    }
