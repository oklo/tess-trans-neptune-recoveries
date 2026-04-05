"""Simple moving-aperture photometry for detector-local TESS bundles."""

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

from kbo_tess.background import subtract_background
from kbo_tess.recovery import (
    align_ephemeris_to_cube_bundle,
    filter_cube_bundle,
    resolve_track_coordinates,
)


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for moving-aperture photometry. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return np


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for moving-aperture photometry. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return pd


def _require_matplotlib() -> Any:
    if plt is None:
        raise ImportError(
            "matplotlib is required for moving-aperture photometry diagnostics. Install project "
            "dependencies with `python -m pip install -e .`."
        )
    return plt


def aperture_photometry_at_position(
    image: Any,
    position: tuple[float, float],
    aperture_radius: float = 1.5,
    annulus_inner_radius: float = 3.0,
    annulus_outer_radius: float = 5.0,
) -> dict[str, float]:
    """Measure net aperture flux and a simple annulus-based aperture S/N."""

    numpy = _require_numpy()
    frame = numpy.asarray(image, dtype=float)
    x0, y0 = float(position[0]), float(position[1])
    yy, xx = numpy.indices(frame.shape, dtype=float)
    rr = numpy.hypot(xx - x0, yy - y0)
    aperture_mask = rr <= float(aperture_radius)
    annulus_mask = (rr >= float(annulus_inner_radius)) & (rr <= float(annulus_outer_radius))

    aperture_values = frame[aperture_mask]
    annulus_values = frame[annulus_mask]
    finite_aperture = aperture_values[numpy.isfinite(aperture_values)]
    finite_annulus = annulus_values[numpy.isfinite(annulus_values)]
    if finite_aperture.size == 0:
        return {
            "background_level": float("nan"),
            "background_sigma": float("nan"),
            "net_flux": float("nan"),
            "aperture_snr": float("nan"),
            "aperture_npix": 0.0,
            "annulus_npix": float(finite_annulus.size),
        }

    background_level = float(numpy.nanmedian(finite_annulus)) if finite_annulus.size else 0.0
    background_sigma = float(numpy.nanstd(finite_annulus, ddof=1)) if finite_annulus.size >= 2 else float("nan")
    net_flux = float(numpy.nansum(finite_aperture - background_level))
    if not numpy.isfinite(background_sigma) or background_sigma == 0.0:
        aperture_snr = net_flux
    else:
        aperture_snr = net_flux / (background_sigma * float(numpy.sqrt(max(1, finite_aperture.size))))
    return {
        "background_level": background_level,
        "background_sigma": background_sigma,
        "net_flux": net_flux,
        "aperture_snr": float(aperture_snr),
        "aperture_npix": float(finite_aperture.size),
        "annulus_npix": float(finite_annulus.size),
    }


def run_moving_aperture_photometry(
    ephem_df: Any,
    cube_bundle: dict[str, Any],
    background_method: str = "none",
    aperture_radius: float = 1.5,
    annulus_inner_radius: float = 3.0,
    annulus_outer_radius: float = 5.0,
) -> dict[str, Any]:
    """Measure direct moving-aperture photometry along the predicted local track."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    filtered_bundle = filter_cube_bundle(cube_bundle)
    cube = numpy.asarray(filtered_bundle["cube"], dtype=float)
    metadata = dict(filtered_bundle.get("metadata", {}))
    aligned_ephem = align_ephemeris_to_cube_bundle(ephem_df, filtered_bundle)
    x, y, track_coordinate_system = resolve_track_coordinates(aligned_ephem)

    baseline_cube, analysis_cube = subtract_background(cube, method=background_method)
    rows: list[dict[str, float]] = []
    for index, frame in enumerate(analysis_cube):
        phot = aperture_photometry_at_position(
            frame,
            position=(float(x[index]), float(y[index])),
            aperture_radius=aperture_radius,
            annulus_inner_radius=annulus_inner_radius,
            annulus_outer_radius=annulus_outer_radius,
        )
        rows.append(
            {
                "frame_index": float(index),
                "time_tdb_jd_minus_2457000": float(aligned_ephem.iloc[index]["time_tdb_jd_minus_2457000"]),
                "local_x": float(x[index]),
                "local_y": float(y[index]),
                "net_flux": float(phot["net_flux"]),
                "aperture_snr": float(phot["aperture_snr"]),
                "background_level": float(phot["background_level"]),
                "background_sigma": float(phot["background_sigma"]),
                "aperture_npix": float(phot["aperture_npix"]),
                "annulus_npix": float(phot["annulus_npix"]),
            }
        )

    lightcurve = pandas.DataFrame(rows)
    representative_index = int(len(analysis_cube) // 2)
    summary = {
        "target": str(aligned_ephem.iloc[0]["target"]) if "target" in aligned_ephem.columns else metadata.get("target", "unknown"),
        "sector": int(aligned_ephem.iloc[0]["sector"]) if "sector" in aligned_ephem.columns else metadata.get("sector"),
        "camera": int(aligned_ephem.iloc[0]["camera"]) if "camera" in aligned_ephem.columns else metadata.get("camera"),
        "ccd": int(aligned_ephem.iloc[0]["ccd"]) if "ccd" in aligned_ephem.columns else metadata.get("ccd"),
        "n_frames": int(len(lightcurve)),
        "time_start": float(lightcurve["time_tdb_jd_minus_2457000"].min()),
        "time_stop": float(lightcurve["time_tdb_jd_minus_2457000"].max()),
        "background_method": background_method,
        "loader_bg_method": metadata.get("loader_bg_method"),
        "track_coordinate_system": track_coordinate_system,
        "median_net_flux": float(numpy.nanmedian(lightcurve["net_flux"])),
        "max_net_flux": float(numpy.nanmax(lightcurve["net_flux"])),
        "median_aperture_snr": float(numpy.nanmedian(lightcurve["aperture_snr"])),
        "max_aperture_snr": float(numpy.nanmax(lightcurve["aperture_snr"])),
        "representative_frame_index": representative_index,
    }
    return {
        "cube_bundle": filtered_bundle,
        "aligned_ephemeris": aligned_ephem,
        "baseline_cube": baseline_cube,
        "analysis_cube": analysis_cube,
        "lightcurve": lightcurve,
        "summary": summary,
        "representative_frame": analysis_cube[representative_index],
        "representative_position": (float(x[representative_index]), float(y[representative_index])),
    }


def plot_moving_aperture_diagnostics(result: dict[str, Any], output_path: str | Path) -> Path:
    """Write a compact moving-aperture diagnostic PNG."""

    plotting = _require_matplotlib()
    numpy = _require_numpy()
    lightcurve = result["lightcurve"]
    frame = numpy.asarray(result["representative_frame"], dtype=float)
    x = numpy.asarray(result["aligned_ephemeris"]["time_tdb_jd_minus_2457000"], dtype=float)
    track_x = numpy.asarray(result["aligned_ephemeris"]["bundle_local_x"], dtype=float) if "bundle_local_x" in result["aligned_ephemeris"].columns else None
    track_y = numpy.asarray(result["aligned_ephemeris"]["bundle_local_y"], dtype=float) if "bundle_local_y" in result["aligned_ephemeris"].columns else None

    figure, axes = plotting.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    finite = frame[numpy.isfinite(frame)]
    vmin, vmax = (numpy.nanpercentile(finite, [5, 99]) if finite.size else (-1.0, 1.0))
    artist = axes[0].imshow(frame, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    if track_x is not None and track_y is not None:
        axes[0].plot(track_x, track_y, color="white", linewidth=1.0)
    axes[0].scatter([result["representative_position"][0]], [result["representative_position"][1]], color="red", marker="x", s=60)
    axes[0].set_title("Representative Frame")
    axes[0].set_xlabel("Local x [pix]")
    axes[0].set_ylabel("Local y [pix]")
    figure.colorbar(artist, ax=axes[0], shrink=0.8)

    axes[1].plot(x, lightcurve["net_flux"].astype(float), marker="o", linewidth=1.0)
    axes[1].set_title("Moving-Aperture Net Flux")
    axes[1].set_xlabel("BTJD")
    axes[1].set_ylabel("Net flux")
    axes[1].grid(alpha=0.3)

    axes[2].plot(x, lightcurve["aperture_snr"].astype(float), marker="o", linewidth=1.0)
    axes[2].set_title("Moving-Aperture S/N")
    axes[2].set_xlabel("BTJD")
    axes[2].set_ylabel("Aperture S/N")
    axes[2].grid(alpha=0.3)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plotting.close(figure)
    return destination


def write_moving_aperture_products(
    result: dict[str, Any],
    outdir: str | Path,
    prefix: str | None = None,
) -> dict[str, Path]:
    """Write moving-aperture summary, light curve CSV, and diagnostic PNG."""

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(result["summary"])
    if prefix is None:
        target = re.sub(r"[^a-z0-9]+", "_", str(summary.get("target", "target")).lower()).strip("_")
        sector = summary.get("sector", "unknown")
        prefix = f"{target}_sector{sector}"

    summary_path = output_dir / f"{prefix}_photometry_summary.json"
    lightcurve_path = output_dir / f"{prefix}_lightcurve.csv"
    diagnostic_path = output_dir / f"{prefix}_photometry.png"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    result["lightcurve"].to_csv(lightcurve_path, index=False)
    plot_moving_aperture_diagnostics(result, diagnostic_path)
    return {
        "summary_json": summary_path,
        "lightcurve_csv": lightcurve_path,
        "diagnostic_png": diagnostic_path,
    }
