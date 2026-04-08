"""Minimal helpers for reproducing the high-level tess_asteroids workflow.

This module does not replace the repo's custom recovery path. It is a bounded
benchmark layer meant to answer:

- does the package's own high-level `make_tpf() -> make_lc() -> plot_lc()` path
  run here at all?
- does that path behave cleanly on the same frozen Pluto detector-local subset?

Optional dependencies remain optional. If tess_asteroids is unavailable or
its network/cached data requirements are not met, the helpers return structured
failure summaries instead of pretending success.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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

from kbo_tess.tess_loader import _import_tess_asteroids, subset_ephemeris_for_build


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for tess_asteroids tutorial benchmarks. Install project "
            "dependencies with `python -m pip install -e .`."
        )
    return np


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for tess_asteroids tutorial benchmarks. Install project "
            "dependencies with `python -m pip install -e .`."
        )
    return pd


def _require_matplotlib() -> Any:
    if plt is None:
        raise ImportError(
            "matplotlib is required for tutorial benchmark plots. Install project dependencies "
            "with `python -m pip install -e .`."
        )
    return plt


def _configure_benchmark_environment() -> None:
    """Keep optional tess_asteroids caches inside the project workspace."""

    project_root = Path(__file__).resolve().parents[2]
    use_system_home = os.environ.get("KBO_TESS_USE_SYSTEM_HOME", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    optional_home = project_root / ".optional_home"
    optional_home.mkdir(parents=True, exist_ok=True)
    cache_root = project_root / ".astropy_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    if not use_system_home:
        os.environ["HOME"] = str(optional_home)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("ASTROPY_CACHE_DIR", str(cache_root))


def _ephem_for_tess_asteroids(normalized_ephem: Any) -> Any:
    pandas = _require_pandas()
    columns = {
        "time": normalized_ephem["time_tdb_jd_minus_2457000"].astype(float),
        "sector": normalized_ephem["sector"].astype(int),
        "camera": normalized_ephem["camera"].astype(int),
        "ccd": normalized_ephem["ccd"].astype(int),
        "column": normalized_ephem["x"].astype(float),
        "row": normalized_ephem["y"].astype(float),
    }
    for optional_name in ("vmag", "hmag", "sun_distance", "obs_distance", "sto_angle"):
        if optional_name in normalized_ephem.columns:
            columns[optional_name] = normalized_ephem[optional_name]
    return pandas.DataFrame(columns)


def _safe_plot_lc(moving: Any, output_path: str | Path) -> Path | None:
    try:
        figure = moving.plot_lc(save=False)
    except Exception:
        return None
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    return destination


def write_status_plot(title: str, message: str, output_path: str | Path) -> Path:
    """Write a simple text-only PNG when the official path is unavailable."""

    plotting = _require_matplotlib()
    figure, axis = plotting.subplots(figsize=(8, 3.5), constrained_layout=True)
    axis.axis("off")
    axis.text(0.02, 0.88, title, fontsize=14, weight="bold", transform=axis.transAxes, va="top")
    axis.text(0.02, 0.72, message, fontsize=10, transform=axis.transAxes, va="top", wrap=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plotting.close(figure)
    return destination


def _lightcurve_method_summary(lightcurve: dict[str, Any]) -> dict[str, Any]:
    numpy = _require_numpy()
    summary: dict[str, Any] = {}
    for method_name, payload in lightcurve.items():
        if not isinstance(payload, dict) or "flux" not in payload:
            continue
        flux = numpy.asarray(payload.get("flux", []), dtype=float)
        time = numpy.asarray(payload.get("time", []), dtype=float)
        quality = numpy.asarray(payload.get("quality", []))
        finite_flux = flux[numpy.isfinite(flux)]
        summary[str(method_name)] = {
            "n_points": int(time.size),
            "finite_flux_points": int(finite_flux.size),
            "flux_median": float(numpy.nanmedian(flux)) if flux.size else float("nan"),
            "flux_max": float(numpy.nanmax(flux)) if flux.size else float("nan"),
            "flux_min": float(numpy.nanmin(flux)) if flux.size else float("nan"),
            "quality_nonzero": int(numpy.sum(quality != 0)) if quality.size else 0,
        }
    return summary


def summarize_movingtpf(moving: Any) -> dict[str, Any]:
    numpy = _require_numpy()
    corr_flux = numpy.asarray(getattr(moving, "corr_flux", []), dtype=float)
    time = numpy.asarray(getattr(moving, "time", []), dtype=float)
    quality = numpy.asarray(getattr(moving, "quality", []))
    summary = {
        "target": str(getattr(moving, "target", "unknown")),
        "shape": list(getattr(moving, "shape", ())),
        "barycentric": bool(getattr(moving, "barycentric", True)),
        "n_cadences": int(time.size),
        "time_start": float(numpy.nanmin(time)) if time.size else float("nan"),
        "time_stop": float(numpy.nanmax(time)) if time.size else float("nan"),
        "quality_nonzero": int(numpy.sum(quality != 0)) if quality.size else 0,
        "corr_flux_shape": list(corr_flux.shape),
        "ap_method": getattr(moving, "ap_method", None),
        "bg_method": getattr(moving, "bg_method", None),
        "lightcurve_methods": _lightcurve_method_summary(getattr(moving, "lc", {})),
    }
    return summary


def attempt_docs_quickstart_benchmark(
    target: str = "1980 VR1",
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    shape: tuple[int, int] = (11, 11),
    outdir: str | Path | None = None,
) -> dict[str, Any]:
    """Try the README-style `from_name()` quickstart benchmark."""

    _configure_benchmark_environment()
    try:
        module = _import_tess_asteroids()
    except Exception as exc:
        return {
            "status": "failed",
            "stage": "import",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    try:
        moving = module.MovingTPF.from_name(target, sector=sector, camera=camera, ccd=ccd)
        moving.make_tpf(shape=shape, save=False)
        moving.make_lc(save=False)
        summary = summarize_movingtpf(moving)
        plot_path = None
        if outdir is not None:
            plot_path = _safe_plot_lc(
                moving,
                Path(outdir) / f"docs_quickstart_{str(target).replace(' ', '_').lower()}_lc.png",
            )
        return {
            "status": "success",
            "target": target,
            "sector": int(sector),
            "camera": int(camera),
            "ccd": int(ccd),
            "summary": summary,
            "plot_path": str(plot_path) if plot_path is not None else None,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "stage": "from_name_or_lightcurve",
            "target": target,
            "sector": int(sector),
            "camera": int(camera),
            "ccd": int(ccd),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def run_custom_ephem_high_level_path(
    ephem_input: Any,
    target: str,
    shape: tuple[int, int] = (21, 21),
    barycentric: bool = True,
    bg_method: str = "rolling",
    outdir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the official high-level MovingTPF path on a local normalized ephemeris."""

    _configure_benchmark_environment()
    normalized_ephem = subset_ephemeris_for_build(ephem_input, target=target)
    tess_ephem = _ephem_for_tess_asteroids(normalized_ephem)
    module = _import_tess_asteroids()
    moving = module.MovingTPF(target=target, ephem=tess_ephem, barycentric=bool(barycentric), metadata={})
    moving.make_tpf(shape=tuple(int(value) for value in shape), bg_method=bg_method, save=False)
    moving.make_lc(save=False)
    plot_path = None
    if outdir is not None:
        plot_path = _safe_plot_lc(
            moving,
            Path(outdir) / f"{str(target).replace(' ', '_').lower()}_official_simple_lc.png",
        )

    lightcurve_tables: dict[str, Any] = {}
    pandas = _require_pandas()
    for method_name, payload in getattr(moving, "lc", {}).items():
        if not isinstance(payload, dict):
            continue
        columns: dict[str, Any] = {}
        n_points = None
        for key, value in payload.items():
            try:
                array = np.asarray(value)
            except Exception:
                continue
            if array.ndim != 1:
                continue
            if n_points is None:
                n_points = array.shape[0]
            if array.shape[0] != n_points:
                continue
            columns[str(key)] = array
        if columns:
            lightcurve_tables[str(method_name)] = pandas.DataFrame(columns)

    return {
        "summary": summarize_movingtpf(moving),
        "plot_path": str(plot_path) if plot_path is not None else None,
        "lightcurve_tables": lightcurve_tables,
        "ephemeris_rows": int(len(tess_ephem)),
    }


def write_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return destination
