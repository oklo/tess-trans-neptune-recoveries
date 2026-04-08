"""Real-data adapters for detector-local TESS moving cutouts.

The authoritative handoff into the recovery pipeline is a local cube bundle.
This module bridges a normalized ephemeris CSV/DataFrame to that bundle using
`tess_asteroids.MovingTPF` when the optional dependency is available.

Expected input ephemeris columns after normalization:

- `time_tdb_jd_minus_2457000`: exact TDB time
- `time_reference`: `spacecraft` or `barycentric` for real-data work
- `sector`, `camera`, `ccd`: one unique detector
- `x`, `y`: one-indexed detector column/row coordinates

Bundle schema written to NPZ:

- `cube`: float array with shape `(n_frames, ny, nx)`
- `time_tdb_jd_minus_2457000`: float cadence array
- `local_x`, `local_y`: predicted source coordinates in cutout-local pixels
- `corner1`, `corner2`: one-indexed lower-left detector column/row of each cutout
- `pos_corr1`, `pos_corr2`: TPF-center offset convention used by `tess_asteroids`
- optional `quality_mask`, `cadence_quality`, `pixel_quality`
- optional timing / geometry diagnostic arrays
- `metadata_json`: JSON-serializable dict stored alongside arrays

The adapter keeps barycentric-vs-spacecraft semantics explicit. If the
ephemeris does not state its TDB reference frame, the real-data build path
raises instead of silently guessing.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    pd = None

from kbo_tess.ephem import (
    get_time_reference,
    load_ephemeris_csv,
    normalize_ephemeris_dataframe,
    validate_single_detector_track,
)

_DOWNLINKS_URL = "https://tess.mit.edu/public/files/TESS_orbit_times.csv"


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for TESS bundle materialization. Install the base project "
            "dependencies with `python -m pip install -e .`."
        )
    return np


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for TESS bundle materialization. Install the base project "
            "dependencies with `python -m pip install -e .`."
        )
    return pd


def _configure_optional_cache_dirs() -> None:
    """Keep optional package caches inside the project workspace when possible."""

    cache_root = Path(__file__).resolve().parents[2] / ".astropy_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("ASTROPY_CACHE_DIR", str(cache_root))


def _import_tess_asteroids() -> Any:
    _configure_optional_cache_dirs()
    try:
        import certifi  # type: ignore[import-not-found]

        if "SSL_CERT_FILE" not in os.environ:
            os.environ["SSL_CERT_FILE"] = certifi.where()
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    try:
        return importlib.import_module("tess_asteroids")
    except Exception as exc:  # pragma: no cover - depends on optional package and network state
        fallback_error: Exception | None = None
        try:
            return _import_tess_asteroids_with_cached_downlinks()
        except Exception as cached_exc:
            fallback_error = cached_exc
        detail = f"Original import error: {exc}"
        if fallback_error is not None:
            detail += f" Cached-downlinks fallback error: {fallback_error}"
        raise ImportError(
            "`tess_asteroids` is required for the real moving-cutout path. "
            "Install it with `python -m pip install -e .[tess]`. "
            "Current upstream releases are typically usable on Python 3.11-3.12, and "
            "the package may need network access on first import to fetch orbit-time metadata. "
            f"{detail}"
        ) from exc


def _resolve_tess_asteroids_cache_path() -> Path:
    spec = importlib.util.find_spec("tess_asteroids")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Could not locate the installed `tess_asteroids` package path.")
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    return package_dir / "data" / "TESS_orbit_times.csv"


def _resolve_lkspacecraft_url_cache_dir() -> Path:
    home_dir = Path(os.environ.get("HOME", str(Path.home())))
    return home_dir / ".lkspacecraft" / "cache" / "download" / "url"


def _build_cached_tess_models_listing() -> Any:
    """Reconstruct the STScI models index from the local lkspacecraft cache."""

    pandas = _require_pandas()
    url_cache_dir = _resolve_lkspacecraft_url_cache_dir()
    listing_rows: list[str] = []
    if url_cache_dir.exists():
        for url_path in sorted(url_cache_dir.glob("*/url")):
            try:
                cached_url = url_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if "https://archive.stsci.edu/missions/tess/models/TESS_EPH_DEF_" not in cached_url:
                continue
            filename = cached_url.rsplit("/", 1)[-1]
            listing_rows.append(f'<td><a href="{filename}">{filename}</a>')
    for legacy_name in ("TESS_EPH_DEF_2018004_01.bsp", "TESS_EPH_DEF_2018080_01.bsp"):
        listing_rows.append(f'<td><a href="{legacy_name}">{legacy_name}</a>')
    if not listing_rows:
        raise RuntimeError(
            "Could not reconstruct the cached TESS models index from the current "
            "lkspacecraft cache. Seed a local cache or use `KBO_TESS_USE_SYSTEM_HOME=1`."
        )
    return pandas.DataFrame(listing_rows)


def _download_downlinks_cache(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    contexts: list[Any] = []
    cert_file = os.environ.get("SSL_CERT_FILE")
    if cert_file:
        contexts.append(ssl.create_default_context(cafile=cert_file))
    contexts.append(ssl._create_unverified_context())
    for context in contexts:
        try:
            with urllib.request.urlopen(_DOWNLINKS_URL, context=context) as response:
                destination.write_bytes(response.read())
                return destination
        except Exception as exc:  # pragma: no cover - depends on external network state
            last_error = exc
    raise RuntimeError(
        "Could not cache the public TESS orbit-times file needed by tess_asteroids."
    ) from last_error


def _import_tess_asteroids_with_cached_downlinks() -> Any:
    _configure_optional_cache_dirs()
    cache_path = _resolve_tess_asteroids_cache_path()
    if not cache_path.exists():
        _download_downlinks_cache(cache_path)

    pandas = _require_pandas()
    original_read_csv = pandas.read_csv

    def patched_read_csv(filepath_or_buffer: Any, *args: Any, **kwargs: Any) -> Any:
        if filepath_or_buffer == _DOWNLINKS_URL:
            filepath_or_buffer = cache_path
        elif filepath_or_buffer == "https://archive.stsci.edu/missions/tess/models/":
            return _build_cached_tess_models_listing()
        return original_read_csv(filepath_or_buffer, *args, **kwargs)

    sys.modules.pop("tess_asteroids", None)
    pandas.read_csv = patched_read_csv
    try:
        return importlib.import_module("tess_asteroids")
    finally:
        pandas.read_csv = original_read_csv


def _normalize_barycentric_mode(value: bool | str | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "barycentric"}:
        return True
    if text in {"false", "0", "no", "n", "spacecraft"}:
        return False
    raise ValueError(
        f"Unsupported barycentric mode {value!r}. Use True/False or 'barycentric'/'spacecraft'."
    )


def _coerce_ephemeris_input(ephem_input: Any, target: str | None = None) -> Any:
    pandas = _require_pandas()
    if isinstance(ephem_input, (str, Path)):
        return load_ephemeris_csv(ephem_input, target=target, require_exact_tdb=True)
    if isinstance(ephem_input, pandas.DataFrame):
        return normalize_ephemeris_dataframe(ephem_input, target=target, require_exact_tdb=True)
    return normalize_ephemeris_dataframe(ephem_input, target=target, require_exact_tdb=True)


def subset_ephemeris_for_build(
    ephem_input: Any,
    target: str | None = None,
    time_start: float | None = None,
    time_stop: float | None = None,
    frame_limit: int | None = None,
) -> Any:
    """Normalize and subset a single-detector ephemeris for a bounded build."""

    normalized = _coerce_ephemeris_input(ephem_input, target=target)
    validate_single_detector_track(normalized, require_exact_tdb=True)

    selected = normalized.copy()
    if time_start is not None:
        selected = selected[selected["time_tdb_jd_minus_2457000"].astype(float) >= float(time_start)]
    if time_stop is not None:
        selected = selected[selected["time_tdb_jd_minus_2457000"].astype(float) <= float(time_stop)]
    if frame_limit is not None:
        if int(frame_limit) < 2:
            raise ValueError(f"`frame_limit` must be at least 2 for a real cutout build; got {frame_limit}.")
        selected = selected.iloc[: int(frame_limit)]

    selected = selected.reset_index(drop=True)
    if len(selected) < 2:
        raise ValueError(
            "Need at least two ephemeris rows after subsetting to build a moving cutout."
        )
    return selected


def _resolve_barycentric_mode(normalized_ephem: Any, barycentric_mode: bool | str | None) -> tuple[bool, str]:
    explicit_mode = _normalize_barycentric_mode(barycentric_mode)
    time_reference = get_time_reference(normalized_ephem, require_known=explicit_mode is None)
    if explicit_mode is not None:
        return explicit_mode, time_reference
    if time_reference == "barycentric":
        return True, time_reference
    if time_reference == "spacecraft":
        return False, time_reference
    raise ValueError(
        "Ephemeris timing is exact TDB but its reference frame is unspecified. "
        "For real-data builds, either export a normalized CSV with `time_reference`, "
        "or pass `barycentric_mode=True/False` explicitly."
    )


def _array_summary(values: Any) -> dict[str, float] | dict[str, int]:
    numpy = _require_numpy()
    array = numpy.asarray(values)
    if array.size == 0:
        return {"n": 0}
    finite = array[numpy.isfinite(array)] if numpy.issubdtype(array.dtype, numpy.number) else array.ravel()
    if finite.size == 0:
        return {"n": int(array.size)}
    if numpy.issubdtype(array.dtype, numpy.number):
        return {
            "n": int(array.size),
            "min": float(numpy.nanmin(array)),
            "max": float(numpy.nanmax(array)),
            "median": float(numpy.nanmedian(array)),
        }
    return {"n": int(array.size)}


def _extract_ra_dec_deg(coords: Any) -> tuple[Any | None, Any | None]:
    numpy = _require_numpy()
    if coords is None:
        return None, None
    ra_values: list[float] = []
    dec_values: list[float] = []
    for coord in coords:
        if not hasattr(coord, "ra") or not hasattr(coord, "dec"):
            return None, None
        ra_values.append(float(coord.ra.value))
        dec_values.append(float(coord.dec.value))
    return numpy.asarray(ra_values, dtype=float), numpy.asarray(dec_values, dtype=float)


def _build_tess_asteroids_ephem(normalized_ephem: Any) -> Any:
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


def build_cube_bundle_from_tess_asteroids(
    ephem_input: Any,
    target: str | None = None,
    shape: tuple[int, int] = (21, 21),
    frame_limit: int | None = None,
    time_start: float | None = None,
    time_stop: float | None = None,
    barycentric_mode: bool | str | None = None,
    bg_method: str = "linear_model",
    metadata: dict[str, Any] | None = None,
    **make_tpf_kwargs: Any,
) -> dict[str, Any]:
    """Build a recovery-ready cube bundle via `tess_asteroids.MovingTPF`.

    Parameters
    ----------
    ephem_input
        Normalized ephemeris DataFrame or CSV path. Real-data rows must carry an
        exact TDB column and a known `time_reference`, or the caller must pass
        `barycentric_mode` explicitly.
    shape
        Moving cutout shape as `(rows, cols)`.
    barycentric_mode
        Forwarded to `MovingTPF(..., barycentric=...)`.
    bg_method
        `tess_asteroids` background method, currently `linear_model` or `rolling`.
    make_tpf_kwargs
        Extra arguments passed through to `MovingTPF.make_tpf()`.
    """

    numpy = _require_numpy()
    module = _import_tess_asteroids()
    normalized_ephem = subset_ephemeris_for_build(
        ephem_input,
        target=target,
        time_start=time_start,
        time_stop=time_stop,
        frame_limit=frame_limit,
    )
    barycentric, time_reference = _resolve_barycentric_mode(normalized_ephem, barycentric_mode)

    if shape[0] < 3 or shape[1] < 3:
        raise ValueError(f"`shape` must be at least (3, 3); got {shape!r}.")

    tess_ephem = _build_tess_asteroids_ephem(normalized_ephem)
    target_name = (
        str(target)
        if target is not None
        else str(normalized_ephem.iloc[0]["target"])
        if "target" in normalized_ephem.columns
        else "target"
    )
    moving = module.MovingTPF(
        target=target_name,
        ephem=tess_ephem,
        barycentric=barycentric,
        metadata=metadata or {},
    )
    ap_method = str(make_tpf_kwargs.pop("ap_method", "threshold"))
    moving.make_tpf(
        shape=tuple(int(value) for value in shape),
        bg_method=bg_method,
        ap_method=ap_method,
        save=False,
        **make_tpf_kwargs,
    )

    cadence_quality = numpy.asarray(getattr(moving, "quality"), dtype=int)
    quality_mask = None
    quality_mask_definition = None
    if hasattr(moving, "_create_spoc_quality_mask"):
        try:
            quality_mask, bad_bits = moving._create_spoc_quality_mask(bad_spoc_bits="default")
            quality_mask = numpy.asarray(quality_mask, dtype=bool)
            quality_mask_definition = f"tess_asteroids_default_bad_spoc_bits={bad_bits!r}"
        except Exception:
            quality_mask = None

    cube = numpy.asarray(moving.corr_flux, dtype=float)
    time_tdb = numpy.asarray(moving.time, dtype=float)
    original_time_tdb = numpy.asarray(getattr(moving, "time_original"), dtype=float)
    timecorr = numpy.asarray(getattr(moving, "timecorr"), dtype=float)
    original_timecorr = numpy.asarray(getattr(moving, "timecorr_original"), dtype=float)
    detector_y = numpy.asarray(moving.ephemeris[:, 0], dtype=float)
    detector_x = numpy.asarray(moving.ephemeris[:, 1], dtype=float)
    corner2 = numpy.asarray(moving.corner[:, 0], dtype=float)
    corner1 = numpy.asarray(moving.corner[:, 1], dtype=float)
    local_x = detector_x - corner1
    local_y = detector_y - corner2
    pos_corr1 = local_x - (0.5 * (cube.shape[2] - 1))
    pos_corr2 = local_y - (0.5 * (cube.shape[1] - 1))
    ra_pred_deg, dec_pred_deg = _extract_ra_dec_deg(getattr(moving, "coords", None))

    bundle_metadata = {
        "target": target_name,
        "sector": int(normalized_ephem.iloc[0]["sector"]),
        "camera": int(normalized_ephem.iloc[0]["camera"]),
        "ccd": int(normalized_ephem.iloc[0]["ccd"]),
        "cutout_shape": [int(cube.shape[1]), int(cube.shape[2])],
        "barycentric_mode": bool(barycentric),
        "ephem_time_reference": time_reference,
        "ephem_time_scale": "tdb",
        "loader_bg_method": bg_method,
        "loader_ap_method": ap_method,
        "quality_mask_definition": quality_mask_definition,
        "time_summary": _array_summary(time_tdb),
        "timecorr_summary": _array_summary(timecorr),
        "original_time_summary": _array_summary(original_time_tdb),
        "original_timecorr_summary": _array_summary(original_timecorr),
        "corner1_summary": _array_summary(corner1),
        "corner2_summary": _array_summary(corner2),
        "pos_corr1_summary": _array_summary(pos_corr1),
        "pos_corr2_summary": _array_summary(pos_corr2),
        "ra_pred_available": ra_pred_deg is not None,
        "dec_pred_available": dec_pred_deg is not None,
        "pixel_quality_available": hasattr(moving, "pixel_quality"),
    }

    bundle = {
        "cube": cube,
        "time_tdb_jd_minus_2457000": time_tdb,
        "original_time_tdb_jd_minus_2457000": original_time_tdb,
        "timecorr_days": timecorr,
        "original_timecorr_days": original_timecorr,
        "local_x": local_x,
        "local_y": local_y,
        "detector_x": detector_x,
        "detector_y": detector_y,
        "corner1": corner1,
        "corner2": corner2,
        "pos_corr1": pos_corr1,
        "pos_corr2": pos_corr2,
        "cadence_quality": cadence_quality,
        "metadata": bundle_metadata,
    }
    if quality_mask is not None:
        bundle["quality_mask"] = quality_mask
    if hasattr(moving, "cadence_number"):
        bundle["cadence_number"] = numpy.asarray(moving.cadence_number, dtype=int)
    if ra_pred_deg is not None and dec_pred_deg is not None:
        bundle["ra_pred_deg"] = ra_pred_deg
        bundle["dec_pred_deg"] = dec_pred_deg
    if hasattr(moving, "pixel_quality"):
        bundle["pixel_quality"] = numpy.asarray(moving.pixel_quality)
    return bundle


def materialize_cube_bundle_to_npz(
    bundle: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a cube bundle to a compressed NPZ file."""

    numpy = _require_numpy()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    for key, value in bundle.items():
        if key == "metadata" or value is None:
            continue
        arrays[key] = numpy.asarray(value)
    arrays["metadata_json"] = json.dumps(bundle.get("metadata", {}), sort_keys=True)
    numpy.savez_compressed(destination, **arrays)
    return destination


def write_cube_bundle_metadata_json(bundle: dict[str, Any], output_path: str | Path) -> Path:
    """Write the cube bundle metadata payload to a sidecar JSON file."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(bundle.get("metadata", {}), handle, indent=2, sort_keys=True)
    return destination
