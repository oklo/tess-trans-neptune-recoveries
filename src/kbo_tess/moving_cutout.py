"""Shared helpers for detector-local moving-cutout workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kbo_tess.ephem import load_ephemeris_csv
from kbo_tess.tutorial_benchmark import _configure_benchmark_environment, _ephem_for_tess_asteroids, _import_tess_asteroids


def load_bundle(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def select_detector_ephem(path: Path, target: str, sector: int, camera: int, ccd: int) -> pd.DataFrame:
    ephem = load_ephemeris_csv(path)
    selected = ephem[
        (ephem["target"].astype(str) == str(target))
        & (ephem["sector"].astype(int) == int(sector))
        & (ephem["camera"].astype(int) == int(camera))
        & (ephem["ccd"].astype(int) == int(ccd))
    ].copy()
    if selected.empty:
        raise ValueError("No ephemeris rows matched the requested target/detector.")
    return selected.sort_values("time_tdb_jd_minus_2457000").reset_index(drop=True)


def build_raw_flux_cube(
    ephem: pd.DataFrame,
    shape: tuple[int, int],
    *,
    bg_method: str = "rolling",
    barycentric: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    _configure_benchmark_environment()
    module = _import_tess_asteroids()
    tess_ephem = _ephem_for_tess_asteroids(ephem)
    moving = module.MovingTPF(target=str(ephem.iloc[0]["target"]), ephem=tess_ephem, barycentric=bool(barycentric), metadata={})
    moving.make_tpf(shape=shape, bg_method=str(bg_method), save=False)
    raw_flux = np.asarray(getattr(moving, "flux"), dtype=float)
    raw_time = np.asarray(getattr(moving, "time"), dtype=float)
    return raw_flux, raw_time


def match_bundle_times_to_raw(raw_time: np.ndarray, bundle_time: np.ndarray) -> np.ndarray:
    match_idx = np.array([int(np.nanargmin(np.abs(raw_time - value))) for value in bundle_time], dtype=int)
    dt_seconds = np.abs(raw_time[match_idx] - bundle_time) * 86400.0
    if float(np.nanmax(dt_seconds)) > 1.0:
        raise ValueError(f"Raw/bundle cadence matching drifted too far; max dt is {float(np.nanmax(dt_seconds)):.3f} s.")
    return match_idx


def local_track_from_bundle(bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    det_x = np.asarray(bundle["detector_x"], dtype=float)
    det_y = np.asarray(bundle["detector_y"], dtype=float)
    corner1 = np.asarray(bundle["corner1"], dtype=float)
    corner2 = np.asarray(bundle["corner2"], dtype=float)
    return det_x - corner1, det_y - corner2
