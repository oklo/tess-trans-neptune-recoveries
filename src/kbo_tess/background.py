"""Simple background estimation for detector-local TESS image cubes."""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    np = None

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    savgol_filter = None


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for background subtraction. Install it with "
            "`python -m pip install numpy` or `python -m pip install -e .`."
        )
    return np


def _require_savgol() -> Any:
    if savgol_filter is None:
        raise ImportError(
            "scipy is required for Savitzky-Golay background subtraction. Install it with "
            "`python -m pip install scipy` or `python -m pip install -e .`."
        )
    return savgol_filter


def _validate_cube(cube: Any) -> Any:
    numpy = _require_numpy()
    array = numpy.asarray(cube, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"Expected cube with shape (n_frames, ny, nx); got {array.shape!r}.")
    return array


def _coerce_odd_window_length(window_length: int, n_frames: int, minimum: int = 3) -> int:
    if n_frames < minimum:
        raise ValueError(
            f"Need at least {minimum} frames for this baseline method; got {n_frames}."
        )
    clipped = max(minimum, min(int(window_length), n_frames))
    if clipped % 2 == 0:
        clipped -= 1
    if clipped < minimum:
        clipped = minimum if minimum % 2 == 1 else minimum + 1
    if clipped > n_frames:
        clipped = n_frames if n_frames % 2 == 1 else n_frames - 1
    return clipped


def median_baseline(
    cube: Any,
    window_length: int = 11,
    chunk_size: int | None = None,
) -> tuple[Any, Any]:
    """Estimate a simple per-pixel baseline with a running or chunked median."""

    numpy = _require_numpy()
    cube_array = _validate_cube(cube)
    n_frames = cube_array.shape[0]

    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive; got {chunk_size}.")
        baseline = numpy.empty_like(cube_array, dtype=float)
        for start in range(0, n_frames, int(chunk_size)):
            stop = min(start + int(chunk_size), n_frames)
            chunk_median = numpy.nanmedian(cube_array[start:stop], axis=0)
            baseline[start:stop] = chunk_median
        residual = cube_array - baseline
        return baseline, residual

    window = _coerce_odd_window_length(window_length, n_frames, minimum=1)
    pad = window // 2
    padded = numpy.pad(cube_array, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    windows = numpy.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=window,
        axis=0,
    )
    baseline = numpy.nanmedian(windows, axis=-1)
    residual = cube_array - baseline
    return baseline, residual


def savgol_baseline(
    cube: Any,
    window_length: int = 11,
    polyorder: int = 2,
) -> tuple[Any, Any]:
    """Estimate a smooth per-pixel baseline with a Savitzky-Golay filter."""

    cube_array = _validate_cube(cube)
    savgol = _require_savgol()
    n_frames = cube_array.shape[0]
    window = _coerce_odd_window_length(window_length, n_frames, minimum=polyorder + 2)
    baseline = savgol(
        cube_array,
        window_length=window,
        polyorder=polyorder,
        axis=0,
        mode="interp",
    )
    residual = cube_array - baseline
    return baseline, residual


def subtract_background(
    cube: Any,
    method: str = "median",
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Dispatch to one of the lightweight baseline-removal methods."""

    cube_array = _validate_cube(cube)
    if method == "none":
        numpy = _require_numpy()
        return numpy.zeros_like(cube_array, dtype=float), cube_array.copy()
    if method == "median":
        return median_baseline(cube_array, **kwargs)
    if method == "savgol":
        return savgol_baseline(cube_array, **kwargs)
    raise ValueError(
        f"Unknown background method {method!r}. Choose 'none', 'median', or 'savgol'."
    )
