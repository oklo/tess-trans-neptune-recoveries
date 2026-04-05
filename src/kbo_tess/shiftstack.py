"""Known-path detector-local shift-and-stack helpers."""

from __future__ import annotations

from typing import Any, Iterable

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    np = None

try:
    from scipy.ndimage import map_coordinates
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    map_coordinates = None


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for shift-and-stack operations. Install it with "
            "`python -m pip install numpy` or `python -m pip install -e .`."
        )
    return np


def _require_linear_interpolator() -> Any:
    if map_coordinates is None:
        raise ImportError(
            "scipy is required for linear interpolation. Install it with "
            "`python -m pip install scipy` or `python -m pip install -e .`, "
            "or rerun with `--interpolation nearest`."
        )
    return map_coordinates


def _validate_inputs(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
) -> tuple[Any, Any, Any]:
    numpy = _require_numpy()
    cube_array = numpy.asarray(cube, dtype=float)
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)

    if cube_array.ndim != 3:
        raise ValueError(f"Expected cube with shape (n_frames, ny, nx); got {cube_array.shape!r}.")
    if x_array.shape != y_array.shape:
        raise ValueError(f"`x` and `y` must have matching shapes; got {x_array.shape!r} and {y_array.shape!r}.")
    if cube_array.shape[0] != x_array.shape[0]:
        raise ValueError(
            f"Track length must match cube frames; got {cube_array.shape[0]} frames and {x_array.shape[0]} track samples."
        )
    return cube_array, x_array, y_array


def _sample_shifted_frame(
    frame: Any,
    shift_x: float,
    shift_y: float,
    interpolation: str,
) -> Any:
    numpy = _require_numpy()
    ny, nx = frame.shape
    yy, xx = numpy.indices((ny, nx), dtype=float)
    sample_x = xx - shift_x
    sample_y = yy - shift_y

    if interpolation == "nearest":
        xi = numpy.rint(sample_x).astype(int)
        yi = numpy.rint(sample_y).astype(int)
        shifted = numpy.full((ny, nx), numpy.nan, dtype=float)
        valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        shifted[valid] = frame[yi[valid], xi[valid]]
        return shifted

    if interpolation == "linear":
        sampler = _require_linear_interpolator()
        coordinates = numpy.asarray([sample_y, sample_x], dtype=float)
        return sampler(
            frame,
            coordinates,
            order=1,
            mode="constant",
            cval=numpy.nan,
            prefilter=False,
        )

    raise ValueError(f"Unknown interpolation mode {interpolation!r}.")


def stack_along_track(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    method: str = "mean",
    interpolation: str = "nearest",
    reference: tuple[float, float] | None = None,
) -> Any:
    """Shift frames so a known detector-local track lands on a common reference pixel."""

    numpy = _require_numpy()
    cube_array, x_array, y_array = _validate_inputs(cube, x, y)
    ref_x, ref_y = reference if reference is not None else (float(x_array[0]), float(y_array[0]))

    aligned_frames = numpy.empty_like(cube_array, dtype=float)
    for index, frame in enumerate(cube_array):
        shift_x = ref_x - float(x_array[index])
        shift_y = ref_y - float(y_array[index])
        aligned_frames[index] = _sample_shifted_frame(frame, shift_x, shift_y, interpolation)

    if method == "mean":
        valid_counts = numpy.sum(numpy.isfinite(aligned_frames), axis=0)
        summed = numpy.nansum(aligned_frames, axis=0)
        stacked = numpy.full(aligned_frames.shape[1:], numpy.nan, dtype=float)
        numpy.divide(summed, valid_counts, out=stacked, where=valid_counts > 0)
        return stacked
    if method == "sum":
        return numpy.nansum(aligned_frames, axis=0)
    if method == "median":
        return numpy.nanmedian(aligned_frames, axis=0)
    raise ValueError(f"Unknown stack method {method!r}. Choose 'mean', 'sum', or 'median'.")


def _position_to_index(position: tuple[float, float], shape: tuple[int, int]) -> tuple[int, int]:
    x, y = position
    ix = int(round(x))
    iy = int(round(y))
    ny, nx = shape
    if not (0 <= ix < nx and 0 <= iy < ny):
        raise ValueError(f"Requested score position {position!r} is outside image shape {shape!r}.")
    return iy, ix


def _radial_masks(
    shape: tuple[int, int],
    position: tuple[float, float],
    aperture_radius: float,
    annulus_inner_radius: float,
    annulus_outer_radius: float,
) -> tuple[Any, Any]:
    numpy = _require_numpy()
    x0, y0 = position
    yy, xx = numpy.indices(shape, dtype=float)
    rr = numpy.hypot(xx - float(x0), yy - float(y0))
    aperture_mask = rr <= float(aperture_radius)
    annulus_mask = (rr >= float(annulus_inner_radius)) & (rr <= float(annulus_outer_radius))
    return aperture_mask, annulus_mask


def score_stack(
    stack: Any,
    method: str = "peak",
    position: tuple[float, float] | None = None,
    signal_radius: int = 1,
    noise_radius: int = 4,
    aperture_radius: float = 1.5,
    annulus_inner_radius: float = 3.0,
    annulus_outer_radius: float = 5.0,
) -> float:
    """Score a stacked image with a peak or local S/N metric."""

    numpy = _require_numpy()
    image = numpy.asarray(stack, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D stacked image; got shape {image.shape!r}.")

    if position is None:
        iy, ix = numpy.unravel_index(numpy.nanargmax(image), image.shape)
    else:
        iy, ix = _position_to_index(position, image.shape)

    y_min = max(0, iy - signal_radius)
    y_max = min(image.shape[0], iy + signal_radius + 1)
    x_min = max(0, ix - signal_radius)
    x_max = min(image.shape[1], ix + signal_radius + 1)
    signal_patch = image[y_min:y_max, x_min:x_max]

    if method == "peak":
        return float(numpy.nanmax(signal_patch))

    if method == "snr":
        noise_y_min = max(0, iy - noise_radius)
        noise_y_max = min(image.shape[0], iy + noise_radius + 1)
        noise_x_min = max(0, ix - noise_radius)
        noise_x_max = min(image.shape[1], ix + noise_radius + 1)
        noise_patch = image[noise_y_min:noise_y_max, noise_x_min:noise_x_max].copy()
        local_y_min = y_min - noise_y_min
        local_y_max = y_max - noise_y_min
        local_x_min = x_min - noise_x_min
        local_x_max = x_max - noise_x_min
        noise_patch[local_y_min:local_y_max, local_x_min:local_x_max] = numpy.nan
        noise_level = float(numpy.nanstd(noise_patch))
        background = float(numpy.nanmedian(noise_patch))
        signal = float(numpy.nanmean(signal_patch))
        if not numpy.isfinite(noise_level) or noise_level == 0.0:
            return signal - background
        return (signal - background) / noise_level

    if method == "aperture_snr":
        score_position = position if position is not None else (float(ix), float(iy))
        aperture_mask, annulus_mask = _radial_masks(
            image.shape,
            score_position,
            aperture_radius=aperture_radius,
            annulus_inner_radius=annulus_inner_radius,
            annulus_outer_radius=annulus_outer_radius,
        )
        aperture_values = image[aperture_mask]
        annulus_values = image[annulus_mask]
        finite_annulus = annulus_values[numpy.isfinite(annulus_values)]
        finite_aperture = aperture_values[numpy.isfinite(aperture_values)]
        if finite_aperture.size == 0:
            return float("-inf")
        background_level = float(numpy.nanmedian(finite_annulus)) if finite_annulus.size else 0.0
        net_signal = float(numpy.nansum(finite_aperture - background_level))
        if finite_annulus.size < 2:
            return net_signal
        noise_sigma = float(numpy.nanstd(finite_annulus, ddof=1))
        if not numpy.isfinite(noise_sigma) or noise_sigma == 0.0:
            return net_signal
        return net_signal / (noise_sigma * float(numpy.sqrt(max(1, finite_aperture.size))))

    raise ValueError(
        f"Unknown score method {method!r}. Choose 'peak', 'snr', or 'aperture_snr'."
    )


def search_local_track_offsets(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    dx_grid: Iterable[float],
    dy_grid: Iterable[float],
    interpolation: str = "nearest",
    stack_method: str = "mean",
    score_method: str = "snr",
    reference: tuple[float, float] | None = None,
    score_position: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Search a small constant dx/dy offset grid around a known detector-local track."""

    cube_array, x_array, y_array = _validate_inputs(cube, x, y)
    fixed_reference = reference if reference is not None else (float(x_array[0]), float(y_array[0]))
    fixed_score_position = score_position if score_position is not None else fixed_reference

    results: list[dict[str, float]] = []
    best_result: dict[str, float] | None = None
    best_stack = None
    for dx in dx_grid:
        for dy in dy_grid:
            stacked = stack_along_track(
                cube_array,
                x_array + float(dx),
                y_array + float(dy),
                method=stack_method,
                interpolation=interpolation,
                reference=fixed_reference,
            )
            score = score_stack(stacked, method=score_method, position=fixed_score_position)
            result = {"dx": float(dx), "dy": float(dy), "score": float(score)}
            results.append(result)
            if best_result is None or score > best_result["score"]:
                best_result = result
                best_stack = stacked

    if best_result is None:
        raise ValueError("Offset search grid is empty.")

    return {
        "reference": fixed_reference,
        "score_position": fixed_score_position,
        "best": {**best_result, "stack": best_stack},
        "results": results,
    }
