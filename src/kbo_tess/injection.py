"""Small real-bundle injection helpers for detector-local control studies."""

from __future__ import annotations

import copy
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    np = None


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for bundle injection. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return np


def inject_moving_gaussian_source(
    cube: Any,
    x: Any,
    y: Any,
    amplitude: float,
    sigma: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
) -> Any:
    """Return a cube with a synthetic moving Gaussian source added frame by frame."""

    numpy = _require_numpy()
    cube_array = numpy.asarray(cube, dtype=float)
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    if cube_array.ndim != 3:
        raise ValueError(f"Expected cube with shape (n_frames, ny, nx); got {cube_array.shape!r}.")
    if x_array.shape != y_array.shape or x_array.shape != (cube_array.shape[0],):
        raise ValueError(
            "Injection track must match the cube frame count; "
            f"got cube frames={cube_array.shape[0]}, x shape={x_array.shape!r}, y shape={y_array.shape!r}."
        )
    if sigma <= 0:
        raise ValueError(f"`sigma` must be positive; got {sigma}.")

    yy, xx = numpy.indices(cube_array.shape[1:], dtype=float)
    injected = cube_array.copy()
    variance = 2.0 * float(sigma) * float(sigma)
    for frame_index in range(cube_array.shape[0]):
        dx_frame = xx - (float(x_array[frame_index]) + float(dx))
        dy_frame = yy - (float(y_array[frame_index]) + float(dy))
        injected[frame_index] += float(amplitude) * numpy.exp(-(dx_frame * dx_frame + dy_frame * dy_frame) / variance)
    return injected


def inject_source_into_cube_bundle(
    cube_bundle: dict[str, Any],
    amplitude: float,
    sigma: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    track_x_key: str = "local_x",
    track_y_key: str = "local_y",
) -> dict[str, Any]:
    """Return a copied cube bundle with a synthetic source injected on the supplied local track."""

    numpy = _require_numpy()
    if track_x_key not in cube_bundle or track_y_key not in cube_bundle:
        raise KeyError(
            f"Cube bundle must contain `{track_x_key}` and `{track_y_key}` for injection."
        )

    injected_bundle: dict[str, Any] = {}
    for key, value in cube_bundle.items():
        if key == "metadata":
            continue
        try:
            injected_bundle[key] = numpy.asarray(value).copy()
        except Exception:
            injected_bundle[key] = copy.deepcopy(value)

    injected_bundle["cube"] = inject_moving_gaussian_source(
        cube=injected_bundle["cube"],
        x=injected_bundle[track_x_key],
        y=injected_bundle[track_y_key],
        amplitude=amplitude,
        sigma=sigma,
        dx=dx,
        dy=dy,
    )
    injected_bundle["injection_local_x"] = numpy.asarray(injected_bundle[track_x_key], dtype=float) + float(dx)
    injected_bundle["injection_local_y"] = numpy.asarray(injected_bundle[track_y_key], dtype=float) + float(dy)

    metadata = copy.deepcopy(cube_bundle.get("metadata", {}))
    injections = list(metadata.get("injections", []))
    injections.append(
        {
            "kind": "moving_gaussian_source",
            "amplitude": float(amplitude),
            "sigma": float(sigma),
            "dx": float(dx),
            "dy": float(dy),
            "track_x_key": track_x_key,
            "track_y_key": track_y_key,
        }
    )
    metadata["injections"] = injections
    metadata["last_injection"] = injections[-1]
    injected_bundle["metadata"] = metadata
    return injected_bundle
