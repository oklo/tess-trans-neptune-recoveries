"""Focused stack-alignment audit helpers for frozen local cube bundles.

The goal of this module is diagnosis rather than production recovery. It keeps
the coordinate bookkeeping explicit so we can answer questions like:

- does a measured-centroid stack behave more physically than the nominal track?
- are integer and subpixel shifts consistent?
- do support maps or NaN coverage explain the preferred offset?
- is there a consistent small frame-by-frame track offset?
"""

from __future__ import annotations

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

from kbo_tess.injection import inject_source_into_cube_bundle
from kbo_tess.photometry import aperture_photometry_at_position
from kbo_tess.shiftstack import score_stack, stack_with_support


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for stack audits. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return np


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for stack audits. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return pd


def _require_matplotlib() -> Any:
    if plt is None:
        raise ImportError(
            "matplotlib is required for stack audit plots. Install project dependencies "
            "with `python -m pip install -e .`."
        )
    return plt


def measure_local_centroids(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    search_radius: float = 3.0,
    annulus_inner_radius: float = 4.0,
    annulus_outer_radius: float = 6.0,
) -> Any:
    """Measure simple weighted centroids near a predicted local track."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    cube_array = numpy.asarray(cube, dtype=float)
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    if cube_array.ndim != 3:
        raise ValueError(f"Expected cube with shape (n_frames, ny, nx); got {cube_array.shape!r}.")
    if x_array.shape != (cube_array.shape[0],) or y_array.shape != (cube_array.shape[0],):
        raise ValueError(
            "Track coordinates must match the cube frame count; "
            f"got cube frames={cube_array.shape[0]}, x shape={x_array.shape!r}, y shape={y_array.shape!r}."
        )

    yy, xx = numpy.indices(cube_array.shape[1:], dtype=float)
    rows: list[dict[str, float | bool]] = []
    for frame_index, frame in enumerate(cube_array):
        x0 = float(x_array[frame_index])
        y0 = float(y_array[frame_index])
        rr = numpy.hypot(xx - x0, yy - y0)
        aperture_mask = rr <= float(search_radius)
        annulus_mask = (rr >= float(annulus_inner_radius)) & (rr <= float(annulus_outer_radius))
        annulus_values = frame[annulus_mask]
        finite_annulus = annulus_values[numpy.isfinite(annulus_values)]
        background = float(numpy.nanmedian(finite_annulus)) if finite_annulus.size else 0.0
        weights = numpy.where(aperture_mask, frame - background, 0.0)
        weights = numpy.where(numpy.isfinite(weights) & (weights > 0.0), weights, 0.0)
        total_weight = float(numpy.sum(weights))
        valid = bool(total_weight > 0.0)
        if valid:
            measured_x = float(numpy.sum(weights * xx) / total_weight)
            measured_y = float(numpy.sum(weights * yy) / total_weight)
            dx = measured_x - x0
            dy = measured_y - y0
        else:
            measured_x = float("nan")
            measured_y = float("nan")
            dx = float("nan")
            dy = float("nan")
        rows.append(
            {
                "frame_index": float(frame_index),
                "predicted_x": x0,
                "predicted_y": y0,
                "measured_x": measured_x,
                "measured_y": measured_y,
                "residual_x": float(dx),
                "residual_y": float(dy),
                "residual_r": float(numpy.hypot(dx, dy)) if valid else float("nan"),
                "total_weight": total_weight,
                "valid_centroid": valid,
            }
        )
    return pandas.DataFrame(rows)


def select_brightest_frames_by_track_photometry(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    subset_size: int = 32,
) -> Any:
    """Return frame indices with the strongest nominal-track aperture S/N."""

    numpy = _require_numpy()
    if subset_size < 2:
        raise ValueError(f"`subset_size` must be at least 2; got {subset_size}.")
    cube_array = numpy.asarray(cube, dtype=float)
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    if subset_size > cube_array.shape[0]:
        raise ValueError(
            f"`subset_size`={subset_size} exceeds available frames={cube_array.shape[0]}."
        )
    scores = numpy.asarray(
        [
            aperture_photometry_at_position(frame, (float(x_array[index]), float(y_array[index])))["aperture_snr"]
            for index, frame in enumerate(cube_array)
        ],
        dtype=float,
    )
    ranked = numpy.argsort(scores)[::-1][:subset_size]
    return numpy.sort(ranked)


def evaluate_frame_offset_photometry(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    offsets: Iterable[tuple[float, float]],
) -> Any:
    """Measure aperture S/N frame by frame on a small offset grid."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    cube_array = numpy.asarray(cube, dtype=float)
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)

    rows: list[dict[str, float | bool]] = []
    offset_list = [(float(dx), float(dy)) for dx, dy in offsets]
    for frame_index, frame in enumerate(cube_array):
        score_rows: list[tuple[float, float, float]] = []
        for dx, dy in offset_list:
            score = float(
                aperture_photometry_at_position(
                    frame,
                    (float(x_array[frame_index] + dx), float(y_array[frame_index] + dy)),
                )["aperture_snr"]
            )
            score_rows.append((dx, dy, score))
        best_dx, best_dy, best_score = max(score_rows, key=lambda item: item[2])
        for dx, dy, score in score_rows:
            rows.append(
                {
                    "frame_index": float(frame_index),
                    "dx": float(dx),
                    "dy": float(dy),
                    "aperture_snr": float(score),
                    "best_dx": float(best_dx),
                    "best_dy": float(best_dy),
                    "best_aperture_snr": float(best_score),
                    "is_best_for_frame": bool(dx == best_dx and dy == best_dy),
                }
            )
    return pandas.DataFrame(rows)


def summarize_offset_preference_counts(offset_photometry: Any) -> dict[str, Any]:
    """Summarize which small offsets win most often frame by frame."""

    pandas = _require_pandas()
    best_rows = pandas.DataFrame(offset_photometry)
    if best_rows.empty:
        return {"counts": [], "nominal_wins": 0}
    best_rows = best_rows[best_rows["is_best_for_frame"].astype(bool)]
    grouped = (
        best_rows.groupby(["dx", "dy"], dropna=False)
        .size()
        .reset_index(name="n_best")
        .sort_values("n_best", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "counts": grouped.to_dict(orient="records"),
        "nominal_wins": int(
            best_rows[(best_rows["dx"].astype(float) == 0.0) & (best_rows["dy"].astype(float) == 0.0)].shape[0]
        ),
    }


def build_stack_variant(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    reference: tuple[float, float],
    interpolation: str = "linear",
    method: str = "mean",
    ranking_position: tuple[float, float] | None = None,
    ranking_method: str = "aperture_snr",
    integer_shifts: bool = False,
) -> dict[str, Any]:
    """Build one aligned stack variant with support and scoring metadata."""

    numpy = _require_numpy()
    stack_data = stack_with_support(
        cube,
        x,
        y,
        method=method,
        interpolation=interpolation,
        reference=reference,
        integer_shifts=integer_shifts,
    )
    stack = numpy.asarray(stack_data["stack"], dtype=float)
    score_position = ranking_position if ranking_position is not None else reference
    peak_y, peak_x = numpy.unravel_index(numpy.nanargmax(stack), stack.shape)
    stack_data["scores"] = {
        "peak": float(score_stack(stack, method="peak", position=score_position)),
        "snr": float(score_stack(stack, method="snr", position=score_position)),
        "aperture_snr": float(score_stack(stack, method="aperture_snr", position=score_position)),
    }
    stack_data["ranking_method"] = ranking_method
    stack_data["ranking_score"] = float(stack_data["scores"][ranking_method])
    stack_data["peak_location"] = {
        "x": float(peak_x),
        "y": float(peak_y),
        "value": float(numpy.nanmax(stack)),
        "peak_aperture_snr": float(score_stack(stack, method="aperture_snr", position=(float(peak_x), float(peak_y)))),
    }
    return stack_data


def run_real_bundle_injection_check(
    cube_bundle: dict[str, Any],
    reference: tuple[float, float],
    subset_indices: Iterable[int],
    amplitude: float = 20.0,
    sigma: float = 0.8,
    injected_dx: float = 0.0,
    injected_dy: float = 0.0,
    ranking_method: str = "aperture_snr",
) -> dict[str, Any]:
    """Inject a synthetic moving PSF on a real bundle subset and stack it."""

    numpy = _require_numpy()
    subset = numpy.asarray(list(subset_indices), dtype=int)
    clipped_bundle: dict[str, Any] = {}
    n_frames = int(numpy.asarray(cube_bundle["cube"]).shape[0])
    for key, value in cube_bundle.items():
        if key == "metadata":
            continue
        array = numpy.asarray(value) if hasattr(value, "shape") else None
        if array is not None and array.ndim >= 1 and array.shape[0] == n_frames:
            clipped_bundle[key] = array[subset].copy()
        else:
            clipped_bundle[key] = value
    clipped_bundle["metadata"] = dict(cube_bundle.get("metadata", {}))
    injected_bundle = inject_source_into_cube_bundle(
        clipped_bundle,
        amplitude=amplitude,
        sigma=sigma,
        dx=injected_dx,
        dy=injected_dy,
    )
    local_x = numpy.asarray(injected_bundle["local_x"], dtype=float)
    local_y = numpy.asarray(injected_bundle["local_y"], dtype=float)
    return build_stack_variant(
        injected_bundle["cube"],
        local_x,
        local_y,
        reference=reference,
        interpolation="linear",
        ranking_position=reference,
        ranking_method=ranking_method,
    )


def plot_stack_audit_summary(
    representative_frame: Any,
    reference: tuple[float, float],
    variants: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write a compact PNG comparing representative frame, stacks, and support."""

    plotting = _require_matplotlib()
    numpy = _require_numpy()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plotting.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    frame = numpy.asarray(representative_frame, dtype=float)
    frame_finite = frame[numpy.isfinite(frame)]
    if frame_finite.size:
        frame_vmin, frame_vmax = numpy.nanpercentile(frame_finite, [5, 99])
    else:
        frame_vmin, frame_vmax = -1.0, 1.0
    axes[0, 0].imshow(frame, origin="lower", cmap="viridis", vmin=frame_vmin, vmax=frame_vmax)
    axes[0, 0].scatter([reference[0]], [reference[1]], marker="x", color="white", s=55, linewidths=1.5)
    axes[0, 0].set_title("Representative Frame")
    axes[0, 0].set_xlabel("x [pix]")
    axes[0, 0].set_ylabel("y [pix]")

    panel_order = [("nominal", "Nominal"), ("x_minus_half", "x - 0.5"), ("centroid", "Centroid")]
    for column_index, (key, title) in enumerate(panel_order, start=1):
        variant = variants[key]
        stack = numpy.asarray(variant["stack"], dtype=float)
        support = numpy.asarray(variant["support_map"], dtype=float)
        finite = stack[numpy.isfinite(stack)]
        if finite.size:
            vmin, vmax = numpy.nanpercentile(finite, [5, 99.5])
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0
        else:
            vmin, vmax = -1.0, 1.0
        axes[0, column_index].imshow(stack, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        axes[0, column_index].scatter([reference[0]], [reference[1]], marker="x", color="white", s=55, linewidths=1.5)
        axes[0, column_index].set_title(f"{title} Stack\nAperture S/N={variant['scores']['aperture_snr']:.2f}")
        axes[0, column_index].set_xlabel("x [pix]")
        axes[0, column_index].set_ylabel("y [pix]")

        axes[1, column_index].imshow(support, origin="lower", cmap="cividis")
        axes[1, column_index].scatter([reference[0]], [reference[1]], marker="x", color="white", s=55, linewidths=1.5)
        axes[1, column_index].set_title(f"{title} Support")
        axes[1, column_index].set_xlabel("x [pix]")
        axes[1, column_index].set_ylabel("y [pix]")

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Reference pixel: ({reference[0]:.3f}, {reference[1]:.3f})",
                "Shift audit uses explicit dx = x_ref - x_i, dy = y_ref - y_i.",
                "Array sampler receives shifts in axis order (dy, dx).",
                "Global stack maxima are shown separately from the fixed scoring location.",
            ]
        ),
        transform=axes[1, 0].transAxes,
        ha="left",
        va="top",
    )

    figure.savefig(destination, dpi=170)
    plotting.close(figure)
    return destination


def plot_centroid_residuals_and_offset_preferences(
    centroids: Any,
    offset_summary: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write centroid residual and offset-preference diagnostics."""

    plotting = _require_matplotlib()
    pandas = _require_pandas()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    centroid_table = pandas.DataFrame(centroids)
    figure, axes = plotting.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(centroid_table["frame_index"], centroid_table["residual_x"], marker="o", linewidth=1.0, label="x")
    axes[0].plot(centroid_table["frame_index"], centroid_table["residual_y"], marker="o", linewidth=1.0, label="y")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_title("Measured Centroid Residuals")
    axes[0].set_xlabel("Subset frame index")
    axes[0].set_ylabel("Measured - predicted [pix]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    counts = offset_summary.get("counts", [])
    if counts:
        labels = [f"({row['dx']:+.1f},{row['dy']:+.1f})" for row in counts]
        values = [row["n_best"] for row in counts]
        positions = list(range(len(labels)))
        axes[1].bar(positions, values)
        axes[1].set_xticks(positions, labels, rotation=45, ha="right")
        axes[1].set_title("Frame-by-Frame Best Offset Counts")
        axes[1].set_ylabel("Frames")
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].axis("off")

    figure.savefig(destination, dpi=170)
    plotting.close(figure)
    return destination
