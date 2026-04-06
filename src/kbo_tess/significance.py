"""Empirical-significance helpers for bounded detector-local bundle studies.

This module keeps the significance logic explicit and conservative:

- The primary statistic is a direct moving-aperture aggregate:
  mean(aperture_snr_i) over cadences along a fixed track.
- The secondary statistic is a stack-based central-source score:
  aperture_snr of the aligned stack at the fixed reference pixel.
- Matched nulls are built from translated pseudo-tracks that preserve the same
  cadence times and relative track shape as the observed target.
- When the observed analysis chooses the better of multiple track hypotheses,
  the null ensemble is allowed the same choice rule.
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

from kbo_tess.photometry import aperture_photometry_at_position
from kbo_tess.stack_audit import build_stack_variant
from kbo_tess.track_calibration import evaluate_track_photometry


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for empirical significance studies. Install project dependencies "
            "with `python -m pip install -e .`."
        )
    return np


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for empirical significance studies. Install project dependencies "
            "with `python -m pip install -e .`."
        )
    return pd


def _require_matplotlib() -> Any:
    if plt is None:
        raise ImportError(
            "matplotlib is required for empirical-significance plots. Install project dependencies "
            "with `python -m pip install -e .`."
        )
    return plt


def empirical_p_value(observed_score: float, null_scores: Iterable[float]) -> float:
    """Return the conservative empirical p-value for a one-sided upper-tail test."""

    numpy = _require_numpy()
    null_array = numpy.asarray(list(null_scores), dtype=float)
    finite = null_array[numpy.isfinite(null_array)]
    if finite.size == 0:
        raise ValueError("Need at least one finite null score to compute an empirical p-value.")
    return float((1 + numpy.sum(finite >= float(observed_score))) / (finite.size + 1))


def generate_translated_pseudo_tracks(
    nominal_x: Iterable[float],
    nominal_y: Iterable[float],
    calibrated_x_offset: float,
    shape: tuple[int, int],
    n_realizations: int = 256,
    edge_margin: float = 5.0,
    exclusion_radius: float = 3.0,
    rng_seed: int = 20260405,
) -> Any:
    """Generate translated blank-sky pseudo-tracks that preserve the Pluto track shape.

    The same constant translation is applied to the nominal and calibrated track
    families, so the null ensemble matches the observed look-elsewhere rule over
    the x-calibrated pair.
    """

    numpy = _require_numpy()
    pandas = _require_pandas()
    nominal_x_array = numpy.asarray(nominal_x, dtype=float)
    nominal_y_array = numpy.asarray(nominal_y, dtype=float)
    calibrated_x_array = nominal_x_array + float(calibrated_x_offset)
    ny, nx = int(shape[0]), int(shape[1])

    def _allowed_bounds(track_x: Any, track_y: Any) -> tuple[float, float, float, float]:
        tx_min = float(edge_margin - numpy.nanmin(track_x))
        tx_max = float((nx - 1 - edge_margin) - numpy.nanmax(track_x))
        ty_min = float(edge_margin - numpy.nanmin(track_y))
        ty_max = float((ny - 1 - edge_margin) - numpy.nanmax(track_y))
        return tx_min, tx_max, ty_min, ty_max

    nominal_bounds = _allowed_bounds(nominal_x_array, nominal_y_array)
    calibrated_bounds = _allowed_bounds(calibrated_x_array, nominal_y_array)
    tx_min = max(nominal_bounds[0], calibrated_bounds[0])
    tx_max = min(nominal_bounds[1], calibrated_bounds[1])
    ty_min = max(nominal_bounds[2], calibrated_bounds[2])
    ty_max = min(nominal_bounds[3], calibrated_bounds[3])
    if not (tx_min < tx_max and ty_min < ty_max):
        raise ValueError(
            "Could not find any valid translated pseudo-track region within the cutout "
            f"for edge_margin={edge_margin}."
        )

    rng = numpy.random.default_rng(int(rng_seed))
    rows: list[dict[str, float]] = []
    attempts = 0
    max_attempts = max(2000, int(n_realizations) * 100)
    while len(rows) < int(n_realizations):
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could only generate {len(rows)} pseudo-tracks after {attempts} attempts; "
                "try reducing the edge margin or exclusion radius."
            )
        attempts += 1
        tx = float(rng.uniform(tx_min, tx_max))
        ty = float(rng.uniform(ty_min, ty_max))
        if numpy.hypot(tx, ty) < float(exclusion_radius):
            continue
        rows.append(
            {
                "realization": float(len(rows)),
                "translation_x": tx,
                "translation_y": ty,
                "translation_radius": float(numpy.hypot(tx, ty)),
            }
        )

    return pandas.DataFrame(rows)


def scrub_track_from_cube_bundle(
    cube_bundle: dict[str, Any],
    x: Iterable[float],
    y: Iterable[float],
    mask_radius: float = 2.0,
    fill_inner_radius: float = 3.0,
    fill_outer_radius: float = 5.0,
    rng_seed: int = 20260405,
) -> dict[str, Any]:
    """Replace a moving-source aperture with local-background resamples frame by frame."""

    numpy = _require_numpy()
    cube = numpy.asarray(cube_bundle["cube"], dtype=float)
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube with shape (n_frames, ny, nx); got {cube.shape!r}.")
    if x_array.shape != (cube.shape[0],) or y_array.shape != (cube.shape[0],):
        raise ValueError("Scrub track coordinates must match the cube frame count.")

    scrubbed = cube.copy()
    yy, xx = numpy.indices(cube.shape[1:], dtype=float)
    rng = numpy.random.default_rng(int(rng_seed))
    total_masked_pixels = 0
    rows: list[dict[str, float]] = []
    for frame_index, frame in enumerate(scrubbed):
        x0 = float(x_array[frame_index])
        y0 = float(y_array[frame_index])
        rr = numpy.hypot(xx - x0, yy - y0)
        mask = rr <= float(mask_radius)
        annulus = (rr >= float(fill_inner_radius)) & (rr <= float(fill_outer_radius))
        annulus_values = frame[annulus]
        finite_annulus = annulus_values[numpy.isfinite(annulus_values)]
        if finite_annulus.size:
            fill_values = rng.choice(finite_annulus, size=int(mask.sum()), replace=True)
            fill_median = float(numpy.nanmedian(finite_annulus))
            fill_sigma = (
                float(numpy.nanstd(finite_annulus, ddof=1)) if finite_annulus.size >= 2 else 0.0
            )
        else:
            fill_values = numpy.zeros(int(mask.sum()), dtype=float)
            fill_median = 0.0
            fill_sigma = 0.0
        frame[mask] = fill_values
        total_masked_pixels += int(mask.sum())
        rows.append(
            {
                "frame_index": float(frame_index),
                "scrub_x": x0,
                "scrub_y": y0,
                "masked_pixels": float(mask.sum()),
                "fill_median": fill_median,
                "fill_sigma": fill_sigma,
            }
        )

    scrubbed_bundle: dict[str, Any] = {}
    n_frames = int(cube.shape[0])
    for key, value in cube_bundle.items():
        if key == "metadata":
            continue
        try:
            array = numpy.asarray(value)
        except Exception:
            scrubbed_bundle[key] = value
            continue
        if array.ndim >= 1 and array.shape[0] == n_frames:
            scrubbed_bundle[key] = array.copy()
        else:
            scrubbed_bundle[key] = value
    scrubbed_bundle["cube"] = scrubbed
    metadata = dict(cube_bundle.get("metadata", {}))
    metadata["scrubbed_source"] = True
    metadata["scrub_mask_radius"] = float(mask_radius)
    metadata["scrub_fill_inner_radius"] = float(fill_inner_radius)
    metadata["scrub_fill_outer_radius"] = float(fill_outer_radius)
    metadata["scrub_rng_seed"] = int(rng_seed)
    scrubbed_bundle["metadata"] = metadata
    return {
        "bundle": scrubbed_bundle,
        "summary": {
            "mask_radius": float(mask_radius),
            "fill_inner_radius": float(fill_inner_radius),
            "fill_outer_radius": float(fill_outer_radius),
            "rng_seed": int(rng_seed),
            "n_frames": int(n_frames),
            "total_masked_pixels": int(total_masked_pixels),
            "frame_table": rows,
        },
    }


def materialize_cube_bundle_to_npz(bundle: dict[str, Any], output_path: str | Path) -> Path:
    """Write a local cube bundle NPZ with metadata_json."""

    numpy = _require_numpy()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for key, value in bundle.items():
        if key == "metadata":
            continue
        payload[key] = numpy.asarray(value)
    payload["metadata_json"] = json.dumps(bundle.get("metadata", {}), sort_keys=True)
    numpy.savez_compressed(destination, **payload)
    return destination


def compute_direct_track_statistic(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    statistic_name: str = "mean_aperture_snr",
) -> dict[str, Any]:
    """Aggregate direct moving-aperture evidence along a fixed local track."""

    table = evaluate_track_photometry(cube, x, y)
    if statistic_name != "mean_aperture_snr":
        raise ValueError(f"Unsupported direct statistic {statistic_name!r}.")
    score = float(table["aperture_snr"].mean())
    return {
        "statistic_name": statistic_name,
        "score": score,
        "mean_aperture_snr": float(table["aperture_snr"].mean()),
        "median_aperture_snr": float(table["aperture_snr"].median()),
        "max_aperture_snr": float(table["aperture_snr"].max()),
        "mean_net_flux": float(table["net_flux"].mean()),
        "median_net_flux": float(table["net_flux"].median()),
        "table": table,
    }


def compute_stack_track_statistic(
    cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    reference: tuple[float, float],
    interpolation: str = "linear",
    ranking_method: str = "aperture_snr",
) -> dict[str, Any]:
    """Evaluate the stack-based central-source statistic for a fixed track."""

    variant = build_stack_variant(
        cube,
        x,
        y,
        reference=reference,
        interpolation=interpolation,
        ranking_position=reference,
        ranking_method=ranking_method,
    )
    return {
        "statistic_name": ranking_method,
        "score": float(variant["ranking_score"]),
        "variant": variant,
    }


def evaluate_track_pair_statistics(
    cube: Any,
    nominal_x: Iterable[float],
    y: Iterable[float],
    calibrated_x_offset: float,
    reference: tuple[float, float],
    interpolation: str = "linear",
    direct_statistic_name: str = "mean_aperture_snr",
    stack_ranking_method: str = "aperture_snr",
) -> dict[str, Any]:
    """Evaluate nominal, calibrated-x, and best-of-two scores on one cube."""

    numpy = _require_numpy()
    nominal_x_array = numpy.asarray(nominal_x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    calibrated_x_array = nominal_x_array + float(calibrated_x_offset)

    direct_nominal = compute_direct_track_statistic(
        cube,
        nominal_x_array,
        y_array,
        statistic_name=direct_statistic_name,
    )
    direct_calibrated = compute_direct_track_statistic(
        cube,
        calibrated_x_array,
        y_array,
        statistic_name=direct_statistic_name,
    )
    stack_nominal = compute_stack_track_statistic(
        cube,
        nominal_x_array,
        y_array,
        reference=reference,
        interpolation=interpolation,
        ranking_method=stack_ranking_method,
    )
    stack_calibrated = compute_stack_track_statistic(
        cube,
        calibrated_x_array,
        y_array,
        reference=reference,
        interpolation=interpolation,
        ranking_method=stack_ranking_method,
    )
    direct_best_label = (
        "calibrated_x" if float(direct_calibrated["score"]) > float(direct_nominal["score"]) else "nominal"
    )
    stack_best_label = (
        "calibrated_x" if float(stack_calibrated["score"]) > float(stack_nominal["score"]) else "nominal"
    )
    return {
        "direct": {
            "nominal": direct_nominal,
            "calibrated_x": direct_calibrated,
            "best_of_two": {
                "label": direct_best_label,
                "score": float(
                    direct_calibrated["score"] if direct_best_label == "calibrated_x" else direct_nominal["score"]
                ),
            },
        },
        "stack": {
            "nominal": stack_nominal,
            "calibrated_x": stack_calibrated,
            "best_of_two": {
                "label": stack_best_label,
                "score": float(
                    stack_calibrated["score"] if stack_best_label == "calibrated_x" else stack_nominal["score"]
                ),
            },
        },
    }


def evaluate_pseudo_track_ensemble(
    cube: Any,
    nominal_x: Iterable[float],
    y: Iterable[float],
    calibrated_x_offset: float,
    translations: Any,
    interpolation: str = "linear",
    direct_statistic_name: str = "mean_aperture_snr",
    stack_ranking_method: str = "aperture_snr",
) -> Any:
    """Evaluate translated pseudo-tracks with the same nominal/calibrated rule."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    nominal_x_array = numpy.asarray(nominal_x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    translation_table = pandas.DataFrame(translations)
    rows: list[dict[str, float | str]] = []
    for row in translation_table.itertuples(index=False):
        tx = float(row.translation_x)
        ty = float(row.translation_y)
        translated_x = nominal_x_array + tx
        translated_y = y_array + ty
        reference = (float(translated_x[0]), float(translated_y[0]))
        scores = evaluate_track_pair_statistics(
            cube,
            translated_x,
            translated_y,
            calibrated_x_offset=calibrated_x_offset,
            reference=reference,
            interpolation=interpolation,
            direct_statistic_name=direct_statistic_name,
            stack_ranking_method=stack_ranking_method,
        )
        rows.append(
            {
                "realization": float(row.realization),
                "translation_x": tx,
                "translation_y": ty,
                "translation_radius": float(row.translation_radius),
                "direct_nominal_score": float(scores["direct"]["nominal"]["score"]),
                "direct_calibrated_score": float(scores["direct"]["calibrated_x"]["score"]),
                "direct_best_of_two_score": float(scores["direct"]["best_of_two"]["score"]),
                "direct_best_of_two_label": str(scores["direct"]["best_of_two"]["label"]),
                "stack_nominal_score": float(scores["stack"]["nominal"]["score"]),
                "stack_calibrated_score": float(scores["stack"]["calibrated_x"]["score"]),
                "stack_best_of_two_score": float(scores["stack"]["best_of_two"]["score"]),
                "stack_best_of_two_label": str(scores["stack"]["best_of_two"]["label"]),
            }
        )
    return pandas.DataFrame(rows)


def build_empirical_summary(
    observed_scores: dict[str, Any],
    null_table: Any,
) -> dict[str, Any]:
    """Summarize empirical p-values for direct and stack statistics."""

    pandas = _require_pandas()
    null_rows = pandas.DataFrame(null_table)
    metrics = [
        ("direct_nominal", "direct_nominal_score", float(observed_scores["direct"]["nominal"]["score"])),
        (
            "direct_calibrated_x",
            "direct_calibrated_score",
            float(observed_scores["direct"]["calibrated_x"]["score"]),
        ),
        (
            "direct_best_of_two",
            "direct_best_of_two_score",
            float(observed_scores["direct"]["best_of_two"]["score"]),
        ),
        ("stack_nominal", "stack_nominal_score", float(observed_scores["stack"]["nominal"]["score"])),
        (
            "stack_calibrated_x",
            "stack_calibrated_score",
            float(observed_scores["stack"]["calibrated_x"]["score"]),
        ),
        (
            "stack_best_of_two",
            "stack_best_of_two_score",
            float(observed_scores["stack"]["best_of_two"]["score"]),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for label, column_name, observed in metrics:
        null_scores = null_rows[column_name].astype(float)
        p_value = empirical_p_value(observed, null_scores)
        rows.append(
            {
                "metric": label,
                "observed_score": float(observed),
                "null_mean": float(null_scores.mean()),
                "null_median": float(null_scores.median()),
                "null_std": float(null_scores.std(ddof=1)) if null_scores.shape[0] >= 2 else 0.0,
                "null_max": float(null_scores.max()),
                "rank_ge": int((null_scores >= float(observed)).sum()),
                "n_nulls": int(null_scores.shape[0]),
                "empirical_p_value": float(p_value),
            }
        )
    return {
        "rows": rows,
        "metrics": {row["metric"]: row for row in rows},
    }


def build_split_half_indices(n_frames: int, mode: str = "even_odd") -> dict[str, Any]:
    """Return deterministic split indices for reproducibility checks."""

    numpy = _require_numpy()
    if int(n_frames) < 4:
        raise ValueError("Need at least 4 frames for split-half reproducibility tests.")
    all_indices = numpy.arange(int(n_frames), dtype=int)
    if mode == "even_odd":
        return {
            "even": all_indices[all_indices % 2 == 0],
            "odd": all_indices[all_indices % 2 == 1],
        }
    if mode == "first_second":
        midpoint = int(n_frames) // 2
        return {
            "first": all_indices[:midpoint],
            "second": all_indices[midpoint:],
        }
    raise ValueError(f"Unsupported split mode {mode!r}.")


def plot_scrubbed_cube_diagnostic(
    original_cube: Any,
    scrubbed_cube: Any,
    x: Iterable[float],
    y: Iterable[float],
    output_path: str | Path,
    representative_indices: Iterable[int] | None = None,
) -> Path:
    """Plot before/after views of a few representative scrubbed frames."""

    numpy = _require_numpy()
    plotting = _require_matplotlib()
    original = numpy.asarray(original_cube, dtype=float)
    scrubbed = numpy.asarray(scrubbed_cube, dtype=float)
    x_array = numpy.asarray(x, dtype=float)
    y_array = numpy.asarray(y, dtype=float)
    if representative_indices is None:
        representative_indices = [0, original.shape[0] // 2, original.shape[0] - 1]
    indices = [int(value) for value in representative_indices]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plotting.subplots(2, len(indices), figsize=(4.5 * len(indices), 7), constrained_layout=True)
    if len(indices) == 1:
        axes = numpy.asarray(axes).reshape(2, 1)

    for column_index, frame_index in enumerate(indices):
        before = original[frame_index]
        after = scrubbed[frame_index]
        finite = before[numpy.isfinite(before)]
        if finite.size:
            vmin, vmax = numpy.nanpercentile(finite, [5, 99])
        else:
            vmin, vmax = -1.0, 1.0
        for row_index, image in enumerate((before, after)):
            axes[row_index, column_index].imshow(image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            axes[row_index, column_index].scatter(
                [x_array[frame_index]],
                [y_array[frame_index]],
                marker="x",
                color="white",
                s=45,
                linewidths=1.2,
            )
            axes[row_index, column_index].set_title(
                f"{'Before' if row_index == 0 else 'After'} frame {frame_index}"
            )
            axes[row_index, column_index].set_xlabel("x [pix]")
            axes[row_index, column_index].set_ylabel("y [pix]")

    figure.savefig(destination, dpi=170)
    plotting.close(figure)
    return destination


def plot_empirical_null_distribution(
    null_table: Any,
    observed_scores: dict[str, Any],
    metric_prefix: str,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot matched-null histograms for nominal, calibrated, and best-of-two metrics."""

    pandas = _require_pandas()
    plotting = _require_matplotlib()
    null_rows = pandas.DataFrame(null_table)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if metric_prefix == "direct":
        columns = [
            ("Nominal", "direct_nominal_score", float(observed_scores["direct"]["nominal"]["score"])),
            (
                "Calibrated x",
                "direct_calibrated_score",
                float(observed_scores["direct"]["calibrated_x"]["score"]),
            ),
            (
                "Best of Two",
                "direct_best_of_two_score",
                float(observed_scores["direct"]["best_of_two"]["score"]),
            ),
        ]
    elif metric_prefix == "stack":
        columns = [
            ("Nominal", "stack_nominal_score", float(observed_scores["stack"]["nominal"]["score"])),
            (
                "Calibrated x",
                "stack_calibrated_score",
                float(observed_scores["stack"]["calibrated_x"]["score"]),
            ),
            (
                "Best of Two",
                "stack_best_of_two_score",
                float(observed_scores["stack"]["best_of_two"]["score"]),
            ),
        ]
    else:
        raise ValueError(f"Unsupported metric_prefix {metric_prefix!r}.")

    figure, axes = plotting.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, (panel_title, column_name, observed) in zip(axes, columns):
        scores = null_rows[column_name].astype(float)
        axis.hist(scores, bins=min(25, max(10, scores.shape[0] // 8)), alpha=0.8, color="tab:blue")
        axis.axvline(observed, color="tab:red", linestyle="--", linewidth=1.4)
        axis.set_title(f"{panel_title}\np={empirical_p_value(observed, scores):.3f}")
        axis.set_xlabel("Score")
        axis.set_ylabel("Null count")
        axis.grid(alpha=0.25)
    figure.suptitle(title)
    figure.savefig(destination, dpi=170)
    plotting.close(figure)
    return destination


def plot_split_half_reproducibility(
    split_summary: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Plot split-half best-of-two observed scores against null medians."""

    plotting = _require_matplotlib()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    labels = list(split_summary)
    direct_obs = [float(split_summary[label]["direct_best_of_two"]["observed_score"]) for label in labels]
    direct_null = [float(split_summary[label]["direct_best_of_two"]["null_median"]) for label in labels]
    stack_obs = [float(split_summary[label]["stack_best_of_two"]["observed_score"]) for label in labels]
    stack_null = [float(split_summary[label]["stack_best_of_two"]["null_median"]) for label in labels]

    figure, axes = plotting.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x_index = list(range(len(labels)))
    axes[0].bar([index - 0.18 for index in x_index], direct_obs, width=0.35, label="observed")
    axes[0].bar([index + 0.18 for index in x_index], direct_null, width=0.35, label="null median")
    axes[0].set_xticks(x_index, labels)
    axes[0].set_title("Split-half direct statistic")
    axes[0].set_ylabel("mean aperture S/N")
    axes[0].legend(loc="best")

    axes[1].bar([index - 0.18 for index in x_index], stack_obs, width=0.35, label="observed")
    axes[1].bar([index + 0.18 for index in x_index], stack_null, width=0.35, label="null median")
    axes[1].set_xticks(x_index, labels)
    axes[1].set_title("Split-half stack statistic")
    axes[1].set_ylabel("stack aperture S/N")
    axes[1].legend(loc="best")

    figure.savefig(destination, dpi=170)
    plotting.close(figure)
    return destination


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON payload with stable formatting."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return destination
