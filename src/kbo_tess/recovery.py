"""Detector-local known-path recovery helpers for local cube bundles."""

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
from kbo_tess.ephem import load_ephemeris_csv, normalize_ephemeris_dataframe, validate_single_detector_track
from kbo_tess.shiftstack import score_stack, search_local_track_offsets, stack_along_track


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for recovery. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return np


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for recovery. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return pd


def _require_matplotlib() -> Any:
    if plt is None:
        raise ImportError(
            "matplotlib is required for recovery diagnostics. Install project dependencies "
            "with `python -m pip install -e .`."
        )
    return plt


def _parse_metadata_payload(raw_value: Any) -> dict[str, Any]:
    if raw_value is None:
        return {}
    payload = raw_value
    if hasattr(payload, "item"):
        payload = payload.item()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            return {}
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("`metadata_json` must decode to a JSON object.")
        return parsed
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported metadata payload type: {type(payload)!r}.")


def load_cube_npz(path: str | Path) -> dict[str, Any]:
    """Load a detector-local cube bundle NPZ.

    Required keys:
    - `cube`: float array with shape `(n_frames, ny, nx)`
    - `time_tdb_jd_minus_2457000`: float array with length `n_frames`

    Any additional arrays in the NPZ are preserved and returned.
    """

    numpy = _require_numpy()
    source_path = Path(path)
    with numpy.load(source_path, allow_pickle=False) as bundle:
        if "cube" not in bundle:
            raise KeyError(f"{source_path} is missing required `cube` array.")
        if "time_tdb_jd_minus_2457000" not in bundle:
            raise KeyError(
                f"{source_path} is missing required `time_tdb_jd_minus_2457000` array."
            )
        loaded: dict[str, Any] = {
            key: numpy.asarray(bundle[key]) for key in bundle.files if key != "metadata_json"
        }
        metadata = _parse_metadata_payload(bundle["metadata_json"]) if "metadata_json" in bundle else {}

    cube = numpy.asarray(loaded["cube"], dtype=float)
    cube_times = numpy.asarray(loaded["time_tdb_jd_minus_2457000"], dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"Expected `cube` with shape (n_frames, ny, nx); got {cube.shape!r}.")
    if cube_times.ndim != 1:
        raise ValueError(
            f"Expected `time_tdb_jd_minus_2457000` with shape (n_frames,); got {cube_times.shape!r}."
        )
    if cube.shape[0] != cube_times.shape[0]:
        raise ValueError(
            "Cube frame count and cube time count do not match; "
            f"got {cube.shape[0]} and {cube_times.shape[0]}."
        )
    if "quality_mask" in loaded:
        quality_mask = numpy.asarray(loaded["quality_mask"], dtype=bool)
        if quality_mask.shape != (cube.shape[0],):
            raise ValueError(
                f"Expected `quality_mask` with shape {(cube.shape[0],)!r}; got {quality_mask.shape!r}."
            )
        loaded["quality_mask"] = quality_mask

    loaded["cube"] = cube
    loaded["time_tdb_jd_minus_2457000"] = cube_times
    loaded["metadata"] = dict(metadata)
    loaded["metadata"].setdefault("cube_npz_path", str(source_path))
    return loaded


def _frame_aligned_array(value: Any, n_frames: int) -> Any | None:
    numpy = _require_numpy()
    try:
        array = numpy.asarray(value)
    except Exception:
        return None
    if array.ndim >= 1 and array.shape[0] == n_frames:
        return array
    return None


def _filter_cube_bundle(cube_bundle: dict[str, Any]) -> dict[str, Any]:
    numpy = _require_numpy()
    cube = numpy.asarray(cube_bundle["cube"], dtype=float)
    cube_times = numpy.asarray(cube_bundle["time_tdb_jd_minus_2457000"], dtype=float)
    quality_mask = cube_bundle.get("quality_mask")
    if quality_mask is None:
        keep_mask = numpy.ones(cube.shape[0], dtype=bool)
    else:
        keep_mask = numpy.asarray(quality_mask, dtype=bool)
    if keep_mask.shape != (cube.shape[0],):
        raise ValueError(
            f"Quality mask must have shape {(cube.shape[0],)!r}; got {keep_mask.shape!r}."
        )
    if not keep_mask.any():
        raise ValueError("Quality mask rejected every frame; nothing remains to recover.")
    filtered_times = cube_times[keep_mask]
    if numpy.any(~numpy.isfinite(filtered_times)):
        raise ValueError("Cube times must be finite.")
    if filtered_times.size > 1 and numpy.any(numpy.diff(filtered_times) <= 0):
        raise ValueError("Cube times must be strictly increasing after quality masking.")

    filtered: dict[str, Any] = {}
    for key, value in cube_bundle.items():
        if key == "metadata":
            continue
        frame_aligned = _frame_aligned_array(value, cube.shape[0])
        if frame_aligned is None:
            filtered[key] = value
        else:
            filtered[key] = frame_aligned[keep_mask]

    metadata = dict(cube_bundle.get("metadata", {}))
    metadata["n_frames_total"] = int(cube.shape[0])
    metadata["n_frames_used"] = int(keep_mask.sum())
    filtered["metadata"] = metadata
    return filtered


def filter_cube_bundle(cube_bundle: dict[str, Any]) -> dict[str, Any]:
    """Public wrapper for applying the bundle quality mask consistently."""

    return _filter_cube_bundle(cube_bundle)


def align_ephemeris_to_cube_times(ephem_df: Any, cube_times: Any) -> Any:
    """Interpolate a detector-local ephemeris onto cube cadence times."""

    numpy = _require_numpy()
    pandas = _require_pandas()
    normalized = normalize_ephemeris_dataframe(ephem_df, require_exact_tdb=True)
    validate_single_detector_track(normalized, require_exact_tdb=True)

    cube_time_array = numpy.asarray(cube_times, dtype=float)
    if cube_time_array.ndim != 1:
        raise ValueError(
            f"Cube times must be one-dimensional; got shape {cube_time_array.shape!r}."
        )

    ephem_times = numpy.asarray(normalized["time_tdb_jd_minus_2457000"], dtype=float)
    if ephem_times.size < 2:
        raise ValueError("Need at least two ephemeris samples to align to cube times.")
    if numpy.any(~numpy.isfinite(ephem_times)) or numpy.any(~numpy.isfinite(cube_time_array)):
        raise ValueError("Ephemeris times and cube times must be finite.")
    if numpy.any(numpy.diff(ephem_times) <= 0):
        raise ValueError("Ephemeris times must be strictly increasing for interpolation.")
    if cube_time_array.min() < ephem_times.min() or cube_time_array.max() > ephem_times.max():
        raise ValueError(
            "Cube times extend outside the ephemeris span. "
            f"Cube span=({cube_time_array.min()}, {cube_time_array.max()}), "
            f"ephemeris span=({ephem_times.min()}, {ephem_times.max()})."
        )

    aligned = pandas.DataFrame(
        {
            "time_tdb_jd_minus_2457000": cube_time_array,
            "x": numpy.interp(cube_time_array, ephem_times, normalized["x"].astype(float)),
            "y": numpy.interp(cube_time_array, ephem_times, normalized["y"].astype(float)),
        }
    )
    for column_name in (
        "target",
        "sector",
        "camera",
        "ccd",
        "time_scale",
        "time_reference",
        "time_reference_source",
    ):
        if column_name in normalized.columns:
            aligned[column_name] = normalized.iloc[0][column_name]
    aligned["alignment_method"] = "linear_interp_in_time"
    return aligned


def align_ephemeris_to_cube_bundle(ephem_df: Any, cube_bundle: dict[str, Any]) -> Any:
    """Interpolate ephemeris to cube times and derive local cutout coordinates when possible."""

    numpy = _require_numpy()
    aligned = align_ephemeris_to_cube_times(ephem_df, cube_bundle["time_tdb_jd_minus_2457000"])

    for bundle_column, aligned_column in (
        ("local_x", "bundle_local_x"),
        ("local_y", "bundle_local_y"),
        ("corner1", "corner1"),
        ("corner2", "corner2"),
        ("pos_corr1", "pos_corr1"),
        ("pos_corr2", "pos_corr2"),
        ("detector_x", "bundle_detector_x"),
        ("detector_y", "bundle_detector_y"),
    ):
        if bundle_column in cube_bundle:
            values = numpy.asarray(cube_bundle[bundle_column], dtype=float)
            if values.shape == (len(aligned),):
                aligned[aligned_column] = values

    if "corner1" in aligned.columns and "corner2" in aligned.columns:
        aligned["local_x_from_corner"] = aligned["x"].astype(float) - aligned["corner1"].astype(float)
        aligned["local_y_from_corner"] = aligned["y"].astype(float) - aligned["corner2"].astype(float)

    cube_shape = numpy.asarray(cube_bundle["cube"]).shape[1:]
    if "pos_corr1" in aligned.columns and "pos_corr2" in aligned.columns:
        center_x = 0.5 * (float(cube_shape[1]) - 1.0)
        center_y = 0.5 * (float(cube_shape[0]) - 1.0)
        aligned["local_x_from_poscorr"] = center_x + aligned["pos_corr1"].astype(float)
        aligned["local_y_from_poscorr"] = center_y + aligned["pos_corr2"].astype(float)

    if "bundle_local_x" in aligned.columns and "local_x_from_corner" in aligned.columns:
        aligned["local_corner_residual_x"] = (
            aligned["local_x_from_corner"].astype(float) - aligned["bundle_local_x"].astype(float)
        )
        aligned["local_corner_residual_y"] = (
            aligned["local_y_from_corner"].astype(float) - aligned["bundle_local_y"].astype(float)
        )
        aligned["local_corner_residual_r"] = numpy.hypot(
            aligned["local_corner_residual_x"].astype(float),
            aligned["local_corner_residual_y"].astype(float),
        )

    if "bundle_local_x" in aligned.columns and "local_x_from_poscorr" in aligned.columns:
        aligned["local_poscorr_residual_x"] = (
            aligned["local_x_from_poscorr"].astype(float) - aligned["bundle_local_x"].astype(float)
        )
        aligned["local_poscorr_residual_y"] = (
            aligned["local_y_from_poscorr"].astype(float) - aligned["bundle_local_y"].astype(float)
        )
        aligned["local_poscorr_residual_r"] = numpy.hypot(
            aligned["local_poscorr_residual_x"].astype(float),
            aligned["local_poscorr_residual_y"].astype(float),
        )

    if "bundle_detector_x" in aligned.columns:
        aligned["detector_residual_x"] = aligned["x"].astype(float) - aligned["bundle_detector_x"].astype(float)
        aligned["detector_residual_y"] = aligned["y"].astype(float) - aligned["bundle_detector_y"].astype(float)
        aligned["detector_residual_r"] = numpy.hypot(
            aligned["detector_residual_x"].astype(float),
            aligned["detector_residual_y"].astype(float),
        )

    return aligned


def _resolve_track_coordinates(aligned_ephem: Any) -> tuple[Any, Any, str]:
    numpy = _require_numpy()
    if {"bundle_local_x", "bundle_local_y"}.issubset(aligned_ephem.columns):
        return (
            numpy.asarray(aligned_ephem["bundle_local_x"], dtype=float),
            numpy.asarray(aligned_ephem["bundle_local_y"], dtype=float),
            "cutout_local_bundle",
        )
    if {"local_x_from_corner", "local_y_from_corner"}.issubset(aligned_ephem.columns):
        return (
            numpy.asarray(aligned_ephem["local_x_from_corner"], dtype=float),
            numpy.asarray(aligned_ephem["local_y_from_corner"], dtype=float),
            "cutout_local_from_corner",
        )
    return (
        numpy.asarray(aligned_ephem["x"], dtype=float),
        numpy.asarray(aligned_ephem["y"], dtype=float),
        "detector_input",
    )


def resolve_track_coordinates(aligned_ephem: Any) -> tuple[Any, Any, str]:
    """Public wrapper for choosing the best available detector-local track coordinates."""

    return _resolve_track_coordinates(aligned_ephem)


def _build_offset_grid(radius: float, step: float) -> Any:
    numpy = _require_numpy()
    if radius < 0:
        raise ValueError(f"`offset_grid_radius` must be non-negative; got {radius}.")
    if radius == 0:
        return numpy.asarray([0.0], dtype=float)
    if step <= 0:
        raise ValueError(f"`offset_grid_step` must be positive; got {step}.")
    return numpy.arange(-radius, radius + 0.5 * step, step, dtype=float)


def _compute_score_dict(stack: Any, position: tuple[float, float]) -> dict[str, float]:
    return {
        "peak": float(score_stack(stack, method="peak", position=position)),
        "snr": float(score_stack(stack, method="snr", position=position)),
        "aperture_snr": float(score_stack(stack, method="aperture_snr", position=position)),
    }


def _growth_checkpoint_counts(n_frames: int) -> list[int]:
    if n_frames <= 5:
        return list(range(2, n_frames + 1))
    counts = {2, n_frames}
    for fraction in (0.25, 0.5, 0.75):
        counts.add(max(2, min(n_frames, int(round(fraction * n_frames)))))
    return sorted(counts)


def _compute_growth_curve(
    residual_cube: Any,
    x: Any,
    y: Any,
    interpolation: str,
    reference_pixel: tuple[float, float],
    best_dx: float,
    best_dy: float,
    ranking_score_method: str,
) -> list[dict[str, float]]:
    growth: list[dict[str, float]] = []
    for count in _growth_checkpoint_counts(int(len(x))):
        nominal_stack = stack_along_track(
            residual_cube[:count],
            x[:count],
            y[:count],
            method="mean",
            interpolation=interpolation,
            reference=reference_pixel,
        )
        best_stack = stack_along_track(
            residual_cube[:count],
            x[:count] + float(best_dx),
            y[:count] + float(best_dy),
            method="mean",
            interpolation=interpolation,
            reference=reference_pixel,
        )
        nominal_scores = _compute_score_dict(nominal_stack, reference_pixel)
        best_scores = _compute_score_dict(best_stack, reference_pixel)
        growth.append(
            {
                "n_frames": float(count),
                "nominal_peak": nominal_scores["peak"],
                "nominal_snr": nominal_scores["snr"],
                "nominal_aperture_snr": nominal_scores["aperture_snr"],
                "best_peak": best_scores["peak"],
                "best_snr": best_scores["snr"],
                "best_aperture_snr": best_scores["aperture_snr"],
                "ranking_score_method": ranking_score_method,
                "nominal_ranking_score": nominal_scores[ranking_score_method],
                "best_ranking_score": best_scores[ranking_score_method],
            }
        )
    return growth


def run_known_path_recovery(
    ephem_df: Any,
    cube_bundle: dict[str, Any],
    background_method: str = "median",
    interpolation: str = "nearest",
    offset_grid_radius: float = 1.0,
    offset_grid_step: float = 0.5,
    ranking_score_method: str = "aperture_snr",
) -> dict[str, Any]:
    """Run detector-local known-path recovery for a cube and ephemeris."""

    numpy = _require_numpy()
    filtered_bundle = _filter_cube_bundle(cube_bundle)
    cube = numpy.asarray(filtered_bundle["cube"], dtype=float)
    cube_times = numpy.asarray(filtered_bundle["time_tdb_jd_minus_2457000"], dtype=float)
    metadata = dict(filtered_bundle.get("metadata", {}))

    aligned_ephem = align_ephemeris_to_cube_bundle(ephem_df, filtered_bundle)
    x, y, track_coordinate_system = _resolve_track_coordinates(aligned_ephem)
    reference_pixel = (float(x[0]), float(y[0]))

    baseline, residual = subtract_background(cube, method=background_method)
    nominal_stack = stack_along_track(
        residual,
        x,
        y,
        method="mean",
        interpolation=interpolation,
        reference=reference_pixel,
    )
    nominal_scores = _compute_score_dict(nominal_stack, reference_pixel)

    dx_grid = _build_offset_grid(offset_grid_radius, offset_grid_step)
    dy_grid = _build_offset_grid(offset_grid_radius, offset_grid_step)
    offset_search = search_local_track_offsets(
        residual,
        x,
        y,
        dx_grid=dx_grid,
        dy_grid=dy_grid,
        interpolation=interpolation,
        stack_method="mean",
        score_method=ranking_score_method,
        reference=reference_pixel,
        score_position=reference_pixel,
    )
    best_stack = numpy.asarray(offset_search["best"]["stack"], dtype=float)
    best_scores = _compute_score_dict(best_stack, reference_pixel)

    representative_index = int(residual.shape[0] // 2)
    representative_position = (float(x[representative_index]), float(y[representative_index]))
    representative_single_frame_scores = _compute_score_dict(
        residual[representative_index],
        representative_position,
    )
    growth_curve = _compute_growth_curve(
        residual,
        x,
        y,
        interpolation=interpolation,
        reference_pixel=reference_pixel,
        best_dx=float(offset_search["best"]["dx"]),
        best_dy=float(offset_search["best"]["dy"]),
        ranking_score_method=ranking_score_method,
    )

    summary = {
        "target": str(aligned_ephem.iloc[0]["target"]) if "target" in aligned_ephem.columns else metadata.get("target", "unknown"),
        "sector": int(aligned_ephem.iloc[0]["sector"]) if "sector" in aligned_ephem.columns else metadata.get("sector"),
        "camera": int(aligned_ephem.iloc[0]["camera"]) if "camera" in aligned_ephem.columns else metadata.get("camera"),
        "ccd": int(aligned_ephem.iloc[0]["ccd"]) if "ccd" in aligned_ephem.columns else metadata.get("ccd"),
        "n_frames": int(residual.shape[0]),
        "n_frames_total": int(metadata.get("n_frames_total", residual.shape[0])),
        "background_method": background_method,
        "loader_bg_method": metadata.get("loader_bg_method"),
        "interpolation_mode": interpolation,
        "alignment_method": str(aligned_ephem.iloc[0]["alignment_method"]),
        "track_coordinate_system": track_coordinate_system,
        "ranking_score_method": ranking_score_method,
        "best_dx": float(offset_search["best"]["dx"]),
        "best_dy": float(offset_search["best"]["dy"]),
        "best_score": float(offset_search["best"]["score"]),
        "nominal_score": float(nominal_scores[ranking_score_method]),
        "nominal_scores": nominal_scores,
        "best_stack_scores": best_scores,
        "representative_single_frame_scores": representative_single_frame_scores,
        "recovered_reference_pixel": {
            "x": float(reference_pixel[0]),
            "y": float(reference_pixel[1]),
        },
    }
    for residual_column, output_name in (
        ("local_corner_residual_r", "geometry_residual_corner_median_px"),
        ("local_poscorr_residual_r", "geometry_residual_poscorr_median_px"),
        ("detector_residual_r", "geometry_residual_detector_median_px"),
    ):
        if residual_column in aligned_ephem.columns:
            summary[output_name] = float(numpy.nanmedian(aligned_ephem[residual_column].astype(float)))

    return {
        "cube": cube,
        "cube_times": cube_times,
        "metadata": metadata,
        "aligned_ephemeris": aligned_ephem,
        "baseline_cube": baseline,
        "residual_cube": residual,
        "representative_frame": residual[representative_index],
        "representative_frame_index": representative_index,
        "representative_position": representative_position,
        "nominal_stack": nominal_stack,
        "offset_search": offset_search,
        "growth_curve": growth_curve,
        "summary": summary,
    }


def recover_from_local_files(
    ephem_csv: str | Path,
    cube_npz: str | Path,
    background_method: str = "median",
    interpolation: str = "nearest",
    offset_grid_radius: float = 1.0,
    offset_grid_step: float = 0.5,
    ranking_score_method: str = "aperture_snr",
) -> dict[str, Any]:
    """Run the local-file recovery path from a recovery-ready ephemeris CSV and NPZ cube."""

    ephem_df = load_ephemeris_csv(ephem_csv, require_exact_tdb=True)
    cube_bundle = load_cube_npz(cube_npz)
    return run_known_path_recovery(
        ephem_df,
        cube_bundle,
        background_method=background_method,
        interpolation=interpolation,
        offset_grid_radius=offset_grid_radius,
        offset_grid_step=offset_grid_step,
        ranking_score_method=ranking_score_method,
    )


def _score_grid_from_results(results: list[dict[str, float]]) -> tuple[Any, Any, Any]:
    numpy = _require_numpy()
    dx_values = numpy.asarray(sorted({float(item["dx"]) for item in results}), dtype=float)
    dy_values = numpy.asarray(sorted({float(item["dy"]) for item in results}), dtype=float)
    score_map = numpy.full((dy_values.size, dx_values.size), numpy.nan, dtype=float)
    dx_index = {float(value): index for index, value in enumerate(dx_values)}
    dy_index = {float(value): index for index, value in enumerate(dy_values)}
    for result in results:
        score_map[dy_index[float(result["dy"])], dx_index[float(result["dx"])]] = float(result["score"])
    return score_map, dx_values, dy_values


def write_recovery_products(
    recovery_result: dict[str, Any],
    outdir: str | Path,
    prefix: str | None = None,
) -> dict[str, Path]:
    """Write summary, search tables, and a diagnostic PNG."""

    pandas = _require_pandas()
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = dict(recovery_result["summary"])
    if prefix is None:
        target = re.sub(r"[^a-z0-9]+", "_", str(summary.get("target", "target")).lower()).strip("_")
        sector = summary.get("sector", "unknown")
        prefix = f"{target}_sector{sector}"

    summary_path = output_dir / f"{prefix}_recovery_summary.json"
    offset_path = output_dir / f"{prefix}_offset_search.csv"
    growth_path = output_dir / f"{prefix}_growth_curve.csv"
    diagnostic_path = output_dir / f"{prefix}_diagnostic.png"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    pandas.DataFrame(recovery_result["offset_search"]["results"]).to_csv(offset_path, index=False)
    pandas.DataFrame(recovery_result["growth_curve"]).to_csv(growth_path, index=False)
    plot_recovery_diagnostics(recovery_result, diagnostic_path)
    return {
        "summary_json": summary_path,
        "offset_csv": offset_path,
        "growth_csv": growth_path,
        "diagnostic_png": diagnostic_path,
    }


def plot_recovery_diagnostics(recovery_result: dict[str, Any], output_path: str | Path) -> Path:
    """Write a recovery diagnostic PNG with stack and score context."""

    plotting = _require_matplotlib()
    numpy = _require_numpy()

    representative_frame = numpy.asarray(recovery_result["representative_frame"], dtype=float)
    nominal_stack = numpy.asarray(recovery_result["nominal_stack"], dtype=float)
    best_stack = numpy.asarray(recovery_result["offset_search"]["best"]["stack"], dtype=float)
    reference_pixel = recovery_result["summary"]["recovered_reference_pixel"]
    representative_position = recovery_result["representative_position"]

    figure, axes = plotting.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], representative_frame, "Representative Residual Frame", representative_position),
        (axes[0, 1], nominal_stack, "Nominal Known-Path Stack", (reference_pixel["x"], reference_pixel["y"])),
        (axes[0, 2], best_stack, "Best Offset Stack", (reference_pixel["x"], reference_pixel["y"])),
    )
    for axis, image, title, marker in panels:
        finite = image[numpy.isfinite(image)]
        if finite.size:
            vmin, vmax = numpy.nanpercentile(finite, [5, 99])
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0
        else:
            vmin, vmax = -1.0, 1.0
        artist = axis.imshow(image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axis.scatter([marker[0]], [marker[1]], marker="x", color="white", s=50, linewidths=1.5)
        axis.set_title(title)
        axis.set_xlabel("x [pix]")
        axis.set_ylabel("y [pix]")
        figure.colorbar(artist, ax=axis, shrink=0.8)

    score_map, dx_values, dy_values = _score_grid_from_results(recovery_result["offset_search"]["results"])
    score_axis = axes[1, 0]
    if dx_values.size > 1 or dy_values.size > 1:
        artist = score_axis.imshow(
            score_map,
            origin="lower",
            aspect="auto",
            extent=(dx_values.min(), dx_values.max(), dy_values.min(), dy_values.max()),
            cmap="magma",
        )
        best = recovery_result["offset_search"]["best"]
        score_axis.scatter([best["dx"]], [best["dy"]], marker="x", color="white", s=60, linewidths=1.5)
        score_axis.set_title("Offset Search Score Map")
        score_axis.set_xlabel("dx [pix]")
        score_axis.set_ylabel("dy [pix]")
        figure.colorbar(artist, ax=score_axis, shrink=0.8)
    else:
        score_axis.axis("off")

    growth_axis = axes[1, 1]
    growth = recovery_result.get("growth_curve", [])
    if growth:
        counts = [row["n_frames"] for row in growth]
        growth_axis.plot(counts, [row["nominal_ranking_score"] for row in growth], marker="o", label="nominal")
        growth_axis.plot(counts, [row["best_ranking_score"] for row in growth], marker="o", label="best offset")
        growth_axis.set_title("Growth Curve")
        growth_axis.set_xlabel("Cumulative frames")
        growth_axis.set_ylabel(recovery_result["summary"]["ranking_score_method"])
        growth_axis.legend(loc="best")
        growth_axis.grid(alpha=0.3)
    else:
        growth_axis.axis("off")

    text_axis = axes[1, 2]
    text_axis.axis("off")
    summary = recovery_result["summary"]
    text_axis.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Frames: {summary['n_frames']}",
                f"Loader BG: {summary.get('loader_bg_method')}",
                f"Temporal BG: {summary['background_method']}",
                f"Track coords: {summary['track_coordinate_system']}",
                f"Best dx,dy: ({summary['best_dx']:.2f}, {summary['best_dy']:.2f})",
                f"Best {summary['ranking_score_method']}: {summary['best_score']:.3f}",
                f"Nominal {summary['ranking_score_method']}: {summary['nominal_score']:.3f}",
                f"Single-frame {summary['ranking_score_method']}: {summary['representative_single_frame_scores'][summary['ranking_score_method']]:.3f}",
            ]
        ),
        transform=text_axis.transAxes,
        va="top",
        ha="left",
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plotting.close(figure)
    return destination
