#!/usr/bin/env python3
"""Experiment: mask bright catalog sources in raw moving-target frames, subtract a simple background, and stack."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kbo_tess.ephem import load_ephemeris_csv
from kbo_tess.shiftstack import apply_support_gate, score_stack, stack_with_support
from kbo_tess.tutorial_benchmark import _configure_benchmark_environment, _ephem_for_tess_asteroids, _import_tess_asteroids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mask bright catalog sources in raw moving-target frames, subtract a simple background, and stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Pluto")
    parser.add_argument("--sector", type=int, default=92)
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--ccd", type=int, default=1)
    parser.add_argument(
        "--ephem-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_calibration" / "pluto_all_detector_inventory.csv",
    )
    parser.add_argument(
        "--bundle-npz",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "pluto_full_duration_conservative_mask"
        / "pluto_sector92_cam2_ccd1_full_cube_bundle_masked.npz",
    )
    parser.add_argument(
        "--bright-csv",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "pluto_full_duration_catalog_sources"
        / "pluto_sector92_cam2_ccd1_tic_crossers_bright.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_catalog_mask_stack",
    )
    parser.add_argument("--cutout-size", type=int, default=21)
    parser.add_argument("--central-box-size", type=int, default=5)
    parser.add_argument("--mag-limit", type=float, default=15.5)
    parser.add_argument("--interpolation", choices=("nearest", "linear"), default="linear")
    parser.add_argument("--support-fraction-min", type=float, default=0.0)
    parser.add_argument("--support-frames-min", type=float, default=0.0)
    return parser


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


def build_raw_flux_cube(ephem: pd.DataFrame, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    _configure_benchmark_environment()
    module = _import_tess_asteroids()
    tess_ephem = _ephem_for_tess_asteroids(ephem)
    moving = module.MovingTPF(target=str(ephem.iloc[0]["target"]), ephem=tess_ephem, barycentric=True, metadata={})
    moving.make_tpf(shape=shape, bg_method="rolling", save=False)
    raw_flux = np.asarray(getattr(moving, "flux"), dtype=float)
    raw_time = np.asarray(getattr(moving, "time"), dtype=float)
    return raw_flux, raw_time


def match_bundle_times_to_raw(raw_time: np.ndarray, bundle_time: np.ndarray) -> np.ndarray:
    match_idx = np.array([int(np.nanargmin(np.abs(raw_time - value))) for value in bundle_time], dtype=int)
    dt_seconds = np.abs(raw_time[match_idx] - bundle_time) * 86400.0
    if float(np.nanmax(dt_seconds)) > 1.0:
        raise ValueError(f"Raw/bundle cadence matching drifted too far; max dt is {float(np.nanmax(dt_seconds)):.3f} s.")
    return match_idx


def mask_radius_from_tmag(tmag: float) -> float:
    if not np.isfinite(tmag):
        return 1.5
    # Simple experimental rule: brighter stars get a larger circular mask.
    return float(np.clip(1.25 + 0.50 * (15.5 - float(tmag)), 1.25, 4.0))


def center_box_mask(shape: tuple[int, int], box_size: int) -> np.ndarray:
    ny, nx = shape
    cx = nx // 2
    cy = ny // 2
    half = int(box_size // 2)
    mask = np.zeros(shape, dtype=bool)
    mask[max(0, cy - half) : min(ny, cy + half + 1), max(0, cx - half) : min(nx, cx + half + 1)] = True
    return mask


def process_cube(
    raw_cube: np.ndarray,
    corner1: np.ndarray,
    corner2: np.ndarray,
    bright_sources: pd.DataFrame,
    central_mask: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    n_frames, ny, nx = raw_cube.shape
    yy, xx = np.indices((ny, nx), dtype=float)

    source_x = bright_sources["detector_x"].astype(float).to_numpy()
    source_y = bright_sources["detector_y"].astype(float).to_numpy()
    mask_radius = bright_sources["mask_radius_pix"].astype(float).to_numpy()
    local_x = source_x[None, :] - corner1[:, None]
    local_y = source_y[None, :] - corner2[:, None]
    max_radius = float(np.nanmax(mask_radius))
    inside = (
        (local_x >= -max_radius)
        & (local_x <= float(nx - 1) + max_radius)
        & (local_y >= -max_radius)
        & (local_y <= float(ny - 1) + max_radius)
    )

    processed = np.full_like(raw_cube, np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    for index in range(n_frames):
        frame = np.asarray(raw_cube[index], dtype=float)
        mask = np.zeros((ny, nx), dtype=bool)
        for source_index in np.flatnonzero(inside[index]):
            rr = np.hypot(xx - float(local_x[index, source_index]), yy - float(local_y[index, source_index]))
            mask |= rr <= float(mask_radius[source_index])
        mask &= ~central_mask

        background_mask = (~mask) & (~central_mask) & np.isfinite(frame)
        background_mean = float(np.nanmean(frame[background_mask])) if np.any(background_mask) else 0.0
        centered = frame - background_mean
        centered[mask] = np.nan
        processed[index] = centered
        rows.append(
            {
                "frame_index": int(index),
                "background_mean": background_mean,
                "n_masked_pixels": int(np.count_nonzero(mask)),
                "n_background_pixels": int(np.count_nonzero(background_mask)),
            }
        )

    summary = {
        "n_frames": int(n_frames),
        "mean_masked_pixels_per_frame": float(np.mean([row["n_masked_pixels"] for row in rows])),
        "max_masked_pixels_per_frame": int(max(row["n_masked_pixels"] for row in rows)),
        "mean_background_pixels_per_frame": float(np.mean([row["n_background_pixels"] for row in rows])),
        "mean_background_mean": float(np.mean([row["background_mean"] for row in rows])),
        "median_background_mean": float(np.median([row["background_mean"] for row in rows])),
    }
    return processed, pd.DataFrame(rows), summary


def plot_stack_heatmap(
    stack: np.ndarray,
    support_map: np.ndarray,
    output_path: Path,
    target: str,
    support_threshold_frames: float | None = None,
) -> None:
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("black")
    vmin = float(np.nanpercentile(stack, 5))
    vmax = float(np.nanpercentile(stack, 99.5))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    im0 = axes[0].imshow(stack, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].scatter([10], [10], marker="x", s=60, linewidths=1.3, color="cyan")
    title = f"Catalog-masked {target} nominal stack"
    if support_threshold_frames is not None:
        title += f"\nSupport gate >= {support_threshold_frames:.1f} frames"
    axes[0].set_title(title)
    axes[0].set_xlabel("Local x [pix]")
    axes[0].set_ylabel("Local y [pix]")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="Mean stacked flux")

    im1 = axes[1].imshow(support_map, origin="lower", cmap="viridis")
    axes[1].scatter([10], [10], marker="x", s=60, linewidths=1.3, color="white")
    axes[1].set_title("Support map")
    axes[1].set_xlabel("Local x [pix]")
    axes[1].set_ylabel("Local y [pix]")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="Contributing frames")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(args.bundle_npz)
    bundle_time = np.asarray(bundle["time_tdb_jd_minus_2457000"], dtype=float)
    local_x = np.asarray(bundle["local_x"], dtype=float)
    local_y = np.asarray(bundle["local_y"], dtype=float)
    corner1 = np.asarray(bundle["corner1"], dtype=float)
    corner2 = np.asarray(bundle["corner2"], dtype=float)
    metadata = json.loads(str(bundle["metadata_json"])) if "metadata_json" in bundle else {}

    prefix = f"{str(args.target).strip().lower()}_sector{int(args.sector)}_cam{int(args.camera)}_ccd{int(args.ccd)}"

    bright = pd.read_csv(args.bright_csv)
    bright = bright[bright["tmag"].fillna(99.0) <= float(args.mag_limit)].copy().reset_index(drop=True)
    bright["mask_radius_pix"] = bright["tmag"].apply(mask_radius_from_tmag)
    bright.to_csv(args.outdir / f"{prefix}_catalog_mask_sources.csv", index=False)

    ephem = select_detector_ephem(args.ephem_csv, args.target, args.sector, args.camera, args.ccd)
    raw_cube, raw_time = build_raw_flux_cube(ephem, shape=(int(args.cutout_size), int(args.cutout_size)))
    match_idx = match_bundle_times_to_raw(raw_time, bundle_time)
    raw_matched = raw_cube[match_idx]

    central_mask = center_box_mask((int(args.cutout_size), int(args.cutout_size)), box_size=int(args.central_box_size))
    processed_cube, frame_table, processing_summary = process_cube(
        raw_matched,
        corner1=corner1,
        corner2=corner2,
        bright_sources=bright,
        central_mask=central_mask,
    )
    frame_table.to_csv(args.outdir / f"{prefix}_catalog_mask_frame_backgrounds.csv", index=False)

    stack_result = stack_with_support(
        processed_cube,
        x=local_x,
        y=local_y,
        method="mean",
        interpolation=str(args.interpolation),
        reference=(10.0, 10.0),
        integer_shifts=False,
    )
    stack = np.asarray(stack_result["stack"], dtype=float)
    support_map = np.asarray(stack_result["support_map"], dtype=float)
    gate_kwargs = {
        "min_support_fraction": float(args.support_fraction_min) if float(args.support_fraction_min) > 0 else None,
        "min_support_frames": float(args.support_frames_min) if float(args.support_frames_min) > 0 else None,
    }
    gated_threshold = None
    if any(value is not None for value in gate_kwargs.values()):
        gated = apply_support_gate(stack, support_map, **gate_kwargs)
        stack = np.asarray(gated["stack"], dtype=float)
        gated_threshold = float(gated["threshold_frames"])
    suffix_parts: list[str] = []
    if float(args.support_fraction_min) > 0:
        suffix_parts.append(f"supportfrac{str(args.support_fraction_min).replace('.', 'p')}")
    if float(args.support_frames_min) > 0:
        suffix_parts.append(f"supportn{str(args.support_frames_min).replace('.', 'p')}")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    heatmap_path = args.outdir / f"{prefix}_catalog_mask_nominal_stack{suffix}_heatmap.png"
    plot_stack_heatmap(stack, support_map, heatmap_path, target=str(args.target), support_threshold_frames=gated_threshold)

    center_score = float(score_stack(stack, method="aperture_snr", position=(10.0, 10.0)))
    peak_y, peak_x = np.unravel_index(np.nanargmax(stack), stack.shape)
    summary = {
        "target": str(args.target),
        "sector": int(args.sector),
        "camera": int(args.camera),
        "ccd": int(args.ccd),
        "bundle_npz": str(args.bundle_npz),
        "bright_csv": str(args.bright_csv),
        "n_bright_sources_masked": int(len(bright)),
        "processing_summary": processing_summary,
        "stack_center_aperture_snr": center_score,
        "stack_peak_x": int(peak_x),
        "stack_peak_y": int(peak_y),
        "stack_peak_value": float(np.nanmax(stack)),
        "support_threshold_frames": gated_threshold,
        "heatmap_png": str(heatmap_path),
        "catalog_mask_sources_csv": str(args.outdir / f"{prefix}_catalog_mask_sources.csv"),
        "frame_backgrounds_csv": str(args.outdir / f"{prefix}_catalog_mask_frame_backgrounds.csv"),
        "bundle_metadata_subset": {
            "loader_bg_method": metadata.get("loader_bg_method"),
            "n_frames_used": metadata.get("n_frames_used"),
            "cutout_shape": metadata.get("cutout_shape"),
        },
    }
    summary_path = args.outdir / f"{prefix}_catalog_mask_stack_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Masked bright sources: {len(bright)}")
    print(
        "Mask radii span "
        f"{bright['mask_radius_pix'].min():.2f} -> {bright['mask_radius_pix'].max():.2f} px "
        f"for Tmag <= {args.mag_limit:.1f}"
    )
    print(f"Mean masked pixels per frame: {processing_summary['mean_masked_pixels_per_frame']:.2f}")
    print(f"Mean subtracted background level: {processing_summary['mean_background_mean']:.2f}")
    print(f"Center aperture S/N on final stack: {center_score:.3f}")
    print(f"Final stack peak: ({peak_x}, {peak_y}) value={float(np.nanmax(stack)):.3f}")
    print(f"Heatmap: {heatmap_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
