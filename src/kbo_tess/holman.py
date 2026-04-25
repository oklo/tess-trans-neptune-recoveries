"""Holman-style image-domain stacking for known moving targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from kbo_tess.shiftstack import stack_with_support


def build_target_excluded_masks(
    shape: tuple[int, int],
    local_x: np.ndarray,
    local_y: np.ndarray,
    target_box_size: int,
) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.indices((ny, nx), dtype=float)
    half = max(0.5, (float(target_box_size) - 1.0) / 2.0)
    masks = np.zeros((len(local_x), ny, nx), dtype=bool)
    for i in range(len(local_x)):
        masks[i] = (np.abs(xx - float(local_x[i])) <= half) & (np.abs(yy - float(local_y[i])) <= half)
    return masks


def build_differential_cube(
    raw_cube: np.ndarray,
    local_x: np.ndarray,
    local_y: np.ndarray,
    keep: np.ndarray,
    *,
    target_box_size: int = 5,
) -> dict[str, Any]:
    target_masks = build_target_excluded_masks(raw_cube.shape[1:], local_x=local_x, local_y=local_y, target_box_size=int(target_box_size))
    outer_median = np.full(raw_cube.shape[0], np.nan, dtype=float)
    scalar_subtracted = np.asarray(raw_cube, dtype=float).copy()
    for i in range(raw_cube.shape[0]):
        outer_values = raw_cube[i][~target_masks[i]]
        outer_median[i] = float(np.nanmedian(outer_values))
        scalar_subtracted[i] = raw_cube[i] - outer_median[i]
    reference_image = np.nanmedian(scalar_subtracted[keep], axis=0)
    differential_cube = scalar_subtracted - reference_image[None, :, :]
    return {
        "target_masks": target_masks,
        "outer_median": outer_median,
        "scalar_subtracted_cube": scalar_subtracted,
        "reference_image": reference_image,
        "differential_cube": differential_cube,
    }


def holman_stack(
    differential_cube: np.ndarray,
    local_x: np.ndarray,
    local_y: np.ndarray,
    keep: np.ndarray,
    *,
    interpolation: str = "linear",
    reference: tuple[float, float] | None = None,
) -> dict[str, Any]:
    ny, nx = differential_cube.shape[1:]
    if reference is None:
        reference = ((nx - 1.0) / 2.0, (ny - 1.0) / 2.0)
    return stack_with_support(
        differential_cube[keep],
        np.asarray(local_x, dtype=float)[keep],
        np.asarray(local_y, dtype=float)[keep],
        method="mean",
        interpolation=str(interpolation),
        reference=reference,
        integer_shifts=False,
    )


def standard_error_from_aligned(aligned_frames: np.ndarray) -> np.ndarray:
    support = np.sum(np.isfinite(aligned_frames), axis=0).astype(float)
    stderr = np.full(aligned_frames.shape[1:], np.nan, dtype=float)
    valid = support > 1
    if np.any(valid):
        std = np.nanstd(aligned_frames, axis=0, ddof=1)
        stderr[valid] = std[valid] / np.sqrt(support[valid])
    return stderr


def snr_map_from_stack(stack: np.ndarray, stderr: np.ndarray) -> np.ndarray:
    snr = np.full_like(stack, np.nan, dtype=float)
    valid = np.isfinite(stack) & np.isfinite(stderr) & (stderr > 0)
    snr[valid] = stack[valid] / stderr[valid]
    return snr


def circular_aperture_mask(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.indices((ny, nx), dtype=float)
    cx, cy = float(center[0]), float(center[1])
    return np.square(xx - cx) + np.square(yy - cy) <= float(radius) * float(radius)


def aperture_stats(
    stack: np.ndarray,
    stderr: np.ndarray,
    center: tuple[float, float],
    radius: float,
) -> dict[str, float]:
    aperture = circular_aperture_mask(stack.shape, center=center, radius=float(radius))
    signal = float(np.nansum(stack[aperture]))
    noise = float(np.sqrt(np.nansum(np.square(stderr[aperture]))))
    return {
        "aperture_radius_pix": float(radius),
        "aperture_signal": signal,
        "aperture_noise": noise,
        "aperture_snr": float(signal / noise) if np.isfinite(noise) and noise > 0 else float("nan"),
    }


def robust_limits(image: np.ndarray, nsigma: float = 4.0) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return -1.0, 1.0
    med = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - med)))
    sigma = max(1.4826 * mad, 1e-6)
    return med - float(nsigma) * sigma, med + float(nsigma) * sigma


def write_holman_stack_figure(
    reference_image: np.ndarray,
    differential_stack: np.ndarray,
    snr_map: np.ndarray,
    support_map: np.ndarray,
    *,
    out_png: Path,
    target: str,
    sector: int,
    camera: int,
    ccd: int,
    keep_count: int,
    total_count: int,
    center: tuple[float, float],
    aperture_radius: float,
    interpolation: str,
    background_mode_label: str,
    aperture_summary: dict[str, float],
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.8), constrained_layout=True)

    ref_vmin, ref_vmax = robust_limits(reference_image, nsigma=4.0)
    stack_vmin, stack_vmax = robust_limits(differential_stack, nsigma=4.0)
    snr_vmax = max(3.0, float(np.nanpercentile(np.abs(snr_map[np.isfinite(snr_map)]), 99.0))) if np.any(np.isfinite(snr_map)) else 5.0
    support_vmin = 0.0
    support_vmax = max(float(np.nanmax(support_map)), 1.0)

    panels = [
        (reference_image, "Reference image", "cividis", ref_vmin, ref_vmax),
        (differential_stack, "Mean differential stack", "RdBu_r", stack_vmin, stack_vmax),
        (snr_map, "Stack S/N map", "RdBu_r", -snr_vmax, snr_vmax),
        (support_map, "Support map", "viridis", support_vmin, support_vmax),
    ]
    for ax, (image, title, cmap, vmin, vmax) in zip(axes, panels, strict=False):
        im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter([center[0]], [center[1]], marker="+", s=55, c="white", linewidths=1.0)
        ax.add_patch(Circle(center, radius=float(aperture_radius), edgecolor="white", facecolor="none", linewidth=0.9, alpha=0.9))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        f"{target} Sector {sector} Cam {camera} CCD {ccd} | Holman-style differential stack\n"
        f"keep {keep_count}/{total_count} cadences, interpolation={interpolation}, background={background_mode_label}, "
        f"center-aperture S/N={aperture_summary['aperture_snr']:.2f}",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_summary_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_stack_npz(
    path: Path,
    *,
    reference_image: np.ndarray,
    differential_stack: np.ndarray,
    stderr_map: np.ndarray,
    snr_map: np.ndarray,
    support_map: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        reference_image=np.asarray(reference_image, dtype=float),
        differential_stack=np.asarray(differential_stack, dtype=float),
        stderr_map=np.asarray(stderr_map, dtype=float),
        snr_map=np.asarray(snr_map, dtype=float),
        support_map=np.asarray(support_map, dtype=float),
    )
