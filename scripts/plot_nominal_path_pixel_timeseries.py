#!/usr/bin/env python3
"""Plot raw detector-pixel time series along a target's nominal path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_pluto_catalog_mask_stack import (
    build_raw_flux_cube,
    load_bundle,
    match_bundle_times_to_raw,
    select_detector_ephem,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot raw pixel time series for detector pixels along a nominal moving-target path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Sedna")
    parser.add_argument("--sector", type=int, default=5)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--ccd", type=int, default=4)
    parser.add_argument(
        "--ephem-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sedna_full_inventory" / "sedna_all_detector_inventory.csv",
    )
    parser.add_argument(
        "--bundle-npz",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "sedna_multisector"
        / "sector5_full"
        / "sedna_sector5_cam1_ccd4_cube_bundle.npz",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sedna_multisector" / "sector5_full_nominal_path_pixel_timeseries_masked.png",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sedna_multisector" / "sector5_full_nominal_path_pixel_timeseries_masked.csv",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sedna_multisector" / "sector5_full_ramp_mask_summary.txt",
    )
    parser.add_argument("--cutout-size", type=int, default=21)
    parser.add_argument("--mask-window", type=int, default=49)
    parser.add_argument("--mask-sigma", type=float, default=5.0)
    parser.add_argument(
        "--extra-mask-times",
        type=float,
        nargs="*",
        default=(),
        help="Additional exact BTJD cadences to exclude after the automatic ramp mask.",
    )
    parser.add_argument("--label-every", type=int, default=3)
    parser.add_argument("--ytick-fontsize", type=int, default=14)
    parser.add_argument("--xtick-fontsize", type=int, default=14)
    parser.add_argument("--axis-label-fontsize", type=int, default=15)
    parser.add_argument("--title-fontsize", type=int, default=15)
    parser.add_argument("--line-alpha", type=float, default=0.78)
    parser.add_argument("--marker-alpha", type=float, default=0.32)
    parser.add_argument("--marker-size", type=float, default=12.0)
    parser.add_argument(
        "--detrend-linear",
        action="store_true",
        help="Subtract a best-fit linear trend from each detector-pixel time series before plotting.",
    )
    parser.add_argument(
        "--offset-scale",
        type=float,
        default=(1.0 / 3.0) * 0.5,
        help="Multiplier applied to the baseline vertical offset step.",
    )
    return parser


def unique_path_pixels(det_x: np.ndarray, det_y: np.ndarray) -> list[tuple[int, int]]:
    pixels: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x_val, y_val in zip(np.rint(det_x).astype(int), np.rint(det_y).astype(int), strict=False):
        pixel = (int(x_val), int(y_val))
        if pixel not in seen:
            seen.add(pixel)
            pixels.append(pixel)
    return pixels


def compute_ramp_mask(raw_cube: np.ndarray, window: int, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    med = np.nanmedian(raw_cube, axis=(1, 2))
    pad = window // 2
    padded = np.pad(med, pad, mode="edge")
    running = np.array([np.median(padded[i : i + window]) for i in range(len(med))], dtype=float)
    resid = med - running
    mad = float(np.median(np.abs(resid - np.median(resid))))
    threshold = float(np.median(resid) + sigma * 1.4826 * mad)
    keep = resid < threshold
    return keep, med, running, mad, threshold


def build_series_rows(
    bundle_time: np.ndarray,
    det_x: np.ndarray,
    det_y: np.ndarray,
    corner1: np.ndarray,
    corner2: np.ndarray,
    raw_matched: np.ndarray,
    keep: np.ndarray,
    path_pixels: list[tuple[int, int]],
    detrend_linear: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order_index, (px, py) in enumerate(path_pixels):
        vals = np.full(len(bundle_time), np.nan, dtype=float)
        centered = (np.rint(det_x).astype(int) == px) & (np.rint(det_y).astype(int) == py) & keep
        for frame_index in range(len(bundle_time)):
            lx = px - int(corner1[frame_index])
            ly = py - int(corner2[frame_index])
            if 0 <= lx < raw_matched.shape[2] and 0 <= ly < raw_matched.shape[1] and keep[frame_index]:
                vals[frame_index] = float(raw_matched[frame_index, ly, lx])
        finite = np.isfinite(vals)
        plotted = np.full(len(bundle_time), np.nan, dtype=float)
        trend_slope = 0.0
        trend_intercept = 0.0
        if np.count_nonzero(finite) >= 2 and detrend_linear:
            trend_slope, trend_intercept = np.polyfit(bundle_time[finite], vals[finite], 1)
            plotted[finite] = vals[finite] - (trend_slope * bundle_time[finite] + trend_intercept)
        else:
            baseline = float(np.nanmedian(vals[finite])) if np.any(finite) else 0.0
            plotted[finite] = vals[finite] - baseline
        rows.append(
            {
                "order_index": int(order_index),
                "pixel": (int(px), int(py)),
                "raw_flux": vals,
                "plot_flux": plotted,
                "centered": centered,
                "trend_slope_per_day": float(trend_slope),
                "trend_intercept": float(trend_intercept),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], bundle_time: np.ndarray, keep: np.ndarray, out_csv: Path) -> None:
    records: list[dict[str, object]] = []
    for row in rows:
        px, py = row["pixel"]  # type: ignore[misc]
        raw_flux = row["raw_flux"]  # type: ignore[assignment]
        plot_flux = row["plot_flux"]  # type: ignore[assignment]
        centered = row["centered"]  # type: ignore[assignment]
        for frame_index, time_value in enumerate(bundle_time):
            if not np.isfinite(raw_flux[frame_index]):
                continue
            records.append(
                {
                    "order_index": int(row["order_index"]),
                    "detector_x": int(px),
                    "detector_y": int(py),
                    "frame_index": int(frame_index),
                    "time_tdb_jd_minus_2457000": float(time_value),
                    "raw_flux": float(raw_flux[frame_index]),
                    "plot_flux": float(plot_flux[frame_index]),
                    "is_centered_pixel": bool(centered[frame_index]),
                    "cadence_kept": bool(keep[frame_index]),
                    "trend_slope_per_day": float(row["trend_slope_per_day"]),
                    "trend_intercept": float(row["trend_intercept"]),
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)


def plot_rows(
    rows: list[dict[str, object]],
    bundle_time: np.ndarray,
    keep: np.ndarray,
    out_png: Path,
    target: str,
    sector: int,
    label_every: int,
    ytick_fontsize: int,
    xtick_fontsize: int,
    axis_label_fontsize: int,
    title_fontsize: int,
    line_alpha: float,
    marker_alpha: float,
    marker_size: float,
    offset_scale: float,
    detrend_linear: bool,
    extra_mask_times: tuple[float, ...],
) -> float:
    all_resid = np.concatenate(
        [row["plot_flux"][np.isfinite(row["plot_flux"])] for row in rows if np.any(np.isfinite(row["plot_flux"]))]  # type: ignore[index]
    )
    amp = float(np.nanpercentile(np.abs(all_resid), 98)) if all_resid.size else 1.0
    base_offset_step = max(8.0, 2.4 * amp)
    offset_step = float(base_offset_step * offset_scale)

    fig_h = max(10.0, 0.16 * len(rows) + 2.5)
    fig, ax = plt.subplots(figsize=(14, fig_h), constrained_layout=True)

    yticks: list[float] = []
    yticklabels: list[str] = []
    for idx, row in enumerate(rows):
        offset = idx * offset_step
        plot_flux = row["plot_flux"]  # type: ignore[assignment]
        centered = row["centered"]  # type: ignore[assignment]
        plot_y = plot_flux + offset
        ax.plot(bundle_time, plot_y, color=(0.1, 0.1, 0.1, line_alpha), linewidth=0.8)
        marker_times = bundle_time[centered]
        marker_vals = plot_y[centered]
        if marker_times.size:
            ax.scatter(
                marker_times,
                marker_vals,
                s=marker_size,
                color="#d94801",
                alpha=marker_alpha,
                edgecolors="none",
                zorder=3,
            )
        if idx % label_every == 0:
            px, py = row["pixel"]  # type: ignore[misc]
            yticks.append(offset)
            yticklabels.append(f"{px},{py}")

    bad_idx = np.flatnonzero(~keep)
    if bad_idx.size:
        groups = np.split(bad_idx, np.where(np.diff(bad_idx) != 1)[0] + 1)
        for grp in groups:
            ax.axvspan(bundle_time[grp[0]], bundle_time[grp[-1]], color="#fdd0a2", alpha=0.22, lw=0)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=ytick_fontsize)
    ax.tick_params(axis="x", labelsize=xtick_fontsize)
    ax.set_xlabel("Time [BTJD]", fontsize=axis_label_fontsize)
    ax.set_ylabel("Detector pixel (x,y), ordered by nominal path visit", fontsize=axis_label_fontsize)
    subtitle = (
        "Each trace has its best-fit linear trend removed; orange markers show cadences where "
        f"{target} is centered in that pixel"
        if detrend_linear
        else f"Each trace is median-subtracted and vertically offset; orange markers show cadences where {target} is centered in that pixel"
    )
    extra_text = ""
    if extra_mask_times:
        formatted = ", ".join(f"{value:.6f}" for value in extra_mask_times)
        extra_text = f"\nManual extra cadence cut(s): {formatted} BTJD"
    ax.set_title(
        f"{target} Sector {sector} raw detector-pixel time series along the nominal path\n"
        f"{subtitle}\n"
        f"Conservative ramp mask removed {int(np.count_nonzero(~keep))} of {len(keep)} cadences{extra_text}",
        fontsize=title_fontsize,
    )
    ax.grid(axis="x", color=(0, 0, 0, 0.08), linewidth=0.6)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return offset_step


def write_summary(
    out_summary: Path,
    path_pixels: list[tuple[int, int]],
    keep: np.ndarray,
    med: np.ndarray,
    mad: float,
    threshold: float,
    rows: list[dict[str, object]],
    offset_step: float,
    extra_mask_times: tuple[float, ...],
    detrend_linear: bool,
) -> None:
    no_centered = [row["pixel"] for row in rows if not np.any(row["centered"])]  # type: ignore[index]
    lines = [
        f"distinct_path_pixels={len(path_pixels)}",
        f"cadences_total={len(keep)}",
        f"cadences_removed={int(np.count_nonzero(~keep))}",
        f"cutout_median_flux_median={float(np.median(med))}",
        f"ramp_mask_mad={mad}",
        f"ramp_mask_threshold={threshold}",
        f"offset_step={offset_step}",
        f"detrend_linear={detrend_linear}",
        f"pixels_with_no_centered_kept_cadences={len(no_centered)}",
    ]
    if extra_mask_times:
        lines.append("extra_mask_times=" + ",".join(f"{value:.12f}" for value in extra_mask_times))
    if no_centered:
        lines.append("no_centered_pixels=" + ",".join(f"{px}:{py}" for px, py in no_centered))
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()

    bundle = load_bundle(args.bundle_npz)
    bundle_time = np.asarray(bundle["time_tdb_jd_minus_2457000"], dtype=float)
    det_x = np.asarray(bundle["detector_x"], dtype=float)
    det_y = np.asarray(bundle["detector_y"], dtype=float)
    corner1 = np.asarray(bundle["corner1"], dtype=int)
    corner2 = np.asarray(bundle["corner2"], dtype=int)

    path_pixels = unique_path_pixels(det_x, det_y)
    ephem = select_detector_ephem(args.ephem_csv, args.target, args.sector, args.camera, args.ccd)
    raw_cube, raw_time = build_raw_flux_cube(ephem, shape=(args.cutout_size, args.cutout_size))
    match_idx = match_bundle_times_to_raw(raw_time, bundle_time)
    raw_matched = raw_cube[match_idx]

    keep, med, _running, mad, threshold = compute_ramp_mask(raw_matched, args.mask_window, args.mask_sigma)
    extra_mask_times = tuple(float(value) for value in args.extra_mask_times)
    if extra_mask_times:
        extra_mask = np.zeros(len(bundle_time), dtype=bool)
        for value in extra_mask_times:
            nearest = int(np.nanargmin(np.abs(bundle_time - value)))
            if abs(float(bundle_time[nearest]) - value) > 1.0 / 86400.0:
                raise ValueError(f"No cadence matched requested extra mask time {value:.12f} within 1 second.")
            extra_mask[nearest] = True
        keep = keep & (~extra_mask)
    rows = build_series_rows(
        bundle_time,
        det_x,
        det_y,
        corner1,
        corner2,
        raw_matched,
        keep,
        path_pixels,
        detrend_linear=bool(args.detrend_linear),
    )
    write_csv(rows, bundle_time, keep, args.out_csv)
    offset_step = plot_rows(
        rows,
        bundle_time,
        keep,
        args.out_png,
        target=str(args.target),
        sector=int(args.sector),
        label_every=args.label_every,
        ytick_fontsize=args.ytick_fontsize,
        xtick_fontsize=args.xtick_fontsize,
        axis_label_fontsize=args.axis_label_fontsize,
        title_fontsize=args.title_fontsize,
        line_alpha=args.line_alpha,
        marker_alpha=args.marker_alpha,
        marker_size=args.marker_size,
        offset_scale=args.offset_scale,
        detrend_linear=bool(args.detrend_linear),
        extra_mask_times=extra_mask_times,
    )
    write_summary(
        args.out_summary,
        path_pixels,
        keep,
        med,
        mad,
        threshold,
        rows,
        offset_step,
        extra_mask_times=extra_mask_times,
        detrend_linear=bool(args.detrend_linear),
    )

    no_centered = [row["pixel"] for row in rows if not np.any(row["centered"])]  # type: ignore[index]
    print(args.out_png)
    print(
        {
            "distinct_path_pixels": len(path_pixels),
            "cadences_removed": int(np.count_nonzero(~keep)),
            "extra_mask_times": extra_mask_times,
            "detrend_linear": bool(args.detrend_linear),
            "pixels_with_no_centered_kept_cadences": len(no_centered),
            "no_centered_pixels": no_centered,
            "offset_step": offset_step,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
