#!/usr/bin/env python3
"""Plot aligned local windows around nominal moving-target pixel crossings."""

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
from matplotlib.gridspec import GridSpec

from plot_nominal_path_pixel_timeseries import build_series_rows, compute_ramp_mask, unique_path_pixels
from run_pluto_catalog_mask_stack import build_raw_flux_cube, load_bundle, match_bundle_times_to_raw, select_detector_ephem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot aligned local windows around nominal pixel crossings, with a summed bottom panel.",
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
        default=PROJECT_ROOT / "outputs" / "sedna_multisector" / "sector5_full" / "sedna_sector5_cam1_ccd4_cube_bundle.npz",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sedna_multisector" / "sector5_full_nominal_path_local_windows_detrended.png",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sedna_multisector" / "sector5_full_nominal_path_local_windows_detrended.csv",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sedna_multisector" / "sector5_full_nominal_path_local_windows_detrended_summary.txt",
    )
    parser.add_argument("--cutout-size", type=int, default=21)
    parser.add_argument("--mask-window", type=int, default=49)
    parser.add_argument("--mask-sigma", type=float, default=5.0)
    parser.add_argument(
        "--extra-mask-times",
        type=float,
        nargs="*",
        default=(1449.9956542677933, 1462.0995072339913),
        help="Additional exact BTJD cadences to exclude after the automatic ramp mask.",
    )
    parser.add_argument(
        "--extra-mask-ranges",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("START", "STOP"),
        help="Additional inclusive BTJD time ranges to exclude after the automatic ramp mask.",
    )
    parser.add_argument("--window-multiplier", type=float, default=4.5)
    parser.add_argument("--sum-bin-hours", type=float, default=0.5)
    parser.add_argument("--label-every", type=int, default=3)
    parser.add_argument("--ytick-fontsize", type=int, default=14)
    parser.add_argument("--xtick-fontsize", type=int, default=14)
    parser.add_argument("--axis-label-fontsize", type=int, default=15)
    parser.add_argument("--title-fontsize", type=int, default=15)
    parser.add_argument("--line-alpha", type=float, default=0.78)
    parser.add_argument("--marker-alpha", type=float, default=0.40)
    parser.add_argument("--marker-size", type=float, default=24.0)
    parser.add_argument("--offset-scale", type=float, default=(1.0 / 3.0) * 0.5)
    parser.add_argument("--bootstrap-draws", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-errorbar-stride", type=int, default=4)
    parser.add_argument(
        "--detrend-degree",
        type=int,
        default=None,
        help="Polynomial detrend degree. Overrides --detrend-linear / --no-detrend-linear.",
    )
    parser.add_argument("--detrend-linear", dest="detrend_linear", action="store_true")
    parser.add_argument("--no-detrend-linear", dest="detrend_linear", action="store_false")
    parser.set_defaults(detrend_linear=True)
    return parser


def apply_extra_mask(
    bundle_time: np.ndarray,
    keep: np.ndarray,
    extra_mask_times: tuple[float, ...],
    extra_mask_ranges: tuple[tuple[float, float], ...],
) -> np.ndarray:
    keep = keep.copy()
    for value in extra_mask_times:
        nearest = int(np.nanargmin(np.abs(bundle_time - value)))
        if abs(float(bundle_time[nearest]) - value) > 1.0 / 86400.0:
            raise ValueError(f"No cadence matched requested extra mask time {value:.12f} within 1 second.")
        keep[nearest] = False
    for start, stop in extra_mask_ranges:
        lo = min(start, stop)
        hi = max(start, stop)
        keep[(bundle_time >= lo) & (bundle_time <= hi)] = False
    return keep


def crossing_time(bundle_time: np.ndarray, det_x: np.ndarray, det_y: np.ndarray, row: dict[str, object]) -> float:
    px, py = row["pixel"]  # type: ignore[misc]
    centered_keep = row["centered"]  # type: ignore[assignment]
    centered_all = (np.rint(det_x).astype(int) == int(px)) & (np.rint(det_y).astype(int) == int(py))
    if np.any(centered_keep):
        return float(np.median(bundle_time[centered_keep]))
    if np.any(centered_all):
        return float(np.median(bundle_time[centered_all]))
    finite = np.isfinite(row["plot_flux"])  # type: ignore[index]
    return float(np.median(bundle_time[finite])) if np.any(finite) else float(np.median(bundle_time))


def local_window_dataframe(
    rows: list[dict[str, object]],
    bundle_time: np.ndarray,
    det_x: np.ndarray,
    det_y: np.ndarray,
    window_width_hours: float,
) -> tuple[pd.DataFrame, float]:
    half_window_days = (window_width_hours / 24.0) / 2.0
    records: list[dict[str, object]] = []
    for row in rows:
        px, py = row["pixel"]  # type: ignore[misc]
        cross_time = crossing_time(bundle_time, det_x, det_y, row)
        rel_hours = (bundle_time - cross_time) * 24.0
        in_window = np.abs(bundle_time - cross_time) <= half_window_days
        plot_flux = row["plot_flux"]  # type: ignore[assignment]
        centered = row["centered"]  # type: ignore[assignment]
        raw_flux = row["raw_flux"]  # type: ignore[assignment]
        for i, ok in enumerate(in_window):
            if not ok or not np.isfinite(plot_flux[i]):
                continue
            records.append(
                {
                    "order_index": int(row["order_index"]),
                    "detector_x": int(px),
                    "detector_y": int(py),
                    "time_tdb_jd_minus_2457000": float(bundle_time[i]),
                    "relative_time_hours": float(rel_hours[i]),
                    "raw_flux": float(raw_flux[i]),
                    "plot_flux": float(plot_flux[i]),
                    "is_centered_pixel": bool(centered[i]),
                    "cross_time_tdb_jd_minus_2457000": float(cross_time),
                    "window_width_hours": float(window_width_hours),
                }
            )
    return pd.DataFrame.from_records(records), half_window_days * 24.0


def offset_step_from_rows(rows: list[dict[str, object]], offset_scale: float) -> float:
    all_vals = np.concatenate(
        [row["plot_flux"][np.isfinite(row["plot_flux"])] for row in rows if np.any(np.isfinite(row["plot_flux"]))]  # type: ignore[index]
    )
    amp = float(np.nanpercentile(np.abs(all_vals), 98)) if all_vals.size else 1.0
    return float(max(8.0, 2.4 * amp) * offset_scale)


def summed_curve(df: pd.DataFrame, half_window_hours: float, bin_width_hours: float) -> tuple[np.ndarray, np.ndarray]:
    x_grid = np.arange(-half_window_hours, half_window_hours + 0.5 * bin_width_hours, bin_width_hours)
    sum_flux = np.full_like(x_grid, np.nan, dtype=float)
    rel = df["relative_time_hours"].to_numpy(dtype=float)
    vals = df["plot_flux"].to_numpy(dtype=float)
    for i, x0 in enumerate(x_grid):
        lo = x0 - 0.5 * bin_width_hours
        hi = x0 + 0.5 * bin_width_hours
        mask = (rel >= lo) & (rel < hi) & np.isfinite(vals)
        if np.any(mask):
            sum_flux[i] = float(np.sum(vals[mask]))
    return x_grid, sum_flux


def row_sum_vectors(
    rows: list[dict[str, object]],
    x_grid: np.ndarray,
    bin_width_hours: float,
) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(x_grid)), dtype=float)
    left_edge = float(x_grid[0] - 0.5 * bin_width_hours)
    n_bins = len(x_grid)
    vectors = np.zeros((len(rows), n_bins), dtype=float)
    for i, row in enumerate(rows):
        x = np.asarray(row["x"], dtype=float)
        y = np.asarray(row["y"], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        x = x[finite]
        y = y[finite]
        indices = np.floor((x - left_edge) / float(bin_width_hours)).astype(int)
        valid = (indices >= 0) & (indices < n_bins)
        if not np.any(valid):
            continue
        np.add.at(vectors[i], indices[valid], y[valid])
    return vectors


def bootstrap_sum_curve(
    row_vectors: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if draws <= 0 or row_vectors.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    n_rows = row_vectors.shape[0]
    boot = np.empty((draws, row_vectors.shape[1]), dtype=float)
    for i in range(draws):
        sample_idx = rng.integers(0, n_rows, size=n_rows)
        boot[i] = np.sum(row_vectors[sample_idx], axis=0)
    low = np.nanpercentile(boot, 16.0, axis=0)
    high = np.nanpercentile(boot, 84.0, axis=0)
    return low, high


def plot_local_windows(
    df: pd.DataFrame,
    out_png: Path,
    target: str,
    sector: int,
    window_multiplier: float,
    window_width_hours: float,
    total_removed: int,
    n_total: int,
    ytick_fontsize: int,
    xtick_fontsize: int,
    axis_label_fontsize: int,
    title_fontsize: int,
    label_every: int,
    line_alpha: float,
    marker_alpha: float,
    marker_size: float,
    offset_step: float,
    sum_bin_hours: float,
    detrend_degree: int,
    extra_mask_times: tuple[float, ...],
    extra_mask_ranges: tuple[tuple[float, float], ...],
    bootstrap_draws: int,
    bootstrap_seed: int,
    bootstrap_errorbar_stride: int,
) -> None:
    rows = []
    for order_index, group in df.groupby("order_index", sort=True):
        group = group.sort_values("relative_time_hours").reset_index(drop=True)
        rows.append(
            {
                "order_index": int(order_index),
                "pixel": (int(group["detector_x"].iloc[0]), int(group["detector_y"].iloc[0])),
                "x": group["relative_time_hours"].to_numpy(dtype=float),
                "y": group["plot_flux"].to_numpy(dtype=float),
                "centered": group["is_centered_pixel"].to_numpy(dtype=bool),
            }
        )

    half_window_hours = window_width_hours / 2.0
    x_grid, sum_flux = summed_curve(df, half_window_hours=half_window_hours, bin_width_hours=sum_bin_hours)
    row_vectors = row_sum_vectors(rows, x_grid=x_grid, bin_width_hours=sum_bin_hours)
    boot_low, boot_high = bootstrap_sum_curve(row_vectors, draws=bootstrap_draws, seed=bootstrap_seed)

    fig_h = max(12.0, 0.16 * len(rows) + 4.0)
    fig = plt.figure(figsize=(13, fig_h), constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[5, 1])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)

    yticks: list[float] = []
    yticklabels: list[str] = []
    for idx, row in enumerate(rows):
        offset = idx * offset_step
        y = row["y"] + offset
        ax_top.plot(row["x"], y, color=(0.1, 0.1, 0.1, line_alpha), linewidth=0.8)
        if np.any(row["centered"]):
            ax_top.scatter(
                row["x"][row["centered"]],
                y[row["centered"]],
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

    ax_top.axvline(0.0, color="tab:blue", alpha=0.25, linewidth=1.0)
    ax_top.set_yticks(yticks)
    ax_top.set_yticklabels(yticklabels, fontsize=ytick_fontsize)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("Detector pixel (x,y), ordered by nominal path visit", fontsize=axis_label_fontsize)
    ax_top.grid(axis="x", color=(0, 0, 0, 0.08), linewidth=0.6)
    if detrend_degree <= 0:
        subtitle = f"Each trace is median-subtracted; orange markers show cadences where {target} is centered in that pixel"
    elif detrend_degree == 1:
        subtitle = f"Each trace has its best-fit linear trend removed; orange markers show cadences where {target} is centered in that pixel"
    elif detrend_degree == 2:
        subtitle = f"Each trace has its best-fit quadratic trend removed; orange markers show cadences where {target} is centered in that pixel"
    else:
        subtitle = (
            f"Each trace has its best-fit polynomial trend (degree {detrend_degree}) removed; "
            f"orange markers show cadences where {target} is centered in that pixel"
        )
    extra_text = ""
    if extra_mask_times:
        extra_text += "\nManual extra cadence cut(s): " + ", ".join(f"{value:.6f}" for value in extra_mask_times)
    if extra_mask_ranges:
        extra_text += "\nManual extra time cut range(s): " + ", ".join(
            f"[{start:.6f}, {stop:.6f}]" for start, stop in extra_mask_ranges
        )
    ax_top.set_title(
        f"{target} Sector {sector} local windows around the nominal pixel crossings\n"
        f"{subtitle}\n"
        f"Window width = {window_multiplier:.1f} x average pixel residence time = {window_width_hours:.2f} h; total masked cadences = {total_removed} of {n_total}{extra_text}",
        fontsize=title_fontsize,
    )

    valid = np.isfinite(sum_flux)
    ax_bottom.axvline(0.0, color="tab:blue", alpha=0.25, linewidth=1.0)
    ax_bottom.axhline(0.0, color="0.35", alpha=0.65, linewidth=1.0)
    if bootstrap_draws > 0 and boot_low.size == sum_flux.size:
        ax_bottom.fill_between(
            x_grid[valid],
            boot_low[valid],
            boot_high[valid],
            color="#74a9cf",
            alpha=0.20,
            linewidth=0.0,
            zorder=1,
        )
        err_idx = np.flatnonzero(valid)[:: max(1, int(bootstrap_errorbar_stride))]
        lower_err = np.maximum(0.0, sum_flux[err_idx] - boot_low[err_idx])
        upper_err = np.maximum(0.0, boot_high[err_idx] - sum_flux[err_idx])
        ax_bottom.errorbar(
            x_grid[err_idx],
            sum_flux[err_idx],
            yerr=np.vstack([lower_err, upper_err]),
            fmt="none",
            ecolor="#3182bd",
            elinewidth=0.8,
            alpha=0.45,
            capsize=0,
            zorder=2,
        )
    ax_bottom.plot(x_grid[valid], sum_flux[valid], color="black", linewidth=1.5)
    ax_bottom.set_xlabel(f"Time relative to {target} crossing [hours]", fontsize=axis_label_fontsize)
    ax_bottom.set_ylabel("Sum", fontsize=axis_label_fontsize)
    ax_bottom.tick_params(axis="x", labelsize=xtick_fontsize)
    ax_bottom.tick_params(axis="y", labelsize=ytick_fontsize)
    ax_bottom.grid(axis="x", color=(0, 0, 0, 0.08), linewidth=0.6)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


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
    extra_mask_ranges = tuple((float(start), float(stop)) for start, stop in args.extra_mask_ranges)
    keep = apply_extra_mask(bundle_time, keep, extra_mask_times, extra_mask_ranges)
    detrend_degree = int(args.detrend_degree) if args.detrend_degree is not None else (1 if bool(args.detrend_linear) else 0)

    rows = build_series_rows(
        bundle_time,
        det_x,
        det_y,
        corner1,
        corner2,
        raw_matched,
        keep,
        path_pixels,
        detrend_degree=detrend_degree,
    )

    residence_counts = [int(np.count_nonzero(row["centered"])) for row in rows if np.count_nonzero(row["centered"])]  # type: ignore[arg-type]
    avg_residence_count = float(np.mean(residence_counts)) if residence_counts else 1.0
    median_cadence_hours = float(np.median(np.diff(bundle_time)) * 24.0)
    window_width_hours = float(args.window_multiplier) * avg_residence_count * median_cadence_hours
    local_df, half_window_hours = local_window_dataframe(rows, bundle_time, det_x, det_y, window_width_hours)
    local_df.to_csv(args.out_csv, index=False)

    offset_step = offset_step_from_rows(rows, float(args.offset_scale))
    plot_local_windows(
        local_df,
        args.out_png,
        target=str(args.target),
        sector=int(args.sector),
        window_multiplier=float(args.window_multiplier),
        window_width_hours=window_width_hours,
        total_removed=int(np.count_nonzero(~keep)),
        n_total=len(keep),
        ytick_fontsize=int(args.ytick_fontsize),
        xtick_fontsize=int(args.xtick_fontsize),
        axis_label_fontsize=int(args.axis_label_fontsize),
        title_fontsize=int(args.title_fontsize),
        label_every=int(args.label_every),
        line_alpha=float(args.line_alpha),
        marker_alpha=float(args.marker_alpha),
        marker_size=float(args.marker_size),
        offset_step=offset_step,
        sum_bin_hours=float(args.sum_bin_hours),
        detrend_degree=detrend_degree,
        extra_mask_times=extra_mask_times,
        extra_mask_ranges=extra_mask_ranges,
        bootstrap_draws=int(args.bootstrap_draws),
        bootstrap_seed=int(args.bootstrap_seed),
        bootstrap_errorbar_stride=int(args.bootstrap_errorbar_stride),
    )

    summary_lines = [
        f"distinct_path_pixels={len(path_pixels)}",
        f"cadences_total={len(keep)}",
        f"cadences_removed={int(np.count_nonzero(~keep))}",
        "extra_mask_times=" + ",".join(f"{value:.12f}" for value in extra_mask_times),
        "extra_mask_ranges="
        + ",".join(f"[{start:.12f},{stop:.12f}]" for start, stop in extra_mask_ranges),
        f"average_residence_count={avg_residence_count}",
        f"median_cadence_hours={median_cadence_hours}",
        f"window_multiplier={float(args.window_multiplier)}",
        f"window_width_hours={window_width_hours}",
        f"half_window_hours={half_window_hours}",
        f"sum_bin_hours={float(args.sum_bin_hours)}",
        f"offset_step={offset_step}",
        f"detrend_degree={detrend_degree}",
        f"bootstrap_draws={int(args.bootstrap_draws)}",
        f"bootstrap_seed={int(args.bootstrap_seed)}",
        f"bootstrap_errorbar_stride={int(args.bootstrap_errorbar_stride)}",
    ]
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(args.out_png)
    print(
        {
            "window_width_hours": window_width_hours,
            "average_residence_count": avg_residence_count,
            "cadences_removed": int(np.count_nonzero(~keep)),
            "n_records": len(local_df),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
