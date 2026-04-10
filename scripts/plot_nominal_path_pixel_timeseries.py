#!/usr/bin/env python3
"""Plot raw detector-pixel time series along a target's nominal path."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--mask-mode",
        choices=("simple", "common_mode_v1"),
        default="simple",
        help="Cadence masking strategy.",
    )
    parser.add_argument("--mask-window", type=int, default=49)
    parser.add_argument("--mask-sigma", type=float, default=5.0)
    parser.add_argument("--target-box-size", type=int, default=5)
    parser.add_argument("--path-box-size", type=int, default=9)
    parser.add_argument("--baseline-hours", type=float, default=24.0)
    parser.add_argument("--slope-smooth-hours", type=float, default=1.0)
    parser.add_argument("--seed-z-outer", type=float, default=4.0)
    parser.add_argument("--seed-z-path", type=float, default=4.0)
    parser.add_argument("--seed-z-q90", type=float, default=3.0)
    parser.add_argument("--seed-z-scatter", type=float, default=3.0)
    parser.add_argument("--seed-z-slope", type=float, default=4.0)
    parser.add_argument("--grow-z", type=float, default=2.0)
    parser.add_argument("--merge-gap-hours", type=float, default=2.0)
    parser.add_argument("--interval-pad-hours", type=float, default=1.0)
    parser.add_argument("--min-interval-hours", type=float, default=6.0)
    parser.add_argument("--edge-window-hours", type=float, default=12.0)
    parser.add_argument("--edge-reference-hours", type=float, default=24.0)
    parser.add_argument("--edge-release-hours", type=float, default=6.0)
    parser.add_argument("--edge-release-z", type=float, default=1.5)
    parser.add_argument(
        "--extra-mask-times",
        type=float,
        nargs="*",
        default=(),
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
    parser.add_argument(
        "--out-mask-csv",
        type=Path,
        default=None,
        help="Optional per-cadence mask diagnostics CSV.",
    )
    parser.add_argument(
        "--out-interval-json",
        type=Path,
        default=None,
        help="Optional interval-summary JSON for the automatic mask.",
    )
    parser.add_argument("--label-every", type=int, default=3)
    parser.add_argument("--ytick-fontsize", type=int, default=14)
    parser.add_argument("--xtick-fontsize", type=int, default=14)
    parser.add_argument("--axis-label-fontsize", type=int, default=15)
    parser.add_argument("--title-fontsize", type=int, default=15)
    parser.add_argument("--line-alpha", type=float, default=0.78)
    parser.add_argument("--line-width", type=float, default=0.8)
    parser.add_argument("--marker-alpha", type=float, default=0.32)
    parser.add_argument("--marker-size", type=float, default=12.0)
    parser.add_argument(
        "--detrend-linear",
        action="store_true",
        help="Subtract a best-fit linear trend from each detector-pixel time series before plotting.",
    )
    parser.add_argument(
        "--detrend-degree",
        type=int,
        default=None,
        help="Polynomial degree to remove from each detector-pixel time series. Overrides --detrend-linear.",
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


def robust_sigma(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return 1.0
    med = float(np.nanmedian(values[finite]))
    mad = float(np.nanmedian(np.abs(values[finite] - med)))
    sigma = 1.4826 * mad
    return float(sigma if sigma > 1e-12 else 1.0)


def window_count_from_hours(bundle_time: np.ndarray, hours: float, minimum: int = 3) -> int:
    if len(bundle_time) <= 1:
        return max(1, minimum)
    dt_hours = float(np.nanmedian(np.diff(bundle_time)) * 24.0)
    if not np.isfinite(dt_hours) or dt_hours <= 0:
        return max(1, minimum)
    count = max(int(np.ceil(hours / dt_hours)), minimum)
    if count % 2 == 0:
        count += 1
    return int(count)


def rolling_median(values: np.ndarray, count: int) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    return series.rolling(window=int(max(1, count)), center=True, min_periods=1).median().to_numpy(dtype=float)


def frame_common_mode_diagnostics(
    raw_cube: np.ndarray,
    local_x: np.ndarray,
    local_y: np.ndarray,
    target_box_size: int,
    path_box_size: int,
) -> dict[str, np.ndarray]:
    n_frames, ny, nx = raw_cube.shape
    yy, xx = np.indices((ny, nx), dtype=float)
    half_target = max(0.5, (float(target_box_size) - 1.0) / 2.0)
    half_path = max(half_target + 1.0, (float(path_box_size) - 1.0) / 2.0)

    m_all = np.full(n_frames, np.nan, dtype=float)
    m_outer = np.full(n_frames, np.nan, dtype=float)
    q90_outer = np.full(n_frames, np.nan, dtype=float)
    s_outer = np.full(n_frames, np.nan, dtype=float)
    m_path = np.full(n_frames, np.nan, dtype=float)

    for i in range(n_frames):
        frame = raw_cube[i]
        target_mask = (np.abs(xx - float(local_x[i])) <= half_target) & (np.abs(yy - float(local_y[i])) <= half_target)
        path_ring = (
            (np.abs(xx - float(local_x[i])) <= half_path)
            & (np.abs(yy - float(local_y[i])) <= half_path)
            & (~target_mask)
        )
        outer_mask = ~target_mask
        outer_values = frame[outer_mask]
        path_values = frame[path_ring]
        m_all[i] = float(np.nanmedian(frame))
        if outer_values.size:
            m_outer[i] = float(np.nanmedian(outer_values))
            q90_outer[i] = float(np.nanpercentile(outer_values, 90.0))
            med_outer = m_outer[i]
            s_outer[i] = float(1.4826 * np.nanmedian(np.abs(outer_values - med_outer)))
        if path_values.size:
            m_path[i] = float(np.nanmedian(path_values))
        else:
            m_path[i] = m_outer[i]

    return {
        "m_all": m_all,
        "m_outer": m_outer,
        "q90_outer": q90_outer,
        "s_outer": s_outer,
        "m_path": m_path,
    }


def standardized_residuals(values: np.ndarray, baseline_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    baseline = rolling_median(values, baseline_count)
    resid = values - baseline
    sigma = robust_sigma(resid)
    z = resid / sigma
    return baseline, resid, z, sigma


def intervals_from_boolean(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    return [(int(group[0]), int(group[-1])) for group in groups if group.size]


def seed_grow_intervals(seed_mask: np.ndarray, grow_mask: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for start, stop in intervals_from_boolean(grow_mask):
        if np.any(seed_mask[start : stop + 1]):
            intervals.append((start, stop))
    return intervals


def merge_intervals(intervals: list[tuple[int, int]], bundle_time: np.ndarray, gap_hours: float) -> list[tuple[int, int]]:
    if not intervals:
        return []
    gap_days = float(gap_hours) / 24.0
    merged = [intervals[0]]
    for start, stop in intervals[1:]:
        prev_start, prev_stop = merged[-1]
        if float(bundle_time[start] - bundle_time[prev_stop]) <= gap_days:
            merged[-1] = (prev_start, max(prev_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def pad_intervals(intervals: list[tuple[int, int]], bundle_time: np.ndarray, pad_hours: float) -> list[tuple[int, int]]:
    if not intervals:
        return []
    pad_days = float(pad_hours) / 24.0
    padded: list[tuple[int, int]] = []
    for start, stop in intervals:
        start_time = float(bundle_time[start] - pad_days)
        stop_time = float(bundle_time[stop] + pad_days)
        start_idx = int(np.searchsorted(bundle_time, start_time, side="left"))
        stop_idx = int(np.searchsorted(bundle_time, stop_time, side="right") - 1)
        padded.append((max(0, start_idx), min(len(bundle_time) - 1, stop_idx)))
    return padded


def filter_short_intervals(
    intervals: list[tuple[int, int]],
    bundle_time: np.ndarray,
    min_hours: float,
) -> list[tuple[int, int]]:
    if not intervals:
        return []
    min_days = float(min_hours) / 24.0
    kept: list[tuple[int, int]] = []
    last_idx = len(bundle_time) - 1
    for start, stop in intervals:
        duration = float(bundle_time[stop] - bundle_time[start])
        if duration >= min_days or start == 0 or stop == last_idx:
            kept.append((start, stop))
    return kept


def intervals_to_mask(intervals: list[tuple[int, int]], n_frames: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in intervals:
        mask[start : stop + 1] = True
    return mask


def edge_interval(
    bundle_time: np.ndarray,
    grow_mask: np.ndarray,
    m_outer: np.ndarray,
    m_path: np.ndarray,
    sigma_outer: float,
    sigma_path: float,
    side: str,
    edge_window_hours: float,
    edge_reference_hours: float,
    edge_release_hours: float,
    edge_release_z: float,
    z_outer: np.ndarray,
    z_path: np.ndarray,
    z_q90: np.ndarray,
    z_scatter: np.ndarray,
) -> tuple[tuple[int, int] | None, dict[str, object]]:
    assert side in {"left", "right"}
    n = len(bundle_time)
    if n == 0:
        return None, {"edge_interval": False}
    edge_days = float(edge_window_hours) / 24.0
    ref_days = float(edge_reference_hours) / 24.0
    release_count = max(1, window_count_from_hours(bundle_time, edge_release_hours, minimum=1))
    if side == "left":
        start_time = float(bundle_time[0])
        edge_mask = bundle_time <= start_time + edge_days
        ref_mask = (bundle_time > start_time + edge_days) & (bundle_time <= start_time + edge_days + ref_days)
    else:
        stop_time = float(bundle_time[-1])
        edge_mask = bundle_time >= stop_time - edge_days
        ref_mask = (bundle_time < stop_time - edge_days) & (bundle_time >= stop_time - edge_days - ref_days)
    edge_fraction = float(np.mean(grow_mask[edge_mask])) if np.any(edge_mask) else 0.0
    z_edge_outer = 0.0
    z_edge_path = 0.0
    if np.any(edge_mask) and np.any(ref_mask):
        z_edge_outer = float((np.nanmedian(m_outer[edge_mask]) - np.nanmedian(m_outer[ref_mask])) / sigma_outer)
        z_edge_path = float((np.nanmedian(m_path[edge_mask]) - np.nanmedian(m_path[ref_mask])) / sigma_path)
    edge_trigger = (edge_fraction > 0.5) or (
        ((z_edge_outer > 3.0) or (z_edge_path > 3.0)) and (edge_fraction > 0.2)
    )
    info: dict[str, object] = {
        "edge_interval": bool(edge_trigger),
        "edge_side": side,
        "edge_fraction_grow": edge_fraction,
        "edge_z_outer": z_edge_outer,
        "edge_z_path": z_edge_path,
    }
    if not edge_trigger:
        return None, info
    calm = (z_outer < edge_release_z) & (z_path < edge_release_z) & (z_q90 < edge_release_z) & (z_scatter < edge_release_z)
    if side == "left":
        release_start = None
        for idx in range(0, max(1, n - release_count + 1)):
            if np.all(calm[idx : idx + release_count]):
                release_start = idx
                break
        if release_start is None:
            return (0, n - 1), info
        return (0, max(0, release_start - 1)), info
    release_stop = None
    for idx in range(n - release_count, -1, -1):
        if np.all(calm[idx : idx + release_count]):
            release_stop = idx + release_count - 1
            break
    if release_stop is None:
        return (0, n - 1), info
    return (min(n - 1, release_stop + 1), n - 1), info


def build_interval_records(
    intervals: list[tuple[int, int]],
    bundle_time: np.ndarray,
    seed_mask: np.ndarray,
    z_outer: np.ndarray,
    z_path: np.ndarray,
    z_q90: np.ndarray,
    z_scatter: np.ndarray,
    z_slope: np.ndarray,
    edge_intervals: list[tuple[int, int]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for interval_id, (start, stop) in enumerate(intervals, start=1):
        reasons: list[str] = []
        seed_slice = seed_mask[start : stop + 1]
        if np.any(z_outer[start : stop + 1] > 4.0):
            reasons.append("z_outer")
        if np.any(z_path[start : stop + 1] > 4.0):
            reasons.append("z_path")
        if np.any((z_q90[start : stop + 1] > 3.0) & (z_scatter[start : stop + 1] > 3.0)):
            reasons.append("z_q90+z_scatter")
        if np.any(np.abs(z_slope[start : stop + 1]) > 4.0):
            reasons.append("z_slope")
        if not reasons and np.any(seed_slice):
            reasons.append("seeded")
        edge_flag = any(start == e0 and stop == e1 for e0, e1 in edge_intervals)
        records.append(
            {
                "interval_id": int(interval_id),
                "start_time": float(bundle_time[start]),
                "stop_time": float(bundle_time[stop]),
                "duration_hours": float((bundle_time[stop] - bundle_time[start]) * 24.0),
                "n_cadences": int(stop - start + 1),
                "edge_interval": bool(edge_flag),
                "seed_reason": reasons,
                "peak_z_outer": float(np.nanmax(z_outer[start : stop + 1])),
                "peak_z_path": float(np.nanmax(z_path[start : stop + 1])),
                "peak_z_q90": float(np.nanmax(z_q90[start : stop + 1])),
                "peak_z_scatter": float(np.nanmax(z_scatter[start : stop + 1])),
                "peak_abs_z_slope": float(np.nanmax(np.abs(z_slope[start : stop + 1]))),
            }
        )
    return records


def compute_mask_bundle(
    raw_cube: np.ndarray,
    bundle_time: np.ndarray,
    local_x: np.ndarray,
    local_y: np.ndarray,
    *,
    mask_mode: str,
    mask_window: int,
    mask_sigma: float,
    target_box_size: int,
    path_box_size: int,
    baseline_hours: float,
    slope_smooth_hours: float,
    seed_z_outer: float,
    seed_z_path: float,
    seed_z_q90: float,
    seed_z_scatter: float,
    seed_z_slope: float,
    grow_z: float,
    merge_gap_hours: float,
    interval_pad_hours: float,
    min_interval_hours: float,
    edge_window_hours: float,
    edge_reference_hours: float,
    edge_release_hours: float,
    edge_release_z: float,
    extra_mask_times: tuple[float, ...],
    extra_mask_ranges: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    spike_keep, m_all_simple, running_simple, simple_mad, simple_threshold = compute_ramp_mask(
        raw_cube, window=int(mask_window), sigma=float(mask_sigma)
    )
    spike_mask = ~spike_keep
    diagnostics = frame_common_mode_diagnostics(
        raw_cube,
        local_x=local_x,
        local_y=local_y,
        target_box_size=int(target_box_size),
        path_box_size=int(path_box_size),
    )
    m_all = diagnostics["m_all"]
    m_outer = diagnostics["m_outer"]
    q90_outer = diagnostics["q90_outer"]
    s_outer = diagnostics["s_outer"]
    m_path = diagnostics["m_path"]
    baseline_count = window_count_from_hours(bundle_time, baseline_hours, minimum=5)
    slope_count = window_count_from_hours(bundle_time, slope_smooth_hours, minimum=3)

    _base_outer, resid_outer, z_outer, sigma_outer = standardized_residuals(m_outer, baseline_count)
    _base_path, resid_path, z_path, sigma_path = standardized_residuals(m_path, baseline_count)
    _base_q90, resid_q90, z_q90, sigma_q90 = standardized_residuals(q90_outer, baseline_count)
    _base_scatter, resid_scatter, z_scatter, sigma_scatter = standardized_residuals(s_outer, baseline_count)
    smooth_outer = rolling_median(m_outer, slope_count)
    deriv_outer = np.gradient(smooth_outer, bundle_time * 24.0)
    _base_slope, resid_slope, z_slope, sigma_slope = standardized_residuals(deriv_outer, baseline_count)

    interval_mask = np.zeros(len(bundle_time), dtype=bool)
    edge_mask = np.zeros(len(bundle_time), dtype=bool)
    intervals: list[tuple[int, int]] = []
    edge_intervals: list[tuple[int, int]] = []
    edge_infos: list[dict[str, object]] = []
    if mask_mode == "common_mode_v1":
        primary_seed_mask = (
            (z_outer > float(seed_z_outer))
            | (z_path > float(seed_z_path))
            | ((z_q90 > float(seed_z_q90)) & (z_scatter > float(seed_z_scatter)))
        )
        broad_support_mask = (
            (z_outer > float(grow_z))
            | (z_path > float(grow_z))
            | ((z_q90 > float(grow_z)) & (z_scatter > float(grow_z)))
        )
        slope_seed_mask = (np.abs(z_slope) > float(seed_z_slope)) & broad_support_mask
        seed_mask = primary_seed_mask | slope_seed_mask
        grow_mask = broad_support_mask
        intervals = seed_grow_intervals(seed_mask=seed_mask, grow_mask=grow_mask)
        intervals = merge_intervals(intervals, bundle_time=bundle_time, gap_hours=float(merge_gap_hours))
        intervals = filter_short_intervals(intervals, bundle_time=bundle_time, min_hours=float(min_interval_hours))
        intervals = pad_intervals(intervals, bundle_time=bundle_time, pad_hours=float(interval_pad_hours))
        intervals = merge_intervals(intervals, bundle_time=bundle_time, gap_hours=float(merge_gap_hours))
        intervals = filter_short_intervals(intervals, bundle_time=bundle_time, min_hours=float(min_interval_hours))

        left_interval, left_info = edge_interval(
            bundle_time=bundle_time,
            grow_mask=grow_mask,
            m_outer=m_outer,
            m_path=m_path,
            sigma_outer=sigma_outer,
            sigma_path=sigma_path,
            side="left",
            edge_window_hours=float(edge_window_hours),
            edge_reference_hours=float(edge_reference_hours),
            edge_release_hours=float(edge_release_hours),
            edge_release_z=float(edge_release_z),
            z_outer=z_outer,
            z_path=z_path,
            z_q90=z_q90,
            z_scatter=z_scatter,
        )
        right_interval, right_info = edge_interval(
            bundle_time=bundle_time,
            grow_mask=grow_mask,
            m_outer=m_outer,
            m_path=m_path,
            sigma_outer=sigma_outer,
            sigma_path=sigma_path,
            side="right",
            edge_window_hours=float(edge_window_hours),
            edge_reference_hours=float(edge_reference_hours),
            edge_release_hours=float(edge_release_hours),
            edge_release_z=float(edge_release_z),
            z_outer=z_outer,
            z_path=z_path,
            z_q90=z_q90,
            z_scatter=z_scatter,
        )
        for interval in (left_interval, right_interval):
            if interval is not None:
                edge_intervals.append(interval)
        edge_infos.extend([left_info, right_info])
        edge_intervals = merge_intervals(edge_intervals, bundle_time=bundle_time, gap_hours=float(merge_gap_hours))
        edge_intervals = pad_intervals(edge_intervals, bundle_time=bundle_time, pad_hours=float(interval_pad_hours))
        edge_mask = intervals_to_mask(edge_intervals, len(bundle_time))
        interval_mask = intervals_to_mask(intervals, len(bundle_time))
        intervals = merge_intervals(intervals + edge_intervals, bundle_time=bundle_time, gap_hours=float(merge_gap_hours))
        intervals = filter_short_intervals(intervals, bundle_time=bundle_time, min_hours=float(min_interval_hours))
        interval_mask = intervals_to_mask(intervals, len(bundle_time))
    else:
        seed_mask = np.zeros(len(bundle_time), dtype=bool)
        grow_mask = np.zeros(len(bundle_time), dtype=bool)

    manual_mask = np.zeros(len(bundle_time), dtype=bool)
    if extra_mask_times:
        for value in extra_mask_times:
            nearest = int(np.nanargmin(np.abs(bundle_time - value)))
            if abs(float(bundle_time[nearest]) - value) > 1.0 / 86400.0:
                raise ValueError(f"No cadence matched requested extra mask time {value:.12f} within 1 second.")
            manual_mask[nearest] = True
    for start, stop in extra_mask_ranges:
        lo = min(start, stop)
        hi = max(start, stop)
        manual_mask |= (bundle_time >= lo) & (bundle_time <= hi)

    keep = ~(spike_mask | interval_mask | manual_mask)
    interval_records = build_interval_records(
        intervals=intervals,
        bundle_time=bundle_time,
        seed_mask=seed_mask,
        z_outer=z_outer,
        z_path=z_path,
        z_q90=z_q90,
        z_scatter=z_scatter,
        z_slope=z_slope,
        edge_intervals=edge_intervals,
    )
    interval_id = np.zeros(len(bundle_time), dtype=int)
    for record in interval_records:
        mask = (bundle_time >= float(record["start_time"])) & (bundle_time <= float(record["stop_time"]))
        interval_id[mask] = int(record["interval_id"])
    diagnostics_df = pd.DataFrame(
        {
            "time_tdb_jd_minus_2457000": bundle_time,
            "keep": keep,
            "masked_by_spike": spike_mask,
            "masked_by_interval": interval_mask,
            "masked_by_manual": manual_mask,
            "interval_id": interval_id,
            "m_all": m_all,
            "m_outer": m_outer,
            "q90_outer": q90_outer,
            "s_outer": s_outer,
            "m_path": m_path,
            "running_median_simple": running_simple,
            "simple_resid": m_all_simple - running_simple,
            "z_outer": z_outer,
            "z_path": z_path,
            "z_q90_outer": z_q90,
            "z_s_outer": z_scatter,
            "d_outer": deriv_outer,
            "z_d_outer": z_slope,
            "seed_mask": seed_mask,
            "grow_mask": grow_mask,
        }
    )
    summary = {
        "mask_mode": mask_mode,
        "n_cadences_total": int(len(bundle_time)),
        "n_masked_spike": int(np.count_nonzero(spike_mask)),
        "n_masked_interval": int(np.count_nonzero(interval_mask)),
        "n_masked_manual": int(np.count_nonzero(manual_mask)),
        "n_masked_total": int(np.count_nonzero(~keep)),
        "mask_window_cadences": int(mask_window),
        "mask_sigma": float(mask_sigma),
        "target_box_size": int(target_box_size),
        "path_box_size": int(path_box_size),
        "baseline_hours": float(baseline_hours),
        "slope_smooth_hours": float(slope_smooth_hours),
        "seed_z_outer": float(seed_z_outer),
        "seed_z_path": float(seed_z_path),
        "seed_z_q90": float(seed_z_q90),
        "seed_z_scatter": float(seed_z_scatter),
        "seed_z_slope": float(seed_z_slope),
        "grow_z": float(grow_z),
        "merge_gap_hours": float(merge_gap_hours),
        "interval_pad_hours": float(interval_pad_hours),
        "min_interval_hours": float(min_interval_hours),
        "edge_window_hours": float(edge_window_hours),
        "edge_reference_hours": float(edge_reference_hours),
        "edge_release_hours": float(edge_release_hours),
        "edge_release_z": float(edge_release_z),
        "simple_mad": float(simple_mad),
        "simple_threshold": float(simple_threshold),
        "sigma_outer": float(sigma_outer),
        "sigma_path": float(sigma_path),
        "sigma_q90": float(sigma_q90),
        "sigma_scatter": float(sigma_scatter),
        "sigma_slope": float(sigma_slope),
        "edge_infos": edge_infos,
    }
    return keep, diagnostics_df, interval_records, summary


def build_series_rows(
    bundle_time: np.ndarray,
    det_x: np.ndarray,
    det_y: np.ndarray,
    corner1: np.ndarray,
    corner2: np.ndarray,
    raw_matched: np.ndarray,
    keep: np.ndarray,
    path_pixels: list[tuple[int, int]],
    detrend_degree: int | bool,
) -> list[dict[str, object]]:
    degree = int(detrend_degree) if not isinstance(detrend_degree, bool) else (1 if detrend_degree else 0)
    if degree < 0:
        raise ValueError(f"`detrend_degree` must be non-negative; got {degree}.")
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
        trend_degree = 0
        trend_coefficients = np.array([], dtype=float)
        trend_slope = np.nan
        trend_intercept = np.nan
        if degree >= 1 and np.count_nonzero(finite) >= degree + 1:
            trend_coefficients = np.polyfit(bundle_time[finite], vals[finite], degree)
            plotted[finite] = vals[finite] - np.polyval(trend_coefficients, bundle_time[finite])
            trend_degree = degree
            if degree == 1:
                trend_slope = float(trend_coefficients[0])
                trend_intercept = float(trend_coefficients[1])
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
                "trend_degree": int(trend_degree),
                "trend_coefficients": trend_coefficients.tolist(),
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
                    "trend_degree": int(row["trend_degree"]),
                    "trend_coefficients_json": json.dumps(row["trend_coefficients"]),  # type: ignore[arg-type]
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
    line_width: float,
    marker_alpha: float,
    marker_size: float,
    offset_scale: float,
    detrend_degree: int,
    extra_mask_times: tuple[float, ...],
    extra_mask_ranges: tuple[tuple[float, float], ...],
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
        ax.plot(bundle_time, plot_y, color=(0.1, 0.1, 0.1, line_alpha), linewidth=line_width)
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
    if detrend_degree <= 0:
        subtitle = (
            f"Each trace is median-subtracted and vertically offset; orange markers show cadences where {target} is centered in that pixel"
        )
    elif detrend_degree == 1:
        subtitle = (
            "Each trace has its best-fit linear trend removed; orange markers show cadences where "
            f"{target} is centered in that pixel"
        )
    elif detrend_degree == 2:
        subtitle = (
            "Each trace has its best-fit quadratic trend removed; orange markers show cadences where "
            f"{target} is centered in that pixel"
        )
    else:
        subtitle = (
            f"Each trace has its best-fit polynomial trend (degree {detrend_degree}) removed; orange markers show cadences where "
            f"{target} is centered in that pixel"
        )
    extra_text = ""
    if extra_mask_times:
        formatted = ", ".join(f"{value:.6f}" for value in extra_mask_times)
        extra_text = f"\nManual extra cadence cut(s): {formatted} BTJD"
    if extra_mask_ranges:
        formatted_ranges = ", ".join(f"[{start:.6f}, {stop:.6f}]" for start, stop in extra_mask_ranges)
        extra_text += f"\nManual extra time cut range(s): {formatted_ranges} BTJD"
    ax.set_title(
        f"{target} Sector {sector} raw detector-pixel time series along the nominal path\n"
        f"{subtitle}\n"
        f"Cadence mask removed {int(np.count_nonzero(~keep))} of {len(keep)} cadences{extra_text}",
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
    diagnostics_df: pd.DataFrame,
    mask_summary: dict[str, object],
    interval_records: list[dict[str, object]],
    rows: list[dict[str, object]],
    offset_step: float,
    extra_mask_times: tuple[float, ...],
    extra_mask_ranges: tuple[tuple[float, float], ...],
    detrend_degree: int,
) -> None:
    no_centered = [row["pixel"] for row in rows if not np.any(row["centered"])]  # type: ignore[index]
    lines = [
        f"distinct_path_pixels={len(path_pixels)}",
        f"cadences_total={len(keep)}",
        f"cadences_removed={int(np.count_nonzero(~keep))}",
        f"mask_mode={mask_summary['mask_mode']}",
        f"cutout_median_flux_median={float(np.nanmedian(diagnostics_df['m_all'].to_numpy(dtype=float)))}",
        f"simple_mad={float(mask_summary['simple_mad'])}",
        f"simple_threshold={float(mask_summary['simple_threshold'])}",
        f"n_masked_spike={int(mask_summary['n_masked_spike'])}",
        f"n_masked_interval={int(mask_summary['n_masked_interval'])}",
        f"n_masked_manual={int(mask_summary['n_masked_manual'])}",
        f"n_intervals={len(interval_records)}",
        f"target_box_size={int(mask_summary['target_box_size'])}",
        f"path_box_size={int(mask_summary['path_box_size'])}",
        f"baseline_hours={float(mask_summary['baseline_hours'])}",
        f"slope_smooth_hours={float(mask_summary['slope_smooth_hours'])}",
        f"seed_z_outer={float(mask_summary['seed_z_outer'])}",
        f"seed_z_path={float(mask_summary['seed_z_path'])}",
        f"seed_z_q90={float(mask_summary['seed_z_q90'])}",
        f"seed_z_scatter={float(mask_summary['seed_z_scatter'])}",
        f"seed_z_slope={float(mask_summary['seed_z_slope'])}",
        f"grow_z={float(mask_summary['grow_z'])}",
        f"merge_gap_hours={float(mask_summary['merge_gap_hours'])}",
        f"interval_pad_hours={float(mask_summary['interval_pad_hours'])}",
        f"min_interval_hours={float(mask_summary['min_interval_hours'])}",
        f"edge_window_hours={float(mask_summary['edge_window_hours'])}",
        f"edge_reference_hours={float(mask_summary['edge_reference_hours'])}",
        f"edge_release_hours={float(mask_summary['edge_release_hours'])}",
        f"edge_release_z={float(mask_summary['edge_release_z'])}",
        f"offset_step={offset_step}",
        f"detrend_degree={detrend_degree}",
        f"pixels_with_no_centered_kept_cadences={len(no_centered)}",
    ]
    if extra_mask_times:
        lines.append("extra_mask_times=" + ",".join(f"{value:.12f}" for value in extra_mask_times))
    if extra_mask_ranges:
        lines.append(
            "extra_mask_ranges="
            + ",".join(f"[{start:.12f},{stop:.12f}]" for start, stop in extra_mask_ranges)
        )
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
    local_x = np.asarray(bundle["local_x"], dtype=float)
    local_y = np.asarray(bundle["local_y"], dtype=float)
    corner1 = np.asarray(bundle["corner1"], dtype=int)
    corner2 = np.asarray(bundle["corner2"], dtype=int)

    path_pixels = unique_path_pixels(det_x, det_y)
    ephem = select_detector_ephem(args.ephem_csv, args.target, args.sector, args.camera, args.ccd)
    raw_cube, raw_time = build_raw_flux_cube(ephem, shape=(args.cutout_size, args.cutout_size))
    match_idx = match_bundle_times_to_raw(raw_time, bundle_time)
    raw_matched = raw_cube[match_idx]

    detrend_degree = int(args.detrend_degree) if args.detrend_degree is not None else (1 if bool(args.detrend_linear) else 0)
    extra_mask_times = tuple(float(value) for value in args.extra_mask_times)
    extra_mask_ranges = tuple((float(start), float(stop)) for start, stop in args.extra_mask_ranges)
    keep, diagnostics_df, interval_records, mask_summary = compute_mask_bundle(
        raw_cube=raw_matched,
        bundle_time=bundle_time,
        local_x=local_x,
        local_y=local_y,
        mask_mode=str(args.mask_mode),
        mask_window=int(args.mask_window),
        mask_sigma=float(args.mask_sigma),
        target_box_size=int(args.target_box_size),
        path_box_size=int(args.path_box_size),
        baseline_hours=float(args.baseline_hours),
        slope_smooth_hours=float(args.slope_smooth_hours),
        seed_z_outer=float(args.seed_z_outer),
        seed_z_path=float(args.seed_z_path),
        seed_z_q90=float(args.seed_z_q90),
        seed_z_scatter=float(args.seed_z_scatter),
        seed_z_slope=float(args.seed_z_slope),
        grow_z=float(args.grow_z),
        merge_gap_hours=float(args.merge_gap_hours),
        interval_pad_hours=float(args.interval_pad_hours),
        min_interval_hours=float(args.min_interval_hours),
        edge_window_hours=float(args.edge_window_hours),
        edge_reference_hours=float(args.edge_reference_hours),
        edge_release_hours=float(args.edge_release_hours),
        edge_release_z=float(args.edge_release_z),
        extra_mask_times=extra_mask_times,
        extra_mask_ranges=extra_mask_ranges,
    )
    if args.out_mask_csv is not None:
        args.out_mask_csv.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_df.to_csv(args.out_mask_csv, index=False)
    if args.out_interval_json is not None:
        args.out_interval_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_interval_json.write_text(
            json.dumps({"mask_summary": mask_summary, "intervals": interval_records}, indent=2) + "\n",
            encoding="utf-8",
        )
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
        line_width=args.line_width,
        marker_alpha=args.marker_alpha,
        marker_size=args.marker_size,
        offset_scale=args.offset_scale,
        detrend_degree=detrend_degree,
        extra_mask_times=extra_mask_times,
        extra_mask_ranges=extra_mask_ranges,
    )
    write_summary(
        args.out_summary,
        path_pixels,
        keep,
        diagnostics_df,
        mask_summary,
        interval_records,
        rows,
        offset_step,
        extra_mask_times=extra_mask_times,
        extra_mask_ranges=extra_mask_ranges,
        detrend_degree=detrend_degree,
    )

    no_centered = [row["pixel"] for row in rows if not np.any(row["centered"])]  # type: ignore[index]
    print(args.out_png)
    print(
        {
            "distinct_path_pixels": len(path_pixels),
            "cadences_removed": int(np.count_nonzero(~keep)),
            "extra_mask_times": extra_mask_times,
            "extra_mask_ranges": extra_mask_ranges,
            "detrend_degree": detrend_degree,
            "mask_mode": str(args.mask_mode),
            "pixels_with_no_centered_kept_cadences": len(no_centered),
            "no_centered_pixels": no_centered,
            "offset_step": offset_step,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
