"""Common-mode cadence masking utilities for moving-cutout workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ramp_mask(raw_cube: np.ndarray, window: int, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    med = np.nanmedian(raw_cube, axis=(1, 2))
    pad = int(window) // 2
    padded = np.pad(med, pad, mode="edge")
    running = np.array([np.median(padded[i : i + int(window)]) for i in range(len(med))], dtype=float)
    resid = med - running
    mad = float(np.median(np.abs(resid - np.median(resid))))
    threshold = float(np.median(resid) + float(sigma) * 1.4826 * mad)
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
    count = max(int(np.ceil(float(hours) / dt_hours)), minimum)
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


def filter_short_intervals(intervals: list[tuple[int, int]], bundle_time: np.ndarray, min_hours: float) -> list[tuple[int, int]]:
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
    edge_trigger = (edge_fraction > 0.5) or (((z_edge_outer > 3.0) or (z_edge_path > 3.0)) and (edge_fraction > 0.2))
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
    seed_z_outer: float,
    seed_z_path: float,
    seed_z_q90: float,
    seed_z_scatter: float,
    seed_z_slope: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for interval_id, (start, stop) in enumerate(intervals, start=1):
        reasons: list[str] = []
        seed_slice = seed_mask[start : stop + 1]
        if np.any(z_outer[start : stop + 1] > float(seed_z_outer)):
            reasons.append("z_outer")
        if np.any(z_path[start : stop + 1] > float(seed_z_path)):
            reasons.append("z_path")
        if np.any((z_q90[start : stop + 1] > float(seed_z_q90)) & (z_scatter[start : stop + 1] > float(seed_z_scatter))):
            reasons.append("z_q90+z_scatter")
        if np.any(np.abs(z_slope[start : stop + 1]) > float(seed_z_slope)):
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
    mask_mode: str = "common_mode_v1",
    mask_window: int = 49,
    mask_sigma: float = 5.0,
    target_box_size: int = 5,
    path_box_size: int = 9,
    baseline_hours: float = 24.0,
    slope_smooth_hours: float = 1.0,
    seed_z_outer: float = 4.0,
    seed_z_path: float = 4.0,
    seed_z_q90: float = 3.0,
    seed_z_scatter: float = 3.0,
    seed_z_slope: float = 4.0,
    grow_z: float = 2.0,
    merge_gap_hours: float = 10.0,
    interval_pad_hours: float = 6.0,
    min_interval_hours: float = 6.0,
    edge_window_hours: float = 12.0,
    edge_reference_hours: float = 24.0,
    edge_release_hours: float = 6.0,
    edge_release_z: float = 1.5,
    extra_mask_times: tuple[float, ...] = (),
    extra_mask_ranges: tuple[tuple[float, float], ...] = (),
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

    _base_outer, _resid_outer, z_outer, sigma_outer = standardized_residuals(m_outer, baseline_count)
    _base_path, _resid_path, z_path, sigma_path = standardized_residuals(m_path, baseline_count)
    _base_q90, _resid_q90, z_q90, sigma_q90 = standardized_residuals(q90_outer, baseline_count)
    _base_scatter, _resid_scatter, z_scatter, sigma_scatter = standardized_residuals(s_outer, baseline_count)
    smooth_outer = rolling_median(m_outer, slope_count)
    deriv_outer = np.gradient(smooth_outer, bundle_time * 24.0)
    _base_slope, _resid_slope, z_slope, sigma_slope = standardized_residuals(deriv_outer, baseline_count)

    interval_mask = np.zeros(len(bundle_time), dtype=bool)
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
        intervals = merge_intervals(intervals + edge_intervals, bundle_time=bundle_time, gap_hours=float(merge_gap_hours))
        intervals = filter_short_intervals(intervals, bundle_time=bundle_time, min_hours=float(min_interval_hours))
        interval_mask = intervals_to_mask(intervals, len(bundle_time))
    else:
        seed_mask = np.zeros(len(bundle_time), dtype=bool)
        grow_mask = np.zeros(len(bundle_time), dtype=bool)
        intervals = []

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
        seed_z_outer=float(seed_z_outer),
        seed_z_path=float(seed_z_path),
        seed_z_q90=float(seed_z_q90),
        seed_z_scatter=float(seed_z_scatter),
        seed_z_slope=float(seed_z_slope),
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
        }
    )
    mask_summary = {
        "mask_mode": str(mask_mode),
        "mask_window": int(mask_window),
        "mask_sigma": float(mask_sigma),
        "simple_mad": float(simple_mad),
        "simple_threshold": float(simple_threshold),
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
        "sigma_outer": float(sigma_outer),
        "sigma_path": float(sigma_path),
        "sigma_q90_outer": float(sigma_q90),
        "sigma_s_outer": float(sigma_scatter),
        "sigma_d_outer": float(sigma_slope),
        "edge_infos": edge_infos,
    }
    return keep, diagnostics_df, interval_records, mask_summary
