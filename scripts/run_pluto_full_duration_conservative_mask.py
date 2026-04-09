#!/usr/bin/env python3
"""Build, mask, and compare the full-duration Pluto Sector 92 workflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
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

from kbo_tess.control_studies import evaluate_null_track_controls, write_null_control_products
from kbo_tess.ephem import load_ephemeris_csv, select_detector_track
from kbo_tess.flux_ledger import (
    compute_pixel_residency_map,
    extract_pixel_time_series,
    select_representative_pixels,
)
from kbo_tess.recovery import (
    align_ephemeris_to_cube_bundle,
    filter_cube_bundle,
    load_cube_npz,
    run_known_path_recovery,
    write_recovery_products,
)
from kbo_tess.tess_loader import (
    build_cube_bundle_from_tess_asteroids,
    materialize_cube_bundle_to_npz,
    write_cube_bundle_metadata_json,
)


@dataclass(frozen=True)
class MaskWindow:
    start: float
    stop: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full-duration Pluto conservative cadence-mask workflow.",
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
        "--clipped-summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_clipped_recovery" / "pluto_sector92_clipped3828p30_summary.json",
    )
    parser.add_argument(
        "--full-bundle-npz",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_full_duration_conservative_mask" / "pluto_sector92_cam2_ccd1_full_cube_bundle.npz",
    )
    parser.add_argument(
        "--full-bundle-metadata-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_full_duration_conservative_mask" / "pluto_sector92_cam2_ccd1_full_cube_bundle_metadata.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_full_duration_conservative_mask",
    )
    parser.add_argument(
        "--note-path",
        type=Path,
        default=PROJECT_ROOT / "notes" / "2026-04-08_pluto_full_duration_conservative_mask_report.txt",
    )
    parser.add_argument("--time-start", type=float, default=3803.5008)
    parser.add_argument("--time-stop", type=float, default=3828.5008)
    parser.add_argument("--cutout-rows", type=int, default=21)
    parser.add_argument("--cutout-cols", type=int, default=21)
    parser.add_argument("--barycentric-mode", action="store_true", default=True)
    parser.add_argument("--loader-bg-method", default="rolling")
    parser.add_argument("--recovery-background-method", choices=("none", "median", "savgol"), default="none")
    parser.add_argument("--interpolation", choices=("nearest", "linear"), default="linear")
    parser.add_argument("--offset-grid-radius", type=float, default=1.0)
    parser.add_argument("--offset-grid-step", type=float, default=0.5)
    parser.add_argument("--mask-start", type=float, default=3813.70)
    parser.add_argument("--mask-stop", type=float, default=3816.10)
    parser.add_argument("--mask-threshold-support", type=float, default=50.0)
    return parser


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ensure_full_bundle(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.full_bundle_npz.exists() and args.full_bundle_metadata_json.exists():
        bundle = load_cube_npz(args.full_bundle_npz)
        return args.full_bundle_npz, args.full_bundle_metadata_json, bundle

    ephem = load_ephemeris_csv(args.ephem_csv, target=args.target, require_exact_tdb=True)
    bundle = build_cube_bundle_from_tess_asteroids(
        ephem,
        target=args.target,
        shape=(int(args.cutout_rows), int(args.cutout_cols)),
        time_start=float(args.time_start),
        time_stop=float(args.time_stop),
        barycentric_mode=bool(args.barycentric_mode),
        bg_method=str(args.loader_bg_method),
    )
    materialize_cube_bundle_to_npz(bundle, args.full_bundle_npz)
    write_cube_bundle_metadata_json(bundle, args.full_bundle_metadata_json)
    return args.full_bundle_npz, args.full_bundle_metadata_json, bundle


def choose_selected_pixels(track_x: np.ndarray, track_y: np.ndarray, shape: tuple[int, int]) -> pd.DataFrame:
    selected = select_representative_pixels(
        track_x,
        track_y,
        shape=shape,
        occupancy_radius=0.5,
        n_track_pixels=1,
    )
    yy, xx = np.indices(shape, dtype=float)
    distance_from_track = np.full(shape, np.inf, dtype=float)
    for x0, y0 in zip(track_x, track_y, strict=False):
        distance_from_track = np.minimum(distance_from_track, np.hypot(xx - float(x0), yy - float(y0)))
    taken = {(int(row.pixel_x), int(row.pixel_y)) for row in selected.itertuples(index=False)}
    extra_rows: list[dict[str, float | str]] = []
    off_rank = 1
    for flat_index in np.argsort(distance_from_track.ravel())[::-1]:
        iy, ix = np.unravel_index(int(flat_index), shape)
        key = (int(ix), int(iy))
        if key in taken:
            continue
        extra_rows.append(
            {
                "label": f"off_track_reference_{off_rank}",
                "pixel_x": float(ix),
                "pixel_y": float(iy),
                "category": "off_track",
                "residency_count": 0.0,
                "residency_rank": float("nan"),
            }
        )
        taken.add(key)
        off_rank += 1
        if off_rank > 2:
            break
    return pd.concat([selected, pd.DataFrame(extra_rows)], ignore_index=True)


def frame_diagnostics(cube: np.ndarray, time: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": np.arange(len(time), dtype=int),
            "time_tdb_jd_minus_2457000": time,
            "cube_median": np.nanmedian(cube, axis=(1, 2)),
            "cube_p95": np.nanpercentile(cube, 95, axis=(1, 2)),
            "cube_max": np.nanmax(cube, axis=(1, 2)),
        }
    )


def build_mask(time: np.ndarray, window: MaskWindow) -> np.ndarray:
    return (time < float(window.start)) | (time > float(window.stop))


def interval_table(time: np.ndarray, keep_mask: np.ndarray) -> list[dict[str, float | int]]:
    bad_idx = np.where(~keep_mask)[0]
    if len(bad_idx) == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(bad_idx[0])
    for idx in bad_idx[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = prev = idx
    groups.append((start, prev))
    rows: list[dict[str, float | int]] = []
    for start, stop in groups:
        rows.append(
            {
                "start_index": start,
                "stop_index": stop,
                "start_time_tdb_jd_minus_2457000": float(time[start]),
                "stop_time_tdb_jd_minus_2457000": float(time[stop]),
                "n_cadences": int(stop - start + 1),
            }
        )
    return rows


def subset_frame_aligned_bundle(bundle: dict[str, Any], keep_mask: np.ndarray, metadata_extra: dict[str, Any]) -> dict[str, Any]:
    full_n = int(np.asarray(bundle["cube"]).shape[0])
    masked: dict[str, Any] = {}
    for key, value in bundle.items():
        if key == "metadata":
            continue
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == full_n:
            masked[key] = array[keep_mask]
        else:
            masked[key] = value
    masked["quality_mask"] = np.ones(int(keep_mask.sum()), dtype=bool)
    metadata = dict(bundle.get("metadata", {}))
    metadata.update(metadata_extra)
    metadata["n_frames_total"] = full_n
    metadata["n_frames_used"] = int(keep_mask.sum())
    masked["metadata"] = metadata
    return masked


def plot_residency_with_selection(
    occupancy: np.ndarray,
    track_x: np.ndarray,
    track_y: np.ndarray,
    selected_pixels: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    im = ax.imshow(occupancy, origin="lower", cmap="magma")
    ax.plot(track_x, track_y, color="white", linewidth=1.2, alpha=0.8)
    for row in selected_pixels.itertuples(index=False):
        color = {"track_residency": "cyan", "neighbor": "lime", "off_track": "tab:orange"}[str(row.category)]
        ax.scatter([row.pixel_x], [row.pixel_y], marker="s", s=50, facecolors="none", edgecolors=color, linewidths=1.5)
        ax.text(float(row.pixel_x) + 0.35, float(row.pixel_y) + 0.2, str(row.label), color=color, fontsize=7)
    ax.set_title("Pluto nominal-path pixel residency")
    ax.set_xlabel("Local x [pix]")
    ax.set_ylabel("Local y [pix]")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Cadence count within 0.5 px")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_selected_pixels_with_mask(pixel_table: pd.DataFrame, mask_window: MaskWindow, output_path: Path) -> None:
    labels = list(pixel_table["label"].drop_duplicates())
    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.3 * nrows), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, label in zip(axes, labels, strict=False):
        subset = pixel_table[pixel_table["label"] == label].copy()
        ax.plot(subset["time_tdb_jd_minus_2457000"], subset["value"], color="black", linewidth=0.8)
        near = subset[subset["nominal_within_0p5"].astype(bool)]
        ax.scatter(near["time_tdb_jd_minus_2457000"], near["value"], s=8, color="tab:blue", label="Pluto <= 0.5 px")
        dropped = subset[~subset["quality_keep"].astype(bool)]
        if not dropped.empty:
            ax.scatter(dropped["time_tdb_jd_minus_2457000"], dropped["value"], s=10, color="red", marker="x", label="masked")
        ax.axvspan(mask_window.start, mask_window.stop, color="red", alpha=0.12)
        ax.set_title(f"{label} ({int(subset.iloc[0]['pixel_x'])}, {int(subset.iloc[0]['pixel_y'])})")
        ax.set_xlabel("BTJD")
        ax.set_ylabel("Pixel value")
        ax.grid(alpha=0.25)
    for ax in axes[len(labels):]:
        ax.axis("off")
    handles, labels_text = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_text, loc="upper right", frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mask_diagnostic(frame_table: pd.DataFrame, mask_window: MaskWindow, threshold: float, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    axes[0].plot(frame_table["time_tdb_jd_minus_2457000"], frame_table["cube_median"], color="black", linewidth=0.9)
    axes[0].axhline(threshold, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].axvspan(mask_window.start, mask_window.stop, color="red", alpha=0.12)
    axes[0].set_ylabel("Cutout median")
    axes[0].set_title("Frame-wide broad-ramp diagnostic")
    axes[0].grid(alpha=0.25)
    axes[1].plot(frame_table["time_tdb_jd_minus_2457000"], frame_table["cube_p95"], color="tab:blue", linewidth=0.9)
    axes[1].axvspan(mask_window.start, mask_window.stop, color="red", alpha=0.12)
    axes[1].set_ylabel("Cutout p95")
    axes[1].set_xlabel("BTJD")
    axes[1].grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summarize_bundle(raw_bundle: dict[str, Any], filtered_bundle: dict[str, Any], path: Path) -> dict[str, Any]:
    raw_cube = np.asarray(raw_bundle["cube"], dtype=float)
    cube = np.asarray(filtered_bundle["cube"], dtype=float)
    time = np.asarray(filtered_bundle["time_tdb_jd_minus_2457000"], dtype=float)
    metadata = dict(filtered_bundle.get("metadata", {}))
    payload = {
        "bundle_path": str(path),
        "time_start_tdb_jd_minus_2457000": float(time.min()),
        "time_stop_tdb_jd_minus_2457000": float(time.max()),
        "n_cadences_total_raw": int(raw_cube.shape[0]),
        "n_cadences_quality_kept": int(cube.shape[0]),
        "cutout_shape": [int(cube.shape[1]), int(cube.shape[2])],
        "barycentric_mode": metadata.get("barycentric_mode"),
        "loader_bg_method": metadata.get("loader_bg_method"),
    }
    return payload


def comparison_payload(
    unmasked_recovery: dict[str, Any],
    unmasked_nulls: dict[str, Any],
    masked_recovery: dict[str, Any],
    masked_nulls: dict[str, Any],
    clipped_summary: dict[str, Any],
) -> dict[str, Any]:
    clipped_none = clipped_summary["results"]["none"]
    clipped_rec = clipped_none["recovery_summary"]
    clipped_null = clipped_none["null_summary"]
    return {
        "unmasked": {
            "nominal_score": float(unmasked_recovery["summary"]["nominal_score"]),
            "best_dx": float(unmasked_recovery["summary"]["best_dx"]),
            "best_dy": float(unmasked_recovery["summary"]["best_dy"]),
            "best_score": float(unmasked_recovery["summary"]["best_score"]),
            "best_near_nominal_dx": float(unmasked_nulls["summary"]["best_near_nominal_dx"]),
            "best_near_nominal_dy": float(unmasked_nulls["summary"]["best_near_nominal_dy"]),
            "best_near_nominal_score": float(unmasked_nulls["summary"]["best_near_nominal_score"]),
            "best_control_score": float(unmasked_nulls["summary"]["best_control_score"]),
            "nominal_minus_best_control": float(unmasked_nulls["summary"]["nominal_minus_best_control"]),
            "exact_nominal_is_best_local": bool(
                np.isclose(float(unmasked_recovery["summary"]["best_dx"]), 0.0)
                and np.isclose(float(unmasked_recovery["summary"]["best_dy"]), 0.0)
            ),
        },
        "masked": {
            "nominal_score": float(masked_recovery["summary"]["nominal_score"]),
            "best_dx": float(masked_recovery["summary"]["best_dx"]),
            "best_dy": float(masked_recovery["summary"]["best_dy"]),
            "best_score": float(masked_recovery["summary"]["best_score"]),
            "best_near_nominal_dx": float(masked_nulls["summary"]["best_near_nominal_dx"]),
            "best_near_nominal_dy": float(masked_nulls["summary"]["best_near_nominal_dy"]),
            "best_near_nominal_score": float(masked_nulls["summary"]["best_near_nominal_score"]),
            "best_control_score": float(masked_nulls["summary"]["best_control_score"]),
            "nominal_minus_best_control": float(masked_nulls["summary"]["nominal_minus_best_control"]),
            "exact_nominal_is_best_local": bool(
                np.isclose(float(masked_recovery["summary"]["best_dx"]), 0.0)
                and np.isclose(float(masked_recovery["summary"]["best_dy"]), 0.0)
            ),
        },
        "bounded_clipped_reference": {
            "nominal_score": float(clipped_rec["nominal_score"]),
            "best_dx": float(clipped_rec["best_dx"]),
            "best_dy": float(clipped_rec["best_dy"]),
            "best_score": float(clipped_rec["best_score"]),
            "best_near_nominal_dx": float(clipped_null["best_near_nominal_dx"]),
            "best_near_nominal_dy": float(clipped_null["best_near_nominal_dy"]),
            "best_near_nominal_score": float(clipped_null["best_near_nominal_score"]),
            "best_control_score": float(clipped_null["best_control_score"]),
            "nominal_minus_best_control": float(clipped_null["nominal_minus_best_control"]),
            "exact_nominal_is_best_local": bool(
                np.isclose(float(clipped_rec["best_dx"]), 0.0)
                and np.isclose(float(clipped_rec["best_dy"]), 0.0)
            ),
        },
    }


def classify_result(masked: dict[str, Any], clipped: dict[str, Any]) -> str:
    masked_nominal = float(masked["nominal_score"])
    masked_best_control = float(masked["best_control_score"])
    masked_best_is_nominal = bool(masked["exact_nominal_is_best_local"])
    clipped_margin = float(clipped["nominal_minus_best_control"])
    masked_margin = float(masked["nominal_minus_best_control"])
    if masked_best_is_nominal and masked_margin > 0.0 and masked_margin >= 0.5 * clipped_margin:
        return "A"
    if masked_nominal > 1.5 * masked_best_control or masked_margin > 0.0 or masked_best_is_nominal:
        return "B"
    return "C"


def plot_recovery_comparison(comparison: dict[str, Any], output_path: Path) -> None:
    labels = ["unmasked", "masked", "bounded_clipped_reference"]
    nominal = [comparison[label]["nominal_score"] for label in labels]
    best = [comparison[label]["best_score"] for label in labels]
    control = [comparison[label]["best_control_score"] for label in labels]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.bar(x - width, nominal, width=width, label="nominal")
    ax.bar(x, best, width=width, label="best local")
    ax.bar(x + width, control, width=width, label="best control")
    ax.set_xticks(x, ["full unmasked", "full masked", "bounded clipped"])
    ax.set_ylabel("Aperture S/N score")
    ax.set_title("Pluto recovery comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_note(
    note_path: Path,
    full_bundle_path: Path,
    selected_pixels: pd.DataFrame,
    mask_summary: dict[str, Any],
    comparison: dict[str, Any],
    classification: str,
) -> None:
    masked = comparison["masked"]
    unmasked = comparison["unmasked"]
    lines = [
        "Pluto Full-Duration Conservative Mask Report",
        "2026-04-08",
        "",
        f"Full-duration bundle path",
        f"- {full_bundle_path}",
        "",
        "Selected path/comparison pixels",
    ]
    for row in selected_pixels.itertuples(index=False):
        lines.append(
            f"- {row.label}: ({int(row.pixel_x)}, {int(row.pixel_y)}) [{row.category}]"
        )
    lines.extend(
        [
            "",
            "Cadence mask choice",
            f"- Rule: keep cadences outside {mask_summary['mask_rule']['remove_intervals'][0]['start']:.2f} -> {mask_summary['mask_rule']['remove_intervals'][0]['stop']:.2f} BTJD",
            f"- Frames before: {mask_summary['n_frames_before']}",
            f"- Frames after: {mask_summary['n_frames_after']}",
            f"- Frames removed: {mask_summary['n_frames_removed']}",
            f"- Justification: {mask_summary['justification']}",
            "",
            "Masked vs unmasked recovery",
            f"- Unmasked nominal score: {unmasked['nominal_score']:.4f}",
            f"- Masked nominal score: {masked['nominal_score']:.4f}",
            f"- Unmasked best dx,dy: ({unmasked['best_dx']:.1f}, {unmasked['best_dy']:.1f})",
            f"- Masked best dx,dy: ({masked['best_dx']:.1f}, {masked['best_dy']:.1f})",
            f"- Unmasked best near-nominal score: {unmasked['best_near_nominal_score']:.4f}",
            f"- Masked best near-nominal score: {masked['best_near_nominal_score']:.4f}",
            f"- Unmasked best control score: {unmasked['best_control_score']:.4f}",
            f"- Masked best control score: {masked['best_control_score']:.4f}",
            f"- Masked nominal minus best control: {masked['nominal_minus_best_control']:.4f}",
            f"- Masked exact nominal is best local solution: {masked['exact_nominal_is_best_local']}",
            "",
            f"Final classification: {classification}",
        ]
    )
    if classification == "A":
        lines.append("- Conservative masking is enough to recover physically sensible full-duration nominal Pluto behavior.")
    elif classification == "B":
        lines.append("- Masking improves things materially but full-duration Pluto is still not clean.")
    else:
        lines.append("- Masking does not materially help; the issue is deeper than the obvious ramps.")
    lines.extend(
        [
            "",
            "Next smallest unresolved issue",
            "- Determine whether the remaining full-duration mismatch is dominated by additional weaker cadence-systematics outside the masked interval or by a deeper track/stack calibration issue.",
            "",
        ]
    )
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    prefix = f"{slug(args.target)}_sector{args.sector}_cam{args.camera}_ccd{args.ccd}"
    full_bundle_path, full_metadata_path, _ = ensure_full_bundle(args)

    raw_full_bundle = load_cube_npz(full_bundle_path)
    full_bundle = filter_cube_bundle(raw_full_bundle)
    full_summary = summarize_bundle(raw_full_bundle, full_bundle, full_bundle_path)
    write_json(full_summary, args.outdir / f"{prefix}_full_bundle_summary.json")

    ephem = load_ephemeris_csv(args.ephem_csv, target=args.target, require_exact_tdb=True)
    detector_track = select_detector_track(
        ephem,
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        target=args.target,
        require_exact_tdb=True,
    )

    cube = np.asarray(full_bundle["cube"], dtype=float)
    time = np.asarray(full_bundle["time_tdb_jd_minus_2457000"], dtype=float)
    aligned = align_ephemeris_to_cube_bundle(detector_track, full_bundle)
    track_x = np.asarray(aligned["bundle_local_x"], dtype=float)
    track_y = np.asarray(aligned["bundle_local_y"], dtype=float)

    selected_pixels = choose_selected_pixels(track_x, track_y, cube.shape[1:])
    selected_pixels.to_csv(args.outdir / f"{prefix}_selected_pixels.csv", index=False)

    residency = compute_pixel_residency_map(cube.shape[1:], track_x, track_y, radius=0.5)
    plot_residency_with_selection(
        residency,
        track_x,
        track_y,
        selected_pixels,
        args.outdir / f"{prefix}_pixel_residency_map.png",
    )

    mask_window = MaskWindow(start=float(args.mask_start), stop=float(args.mask_stop))
    keep_mask = build_mask(time, mask_window)
    pixel_table = extract_pixel_time_series(
        cube,
        time,
        selected_pixels=selected_pixels,
        nominal_x=track_x,
        nominal_y=track_y,
        calibrated_x=track_x,
        calibrated_y=track_y,
        quality_mask=keep_mask,
    )
    pixel_csv = args.outdir / f"{prefix}_selected_pixel_timeseries.csv"
    pixel_table.to_csv(pixel_csv, index=False)
    plot_selected_pixels_with_mask(
        pixel_table,
        mask_window,
        args.outdir / f"{prefix}_pixel_timeseries_masked.png",
    )

    frame_table = frame_diagnostics(cube, time)
    frame_table["keep"] = keep_mask
    frame_table.to_csv(args.outdir / f"{prefix}_cadence_mask.csv", index=False)
    plot_mask_diagnostic(
        frame_table,
        mask_window,
        float(args.mask_threshold_support),
        args.outdir / f"{prefix}_mask_diagnostic.png",
    )

    mask_summary = {
        "bundle_path": str(full_bundle_path),
        "mask_rule": {
            "type": "hard_time_cut",
            "remove_intervals": [{"start": float(mask_window.start), "stop": float(mask_window.stop)}],
            "supporting_cutout_median_threshold": float(args.mask_threshold_support),
        },
        "n_frames_before": int(len(time)),
        "n_frames_after": int(keep_mask.sum()),
        "n_frames_removed": int((~keep_mask).sum()),
        "removed_intervals_realized": interval_table(time, keep_mask),
        "justification": (
            "The selected Pluto path pixel, its immediate neighbors, and off-track comparison pixels all show a "
            "broad simultaneous flux-ramp episode centered near BTJD 3815.84. The frame-wide cutout median rises "
            "from a near-zero baseline to very large positive values across the same interval, so the broad event "
            "is treated as detector-wide contamination rather than physically sensible Pluto behavior."
        ),
    }
    write_json(mask_summary, args.outdir / f"{prefix}_cadence_mask_summary.json")

    masked_bundle = subset_frame_aligned_bundle(
        full_bundle,
        keep_mask,
        metadata_extra={
            "source_full_bundle": str(full_bundle_path),
            "mask_rule": "hard_time_cut",
            "mask_start_tdb_jd_minus_2457000": float(mask_window.start),
            "mask_stop_tdb_jd_minus_2457000": float(mask_window.stop),
        },
    )
    masked_bundle_path = args.outdir / f"{prefix}_full_cube_bundle_masked.npz"
    masked_meta_path = args.outdir / f"{prefix}_full_cube_bundle_masked_metadata.json"
    materialize_cube_bundle_to_npz(masked_bundle, masked_bundle_path)
    write_cube_bundle_metadata_json(masked_bundle, masked_meta_path)

    masked_track = align_ephemeris_to_cube_bundle(detector_track, masked_bundle)
    masked_track_path = args.outdir / f"{prefix}_masked_bundle_track.csv"
    masked_track.to_csv(masked_track_path, index=False)

    unmasked_recovery = run_known_path_recovery(
        detector_track,
        full_bundle,
        background_method=args.recovery_background_method,
        interpolation=args.interpolation,
        offset_grid_radius=args.offset_grid_radius,
        offset_grid_step=args.offset_grid_step,
        ranking_score_method="aperture_snr",
    )
    unmasked_recovery_paths = write_recovery_products(
        unmasked_recovery,
        args.outdir / "recovery_none_unmasked",
        prefix=f"{prefix}_full_unmasked_none",
    )
    unmasked_nulls = evaluate_null_track_controls(unmasked_recovery)
    unmasked_null_paths = write_null_control_products(
        unmasked_nulls,
        args.outdir / "null_controls_none_unmasked",
        prefix=f"{prefix}_full_unmasked_none_null_controls",
    )

    masked_recovery = run_known_path_recovery(
        detector_track,
        masked_bundle,
        background_method=args.recovery_background_method,
        interpolation=args.interpolation,
        offset_grid_radius=args.offset_grid_radius,
        offset_grid_step=args.offset_grid_step,
        ranking_score_method="aperture_snr",
    )
    masked_recovery_paths = write_recovery_products(
        masked_recovery,
        args.outdir / "recovery_none_masked",
        prefix=f"{prefix}_full_masked_none",
    )
    masked_nulls = evaluate_null_track_controls(masked_recovery)
    masked_null_paths = write_null_control_products(
        masked_nulls,
        args.outdir / "null_controls_none_masked",
        prefix=f"{prefix}_full_masked_none_null_controls",
    )

    clipped_summary = json.loads(args.clipped_summary_json.read_text(encoding="utf-8"))
    comparison = comparison_payload(unmasked_recovery, unmasked_nulls, masked_recovery, masked_nulls, clipped_summary)
    classification = classify_result(comparison["masked"], comparison["bounded_clipped_reference"])
    comparison["classification"] = classification
    comparison["full_bundle_path"] = str(full_bundle_path)
    comparison["masked_bundle_path"] = str(masked_bundle_path)
    write_json(comparison, args.outdir / f"{prefix}_masked_vs_unmasked_comparison.json")
    plot_recovery_comparison(comparison, args.outdir / f"{prefix}_masked_vs_unmasked_comparison.png")

    write_note(
        args.note_path,
        full_bundle_path,
        selected_pixels,
        mask_summary,
        comparison,
        classification,
    )

    report = {
        "full_bundle_path": str(full_bundle_path),
        "full_bundle_metadata_json": str(full_metadata_path),
        "full_bundle_summary_json": str(args.outdir / f"{prefix}_full_bundle_summary.json"),
        "selected_pixels_csv": str(args.outdir / f"{prefix}_selected_pixels.csv"),
        "pixel_timeseries_csv": str(pixel_csv),
        "pixel_residency_png": str(args.outdir / f"{prefix}_pixel_residency_map.png"),
        "pixel_timeseries_png": str(args.outdir / f"{prefix}_pixel_timeseries_masked.png"),
        "mask_csv": str(args.outdir / f"{prefix}_cadence_mask.csv"),
        "mask_summary_json": str(args.outdir / f"{prefix}_cadence_mask_summary.json"),
        "masked_bundle_path": str(masked_bundle_path),
        "masked_bundle_metadata_json": str(masked_meta_path),
        "masked_track_csv": str(masked_track_path),
        "unmasked_recovery_summary_json": str(unmasked_recovery_paths["summary_json"]),
        "unmasked_null_summary_json": str(unmasked_null_paths["summary_json"]),
        "masked_recovery_summary_json": str(masked_recovery_paths["summary_json"]),
        "masked_null_summary_json": str(masked_null_paths["summary_json"]),
        "comparison_json": str(args.outdir / f"{prefix}_masked_vs_unmasked_comparison.json"),
        "comparison_png": str(args.outdir / f"{prefix}_masked_vs_unmasked_comparison.png"),
        "note_path": str(args.note_path),
        "classification": classification,
    }
    write_json(report, args.outdir / f"{prefix}_workflow_report.json")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
