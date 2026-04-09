#!/usr/bin/env python3
"""Diagnose the strongest off-center source in the clipped Pluto nominal stack."""

from __future__ import annotations

import argparse
import json
import os
import re
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

from kbo_tess.background import subtract_background
from kbo_tess.recovery import filter_cube_bundle, load_cube_npz
from kbo_tess.shiftstack import stack_with_support


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace which cadences feed the strongest off-center Pluto stack source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cube-npz",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_clipped_recovery" / "pluto_sector92_cam2_ccd1_cube_bundle_clipped_3828p30.npz",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_edge_source_diagnostic",
    )
    parser.add_argument(
        "--catalog-csv",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "pluto_full_duration_catalog_sources"
        / "pluto_sector92_cam2_ccd1_tic_crossers_deduped.csv",
    )
    parser.add_argument("--background-method", choices=("none", "median", "savgol"), default="none")
    parser.add_argument("--interpolation", choices=("nearest", "linear"), default="linear")
    parser.add_argument("--support-fraction-min", type=float, default=0.9)
    parser.add_argument("--min-distance-from-target", type=float, default=4.0)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--catalog-top-n", type=int, default=8)
    return parser


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def find_supported_offcenter_feature(
    stack: np.ndarray,
    support: np.ndarray,
    reference: tuple[float, float],
    support_fraction_min: float,
    min_distance_from_target: float,
) -> tuple[int, int, float, float]:
    refx, refy = reference
    yy, xx = np.indices(stack.shape, dtype=float)
    rr = np.hypot(xx - refx, yy - refy)
    mask = rr >= float(min_distance_from_target)
    mask &= support >= float(support_fraction_min) * float(np.nanmax(support))
    masked = np.where(mask, stack, np.nan)
    if not np.isfinite(masked).any():
        raise ValueError("No supported off-center feature found under the requested thresholds.")
    peak_flat = int(np.nanargmax(masked))
    py, px = np.unravel_index(peak_flat, stack.shape)
    return int(px), int(py), float(stack[py, px]), float(support[py, px])


def main() -> int:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    bundle = filter_cube_bundle(load_cube_npz(args.cube_npz))
    cube = np.asarray(bundle["cube"], dtype=float)
    time = np.asarray(bundle["time_tdb_jd_minus_2457000"], dtype=float)
    x = np.asarray(bundle["local_x"], dtype=float)
    y = np.asarray(bundle["local_y"], dtype=float)
    residual = subtract_background(cube, method=args.background_method)[1]
    reference = (float(x[0]), float(y[0]))

    stack_data = stack_with_support(
        residual,
        x,
        y,
        method="mean",
        interpolation=args.interpolation,
        reference=reference,
    )
    stack = np.asarray(stack_data["stack"], dtype=float)
    aligned_frames = np.asarray(stack_data["aligned_frames"], dtype=float)
    support = np.asarray(stack_data["support_map"], dtype=float)

    feature_x, feature_y, feature_value, feature_support = find_supported_offcenter_feature(
        stack,
        support,
        reference=reference,
        support_fraction_min=float(args.support_fraction_min),
        min_distance_from_target=float(args.min_distance_from_target),
    )

    x0 = max(0, feature_x - 1)
    x1 = min(stack.shape[1], feature_x + 2)
    y0 = max(0, feature_y - 1)
    y1 = min(stack.shape[0], feature_y + 2)
    patch_contribution = np.nanmean(aligned_frames[:, y0:y1, x0:x1], axis=(1, 2))

    rows: list[dict[str, float]] = []
    for idx in range(len(time)):
        shift_row = stack_data["shift_rows"][idx]
        raw_feature_x = float(feature_x - shift_row["applied_shift_x"])
        raw_feature_y = float(feature_y - shift_row["applied_shift_y"])
        frame = residual[idx]
        rows.append(
            {
                "frame_index": float(idx),
                "time_tdb_jd_minus_2457000": float(time[idx]),
                "feature_patch_contribution": float(patch_contribution[idx]),
                "raw_feature_x": raw_feature_x,
                "raw_feature_y": raw_feature_y,
                "frame_median": float(np.nanmedian(frame)),
                "frame_p95": float(np.nanpercentile(frame, 95)),
            }
        )
    table = pd.DataFrame(rows).sort_values("feature_patch_contribution", ascending=False).reset_index(drop=True)
    csv_path = args.outdir / "pluto_clipped_edge_source_contributions.csv"
    table.to_csv(csv_path, index=False)

    top = table.head(int(args.top_n)).copy()
    top_idx = top["frame_index"].astype(int).to_numpy()
    corner1 = np.asarray(bundle["corner1"], dtype=float)
    corner2 = np.asarray(bundle["corner2"], dtype=float)
    feature_detector_x = table["raw_feature_x"].to_numpy() + corner1[table["frame_index"].astype(int).to_numpy()]
    feature_detector_y = table["raw_feature_y"].to_numpy() + corner2[table["frame_index"].astype(int).to_numpy()]
    top_feature_detector_x = top["raw_feature_x"].to_numpy() + corner1[top_idx]
    top_feature_detector_y = top["raw_feature_y"].to_numpy() + corner2[top_idx]

    catalog_matches = pd.DataFrame()
    best_catalog_match: dict[str, Any] | None = None
    if args.catalog_csv.exists():
        catalog = pd.read_csv(args.catalog_csv)
        match_rows: list[dict[str, Any]] = []
        for row in catalog.itertuples(index=False):
            detector_x = float(row.detector_x)
            detector_y = float(row.detector_y)
            local_track_x = detector_x - corner1
            local_track_y = detector_y - corner2
            same_frame_distance = np.hypot(
                local_track_x[table["frame_index"].astype(int).to_numpy()] - table["raw_feature_x"].to_numpy(),
                local_track_y[table["frame_index"].astype(int).to_numpy()] - table["raw_feature_y"].to_numpy(),
            )
            top_same_frame_distance = np.hypot(
                local_track_x[top_idx] - top["raw_feature_x"].to_numpy(),
                local_track_y[top_idx] - top["raw_feature_y"].to_numpy(),
            )
            match_rows.append(
                {
                    "source_key": row.source_key,
                    "tic_id": row.tic_id,
                    "gaia_id": row.gaia_id,
                    "tmag": float(row.tmag) if np.isfinite(row.tmag) else np.nan,
                    "detector_x": detector_x,
                    "detector_y": detector_y,
                    "distance_to_feature_detector_mean": float(
                        np.hypot(detector_x - np.mean(top_feature_detector_x), detector_y - np.mean(top_feature_detector_y))
                    ),
                    "mean_same_frame_distance_all": float(np.mean(same_frame_distance)),
                    "mean_same_frame_distance_top": float(np.mean(top_same_frame_distance)),
                    "max_same_frame_distance_top": float(np.max(top_same_frame_distance)),
                    "min_same_frame_distance_all": float(np.min(same_frame_distance)),
                    "n_top_within_0p5": int(np.count_nonzero(top_same_frame_distance <= 0.5)),
                    "n_top_within_1p0": int(np.count_nonzero(top_same_frame_distance <= 1.0)),
                }
            )
        catalog_matches = pd.DataFrame(match_rows).sort_values(
            ["mean_same_frame_distance_top", "tmag"],
            ascending=[True, True],
            na_position="last",
        ).reset_index(drop=True)
        catalog_matches.to_csv(args.outdir / "pluto_clipped_edge_source_catalog_matches.csv", index=False)
        if not catalog_matches.empty:
            best_catalog_match = catalog_matches.iloc[0].to_dict()

    top_summary = {
        "cube_npz": str(args.cube_npz),
        "catalog_csv": str(args.catalog_csv) if args.catalog_csv.exists() else "",
        "background_method": args.background_method,
        "interpolation": args.interpolation,
        "reference_x": float(reference[0]),
        "reference_y": float(reference[1]),
        "feature_x": int(feature_x),
        "feature_y": int(feature_y),
        "feature_value": float(feature_value),
        "feature_support": float(feature_support),
        "n_frames": int(len(time)),
        "top_n": int(args.top_n),
        "top_time_min": float(top["time_tdb_jd_minus_2457000"].min()),
        "top_time_max": float(top["time_tdb_jd_minus_2457000"].max()),
        "top_raw_feature_x_mean": float(top["raw_feature_x"].mean()),
        "top_raw_feature_y_mean": float(top["raw_feature_y"].mean()),
        "top_raw_feature_x_std": float(top["raw_feature_x"].std(ddof=0)),
        "top_raw_feature_y_std": float(top["raw_feature_y"].std(ddof=0)),
        "feature_detector_x_mean": float(np.mean(top_feature_detector_x)),
        "feature_detector_y_mean": float(np.mean(top_feature_detector_y)),
        "feature_detector_x_std": float(np.std(top_feature_detector_x, ddof=0)),
        "feature_detector_y_std": float(np.std(top_feature_detector_y, ddof=0)),
        "best_catalog_match": best_catalog_match,
    }
    write_json(top_summary, args.outdir / "pluto_clipped_edge_source_summary.json")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    finite = stack[np.isfinite(stack)]
    vmin, vmax = np.nanpercentile(finite, [5, 99.5]) if finite.size else (-1.0, 1.0)
    im = axes[0].imshow(stack, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].scatter([reference[0]], [reference[1]], marker="x", color="white", s=55, linewidths=1.5, label="Pluto")
    axes[0].scatter([feature_x], [feature_y], marker="s", facecolors="none", edgecolors="red", s=55, linewidths=1.5, label="off-center feature")
    axes[0].set_title("Clipped nominal stack")
    axes[0].set_xlabel("x [pix]")
    axes[0].set_ylabel("y [pix]")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.colorbar(im, ax=axes[0], shrink=0.82)

    axes[1].plot(table["time_tdb_jd_minus_2457000"], table["feature_patch_contribution"], ".", color="black", markersize=3)
    axes[1].scatter(top["time_tdb_jd_minus_2457000"], top["feature_patch_contribution"], color="tab:red", s=14)
    axes[1].set_title("Feature-patch contribution by cadence")
    axes[1].set_xlabel("BTJD")
    axes[1].set_ylabel("3x3 aligned patch mean")
    axes[1].grid(alpha=0.25)

    axes[2].scatter(table["raw_feature_x"], table["raw_feature_y"], s=8, color="0.75", alpha=0.6)
    axes[2].scatter(top["raw_feature_x"], top["raw_feature_y"], s=18, color="tab:red", alpha=0.85)
    if not catalog_matches.empty:
        palette = ["tab:blue", "tab:orange", "tab:green"]
        for color, row in zip(palette, catalog_matches.head(3).itertuples(index=False), strict=False):
            source_local_x = float(row.detector_x) - corner1
            source_local_y = float(row.detector_y) - corner2
            axes[2].plot(source_local_x, source_local_y, color=color, linewidth=1.2, alpha=0.9)
            label = f"TIC {row.tic_id} (T={row.tmag:.1f}, mean d={row.mean_same_frame_distance_top:.2f})"
            axes[2].text(
                float(source_local_x[-1]) + 0.15,
                float(source_local_y[-1]) + 0.1,
                label,
                color=color,
                fontsize=7,
            )
    axes[2].set_title("Back-projected raw cutout coordinates")
    axes[2].set_xlabel("raw local x [pix]")
    axes[2].set_ylabel("raw local y [pix]")
    axes[2].grid(alpha=0.25)

    fig.savefig(args.outdir / "pluto_clipped_edge_source_diagnostic.png", dpi=160)
    plt.close(fig)

    print(json.dumps(top_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
