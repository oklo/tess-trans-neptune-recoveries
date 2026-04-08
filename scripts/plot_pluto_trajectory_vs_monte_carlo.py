#!/usr/bin/env python3
"""Plot the observed Pluto track against the empirical-significance pseudo-tracks."""

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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kbo_tess.recovery import filter_cube_bundle, load_cube_npz  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the frozen Pluto track against the Monte Carlo pseudo-tracks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "pluto_empirical_significance"
        / "pluto_sector92_cam2_ccd1_empirical_significance_summary.json",
        help="Empirical-significance summary JSON.",
    )
    parser.add_argument(
        "--pseudo-tracks-csv",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "pluto_empirical_significance"
        / "pluto_sector92_cam2_ccd1_pseudo_tracks.csv",
        help="CSV of translated pseudo-tracks.",
    )
    parser.add_argument(
        "--cube-npz",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "pluto_calibration"
        / "primary"
        / "pluto_sector92_cam2_ccd1_cube_bundle.npz",
        help="Frozen Pluto bundle NPZ.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=48,
        help="Number of pseudo-tracks to draw individually in the overlay panel.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260406,
        help="Deterministic RNG seed for pseudo-track sampling.",
    )
    parser.add_argument(
        "--outpath",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "pluto_empirical_significance"
        / "pluto_sector92_cam2_ccd1_trajectory_vs_monte_carlo.png",
        help="PNG output path.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _extract_local_track(bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    for x_key, y_key in (("local_x", "local_y"), ("bundle_local_x", "bundle_local_y")):
        if x_key in bundle and y_key in bundle:
            return np.asarray(bundle[x_key], dtype=float), np.asarray(bundle[y_key], dtype=float)
    raise KeyError("Cube bundle does not contain local_x/local_y track coordinates.")


def _build_occupancy_map(
    pseudo_tracks: pd.DataFrame,
    nominal_x: np.ndarray,
    nominal_y: np.ndarray,
    calibrated_x_offset: float,
    shape: tuple[int, int],
) -> np.ndarray:
    ny, nx = shape
    occupancy = np.zeros((ny, nx), dtype=float)
    for row in pseudo_tracks.itertuples(index=False):
        translated_x = nominal_x + float(row.translation_x)
        translated_y = nominal_y + float(row.translation_y)
        calibrated_x = translated_x + float(calibrated_x_offset)
        for x_track in (translated_x, calibrated_x):
            pixel_x = np.clip(np.rint(x_track).astype(int), 0, nx - 1)
            pixel_y = np.clip(np.rint(translated_y).astype(int), 0, ny - 1)
            occupancy[pixel_y, pixel_x] += 1.0
    return occupancy


def main() -> int:
    args = build_parser().parse_args()
    summary = _load_json(args.summary_json)
    pseudo_tracks = pd.read_csv(args.pseudo_tracks_csv)
    bundle = filter_cube_bundle(load_cube_npz(args.cube_npz))
    cube = np.asarray(bundle["cube"], dtype=float)
    nominal_x, nominal_y = _extract_local_track(bundle)
    calibrated_x_offset = float(summary["calibrated_x_offset"])
    calibrated_x = nominal_x + calibrated_x_offset

    median_frame = np.nanmedian(cube, axis=0)
    occupancy = _build_occupancy_map(
        pseudo_tracks,
        nominal_x=nominal_x,
        nominal_y=nominal_y,
        calibrated_x_offset=calibrated_x_offset,
        shape=median_frame.shape,
    )

    rng = np.random.default_rng(int(args.seed))
    if args.sample_count >= len(pseudo_tracks):
        sample_table = pseudo_tracks.copy()
    else:
        sample_table = pseudo_tracks.iloc[np.sort(rng.choice(len(pseudo_tracks), size=args.sample_count, replace=False))]

    args.outpath.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)

    finite = median_frame[np.isfinite(median_frame)]
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [5, 99])
    else:
        vmin, vmax = -1.0, 1.0

    axes[0].imshow(median_frame, origin="lower", cmap="Greys_r", vmin=vmin, vmax=vmax)
    for row in sample_table.itertuples(index=False):
        translated_x = nominal_x + float(row.translation_x)
        translated_y = nominal_y + float(row.translation_y)
        axes[0].plot(translated_x, translated_y, color="lightgray", alpha=0.22, linewidth=0.8)
    axes[0].plot(nominal_x, nominal_y, color="#ff8c00", linewidth=2.0, linestyle="--", label="Nominal Pluto")
    axes[0].plot(calibrated_x, nominal_y, color="#00bcd4", linewidth=2.2, label="Best stack attempt")
    axes[0].scatter([nominal_x[0]], [nominal_y[0]], color="#ff8c00", s=32, marker="o")
    axes[0].scatter([calibrated_x[0]], [nominal_y[0]], color="#00bcd4", s=38, marker="x", linewidths=1.6)
    axes[0].set_title(f"Observed vs sampled null trajectories\n{len(sample_table)} of {len(pseudo_tracks)} MC tracks shown")
    axes[0].set_xlabel("Local x [pix]")
    axes[0].set_ylabel("Local y [pix]")
    axes[0].legend(loc="upper right", frameon=True)

    occupancy_artist = axes[1].imshow(occupancy, origin="lower", cmap="magma")
    axes[1].plot(nominal_x, nominal_y, color="white", linewidth=1.8, linestyle="--", label="Nominal Pluto")
    axes[1].plot(calibrated_x, nominal_y, color="#00e5ff", linewidth=2.0, label="Best stack attempt")
    axes[1].set_title("All Monte Carlo track occupancy\n(two observed track families per null)")
    axes[1].set_xlabel("Local x [pix]")
    axes[1].set_ylabel("Local y [pix]")
    axes[1].legend(loc="upper right", frameon=True)
    figure.colorbar(occupancy_artist, ax=axes[1], shrink=0.85, label="Pseudo-track samples")

    axes[2].scatter(
        pseudo_tracks["translation_x"].astype(float),
        pseudo_tracks["translation_y"].astype(float),
        s=22,
        color="lightgray",
        edgecolors="none",
        alpha=0.85,
        label="Monte Carlo translations",
    )
    exclusion_circle = plt.Circle(
        (0.0, 0.0),
        float(summary["null_design"]["exclusion_radius"]),
        color="tab:red",
        fill=False,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="Excluded around Pluto",
    )
    axes[2].add_patch(exclusion_circle)
    axes[2].scatter([0.0], [0.0], color="#ff8c00", s=55, marker="o", label="Nominal Pluto")
    axes[2].scatter([calibrated_x_offset], [0.0], color="#00bcd4", s=75, marker="x", linewidths=2.0, label="Best stack attempt")
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[2].axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_title("Monte Carlo translation offsets")
    axes[2].set_xlabel("Translation dx [pix]")
    axes[2].set_ylabel("Translation dy [pix]")
    axes[2].legend(loc="upper left", frameon=True)

    direct_best = summary["empirical"]["metrics"]["direct_best_of_two"]
    stack_best = summary["empirical"]["metrics"]["stack_best_of_two"]
    figure.suptitle(
        "\n".join(
            [
                f"Pluto trajectory vs Monte Carlo null trajectories: Sector {summary['sector']} / Camera {summary['camera']} / CCD {summary['ccd']}",
                (
                    f"Direct best-of-two p={direct_best['empirical_p_value']:.3f} "
                    f"(label={summary['observed_scores']['direct_best_of_two_label']}), "
                    f"Stack best-of-two p={stack_best['empirical_p_value']:.3f} "
                    f"(label={summary['observed_scores']['stack_best_of_two_label']})"
                ),
            ]
        ),
        fontsize=11,
    )

    figure.savefig(args.outpath, dpi=180)
    plt.close(figure)
    print(f"Wrote {args.outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
