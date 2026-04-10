#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot highly transparent moving 21x21 cutout grids in detector coordinates."
    )
    parser.add_argument("--bundle-npz", type=Path, required=True, help="Path to cube bundle NPZ.")
    parser.add_argument("--out-png", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", default="", help="Optional plot title.")
    parser.add_argument(
        "--grid-alpha",
        type=float,
        default=0.010,
        help="Alpha value for each individual grid line.",
    )
    parser.add_argument(
        "--track-color",
        default="#e67e22",
        help="Color for the target center track.",
    )
    parser.add_argument(
        "--grid-color",
        default="black",
        help="Color for the transparent cutout grid lines.",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=0.22,
        help="Line width for individual grid lines.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Saved PNG DPI.",
    )
    parser.add_argument(
        "--stars-csv",
        type=Path,
        default=None,
        help="Optional catalog-source CSV with detector_x/detector_y/tmag columns.",
    )
    parser.add_argument(
        "--mag-threshold",
        type=float,
        default=17.0,
        help="Plot only catalog sources brighter than this Tmag threshold.",
    )
    parser.add_argument(
        "--star-cmap",
        default="viridis_r",
        help="Colormap used for star-circle edge colors.",
    )
    parser.add_argument(
        "--star-alpha",
        type=float,
        default=0.72,
        help="Alpha for star circles.",
    )
    parser.add_argument(
        "--star-linewidth",
        type=float,
        default=0.95,
        help="Line width for star circles.",
    )
    parser.add_argument(
        "--shade-path-pixels",
        action="store_true",
        help="Shade the detector pixels traversed by the target center.",
    )
    parser.add_argument(
        "--path-pixel-alpha",
        type=float,
        default=0.14,
        help="Alpha for the traversed-path pixel shading.",
    )
    parser.add_argument(
        "--path-pixel-edge-alpha",
        type=float,
        default=0.22,
        help="Alpha for the traversed-path pixel outlines.",
    )
    parser.add_argument(
        "--path-pixel-linewidth",
        type=float,
        default=0.32,
        help="Line width for the traversed-path pixel outlines.",
    )
    return parser.parse_args()


def load_bundle(bundle_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(bundle_path, allow_pickle=True) as bundle:
        corner1 = np.asarray(bundle["corner1"], dtype=float)
        corner2 = np.asarray(bundle["corner2"], dtype=float)
        detector_x = np.asarray(bundle["detector_x"], dtype=float)
        detector_y = np.asarray(bundle["detector_y"], dtype=float)
        metadata = json.loads(str(bundle["metadata_json"]))
    return corner1, corner2, detector_x, detector_y, metadata


def load_catalog_sources(csv_path: Path, mag_threshold: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                tmag = float(row["tmag"])
                detector_x = float(row["detector_x"])
                detector_y = float(row["detector_y"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(tmag) or tmag >= mag_threshold:
                continue
            rows.append(
                {
                    "tmag": tmag,
                    "detector_x": detector_x,
                    "detector_y": detector_y,
                }
            )
    return rows


def mask_radius_from_tmag(tmag: float) -> float:
    if not np.isfinite(tmag):
        return 1.5
    return float(np.clip(1.25 + 0.50 * (15.5 - float(tmag)), 1.25, 4.0))


def alpha_from_tmag(tmag: np.ndarray, threshold: float) -> np.ndarray:
    brightness = np.clip(threshold - tmag, 0.0, None)
    if np.nanmax(brightness) <= 0:
        return np.full_like(brightness, 0.35, dtype=float)
    scaled = brightness / float(np.nanmax(brightness))
    return 0.22 + 0.68 * np.power(scaled, 0.85)


def build_fill_colormap(base_cmap_name: str, mag_threshold: float, mag_min: float, star_alpha: float) -> ListedColormap:
    sample_tmag = np.linspace(mag_min, mag_threshold, 256)
    base_cmap = plt.get_cmap(base_cmap_name)
    norm = Normalize(vmin=mag_min, vmax=mag_threshold)
    colors = base_cmap(norm(sample_tmag))
    fill_alpha = alpha_from_tmag(sample_tmag, mag_threshold) * float(star_alpha) * 0.30
    colors[:, 3] = fill_alpha
    return ListedColormap(colors)


def build_grid_segments(
    corner1: np.ndarray,
    corner2: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    x_edges = np.arange(nx + 1, dtype=float) - 0.5
    y_edges = np.arange(ny + 1, dtype=float) - 0.5
    segments: list[list[tuple[float, float]]] = []
    for c1, c2 in zip(corner1, corner2, strict=True):
        for x_edge in x_edges:
            x = float(c1 + x_edge)
            y0 = float(c2 - 0.5)
            y1 = float(c2 + ny - 0.5)
            segments.append([(x, y0), (x, y1)])
        for y_edge in y_edges:
            y = float(c2 + y_edge)
            x0 = float(c1 - 0.5)
            x1 = float(c1 + nx - 0.5)
            segments.append([(x0, y), (x1, y)])
    return np.asarray(segments, dtype=float)


def unique_path_pixels(detector_x: np.ndarray, detector_y: np.ndarray) -> list[tuple[int, int]]:
    pixels: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x_val, y_val in zip(np.rint(detector_x).astype(int), np.rint(detector_y).astype(int), strict=True):
        pixel = (int(x_val), int(y_val))
        if pixel not in seen:
            seen.add(pixel)
            pixels.append(pixel)
    return pixels


def main() -> None:
    args = parse_args()
    bundle_path = args.bundle_npz.resolve()
    out_path = args.out_png.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corner1, corner2, detector_x, detector_y, metadata = load_bundle(bundle_path)
    nx, ny = metadata["cutout_shape"][1], metadata["cutout_shape"][0]
    segments = build_grid_segments(corner1, corner2, nx=nx, ny=ny)

    fig = plt.figure(figsize=(11.8, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.030], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1]) if args.stars_csv is not None else None

    grid_collection = LineCollection(
        segments,
        colors=args.grid_color,
        linewidths=args.linewidth,
        alpha=args.grid_alpha,
        zorder=1,
    )
    ax.add_collection(grid_collection)

    if args.shade_path_pixels:
        for px, py in unique_path_pixels(detector_x, detector_y):
            ax.add_patch(
                Rectangle(
                    (px - 0.5, py - 0.5),
                    1.0,
                    1.0,
                    facecolor=(0.45, 0.45, 0.45, float(args.path_pixel_alpha)),
                    edgecolor=(0.34, 0.34, 0.34, float(args.path_pixel_edge_alpha)),
                    linewidth=float(args.path_pixel_linewidth),
                    zorder=0.5,
                )
            )

    ax.plot(
        detector_x,
        detector_y,
        color=args.track_color,
        lw=1.8,
        alpha=0.95,
        zorder=3,
    )
    ax.scatter(
        [detector_x[0], detector_x[-1]],
        [detector_y[0], detector_y[-1]],
        c=[args.track_color, args.track_color],
        s=[54, 64],
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )
    ax.annotate(
        "start",
        (detector_x[0], detector_y[0]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10,
        color=args.track_color,
    )
    ax.annotate(
        "end",
        (detector_x[-1], detector_y[-1]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10,
        color=args.track_color,
    )

    n_star_sources = 0
    if args.stars_csv is not None:
        sources = load_catalog_sources(args.stars_csv.resolve(), args.mag_threshold)
        if sources:
            n_star_sources = len(sources)
            tmag = np.asarray([row["tmag"] for row in sources], dtype=float)
            star_x = np.asarray([row["detector_x"] for row in sources], dtype=float)
            star_y = np.asarray([row["detector_y"] for row in sources], dtype=float)
            radii = np.asarray([mask_radius_from_tmag(value) for value in tmag], dtype=float)
            norm = Normalize(vmin=float(np.min(tmag)), vmax=args.mag_threshold)
            cmap = plt.get_cmap(args.star_cmap)
            edgecolors = cmap(norm(tmag))
            alphas = alpha_from_tmag(tmag, args.mag_threshold) * float(args.star_alpha)
            for x, y, radius, color, alpha in zip(star_x, star_y, radii, edgecolors, alphas, strict=True):
                ax.add_patch(
                    Circle(
                        (float(x), float(y)),
                        radius=float(radius),
                        facecolor=(float(color[0]), float(color[1]), float(color[2]), float(alpha * 0.30)),
                        edgecolor=(float(color[0]), float(color[1]), float(color[2]), float(alpha)),
                        linewidth=args.star_linewidth,
                        zorder=2,
                    )
                )

            fill_cmap = build_fill_colormap(
                base_cmap_name=args.star_cmap,
                mag_threshold=float(args.mag_threshold),
                mag_min=float(np.min(tmag)),
                star_alpha=float(args.star_alpha),
            )
            sm = ScalarMappable(norm=norm, cmap=fill_cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize=9)
            cbar.ax.xaxis.set_label_position("bottom")
            cbar.ax.set_xlabel("Tmag", fontsize=11, labelpad=8)

    xmin = float(np.nanmin(corner1) - 1.0)
    xmax = float(np.nanmax(corner1) + nx)
    ymin = float(np.nanmin(corner2) - 1.0)
    ymax = float(np.nanmax(corner2) + ny)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Detector X (column)", fontsize=12)
    ax.set_ylabel("Detector Y (row)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    if args.title:
        ax.set_title(args.title, fontsize=13)
    elif cax is not None:
        cax.set_anchor("C")

    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {out_path}")
    print(f"Frames plotted: {len(corner1)}")
    print(f"Cutout shape: {ny}x{nx}")
    if args.stars_csv is not None:
        print(f"Stars plotted: {n_star_sources}")
    print(
        "Detector bounds: "
        f"x=({xmin:.1f}, {xmax:.1f}), "
        f"y=({ymin:.1f}, {ymax:.1f})"
    )


if __name__ == "__main__":
    main()
