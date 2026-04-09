#!/usr/bin/env python3
"""Make focused heatmap and S/N accumulation plots for a detector-local recovery run."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from kbo_tess.recovery import recover_from_local_files, resolve_track_coordinates  # noqa: E402
from kbo_tess.shiftstack import apply_support_gate, stack_with_support  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a focused detection heatmap and S/N accumulation plot from a local recovery run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Sedna", help="Target designation.")
    parser.add_argument("--sector", type=int, required=True, help="TESS sector.")
    parser.add_argument("--camera", type=int, required=True, help="TESS camera.")
    parser.add_argument("--ccd", type=int, required=True, help="TESS CCD.")
    parser.add_argument("--ephem-csv", type=Path, required=True, help="Normalized ephemeris CSV.")
    parser.add_argument("--cube-npz", type=Path, required=True, help="Local cube bundle NPZ.")
    parser.add_argument("--background-method", choices=("none", "median", "savgol"), default="none", help="Temporal background method for the focused rerun.")
    parser.add_argument("--interpolation", choices=("nearest", "linear"), default="linear", help="Shift interpolation mode.")
    parser.add_argument("--offset-grid-radius", type=float, default=1.0, help="Offset-search half-width in pixels.")
    parser.add_argument("--offset-grid-step", type=float, default=0.5, help="Offset-search step size in pixels.")
    parser.add_argument(
        "--support-fraction-min",
        type=float,
        default=0.0,
        help="If > 0, mask stack pixels with support below this fraction of the maximum support.",
    )
    parser.add_argument(
        "--support-frames-min",
        type=float,
        default=0.0,
        help="If > 0, mask stack pixels with support below this many contributing frames.",
    )
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs", help="Directory for derived plots.")
    return parser


def _prefix(args: argparse.Namespace) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", args.target.lower()).strip("_")
    return f"{slug}_sector{args.sector}_cam{args.camera}_ccd{args.ccd}_{args.background_method}"


def _image_limits(image):
    import numpy as np

    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [5, 99.5])
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    return float(vmin), float(vmax)


def _support_gate_kwargs(args: argparse.Namespace) -> dict[str, float | None]:
    return {
        "min_support_fraction": float(args.support_fraction_min) if float(args.support_fraction_min) > 0 else None,
        "min_support_frames": float(args.support_frames_min) if float(args.support_frames_min) > 0 else None,
    }


def _support_suffix(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if float(args.support_fraction_min) > 0:
        parts.append(f"supportfrac{str(args.support_fraction_min).replace('.', 'p')}")
    if float(args.support_frames_min) > 0:
        parts.append(f"supportn{str(args.support_frames_min).replace('.', 'p')}")
    return ("_" + "_".join(parts)) if parts else ""


def _build_support_aware_stacks(recovery_result: dict[str, object], args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    residual_cube = recovery_result["residual_cube"]
    aligned_ephem = recovery_result["aligned_ephemeris"]
    x, y, _ = resolve_track_coordinates(aligned_ephem)
    summary = recovery_result["summary"]
    reference = (
        float(summary["recovered_reference_pixel"]["x"]),
        float(summary["recovered_reference_pixel"]["y"]),
    )
    nominal = stack_with_support(
        residual_cube,
        x,
        y,
        method="mean",
        interpolation=args.interpolation,
        reference=reference,
        integer_shifts=False,
    )
    best = stack_with_support(
        residual_cube,
        x + float(summary["best_dx"]),
        y + float(summary["best_dy"]),
        method="mean",
        interpolation=args.interpolation,
        reference=reference,
        integer_shifts=False,
    )
    return nominal, best


def plot_heatmap(recovery_result: dict[str, object], output_path: Path, args: argparse.Namespace) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    nominal_support, best_support = _build_support_aware_stacks(recovery_result, args)
    nominal_stack = np.asarray(nominal_support["stack"], dtype=float)
    best_stack = np.asarray(best_support["stack"], dtype=float)
    gate_kwargs = _support_gate_kwargs(args)
    gate_active = any(value is not None for value in gate_kwargs.values())
    if gate_active:
        nominal_gate = apply_support_gate(nominal_stack, nominal_support["support_map"], **gate_kwargs)
        best_gate = apply_support_gate(best_stack, best_support["support_map"], **gate_kwargs)
        nominal_stack = np.asarray(nominal_gate["stack"], dtype=float)
        best_stack = np.asarray(best_gate["stack"], dtype=float)
        nominal_threshold = float(nominal_gate["threshold_frames"])
    else:
        nominal_threshold = 0.0
    summary = recovery_result["summary"]
    ref = summary["recovered_reference_pixel"]
    title_suffix = f"\nSupport gate >= {nominal_threshold:.1f} frames" if gate_active else ""

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    panels = [
        (axes[0], nominal_stack, f"Nominal Stack\nS/N={summary['nominal_score']:.2f}{title_suffix}"),
        (
            axes[1],
            best_stack,
            f"Best Stack\nS/N={summary['best_score']:.2f} at ({summary['best_dx']:.1f},{summary['best_dy']:.1f}){title_suffix}",
        ),
    ]
    for axis, image, title in panels:
        vmin, vmax = _image_limits(image)
        artist = axis.imshow(image, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        axis.scatter([ref["x"]], [ref["y"]], marker="x", color="white", s=70, linewidths=1.5)
        axis.set_title(title)
        axis.set_xlabel("Local x [pix]")
        axis.set_ylabel("Local y [pix]")
        figure.colorbar(artist, ax=axis, shrink=0.85)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def plot_growth(recovery_result: dict[str, object], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    growth_curve = recovery_result["growth_curve"]
    summary = recovery_result["summary"]

    counts = [float(row["n_frames"]) for row in growth_curve]
    nominal = [float(row["nominal_ranking_score"]) for row in growth_curve]
    best = [float(row["best_ranking_score"]) for row in growth_curve]

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(counts, nominal, marker="o", linewidth=1.8, label="Nominal track")
    axis.plot(counts, best, marker="o", linewidth=1.8, label="Best local offset")
    axis.set_title(
        f"{summary['target']} S/N Accumulation ({summary['background_method']})"
    )
    axis.set_xlabel("Cumulative frames")
    axis.set_ylabel(summary["ranking_score_method"])
    axis.grid(alpha=0.3)
    axis.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def main() -> int:
    args = build_parser().parse_args()
    recovery = recover_from_local_files(
        args.ephem_csv,
        args.cube_npz,
        background_method=args.background_method,
        interpolation=args.interpolation,
        offset_grid_radius=args.offset_grid_radius,
        offset_grid_step=args.offset_grid_step,
    )
    prefix = _prefix(args)
    heatmap_path = plot_heatmap(recovery, args.outdir / f"{prefix}{_support_suffix(args)}_heatmap.png", args)
    growth_path = plot_growth(recovery, args.outdir / f"{prefix}_snr_accumulation.png")

    summary = recovery["summary"]
    print("Focused recovery plots:")
    print(f"  target: {summary['target']}")
    print(f"  background_method: {summary['background_method']}")
    print(f"  nominal_score: {summary['nominal_score']:.3f}")
    print(f"  best_score: {summary['best_score']:.3f}")
    print(f"  best_dxdy: ({summary['best_dx']:.2f},{summary['best_dy']:.2f})")
    print(f"  heatmap_png: {heatmap_path}")
    print(f"  snr_accumulation_png: {growth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
