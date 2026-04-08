#!/usr/bin/env python3
"""Plot a stacked-flux histogram for Pluto versus matched Monte Carlo trajectories."""

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

from kbo_tess.photometry import aperture_photometry_at_position  # noqa: E402
from kbo_tess.recovery import filter_cube_bundle, load_cube_npz  # noqa: E402
from kbo_tess.stack_audit import build_stack_variant  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the Pluto stacked-flux distribution against matched Monte Carlo null tracks.",
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
        "--null-cube-npz",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "pluto_empirical_significance"
        / "pluto_sector92_cam2_ccd1_scrubbed_cube_bundle.npz",
        help="Source-scrubbed cube bundle used for the null ensemble.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "linear"),
        default="linear",
        help="Shift interpolation mode, kept consistent with the significance study.",
    )
    parser.add_argument(
        "--outpath",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "pluto_empirical_significance"
        / "pluto_sector92_cam2_ccd1_stack_flux_histogram.png",
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


def _stack_net_flux(
    cube: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    reference: tuple[float, float],
    interpolation: str,
) -> dict[str, float]:
    variant = build_stack_variant(
        cube,
        x,
        y,
        reference=reference,
        interpolation=interpolation,
        ranking_position=reference,
        ranking_method="aperture_snr",
    )
    phot = aperture_photometry_at_position(np.asarray(variant["stack"], dtype=float), reference)
    return {
        "net_flux": float(phot["net_flux"]),
        "aperture_snr": float(phot["aperture_snr"]),
        "support_at_reference": float(
            variant["support_map"][int(round(reference[1])), int(round(reference[0]))]
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    summary = _load_json(args.summary_json)
    calibrated_x_offset = float(summary["calibrated_x_offset"])
    pseudo_tracks = pd.read_csv(args.pseudo_tracks_csv)

    observed_bundle = filter_cube_bundle(load_cube_npz(args.cube_npz))
    null_bundle = filter_cube_bundle(load_cube_npz(args.null_cube_npz))
    observed_cube = np.asarray(observed_bundle["cube"], dtype=float)
    null_cube = np.asarray(null_bundle["cube"], dtype=float)
    nominal_x, nominal_y = _extract_local_track(observed_bundle)
    reference = (float(nominal_x[0]), float(nominal_y[0]))

    nominal_flux = _stack_net_flux(
        observed_cube,
        nominal_x,
        nominal_y,
        reference=reference,
        interpolation=args.interpolation,
    )
    calibrated_flux = _stack_net_flux(
        observed_cube,
        nominal_x + calibrated_x_offset,
        nominal_y,
        reference=reference,
        interpolation=args.interpolation,
    )

    rows: list[dict[str, float | str]] = []
    for row in pseudo_tracks.itertuples(index=False):
        translated_x = nominal_x + float(row.translation_x)
        translated_y = nominal_y + float(row.translation_y)
        translated_reference = (float(translated_x[0]), float(translated_y[0]))
        nominal_null = _stack_net_flux(
            null_cube,
            translated_x,
            translated_y,
            reference=translated_reference,
            interpolation=args.interpolation,
        )
        calibrated_null = _stack_net_flux(
            null_cube,
            translated_x + calibrated_x_offset,
            translated_y,
            reference=translated_reference,
            interpolation=args.interpolation,
        )
        best_label = "calibrated_x" if calibrated_null["net_flux"] > nominal_null["net_flux"] else "nominal"
        best_flux = calibrated_null["net_flux"] if best_label == "calibrated_x" else nominal_null["net_flux"]
        rows.append(
            {
                "realization": float(row.realization),
                "translation_x": float(row.translation_x),
                "translation_y": float(row.translation_y),
                "translation_radius": float(row.translation_radius),
                "nominal_flux": float(nominal_null["net_flux"]),
                "calibrated_flux": float(calibrated_null["net_flux"]),
                "best_of_two_flux": float(best_flux),
                "best_of_two_label": str(best_label),
            }
        )
    null_flux_table = pd.DataFrame(rows)

    observed_best_label = "calibrated_x" if calibrated_flux["net_flux"] > nominal_flux["net_flux"] else "nominal"
    observed_best_flux = calibrated_flux["net_flux"] if observed_best_label == "calibrated_x" else nominal_flux["net_flux"]
    empirical_p = float((1 + (null_flux_table["best_of_two_flux"].astype(float) >= observed_best_flux).sum()) / (len(null_flux_table) + 1))

    args.outpath.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    values = null_flux_table["best_of_two_flux"].astype(float)
    axis.hist(values, bins=min(28, max(12, len(values) // 10)), color="lightgray", edgecolor="gray", alpha=0.95)
    axis.axvline(
        nominal_flux["net_flux"],
        color="#ff8c00",
        linestyle="--",
        linewidth=2.0,
        label=f"Nominal Pluto: {nominal_flux['net_flux']:.3f}",
    )
    axis.axvline(
        calibrated_flux["net_flux"],
        color="#00bcd4",
        linestyle="-",
        linewidth=2.2,
        label=f"Best stack attempt: {calibrated_flux['net_flux']:.3f}",
    )
    axis.axvline(
        float(values.median()),
        color="black",
        linestyle=":",
        linewidth=1.4,
        label=f"Null median: {float(values.median()):.3f}",
    )
    axis.set_title(
        "\n".join(
            [
                "Pluto stacked-flux vs matched Monte Carlo trajectories",
                (
                    f"Histogram uses best-of-two null fluxes from the scrubbed cube; "
                    f"observed best label = {observed_best_label}, empirical p = {empirical_p:.3f}"
                ),
            ]
        )
    )
    axis.set_xlabel("Stacked aperture net flux")
    axis.set_ylabel("Number of trajectories")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", frameon=True)

    figure.savefig(args.outpath, dpi=180)
    plt.close(figure)
    print(f"Wrote {args.outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
