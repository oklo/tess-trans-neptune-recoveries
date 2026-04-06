#!/usr/bin/env python3
"""Run a bounded empirical-significance study on the frozen Pluto bundle."""

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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kbo_tess.background import subtract_background  # noqa: E402
from kbo_tess.recovery import filter_cube_bundle, load_cube_npz  # noqa: E402
from kbo_tess.significance import (  # noqa: E402
    build_empirical_summary,
    build_split_half_indices,
    compute_direct_track_statistic,
    empirical_p_value,
    evaluate_track_pair_statistics,
    evaluate_pseudo_track_ensemble,
    generate_translated_pseudo_tracks,
    materialize_cube_bundle_to_npz,
    plot_empirical_null_distribution,
    plot_scrubbed_cube_diagnostic,
    plot_split_half_reproducibility,
    scrub_track_from_cube_bundle,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a matched-null empirical-significance study on the frozen Pluto bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Pluto", help="Target designation for reporting.")
    parser.add_argument("--sector", type=int, default=92, help="TESS sector.")
    parser.add_argument("--camera", type=int, default=2, help="TESS camera.")
    parser.add_argument("--ccd", type=int, default=1, help="TESS CCD.")
    parser.add_argument(
        "--cube-npz",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_calibration" / "primary" / "pluto_sector92_cam2_ccd1_cube_bundle.npz",
        help="Frozen detector-local Pluto cube bundle.",
    )
    parser.add_argument(
        "--x-calibration-summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_x_calibration" / "pluto_sector92_cam2_ccd1_x_calibration_summary.json",
        help="Pluto x-calibration summary JSON used to read the calibrated x offset.",
    )
    parser.add_argument(
        "--background-method",
        choices=("none", "median", "savgol"),
        default="none",
        help="Temporal background handling applied consistently to observed and null analyses.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "linear"),
        default="linear",
        help="Shift interpolation mode for the stack statistic.",
    )
    parser.add_argument(
        "--n-nulls",
        type=int,
        default=256,
        help="Number of translated pseudo-track null realizations.",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=5.0,
        help="Minimum edge margin in pixels for pseudo-tracks so the aperture annulus stays inside the cutout.",
    )
    parser.add_argument(
        "--exclusion-radius",
        type=float,
        default=3.0,
        help="Minimum translation radius away from the real Pluto path for pseudo-tracks.",
    )
    parser.add_argument(
        "--scrub-mask-radius",
        type=float,
        default=2.0,
        help="Radius in pixels used to scrub the Pluto path from the null cube.",
    )
    parser.add_argument(
        "--scrub-fill-inner-radius",
        type=float,
        default=3.0,
        help="Inner radius of the local-background annulus for scrubbing.",
    )
    parser.add_argument(
        "--scrub-fill-outer-radius",
        type=float,
        default=5.0,
        help="Outer radius of the local-background annulus for scrubbing.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("even_odd", "first_second"),
        default="even_odd",
        help="Cadence split used for reproducibility checks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260405,
        help="Deterministic RNG seed for scrubbing and pseudo-track generation.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_empirical_significance",
        help="Directory for derived significance products.",
    )
    return parser


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_local_track(cube_bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    for x_key, y_key, label in (
        ("local_x", "local_y", "bundle_local"),
        ("bundle_local_x", "bundle_local_y", "bundle_local"),
    ):
        if x_key in cube_bundle and y_key in cube_bundle:
            return (
                np.asarray(cube_bundle[x_key], dtype=float),
                np.asarray(cube_bundle[y_key], dtype=float),
                label,
            )
    raise KeyError("Cube bundle does not contain local_x/local_y track coordinates.")


def _summarize_split(
    observed_scores: dict[str, Any],
    empirical_summary: dict[str, Any],
) -> dict[str, Any]:
    metrics = empirical_summary["metrics"]
    return {
        "direct_best_of_two": {
            "best_label": str(observed_scores["direct"]["best_of_two"]["label"]),
            "observed_score": float(observed_scores["direct"]["best_of_two"]["score"]),
            "null_median": float(metrics["direct_best_of_two"]["null_median"]),
            "empirical_p_value": float(metrics["direct_best_of_two"]["empirical_p_value"]),
        },
        "stack_best_of_two": {
            "best_label": str(observed_scores["stack"]["best_of_two"]["label"]),
            "observed_score": float(observed_scores["stack"]["best_of_two"]["score"]),
            "null_median": float(metrics["stack_best_of_two"]["null_median"]),
            "empirical_p_value": float(metrics["stack_best_of_two"]["empirical_p_value"]),
        },
    }


def _classify_result(summary: dict[str, Any], split_summary: dict[str, Any]) -> str:
    direct_best_p = float(summary["empirical"]["metrics"]["direct_best_of_two"]["empirical_p_value"])
    stack_best_p = float(summary["empirical"]["metrics"]["stack_best_of_two"]["empirical_p_value"])
    direct_split_directional = all(
        float(split["direct_best_of_two"]["observed_score"]) > float(split["direct_best_of_two"]["null_median"])
        for split in split_summary.values()
    )
    stack_split_supportive = all(
        float(split["stack_best_of_two"]["observed_score"]) > float(split["stack_best_of_two"]["null_median"])
        for split in split_summary.values()
    )
    if direct_best_p <= 0.01 and direct_split_directional and stack_best_p <= 0.10 and stack_split_supportive:
        return "A"
    if direct_best_p <= 0.05 and direct_split_directional:
        return "B"
    if direct_best_p <= 0.20 or stack_best_p <= 0.20:
        return "C"
    return "D"


def main() -> int:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_slug(args.target)}_sector{args.sector}_cam{args.camera}_ccd{args.ccd}"

    xcal_summary = _load_json(args.x_calibration_summary_json)
    calibrated_x_offset = float(xcal_summary["full_best_x_offset"])
    frozen_bundle_path = Path(str(xcal_summary["frozen_bundle_path"]))
    if frozen_bundle_path.resolve() != args.cube_npz.resolve():
        raise ValueError(
            "The requested cube bundle does not match the x-calibration frozen bundle. "
            f"xcal={frozen_bundle_path}, requested={args.cube_npz}"
        )

    cube_bundle = filter_cube_bundle(load_cube_npz(args.cube_npz))
    nominal_x, nominal_y, track_coordinate_system = _extract_local_track(cube_bundle)
    reference = (float(nominal_x[0]), float(nominal_y[0]))

    _, analysis_cube = subtract_background(np.asarray(cube_bundle["cube"], dtype=float), method=args.background_method)
    analysis_bundle = dict(cube_bundle)
    analysis_bundle["cube"] = np.asarray(analysis_cube, dtype=float)
    analysis_bundle["metadata"] = dict(cube_bundle.get("metadata", {}))
    analysis_bundle["metadata"]["significance_background_method"] = args.background_method

    scrubbed_result = scrub_track_from_cube_bundle(
        analysis_bundle,
        nominal_x + calibrated_x_offset,
        nominal_y,
        mask_radius=args.scrub_mask_radius,
        fill_inner_radius=args.scrub_fill_inner_radius,
        fill_outer_radius=args.scrub_fill_outer_radius,
        rng_seed=args.seed,
    )
    scrubbed_bundle = scrubbed_result["bundle"]
    scrubbed_cube = np.asarray(scrubbed_bundle["cube"], dtype=float)

    pseudo_tracks = generate_translated_pseudo_tracks(
        nominal_x,
        nominal_y,
        calibrated_x_offset=calibrated_x_offset,
        shape=analysis_cube.shape[1:],
        n_realizations=args.n_nulls,
        edge_margin=args.edge_margin,
        exclusion_radius=args.exclusion_radius,
        rng_seed=args.seed,
    )

    observed_scores = evaluate_track_pair_statistics(
        analysis_cube,
        nominal_x,
        nominal_y,
        calibrated_x_offset=calibrated_x_offset,
        reference=reference,
        interpolation=args.interpolation,
    )
    null_table = evaluate_pseudo_track_ensemble(
        scrubbed_cube,
        nominal_x,
        nominal_y,
        calibrated_x_offset=calibrated_x_offset,
        translations=pseudo_tracks,
        interpolation=args.interpolation,
    )
    empirical_summary = build_empirical_summary(observed_scores, null_table)

    split_indices = build_split_half_indices(len(analysis_cube), mode=args.split_mode)
    split_summary: dict[str, Any] = {}
    for split_label, indices in split_indices.items():
        subset_observed = evaluate_track_pair_statistics(
            analysis_cube[indices],
            nominal_x[indices],
            nominal_y[indices],
            calibrated_x_offset=calibrated_x_offset,
            reference=(float(nominal_x[indices][0]), float(nominal_y[indices][0])),
            interpolation=args.interpolation,
        )
        subset_null = evaluate_pseudo_track_ensemble(
            scrubbed_cube[indices],
            nominal_x[indices],
            nominal_y[indices],
            calibrated_x_offset=calibrated_x_offset,
            translations=pseudo_tracks,
            interpolation=args.interpolation,
        )
        subset_empirical = build_empirical_summary(subset_observed, subset_null)
        split_summary[split_label] = _summarize_split(subset_observed, subset_empirical)

    significance_summary = {
        "target": args.target,
        "sector": int(args.sector),
        "camera": int(args.camera),
        "ccd": int(args.ccd),
        "frozen_bundle_path": str(args.cube_npz.resolve()),
        "track_coordinate_system": track_coordinate_system,
        "background_method": args.background_method,
        "interpolation": args.interpolation,
        "calibrated_x_offset": calibrated_x_offset,
        "direct_statistic": {
            "name": "mean_aperture_snr",
            "formula": "mean of per-frame aperture_snr along the fixed local track",
        },
        "stack_statistic": {
            "name": "aperture_snr",
            "formula": "aperture_snr of the aligned stack at the fixed nominal reference pixel",
        },
        "null_design": {
            "pseudo_track_family": "constant translations of the Pluto local track through the same cadences",
            "matched_optimization_rule": "nominal, calibrated-x, and best-of-two are all evaluated for observed and null tracks",
            "n_null_realizations": int(args.n_nulls),
            "edge_margin": float(args.edge_margin),
            "exclusion_radius": float(args.exclusion_radius),
            "scrubbed_cube_used": True,
        },
        "scrub_summary": dict(scrubbed_result["summary"]),
        "observed_scores": {
            "direct_nominal": float(observed_scores["direct"]["nominal"]["score"]),
            "direct_calibrated_x": float(observed_scores["direct"]["calibrated_x"]["score"]),
            "direct_best_of_two": float(observed_scores["direct"]["best_of_two"]["score"]),
            "direct_best_of_two_label": str(observed_scores["direct"]["best_of_two"]["label"]),
            "stack_nominal": float(observed_scores["stack"]["nominal"]["score"]),
            "stack_calibrated_x": float(observed_scores["stack"]["calibrated_x"]["score"]),
            "stack_best_of_two": float(observed_scores["stack"]["best_of_two"]["score"]),
            "stack_best_of_two_label": str(observed_scores["stack"]["best_of_two"]["label"]),
        },
        "empirical": empirical_summary,
        "split_mode": args.split_mode,
        "split_summary": split_summary,
        "split_flags": {
            "direct_directionally_reproducible": bool(
                all(
                    float(split["direct_best_of_two"]["observed_score"])
                    > float(split["direct_best_of_two"]["null_median"])
                    for split in split_summary.values()
                )
            ),
            "stack_directionally_reproducible": bool(
                all(
                    float(split["stack_best_of_two"]["observed_score"])
                    > float(split["stack_best_of_two"]["null_median"])
                    for split in split_summary.values()
                )
            ),
        },
    }
    significance_summary["classification"] = _classify_result(significance_summary, split_summary)

    summary_json = args.outdir / f"{prefix}_empirical_significance_summary.json"
    null_csv = args.outdir / f"{prefix}_null_statistics.csv"
    empirical_csv = args.outdir / f"{prefix}_empirical_summary.csv"
    direct_png = args.outdir / f"{prefix}_direct_null_distribution.png"
    stack_png = args.outdir / f"{prefix}_stack_null_distribution.png"
    split_png = args.outdir / f"{prefix}_split_half_reproducibility.png"
    scrub_png = args.outdir / f"{prefix}_scrubbed_cube_diagnostic.png"
    scrub_json = args.outdir / f"{prefix}_scrub_summary.json"
    scrub_npz = args.outdir / f"{prefix}_scrubbed_cube_bundle.npz"
    pseudo_csv = args.outdir / f"{prefix}_pseudo_tracks.csv"

    write_json(summary_json, significance_summary)
    write_json(scrub_json, dict(scrubbed_result["summary"]))
    materialize_cube_bundle_to_npz(scrubbed_bundle, scrub_npz)
    pseudo_tracks.to_csv(pseudo_csv, index=False)
    null_table.to_csv(null_csv, index=False)
    pd.DataFrame(empirical_summary["rows"]).to_csv(empirical_csv, index=False)
    plot_scrubbed_cube_diagnostic(
        analysis_cube,
        scrubbed_cube,
        nominal_x + calibrated_x_offset,
        nominal_y,
        scrub_png,
    )
    plot_empirical_null_distribution(
        null_table,
        observed_scores,
        metric_prefix="direct",
        output_path=direct_png,
        title="Direct moving-aperture significance",
    )
    plot_empirical_null_distribution(
        null_table,
        observed_scores,
        metric_prefix="stack",
        output_path=stack_png,
        title="Stack-based significance",
    )
    plot_split_half_reproducibility(split_summary, split_png)

    print("Pluto empirical-significance summary:")
    print(f"  frozen_bundle: {args.cube_npz}")
    print(f"  calibrated_x_offset: {calibrated_x_offset:.3f}")
    print(f"  n_nulls: {args.n_nulls}")
    print(
        "  direct p-values: "
        f"nominal={empirical_summary['metrics']['direct_nominal']['empirical_p_value']:.4f}, "
        f"calibrated={empirical_summary['metrics']['direct_calibrated_x']['empirical_p_value']:.4f}, "
        f"best_of_two={empirical_summary['metrics']['direct_best_of_two']['empirical_p_value']:.4f}"
    )
    print(
        "  stack p-values: "
        f"nominal={empirical_summary['metrics']['stack_nominal']['empirical_p_value']:.4f}, "
        f"calibrated={empirical_summary['metrics']['stack_calibrated_x']['empirical_p_value']:.4f}, "
        f"best_of_two={empirical_summary['metrics']['stack_best_of_two']['empirical_p_value']:.4f}"
    )
    print(f"  classification: {significance_summary['classification']}")
    print(f"  summary_json: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
