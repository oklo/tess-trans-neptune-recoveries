#!/usr/bin/env python3
"""Run the nominal-path workflow on a batch of named outer Solar System bodies.

This script reuses the generic wrapper, adds lightweight apparent-brightness
estimates from the selected detector-local ephemeris, computes comparison
metrics from the final summed local-window curve, and writes one survey-wide
summary figure/table.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TARGETS = (
    "Eris",
    "Makemake",
    "Gonggong",
    "Orcus",
    "Quaoar",
    "Chiminigagua",
    "Aya",
    "G!kun||'homdima",
    "Varda",
    "Ixion",
    "Máni",
    "Ritona",
    "Achlys",
    "Varuna",
    "Dziewanna",
    "Salacia",
    "Chaos",
)

DEFAULT_REUSE_SUMMARIES = (
    PROJECT_ROOT / "outputs" / "nominal_path_diagnostics" / "sedna" / "sedna_nominal_path_workflow_summary.json",
    PROJECT_ROOT / "outputs" / "nominal_path_diagnostics" / "haumea" / "haumea_nominal_path_workflow_summary.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bright-KBO nominal-path survey and build a compact comparison summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="Targets to run through the workflow. Existing reuse summaries are added separately.",
    )
    parser.add_argument(
        "--reuse-summary",
        type=Path,
        action="append",
        default=list(DEFAULT_REUSE_SUMMARIES),
        help="Existing workflow summary JSONs to include without rerunning.",
    )
    parser.add_argument(
        "--workflow-outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "nominal_path_diagnostics",
        help="Shared root used by the generic workflow wrapper.",
    )
    parser.add_argument(
        "--survey-outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "nominal_path_diagnostics" / "surveys" / "bright_kbos_2026_04_11",
        help="Output directory for the survey-wide summary products.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=1,
        help="Maximum viable detector opportunities to run per target.",
    )
    parser.add_argument(
        "--star-mag-threshold",
        type=float,
        default=17.0,
        help="Bright-star threshold used in the overlay plots.",
    )
    parser.add_argument(
        "--tmag-offset",
        type=float,
        default=0.3,
        help="Best-effort TESS-band correction relative to detector-local V magnitude: T ~= V - offset.",
    )
    parser.add_argument(
        "--cleanup-secondary",
        action="store_true",
        help="Delete clearly secondary outputs after successful runs (preview PNGs and bulky catalog tables).",
    )
    return parser


def slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text).strip()).strip("_")


def parse_key_value_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def coerce_number(value: str | None) -> float | int | None:
    if value is None or value == "":
        return None
    lowered = value.lower()
    if lowered in {"nan", "none"}:
        return None
    try:
        integer = int(value)
        if str(integer) == value:
            return integer
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return None


def run_wrapper(target: str, workflow_outdir: Path, max_candidates: int, star_mag_threshold: float) -> Path:
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_target_nominal_path_diagnostics.py"),
            "--target",
            str(target),
            "--outdir",
            str(workflow_outdir),
            "--max-candidates",
            str(int(max_candidates)),
            "--star-mag-threshold",
            str(float(star_mag_threshold)),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    return workflow_outdir / slug(target) / f"{slug(target)}_nominal_path_workflow_summary.json"


def cleanup_secondary_outputs(summary_json: Path) -> list[str]:
    removed: list[str] = []
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    for run in payload.get("runs", []):
        bundle_metadata = Path(run["bundle_metadata_json"])
        preview_png = bundle_metadata.with_name(bundle_metadata.name.replace("_metadata.json", ".png"))
        for path in (
            preview_png,
            Path(run["bright_crosser_csv"]).with_name(Path(run["bright_crosser_csv"]).name.replace("_bright.csv", "_full.csv")),
            Path(run["bright_crosser_csv"]).with_name(Path(run["bright_crosser_csv"]).name.replace("_bright.csv", "_deduped.csv")),
        ):
            if path.exists():
                path.unlink()
                removed.append(str(path))
    return removed


def summed_curve_metrics(local_csv: Path, local_summary_txt: Path) -> dict[str, float]:
    df = pd.read_csv(local_csv)
    summary = parse_key_value_file(local_summary_txt)
    half_window_hours = float(summary["half_window_hours"])
    bin_width_hours = float(summary["sum_bin_hours"])
    average_residence_count = float(summary["average_residence_count"])
    median_cadence_hours = float(summary["median_cadence_hours"])
    residence_hours = average_residence_count * median_cadence_hours

    x_grid = np.arange(-half_window_hours, half_window_hours + 0.5 * bin_width_hours, bin_width_hours)
    values = df["plot_flux"].to_numpy(dtype=float)
    rel = df["relative_time_hours"].to_numpy(dtype=float)
    sum_flux = np.full_like(x_grid, np.nan, dtype=float)
    for i, x0 in enumerate(x_grid):
        lo = x0 - 0.5 * bin_width_hours
        hi = x0 + 0.5 * bin_width_hours
        mask = (rel >= lo) & (rel < hi) & np.isfinite(values)
        if np.any(mask):
            sum_flux[i] = float(np.sum(values[mask]))

    signal_half = float(np.clip(0.5 * residence_hours, 1.0, 6.0))
    signal_mask = np.isfinite(sum_flux) & (np.abs(x_grid) <= signal_half)
    if not np.any(signal_mask):
        signal_mask = np.isfinite(sum_flux) & (np.abs(x_grid) <= max(signal_half, bin_width_hours))
    peak_idx_local = int(np.nanargmax(sum_flux[signal_mask]))
    signal_x = x_grid[signal_mask]
    signal_y = sum_flux[signal_mask]
    peak_sum = float(signal_y[peak_idx_local])
    peak_time_hours = float(signal_x[peak_idx_local])

    baseline_mask = np.isfinite(sum_flux) & (np.abs(x_grid) >= max(3.0 * signal_half, 0.35 * half_window_hours))
    if not np.any(baseline_mask):
        baseline_mask = np.isfinite(sum_flux) & (np.abs(x_grid) >= max(2.0 * signal_half, 0.25 * half_window_hours))
    baseline = sum_flux[baseline_mask]
    baseline_median = float(np.nanmedian(baseline)) if baseline.size else 0.0
    baseline_mad = float(np.nanmedian(np.abs(baseline - baseline_median))) if baseline.size else float("nan")
    baseline_sigma = 1.4826 * baseline_mad if np.isfinite(baseline_mad) else float("nan")
    if not np.isfinite(baseline_sigma) or baseline_sigma <= 0.0:
        baseline_sigma = float(np.nanstd(baseline)) if baseline.size else float("nan")
    contrast_snr = float((peak_sum - baseline_median) / baseline_sigma) if np.isfinite(baseline_sigma) and baseline_sigma > 0 else float("nan")

    zero_idx = int(np.nanargmin(np.abs(x_grid)))
    zero_sum = float(sum_flux[zero_idx]) if np.isfinite(sum_flux[zero_idx]) else float("nan")

    return {
        "signal_half_hours": signal_half,
        "residence_hours": residence_hours,
        "peak_sum": peak_sum,
        "peak_time_hours": peak_time_hours,
        "zero_sum": zero_sum,
        "baseline_median": baseline_median,
        "baseline_sigma": baseline_sigma,
        "contrast_snr": contrast_snr,
    }


def collect_run_metrics(summary_json: Path, *, source_label: str, tmag_offset: float) -> list[dict[str, object]]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []
    for run in payload.get("runs", []):
        inventory_csv = Path(run["inventory_csv"])
        inv = pd.read_csv(inventory_csv)
        path_summary = parse_key_value_file(Path(run["path_summary_txt"]))
        curve = summed_curve_metrics(Path(run["local_windows_csv"]), Path(run["local_windows_summary_txt"]))

        distinct_path_pixels = int(coerce_number(path_summary.get("distinct_path_pixels")) or 0)
        pixels_no_centered = int(coerce_number(path_summary.get("pixels_with_no_centered_kept_cadences")) or 0)
        cadences_total = int(coerce_number(path_summary.get("cadences_total")) or 0)
        cadences_removed = int(coerce_number(path_summary.get("cadences_removed")) or 0)
        cadence_keep_fraction = float((cadences_total - cadences_removed) / cadences_total) if cadences_total else float("nan")
        path_keep_fraction = float((distinct_path_pixels - pixels_no_centered) / distinct_path_pixels) if distinct_path_pixels else float("nan")

        vmag_median = float(inv["vmag"].median()) if "vmag" in inv.columns else float("nan")
        vmag_min = float(inv["vmag"].min()) if "vmag" in inv.columns else float("nan")
        vmag_max = float(inv["vmag"].max()) if "vmag" in inv.columns else float("nan")
        tmag_est_median = float(vmag_median - tmag_offset) if np.isfinite(vmag_median) else float("nan")
        tmag_est_min = float(vmag_min - tmag_offset) if np.isfinite(vmag_min) else float("nan")
        tmag_est_max = float(vmag_max - tmag_offset) if np.isfinite(vmag_max) else float("nan")

        results.append(
            {
                "target": str(run["target"]),
                "target_slug": slug(str(run["target"])),
                "source": source_label,
                "sector": int(run["sector"]),
                "camera": int(run["camera"]),
                "ccd": int(run["ccd"]),
                "duration_days": float(run["duration_days"]),
                "ephem_rows": int(run["ephem_rows"]),
                "inventory_csv": str(inventory_csv),
                "overlay_png": str(run["cutout_grid_overlay_png"]),
                "path_png": str(run["path_timeseries_png"]),
                "local_png": str(run["local_windows_png"]),
                "workflow_summary_json": str(summary_json),
                "vmag_median": vmag_median,
                "vmag_min": vmag_min,
                "vmag_max": vmag_max,
                "tmag_est_median": tmag_est_median,
                "tmag_est_min": tmag_est_min,
                "tmag_est_max": tmag_est_max,
                "cadences_total": cadences_total,
                "cadences_removed": cadences_removed,
                "cadence_keep_fraction": cadence_keep_fraction,
                "distinct_path_pixels": distinct_path_pixels,
                "pixels_no_centered_kept": pixels_no_centered,
                "path_keep_fraction": path_keep_fraction,
                "peak_sum": curve["peak_sum"],
                "peak_time_hours": curve["peak_time_hours"],
                "zero_sum": curve["zero_sum"],
                "baseline_median": curve["baseline_median"],
                "baseline_sigma": curve["baseline_sigma"],
                "contrast_snr": curve["contrast_snr"],
                "signal_half_hours": curve["signal_half_hours"],
                "residence_hours": curve["residence_hours"],
            }
        )
    return results


def build_summary_figure(df: pd.DataFrame, out_png: Path) -> None:
    success = df[df["status"] == "success"].copy()
    if success.empty:
        raise ValueError("No successful runs available for the summary figure.")

    success = success.sort_values(["tmag_est_median", "contrast_snr"], ascending=[True, False]).reset_index(drop=True)

    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.18)

    ax0 = fig.add_subplot(gs[0, 0])
    sizes = 40.0 + 16.0 * np.sqrt(np.clip(success["duration_days"].to_numpy(dtype=float), 0.0, None))
    sc = ax0.scatter(
        success["tmag_est_median"],
        success["contrast_snr"],
        s=sizes,
        c=success["path_keep_fraction"],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.88,
        edgecolors="black",
        linewidths=0.6,
    )
    for i, row in success.iterrows():
        dx = -0.06 if i % 2 else 0.06
        ax0.annotate(
            f"{row['target']} S{int(row['sector'])}",
            (row["tmag_est_median"], row["contrast_snr"]),
            xytext=(8 if dx >= 0 else -8, 8 + 4 * (i % 3)),
            textcoords="offset points",
            fontsize=8,
            ha="left" if dx >= 0 else "right",
            va="bottom",
        )
    ax0.set_xlabel("Estimated Tmag During Best Opportunity", fontsize=13)
    ax0.set_ylabel("Final Summed-Curve Contrast S/N", fontsize=13)
    ax0.invert_xaxis()
    ax0.grid(alpha=0.25, linewidth=0.6)
    cbar = fig.colorbar(sc, ax=ax0, fraction=0.045, pad=0.03)
    cbar.set_label("Centered Path-Pixel Keep Fraction", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    ax0.set_title("Brightness vs Final Nominal-Path Recovery", fontsize=15)

    ranked = success.sort_values("contrast_snr", ascending=True).reset_index(drop=True)
    ax1 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(ranked))
    ax1.barh(y + 0.18, ranked["cadence_keep_fraction"], height=0.32, color="#6baed6", alpha=0.75, label="Cadence keep fraction")
    ax1.barh(y - 0.18, ranked["path_keep_fraction"], height=0.32, color="#fd8d3c", alpha=0.80, label="Centered path-pixel keep fraction")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xlabel("Fraction Retained After Automasking", fontsize=13)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{row.target} S{int(row.sector)}" for row in ranked.itertuples()], fontsize=10)
    ax1.tick_params(axis="x", labelsize=10)
    ax1.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax1.legend(loc="lower right", fontsize=10, frameon=False)
    ax1.set_title("Mask Retention Context", fontsize=15)

    fig.suptitle("Bright KBO Nominal-Path Survey", fontsize=17, y=0.98)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()

    workflow_outdir = Path(args.workflow_outdir)
    survey_outdir = Path(args.survey_outdir)
    survey_outdir.mkdir(parents=True, exist_ok=True)

    statuses: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    removed_files: list[str] = []

    for summary_path in args.reuse_summary:
        summary_path = Path(summary_path)
        if not summary_path.exists():
            statuses.append({"target": summary_path.stem, "status": "missing_reuse_summary", "reason": str(summary_path)})
            continue
        for record in collect_run_metrics(summary_path, source_label="existing", tmag_offset=float(args.tmag_offset)):
            record["status"] = "success"
            metrics.append(record)
            statuses.append({"target": record["target"], "status": "success", "source": "existing", "workflow_summary_json": str(summary_path)})

    reused_targets = {str(item["target"]) for item in metrics}
    for target in args.targets:
        if str(target) in reused_targets:
            continue
        try:
            summary_json = run_wrapper(
                target=str(target),
                workflow_outdir=workflow_outdir,
                max_candidates=int(args.max_candidates),
                star_mag_threshold=float(args.star_mag_threshold),
            )
            if args.cleanup_secondary:
                removed_files.extend(cleanup_secondary_outputs(summary_json))
            target_metrics = collect_run_metrics(summary_json, source_label="new", tmag_offset=float(args.tmag_offset))
            for record in target_metrics:
                record["status"] = "success"
                metrics.append(record)
            statuses.append({"target": str(target), "status": "success", "source": "new", "workflow_summary_json": str(summary_json)})
        except subprocess.CalledProcessError as exc:
            statuses.append({"target": str(target), "status": "failed", "reason": f"wrapper exit code {exc.returncode}"})
        except Exception as exc:  # pragma: no cover - defensive top-level guard
            statuses.append({"target": str(target), "status": "failed", "reason": str(exc)})

    metrics_df = pd.DataFrame.from_records(metrics)
    status_df = pd.DataFrame.from_records(statuses)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["contrast_snr", "tmag_est_median"], ascending=[False, True]).reset_index(drop=True)

    summary_csv = survey_outdir / "bright_kbo_survey_summary.csv"
    summary_json = survey_outdir / "bright_kbo_survey_summary.json"
    status_csv = survey_outdir / "bright_kbo_survey_status.csv"
    figure_png = survey_outdir / "bright_kbo_survey_overview.png"

    metrics_df.to_csv(summary_csv, index=False)
    status_df.to_csv(status_csv, index=False)
    payload = {
        "targets_requested": list(args.targets),
        "reuse_summaries": [str(Path(path)) for path in args.reuse_summary],
        "tmag_offset": float(args.tmag_offset),
        "cleanup_secondary": bool(args.cleanup_secondary),
        "n_success": int((status_df["status"] == "success").sum()) if not status_df.empty else 0,
        "n_failed": int((status_df["status"] != "success").sum()) if not status_df.empty else 0,
        "removed_secondary_files": removed_files,
        "rows": metrics_df.to_dict(orient="records"),
        "status_rows": status_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if not metrics_df.empty:
        build_summary_figure(metrics_df, figure_png)

    print(summary_csv)
    print(status_csv)
    if not metrics_df.empty:
        print(figure_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
