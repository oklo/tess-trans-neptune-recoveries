#!/usr/bin/env python3
"""Run the generic nominal-path diagnostic workflow for one or more target detector tracks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from kbo_tess.candidates import summarize_detector_candidates
from kbo_tess.ephem import export_ephemeris_csv, load_ephemeris_csv, query_tess_ephem, select_detector_track


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query/reuse target ephemerides, select viable detector tracks, and run the nominal-path diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", required=True, help="Target designation.")
    parser.add_argument(
        "--ephem-csv",
        type=Path,
        help="Optional existing normalized detector inventory CSV. If omitted, query tess_ephem.",
    )
    parser.add_argument(
        "--sectors",
        type=int,
        nargs="*",
        default=(),
        help="Optional list of sectors to run. If omitted, use the best viable detector windows.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=2,
        help="Maximum viable detector tracks to run when --sectors is not provided and --run-all-viable is not set.",
    )
    parser.add_argument(
        "--run-all-viable",
        action="store_true",
        help="Run every viable detector opportunity instead of only the top-ranked ones.",
    )
    parser.add_argument(
        "--run-all-detectors-in-sector",
        action="store_true",
        help="If a requested sector has multiple viable detector tracks, run them all instead of only the best one.",
    )
    parser.add_argument(
        "--min-ephem-rows",
        type=int,
        default=5,
        help="Lax minimum number of inventory ephemeris rows required to consider a detector track worth running.",
    )
    parser.add_argument(
        "--min-duration-days",
        type=float,
        default=0.5,
        help="Lax minimum on-detector duration required to consider a detector track worth running.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLS"),
        default=(21, 21),
        help="Moving cutout shape for bundle builds.",
    )
    parser.add_argument(
        "--bg-method",
        choices=("linear_model", "rolling"),
        default="rolling",
        help="tess_asteroids background method for bundle builds.",
    )
    parser.add_argument(
        "--barycentric-mode",
        choices=("true", "false"),
        default="true",
        help="Explicit tess_asteroids barycentric mode for bundle builds.",
    )
    parser.add_argument(
        "--force-rebuild-bundle",
        action="store_true",
        help="Rebuild bundles even if the output NPZ already exists.",
    )
    parser.add_argument(
        "--star-mag-threshold",
        type=float,
        default=17.0,
        help="Tmag threshold for catalog sources shown on the cutout-grid overlay.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "nominal_path_diagnostics",
        help="Root directory for workflow outputs.",
    )
    return parser


def slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text).strip()).strip("_")


def run_cmd(args: list[str], *, env: dict[str, str], workdir: Path) -> None:
    subprocess.run(args, check=True, cwd=workdir, env=env)


def choose_local_offset_scale(median_sample_spacing_days: float) -> float:
    hours = float(median_sample_spacing_days) * 24.0
    return 0.9 if hours < 0.10 else (1.0 / 3.0) * 0.5


def choose_path_plot_kwargs(median_sample_spacing_days: float) -> list[str]:
    hours = float(median_sample_spacing_days) * 24.0
    if hours < 0.10:
        return [
            "--label-every",
            "4",
            "--ytick-fontsize",
            "13",
            "--xtick-fontsize",
            "13",
            "--axis-label-fontsize",
            "14",
            "--title-fontsize",
            "14",
            "--line-alpha",
            "0.42",
            "--line-width",
            "0.5",
            "--marker-alpha",
            "0.22",
            "--marker-size",
            "8",
            "--offset-scale",
            "1.0",
        ]
    return []


def choose_local_plot_kwargs(median_sample_spacing_days: float) -> list[str]:
    return ["--offset-scale", f"{choose_local_offset_scale(median_sample_spacing_days):.12g}"]


def load_or_query_ephem(target: str, ephem_csv: Path | None) -> tuple[pd.DataFrame, str]:
    if ephem_csv is not None:
        return load_ephemeris_csv(ephem_csv, target=target, require_exact_tdb=True), "local_csv"
    return query_tess_ephem(target=target, require_exact_tdb=True), "tess_ephem"


def candidate_is_viable(row: pd.Series, *, min_ephem_rows: int, min_duration_days: float) -> bool:
    return int(row["ephem_rows"]) >= int(min_ephem_rows) and float(row["duration_days"]) >= float(min_duration_days)


def select_candidates(
    candidate_table: pd.DataFrame,
    *,
    sectors: tuple[int, ...],
    run_all_viable: bool,
    run_all_detectors_in_sector: bool,
    max_candidates: int,
    min_ephem_rows: int,
    min_duration_days: float,
) -> pd.DataFrame:
    table = candidate_table.copy()
    table["worth_running"] = table.apply(
        candidate_is_viable,
        axis=1,
        min_ephem_rows=min_ephem_rows,
        min_duration_days=min_duration_days,
    )
    table = table[table["worth_running"]].copy()
    if sectors:
        table = table[table["sector"].astype(int).isin([int(value) for value in sectors])].copy()
        if not run_all_detectors_in_sector:
            table = table.groupby("sector", sort=True, as_index=False).head(1).reset_index(drop=True)
        return table
    if run_all_viable:
        return table.reset_index(drop=True)
    return table.head(int(max_candidates)).reset_index(drop=True)


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_json(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_one_candidate(
    row: pd.Series,
    ephem_df: pd.DataFrame,
    *,
    target: str,
    env: dict[str, str],
    outdir: Path,
    shape: tuple[int, int],
    bg_method: str,
    barycentric_mode: str,
    force_rebuild_bundle: bool,
    star_mag_threshold: float,
) -> dict[str, object]:
    sector = int(row["sector"])
    camera = int(row["camera"])
    ccd = int(row["ccd"])
    prefix = f"{slug(target)}_sector{sector}_cam{camera}_ccd{ccd}"
    run_dir = outdir / slug(target) / f"sector{sector}_cam{camera}_ccd{ccd}"
    run_dir.mkdir(parents=True, exist_ok=True)

    detector_track = select_detector_track(ephem_df, sector=sector, camera=camera, ccd=ccd, target=target, require_exact_tdb=True)
    sector_ephem_csv = export_ephemeris_csv(detector_track, run_dir / f"{prefix}_inventory.csv")

    bundle_npz = run_dir / f"{prefix}_cube_bundle.npz"
    bundle_metadata = run_dir / f"{prefix}_cube_bundle_metadata.json"
    if force_rebuild_bundle or not bundle_npz.exists():
        run_cmd(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_sedna_sector5_cube.py"),
                "--target",
                str(target),
                "--sector",
                str(sector),
                "--camera",
                str(camera),
                "--ccd",
                str(ccd),
                "--ephem-csv",
                str(sector_ephem_csv),
                "--shape",
                str(int(shape[0])),
                str(int(shape[1])),
                "--barycentric-mode",
                str(barycentric_mode),
                "--bg-method",
                str(bg_method),
                "--outdir",
                str(run_dir),
            ],
            env=env,
            workdir=PROJECT_ROOT,
        )

    overlay_png = run_dir / f"{prefix}_cutout_grid_overlay.png"
    stars_outdir = run_dir / "catalog_sources"
    run_cmd(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "list_pluto_catalog_crossers.py"),
            "--target",
            str(target),
            "--bundle-npz",
            str(bundle_npz),
            "--outdir",
            str(stars_outdir),
            "--target-mag-limit",
            str(float(star_mag_threshold)),
        ],
        env=env,
        workdir=PROJECT_ROOT,
    )
    bright_csv = stars_outdir / f"{prefix}_tic_crossers_bright.csv"
    run_cmd(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "plot_cutout_grid_overlay.py"),
            "--bundle-npz",
            str(bundle_npz),
            "--out-png",
            str(overlay_png),
            "--stars-csv",
            str(bright_csv),
            "--mag-threshold",
            str(float(star_mag_threshold)),
            "--shade-path-pixels",
        ],
        env=env,
        workdir=PROJECT_ROOT,
    )

    path_png = run_dir / f"{prefix}_nominal_path_pixel_timeseries_automask.png"
    path_csv = run_dir / f"{prefix}_nominal_path_pixel_timeseries_automask.csv"
    path_summary = run_dir / f"{prefix}_automask_summary.txt"
    mask_csv = run_dir / f"{prefix}_automask_diagnostics.csv"
    interval_json = run_dir / f"{prefix}_automask_intervals.json"
    path_args = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "plot_nominal_path_pixel_timeseries.py"),
        "--target",
        str(target),
        "--sector",
        str(sector),
        "--camera",
        str(camera),
        "--ccd",
        str(ccd),
        "--ephem-csv",
        str(sector_ephem_csv),
        "--bundle-npz",
        str(bundle_npz),
        "--mask-mode",
        "common_mode_v1",
        "--target-box-size",
        "5",
        "--path-box-size",
        "9",
        "--baseline-hours",
        "24",
        "--slope-smooth-hours",
        "1",
        "--merge-gap-hours",
        "10",
        "--interval-pad-hours",
        "6",
        "--min-interval-hours",
        "6",
        "--out-png",
        str(path_png),
        "--out-csv",
        str(path_csv),
        "--out-summary",
        str(path_summary),
        "--out-mask-csv",
        str(mask_csv),
        "--out-interval-json",
        str(interval_json),
    ]
    path_args.extend(choose_path_plot_kwargs(float(row["median_sample_spacing_days"])))
    run_cmd(path_args, env=env, workdir=PROJECT_ROOT)

    local_png = run_dir / f"{prefix}_nominal_path_local_windows_automask.png"
    local_csv = run_dir / f"{prefix}_nominal_path_local_windows_automask.csv"
    local_summary = run_dir / f"{prefix}_nominal_path_local_windows_automask_summary.txt"
    local_args = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "plot_nominal_path_local_windows.py"),
        "--target",
        str(target),
        "--sector",
        str(sector),
        "--camera",
        str(camera),
        "--ccd",
        str(ccd),
        "--ephem-csv",
        str(sector_ephem_csv),
        "--bundle-npz",
        str(bundle_npz),
        "--mask-csv",
        str(mask_csv),
        "--detrend-degree",
        "2",
        "--bootstrap-draws",
        "500",
        "--bootstrap-seed",
        "0",
        "--bootstrap-errorbar-stride",
        "4",
        "--out-png",
        str(local_png),
        "--out-csv",
        str(local_csv),
        "--out-summary",
        str(local_summary),
    ]
    local_args.extend(choose_local_plot_kwargs(float(row["median_sample_spacing_days"])))
    run_cmd(local_args, env=env, workdir=PROJECT_ROOT)

    return {
        "target": str(target),
        "sector": sector,
        "camera": camera,
        "ccd": ccd,
        "ephem_rows": int(row["ephem_rows"]),
        "duration_days": float(row["duration_days"]),
        "edge_margin_min": float(row["edge_margin_min"]),
        "edge_margin_median": float(row["edge_margin_median"]),
        "median_sample_spacing_days": float(row["median_sample_spacing_days"]),
        "inventory_csv": str(sector_ephem_csv),
        "bundle_npz": str(bundle_npz),
        "bundle_metadata_json": str(bundle_metadata),
        "bright_crosser_csv": str(bright_csv),
        "cutout_grid_overlay_png": str(overlay_png),
        "path_timeseries_png": str(path_png),
        "path_timeseries_csv": str(path_csv),
        "path_summary_txt": str(path_summary),
        "mask_csv": str(mask_csv),
        "interval_json": str(interval_json),
        "local_windows_png": str(local_png),
        "local_windows_csv": str(local_csv),
        "local_windows_summary_txt": str(local_summary),
    }


def main() -> int:
    args = build_parser().parse_args()

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
    env.setdefault("KBO_TESS_USE_SYSTEM_HOME", "1")

    ephem_df, ephem_source = load_or_query_ephem(str(args.target), args.ephem_csv)
    out_root = Path(args.outdir)
    target_root = out_root / slug(args.target)
    target_root.mkdir(parents=True, exist_ok=True)

    full_inventory_csv = export_ephemeris_csv(ephem_df, target_root / f"{slug(args.target)}_all_detector_inventory.csv")
    candidate_table = summarize_detector_candidates(ephem_df, target=str(args.target))
    candidate_csv = write_table(candidate_table, target_root / f"{slug(args.target)}_detector_candidates.csv")

    selected = select_candidates(
        candidate_table,
        sectors=tuple(int(value) for value in args.sectors),
        run_all_viable=bool(args.run_all_viable),
        run_all_detectors_in_sector=bool(args.run_all_detectors_in_sector),
        max_candidates=int(args.max_candidates),
        min_ephem_rows=int(args.min_ephem_rows),
        min_duration_days=float(args.min_duration_days),
    )
    if selected.empty:
        raise SystemExit(
            f"No viable detector tracks found for {args.target} with min_ephem_rows={args.min_ephem_rows} "
            f"and min_duration_days={args.min_duration_days}."
        )
    selected_csv = write_table(selected, target_root / f"{slug(args.target)}_selected_detector_candidates.csv")

    run_records = []
    for _, row in selected.iterrows():
        run_records.append(
            run_one_candidate(
                row,
                ephem_df=ephem_df,
                target=str(args.target),
                env=env,
                outdir=out_root,
                shape=(int(args.shape[0]), int(args.shape[1])),
                bg_method=str(args.bg_method),
                barycentric_mode=str(args.barycentric_mode),
                force_rebuild_bundle=bool(args.force_rebuild_bundle),
                star_mag_threshold=float(args.star_mag_threshold),
            )
        )

    summary_json = write_json(
        {
            "target": str(args.target),
            "ephem_source": ephem_source,
            "full_inventory_csv": str(full_inventory_csv),
            "candidate_csv": str(candidate_csv),
            "selected_candidate_csv": str(selected_csv),
            "min_ephem_rows": int(args.min_ephem_rows),
            "min_duration_days": float(args.min_duration_days),
            "bg_method": str(args.bg_method),
            "barycentric_mode": str(args.barycentric_mode),
            "star_mag_threshold": float(args.star_mag_threshold),
            "shape": [int(args.shape[0]), int(args.shape[1])],
            "n_runs": int(len(run_records)),
            "runs": run_records,
        },
        target_root / f"{slug(args.target)}_nominal_path_workflow_summary.json",
    )

    print(summary_json)
    for record in run_records:
        print(record["local_windows_png"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
