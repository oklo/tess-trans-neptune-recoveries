#!/usr/bin/env python3
"""Run the Holman-style workflow for one or more target detector tracks."""

from __future__ import annotations

# ruff: noqa: E402

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
        description="Query/reuse target ephemerides, select viable detector tracks, and run the Holman-style diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--ephem-csv", type=Path, help="Optional existing normalized detector inventory CSV.")
    parser.add_argument("--sectors", type=int, nargs="*", default=())
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--run-all-viable", action="store_true")
    parser.add_argument("--run-all-detectors-in-sector", action="store_true")
    parser.add_argument("--min-ephem-rows", type=int, default=5)
    parser.add_argument("--min-duration-days", type=float, default=0.5)
    parser.add_argument("--shape", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(21, 21))
    parser.add_argument("--bg-method", choices=("linear_model", "rolling"), default="rolling")
    parser.add_argument("--barycentric-mode", choices=("true", "false"), default="true")
    parser.add_argument("--force-rebuild-bundle", action="store_true")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "holman_diagnostics")
    return parser


def slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text).strip()).strip("_")


def run_cmd(args: list[str], *, env: dict[str, str], workdir: Path) -> None:
    subprocess.run(args, check=True, cwd=workdir, env=env)


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

    run_cmd(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_holman_detector_stack.py"),
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
            "--bg-method",
            str(bg_method),
            "--barycentric-mode",
            str(barycentric_mode),
            "--outdir",
            str(run_dir),
        ],
        env=env,
        workdir=PROJECT_ROOT,
    )

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
        "holman_stack_png": str(run_dir / f"{prefix}_holman_stack.png"),
        "holman_stack_npz": str(run_dir / f"{prefix}_holman_stack.npz"),
        "holman_summary_json": str(run_dir / f"{prefix}_holman_summary.json"),
        "holman_mask_csv": str(run_dir / f"{prefix}_holman_mask.csv"),
        "holman_interval_json": str(run_dir / f"{prefix}_holman_mask_intervals.json"),
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
            "shape": [int(args.shape[0]), int(args.shape[1])],
            "n_runs": int(len(run_records)),
            "runs": run_records,
        },
        target_root / f"{slug(args.target)}_holman_workflow_summary.json",
    )

    print(summary_json)
    for record in run_records:
        print(record["holman_stack_png"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
