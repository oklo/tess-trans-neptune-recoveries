#!/usr/bin/env python3
"""Build or summarize a detector-local target ephemeris inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json

from kbo_tess.candidates import select_bounded_candidate_windows, summarize_detector_candidates  # noqa: E402
from kbo_tess.ephem import (  # noqa: E402
    export_ephemeris_csv,
    load_ephemeris_csv,
    query_tess_ephem,
    select_detector_track,
    summarize_track,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query or normalize a detector-local target ephemeris inventory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="Sedna", help="Target designation to query or label.")
    parser.add_argument("--sector", type=int, default=5, help="TESS observing sector.")
    parser.add_argument("--camera", type=int, default=1, help="TESS camera number.")
    parser.add_argument("--ccd", type=int, default=4, help="TESS CCD number.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Optional local ephemeris CSV to normalize instead of querying tess_ephem.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for derived outputs.",
    )
    parser.add_argument(
        "--all-detectors",
        action="store_true",
        help="Summarize all detector-local opportunities for the queried target instead of selecting one detector.",
    )
    parser.add_argument(
        "--window-days",
        type=float,
        default=2.0,
        help="Bounded candidate-window size in days when --all-detectors is used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned action and summary without writing an output CSV.",
    )
    return parser


def _inventory_filename(target: str, sector: int) -> str:
    return f"{target.lower()}_sector{sector}_inventory.csv"


def _full_inventory_filename(target: str) -> str:
    return f"{target.lower()}_all_detector_inventory.csv"


def _candidate_filename(target: str) -> str:
    return f"{target.lower()}_candidate_table.csv"


def _selection_json_filename(target: str) -> str:
    return f"{target.lower()}_selection_summary.json"


def _selection_note_filename(target: str) -> str:
    return f"{target.lower()}_selection_note.txt"


def _print_summary(summary: dict[str, object]) -> None:
    print("Ephemeris summary:")
    print(f"  rows: {summary.get('rows')}")
    if "time_scale" in summary or "time_reference" in summary:
        print(
            f"  time semantics: scale={summary.get('time_scale', 'unknown')}, "
            f"reference={summary.get('time_reference', 'unknown')}"
        )
    if "time_start" in summary and "time_stop" in summary:
        print(
            "  time_tdb_jd_minus_2457000: "
            f"{summary['time_start']:.8f} -> {summary['time_stop']:.8f}"
        )
    if "x_min" in summary and "x_max" in summary:
        print(f"  x_span: {summary['x_min']:.3f} -> {summary['x_max']:.3f}")
    if "y_min" in summary and "y_max" in summary:
        print(f"  y_span: {summary['y_min']:.3f} -> {summary['y_max']:.3f}")


def load_inventory(args: argparse.Namespace):
    if args.input_csv is not None:
        return load_ephemeris_csv(
            args.input_csv,
            target=args.target,
            require_exact_tdb=True,
        )
    query_kwargs = {"require_exact_tdb": True}
    if not args.all_detectors:
        query_kwargs["sector"] = args.sector
    return query_tess_ephem(target=args.target, **query_kwargs)


def _jsonable_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _build_selection_summary(target: str, candidate_table, selected_windows) -> dict[str, object]:
    summary: dict[str, object] = {
        "target": target,
        "n_detector_candidates": int(len(candidate_table)),
        "n_selected_windows": int(len(selected_windows)),
        "primary_candidate": None,
        "backup_candidate": None,
    }
    for _, row in selected_windows.iterrows():
        payload = {key: _jsonable_value(value) for key, value in row.to_dict().items()}
        if row["candidate_label"] == "primary":
            summary["primary_candidate"] = payload
        elif row["candidate_label"] == "backup":
            summary["backup_candidate"] = payload
    return summary


def _write_selection_note(target: str, selection_summary: dict[str, object], output_path: Path) -> Path:
    primary = selection_summary.get("primary_candidate")
    backup = selection_summary.get("backup_candidate")
    lines = [
        f"{target} calibration candidate selection",
        "",
        f"Detector opportunities found: {selection_summary['n_detector_candidates']}",
        f"Bounded windows selected: {selection_summary['n_selected_windows']}",
        "",
    ]
    if isinstance(primary, dict):
        lines.extend(
            [
                "Primary candidate:",
                f"  sector/camera/ccd: {primary['sector']}/{primary['camera']}/{primary['ccd']}",
                f"  time span: {primary['time_start']:.6f} -> {primary['time_stop']:.6f}",
                f"  edge margin min/median: {primary['edge_margin_min']:.2f} / {primary['edge_margin_median']:.2f}",
                f"  x span: {primary['x_min']:.2f} -> {primary['x_max']:.2f}",
                f"  y span: {primary['y_min']:.2f} -> {primary['y_max']:.2f}",
                "",
            ]
        )
    if isinstance(backup, dict):
        lines.extend(
            [
                "Backup candidate:",
                f"  sector/camera/ccd: {backup['sector']}/{backup['camera']}/{backup['ccd']}",
                f"  time span: {backup['time_start']:.6f} -> {backup['time_stop']:.6f}",
                f"  edge margin min/median: {backup['edge_margin_min']:.2f} / {backup['edge_margin_median']:.2f}",
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> int:
    args = build_parser().parse_args()

    try:
        ephem_df = load_inventory(args)
        if args.all_detectors:
            candidate_table = summarize_detector_candidates(ephem_df, target=args.target)
            selected_windows = select_bounded_candidate_windows(
                ephem_df,
                target=args.target,
                window_days=args.window_days,
                max_candidates=2,
            )
            selection_summary = _build_selection_summary(args.target, candidate_table, selected_windows)
            print("Candidate summary:")
            print(f"  detector opportunities: {len(candidate_table)}")
            print(f"  selected windows: {len(selected_windows)}")
            if not candidate_table.empty:
                best = candidate_table.iloc[0]
                print(
                    f"  best detector: sector/camera/ccd={int(best['sector'])}/{int(best['camera'])}/{int(best['ccd'])}, "
                    f"time span={best['time_start']:.6f}->{best['time_stop']:.6f}, "
                    f"edge margin min/median={best['edge_margin_min']:.2f}/{best['edge_margin_median']:.2f}"
                )
            if not selected_windows.empty:
                primary = selected_windows.iloc[0]
                print(
                    f"  primary window: sector/camera/ccd={int(primary['sector'])}/{int(primary['camera'])}/{int(primary['ccd'])}, "
                    f"time span={primary['time_start']:.6f}->{primary['time_stop']:.6f}"
                )
            print(f"Full inventory path: {args.outdir / _full_inventory_filename(args.target)}")
            print(f"Candidate table path: {args.outdir / _candidate_filename(args.target)}")
            print(f"Selection summary path: {args.outdir / _selection_json_filename(args.target)}")
        else:
            output_path = args.outdir / _inventory_filename(args.target, args.sector)
            detector_track = select_detector_track(
                ephem_df,
                sector=args.sector,
                camera=args.camera,
                ccd=args.ccd,
                target=args.target,
                require_exact_tdb=True,
            )
            summary = summarize_track(detector_track)
            _print_summary(summary)
            print(f"Output path: {output_path}")
    except (ImportError, KeyError, NotImplementedError, ValueError) as exc:
        if args.dry_run:
            print("Dry run only: could not build a normalized ephemeris in this environment.")
            print(f"Reason: {exc}")
            if args.all_detectors:
                print(f"Planned full inventory path: {args.outdir / _full_inventory_filename(args.target)}")
                print(f"Planned candidate path: {args.outdir / _candidate_filename(args.target)}")
            else:
                print(
                    "Planned request: "
                    f"target={args.target}, sector={args.sector}, camera={args.camera}, ccd={args.ccd}"
                )
                print(f"Planned output path: {args.outdir / _inventory_filename(args.target, args.sector)}")
            return 0
        raise

    if args.dry_run:
        print("Dry run: no CSV written.")
        return 0

    if args.all_detectors:
        full_inventory_path = export_ephemeris_csv(ephem_df, args.outdir / _full_inventory_filename(args.target))
        candidate_table.to_csv(args.outdir / _candidate_filename(args.target), index=False)
        selection_summary_path = args.outdir / _selection_json_filename(args.target)
        with selection_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(selection_summary, handle, indent=2, sort_keys=True)
        selection_note_path = _write_selection_note(
            args.target,
            selection_summary,
            args.outdir / _selection_note_filename(args.target),
        )
        print(f"Wrote {full_inventory_path}")
        print(f"Wrote {args.outdir / _candidate_filename(args.target)}")
        print(f"Wrote {selection_summary_path}")
        print(f"Wrote {selection_note_path}")
    else:
        written = export_ephemeris_csv(detector_track, args.outdir / _inventory_filename(args.target, args.sector))
        print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
