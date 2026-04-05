#!/usr/bin/env python3
"""Build or summarize a detector-local Sedna ephemeris inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kbo_tess.ephem import (  # noqa: E402
    export_ephemeris_csv,
    load_ephemeris_csv,
    query_tess_ephem,
    select_detector_track,
    summarize_track,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query or normalize a detector-local Sedna ephemeris inventory.",
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
        "--dry-run",
        action="store_true",
        help="Print the planned action and summary without writing an output CSV.",
    )
    return parser


def _inventory_filename(target: str, sector: int) -> str:
    return f"{target.lower()}_sector{sector}_inventory.csv"


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
    return query_tess_ephem(
        target=args.target,
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        require_exact_tdb=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    output_path = args.outdir / _inventory_filename(args.target, args.sector)

    try:
        ephem_df = load_inventory(args)
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
            print(
                "Planned request: "
                f"target={args.target}, sector={args.sector}, camera={args.camera}, ccd={args.ccd}"
            )
            print(f"Planned output path: {output_path}")
            return 0
        raise

    if args.dry_run:
        print("Dry run: no CSV written.")
        return 0

    written = export_ephemeris_csv(detector_track, output_path)
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
