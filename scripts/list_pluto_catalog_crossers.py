#!/usr/bin/env python3
"""List TIC/Gaia catalog sources that pass through a retained moving-target cutout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _require_numpy() -> Any:
    import numpy as np

    return np


def _require_pandas() -> Any:
    import pandas as pd

    return pd


def _require_astropy() -> tuple[Any, Any]:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    return SkyCoord, u


def _require_catalog_tools() -> tuple[Any, Any]:
    import tesswcs
    from astroquery.mast import Catalogs

    return tesswcs, Catalogs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Identify TIC/Gaia catalog sources that pass through a retained 21x21 moving-target cutout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bundle-npz",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "pluto_full_duration_conservative_mask"
        / "pluto_sector92_cam2_ccd1_full_cube_bundle_masked.npz",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pluto_full_duration_catalog_sources",
    )
    parser.add_argument("--target", default="Pluto")
    parser.add_argument("--target-mag-limit", type=float, default=15.5)
    parser.add_argument("--query-radius-deg", type=float, default=0.0)
    parser.add_argument("--bbox-margin-pix", type=float, default=1.0)
    parser.add_argument("--central-radius-pix", type=float, default=5.0)
    return parser


def load_bundle(bundle_path: Path) -> dict[str, Any]:
    np = _require_numpy()

    loaded = np.load(bundle_path, allow_pickle=True)
    bundle = {key: loaded[key] for key in loaded.files}
    metadata = json.loads(str(bundle["metadata_json"])) if "metadata_json" in bundle else {}
    bundle["metadata"] = metadata
    return bundle


def choose_query_region(bundle: dict[str, Any], explicit_radius_deg: float) -> tuple[Any, float]:
    np = _require_numpy()
    SkyCoord, u = _require_astropy()

    ra = np.asarray(bundle["ra_pred_deg"], dtype=float)
    dec = np.asarray(bundle["dec_pred_deg"], dtype=float)
    center = SkyCoord(float(np.nanmedian(ra)) * u.deg, float(np.nanmedian(dec)) * u.deg)
    if explicit_radius_deg > 0:
        return center, float(explicit_radius_deg)

    path = SkyCoord(ra * u.deg, dec * u.deg)
    max_sep = float(np.nanmax(center.separation(path).deg))
    # TESS pixels are about 21 arcsec. Add a little extra margin beyond the
    # half-diagonal of the 21x21 cutout so the catalog query fully covers the
    # retained union footprint.
    half_diagonal_pix = ((21.0 / 2.0) ** 2 + (21.0 / 2.0) ** 2) ** 0.5
    cutout_margin_deg = (half_diagonal_pix * 21.0 + 42.0) / 3600.0
    return center, float(max_sep + cutout_margin_deg)


def query_tic_sources(center: Any, radius_deg: float) -> Any:
    _, Catalogs = _require_catalog_tools()
    _, u = _require_astropy()

    return Catalogs.query_region(center, radius=radius_deg * u.deg, catalog="TIC")


def project_sources_to_detector(table: Any, sector: int, camera: int, ccd: int) -> tuple[Any, Any]:
    np = _require_numpy()
    tesswcs, _ = _require_catalog_tools()

    wcs = tesswcs.WCS.from_sector(sector=int(sector), camera=int(camera), ccd=int(ccd))
    x0, y0 = wcs.world_to_pixel_values(
        np.asarray(table["ra"], dtype=float),
        np.asarray(table["dec"], dtype=float),
    )
    return np.asarray(x0, dtype=float) + 1.0, np.asarray(y0, dtype=float) + 1.0


def compute_crossers(
    table: Any,
    detector_x: Any,
    detector_y: Any,
    bundle: dict[str, Any],
    mag_limit: float,
    bbox_margin_pix: float,
    central_radius_pix: float,
) -> tuple[Any, dict[str, Any]]:
    np = _require_numpy()
    pd = _require_pandas()

    corner1 = np.asarray(bundle["corner1"], dtype=float)
    corner2 = np.asarray(bundle["corner2"], dtype=float)
    local_pluto_x = np.asarray(bundle["local_x"], dtype=float)
    local_pluto_y = np.asarray(bundle["local_y"], dtype=float)
    time = np.asarray(bundle["time_tdb_jd_minus_2457000"], dtype=float)

    ny, nx = np.asarray(bundle["cube"]).shape[1:]
    xmax_frame = corner1 + float(nx - 1)
    ymax_frame = corner2 + float(ny - 1)
    xmin = float(np.nanmin(corner1) - bbox_margin_pix)
    xmax = float(np.nanmax(xmax_frame) + bbox_margin_pix)
    ymin = float(np.nanmin(corner2) - bbox_margin_pix)
    ymax = float(np.nanmax(ymax_frame) + bbox_margin_pix)

    pre = (
        np.isfinite(detector_x)
        & np.isfinite(detector_y)
        & (detector_x >= xmin)
        & (detector_x <= xmax)
        & (detector_y >= ymin)
        & (detector_y <= ymax)
    )
    candidate_idx = np.flatnonzero(pre)

    rows: list[dict[str, Any]] = []
    for j in candidate_idx:
        x_star = float(detector_x[j])
        y_star = float(detector_y[j])
        inside = (
            (x_star >= corner1)
            & (x_star <= xmax_frame)
            & (y_star >= corner2)
            & (y_star <= ymax_frame)
        )
        n_inside = int(np.count_nonzero(inside))
        if n_inside == 0:
            continue
        local_x = x_star - corner1
        local_y = y_star - corner2
        distance_to_pluto = np.hypot(local_x - local_pluto_x, local_y - local_pluto_y)
        central = inside & (distance_to_pluto <= float(central_radius_pix))

        gaia_value = str(table["GAIA"][j]).strip() if "GAIA" in table.colnames else ""
        gaia_id = gaia_value if gaia_value not in {"", "--", "nan", "None"} else ""
        tic_id = str(table["ID"][j])
        source_key = f"gaia:{gaia_id}" if gaia_id else f"tic:{tic_id}"
        tmag = float(table["Tmag"][j]) if "Tmag" in table.colnames and np.isfinite(table["Tmag"][j]) else float("nan")
        rows.append(
            {
                "source_key": source_key,
                "tic_id": tic_id,
                "gaia_id": gaia_id,
                "ra_deg": float(table["ra"][j]),
                "dec_deg": float(table["dec"][j]),
                "tmag": tmag,
                "detector_x": x_star,
                "detector_y": y_star,
                "n_cadences_in_frame": n_inside,
                "n_cadences_within_central_radius": int(np.count_nonzero(central)),
                "first_time_in_frame": float(time[inside][0]),
                "last_time_in_frame": float(time[inside][-1]),
                "first_local_x": float(local_x[inside][0]),
                "first_local_y": float(local_y[inside][0]),
                "last_local_x": float(local_x[inside][-1]),
                "last_local_y": float(local_y[inside][-1]),
                "min_distance_to_pluto_pix": float(np.nanmin(distance_to_pluto)),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        summary = {
            "n_catalog_rows": int(len(table)),
            "n_bbox_candidates": int(len(candidate_idx)),
            "n_crossers": 0,
            "n_bright_crossers": 0,
        }
        return frame, summary

    frame = frame.sort_values(
        ["tmag", "n_cadences_in_frame", "min_distance_to_pluto_pix"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    deduped = (
        frame.sort_values(
            ["source_key", "tmag", "n_cadences_in_frame"],
            ascending=[True, True, False],
            na_position="last",
        )
        .drop_duplicates(subset=["source_key"], keep="first")
        .sort_values(["tmag", "n_cadences_in_frame"], ascending=[True, False], na_position="last")
        .reset_index(drop=True)
    )
    bright = deduped[deduped["tmag"].fillna(99.0) <= float(mag_limit)].copy().reset_index(drop=True)

    # Likely late-run bright contaminant near the off-center raw-frame feature.
    late_feature_x = float(corner1[-1] + 8.0)
    late_feature_y = float(corner2[-1] + 12.0)
    feature_distance = np.hypot(deduped["detector_x"] - late_feature_x, deduped["detector_y"] - late_feature_y)
    nearest_feature = (
        deduped.assign(distance_to_late_feature_pix=feature_distance)
        .sort_values(["distance_to_late_feature_pix", "tmag"], ascending=[True, True], na_position="last")
        .head(5)
    )

    summary = {
        "n_catalog_rows": int(len(table)),
        "n_bbox_candidates": int(len(candidate_idx)),
        "n_crossers": int(len(frame)),
        "n_unique_crossers": int(len(deduped)),
        "n_bright_crossers": int(len(bright)),
        "bright_mag_limit": float(mag_limit),
        "late_feature_detector_x": late_feature_x,
        "late_feature_detector_y": late_feature_y,
        "late_feature_nearest_sources": nearest_feature.to_dict(orient="records"),
    }
    return (frame, {"summary": summary, "deduped": deduped, "bright": bright})


def write_outputs(frame: Any, aux: dict[str, Any], outdir: Path, prefix: str) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    full_csv = outdir / f"{prefix}_tic_crossers_full.csv"
    dedup_csv = outdir / f"{prefix}_tic_crossers_deduped.csv"
    bright_csv = outdir / f"{prefix}_tic_crossers_bright.csv"
    summary_json = outdir / f"{prefix}_tic_crossers_summary.json"
    frame.to_csv(full_csv, index=False)
    aux["deduped"].to_csv(dedup_csv, index=False)
    aux["bright"].to_csv(bright_csv, index=False)
    summary_json.write_text(json.dumps(aux["summary"], indent=2), encoding="utf-8")
    return {
        "full_csv": str(full_csv),
        "dedup_csv": str(dedup_csv),
        "bright_csv": str(bright_csv),
        "summary_json": str(summary_json),
    }


def output_prefix(metadata: dict[str, Any]) -> str:
    target = str(metadata.get("target", "target")).strip().lower()
    sector = int(metadata["sector"])
    camera = int(metadata["camera"])
    ccd = int(metadata["ccd"])
    return f"{target}_sector{sector}_cam{camera}_ccd{ccd}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    bundle = load_bundle(args.bundle_npz)
    metadata = bundle["metadata"]
    center, radius_deg = choose_query_region(bundle, explicit_radius_deg=float(args.query_radius_deg))
    table = query_tic_sources(center, radius_deg=radius_deg)
    detector_x, detector_y = project_sources_to_detector(
        table,
        sector=int(metadata["sector"]),
        camera=int(metadata["camera"]),
        ccd=int(metadata["ccd"]),
    )
    frame, aux = compute_crossers(
        table,
        detector_x,
        detector_y,
        bundle,
        mag_limit=float(args.target_mag_limit),
        bbox_margin_pix=float(args.bbox_margin_pix),
        central_radius_pix=float(args.central_radius_pix),
    )
    prefix = output_prefix(metadata)
    if frame.empty:
        print("No TIC sources were found inside the retained cutout footprint.")
        return 0

    outputs = write_outputs(frame, aux, args.outdir, prefix=prefix)
    summary = aux["summary"]
    print(f"Queried TIC rows: {summary['n_catalog_rows']}")
    print(f"Crossers in retained 21x21 footprint: {summary['n_crossers']}")
    print(f"Unique source keys after Gaia/TIC dedupe: {summary['n_unique_crossers']}")
    print(f"Bright subset (Tmag <= {summary['bright_mag_limit']:.1f}): {summary['n_bright_crossers']}")
    print("Output files:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    print("Top bright crossers:")
    bright = aux["bright"].head(20)
    for row in bright.itertuples(index=False):
        gaia = str(row.gaia_id) if str(row.gaia_id) else "-"
        print(
            f"  TIC {row.tic_id} | Gaia {gaia} | Tmag {row.tmag:.3f} | "
            f"detector=({row.detector_x:.2f},{row.detector_y:.2f}) | "
            f"n_in_frame={row.n_cadences_in_frame} | min_d_pluto={row.min_distance_to_pluto_pix:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
