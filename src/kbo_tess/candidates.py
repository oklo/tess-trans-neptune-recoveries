"""Small helpers for ranking detector-local calibration candidates."""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    pd = None

from kbo_tess.ephem import normalize_ephemeris_dataframe


def _require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is required for candidate ranking. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return np


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for candidate ranking. Install project dependencies with "
            "`python -m pip install -e .`."
        )
    return pd


def _edge_margins(
    detector_track: Any,
    detector_shape: tuple[int, int] = (2048, 2048),
) -> Any:
    numpy = _require_numpy()
    x = detector_track["x"].astype(float).to_numpy()
    y = detector_track["y"].astype(float).to_numpy()
    width = float(detector_shape[0])
    height = float(detector_shape[1])
    return numpy.minimum.reduce(
        [
            x - 1.0,
            width - x,
            y - 1.0,
            height - y,
        ]
    )


def summarize_detector_candidates(
    ephem_df: Any,
    target: str | None = None,
    detector_shape: tuple[int, int] = (2048, 2048),
) -> Any:
    """Summarize one row per observed detector track."""

    pandas = _require_pandas()
    numpy = _require_numpy()
    normalized = normalize_ephemeris_dataframe(ephem_df, target=target, require_exact_tdb=True)

    rows: list[dict[str, Any]] = []
    for (sector, camera, ccd), group in normalized.groupby(["sector", "camera", "ccd"], sort=True):
        group = group.sort_values("time_tdb_jd_minus_2457000").reset_index(drop=True)
        margins = _edge_margins(group, detector_shape=detector_shape)
        times = group["time_tdb_jd_minus_2457000"].astype(float).to_numpy()
        median_spacing = float(numpy.nanmedian(numpy.diff(times))) if len(times) > 1 else float("nan")
        duration_days = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        rows.append(
            {
                "target": str(group.iloc[0]["target"]) if "target" in group.columns else target or "target",
                "sector": int(sector),
                "camera": int(camera),
                "ccd": int(ccd),
                "ephem_rows": int(len(group)),
                "time_start": float(times[0]),
                "time_stop": float(times[-1]),
                "duration_days": duration_days,
                "median_sample_spacing_days": median_spacing,
                "x_min": float(group["x"].min()),
                "x_max": float(group["x"].max()),
                "x_span": float(group["x"].max() - group["x"].min()),
                "y_min": float(group["y"].min()),
                "y_max": float(group["y"].max()),
                "y_span": float(group["y"].max() - group["y"].min()),
                "edge_margin_min": float(numpy.nanmin(margins)),
                "edge_margin_median": float(numpy.nanmedian(margins)),
                "edge_margin_max": float(numpy.nanmax(margins)),
                "time_scale": str(group.iloc[0]["time_scale"]) if "time_scale" in group.columns else "unknown",
                "time_reference": str(group.iloc[0]["time_reference"]) if "time_reference" in group.columns else "unknown",
                # Transparent heuristic: prioritize staying far from edges, then duration.
                "calibration_score": float(numpy.nanmin(margins) + 0.25 * numpy.nanmedian(margins) + 2.0 * duration_days),
            }
        )

    if not rows:
        return pandas.DataFrame(
            columns=[
                "target",
                "sector",
                "camera",
                "ccd",
                "ephem_rows",
                "time_start",
                "time_stop",
                "duration_days",
                "median_sample_spacing_days",
                "x_min",
                "x_max",
                "x_span",
                "y_min",
                "y_max",
                "y_span",
                "edge_margin_min",
                "edge_margin_median",
                "edge_margin_max",
                "time_scale",
                "time_reference",
                "calibration_score",
            ]
        )

    summary = pandas.DataFrame(rows)
    return summary.sort_values(
        ["calibration_score", "edge_margin_min", "duration_days", "ephem_rows"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def select_bounded_candidate_windows(
    ephem_df: Any,
    target: str | None = None,
    window_days: float = 2.0,
    max_candidates: int = 2,
    detector_shape: tuple[int, int] = (2048, 2048),
    min_separation_days: float | None = None,
) -> Any:
    """Select bounded primary/backup windows from one or more detector tracks."""

    pandas = _require_pandas()
    numpy = _require_numpy()
    if window_days <= 0:
        raise ValueError(f"`window_days` must be positive; got {window_days}.")
    if max_candidates < 1:
        raise ValueError(f"`max_candidates` must be at least 1; got {max_candidates}.")
    if min_separation_days is None:
        min_separation_days = 0.75 * float(window_days)

    normalized = normalize_ephemeris_dataframe(ephem_df, target=target, require_exact_tdb=True)
    candidate_rows: list[dict[str, Any]] = []
    half_window = 0.5 * float(window_days)

    for (sector, camera, ccd), group in normalized.groupby(["sector", "camera", "ccd"], sort=True):
        group = group.sort_values("time_tdb_jd_minus_2457000").reset_index(drop=True)
        margins = _edge_margins(group, detector_shape=detector_shape)
        times = group["time_tdb_jd_minus_2457000"].astype(float).to_numpy()
        for index, center_time in enumerate(times):
            mask = (times >= center_time - half_window) & (times <= center_time + half_window)
            if int(mask.sum()) < 2:
                continue
            subset = group.loc[mask].reset_index(drop=True)
            subset_margins = margins[mask]
            candidate_rows.append(
                {
                    "target": str(subset.iloc[0]["target"]) if "target" in subset.columns else target or "target",
                    "sector": int(sector),
                    "camera": int(camera),
                    "ccd": int(ccd),
                    "window_center_time": float(center_time),
                    "time_start": float(subset["time_tdb_jd_minus_2457000"].min()),
                    "time_stop": float(subset["time_tdb_jd_minus_2457000"].max()),
                    "duration_days": float(subset["time_tdb_jd_minus_2457000"].max() - subset["time_tdb_jd_minus_2457000"].min()),
                    "ephem_rows": int(len(subset)),
                    "x_min": float(subset["x"].min()),
                    "x_max": float(subset["x"].max()),
                    "x_span": float(subset["x"].max() - subset["x"].min()),
                    "y_min": float(subset["y"].min()),
                    "y_max": float(subset["y"].max()),
                    "y_span": float(subset["y"].max() - subset["y"].min()),
                    "edge_margin_min": float(numpy.nanmin(subset_margins)),
                    "edge_margin_median": float(numpy.nanmedian(subset_margins)),
                    "edge_margin_max": float(numpy.nanmax(subset_margins)),
                    "window_score": float(numpy.nanmin(subset_margins) + 0.25 * numpy.nanmedian(subset_margins) + 0.5 * len(subset)),
                    "time_scale": str(subset.iloc[0]["time_scale"]) if "time_scale" in subset.columns else "unknown",
                    "time_reference": str(subset.iloc[0]["time_reference"]) if "time_reference" in subset.columns else "unknown",
                    "detector_key": f"{int(sector)}-{int(camera)}-{int(ccd)}",
                }
            )

    if not candidate_rows:
        return pandas.DataFrame()

    ranked = pandas.DataFrame(candidate_rows).sort_values(
        ["window_score", "edge_margin_min", "duration_days", "ephem_rows"],
        ascending=[False, False, False, False],
    )

    selected_rows: list[dict[str, Any]] = []
    for row in ranked.to_dict(orient="records"):
        conflicts = False
        for existing in selected_rows:
            if row["detector_key"] != existing["detector_key"]:
                continue
            if abs(float(row["window_center_time"]) - float(existing["window_center_time"])) < float(min_separation_days):
                conflicts = True
                break
        if conflicts:
            continue
        selected_rows.append(row)
        if len(selected_rows) >= int(max_candidates):
            break

    selected = pandas.DataFrame(selected_rows)
    if selected.empty:
        return selected

    labels = ["primary", "backup", "candidate3", "candidate4", "candidate5"]
    selected = selected.reset_index(drop=True)
    selected["candidate_rank"] = selected.index + 1
    selected["candidate_label"] = [
        labels[index] if index < len(labels) else f"candidate{index + 1}"
        for index in range(len(selected))
    ]
    ordered_columns = [
        "candidate_rank",
        "candidate_label",
        "target",
        "sector",
        "camera",
        "ccd",
        "time_start",
        "time_stop",
        "window_center_time",
        "duration_days",
        "ephem_rows",
        "x_min",
        "x_max",
        "x_span",
        "y_min",
        "y_max",
        "y_span",
        "edge_margin_min",
        "edge_margin_median",
        "edge_margin_max",
        "window_score",
        "time_scale",
        "time_reference",
    ]
    return selected[ordered_columns]
