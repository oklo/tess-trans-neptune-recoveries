"""Helpers for exact, detector-local TESS ephemeris normalization.

Real-data paths in this project require an exact canonical
`time_tdb_jd_minus_2457000` column. This module will derive it from one of:

- a direct TDB column such as `time_tdb_jd` or `btjd`
- `time_utc_jd` plus an exact `tdb_ut_seconds` correction column

For real ephemerides we do not approximate UTC->TDB. If a file only provides
UTC without an exact TDB-UT correction, the caller gets a clear error so the
data can be repaired upstream.

The normalized CSV handoff also carries explicit timing semantics:

- `time_scale`: expected to be `tdb` for real recovery-ready rows
- `time_reference`: one of `spacecraft`, `barycentric`, or `unspecified`

When `time_tdb_jd_minus_2457000` is derived from `time_utc_jd` plus
`tdb_ut_seconds`, the time reference is set to `spacecraft` because this is the
convention used by `tess_ephem` / `tess_asteroids` custom ephemerides. If a CSV
already contains a TDB column but does not state whether that time is measured
at the spacecraft or the solar-system barycenter, the normalized result keeps
`time_reference=unspecified` so downstream real-data code can stop rather than
silently guess.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    pd = None

try:
    import tess_ephem  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    tess_ephem = None

TESS_EPOCH_JD = 2457000.0

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "target": ("target", "name", "object", "designation"),
    "sector": ("sector",),
    "camera": ("camera", "cam"),
    "ccd": ("ccd",),
    "x": ("x", "column", "col", "xpix", "x_pixel", "pixel_x"),
    "y": ("y", "row", "ypix", "y_pixel", "pixel_y"),
    "time_utc_jd": ("time_utc_jd", "utc_jd", "jd_utc", "ut_jd", "jdutc", "time_jd_utc"),
    "time_tdb_jd": ("time_tdb_jd", "tdb_jd", "jd_tdb", "jdtdb", "bjd_tdb"),
    "time_tdb_jd_minus_2457000": (
        "time_tdb_jd_minus_2457000",
        "tdb_jd_minus_2457000",
        "btjd",
        "time_btjd",
        "bjd_tdb_minus_2457000",
        "bjd_tdb_2457000",
    ),
    "tdb_ut_seconds": (
        "tdb_ut_seconds",
        "tdb_ut",
        "tdb_minus_ut",
        "tdb_minus_ut_seconds",
        "tdbut",
        "tdb_ut_sec",
        "tdb_ut_s",
    ),
    "time_scale": ("time_scale", "timescale", "scale"),
    "time_reference": (
        "time_reference",
        "tdb_reference",
        "time_frame",
        "time_origin_reference",
        "barycentric_mode",
    ),
}


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for ephemeris handling. Install project dependencies with "
            "`python -m pip install pandas` or `python -m pip install -e .`."
        )
    return pd


def _require_tess_ephem() -> Any:
    if tess_ephem is None:
        raise ImportError(
            "tess_ephem is not installed. Install it with "
            "`python -m pip install tess-ephem` or `python -m pip install -e .[ephem]`, "
            "or pass a local ephemeris CSV instead."
        )
    return tess_ephem


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in dependency-light environments
        raise ImportError(
            "numpy is required for ephemeris time conversion. Install it with "
            "`python -m pip install numpy` or `python -m pip install -e .`."
        ) from exc
    return np


def _clean_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _normalize_time_reference_value(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null", "unspecified", "unknown"}:
        return "unspecified"
    if text in {"true", "1", "barycentric", "ssb", "solar_system_barycenter", "solar system barycenter"}:
        return "barycentric"
    if text in {"false", "0", "spacecraft", "observer", "tess", "spacecraft_frame", "spacecraft frame"}:
        return "spacecraft"
    return "unspecified"


def exact_utc_jd_to_tdb_jd_minus_2457000(
    jd_utc: Iterable[float],
    tdb_ut_seconds: Iterable[float],
) -> Any:
    """Convert UTC Julian dates to exact TDB JD-2457000 using a provided offset."""

    np = _require_numpy()
    jd_utc_array = np.asarray(jd_utc, dtype=float)
    tdb_ut_array = np.asarray(tdb_ut_seconds, dtype=float)
    if jd_utc_array.shape != tdb_ut_array.shape:
        raise ValueError(
            "UTC Julian dates and TDB-UT offsets must have matching shapes; "
            f"got {jd_utc_array.shape!r} and {tdb_ut_array.shape!r}."
        )
    return jd_utc_array + (tdb_ut_array / 86400.0) - TESS_EPOCH_JD


def _first_matching_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    cleaned_to_original = {_clean_column_name(column): column for column in columns}
    for alias in aliases:
        match = cleaned_to_original.get(_clean_column_name(alias))
        if match is not None:
            return match
    return None


def _normalize_column_names(ephem_df: Any) -> Any:
    pandas = _require_pandas()
    if not isinstance(ephem_df, pandas.DataFrame):
        ephem_df = pandas.DataFrame(ephem_df)
    normalized = ephem_df.copy()
    if "time" not in normalized.columns and getattr(normalized.index, "name", None) == "time":
        normalized = normalized.reset_index()

    rename_map: dict[str, str] = {}
    for canonical_name, aliases in _COLUMN_ALIASES.items():
        if canonical_name in normalized.columns:
            continue
        match = _first_matching_column(normalized.columns, aliases)
        if match is not None:
            rename_map[match] = canonical_name
    if rename_map:
        normalized = normalized.rename(columns=rename_map)
    return normalized


def _coerce_time_series_to_jd(series: Any) -> Any:
    pandas = _require_pandas()
    sample = None
    for value in series:
        if value is None:
            continue
        if pandas.isna(value):
            continue
        sample = value
        break
    if sample is None:
        return series.astype(float)
    if hasattr(sample, "jd"):
        return pandas.Series(
            [float(value.jd) if value is not None and not pandas.isna(value) else float("nan") for value in series],
            index=series.index,
            dtype=float,
        )
    return series.astype(float)


def _ensure_exact_tdb_column(normalized: Any, require_exact_tdb: bool) -> Any:
    derived_from_utc = False
    if "time_tdb_jd_minus_2457000" in normalized.columns:
        normalized["time_tdb_jd_minus_2457000"] = normalized[
            "time_tdb_jd_minus_2457000"
        ].astype(float)
    elif "time_tdb_jd" in normalized.columns:
        normalized["time_tdb_jd"] = _coerce_time_series_to_jd(normalized["time_tdb_jd"])
        normalized["time_tdb_jd_minus_2457000"] = normalized["time_tdb_jd"] - TESS_EPOCH_JD
    elif "time_utc_jd" in normalized.columns and "tdb_ut_seconds" in normalized.columns:
        normalized["time_utc_jd"] = _coerce_time_series_to_jd(normalized["time_utc_jd"])
        normalized["tdb_ut_seconds"] = normalized["tdb_ut_seconds"].astype(float)
        normalized["time_tdb_jd_minus_2457000"] = exact_utc_jd_to_tdb_jd_minus_2457000(
            normalized["time_utc_jd"],
            normalized["tdb_ut_seconds"],
        )
        derived_from_utc = True
    elif require_exact_tdb:
        available = ", ".join(str(column) for column in normalized.columns)
        raise ValueError(
            "Ephemeris is missing an exact TDB time definition. Provide either a direct TDB "
            "column such as `time_tdb_jd`/`btjd`, or provide `time_utc_jd` together with "
            "`tdb_ut_seconds` (or an equivalent column variant). "
            f"Available columns: {available}"
        )

    if "time_tdb_jd_minus_2457000" in normalized.columns and "time_tdb_jd" not in normalized.columns:
        normalized["time_tdb_jd"] = normalized["time_tdb_jd_minus_2457000"].astype(float) + TESS_EPOCH_JD

    if "time_tdb_jd_minus_2457000" in normalized.columns:
        normalized["time_scale"] = "tdb"
        if "time_reference" in normalized.columns:
            normalized["time_reference"] = normalized["time_reference"].map(_normalize_time_reference_value)
            if derived_from_utc:
                normalized.loc[
                    normalized["time_reference"] == "unspecified",
                    "time_reference",
                ] = "spacecraft"
        elif derived_from_utc:
            normalized["time_reference"] = "spacecraft"
        else:
            normalized["time_reference"] = "unspecified"

        if derived_from_utc:
            normalized["time_reference_source"] = "utc_plus_tdb_ut_seconds"
        elif "time_reference_source" not in normalized.columns:
            normalized["time_reference_source"] = "input_ephemeris"

    return normalized


def normalize_ephemeris_dataframe(
    ephem_df: Any,
    target: str | None = None,
    require_exact_tdb: bool = False,
) -> Any:
    """Return a DataFrame with canonical detector-local ephemeris columns."""

    normalized = _normalize_column_names(ephem_df)

    if target is not None and "target" not in normalized.columns:
        normalized["target"] = target

    normalized = _ensure_exact_tdb_column(normalized, require_exact_tdb=require_exact_tdb)

    for numeric_column in ("sector", "camera", "ccd", "x", "y", "time_utc_jd", "tdb_ut_seconds"):
        if numeric_column in normalized.columns:
            if numeric_column == "time_utc_jd":
                normalized[numeric_column] = _coerce_time_series_to_jd(normalized[numeric_column])
            else:
                normalized[numeric_column] = normalized[numeric_column].astype(float)

    if "sector" in normalized.columns:
        normalized["sector"] = normalized["sector"].astype(int)
    if "camera" in normalized.columns:
        normalized["camera"] = normalized["camera"].astype(int)
    if "ccd" in normalized.columns:
        normalized["ccd"] = normalized["ccd"].astype(int)

    if "time_tdb_jd_minus_2457000" in normalized.columns:
        normalized = normalized.sort_values("time_tdb_jd_minus_2457000").reset_index(drop=True)
    return normalized


def get_time_reference(ephem_df: Any, require_known: bool = False) -> str:
    """Return the unique normalized time-reference label for an ephemeris."""

    normalized = normalize_ephemeris_dataframe(ephem_df, require_exact_tdb=False)
    if "time_reference" not in normalized.columns:
        if require_known:
            raise ValueError(
                "Ephemeris is missing the normalized `time_reference` column. "
                "Re-export a normalized CSV or provide an explicit barycentric choice."
            )
        return "unspecified"

    references = {
        str(value)
        for value in normalized["time_reference"].map(_normalize_time_reference_value).dropna().unique().tolist()
    }
    if not references:
        if require_known:
            raise ValueError(
                "Ephemeris `time_reference` is empty. Provide `spacecraft` or `barycentric`."
            )
        return "unspecified"
    if len(references) != 1:
        raise ValueError(
            "Ephemeris mixes multiple `time_reference` values. "
            f"Found {sorted(references)!r}."
        )
    reference = next(iter(references))
    if require_known and reference == "unspecified":
        raise ValueError(
            "Ephemeris time reference is unspecified. For real-data work, export a normalized "
            "CSV with `time_reference=spacecraft` or `time_reference=barycentric`, or pass an explicit override."
        )
    return reference


def load_ephemeris_csv(
    path: str | Path,
    target: str | None = None,
    require_exact_tdb: bool = True,
) -> Any:
    """Load an ephemeris CSV and normalize it into a recovery-ready DataFrame."""

    pandas = _require_pandas()
    loaded = pandas.read_csv(Path(path))
    return normalize_ephemeris_dataframe(
        loaded,
        target=target,
        require_exact_tdb=require_exact_tdb,
    )


def _coerce_query_result_to_dataframe(
    result: Any,
    target: str | None = None,
    require_exact_tdb: bool = True,
) -> Any:
    pandas = _require_pandas()
    metadata: dict[str, Any] = {}
    if isinstance(result, tuple) and result:
        first = result[0]
        if len(result) > 1 and isinstance(result[1], dict):
            metadata = dict(result[1])
        result = first
    if isinstance(result, pandas.DataFrame):
        dataframe = result.copy()
        if "time" not in dataframe.columns and getattr(dataframe.index, "name", None) == "time":
            dataframe = dataframe.reset_index()
        if (
            "time" in dataframe.columns
            and "time_utc_jd" not in dataframe.columns
            and "time_tdb_jd" not in dataframe.columns
            and "time_tdb_jd_minus_2457000" not in dataframe.columns
        ):
            dataframe = dataframe.rename(columns={"time": "time_utc_jd"})
        normalized = normalize_ephemeris_dataframe(
            dataframe,
            target=target,
            require_exact_tdb=require_exact_tdb,
        )
        for key, value in metadata.items():
            if key not in normalized.columns:
                normalized[key] = value
        return normalized
    if hasattr(result, "to_pandas"):
        return normalize_ephemeris_dataframe(
            result.to_pandas(),
            target=target,
            require_exact_tdb=require_exact_tdb,
        )
    if isinstance(result, dict):
        return normalize_ephemeris_dataframe(
            pandas.DataFrame(result),
            target=target,
            require_exact_tdb=require_exact_tdb,
        )
    if isinstance(result, (list, tuple)):
        return normalize_ephemeris_dataframe(
            pandas.DataFrame(result),
            target=target,
            require_exact_tdb=require_exact_tdb,
        )
    raise TypeError(
        f"Unsupported tess_ephem result type: {type(result)!r}. "
        "Expected a DataFrame-like object or a mapping/list of rows."
    )


def query_tess_ephem(
    target: str = "Sedna",
    require_exact_tdb: bool = True,
    **query_kwargs: Any,
) -> Any:
    """Query a target ephemeris through tess_ephem if the package is available."""

    module = _require_tess_ephem()
    _require_pandas()

    module_ephem = getattr(module, "ephem", None)
    if callable(module_ephem):
        direct_kwargs = {
            name: value
            for name, value in query_kwargs.items()
            if name in {"time", "sector", "time_step", "id_type", "interpolation_step", "orbital_elements"}
        }
        result = module_ephem(target=target, **direct_kwargs)
        return _coerce_query_result_to_dataframe(
            result,
            target=target,
            require_exact_tdb=require_exact_tdb,
        )

    candidate_callables: list[tuple[str, Any]] = []
    for container_name in ("", "core", "ephem"):
        container = module if not container_name else getattr(module, container_name, None)
        if container is None:
            continue
        for function_name in (
            "query_ephemeris",
            "get_ephemeris",
            "ephemeris",
            "query_visibility",
            "get_visibility",
            "tess_ephem",
        ):
            func = getattr(container, function_name, None)
            if callable(func):
                label = f"tess_ephem{'.' + container_name if container_name else ''}.{function_name}"
                candidate_callables.append((label, func))

    attempted: list[str] = []
    for label, func in candidate_callables:
        call_patterns = (
            lambda: func(target=target, **query_kwargs),
            lambda: func(target, **query_kwargs),
        )
        for call in call_patterns:
            try:
                result = call()
            except TypeError as exc:
                attempted.append(f"{label}: {exc}")
                continue
            return _coerce_query_result_to_dataframe(
                result,
                target=target,
                require_exact_tdb=require_exact_tdb,
            )

    if candidate_callables:
        attempted_message = "; ".join(attempted[-4:])
        raise NotImplementedError(
            "tess_ephem is installed, but this wrapper could not find a compatible query "
            f"function. Recent attempts: {attempted_message}. "
            "Pass a local CSV for now or extend `query_tess_ephem` for your environment."
        )

    raise NotImplementedError(
        "tess_ephem is installed, but no known query function was found. "
        "Pass a local CSV for now or extend `query_tess_ephem` for your environment."
    )


def select_detector_track(
    ephem_df: Any,
    sector: int,
    camera: int,
    ccd: int,
    target: str | None = None,
    require_exact_tdb: bool = True,
    require_single_detector: bool = True,
) -> Any:
    """Filter an ephemeris to a single detector-local track."""

    pandas = _require_pandas()
    normalized = normalize_ephemeris_dataframe(
        ephem_df,
        target=target,
        require_exact_tdb=require_exact_tdb,
    )
    filtered = normalized.copy()

    if "target" in filtered.columns and target is not None:
        filtered = filtered[filtered["target"].astype(str).str.lower() == target.lower()]
    for column_name, value in (("sector", sector), ("camera", camera), ("ccd", ccd)):
        if column_name not in filtered.columns:
            raise KeyError(
                f"Ephemeris is missing required `{column_name}` column after normalization."
            )
        filtered = filtered[filtered[column_name].astype(int) == int(value)]

    filtered = filtered.reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            f"No ephemeris rows matched sector={sector}, camera={camera}, ccd={ccd}."
        )

    if require_single_detector:
        validate_single_detector_track(filtered, require_exact_tdb=require_exact_tdb)

    return pandas.DataFrame(filtered)


def validate_single_detector_track(
    ephem_df: Any,
    require_exact_tdb: bool = True,
) -> None:
    """Raise if the ephemeris does not stay on a single detector or lacks key columns."""

    normalized = normalize_ephemeris_dataframe(ephem_df, require_exact_tdb=require_exact_tdb)
    if normalized.empty:
        raise ValueError("Ephemeris selection is empty; nothing to validate.")

    for column_name in ("sector", "camera", "ccd"):
        if column_name not in normalized.columns:
            raise KeyError(f"Ephemeris is missing `{column_name}`.")
        unique_count = normalized[column_name].nunique(dropna=False)
        if unique_count != 1:
            raise ValueError(
                f"Selected ephemeris spans multiple `{column_name}` values; expected a single detector track."
            )

    if require_exact_tdb and "time_tdb_jd_minus_2457000" not in normalized.columns:
        raise ValueError(
            "Ephemeris is missing the canonical `time_tdb_jd_minus_2457000` column after normalization."
        )

    for coordinate in ("x", "y"):
        if coordinate not in normalized.columns:
            raise KeyError(f"Ephemeris is missing `{coordinate}` pixel coordinates.")
        if normalized[coordinate].isna().any():
            raise ValueError(f"Ephemeris contains missing `{coordinate}` pixel coordinates.")


def summarize_track(ephem_df: Any) -> dict[str, Any]:
    """Return a concise summary dictionary for a detector-local ephemeris track."""

    normalized = normalize_ephemeris_dataframe(ephem_df, require_exact_tdb=False)
    summary: dict[str, Any] = {"rows": int(len(normalized))}
    for column_name in ("target", "sector", "camera", "ccd", "time_scale", "time_reference"):
        if column_name in normalized.columns and not normalized.empty:
            summary[column_name] = normalized.iloc[0][column_name]
    if "time_tdb_jd_minus_2457000" in normalized.columns:
        summary["time_start"] = float(normalized["time_tdb_jd_minus_2457000"].min())
        summary["time_stop"] = float(normalized["time_tdb_jd_minus_2457000"].max())
    if "x" in normalized.columns:
        summary["x_min"] = float(normalized["x"].min())
        summary["x_max"] = float(normalized["x"].max())
    if "y" in normalized.columns:
        summary["y_min"] = float(normalized["y"].min())
        summary["y_max"] = float(normalized["y"].max())
    return summary


def export_ephemeris_csv(ephem_df: Any, output_path: str | Path) -> Path:
    """Write a normalized, recovery-ready ephemeris CSV to disk."""

    normalized = normalize_ephemeris_dataframe(ephem_df, require_exact_tdb=True)
    canonical_first = [
        "target",
        "sector",
        "camera",
        "ccd",
        "time_tdb_jd_minus_2457000",
        "time_tdb_jd",
        "time_scale",
        "time_reference",
        "time_reference_source",
        "time_utc_jd",
        "tdb_ut_seconds",
        "x",
        "y",
    ]
    ordered_columns = [
        column for column in canonical_first if column in normalized.columns
    ] + [
        column for column in normalized.columns if column not in canonical_first
    ]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized.loc[:, ordered_columns].to_csv(destination, index=False)
    return destination
