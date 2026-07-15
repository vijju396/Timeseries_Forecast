import hashlib
import json
import math
from pathlib import Path

from services.data_service import save_json


DATA_STUDIO_ANALYTICS_SCHEMA_VERSION = "data-studio-analytics-v1"
MISSING_DIMENSION_TOKEN = "__data_studio_missing__"


def build_dataset_profile(cleaned_df, raw_df=None, cleaning_stats=None):
    import numpy as np
    import pandas as pd

    stats = dict(cleaning_stats or {})
    frame = cleaned_df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    target = pd.to_numeric(frame["Click Count"], errors="coerce")
    raw_rows = int(stats.get("raw_rows", len(raw_df) if raw_df is not None else len(frame)))
    raw_columns = int(stats.get("raw_columns", len(raw_df.columns) if raw_df is not None else len(frame.columns)))
    missing_values = int(stats.get("source_missing_values", raw_df.isna().sum().sum() if raw_df is not None else 0))
    raw_rows = int(stats.get("raw_rows", raw_rows))
    valid_rows = int(stats.get("valid_rows", raw_rows))
    rows_removed = int(stats.get("rows_removed", max(raw_rows - valid_rows, 0)))
    rows_aggregated = int(stats.get("rows_aggregated", 0))
    calendar_rows_generated = int(stats.get("calendar_rows_generated", 0))
    unresolved_generated_rows_removed = int(stats.get("unresolved_generated_rows_removed", 0))
    training_rows = int(stats.get("training_rows", len(frame)))
    expected_training_rows = valid_rows - rows_aggregated + calendar_rows_generated - unresolved_generated_rows_removed

    profile = {
        "columns": {
            "names": [str(column) for column in (raw_df.columns if raw_df is not None else frame.columns)],
            "physical_types": {str(column): str(dtype) for column, dtype in (raw_df.dtypes.items() if raw_df is not None else frame.dtypes.items())},
            "missing_counts": {str(column): int(count) for column, count in (raw_df.isna().sum().items() if raw_df is not None else frame.isna().sum().items())},
            "distinct_counts": {str(column): int(count) for column, count in (raw_df.nunique(dropna=True).items() if raw_df is not None else frame.nunique(dropna=True).items())},
        },
        "summary": {
            "total_rows": raw_rows,
            "total_columns": raw_columns,
            "raw_rows": raw_rows,
            "valid_rows": valid_rows,
            "rows_removed": rows_removed,
            "rows_aggregated": rows_aggregated,
            "calendar_rows_generated": calendar_rows_generated,
            "training_rows": training_rows,
            "missing_values_imputed": int(stats.get("missing_values_imputed", 0)),
            "missing_timestamps_generated": int(stats.get("missing_timestamps_generated", calendar_rows_generated)),
            "unresolved_generated_rows_removed": unresolved_generated_rows_removed,
            "source_missing_values": missing_values,
            "row_accounting_expected": expected_training_rows,
            "row_accounting_reconciled": training_rows == expected_training_rows,
            # Compatibility fields remain numerically honest for older clients.
            "cleaned_rows": training_rows,
            "removed_invalid_rows": rows_removed,
            "missing_values": missing_values,
            "duplicate_date_rows": 0,
            "filled_gap_rows": calendar_rows_generated,
            "negative_target_rows": int(stats.get("negative_target_rows", 0)),
            "series_count": int(stats.get("series_count", 1 if len(frame) else 0)),
            "stores": int(frame["Store"].nunique()) if "Store" in frame else 0,
            "items": int(frame["Item"].nunique()) if "Item" in frame else 0,
        },
        "preprocessing": stats,
        "date_range": {
            "start": frame["Date"].min().date().isoformat(),
            "end": frame["Date"].max().date().isoformat(),
            "frequency": _infer_frequency(frame["Date"]),
        },
        "target_stats": {
            "min": _round(target.min()),
            "max": _round(target.max()),
            "average": _round(target.mean()),
            "median": _round(target.median()),
            "standard_deviation": _round(target.std()),
            "q1": _round(target.quantile(.25)),
            "q3": _round(target.quantile(.75)),
            "zero_ratio": round(float((target == 0).mean()), 4) if len(target) else 0,
            "negative_ratio": round(float((target < 0).mean()), 4) if len(target) else 0,
            "extreme_value_ratio": round(float((target.abs() > target.abs().quantile(.99)).mean()), 4) if len(target) else 0,
        },
        "profiling": {
            "duplicate_rows": int(raw_df.duplicated().sum()) if raw_df is not None else int(frame.duplicated().sum()),
            "numeric_columns": [str(column) for column in (raw_df.select_dtypes(include=np.number).columns if raw_df is not None else frame.select_dtypes(include=np.number).columns)],
            "candidate_identifiers": [str(column) for column in (raw_df.columns if raw_df is not None else frame.columns) if (raw_df[column].nunique(dropna=True) if raw_df is not None else frame[column].nunique(dropna=True)) >= max(20, len(frame) * .8)],
            "rows_per_series": int(round(len(frame) / max(int(stats.get("series_count", 1)), 1))),
        },
    }
    profile["quality"] = {"score": calculate_data_quality_score(profile), "label": "Data quality"}
    profile["preprocessing_metrics"] = _preprocessing_metrics(profile)
    profile["preprocessing_explanation"] = _preprocessing_explanation(profile)
    return profile


def calculate_data_quality_score(profile):
    summary = profile.get("summary", {})
    rows = max(int(summary.get("total_rows", 0)), 1)
    cells = max(rows * int(summary.get("total_columns", 1)), 1)
    penalties = (
        min(summary.get("missing_values", 0) / cells, 0.35) * 45
        + min(summary.get("rows_removed", summary.get("removed_invalid_rows", 0)) / rows, 0.35) * 35
        + min(summary.get("negative_target_rows", 0) / rows, 0.2) * 25
    )
    return int(round(max(0, min(100, 100 - penalties))))


def _preprocessing_metrics(profile):
    summary = profile["summary"]
    preprocessing = profile.get("preprocessing", {})
    date_range = profile["date_range"]
    target = profile["target_stats"]
    aggregation = str(preprocessing.get("aggregation", "sum")).upper()
    frequency = preprocessing.get("frequency") or date_range.get("frequency")
    dimensions = [item.get("display_name") for item in preprocessing.get("dimension_schema", [])]
    grain = f"{frequency} by {', '.join(dimensions)}" if dimensions else f"{frequency} single series"
    return [
        {"key": "raw_rows", "label": "Raw Rows", "value": summary["raw_rows"], "description": "Rows in the uploaded dataset.", "calculation": "Count of source records before parsing."},
        {"key": "valid_rows", "label": "Valid Rows", "value": summary["valid_rows"], "description": "Rows with a valid mapped timestamp and target.", "calculation": "Raw rows minus invalid mandatory records."},
        {"key": "rows_removed", "label": "Rows Removed", "value": summary["rows_removed"], "description": "Invalid dates, missing mandatory targets, or parsing failures.", "calculation": "Raw rows − valid rows."},
        {"key": "rows_aggregated", "label": "Rows Aggregated", "value": summary["rows_aggregated"], "description": f"Observations merged at forecast grain using {aggregation}.", "calculation": "Valid rows − unique timestamp/series grain rows."},
        {"key": "calendar_rows_generated", "label": "Calendar Rows Generated", "value": summary["calendar_rows_generated"], "description": "Synthetic rows created for missing timestamps.", "calculation": "Expanded calendar rows − post-aggregation rows."},
        {"key": "training_rows", "label": "Training Rows", "value": summary["training_rows"], "description": "Final rows available to model training.", "calculation": "Valid − aggregated + calendar-generated − unresolved generated rows."},
        {"key": "missing_values_imputed", "label": "Missing Values Imputed", "value": summary["missing_values_imputed"], "description": "Existing NULL values filled by the configured policy.", "calculation": "Filled values on source-backed rows only."},
        {"key": "missing_timestamps_generated", "label": "Missing Timestamps Generated", "value": summary["missing_timestamps_generated"], "description": "Entirely new timestamps introduced by calendar completion.", "calculation": "Count of synthetic timestamp rows."},
        {"key": "frequency", "label": "Frequency", "value": frequency, "description": "Inferred or configured observation cadence.", "calculation": "Date-spacing analysis after aggregation."},
        {"key": "forecast_grain", "label": "Forecast Grain", "value": grain, "description": "Timestamp and mapped dimensions defining one observation.", "calculation": "Frequency plus mapped dimension roles."},
        {"key": "date_range", "label": "Date Range", "value": f"{date_range.get('start')} to {date_range.get('end')}", "description": "Training data time span.", "calculation": "Minimum and maximum training timestamps."},
        {"key": "series_count", "label": "Series Count", "value": summary["series_count"], "description": "Distinct mapped dimension combinations.", "calculation": "Unique canonical series keys."},
        {"key": "target_statistics", "label": "Target Statistics", "value": f"Avg {_compact(target.get('average') or 0)}", "description": f"Range {_compact(target.get('min') or 0)} to {_compact(target.get('max') or 0)}.", "calculation": "Descriptive statistics over final training rows."},
    ]


def _preprocessing_explanation(profile):
    summary = profile["summary"]
    preprocessing = profile.get("preprocessing", {})
    aggregation = str(preprocessing.get("aggregation", "sum")).upper()
    imputation = preprocessing.get("imputation", {}).get("method", "leave_missing").replace("_", " ")
    reasons = preprocessing.get("rows_removed_reasons", {})
    removal_detail = (
        f" ({reasons.get('invalid_date_only', 0):,} invalid date only, "
        f"{reasons.get('invalid_target_only', 0):,} invalid target only, "
        f"{reasons.get('invalid_date_and_target', 0):,} invalid in both fields)"
    ) if reasons else ""
    parts = [
        f"Uploaded rows: {summary['raw_rows']:,}.",
        f"Rows removed: {summary['rows_removed']:,} invalid mandatory records{removal_detail}.",
        f"Rows aggregated: {summary['rows_aggregated']:,} observations combined at forecast grain using {aggregation}.",
        f"Calendar rows generated: {summary['calendar_rows_generated']:,} synthetic timestamps required for continuous {profile['date_range'].get('frequency')} frequency.",
        f"Training dataset: {summary['training_rows']:,} rows.",
        f"Existing missing values imputed: {summary['missing_values_imputed']:,} using {imputation}.",
    ]
    if summary.get("unresolved_generated_rows_removed"):
        parts.append(f"Unresolved generated rows removed: {summary['unresolved_generated_rows_removed']:,}.")
    parts.append("Row accounting reconciles exactly." if summary.get("row_accounting_reconciled") else "Row accounting requires review.")
    return " ".join(parts)


def build_enrichment_payload(cleaned_df, profile=None, raw_df=None):
    import numpy as np
    import pandas as pd

    frame = cleaned_df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Click Count"] = pd.to_numeric(frame["Click Count"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Click Count"])
    profile = profile or build_dataset_profile(frame, raw_df=raw_df)

    trend = frame.groupby("Date", as_index=False)["Click Count"].sum().sort_values("Date")
    if len(trend) > 500:
        sample_step = int(math.ceil(len(trend) / 500))
        trend = trend.iloc[::sample_step]

    values = frame["Click Count"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    try:
        if len(values) and float(values.min()) == float(values.max()):
            edge = float(values[0])
            counts = np.array([len(values)])
            edges = np.array([edge, np.nextafter(edge, math.inf)])
        else:
            counts, edges = np.histogram(values, bins=min(20, max(5, int(math.sqrt(len(values))))))
    except (ValueError, OverflowError):
        counts = np.array([len(values)])
        edge = float(values[0]) if len(values) else 0.0
        edges = np.array([edge, np.nextafter(edge, math.inf)])
    seasonality = frame.assign(period=frame["Date"].dt.month_name().str[:3]).groupby("period", as_index=False)["Click Count"].mean()
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    seasonality["order"] = seasonality["period"].map({name: index for index, name in enumerate(month_order)})
    seasonality = seasonality.sort_values("order")

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_count = int(((values < lower) | (values > upper)).sum())

    payload = {
        **profile,
        "missing_by_column": [
            {"column": str(column), "count": int(count)}
            for column, count in (raw_df.isna().sum().items() if raw_df is not None else frame.isna().sum().items())
        ],
        "trend": [{"date": row.Date.date().isoformat(), "value": _round(row._2)} for row in trend.itertuples()],
        "distribution": [
            {"label": f"{_compact(edges[index])} - {_compact(edges[index + 1])}", "count": int(count)}
            for index, count in enumerate(counts)
        ],
        "seasonality": [{"period": row.period, "value": _round(row._2)} for row in seasonality[["period", "Click Count"]].itertuples()],
        "top_stores": _top_dimension(frame, "Store"),
        "top_items": _top_dimension(frame, "Item"),
        "top_dimensions": _top_mapped_dimensions(frame, profile.get("preprocessing", {}).get("dimension_schema", [])),
        "outliers": {"count": outlier_count, "lower_bound": _round(lower), "upper_bound": _round(upper)},
    }
    return payload


def save_enrichment(payload, dataset_id=None):
    enriched = {**payload, "dataset_id": dataset_id}
    return save_json("enrichment.json", enriched)


def build_data_studio_analytics(dataset, params=None):
    import pandas as pd

    params = params or {}
    adapted = dataset.get("adapted") or {}
    dataset_id = dataset.get("id") or adapted.get("dataset_id")
    artifact_id = adapted.get("artifact_id")
    if params.get("dataset_id") and params["dataset_id"] != dataset_id:
        raise ValueError("The requested dataset is no longer active.")
    if params.get("artifact_id") and params["artifact_id"] != artifact_id:
        raise ValueError("The requested processed artifact is no longer active.")
    cleaned_path = Path(adapted.get("cleaned_path", ""))
    if not artifact_id or not cleaned_path.is_file():
        raise ValueError("Map the active dataset before loading Data Studio analytics.")

    dimension_schema = (adapted.get("dimension_schema") or [])[:2]
    columns = ["Date", "Click Count"] + [item["canonical_column"] for item in dimension_schema]
    frame = pd.read_csv(cleaned_path, usecols=lambda column: column in columns)
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Processed analytics columns are unavailable: {', '.join(missing_columns)}")

    requested_raw = params.get("dimensions") or params.get("dimension_filters") or "{}"
    try:
        requested_raw = json.loads(requested_raw) if isinstance(requested_raw, str) else requested_raw
    except (TypeError, ValueError) as exc:
        raise ValueError("Dimension filters are malformed.") from exc
    requested_raw = requested_raw if isinstance(requested_raw, dict) else {}
    allowed_ids = {item["id"] for item in dimension_schema}
    unknown = sorted(set(requested_raw) - allowed_ids)
    if unknown:
        raise ValueError(f"Unknown mapped dimension filter: {', '.join(unknown)}")
    requested = {item["id"]: requested_raw.get(item["id"]) or None for item in dimension_schema}

    option_schema = []
    option_frame = frame
    for index, dimension in enumerate(dimension_schema):
        column = dimension["canonical_column"]
        if index:
            previous = dimension_schema[index - 1]
            previous_value = requested.get(previous["id"])
            if previous_value:
                option_frame = _filter_dimension(option_frame, previous["canonical_column"], previous_value)
        option_schema.append({
            **dimension,
            "column_id": dimension["id"],
            "aggregate_label": _aggregate_label(dimension.get("display_name")),
            "values": _dimension_values(option_frame[column]),
        })

    filtered = frame
    resolved = {}
    for dimension in dimension_schema:
        value = requested.get(dimension["id"])
        if value:
            filtered = _filter_dimension(filtered, dimension["canonical_column"], value)
            if not filtered.empty:
                resolved[dimension["id"]] = value

    normalized_request = {key: requested[key] for key in sorted(requested) if requested[key]}
    cache_source = {
        "dataset_id": dataset_id, "artifact_id": artifact_id,
        "dimensions": normalized_request, "schema": DATA_STUDIO_ANALYTICS_SCHEMA_VERSION,
    }
    cache_key = hashlib.sha256(json.dumps(cache_source, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    date_values = pd.to_datetime(filtered["Date"], errors="coerce").dropna()
    dimension_columns = [item["canonical_column"] for item in dimension_schema]
    matched_series = int(filtered[dimension_columns].drop_duplicates().shape[0]) if dimension_columns and len(filtered) else (1 if len(filtered) else 0)
    ownership = {
        "ok": True, "available": not filtered.empty, "request_id": params.get("request_id"),
        "dataset_id": dataset_id, "artifact_id": artifact_id,
        "requested_dimension_filters": normalized_request,
        "resolved_dimension_filters": resolved,
        "matched_row_count": int(len(filtered)), "matched_series_count": matched_series,
        "effective_date_range": {
            "start": date_values.min().date().isoformat() if not date_values.empty else None,
            "end": date_values.max().date().isoformat() if not date_values.empty else None,
        },
        "dimension_schema": option_schema, "cache_key": cache_key,
        "analytics_schema_version": DATA_STUDIO_ANALYTICS_SCHEMA_VERSION,
    }
    if filtered.empty:
        return {**ownership, "code": "no_data", "message": "No analytical data matches the selected series.", "analytics": _empty_analytics()}
    analytics = build_enrichment_payload(filtered, profile={"preprocessing": {"dimension_schema": dimension_schema}})
    return {**ownership, "analytics": {key: analytics[key] for key in ("trend", "distribution", "missing_by_column", "seasonality", "top_dimensions", "top_stores", "top_items", "outliers")}}


def _filter_dimension(frame, column, value):
    if value == MISSING_DIMENSION_TOKEN:
        return frame[frame[column].isna()]
    return frame[frame[column].notna() & frame[column].astype(str).eq(str(value))]


def _dimension_values(series):
    values = [{"value": str(value), "label": str(value)} for value in sorted(series.dropna().unique(), key=lambda value: str(value).casefold())]
    if series.isna().any():
        values.append({"value": MISSING_DIMENSION_TOKEN, "label": "(Missing)"})
    return values


def _aggregate_label(display_name):
    label = str(display_name or "").strip()
    if not label:
        return "All Series"
    lower = label.lower()
    if lower.endswith("y") and len(label) > 1 and lower[-2] not in "aeiou":
        plural = f"{label[:-1]}ies"
    elif lower.endswith(("s", "x", "z", "ch", "sh")):
        plural = f"{label}es" if not lower.endswith("s") else label
    else:
        plural = f"{label}s"
    return f"All {plural}"


def _empty_analytics():
    return {"trend": [], "distribution": [], "missing_by_column": [], "seasonality": [], "top_dimensions": [], "top_stores": [], "top_items": [], "outliers": {"count": 0, "lower_bound": None, "upper_bound": None}}


def _top_dimension(frame, column):
    if column not in frame or frame[column].nunique() <= 1:
        return []
    grouped = frame.groupby(column, as_index=False)["Click Count"].sum().nlargest(8, "Click Count")
    return [{"label": str(row[0]), "value": _round(row[1])} for row in grouped.itertuples(index=False, name=None)]


def _top_mapped_dimensions(frame, dimension_schema):
    result = []
    for dimension in dimension_schema or []:
        column = dimension.get("canonical_column")
        if not column or column not in frame or frame[column].nunique(dropna=False) <= 1:
            continue
        result.append({
            "id": dimension.get("id"),
            "display_name": dimension.get("display_name") or dimension.get("source_column") or dimension.get("id"),
            "values": _top_dimension(frame, column),
        })
    return result


def _infer_frequency(dates):
    import pandas as pd

    unique = pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values()
    if len(unique) >= 3:
        inferred = pd.infer_freq(unique)
        if inferred:
            return inferred
    gaps = unique.diff().dt.total_seconds().dropna()
    if gaps.empty:
        return "D"
    median = gaps.median()
    if median >= 27 * 86400:
        return "MS"
    if median >= 6 * 86400:
        return "W"
    if median >= 86400:
        return "D"
    if median >= 3600:
        return "h"
    if median >= 60:
        return "min"
    return "D"


def _round(value):
    return round(float(value), 2) if value is not None and not math.isnan(float(value)) else None


def _compact(value):
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.1f}"
