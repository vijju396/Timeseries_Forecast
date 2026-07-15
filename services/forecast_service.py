import math
import json
import hashlib
import os
import re
import uuid
from collections import OrderedDict
from datetime import datetime

from services.metrics_service import calculate_metrics, get_champion, get_metric_lookup, get_model_metrics
from services.data_service import current_dataset_id, load_json, preprocessed_path
from services.series_service import aggregate_series_identity, canonical_dimension_filters, canonical_series_key, series_key_hash


class ForecastRequestError(ValueError):
    status_code = 400


DEFAULT_FILTERS = {
    "datasets": ["Current Dataset"],
    "stores": ["All Stores"],
    "items": ["All Items"],
    "granularities": ["Daily", "Weekly", "Monthly"],
    "horizons": ["7 days", "14 days", "30 days", "60 days", "90 days", "4 weeks", "13 weeks", "26 weeks", "52 weeks", "6 months", "12 months", "18 months", "24 months"],
    "start_date": "",
    "end_date": "",
}
HORIZONS_BY_GRANULARITY = {
    "Hourly": [(24, "hours"), (48, "hours"), (7, "days"), (14, "days")],
    "Daily": [(7, "days"), (14, "days"), (30, "days"), (60, "days"), (90, "days"), (4, "weeks"), (13, "weeks"), (26, "weeks"), (52, "weeks")],
    "Weekly": [(4, "weeks"), (13, "weeks"), (26, "weeks"), (52, "weeks")],
    "Monthly": [(6, "months"), (12, "months"), (18, "months"), (24, "months")],
    "Quarterly": [(4, "quarters"), (8, "quarters"), (12, "quarters")],
    "Yearly": [(1, "years"), (2, "years"), (3, "years")],
}
FORECAST_RESPONSE_SCHEMA_VERSION = "forecast-explorer-v3"


def get_filter_options():
    payload = load_json("forecast_data.json", {})
    filters = {**DEFAULT_FILTERS, **payload.get("filters", {})}
    filters.update({key: payload.get(key) for key in ("dataset_id", "artifact_id", "job_id")})
    filters["models"] = [metric["model"] for metric in get_model_metrics()]
    dimension_schema = payload.get("dimension_schema") or []
    filters["dimension_labels"] = [
        dimension.get("display_name") or dimension.get("source_column") or dimension.get("id")
        for dimension in dimension_schema[:2]
    ]
    options = payload.get("dimension_options") or {}
    filters["dimensions"] = [
        {"id": dimension["id"], "label": dimension.get("display_name") or dimension.get("source_column") or dimension["id"], "values": options.get(dimension["id"], [])}
        for dimension in dimension_schema
    ]
    filters["horizons_by_granularity"] = _horizon_options()
    filters["horizons"] = [row["label"] for row in filters["horizons_by_granularity"].get("Daily", [])]
    aggregate_hash = aggregate_series_identity()["series_key_hash"]
    filters["model_options"] = _model_options(payload, aggregate_hash)
    filters["global_champion"] = _global_champion_metadata(payload, filters["model_options"])
    return filters


def get_dependent_filter_options(params=None):
    import pandas as pd

    params = params or {}
    payload = load_json("forecast_data.json", {})
    active_id = current_dataset_id()
    if not active_id or (params.get("dataset_id") and params.get("dataset_id") != active_id):
        raise ForecastRequestError("The requested dataset is no longer active.")
    for key in ("artifact_id", "job_id"):
        if params.get(key) and params.get(key) != payload.get(key):
            raise ForecastRequestError(f"The requested {key} does not own the active forecast artifact.")
    schema = _effective_dimension_schema(payload)
    requested = canonical_dimension_filters(_parse_dimensions(params.get("dimensions")), schema)
    requested_hash = series_key_hash(canonical_series_key(requested, schema))
    if not schema:
        return _filter_model_payload(payload, active_id, requested_hash, [], params)
    try:
        frame = pd.read_csv(preprocessed_path("cleaned_training_input.csv"), usecols=[dimension["canonical_column"] for dimension in schema])
    except (OSError, ValueError) as exc:
        raise ForecastRequestError("The active mapped dimensions are unavailable.") from exc
    dimensions = []
    preceding = frame
    for dimension in schema:
        column = dimension["canonical_column"]
        values = sorted(preceding[column].dropna().astype(str).unique().tolist())
        dimensions.append({"id": dimension["id"], "label": dimension.get("display_name") or dimension.get("source_column") or dimension["id"], "values": values, "selected": requested.get(dimension["id"])})
        selected = requested.get(dimension["id"])
        if selected is not None:
            preceding = preceding[preceding[column].astype(str) == str(selected)]
    return _filter_model_payload(payload, active_id, requested_hash, dimensions, params)


def get_forecast(params=None):
    params = params or {}
    payload = load_json("forecast_data.json", {})
    filters = get_filter_options()
    models = filters.get("models", [])
    if not models:
        return _empty_forecast()

    requested_model = params.get("model_id") or params.get("model")
    if requested_model and requested_model not in models:
        raise ForecastRequestError(f"Model '{requested_model}' is not available for the active dataset.")
    dataset = params.get("dataset") or filters["datasets"][0]
    request_dataset_id = params.get("dataset_id")
    active_id = current_dataset_id()
    if request_dataset_id and request_dataset_id != active_id:
        raise ForecastRequestError("The requested dataset is no longer active.")
    store = params.get("store") or filters["stores"][0]
    item = params.get("item") or filters["items"][0]
    dimensions = _parse_dimensions(params.get("dimensions"))
    if dimensions:
        store = dimensions.get("store", dimensions.get("Store", store))
        item = dimensions.get("item", dimensions.get("Item", item))
    granularity = params.get("granularity") or "Daily"
    horizon_spec = _horizon_spec(params, granularity)
    horizon = horizon_spec["label"]
    horizon_points = horizon_spec["points"]
    start_date = params.get("historical_start") or params.get("startDate") or params.get("start_date") or ""
    end_date = params.get("historical_end") or params.get("endDate") or params.get("end_date") or ""
    dimension_schema = _effective_dimension_schema(payload)
    requested_dimensions = dict(dimensions)
    if dimension_schema:
        requested_dimensions.setdefault("store", store)
        requested_dimensions.setdefault("item", item)
    dimension_filters = canonical_dimension_filters(requested_dimensions, dimension_schema)
    requested_series_key = canonical_series_key(dimension_filters, dimension_schema)
    requested_series_hash = series_key_hash(requested_series_key)
    aggregate_identity = aggregate_series_identity()
    is_aggregate_request = requested_series_hash == aggregate_identity["series_key_hash"]
    selection_scope = "aggregate" if is_aggregate_request else "series"
    resolution = _resolve_model_selection(payload, requested_series_hash, requested_model, params.get("selection_source"), selection_scope)
    model = resolution["resolved_model_id"]
    selection_source = resolution["selection_source"]

    for requested_key, payload_key in (("artifact_id", "artifact_id"), ("job_id", "job_id")):
        requested = params.get(requested_key)
        if requested and requested != payload.get(payload_key):
            raise ForecastRequestError(f"The requested {requested_key} does not own the active forecast artifact.")

    _validate_choice(store, filters.get("stores", []), "store")
    _validate_choice(item, filters.get("items", []), "item")
    historical, validation, future = _sections(payload)
    last_actual_date = payload.get("last_actual_date") or (historical[-1].get("date") if historical else None)
    forecast_start_date = payload.get("forecast_start_date") or (future[0].get("date") if future else None)
    selected = historical if is_aggregate_request else _load_series_history(dimension_filters, dimension_schema)
    historical = _date_filter(selected, start_date, end_date)
    if not historical and (start_date or end_date):
        raise ForecastRequestError("The selected historical date range has no observations.")

    metric = get_metric_lookup().get(model, {})
    actual_rows = _aggregate([
        {"date": row.get("date"), "value": _float(row.get("actual"))} for row in historical
    ], granularity)

    selected_backtests = _select_backtests(payload, validation, model, requested_series_hash, is_aggregate_request)
    selected_backtests = _date_filter(selected_backtests, start_date, end_date)
    selected_backtests = _dedupe_backtests(selected_backtests)
    all_fitted_rows = _aggregate_prediction_rows(selected_backtests, granularity)
    backtest_total_count = len(all_fitted_rows)
    display_limit = _backtest_display_limit(granularity, len(actual_rows), horizon_points)
    fitted_rows = all_fitted_rows[-display_limit:]

    selected_future = _select_future_predictions(payload, future, model, requested_series_hash, is_aggregate_request)
    future_rows = _aggregate_prediction_rows(selected_future, granularity)[:horizon_points]

    actual_map = {row["key"]: row for row in actual_rows}
    fitted_map = {row["key"]: row for row in fitted_rows}
    future_map = {row["key"]: row for row in future_rows}
    keys = sorted(set(actual_map) | set(fitted_map) | set(future_map))
    labels = [_label_for_key(key, actual_map, fitted_map, future_map) for key in keys]
    actual = [_round(actual_map.get(key, {}).get("value")) for key in keys]
    fitted = [_round(fitted_map.get(key, {}).get("value")) for key in keys]
    future_values = [_round(future_map.get(key, {}).get("value")) for key in keys]
    lower = [_round(future_map.get(key, {}).get("lower")) for key in keys]
    upper = [_round(future_map.get(key, {}).get("upper")) for key in keys]
    forecast_start_index = next((index for index, value in enumerate(future_values) if value is not None), None)

    identity = {
        "dataset_id": payload.get("dataset_id") or active_id,
        "artifact_id": payload.get("artifact_id"),
        "job_id": payload.get("job_id"),
        "model_id": model,
        "dimension_filters": dimension_filters,
        "series_key": requested_series_key,
        "series_key_hash": requested_series_hash,
    }
    actual_identity = {key: value for key, value in identity.items() if key != "model_id"}
    historical_actual = [{"timestamp": row["date"], "value": _round(row["value"]), **actual_identity} for row in actual_rows]
    historical_prediction = [
        {
            "timestamp": row["date"], "value": _round(row["value"]),
            "actual": _round(row.get("actual")), "lower": _round(row.get("lower")),
            "upper": _round(row.get("upper")), "forecast_origin": row.get("forecast_origin"),
            "fold": row.get("fold"), "horizon_step": row.get("horizon_step"),
            "validation_method": row.get("validation_method") or "historical_backtest",
            "prediction_type": "historical_backtest", **identity,
        }
        for row in fitted_rows
    ]
    future_prediction = [
        {
            "timestamp": row["date"], "value": _round(row["value"]),
            "lower": _round(row.get("lower")), "upper": _round(row.get("upper")),
            "horizon_step": row.get("horizon_step") or index + 1,
            "prediction_type": "future_forecast", **identity,
        }
        for index, row in enumerate(future_rows)
    ]
    timeline = [{"timestamp": _label_for_key(key, actual_map, fitted_map, future_map), "actual": actual[index], "historical_prediction": fitted[index], "future_prediction": future_values[index], "lower": lower[index], "upper": upper[index]} for index, key in enumerate(keys)]

    comparable = [
        (actual_map[key]["value"], fitted_map[key]["value"])
        for key in keys if key in actual_map and key in fitted_map
    ]
    total_actual = sum(pair[0] for pair in comparable)
    total_fitted = sum(pair[1] for pair in comparable)
    variance = ((total_fitted - total_actual) / total_actual * 100) if total_actual else 0

    backtest_summary = _backtest_summary(actual_rows, all_fitted_rows)
    forecast_summary = _forecast_summary(future_rows)
    uncertainty = _uncertainty_summary(future_rows)
    warnings = _forecast_warnings(
        backtest_summary, historical_prediction, future_prediction, uncertainty,
        len(actual_rows), horizon_points, granularity,
    )
    seasonality = _seasonality(historical, granularity)
    model_context = _model_context(model, metric, payload, bool(future_prediction))
    model_context.update({"requested_model_id": requested_model or model, "resolved_model_id": model, "selection_source": selection_source, "selection_scope": selection_scope})
    if not is_aggregate_request and model_context.get("scope") == "aggregate_only":
        warnings.append("The selected model is aggregate-only and cannot publish predictions for an individual series.")
    historical_summary = _historical_summary(actual_rows)
    plain_language_explanation = _plain_language_explanation(
        historical_summary, backtest_summary, forecast_summary, uncertainty, model_context, warnings
    )
    explanation = {
        "direction": forecast_summary.get("trend", "stable"),
        "summary": plain_language_explanation,
        "reasons": warnings,
    }
    confidence_available = any(row.get("lower") is not None and row.get("upper") is not None for row in historical_prediction + future_prediction)
    scope_reason = "The selected model does not support this exact series." if not is_aggregate_request and model_context.get("scope") == "aggregate_only" else None
    unavailable_reasons = {
        "historical_backtest": None if historical_prediction else scope_reason or "No owned historical backtest rows match this series and model.",
        "future_forecast": None if future_prediction else scope_reason or "No owned future forecast rows match this series and model.",
        "prediction_interval": None if confidence_available else "The selected model has no owned prediction intervals for this series.",
    }
    return {
        "ok": True,
        "response_schema_version": FORECAST_RESPONSE_SCHEMA_VERSION,
        "request_id": params.get("request_id") or uuid.uuid4().hex,
        "dataset_id": payload.get("dataset_id") or active_id,
        "artifact_id": payload.get("artifact_id"),
        "job_id": payload.get("job_id"),
        "model_id": model,
        "requested_model_id": requested_model or model,
        "resolved_model_id": model,
        "selection_source": selection_source,
        "selection_scope": selection_scope,
        "series_key": requested_series_key,
        "requested_series_key": requested_series_key,
        "resolved_series_key": requested_series_key,
        "series_key_hash": requested_series_hash,
        "effective_filters": {"historical_start": start_date or (historical[0]["date"] if historical else None), "historical_end": end_date or (historical[-1]["date"] if historical else None), "dimension_filters": dimension_filters, "series_key_hash": requested_series_hash, "granularity": granularity, "future_horizon": horizon_points},
        "effective_granularity": granularity,
        "effective_horizon": {key: horizon_spec[key] for key in ("value", "unit", "points")},
        "metadata": {
            "target_display_name": payload.get("target_display_name", "Target"),
            "target_unit": payload.get("target_unit"), "frequency": payload.get("frequency", "unknown"),
            "aggregation": payload.get("aggregation", "sum"),
            "last_actual_timestamp": actual_rows[-1]["date"] if actual_rows else None,
            "first_historical_prediction_timestamp": fitted_rows[0]["date"] if fitted_rows else None,
            "first_future_timestamp": future_rows[0]["date"] if future_rows else None,
            "historical_point_count": len(actual_rows),
            "historical_prediction_count": len(fitted_rows),
            "historical_prediction_total_count": backtest_total_count,
            "historical_prediction_displayed_count": len(fitted_rows),
            "historical_prediction_coverage_percentage": backtest_summary.get("coverage_percentage", 0),
            "historical_prediction_source": _backtest_source(selected_backtests),
            "backtest_duplicate_policy": payload.get("backtest_duplicate_policy", "shortest_horizon"),
            "future_prediction_count": len(future_rows),
            "actual_start": actual_rows[0]["date"] if actual_rows else None,
            "actual_end": actual_rows[-1]["date"] if actual_rows else None,
            "historical_prediction_start": fitted_rows[0]["date"] if fitted_rows else None,
            "historical_prediction_end": fitted_rows[-1]["date"] if fitted_rows else None,
            "future_prediction_start": future_rows[0]["date"] if future_rows else None,
            "future_prediction_end": future_rows[-1]["date"] if future_rows else None,
            "confidence_level": payload.get("confidence_level") if confidence_available else None,
            "selected_model_id": model, "selected_dimension_filters": identity["dimension_filters"],
        },
        "filters": {"dataset": dataset, "store": store, "item": item, "dimensions": dimension_filters, "series_key_hash": requested_series_hash, "granularity": granularity, "horizon": horizon, "model": model, "model_id": model},
        "historical_actual": historical_actual,
        "historical_prediction": historical_prediction,
        "future_prediction": future_prediction,
        "timeline": timeline,
        "labels": labels,
        "actual": actual,
        "fitted": fitted,
        "future_forecast": future_values,
        "predicted": [future_values[i] if future_values[i] is not None else fitted[i] for i in range(len(keys))],
        "lower": lower,
        "upper": upper,
        "forecast_start_index": forecast_start_index,
        "last_actual_date": last_actual_date,
        "forecast_start_date": forecast_start_date,
        "forecast_end_date": future_rows[-1]["date"] if future_rows else None,
        "effective": {"matched_rows": len(historical), "historical_start": historical[0]["date"] if historical else None, "historical_end": historical[-1]["date"] if historical else None, "historical_points": len(actual_rows), "forecast_points": len(future_rows), "aggregation": "sum"},
        "sections": {"historical_points": len(actual_rows), "validation_points": len(fitted_rows), "future_points": len(future_rows)},
        "summary": {
            "points": len(keys),
            "future_points": len(future_rows),
            "total_actual": round(total_actual, 2),
            "total_predicted": round(total_fitted, 2),
            "variance_pct": round(variance, 2),
            "mape": metric.get("mape"),
            "accuracy": metric.get("accuracy"),
        },
        "backtest_summary": backtest_summary,
        "historical_summary": historical_summary,
        "forecast_summary": forecast_summary,
        "uncertainty": uncertainty,
        "drift_summary": {},
        "model_context": model_context,
        "plain_language_explanation": plain_language_explanation,
        "seasonality": seasonality,
        "explanation": explanation,
        "prediction_availability": {"model_scope": model_context.get("scope"), "historical_backtest": bool(historical_prediction), "future_forecast": bool(future_prediction), "series_supported": bool(is_aggregate_request or historical_prediction or future_prediction)},
        "availability": {"historical_backtest": bool(historical_prediction), "future_forecast": bool(future_prediction), "prediction_interval": confidence_available},
        "unavailable_reasons": unavailable_reasons,
        "model_scope": model_context.get("scope"),
        "historical_prediction_status": "available" if historical_prediction else "historical_prediction_unavailable",
        "future_prediction_status": "available" if future_prediction else "future_prediction_unavailable",
        "warnings": warnings,
        "cache_key": _cache_key(payload, dimension_filters, requested_series_hash, start_date, end_date, granularity, horizon_spec, model, selection_scope),
    }


def _select_backtests(payload, legacy_rows, model, requested_series_hash, is_aggregate_request):
    long_rows = payload.get("backtest_predictions") or []
    if long_rows:
        selected = []
        for row in long_rows:
            row_hash = row.get("series_key_hash") or aggregate_series_identity()["series_key_hash"]
            if row.get("model_id") != model or row_hash != requested_series_hash or not _row_owned_by_payload(row, payload):
                continue
            selected.append(
                {
                    "date": row.get("date") or row.get("timestamp"),
                    "value": _float(row.get("prediction", row.get("predicted", row.get("value")))),
                    "actual": _float(row.get("actual")),
                    "lower": _float(row.get("lower")),
                    "upper": _float(row.get("upper")),
                    "forecast_origin": row.get("forecast_origin"),
                    "fold": row.get("fold"),
                    "horizon_step": row.get("horizon_step"),
                    "validation_method": row.get("validation_method") or "rolling_origin",
                }
            )
        return selected

    if not is_aggregate_request:
        return []
    return [
        {
            "date": row.get("date"), "value": _prediction(row, model),
            "actual": _float(row.get("actual")),
            "lower": _confidence_bound(row, model, "lower"),
            "upper": _confidence_bound(row, model, "upper"),
            "forecast_origin": row.get("forecast_origin"), "fold": row.get("fold"),
            "horizon_step": row.get("horizon_step"),
            "validation_method": row.get("validation_method") or "chronological_holdout",
        }
        for row in legacy_rows
    ]


def _select_future_predictions(payload, legacy_rows, model, requested_series_hash, is_aggregate_request):
    long_rows = payload.get("future_predictions") or []
    if long_rows:
        return [
            {
                "date": row.get("date") or row.get("timestamp"),
                "value": _float(row.get("prediction", row.get("predicted", row.get("value")))),
                "lower": _float(row.get("lower")), "upper": _float(row.get("upper")),
                "horizon_step": row.get("horizon_step"),
            }
            for row in long_rows
            if row.get("model_id") == model and row.get("series_key_hash") == requested_series_hash and _row_owned_by_payload(row, payload)
        ]
    if not is_aggregate_request:
        return []
    return [
        {"date": row.get("date"), "value": _prediction(row, model), "lower": _confidence_bound(row, model, "lower"), "upper": _confidence_bound(row, model, "upper"), "horizon_step": row.get("horizon_step")}
        for row in legacy_rows
    ]


def _row_owned_by_payload(row, payload):
    return all(row.get(key) is not None and row.get(key) == payload.get(key) for key in ("dataset_id", "artifact_id", "job_id"))


def _dedupe_backtests(rows):
    """For duplicate timestamps, keep the shortest-horizon honest prediction."""
    selected = {}
    for row in rows:
        if not row.get("date") or row.get("value") is None:
            continue
        key = row["date"]
        step = row.get("horizon_step")
        rank = int(step) if str(step or "").isdigit() else 10**9
        current = selected.get(key)
        current_step = current.get("horizon_step") if current else None
        current_rank = int(current_step) if str(current_step or "").isdigit() else 10**9
        if current is None or rank < current_rank or (rank == current_rank and str(row.get("forecast_origin") or "") > str(current.get("forecast_origin") or "")):
            selected[key] = row
    return [selected[key] for key in sorted(selected)]


def _aggregate_prediction_rows(rows, granularity):
    buckets = OrderedDict()
    for row in rows:
        value = _float(row.get("value"))
        if not row.get("date") or value is None:
            continue
        lower, upper = _float(row.get("lower")), _float(row.get("upper"))
        if lower is not None and upper is not None and not (lower <= value <= upper):
            lower = upper = None
        key, label = _bucket(row["date"], granularity)
        bucket = buckets.setdefault(
            key,
            {
                "key": key, "label": label, "date": row["date"], "value": 0.0,
                "actual": 0.0, "actual_count": 0, "lower": 0.0, "upper": 0.0,
                "bound_count": 0, "value_count": 0, "forecast_origin": None,
                "fold": None, "horizon_step": None, "validation_method": row.get("validation_method"),
            },
        )
        bucket["value"] += value
        bucket["value_count"] += 1
        actual = _float(row.get("actual"))
        if actual is not None:
            bucket["actual"] += actual
            bucket["actual_count"] += 1
        if lower is not None and upper is not None:
            bucket["lower"] += lower
            bucket["upper"] += upper
            bucket["bound_count"] += 1
        bucket["date"] = max(bucket["date"], row["date"])
        origin = row.get("forecast_origin")
        if origin and (bucket["forecast_origin"] is None or origin > bucket["forecast_origin"]):
            bucket["forecast_origin"] = origin
        if row.get("fold") is not None:
            bucket["fold"] = row.get("fold") if bucket["fold"] is None else max(bucket["fold"], row.get("fold"))
        step = row.get("horizon_step")
        if step is not None:
            bucket["horizon_step"] = step if bucket["horizon_step"] is None else min(bucket["horizon_step"], step)
    result = []
    for bucket in buckets.values():
        if bucket.pop("actual_count") == 0:
            bucket["actual"] = None
        if bucket.pop("bound_count") != bucket["value_count"]:
            bucket["lower"] = bucket["upper"] = None
        bucket.pop("value_count", None)
        result.append(bucket)
    return result


def _backtest_display_limit(granularity, historical_points, horizon_points):
    defaults = {"Hourly": 168, "Daily": 90, "Weekly": 52, "Monthly": 24, "Quarterly": 16, "Yearly": 10}
    configured = os.getenv(f"FORECAST_BACKTEST_DISPLAY_{str(granularity).upper()}")
    default = defaults.get(granularity, max(12, min(90, horizon_points * 2)))
    limit = int(configured) if configured and configured.isdigit() else default
    return max(1, min(limit, max(1, historical_points)))


def _backtest_summary(actual_rows, prediction_rows):
    actual_map = {row["date"]: row["value"] for row in actual_rows if row.get("value") is not None}
    comparable = [
        (row["date"], actual_map[row["date"]], row["value"])
        for row in prediction_rows if row.get("date") in actual_map and row.get("value") is not None
    ]
    metrics = calculate_metrics([row[1] for row in comparable], [row[2] for row in comparable])
    absolute_errors = [abs(actual - predicted) for _date, actual, predicted in comparable]
    denominator = sum(abs(actual) for _date, actual, _predicted in comparable)
    wape = sum(absolute_errors) / denominator * 100 if denominator else None
    smape_parts = [2 * abs(actual - predicted) / (abs(actual) + abs(predicted)) for _date, actual, predicted in comparable if abs(actual) + abs(predicted)]
    smape = sum(smape_parts) / len(smape_parts) * 100 if smape_parts else None
    coverage = len({row[0] for row in comparable}) / len(actual_map) * 100 if actual_map else 0
    largest_index = max(range(len(absolute_errors)), key=absolute_errors.__getitem__) if absolute_errors else None
    minimum = max(2, int(os.getenv("FORECAST_MIN_BACKTEST_POINTS", "5")))
    return {
        "point_count": len(comparable), "mae": metrics.get("mae"), "rmse": metrics.get("rmse"),
        "wape": _round(wape), "smape": _round(smape), "bias": metrics.get("bias"),
        "largest_error_timestamp": comparable[largest_index][0] if largest_index is not None else None,
        "largest_absolute_error": _round(absolute_errors[largest_index]) if largest_index is not None else None,
        "coverage_percentage": round(coverage, 2), "minimum_reliable_points": minimum,
        "reliable": len(comparable) >= minimum,
    }


def _forecast_summary(rows):
    points = [(row.get("date"), _float(row.get("value"))) for row in rows]
    points = [(date, value) for date, value in points if date and value is not None]
    if not points:
        return {"first_value": None, "last_value": None, "absolute_change": None, "percentage_change": None, "minimum_value": None, "maximum_value": None, "peak_timestamp": None, "lowest_timestamp": None, "trend": "unavailable"}
    first, last = points[0][1], points[-1][1]
    absolute_change = last - first
    percentage_change = absolute_change / abs(first) * 100 if abs(first) > 1e-12 else None
    minimum = min(points, key=lambda row: row[1])
    maximum = max(points, key=lambda row: row[1])
    if percentage_change is None:
        trend = "increasing" if absolute_change > 0 else "decreasing" if absolute_change < 0 else "stable"
    elif abs(percentage_change) <= 2:
        spread = (maximum[1] - minimum[1]) / max(abs(first), 1e-12) * 100
        trend = "mixed" if spread > 10 else "stable"
    else:
        trend = "increasing" if percentage_change > 0 else "decreasing"
    return {"first_value": _round(first), "last_value": _round(last), "absolute_change": _round(absolute_change), "percentage_change": _round(percentage_change), "minimum_value": _round(minimum[1]), "maximum_value": _round(maximum[1]), "peak_timestamp": maximum[0], "lowest_timestamp": minimum[0], "trend": trend}


def _uncertainty_summary(rows):
    widths = [
        float(row["upper"]) - float(row["lower"])
        for row in rows
        if row.get("lower") is not None and row.get("upper") is not None and row["lower"] <= row["value"] <= row["upper"]
    ]
    if not widths:
        return {"available": False, "early_average_width": None, "late_average_width": None, "direction": "unavailable", "percentage_change": None}
    group = max(1, len(widths) // 3)
    early = sum(widths[:group]) / len(widths[:group])
    late = sum(widths[-group:]) / len(widths[-group:])
    change = (late - early) / abs(early) * 100 if abs(early) > 1e-12 else None
    direction = "stable" if change is None or abs(change) <= 5 else "widening" if change > 0 else "narrowing"
    return {"available": True, "early_average_width": _round(early), "late_average_width": _round(late), "direction": direction, "percentage_change": _round(change)}


def _forecast_warnings(backtest, historical_prediction, future_prediction, uncertainty, historical_count, horizon_count, granularity):
    warnings = []
    if not historical_prediction:
        warnings.append("No historical backtest predictions available for this selected series.")
    elif not backtest.get("reliable"):
        warnings.append(f"Only {backtest.get('point_count', 0)} historical prediction point(s) are available; accuracy is not yet reliable.")
    if historical_prediction and backtest.get("coverage_percentage", 0) < 20:
        warnings.append(f"Historical predictions cover only {backtest['coverage_percentage']:.1f}% of the selected period. Accuracy may not represent the full history.")
    if not any(row.get("lower") is not None and row.get("upper") is not None for row in historical_prediction + future_prediction):
        warnings.append("Prediction interval unavailable for this model.")
    if not future_prediction:
        warnings.append("No future forecast is available for this selected series and model.")
    if historical_count and horizon_count > historical_count:
        warnings.append("The forecast horizon is long relative to the selected historical range.")
    if granularity != "Daily":
        warnings.append(f"Values are aggregated at {granularity.lower()} granularity using the configured target aggregation.")
    if uncertainty.get("direction") == "widening":
        warnings.append("Forecast uncertainty increases at longer horizons.")
    return warnings


def _model_context(model, metric, payload, has_future):
    uses_exogenous = "_exog_" in model.lower()
    registry = next((row for row in payload.get("model_registry", []) if row.get("model_id") == model), {})
    limitations = []
    if not has_future:
        limitations.append("Future forecast is unavailable for the selected model.")
    if uses_exogenous:
        limitations.append("Only generated calendar features are available for future periods unless mapped scenarios are supplied.")
    return {
        "requested_model_id": model, "resolved_model_id": model, "model_id": model,
        "display_name": metric.get("model_label") or model.replace("_Predictions", "").replace("_", " "),
        "uses_exogenous_variables": uses_exogenous,
        "included_feature_groups": ["calendar"] if uses_exogenous else [],
        "future_feature_availability": {"calendar": "generated"} if uses_exogenous else {},
        "limitations": limitations,
        "scope": registry.get("scope") or metric.get("model_scope") or "aggregate_only",
        "supported_series_count": registry.get("supported_series_count", metric.get("supported_series_count", 0)),
        "total_series_count": registry.get("total_series_count", metric.get("total_series_count", payload.get("series_count", 1))),
        "series_failure_count": registry.get("series_failure_count", metric.get("series_failure_count", 0)),
        "dataset_id": payload.get("dataset_id"), "artifact_id": payload.get("artifact_id"), "job_id": payload.get("job_id"),
    }


def _historical_summary(rows):
    values = [_float(row.get("value")) for row in rows[-12:]]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return {"trend": "insufficient_data", "percentage_change": None}
    change = values[-1] - values[0]
    percentage = change / abs(values[0]) * 100 if abs(values[0]) > 1e-12 else None
    trend = "stable" if percentage is not None and abs(percentage) <= 2 else "increasing" if change > 0 else "decreasing" if change < 0 else "stable"
    return {"trend": trend, "percentage_change": _round(percentage)}


def _plain_language_explanation(history, backtest, forecast, uncertainty, model_context, warnings):
    history_text = {
        "increasing": "Recent historical values show an upward pattern.",
        "decreasing": "Recent historical values show a downward pattern.",
        "stable": "Recent historical values are broadly stable.",
    }.get(history.get("trend"), "There is not enough recent history to characterize the current pattern.")
    if backtest.get("reliable") and backtest.get("wape") is not None:
        backtest_text = f"The selected model follows the available historical backtest with WAPE of {backtest['wape']:.1f}%."
    elif backtest.get("point_count"):
        backtest_text = f"Only {backtest['point_count']} out-of-sample historical prediction point(s) are available, so accuracy should be treated cautiously."
    else:
        backtest_text = "Out-of-sample historical predictions are unavailable for this selection."
    trend = forecast.get("trend", "unavailable")
    change = forecast.get("percentage_change")
    change_text = f", ending approximately {abs(change):.1f}% {'above' if change >= 0 else 'below'} the first forecast point" if change is not None else ""
    forecast_text = f"The future forecast is {trend}{change_text}." if trend != "unavailable" else "A future forecast is unavailable."
    uncertainty_text = {
        "widening": "Forecast uncertainty increases at longer horizons.",
        "narrowing": "The model's empirical prediction interval narrows over the horizon.",
        "stable": "Forecast uncertainty remains broadly stable across the horizon.",
    }.get(uncertainty.get("direction"), "A prediction interval is unavailable, so uncertainty width cannot be assessed.")
    feature_text = " Generated calendar effects are included; future business or external drivers are not assumed without mapped scenario data." if model_context.get("uses_exogenous_variables") else " No external-variable effect is claimed by this model context."
    attention = f" Pay attention to: {warnings[0]}" if warnings else ""
    return f"{history_text} {backtest_text} {forecast_text} {uncertainty_text}{feature_text}{attention}"


def _backtest_source(rows):
    methods = {row.get("validation_method") for row in rows if row.get("validation_method")}
    return next(iter(methods)) if len(methods) == 1 else "multiple_validation_methods" if methods else None


def _confidence_bound(row, model, bound):
    bands = row.get("confidence_bands") or {}
    model_band = bands.get(model) if isinstance(bands, dict) else None
    if isinstance(model_band, dict):
        return _float(model_band.get(bound))
    return _float(row.get(bound))


def _cache_key(payload, dimension_filters, requested_series_hash, start_date, end_date, granularity, horizon, model, scope=None):
    parts = {
        "dataset_id": payload.get("dataset_id"), "artifact_id": payload.get("artifact_id"), "job_id": payload.get("job_id"),
        "dimension_filters": {key: dimension_filters[key] for key in sorted(dimension_filters)},
        "series_key_hash": requested_series_hash, "start": start_date, "end": end_date,
        "granularity": granularity, "horizon_value": horizon["value"], "horizon_unit": horizon["unit"],
        "model_id": model, "scope": scope or ("aggregate" if requested_series_hash == aggregate_series_identity()["series_key_hash"] else "series"), "response_schema_version": FORECAST_RESPONSE_SCHEMA_VERSION,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _horizon_options():
    return {
        granularity: [{"value": value, "unit": unit, "label": f"{value} {unit}"} for value, unit in options]
        for granularity, options in HORIZONS_BY_GRANULARITY.items()
    }


def _horizon_spec(params, granularity):
    if granularity not in HORIZONS_BY_GRANULARITY:
        raise ForecastRequestError(f"Unsupported granularity '{granularity}'.")
    raw_value = params.get("horizon_value")
    raw_unit = params.get("horizon_unit")
    if raw_value not in (None, "") or raw_unit not in (None, ""):
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ForecastRequestError("horizon_value must be a positive integer.") from exc
        unit = str(raw_unit or "").strip().lower()
    else:
        text = params.get("future_horizon") or params.get("horizon")
        if not text:
            value, unit = HORIZONS_BY_GRANULARITY[granularity][0]
        else:
            match = re.fullmatch(r"\s*(\d+)\s+([a-zA-Z]+)\s*", str(text))
            if not match:
                raise ForecastRequestError("The forecast horizon is invalid.")
            value, unit = int(match.group(1)), match.group(2).lower()
    allowed = set(HORIZONS_BY_GRANULARITY[granularity])
    if value <= 0 or (value, unit) not in allowed:
        valid = ", ".join(f"{number} {name}" for number, name in HORIZONS_BY_GRANULARITY[granularity])
        raise ForecastRequestError(f"Horizon '{value} {unit}' is incompatible with {granularity.lower()} granularity. Choose: {valid}.")
    multiplier = {("Hourly", "days"): 24, ("Daily", "weeks"): 7}.get((granularity, unit), 1)
    return {"value": value, "unit": unit, "points": value * multiplier, "label": f"{value} {unit}"}


def _model_options(payload, requested_series_hash):
    aggregate_hash = aggregate_series_identity()["series_key_hash"]
    metric_rows = get_model_metrics()
    metric_lookup = {row.get("model"): row for row in metric_rows}
    registries = payload.get("model_registry") or [
        {"model_id": row.get("model"), "display_name": row.get("model_label"), "scope": row.get("model_scope") or "aggregate_only"}
        for row in metric_rows
    ]
    options = []
    for registry in registries:
        model_id = registry.get("model_id")
        if not model_id:
            continue
        owned_backtests = [row for row in payload.get("backtest_predictions") or [] if row.get("model_id") == model_id and row.get("series_key_hash") == requested_series_hash and _row_owned_by_payload(row, payload)]
        owned_future = [row for row in payload.get("future_predictions") or [] if row.get("model_id") == model_id and row.get("series_key_hash") == requested_series_hash and _row_owned_by_payload(row, payload)]
        scope = registry.get("scope") or "aggregate_only"
        supported = bool(owned_backtests or owned_future) or (requested_series_hash == aggregate_hash and scope in {"aggregate_only", "single_series"})
        scoped_metrics = calculate_metrics(
            [row.get("actual") for row in owned_backtests],
            [row.get("predicted") for row in owned_backtests],
        ) if owned_backtests else {}
        ranking_value = scoped_metrics.get("mape")
        if ranking_value is None:
            ranking_value = metric_lookup.get(model_id, {}).get("mape")
        options.append({
            "model_id": model_id, "display_name": registry.get("display_name") or model_id,
            "scope": scope, "supports_selected_series": supported,
            "supports_backtest": bool(owned_backtests), "supports_future_forecast": bool(owned_future),
            "supports_intervals": any(row.get("lower") is not None and row.get("upper") is not None for row in owned_backtests + owned_future),
            "ranking_metric": "mape", "ranking_value": ranking_value,
        })
    return sorted(options, key=lambda row: (not row["supports_selected_series"], _finite_rank(row.get("ranking_value")), row["display_name"]))


def _filter_model_payload(payload, active_id, requested_hash, dimensions, params):
    options = _model_options(payload, requested_hash)
    aggregate_hash = aggregate_series_identity()["series_key_hash"]
    scope = "aggregate" if requested_hash == aggregate_hash else "series"
    champion = _global_champion_metadata(payload, options if scope == "aggregate" else _model_options(payload, aggregate_hash))
    recommended = next((row["model_id"] for row in options if row["supports_selected_series"]), None)
    return {
        "ok": True, "request_id": params.get("request_id"),
        "dataset_id": active_id, "artifact_id": payload.get("artifact_id"), "job_id": payload.get("job_id"),
        "dimensions": dimensions, "models": options, "horizons_by_granularity": _horizon_options(),
        "selection_scope": scope, "series_key_hash": requested_hash,
        "global_champion": champion, "recommended_model_id": champion.get("model_id") if scope == "aggregate" else recommended,
    }


def _global_champion_metadata(payload, aggregate_options=None):
    aggregate_options = aggregate_options if aggregate_options is not None else _model_options(payload, aggregate_series_identity()["series_key_hash"])
    supported = {row["model_id"]: row for row in aggregate_options if row.get("supports_selected_series")}
    champion = get_champion() or {}
    champion_id = champion.get("model")
    ownership_valid = all(not champion.get(key) or champion.get(key) == payload.get(key) for key in ("dataset_id", "artifact_id", "job_id"))
    selected = supported.get(champion_id) if ownership_valid else None
    if not selected:
        selected = next(iter(supported.values()), None)
        champion = next((row for row in get_model_metrics() if row.get("model") == (selected or {}).get("model_id")), {})
    if not selected:
        return {}
    return {
        "model_id": selected["model_id"], "display_name": selected["display_name"], "scope": "aggregate",
        "ranking_metric": "mape", "ranking_value": champion.get("mape", selected.get("ranking_value")),
        "dataset_id": payload.get("dataset_id"), "artifact_id": payload.get("artifact_id"), "job_id": payload.get("job_id"),
    }


def _resolve_model_selection(payload, requested_series_hash, requested_model, requested_source, scope):
    options = _model_options(payload, requested_series_hash)
    compatible = {row["model_id"]: row for row in options if row.get("supports_selected_series")}
    if requested_model in compatible:
        return {"resolved_model_id": requested_model, "selection_source": requested_source or ("global_champion" if scope == "aggregate" else "automatic_series_fallback")}
    if scope == "aggregate":
        selected = _global_champion_metadata(payload, options).get("model_id")
        source = "global_champion"
    else:
        selected = next(iter(compatible), None)
        source = "automatic_series_fallback"
    if not selected:
        if requested_model and any(row["model_id"] == requested_model for row in options):
            return {"resolved_model_id": requested_model, "selection_source": requested_source or source}
        raise ForecastRequestError("No compatible completed model is available for the selected series scope.")
    return {"resolved_model_id": selected, "selection_source": source}


def _finite_rank(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else math.inf
    except (TypeError, ValueError):
        return math.inf


def _parse_dimensions(value):
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k).strip(): v for k, v in value.items() if str(k).strip()}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        raise ForecastRequestError("dimensions must be valid JSON.")
    if not isinstance(parsed, dict):
        raise ForecastRequestError("dimensions must be an object.")
    return {str(k).strip(): v for k, v in parsed.items() if str(k).strip()}


def _effective_dimension_schema(payload):
    schema = payload.get("dimension_schema") or []
    if schema:
        return schema
    # Read-only compatibility for artifacts created before canonical dimension IDs existed.
    filters = payload.get("filters") or {}
    fallback = []
    if len(filters.get("stores") or []) > 1:
        fallback.append({"id": "dimension_1", "canonical_column": "Store", "source_column": "Store", "display_name": "Store / Region"})
    if len(filters.get("items") or []) > 1:
        fallback.append({"id": "dimension_2", "canonical_column": "Item", "source_column": "Item", "display_name": "Item / SKU"})
    return fallback


def _validate_choice(value, allowed, label):
    if value in (None, "", "All Stores", "All Items"):
        return
    if allowed and value not in allowed:
        raise ForecastRequestError(f"Unknown {label} '{value}'.")


def _load_series_history(dimension_filters, dimension_schema):
    """Load observed history for exactly one canonical series; never allocate aggregate predictions."""
    import pandas as pd

    try:
        frame = pd.read_csv(preprocessed_path("cleaned_training_input.csv"))
    except (OSError, ValueError):
        raise ForecastRequestError("The active mapped data is unavailable.")
    selected = frame
    schema_lookup = {dimension["id"]: dimension["canonical_column"] for dimension in dimension_schema or []}
    for dimension_id, requested_value in dimension_filters.items():
        column = schema_lookup.get(dimension_id)
        if not column or column not in selected:
            raise ForecastRequestError(f"Mapped dimension '{dimension_id}' is unavailable in the active artifact.")
        selected = selected[selected[column].astype(str) == str(requested_value)]
    if selected.empty:
        raise ForecastRequestError("No observations match the selected dimensions.")
    date_col = "Date" if "Date" in selected.columns else "date"
    target_col = "Click Count" if "Click Count" in selected.columns else "target"
    selected[date_col] = pd.to_datetime(selected[date_col], errors="coerce")
    selected[target_col] = pd.to_numeric(selected[target_col], errors="coerce")
    selected = selected.dropna(subset=[date_col, target_col])
    return [{"date": _timestamp_text(row[date_col]), "actual": float(row[target_col])} for _, row in selected.iterrows()]


def _sections(payload):
    if any(key in payload for key in ("historical", "validation", "future")):
        return payload.get("historical", []), payload.get("validation", []), payload.get("future", [])
    rows = payload.get("series", [])
    validation = [row for row in rows if row.get("actual") is not None]
    future = [row for row in rows if row.get("actual") is None]
    return validation, validation, future


def _prediction(row, model):
    predictions = row.get("predictions", {})
    return _float(predictions.get(model))


def _date_filter(rows, start_date, end_date):
    return [
        row for row in rows
        if (not start_date or row.get("date", "") >= start_date) and (not end_date or row.get("date", "") <= end_date)
    ]


def _aggregate(rows, granularity):
    buckets = OrderedDict()
    for row in rows:
        if not row.get("date") or row.get("value") is None:
            continue
        key, label = _bucket(row["date"], granularity)
        bucket = buckets.setdefault(key, {"key": key, "label": label, "date": row["date"], "value": 0})
        bucket["value"] += row["value"]
        bucket["date"] = max(bucket["date"], row["date"])
        if row.get("forecast_origin") is not None:
            bucket.setdefault("forecast_origin", row.get("forecast_origin"))
        if row.get("fold") is not None:
            bucket.setdefault("fold", row.get("fold"))
    return list(buckets.values())


def _bucket(date_value, granularity):
    parsed = datetime.fromisoformat(date_value)
    if granularity == "Weekly":
        iso = parsed.isocalendar()
        return f"{iso.year}-W{iso.week:02d}", f"W{iso.week:02d} {iso.year}"
    if granularity == "Monthly":
        return parsed.strftime("%Y-%m"), parsed.strftime("%b %Y")
    return date_value, parsed.strftime("%b %d, %Y")


def _point_limit(horizon, granularity):
    match = re.search(r"(\d+)\s*(day|days|week|weeks|month|months|year|years)?", str(horizon).lower())
    amount = int(match.group(1)) if match else 13
    unit = (match.group(2) or "weeks") if match else "weeks"
    if granularity == "Weekly" and unit in {"month", "months"}:
        return amount * 4
    if granularity == "Monthly" and unit in {"month", "months"}:
        return amount
    if granularity == "Monthly" and unit in {"week", "weeks"}:
        return max(1, math.ceil(amount / 4))
    if granularity == "Weekly" and unit in {"week", "weeks"}:
        return amount
    if granularity == "Daily" and unit in {"day", "days"}:
        return amount
    days = amount * {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30, "year": 365, "years": 365}[unit]
    if granularity == "Weekly":
        return max(1, math.ceil(days / 7))
    if granularity == "Monthly":
        return max(1, math.ceil(days / 30))
    return days


def _seasonality(rows, granularity):
    values = [(row.get("date"), _float(row.get("actual"))) for row in rows]
    values = [(date, value) for date, value in values if date and value is not None]
    if len(values) < 4:
        return {"frequency": granularity, "strength": 0, "strongest_period": None, "weakest_period": None, "summary": "Seasonality is not yet strong enough to detect."}
    groups = {}
    for date, value in values:
        parsed = datetime.fromisoformat(date)
        period = parsed.strftime("%A") if granularity == "Daily" else parsed.strftime("%B")
        groups.setdefault(period, []).append(value)
    means = {period: sum(items) / len(items) for period, items in groups.items()}
    strongest, weakest = max(means, key=means.get), min(means, key=means.get)
    average = sum(means.values()) / len(means) or 1
    strength = round(min(100, max(0, (max(means.values()) - min(means.values())) / average * 100)), 2)
    summary = f"Demand is strongest on {strongest} and weakest on {weakest}." if granularity == "Daily" else f"Forecast shows seasonal lift in {strongest}."
    return {"frequency": granularity, "strength": strength, "strongest_period": strongest, "weakest_period": weakest, "summary": summary}


def _label_for_key(key, *maps):
    for mapping in maps:
        if key in mapping:
            return mapping[key]["label"]
    return key


def _float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _round(value):
    return round(value, 2) if value is not None else None


def _timestamp_text(value):
    timestamp = datetime.fromisoformat(str(value)) if not isinstance(value, datetime) else value
    if timestamp.hour == timestamp.minute == timestamp.second == timestamp.microsecond == 0:
        return timestamp.date().isoformat()
    return timestamp.isoformat()


def _empty_forecast():
    return {
        "ok": False,
        "filters": {}, "labels": [], "actual": [], "fitted": [], "future_forecast": [],
        "historical_actual": [], "historical_prediction": [], "future_prediction": [],
        "predicted": [], "lower": [], "upper": [], "forecast_start_index": None,
        "last_actual_date": None, "forecast_start_date": None, "forecast_end_date": None,
        "sections": {"historical_points": 0, "validation_points": 0, "future_points": 0},
        "summary": {"points": 0, "future_points": 0, "total_actual": 0, "total_predicted": 0, "variance_pct": 0, "mape": None, "accuracy": None},
        "seasonality": {"frequency": None, "strength": 0, "strongest_period": None, "weakest_period": None, "summary": "No forecast data available."},
        "explanation": {"direction": "stable", "summary": "No forecast explanation is available yet.", "reasons": []},
        "metadata": {"last_actual_timestamp": None, "first_historical_prediction_timestamp": None, "first_future_timestamp": None, "historical_prediction_total_count": 0, "historical_prediction_displayed_count": 0, "historical_prediction_coverage_percentage": 0, "confidence_level": None, "target_unit": None},
        "backtest_summary": {"point_count": 0, "reliable": False, "coverage_percentage": 0},
        "forecast_summary": {"trend": "unavailable"},
        "uncertainty": {"available": False, "direction": "unavailable"},
        "drift_summary": {}, "model_context": {}, "warnings": [],
        "plain_language_explanation": "No forecast explanation is available yet.",
    }
