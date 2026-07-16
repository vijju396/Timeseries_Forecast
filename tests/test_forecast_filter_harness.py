"""Deterministic filter-flow harness; uses a tiny mapped artifact, not production training."""
import hashlib
import json
import tempfile
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import drift_service, forecast_service, series_service, training_service


class _Patch:
    def __init__(self):
        self.originals = []

    def setattr(self, obj, name, value):
        self.originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def close(self):
        for obj, name, value in reversed(self.originals):
            setattr(obj, name, value)


def test_forecast_filter_ui_is_dynamic_and_prediction_ownership_is_strict():
    root = Path(__file__).resolve().parents[1]
    template = (root / "templates" / "forecast_explorer.html").read_text(encoding="utf-8")
    script = (root / "static" / "js" / "forecast_explorer.js").read_text(encoding="utf-8")
    assert "data-forecast-dimension" in template
    assert 'select name="store"' not in template and 'select name="item"' not in template
    assert "/api/forecast-filters" in script and "clearDependentDimensions" in script
    assert "canonicalDriftResult" in script
    payload = {"dataset_id": "d", "artifact_id": "a", "job_id": "j"}
    assert forecast_service._row_owned_by_payload({**payload}, payload)
    assert not forecast_service._row_owned_by_payload({"dataset_id": "d", "artifact_id": "a", "job_id": "old"}, payload)
    assert not forecast_service._row_owned_by_payload({"dataset_id": "d"}, payload)


def test_daily_forecast_defaults_to_32_points():
    spec = forecast_service._horizon_spec({}, "Daily")
    explicit = forecast_service._horizon_spec(
        {"horizon_value": "32", "horizon_unit": "days"}, "Daily"
    )
    assert spec == {"value": 32, "unit": "days", "points": 32, "label": "32 days"}
    assert explicit == spec
    assert (32, "days") in forecast_service.HORIZONS_BY_GRANULARITY["Daily"]


def test_four_filter_combinations_change_the_response(monkeypatch, tmp_path):
    rows = []
    for day in pd.date_range("2024-01-01", periods=12, freq="D"):
        for store, item, base in (("North Hub", "A 1", 10), ("South/Hub", "A 1", 30), ("North Hub", "B&2", 60)):
            rows.append({"Date": day, "Click Count": base + day.day, "Store": store, "Item": item})
    cleaned = tmp_path / "cleaned_training_input.csv"
    pd.DataFrame(rows).to_csv(cleaned, index=False)
    dates = [day.date().isoformat() for day in pd.date_range("2024-01-13", periods=40, freq="D")]
    dimension_schema = [
        {"id": "dimension_1", "canonical_column": "Store", "source_column": "store_source", "display_name": "Location"},
        {"id": "dimension_2", "canonical_column": "Item", "source_column": "product_source", "display_name": "Product"},
    ]
    ownership = {"dataset_id": "dataset-test", "artifact_id": "artifact-test", "job_id": "job-test"}
    series_values = [("North Hub", "A 1", 100), ("South/Hub", "A 1", 300), ("North Hub", "B&2", 600)]
    future_predictions = []
    backtest_predictions = []
    for store_value, item_value, base in series_values:
        series_key = {"dimension_1": store_value, "dimension_2": item_value}
        series_hash = series_service.series_key_hash(series_key)
        for model, multiplier in (("model_a", 1), ("model_b", 2)):
            for step, date in enumerate(dates, start=1):
                value = base * multiplier + step
                future_predictions.append({**ownership, "date": date, "model_id": model, "series_key": series_key, "series_key_hash": series_hash, "prediction_type": "future_forecast", "horizon_step": step, "predicted": value, "lower": value - 5, "upper": value + 5})
            for fold, date in enumerate(["2024-01-04", "2024-01-05"], start=1):
                actual = base / 10 + int(date[-2:])
                backtest_predictions.append({**ownership, "date": date, "model_id": model, "series_key": series_key, "series_key_hash": series_hash, "prediction_type": "historical_backtest", "forecast_origin": "2024-01-03", "fold": fold, "horizon_step": 1, "validation_method": "rolling_origin", "actual": actual, "predicted": actual * multiplier, "lower": actual * multiplier - 2, "upper": actual * multiplier + 2})
    payload = {
        "dataset_id": "dataset-test", "artifact_id": "artifact-test", "job_id": "job-test",
        "dimension_schema": dimension_schema,
        "filters": {"datasets": ["test.csv"], "stores": ["All Stores", "North Hub", "South/Hub"], "items": ["All Items", "A 1", "B&2"], "granularities": ["Daily", "Weekly", "Monthly"], "horizons": ["7 days", "4 weeks", "6 months"]},
        "historical": [{"date": "2024-01-01", "actual": 100, "store": "All Stores", "item": "All Items"}],
        "validation": [], "backtest_predictions": backtest_predictions, "future_predictions": future_predictions,
        "future": [{"date": date, "actual": None, "store": "All Stores", "item": "All Items", "predictions": {"model_a": 100, "model_b": 200}} for date in dates],
        "last_actual_date": "2024-01-12", "forecast_start_date": "2024-01-13",
        "model_registry": [{"model_id": "model_a", "scope": "per_series"}, {"model_id": "model_b", "scope": "per_series"}],
    }
    monkeypatch.setattr(forecast_service, "load_json", lambda name, default=None: payload if name == "forecast_data.json" else default)
    monkeypatch.setattr(forecast_service, "current_dataset_id", lambda: "dataset-test")
    monkeypatch.setattr(forecast_service, "preprocessed_path", lambda name: str(cleaned))
    monkeypatch.setattr(forecast_service, "get_model_metrics", lambda: [{"model": "model_a", "mape": 5, "mae": 1, "rmse": 2, "accuracy": 95}, {"model": "model_b", "mape": 10, "mae": 2, "rmse": 3, "accuracy": 90}])
    monkeypatch.setattr(forecast_service, "get_champion", lambda: {"model": "model_a"})
    monkeypatch.setattr(forecast_service, "get_metric_lookup", lambda: {"model_a": {"mape": 5, "accuracy": 95}, "model_b": {"mape": 10, "accuracy": 90}})

    combos = [
        {"store": "North Hub", "item": "A 1", "granularity": "Daily", "horizon": "7 days", "model": "model_a", "startDate": "2024-01-01", "endDate": "2024-01-05"},
        {"store": "South/Hub", "item": "A 1", "granularity": "Daily", "horizon": "7 days", "model": "model_a", "startDate": "2024-01-01", "endDate": "2024-01-05"},
        {"store": "North Hub", "item": "B&2", "granularity": "Monthly", "horizon": "6 months", "model": "model_a"},
        {"store": "North Hub", "item": "A 1", "granularity": "Weekly", "horizon": "4 weeks", "model": "model_b"},
    ]
    cold_start = time.perf_counter()
    cold_response = forecast_service.get_forecast({**combos[0], "dataset_id": "dataset-test"})
    cold_ms = (time.perf_counter() - cold_start) * 1000
    warm_start = time.perf_counter()
    warm_response = forecast_service.get_forecast({**combos[0], "dataset_id": "dataset-test"})
    warm_ms = (time.perf_counter() - warm_start) * 1000
    assert cold_response["cache_key"] == warm_response["cache_key"]
    default_params = {key: value for key, value in combos[0].items() if key != "horizon"}
    default_response = forecast_service.get_forecast({**default_params, "dataset_id": "dataset-test"})
    assert default_response["filters"]["horizon"] == "32 days"
    assert len(default_response["future_prediction"]) == 32
    assert default_response["metadata"]["future_prediction_count"] == 32
    responses = [cold_response, forecast_service.get_forecast({**combos[1], "dataset_id": "dataset-test"}), forecast_service.get_forecast({**combos[2], "dataset_id": "dataset-test"}), forecast_service.get_forecast({**combos[3], "dataset_id": "dataset-test"})]
    assert [response["filters"]["model_id"] for response in responses] == ["model_a", "model_a", "model_a", "model_b"]
    assert responses[0]["effective"]["matched_rows"] == 5
    assert responses[0]["actual"] != responses[1]["actual"]
    assert responses[0]["historical_prediction"] != responses[1]["historical_prediction"]
    assert responses[2]["effective"]["historical_points"] == 1
    assert responses[3]["effective"]["forecast_points"] >= 1
    assert {"historical_actual", "historical_prediction", "future_prediction"}.issubset(responses[0])
    assert responses[0]["future_prediction"][0]["timestamp"] > responses[0]["historical_actual"][-1]["timestamp"]
    model_b_response = forecast_service.get_forecast({**combos[0], "model": "model_b", "dataset_id": "dataset-test"})
    assert model_b_response["filters"]["model_id"] == "model_b"
    assert model_b_response["actual"] == responses[0]["actual"]
    assert model_b_response["future_forecast"] != responses[0]["future_forecast"]
    signatures = [hashlib.sha256(json.dumps({"filters": r["filters"], "labels": r["labels"], "actual": r["actual"], "future": r["future_forecast"]}, sort_keys=True).encode()).hexdigest() for r in responses]
    assert len(set(signatures)) == 4
    assert len({response["cache_key"] for response in responses}) == 4
    north_filters = forecast_service.get_dependent_filter_options({"dataset_id": "dataset-test", "dimensions": json.dumps({"dimension_1": "North Hub"})})
    south_filters = forecast_service.get_dependent_filter_options({"dataset_id": "dataset-test", "dimensions": json.dumps({"dimension_1": "South/Hub"})})
    assert north_filters["dimensions"][1]["values"] == ["A 1", "B&2"]
    assert south_filters["dimensions"][1]["values"] == ["A 1"]
    monkeypatch.setattr(drift_service, "save_json", lambda *_args, **_kwargs: None)
    drift = drift_service.build_drift_payload({**combos[0], "dataset_id": "dataset-test"})
    assert drift["dataset_id"] == "dataset-test"
    assert drift["drift_cache_key"]
    assert all(0 <= row["score"] <= 1 and row["severity"] in {"stable", "moderate", "high", "critical"} for row in drift["drift_series"])
    result = drift["drift_result"]
    assert result["target_drift"] == drift["target_drift"] == drift["past"]
    assert result["future_drift"] == drift["future_drift"] == drift["future"]
    assert result["target_drift"]["status"] == result["summary"]["overall_severity"]
    assert result["target_drift"]["mean_change"] is not None
    assert result["future_drift"]["available"] is True
    assert result["future_drift"]["forecast_change"] is not None


def test_drift_unavailable_reasons_are_explicit():
    future = drift_service.calculate_future_drift({"historical": [{"actual": 10}], "future": []})
    result = drift_service._canonical_drift_result([], [{"timestamp": "2024-01-01", "value": 10}], future)
    assert result["target_drift"]["reason"] == "insufficient history"
    assert result["future_drift"]["reason"] == "no forecast"


def test_training_artifact_marks_backtest_and_future_without_leakage():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    cleaned = pd.DataFrame({"Date": dates, "Click Count": [10, 11, 12, 13, 14, 15], "Store": "All Stores", "Item": "All Items"})
    prediction_frame = pd.DataFrame({"date": dates[3:], "target": [13, 14, pd.NA], "model_a": [12.8, 13.7, 15.2]})
    payload = training_service._forecast_payload(prediction_frame, [{"model": "model_a", "mape": 5}], {"source_file": "test.csv", "target_column": "target"}, cleaned, "D")
    backtest = [row for row in payload["validation"] if row["prediction_type"] == "historical_backtest"]
    future = [row for row in payload["future"] if row["prediction_type"] == "future_forecast"]
    assert backtest and future
    assert all(row["forecast_origin"] < row["date"] for row in backtest)
    assert future[0]["date"] > backtest[-1]["date"]
    assert future[0]["horizon_step"] == 1


def test_all_rolling_folds_are_persisted_with_empirical_intervals():
    dates = pd.date_range("2018-01-01", periods=100, freq="MS")
    series = pd.DataFrame({"date": dates, "target": [100 + index * 1.5 for index in range(100)]})

    def lightweight_fitter(train, validation):
        last = float(train["target"].iloc[-1])
        return [last + (step + 1) * 1.4 for step in range(len(validation))]

    evaluation = training_service._rolling_origin_evaluation(lightweight_fitter, series)
    assert evaluation["folds"] == 3
    assert len(evaluation["predictions"]) == 60
    assert {row["fold"] for row in evaluation["predictions"]} == {1, 2, 3}
    assert all(row["forecast_origin"] < row["date"] for row in evaluation["predictions"])

    output_dates = list(dates[-20:]) + list(pd.date_range(dates[-1] + pd.offsets.MonthBegin(), periods=4, freq="MS"))
    prediction_frame = pd.DataFrame({
        "date": output_dates,
        "target": list(series["target"].iloc[-20:]) + [pd.NA] * 4,
        "model_a": [float(value) - .2 for value in series["target"].iloc[-20:]] + [251, 252, 253, 254],
    })
    cleaned = pd.DataFrame({"Date": dates, "Click Count": series["target"], "Store": "All Stores", "Item": "All Items"})
    payload = training_service._forecast_payload(
        prediction_frame,
        [{"model": "model_a", "mape": 2}],
        {"source_file": "monthly.csv", "target_column": "demand"},
        cleaned,
        "MS",
        backtest_predictions={"model_a": evaluation["predictions"]},
    )
    persisted = payload["backtest_predictions"]
    assert len(persisted) == 60
    assert {row["fold"] for row in persisted} == {1, 2, 3}
    assert all(row["lower"] <= row["predicted"] <= row["upper"] for row in persisted)
    assert all(row["validation_method"] == "rolling_origin" for row in persisted)
    assert payload["backtest_duplicate_policy"] == "shortest_horizon"


def test_per_series_training_artifacts_preserve_owned_series_keys():
    rows = []
    for series_name, offset in (("alpha", 0), ("beta", 100)):
        for index, timestamp in enumerate(pd.date_range("2022-01-01", periods=100, freq="D")):
            rows.append({"Date": timestamp, "Click Count": offset + index + 1, "dimension_1": series_name})
    cleaned = pd.DataFrame(rows)
    schema = [{"id": "dimension_1", "canonical_column": "dimension_1", "source_column": "business series", "display_name": "Business Series"}]

    def lightweight_fitter(train, output):
        return [float(train["target"].iloc[-1])] * len(output)

    artifacts, scopes = training_service._generate_series_artifacts(
        cleaned,
        {"dimension_schema": schema, "series_count": 2},
        [{"model": "exp_additive_Predictions"}],
        {"_fit_exp_additive": lightweight_fitter},
    )
    backtests = artifacts["backtest_predictions"]
    futures = artifacts["future_predictions"]
    hashes = {row["series_key_hash"] for row in backtests}
    assert len(hashes) == 2
    assert series_service.series_key_hash({}) not in hashes
    assert all(row["series_key"] and row["model_id"] == "exp_additive_Predictions" for row in backtests + futures)
    assert all(row.get("timestamp") and "prediction" in row for row in backtests + futures)
    assert all(row["prediction_type"] == "historical_backtest" for row in backtests)
    assert all(row["prediction_type"] == "future_forecast" for row in futures)
    assert scopes["exp_additive_Predictions"]["scope"] == "per_series"
    assert scopes["exp_additive_Predictions"]["supported_series_count"] == 2
    assert scopes["exp_additive_Predictions"]["series_failure_count"] == 0
    assert artifacts["failures"] == []
    assert sorted(sum(row["series_key_hash"] == series_hash for row in backtests) for series_hash in hashes) == [60, 60]


def test_backtest_summaries_duplicate_policy_and_uncertainty():
    historical = [
        {"date": day.date().isoformat(), "actual": 100 + index * 10, "store": "All Stores", "item": "All Items"}
        for index, day in enumerate(pd.date_range("2024-01-01", periods=10, freq="D"))
    ]
    backtests = []
    for index in range(4, 10):
        timestamp = historical[index]["date"]
        backtests.append({"date": timestamp, "actual": historical[index]["actual"], "predicted": historical[index]["actual"] - 5, "lower": historical[index]["actual"] - 12, "upper": historical[index]["actual"] + 2, "forecast_origin": historical[index - 1]["date"], "fold": 2, "horizon_step": 1, "validation_method": "rolling_origin", "model_id": "model_a", "store": "All Stores", "item": "All Items"})
    backtests.append({**backtests[-1], "predicted": 999, "horizon_step": 3, "forecast_origin": historical[5]["date"]})
    backtests = [{**row, "dataset_id": "dataset-summary", "artifact_id": "artifact-summary", "job_id": "job-summary"} for row in backtests]
    future = []
    for step, day in enumerate(pd.date_range("2024-01-11", periods=6, freq="D"), start=1):
        value = 200 + step * 10
        width = step * 4
        future.append({"date": day.date().isoformat(), "actual": None, "store": "All Stores", "item": "All Items", "horizon_step": step, "predictions": {"model_a": value}, "confidence_bands": {"model_a": {"lower": value - width, "upper": value + width}}})
    payload = {
        "dataset_id": "dataset-summary", "artifact_id": "artifact-summary", "job_id": "job-summary",
        "filters": {"datasets": ["summary.csv"], "stores": ["All Stores"], "items": ["All Items"], "granularities": ["Daily"], "horizons": ["7 days"]},
        "historical": historical, "validation": [], "backtest_predictions": backtests, "future": future,
        "confidence_level": 95, "backtest_duplicate_policy": "shortest_horizon",
    }
    patcher = _Patch()
    try:
        patcher.setattr(forecast_service, "load_json", lambda name, default=None: payload if name == "forecast_data.json" else default)
        patcher.setattr(forecast_service, "current_dataset_id", lambda: "dataset-summary")
        patcher.setattr(forecast_service, "get_model_metrics", lambda: [{"model": "model_a", "model_label": "Model A", "mape": 4, "mae": 5, "rmse": 5, "accuracy": 96}])
        patcher.setattr(forecast_service, "get_champion", lambda: {"model": "model_a"})
        patcher.setattr(forecast_service, "get_metric_lookup", lambda: {"model_a": {"model_label": "Model A", "mape": 4, "mae": 5, "rmse": 5, "accuracy": 96}})
        response = forecast_service.get_forecast({"dataset_id": "dataset-summary", "model": "model_a", "horizon": "7 days"})
    finally:
        patcher.close()
    assert len(response["historical_prediction"]) == 6
    assert response["historical_prediction"][-1]["value"] == historical[-1]["actual"] - 5
    assert response["backtest_summary"]["coverage_percentage"] == 60.0
    assert response["backtest_summary"]["mae"] == 5.0
    assert response["backtest_summary"]["wape"] is not None
    assert response["backtest_summary"]["smape"] is not None
    assert response["backtest_summary"]["reliable"] is True
    assert response["forecast_summary"]["first_value"] == 210.0
    assert response["forecast_summary"]["last_value"] == 260.0
    assert response["forecast_summary"]["trend"] == "increasing"
    assert response["uncertainty"]["direction"] == "widening"
    assert all(row["lower"] <= row["value"] <= row["upper"] for row in response["historical_prediction"] + response["future_prediction"])


def test_one_point_backtest_is_disclosed_as_unreliable():
    summary = forecast_service._backtest_summary(
        [{"date": "2024-01-01", "value": 10}, {"date": "2024-01-02", "value": 12}],
        [{"date": "2024-01-02", "value": 11, "actual": 12}],
    )
    warnings = forecast_service._forecast_warnings(summary, [{"timestamp": "2024-01-02"}], [], {"available": False}, 2, 1, "Daily")
    assert summary["point_count"] == 1 and summary["reliable"] is False
    assert any("accuracy is not yet reliable" in warning for warning in warnings)


def test_aggregate_predictions_never_fall_back_to_selected_series(tmp_path):
    cleaned = tmp_path / "cleaned_training_input.csv"
    pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=5, freq="D"), "Click Count": [1, 2, 3, 4, 5], "dimension_1": "series-a", "Store": "series-a", "Item": "All Items"}).to_csv(cleaned, index=False)
    aggregate_hash = series_service.series_key_hash({})
    schema = [{"id": "dimension_1", "canonical_column": "dimension_1", "source_column": "business unit", "display_name": "Business Unit"}]
    payload = {
        "dataset_id": "dataset-owned", "artifact_id": "artifact-owned", "job_id": "job-owned",
        "dimension_schema": schema,
        "filters": {"datasets": ["owned.csv"], "stores": ["All Series", "series-a"], "items": ["All Series"], "granularities": ["Daily"], "horizons": ["7 days"]},
        "historical": [{"date": f"2024-01-0{index}", "actual": index, "series_key": {}, "series_key_hash": aggregate_hash} for index in range(1, 6)],
        "backtest_predictions": [{"dataset_id": "dataset-owned", "artifact_id": "artifact-owned", "job_id": "job-owned", "date": "2024-01-05", "actual": 5, "predicted": 5, "model_id": "aggregate_model", "series_key": {}, "series_key_hash": aggregate_hash, "forecast_origin": "2024-01-04", "fold": 1, "horizon_step": 1}],
        "future_predictions": [{"dataset_id": "dataset-owned", "artifact_id": "artifact-owned", "job_id": "job-owned", "date": "2024-01-06", "predicted": 6, "lower": 5, "upper": 7, "model_id": "aggregate_model", "series_key": {}, "series_key_hash": aggregate_hash, "horizon_step": 1}],
        "model_registry": [{"model_id": "aggregate_model", "scope": "aggregate_only", "supported_series_count": 0, "total_series_count": 1}],
    }
    patcher = _Patch()
    try:
        patcher.setattr(forecast_service, "load_json", lambda name, default=None: payload if name == "forecast_data.json" else default)
        patcher.setattr(forecast_service, "current_dataset_id", lambda: "dataset-owned")
        patcher.setattr(forecast_service, "preprocessed_path", lambda _name: str(cleaned))
        patcher.setattr(forecast_service, "get_model_metrics", lambda: [{"model": "aggregate_model", "model_label": "Aggregate Model", "mape": 5, "mae": 1, "rmse": 1, "accuracy": 95, "model_scope": "aggregate_only"}])
        patcher.setattr(forecast_service, "get_champion", lambda: {"model": "aggregate_model"})
        patcher.setattr(forecast_service, "get_metric_lookup", lambda: {"aggregate_model": {"model_label": "Aggregate Model", "mape": 5, "mae": 1, "rmse": 1, "accuracy": 95, "model_scope": "aggregate_only"}})
        series_response = forecast_service.get_forecast({"dataset_id": "dataset-owned", "model": "aggregate_model", "store": "series-a", "horizon": "7 days"})
        aggregate_response = forecast_service.get_forecast({"dataset_id": "dataset-owned", "model": "aggregate_model", "store": "All Series", "horizon": "7 days"})
        patcher.setattr(drift_service, "save_json", lambda *_args, **_kwargs: None)
        drift = drift_service.build_drift_payload({"dataset_id": "dataset-owned", "model": "aggregate_model", "store": "series-a", "horizon": "7 days"})
    finally:
        patcher.close()
    assert series_response["historical_actual"]
    assert series_response["historical_prediction"] == []
    assert series_response["future_prediction"] == []
    assert series_response["historical_prediction_status"] == "historical_prediction_unavailable"
    assert series_response["forecast_summary"]["trend"] == "unavailable"
    assert series_response["uncertainty"]["available"] is False
    assert series_response["prediction_availability"]["model_scope"] == "aggregate_only"
    assert "No historical backtest predictions available for this selected series." in series_response["warnings"]
    assert any("aggregate-only" in warning for warning in series_response["warnings"])
    assert series_response["series_key_hash"] != aggregate_response["series_key_hash"]
    assert series_response["cache_key"] != aggregate_response["cache_key"]
    assert drift["series_key_hash"] == series_response["series_key_hash"]
    assert drift["residual_drift"]["overall_severity"] == "insufficient_data"


def test_historical_prediction_response_overlaps_actual_and_preserves_identity():
    payload = {
        "dataset_id": "dataset-hp", "artifact_id": "artifact-hp", "job_id": "job-hp",
        "filters": {"datasets": ["test.csv"], "stores": ["All Stores"], "items": ["All Items"], "granularities": ["Daily"], "horizons": ["7 days"]},
        "historical": [
            {"date": "2024-01-01", "actual": 10, "store": "All Stores", "item": "All Items"},
            {"date": "2024-01-02", "actual": 12, "store": "All Stores", "item": "All Items"},
            {"date": "2024-01-03", "actual": 14, "store": "All Stores", "item": "All Items"},
            {"date": "2024-01-04", "actual": 16, "store": "All Stores", "item": "All Items"},
        ],
        "validation": [
            {"date": "2024-01-03", "actual": 14, "store": "All Stores", "item": "All Items", "forecast_origin": "2024-01-02", "fold": 1, "predictions": {"model_a": 13.5, "model_b": 15.0}},
            {"date": "2024-01-04", "actual": 16, "store": "All Stores", "item": "All Items", "forecast_origin": "2024-01-02", "fold": 1, "predictions": {"model_a": 15.5, "model_b": 17.0}},
        ],
        "future": [
            {"date": "2024-01-05", "actual": None, "store": "All Stores", "item": "All Items", "predictions": {"model_a": 17.5, "model_b": 20.0}},
            {"date": "2024-01-06", "actual": None, "store": "All Stores", "item": "All Items", "predictions": {"model_a": 18.5, "model_b": 21.0}},
        ],
    }
    patcher = _Patch()
    try:
        patcher.setattr(forecast_service, "load_json", lambda name, default=None: payload if name == "forecast_data.json" else default)
        patcher.setattr(forecast_service, "current_dataset_id", lambda: "dataset-hp")
        patcher.setattr(forecast_service, "get_model_metrics", lambda: [{"model": "model_a", "mape": 5, "mae": 1, "rmse": 2, "accuracy": 95}, {"model": "model_b", "mape": 7, "mae": 1, "rmse": 2, "accuracy": 93}])
        patcher.setattr(forecast_service, "get_champion", lambda: {"model": "model_a"})
        patcher.setattr(forecast_service, "get_metric_lookup", lambda: {"model_a": {"mape": 5, "accuracy": 95}, "model_b": {"mape": 7, "accuracy": 93}})
        response_a = forecast_service.get_forecast({"dataset_id": "dataset-hp", "model": "model_a", "horizon": "7 days"})
        response_b = forecast_service.get_forecast({"dataset_id": "dataset-hp", "model": "model_b", "horizon": "7 days"})
    finally:
        patcher.close()
    actual_timestamps = {row["timestamp"] for row in response_a["historical_actual"]}
    prediction_timestamps = {row["timestamp"] for row in response_a["historical_prediction"]}
    assert response_a["historical_prediction"]
    assert prediction_timestamps.issubset(actual_timestamps)
    assert all(row["forecast_origin"] < row["timestamp"] for row in response_a["historical_prediction"])
    assert all(row["dataset_id"] == "dataset-hp" and row["artifact_id"] == "artifact-hp" and row["job_id"] == "job-hp" and row["model_id"] == "model_a" for row in response_a["historical_prediction"])
    assert response_a["metadata"]["last_actual_timestamp"] == response_a["historical_actual"][-1]["timestamp"]
    assert response_a["future_prediction"][0]["timestamp"] > response_a["metadata"]["last_actual_timestamp"]
    assert response_a["historical_actual"] == response_b["historical_actual"]
    assert response_a["historical_prediction"] != response_b["historical_prediction"]
    assert response_a["future_prediction"] != response_b["future_prediction"]


def test_model_resolution_restores_aggregate_champion(tmp_path):
    cleaned = tmp_path / "cleaned_training_input.csv"
    pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=5, freq="D"), "Click Count": [1, 2, 3, 4, 5], "dimension_1": "series-a"}).to_csv(cleaned, index=False)
    aggregate_hash = series_service.series_key_hash({})
    series_key = {"dimension_1": "series-a"}
    series_hash = series_service.series_key_hash(series_key)
    ownership = {"dataset_id": "dataset-model-state", "artifact_id": "artifact-model-state", "job_id": "job-model-state"}
    payload = {
        **ownership,
        "dimension_schema": [{"id": "dimension_1", "canonical_column": "dimension_1", "source_column": "Department", "display_name": "Department"}],
        "dimension_options": {"dimension_1": ["series-a"]},
        "filters": {"datasets": ["state.csv"], "stores": ["All Series"], "items": ["All Series"], "granularities": ["Daily"], "horizons": ["7 days"]},
        "historical": [{"date": f"2024-01-0{index}", "actual": index} for index in range(1, 6)],
        "backtest_predictions": [
            {**ownership, "date": "2024-01-05", "actual": 5, "predicted": 5, "model_id": "auto_arima", "series_key": {}, "series_key_hash": aggregate_hash, "forecast_origin": "2024-01-04", "fold": 1, "horizon_step": 1},
            {**ownership, "date": "2024-01-05", "actual": 5, "predicted": 5.5, "model_id": "exp_additive", "series_key": series_key, "series_key_hash": series_hash, "forecast_origin": "2024-01-04", "fold": 1, "horizon_step": 1},
        ],
        "future_predictions": [
            {**ownership, "date": "2024-01-06", "predicted": 6, "model_id": "auto_arima", "series_key": {}, "series_key_hash": aggregate_hash, "horizon_step": 1},
            {**ownership, "date": "2024-01-06", "predicted": 7, "model_id": "exp_additive", "series_key": series_key, "series_key_hash": series_hash, "horizon_step": 1},
        ],
        "model_registry": [
            {"model_id": "auto_arima", "display_name": "Auto ARIMA", "scope": "aggregate_only"},
            {"model_id": "exp_additive", "display_name": "Exponential Smoothing", "scope": "per_series"},
        ],
    }
    metrics = [
        {**ownership, "model": "auto_arima", "model_label": "Auto ARIMA", "mape": 4, "mae": 1, "rmse": 1, "accuracy": 96},
        {**ownership, "model": "exp_additive", "model_label": "Exponential Smoothing", "mape": 8, "mae": 2, "rmse": 2, "accuracy": 92},
    ]
    patcher = _Patch()
    try:
        patcher.setattr(forecast_service, "load_json", lambda name, default=None: payload if name == "forecast_data.json" else default)
        patcher.setattr(forecast_service, "current_dataset_id", lambda: ownership["dataset_id"])
        patcher.setattr(forecast_service, "preprocessed_path", lambda _name: str(cleaned))
        patcher.setattr(forecast_service, "get_model_metrics", lambda: metrics)
        patcher.setattr(forecast_service, "get_champion", lambda: metrics[0])
        patcher.setattr(forecast_service, "get_metric_lookup", lambda: {row["model"]: row for row in metrics})
        aggregate_options = forecast_service.get_dependent_filter_options({"dataset_id": ownership["dataset_id"]})
        series_options = forecast_service.get_dependent_filter_options({"dataset_id": ownership["dataset_id"], "dimensions": json.dumps(series_key)})
        series_response = forecast_service.get_forecast({"dataset_id": ownership["dataset_id"], "dimensions": json.dumps(series_key), "model_id": "auto_arima", "selection_source": "global_champion", "horizon": "7 days"})
        aggregate_response = forecast_service.get_forecast({"dataset_id": ownership["dataset_id"], "model_id": "exp_additive", "selection_source": "automatic_series_fallback", "horizon": "7 days"})
    finally:
        patcher.close()
    assert aggregate_options["global_champion"]["model_id"] == "auto_arima"
    assert aggregate_options["recommended_model_id"] == "auto_arima"
    assert series_options["recommended_model_id"] == "exp_additive"
    assert series_response["resolved_model_id"] == "exp_additive" and series_response["selection_source"] == "automatic_series_fallback"
    assert aggregate_response["resolved_model_id"] == "auto_arima" and aggregate_response["selection_source"] == "global_champion"
    assert series_response["model_context"]["resolved_model_id"] == "exp_additive"
    assert all(row["model_id"] == "exp_additive" for row in series_response["historical_prediction"] + series_response["future_prediction"])
    assert series_response["cache_key"] != aggregate_response["cache_key"]
if __name__ == "__main__":
    test_daily_forecast_defaults_to_32_points()
    with tempfile.TemporaryDirectory() as directory:
        patcher = _Patch()
        try:
            test_four_filter_combinations_change_the_response(patcher, Path(directory))
        finally:
            patcher.close()
    test_training_artifact_marks_backtest_and_future_without_leakage()
    test_all_rolling_folds_are_persisted_with_empirical_intervals()
    test_per_series_training_artifacts_preserve_owned_series_keys()
    test_backtest_summaries_duplicate_policy_and_uncertainty()
    test_one_point_backtest_is_disclosed_as_unreliable()
    with tempfile.TemporaryDirectory() as directory:
        test_aggregate_predictions_never_fall_back_to_selected_series(Path(directory))
    test_historical_prediction_response_overlaps_actual_and_preserves_identity()
    print("forecast filter harness passed")
