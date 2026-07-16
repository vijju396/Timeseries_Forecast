"""Deterministic preprocessing accounting tests across dataset shapes and frequencies."""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import dataset_adapter, enrichment_service, preprocessing_service, training_service


class _Patch:
    def __init__(self):
        self.originals = []

    def set(self, obj, name, value):
        self.originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def close(self):
        for obj, name, value in reversed(self.originals):
            setattr(obj, name, value)


def _adapt(frame, mapping, directory, dataset_id):
    source = directory / f"{dataset_id}.csv"
    frame.to_csv(source, index=False)
    patch = _Patch()
    patch.set(dataset_adapter, "preprocessed_path", lambda name: directory / name)
    patch.set(preprocessing_service, "preprocessed_path", lambda name: directory / name)
    patch.set(enrichment_service, "save_json", lambda _name, payload: payload)
    try:
        return dataset_adapter.adapt_dataset(source, {"csv_file": str(source), **mapping}, dataset_id=dataset_id)
    finally:
        patch.close()


def _assert_reconciled(adapted):
    summary = adapted["profile"]["summary"]
    preprocessing = adapted["preprocessing_metrics"]
    expected = (
        summary["valid_rows"] - summary.get("exact_duplicate_rows_removed", 0)
        - summary["rows_aggregated"] + summary["calendar_rows_generated"]
        - summary["unresolved_generated_rows_removed"]
    )
    assert summary["training_rows"] == expected
    assert summary["row_accounting_expected"] == expected
    assert summary["row_accounting_reconciled"] is True
    assert summary["missing_values_imputed"] >= 0
    assert summary["missing_timestamps_generated"] == summary["calendar_rows_generated"]
    reasons = preprocessing.get("rows_removed_reasons", {})
    assert sum(reasons.values()) == summary["rows_removed"]
    labels = {metric["label"] for metric in adapted["profile"]["preprocessing_metrics"]}
    assert {
        "Raw Rows", "Valid Rows", "Rows Removed", "Rows Aggregated",
        "Calendar Rows Generated", "Training Rows", "Missing Values Imputed",
        "Missing Timestamps Generated", "Frequency", "Forecast Grain",
        "Date Range", "Series Count", "Target Statistics",
    }.issubset(labels)
    assert adapted["profile"]["preprocessing_explanation"].endswith("Row accounting reconciles exactly.")


def test_transaction_accounting(directory):
    frame = pd.DataFrame([
        {"when": "2024-01-01", "demand": 10, "location code": "A", "product code": "X"},
        {"when": "2024-01-01", "demand": 5, "location code": "A", "product code": "X"},
        {"when": "2024-01-03", "demand": 20, "location code": "A", "product code": "X"},
        {"when": "2024-01-01", "demand": 7, "location code": "B", "product code": "Y"},
        {"when": "2024-01-02", "demand": 8, "location code": "B", "product code": "Y"},
        {"when": "2024-01-03", "demand": 9, "location code": "B", "product code": "Y"},
        {"when": "not-a-date", "demand": 4, "location code": "A", "product code": "X"},
        {"when": "2024-01-02", "demand": None, "location code": "A", "product code": "X"},
    ])
    adapted = _adapt(frame, {"date_column": "when", "target_column": "demand", "dimension_columns": ["location code", "product code"]}, directory, "transactions")
    summary = adapted["profile"]["summary"]
    assert summary["raw_rows"] == 8
    assert summary["valid_rows"] == 6
    assert summary["rows_removed"] == 2
    assert summary["rows_aggregated"] == 1
    assert summary["calendar_rows_generated"] == 1
    assert summary["training_rows"] == 6
    assert summary["missing_values_imputed"] == 0
    assert summary["series_count"] == 2
    assert len(adapted["dimension_schema"]) == 2
    _assert_reconciled(adapted)


def test_frequency_and_schema_compatibility(directory):
    cases = [
        ("weekly", pd.DataFrame({"period": ["2024-01-01", "2024-01-08", "2024-01-15"], "quantity": [1, 2, 3], "region name": ["R", "R", "R"]}), "W-", ["region name"]),
        ("monthly_inventory", pd.DataFrame({"period": ["2024-01-01", "2024-03-01", "2024-01-01"], "inventory level": [10, 12, 4], "warehouse": ["W1", "W1", "W2"]}), "MS", ["warehouse"]),
        ("hourly_sensor", pd.DataFrame({"observed_at": ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"], "temperature": [20.0, 20.5, 21.0], "sensor zone": ["Z", "Z", "Z"]}), "h", ["sensor zone"]),
        ("irregular_sensor", pd.DataFrame({"observed_at": ["2024-01-01 00:00", "2024-01-01 00:07", "2024-01-01 00:19"], "pressure": [20.0, 20.5, 21.0], "device": ["D1", "D1", "D1"]}), "min", ["device"]),
        ("three_dimensions", pd.DataFrame({"ts": ["2024-01-01", "2024-01-02", "2024-01-03"], "events": [2, 3, 4], "site": ["A", "A", "A"], "channel": ["web", "web", "web"], "segment": ["new", "new", "new"]}), "D", ["site", "channel", "segment"]),
    ]
    for name, frame, expected_frequency, dimensions in cases:
        date_column = frame.columns[0]
        target_column = frame.columns[1]
        adapted = _adapt(frame, {"date_column": date_column, "target_column": target_column, "dimension_columns": dimensions}, directory, name)
        assert str(adapted["frequency"]).startswith(expected_frequency)
        assert len(adapted["dimension_schema"]) == len(dimensions)
        _assert_reconciled(adapted)


def test_feature_lineage_duplicates_and_model_matrix(directory):
    rows = []
    for day in range(1, 7):
        for department, team in (("A", "red"), ("B", "blue")):
            rows.append({
                "when": f"2024-01-{day:02d}", "workers": 10 + day,
                "department": department, "team": team, "price": 100 + day,
                "promotion": day % 2 == 0, "status": "open", "capacity": 20 + day,
                "future_target": 99, "ignored_a": "x", "ignored_b": day, "ignored_c": "z",
                "ignored_d": 1,
            })
    rows.append({**rows[0], "workers": 3, "price": 90})  # legitimate grain collision
    rows.append(dict(rows[1]))  # exact duplicate
    frame = pd.DataFrame(rows)
    adapted = _adapt(frame, {
        "date_column": "when", "target_column": "workers",
        "dimension_columns": ["department", "team"],
        "exogenous_columns": ["price", "promotion", "status", "future_target"],
    }, directory, "feature_lineage")
    metrics = adapted["preprocessing_metrics"]
    assert metrics["exact_duplicate_rows_removed"] == 1
    assert metrics["grain_collision_groups"] == 1
    assert metrics["rows_combined_by_aggregation"] == 1
    assert metrics["post_aggregation_rows"] == 12
    assert metrics["training_rows"] == 12
    assert len(adapted["column_lineage"]) == 13
    assert all(row["retained"] or row["reason"] for row in adapted["column_lineage"])
    leaked = next(row for row in adapted["column_lineage"] if row["original_column"] == "future_target")
    assert leaked["retained"] is False and "leakage" in leaked["reason"]

    model_ready = pd.read_csv(directory / "ts_preprocessed.csv")
    assert len(model_ready) == 12
    assert {"dimension_1", "dimension_2", "exogenous_1", "exogenous_2", "exogenous_3"}.issubset(model_ready.columns)
    cleaned = pd.read_csv(directory / "cleaned_training_input.csv")
    cleaned["Date"] = pd.to_datetime(cleaned["Date"])
    aggregate = training_service._training_series(cleaned)
    assert len(aggregate) == 6
    matrix, target = training_service._xgb_training_matrix(aggregate, use_exog=True)
    assert len(matrix) == len(target) == 5
    assert any(column.startswith("exogenous_1") for column in matrix.columns)
    assert matrix.shape[1] > 13  # source features augment calendar, lag, rolling and trend features
    summary = training_service._model_input_summary("xgboost_exog", "_fit_xgboost_exog", aggregate)
    assert summary["source_exogenous_feature_count"] == 3
    assert summary["final_feature_count"] > summary["generated_feature_count"]

    future = preprocessing_service.add_exogenous_features(pd.DataFrame({
        "date": pd.date_range("2024-01-07", periods=2, freq="D"), "target": [None, None]
    }))
    try:
        training_service._exog_pair(aggregate, future)
    except training_service.FutureExogenousUnavailable:
        pass
    else:
        raise AssertionError("Future exogenous inputs must not be silently forward-filled.")


def main():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        test_transaction_accounting(directory)
        test_frequency_and_schema_compatibility(directory)
        test_feature_lineage_duplicates_and_model_matrix(directory)
    print("preprocessing accounting harness passed")


if __name__ == "__main__":
    main()
