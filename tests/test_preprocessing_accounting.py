"""Deterministic preprocessing accounting tests across dataset shapes and frequencies."""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import dataset_adapter, enrichment_service, preprocessing_service


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
    expected = summary["valid_rows"] - summary["rows_aggregated"] + summary["calendar_rows_generated"] - summary["unresolved_generated_rows_removed"]
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


def main():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        test_transaction_accounting(directory)
        test_frequency_and_schema_compatibility(directory)
    print("preprocessing accounting harness passed")


if __name__ == "__main__":
    main()
