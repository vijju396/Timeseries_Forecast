"""End-to-end healthcare workbook harness for upload, EDA, training, and forecast APIs."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "manpower_healthcare.xlsx"
sys.path.insert(0, str(ROOT))


def main(full_training=False):
    assert WORKBOOK.is_file(), "Healthcare workbook is missing."
    with tempfile.TemporaryDirectory() as temp_dir:
        runtime_root = Path(temp_dir)
        os.environ["FORECAST_DATA_DIR"] = str(runtime_root / "Data" / "runtime")
        os.environ["CLEAR_RUNTIME_ON_START"] = "true"
        os.environ["FORECAST_LSTM_EPOCHS"] = "1"
        os.environ["FORECAST_LSTM_WINDOW"] = "14"
        os.environ["FORECAST_LSTM_BATCH_SIZE"] = "32"
        os.environ["FORECAST_XGB_N_JOBS"] = "1"
        os.environ["FORECAST_PER_SERIES_MODEL_LIMIT"] = "2"
        os.environ["FORECAST_SERIES_PHASE_BUDGET_SECONDS"] = "300"

        from services import data_service

        data_service.BASE_DIR = runtime_root
        data_service.DATA_DIR = runtime_root / "Data"
        data_service.PREPROCESSED_DIR = data_service.DATA_DIR / "preprocessed"
        data_service.UPLOADS_DIR = data_service.DATA_DIR / "uploads"
        data_service.RUNTIME_DATA_DIR = data_service.DATA_DIR / "runtime"
        data_service.ensure_data_directories()

        from app import app

        app.config.update(TESTING=True)
        client = app.test_client()
        with WORKBOOK.open("rb") as handle:
            upload = client.post(
                "/dataset/upload",
                data={"dataset_file": (handle, WORKBOOK.name)},
                content_type="multipart/form-data",
            )
        assert upload.status_code == 200, upload.get_data(as_text=True)
        dataset = upload.get_json()["dataset"]
        mapping = dataset["suggested_mapping"]
        assert dataset["raw_preview"]["total_raw_rows"] == 396
        assert len(dataset["raw_schema"]) == 24
        assert mapping["date_column"] == "date"
        assert mapping["target_column"] == "manpower_required"
        assert mapping["domain"] == "healthcare_staffing"
        assert mapping["target_unit"] == "staff members"
        assert dataset["requires_mapping"] is False
        assert [dataset["raw_schema"][int(value.split("_")[1])]["source_name"] for value in mapping["dimension_column_ids"]] == ["department", "facility_id"]
        assert len(mapping["exogenous_column_ids"]) == 10

        mapped = client.post(
            "/dataset/map",
            json={
                "dataset_id": dataset["id"],
                "timestamp_column_id": mapping["timestamp_column_id"],
                "target_column_id": mapping["target_column_id"],
                "dimension_column_ids": mapping["dimension_column_ids"],
                "exogenous_column_ids": mapping["exogenous_column_ids"],
            },
        )
        assert mapped.status_code == 200, mapped.get_data(as_text=True)
        adapted = mapped.get_json()["adapted"]
        assert adapted["rows_read"] == adapted["rows_used"] == 396
        assert adapted["frequency"] == "D"
        assert adapted["start_date"] == "2025-01-01"
        assert adapted["end_date"] == "2026-01-31"
        assert adapted["target_display_name"] == "Manpower Required"
        assert adapted["target_unit"] == "staff members"
        assert len(adapted["column_lineage"]) == 24
        roles = {row["original_column"]: row["mapped_role"] for row in adapted["column_lineage"]}
        assert roles["manpower_required"] == "target"
        assert roles["department"] == roles["facility_id"] == "series_dimension"
        assert roles["patient_census"] == roles["patient_acuity_index"] == "exogenous"
        assert roles["scheduled_staff"] == roles["overtime_hours"] == "excluded_leakage"
        assert roles["data_origin"] == roles["source_url"] == "provenance"

        import pandas as pd

        cleaned = pd.read_csv(adapted["cleaned_path"])
        dates = pd.to_datetime(cleaned["Date"])
        assert len(cleaned) == 396 and dates.is_monotonic_increasing
        assert cleaned["Click Count"].between(8, 15).all()
        assert cleaned.filter(regex=r"^exogenous_").shape[1] == 10

        preview = client.get(f"/api/dataset-preview?dataset_id={dataset['id']}&limit=10")
        assert preview.status_code == 200 and preview.get_json()["preview_type"] == "raw"
        enrichment = client.get("/api/enrichment")
        enrichment_payload = enrichment.get_json()
        assert enrichment_payload["available"] is True
        assert enrichment_payload["summary"]["training_rows"] == 396
        assert enrichment_payload["summary"]["row_accounting_reconciled"] is True
        analytics = client.get(
            "/api/data-studio-analytics",
            query_string={
                "dataset_id": dataset["id"],
                "artifact_id": adapted["artifact_id"],
                "dimensions": json.dumps({}),
                "request_id": "healthcare-test",
            },
        )
        assert analytics.status_code == 200
        assert analytics.get_json()["matched_row_count"] == 396

        for path in ("/", "/dataset", "/dashboard", "/training-pipeline", "/model-metrics", "/forecast-explorer"):
            response = client.get(path)
            assert response.status_code == 200, path

        if not full_training:
            print("healthcare workbook mapping and EDA pipeline passed")
            return

        started = client.post("/train", json={"dataset_id": dataset["id"]})
        assert started.status_code == 202, started.get_data(as_text=True)
        terminal = {"completed", "completed_with_warnings", "failed", "cancelled", "budget_exceeded"}
        deadline = time.monotonic() + 1200
        status = started.get_json()["status"]
        while status.get("status") not in terminal and time.monotonic() < deadline:
            time.sleep(1)
            status = client.get("/training-status").get_json()
        assert status.get("status") in {"completed", "completed_with_warnings"}, status
        assert status.get("forecast_ready") is True
        assert status.get("rows_used_for_training") == 396
        assert len(status.get("completed_models") or []) >= 8
        failed_model_ids = {row.get("model") for row in status.get("failed_models") or []}
        assert not failed_model_ids.intersection({"var_Predictions", "var_exog_Predictions"})

        forecast = client.get("/api/forecast", query_string={"horizon": "30 days", "granularity": "Daily"})
        assert forecast.status_code == 200, forecast.get_data(as_text=True)
        forecast_payload = forecast.get_json()
        assert forecast_payload["ok"] is True
        assert forecast_payload["metadata"]["target_display_name"] == "Manpower Required"
        assert forecast_payload["metadata"]["target_unit"] == "staff members"
        assert forecast_payload["metadata"]["last_actual_timestamp"] == "2026-01-31"
        assert forecast_payload["metadata"]["first_future_timestamp"] == "2026-02-01"
        assert len(forecast_payload["future_prediction"]) == 30
        assert all(point["value"] is not None for point in forecast_payload["future_prediction"])
        assert client.get("/api/models").get_json()
        print(
            f"healthcare full pipeline passed: {len(status['completed_models'])} models, "
            f"{len(forecast_payload['future_prediction'])} future points"
        )


if __name__ == "__main__":
    main(full_training="--full" in sys.argv)
