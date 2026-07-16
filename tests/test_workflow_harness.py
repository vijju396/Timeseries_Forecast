"""Focused upload/mapping/training harness for the dataset lifecycle patch."""

import io
import json
import sys
import tempfile
import threading
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = PROJECT / "venv" / "Lib" / "site-packages"
sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(PROJECT))


def _csv(rows=40):
    return "Date,Demand,Store,Item\n" + "\n".join(
        f"2025-01-{index + 1:02d},{100 + index},North,Widget" for index in range(rows)
    )


def _large_csv(rows=40):
    return "Date,Demand,Store,Item\n" + "\n".join(
        f"2025-01-{index + 1:02d},{10 ** 100},North,Widget" for index in range(rows)
    )


def _wait_for_terminal(client, dataset_id, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/training-status").get_json()
        if status.get("job_id") == job_id and status.get("dataset_id") == dataset_id and status.get("status") != "running":
            return status
        time.sleep(0.05)
    raise AssertionError("training did not reach a terminal state")


def _join_training(training_service):
    thread = training_service._TRAINING_THREAD
    if thread:
        thread.join(timeout=5)
        assert not thread.is_alive(), "training thread did not stop after terminal status"


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        import services.data_service as data_service

        data_service.BASE_DIR = root
        data_service.DATA_DIR = root / "Data"
        data_service.PREPROCESSED_DIR = data_service.DATA_DIR / "Preprocessed"
        data_service.UPLOADS_DIR = data_service.DATA_DIR / "uploads"
        data_service.RUNTIME_DATA_DIR = data_service.DATA_DIR / "runtime"
        data_service.ensure_data_directories()
        locked = data_service.RUNTIME_DATA_DIR / "sessions" / "locked"
        locked.mkdir(parents=True)
        (locked / "state.json").write_text("{}", encoding="utf-8")
        original_rmtree = data_service.shutil.rmtree
        data_service.shutil.rmtree = lambda _path: (_ for _ in ()).throw(PermissionError("simulated OneDrive lock"))
        try:
            assert data_service._remove_generated_path(locked, attempts=2, delay=0) is False
            assert locked.exists()
        finally:
            data_service.shutil.rmtree = original_rmtree
        assert data_service._remove_generated_path(locked) is True
        (data_service.UPLOADS_DIR / "stale.csv").write_text("stale", encoding="utf-8")
        data_service.preprocessed_path("stale.csv").write_text("stale", encoding="utf-8")
        data_service.save_json("current_dataset.json", {"dataset_id": "stale"})
        data_service.save_json("enrichment.json", {"dataset_id": "stale"})
        data_service.initialize_runtime_state()
        assert not list(data_service.UPLOADS_DIR.iterdir())
        assert not list(data_service.PREPROCESSED_DIR.iterdir())
        assert data_service.current_dataset_id() is None

        import app
        from services import dataset_adapter, drift_service, preprocessing_service, training_service
        import pandas as pd

        parsed, audit = dataset_adapter._numeric_series(pd.Series(["100", "-100", "(100)", "1,000", "1 000", "$1,250.50", "bad"]), return_audit=True)
        assert parsed.iloc[:6].tolist() == [100.0, -100.0, -100.0, 1000.0, 1000.0, 1250.5]
        assert audit["malformed_values"] == 1
        frame = pd.DataFrame({"value": [1.0, None, 3.0, None, 99.0]})
        imputed, imputation_audit = preprocessing_service.impute_training_partition(frame, ["value"], "mean", train_end=3)
        assert imputed.loc[1, "value"] == 2.0 and pd.isna(imputed.loc[3, "value"])
        assert imputation_audit["columns"]["value"]["training_statistic"] == 2.0
        drift = drift_service.calculate_past_drift({"historical": [{"date": f"2025-01-{i:02d}", "actual": i} for i in range(1, 11)]})
        assert drift["status"] in {"Stable", "Watch", "Drift Detected"} and drift["explanation"]

        client = app.app.test_client()
        for page in ("/dataset", "/dashboard", "/forecast-explorer", "/model-metrics", "/training-pipeline"):
            assert client.get(page).status_code == 200, page
        uploaded = client.post(
            "/dataset/upload",
            data={"dataset_file": (io.BytesIO(_csv().encode()), "workflow.csv")},
            content_type="multipart/form-data",
        )
        assert uploaded.status_code == 200, uploaded.get_json()
        dataset = uploaded.get_json()["dataset"]
        dataset_id = dataset["id"]
        assert client.get("/api/enrichment").get_json()["available"] is False

        mapping = dataset["mapping"]
        mapped = client.post("/dataset/map", json={"dataset_id": dataset_id, **mapping})
        assert mapped.status_code == 200, mapped.get_json()
        assert client.get("/api/enrichment").get_json()["dataset_id"] == dataset_id
        raw_preview_before_training = client.get(f"/api/dataset-preview?dataset_id={dataset_id}").get_json()

        original_adapt = dataset_adapter.adapt_dataset
        adapt_calls = []

        def counted_adapt(*args, **kwargs):
            adapt_calls.append(1)
            return original_adapt(*args, **kwargs)

        dataset_adapter.adapt_dataset = counted_adapt
        original_specs = training_service.MODEL_SPECS
        original_fitters = training_service._fitters

        def fast_fitter(train_df, output_df):
            return [float(train_df["target"].iloc[-1])] * len(output_df)

        training_service.MODEL_SPECS = [("fast_Predictions", "Fast test model", "_fit_fast")]
        training_service._fitters = lambda: {"_fit_fast": fast_fitter}
        started = client.post("/train", json={"dataset_id": dataset_id})
        assert started.status_code == 202, started.get_json()
        start_payload = started.get_json()
        status = _wait_for_terminal(client, dataset_id, start_payload["job_id"])
        _join_training(training_service)
        assert status["status"] == "completed", status
        assert status["forecast_ready"] is True and status["usable_forecast_count"] == 1
        manifest = json.loads(data_service.data_path("forecast_manifest.json").read_text(encoding="utf-8"))
        assert manifest["job_id"] == start_payload["job_id"] and manifest["forecast_ready"] is True
        assert client.get(f"/api/dataset-preview?dataset_id={dataset_id}").get_json() == raw_preview_before_training
        assert not adapt_calls, "Train remapped an already enriched dataset"
        metrics = json.loads(data_service.data_path("model_metrics.json").read_text(encoding="utf-8"))
        assert metrics and all(row["dataset_id"] == dataset_id and row["job_id"] == start_payload["job_id"] for row in metrics)

        # A fresh browser must not inherit another browser's dataset or trained outputs.
        other_client = app.app.test_client()
        assert other_client.get("/api/current-dataset").get_json()["available"] is False
        assert other_client.get("/api/enrichment").get_json()["available"] is False
        assert other_client.get("/training-status").get_json()["available"] is False
        assert other_client.get("/api/forecast").status_code == 409
        assert client.get("/api/current-dataset").get_json()["dataset_id"] == dataset_id

        repeated = client.post("/train", json={"dataset_id": dataset_id})
        assert repeated.status_code == 202, repeated.get_json()
        repeated_status = _wait_for_terminal(client, dataset_id, repeated.get_json()["job_id"])
        _join_training(training_service)
        assert repeated_status["status"] == "completed", repeated_status

        large_upload = client.post(
            "/dataset/upload",
            data={"dataset_file": (io.BytesIO(_large_csv().encode()), "large.csv")},
            content_type="multipart/form-data",
        )
        assert large_upload.status_code == 200, large_upload.get_json()
        large_dataset = large_upload.get_json()["dataset"]
        large_id = large_dataset["id"]
        large_map = client.post("/dataset/map", json={"dataset_id": large_id, **large_dataset["mapping"]})
        assert large_map.status_code == 200, large_map.get_json()

        def overflow_fitter(_train_df, output_df):
            return [10 ** 200] * len(output_df)

        training_service.MODEL_SPECS = [
            ("good_large_Predictions", "Good large-target model", "_fit_fast"),
            ("overflow_Predictions", "Overflow model", "_fit_overflow"),
        ]
        training_service._fitters = lambda: {"_fit_fast": fast_fitter, "_fit_overflow": overflow_fitter}
        large_start = client.post("/train", json={"dataset_id": large_id})
        assert large_start.status_code == 202, large_start.get_json()
        large_status = _wait_for_terminal(client, large_id, large_start.get_json()["job_id"])
        _join_training(training_service)
        assert large_status["status"] == "completed_with_warnings", large_status
        assert large_status["forecast_ready"] is True
        assert any(item["model"] == "overflow_Predictions" for item in large_status["failed_models"])
        assert any(item["model"] == "good_large_Predictions" for item in large_status["completed_models"])

        # A second upload is rejected while the active job is running.
        original_run = training_service.run_real_training_pipeline

        def slow_job(_mapping, job_dataset_id, job_id, _artifact_id):
            time.sleep(0.5)
            training_service._write_status({"dataset_id": job_dataset_id, "job_id": job_id, "status": "completed", "completed_models": [], "failed_models": [], "total_models": 1})

        training_service.run_real_training_pipeline = slow_job
        started = client.post("/train", json={"dataset_id": large_id})
        assert started.status_code == 202
        blocked = client.post(
            "/dataset/upload",
            data={"dataset_file": (io.BytesIO(_csv().encode()), "replacement.csv")},
            content_type="multipart/form-data",
        )
        assert blocked.status_code == 409, blocked.get_json()
        independent_upload = other_client.post(
            "/dataset/upload",
            data={"dataset_file": (io.BytesIO(_csv().encode()), "independent.csv")},
            content_type="multipart/form-data",
        )
        assert independent_upload.status_code == 200, independent_upload.get_json()
        assert client.get("/api/current-dataset").get_json()["dataset_id"] == large_id
        stale_thread = training_service._TRAINING_THREAD
        data_service.clear_generated_dataset_state()
        training_service.reset_training_state(force=True)
        stale_thread.join(timeout=5)
        assert not data_service.data_path("training_status.json").exists(), "obsolete job published status"
        training_service.run_real_training_pipeline = original_run
        training_service.MODEL_SPECS = original_specs
        training_service._fitters = original_fitters
        dataset_adapter.adapt_dataset = original_adapt

        data_service.save_json("training_status.json", {"dataset_id": dataset_id, "job_id": "old", "status": "completed"})
        data_service.save_json("training_log.json", {"dataset_id": dataset_id})
        data_service.save_json("model_metrics.json", [{"dataset_id": dataset_id}])
        data_service.save_json("forecast_data.json", {"dataset_id": dataset_id})
        data_service.initialize_runtime_state()
        assert client.get("/api/current-dataset").get_json()["available"] is False
        assert client.get("/api/enrichment").get_json()["available"] is False
        assert client.post("/train", json={"dataset_id": dataset_id}).status_code == 400

    print("workflow harness passed")


if __name__ == "__main__":
    main()
