import os

from flask import Flask, jsonify, render_template, request

from services import dataset_adapter, drift_service, enrichment_service, forecast_service, metrics_service, training_service, upload_service
from services.raw_preview_service import RawPreviewError
from services.data_service import (
    clear_generated_dataset_state,
    current_dataset_id,
    initialize_runtime_state,
    load_json,
)


initialize_runtime_state()
training_service.reset_training_state(force=True)
app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", champion=metrics_service.get_champion())


@app.get("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        kpis=metrics_service.get_kpis(),
        filter_options=forecast_service.get_filter_options(),
    )


@app.get("/dataset")
def dataset():
    return render_template("dataset.html")


@app.post("/dataset/upload")
def dataset_upload():
    try:
        training_service.reject_if_training_active()
        with upload_service.UPLOAD_LOCK:
            clear_generated_dataset_state()
            result = upload_service.save_uploaded_dataset(request.files.get("dataset_file"))
            training_service.reset_training_state()
        return jsonify({"ok": True, **result})
    except training_service.TrainingConflictError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/dataset/map")
def dataset_map():
    try:
        payload = request.get_json(silent=True) if request.is_json else request.form
        payload = payload or {}
        training_service.reject_if_training_active()
        result = dataset_adapter.apply_dataset_mapping(
            dataset_id=payload.get("dataset_id", ""),
            mapping={
                "csv_file": payload.get("csv_file", ""),
                "timestamp_column_id": payload.get("timestamp_column_id", ""),
                "target_column_id": payload.get("target_column_id", ""),
                "primary_dimension_column_id": payload.get("primary_dimension_column_id", ""),
                "secondary_dimension_column_id": payload.get("secondary_dimension_column_id", ""),
                "dimension_column_ids": payload.get("dimension_column_ids") or [
                    column for column in (
                        payload.get("primary_dimension_column_id"),
                        payload.get("secondary_dimension_column_id"),
                    ) if column
                ],
                "exogenous_column_ids": payload.get("exogenous_column_ids") or [],
                "date_column": payload.get("date_column", ""),
                "target_column": payload.get("target_column", ""),
                "store_column": payload.get("store_column", ""),
                "item_column": payload.get("item_column", ""),
                "dimension_columns": payload.get("dimension_columns") or [
                    column for column in (payload.get("store_column"), payload.get("item_column")) if column
                ],
            },
        )
        if not result.get("reused"):
            training_service.reset_training_state()
        return jsonify({"ok": True, **result})
    except training_service.TrainingConflictError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/train")
def train():
    payload = request.get_json(silent=True) if request.is_json else {}
    try:
        status = training_service.start_background_training(payload or {})
        return jsonify({"ok": True, "status": status, "job_id": status.get("job_id"), "dataset_id": status.get("dataset_id")}), 202
    except training_service.TrainingValidationError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except training_service.TrainingConflictError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409


@app.get("/training-status")
def training_status():
    return jsonify(_validated_training_status())


@app.get("/training-pipeline")
def training_pipeline():
    return render_template("training_pipeline.html", status=_validated_training_status())


@app.get("/model-metrics")
def model_metrics():
    return render_template(
        "model_metrics.html",
        metrics=metrics_service.get_model_metrics(),
        champion=metrics_service.get_champion(),
        failed_models=metrics_service.get_failed_models(),
    )


@app.get("/forecast-explorer")
def forecast_explorer():
    return render_template(
        "forecast_explorer.html",
        filter_options=forecast_service.get_filter_options(),
    )


@app.get("/api/kpis")
def api_kpis():
    return jsonify(metrics_service.get_kpis())


@app.get("/api/models")
def api_models():
    return jsonify(metrics_service.get_model_metrics())


@app.get("/api/forecast")
def api_forecast():
    status = training_service.get_training_status()
    manifest = load_json("forecast_manifest.json", {})
    terminal_ready = status.get("status") in {"completed", "completed_with_warnings"} and status.get("forecast_ready") is True
    owned_manifest = all(manifest.get(key) == status.get(key) for key in ("dataset_id", "artifact_id", "job_id"))
    if not terminal_ready or not manifest.get("forecast_ready") or not owned_manifest:
        return jsonify({"ok": False, "error": "Forecast results are not ready for the active training job.", "labels": [], "actual": [], "fitted": [], "future_forecast": []}), 409
    try:
        return jsonify(forecast_service.get_forecast(request.args.to_dict()))
    except forecast_service.ForecastRequestError as exc:
        return jsonify({"ok": False, "error": str(exc), "filters": request.args.to_dict(), "labels": [], "actual": [], "fitted": [], "future_forecast": []}), exc.status_code


@app.get("/api/forecast-filters")
def api_forecast_filters():
    try:
        return jsonify(forecast_service.get_dependent_filter_options(request.args.to_dict()))
    except forecast_service.ForecastRequestError as exc:
        return jsonify({"ok": False, "error": str(exc), "dimensions": []}), exc.status_code


@app.get("/api/drift")
def api_drift():
    try:
        payload = drift_service.build_drift_payload(request.args.to_dict())
    except forecast_service.ForecastRequestError as exc:
        return jsonify({"ok": False, "available": False, "error": str(exc), "drift_series": [], "summary": {}}), exc.status_code
    if not current_dataset_id() or payload.get("dataset_id") != current_dataset_id():
        return jsonify({"ok": False, "available": False, "message": "No drift report is available for the active dataset.", "drift_series": [], "summary": {}})
    return jsonify({**payload, "available": True})


@app.get("/api/training-status")
def api_training_status():
    return jsonify(_validated_training_status())


@app.get("/api/datasets")
def api_datasets():
    payload = load_json("datasets.json", {"datasets": []})
    payload["active_dataset_id"] = current_dataset_id()
    return jsonify(payload)


@app.get("/api/current-dataset")
def api_current_dataset():
    dataset_id = current_dataset_id()
    if not dataset_id:
        return jsonify({"available": False, "message": "No active dataset."})
    dataset = dataset_adapter.get_dataset(dataset_id)
    if not dataset:
        return jsonify({"available": False, "message": "Active dataset metadata is unavailable."})
    return jsonify({"available": True, "dataset_id": dataset_id, "dataset": dataset})


@app.get("/api/dataset-preview")
def api_dataset_preview():
    dataset_id = request.args.get("dataset_id", "")
    if not current_dataset_id():
        return jsonify({"ok": False, "code": "dataset_unavailable", "error": "No active raw dataset is available."}), 404
    try:
        return jsonify(dataset_adapter.get_raw_preview(dataset_id, request.args.get("limit", 10)))
    except RawPreviewError as exc:
        return jsonify({"ok": False, "code": "raw_preview_unavailable", "error": str(exc)}), 400


@app.get("/api/enrichment")
def api_enrichment():
    dataset_id = current_dataset_id()
    if not dataset_id:
        return jsonify({"available": False, "message": "No active dataset. Upload and map a dataset to generate enrichment insights."})
    try:
        payload = load_json("enrichment.json", {})
    except (OSError, ValueError):
        payload = {}
    active_dataset = dataset_adapter.get_active_dataset()
    expected_artifact = (active_dataset or {}).get("adapted", {}).get("artifact_id")
    if not payload or not payload.get("summary") or payload.get("dataset_id") != dataset_id or (expected_artifact and payload.get("artifact_id") != expected_artifact):
        return jsonify({"available": False, "message": "No enrichment profile yet."})
    return jsonify({**payload, "available": True})


@app.get("/api/data-studio-analytics")
def api_data_studio_analytics():
    dataset = dataset_adapter.get_active_dataset()
    if not dataset:
        return jsonify({"ok": False, "available": False, "code": "dataset_unavailable", "message": "No mapped dataset is available."}), 404
    try:
        return jsonify(enrichment_service.build_data_studio_analytics(dataset, request.args.to_dict()))
    except (ValueError, OSError) as exc:
        return jsonify({"ok": False, "available": False, "code": "analytics_unavailable", "message": str(exc)}), 400


def _validated_training_status():
    dataset_id = current_dataset_id()
    status = training_service.get_training_status()
    if not dataset_id or status.get("dataset_id") != dataset_id:
        return {
            "available": False,
            "dataset_id": dataset_id,
            "status": "idle",
            "training_mode": "Full ML Training",
            "current_step": "No training run for the active dataset",
            "current_model": "",
            "duration_display": "0 sec",
            "completed_models": [],
            "failed_models": [],
            "total_models": 13,
            "rows_used_for_training": 0,
            "evaluation_mode": "--",
            "message": "Upload and map a dataset before training.",
        }
    return {**status, "available": True}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
