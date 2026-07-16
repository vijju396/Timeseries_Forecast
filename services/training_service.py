import copy
import logging
import math
import os
import threading
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path

from services.data_service import DATA_DIR, current_dataset_id, current_runtime_namespace, load_json, preprocessed_path, save_json, set_runtime_namespace
from services import dependency_service
from services.metrics_service import calculate_metrics, is_valid_metric
from services.series_service import aggregate_series_identity, row_series_identity, series_count


os.environ.setdefault("MPLCONFIGDIR", str(DATA_DIR / ".matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
CALENDAR_EXOG_COLUMNS = ["week_exog", "dayOfWeek_exog", "month_exog"]
SOURCE_EXOG_PREFIX = "exogenous_"
TRAINING_MODE = "Full ML Training"
MODEL_SPECS = [
    ("sarimax_Predictions", "SARIMAX", "_fit_sarimax"),
    ("sarimax_exog_Predictions", "SARIMAX with exogenous variables", "_fit_sarimax_exog"),
    ("autoArima_Predictions", "Auto ARIMA", "_fit_auto_arima"),
    ("autoArima_exog_Predictions", "Auto ARIMA with exogenous variables", "_fit_auto_arima_exog"),
    ("xgboost_Predictions", "XGBoost", "_fit_xgboost"),
    ("xgboost_exog_Predictions", "XGBoost with exogenous variables", "_fit_xgboost_exog"),
    ("exp_additive_Predictions", "Exponential Smoothing Additive", "_fit_exp_additive"),
    ("exp_additive_damped_Predictions", "Exponential Smoothing Additive Damped", "_fit_exp_additive_damped"),
    ("exp_multiplicative_Predictions", "Exponential Smoothing Multiplicative", "_fit_exp_multiplicative"),
    ("exp_multiplicative_damped_Predictions", "Exponential Smoothing Multiplicative Damped", "_fit_exp_multiplicative_damped"),
    ("var_Predictions", "VAR", "_fit_var"),
    ("var_exog_Predictions", "VAR with exogenous variables", "_fit_var_exog"),
    ("lstm_Predictions", "LSTM", "_fit_lstm"),
]

_THREAD_LOCK = threading.Lock()
_STATUS_LOCK = threading.Lock()
_TRAINING_THREAD = None
_TRAINING_THREAD_NAMESPACE = None
_IN_MEMORY_STATUSES = {}


def _status_namespace_key():
    return current_runtime_namespace() or "__global__"


def _memory_status():
    return _IN_MEMORY_STATUSES.get(_status_namespace_key())


def _clear_memory_status():
    _IN_MEMORY_STATUSES.pop(_status_namespace_key(), None)


class TrainingValidationError(ValueError):
    pass


class TrainingConflictError(RuntimeError):
    pass


class StaleTrainingJob(RuntimeError):
    pass


class FutureExogenousUnavailable(ValueError):
    pass


def _job_is_current(dataset_id, job_id):
    if dataset_id != current_dataset_id():
        return False
    with _STATUS_LOCK:
        active_job = (_memory_status() or {}).get("job_id")
    return active_job in (None, job_id)


def _ensure_job_active(dataset_id, job_id):
    if not _job_is_current(dataset_id, job_id):
        raise StaleTrainingJob(f"Training job {job_id} is no longer active for dataset {dataset_id}.")


def _set_dataset_phase(dataset_id, phase, job_id):
    if dataset_id != current_dataset_id():
        return
    from services.dataset_adapter import append_or_update_dataset, get_dataset

    dataset = get_dataset(dataset_id)
    if not dataset:
        return
    dataset["status"] = phase
    dataset["training_job_id"] = job_id
    append_or_update_dataset(dataset)


def reject_if_training_active():
    if (
        _TRAINING_THREAD
        and _TRAINING_THREAD.is_alive()
        and _TRAINING_THREAD_NAMESPACE == _status_namespace_key()
    ):
        raise TrainingConflictError("Training is already running. Wait for it to finish before replacing the active dataset.")


def reset_training_state(force=False):
    global _TRAINING_THREAD, _TRAINING_THREAD_NAMESPACE
    if not force:
        reject_if_training_active()
    with _STATUS_LOCK:
        _clear_memory_status()
    thread_alive = bool(_TRAINING_THREAD and _TRAINING_THREAD.is_alive())
    if force or not thread_alive or _TRAINING_THREAD_NAMESPACE == _status_namespace_key():
        _TRAINING_THREAD = None
        _TRAINING_THREAD_NAMESPACE = None


def _active_dataset_for_training(dataset_id):
    from services.dataset_adapter import _mapping_hash, _source_signature, get_dataset

    if not dataset_id or dataset_id != current_dataset_id():
        raise TrainingValidationError("Upload and map the active dataset before training.")
    dataset = get_dataset(dataset_id)
    if not dataset:
        raise TrainingValidationError("The active dataset metadata is unavailable. Upload it again.")
    mapping = dataset.get("mapping") or {}
    if not mapping.get("csv_file") or not mapping.get("date_column") or not mapping.get("target_column"):
        raise TrainingValidationError("Confirm the Date and Target columns before training.")
    adapted = dataset.get("adapted") or {}
    cleaned_path = Path(adapted.get("cleaned_path", ""))
    source_file = mapping.get("csv_file")
    if not source_file or not Path(source_file).is_file():
        raise TrainingValidationError("The uploaded source file is unavailable. Upload the dataset again.")
    if (
        adapted.get("dataset_id") != dataset_id
        or not adapted.get("artifact_id")
        or adapted.get("mapping_hash") != _mapping_hash(mapping)
        or adapted.get("source_signature") != _source_signature(source_file)
        or not cleaned_path.is_file()
        or cleaned_path.stat().st_size == 0
    ):
        raise TrainingValidationError("Apply mapping and enrichment for this dataset before training.")
    enrichment = load_json("enrichment.json", {})
    if enrichment.get("dataset_id") != dataset_id or enrichment.get("artifact_id") != adapted.get("artifact_id") or enrichment.get("mapping_hash") != adapted.get("mapping_hash"):
        raise TrainingValidationError("Enrichment is not ready for the active dataset. Apply mapping again.")
    return dataset, adapted


def start_background_training(dataset_mapping=None):
    global _TRAINING_THREAD, _TRAINING_THREAD_NAMESPACE
    with _THREAD_LOCK:
        dataset_mapping = dataset_mapping or {}
        dataset_id = dataset_mapping.get("dataset_id") or current_dataset_id()
        _dataset, adapted = _active_dataset_for_training(dataset_id)
        if _TRAINING_THREAD and _TRAINING_THREAD.is_alive():
            if _TRAINING_THREAD_NAMESPACE == _status_namespace_key():
                raise TrainingConflictError("Training is already running for the active dataset.")
            raise TrainingConflictError("Training capacity is currently in use by another browser. You can upload and map data now, then start training when the active job finishes.")
        job_id = uuid.uuid4().hex
        _set_dataset_phase(dataset_id, "training", job_id)
        with _STATUS_LOCK:
            _clear_memory_status()
        _write_status(_initial_status(dataset_id=dataset_id, job_id=job_id, artifact_id=adapted.get("artifact_id")))
        runtime_namespace = current_runtime_namespace()
        _TRAINING_THREAD_NAMESPACE = _status_namespace_key()
        _TRAINING_THREAD = threading.Thread(
            target=_run_training_in_namespace,
            args=(runtime_namespace, dataset_mapping, dataset_id, job_id, adapted.get("artifact_id")),
            daemon=True,
        )
        _TRAINING_THREAD.start()
        return get_training_status()


def _run_training_in_namespace(runtime_namespace, dataset_mapping, dataset_id, job_id, artifact_id):
    set_runtime_namespace(runtime_namespace)
    return run_real_training_pipeline(dataset_mapping, dataset_id, job_id, artifact_id)


def get_training_status():
    with _STATUS_LOCK:
        memory_status = _memory_status()
        status = copy.deepcopy(memory_status) if memory_status else None
    if status is None:
        try:
            status = load_json("training_status.json", _idle_status())
        except (OSError, ValueError):
            status = _idle_status()
        _set_memory_status(status)
    status = _recover_stale_training_status(status)
    status["background_thread_alive"] = bool(
        _TRAINING_THREAD
        and _TRAINING_THREAD.is_alive()
        and _TRAINING_THREAD_NAMESPACE == _status_namespace_key()
    )
    status["active_lock_owner"] = status.get("job_id") if _THREAD_LOCK.locked() else None
    status["exact_last_function"] = status.get("phase") or status.get("current_step")
    if status.get("status") == "running" and status.get("start_time"):
        elapsed = max(0, (datetime.now() - datetime.fromisoformat(status["start_time"])).total_seconds())
        status["duration_seconds"] = round(elapsed, 2)
        status["duration_display"] = _duration_display(elapsed)
    return status


def _recover_stale_training_status(status):
    active_thread = bool(
        _TRAINING_THREAD
        and _TRAINING_THREAD.is_alive()
        and _TRAINING_THREAD_NAMESPACE == _status_namespace_key()
    )
    if status.get("status") != "running" or active_thread:
        return status
    terminal_models = len(status.get("completed_models") or []) + len(status.get("failed_models") or []) + len(status.get("skipped_models") or [])
    expected_models = status.get("models_expected") or status.get("total_models") or 0
    if terminal_models < expected_models:
        return status
    heartbeat = status.get("last_heartbeat") or status.get("start_time")
    try:
        stale_seconds = (datetime.now() - datetime.fromisoformat(heartbeat)).total_seconds()
    except (TypeError, ValueError):
        stale_seconds = float("inf")
    if stale_seconds < max(1, int(os.getenv("FORECAST_STALE_JOB_SECONDS", "120"))):
        return status
    manifest = load_json("forecast_manifest.json", {})
    ownership = {key: status.get(key) for key in ("dataset_id", "artifact_id", "job_id")}
    if not manifest or any(manifest.get(key) != value for key, value in ownership.items()):
        forecast_payload = load_json("forecast_data.json", {})
        if forecast_payload and all(forecast_payload.get(key) == value for key, value in ownership.items()):
            manifest = _build_forecast_manifest(forecast_payload)
            save_json("forecast_manifest.json", manifest)
    finished = datetime.now().isoformat(timespec="seconds")
    if manifest.get("forecast_ready") and all(manifest.get(key) == value for key, value in ownership.items()):
        progress = manifest.get("forecast_progress") or {}
        warning_state = bool(manifest.get("warnings") or progress.get("failed") or progress.get("skipped") or progress.get("budget_exceeded"))
        status.update({
            "status": "completed_with_warnings" if warning_state else "completed", "phase": "completed",
            "current_step": "Recovered completed forecast publication", "forecast_ready": True,
            "usable_model_count": manifest.get("usable_model_count", 0), "usable_forecast_count": manifest.get("usable_forecast_count", 0),
            "forecast_manifest_id": manifest.get("forecast_manifest_id"), "message": "Recovered an orphaned completed training job.",
        })
    else:
        status.update({"status": "failed", "phase": "failed", "current_step": "Failed", "forecast_ready": False, "message": "orphaned_job: no readable owned forecast manifest was available."})
    status.update({"current_model": "", "current_artifact": "", "finished_at": finished, "end_time": finished, "last_heartbeat": finished})
    _write_status(status)
    return status


def run_real_training_pipeline(dataset_mapping=None, dataset_id=None, job_id=None, artifact_id=None):
    global _TRAINING_THREAD, _TRAINING_THREAD_NAMESPACE
    start_wall = datetime.now()
    start_timer = time.perf_counter()
    status = _initial_status(start_wall, dataset_id=dataset_id, job_id=job_id, artifact_id=artifact_id)
    log_entries = []
    _write_status(status)
    _write_log(log_entries, dataset_id, job_id)

    def log(level, message, model=None):
        entry = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "level": level,
            "model": model,
            "message": message,
        }
        log_entries.append(entry)
        _write_log(log_entries, dataset_id, job_id)

    def update(**changes):
        _ensure_job_active(dataset_id, job_id)
        now = datetime.now()
        if changes.get("phase") and changes["phase"] != status.get("phase"):
            changes.setdefault("phase_started_at", now.isoformat(timespec="seconds"))
        status.update(changes)
        status["last_heartbeat"] = now.isoformat(timespec="seconds")
        try:
            status["elapsed_phase_seconds"] = round((now - datetime.fromisoformat(status["phase_started_at"])).total_seconds(), 2)
        except (KeyError, TypeError, ValueError):
            status["elapsed_phase_seconds"] = 0
        elapsed = time.perf_counter() - start_timer
        status["duration_seconds"] = round(elapsed, 2)
        status["duration_display"] = _duration_display(elapsed)
        _write_status(status)

    try:
        update(phase="queued", current_step="Preparing full training data", current_model="")
        dataset_info = _ensure_training_input(dataset_mapping or {}, dataset_id)
        if dataset_info.get("artifact_id") != artifact_id:
            raise StaleTrainingJob("The mapped artifact changed before training started.")
        update(dataset_id=dataset_id)
        cleaned_df = _load_cleaned_frame()
        training_df = _training_series(cleaned_df)
        if len(training_df) < 10:
            raise ValueError("Not enough cleaned rows to train full ML forecasting models.")

        train_df, output_df, holdout_size, frequency = _split_training_output(training_df)
        holdout_df = output_df.iloc[:holdout_size].copy()
        future_df = output_df.iloc[holdout_size:].copy()
        last_actual_date = _timestamp_text(training_df["date"].max())
        forecast_start_date = _timestamp_text(future_df["date"].min())
        forecast_end_date = _timestamp_text(future_df["date"].max())
        evaluation_mode = _evaluation_mode(len(training_df), _fold_validation_size(len(training_df)))
        update(
            phase="training_models",
            current_step="Training full ML forecasting models",
            total_models=len(MODEL_SPECS),
            models_expected=len(MODEL_SPECS), models_terminal=0,
            raw_rows=dataset_info.get("rows_read", len(cleaned_df)),
            cleaned_rows=len(cleaned_df),
            rows_used_for_training=len(training_df),
            evaluation_mode=evaluation_mode,
            last_data_date=last_actual_date,
            forecast_start_date=forecast_start_date,
            forecast_end_date=forecast_end_date,
            forecast_horizon=len(future_df),
        )

        prediction_frame = _prediction_frame(output_df)
        metric_rows = []
        completed_models = []
        failed_models = []
        skipped_models = []
        backtest_predictions = {}
        fitters = _fitters()

        for model_name, label, fitter_name in MODEL_SPECS:
            model_timer = time.perf_counter()
            update(current_step=f"Training {label}", current_model=label)
            eligibility = _model_eligibility(model_name, fitter_name, training_df)
            if not eligibility["eligible"]:
                skipped = {
                    "model": model_name, "model_label": label, "status": "skipped",
                    "runtime_seconds": 0, "reason_code": eligibility["reason_code"],
                    "reason": eligibility["reason"],
                }
                skipped_models.append(skipped)
                log("warning", f"Model skipped: {skipped['reason']}", model_name)
                update(
                    completed_models=completed_models, failed_models=failed_models, skipped_models=skipped_models,
                    models_terminal=len(completed_models) + len(failed_models) + len(skipped_models),
                )
                continue
            try:
                input_summary = _model_input_summary(model_name, fitter_name, training_df)
                log("info", _input_summary_text(input_summary), model_name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    backtest = _rolling_origin_evaluation(fitters[fitter_name], training_df)
                    holdout_predictions = fitters[fitter_name](train_df, holdout_df)
                    future_unavailable_reason = None
                    try:
                        future_predictions = fitters[fitter_name](training_df, future_df)
                    except FutureExogenousUnavailable as exc:
                        future_predictions = [None] * len(future_df)
                        future_unavailable_reason = str(exc)

                predictions = list(holdout_predictions) + list(future_predictions)
                prediction_frame[model_name] = _clean_predictions(predictions, len(output_df))
                holdout_metrics = calculate_metrics(
                    output_df["target"].iloc[:holdout_size],
                    prediction_frame[model_name].iloc[:holdout_size],
                )
                ranking_metrics = backtest["metrics"]
                if ranking_metrics["mape"] is None:
                    ranking_metrics = holdout_metrics
                if ranking_metrics["mape"] is None:
                    raise ValueError("Model produced no comparable validation predictions.")

                runtime = round(time.perf_counter() - model_timer, 2)
                metric_row = {
                        "model": model_name,
                        "model_label": label,
                        **ranking_metrics,
                        "holdout_mape": holdout_metrics["mape"],
                        "holdout_accuracy": holdout_metrics["accuracy"],
                        "holdout_mae": holdout_metrics["mae"],
                        "holdout_rmse": holdout_metrics["rmse"],
                        "holdout_wape": holdout_metrics["wape"],
                        "holdout_bias": holdout_metrics["bias"],
                        "evaluation_mode": backtest["mode"],
                        "folds": backtest["folds"],
                        "runtime_seconds": runtime,
                        "status": "completed",
                        "input_summary": input_summary,
                        "future_forecast_available": future_unavailable_reason is None,
                        "future_unavailable_reason": future_unavailable_reason,
                    }
                if not is_valid_metric(metric_row):
                    raise ValueError("Model produced invalid or overflowed finite metrics.")
                metric_rows.append(metric_row)
                backtest_predictions[model_name] = backtest.get("predictions", [])
                completed_models.append(
                    {
                        "model": model_name,
                        "model_label": label,
                        "status": "completed",
                        "runtime_seconds": runtime,
                    }
                )
                if backtest.get("reason"):
                    log("info", backtest["reason"], model_name)
                if future_unavailable_reason:
                    log("warning", future_unavailable_reason, model_name)
                log("info", "Model evaluated, then refit on full history for future forecasting.", model_name)
            except Exception as exc:
                runtime = round(time.perf_counter() - model_timer, 2)
                failure = {
                    "model": model_name,
                    "model_label": label,
                    "status": "failed",
                    "runtime_seconds": runtime,
                    "reason": _friendly_error(exc),
                }
                failed_models.append(failure)
                log("warning", f"Model failed, continuing remaining models: {failure['reason']}", model_name)
            update(
                completed_models=completed_models, failed_models=failed_models, skipped_models=skipped_models,
                models_terminal=len(completed_models) + len(failed_models) + len(skipped_models),
            )

        if not metric_rows:
            raise RuntimeError("All full ML model training attempts failed. Check dependencies and training log.")

        update(phase="generating_forecasts", current_step="Generating forecasts", current_model="")

        def artifact_progress(snapshot):
            update(
                phase="generating_forecasts",
                current_step="Generating forecasts",
                current_artifact=snapshot.get("current_artifact", ""),
                artifact_progress=snapshot,
            )

        series_artifacts, scope_by_model = _generate_series_artifacts(
            cleaned_df, {**dataset_info, "job_id": job_id}, metric_rows, fitters, progress=artifact_progress
        )
        for row in metric_rows:
            scope = scope_by_model.get(row["model"], {})
            row["model_scope"] = scope.get("scope", "aggregate_only")
            row["supported_series_count"] = scope.get("supported_series_count", 0)
            row["total_series_count"] = scope.get("total_series_count", dataset_info.get("series_count", 1))
            row["series_failure_count"] = scope.get("series_failure_count", 0)
            row["scope_reason"] = scope.get("scope_reason")
            row["supported_series_hashes"] = scope.get("supported_series_hashes", [])
        for row in completed_models:
            scope = scope_by_model.get(row["model"], {})
            row["model_scope"] = scope.get("scope", "aggregate_only")

        metric_rows = _rank_metrics(metric_rows)
        champion = metric_rows[0]
        prediction_frame = _order_prediction_columns(prediction_frame, metric_rows)
        artifact_summary = series_artifacts.get("progress", {})
        artifact_warnings = []
        if artifact_summary.get("failed_artifacts"):
            artifact_warnings.append(f"{artifact_summary['failed_artifacts']} series artifact(s) failed.")
        if artifact_summary.get("skipped_artifacts"):
            artifact_warnings.append(f"{artifact_summary['skipped_artifacts']} unsupported or ineligible series artifact(s) were skipped.")
        if artifact_summary.get("budget_exceeded"):
            artifact_warnings.append("Series artifact generation reached its configured time budget.")
        update(
            phase="validating_forecasts", current_step="Validating forecasts", current_model="",
            current_artifact="", artifact_warnings=artifact_warnings,
            artifact_progress=artifact_summary,
            forecasts_expected=len(metric_rows), forecasts_generated=0, forecasts_failed=0, forecasts_skipped=0,
            artifacts_expected=2, artifacts_published=0, artifacts_failed=0, artifacts_skipped=0,
            champion_model=champion["model"], champion_mape=champion["mape"],
        )
        update(phase="publishing_forecasts", current_step="Publishing forecasts", phase_started_at=datetime.now().isoformat(timespec="seconds"))
        _ensure_job_active(dataset_id, job_id)
        publication = _write_training_outputs(
            prediction_frame,
            metric_rows,
            dataset_info,
            cleaned_df,
            frequency,
            status,
            backtest_predictions,
            series_artifacts,
        )
        _ensure_job_active(dataset_id, job_id)
        forecast_progress = publication["forecast_progress"]
        publication_progress = publication["artifact_publication"]
        artifact_warnings.extend(publication.get("warnings") or [])
        update(
            forecast_ready=publication["forecast_ready"], usable_model_count=publication["usable_model_count"],
            usable_forecast_count=publication["usable_forecast_count"], forecast_manifest_id=publication["forecast_manifest_id"],
            forecasts_expected=forecast_progress["expected"], forecasts_generated=forecast_progress["generated"],
            forecasts_failed=forecast_progress["failed"], forecasts_skipped=forecast_progress["skipped"],
            forecast_budget_exceeded=forecast_progress["budget_exceeded"],
            artifacts_expected=publication_progress["expected"], artifacts_published=publication_progress["published"],
            artifacts_failed=publication_progress["failed"], artifacts_skipped=publication_progress["skipped"],
            artifact_warnings=artifact_warnings,
        )
        if not publication["forecast_ready"]:
            raise RuntimeError("No usable future forecast was produced by the completed models.")
        elapsed = time.perf_counter() - start_timer
        terminal_status = "completed_with_warnings" if failed_models or skipped_models or artifact_warnings else "completed"
        status.update({
            "status": terminal_status, "phase": "completed", "current_step": "Completed with warnings" if artifact_warnings or failed_models or skipped_models else "Completed",
            "current_model": "", "current_artifact": "", "end_time": datetime.now().isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"), "last_heartbeat": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": round(elapsed, 2), "duration_display": _duration_display(elapsed),
            "message": "Full ML Training completed with warnings." if terminal_status == "completed_with_warnings" else "Full ML Training completed.",
            "completed_models": completed_models, "failed_models": failed_models, "skipped_models": skipped_models, "total_models": len(MODEL_SPECS),
        })
        _set_dataset_phase(dataset_id, "completed", job_id)
        _write_status(status)
        _refresh_runtime_kpi(status["duration_display"], dataset_id, job_id)
        log("info", status["message"])
        return status
    except StaleTrainingJob:
        status.update({"status": "cancelled", "phase": "cancelled", "current_step": "Cancelled", "current_model": ""})
        logging.getLogger(__name__).info("Ignored obsolete training job %s for dataset %s.", job_id, dataset_id)
        return status
    except Exception as exc:
        elapsed = time.perf_counter() - start_timer
        status.update(
            {
                "status": "failed",
                "phase": "failed",
                "current_step": "Full ML Training failed",
                "current_model": "",
                "end_time": datetime.now().isoformat(timespec="seconds"),
                "duration_seconds": round(elapsed, 2),
                "duration_display": _duration_display(elapsed),
                "message": _friendly_error(exc),
                "failed_stage": status.get("phase"),
            }
        )
        if _job_is_current(dataset_id, job_id):
            _set_dataset_phase(dataset_id, "failed", job_id)
        _write_status(status)
        log("error", status["message"])
        return status
    finally:
        terminal_states = {"completed", "completed_with_warnings", "failed", "cancelled", "budget_exceeded"}
        if status.get("status") not in terminal_states:
            status.update({"status": "failed", "phase": "failed", "current_step": "Failed during finalization", "message": "Training exited without publishing a terminal state."})
        finished = datetime.now().isoformat(timespec="seconds")
        status.update({"current_model": "", "current_artifact": "", "end_time": status.get("end_time") or finished, "finished_at": status.get("finished_at") or finished, "last_heartbeat": finished})
        if _job_is_current(dataset_id, job_id):
            _write_status(status)
        with _THREAD_LOCK:
            if _TRAINING_THREAD is threading.current_thread():
                _TRAINING_THREAD = None
                _TRAINING_THREAD_NAMESPACE = None


def _fitters():
    return {
        "_fit_sarimax": _fit_sarimax,
        "_fit_sarimax_exog": _fit_sarimax_exog,
        "_fit_auto_arima": _fit_auto_arima,
        "_fit_auto_arima_exog": _fit_auto_arima_exog,
        "_fit_xgboost": _fit_xgboost,
        "_fit_xgboost_exog": _fit_xgboost_exog,
        "_fit_exp_additive": _fit_exp_additive,
        "_fit_exp_additive_damped": _fit_exp_additive_damped,
        "_fit_exp_multiplicative": _fit_exp_multiplicative,
        "_fit_exp_multiplicative_damped": _fit_exp_multiplicative_damped,
        "_fit_var": _fit_var,
        "_fit_var_exog": _fit_var_exog,
        "_fit_lstm": _fit_lstm,
    }


def _model_eligibility(model_id, fitter_name, training_df, dependency_registry=None):
    dependency = dependency_service.model_dependency_status(model_id, dependency_registry)
    if not dependency["available"]:
        module = dependency["module"] or "required package"
        return {
            "eligible": False,
            "reason_code": "dependency_unavailable",
            "reason": f"{module} is not installed in the active server environment: {dependency.get('import_error') or 'import unavailable'}.",
        }
    if model_id in {"var_Predictions", "var_exog_Predictions"} and len(_endogenous_columns(training_df)) < 2:
        return {
            "eligible": False,
            "reason_code": "insufficient_endogenous_series",
            "reason": "VAR requires at least two aligned mapped target series.",
        }
    return {"eligible": True, "reason_code": None, "reason": None}


def _ensure_training_input(dataset_mapping, dataset_id):
    _dataset, adapted = _active_dataset_for_training(dataset_id)
    return {**adapted, "dataset_id": dataset_id}


def _load_cleaned_frame():
    import pandas as pd

    df = pd.read_csv(preprocessed_path("cleaned_training_input.csv"))
    date_column = "Date" if "Date" in df.columns else "date"
    target_column = "Click Count" if "Click Count" in df.columns else "target"
    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame[date_column], errors="coerce")
    frame["Click Count"] = pd.to_numeric(frame[target_column], errors="coerce")
    dimension_columns = sorted(column for column in frame.columns if str(column).startswith("dimension_"))
    for column in dimension_columns:
        frame[column] = frame[column].fillna("(missing)").astype(str)
    sort_columns = ["Date"] + dimension_columns
    return frame.dropna(subset=["Date", "Click Count"]).sort_values(sort_columns).reset_index(drop=True)


def _training_series(cleaned_df):
    import pandas as pd

    source_exog = _source_exog_columns(cleaned_df)
    dimension_columns = sorted(column for column in cleaned_df.columns if str(column).startswith("dimension_"))
    panel = None
    if dimension_columns:
        panel = cleaned_df.pivot_table(
            index="Date", columns=dimension_columns, values="Click Count", aggfunc="sum", dropna=True, observed=True,
        ).sort_index()
        if panel.shape[1] >= 2:
            panel.columns = [f"endogenous_{index + 1}" for index in range(panel.shape[1])]
            panel = panel.reset_index()
        else:
            panel = None
    aggregations = {"Click Count": "sum"}
    for column in source_exog:
        numeric = pd.to_numeric(cleaned_df[column], errors="coerce")
        aggregations[column] = "mean" if numeric.notna().mean() >= 0.8 else _mode_or_last
    frame = cleaned_df.groupby("Date", as_index=False, dropna=False).agg(aggregations)
    frame = frame.rename(columns={"Date": "date", "Click Count": "target"}).sort_values("date")
    frame["date"] = pd.to_datetime(frame["date"])
    if panel is not None:
        panel = panel.rename(columns={"Date": "date"})
        panel["date"] = pd.to_datetime(panel["date"])
        frame = frame.merge(panel, on="date", how="left", validate="one_to_one")
    from services.preprocessing_service import add_exogenous_features

    return add_exogenous_features(frame.reset_index(drop=True))


def _mode_or_last(values):
    modes = values.dropna().mode()
    if len(modes):
        return modes.iloc[0]
    non_null = values.dropna()
    return non_null.iloc[-1] if len(non_null) else None


def _generate_series_artifacts(cleaned_df, dataset_info, metrics, fitters, progress=None):
    """Generate owned per-series artifacts only when the portfolio can do so safely.

    Large panels remain explicitly aggregate-only instead of receiving allocated aggregate forecasts.
    """
    import pandas as pd

    dimension_schema = dataset_info.get("dimension_schema") or []
    model_names = [row["model"] for row in metrics]
    model_labels = {row["model"]: row.get("model_label") or row["model"] for row in metrics}
    total_series = series_count(cleaned_df, dimension_schema)
    planned_total = total_series * len(model_names)
    expected_artifacts = 0
    not_applicable_count = planned_total
    skip_records = []

    def emit(generated=0, failed=0, skipped=0, current_artifact="", budget_exceeded=False):
        snapshot = {
            "expected_artifacts": expected_artifacts, "generated_artifacts": generated,
            "failed_artifacts": failed, "skipped_artifacts": skipped,
            "terminal_artifacts": generated + failed + skipped,
            "planned_total": planned_total, "not_applicable_artifacts": not_applicable_count,
            "expected_backtest_artifacts": expected_artifacts, "expected_future_artifacts": expected_artifacts,
            "generated_backtest_artifacts": generated, "generated_future_artifacts": generated,
            "expected_interval_artifacts": 0,
            "current_artifact": current_artifact, "budget_exceeded": budget_exceeded,
        }
        if progress:
            progress(snapshot)
        return snapshot

    if not dimension_schema:
        scopes = {model: {"scope": "single_series", "supported_series_count": 1, "total_series_count": 1, "series_failure_count": 0} for model in model_names}
        return {"backtest_predictions": [], "future_predictions": [], "failures": [], "skips": [], "progress": emit()}, scopes

    max_series = max(0, int(os.getenv("FORECAST_MAX_PER_SERIES_MODELS", "250")))
    if total_series > max_series or max_series == 0:
        scopes = {model: {"scope": "aggregate_only", "scope_reason": "series_limit_exceeded", "supported_series_count": 0, "total_series_count": total_series, "series_failure_count": 0, "supported_series_hashes": []} for model in model_names}
        return {"backtest_predictions": [], "future_predictions": [], "failures": [], "skips": [], "progress": emit()}, scopes

    per_series_model_limit = max(1, int(os.getenv("FORECAST_PER_SERIES_MODEL_LIMIT", "4")))
    default_ids = "exp_additive_Predictions,exp_additive_damped_Predictions,exp_multiplicative_Predictions,exp_multiplicative_damped_Predictions"
    allowed_models = {value.strip() for value in os.getenv("FORECAST_PER_SERIES_MODEL_IDS", default_ids).split(",") if value.strip()}
    ranked_models = sorted((row for row in metrics if row["model"] in allowed_models and row.get("model_scope") != "aggregate_only"), key=lambda row: float(row.get("mape", float("inf"))))
    eligible_models = {row["model"] for row in ranked_models[:per_series_model_limit]}
    expected_artifacts = total_series * len(eligible_models)
    not_applicable_count = planned_total - expected_artifacts
    spec_lookup = {model: fitter_name for model, _label, fitter_name in MODEL_SPECS}
    dimension_columns = [dimension["canonical_column"] for dimension in dimension_schema]
    backtests = []
    futures = []
    supported = {model: set() for model in model_names}
    failures = []
    generated_count = failed_count = 0
    skipped_count = 0
    phase_started = time.perf_counter()
    phase_budget = max(1.0, float(os.getenv("FORECAST_SERIES_PHASE_BUDGET_SECONDS", "300")))
    artifact_budget = max(1.0, float(os.getenv("FORECAST_SERIES_ARTIFACT_TIMEOUT_SECONDS", "30")))
    budget_exceeded = False
    emit(generated_count, failed_count, skipped_count)
    grouped = cleaned_df.groupby(dimension_columns, dropna=False, sort=False)
    if grouped.ngroups != total_series:
        raise RuntimeError("Observed series count does not match grouped series construction.")
    observed_groups = list(grouped)
    artifact_plan = []
    plan_lookup = {}
    for group_key, _group in observed_groups:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        key_source = {dimension["id"]: str(value) for dimension, value in zip(dimension_schema, key_values)}
        identity = row_series_identity(key_source, dimension_schema)
        for model in model_names:
            applicable = model in eligible_models
            item = {
                "model_id": model, **identity, "prediction_type": "historical_backtest+future_forecast",
                "required": applicable, "applicable": applicable,
                "status": "planned" if applicable else "not_applicable",
                "reason_code": None if applicable else "unsupported_series_scope",
                "reason": None if applicable else "The configured model scope does not require an individual-series artifact.",
            }
            artifact_plan.append(item)
            plan_lookup[(model, identity["series_key_hash"])] = item
    attempted = set()
    for group_key, group in observed_groups:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        key_source = {dimension["id"]: str(value) for dimension, value in zip(dimension_schema, key_values)}
        identity = row_series_identity(key_source, dimension_schema)
        series_frame = _training_series(group)
        frequency = _infer_frequency(series_frame)
        future_frame = _future_frame(series_frame, frequency, history_length=len(series_frame))
        future_frame["target"] = pd.NA
        for model in model_names:
            if model not in eligible_models:
                continue
            current_artifact = f"{model}:{identity['series_key_hash']}"
            if time.perf_counter() - phase_started >= phase_budget:
                budget_exceeded = True
                skipped_count = expected_artifacts - generated_count - failed_count
                emit(generated_count, failed_count, skipped_count, current_artifact, True)
                break
            attempt_key = (model, identity["series_key_hash"], "series_forecast")
            if attempt_key in attempted:
                raise RuntimeError(f"Duplicate series artifact attempt detected: {current_artifact}")
            attempted.add(attempt_key)
            reason_code = _series_ineligibility(model, series_frame, frequency)
            if reason_code:
                skipped_count += 1
                diagnostic = _series_artifact_diagnostic(dataset_info, model, model_labels[model], identity, series_frame, frequency, reason_code)
                plan_lookup[(model, identity["series_key_hash"])].update(status="skipped", reason_code=reason_code, reason=diagnostic["reason"])
                skip_records.append(diagnostic)
                emit(generated_count, failed_count, skipped_count, current_artifact)
                continue
            fitter_name = spec_lookup.get(model)
            if not fitter_name:
                skipped_count += 1
                diagnostic = _series_artifact_diagnostic(dataset_info, model, model_labels[model], identity, series_frame, frequency, "model_dependency_unavailable")
                plan_lookup[(model, identity["series_key_hash"])].update(status="skipped", reason_code="model_dependency_unavailable", reason=diagnostic["reason"])
                skip_records.append(diagnostic)
                emit(generated_count, failed_count, skipped_count, current_artifact)
                continue
            artifact_started = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    evaluation = _rolling_origin_evaluation(fitters[fitter_name], series_frame)
                    future_values = _clean_predictions(fitters[fitter_name](series_frame, future_frame), len(future_frame))
                offsets = _empirical_interval_offsets(evaluation.get("predictions", []))
                for row in evaluation.get("predictions", []):
                    interval = _apply_interval(row["predicted"], offsets) if row.get("predicted") is not None and offsets else {}
                    backtests.append({
                        **row, **interval, **identity,
                        "timestamp": row.get("date"), "prediction": row.get("predicted"),
                        "model_id": model, "model_scope": "per_series", "prediction_type": "historical_backtest",
                        "source_frequency": frequency,
                    })
                for horizon_step, (timestamp, prediction) in enumerate(zip(future_frame["date"], future_values), start=1):
                    value = _none_or_float(prediction)
                    if value is None:
                        continue
                    interval = _apply_interval(value, offsets) if offsets else {}
                    timestamp_text = _timestamp_text(timestamp)
                    futures.append({
                        **identity, **interval, "model_id": model,
                        "model_scope": "per_series", "prediction_type": "future_forecast", "date": timestamp_text,
                        "timestamp": timestamp_text, "forecast_origin": None, "fold": None,
                        "horizon_step": horizon_step, "actual": None,
                        "predicted": value, "prediction": value, "source_frequency": frequency,
                    })
                if evaluation.get("predictions") and len(future_values):
                    supported[model].add(identity["series_key_hash"])
                    generated_count += 1
                    plan_lookup[(model, identity["series_key_hash"])]["status"] = "generated"
                else:
                    skipped_count += 1
                    diagnostic = _series_artifact_diagnostic(dataset_info, model, model_labels[model], identity, series_frame, frequency, "no_valid_future_forecast")
                    plan_lookup[(model, identity["series_key_hash"])].update(status="skipped", reason_code="no_valid_future_forecast", reason=diagnostic["reason"])
                    skip_records.append(diagnostic)
            except Exception as exc:
                failed_count += 1
                plan_lookup[(model, identity["series_key_hash"])].update(status="failed", reason_code="generation_exception", reason=_friendly_error(exc))
                failures.append({
                    **identity,
                    "model_id": model,
                    "model_display_name": model_labels[model], "model_scope": "per_series",
                    "prediction_type": "historical_backtest+future_forecast", "reason_code": "generation_exception",
                    "reason": _friendly_error(exc),
                })
            if time.perf_counter() - artifact_started > artifact_budget:
                budget_exceeded = True
            emit(generated_count, failed_count, skipped_count, current_artifact, budget_exceeded)
        if budget_exceeded:
            skipped_count = expected_artifacts - generated_count - failed_count
            break
    if budget_exceeded:
        for group_key, group in observed_groups:
            key_values = group_key if isinstance(group_key, tuple) else (group_key,)
            key_source = {dimension["id"]: str(value) for dimension, value in zip(dimension_schema, key_values)}
            identity = row_series_identity(key_source, dimension_schema)
            series_frame = _training_series(group)
            frequency = _infer_frequency(series_frame)
            for model in eligible_models:
                attempt_key = (model, identity["series_key_hash"], "series_forecast")
                if attempt_key not in attempted:
                    attempted.add(attempt_key)
                    diagnostic = _series_artifact_diagnostic(dataset_info, model, model_labels[model], identity, series_frame, frequency, "resource_limit")
                    plan_lookup[(model, identity["series_key_hash"])].update(status="skipped", reason_code="resource_limit", reason=diagnostic["reason"])
                    skip_records.append(diagnostic)
    scopes = {
        model: {
            "scope": "per_series" if supported[model] else "aggregate_only",
            "scope_reason": None if supported[model] else "not_selected_for_per_series_generation" if model not in eligible_models else "series_generation_failed",
            "supported_series_count": len(supported[model]),
            "total_series_count": total_series,
            "series_failure_count": sum(row["model_id"] == model for row in failures),
            "supported_series_hashes": sorted(supported[model]),
        }
        for model in model_names
    }
    summary = emit(generated_count, failed_count, skipped_count, "", budget_exceeded)
    summary["unique_skipped_model_series_pairs"] = len({(row["model_id"], row["series_key_hash"]) for row in skip_records})
    summary["skip_reason_counts"] = {code: sum(row["reason_code"] == code for row in skip_records) for code in sorted({row["reason_code"] for row in skip_records})}
    if summary["terminal_artifacts"] != summary["expected_artifacts"]:
        raise RuntimeError("Series artifact terminal counters did not reconcile with the expected count.")
    if summary["unique_skipped_model_series_pairs"] != summary["skipped_artifacts"]:
        raise RuntimeError("Skipped series artifact diagnostics did not reconcile with the unique skipped count.")
    if len(artifact_plan) != planned_total or any(row["status"] == "planned" for row in artifact_plan):
        raise RuntimeError("Series artifact plan did not reach a terminal status for every model-series pair.")
    return {"backtest_predictions": backtests, "future_predictions": futures, "failures": failures, "skips": skip_records, "plan": artifact_plan, "progress": summary}, scopes


def _series_ineligibility(model_id, series_frame, frequency):
    values = series_frame["target"].dropna().astype(float)
    if values.empty:
        return "invalid_series"
    if len(values) < 10:
        return "insufficient_history"
    if "exp_" in model_id:
        period = _seasonal_period(frequency)
        if period and len(values) < max(24, period * 2):
            return "insufficient_seasonal_cycles"
    if "multiplicative" in model_id and (values <= 0).any():
        return "incompatible_nonpositive_values"
    return None


def _series_artifact_diagnostic(dataset_info, model_id, model_label, identity, series_frame, frequency, reason_code):
    values = series_frame["target"].dropna().astype(float)
    period = _seasonal_period(frequency)
    required = max(24, period * 2) if "exp_" in model_id and period else 10
    reason_text = {
        "insufficient_history": "The observed series has too few valid timestamps for an honest backtest.",
        "insufficient_seasonal_cycles": "The observed series does not contain enough complete seasonal cycles.",
        "incompatible_nonpositive_values": "Multiplicative forecasting requires strictly positive target values.",
        "model_dependency_unavailable": "The configured model fitter is unavailable.",
        "no_valid_future_forecast": "The model produced no valid owned future forecast.",
        "invalid_series": "The observed series contains no valid target values.",
        "resource_limit": "The configured series-artifact generation budget was exhausted.",
    }.get(reason_code, reason_code.replace("_", " ").capitalize())
    return {
        "dataset_id": dataset_info.get("dataset_id"), "artifact_id": dataset_info.get("artifact_id"), "job_id": dataset_info.get("job_id"),
        "model_id": model_id, "model_display_name": model_label, "model_scope": "per_series",
        **identity, "prediction_type": "historical_backtest+future_forecast", "reason_code": reason_code, "reason": reason_text,
        "observations_available": len(values), "observations_required": required, "detected_frequency": frequency,
        "seasonal_period": period, "positive_target_count": int((values > 0).sum()), "zero_target_count": int((values == 0).sum()),
        "negative_target_count": int((values < 0).sum()), "exogenous_availability": "calendar_only",
        "future_exogenous_availability": "generated_calendar_only", "aggregate_fallback_attempted": False,
    }


def _split_training_output(df):
    import pandas as pd

    holdout_size = _holdout_size(len(df))
    train_df = df.iloc[:-holdout_size].copy()
    holdout_df = df.iloc[-holdout_size:].copy()
    frequency = _infer_frequency(df)
    future_df = _future_frame(df, frequency)
    future_df["target"] = pd.NA
    output_df = pd.concat([holdout_df, future_df], ignore_index=True)
    return train_df, output_df, len(holdout_df), frequency


def _prediction_frame(output_df):
    frame = output_df[["date", "target"]].copy()
    frame["date_var"] = frame["date"]
    return frame[["date", "date_var", "target"]]


def _rolling_origin_evaluation(fitter, df):
    validation_size = _fold_validation_size(len(df))
    folds = _rolling_folds(len(df), validation_size)
    if len(folds) < 3:
        train_df, output_df, holdout_size, _frequency = _split_training_output(df)
        validation = output_df.iloc[:holdout_size].copy()
        predictions = _clean_predictions(fitter(train_df, validation), len(validation))
        return {
            "mode": "Holdout fallback",
            "folds": 1,
            "reason": "Dataset too small for 3 rolling-origin folds; used honest holdout fallback.",
            "metrics": calculate_metrics(validation["target"], predictions),
            "predictions": _backtest_records(train_df, validation, predictions, 1, "chronological_holdout"),
        }

    fold_metrics = []
    prediction_records = []
    for fold_number, (train_end, validation_end) in enumerate(folds, start=1):
        fold_train = df.iloc[:train_end].copy()
        fold_validation = df.iloc[train_end:validation_end].copy()
        predictions = _clean_predictions(fitter(fold_train, fold_validation), len(fold_validation))
        fold_metrics.append(calculate_metrics(fold_validation["target"], predictions))
        prediction_records.extend(
            _backtest_records(fold_train, fold_validation, predictions, fold_number, "rolling_origin")
        )
    return {
        "mode": "Rolling-origin evaluation",
        "folds": len(fold_metrics),
        "reason": "",
        "metrics": _average_metrics(fold_metrics),
        "predictions": prediction_records,
    }


def _backtest_records(train_df, validation_df, predictions, fold, validation_method):
    """Persist every honest validation prediction; training always ends before its timestamp."""
    origin = train_df["date"].max()
    origin_text = _timestamp_text(origin)
    records = []
    for horizon_step, ((_, row), predicted) in enumerate(zip(validation_df.iterrows(), predictions), start=1):
        timestamp = row["date"]
        timestamp_text = _timestamp_text(timestamp)
        if origin_text >= timestamp_text:
            raise ValueError("Backtest forecast origin must precede its prediction timestamp.")
        records.append(
            {
                "date": timestamp_text,
                "forecast_origin": origin_text,
                "fold": fold,
                "horizon_step": horizon_step,
                "validation_method": validation_method,
                "actual": _none_or_float(row.get("target")),
                "predicted": _none_or_float(predicted),
            }
        )
    return records


def _fit_sarimax(train_df, output_df):
    return _sarimax_forecast(train_df, output_df, use_exog=False)


def _fit_sarimax_exog(train_df, output_df):
    return _sarimax_forecast(train_df, output_df, use_exog=True)


def _sarimax_forecast(train_df, output_df, use_exog):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    _require_rows(train_df, 12, "SARIMAX")
    exog_train, exog_future = _exog_pair(train_df, output_df) if use_exog else (None, None)
    model = SARIMAX(
        train_df["target"].astype(float),
        exog=exog_train,
        order=(2, 1, 1),
        seasonal_order=(0, 0, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return model.forecast(steps=len(output_df), exog=exog_future)


def _fit_auto_arima(train_df, output_df):
    return _auto_arima_forecast(train_df, output_df, use_exog=False)


def _fit_auto_arima_exog(train_df, output_df):
    return _auto_arima_forecast(train_df, output_df, use_exog=True)


def _auto_arima_forecast(train_df, output_df, use_exog):
    import pmdarima as pm

    _require_rows(train_df, 24, "Auto ARIMA")
    x_train, x_future = _exog_pair(train_df, output_df) if use_exog else (None, None)
    model = pm.auto_arima(
        train_df["target"].astype(float),
        X=x_train,
        start_p=1,
        start_q=1,
        max_p=3,
        max_q=3,
        max_d=2,
        m=12,
        seasonal=True,
        max_P=1,
        max_D=2,
        max_Q=1,
        max_order=8,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        trace=False,
    )
    return model.predict(n_periods=len(output_df), X=x_future)


def _fit_xgboost(train_df, output_df):
    return _xgboost_forecast(train_df, output_df, use_exog=False)


def _fit_xgboost_exog(train_df, output_df):
    return _xgboost_forecast(train_df, output_df, use_exog=True)


def _xgboost_forecast(train_df, output_df, use_exog):
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBRegressor

    x_train, y_train = _xgb_training_matrix(train_df, use_exog)
    if len(x_train) < 5:
        raise ValueError("Not enough lagged rows for XGBoost.")
    params = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "min_child_weight": [1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "n_estimators": [100, 200],
    }
    # Keep the search bounded and avoid nested process/thread oversubscription
    # inside the already-backgrounded Flask training job.
    n_jobs = max(1, int(os.getenv("FORECAST_XGB_N_JOBS", "1")))
    reg = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=n_jobs)
    grid = GridSearchCV(reg, params, cv=2, n_jobs=n_jobs, verbose=0)
    grid.fit(x_train, y_train)
    model = grid.best_estimator_

    history = train_df["target"].astype(float).tolist()
    predictions = []
    feature_columns = list(x_train.columns)
    exog_schema = x_train.attrs.get("exog_schema", [])
    for _index, row in output_df.iterrows():
        features = pd.DataFrame([_xgb_features(row["date"], history, use_exog, row=row, exog_schema=exog_schema)])
        features = features.reindex(columns=feature_columns, fill_value=0.0)
        prediction = float(model.predict(features)[0])
        predictions.append(prediction)
        history.append(float(row["target"]) if pd.notna(row.get("target")) else prediction)
    return predictions


def _fit_exp_additive(train_df, output_df):
    return _exp_forecast(train_df, output_df, seasonal="add", damped=False)


def _fit_exp_additive_damped(train_df, output_df):
    return _exp_forecast(train_df, output_df, seasonal="add", damped=True)


def _fit_exp_multiplicative(train_df, output_df):
    return _exp_forecast(train_df, output_df, seasonal="mul", damped=False)


def _fit_exp_multiplicative_damped(train_df, output_df):
    return _exp_forecast(train_df, output_df, seasonal="mul", damped=True)


def _exp_forecast(train_df, output_df, seasonal, damped):
    import pandas as pd
    from statsmodels.tsa.api import ExponentialSmoothing

    _require_rows(train_df, 24, "Exponential Smoothing")
    y = pd.Series(train_df["target"].astype(float).values)
    if seasonal == "mul" and (y <= 0).any():
        raise ValueError("Multiplicative forecasting requires strictly positive target values.")
    seasonal_period = _seasonal_period(_infer_frequency(train_df))
    if not seasonal_period or len(y) < seasonal_period * 2:
        raise ValueError("Exponential Smoothing requires at least two complete seasonal cycles.")
    try:
        model = ExponentialSmoothing(
            y,
            seasonal_periods=seasonal_period,
            trend="add",
            seasonal=seasonal,
            damped_trend=damped,
            initialization_method="estimated",
        ).fit(optimized=True)
    except TypeError:
        model = ExponentialSmoothing(
            y,
            seasonal_periods=seasonal_period,
            trend="add",
            seasonal=seasonal,
            damped=damped,
        ).fit(optimized=True)
    return model.forecast(len(output_df))


def _fit_var(train_df, output_df):
    from statsmodels.tsa.api import VAR

    endog = _var_endog(train_df)
    if endog.shape[1] < 2:
        raise ValueError("VAR requires at least two numeric time-series columns.")
    maxlags = min(7, max(1, len(endog) // 10))
    model = VAR(endog).fit(maxlags=maxlags, ic=None, trend="c")
    lag_order = model.k_ar
    forecast_input = endog.values[-lag_order:]
    return model.forecast(forecast_input, steps=len(output_df)).sum(axis=1)


def _fit_var_exog(train_df, output_df):
    from statsmodels.tsa.statespace.varmax import VARMAX

    exog_train, exog_future = _exog_pair(train_df, output_df)
    endog = _var_endog(train_df)
    if endog.shape[1] < 2:
        raise ValueError("VAR exogenous model requires at least two endogenous series.")
    model = VARMAX(
        endog,
        exog=exog_train,
        order=(1, 0),
        trend="c",
        enforce_stationarity=False,
    ).fit(disp=False, maxiter=100)
    forecast = model.forecast(steps=len(output_df), exog=exog_future)
    return forecast.sum(axis=1)


def _fit_lstm(train_df, output_df):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.models import Sequential

    sequence_length = min(_lstm_window(), max(3, len(train_df) // 4))
    if len(train_df) <= sequence_length + 2:
        raise ValueError("Not enough rows for LSTM sequence training.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_df[["target"]].astype(float)).reshape(-1)
    x_train = []
    y_train = []
    for index in range(sequence_length, len(scaled)):
        x_train.append(scaled[index - sequence_length:index])
        y_train.append(scaled[index])
    x_train = np.array(x_train).reshape((-1, sequence_length, 1))
    y_train = np.array(y_train)

    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.fit(
        x_train,
        y_train,
        epochs=_lstm_epochs(),
        batch_size=_lstm_batch_size(),
        verbose=0,
        shuffle=False,
    )

    history = list(scaled)
    predictions = []
    for _index, row in output_df.iterrows():
        window = np.array(history[-sequence_length:]).reshape((1, sequence_length, 1))
        predicted_scaled = float(model.predict(window, verbose=0)[0][0])
        predicted = float(scaler.inverse_transform([[predicted_scaled]])[0][0])
        predictions.append(predicted)
        if pd.notna(row.get("target")):
            actual_scaled = float(scaler.transform([[float(row["target"])]])[0][0])
            history.append(actual_scaled)
        else:
            history.append(predicted_scaled)
    return predictions


def _xgb_training_matrix(train_df, use_exog):
    import pandas as pd

    history = train_df["target"].astype(float).tolist()
    exog_schema = _fit_source_exog_schema(train_df) if use_exog else []
    rows = []
    targets = []
    for index in range(1, len(train_df)):
        row = train_df.iloc[index]
        rows.append(_xgb_features(row["date"], history[:index], use_exog, row=row, exog_schema=exog_schema))
        targets.append(history[index])
    matrix = pd.DataFrame(rows)
    matrix.attrs["exog_schema"] = exog_schema
    return matrix, pd.Series(targets)


def _xgb_features(date_value, history, use_exog, row=None, exog_schema=None):
    import pandas as pd

    date_value = pd.Timestamp(date_value)
    features = {
        "date_ordinal": date_value.toordinal(),
        "dayofweek": date_value.dayofweek,
        "month": date_value.month,
        "dayofyear": date_value.dayofyear,
        "lag_1": _lag(history, 1),
        "lag_7": _lag(history, 7),
        "lag_14": _lag(history, 14),
        "lag_28": _lag(history, 28),
        "rolling_7": _rolling(history, 7),
        "rolling_28": _rolling(history, 28),
        "rolling_median_7": _rolling_median(history, 7),
        "rolling_std_7": _rolling_std(history, 7),
        "trend_index": len(history),
    }
    if use_exog:
        features.update(
            {
                "week_exog": int(date_value.isocalendar().week),
                "dayOfWeek_exog": date_value.dayofweek,
                "month_exog": date_value.month,
            }
        )
        features.update(_encoded_source_exog(row, exog_schema or []))
    return features


def _write_training_outputs(
    prediction_frame,
    metrics,
    dataset_info,
    cleaned_df,
    frequency,
    status,
    backtest_predictions=None,
    series_artifacts=None,
):
    import pandas as pd

    dataset_id = dataset_info.get("dataset_id") or status.get("dataset_id") or current_dataset_id()
    job_id = status.get("job_id")
    artifact_id = status.get("artifact_id") or dataset_info.get("artifact_id")
    _ensure_job_active(dataset_id, job_id)

    prediction_out = prediction_frame.copy()
    prediction_out["date"] = pd.to_datetime(prediction_out["date"]).dt.date.astype(str)
    prediction_out["date_var"] = pd.to_datetime(prediction_out["date_var"]).dt.date.astype(str)
    prediction_path = preprocessed_path("prediction_new.csv")
    prediction_out.to_csv(prediction_path, index=False)

    accuracy_rows = [
        {
            "Models": row["model"],
            "MAPE": row["mape"],
            "RMSE Errors": row["rmse"],
            "Accuracy": row["accuracy"],
            "MAE": row["mae"],
            "WAPE": row["wape"],
            "Bias": row["bias"],
            "Holdout MAPE": row.get("holdout_mape"),
            "Evaluation Mode": row.get("evaluation_mode"),
            "Folds": row.get("folds"),
            "Runtime Seconds": row.get("runtime_seconds"),
            "frequency": "daily" if str(frequency).startswith("D") else str(frequency),
        }
        for row in metrics
    ]
    accuracy_df = pd.DataFrame(accuracy_rows)
    accuracy_df.to_csv(preprocessed_path("Accuracy.csv"), index=False)
    accuracy_df.to_csv(preprocessed_path("MAPE_Report.csv"), index=False)

    champion = metrics[0]
    runtime = status.get("duration_display", "0 sec")
    enrichment = load_json("enrichment.json", {})
    quality_score = enrichment.get("quality", {}).get("score", "--")
    forecast_meta = _forecast_dates(prediction_frame)
    kpis = [
        {"label": "Best Model", "value": champion["model"], "caption": "Champion model"},
        {"label": "MAPE", "value": f"{champion['mape']:.2f}%", "caption": champion.get("evaluation_mode", "Rolling-origin evaluation")},
        {"label": "Forecast Accuracy", "value": f"{champion['accuracy']:.2f}%", "caption": "100 - MAPE"},
        {"label": "MAE", "value": f"{champion['mae']:.2f}", "caption": "Mean absolute error"},
        {"label": "RMSE", "value": f"{champion['rmse']:.2f}", "caption": "Root mean squared error"},
        {"label": "WAPE", "value": f"{champion['wape']:.2f}%", "caption": "Weighted absolute percentage error"},
        {"label": "Bias", "value": f"{champion['bias']:.2f}%", "caption": "Forecast tendency"},
        {"label": "Last Training Runtime", "value": runtime, "caption": "Most recent run"},
        {"label": "Data Quality Score", "value": f"{quality_score}/100" if quality_score != "--" else "--", "caption": "Cleanliness and completeness"},
        {"label": "Forecast Start Date", "value": forecast_meta["forecast_start_date"] or "--", "caption": "First future prediction"},
        {"label": "Forecast Horizon", "value": str(forecast_meta["future_points"]), "caption": f"Future points at {frequency} frequency"},
    ]

    save_json("model_metrics.json", [{**row, "dataset_id": dataset_id, "job_id": job_id, "artifact_id": artifact_id} for row in metrics])
    save_json("kpis.json", [{**row, "dataset_id": dataset_id, "job_id": job_id, "artifact_id": artifact_id} for row in kpis])
    forecast_payload = _forecast_payload(
        prediction_frame,
        metrics,
        dataset_info,
        cleaned_df,
        frequency,
        backtest_predictions=backtest_predictions,
    )
    ownership = {"dataset_id": dataset_id, "artifact_id": artifact_id, "job_id": job_id}
    aggregate_identity = aggregate_series_identity()
    for row in forecast_payload.get("backtest_predictions", []):
        row.update({
            **ownership, **aggregate_identity,
            "timestamp": row.get("timestamp") or row.get("date"),
            "prediction": row.get("prediction", row.get("predicted")),
            "model_scope": "aggregate_only", "source_frequency": frequency,
        })
    long_future = []
    for row in forecast_payload.get("future", []):
        for model_id, prediction in (row.get("predictions") or {}).items():
            if prediction is None:
                continue
            band = (row.get("confidence_bands") or {}).get(model_id, {})
            long_future.append(
                {
                    **ownership,
                    **aggregate_identity,
                    "model_id": model_id,
                    "model_scope": "aggregate_only", "source_frequency": frequency,
                    "prediction_type": "future_forecast",
                    "date": row.get("date"),
                    "timestamp": row.get("date"),
                    "forecast_origin": None,
                    "fold": None,
                    "horizon_step": row.get("horizon_step"),
                    "actual": None,
                    "store": row.get("store"),
                    "item": row.get("item"),
                    "predicted": prediction,
                    "prediction": prediction,
                    "lower": band.get("lower"),
                    "upper": band.get("upper"),
                }
            )
    series_artifacts = series_artifacts or {}
    for row in series_artifacts.get("backtest_predictions", []):
        forecast_payload.setdefault("backtest_predictions", []).append({
            **row, **ownership,
            "timestamp": row.get("timestamp") or row.get("date"),
            "prediction": row.get("prediction", row.get("predicted")),
        })
    long_future.extend({
        **row, **ownership,
        "timestamp": row.get("timestamp") or row.get("date"),
        "prediction": row.get("prediction", row.get("predicted")),
    } for row in series_artifacts.get("future_predictions", []))
    forecast_payload["future_predictions"] = long_future
    forecast_payload["series_generation_failures"] = [
        {**row, **ownership} for row in series_artifacts.get("failures", [])
    ]
    forecast_payload["series_generation_skips"] = [
        {**row, **ownership} for row in series_artifacts.get("skips", [])
    ]
    forecast_payload["series_artifact_plan"] = [
        {**row, **ownership} for row in series_artifacts.get("plan", [])
    ]
    forecast_payload["model_registry"] = [
        {
            "model_id": row["model"], "display_name": row.get("model_label") or row["model"],
            "status": row.get("status", "completed"), "scope": row.get("model_scope", "aggregate_only"),
            "supported_series_count": row.get("supported_series_count", 0),
            "total_series_count": row.get("total_series_count", dataset_info.get("series_count", 1)),
            "series_failure_count": row.get("series_failure_count", 0),
            "scope_reason": row.get("scope_reason"),
            "supported_series_hashes": row.get("supported_series_hashes", []),
            **ownership,
        }
        for row in metrics
    ]
    _ensure_job_active(dataset_id, job_id)
    published_payload = {**forecast_payload, "dataset_id": dataset_id, "job_id": job_id, "artifact_id": artifact_id}
    save_json("forecast_data.json", published_payload)
    readable = load_json("forecast_data.json", {})
    if any(readable.get(key) != ownership[key] for key in ownership):
        raise RuntimeError("Published forecast artifact failed ownership validation.")
    manifest = _build_forecast_manifest(published_payload)
    save_json("forecast_manifest.json", manifest)
    readable_manifest = load_json("forecast_manifest.json", {})
    if readable_manifest.get("forecast_manifest_id") != manifest["forecast_manifest_id"]:
        raise RuntimeError("Forecast manifest publication could not be validated.")
    return {
        "forecast_ready": manifest["forecast_ready"],
        "usable_model_count": manifest["usable_model_count"],
        "usable_forecast_count": manifest["usable_forecast_count"],
        "forecast_manifest_id": manifest["forecast_manifest_id"],
        "forecast_progress": manifest["forecast_progress"],
        "artifact_publication": {"expected": 2, "published": 2, "failed": 0, "skipped": 0},
        "warnings": manifest["warnings"],
    }


def _build_forecast_manifest(payload):
    ownership = {key: payload.get(key) for key in ("dataset_id", "artifact_id", "job_id")}
    backtests = payload.get("backtest_predictions") or []
    futures = payload.get("future_predictions") or []
    failures = payload.get("series_generation_failures") or []
    entries = []
    for model in payload.get("model_registry") or []:
        model_id = model.get("model_id")
        model_backtests = [row for row in backtests if row.get("model_id") == model_id and _valid_owned_prediction(row, ownership)]
        model_futures = [row for row in futures if row.get("model_id") == model_id and _valid_owned_prediction(row, ownership)]
        model_failures = [row for row in failures if row.get("model_id") == model_id]
        status = "generated" if model_futures else "failed" if model_failures else "skipped"
        entries.append({
            **ownership, "model_id": model_id, "model_scope": model.get("scope", "aggregate_only"),
            "prediction_type": "historical_backtest+future_forecast" if model_backtests and model_futures else "future_forecast" if model_futures else "historical_backtest" if model_backtests else None,
            "supported_series_keys": model.get("supported_series_hashes", []),
            "historical_backtest_available": bool(model_backtests), "future_forecast_available": bool(model_futures),
            "prediction_interval_available": any(row.get("lower") is not None and row.get("upper") is not None for row in model_backtests + model_futures),
            "output_artifact_location": "forecast_data.json" if model_futures else None,
            "row_count": len(model_backtests) + len(model_futures), "status": status,
            "failure_reason": model_failures[0].get("reason") if model_failures and not model_futures else None,
        })
    generated = sum(row["status"] == "generated" for row in entries)
    failed = sum(row["status"] == "failed" for row in entries)
    skipped = sum(row["status"] == "skipped" for row in entries)
    if generated + failed + skipped != len(entries):
        raise RuntimeError("Forecast terminal counters exceeded the expected model count.")
    warnings = []
    if failed:
        warnings.append(f"{failed} model forecast(s) failed validation.")
    if skipped:
        warnings.append(f"{skipped} completed model(s) produced no usable future forecast.")
    return {
        **ownership, "forecast_manifest_id": uuid.uuid4().hex,
        "created_at": datetime.now().isoformat(timespec="seconds"), "status": "ready" if generated else "unavailable",
        "forecast_ready": generated > 0, "usable_model_count": generated, "usable_forecast_count": generated,
        "forecast_progress": {"expected": len(entries), "generated": generated, "failed": failed, "skipped": skipped, "budget_exceeded": 0, "terminal": generated + failed + skipped},
        "models": entries, "warnings": warnings,
    }


def _valid_owned_prediction(row, ownership):
    if any(row.get(key) != value for key, value in ownership.items()):
        return False
    try:
        value = float(row.get("prediction", row.get("predicted")))
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False


def _forecast_payload(prediction_frame, metrics, dataset_info, cleaned_df, frequency, backtest_predictions=None):
    model_names = [row["model"] for row in metrics]
    source_name = _dataset_name(dataset_info)
    historical_frame = _training_series(cleaned_df)
    backtest_predictions = backtest_predictions or {}
    dimension_schema = dataset_info.get("dimension_schema") or []
    configured_options = dataset_info.get("dimension_options") or {}
    aggregate_identity = aggregate_series_identity()
    interval_offsets = {
        model: _empirical_interval_offsets(backtest_predictions.get(model, []))
        for model in model_names
    }
    validation_dates = prediction_frame.loc[prediction_frame["target"].notna(), "date"]
    forecast_origin = historical_frame.loc[historical_frame["date"] < validation_dates.min(), "date"].max() if not validation_dates.empty else None
    validation = []
    future = []
    future_step = 0
    for _index, row in prediction_frame.iterrows():
        output_row = {
            "date": _timestamp_text(row["date"]),
            "store": "All Stores",
            "item": "All Items",
            "actual": _none_or_float(row["target"]),
            "predictions": {model: _none_or_float(row.get(model)) for model in model_names},
            **aggregate_identity,
        }
        if output_row["actual"] is None:
            future_step += 1
            output_row.update({"prediction_type": "future_forecast", "horizon_step": future_step})
            output_row["confidence_bands"] = {}
            for model, prediction in output_row["predictions"].items():
                if prediction is None:
                    continue
                offsets = interval_offsets.get(model)
                if offsets:
                    output_row["confidence_bands"][model] = _apply_interval(prediction, offsets)
        if output_row["actual"] is not None:
            output_row.update({"prediction_type": "historical_backtest", "forecast_origin": _timestamp_text(forecast_origin) if forecast_origin is not None else None, "fold": 1})
            validation.append(output_row)
        else:
            future.append(output_row)

    historical = [
        {
            "date": _timestamp_text(row.date),
            "actual": _none_or_float(row.target),
            "store": "All Stores",
            "item": "All Items",
        }
        for row in historical_frame.itertuples()
    ]
    last_actual_date = historical[-1]["date"] if historical else None
    forecast_start_date = future[0]["date"] if future else None
    forecast_end_date = future[-1]["date"] if future else None

    first_options = configured_options.get(dimension_schema[0]["id"], []) if dimension_schema else []
    second_options = configured_options.get(dimension_schema[1]["id"], []) if len(dimension_schema) > 1 else []
    stores = ["All Series"] + first_options
    items = ["All Series"] + second_options
    persisted_backtests = []
    for model in model_names:
        offsets = interval_offsets.get(model)
        for record in backtest_predictions.get(model, []):
            prediction = record.get("predicted")
            interval = _apply_interval(prediction, offsets) if prediction is not None and offsets else {}
            persisted_backtests.append(
                {
                    **record,
                    **interval,
                    "model_id": model,
                    "prediction_type": "historical_backtest",
                    **aggregate_identity,
                }
            )

    return {
        "mode": "trained",
        "source": dataset_info.get("source_file") or dataset_info.get("cleaned_path", ""),
        "frequency": frequency,
        "target_display_name": dataset_info.get("target_display_name") or dataset_info.get("target_column") or "Target",
        "target_unit": dataset_info.get("target_unit"),
        "domain": dataset_info.get("domain", "general"),
        "aggregation": dataset_info.get("aggregation", "sum"),
        "dimension_schema": dimension_schema,
        "dimension_options": configured_options,
        "series_count": dataset_info.get("series_count", 1),
        "filters": {
            "datasets": [source_name],
            "stores": stores,
            "items": items,
            "granularities": ["Daily", "Weekly", "Monthly"],
            "horizons": ["7 days", "14 days", "30 days", "60 days", "90 days", "4 weeks", "13 weeks", "26 weeks", "52 weeks", "6 months", "12 months", "18 months", "24 months"],
            "start_date": historical[0]["date"] if historical else None,
            "end_date": last_actual_date,
        },
        "historical": historical,
        "validation": validation,
        "backtest_predictions": persisted_backtests,
        "future": future,
        "series": validation + future,
        "last_actual_date": last_actual_date,
        "forecast_start_date": forecast_start_date,
        "forecast_end_date": forecast_end_date,
        "forecast_horizon": len(future),
        "backtest_duplicate_policy": "shortest_horizon",
        "confidence_level": 95 if any(interval_offsets.values()) else None,
        "seasonality": _seasonality_payload(
            historical_frame,
            frequency,
            dataset_info.get("target_display_name") or dataset_info.get("target_column") or "Target",
        ),
        "explanation": {"direction": "stable", "summary": "Forecast explanation is available in the Forecast Explorer.", "reasons": []},
    }


def _empirical_interval_offsets(records):
    """Return a 95% empirical residual interval from out-of-sample errors only."""
    import numpy as np

    residuals = [
        float(row["actual"]) - float(row["predicted"])
        for row in records
        if row.get("actual") is not None and row.get("predicted") is not None
    ]
    if len(residuals) < 5:
        return None
    low, high = np.quantile(residuals, [0.025, 0.975])
    return float(low), float(high)


def _apply_interval(prediction, offsets):
    low_offset, high_offset = offsets
    lower = min(float(prediction), float(prediction) + low_offset)
    upper = max(float(prediction), float(prediction) + high_offset)
    return {"lower": round(lower, 2), "upper": round(upper, 2)}


def _forecast_dates(prediction_frame):
    future = prediction_frame[prediction_frame["target"].isna()]
    return {
        "forecast_start_date": _timestamp_text(future["date"].min()) if not future.empty else None,
        "forecast_end_date": _timestamp_text(future["date"].max()) if not future.empty else None,
        "future_points": len(future),
    }


def _seasonality_payload(frame, frequency, target_display_name="Target"):
    import pandas as pd

    grouped = frame.copy()
    grouped["period"] = grouped["date"].dt.strftime("%A" if str(frequency).startswith("D") else "%B")
    means = grouped.groupby("period")["target"].mean().dropna().to_dict()
    if not means:
        return {"frequency": str(frequency), "strength": 0, "strongest_period": None, "weakest_period": None, "summary": "Seasonality is not yet strong enough to detect."}
    strongest, weakest = max(means, key=means.get), min(means, key=means.get)
    average = sum(means.values()) / len(means) or 1
    strength = round(min(100, max(0, (means[strongest] - means[weakest]) / average * 100)), 2)
    label = str(target_display_name or "Target")
    summary = f"{label} is strongest on {strongest} and weakest on {weakest}." if str(frequency).startswith("D") else f"{label} shows seasonal lift in {strongest}."
    return {"frequency": str(frequency), "strength": strength, "strongest_period": strongest, "weakest_period": weakest, "summary": summary}


def _rank_metrics(metrics):
    ranked = sorted(metrics, key=lambda row: row.get("mape") if row.get("mape") is not None else 999999)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _order_prediction_columns(prediction_frame, metrics):
    columns = ["date", "date_var", "target"] + [row["model"] for row in metrics if row["model"] in prediction_frame.columns]
    return prediction_frame[columns]


def _clean_predictions(predictions, expected_length):
    import numpy as np
    import pandas as pd

    values = pd.Series(predictions).astype(float).reset_index(drop=True)
    if len(values) != expected_length:
        raise ValueError(f"Expected {expected_length} predictions, got {len(values)}.")
    values = values.replace([np.inf, -np.inf], np.nan)
    return values


def _future_frame(df, frequency, history_length=None):
    import pandas as pd
    from pandas.tseries.frequencies import to_offset

    periods = _future_periods(frequency)
    if history_length:
        periods = min(periods, max(1, int(history_length) // 2))
    try:
        offset = to_offset(frequency)
    except ValueError:
        frequency = "D"
        offset = to_offset(frequency)
    start = pd.Timestamp(df["date"].iloc[-1]) + offset
    future = pd.DataFrame({"date": pd.date_range(start=start, periods=periods, freq=frequency)})
    from services.preprocessing_service import add_exogenous_features

    return add_exogenous_features(future)


def _infer_frequency(df):
    import pandas as pd

    inferred = pd.infer_freq(pd.to_datetime(df["date"]))
    if inferred:
        return inferred
    gaps = pd.to_datetime(df["date"]).diff().dt.days.dropna()
    if gaps.empty:
        return "D"
    median_gap = gaps.median()
    if median_gap >= 27:
        return "MS"
    if median_gap >= 6:
        first = pd.Timestamp(df["date"].iloc[0])
        return f"W-{first.day_name()[:3].upper()}"
    return "D"


def _future_periods(frequency):
    text = str(frequency).upper()
    if text.startswith("W"):
        return 52
    if "M" in text:
        return 13
    return 364


def _seasonal_period(frequency):
    text = str(frequency).upper()
    if text.startswith("H"):
        return 24
    if text.startswith("D"):
        return 7
    if text.startswith("W"):
        return 52
    if "Q" in text:
        return 4
    if "M" in text:
        return 12
    return None


def _holdout_size(length):
    if length < 10:
        return max(1, length // 3)
    return min(28, max(5, length // 5))


def _fold_validation_size(length):
    # A wider configurable fold preserves a useful backtest trace after weekly/monthly chart aggregation.
    configured = int(os.getenv("FORECAST_FOLD_VALIDATION_SIZE", "120"))
    return min(configured, max(1, length // 5))


def _rolling_folds(length, validation_size):
    min_train = max(30, validation_size * 2)
    if length < min_train + validation_size * 3:
        return []
    folds = []
    for fold_index in range(3):
        validation_start = length - validation_size * (3 - fold_index)
        validation_end = validation_start + validation_size
        folds.append((validation_start, validation_end))
    return folds


def _evaluation_mode(length, validation_size):
    return "Rolling-origin evaluation" if len(_rolling_folds(length, validation_size)) >= 3 else "Holdout fallback"


def _average_metrics(metrics):
    averaged = {}
    for key in ("mape", "accuracy", "mae", "rmse", "wape", "bias"):
        values = [metric[key] for metric in metrics if metric.get(key) is not None]
        averaged[key] = round(sum(values) / len(values), 2) if values else None
    return averaged


def _exog(df):
    """Compatibility helper for diagnostics; paired fitting uses fold-fitted encoding."""
    encoded, _ = _exog_pair(df, df)
    return encoded


def _exog_pair(train_df, output_df):
    import numpy as np
    import pandas as pd

    schema = _fit_source_exog_schema(train_df)
    if not schema and not any(column in train_df for column in CALENDAR_EXOG_COLUMNS):
        raise ValueError("Exogenous model requires mapped source or deterministic calendar features.")
    future_output = bool(len(output_df)) and output_df.get("target", pd.Series(dtype=float)).isna().all()
    train_rows = []
    output_rows = []
    for _index, row in train_df.iterrows():
        values = _encoded_source_exog(row, schema)
        values.update({column: float(row[column]) for column in CALENDAR_EXOG_COLUMNS if column in row})
        train_rows.append(values)
    for _index, row in output_df.iterrows():
        try:
            values = _encoded_source_exog(row, schema, require_values=future_output)
        except FutureExogenousUnavailable:
            raise
        values.update({column: float(row[column]) for column in CALENDAR_EXOG_COLUMNS if column in row})
        output_rows.append(values)
    train = pd.DataFrame(train_rows)
    output = pd.DataFrame(output_rows).reindex(columns=train.columns, fill_value=0.0)
    if not np.isfinite(train.to_numpy(dtype=float)).all() or not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ValueError("Exogenous matrix contains non-finite values after fold-fitted preprocessing.")
    return train.astype(float), output.astype(float)


def _available_exog(df):
    return _source_exog_columns(df) + [column for column in CALENDAR_EXOG_COLUMNS if column in df.columns]


def _source_exog_columns(df):
    return sorted(column for column in df.columns if str(column).startswith(SOURCE_EXOG_PREFIX))


def _fit_source_exog_schema(train_df):
    import pandas as pd

    schema = []
    max_categories = max(2, int(os.getenv("FORECAST_MAX_EXOG_CATEGORIES", "50")))
    for column in _source_exog_columns(train_df):
        values = train_df[column]
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            fill = float(numeric.median()) if numeric.notna().any() else 0.0
            schema.append({"column": column, "kind": "numeric", "fill": fill, "features": [column]})
            continue
        categories = sorted(values.dropna().astype(str).unique().tolist())
        if len(categories) > max_categories:
            raise ValueError(f"Exogenous feature {column} exceeds the configured categorical cardinality limit.")
        schema.append({
            "column": column, "kind": "categorical", "categories": categories,
            "features": [f"{column}__{index}" for index in range(len(categories))],
        })
    return schema


def _encoded_source_exog(row, schema, require_values=False):
    import pandas as pd

    values = {}
    for feature in schema:
        column = feature["column"]
        raw = row.get(column) if row is not None else None
        missing = raw is None or pd.isna(raw)
        if missing and require_values:
            raise FutureExogenousUnavailable(
                f"Future forecast is unavailable because mapped feature '{column}' has no future values. Historical backtesting remains valid."
            )
        if feature["kind"] == "numeric":
            parsed = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
            values[column] = feature["fill"] if pd.isna(parsed) else float(parsed)
        else:
            text = None if missing else str(raw)
            for index, category in enumerate(feature["categories"]):
                values[f"{column}__{index}"] = 1.0 if text == category else 0.0
    return values


def _var_endog(df):
    columns = _endogenous_columns(df)
    return df[columns].astype(float) if len(columns) >= 2 else df[["target"]].astype(float)


def _endogenous_columns(df):
    return sorted(column for column in df.columns if str(column).startswith("endogenous_"))


def _require_rows(df, minimum, model_name):
    if len(df) < minimum:
        raise ValueError(f"{model_name} requires at least {minimum} training rows.")


def _lag(history, steps):
    if not history:
        return 0
    return history[-steps] if len(history) >= steps else _rolling(history, min(len(history), steps))


def _rolling(history, window):
    if not history:
        return 0
    tail = history[-window:] if len(history) >= window else history
    return sum(tail) / len(tail)


def _rolling_median(history, window):
    import statistics

    if not history:
        return 0
    return float(statistics.median(history[-window:]))


def _rolling_std(history, window):
    import statistics

    if len(history) < 2:
        return 0
    tail = history[-window:]
    return float(statistics.pstdev(tail)) if len(tail) > 1 else 0


def _model_input_summary(model_id, fitter_name, training_df):
    source = _source_exog_columns(training_df)
    uses_exog = "_exog" in fitter_name
    exogenous_mode = "mapped_source_plus_calendar" if uses_exog and source else "deterministic_calendar" if uses_exog else "none"
    generated = list(CALENDAR_EXOG_COLUMNS) if uses_exog else []
    encoded = []
    if uses_exog and source:
        for feature in _fit_source_exog_schema(training_df):
            encoded.extend(feature["features"])
    if "xgboost" in fitter_name:
        generated += [
            "date_ordinal", "dayofweek", "month", "dayofyear", "lag_1", "lag_7", "lag_14", "lag_28",
            "rolling_7", "rolling_28", "rolling_median_7", "rolling_std_7", "trend_index",
        ]
    final_features = (encoded + generated) if uses_exog else (generated if "xgboost" in fitter_name else [])
    rows = max(0, len(training_df) - 1) if "xgboost" in fitter_name else len(training_df)
    return {
        "model_id": model_id,
        "model_type": fitter_name,
        "rows": len(training_df),
        "target_column": "target",
        "source_exogenous_features": source,
        "source_exogenous_feature_count": len(source),
        "generated_features": generated,
        "generated_feature_count": len(generated),
        "final_feature_names": final_features[:100],
        "final_feature_count": len(final_features),
        "matrix_shape": [rows, len(final_features)],
        "series_count": max(1, len(_endogenous_columns(training_df))),
        "exogenous_mode": exogenous_mode,
        "frequency": _infer_frequency(training_df),
        "validation_folds": len(_rolling_folds(len(training_df), _fold_validation_size(len(training_df)))),
        "future_feature_availability": {column: False for column in source},
    }


def _input_summary_text(summary):
    return (
        f"Input rows={summary['rows']}; target={summary['target_column']}; "
        f"source_exogenous={summary['source_exogenous_feature_count']}; "
        f"generated={summary['generated_feature_count']}; matrix={summary['matrix_shape']}."
    )


def _lstm_window():
    return int(os.getenv("FORECAST_LSTM_WINDOW", "30"))


def _lstm_epochs():
    return int(os.getenv("FORECAST_LSTM_EPOCHS", "25"))


def _lstm_batch_size():
    return int(os.getenv("FORECAST_LSTM_BATCH_SIZE", "32"))


def _none_or_float(value):
    try:
        import pandas as pd

        if pd.isna(value):
            return None
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _timestamp_text(value):
    import pandas as pd

    timestamp = pd.Timestamp(value)
    if timestamp.hour == timestamp.minute == timestamp.second == timestamp.microsecond == 0:
        return timestamp.date().isoformat()
    return timestamp.isoformat()


def _dataset_name(dataset_info):
    source = dataset_info.get("source_file") or dataset_info.get("cleaned_path") or "Current Dataset"
    return str(source).split("\\")[-1].split("/")[-1]


def _initial_status(start_time=None, dataset_id=None, job_id=None, artifact_id=None):
    now = start_time or datetime.now()
    return {
        "dataset_id": dataset_id or current_dataset_id(),
        "job_id": job_id,
        "artifact_id": artifact_id,
        "status": "running",
        "phase": "queued",
        "training_mode": TRAINING_MODE,
        "start_time": now.isoformat(timespec="seconds"),
        "end_time": None,
        "duration_seconds": 0,
        "duration_display": "0 sec",
        "current_step": "Queued",
        "current_model": "",
        "completed_models": [],
        "failed_models": [],
        "skipped_models": [],
        "total_models": len(MODEL_SPECS),
        "raw_rows": 0,
        "cleaned_rows": 0,
        "rows_used_for_training": 0,
        "evaluation_mode": "Rolling-origin evaluation",
        "last_data_date": None,
        "forecast_start_date": None,
        "forecast_end_date": None,
        "forecast_horizon": 0,
        "message": "Full ML Training started.",
        "last_heartbeat": now.isoformat(timespec="seconds"),
        "finished_at": None,
        "current_artifact": "",
        "artifact_progress": {"expected_artifacts": 0, "generated_artifacts": 0, "failed_artifacts": 0, "skipped_artifacts": 0, "terminal_artifacts": 0},
        "models_expected": len(MODEL_SPECS), "models_terminal": 0,
        "forecasts_expected": 0, "forecasts_generated": 0, "forecasts_failed": 0, "forecasts_skipped": 0, "forecast_budget_exceeded": 0,
        "artifacts_expected": 0, "artifacts_published": 0, "artifacts_failed": 0, "artifacts_skipped": 0,
        "forecast_ready": False, "usable_model_count": 0, "usable_forecast_count": 0, "forecast_manifest_id": None,
        "phase_started_at": now.isoformat(timespec="seconds"),
    }


def _idle_status():
    return {
        "dataset_id": current_dataset_id(),
        "job_id": None,
        "artifact_id": None,
        "status": "idle",
        "training_mode": TRAINING_MODE,
        "start_time": None,
        "end_time": None,
        "duration_seconds": 0,
        "duration_display": "0 sec",
        "current_step": "No training run started",
        "current_model": "",
        "completed_models": [],
        "failed_models": [],
        "skipped_models": [],
        "total_models": len(MODEL_SPECS),
        "raw_rows": 0,
        "cleaned_rows": 0,
        "rows_used_for_training": 0,
        "evaluation_mode": "Rolling-origin evaluation",
        "last_data_date": None,
        "forecast_start_date": None,
        "forecast_end_date": None,
        "forecast_horizon": 0,
        "message": "Upload a dataset, then start Full ML Training.",
    }


def _write_status(status):
    if status.get("dataset_id") and status.get("job_id") and not _job_is_current(status["dataset_id"], status["job_id"]):
        return False
    _set_memory_status(status)
    try:
        save_json("training_status.json", status)
        return True
    except Exception as exc:
        warning = f"Training status remains in memory because JSON persistence failed: {exc}"
        logging.getLogger(__name__).warning(warning)
        _set_memory_status(status)
        return False


def _set_memory_status(status):
    with _STATUS_LOCK:
        _IN_MEMORY_STATUSES[_status_namespace_key()] = copy.deepcopy(status)


def _write_log(entries, dataset_id=None, job_id=None):
    try:
        if dataset_id and job_id:
            _ensure_job_active(dataset_id, job_id)
        save_json("training_log.json", {"dataset_id": dataset_id or current_dataset_id(), "job_id": job_id, "entries": entries})
    except Exception as exc:
        logging.getLogger(__name__).warning("Training log persistence failed; training will continue: %s", exc)


def _refresh_runtime_kpi(duration_display, dataset_id=None, job_id=None):
    if dataset_id and job_id:
        _ensure_job_active(dataset_id, job_id)
    kpis = load_json("kpis.json", [])
    refreshed = []
    runtime_written = False
    for card in kpis:
        item = dict(card)
        if item.get("label") == "Last Training Runtime":
            item["value"] = duration_display
            runtime_written = True
        refreshed.append(item)
    if not runtime_written:
        refreshed.append({"label": "Last Training Runtime", "value": duration_display, "caption": "Most recent run", "dataset_id": dataset_id or current_dataset_id()})
    save_json("kpis.json", refreshed)


def _duration_display(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} hr {minutes} min {seconds} sec"
    if minutes:
        return f"{minutes} min {seconds} sec"
    return f"{seconds} sec"


def _friendly_error(exc):
    text = str(exc).strip()
    return text or exc.__class__.__name__
