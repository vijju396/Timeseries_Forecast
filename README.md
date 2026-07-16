# ACI Healthcare Manpower Forecasting

Flask forecasting workspace configured for `manpower_healthcare.xlsx`. The app profiles all 24 source columns, preserves department/facility identity, prepares daily staffing requirements, trains the full model portfolio, evaluates candidates with rolling-origin validation, and produces a future manpower forecast beyond the final actual date.

## Healthcare workbook mapping

The included workbook contains 396 daily records from 2025-01-01 through 2026-01-31. Rows may arrive out of order; preprocessing sorts them by date and mapped series before training.

- Timestamp: `date`
- Forecast target: `manpower_required` (staff members)
- Series dimensions: `department`, then `facility_id`
- Suggested operational drivers: `day_of_week`, `is_weekend`, `is_holiday`, `holiday_name`, `season`, `patient_census`, `occupancy_rate`, `patient_acuity_index`, `admissions`, and `discharges`
- Explicit leakage exclusions: `total_staff_hours`, `scheduled_staff`, `absent_staff`, `available_staff`, and `overtime_hours`
- Context/provenance retained in the immutable raw preview: facility/location fields, `data_origin`, and `source_url`

The Data Studio shows the disposition of every source column after mapping. Source drivers support historical evaluation for exogenous models. Future exogenous forecasts are only published when future driver values are genuinely available; standard time-series models continue to publish future manpower forecasts without inventing patient or staffing inputs.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`.

## Workflow

1. Open `/dataset` and upload `manpower_healthcare.xlsx` (or another `.csv`, `.tsv`, `.txt`, or `.xlsx` file).
2. Confirm the detected timestamp, manpower target, department/facility dimensions, and operational drivers.
3. Apply mapping to clean the dataset and view quality and enrichment insights.
4. Start **Full ML Training** and monitor live model status and runtime.
5. Review the champion on `/dashboard` and inspect future forecasts on `/forecast-explorer`.

Mapped training data is written to `Data/Preprocessed/`. Preprocessing reports raw, valid, removed, aggregated, calendar-generated, and final training rows separately so synthetic timestamps are never presented as source observations. Active JSON state is isolated under `Data/runtime/` by default. With `CLEAR_RUNTIME_ON_START=true` (the default), uploaded datasets and generated runtime/preprocessed artifacts are removed when the server starts.

Dataset Preview reads only the immutable active raw upload. Its schema uses stable positional column IDs while displaying the original headers and values, including duplicate or Unicode headers. Applying mapping creates a separate processed preview and never replaces the raw table. Exact raw row counts are calculated for text uploads up to `RAW_PREVIEW_EXACT_COUNT_MAX_BYTES`; larger uploads return a bounded preview without scanning the whole file for display.

Prediction artifacts carry dataset, preprocessing artifact, training job, model, and canonical series-key ownership. Aggregate forecasts are never allocated to a selected series. For bounded panels the app can train owned per-series artifacts; panels above `FORECAST_MAX_PER_SERIES_MODELS` remain explicitly aggregate-only unless the limit is deliberately raised for available compute.

The model portfolio includes SARIMAX, Auto ARIMA, XGBoost GridSearchCV, four Exponential Smoothing variants, VAR, and TensorFlow LSTM, with exogenous variants where the data structure supports them. Failed models are recorded honestly while the remaining candidates continue.

Validation uses rolling-origin folds where the dataset is large enough. Each completed model is then refit on the complete cleaned history, and future dates begin strictly after the maximum uploaded Date.

## Render deployment

The repository includes a Docker-based `render.yaml`. In Render:

1. Choose **New > Blueprint**.
2. Connect `vijju396/Timeseries_Forecast` and select the `healthcare-forecasting` branch.
3. Confirm the Blueprint and create the `timeseries-forecast` service.
4. Wait for `/health` to pass, then open the generated `onrender.com` URL.

The Docker image fixes Python at 3.12 and installs CPU-only TensorFlow and XGBoost wheels. The Blueprint starts on Render's free instance type to avoid unexpected charges. The application UI can deploy there, but the complete TensorFlow/XGBoost/Auto-ARIMA portfolio can exceed its 512 MB memory limit. Upgrade the web service to at least the Standard instance type for full-model training.

Environment variables:

```text
FLASK_DEBUG=0
CLEAR_RUNTIME_ON_START=true
FORECAST_DATA_DIR=Data/runtime
FORECAST_AGGREGATION=
FORECAST_TARGET_IMPUTATION=auto
FORECAST_IMPUTATION_MAX_GAP=3
FORECAST_MAX_NUMERIC_FAILURE_RATE=0.25
FORECAST_MAX_PER_SERIES_MODELS=12
FORECAST_XGB_N_JOBS=1
RAW_PREVIEW_EXACT_COUNT_MAX_BYTES=10485760
```

Render's filesystem is ephemeral unless a persistent disk is configured. Use one worker so the background training thread and runtime JSON state remain consistent. Full ML training includes TensorFlow and requires a memory-sufficient Render instance; a paid instance may be necessary for larger datasets.
