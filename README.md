# ACI Forecast Intelligence

Premium Flask forecasting workspace for uploaded operational datasets. The app profiles source data, preserves mapped series dimensions, trains the full ML model portfolio, evaluates candidates with rolling-origin validation, and produces a future forecast beyond the final actual date.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Workflow

1. Open `/dataset` and upload a `.csv`, `.tsv`, `.txt`, or `.xlsx` file.
2. Confirm the detected timestamp, target, and optional series-dimension columns.
3. Apply mapping to clean the dataset and view quality and enrichment insights.
4. Start **Full ML Training** and monitor live model status and runtime.
5. Review the champion on `/dashboard` and inspect future forecasts on `/forecast-explorer`.

Mapped training data is written to `Data/Preprocessed/`. Preprocessing reports raw, valid, removed, aggregated, calendar-generated, and final training rows separately so synthetic timestamps are never presented as source observations. Active JSON state is isolated under `Data/runtime/` by default. With `CLEAR_RUNTIME_ON_START=true` (the default), uploaded datasets and generated runtime/preprocessed artifacts are removed when the server starts.

Dataset Preview reads only the immutable active raw upload. Its schema uses stable positional column IDs while displaying the original headers and values, including duplicate or Unicode headers. Applying mapping creates a separate processed preview and never replaces the raw table. Exact raw row counts are calculated for text uploads up to `RAW_PREVIEW_EXACT_COUNT_MAX_BYTES`; larger uploads return a bounded preview without scanning the whole file for display.

Prediction artifacts carry dataset, preprocessing artifact, training job, model, and canonical series-key ownership. Aggregate forecasts are never allocated to a selected series. For bounded panels the app can train owned per-series artifacts; panels above `FORECAST_MAX_PER_SERIES_MODELS` remain explicitly aggregate-only unless the limit is deliberately raised for available compute.

The model portfolio includes SARIMAX, Auto ARIMA, XGBoost GridSearchCV, four Exponential Smoothing variants, VAR, and TensorFlow LSTM, with exogenous variants where the data structure supports them. Failed models are recorded honestly while the remaining candidates continue.

Validation uses rolling-origin folds where the dataset is large enough. Each completed model is then refit on the complete cleaned history, and future dates begin strictly after the maximum uploaded Date.

## Render deployment

Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 600
```

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
