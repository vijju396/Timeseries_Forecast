from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    ranking = (ROOT / "static/js/model_metrics.js").read_text(encoding="utf-8")
    template = (ROOT / "templates/model_metrics.html").read_text(encoding="utf-8")
    explorer_template = (ROOT / "templates/forecast_explorer.html").read_text(encoding="utf-8")
    explorer_script = (ROOT / "static/js/forecast_explorer.js").read_text(encoding="utf-8")
    command_template = (ROOT / "templates/dashboard.html").read_text(encoding="utf-8")
    command_script = (ROOT / "static/js/dashboard.js").read_text(encoding="utf-8")
    dataset_template = (ROOT / "templates/dataset.html").read_text(encoding="utf-8")
    dataset_script = (ROOT / "static/js/dataset.js").read_text(encoding="utf-8")
    training_template = (ROOT / "templates/training_pipeline.html").read_text(encoding="utf-8")
    dashboard = (ROOT / "static/js/dashboard.js").read_text(encoding="utf-8")
    explorer = (ROOT / "static/js/forecast_explorer.js").read_text(encoding="utf-8")
    assert 'indexAxis: "x"' in ranking
    assert 'text: "Model"' in ranking and 'text: "MAPE %"' in ranking
    assert 'sort((a, b) => Number(a.mape) - Number(b.mape))' in ranking
    assert 'index === 0 ? "#12a875"' in ranking
    assert 'status !== "failed"' not in ranking or 'validMetric' in ranking
    assert 'id="forecastDriftChart"' in explorer_template and 'Drift Analysis' in explorer_template
    assert 'forecastDriftChart' in explorer_script and 'fetch(`/api/drift?' in explorer_script
    assert 'AbortController' in explorer_script and 'requestSequence' in explorer_script and 'drift.request_id' in explorer_script
    assert 'forecastDriftChart' not in command_template and 'forecastDriftChart' not in command_script
    assert 'driftChart' not in template and 'api/drift' not in (ROOT / "static/js/model_metrics.js").read_text(encoding="utf-8")
    assert 'label: "Actual"' in dashboard and 'label: "Historical Prediction"' in dashboard and 'label: "Future Prediction"' in dashboard
    assert 'label: "Actual"' in explorer and 'label: "Historical Prediction"' in explorer and 'label: "Future Prediction"' in explorer
    assert 'toPoints(data.historical_actual || [], "actual")' in explorer
    assert 'toPoints(data.historical_prediction || [], "historical")' in explorer
    assert 'toPoints(data.future_prediction || [], "future")' in explorer
    assert "forecastMarkerTimestamp" in explorer and "metadata?.last_actual_timestamp" in explorer
    assert "backtestMarkerTimestamp" in explorer and "first_historical_prediction_timestamp" in explorer
    assert 'borderDash: [6, 4]' in explorer and "borderWidth: 2.6" in explorer
    assert "Historical prediction is unavailable for the selected model." in explorer
    assert '"_Future confidence lower"' in explorer and '"_Historical confidence lower"' in explorer
    assert '"Future confidence band"' in explorer and '"Historical confidence band"' in explorer
    assert 'fill: "-1"' in explorer and '_intervalHelper' in explorer and '_intervalBand' in explorer
    assert 'id="forecastChartMode"' in explorer_template and "Backtest focus" in explorer_template and "Forecast focus" in explorer_template
    assert "function applyChartMode()" in explorer and 'mode === "backtest"' in explorer and 'mode === "forecast"' in explorer
    assert "Absolute error:" in explorer and "Percentage error:" in explorer and "Forecast origin:" in explorer
    assert "Major backtest errors" in explorer and "largestErrorPoints" in explorer
    assert "forecastExplorerVisibility" in explorer
    assert "Actual, backtest & forecast" in explorer_template and "Actual, fitted & forecast" not in explorer_template
    assert 'id="preprocessingExplanation"' in dataset_template
    assert dataset_template.index('id="preprocessingExplanation"') < dataset_template.index('id="dataStudioFilters"') < dataset_template.index('id="trendChart"')
    assert "/api/data-studio-analytics" in dataset_script and "canonicalDataStudioFilterState" in dataset_script
    assert "AbortController" in dataset_script and "dataStudioAnalyticsSequence" in dataset_script
    assert "preprocessing_metrics" in dataset_script and "preprocessing_explanation" in dataset_script
    assert "Cleaned Rows" not in dataset_script and "Duplicate Rows" not in dataset_script
    assert "Primary series dimension" in dataset_script and "Secondary series dimension" in dataset_script
    assert "Training rows" in training_template and "Cleaned rows" not in training_template
    print("chart contract passed")


if __name__ == "__main__":
    main()
