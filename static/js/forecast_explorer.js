const forecastExplorerRoot = document.getElementById("forecastExplorerPage");

if (forecastExplorerRoot) {
  const form = document.getElementById("forecastFilters");
  let forecastChart;
  let forecastDriftChart;
  let requestSequence = 0;
  let filterRequestSequence = 0;
  let activeRequest;
  let filterRequest;
  let latestForecastPayload;
  const colors = { ink: "#132238", cyan: "#16b8d4", violet: "#6d5dfc", grid: "rgba(19,34,56,.07)" };
  const marker = forecastMarkers();
  const modelSelectionState = {
    selected_model_id: form.elements.model?.value || "",
    selection_source: "global_champion",
    selection_scope: "aggregate",
    series_key_hash: "",
    dataset_id: form.elements.dataset_id?.value || "",
    artifact_id: form.elements.artifact_id?.value || "",
    job_id: form.elements.job_id?.value || "",
    manual_aggregate_model_id: null,
    compatibility_message: ""
  };

  form.addEventListener("change", async event => {
    if (event.target.name === "model") {
      const scope = currentForecastScope();
      modelSelectionState.selected_model_id = event.target.value;
      modelSelectionState.selection_source = "manual_user_selection";
      modelSelectionState.selection_scope = scope;
      if (scope === "aggregate") modelSelectionState.manual_aggregate_model_id = event.target.value;
      loadPage();
      return;
    }
    const dimension = event.target.closest("[data-forecast-dimension]");
    if (dimension) {
      clearDependentDimensions(Number(dimension.dataset.dimensionIndex));
      if (!await refreshDependentFilters()) return;
    } else if (event.target.name === "granularity") {
      if (!await refreshDependentFilters()) return;
    }
    loadPage();
  });
  form.querySelector(".filter-button").addEventListener("click", async () => {
    if (await refreshDependentFilters()) loadPage();
  });
  document.getElementById("forecastChartMode")?.addEventListener("change", applyChartMode);
  refreshDependentFilters().then(ready => { if (ready) loadPage(); });

  async function refreshDependentFilters() {
    if (filterRequest) filterRequest.abort();
    filterRequest = new AbortController();
    const sequence = ++filterRequestSequence;
    const state = canonicalFilters();
    const query = new URLSearchParams({
      dataset_id: state.dataset_id,
      artifact_id: state.artifact_id,
      job_id: state.job_id,
      dimensions: JSON.stringify(state.dimension_filters),
      request_id: String(sequence)
    });
    try {
      const response = await fetch(`/api/forecast-filters?${query}`, { signal: filterRequest.signal });
      const payload = await response.json();
      if (sequence !== filterRequestSequence || (payload.request_id && payload.request_id !== String(sequence))) return false;
      if (!response.ok || payload.ok === false) throw new Error(payload.error || "Filter values are unavailable.");
      (payload.dimensions || []).forEach(dimension => {
        const select = [...form.querySelectorAll("[data-forecast-dimension]")].find(node => node.name === dimension.id);
        if (!select) return;
        const current = select.value;
        select.replaceChildren(new Option(`All ${dimension.label}`, ""), ...(dimension.values || []).map(value => new Option(value, value)));
        select.value = (dimension.values || []).includes(current) ? current : "";
      });
      refreshModelOptions(payload.models || [], payload, currentForecastScope());
      refreshHorizonOptions(payload.horizons_by_granularity || {}, state.granularity);
      return true;
    } catch (error) {
      if (error.name === "AbortError") return false;
      clearChart(forecastChart);
      clearChart(forecastDriftChart);
      setText("forecastExplorerNotice", error.message || "Filter values are unavailable.");
      return false;
    }
  }

  function clearDependentDimensions(index) {
    form.querySelectorAll("[data-forecast-dimension]").forEach(select => {
      if (Number(select.dataset.dimensionIndex) > index) select.value = "";
    });
  }

  function refreshModelOptions(models, metadata, scope) {
    const select = form.elements.model;
    if (!select || !models.length) return;
    const previous = modelSelectionState.selected_model_id || select.value;
    const options = models.map(model => {
      const suffix = model.supports_selected_series ? "" : ` (${model.scope === "aggregate_only" ? "aggregate only" : "unavailable"})`;
      const option = new Option(`${model.display_name}${suffix}`, model.model_id);
      option.disabled = !model.supports_selected_series;
      return option;
    });
    select.replaceChildren(...options);
    const resolution = resolveForecastModelSelection(models, metadata, modelSelectionState, scope);
    select.value = resolution.selected_model_id || "";
    if (resolution.selection_source === "automatic_series_fallback" && previous && previous !== resolution.selected_model_id) {
      const selected = models.find(model => model.model_id === resolution.selected_model_id);
      modelSelectionState.compatibility_message = `The previous model is unavailable for this series. ${selected?.display_name || "A compatible model"} was selected automatically.`;
    } else {
      modelSelectionState.compatibility_message = "";
    }
    Object.assign(modelSelectionState, resolution, {
      series_key_hash: metadata.series_key_hash || "",
      dataset_id: metadata.dataset_id || modelSelectionState.dataset_id,
      artifact_id: metadata.artifact_id || modelSelectionState.artifact_id,
      job_id: metadata.job_id || modelSelectionState.job_id
    });
  }

  function refreshHorizonOptions(optionsByGranularity, granularity) {
    const select = form.elements.horizon;
    const options = optionsByGranularity[granularity] || [];
    if (!select || !options.length) return;
    const currentValue = Number(select.selectedOptions[0]?.dataset.horizonValue);
    const currentUnit = select.selectedOptions[0]?.dataset.horizonUnit;
    select.replaceChildren(...options.map(item => {
      const option = new Option(item.label, item.label);
      option.dataset.horizonValue = item.value;
      option.dataset.horizonUnit = item.unit;
      return option;
    }));
    const compatible = options.find(item => item.value === currentValue && item.unit === currentUnit);
    select.value = compatible?.label || options[0].label;
  }

  async function loadPage() {
    const state = canonicalFilters();
    const sequence = ++requestSequence;
    if (activeRequest) activeRequest.abort();
    activeRequest = new AbortController();
    const { dimension_filters: dimensionFilters, ...queryState } = state;
    const query = new URLSearchParams({ ...queryState, dimensions: JSON.stringify(dimensionFilters), request_id: String(sequence) });
    try {
      const [forecastResponse, driftResponse] = await Promise.all([
        fetch(`/api/forecast?${query}`, { signal: activeRequest.signal }),
        fetch(`/api/drift?${query}`, { signal: activeRequest.signal })
      ]);
      const forecast = await forecastResponse.json();
      const drift = await driftResponse.json();
      if (sequence !== requestSequence) return;
      if (!forecastResponse.ok || forecast.ok === false) throw new Error(forecast.error || "Forecast data is unavailable for this selection.");
      if (forecast.request_id !== String(sequence)) return;
      if (forecast.resolved_model_id !== state.model_id || forecast.selection_scope !== state.selection_scope) return;
      if (forecast.dataset_id !== state.dataset_id || forecast.artifact_id !== state.artifact_id || forecast.job_id !== state.job_id || (modelSelectionState.series_key_hash && forecast.series_key_hash !== modelSelectionState.series_key_hash)) return;
      if (drift.request_id && drift.request_id !== String(sequence)) return;
      const driftResult = canonicalDriftResult(drift);
      renderForecast(forecast, driftResult);
      renderDrift(driftResult, state);
    } catch (error) {
      if (error.name === "AbortError" || sequence !== requestSequence) return;
      clearChart(forecastChart);
      clearChart(forecastDriftChart);
      setText("forecastDriftExplanation", error.message || "The selected forecast is unavailable.");
    }
  }

  function canonicalFilters() {
    const values = Object.fromEntries(new FormData(form).entries());
    const horizonOption = form.elements.horizon?.selectedOptions[0];
    const dimensionFilters = {};
    form.querySelectorAll("select").forEach(select => {
      if (!["dataset", "granularity", "horizon", "model"].includes(select.name)) dimensionFilters[select.name] = select.value || null;
    });
    return {
      dataset_id: values.dataset_id || "",
      artifact_id: values.artifact_id || "",
      job_id: values.job_id || "",
      historical_start: values.startDate || "",
      historical_end: values.endDate || "",
      dimension_filters: dimensionFilters,
      granularity: values.granularity || "Daily",
      future_horizon: values.horizon || "",
      horizon_value: horizonOption?.dataset.horizonValue || "",
      horizon_unit: horizonOption?.dataset.horizonUnit || "",
      model_id: values.model || "",
      selection_source: modelSelectionState.selection_source,
      selection_scope: currentForecastScope()
    };
  }

  function currentForecastScope() {
    return [...form.querySelectorAll("[data-forecast-dimension]")].some(select => Boolean(select.value)) ? "series" : "aggregate";
  }

  function canonicalDriftResult(payload) {
    if (payload?.drift_result) return payload.drift_result;
    const rows = payload?.drift_series || [];
    const targetRows = rows.filter(row => row.drift_type === "target").sort((a, b) => String(a.timestamp).localeCompare(String(b.timestamp)));
    const strongest = targetRows.reduce((best, row) => !best || finite(row.score) > finite(best.score) ? row : best, null);
    const firstDetected = targetRows.find(row => row.severity !== "stable");
    const past = payload?.past || payload?.series_drift || {};
    const future = payload?.future || {};
    const forecastChange = finite(future.forecast_change ?? future.forecast_change_pct);
    const futureAvailable = future.available ?? forecastChange !== null;
    const futureRisk = future.risk || (future.status === "Drift Expected" ? "high" : future.status === "Watch" ? "moderate" : futureAvailable ? "low" : "unavailable");
    return {
      ok: payload?.ok,
      available: payload?.available,
      drift_series: rows,
      summary: payload?.summary || {},
      target_drift: payload?.target_drift || {
        available: Boolean(strongest),
        reason: strongest ? null : "insufficient history",
        status: strongest?.severity || "insufficient_data",
        severity: finite(strongest?.score),
        detected: Boolean(firstDetected),
        first_detected: firstDetected?.timestamp || null,
        mean_change: finite(strongest?.mean_change ?? past.mean_change_pct),
        volatility_change: finite(strongest?.volatility_change ?? past.volatility_change_pct),
        type: "target",
        confidence: past.confidence || payload?.summary?.confidence || "insufficient"
      },
      future_drift: payload?.future_drift || {
        available: futureAvailable,
        reason: futureAvailable ? null : (future.reason || "no forecast"),
        status: future.status || "insufficient_data",
        risk: futureRisk,
        forecast_change: forecastChange,
        forecast_volatility: finite(future.forecast_volatility),
        confidence: future.confidence || "insufficient"
      }
    };
  }

  function renderForecast(data, drift) {
    latestForecastPayload = data;
    const actualPoints = toPoints(data.historical_actual || [], "actual");
    const historicalRows = validIntervalRows(data.historical_prediction || []);
    const futureRows = validIntervalRows(data.future_prediction || []);
    const historicalPredictionPoints = toPoints(data.historical_prediction || [], "historical");
    const futurePredictionPoints = toPoints(data.future_prediction || [], "future");
    const historicalLower = toBoundPoints(historicalRows, "lower", "historical");
    const historicalUpper = toBoundPoints(historicalRows, "upper", "historical");
    const futureLower = toBoundPoints(futureRows, "lower", "future");
    const futureUpper = toBoundPoints(futureRows, "upper", "future");
    const labels = uniqueSorted([
      ...actualPoints.map(point => point.x),
      ...historicalPredictionPoints.map(point => point.x),
      ...futurePredictionPoints.map(point => point.x)
    ]);
    const lastActualTimestamp = data.metadata?.last_actual_timestamp || actualPoints.at(-1)?.x || null;
    const firstBacktestTimestamp = data.metadata?.first_historical_prediction_timestamp || historicalPredictionPoints[0]?.x || null;
    const datasets = [];

    if (historicalLower.length && historicalUpper.length) {
      datasets.push(
        intervalHelper("_Historical confidence lower", historicalLower, 30),
        intervalBand("Historical confidence band", historicalUpper, "rgba(109,93,252,.11)", 29)
      );
    }
    if (futureLower.length && futureUpper.length) {
      datasets.push(
        intervalHelper("_Future confidence lower", futureLower, 28),
        intervalBand("Future confidence band", futureUpper, "rgba(22,184,212,.13)", 27)
      );
    }
    datasets.push({ label: "Actual", _seriesType: "actual", data: actualPoints, parsing: false, borderColor: colors.ink, backgroundColor: colors.ink, borderWidth: 2.2, pointRadius: 0, pointHoverRadius: 4, tension: .18, order: 12 });
    if (historicalPredictionPoints.length) {
      datasets.push({ label: "Historical Prediction", _seriesType: "historical", data: historicalPredictionPoints, parsing: false, borderColor: colors.violet, backgroundColor: colors.violet, borderDash: [6, 4], borderWidth: 2.6, pointRadius: 1.2, pointHoverRadius: 5, tension: .18, order: 10 });
    }
    if (futurePredictionPoints.length) {
      datasets.push({ label: "Future Prediction", _seriesType: "future", data: futurePredictionPoints, parsing: false, borderColor: colors.cyan, backgroundColor: colors.cyan, borderWidth: 2.8, pointRadius: 1.3, pointHoverRadius: 5, tension: .18, order: 9 });
    }

    const majorErrors = largestErrorPoints(historicalPredictionPoints, 3);
    if (majorErrors.length) {
      datasets.push({ label: "Major backtest errors", _seriesType: "error", data: majorErrors, parsing: false, showLine: false, pointStyle: "crossRot", pointRadius: 5, pointHoverRadius: 7, pointBorderWidth: 1.5, borderColor: "#d28a25", backgroundColor: "#d28a25", order: 4 });
    }
    const driftMarkers = driftMarkerPoints(drift, actualPoints);
    if (driftMarkers.length) {
      datasets.push({ label: "Drift markers", _seriesType: "drift", data: driftMarkers, parsing: false, showLine: false, pointStyle: "triangle", pointRadius: 5, pointHoverRadius: 7, borderColor: "#d45165", backgroundColor: "rgba(212,81,101,.75)", order: 3 });
    }

    if (forecastChart) forecastChart.destroy();
    forecastChart = new Chart(document.getElementById("forecastExplorerChart"), {
      type: "line",
      plugins: [marker],
      data: { labels, datasets },
      options: explorerChartOptions({ forecast: futurePredictionPoints.length ? lastActualTimestamp : null, backtest: firstBacktestTimestamp, targetUnit: data.metadata?.target_unit, model: data.model_context?.display_name || humanModel(data.model_id) })
    });
    forecastChart.$ranges = { actualPoints, historicalPredictionPoints, futurePredictionPoints };
    restoreVisibility(forecastChart);
    applyChartMode();

    const backtest = data.backtest_summary || {};
    const forecast = data.forecast_summary || {};
    const uncertainty = data.uncertainty || {};
    const modelName = data.model_context?.display_name || humanModel(data.model_id || data.filters?.model_id || "--");
    const driftStatus = drift?.summary?.overall_severity || "Insufficient data";
    const targetDrift = drift?.target_drift || {};
    const futureDrift = drift?.future_drift || {};
    const intervalText = uncertainty.available ? (data.metadata?.confidence_level ? `${data.metadata.confidence_level}% interval` : "Model interval") : "Interval unavailable";
    const trendText = `${titleCase(forecast.trend || "unavailable")}${forecast.percentage_change == null ? "" : ` ${signedPercent(forecast.percentage_change)}`}`;
    const horizonText = String(data.filters?.horizon || data.effective_filters?.future_horizon || data.metadata?.future_prediction_count || "--");

    setText("explorerModel", modelName);
    setText("lastActualDate", lastActualTimestamp || "--");
    setText("forecastStartDate", futurePredictionPoints[0]?.x || "--");
    setText("forecastEndDate", futurePredictionPoints.at(-1)?.x || "--");
    setText("chartMetricModel", modelName);
    setText("chartMetricHorizon", horizonText);
    setText("chartMetricAccuracy", backtest.reliable && backtest.wape != null ? `WAPE ${formatPercent(backtest.wape)}` : "Limited backtest");
    setText("chartMetricTrend", trendText);
    setText("chartMetricConfidence", intervalText);
    setText("chartMetricDrift", titleCase(driftStatus));
    setText("pastDriftStatus", targetDrift.available ? titleCase(targetDrift.status) : "Unavailable");
    setText("pastDriftDetail", targetDrift.available
      ? `Mean change ${formatPercent(targetDrift.mean_change)} · Volatility change ${formatPercent(targetDrift.volatility_change)}`
      : titleCase(targetDrift.reason || "insufficient history"));
    setText("futureDriftStatus", futureDrift.available ? titleCase(futureDrift.risk || futureDrift.status) : "Unavailable");
    setText("futureDriftDetail", futureDrift.available
      ? `Forecast change ${formatPercent(futureDrift.forecast_change)}`
      : titleCase(futureDrift.reason || "no forecast"));
    setText("confidenceBadge", uncertainty.available ? (data.metadata?.confidence_level ? `${data.metadata.confidence_level}% empirical band` : "Model interval") : "Interval unavailable");
    setText("summaryMape", backtest.wape == null ? "--" : formatPercent(backtest.wape));
    setText("summaryAccuracy", data.metadata?.historical_prediction_displayed_count ?? 0);
    setText("summaryVariance", `${backtest.coverage_percentage ?? 0}%`);
    setText("summaryPoints", data.metadata?.future_prediction_count ?? 0);
    setText("seasonalitySummary", data.seasonality?.summary || "--");
    setText("seasonalityDetail", `Strength ${data.seasonality?.strength ?? 0}%`);
    setText("explanationDirection", data.explanation?.direction || "--");
    setText("explanationSummary", data.plain_language_explanation || data.explanation?.summary || "--");
    const notices = [...(data.warnings || [])];
    if (modelSelectionState.compatibility_message) notices.unshift(modelSelectionState.compatibility_message);
    if (!historicalPredictionPoints.length && !notices.some(item => item.includes("Historical prediction"))) notices.unshift("Historical prediction is unavailable for the selected model.");
    setText("forecastExplorerNotice", [data.plain_language_explanation, drift?.summary?.plain_language_explanation, ...notices].filter(Boolean).join(" "));
    modelSelectionState.compatibility_message = "";
  }

  function renderDrift(drift, state) {
    if (!drift.ok || !drift.drift_series?.length) { clearChart(forecastDriftChart); setText("forecastDriftExplanation", drift.summary?.plain_language_explanation || "Drift is unavailable for this filter combination."); return; }
    const labels = [...new Set(drift.drift_series.map(row => row.timestamp))].sort();
    const types = [...new Set(drift.drift_series.map(row => row.drift_type))];
    const datasets = types.map(type => {
      const rows = drift.drift_series.filter(row => row.drift_type === type);
      return { label: `${type} drift`, data: labels.map(label => rows.find(row => row.timestamp === label)?.score ?? null), borderColor: type === "residual" ? colors.violet : colors.cyan, backgroundColor: "transparent", tension: .2, pointRadius: labels.map(label => rows.find(row => row.timestamp === label)?.change_point ? 5 : 2), _rows: rows };
    });
    [0.25, 0.5, 0.75].forEach((threshold, index) => datasets.push({ label: ["Moderate threshold", "High threshold", "Critical threshold"][index], data: labels.map(() => threshold), borderColor: ["rgba(22,184,212,.35)", "rgba(109,93,252,.35)", "rgba(220,90,90,.35)"][index], borderDash: [5, 4], pointRadius: 0, borderWidth: 1 }));
    if (forecastDriftChart) forecastDriftChart.destroy();
    forecastDriftChart = new Chart(document.getElementById("forecastDriftChart"), { type: "line", data: { labels, datasets }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, scales: { x: { grid: { display: false }, ticks: { color: "#8795a7", font: { size: 9 }, maxTicksLimit: 10 } }, y: { min: 0, max: 1, title: { display: true, text: "Normalized drift severity (0–1)" }, grid: { color: colors.grid }, ticks: { color: "#8795a7", font: { size: 9 } } } }, plugins: { legend: { display: true, labels: { usePointStyle: true, boxWidth: 7, color: "#68768a", font: { size: 10 } } }, tooltip: { callbacks: { label: context => { const row = context.dataset._rows?.find(item => item.timestamp === context.label); return row ? [`${row.drift_type}: ${row.score}`, `Severity: ${row.severity}`, `Effect: ${row.effect_size}`, `Variable: ${row.affected_variable}`, `Action: ${row.recommended_action}`] : `${context.dataset.label}: ${context.raw}`; } } } } } });
    setText("forecastDriftExplanation", drift.summary?.plain_language_explanation || "Drift is calculated from the active historical window.");
  }

  function explorerChartOptions(markers) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      forecastMarkerTimestamp: markers.forecast,
      backtestMarkerTimestamp: markers.backtest,
      targetUnit: markers.targetUnit,
      selectedModelName: markers.model,
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: {
          onClick(event, item, legend) {
            Chart.defaults.plugins.legend.onClick(event, item, legend);
            saveVisibility(legend.chart);
          },
          labels: {
            usePointStyle: true, boxWidth: 7, color: "#68768a", font: { size: 10 },
            filter: item => !String(item.text || "").startsWith("_")
          }
        },
        tooltip: {
          filter: context => !context.dataset._intervalHelper && !context.dataset._intervalBand,
          callbacks: {
            title: contexts => contexts[0]?.raw?.x || contexts[0]?.label || "",
            label: forecastTooltip
          }
        }
      },
      scales: {
        x: { type: "category", grid: { display: false }, ticks: { color: "#8795a7", font: { size: 9 }, maxTicksLimit: 12 } },
        y: { beginAtZero: true, grid: { color: colors.grid }, ticks: { color: "#8795a7", font: { size: 9 }, maxTicksLimit: 8, callback: value => compactNumber(value) } }
      }
    };
  }

  function forecastTooltip(context) {
    const point = context.raw || {};
    const unit = context.chart.options.targetUnit;
    const value = readableNumber(point.y, unit);
    if (context.dataset._seriesType === "actual") return `Actual: ${value}`;
    if (context.dataset._seriesType === "historical" || context.dataset._seriesType === "error") {
      const actual = finite(point.actual);
      const absoluteError = actual == null ? null : Math.abs(actual - point.y);
      const percentageError = actual == null || actual === 0 ? null : absoluteError / Math.abs(actual) * 100;
      return [
        `${context.dataset._seriesType === "error" ? "Largest error prediction" : "Predicted"}: ${value}`,
        `Actual: ${readableNumber(actual, unit)}`,
        `Absolute error: ${readableNumber(absoluteError, unit)}`,
        `Percentage error: ${percentageError == null ? "Unavailable" : formatPercent(percentageError)}`,
        `Forecast origin: ${point.forecast_origin || "Unavailable"}`,
        `Horizon step: ${point.horizon_step ?? "--"}`,
        `Interval: ${intervalText(point, unit)}`,
        `Validation fold: ${point.fold ?? "--"}`
      ];
    }
    if (context.dataset._seriesType === "future") {
      return [
        `Predicted: ${value}`,
        `Interval: ${intervalText(point, unit)}`,
        `Horizon step: ${point.horizon_step ?? "--"}`,
        `Model: ${context.chart.options.selectedModelName || point.model_id || "--"}`,
        `Confidence: ${latestForecastPayload?.metadata?.confidence_level ? `${latestForecastPayload.metadata.confidence_level}% empirical interval` : (point.lower != null && point.upper != null ? "Level unavailable" : "Unavailable")}`
      ];
    }
    if (context.dataset._seriesType === "drift") return [`Drift: ${point.severity || "detected"}`, point.explanation || "Review the Drift Analysis panel."];
    return `${context.dataset.label}: ${value}`;
  }

  function applyChartMode() {
    if (!forecastChart?.$ranges) return;
    const mode = document.getElementById("forecastChartMode")?.value || "full";
    const { actualPoints, historicalPredictionPoints, futurePredictionPoints } = forecastChart.$ranges;
    let min;
    let max;
    if (mode === "backtest" && historicalPredictionPoints.length) {
      min = historicalPredictionPoints[0].x;
      max = historicalPredictionPoints.at(-1).x;
    } else if (mode === "forecast" && futurePredictionPoints.length) {
      const contextCount = Math.min(24, actualPoints.length);
      min = actualPoints.at(-contextCount)?.x || actualPoints[0]?.x;
      max = futurePredictionPoints.at(-1)?.x;
    }
    if (min) forecastChart.options.scales.x.min = min; else delete forecastChart.options.scales.x.min;
    if (max) forecastChart.options.scales.x.max = max; else delete forecastChart.options.scales.x.max;
    forecastChart.update("none");
  }

  function saveVisibility(chart) {
    try {
      const state = Object.fromEntries(chart.data.datasets.filter(dataset => !dataset._intervalHelper).map((dataset, index) => [dataset.label, !chart.isDatasetVisible(index)]));
      sessionStorage.setItem("forecastExplorerVisibility", JSON.stringify(state));
    } catch (_error) { /* Browser storage is optional. */ }
  }

  function restoreVisibility(chart) {
    try {
      const state = JSON.parse(sessionStorage.getItem("forecastExplorerVisibility") || "{}");
      chart.data.datasets.forEach((dataset, index) => {
        if (Object.prototype.hasOwnProperty.call(state, dataset.label)) chart.getDatasetMeta(index).hidden = Boolean(state[dataset.label]);
      });
      chart.update("none");
    } catch (_error) { /* Browser storage is optional. */ }
  }
  function clearChart(chart) { if (chart) { chart.data.labels = []; chart.data.datasets.forEach(dataset => { dataset.data = []; }); chart.update(); } }
}

function resolveForecastModelSelection(models, metadata, state, scope) {
  const compatible = modelId => models.some(model => model.model_id === modelId && model.supports_selected_series);
  if (scope === "aggregate") {
    const manualAggregate = state.manual_aggregate_model_id;
    if (manualAggregate && compatible(manualAggregate)) {
      return { selected_model_id: manualAggregate, selection_source: "manual_user_selection", selection_scope: scope };
    }
    if (state.selection_scope === "aggregate" && state.selection_source === "manual_user_selection" && compatible(state.selected_model_id)) {
      return { selected_model_id: state.selected_model_id, selection_source: "manual_user_selection", selection_scope: scope };
    }
    const champion = metadata.global_champion?.model_id;
    const selected = compatible(champion) ? champion : models.find(model => model.supports_selected_series)?.model_id || "";
    return { selected_model_id: selected, selection_source: "global_champion", selection_scope: scope };
  }
  if (compatible(state.selected_model_id)) {
    return { selected_model_id: state.selected_model_id, selection_source: state.selection_source, selection_scope: scope };
  }
  const recommended = metadata.recommended_model_id;
  const selected = compatible(recommended) ? recommended : models.find(model => model.supports_selected_series)?.model_id || "";
  return { selected_model_id: selected, selection_source: "automatic_series_fallback", selection_scope: scope };
}

function forecastMarkers() { return { id: "forecastExplorerMarkers", afterDraw(chart) { drawMarker(chart, chart.options.backtestMarkerTimestamp, "Backtest period begins", "#6d5dfc", 26); drawMarker(chart, chart.options.forecastMarkerTimestamp, "Forecast begins", "#16b8d4", 12); } }; }
function drawMarker(chart, timestamp, label, color, topOffset) { if (!timestamp) return; const index = chart.data.labels.indexOf(timestamp); if (index < 0) return; const x = chart.scales.x.getPixelForValue(index); const { top, bottom, right } = chart.chartArea; const ctx = chart.ctx; ctx.save(); ctx.strokeStyle = color; ctx.globalAlpha = .8; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke(); ctx.fillStyle = color; ctx.font = "700 10px DM Sans"; ctx.fillText(label, Math.min(x + 7, right - ctx.measureText(label).width), top + topOffset); ctx.restore(); }
function finite(value) { const number = Number(value); return Number.isFinite(number) && Math.abs(number) < 1e100 ? number : null; }
function toPoints(rows, type) { return rows.map(row => ({ x: row.timestamp, y: finite(row.value), actual: type === "actual" ? finite(row.value) : finite(row.actual), lower: finite(row.lower), upper: finite(row.upper), model_id: row.model_id, dataset_id: row.dataset_id, artifact_id: row.artifact_id, job_id: row.job_id, forecast_origin: row.forecast_origin, fold: row.fold, horizon_step: row.horizon_step, validation_method: row.validation_method })).filter(point => point.x && point.y !== null).sort((a, b) => String(a.x).localeCompare(String(b.x))); }
function toBoundPoints(rows, bound, type) { return rows.map(row => ({ x: row.timestamp, y: finite(row[bound]), actual: finite(row.actual), lower: finite(row.lower), upper: finite(row.upper), _intervalType: type })).filter(point => point.x && point.y !== null).sort((a, b) => String(a.x).localeCompare(String(b.x))); }
function validIntervalRows(rows) { return rows.filter(row => { const value = finite(row.value); const lower = finite(row.lower); const upper = finite(row.upper); return value !== null && lower !== null && upper !== null && lower <= value && value <= upper; }); }
function intervalHelper(label, data, order) { return { label, _intervalHelper: true, data, parsing: false, borderColor: "transparent", backgroundColor: "transparent", pointRadius: 0, fill: false, order }; }
function intervalBand(label, data, backgroundColor, order) { return { label, _intervalBand: true, data, parsing: false, borderColor: "transparent", backgroundColor, pointRadius: 0, fill: "-1", order }; }
function largestErrorPoints(points, limit) { return points.filter(point => point.actual !== null).map(point => ({ ...point, absolute_error: Math.abs(point.actual - point.y) })).sort((a, b) => b.absolute_error - a.absolute_error).slice(0, limit); }
function driftMarkerPoints(drift, actualPoints) { const actual = new Map(actualPoints.map(point => [point.x, point.y])); return (drift?.drift_series || []).filter(row => row.change_point && actual.has(row.timestamp)).map(row => ({ x: row.timestamp, y: actual.get(row.timestamp), severity: row.severity, explanation: row.explanation })); }
function uniqueSorted(values) { return [...new Set(values.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b))); }
function setText(id, value) { const element = document.getElementById(id); if (element) element.textContent = value; }
function humanModel(value) { return String(value || "").replace("_Predictions", "").replaceAll("_", " "); }
function intervalText(point, unit) { return point.lower == null || point.upper == null ? "Unavailable" : `${readableNumber(point.lower, unit)} – ${readableNumber(point.upper, unit)}`; }
function readableNumber(value, unit) { const number = finite(value); if (number === null) return "Unavailable"; const formatted = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(number); return unit ? `${formatted} ${unit}` : formatted; }
function compactNumber(value) { const number = finite(value); if (number === null) return ""; return new Intl.NumberFormat(undefined, { notation: Math.abs(number) >= 1e6 ? "compact" : "standard", maximumFractionDigits: 1 }).format(number); }
function formatPercent(value) { const number = finite(value); return number === null ? "--" : `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(number)}%`; }
function signedPercent(value) { const number = finite(value); return number === null ? "" : `${number >= 0 ? "+" : ""}${formatPercent(number)}`; }
function titleCase(value) { return String(value || "").replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase()); }
