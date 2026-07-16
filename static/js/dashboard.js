const dashboardPage = document.querySelector('[data-page="dashboard"]');

if (dashboardPage) {
  const form = document.getElementById("dashboardFilters");
  let actualChart, mapeChart, confidenceChart;
  let requestSequence = 0;
  let activeRequest;
  const colors = { ink: "#132238", cyan: "#16b8d4", violet: "#6d5dfc", green: "#12a875", grid: "rgba(19,34,56,.07)" };
  const markerPlugin = forecastMarker();

  form.addEventListener("change", loadForecast);
  form.querySelector(".filter-button").addEventListener("click", loadForecast);
  loadForecast();
  loadMape();

  async function loadForecast() {
    const requestId = ++requestSequence;
    if (activeRequest) activeRequest.abort();
    activeRequest = new AbortController();
    const filterState = Object.fromEntries(new FormData(form).entries());
    const dimensions = {};
    form.querySelectorAll("[data-dashboard-dimension]").forEach(select => {
      if (select.value) dimensions[select.name] = select.value;
      delete filterState[select.name];
    });
    filterState.dimensions = JSON.stringify(dimensions);
    filterState.request_id = String(requestId);
    try {
      const response = await fetch(`/api/forecast?${new URLSearchParams(filterState)}`, { signal: activeRequest.signal });
      const data = await response.json();
      if (requestId !== requestSequence) return;
      if (!response.ok || data.ok === false) throw new Error(data.error || "Unable to load this forecast selection.");
      const clean = value => Number.isFinite(Number(value)) && Math.abs(Number(value)) < 1e100 ? Number(value) : null;
      ["actual", "fitted", "future_forecast", "lower", "upper"].forEach(key => { data[key] = (data[key] || []).map(clean); });
      renderTimeline(data);
      renderConfidence(data);
      document.getElementById("forecastDateBadge").textContent = data.forecast_start_date ? `Starts ${data.forecast_start_date}` : "Future only";
    } catch (error) {
      if (error.name === "AbortError" || requestId !== requestSequence) return;
      [actualChart, confidenceChart].forEach(chart => { if (chart) { chart.data.labels = []; chart.data.datasets.forEach(dataset => { dataset.data = []; }); chart.update(); } });
    }
  }

  async function loadMape() {
    const metrics = (await (await fetch("/api/models")).json()).filter((x) => Number.isFinite(Number(x.mape)) && Number.isFinite(Number(x.mae)) && Number.isFinite(Number(x.rmse)) && Number(x.mape) >= 0 && Number(x.mape) <= 1000 && Number(x.mae) <= 1e12 && Number(x.rmse) <= 1e12);
    if (mapeChart) mapeChart.destroy();
    mapeChart = new Chart(document.getElementById("mapeChart"), { type: "bar", data: { labels: metrics.map((x) => x.model_label || x.model), datasets: [{ data: metrics.map((x) => x.mape), backgroundColor: metrics.map((_, i) => i ? "rgba(109,93,252,.64)" : colors.green), borderRadius: 5 }] }, options: chartOptions(true, false) });
  }

  function renderTimeline(data) {
    if (actualChart) actualChart.destroy();
    actualChart = new Chart(document.getElementById("actualPredictedChart"), { type: "line", plugins: [markerPlugin], data: { labels: data.labels, datasets: [
      { label: "Actual manpower required", data: data.actual, borderColor: colors.ink, backgroundColor: "rgba(19,34,56,.06)", borderWidth: 2, pointRadius: 0, tension: .2 },
      { label: "Historical backtest", data: data.fitted, borderColor: colors.violet, borderDash: [5,4], borderWidth: 2, pointRadius: 0, tension: .2 },
      { label: "Future manpower required", data: data.future_forecast, borderColor: colors.cyan, backgroundColor: "rgba(22,184,212,.12)", borderWidth: 2.5, pointRadius: 0, tension: .2 }
    ] }, options: { ...chartOptions(false, true), forecastStartIndex: data.forecast_start_index } });
  }

  function renderConfidence(data) {
    if (confidenceChart) confidenceChart.destroy();
    confidenceChart = new Chart(document.getElementById("confidenceChart"), { type: "line", plugins: [markerPlugin], data: { labels: data.labels, datasets: [
      { label: "Prediction interval lower", data: data.lower, borderColor: "transparent", pointRadius: 0, fill: false },
      { label: "Prediction interval upper", data: data.upper, borderColor: "transparent", backgroundColor: "rgba(22,184,212,.14)", pointRadius: 0, fill: "-1" },
      { label: "Future Prediction", data: data.future_forecast, borderColor: colors.cyan, borderWidth: 2.5, pointRadius: 0, tension: .2 }
    ] }, options: { ...chartOptions(false, true), forecastStartIndex: data.forecast_start_index } });
  }

  function chartOptions(horizontal, legend) {
    return { responsive: true, maintainAspectRatio: false, indexAxis: horizontal ? "y" : "x", interaction: { mode: "index", intersect: false }, plugins: { legend: { display: legend, labels: { usePointStyle: true, boxWidth: 7, color: "#68768a", font: { size: 10 } } } }, scales: { x: { grid: { display: false }, ticks: { color: "#8795a7", font: { size: 9 }, maxTicksLimit: 10 } }, y: { beginAtZero: true, grid: { color: colors.grid }, ticks: { color: "#8795a7", font: { size: 9 }, maxTicksLimit: 7 } } } };
  }
}

function forecastMarker() {
  return { id: "forecastMarker", afterDraw(chart) { const index = chart.options.forecastStartIndex; if (index == null || index < 0) return; const x = chart.scales.x.getPixelForValue(index); const { top, bottom } = chart.chartArea; const ctx = chart.ctx; ctx.save(); ctx.strokeStyle = "rgba(22,184,212,.72)"; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke(); ctx.fillStyle = "#187f9f"; ctx.font = "700 10px DM Sans"; ctx.fillText("Forecast begins", Math.min(x + 7, chart.chartArea.right - 82), top + 12); ctx.restore(); } };
}
