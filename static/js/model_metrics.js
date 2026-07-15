const modelMetricsPage = document.querySelector('[data-page="model-metrics"]');

if (modelMetricsPage) initializeModelInsights();

async function initializeModelInsights() {
  try {
    const [metricsResponse, statusResponse] = await Promise.all([fetch("/api/models"), fetch("/api/training-status")]);
    const metrics = (await metricsResponse.json()).filter(validMetric).sort((a, b) => Number(a.mape) - Number(b.mape));
    const status = await statusResponse.json();
    renderMetricCards(metrics, status);
    renderStatusStrip(metrics, status);
    renderRankingChart(metrics);
    renderAccuracyErrorChart(metrics);
  } catch (_error) {
    ["modelRankingChart", "accuracyErrorChart"].forEach(id => showChartEmpty(id, "Model visualization data is unavailable."));
  }
}

function renderMetricCards(metrics, status) {
  if (!metrics.length) { ["modelRankingChart", "accuracyErrorChart"].forEach(id => showChartEmpty(id, "No successful finite model metrics are available yet.")); return; }
  const bestMape = metrics[0];
  const bestAccuracy = metrics.reduce((best, row) => Number(row.accuracy || -Infinity) > Number(best.accuracy || -Infinity) ? row : best);
  const lowestRmse = metrics.reduce((best, row) => Number(row.rmse) < Number(best.rmse) ? row : best);
  const lowestBias = metrics.reduce((best, row) => Math.abs(Number(row.bias || 0)) < Math.abs(Number(best.bias || 0)) ? row : best);
  setText("bestMape", `${Number(bestMape.mape).toFixed(2)}%`);
  setText("bestAccuracy", Number.isFinite(Number(bestAccuracy.accuracy)) ? `${Number(bestAccuracy.accuracy).toFixed(2)}%` : "--");
  setText("lowestRmse", compactNumber(lowestRmse.rmse));
  setText("lowestBias", `${Number(lowestBias.bias || 0).toFixed(2)}%`);
  setText("modelsCompleted", status.completed_models?.length ?? metrics.length);
  setText("modelsFailed", status.failed_models?.length ?? 0);
}

function renderStatusStrip(metrics, status) {
  setText("statusCompleted", status.completed_models?.length ?? metrics.length);
  setText("statusFailed", status.failed_models?.length ?? 0);
  setText("statusEvaluation", status.evaluation_mode || metrics[0]?.evaluation_mode || "--");
  setText("statusRuntime", status.duration_display || "--");
}

function renderRankingChart(metrics) {
  if (!metrics.length) return showChartEmpty("modelRankingChart", "No successful finite MAPE values are available yet.");
  const labels = metrics.map(row => row.model_label || humanModel(row.model));
  new Chart(document.getElementById("modelRankingChart"), {
    type: "bar",
    data: { labels, datasets: [{ label: "MAPE %", data: metrics.map(row => Number(row.mape)), backgroundColor: metrics.map((_row, index) => index === 0 ? "#12a875" : "rgba(109, 93, 252, .68)"), borderRadius: 6, borderSkipped: false, maxBarThickness: 24 }] },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: "x",
      scales: {
        x: { title: { display: true, text: "Model" }, grid: { display: false }, ticks: { color: "#8795a7", font: { size: 9 }, maxRotation: 45, minRotation: 0, autoSkip: true, autoSkipPadding: 12 } },
        y: { title: { display: true, text: "MAPE %" }, beginAtZero: true, grid: { color: "rgba(19,34,56,.07)" }, ticks: { color: "#8795a7", font: { size: 9 } } }
      },
      plugins: { legend: { display: false }, tooltip: { callbacks: { title: items => labels[items[0].dataIndex], label: context => { const row = metrics[context.dataIndex]; return [`MAPE: ${Number(row.mape).toFixed(2)}%`, `Rank: ${context.dataIndex + 1}`, `Champion: ${context.dataIndex === 0 ? "Yes" : "No"}`]; } } } }
    }
  });
}

function renderAccuracyErrorChart(metrics) {
  if (!metrics.length) return showChartEmpty("accuracyErrorChart", "No valid completed model metrics are available yet.");
  const maxMae = Math.max(...metrics.map(row => Number(row.mae) || 0), 1);
  const bubbles = metrics.map((row, index) => ({ x: Number(row.mape), y: Number(row.rmse), r: 6 + 12 * Math.sqrt((Number(row.mae) || 0) / maxMae), model: row.model_label || humanModel(row.model), champion: index === 0 }));
  new Chart(document.getElementById("accuracyErrorChart"), { type: "bubble", data: { datasets: [{ label: "Completed models", data: bubbles, backgroundColor: bubbles.map(point => point.champion ? "rgba(18, 168, 117, .78)" : "rgba(22, 184, 212, .55)"), borderColor: bubbles.map(point => point.champion ? "#087956" : "#168aa0"), borderWidth: 1.5 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: context => `${context.raw.model}: MAPE ${context.raw.x}% / RMSE ${compactNumber(context.raw.y)}` } } }, scales: { x: { title: { display: true, text: "MAPE %" }, beginAtZero: true }, y: { title: { display: true, text: "RMSE" }, beginAtZero: true } } } });
}

function validMetric(row) {
  const status = String(row.status || "completed").toLowerCase();
  return !["failed", "skipped", "unavailable", "error"].includes(status) && [row.mape, row.mae, row.rmse].every(value => Number.isFinite(Number(value))) && Number(row.mape) >= 0 && Number(row.mape) <= 1000 && Number(row.mae) >= 0 && Number(row.mae) <= 1e12 && Number(row.rmse) >= 0 && Number(row.rmse) <= 1e12;
}

function showChartEmpty(id, message) { const canvas = document.getElementById(id); if (canvas && !canvas.previousElementSibling?.classList.contains("chart-empty")) canvas.insertAdjacentHTML("beforebegin", `<div class="chart-empty">${message}</div>`); }
function setText(id, value) { const element = document.getElementById(id); if (element) element.textContent = value; }
function compactNumber(value) { return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 2 }).format(value || 0); }
function humanModel(value) { return String(value || "").replace("_Predictions", "").replaceAll("_", " "); }
