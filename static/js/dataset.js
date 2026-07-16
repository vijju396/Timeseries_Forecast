const datasetPage = document.querySelector('[data-page="dataset"]');
const trainingPage = document.querySelector('[data-page="training-pipeline"]');
const trainingPanel = document.getElementById("training-panel") || document.getElementById("trainingStatus");
const charts = {};
let latestDataset = null;
let trainingPoll = null;
let mappingDirty = false;
let activeTrainingJobId = null;
let activeTrainingDatasetId = null;
let uploadSequence = 0;
let activeUploadRequest = null;
let dataStudioAnalyticsRequest = null;
let dataStudioAnalyticsSequence = 0;
const terminalTrainingStatuses = new Set(["completed", "completed_with_warnings", "failed", "cancelled", "budget_exceeded"]);

if (datasetPage) {
  const uploadForm = document.getElementById("datasetUploadForm");
  const fileInput = document.getElementById("datasetFile");
  const dropZone = document.getElementById("dropZone");
  const status = document.getElementById("datasetStatus");
  const trainButton = document.getElementById("trainButton");

  fileInput.addEventListener("change", () => renderFileMeta(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  }));
  dropZone.addEventListener("drop", (event) => {
    if (!event.dataTransfer.files.length) return;
    fileInput.files = event.dataTransfer.files;
    renderFileMeta(fileInput.files[0]);
  });

  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const sequence = ++uploadSequence;
    activeUploadRequest?.abort();
    activeUploadRequest = new AbortController();
    resetDataStudioForNewDataset();
    setNotice("Profiling source data and detecting forecast fields...", "info-state");
    try {
      const response = await fetch("/dataset/upload", { method: "POST", body: new FormData(uploadForm), signal: activeUploadRequest.signal });
      const data = await response.json();
      if (sequence !== uploadSequence) return;
      if (!response.ok || data.ok === false) throw new Error(data.error || "Upload failed.");
      latestDataset = data.dataset;
      renderDataset(data.dataset);
      await loadRawPreview(data.dataset);
      updateUploadedRows(data.dataset.raw_preview?.total_raw_rows);
      setNotice(data.dataset.message || "Dataset uploaded successfully.", data.dataset.requires_mapping ? "info-state" : "success-state");
      if (data.dataset.adapted) await loadEnrichment();
    } catch (error) {
      if (error.name === "AbortError" || sequence !== uploadSequence) return;
      resetEnrichmentState("Upload a dataset to generate enrichment insights.");
      setNotice(error.message, "error-state");
    }
  });
  trainButton.addEventListener("click", startTraining);
}

document.addEventListener("DOMContentLoaded", () => {
  initDataStudioState();
  if (datasetPage) initWorkflowNavigation();
});

async function initDataStudioState() {
  try {
    const requests = [fetch("/api/enrichment"), fetch("/api/training-status")];
    if (datasetPage) requests.push(fetch("/api/current-dataset"));
    const responses = await Promise.all(requests);
    const enrichment = await responses[0].json();
    const trainingStatus = await responses[1].json();

    if (datasetPage) {
      const currentPayload = await responses[2].json();
      const currentDataset = currentPayload.available ? currentPayload.dataset : null;
      if (currentDataset) {
        latestDataset = currentDataset;
        renderDataset(currentDataset);
        await loadRawPreview(currentDataset);
        restoreUploadedMeta(currentDataset);
        setNotice(currentDataset.message || "Saved dataset workspace restored.", currentDataset.adapted ? "success-state" : "info-state");
      }
      if (!currentDataset) resetDataStudioForNewDataset();
      renderEnrichment(enrichment);
    }

    renderTrainingStatus(trainingStatus);
    loadTrainingLog();
    if (trainingStatus.status === "running") beginTrainingPoll();
  } catch (_error) {
    if (datasetPage) setNotice("Saved workspace state could not be restored. Uploads remain available for a new run.", "error-state");
  }
}

function initWorkflowNavigation() {
  const steps = [...document.querySelectorAll(".workflow-step")];
  steps.forEach((step) => step.addEventListener("click", () => {
    if (!step.disabled) document.getElementById(step.dataset.target)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }));
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    steps.forEach((step) => step.classList.toggle("active", step.dataset.target === visible.target.id));
  }, { rootMargin: "-20% 0px -65% 0px", threshold: [0.05, 0.25, 0.5] });
  document.querySelectorAll(".studio-section").forEach((section) => observer.observe(section));
}

function setWorkflowAvailability({ mapping = false, enrichment = false, training = false } = {}) {
  const availability = { "mapping-section": mapping, "enrichment-section": enrichment, "training-section": training };
  const order = ["upload-section", "mapping-section", "enrichment-section", "training-section"];
  const highest = Math.max(...order.map((id, index) => id === "upload-section" || availability[id] ? index : 0));
  document.querySelectorAll(".workflow-step").forEach((step) => {
    const enabled = step.dataset.target === "upload-section" || availability[step.dataset.target];
    step.disabled = !enabled;
    step.classList.toggle("completed", enabled && order.indexOf(step.dataset.target) < highest);
    if (enabled) step.title = `Go to ${step.textContent.trim()}`;
  });
}

function resetDataStudioForNewDataset() {
  latestDataset = null;
  mappingDirty = false;
  activeTrainingJobId = null;
  activeTrainingDatasetId = null;
  resetEnrichmentState("Upload a dataset to generate enrichment insights.");
  resetTrainingPanel();
  const mappingPanel = document.getElementById("mappingPanel");
  if (mappingPanel) {
    mappingPanel.className = "panel-empty";
    mappingPanel.innerHTML = '<i data-lucide="waypoints"></i><p>Column controls appear after upload.</p>';
  }
  const confidence = document.getElementById("mappingConfidence");
  if (confidence) {
    confidence.textContent = "Awaiting data";
    confidence.className = "confidence-chip";
  }
  const trainButton = document.getElementById("trainButton");
  if (trainButton) trainButton.disabled = true;
  setWorkflowAvailability();
  lucide.createIcons();
}

function resetEnrichmentState(message, { preservePreview = false } = {}) {
  dataStudioAnalyticsRequest?.abort();
  dataStudioAnalyticsRequest = null;
  Object.keys(charts).forEach((key) => {
    charts[key]?.destroy();
    delete charts[key];
  });
  const empty = document.getElementById("enrichment-empty-state");
  const content = document.getElementById("enrichment-content");
  const cards = document.getElementById("enrichment-cards");
  const preview = document.getElementById("previewTable");
  const previewSurface = document.getElementById("previewSurface");
  const orbit = document.getElementById("qualityOrbit");
  if (empty) {
    empty.classList.remove("is-hidden");
    const copy = empty.querySelector("p");
    if (copy) copy.textContent = message || "Upload a dataset to generate enrichment insights.";
  }
  content?.classList.add("is-hidden");
  orbit?.classList.add("is-hidden");
  if (!preservePreview) previewSurface?.classList.add("is-hidden");
  if (cards) cards.innerHTML = "";
  if (preview && !preservePreview) preview.innerHTML = "";
  const score = document.getElementById("qualityScore");
  if (score) score.textContent = "--";
  const explanation = document.getElementById("preprocessingExplanation");
  if (explanation) explanation.textContent = "";
  const filters = document.getElementById("dataStudioFilters");
  if (filters) { filters.replaceChildren(); filters.classList.add("is-hidden"); }
  const analyticsNotice = document.getElementById("dataStudioAnalyticsNotice");
  if (analyticsNotice) { analyticsNotice.textContent = ""; analyticsNotice.classList.add("is-hidden"); }
  const lineageSurface = document.getElementById("columnLineageSurface");
  const lineageTable = document.getElementById("columnLineageTable");
  lineageSurface?.classList.add("is-hidden");
  if (lineageTable) lineageTable.replaceChildren();
}

function resetTrainingPanel() {
  if (trainingPanel) trainingPanel.innerHTML = '<div class="training-empty-state">Upload and map a dataset to begin Full ML Training.</div>';
  const log = document.getElementById("trainingLogPreview");
  if (log) log.innerHTML = "";
}

function renderFileMeta(file) {
  const meta = document.getElementById("uploadMeta");
  if (!meta || !file) return;
  meta.innerHTML = `<i data-lucide="file-check-2"></i><span>${escapeHtml(file.name)} / ${formatBytes(file.size)}</span>`;
  lucide.createIcons();
}

function updateUploadedRows(rows) {
  const label = document.querySelector("#uploadMeta span");
  if (label && rows != null && !label.textContent.includes("rows")) label.textContent += ` / ${numberFormat(rows)} rows`;
}

function restoreUploadedMeta(dataset) {
  const meta = document.getElementById("uploadMeta");
  if (!meta) return;
  const rows = dataset.raw_preview?.total_raw_rows;
  meta.innerHTML = `<i data-lucide="database"></i><span>${escapeHtml(dataset.name || "Saved dataset")}${rows != null ? ` / ${numberFormat(rows)} rows` : ""}</span>`;
  lucide.createIcons();
}

function renderDataset(dataset) {
  const panel = document.getElementById("mappingPanel");
  const columns = dataset.raw_preview?.columns || [];
  const mapping = dataset.mapping || dataset.suggested_mapping || {};
  const confidence = document.getElementById("mappingConfidence");
  confidence.textContent = dataset.requires_mapping ? "Review required" : "Auto-detected";
  confidence.className = `confidence-chip ${dataset.requires_mapping ? "warning" : "ready"}`;
  panel.className = "";
  panel.innerHTML = renderMapping(dataset, columns, mapping);
  mappingDirty = !dataset.adapted;
  document.getElementById("mappingForm")?.addEventListener("submit", saveCurrentMapping);
  document.getElementById("mappingForm")?.addEventListener("change", () => { mappingDirty = true; });
  const button = document.getElementById("trainButton");
  if (button) button.disabled = !dataset.adapted;
  setWorkflowAvailability({ mapping: true, enrichment: Boolean(dataset.adapted), training: Boolean(dataset.adapted) });
  lucide.createIcons();
}

function renderMapping(dataset, columns, mapping) {
  const columnLabel = (column) => `${column.display_name || "(unnamed column)"}${column.duplicate_source_name ? ` [column ${column.position + 1}]` : ""}`;
  const options = (selected, optional = false) => `${optional ? '<option value="">None</option>' : ""}${columns.map((column) => `<option value="${escapeHtml(column.column_id)}"${column.column_id === selected ? " selected" : ""}>${escapeHtml(columnLabel(column))}</option>`).join("")}`;
  const timestampId = mapping.timestamp_column_id || "";
  const targetId = mapping.target_column_id || "";
  const dimensions = mapping.dimension_column_ids || [];
  const exogenous = new Set(mapping.exogenous_column_ids || []);
  const reserved = new Set([timestampId, targetId, ...dimensions].filter(Boolean));
  const sourceName = (columnId) => columns.find(column => column.column_id === columnId)?.display_name || "Select a column";
  const driverOptions = columns
    .filter(column => !reserved.has(column.column_id))
    .map(column => `<label class="driver-option"><input type="checkbox" name="exogenous_column_ids" value="${escapeHtml(column.column_id)}"${exogenous.has(column.column_id) ? " checked" : ""}><span>${escapeHtml(columnLabel(column))}<small>${escapeHtml(column.physical_type || "unknown")}</small></span></label>`)
    .join("");
  return `
    <div class="detected-note"><span>Date signal<strong>${escapeHtml(sourceName(timestampId))}</strong></span><span>Staffing target<strong>${escapeHtml(sourceName(targetId))}</strong></span><span>Source columns<strong>${columns.length} profiled</strong></span><span>Suggested drivers<strong>${exogenous.size} selected</strong></span></div>
    <form class="mapping-form" id="mappingForm">
      <input type="hidden" name="dataset_id" value="${escapeHtml(dataset.id || "")}">
      <input type="hidden" name="csv_file" value="${escapeHtml(mapping.csv_file || dataset.source_file || "")}">
      <label>Date column<select name="timestamp_column_id" required>${options(timestampId)}</select></label>
      <label>Target column<select name="target_column_id" required>${options(targetId)}</select></label>
      <label>Primary series dimension <small>optional</small><select name="primary_dimension_column_id">${options(dimensions[0] || "", true)}</select></label>
      <label>Secondary series dimension <small>optional</small><select name="secondary_dimension_column_id">${options(dimensions[1] || "", true)}</select></label>
      <fieldset class="driver-mapping"><legend>Operational drivers <small>optional; used for historical exogenous-model evaluation</small></legend><div class="driver-grid">${driverOptions || '<span class="muted-copy">No additional columns are available.</span>'}</div></fieldset>
      <button class="primary-button" type="submit"><i data-lucide="wand-sparkles"></i>Apply mapping & enrich</button>
    </form>`;
}

async function saveCurrentMapping(event) {
  if (event) event.preventDefault();
  const form = document.getElementById("mappingForm");
  if (!form) return latestDataset;
  resetEnrichmentState("No enrichment profile yet.", { preservePreview: true });
  resetTrainingPanel();
  setNotice("Cleaning, filling time gaps, and building enrichment insights...", "info-state");
  try {
    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());
    payload.exogenous_column_ids = formData.getAll("exogenous_column_ids");
    const data = await postJson("/dataset/map", payload);
    latestDataset = { ...latestDataset, ...data.dataset, adapted: data.adapted };
    document.getElementById("trainButton").disabled = false;
    setNotice(`Training input ready: ${numberFormat(data.adapted.rows_used)} reconciled training rows at ${data.adapted.frequency} frequency.`, "success-state");
    await loadEnrichment();
    mappingDirty = false;
    return latestDataset;
  } catch (error) {
    resetEnrichmentState("No enrichment profile yet.", { preservePreview: true });
    setNotice(error.message, "error-state");
    throw error;
  }
}

async function loadRawPreview(dataset = latestDataset) {
  if (!dataset?.id || !dataset?.raw_artifact_id) throw new Error("Raw preview ownership metadata is unavailable.");
  const query = new URLSearchParams({ dataset_id: dataset.id, limit: "10" });
  const response = await fetch(`/api/dataset-preview?${query}`);
  const preview = await response.json();
  if (!response.ok || preview.ok === false) {
    clearRawPreview();
    throw new Error(preview.error || "Raw dataset preview is unavailable.");
  }
  if (preview.dataset_id !== latestDataset?.id || preview.raw_artifact_id !== latestDataset?.raw_artifact_id || preview.source_file_hash !== latestDataset?.source_file_hash || preview.preview_type !== "raw") {
    clearRawPreview();
    throw new Error("A stale dataset preview response was rejected.");
  }
  latestDataset.raw_preview = preview;
  renderPreview(preview);
}

function renderPreview(preview) {
  const target = document.getElementById("previewTable");
  const columns = preview?.columns || [];
  const rows = preview?.rows || [];
  if (!target) return;
  if (!columns.length) {
    clearRawPreview();
    return;
  }
  target.innerHTML = `<table><thead><tr>${columns.map((column) => `<th>${escapeHtml(column.display_name || "")}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${columns.map((column) => `<td>${escapeHtml(row[column.column_id] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  const badge = document.getElementById("previewBadge");
  if (badge) {
    badge.textContent = `First ${preview.preview_row_count} rows${preview.warnings?.length ? " · schema warning" : ""}`;
    badge.title = (preview.warnings || []).join(" ");
  }
  document.getElementById("previewSurface")?.classList.remove("is-hidden");
}

function clearRawPreview() {
  const target = document.getElementById("previewTable");
  if (target) target.innerHTML = "";
  document.getElementById("previewSurface")?.classList.add("is-hidden");
}

async function loadEnrichment() {
  const response = await fetch("/api/enrichment");
  renderEnrichment(await response.json());
}

function renderEnrichment(data) {
  const empty = document.getElementById("enrichment-empty-state");
  const content = document.getElementById("enrichment-content");
  if (!empty || !content) return;
  if (!data || data.available === false || !data.summary || (latestDataset && data.dataset_id !== latestDataset.id)) {
    resetEnrichmentState(data?.message || "Upload a dataset to generate enrichment insights.", { preservePreview: Boolean(latestDataset) });
    setWorkflowAvailability({ mapping: Boolean(latestDataset), enrichment: false, training: Boolean(latestDataset?.adapted) });
    return;
  }
  empty.classList.add("is-hidden");
  content.classList.remove("is-hidden");
  document.getElementById("qualityOrbit")?.classList.remove("is-hidden");
  renderQuality(data);
  renderEnrichmentCharts(data);
  loadDataStudioAnalytics();
  setWorkflowAvailability({ mapping: Boolean(latestDataset), enrichment: true, training: Boolean(latestDataset?.adapted) });
  const button = document.getElementById("trainButton");
  if (button) button.disabled = !latestDataset?.adapted;
}

function renderQuality(data) {
  const score = data.quality?.score ?? 0;
  const orbit = document.querySelector(".quality-orbit");
  orbit.style.setProperty("--quality", `${score}%`);
  document.getElementById("qualityScore").textContent = score;
  const cards = data.preprocessing_metrics || [];
  document.getElementById("enrichment-cards").innerHTML = cards.map((metric) => `<article class="insight-card" data-metric="${escapeHtml(metric.key)}"><span>${escapeHtml(metric.label)}</span><strong>${formatMetricValue(metric.value)}</strong><small>${escapeHtml(metric.description)} ${escapeHtml(metric.calculation)}</small></article>`).join("");
  const explanation = document.getElementById("preprocessingExplanation");
  if (explanation) explanation.textContent = data.preprocessing_explanation || "Preprocessing accounting is unavailable.";
  renderColumnLineage(data.preprocessing?.column_lineage || []);
}

function renderColumnLineage(lineage) {
  const target = document.getElementById("columnLineageTable");
  const surface = document.getElementById("columnLineageSurface");
  const summary = document.getElementById("columnLineageSummary");
  if (!target || !surface) return;
  if (!lineage.length) {
    surface.classList.add("is-hidden");
    target.replaceChildren();
    return;
  }
  const retained = lineage.filter(column => column.retained).length;
  if (summary) summary.textContent = `${lineage.length} source columns · ${retained} retained roles`;
  target.innerHTML = `<table><thead><tr><th>Source column</th><th>Detected type</th><th>Forecast role</th><th>Status</th><th>Model usage / rationale</th></tr></thead><tbody>${lineage.map(column => {
    const usage = (column.used_by || []).join(", ") || column.reason || "Documented only";
    const status = column.retained ? "Retained" : "Excluded";
    return `<tr><td><strong>${escapeHtml(column.original_column)}</strong></td><td>${escapeHtml(column.physical_type)}</td><td>${escapeHtml(String(column.mapped_role || "ignored").replaceAll("_", " "))}</td><td><span class="model-status ${column.retained ? "completed" : "excluded"}">${status}</span></td><td class="lineage-reason">${escapeHtml(usage)}</td></tr>`;
  }).join("")}</tbody></table>`;
  surface.classList.remove("is-hidden");
}

function renderEnrichmentCharts(data) {
  const palette = { ink: "#132238", cyan: "#16b8d4", violet: "#6d5dfc", green: "#12a875", grid: "rgba(19,34,56,.07)" };
  chart("trendChart", "line", data.trend.map((x) => x.date), [{ label: "Target", data: data.trend.map((x) => x.value), borderColor: palette.cyan, backgroundColor: "rgba(22,184,212,.10)", fill: true, pointRadius: 0, borderWidth: 2, tension: .25 }]);
  chart("distributionChart", "bar", data.distribution.map((x) => x.label), [{ label: "Records", data: data.distribution.map((x) => x.count), backgroundColor: "rgba(109,93,252,.72)", borderRadius: 5 }]);
  chart("missingChart", "bar", data.missing_by_column.map((x) => x.column), [{ label: "Missing", data: data.missing_by_column.map((x) => x.count), backgroundColor: "rgba(228,154,34,.72)", borderRadius: 5 }], true);
  chart("seasonalityChart", "line", data.seasonality.map((x) => x.period), [{ label: "Average target", data: data.seasonality.map((x) => x.value), borderColor: palette.green, backgroundColor: "rgba(18,168,117,.10)", fill: true, tension: .35 }]);
  const mappedDimension = (data.top_dimensions || []).find(dimension => dimension.values?.length);
  const dimensions = mappedDimension?.values || (data.top_stores?.length ? data.top_stores : data.top_items || []);
  document.getElementById("dimensionTitle").textContent = mappedDimension ? `Top ${mappedDimension.display_name}` : dimensions.length ? "Top mapped dimension" : "Dimension contribution";
  chart("dimensionChart", "bar", dimensions.map((x) => x.label), [{ label: "Target volume", data: dimensions.map((x) => x.value), backgroundColor: "rgba(22,184,212,.72)", borderRadius: 5 }], true);

  function chart(id, type, labels, datasets, horizontal = false) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(document.getElementById(id), { type, data: { labels, datasets }, options: { responsive: true, maintainAspectRatio: false, indexAxis: horizontal ? "y" : "x", plugins: { legend: { display: false }, tooltip: { mode: "index", intersect: false } }, scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 8, color: "#8795a7", font: { size: 9 } } }, y: { grid: { color: palette.grid }, ticks: { maxTicksLimit: 6, color: "#8795a7", font: { size: 9 } } } } } });
  }
}

async function loadDataStudioAnalytics() {
  if (!datasetPage || !latestDataset?.adapted?.artifact_id) return;
  const state = canonicalDataStudioFilterState();
  const sequence = ++dataStudioAnalyticsSequence;
  dataStudioAnalyticsRequest?.abort();
  dataStudioAnalyticsRequest = new AbortController();
  const query = new URLSearchParams({
    dataset_id: state.dataset_id,
    artifact_id: state.artifact_id,
    dimensions: JSON.stringify(state.dimension_filters),
    request_id: String(sequence)
  });
  try {
    const response = await fetch(`/api/data-studio-analytics?${query}`, { signal: dataStudioAnalyticsRequest.signal });
    const payload = await response.json();
    if (sequence !== dataStudioAnalyticsSequence) return;
    if (!response.ok || payload.ok === false) throw new Error(payload.message || "Data Studio analytics are unavailable.");
    if (payload.request_id !== String(sequence) || payload.dataset_id !== state.dataset_id || payload.artifact_id !== state.artifact_id) return;
    const expectedFilters = Object.fromEntries(Object.entries(state.dimension_filters).filter(([, value]) => value).sort(([left], [right]) => left.localeCompare(right)));
    if (JSON.stringify(payload.requested_dimension_filters || {}) !== JSON.stringify(expectedFilters)) return;
    const selectionReset = renderDataStudioFilters(payload.dimension_schema || [], state);
    if (selectionReset) { loadDataStudioAnalytics(); return; }
    if (!payload.available) {
      clearDataStudioCharts();
      showDataStudioAnalyticsNotice(payload.message || "No analytical data matches the selected series.");
      return;
    }
    hideDataStudioAnalyticsNotice();
    renderEnrichmentCharts(payload.analytics || {});
  } catch (error) {
    if (error.name === "AbortError" || sequence !== dataStudioAnalyticsSequence) return;
    clearDataStudioCharts();
    showDataStudioAnalyticsNotice(error.message || "Data Studio analytics are unavailable.");
  }
}

function canonicalDataStudioFilterState() {
  const controls = [...document.querySelectorAll("[data-studio-dimension]")];
  const dimensions = Object.fromEntries(controls.map(select => [select.dataset.columnId, select.value || null]));
  return {
    dataset_id: latestDataset?.id || "",
    artifact_id: latestDataset?.adapted?.artifact_id || "",
    primary_dimension: controls[0] ? { column_id: controls[0].dataset.columnId, value: controls[0].value || null } : null,
    secondary_dimension: controls[1] ? { column_id: controls[1].dataset.columnId, value: controls[1].value || null } : null,
    dimension_filters: dimensions
  };
}

function renderDataStudioFilters(schema, state) {
  const container = document.getElementById("dataStudioFilters");
  if (!container) return false;
  if (!schema.length) {
    container.replaceChildren();
    container.classList.add("is-hidden");
    return false;
  }
  let selectionReset = false;
  const fragment = document.createDocumentFragment();
  schema.slice(0, 2).forEach((dimension, index) => {
    const label = document.createElement("label");
    label.append(document.createTextNode(dimension.display_name || `Series dimension ${index + 1}`));
    const select = document.createElement("select");
    select.dataset.studioDimension = "";
    select.dataset.columnId = dimension.column_id || dimension.id;
    select.dataset.dimensionIndex = String(index);
    select.append(new Option(dimension.aggregate_label || "All Series", ""));
    (dimension.values || []).forEach(option => select.append(new Option(option.label, option.value)));
    const previous = state.dimension_filters[select.dataset.columnId] || "";
    if (previous && [...select.options].some(option => option.value === previous)) select.value = previous;
    else if (previous) selectionReset = true;
    select.addEventListener("change", loadDataStudioAnalytics);
    label.append(select);
    fragment.append(label);
  });
  container.replaceChildren(fragment);
  container.classList.remove("is-hidden");
  return selectionReset;
}

function clearDataStudioCharts() {
  ["trendChart", "distributionChart", "missingChart", "seasonalityChart", "dimensionChart"].forEach(id => {
    charts[id]?.destroy();
    delete charts[id];
  });
}

function showDataStudioAnalyticsNotice(message) {
  const notice = document.getElementById("dataStudioAnalyticsNotice");
  if (notice) { notice.textContent = message; notice.classList.remove("is-hidden"); }
}

function hideDataStudioAnalyticsNotice() {
  const notice = document.getElementById("dataStudioAnalyticsNotice");
  if (notice) { notice.textContent = ""; notice.classList.add("is-hidden"); }
}

async function startTraining() {
  const button = document.getElementById("trainButton");
  try {
    button.disabled = true;
    button.innerHTML = '<i data-lucide="loader-circle"></i>Training in progress';
    if (document.getElementById("mappingForm") && (mappingDirty || !latestDataset?.adapted)) await saveCurrentMapping();
    const data = await postJson("/train", { dataset_id: latestDataset?.id });
    activeTrainingJobId = data.job_id || data.status?.job_id;
    activeTrainingDatasetId = data.dataset_id || data.status?.dataset_id || latestDataset?.id;
    renderTrainingStatus(data.status);
    beginTrainingPoll(activeTrainingJobId, activeTrainingDatasetId);
  } catch (error) {
    setNotice(error.message, "error-state");
    button.disabled = false;
    button.textContent = "Train Full ML Models";
  }
  lucide.createIcons();
}

async function loadTrainingStatus() {
  try {
    const response = await fetch("/training-status");
    const status = await response.json();
    renderTrainingStatus(status);
    if (status.status === "running") beginTrainingPoll(status.job_id, status.dataset_id);
  } catch (_) { return; }
}

function beginTrainingPoll(jobId = activeTrainingJobId, datasetId = activeTrainingDatasetId) {
  activeTrainingJobId = jobId || activeTrainingJobId;
  activeTrainingDatasetId = datasetId || activeTrainingDatasetId;
  clearInterval(trainingPoll);
  trainingPoll = setInterval(async () => {
    const [statusResponse] = await Promise.all([fetch("/training-status"), loadTrainingLog(activeTrainingJobId, activeTrainingDatasetId)]);
    const status = await statusResponse.json();
    if ((activeTrainingJobId && status.job_id !== activeTrainingJobId) || (activeTrainingDatasetId && status.dataset_id !== activeTrainingDatasetId)) return;
    renderTrainingStatus(status);
    if (terminalTrainingStatuses.has(status.status)) {
      clearInterval(trainingPoll);
      const button = document.getElementById("trainButton");
      if (button) { button.disabled = false; button.innerHTML = '<i data-lucide="cpu"></i>Train Full ML Models'; }
      lucide.createIcons();
    }
  }, 2000);
}

function renderTrainingStatus(status) {
  if (!trainingPanel || !status) return;
  window.updateSystemStatus?.(status);
  syncTrainingButton(status);
  if (status.available === false) {
    resetTrainingPanel();
    return;
  }
  const completed = status.completed_models || [];
  const failed = status.failed_models || [];
  const skipped = status.skipped_models || [];
  const totalModels = status.total_models || 13;
  const applicableTotal = Math.max(completed.length, totalModels - skipped.length);
  const successfulTerminal = ["completed", "completed_with_warnings"].includes(status.status);
  const visualStatus = status.status === "completed_with_warnings" || (status.status === "completed" && failed.length) ? "partial" : (status.status || "idle");
  const warnings = [status.environment_warning, status.persistence_warning, ...(status.artifact_warnings || [])].filter(Boolean);
  trainingPanel.innerHTML = `<div class="pipeline-summary status-${visualStatus}"><h3>${escapeHtml(status.current_step || "Training status")}</h3><div class="pipeline-grid"><span><strong>${escapeHtml(status.status || "idle")}</strong>Status</span><span><strong>${escapeHtml(status.duration_display || "0 sec")}</strong>Runtime</span><span><strong>${escapeHtml(status.current_model || "--")}</strong>Current model</span><span><strong>${completed.length} / ${applicableTotal}</strong>Applicable completed</span><span><strong>${skipped.length}</strong>Not applicable</span><span><strong>${failed.length}</strong>Failed</span><span><strong>${numberFormat(status.rows_used_for_training)}</strong>Rows used</span><span><strong>${escapeHtml(status.evaluation_mode || "--")}</strong>Evaluation</span><span><strong>${escapeHtml(status.last_data_date || "--")}</strong>Last actual</span><span><strong>${escapeHtml(status.forecast_start_date || "--")}</strong>Forecast starts</span><span><strong>${escapeHtml(status.forecast_horizon || "--")}</strong>Future points</span></div><p>${escapeHtml(status.message || "")}</p>${warnings.length ? `<div class="failure-list muted">${warnings.map((warning) => `<span>${escapeHtml(warning)}</span>`).join("")}</div>` : ""}${failed.length ? `<div class="failure-list">${failed.map((item) => `<span>${escapeHtml(item.model_label || item.model)}: ${escapeHtml(item.reason)}</span>`).join("")}</div>` : ""}${successfulTerminal ? '<div class="result-actions"><a href="/dashboard">Open Command Center</a><a href="/model-metrics">Open Model Leaderboard</a><a href="/forecast-explorer">Open Forecast Explorer</a></div>' : ""}</div>`;
  if (datasetPage && successfulTerminal) document.querySelector('[data-target="training-section"]')?.classList.add("completed");
}

function syncTrainingButton(status) {
  const button = document.getElementById("trainButton");
  if (!button) return;
  if (status.status === "running") {
    button.disabled = true;
    button.innerHTML = '<i data-lucide="loader-circle"></i>Training in progress';
  } else if (terminalTrainingStatuses.has(status.status)) {
    button.disabled = false;
    button.innerHTML = '<i data-lucide="cpu"></i>Train Full ML Models';
  }
}

async function loadTrainingLog(jobId = activeTrainingJobId, datasetId = activeTrainingDatasetId) {
  const target = document.getElementById("trainingLogPreview");
  if (!target) return;
  const statusResponse = await fetch("/api/training-status");
  if (!statusResponse.ok) return;
  const status = await statusResponse.json();
  if ((jobId && status.job_id !== jobId) || (datasetId && status.dataset_id !== datasetId)) return;
  const entries = [...(status.completed_models || []).map((x) => ({ model: x.model_label || x.model, message: "Completed", time: `${x.runtime_seconds}s` })), ...(status.failed_models || []).map((x) => ({ model: x.model_label || x.model, message: x.reason, time: `${x.runtime_seconds}s` }))].slice(-6);
  target.innerHTML = entries.map((entry) => `<div class="log-entry"><span>${escapeHtml(entry.time)}</span><strong>${escapeHtml(entry.model)}</strong><span>${escapeHtml(entry.message)}</span></div>`).join("");
}

async function postJson(url, payload) {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || "Request failed.");
  return data;
}

function setNotice(message, state) {
  const target = document.getElementById("datasetStatus");
  if (!target) return;
  target.className = `notice ${state}`;
  target.textContent = message;
}
function numberFormat(value) { return new Intl.NumberFormat().format(value || 0); }
function compactNumber(value) { return value == null ? "--" : new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 2 }).format(value); }
function formatMetricValue(value) { return typeof value === "number" ? numberFormat(value) : escapeHtml(value ?? "--"); }
function formatBytes(bytes) { return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`; }
function escapeHtml(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;"); }
