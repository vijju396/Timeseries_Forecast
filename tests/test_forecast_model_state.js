const fs = require("fs");
const vm = require("vm");

global.document = { getElementById: () => null };
const source = fs.readFileSync("static/js/forecast_explorer.js", "utf8");
vm.runInThisContext(`${source}\nglobal.resolveForecastModelSelectionForTest = resolveForecastModelSelection;`);

const choose = global.resolveForecastModelSelectionForTest;
const aggregateModels = [
  { model_id: "auto_arima", supports_selected_series: true },
  { model_id: "exp_additive", supports_selected_series: true }
];
const seriesModels = [
  { model_id: "exp_additive", supports_selected_series: true },
  { model_id: "auto_arima", supports_selected_series: false }
];
const champion = { global_champion: { model_id: "auto_arima" }, recommended_model_id: "auto_arima" };

let state = { selected_model_id: "", selection_source: "global_champion", selection_scope: "aggregate", manual_aggregate_model_id: null };
state = { ...state, ...choose(aggregateModels, champion, state, "aggregate") };
if (state.selected_model_id !== "auto_arima" || state.selection_source !== "global_champion") throw new Error("Initial aggregate champion was not selected.");

state = { ...state, ...choose(seriesModels, { recommended_model_id: "exp_additive" }, state, "series") };
if (state.selected_model_id !== "exp_additive" || state.selection_source !== "automatic_series_fallback") throw new Error("Series fallback was not selected.");

state = { ...state, ...choose(aggregateModels, champion, state, "aggregate") };
if (state.selected_model_id !== "auto_arima" || state.selection_source !== "global_champion") throw new Error("Aggregate champion was not restored.");

const compatibleSeries = aggregateModels.map(model => ({ ...model, supports_selected_series: true }));
state = { selected_model_id: "auto_arima", selection_source: "global_champion", selection_scope: "aggregate", manual_aggregate_model_id: null };
state = { ...state, ...choose(compatibleSeries, { recommended_model_id: "exp_additive" }, state, "series") };
if (state.selected_model_id !== "auto_arima") throw new Error("Compatible champion should be retained.");

state = { selected_model_id: "exp_additive", selection_source: "manual_user_selection", selection_scope: "aggregate", manual_aggregate_model_id: "exp_additive" };
state = { ...state, ...choose(compatibleSeries, {}, state, "series") };
state = { ...state, ...choose(aggregateModels, champion, state, "aggregate") };
if (state.selected_model_id !== "exp_additive" || state.selection_source !== "manual_user_selection") throw new Error("Manual aggregate selection was not retained.");

console.log("forecast model state tests passed: 5");
