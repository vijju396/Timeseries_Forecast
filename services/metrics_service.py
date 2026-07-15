import math
from statistics import mean

from services.data_service import load_json


def calculate_metrics(actuals, predictions):
    pairs = []
    for actual, predicted in zip(actuals, predictions):
        actual_value = _to_float(actual)
        predicted_value = _to_float(predicted)
        if actual_value is not None and predicted_value is not None:
            pairs.append((actual_value, predicted_value))

    if not pairs:
        return {"mape": None, "accuracy": None, "mae": None, "rmse": None, "wape": None, "bias": None}

    abs_errors = [abs(actual - predicted) for actual, predicted in pairs]
    squared_errors = [(actual - predicted) ** 2 for actual, predicted in pairs]
    nonzero_pairs = [(actual, predicted) for actual, predicted in pairs if actual != 0]
    total_actual = sum(actual for actual, _predicted in pairs)

    mape = (
        mean(abs((actual - predicted) / actual) for actual, predicted in nonzero_pairs) * 100
        if nonzero_pairs
        else None
    )
    mae = mean(abs_errors)
    rmse = math.sqrt(mean(squared_errors))
    wape = (sum(abs_errors) / total_actual * 100) if total_actual else None
    bias = (sum(predicted - actual for actual, predicted in pairs) / total_actual * 100) if total_actual else None
    accuracy = max(100 - mape, 0) if mape is not None else None

    return {
        "mape": _round_or_none(mape),
        "accuracy": _round_or_none(accuracy),
        "mae": _round_or_none(mae),
        "rmse": _round_or_none(rmse),
        "wape": _round_or_none(wape),
        "bias": _round_or_none(bias),
    }


def get_model_metrics():
    metrics = load_json("model_metrics.json", [])
    return sorted((row for row in metrics if is_valid_metric(row)), key=lambda row: row["mape"])


def is_valid_metric(row):
    if not isinstance(row, dict) or str(row.get("status", "")).lower() == "failed":
        return False
    values = [_to_float(row.get(key)) for key in ("mape", "mae", "rmse")]
    if any(value is None or not math.isfinite(value) for value in values):
        return False
    mape, mae, rmse = values
    return 0 <= mape <= 1000 and 0 <= mae <= 1e12 and 0 <= rmse <= 1e12


def get_failed_model_metrics():
    metrics = load_json("model_metrics.json", [])
    failed = [row for row in metrics if not is_valid_metric(row)]
    known = {(row.get("model"), row.get("model_label")) for row in failed}
    status = load_json("training_status.json", {})
    for row in status.get("failed_models", []) or []:
        key = (row.get("model"), row.get("model_label"))
        if key not in known:
            failed.append(row)
    return failed


def get_champion():
    metrics = get_model_metrics()
    return metrics[0] if metrics else {}


def get_kpis():
    cards = load_json("kpis.json", [])
    champion = get_champion()
    training_status = load_json("training_status.json", {})
    if not champion:
        return cards

    refreshed = []
    has_runtime = False
    for card in cards:
        item = dict(card)
        if item.get("label") == "Best Model":
            item["value"] = champion.get("model", item.get("value", ""))
        if item.get("label") == "Last Training Runtime":
            item["value"] = training_status.get("duration_display", item.get("value", "0 sec"))
            has_runtime = True
        refreshed.append(item)
    if not has_runtime:
        refreshed.append(
            {
                "label": "Last Training Runtime",
                "value": training_status.get("duration_display", "0 sec"),
                "caption": "Most recent run",
            }
        )
    labels = {card.get("label") for card in refreshed}
    enrichment = load_json("enrichment.json", {})
    forecast = load_json("forecast_data.json", {})
    additions = [
        {
            "label": "Data Quality Score",
            "value": f"{enrichment.get('quality', {}).get('score')}/100" if enrichment.get("quality", {}).get("score") is not None else "--",
            "caption": "Cleanliness and completeness",
        },
        {
            "label": "Forecast Start Date",
            "value": forecast.get("forecast_start_date") or "--",
            "caption": "First future prediction",
        },
        {
            "label": "Forecast Horizon",
            "value": str(forecast.get("forecast_horizon", 0)),
            "caption": "Future forecast points",
        },
    ]
    refreshed.extend(card for card in additions if card["label"] not in labels)
    return refreshed


def get_metric_lookup():
    return {metric["model"]: metric for metric in get_model_metrics()}


def get_failed_models():
    status = load_json("training_status.json", {})
    rows = list(status.get("failed_models", []) or [])
    known = {(row.get("model"), row.get("model_label")) for row in rows}
    rows.extend(row for row in get_failed_model_metrics() if (row.get("model"), row.get("model_label")) not in known)
    return rows


def _to_float(value):
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_or_none(value):
    return round(value, 2) if value is not None else None
