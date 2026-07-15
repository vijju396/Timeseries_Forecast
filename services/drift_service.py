import math
import hashlib
import json
from statistics import mean, pstdev

from services.data_service import load_json, save_json


def calculate_past_drift(cleaned_or_forecast_data):
    rows = cleaned_or_forecast_data.get("historical", []) if isinstance(cleaned_or_forecast_data, dict) else []
    values = [_number(row.get("actual")) for row in rows]
    values = [value for value in values if value is not None]
    if len(values) < 4:
        return {"status": "insufficient_data", "mean_change_pct": None, "std_deviation_change_pct": None, "trend_change_pct": None, "volatility_change_pct": None, "anomaly_count": 0, "severity": None, "confidence": "insufficient", "explanation": "Insufficient observations to calculate drift.", "recommended_action": "Collect more history before interpreting drift.", "reference_window": {"start": None, "end": None}, "current_window": {"start": None, "end": None}}
    split = max(1, len(values) // 2)
    reference, current = values[:split], values[-split:]
    result = _drift_stats(reference, current, "past")
    result["reference_window"] = {"start": rows[0].get("date") if rows else None, "end": rows[max(0, split - 1)].get("date") if rows else None}
    result["current_window"] = {"start": rows[-split].get("date") if rows else None, "end": rows[-1].get("date") if rows else None}
    return result


def calculate_future_drift(forecast_data):
    historical = forecast_data.get("historical", [])
    future = forecast_data.get("future", [])
    actual = [_number(row.get("actual")) for row in historical]
    predicted = [_number(row.get("value", next(iter(row.get("predictions", {}).values()), None))) for row in future]
    actual = [value for value in actual if value is not None][-7:]
    predicted = [value for value in predicted if value is not None]
    if not predicted:
        return {"available": False, "reason": "no forecast", "status": "insufficient_data", "risk": "unavailable", "recent_actual_mean": round(mean(actual), 2) if actual else None, "future_forecast_mean": None, "forecast_change": None, "forecast_change_pct": None, "forecast_volatility": None, "confidence_band_widening": None, "expected_trend_direction": "unavailable", "confidence": "insufficient", "explanation": "No owned future forecast is available for this series.", "recommended_action": "Train a model that supports the selected series."}
    recent_mean = mean(actual) if actual else 0
    future_mean = mean(predicted) if predicted else recent_mean
    change = _percent_change(recent_mean, future_mean)
    widths = []
    for row in future:
        for band in row.get("confidence_bands", {}).values():
            width = (_number(band.get("upper")) or 0) - (_number(band.get("lower")) or 0)
            widths.append(width)
    status = "Drift Expected" if abs(change) >= 20 else "Watch" if abs(change) >= 8 else "Stable"
    direction = "increasing" if change > 2 else "decreasing" if change < -2 else "stable"
    risk = "high" if status == "Drift Expected" else "moderate" if status == "Watch" else "low"
    return {"available": True, "reason": "insufficient forecast horizon" if len(predicted) < 2 else None, "status": status, "risk": risk, "recent_actual_mean": round(recent_mean, 2), "future_forecast_mean": round(future_mean, 2), "forecast_change": round(change, 2), "forecast_change_pct": round(change, 2), "forecast_volatility": round(pstdev(predicted), 2) if len(predicted) > 1 else None, "confidence_band_widening": bool(widths and widths[-1] > widths[0]), "expected_trend_direction": direction, "confidence": "high" if len(actual) >= 7 and len(predicted) >= 7 else "limited", "explanation": f"Future forecast is {direction} versus the recent actual window ({change:.1f}% change).", "recommended_action": "Review drivers and prediction intervals." if status != "Stable" else "Continue monitoring."}


DRIFT_CONFIG_VERSION = "drift-v2"


def build_drift_payload(params=None):
    from services.forecast_service import get_forecast

    params = params or {}
    forecast = get_forecast(params)
    actual = forecast.get("historical_actual", [])
    predictions = forecast.get("historical_prediction", [])
    windows = _window_drift(actual, predictions, forecast.get("filters", {}).get("granularity", "Daily"))
    aggregate_payload = load_json("forecast_data.json", {})
    aggregate_historical = aggregate_payload.get("historical", [])
    dataset_past = calculate_past_drift({"historical": aggregate_historical})
    residual_windows = [row for row in windows if row.get("drift_type") == "residual"]
    residual_summary = _summary(residual_windows, actual) if predictions else {"overall_severity": "insufficient_data", "plain_language_explanation": "Residual drift is unavailable because this selected series has no historical backtest predictions.", "recommended_action": "Train a model with owned backtest predictions for this series.", "affected_variables": ["residual"]}
    future_drift = calculate_future_drift({"historical": [{"actual": row.get("value")} for row in actual], "future": forecast.get("future_prediction", [])})
    drift_result = _canonical_drift_result(windows, actual, future_drift)
    payload = {
        "ok": bool(windows), "available": bool(windows), "request_id": forecast.get("request_id"),
        "dataset_id": forecast.get("dataset_id"), "job_id": forecast.get("job_id"), "artifact_id": forecast.get("artifact_id"),
        "series_key": forecast.get("filters", {}).get("dimensions", {}),
        "series_key_hash": forecast.get("series_key_hash"),
        "granularity": forecast.get("filters", {}).get("granularity"),
        "target": forecast.get("filters", {}).get("target", "mapped target"),
        "filters": forecast.get("filters", {}),
        "effective_filters": forecast.get("effective_filters", {}),
        "drift_cache_key": _drift_cache_key(forecast, params),
        "drift_result": drift_result,
        "drift_series": drift_result["drift_series"],
        "summary": drift_result["summary"],
        "target_drift": drift_result["target_drift"],
        "future_drift": drift_result["future_drift"],
        "dataset_drift": dataset_past,
        "series_drift": drift_result["target_drift"],
        "residual_drift": residual_summary,
        "past": drift_result["target_drift"],
        "future": drift_result["future_drift"],
    }
    save_json("drift.json", payload)
    return payload


def _canonical_drift_result(windows, actual, future_drift):
    target_rows = sorted((row for row in windows if row.get("drift_type") == "target"), key=lambda row: row.get("timestamp") or "")
    strongest = max(target_rows, key=lambda row: row.get("score", 0), default=None)
    detected_rows = [row for row in target_rows if row.get("severity") != "stable"]
    target_drift = {
        "available": strongest is not None,
        "reason": None if strongest else "insufficient history",
        "status": strongest.get("severity") if strongest else "insufficient_data",
        "severity": strongest.get("score") if strongest else None,
        "detected": bool(detected_rows),
        "first_detected": detected_rows[0].get("timestamp") if detected_rows else None,
        "mean_change": strongest.get("mean_change") if strongest else None,
        "volatility_change": strongest.get("volatility_change") if strongest else None,
        "type": "target",
        "confidence": "high" if len(actual) >= 12 else "limited" if strongest else "insufficient",
    }
    return {
        "ok": bool(windows),
        "available": bool(windows),
        "drift_series": windows,
        "summary": _summary(windows, actual),
        "target_drift": target_drift,
        "future_drift": future_drift,
    }


def _window_drift(actual, predictions, granularity):
    values = [(row.get("timestamp"), _number(row.get("value"))) for row in actual]
    values = [(timestamp, value) for timestamp, value in values if timestamp and value is not None]
    if len(values) < 4:
        return []
    size = max(2, len(values) // 4)
    windows = []
    prediction_map = {row.get("timestamp"): _number(row.get("value")) for row in predictions}
    for index in range(size, len(values), size):
        reference = [value for _, value in values[max(0, index - size):index]]
        current_pairs = values[index:index + size]
        current = [value for _, value in current_pairs]
        if len(current) < 2:
            continue
        score, effect = _normalized_effect(reference, current)
        severity = _severity(score)
        timestamp = current_pairs[0][0]
        reference_std = pstdev(reference) if len(reference) > 1 else 0
        current_std = pstdev(current) if len(current) > 1 else 0
        mean_change = _percent_change(mean(reference), mean(current))
        volatility_change = _percent_change(reference_std, current_std)
        action = "Investigate the affected series and review model assumptions." if severity != "stable" else "Continue monitoring the next analysis window."
        windows.append({"timestamp": timestamp, "drift_type": "target", "score": round(score, 4), "severity": severity, "effect_size": round(effect, 4), "mean_change": round(mean_change, 2), "volatility_change": round(volatility_change, 2), "change_point": bool(score >= .65), "explanation": f"Target level and distribution changed by approximately {effect * 100:.1f}% versus the preceding window.", "recommended_action": action, "affected_variable": "target", "granularity": granularity})
        residuals_reference = [_number(prediction_map.get(ts)) - value for ts, value in values[max(0, index - size):index] if prediction_map.get(ts) is not None]
        residuals_current = [_number(prediction_map.get(ts)) - value for ts, value in current_pairs if prediction_map.get(ts) is not None]
        if len(residuals_reference) >= 2 and len(residuals_current) >= 2:
            residual_score, residual_effect = _normalized_effect(residuals_reference, residuals_current)
            windows.append({"timestamp": timestamp, "drift_type": "residual", "score": round(residual_score, 4), "severity": _severity(residual_score), "effect_size": round(residual_effect, 4), "change_point": bool(residual_score >= .65), "explanation": "Out-of-sample residual behavior changed relative to the preceding window.", "recommended_action": "Review the selected model's recent backtest errors.", "affected_variable": "residual", "granularity": granularity})
    return windows


def _normalized_effect(reference, current):
    reference_mean, current_mean = mean(reference), mean(current)
    reference_std = pstdev(reference) if len(reference) > 1 else 0
    current_std = pstdev(current) if len(current) > 1 else 0
    mean_effect = abs(_percent_change(reference_mean, current_mean)) / 100
    volatility_effect = abs(_percent_change(reference_std, current_std)) / 100 if reference_std else 0
    effect = min(1.0, max(mean_effect, volatility_effect))
    return min(1.0, effect), effect


def _severity(score):
    return "critical" if score >= .75 else "high" if score >= .5 else "moderate" if score >= .25 else "stable"


def _summary(windows, actual):
    if not windows:
        return {"overall_severity": "insufficient_data", "start_period": None, "drift_start": None, "affected_variables": [], "plain_language_explanation": "Insufficient historical observations to calculate time-window drift.", "recommended_action": "Collect more history before interpreting drift."}
    strongest = max(windows, key=lambda row: row.get("score", 0))
    variables = sorted({row.get("affected_variable") for row in windows if row.get("affected_variable")})
    gradual_or_sudden = "sudden" if strongest["change_point"] else "gradual"
    return {"overall_severity": strongest["severity"], "start_period": strongest["timestamp"], "drift_start": strongest["timestamp"], "gradual_or_sudden": gradual_or_sudden, "affected_variables": variables, "forecast_impact": "Forecast uncertainty may be elevated; review the selected model." if strongest["severity"] != "stable" else "No material forecast impact is indicated.", "plain_language_explanation": f"{strongest['severity'].title()} {strongest['drift_type']} drift was detected from {strongest['timestamp']}. The change appears {gradual_or_sudden} and affects {', '.join(variables)}.", "recommended_action": strongest["recommended_action"], "confidence": "high" if len(actual) >= 12 else "limited"}


def _drift_cache_key(forecast, params):
    key = {"dataset_id": forecast.get("dataset_id"), "artifact_id": forecast.get("artifact_id"), "job_id": forecast.get("job_id"), "series_key_hash": forecast.get("series_key_hash"), "filters": forecast.get("filters", {}), "target": forecast.get("filters", {}).get("target", "mapped target"), "config_version": DRIFT_CONFIG_VERSION, "request": params}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _drift_stats(reference, current, _kind):
    reference_mean, current_mean = (mean(values) if values else 0 for values in (reference, current))
    reference_std = pstdev(reference) if len(reference) > 1 else 0
    current_std = pstdev(current) if len(current) > 1 else 0
    mean_change = _percent_change(reference_mean, current_mean)
    volatility_change = _percent_change(reference_std, current_std)
    trend_change = _percent_change(reference[-1] - reference[0], current[-1] - current[0]) if len(reference) > 1 and len(current) > 1 else 0
    anomalies = 0
    if reference_std:
        anomalies = sum(abs(value - reference_mean) > 3 * reference_std for value in current)
    status = "Drift Detected" if max(abs(mean_change), abs(volatility_change)) >= 20 else "Watch" if max(abs(mean_change), abs(volatility_change)) >= 8 else "Stable"
    severity = max(abs(mean_change), abs(volatility_change))
    explanation = f"Recent historical values changed by {mean_change:.1f}% in mean and {volatility_change:.1f}% in volatility versus the reference window."
    return {"status": status, "mean_change_pct": round(mean_change, 2), "std_deviation_change_pct": round(volatility_change, 2), "trend_change_pct": round(trend_change, 2), "volatility_change_pct": round(volatility_change, 2), "anomaly_count": int(anomalies), "severity": round(severity, 2), "confidence": "high" if len(reference) >= 10 and len(current) >= 10 else "limited", "explanation": explanation, "recommended_action": "Investigate recent data and model residuals." if status != "Stable" else "Continue monitoring."}


def _percent_change(old, new):
    return (new - old) / abs(old) * 100 if old else (100 if new else 0)


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None
