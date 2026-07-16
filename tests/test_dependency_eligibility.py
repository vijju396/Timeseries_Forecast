"""Focused dependency, interpreter, and model-eligibility checks."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import dependency_service, training_service


def _registry(missing=None):
    missing = set(missing or [])
    return {
        name: {
            "available": name not in missing,
            "installed_version": None if name in missing else "test",
            "import_error": f"No module named '{name}'" if name in missing else None,
            "supported_models": sorted(specification["models"]),
        }
        for name, specification in dependency_service.DEPENDENCIES.items()
    }


def _frame(with_exogenous=False):
    frame = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=40, freq="D"),
        "target": range(40), "week_exog": 1, "dayOfWeek_exog": 1, "month_exog": 1,
    })
    if with_exogenous:
        frame["exogenous_1"] = range(40)
    return frame


def main():
    available = _registry()
    missing = _registry({"pmdarima"})
    assert all(state["available"] for state in dependency_service.dependency_registry().values())
    assert training_service._model_eligibility("sarimax_Predictions", "_fit_sarimax", _frame(), missing)["eligible"]
    unavailable = training_service._model_eligibility("autoArima_Predictions", "_fit_auto_arima", _frame(), missing)
    assert unavailable["reason_code"] == "dependency_unavailable"

    no_exog = training_service._model_eligibility("sarimax_exog_Predictions", "_fit_sarimax_exog", _frame(), available)
    assert no_exog["eligible"] is True
    assert training_service._model_input_summary("sarimax_exog_Predictions", "_fit_sarimax_exog", _frame())["exogenous_mode"] == "deterministic_calendar"
    assert training_service._model_eligibility("sarimax_exog_Predictions", "_fit_sarimax_exog", _frame(True), available)["eligible"]
    panel = _frame()
    panel["endogenous_1"] = panel["target"] * 0.4
    panel["endogenous_2"] = panel["target"] * 0.6
    for model_id, _label, fitter_name in training_service.MODEL_SPECS:
        assert training_service._model_eligibility(model_id, fitter_name, panel, available)["eligible"] is True

    diagnostics = dependency_service.environment_diagnostics(ROOT, executable=ROOT / "not-the-project-python.exe")
    assert diagnostics["interpreter_mismatch"] is True
    assert ".venv" in (diagnostics["expected_project_interpreter"] or "")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    procfile = (ROOT / "Procfile").read_text(encoding="utf-8")
    assert ".\\.venv\\Scripts\\python.exe app.py" in readme
    assert "python -m gunicorn" in dockerfile and "python -m gunicorn" in procfile
    print("dependency and eligibility tests passed: 10 checks")


if __name__ == "__main__":
    main()
