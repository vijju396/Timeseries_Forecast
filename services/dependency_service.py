"""Cached model-dependency and interpreter diagnostics."""
import importlib.util
import logging
import os
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path


DEPENDENCIES = {
    "statsmodels": {
        "distribution": "statsmodels",
        "models": {"sarimax_Predictions", "sarimax_exog_Predictions", "exp_additive_Predictions", "exp_additive_damped_Predictions", "exp_multiplicative_Predictions", "exp_multiplicative_damped_Predictions", "var_Predictions", "var_exog_Predictions"},
    },
    "pmdarima": {"distribution": "pmdarima", "models": {"autoArima_Predictions", "autoArima_exog_Predictions"}},
    "xgboost": {"distribution": "xgboost-cpu", "models": {"xgboost_Predictions", "xgboost_exog_Predictions"}},
    "tensorflow": {"distribution": "tensorflow-cpu", "models": {"lstm_Predictions"}},
}


@lru_cache(maxsize=1)
def dependency_registry():
    registry = {}
    for module_name, specification in DEPENDENCIES.items():
        available = importlib.util.find_spec(module_name) is not None
        try:
            version = metadata.version(specification["distribution"]) if available else None
        except metadata.PackageNotFoundError:
            try:
                version = metadata.version(module_name) if available else None
            except metadata.PackageNotFoundError:
                version = None
        registry[module_name] = {
            "available": bool(available),
            "installed_version": version,
            "import_error": None if available else f"No module named '{module_name}'",
            "supported_models": sorted(specification["models"]),
        }
    return registry


def model_dependency_status(model_id, registry=None):
    registry = registry or dependency_registry()
    for module_name, specification in DEPENDENCIES.items():
        if model_id in specification["models"]:
            state = registry[module_name]
            return {"module": module_name, **state}
    return {"module": None, "available": True, "installed_version": None, "import_error": None, "supported_models": []}


def environment_diagnostics(project_root, executable=None):
    project = Path(project_root).resolve()
    active = Path(executable or sys.executable).resolve()
    expected = project / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    expected = expected.resolve() if expected.exists() else None
    pip_name = "pip.exe" if os.name == "nt" else "pip"
    pip_path = active.parent / pip_name
    return {
        "python_executable": str(active),
        "python_version": sys.version.split()[0],
        "working_directory": str(Path.cwd().resolve()),
        "virtual_environment": os.environ.get("VIRTUAL_ENV") or (str(Path(sys.prefix).resolve()) if sys.prefix != sys.base_prefix else None),
        "pip_executable": str(pip_path) if pip_path.exists() else f"{active} -m pip",
        "project_path": str(project),
        "app_path": str((project / "app.py").resolve()),
        "training_service_path": str((project / "services/training_service.py").resolve()),
        "expected_project_interpreter": str(expected) if expected else None,
        "interpreter_mismatch": bool(expected and active != expected),
        "sys_path": list(sys.path),
    }


def log_startup_diagnostics(project_root, development=False):
    logger = logging.getLogger("forecast.startup")
    dependencies = dependency_registry()
    summary = ", ".join(
        f"{name}={state.get('installed_version') or 'unavailable'}"
        for name, state in dependencies.items()
    )
    logger.info("Model dependencies: %s", summary)
    diagnostics = environment_diagnostics(project_root)
    if development:
        logger.info(
            "Python=%s version=%s venv=%s project=%s pip=%s mismatch=%s",
            diagnostics["python_executable"], diagnostics["python_version"],
            diagnostics["virtual_environment"], diagnostics["project_path"],
            diagnostics["pip_executable"], diagnostics["interpreter_mismatch"],
        )
    elif diagnostics["interpreter_mismatch"]:
        logger.warning("Server interpreter differs from the available project virtual environment.")
    return {"environment": diagnostics, "dependencies": dependencies}
