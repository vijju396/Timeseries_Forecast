import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
PREPROCESSED_DIR = DATA_DIR / "Preprocessed"
UPLOADS_DIR = DATA_DIR / "uploads"

_configured_runtime_dir = Path(os.environ.get("FORECAST_DATA_DIR", "Data/runtime"))
RUNTIME_DATA_DIR = _configured_runtime_dir if _configured_runtime_dir.is_absolute() else BASE_DIR / _configured_runtime_dir
ONEDRIVE_WARNING = ""
_runtime_write_lock = threading.Lock()
_RUNTIME_MEMORY = {}
DATASET_OUTPUT_FILES = {
    "enrichment.json",
    "training_status.json",
    "training_log.json",
    "model_metrics.json",
    "kpis.json",
    "forecast_data.json",
    "forecast_manifest.json",
    "drift.json",
}


def initialize_runtime_state():
    ensure_data_directories()
    if os.environ.get("CLEAR_RUNTIME_ON_START", "true").strip().lower() in {"1", "true", "yes", "on"}:
        clear_generated_dataset_state()
    else:
        _cleanup_stale_temp_files()


def _cleanup_stale_temp_files():
    for path in RUNTIME_DATA_DIR.glob("*.tmp"):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logging.getLogger(__name__).warning("Could not remove stale runtime temp file %s: %s", path, exc)


def ensure_data_directories():
    for directory in (DATA_DIR, RUNTIME_DATA_DIR, PREPROCESSED_DIR, UPLOADS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _is_project_child(path, allow_project_root=False):
    candidate = Path(path).resolve()
    root = BASE_DIR.resolve()
    if candidate == root:
        return allow_project_root
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _remove_generated_contents(directory):
    directory = Path(directory).resolve()
    if not _is_project_child(directory):
        raise RuntimeError(f"Refusing cleanup outside project directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if not _is_project_child(child):
            raise RuntimeError(f"Refusing cleanup outside project directory: {child}")
        if child.is_dir():
            import shutil

            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def clear_generated_dataset_state():
    """Remove only generated dataset/runtime state inside this project."""
    _remove_generated_contents(RUNTIME_DATA_DIR)
    _remove_generated_contents(UPLOADS_DIR)
    _remove_generated_contents(PREPROCESSED_DIR)
    legacy_click_counts = DATA_DIR / "Click Counts.txt"
    if _is_project_child(legacy_click_counts):
        legacy_click_counts.unlink(missing_ok=True)
    _RUNTIME_MEMORY.clear()


def clear_runtime_json(exclude=None):
    excluded = set(exclude or ())
    RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in RUNTIME_DATA_DIR.glob("*.json"):
        if path.name not in excluded:
            path.unlink(missing_ok=True)


def clear_dataset_outputs():
    for filename in DATASET_OUTPUT_FILES:
        data_path(filename).unlink(missing_ok=True)


def clear_active_runtime_state():
    clear_dataset_outputs()
    data_path("current_dataset.json").unlink(missing_ok=True)


def activate_dataset(dataset_id):
    clear_dataset_outputs()
    save_json("current_dataset.json", {"dataset_id": dataset_id})
    return dataset_id


def current_dataset_id():
    return load_json("current_dataset.json", {}).get("dataset_id")


def data_path(filename):
    RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DATA_DIR / filename


def preprocessed_path(filename):
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PREPROCESSED_DIR / filename


def uploads_path(filename=""):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR / filename


def load_json(filename, default=None):
    path = data_path(filename)
    if not path.exists():
        return _RUNTIME_MEMORY.get(filename, default if default is not None else {})

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
            _RUNTIME_MEMORY[filename] = value
            return value
    except (OSError, ValueError):
        return _RUNTIME_MEMORY.get(filename, default if default is not None else {})


def save_json(filename, payload):
    path = data_path(filename)
    safe_write_json(path, payload)
    _RUNTIME_MEMORY[filename] = payload
    return payload


def safe_write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.{uuid.uuid4().hex}.tmp")

    with _runtime_write_lock:
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            for attempt in range(5):
                try:
                    os.replace(temporary, target)
                    return
                except PermissionError:
                    if attempt == 4:
                        break
                    time.sleep(0.05 * (attempt + 1))

            logging.getLogger(__name__).warning("Could not persist runtime JSON %s after 5 retries; keeping in-memory state.", target)
            return False
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
