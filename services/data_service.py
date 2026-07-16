import json
import logging
import os
import shutil
import threading
import time
import uuid
from contextvars import ContextVar
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
_RUNTIME_NAMESPACE = ContextVar("forecast_runtime_namespace", default="")
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
        clear_all_generated_dataset_state()
    else:
        _cleanup_stale_temp_files()


def _cleanup_stale_temp_files():
    for path in RUNTIME_DATA_DIR.rglob("*.tmp"):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logging.getLogger(__name__).warning("Could not remove stale runtime temp file %s: %s", path, exc)


def ensure_data_directories():
    for directory in (DATA_DIR, RUNTIME_DATA_DIR, PREPROCESSED_DIR, UPLOADS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for directory in (_runtime_directory(), _preprocessed_directory(), _uploads_directory()):
        directory.mkdir(parents=True, exist_ok=True)


def set_runtime_namespace(namespace):
    """Bind generated state to one signed browser session or background job."""
    value = str(namespace or "").strip()
    if value and (len(value) > 64 or not value.replace("-", "").replace("_", "").isalnum()):
        raise ValueError("Invalid runtime namespace.")
    _RUNTIME_NAMESPACE.set(value)
    return value


def current_runtime_namespace():
    return _RUNTIME_NAMESPACE.get()


def _scoped_directory(directory):
    namespace = current_runtime_namespace()
    return Path(directory) / "sessions" / namespace if namespace else Path(directory)


def _runtime_directory():
    return _scoped_directory(RUNTIME_DATA_DIR)


def _preprocessed_directory():
    return _scoped_directory(PREPROCESSED_DIR)


def _uploads_directory():
    return _scoped_directory(UPLOADS_DIR)


def _memory_key(filename):
    return (current_runtime_namespace() or "__global__", filename)


def _clear_namespace_memory():
    namespace = current_runtime_namespace() or "__global__"
    for key in [key for key in _RUNTIME_MEMORY if key[0] == namespace]:
        _RUNTIME_MEMORY.pop(key, None)


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


def _remove_generated_path(path, attempts=5, delay=0.05):
    path = Path(path)
    for attempt in range(max(1, attempts)):
        try:
            if not path.exists() and not path.is_symlink():
                return True
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            return True
        except FileNotFoundError:
            return True
        except (PermissionError, OSError) as exc:
            if attempt + 1 < attempts:
                time.sleep(delay * (attempt + 1))
                continue
            logging.getLogger(__name__).warning(
                "Could not remove locked generated path after %s attempts; startup will continue: %s (%s)",
                attempts, path, exc,
            )
            return False


def _remove_generated_contents(directory):
    directory = Path(directory).resolve()
    if not _is_project_child(directory):
        raise RuntimeError(f"Refusing cleanup outside project directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if not _is_project_child(child):
            raise RuntimeError(f"Refusing cleanup outside project directory: {child}")
        _remove_generated_path(child)


def clear_generated_dataset_state():
    """Remove generated state owned by the current browser session."""
    _remove_generated_contents(_runtime_directory())
    _remove_generated_contents(_uploads_directory())
    _remove_generated_contents(_preprocessed_directory())
    _clear_namespace_memory()


def clear_all_generated_dataset_state():
    """Startup-only cleanup of all generated state inside this project."""
    _remove_generated_contents(RUNTIME_DATA_DIR)
    _remove_generated_contents(UPLOADS_DIR)
    _remove_generated_contents(PREPROCESSED_DIR)
    legacy_click_counts = DATA_DIR / "Click Counts.txt"
    if _is_project_child(legacy_click_counts):
        _remove_generated_path(legacy_click_counts)
    _RUNTIME_MEMORY.clear()


def clear_runtime_json(exclude=None):
    excluded = set(exclude or ())
    runtime_directory = _runtime_directory()
    runtime_directory.mkdir(parents=True, exist_ok=True)
    for path in runtime_directory.glob("*.json"):
        if path.name not in excluded:
            path.unlink(missing_ok=True)
            _RUNTIME_MEMORY.pop(_memory_key(path.name), None)


def clear_dataset_outputs():
    for filename in DATASET_OUTPUT_FILES:
        data_path(filename).unlink(missing_ok=True)
        _RUNTIME_MEMORY.pop(_memory_key(filename), None)


def clear_active_runtime_state():
    clear_dataset_outputs()
    data_path("current_dataset.json").unlink(missing_ok=True)
    _RUNTIME_MEMORY.pop(_memory_key("current_dataset.json"), None)


def activate_dataset(dataset_id):
    clear_dataset_outputs()
    save_json("current_dataset.json", {"dataset_id": dataset_id})
    return dataset_id


def current_dataset_id():
    return load_json("current_dataset.json", {}).get("dataset_id")


def data_path(filename):
    directory = _runtime_directory()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def preprocessed_path(filename):
    directory = _preprocessed_directory()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def uploads_path(filename=""):
    directory = _uploads_directory()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def load_json(filename, default=None):
    path = data_path(filename)
    memory_key = _memory_key(filename)
    if not path.exists():
        return _RUNTIME_MEMORY.get(memory_key, default if default is not None else {})

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
            _RUNTIME_MEMORY[memory_key] = value
            return value
    except (OSError, ValueError):
        return _RUNTIME_MEMORY.get(memory_key, default if default is not None else {})


def save_json(filename, payload):
    path = data_path(filename)
    safe_write_json(path, payload)
    _RUNTIME_MEMORY[_memory_key(filename)] = payload
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
