"""Focused browser-session storage isolation checks."""

import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import data_service


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = (data_service.BASE_DIR, data_service.DATA_DIR, data_service.PREPROCESSED_DIR, data_service.UPLOADS_DIR, data_service.RUNTIME_DATA_DIR)
        try:
            data_service.BASE_DIR = root
            data_service.DATA_DIR = root / "Data"
            data_service.PREPROCESSED_DIR = data_service.DATA_DIR / "Preprocessed"
            data_service.UPLOADS_DIR = data_service.DATA_DIR / "uploads"
            data_service.RUNTIME_DATA_DIR = data_service.DATA_DIR / "runtime"
            data_service.set_runtime_namespace("browser-a")
            data_service.save_json("current_dataset.json", {"dataset_id": "dataset-a"})
            a_upload = data_service.uploads_path("a.csv")
            a_upload.write_text("a", encoding="utf-8")

            data_service.set_runtime_namespace("browser-b")
            assert data_service.current_dataset_id() is None
            assert not data_service.uploads_path("a.csv").exists()
            data_service.save_json("current_dataset.json", {"dataset_id": "dataset-b"})
            data_service.clear_generated_dataset_state()
            assert data_service.current_dataset_id() is None

            data_service.set_runtime_namespace("browser-a")
            assert data_service.current_dataset_id() == "dataset-a"
            assert a_upload.exists()

            from services import training_service

            training_service._set_memory_status({"dataset_id": "dataset-a", "status": "completed"})
            data_service.set_runtime_namespace("browser-b")
            training_service._set_memory_status({"dataset_id": "dataset-b", "status": "idle"})
            data_service.set_runtime_namespace("browser-a")
            assert training_service._memory_status()["dataset_id"] == "dataset-a"
            print("session storage isolation: PASS")
        finally:
            data_service.set_runtime_namespace("")
            data_service.BASE_DIR, data_service.DATA_DIR, data_service.PREPROCESSED_DIR, data_service.UPLOADS_DIR, data_service.RUNTIME_DATA_DIR = original


if __name__ == "__main__":
    main()
