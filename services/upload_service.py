import time
import threading
import uuid
from pathlib import Path

from werkzeug.utils import secure_filename

from services.data_service import uploads_path
from services.dataset_adapter import SUPPORTED_EXTENSIONS, create_dataset_record


UPLOAD_LOCK = threading.Lock()


def save_uploaded_dataset(file_storage):
    if not file_storage or not file_storage.filename:
        raise ValueError("Select a CSV, TSV, TXT, or XLSX file to upload.")

    original_name = file_storage.filename
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload .csv, .tsv, .txt, or .xlsx.")

    safe_name = secure_filename(original_name) or f"dataset{suffix}"
    target_path = uploads_path(f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{safe_name}")
    file_storage.save(target_path)
    try:
        dataset = create_dataset_record(
            name=Path(original_name).stem.replace("_", " ").replace("-", " ").title(),
            source_file=target_path,
            source_type="upload",
            original_filename=original_name,
        )
    except Exception:
        target_path.unlink(missing_ok=True)
        raise
    dataset["uploaded_file"] = str(target_path)
    dataset["message"] = dataset.get("message") or "Dataset uploaded successfully."
    return {"dataset": dataset}
