import math
import hashlib
import json
import os
import re
import time
import warnings
from pathlib import Path

from services.data_service import DATA_DIR, activate_dataset, clear_dataset_outputs, current_dataset_id, load_json, preprocessed_path, save_json, uploads_path
from services import raw_preview_service
from services.series_service import build_dimension_schema, dimension_options, series_count


DATE_NAME_HINTS = {"date", "order date", "sales date", "ds", "timestamp", "week", "month"}
TARGET_NAME_HINTS = {"sales", "demand", "quantity", "qty", "revenue", "target", "y", "clicks", "click count"}
STORE_HINTS = ("store id", "store", "region", "site", "outlet", "location", "market", "branch")
ITEM_HINTS = ("item id", "item", "sku", "product", "article", "material")
SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx"}
PREPROCESSING_VERSION = "2026-07-dynamic-v1"


def analyze_file(path, original_filename=None, dataset_id=None, source_hash=None):
    dataset_id = dataset_id or _dataset_id(path, "upload", source_hash)
    preview, raw_schema, df = raw_preview_service.build_raw_preview(
        path,
        original_filename or Path(path).name,
        dataset_id,
        source_hash=source_hash,
    )
    mapping = detect_mapping(df)
    mapping = _mapping_with_column_ids(mapping, raw_schema)
    return {
        "raw_preview": preview,
        "raw_schema": raw_schema,
        "mapping": mapping,
        "suggested_mapping": mapping,
        "requires_mapping": mapping.get("requires_mapping", True),
    }


def preview_file(path, limit=10):
    source_hash = raw_preview_service.source_file_hash(path)
    dataset_id = _dataset_id(path, "preview", source_hash)
    preview, _schema, _frame = raw_preview_service.build_raw_preview(path, Path(path).name, dataset_id, limit=limit, source_hash=source_hash)
    return preview


def detect_mapping(df):
    columns = [str(column) for column in df.columns]
    date_column, date_ratio = _detect_date_column(df, columns)
    target_column, target_ratio, numeric_candidates = _detect_target_column(df, columns, date_column)
    store_column = _first_name_match(columns, STORE_HINTS)
    item_column = _first_name_match(columns, ITEM_HINTS)

    target_named = _normalized(target_column) in TARGET_NAME_HINTS if target_column else False
    target_confident = bool(target_column) and target_ratio >= 0.6 and (
        target_named or len(numeric_candidates) == 1
    )
    date_confident = bool(date_column) and date_ratio >= 0.6
    messages = []
    if not date_confident:
        messages.append("Could not detect date column. Please select it manually.")
    if not target_confident:
        messages.append("Could not detect target column. Please select it manually.")

    return {
        "csv_file": "",
        "date_column": date_column or "",
        "target_column": target_column or "",
        "store_column": store_column or "",
        "item_column": item_column or "",
        "dimension_columns": [column for column in (store_column, item_column) if column],
        "date_confidence": round(date_ratio, 2),
        "target_confidence": round(target_ratio, 2),
        "requires_mapping": not (date_confident and target_confident),
        "messages": messages,
    }


def adapt_dataset(source_file, mapping, dataset_id=None):
    source_path = Path(source_file)
    if not source_path.exists():
        raise ValueError("Dataset file was not found.")

    df = _read_tabular(source_path)
    date_column = mapping.get("date_column", "")
    target_column = mapping.get("target_column", "")
    if date_column not in df.columns:
        raise ValueError("Could not detect date column. Please select it manually.")
    if target_column not in df.columns:
        raise ValueError("Could not detect target column. Please select it manually.")

    import pandas as pd

    dimension_schema = build_dimension_schema(mapping, df.columns)
    parsed_dates = _parse_dates_best(df[date_column], strict=True)
    target, numeric_audit = _numeric_series(df[target_column], return_audit=True, allow_percent=("percent" in _normalized(target_column) or os.environ.get("FORECAST_TARGET_IS_PERCENT", "false").lower() == "true"))
    parse_denominator = max(1, len(df) - numeric_audit["missing_values"])
    numeric_audit["malformed_rate"] = round(numeric_audit["malformed_values"] / parse_denominator, 6)
    if numeric_audit["malformed_rate"] > float(os.environ.get("FORECAST_MAX_NUMERIC_FAILURE_RATE", "0.25")):
        raise ValueError(f"Target column '{target_column}' contains too many malformed numeric values ({numeric_audit['malformed_values']} rows).")
    raw_missing_values = int(df.isna().sum().sum())
    invalid_date_rows = int(parsed_dates.isna().sum())
    invalid_target_rows = int(target.isna().sum())
    invalid_date_mask = parsed_dates.isna()
    invalid_target_mask = target.isna()
    removal_reasons = {
        "invalid_date_only": int((invalid_date_mask & ~invalid_target_mask).sum()),
        "invalid_target_only": int((~invalid_date_mask & invalid_target_mask).sum()),
        "invalid_date_and_target": int((invalid_date_mask & invalid_target_mask).sum()),
    }
    negative_target_rows = int((target < 0).fillna(False).sum())
    clean = pd.DataFrame({"Date": parsed_dates, "Click Count": target})
    for dimension in dimension_schema:
        clean[dimension["canonical_column"]] = _dimension_series(df, dimension["source_column"], "")
    before_rows = len(clean)
    valid_mask = clean["Date"].notna() & clean["Click Count"].notna()
    valid_rows = int(valid_mask.sum())
    rows_removed = int(before_rows - valid_rows)
    clean = clean.dropna(subset=["Date", "Click Count"])
    clean["Date"] = pd.to_datetime(clean["Date"])
    for dimension in dimension_schema:
        column = dimension["canonical_column"]
        clean[column] = clean[column].fillna("(missing)").astype(str).str.strip().replace("", "(missing)")
    grain_columns = ["Date"] + [dimension["canonical_column"] for dimension in dimension_schema]
    aggregation = _aggregation_method(target_column)
    before_aggregation_rows = len(clean)
    clean = _aggregate_clean(clean, aggregation, dimension_schema).sort_values(grain_columns)
    after_aggregation_rows = len(clean)
    rows_aggregated = int(before_aggregation_rows - after_aggregation_rows)

    if len(clean) < 3:
        raise ValueError("Not enough valid date/target rows to train a forecast.")

    freq = _infer_frequency(clean["Date"])
    imputation = _imputation_policy(target_column)
    clean, imputation_audit = _fill_group_date_gaps(clean, freq, imputation, dimension_schema)
    calendar_rows_generated = int(imputation_audit["calendar_rows_generated"])
    unresolved_generated_rows_removed = int(imputation_audit["unresolved_generated_rows_removed"])
    training_rows = int(len(clean))

    # Preserve the two existing UI aliases while runtime identity uses the mapped dimension schema.
    clean["Store"] = clean[dimension_schema[0]["canonical_column"]] if dimension_schema else "All Stores"
    clean["Item"] = clean[dimension_schema[1]["canonical_column"]] if len(dimension_schema) > 1 else "All Items"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = preprocessed_path("cleaned_training_input.csv")
    # Keep generated compatibility output dataset-scoped. A fixed file name can
    # remain locked by OneDrive/Excel/antivirus while a new upload is processed.
    click_counts_path = preprocessed_path(f"click_counts_{dataset_id}.txt")
    output_columns = ["Date", "Click Count"] + [dimension["canonical_column"] for dimension in dimension_schema] + ["Store", "Item"]
    clean[output_columns].to_csv(cleaned_path, index=False)
    clean[output_columns].to_csv(click_counts_path, sep="\t", index=False)

    from services.enrichment_service import build_dataset_profile, build_enrichment_payload, save_enrichment
    from services.preprocessing_service import build_preprocessed_files

    preprocessing = build_preprocessed_files(clean)
    cleaning_stats = {
        "raw_rows": before_rows,
        "raw_columns": len(df.columns),
        "valid_rows": valid_rows,
        "rows_removed": rows_removed,
        "rows_aggregated": rows_aggregated,
        "calendar_rows_generated": calendar_rows_generated,
        "training_rows": training_rows,
        "missing_values_imputed": int(imputation_audit["existing_values_imputed"]),
        "missing_timestamps_generated": calendar_rows_generated,
        "unresolved_generated_rows_removed": unresolved_generated_rows_removed,
        "row_accounting_expected": valid_rows - rows_aggregated + calendar_rows_generated - unresolved_generated_rows_removed,
        "row_accounting_reconciled": training_rows == valid_rows - rows_aggregated + calendar_rows_generated - unresolved_generated_rows_removed,
        "invalid_date_rows": invalid_date_rows,
        "invalid_target_rows": invalid_target_rows,
        "rows_removed_reasons": removal_reasons,
        "negative_target_rows": negative_target_rows,
        "source_missing_values": raw_missing_values,
        "numeric_audit": numeric_audit,
        "aggregation": aggregation,
        "imputation": imputation_audit,
        "frequency": freq,
        "target_display_name": target_column,
        "dimension_schema": dimension_schema,
        "series_count": series_count(clean, dimension_schema),
        "forecast_grain": {"frequency": freq, "dimensions": [dimension["id"] for dimension in dimension_schema]},
    }
    mapping_hash = _mapping_hash(mapping)
    source_signature = _source_signature(source_path)
    preprocessing_config = _preprocessing_config(target_column, freq)
    preprocessing_config_hash = hashlib.sha256(json.dumps(preprocessing_config, sort_keys=True).encode("utf-8")).hexdigest()
    artifact_id = hashlib.sha256(f"{dataset_id}|{source_signature}|{mapping_hash}|{preprocessing_config_hash}".encode("utf-8")).hexdigest()[:20]
    profile = build_dataset_profile(clean, raw_df=df, cleaning_stats=cleaning_stats)
    enriched = build_enrichment_payload(clean, profile=profile, raw_df=df)
    enriched.update({"artifact_id": artifact_id, "mapping_hash": mapping_hash, "preprocessing_config_hash": preprocessing_config_hash})
    save_enrichment(enriched, dataset_id=dataset_id)
    return {
        "dataset_id": dataset_id,
        "mapping_hash": mapping_hash,
        "source_signature": source_signature,
        "preprocessing_config": preprocessing_config,
        "preprocessing_config_hash": preprocessing_config_hash,
        "artifact_id": artifact_id,
        "source_file": str(source_path),
        "target_column": target_column,
        "date_column": date_column,
        "cleaned_path": str(cleaned_path),
        "click_counts_path": str(click_counts_path),
        "rows_read": before_rows,
        "rows_used": training_rows,
        "stores": sorted(clean["Store"].dropna().astype(str).unique().tolist()),
        "items": sorted(clean["Item"].dropna().astype(str).unique().tolist()),
        "frequency": freq,
        "dimension_schema": dimension_schema,
        "dimension_options": dimension_options(clean, dimension_schema),
        "series_count": series_count(clean, dimension_schema),
        "preprocessing_metrics": cleaning_stats,
        "quality_score": profile["quality"]["score"],
        "profile": profile,
        "numeric_audit": numeric_audit,
        "imputation": imputation_audit,
        "aggregation": aggregation,
        "start_date": _timestamp_text(clean["Date"].min()),
        "end_date": _timestamp_text(clean["Date"].max()),
        "processed_preview": {"preview_type": "processed", **_preview_from_frame(clean.head(10))},
        "preprocessing": preprocessing,
    }


def apply_dataset_mapping(dataset_id, mapping):
    if not dataset_id or dataset_id != current_dataset_id():
        raise ValueError("This dataset is no longer active. Upload it again before mapping.")
    dataset = get_dataset(dataset_id)
    if not dataset:
        raise ValueError("Dataset metadata was not found.")

    active_mapping = _resolve_mapping(dataset, mapping)
    if (
        dataset.get("status") in {"ready_to_train", "training", "completed", "failed"}
        and dataset.get("adapted", {}).get("dataset_id") == dataset_id
        and dataset.get("adapted", {}).get("mapping_hash") == _mapping_hash(active_mapping)
        and dataset.get("adapted", {}).get("source_signature") == _source_signature(active_mapping["csv_file"])
        and dataset.get("adapted", {}).get("artifact_id")
        and dataset.get("adapted", {}).get("preprocessing_config") == _preprocessing_config(active_mapping["target_column"], dataset.get("adapted", {}).get("frequency"))
        and Path(dataset.get("adapted", {}).get("cleaned_path", "")).exists()
    ):
        return {"dataset": dataset, "adapted": dataset["adapted"], "reused": True}

    clear_dataset_outputs()
    adapted = adapt_dataset(active_mapping["csv_file"], active_mapping, dataset_id=dataset_id)
    dataset.update(
        {
            "mapping": active_mapping,
            "adapted": adapted,
            "status": "ready_to_train",
            "message": "Dataset cleaned and converted. Ready to train real forecasting models.",
        }
    )
    append_or_update_dataset(dataset)
    save_json("current_dataset.json", {"dataset_id": dataset_id})
    return {"dataset": dataset, "adapted": adapted}


def create_dataset_record(name, source_file, source_type, original_filename=None):
    source_hash = raw_preview_service.source_file_hash(source_file)
    dataset_id = _dataset_id(source_file, source_type, source_hash)
    analysis = analyze_file(source_file, original_filename=original_filename, dataset_id=dataset_id, source_hash=source_hash)
    mapping = dict(analysis["mapping"])
    mapping["csv_file"] = str(source_file)
    source_stat = Path(source_file).stat()
    dataset = {
        "id": dataset_id,
        "name": name,
        "original_filename": original_filename or Path(source_file).name,
        "source_file_hash": source_hash,
        "raw_artifact_id": analysis["raw_preview"]["raw_artifact_id"],
        "schema_version": raw_preview_service.RAW_SCHEMA_VERSION,
        "upload_timestamp": raw_preview_service.upload_timestamp(),
        "raw_file_size": source_stat.st_size,
        "raw_file_mtime_ns": source_stat.st_mtime_ns,
        "source_type": source_type,
        "path": str(Path(source_file).parent),
        "source_file": str(source_file),
        "csv_files": [str(source_file)],
        "preview_file": str(source_file),
        "raw_preview": analysis["raw_preview"],
        "raw_schema": analysis["raw_schema"],
        "mapping": mapping,
        "suggested_mapping": mapping,
        "requires_mapping": analysis["requires_mapping"],
        "status": "needs_mapping",
        "message": "Confirm mapping before training.",
        "uploaded_at": int(time.time()),
    }
    activate_dataset(dataset_id)
    append_or_update_dataset(dataset)
    save_json("current_dataset.json", {"dataset_id": dataset["id"]})
    return dataset


def get_raw_preview(dataset_id, limit=10):
    if not dataset_id or dataset_id != current_dataset_id():
        raise raw_preview_service.RawPreviewError("Dataset preview is unavailable because this upload is not active.")
    dataset = get_dataset(dataset_id)
    if not dataset:
        raise raw_preview_service.RawPreviewError("Dataset preview metadata is unavailable.")
    source = Path(dataset.get("source_file", "")).resolve()
    upload_root = Path(uploads_path()).resolve()
    try:
        source.relative_to(upload_root)
    except ValueError as exc:
        raise raw_preview_service.RawPreviewError("Raw preview source is outside the upload directory.") from exc
    preview, schema = raw_preview_service.reload_raw_preview(source, dataset, limit=limit)
    stored_schema = dataset.get("raw_schema") or []
    stored_identity = [(column.get("column_id"), column.get("source_name"), column.get("position")) for column in stored_schema]
    current_identity = [(column.get("column_id"), column.get("source_name"), column.get("position")) for column in schema]
    if stored_identity and stored_identity != current_identity:
        raise raw_preview_service.RawPreviewError("Raw schema no longer matches the active upload.")
    return preview


def _mapping_with_column_ids(mapping, raw_schema):
    id_by_internal = {column["internal_name"]: column["column_id"] for column in raw_schema}
    result = dict(mapping)
    result["timestamp_column_id"] = id_by_internal.get(mapping.get("date_column"), "")
    result["target_column_id"] = id_by_internal.get(mapping.get("target_column"), "")
    result["dimension_column_ids"] = [
        id_by_internal[column]
        for column in mapping.get("dimension_columns", [])
        if column in id_by_internal
    ]
    return result


def _resolve_mapping(dataset, mapping):
    schema = dataset.get("raw_schema") or []
    if not schema:
        raise ValueError("Raw schema metadata is unavailable. Upload the dataset again.")
    timestamp_id = mapping.get("timestamp_column_id") or mapping.get("date_column")
    target_id = mapping.get("target_column_id") or mapping.get("target_column")
    dimension_ids = mapping.get("dimension_column_ids") or []
    if isinstance(dimension_ids, str):
        try:
            dimension_ids = json.loads(dimension_ids)
        except (TypeError, ValueError):
            dimension_ids = [value.strip() for value in dimension_ids.split(",") if value.strip()]
    if not dimension_ids:
        dimension_ids = [
            value for value in (
                mapping.get("primary_dimension_column_id") or mapping.get("store_column"),
                mapping.get("secondary_dimension_column_id") or mapping.get("item_column"),
            ) if value
        ]
    exogenous_ids = mapping.get("exogenous_column_ids") or []
    if isinstance(exogenous_ids, str):
        try:
            exogenous_ids = json.loads(exogenous_ids)
        except (TypeError, ValueError):
            exogenous_ids = [value.strip() for value in exogenous_ids.split(",") if value.strip()]
    timestamp = _resolve_schema_column(timestamp_id, schema, required=True, role="timestamp")
    target = _resolve_schema_column(target_id, schema, required=True, role="target")
    dimensions = []
    for value in dimension_ids:
        column = _resolve_schema_column(value, schema, required=False, role="dimension")
        if column and column["column_id"] not in {item["column_id"] for item in dimensions}:
            dimensions.append(column)
    exogenous = []
    reserved_ids = {timestamp["column_id"], target["column_id"], *(column["column_id"] for column in dimensions)}
    for value in exogenous_ids:
        column = _resolve_schema_column(value, schema, required=False, role="exogenous")
        if column and column["column_id"] not in reserved_ids and column["column_id"] not in {item["column_id"] for item in exogenous}:
            exogenous.append(column)
    mapped_roles = {
        "timestamp": _role_metadata(timestamp),
        "target": _role_metadata(target),
        "dimensions": [_role_metadata(column) for column in dimensions],
        "exogenous": [_role_metadata(column) for column in exogenous],
    }
    return {
        "csv_file": dataset.get("source_file"),
        "timestamp_column_id": timestamp["column_id"],
        "target_column_id": target["column_id"],
        "dimension_column_ids": [column["column_id"] for column in dimensions],
        "exogenous_column_ids": [column["column_id"] for column in exogenous],
        "date_column": timestamp["internal_name"],
        "target_column": target["internal_name"],
        "store_column": dimensions[0]["internal_name"] if dimensions else "",
        "item_column": dimensions[1]["internal_name"] if len(dimensions) > 1 else "",
        "dimension_columns": [column["internal_name"] for column in dimensions],
        "mapped_roles": mapped_roles,
    }


def _resolve_schema_column(value, schema, required, role):
    if value in (None, ""):
        if required:
            raise ValueError(f"Select a {role} column from the active raw schema.")
        return None
    direct = next((column for column in schema if column.get("column_id") == value), None)
    if direct:
        return direct
    matches = [
        column for column in schema
        if value in {column.get("internal_name"), column.get("source_name")}
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"The {role} header is duplicated; select it by stable column ID.")
    raise ValueError(f"The selected {role} column does not belong to the active raw schema.")


def _role_metadata(column):
    return {
        "column_id": column["column_id"],
        "source_name": column["source_name"],
        "position": column["position"],
    }


def get_dataset(dataset_id):
    payload = load_json("datasets.json", {"datasets": []})
    for dataset in payload.get("datasets", []):
        if dataset.get("id") == dataset_id:
            return dataset
    return None


def get_active_dataset():
    dataset_id = current_dataset_id()
    if dataset_id:
        dataset = get_dataset(dataset_id)
        if dataset:
            return dataset
    return None


def _dataset_id(source_file, source_type, source_hash=None):
    token = f"{source_type}:{source_hash or Path(source_file).name}:{time.time_ns()}".encode("utf-8")
    return f"{source_type}_{hashlib.sha256(token).hexdigest()[:16]}"


def append_or_update_dataset(dataset):
    payload = load_json("datasets.json", {"datasets": []})
    datasets = [item for item in payload.get("datasets", []) if item.get("id") != dataset.get("id")]
    datasets.append(_json_safe(dataset))
    save_json("datasets.json", {"datasets": datasets})


def _mapping_hash(mapping):
    values = [str(mapping.get(key, "")) for key in ("csv_file", "date_column", "target_column", "store_column", "item_column")]
    values.append(json.dumps(mapping.get("dimension_columns") or [], sort_keys=True, default=str))
    values.extend(str(mapping.get(key, "")) for key in ("timestamp_column_id", "target_column_id"))
    values.append(json.dumps(mapping.get("dimension_column_ids") or [], sort_keys=True, default=str))
    values.append(json.dumps(mapping.get("exogenous_column_ids") or [], sort_keys=True, default=str))
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


def _source_signature(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_tabular(path, nrows=None):
    frame, _headers, _metadata, _total, _exact = raw_preview_service.read_dataframe(path, nrows=nrows)
    return frame


def _preview_from_frame(df):
    rows = []
    for row in df.where(df.notna(), "").to_dict("records"):
        rows.append({str(key): _display_value(value) for key, value in row.items()})
    return {"columns": [str(column) for column in df.columns], "rows": rows}


def _detect_date_column(df, columns):
    scored = []
    for column in columns:
        parsed = _parse_dates_best(df[column])
        ratio = parsed.notna().mean() if len(parsed) else 0
        name_score = 2 if _normalized(column) in DATE_NAME_HINTS else 0
        if "date" in _normalized(column) or "timestamp" in _normalized(column):
            name_score += 2
        scored.append((name_score + ratio, ratio, column))
    scored.sort(reverse=True)
    return (scored[0][2], scored[0][1]) if scored and scored[0][1] > 0 else (None, 0)


def _detect_target_column(df, columns, date_column):
    scored = []
    numeric_candidates = []
    for column in columns:
        if column == date_column:
            continue
        numeric = _numeric_series(df[column])
        ratio = numeric.notna().mean() if len(numeric) else 0
        if ratio >= 0.6:
            numeric_candidates.append(column)
        normalized = _normalized(column)
        name_score = 3 if normalized in TARGET_NAME_HINTS else 0
        if any(hint in normalized for hint in TARGET_NAME_HINTS):
            name_score += 1
        scored.append((name_score + ratio, ratio, column))
    scored.sort(reverse=True)
    return (scored[0][2], scored[0][1], numeric_candidates) if scored and scored[0][1] > 0 else (None, 0, numeric_candidates)


def _parse_dates_best(series, strict=False):
    import pandas as pd

    text = series.astype(str).str.strip()
    iso_ratio = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$", na=False).mean() if len(text) else 0
    if iso_ratio >= 0.8:
        return pd.to_datetime(series, errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = [
            pd.to_datetime(series, errors="coerce"),
            pd.to_datetime(series, errors="coerce", dayfirst=True),
        ]
    if strict and len(series):
        ambiguous = (candidates[0].notna() & candidates[1].notna() & (candidates[0] != candidates[1])).mean()
        if ambiguous > 0.5:
            raise ValueError("Date column contains ambiguous day/month values. Confirm an unambiguous date format before mapping.")
    return max(candidates, key=lambda item: (item.notna().mean(), _monotonic_score(item)))


def _numeric_series(series, return_audit=False, allow_percent=False):
    import pandas as pd

    parsed = []
    malformed = 0
    nulls = {"", "na", "n/a", "nan", "null", "none", "-"}
    for value in series.tolist():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            parsed.append(float("nan"))
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed.append(float(value))
            continue
        text = str(value).strip()
        if text.lower() in nulls:
            parsed.append(float("nan"))
            continue
        negative = text.startswith("(") and text.endswith(")")
        percent = text.endswith("%")
        text = text.strip("()") if negative else text
        text = re.sub(r"^[^0-9+\-.,]+", "", text)
        text = text.replace(",", "").replace(" ", "").replace("$", "").replace("€", "").replace("£", "")
        try:
            number = float(text.rstrip("%"))
            parsed.append((-number if negative else number) / (100 if percent and allow_percent else 1))
        except (TypeError, ValueError):
            malformed += 1
            parsed.append(float("nan"))
    result = pd.Series(parsed, index=series.index, dtype="float64")
    audit = {"parsed_values": int(result.notna().sum()), "missing_values": int(result.isna().sum()), "malformed_values": malformed, "rejected_rows": malformed}
    return (result, audit) if return_audit else result


def _dimension_series(df, column, default):
    if column and column in df.columns:
        return df[column].astype(str)
    return default


def _fill_group_date_gaps(clean, freq, policy=None, dimension_schema=None):
    import pandas as pd

    policy = policy or {"method": "leave_missing", "max_gap": 0}
    if policy.get("method") == "leave_missing":
        return clean, {"method": "leave_missing", "existing_values_imputed": 0, "calendar_rows_generated": 0, "generated_values_filled": 0, "unresolved_generated_rows_removed": 0, "future_dependent": False, "max_gap": 0}
    dimension_columns = [dimension["canonical_column"] for dimension in dimension_schema or []]
    filled = []
    existing_values_imputed = 0
    calendar_rows_generated = 0
    generated_values_filled = 0
    unresolved_generated_rows_removed = 0
    groups = clean.groupby(dimension_columns, dropna=False, sort=False) if dimension_columns else [((), clean)]
    for group_key, group in groups:
        group = group.sort_values("Date")
        group["_source_row"] = True
        full_dates = pd.date_range(group["Date"].min(), group["Date"].max(), freq=freq)
        group = group.set_index("Date").reindex(full_dates)
        group.index.name = "Date"
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        for column, value in zip(dimension_columns, key_values):
            group[column] = value
        group["Click Count"] = pd.to_numeric(group["Click Count"], errors="coerce")
        generated_mask = group["_source_row"].isna()
        existing_missing_mask = ~generated_mask & group["Click Count"].isna()
        calendar_rows_generated += int(generated_mask.sum())
        if policy["method"] == "zero":
            group["Click Count"] = group["Click Count"].fillna(0)
        elif policy["method"] == "forward_fill":
            group["Click Count"] = group["Click Count"].ffill(limit=policy.get("max_gap", 3))
        elif policy["method"] in {"linear", "time"}:
            group["Click Count"] = group["Click Count"].interpolate(limit=policy.get("max_gap", 3), limit_direction="forward")
        existing_values_imputed += int((existing_missing_mask & group["Click Count"].notna()).sum())
        generated_values_filled += int((generated_mask & group["Click Count"].notna()).sum())
        unresolved = generated_mask & group["Click Count"].isna()
        unresolved_generated_rows_removed += int(unresolved.sum())
        group = group.loc[~unresolved].drop(columns=["_source_row"])
        filled.append(group.reset_index())
    sort_columns = ["Date"] + dimension_columns
    result = pd.concat(filled, ignore_index=True).sort_values(sort_columns)
    return result, {"method": policy["method"], "existing_values_imputed": existing_values_imputed, "calendar_rows_generated": calendar_rows_generated, "generated_values_filled": generated_values_filled, "unresolved_generated_rows_removed": unresolved_generated_rows_removed, "future_dependent": False, "max_gap": policy.get("max_gap", 0)}


def _aggregate_clean(clean, method, dimension_schema=None):
    grain = ["Date"] + [dimension["canonical_column"] for dimension in dimension_schema or []]
    grouped = clean.groupby(grain, as_index=False, dropna=False)
    if method == "mean":
        return grouped["Click Count"].mean()
    if method == "median":
        return grouped["Click Count"].median()
    if method == "last":
        return grouped["Click Count"].last()
    if method == "first":
        return grouped["Click Count"].first()
    if method == "min":
        return grouped["Click Count"].min()
    if method == "max":
        return grouped["Click Count"].max()
    if method == "count":
        return grouped["Click Count"].count()
    return grouped["Click Count"].sum()


def _aggregation_method(target_column):
    name = _normalized(target_column)
    configured = os.environ.get("FORECAST_AGGREGATION", "").strip().lower()
    if configured in {"sum", "mean", "median", "min", "max", "first", "last", "count"}:
        return configured
    if any(token in name for token in ("price", "rate", "temperature", "temp", "sensor", "percent")):
        return "mean"
    if any(token in name for token in ("balance", "stock", "inventory", "level")):
        return "last"
    return "sum"


def _imputation_policy(target_column):
    configured = os.environ.get("FORECAST_TARGET_IMPUTATION", "auto").strip().lower()
    if configured in {"zero", "forward_fill", "linear", "time", "leave_missing"}:
        return {"method": configured, "max_gap": int(os.environ.get("FORECAST_IMPUTATION_MAX_GAP", "3"))}
    name = _normalized(target_column)
    if any(token in name for token in ("demand", "sales", "quantity", "count", "orders", "transactions", "events")):
        return {"method": "zero", "max_gap": 0, "reason": "count-like target semantics"}
    if any(token in name for token in ("balance", "stock", "inventory", "level")):
        return {"method": "forward_fill", "max_gap": int(os.environ.get("FORECAST_IMPUTATION_MAX_GAP", "3")), "reason": "state-like target semantics"}
    return {"method": "leave_missing", "max_gap": 0, "reason": "ambiguous target semantics"}


def _preprocessing_config(target_column, frequency):
    return {"aggregation": _aggregation_method(target_column), "frequency": frequency, "imputation": _imputation_policy(target_column), "version": PREPROCESSING_VERSION}


def _infer_frequency(dates):
    import pandas as pd

    unique = pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values()
    inferred = pd.infer_freq(unique) if len(unique) >= 3 else None
    if inferred:
        return inferred
    gaps = unique.diff().dt.total_seconds().dropna()
    if gaps.empty:
        return "D"
    median_gap = gaps.median()
    if median_gap >= 27 * 86400:
        return "MS"
    if median_gap >= 6 * 86400:
        first = unique.min()
        return f"W-{first.day_name()[:3].upper()}"
    if median_gap >= 86400:
        return "D"
    if median_gap >= 3600:
        return "h"
    if median_gap >= 60:
        return "min"
    return "D"


def _first_name_match(columns, hints):
    for column in columns:
        normalized = _normalized(column)
        if any(hint in normalized for hint in hints):
            return column
    return ""


def _normalized(value):
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))


def _monotonic_score(series):
    clean = series.dropna()
    if len(clean) < 2:
        return 0
    return 1 if clean.is_monotonic_increasing else 0


def _display_value(value):
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except TypeError:
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _timestamp_text(value):
    import pandas as pd

    timestamp = pd.Timestamp(value)
    if timestamp.hour == timestamp.minute == timestamp.second == timestamp.microsecond == 0:
        return timestamp.date().isoformat()
    return timestamp.isoformat()
