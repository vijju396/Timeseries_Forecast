import hashlib
import json
import re


SERIES_KEY_VERSION = "series-key-v1"
AGGREGATE_SERIES_KEY = {}


def build_dimension_schema(mapping, available_columns):
    """Return stable mapped dimension IDs without coupling runtime logic to source names."""
    available = {str(column) for column in available_columns}
    configured = mapping.get("dimension_columns") or []
    if isinstance(configured, str):
        try:
            configured = json.loads(configured)
        except (TypeError, ValueError):
            configured = [value.strip() for value in configured.split(",") if value.strip()]
    if isinstance(configured, dict):
        configured = list(configured.values())
    legacy = [mapping.get("store_column"), mapping.get("item_column")]
    source_columns = []
    for value in list(configured) + legacy:
        source = value.get("source_column") if isinstance(value, dict) else value
        if source and str(source) in available and str(source) not in source_columns:
            source_columns.append(str(source))
    return [
        {
            "id": f"dimension_{index + 1}",
            "source_column": source,
            "display_name": source,
            "canonical_column": f"dimension_{index + 1}",
        }
        for index, source in enumerate(source_columns)
    ]


def canonical_series_key(values, dimension_schema):
    key = {}
    values = values or {}
    for dimension in dimension_schema or []:
        dimension_id = dimension["id"]
        value = values.get(dimension_id)
        if value is None:
            value = values.get(dimension.get("canonical_column"))
        if value is None:
            value = values.get(dimension.get("source_column"))
        if value is not None and not is_all_value(value):
            key[dimension_id] = str(value)
    return {key: key_value for key, key_value in sorted(key.items())}


def series_key_hash(series_key):
    canonical = json.dumps(series_key or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(f"{SERIES_KEY_VERSION}|{canonical}".encode("utf-8")).hexdigest()[:24]


def aggregate_series_identity():
    return {"series_key": {}, "series_key_hash": series_key_hash({})}


def row_series_identity(row, dimension_schema):
    key = canonical_series_key(row, dimension_schema)
    return {"series_key": key, "series_key_hash": series_key_hash(key)}


def canonical_dimension_filters(filters, dimension_schema):
    filters = filters or {}
    aliases = {}
    for index, dimension in enumerate(dimension_schema or []):
        aliases[dimension["id"]] = dimension["id"]
        aliases[dimension.get("canonical_column")] = dimension["id"]
        aliases[dimension.get("source_column")] = dimension["id"]
        aliases[dimension.get("display_name")] = dimension["id"]
        # Compatibility for the two existing filter controls; artifacts never store these labels.
        if index == 0:
            aliases["store"] = dimension["id"]
        elif index == 1:
            aliases["item"] = dimension["id"]
    canonical = {}
    for key, value in filters.items():
        dimension_id = aliases.get(key)
        if dimension_id and value not in (None, "") and not is_all_value(value):
            canonical[dimension_id] = str(value)
    return {key: canonical[key] for key in sorted(canonical)}


def is_all_value(value):
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return not text or text == "all" or text.startswith("all ") or text in {"enterprise rollup", "__all__"}


def dimension_options(frame, dimension_schema):
    options = {}
    for dimension in dimension_schema or []:
        column = dimension["canonical_column"]
        if column in frame:
            options[dimension["id"]] = sorted(frame[column].dropna().astype(str).unique().tolist())
    return options


def series_count(frame, dimension_schema):
    columns = [dimension["canonical_column"] for dimension in dimension_schema or [] if dimension["canonical_column"] in frame]
    return int(frame[columns].drop_duplicates().shape[0]) if columns else (1 if len(frame) else 0)
