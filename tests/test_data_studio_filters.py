import json
import tempfile
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.enrichment_service import MISSING_DIMENSION_TOKEN, build_data_studio_analytics


def _dataset(path, dimensions):
    return {"id": "dataset-test", "adapted": {"dataset_id": "dataset-test", "artifact_id": "artifact-test", "cleaned_path": str(path), "dimension_schema": dimensions}}


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cleaned.csv"
        pd.DataFrame([
            {"Date": "2024-01-01", "Click Count": 10, "dimension_1": "Asia-Pacific", "dimension_2": "India"},
            {"Date": "2024-01-02", "Click Count": 20, "dimension_1": "Asia-Pacific", "dimension_2": "Japan"},
            {"Date": "2024-01-01", "Click Count": 40, "dimension_1": "Africa", "dimension_2": "South Africa"},
        ]).to_csv(path, index=False)
        dimensions = [
            {"id": "dimension_1", "canonical_column": "dimension_1", "source_column": "Geography", "display_name": "Region"},
            {"id": "dimension_2", "canonical_column": "dimension_2", "source_column": "Nation", "display_name": "Country"},
        ]
        dataset = _dataset(path, dimensions)
        aggregate = build_data_studio_analytics(dataset, {})
        asia = build_data_studio_analytics(dataset, {"dimensions": json.dumps({"dimension_1": "Asia-Pacific"})})
        india = build_data_studio_analytics(dataset, {"dimensions": json.dumps({"dimension_1": "Asia-Pacific", "dimension_2": "India"})})
        africa = build_data_studio_analytics(dataset, {"dimensions": json.dumps({"dimension_1": "Africa", "dimension_2": "South Africa"})})
        assert aggregate["matched_row_count"] == 3 and aggregate["matched_series_count"] == 3
        assert [item["label"] for item in asia["dimension_schema"][1]["values"]] == ["India", "Japan"]
        assert india["matched_row_count"] == 1 and india["analytics"]["trend"][0]["value"] == 10
        assert africa["matched_row_count"] == 1 and africa["analytics"]["trend"][0]["value"] == 40
        assert len({aggregate["cache_key"], asia["cache_key"], india["cache_key"], africa["cache_key"]}) == 4
        assert india["requested_dimension_filters"] == india["resolved_dimension_filters"]
        assert "preprocessing_metrics" not in india and "summary" not in india

        no_dimensions = build_data_studio_analytics(_dataset(path, []), {})
        assert no_dimensions["dimension_schema"] == [] and no_dimensions["matched_row_count"] == 3

        one_dimension = build_data_studio_analytics(_dataset(path, dimensions[:1]), {"dimensions": json.dumps({"dimension_1": "Africa"})})
        assert len(one_dimension["dimension_schema"]) == 1 and one_dimension["matched_row_count"] == 1
        assert one_dimension["analytics"]["top_dimensions"] == [{
            "id": "dimension_1", "display_name": "Region",
            "values": [{"label": "Africa", "value": 40.0}],
        }]

        invalid = build_data_studio_analytics(dataset, {"dimensions": json.dumps({"dimension_1": "Asia-Pacific", "dimension_2": "South Africa"})})
        assert invalid["available"] is False and invalid["code"] == "no_data" and invalid["analytics"]["trend"] == []

        typed_path = Path(directory) / "typed.csv"
        pd.DataFrame([
            {"Date": "2024-02-01", "Click Count": 1, "dimension_1": 10, "dimension_2": "München"},
            {"Date": "2024-02-02", "Click Count": 2, "dimension_1": 10, "dimension_2": "東京"},
            {"Date": "2024-02-03", "Click Count": 3, "dimension_1": 20, "dimension_2": None},
        ]).to_csv(typed_path, index=False)
        typed_dataset = _dataset(typed_path, dimensions)
        numeric = build_data_studio_analytics(typed_dataset, {"dimensions": json.dumps({"dimension_1": "10"})})
        missing = build_data_studio_analytics(typed_dataset, {"dimensions": json.dumps({"dimension_1": "20", "dimension_2": MISSING_DIMENSION_TOKEN})})
        assert [item["label"] for item in numeric["dimension_schema"][1]["values"]] == ["München", "東京"]
        assert missing["matched_row_count"] == 1 and missing["analytics"]["trend"][0]["value"] == 3
        assert missing["analytics"]["top_dimensions"][1]["values"] == [{"label": "(Missing)", "value": 3.0}]
    print("data studio filter tests passed: 13")


if __name__ == "__main__":
    main()
