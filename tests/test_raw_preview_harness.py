"""Raw upload preview ownership and schema-preservation harness."""
import io
import sys
import tempfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = PROJECT / "venv" / "Lib" / "site-packages"
sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(PROJECT))


def _upload(client, filename, content, expected_status=200):
    response = client.post(
        "/dataset/upload",
        data={"dataset_file": (io.BytesIO(content), filename)},
        content_type="multipart/form-data",
    )
    assert response.status_code == expected_status, response.get_json()
    return response.get_json()


def _preview(client, dataset, expected_status=200):
    response = client.get(f"/api/dataset-preview?dataset_id={dataset['id']}&limit=10")
    assert response.status_code == expected_status, response.get_json()
    return response.get_json()


def _headers(preview):
    return [column["display_name"] for column in preview["columns"]]


def _column_id(dataset, source_name, occurrence=0):
    matches = [column for column in dataset["raw_schema"] if column["source_name"] == source_name]
    return matches[occurrence]["column_id"]


def _assert_owned_raw(preview, dataset, expected_headers):
    assert preview["ok"] is True and preview["preview_type"] == "raw"
    assert preview["dataset_id"] == dataset["id"]
    assert preview["raw_artifact_id"] == dataset["raw_artifact_id"]
    assert preview["source_file_hash"] == dataset["source_file_hash"]
    assert preview["original_filename"] == dataset["original_filename"]
    assert preview["schema_version"] == dataset["schema_version"]
    assert _headers(preview) == expected_headers
    assert [column["position"] for column in preview["columns"]] == list(range(len(expected_headers)))
    assert all(column["column_id"] == f"col_{index}" for index, column in enumerate(preview["columns"]))
    assert "cleaned_training_input" not in str(preview)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        import services.data_service as data_service

        data_service.BASE_DIR = root
        data_service.DATA_DIR = root / "Data"
        data_service.PREPROCESSED_DIR = data_service.DATA_DIR / "Preprocessed"
        data_service.UPLOADS_DIR = data_service.DATA_DIR / "uploads"
        data_service.RUNTIME_DATA_DIR = data_service.DATA_DIR / "runtime"
        data_service.ensure_data_directories()

        import app

        client = app.app.test_client()

        workforce_headers = ["event_time", "site_code", "business_unit", "geography", "required_staff", "actual_staff"]
        workforce_csv = (
            ",".join(workforce_headers) + "\n"
            "2025-01-01,S-01,Operations,North,12,10\n"
            "2025-01-02,S-01,Operations,North,13,12\n"
            "2025-01-03,S-01,Operations,North,14,13\n"
            "2025-01-04,S-01,Operations,North,15,15\n"
        ).encode("utf-8")
        workforce = _upload(client, "workforce.csv", workforce_csv)["dataset"]
        workforce_preview = _preview(client, workforce)
        _assert_owned_raw(workforce_preview, workforce, workforce_headers)
        assert workforce_preview["total_raw_rows"] == 4 and workforce_preview["row_count_exact"] is True

        mapping_response = client.post("/dataset/map", json={
            "dataset_id": workforce["id"],
            "timestamp_column_id": _column_id(workforce, "event_time"),
            "target_column_id": _column_id(workforce, "required_staff"),
            "dimension_column_ids": [
                _column_id(workforce, "site_code"),
                _column_id(workforce, "business_unit"),
            ],
            "exogenous_column_ids": [_column_id(workforce, "actual_staff")],
        })
        assert mapping_response.status_code == 200, mapping_response.get_json()
        mapped_payload = mapping_response.get_json()
        assert "preview" not in mapped_payload["adapted"]
        assert mapped_payload["adapted"]["processed_preview"]["preview_type"] == "processed"
        mapped_preview = _preview(client, workforce)
        assert mapped_preview["columns"] == workforce_preview["columns"]
        assert mapped_preview["rows"] == workforce_preview["rows"]
        assert mapped_payload["dataset"]["mapping"]["mapped_roles"]["timestamp"]["source_name"] == "event_time"
        assert mapped_payload["dataset"]["mapping"]["mapped_roles"]["exogenous"][0]["source_name"] == "actual_staff"

        schemas = [
            (
                "retail.txt",
                ["period", "branch", "product", "quantity", "unit_price"],
                "period;branch;product;quantity;unit_price\n2025-01-01;B1;P1;2;4.50\n2025-01-02;B2;P2;3;8.25\n".encode("utf-8"),
            ),
            (
                "sensor.tsv",
                ["observed_at", "device", "temperature", "pressure", "status"],
                "observed_at\tdevice\ttemperature\tpressure\tstatus\n2025-01-01 00:00\tD1\t21.4\t1012.2\tok\n2025-01-01 01:00\tD1\t21.8\t1011.7\t\n".encode("utf-8"),
            ),
            (
                "finance.csv",
                ["fiscal_period", "account", "debit", "credit", "currency_code"],
                "fiscal_period,account,debit,credit,currency_code\n2025-Q1,A-1,1200.00,0,EUR\n2025-Q2,A-2,0,800.00,GBP\n".encode("utf-8"),
            ),
            ("single.csv", ["only one"], "only one\nalpha\nbeta\n".encode("utf-8")),
        ]
        previous_headers = set(workforce_headers)
        for filename, expected_headers, content in schemas:
            dataset = _upload(client, filename, content)["dataset"]
            preview = _preview(client, dataset)
            _assert_owned_raw(preview, dataset, expected_headers)
            assert not previous_headers.intersection(_headers(preview))
            if filename == "sensor.tsv":
                type_lookup = {column["source_name"]: column["physical_type"] for column in preview["columns"]}
                assert type_lookup["observed_at"] == "datetime" and type_lookup["temperature"] == "number"
                assert preview["rows"][1][_column_id(dataset, "status")] is None
            previous_headers = set(expected_headers)

        import pandas as pd

        excel_buffer = io.BytesIO()
        excel_headers = ["as_of", "desk", "exposure"]
        pd.DataFrame([["2025-01-01", "D-1", 125.5]], columns=excel_headers).to_excel(excel_buffer, index=False)
        excel = _upload(client, "positions.xlsx", excel_buffer.getvalue())["dataset"]
        _assert_owned_raw(_preview(client, excel), excel, excel_headers)

        many_headers = [f"field {index}" for index in range(30)]
        many = _upload(client, "many.csv", (",".join(many_headers) + "\n" + ",".join(str(index) for index in range(30)) + "\n").encode())["dataset"]
        assert _headers(_preview(client, many)) == many_headers

        duplicate = _upload(client, "duplicates.csv", b"Amount,Amount,When\n10,20,2025-01-01\n30,40,2025-01-02\n50,60,2025-01-03\n")["dataset"]
        duplicate_preview = _preview(client, duplicate)
        assert _headers(duplicate_preview) == ["Amount", "Amount", "When"]
        assert duplicate_preview["columns"][0]["column_id"] != duplicate_preview["columns"][1]["column_id"]
        assert all(column["duplicate_source_name"] for column in duplicate_preview["columns"][:2])
        assert any("Duplicate source headers" in warning for warning in duplicate_preview["warnings"])
        duplicate_mapping = client.post("/dataset/map", json={
            "dataset_id": duplicate["id"],
            "timestamp_column_id": _column_id(duplicate, "When"),
            "target_column_id": _column_id(duplicate, "Amount", occurrence=1),
        })
        assert duplicate_mapping.status_code == 200, duplicate_mapping.get_json()
        assert duplicate_mapping.get_json()["dataset"]["mapping"]["mapped_roles"]["target"]["position"] == 1
        assert _headers(_preview(client, duplicate)) == ["Amount", "Amount", "When"]

        unusual_headers = [" Event Time ", "Ünit/Code", "Mixed.Case", "value (%)"]
        unusual_content = (",".join(unusual_headers) + "\n2025-01-01,Å-1,KeepCase,12.5\n").encode("utf-8-sig")
        unusual = _upload(client, "üñíçødé.csv", unusual_content)["dataset"]
        unusual_preview = _preview(client, unusual)
        assert _headers(unusual_preview) == unusual_headers

        large_content = "timestamp,value\n" + "\n".join(f"2025-01-{(index % 28) + 1:02d},{index}" for index in range(500))
        large = _upload(client, "bounded.csv", large_content.encode("utf-8"))["dataset"]
        large_preview = _preview(client, large)
        assert large_preview["preview_row_count"] == 10 and len(large_preview["rows"]) == 10
        assert large_preview["total_raw_rows"] == 500 and large_preview["row_count_exact"] is True

        first = _upload(client, "same-name.csv", workforce_csv)["dataset"]
        first_preview = _preview(client, first)
        second = _upload(client, "same-name.csv", schemas[0][2])["dataset"]
        second_preview = _preview(client, second)
        assert first["id"] != second["id"] and first["raw_artifact_id"] != second["raw_artifact_id"]
        assert _headers(first_preview) == workforce_headers
        assert _headers(second_preview) == schemas[0][1]
        assert client.get(f"/api/dataset-preview?dataset_id={first['id']}").status_code == 400

        failed = _upload(client, "broken.csv", b"header\x00binary\nvalue", expected_status=400)
        assert failed["ok"] is False
        unavailable = client.get(f"/api/dataset-preview?dataset_id={second['id']}")
        assert unavailable.status_code == 404
        assert unavailable.get_json()["code"] == "dataset_unavailable"
        assert client.get("/api/current-dataset").get_json()["available"] is False

        dataset_template = (PROJECT / "templates" / "dataset.html").read_text(encoding="utf-8")
        dataset_script = (PROJECT / "static" / "js" / "dataset.js").read_text(encoding="utf-8")
        assert 'id="previewSurface"' in dataset_template and 'class="table-wrap"' in dataset_template
        assert "preview?.columns" in dataset_script and "column.column_id" in dataset_script
        assert "data.adapted.preview" not in dataset_script
        assert "stale dataset preview response was rejected" in dataset_script
        assert "uploadSequence" in dataset_script and "AbortController" in dataset_script

    print("raw preview harness passed: 12 schema/ownership scenarios")


if __name__ == "__main__":
    main()
