from __future__ import annotations

import csv
from pathlib import Path

from straintrace_campaign.adapters import discover
from straintrace_campaign.adapters.desy import (
    classify_desy_path,
    discover as discover_desy,
    parse_cbf_header_text,
    parse_fio_text,
    parse_load_table_text,
    parse_time_value,
)
from straintrace_campaign.adapters.generic import _is_excluded, _parse_cell, read_explicit_frame_mapping


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_generic_discovery_keeps_counter_gaps_dedupes_identical_and_quarantines_conflicts(tmp_path: Path) -> None:
    root = tmp_path / "images"
    _write(root / "batch_a" / "DIC" / "dic_000001.tif", "same-pixels")
    _write(root / "batch_a" / "copies" / "duplicate_000001.tif", "same-pixels")
    _write(root / "batch_a" / "DIC" / "dic_000003.tif", "third")
    _write(root / "batch_b" / "left" / "dic_000002.tif", "left")
    _write(root / "batch_b" / "right" / "other_000002.tif", "right")

    records = discover(
        str(root),
        adapter="generic",
        pattern={"image_patterns": ["**/*.tif"], "acquisition_parent_levels": 1},
    )
    by_path = {item["relative_path"]: item for item in records}
    a = by_path["batch_a"]
    b = by_path["batch_b"]

    assert [frame["frame"] for frame in a["image_frames"]] == [1, 3]
    assert a["metadata"]["missing_frame_ranges"] == [[2, 2]]
    assert "DUPLICATE_IDENTICAL_CONTENT" in {issue["code"] for issue in a["discovery_issues"]}
    assert len(a["image_frames"][0]["duplicate_source_paths"]) == 1
    assert b["image_frames"] == []
    conflict = next(issue for issue in b["discovery_issues"] if issue["code"] == "DUPLICATE_NATIVE_COUNTER_CONTENT_CONFLICT")
    assert len(conflict["records"]) == 2
    assert len(b["unmapped_images"]) == 2


def test_generic_frame_metadata_and_measurement_map_join_only_by_named_counters(tmp_path: Path) -> None:
    sample = tmp_path / "root" / "sample"
    _write(sample / "DIC" / "dic_000004.tif", "4")
    _write(sample / "DIC" / "dic_000006.tif", "6")
    with (sample / "frame_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "filename", "frame", "native_frame", "timestamp_unix_s", "estimated_timestamp_unix_s",
            "time_s", "scan_id", "scan_id_is_global_fio", "clock_id",
        ])
        writer.writeheader()
        writer.writerow({
            "filename": "dic_000004.tif", "frame": 40, "native_frame": 4,
            "timestamp_unix_s": 1_700_000_004, "estimated_timestamp_unix_s": "",
            "time_s": "", "scan_id": "", "scan_id_is_global_fio": "false", "clock_id": "",
        })
        writer.writerow({
            "filename": "dic_000006.tif", "frame": 60, "native_frame": 6,
            "timestamp_unix_s": "", "estimated_timestamp_unix_s": 1_700_000_006,
            "time_s": "", "scan_id": "", "scan_id_is_global_fio": "false", "clock_id": "",
        })
    with (sample / "measurement_frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "frame", "native_cbf_frame", "native_dic_frame", "scan_id", "scan_id_is_global_fio",
            "scan_id_namespace", "timestamp_unix_s", "timestamp_status",
        ])
        writer.writeheader()
        writer.writerow({
            "frame": 101, "native_cbf_frame": 9001, "native_dic_frame": 4,
            "scan_id": 88, "scan_id_is_global_fio": "true",
            "scan_id_namespace": "global_fio_scan_id", "timestamp_unix_s": 1_700_000_004,
            "timestamp_status": "RECORDED_ABSOLUTE_EPOCH",
        })
        writer.writerow({
            "frame": 102, "native_cbf_frame": 9003, "native_dic_frame": 6,
            "scan_id": 88, "scan_id_is_global_fio": "true",
            "scan_id_namespace": "global_fio_scan_id", "timestamp_unix_s": 1_700_000_006,
            "timestamp_status": "RECORDED_ABSOLUTE_EPOCH",
        })

    acquisition = discover(
        str(tmp_path / "root"),
        pattern={"image_patterns": ["**/*.tif"], "acquisition_parent_levels": 1},
    )[0]
    assert [(frame["frame"], frame["native_frame"]) for frame in acquisition["image_frames"]] == [(40, 4), (60, 6)]
    assert acquisition["image_frames"][0]["timestamp_unix_s"] == 1_700_000_004
    assert acquisition["image_frames"][1]["timestamp_unix_s"] is None
    assert acquisition["image_frames"][1]["estimated_timestamp_unix_s"] == 1_700_000_006
    assert [(row["frame"], row["native_cbf_frame"]) for row in acquisition["measurement_frames"]] == [
        (101, 9001), (102, 9003),
    ]
    assert all(row["scan_id_is_global_fio"] for row in acquisition["measurement_frames"])


def test_duplicate_explicit_measurement_frames_are_reported_and_not_admitted(tmp_path: Path) -> None:
    path = _write(tmp_path / "mapping.csv", (
        "frame,native_cbf_frame,scan_id,scan_id_is_global_fio\n"
        "8,100,12,true\n"
        "8,101,12,true\n"
        "9,102,,false\n"
    ))
    rows, issues = read_explicit_frame_mapping(path)
    assert [row["frame"] for row in rows] == [9]
    assert rows[0]["native_cbf_frame"] == 102
    assert rows[0]["scan_id"] is None
    assert rows[0]["scan_id_is_global_fio"] is False
    assert next(issue for issue in issues if issue["code"] == "DUPLICATE_MAPPING_CONFLICT")["frame"] == 8


def test_desy_discovery_separates_delun_dic_camera_tiff_and_global_fio(tmp_path: Path) -> None:
    root = tmp_path / "desy"
    _write(root / "Grip2" / "Sample_01" / "DIC" / "dic_000001.tif", "dic")
    camera_dir = root / "DIC_images" / "FIO119" / "1236B_1"
    _write(camera_dir / "dir_000003.tif", "camera")
    _write(root / "DIC_images" / "FIO119" / "eh3scan_900.fio",
           "%d\n Col 1 Varex_1 DOUBLE\n Col 2 timestamp DOUBLE\n 3 1700000000\n")
    cbf_dir = root / "camera_scans" / "scan_x"
    _write(cbf_dir / "cbf_000007.cbf", "cbf")
    _write(cbf_dir / "eh3scan_901.fio",
           "%d\n Col 1 Varex_1 DOUBLE\n Col 2 timestamp DOUBLE\n 7 1700000001\n")

    batches = discover_desy(root)
    by_kind = {(item["source_kind"], item["relative_path"]): item for item in batches}
    delun = by_kind[("desy_delun_dic", "Grip2/Sample_01")]
    camera = by_kind[("desy_camera_dic", "DIC_images/FIO119")]
    cbf = by_kind[("desy_camera_cbf", "camera_scans/scan_x")]

    assert delun["image_frames"][0]["frame_namespace"] == "delun_dic_filename_counter"
    assert delun["measurement_frames"] == []
    assert camera["metadata"]["scan_id_candidate"] == 119
    assert camera["metadata"]["scan_id_is_global_fio"] is True
    assert camera["metadata"]["scan_id"] == 900
    assert cbf["measurement_frames"][0]["native_cbf_frame"] == 7
    assert cbf["measurement_frames"][0]["scan_id"] == 901
    assert cbf["measurement_frames"][0]["scan_id_is_global_fio"] is True
    assert classify_desy_path(root / "Grip2" / "Sample_01" / "DIC" / "dic_1.tif")["source_kind"] == "desy_delun_dic"

    counter_mapped = discover_desy(root, pattern={"fio_counter_column": "Varex_1"})
    cbf_mapped = next(item for item in counter_mapped if item["source_kind"] == "desy_camera_cbf")
    mapped_frame = cbf_mapped["measurement_frames"][0]
    assert mapped_frame["timestamp_unix_s"] == 1_700_000_001
    assert mapped_frame["timestamp_source"] == "FIO_LOG"
    assert mapped_frame["timestamp_basis_namespace"] == "Varex_1"
    assert cbf_mapped["metadata"]["fio_clock_mapping"]["join_method"] == "EXACT_NATIVE_COUNTER_EQUALITY"


def test_fio_and_load_parsers_keep_clock_and_force_units_explicit() -> None:
    fio = parse_fio_text(
        "! scan metadata\n%d\n Col 1 Varex_1 DOUBLE\n Col 2 1236B_1 DOUBLE\n"
        " Col 3 timestamp DOUBLE\n Col 4 deben_tension DOUBLE\n"
        "1 <no-data> 12.5 -100\n<no-data> 2 1700000000.25 None\n",
        source_path="eh3scan_42.fio",
    )
    assert fio["issues"] == []
    assert fio["scan_id"] == 42 and fio["scan_id_is_global_fio"] is True
    assert fio["rows"][0]["Varex_1"] == 1
    assert fio["rows"][0]["1236B_1"] is None
    assert parse_time_value(fio["rows"][0]["timestamp"])["timestamp_status"] == "RECORDED_RELATIVE_CLOCK"
    assert parse_time_value(fio["rows"][1]["timestamp"])["timestamp_status"] == "RECORDED_ABSOLUTE_EPOCH"

    parsed = parse_load_table_text(
        "time,load,unit\n2026-10-15T10:00:00+02:00,2,kN\n2026-10-15T10:00:01+02:00,3,lbf?\n",
        time_column="time", force_column="load",
    )
    assert parsed["rows"][0]["timestamp_status"] == "RECORDED_ABSOLUTE_TIMEZONE_AWARE"
    assert parsed["rows"][0]["force_N"] == 2000
    assert parsed["rows"][1]["force_N"] is None
    assert parsed["rows"][1]["force_status"] == "UNKNOWN_FORCE_UNIT"

    naive = parse_time_value("2026-10-15T10:00:00")
    assert naive["timestamp_unix_s"] is None
    assert naive["timestamp_status"] == "NAIVE_DATETIME_TIMEZONE_UNKNOWN"


def test_cbf_header_time_requires_explicit_timezone_and_no_interval_is_inferred() -> None:
    aware = parse_cbf_header_text("# timestamp: 2026-10-15T10:00:00+02:00\n# Exposure_time: 0.25\n")
    assert aware["timestamp_status"] == "RECORDED_ABSOLUTE_TIMEZONE_AWARE"
    assert aware["timestamp_unix_s"] is not None
    assert aware["exposure_s"] == 0.25
    absent = parse_cbf_header_text("# Detector: synthetic\n")
    assert absent["timestamp_unix_s"] is None
    assert absent["exposure_s"] is None


def test_parse_cell_preserves_newton_unit_and_rejects_nonfinite() -> None:
    assert _parse_cell("N") == "N"
    assert _parse_cell("n") == "n"
    assert _parse_cell("y") == "y"
    assert _parse_cell("true") is True
    assert _parse_cell("false") is False
    assert _parse_cell("inf") is None
    assert _parse_cell("-inf") is None
    assert _parse_cell("infinity") is None
    assert _parse_cell("+nan") is None


def test_generic_mapping_metadata_rejects_booleans_and_matches_relative_paths(tmp_path: Path) -> None:
    mapping_csv = _write(
        tmp_path / "mapping.csv",
        "frame,native_cbf_frame,scan_id,scan_id_is_global_fio,timestamp_unix_s\n"
        "true,10,5,true,1700000000\n"
        "1,false,5,true,1700000000\n"
        "2,20,5,maybe,not_a_number\n",
    )
    rows, issues = read_explicit_frame_mapping(mapping_csv)
    assert [row["frame"] for row in rows] == [2]
    assert rows[0]["timestamp_unix_s"] is None
    assert rows[0]["scan_id_is_global_fio"] is False
    codes = {issue["code"] for issue in issues}
    assert {"MAPPING_FRAME_ID_MISSING", "MAPPING_NATIVE_CBF_COUNTER_INVALID", "MAPPING_TIMESTAMP_INVALID", "MAPPING_SCAN_ID_FLAG_INVALID"}.issubset(codes)

    root = tmp_path / "rel_match"
    _write(root / "sample" / "DIC" / "dic_000001.tif", "one")
    _write(root / "sample" / "DIC" / "dic_000002.tif", "two")
    _write(root / "sample" / "other" / "dic_000002.tif", "conflict-two")
    _write(
        root / "sample" / "frame_metadata.csv",
        "filename,timestamp_unix_s\nDIC/dic_000001.tif,1700000001\n",
    )
    batch = discover(str(root), pattern={"image_patterns": ["**/*.tif"], "acquisition_parent_levels": 1})[0]
    assert len(batch["image_frames"]) == 1
    assert batch["image_frames"][0]["timestamp_unix_s"] == 1_700_000_001
    issue_codes = {issue["code"] for issue in batch["discovery_issues"]}
    assert "DUPLICATE_NATIVE_COUNTER_CONTENT_CONFLICT" in issue_codes
    assert "UNPARSED_NATIVE_COUNTER_FILES" not in issue_codes
    assert _is_excluded(tmp_path / "outside.tif", root, ()) is True


def test_desy_discovery_ignores_ancestors_handles_root_brackets_mixed_kinds_and_fio_csv(tmp_path: Path) -> None:
    # Ancestor directories named DIC and FIO999 must not affect classification or scan_id_candidate.
    root = tmp_path / "DIC" / "FIO999" / "campaign_root"
    _write(root / "plain_000001.tif", "root-img")
    _write(root / "mixed" / "sample[1]_000002.tif", "bracket-tif")
    _write(root / "mixed" / "cam_000002.cbf", "cam-cbf")
    _write(root / "mixed" / "eh3scan_77.fio", "%d\n Col 1 Varex_1 DOUBLE\n Col 2 timestamp DOUBLE\n 2 1700000002\n")
    _write(
        root / "mixed" / "measurement_frames.csv",
        "frame,native_cbf_frame,scan_id,scan_id_is_global_fio\n2,2,77,true\n",
    )

    batches = discover_desy(root)
    rel_paths = [item["relative_path"] for item in batches]
    assert "." not in rel_paths
    assert len(set(rel_paths)) == len(rel_paths)

    root_batch = next(item for item in batches if item["relative_path"] == "_root_sequence")
    assert root_batch["source_kind"] == "desy_other_image"
    assert root_batch["metadata"]["scan_id_candidate"] is None

    mixed_tif = next(item for item in batches if item["relative_path"] == "mixed__desy_other_image")
    assert [f["filename"] for f in mixed_tif["image_frames"]] == ["sample[1]_000002.tif"]

    mixed_cbf = next(item for item in batches if item["relative_path"] == "mixed__desy_camera_cbf")
    assert [row["frame"] for row in mixed_cbf["measurement_frames"]] == [2]


def test_fio_iso_duplicate_columns_quoted_load_csv_and_space_cbf_headers() -> None:
    fio = parse_fio_text(
        "%d\n Col 1 Varex_1 DOUBLE\n Col 2 Varex_1 DOUBLE\n Col 3 timestamp STRING\n"
        " 1 10 2026-10-15T10:00:00+02:00\n",
        source_path="eh3scan_5.fio",
    )
    assert "FIO_DUPLICATE_COLUMN_NAMES" in {issue["code"] for issue in fio["issues"]}
    assert "FIO_NONNUMERIC_VALUE" not in {issue["code"] for issue in fio["issues"]}
    assert fio["rows"][0]["Varex_1"] == 1
    assert fio["rows"][0]["Varex_1__col2"] == 10
    assert fio["rows"][0]["timestamp"] == "2026-10-15T10:00:00+02:00"

    load_parsed = parse_load_table_text(
        'time,force_N,note\n1700000000,150,"line1\nline2"\n',
        time_column="time",
        force_column="force_N",
    )
    assert len(load_parsed["rows"]) == 1
    assert load_parsed["rows"][0]["force_N"] == 150.0

    cbf_s = parse_cbf_header_text("# timestamp 2026-10-15T10:00:00+02:00\n# Exposure_time 0.5 s\n")
    assert cbf_s["timestamp_status"] == "RECORDED_ABSOLUTE_TIMEZONE_AWARE"
    assert cbf_s["exposure_s"] == 0.5
    cbf_ms = parse_cbf_header_text("# Exposure_time 250 ms\n")
    assert cbf_ms["exposure_s"] == 0.25
