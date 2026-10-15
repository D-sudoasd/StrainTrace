from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from straintrace_campaign import alignment
from straintrace_campaign.models import CampaignError
from straintrace_campaign.storage import read_jsonl, relative_artifact, write_jsonl_atomic


T0 = 1_700_000_000.0


def _load(path: Path, rows: list[dict], columns: list[str] | None = None) -> Path:
    fields = columns or list(rows[0])
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join("" if row.get(field) is None else str(row[field]) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _dic_row(frame: int, time: float, values: tuple[float, float, float]) -> dict:
    return {
        "frame": frame,
        "native_dic_frame": frame,
        "timestamp_unix_s": time,
        "timestamp_source": "TEST_RECORDED_TIME",
        "timestamp_status": "RECORDED_ABSOLUTE_EPOCH",
        "valid_gauge_count": 3,
        "quality_status": "PASS",
        "G01_engineering_strain": values[0],
        "G02_engineering_strain": values[1],
        "G03_engineering_strain": values[2],
        "engineering_strain_mean": sum(values) / 3,
        "true_strain_mean": math.log1p(sum(values) / 3),
    }


def _acquisition(tmp_path: Path, dic_rows: list[dict], measurement_rows: list[dict] | None = None,
                 *, dimensions: dict | None = None, status: str = "COMPLETE") -> dict:
    source = tmp_path / "source.jsonl"
    write_jsonl_atomic(source, dic_rows)
    return {
        "acquisition_id": "acq-test-001",
        "name": "synthetic",
        "relative_path": "synthetic",
        "source_kind": "generic_image_sequence",
        "physical_specimen_id": None,
        "dimensions": dimensions or {
            "area_mm2": 5.0, "width_mm": None, "thickness_mm": None,
            "status": "confirmed", "source": "synthetic recorded area",
        },
        "analysis_mode": "extensometer",
        "image_frames": [],
        "measurement_frames": measurement_rows or [],
        "result": {
            "status": status,
            "fingerprint": "source-fingerprint-001",
            "rows_path": relative_artifact(tmp_path, source),
            "artifacts": [relative_artifact(tmp_path, source)],
            "checksums": {},
            "provenance": {},
            "summary": {},
        },
        "alignment": None,
        "roi": None,
        "approval": None,
        "analysis_config": {},
        "metadata": {},
    }


def _read_aligned(project: Path, record: dict) -> list[dict]:
    return read_jsonl(project / record["rows_path"])


def test_instantaneous_alignment_uses_explicit_frames_and_bounded_support(tmp_path: Path) -> None:
    acquisition = _acquisition(
        tmp_path,
        [
            _dic_row(1, T0 + 0.5, (0.01, 0.02, 0.03)),
            _dic_row(2, T0 + 1.5, (0.03, 0.04, 0.05)),
        ],
        measurement_rows=[{
            "frame": 80,
            "native_cbf_frame": 436,
            "scan_id": 123,
            "scan_id_is_global_fio": True,
            "scan_id_namespace": "global_fio_scan_id",
            "timestamp_unix_s": T0 + 1,
            "timestamp_status": "RECORDED_ABSOLUTE_EPOCH",
            "timestamp_source": "synthetic CBF header",
        }],
    )
    load_path = _load(tmp_path / "force.csv", [
        {"timestamp_unix_s": T0, "force_N": 100},
        {"timestamp_unix_s": T0 + 2, "force_N": 300},
    ])

    record = alignment.align(acquisition, tmp_path, load_path, max_gap_s=3)
    row = _read_aligned(tmp_path, record)[0]

    assert row["frame"] == 80
    assert row["native_cbf_frame"] == 436
    assert row["scan_id"] == 123 and row["scan_id_is_global_fio"] is True
    assert row["force_N"] == pytest.approx(200)
    assert row["stress_MPa"] == pytest.approx(40)
    assert row["force_interpolation_status"] == "LINEAR_TIME_INTERPOLATION"
    assert row["force_support_indices"] == [0, 1]
    assert row["force_support_weights"] == pytest.approx([0.5, 0.5])
    assert row["dic_support_status"] == "DIC_BOUNDED_INTERPOLATION"
    assert row["dic_source_frames"] == [1, 2]
    assert row["engineering_strain"] == pytest.approx(0.03)
    assert row["true_strain"] == pytest.approx(math.log1p(0.03))
    assert row["valid_gauge_count"] == 3
    assert row["engineering_strain_sd"] == pytest.approx(0.01)
    assert record["checksums"][record["rows_path"]]


def test_invalid_load_row_is_barrier_and_extrapolation_is_missing(tmp_path: Path) -> None:
    acquisition = _acquisition(tmp_path, [], [
        {"frame": 1, "native_cbf_frame": 1, "scan_id": None, "scan_id_is_global_fio": False,
         "timestamp_unix_s": T0 + 0.5},
        {"frame": 2, "native_cbf_frame": 2, "scan_id": None, "scan_id_is_global_fio": False,
         "timestamp_unix_s": T0 - 1},
    ])
    load_path = _load(tmp_path / "force.csv", [
        {"timestamp_unix_s": T0, "force_N": 10, "is_force_valid": True},
        {"timestamp_unix_s": T0 + 1, "force_N": 20, "is_force_valid": False},
        {"timestamp_unix_s": T0 + 2, "force_N": 30, "is_force_valid": True},
    ])

    record = alignment.align(acquisition, tmp_path, load_path, max_gap_s=3)
    rows = _read_aligned(tmp_path, record)

    assert rows[0]["force_N"] is None
    assert rows[0]["force_interpolation_status"] == "INVALID_BRACKET"
    assert rows[1]["force_N"] is None
    assert rows[1]["force_interpolation_status"] == "OUTSIDE_TIME_RANGE"


def test_window_mean_integrates_piecewise_linear_force_and_records_weights(tmp_path: Path) -> None:
    acquisition = _acquisition(tmp_path, [], [{
        "frame": 7,
        "native_cbf_frame": 77,
        "scan_id": None,
        "scan_id_is_global_fio": False,
        "timestamp_unix_s": T0 + 1,
        "exposure_start_unix_s": T0 + 0.5,
        "exposure_end_unix_s": T0 + 1.5,
    }])
    load_path = _load(tmp_path / "force.csv", [
        {"timestamp_unix_s": T0, "force_N": 0},
        {"timestamp_unix_s": T0 + 1, "force_N": 10},
        {"timestamp_unix_s": T0 + 2, "force_N": 20},
    ])

    record = alignment.align(acquisition, tmp_path, load_path, mode="window_mean", max_gap_s=2)
    row = _read_aligned(tmp_path, record)[0]

    assert row["force_N"] == pytest.approx(10)
    assert row["force_interpolation_status"] == "EXPOSURE_WINDOW_MEAN"
    assert row["force_support_indices"] == [0, 1, 2]
    assert row["force_support_weights"] == pytest.approx([0.125, 0.75, 0.125])
    assert row["stress_MPa"] == pytest.approx(2)


def test_estimated_time_is_not_used_until_selected_and_relative_clocks_must_match(tmp_path: Path) -> None:
    dic = [{
        "frame": 1, "native_dic_frame": 1, "estimated_timestamp_unix_s": T0,
        "timestamp_unix_s": None, "timestamp_status": "ESTIMATED_FROM_PROTOCOL",
        "engineering_strain": 0.01, "valid_gauge_count": 3,
    }]
    acquisition = _acquisition(tmp_path, dic, status="COMPLETE")
    load_path = _load(tmp_path / "force.csv", [{"timestamp_unix_s": T0, "force_N": 100}])

    default_record = alignment.align(acquisition, tmp_path, load_path)
    default_row = _read_aligned(tmp_path, default_record)[0]
    assert default_row["force_N"] is None
    assert default_row["force_interpolation_status"] == "RECORDED_TIMESTAMP_MISSING_ESTIMATE_SEPARATE"

    explicit_record = alignment.align(
        acquisition, tmp_path, load_path,
        measurement_time_field="estimated_timestamp_unix_s",
        load_time_column="timestamp_unix_s",
    )
    explicit_row = _read_aligned(tmp_path, explicit_record)[0]
    assert explicit_row["force_N"] == pytest.approx(100)
    assert explicit_record["provenance"]["time_column_selection_is_explicit"] is True

    relative_acquisition = _acquisition(tmp_path, [], [{
        "frame": 1, "native_cbf_frame": 1, "scan_id": None, "scan_id_is_global_fio": False,
        "time_s": 1.0, "clock_id": "camera-clock",
    }])
    relative_load = _load(tmp_path / "relative.csv", [
        {"time_s": 0.0, "clock_id": "load-clock", "force_N": 10},
        {"time_s": 2.0, "clock_id": "load-clock", "force_N": 20},
    ])
    relative_record = alignment.align(
        relative_acquisition, tmp_path, relative_load,
        time_column="time_s", load_time_column="time_s",
    )
    relative_row = _read_aligned(tmp_path, relative_record)[0]
    assert relative_row["force_N"] is None
    assert relative_row["force_interpolation_status"] == "RELATIVE_CLOCK_DOMAIN_MISMATCH"


def test_candidate_area_never_enters_new_stress_and_unknown_force_units_stay_pending(tmp_path: Path) -> None:
    acquisition = _acquisition(
        tmp_path,
        [],
        [{
            "frame": 1, "native_cbf_frame": 1, "scan_id": None, "scan_id_is_global_fio": False,
            "timestamp_unix_s": T0,
        }],
        dimensions={
            "area_mm2": 5.0, "width_mm": 2.0, "thickness_mm": 2.5,
            "status": "candidate", "source": "unconfirmed notes",
        },
    )
    load_path = _load(tmp_path / "force.csv", [{"timestamp_unix_s": T0, "load": 100, "unit": "lbf?"}])
    record = alignment.align(acquisition, tmp_path, load_path, force_column="load")
    row = _read_aligned(tmp_path, record)[0]
    assert row["area_mm2"] is None
    assert row["area_mm2_candidate"] == 5
    assert row["force_N"] is None
    assert row["load_force_unit_status"] == "UNKNOWN_OR_UNSUPPORTED_UNIT"
    assert row["stress_MPa"] is None
    assert row["stress_status"] == "MISSING_CONFIRMED_AREA"


@pytest.mark.parametrize(
    ("right_frame", "right_time", "right_gauges", "dic_gap", "expected_status"),
    [
        (2, T0 + 2, (0.03, 0.04, 0.05), 1.0, "DIC_LONG_TIME_GAP"),
        (2, T0 + 1, (0.03, 0.04, 0.05), 2.0, "DIC_INVALID_BRACKET"),
        (3, T0 + 1, (0.03, 0.04, 0.05), 2.0, "DIC_NATIVE_FRAME_GAP"),
    ],
)
def test_dic_resampling_respects_gap_quality_and_native_counter_boundaries(
    tmp_path: Path,
    right_frame: int,
    right_time: float,
    right_gauges: tuple[float, float, float],
    dic_gap: float,
    expected_status: str,
) -> None:
    left = _dic_row(1, T0, (0.01, 0.02, 0.03))
    right = _dic_row(right_frame, right_time, right_gauges)
    if expected_status == "DIC_INVALID_BRACKET":
        right["valid_gauge_count"] = 2
        right["G03_engineering_strain"] = None
    acquisition = _acquisition(tmp_path, [left, right], measurement_rows=[{
        "frame": 10, "native_cbf_frame": 10, "native_dic_frame": None,
        "scan_id": None, "scan_id_is_global_fio": False, "timestamp_unix_s": T0 + 0.5,
    }])
    load_path = _load(tmp_path / "force.csv", [
        {"timestamp_unix_s": T0, "force_N": 10},
        {"timestamp_unix_s": T0 + 1, "force_N": 20},
    ])

    record = alignment.align(acquisition, tmp_path, load_path, max_gap_s=2, dic_max_gap_s=dic_gap)
    row = _read_aligned(tmp_path, record)[0]

    assert row["dic_support_status"] == expected_status
    assert row["engineering_strain"] is None


def test_imported_accepted_values_are_retained_while_alignment_adds_force(tmp_path: Path) -> None:
    source = [{
        "frame": 4, "native_dic_frame": 4, "timestamp_unix_s": T0,
        "force_N": 123.0, "stress_MPa": 24.6, "engineering_strain": 0.04,
        "true_strain": math.log1p(0.04), "valid_gauge_count": 3,
        "strain_status": "TRACKED",
    }]
    acquisition = _acquisition(
        tmp_path, source,
        dimensions={"area_mm2": 5.0, "width_mm": None, "thickness_mm": None,
                    "status": "candidate", "source": "candidate only"},
        status="IMPORTED_ACCEPTED",
    )
    load_path = _load(tmp_path / "force.csv", [{"timestamp_unix_s": T0, "force_N": 200}])
    source_before = json.loads((tmp_path / "source.jsonl").read_text(encoding="utf-8"))

    record = alignment.align(acquisition, tmp_path, load_path)
    row = _read_aligned(tmp_path, record)[0]

    assert row["source_accepted_force_N"] == 123
    assert row["force_N"] == 123
    assert row["force_N_candidate"] == 200
    assert row["source_accepted_stress_MPa"] == 24.6
    assert row["stress_MPa"] == 24.6
    assert row["stress_MPa_candidate"] is None
    assert row["engineering_strain"] == 0.04
    assert json.loads((tmp_path / "source.jsonl").read_text(encoding="utf-8")) == source_before


def test_imported_measurement_axis_joins_accepted_values_by_native_identity_without_time(tmp_path: Path) -> None:
    accepted = [{
        "frame": 72,
        "native_cbf_frame": 436,
        "scan_id": 41,
        "scan_id_is_global_fio": True,
        "force_N": 600.0,
        "stress_MPa": 120.0,
        "engineering_strain": 0.025,
        "true_strain": math.log1p(0.025),
        "valid_gauge_count": 4,
        "strain_status": "TRACKED",
        "quality_status": "PASS",
    }]
    acquisition = _acquisition(
        tmp_path,
        accepted,
        measurement_rows=[{
            "frame": 72,
            "native_cbf_frame": 436,
            "scan_id": 41,
            "scan_id_is_global_fio": True,
            "scan_id_namespace": "global_fio_scan_id",
        }],
        status="IMPORTED_ACCEPTED",
    )
    load_path = _load(tmp_path / "force.csv", [{"timestamp_unix_s": T0, "force_N": 700}])

    record = alignment.align(acquisition, tmp_path, load_path)
    row = _read_aligned(tmp_path, record)[0]

    assert row["dic_support_status"] == "IMPORTED_ACCEPTED_EXACT_IDENTITY"
    assert row["force_interpolation_status"] == "TIME_VALUE_MISSING"
    assert row["force_N"] == 600
    assert row["stress_MPa"] == 120
    assert row["engineering_strain"] == 0.025
    assert row["true_strain"] == pytest.approx(math.log1p(0.025))
    assert row["force_N_candidate"] is None
    assert row["stress_MPa_candidate"] is None


def test_alignment_rejects_nonmonotonic_load_axis(tmp_path: Path) -> None:
    acquisition = _acquisition(tmp_path, [], [{
        "frame": 1, "native_cbf_frame": 1, "scan_id": None, "scan_id_is_global_fio": False,
        "timestamp_unix_s": T0,
    }])
    load_path = _load(tmp_path / "force.csv", [
        {"timestamp_unix_s": T0 + 1, "force_N": 1},
        {"timestamp_unix_s": T0, "force_N": 2},
    ])
    with pytest.raises(CampaignError, match="unique and increasing"):
        alignment.align(acquisition, tmp_path, load_path)
