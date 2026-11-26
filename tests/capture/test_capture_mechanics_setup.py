import csv
import json

import cv2
import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, ExperimentReader
from straintrace_capture.cameras.base import Device, Image
from straintrace_capture.live import LiveAnalysis
from straintrace_capture.live_results import LiveResultLog, export_live_csv
from straintrace_capture.mechanics_setup import inspect_load_csv, preview_mechanics, suggested_capture_basis, table_alignment
from straintrace_capture.storage import DATABASE, ExperimentWriter, sha256_file
from straintrace_capture.synchronization import load_alignment_settings


def capture_with_known_receipts(root):
    settings = CaptureConfig.from_dict({
        "config_version": 1, "name": "Known clock fixture", "cameras": [{"id": "sim:0"}],
        "phases": [{"mode": "single", "role": "reference"}, {"mode": "manual", "frame_count": 3}],
    })
    pixels = cv2.GaussianBlur(np.random.default_rng(70).integers(50, 3950, (144, 176), dtype=np.uint16), (5, 5), .85)
    writer = ExperimentWriter(root, settings)
    try:
        writer.register_camera(Device("sim:0", "sim", "Fixture", "Known clock", "0", "unknown"), {})
        rows = []
        for seq, delta in enumerate((0, 500_000_000, 1_000_000_000, 5_000_000_000), 1):
            image = Image(np.roll(pixels, seq - 1, axis=1), "Mono12", 12, "mono",
                          1_800_000_000_000_000_000 + delta, 100_000_000_000 + delta)
            rows.append(writer.prepare_frame("sim:0", seq, 1 if seq == 1 else 2, "reference" if seq == 1 else "test",
                                             image, None, None, None, 0))
        writer.commit_frames(rows)
        writer.set_state("completed")
    finally:
        writer.close()
    return root


def _options(info, **overrides):
    hints = info.suggest()
    return {"time_column": hints["time"], "time_format": hints["time_format"], "time_unit": hints["time_unit"],
            "force_column": hints["force"], "force_unit": hints["force_unit"],
            "displacement_column": hints["displacement"], "displacement_unit": hints["displacement_unit"],
            "basis": "host-receive", "max_gap_s": 1.5, **overrides}


def test_relative_load_csv_uses_the_measured_event_and_keeps_evidence(tmp_path):
    root = capture_with_known_receipts(tmp_path / "run")
    table = tmp_path / "载荷.csv"
    table.write_text("试验时间 (ms);试验力 (kN);横梁位移 (µm)\n"
                     "1000;0.1;0\n2000;0.3;200\n3000;0.5;400\n7000;0.7;1000\n", encoding="gb18030")
    info = inspect_load_csv(table)
    assert info.encoding == "gb18030"
    assert info.suggest() == {"time": "试验时间 (ms)", "time_format": "relative", "time_unit": "ms", "gap": 3,
                              "force": "试验力 (kN)", "force_unit": "kN", "displacement": "横梁位移 (µm)",
                              "displacement_unit": "um"}
    assert suggested_capture_basis(root, "sim:0") == "host-receive"
    with pytest.raises(CaptureError, match="所选相机尚无序号"):
        table_alignment(info, root, "sim:0", **_options(info, sync_frame="999", sync_time="2000"))
    with pytest.raises(CaptureError, match="没有可用于对齐"):
        table_alignment(info, root, "sim:0", **_options(info, basis="exposure-midpoint", sync_frame="2", sync_time="2000"))
    bad = table_alignment(info, root, "sim:0", **_options(info, sync_frame="2", sync_time="20"))
    with pytest.raises(CaptureError, match="不在载荷表记录范围"):
        preview_mechanics(bad, root, "sim:0")
    channel = table_alignment(info, root, "sim:0", **_options(info, sync_frame="2", sync_time="2000"))
    _, preview = preview_mechanics(channel, root, "sim:0")
    assert [row["frame"] for row in preview] == [1, 2, 4]
    assert preview[0]["load_time_s"] == 1.5
    assert preview[0]["force_N"] == 200 and preview[0]["displacement_mm"] == .1
    assert preview[1]["force_support"]["status"] == "EXACT_TIME_SAMPLE"
    assert preview[-1]["force_support"]["status"] == "LONG_TIME_GAP"

    with ExperimentReader(root) as reader:
        original = reader.frames()
        frame = original[2]
        mismatch = {**frame, "timestamps": {**frame["timestamps"], "received_monotonic": {
            **frame["timestamps"]["received_monotonic"], "clock_id": "other-session"}}}
        assert channel.at(mismatch)["force_support"]["status"] == "LOAD_SYNC_CLOCK_MISMATCH"
        foreign = {**frame, "frame_id": "different-experiment/sim:0/3"}
        assert channel.at(foreign)["force_N"] is None

    gauges = [{"name": "G01", "roi1": (45, 25, 25, 25), "roi2": (45, 83, 25, 25), "strain_mode": "y"}]
    record = LiveResultLog(root)
    original_ledger = sha256_file(root / DATABASE)
    with LiveAnalysis(root, gauges=gauges, latest=False, load_alignment=channel) as live:
        while row := live.poll():
            record.append(row, live, "ordered")
    assert record.error is None
    rows = [json.loads(line) for line in record.path.read_text(encoding="utf-8").splitlines()]
    assert [row["mechanics"]["force_N"] for row in rows] == [200, 300, 400, None]
    assert rows[0]["settings"]["mechanics"]["load_clock_origin"]["frame_id"] == original[1]["frame_id"]
    assert rows[2]["mechanics"]["mechanics_source"]["sha256"] == sha256_file(table)
    assert sha256_file(root / DATABASE) == original_ledger
    export = tmp_path / "alignment.csv"
    export_live_csv(record.path, export)
    with export.open(encoding="utf-8-sig", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert [row["载荷 (N)"] for row in exported] == ["200.0", "300.0", "400.0", ""]
    assert all(row["载荷同步帧标识"] == original[1]["frame_id"] for row in exported)
    assert exported[2]["载荷表匹配时间 (s)"] == "2.5"

    text = table.read_text(encoding="gb18030")
    table.write_text(text + "8000;0.8;1200\n", encoding="gb18030")
    assert channel.at(frame)["force_N"] == 400
    table.write_text(text.replace("2000;0.3;200", "2000;0.4;200"), encoding="gb18030")
    with pytest.raises(CaptureError, match="载荷表已重写"):
        channel.at(frame)


def test_absolute_load_accepts_header_before_data_and_recovers_after_bad_sample(tmp_path):
    root = capture_with_known_receipts(tmp_path / "run")
    table = tmp_path / "load.tsv"
    header = "timestamp_unix_ms\tLoad (N)\n"
    table.write_text(header, encoding="utf-16")
    info = inspect_load_csv(table)
    assert (info.encoding, info.suggest()["time_format"], info.suggest()["time_unit"]) == ("utf-16", "absolute", "ms")
    channel = table_alignment(info, root, "sim:0", **_options(info))
    summary, rows = preview_mechanics(channel, root, "sim:0")
    assert (summary["rows"], rows) == (0, [])

    gauges = [{"name": "G01", "roi1": (45, 25, 25, 25), "roi2": (45, 83, 25, 25), "strain_mode": "y"}]
    with LiveAnalysis(root, gauges=gauges, latest=False, load_alignment=channel) as live:
        assert live.poll()["mechanics"]["force_support"]["status"] == "LOAD_SOURCE_EMPTY"
        assert live.record_context()[1]["mechanics"]["time_format"] == "absolute"
        table.write_text(header + "unknown\t150\n", encoding="utf-16")
        assert live.poll()["mechanics"]["force_support"]["status"] == "LOAD_TIME_AXIS_INVALID"
        table.write_text(header + "1800000000000\t100\n1800000001000\t200\n", encoding="utf-16")
        result = live.poll()
        assert result["sequence"] == 3 and result["mechanics"]["force_N"] == 200
        assert result["mechanics"]["load_time_s"] == 1_800_000_001
        assert live.poll()["mechanics"]["force_support"]["status"] == "OUTSIDE_TIME_RANGE"


def test_guess_leaves_ambiguous_columns_and_unstated_units_for_confirmation(tmp_path):
    table = tmp_path / "unknown.csv"
    table.write_text("Time,Load,Force,Stroke\n0.1,5,8,1\n0.2,7,14,2\n", encoding="utf-8")
    suggested = inspect_load_csv(table).suggest()
    assert suggested["time"] == "Time" and not suggested["time_format"] and not suggested["time_unit"]
    assert not suggested["force"] and not suggested["force_unit"]
    assert suggested["displacement"] == "Stroke" and not suggested["displacement_unit"]


def test_imported_clock_basis_is_checked_even_before_the_first_image(tmp_path):
    with pytest.raises(CaptureError) as error:
        load_alignment_settings({"source": "load.csv", "basis": "device_counter"}, base_dir=tmp_path)
    assert error.value.code == "CLOCK_BASIS_INVALID"
