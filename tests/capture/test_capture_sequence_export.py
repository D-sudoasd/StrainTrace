import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from straintrace_capture import (
    CaptureConfig, CaptureError, ExperimentReader, SequenceSelection, export_sequence, inspect_sequence, verify_sequence,
)
from straintrace_capture.cameras._gentl_multipart import IMAGE_PART
from straintrace_capture.cameras.base import Device, Image
from straintrace_capture.cli import main
from straintrace_capture.sequence_export import SEQUENCE_INDEX, SEQUENCE_MANIFEST
from straintrace_capture.storage import ExperimentWriter, sha256_file


BASE_UTC = 1795570200123456789
BASE_MONOTONIC = 790059000002
PART = "2/16/0/Mono8"
PHASES = [
    {"mode": "single", "role": "reference"},
    {"mode": "manual", "role": "calibration"},
    {"mode": "timed", "fps": 10, "frame_count": 2},
    {"mode": "timed", "fps": 10, "frame_count": 2},
]
FRAMES = [(1, "reference"), (2, "calibration"), (3, "test"), (3, "test"), (4, "test"), (4, "test")]


def experiment(root, *, cameras=("sim:0", "sim:1"), phases=None, frames=None, offsets=None, state="completed",
               commit_order=None):
    plan = CaptureConfig.from_dict({
        "config_version": 1, "name": "导出试验", "cameras": [{"id": camera} for camera in cameras],
        "phases": PHASES if phases is None else phases, "segment_frames": 2,
    })
    writer = ExperimentWriter(root, plan)
    try:
        for camera in cameras:
            writer.register_camera(Device(camera, "sim", "模拟散斑", "Sim", camera[-1], ""), {})
        for phase, offset in (offsets or {}).items():
            writer.event("phase_started", {
                "phase": phase, "camera_features": {camera: {"offset_x": {"value": offset}} for camera in cameras},
            })
        phase_frames = FRAMES if frames is None else frames
        for sequence in commit_order or range(1, len(phase_frames) + 1):
            phase, role = phase_frames[sequence - 1]
            for number, camera in enumerate(cameras):
                times = BASE_UTC + sequence * 71 + number, BASE_MONOTONIC + sequence
                mono = Image(np.full((16, 24), sequence + number * 100, np.uint16), "Mono12", 12, "mono", *times)
                part = Image(np.full((8, 12), sequence, np.uint8), "Mono8", 8, "mono", *times,
                             metadata={IMAGE_PART: {"selector": PART}})
                frame = writer.prepare_frame(camera, sequence, phase, role, replace(mono, auxiliary_images=(part,)),
                                             None, None, None, 0)
                writer.commit_frames([frame])
        writer.set_state(state)
    finally:
        writer.close()
    return root


def hashes(root):
    return {path.relative_to(root): sha256_file(path) for path in root.rglob("*") if path.is_file()}


def scan(folder):
    with (folder / SEQUENCE_INDEX).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_range_preserves_reference_raw_tiff_and_time_and_can_verify_without_source(tmp_path):
    import cv2

    source = experiment(tmp_path / "源实验")
    target = tmp_path / "选段序列"
    selection = SequenceSelection(camera_id="sim:1", phase=4, first_sequence=6, last_sequence=6)
    before = hashes(source)
    report = inspect_sequence(source, target, selection)
    assert report["reference_sequence"] == 1 and report["included_phases"] == [1, 4]
    assert report["frames"] == 2 and report["available_bytes"] > report["image_bytes"]
    assert not target.exists()
    result = export_sequence(source, target, selection)
    assert result == verify_sequence(target)
    rows = scan(target)
    assert [(r["index"], r["sequence_role"], r["source_sequence"]) for r in rows] == [
        ("0", "reference", "1"), ("1", "test", "6"),
    ]
    with ExperimentReader(source, immutable=True) as reader:
        for row, seq in zip(rows, (1, 6), strict=True):
            original = reader.frame_at_sequence("sim:1", seq)
            assert row["file"] == f"images/frame_{int(row['index']):012d}.tif"
            assert row["frame_id"] == original["frame_id"]
            assert row["source_path"] == original["path"]
            assert row["received_utc_ns"] == str(BASE_UTC + seq * 71 + 1)
            assert row["received_monotonic_ns"] == str(BASE_MONOTONIC + seq)
            assert row["received_utc_iso"].endswith(f".{(BASE_UTC + seq * 71 + 1) % 10**9:09d}Z")
            assert json.loads(row["clock_readings_json"]) == original["timestamps"]
            assert (target / row["file"]).read_bytes() == (source / original["path"]).read_bytes()
            pixels = cv2.imdecode(np.fromfile(target / row["file"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            assert pixels.dtype == np.uint16 and pixels.shape == (16, 24) and np.all(pixels == seq + 100)
    assert hashes(source) == before
    source.rename(tmp_path / "moved-source")
    assert verify_sequence(target)["ok"]


def test_part_selection_and_first_test_reference(tmp_path):
    source = experiment(tmp_path / "single", cameras=("sim:0",), phases=[PHASES[2]],
                        frames=[(1, "test"), (1, "test")])
    result = export_sequence(source, tmp_path / "selected", SequenceSelection(image_part=PART))
    assert result["image_part"] == PART and result["reference_source_role"] == "test"
    assert result["sample_layout"]["shape"] == [8, 12]
    assert result["sensor_geometry_by_phase"]["1"]["offset_x"] == 16
    for entry in scan(Path(result["output"])):
        assert "-part-02." in entry["source_path"]
        assert sha256_file(Path(result["output"]) / entry["file"]) == sha256_file(source / entry["source_path"])


def test_backfilled_frames_follow_original_sequence_including_automatic_reference(tmp_path):
    source = experiment(tmp_path / "recovered", cameras=("sim:0",), phases=[PHASES[2]],
                        frames=[(1, "test")] * 4, commit_order=(2, 4, 1, 3))
    target = tmp_path / "restored-order"
    assert export_sequence(source, target)["reference_sequence"] == 1
    assert [int(row["source_sequence"]) for row in scan(target)] == [1, 2, 3, 4]


def test_rejects_bad_scope_and_inconsistent_sensor_windows(tmp_path):
    source = experiment(tmp_path / "run", offsets={3: 8, 4: 16})
    output = tmp_path / "target"
    cases = [
        ("CHOOSE_CAMERA", SequenceSelection()),
        ("REFERENCE_UNAVAILABLE", SequenceSelection(camera_id="sim:0", reference_sequence=2)),
        ("SEQUENCE_PHASE", SequenceSelection(camera_id="sim:0", phase=2)),
        ("SEQUENCE_EMPTY", SequenceSelection(camera_id="sim:0", first_sequence=9)),
        ("SEQUENCE_SENSOR_CHANGED", SequenceSelection(camera_id="sim:0")),
    ]
    for code, options in cases:
        with pytest.raises(CaptureError) as caught:
            export_sequence(source, output, options)
        assert caught.value.code == code and not output.exists()
    assert export_sequence(source, output, SequenceSelection(camera_id="sim:0", phase=3))["frames"] == 3


def test_rejects_active_ambiguous_or_swapped_source(tmp_path):
    active = experiment(tmp_path / "active", state="recording")
    with pytest.raises(CaptureError, match="capture state is recording"):
        export_sequence(active, tmp_path / "out", SequenceSelection(camera_id="sim:0"))
    source = experiment(tmp_path / "multiple", cameras=("sim:0",), phases=[PHASES[0], *PHASES],
                        frames=[(1, "reference"), (2, "reference"), (4, "test")])
    with pytest.raises(CaptureError, match="selected experiment identity changed"):
        inspect_sequence(source, tmp_path / "out", expected_id="another-run")
    with pytest.raises(CaptureError, match="multiple reference images exist"):
        export_sequence(source, tmp_path / "out")
    assert export_sequence(source, tmp_path / "out", SequenceSelection(reference_sequence=2))["reference_sequence"] == 2


def test_cancellation_and_competing_destinations_do_not_publish_partial_results(tmp_path):
    source = experiment(tmp_path / "run", cameras=("sim:0",))
    output = tmp_path / "out"
    cancelled = False

    def cancel_on_verify(update):
        nonlocal cancelled
        if update.stage == "checking":
            cancelled = True

    with pytest.raises(CaptureError) as interrupted:
        export_sequence(source, output, cancelled=lambda: cancelled, progress=cancel_on_verify)
    assert interrupted.value.code == "SEQUENCE_CANCELLED"
    assert not output.exists() and not list(tmp_path.glob(".capture-sequence-*"))

    def insert_target(update):
        if update.stage == "publishing":
            output.mkdir()
            (output / "existing.txt").write_bytes(b"keep")

    with pytest.raises(CaptureError) as collision:
        export_sequence(source, output, progress=insert_target)
    assert collision.value.code == "OUTPUT_EXISTS"
    assert [path.name for path in output.iterdir()] == ["existing.txt"]
    assert (output / "existing.txt").read_bytes() == b"keep"
    assert not list(tmp_path.glob(".capture-sequence-*"))


def test_cli_and_integrity_checks_identify_source_and_destination_tampering(tmp_path, capsys):
    source = experiment(tmp_path / "run", cameras=("sim:0",))
    target = tmp_path / "out"
    args = ["export-sequence", str(source), "--output", str(target), "--phase", "4", "--sequence", "6"]
    assert main([*args, "--check"]) == 0 and json.loads(capsys.readouterr().out)["frames"] == 2
    assert main(args) == 0 and json.loads(capsys.readouterr().out)["ok"]
    assert main(["verify-sequence", str(target)]) == 0 and json.loads(capsys.readouterr().out)["ok"]
    frame = scan(target)[-1]
    for path, command in ((target / frame["file"], ["verify-sequence", str(target)]),
                          (source / frame["source_path"], [*args[:3], str(tmp_path / "bad-copy"), *args[4:]])):
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        assert main(command) == (4 if path.is_relative_to(target) else 3)
        assert json.loads(capsys.readouterr().out)["error"]["code"] == "TRANSFER_CHANGED"
        path.write_bytes(original)
    assert not (tmp_path / "bad-copy").exists()
    (target / "images" / "extra.tif").write_bytes(b"extra")
    assert main(["verify-sequence", str(target)]) == 4
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "SEQUENCE_INDEX"
    manifest = json.loads((target / SEQUENCE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["index"]["sha256"] == sha256_file(target / SEQUENCE_INDEX)
