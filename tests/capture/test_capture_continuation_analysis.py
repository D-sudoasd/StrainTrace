import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, ExperimentReader
from straintrace_capture.cameras.base import Device, Image, receipt_times
from straintrace_capture.continuation_analysis import (
    continuation_snapshot, offline_from_live_reference, prepare_continuation_binding, prepare_live_reference,
    preview_continuation, read_continuation_binding, read_live_reference,
)
from straintrace_capture.handoff import analysis_paths, analysis_settings, derive_analysis_inputs, write_snapshot
from straintrace_capture.image_inputs import decode_samples
from straintrace_capture.live import LiveAnalysis
from straintrace_capture.live_results import LiveResultLog, export_live_csv
from straintrace_capture.storage import DATABASE, ExperimentWriter, sha256_file


def _record(root, samples, role, *, parent=None, offset=0, serial="same", color="mono"):
    camera = "sim:0"
    config = CaptureConfig.from_dict({
        "config_version": 1, "name": root.name, "cameras": [{"id": camera}],
        "phases": [{"mode": "manual", "role": role, "frame_count": len(samples)}],
        "metadata": {"continuation_of": parent} if parent else {},
    })
    writer = ExperimentWriter(root, config)
    try:
        features = {name: {"value": value} for name, value in (("width", 176), ("height", 144), ("offset_x", offset), ("offset_y", 0))}
        writer.register_camera(Device(camera, "sim", "Fixture", "Speckle camera", serial, "fixture"), features)
        writer.event("phase_started", {"phase": 1, "camera_features": {camera: features}})
        for sequence, pixels in enumerate(samples, 1):
            stamp, monotonic = receipt_times()
            image = Image(pixels, "RGBa8" if color == "rgba" else "Mono8", 8, color, stamp, monotonic)
            writer.commit_frames([writer.prepare_frame(camera, sequence, 1, role, image, None, None, None, 0)])
        writer.event("phase_ended", {"phase": 1, "stop_requested": False})
        writer.set_state("completed")
    finally:
        writer.close()
    with ExperimentReader(root) as reader:
        return reader.experiment["experiment_id"]


def _images():
    pattern = np.random.default_rng(69).integers(25, 226, (144, 176), dtype=np.uint8)
    pattern = cv2.GaussianBlur(pattern, (5, 5), .85)
    return pattern, np.roll(pattern, 3, axis=1), np.roll(pattern, 4, axis=1)


def _files(root):
    return {file.relative_to(root).as_posix(): sha256_file(file)
            for file in (*root.rglob("images/**/*.tiff"), *root.rglob(DATABASE))}


def test_borrowed_reference_is_first_in_headless_dic_and_reopens_after_moving_tree(tmp_path):
    home = tmp_path / "双相机实验"
    home.mkdir()
    parent, child = home / "参考", home / "续采"
    reference, one, two = _images()
    parent_id = _record(parent, (reference,), "reference")
    child_id = _record(child, (one, two), "test", parent=parent_id)
    original = _files(home)
    assert len(original) == 5

    selection = continuation_snapshot(child, parent, None, 1)
    for preview, expected in zip(preview_continuation(selection), (reference, one), strict=True):
        np.testing.assert_array_equal(preview.pixels, expected)
    saved = prepare_continuation_binding(selection)
    snapshot, _digest = read_continuation_binding(saved)
    assert [row["sequence"] for row in snapshot["frames"]] == [1, 1, 2]
    assert selection["reference_frame_id"] == f"{parent_id}/sim:0/1"
    assert [row["frame_id"] for row in snapshot["frames"][1:]] == [f"{child_id}/sim:0/{i}" for i in (1, 2)]
    assert _files(home) == original

    assert not saved.parent.joinpath("000000000001.tiff").exists()
    moved = tmp_path / "归档"
    shutil.copytree(home, moved)
    reopened, _digest = read_continuation_binding(moved / saved.relative_to(home))
    assert [Path(path).is_relative_to(moved) for path in analysis_paths(reopened)] == [True, True, True]

    template = {"schema_version": 1, "analysis_mode": "fullfield", "output_dir": str(tmp_path / "results"),
                "field_roi": [32, 28, 96, 84], "solver": {"subset_size_px": 21, "step_px": 14}}
    path = write_snapshot(template, tmp_path / "template.json")
    config_path = tmp_path / "analysis.json"
    gather = [sys.executable, "-B", "-m", "straintrace_capture", "prepare-analysis", str(child),
              "--reference-experiment", str(parent), "--reference-sequence", "1", "--template", str(path),
              "--run-config", str(config_path), "--output", str(tmp_path / "snapshot.json")]
    command = subprocess.run(gather, capture_output=True, text=True, timeout=45)
    assert command.returncode == 0, command.stdout + command.stderr
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    assert settings["capture_source"] == analysis_settings(reopened, template)["capture_source"]
    run = [sys.executable, "-B", "-m", "ezdic_cli", "run", "--config", str(config_path)]
    result = subprocess.run(run, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["scientific_ok"]
    manifest = json.loads((tmp_path / "results/run_manifest.json").read_text(encoding="utf-8"))
    source = manifest["config"]["capture_source"]
    assert source["experiment_id"] == child_id and source["frames"][0]["experiment_id"] == parent_id
    assert manifest["inputs"][0]["sha256"] == snapshot["frames"][0]["sha256"]
    write_snapshot({**settings, "start_frame_1based": 2, "reference_frame_1based": 2,
                    "field_roi_reference_frame_1based": 2}, tmp_path / "wrong-zero.json")
    wrong = subprocess.run([*run[:-1], str(tmp_path / "wrong-zero.json")], capture_output=True, timeout=45)
    assert wrong.returncode != 0
    assert json.loads(wrong.stderr)["error_code"] == "CAPTURE_REFERENCE_RANGE"
    assert _files(home) == original


def test_same_image_size_cannot_hide_changed_sensor_offset_or_substituted_reference(tmp_path):
    reference, sample, _ = _images()
    parent = tmp_path / "parent"
    identity = _record(parent, (reference,), "reference")
    changed, wrong_camera, normal = (tmp_path / label for label in ("roi-changed", "other-camera", "next"))
    _record(changed, (sample,), "test", parent=identity, offset=12)
    _record(wrong_camera, (sample,), "test", parent=identity, serial="replaced")
    _record(normal, (sample,), "test", parent=identity)
    with pytest.raises(CaptureError) as problem:
        continuation_snapshot(changed, parent, "sim:0", 1)
    assert problem.value.code == "CONTINUATION_REFERENCE_GEOMETRY"
    with pytest.raises(CaptureError) as problem:
        continuation_snapshot(wrong_camera, parent, "sim:0", 1)
    assert problem.value.code == "CONTINUATION_REFERENCE_CAMERA"
    with pytest.raises(CaptureError) as problem:
        continuation_snapshot(normal, changed, "sim:0", 1)
    assert problem.value.code == "CONTINUATION_REFERENCE_SOURCE"

    selection = continuation_snapshot(normal, parent, "sim:0", 1)
    saved = prepare_continuation_binding(selection)
    file = Path(analysis_paths(selection)[0])
    data = file.read_bytes()
    file.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(CaptureError) as problem:
        read_continuation_binding(saved)
    assert problem.value.code == "CAPTURE_IMAGE_CHANGED"


def test_color_transform_uses_analysis_order_when_ledger_sequences_repeat(tmp_path):
    parent, child = tmp_path / "parent", tmp_path / "next"
    samples = [np.stack((image, image, image, np.full_like(image, 230)), axis=-1) for image in _images()]
    identity = _record(parent, samples[:1], "reference", color="rgba")
    _record(child, samples[1:], "test", parent=identity, color="rgba")
    selected = continuation_snapshot(child, parent, "sim:0", 1)
    binding = prepare_continuation_binding(selected)
    snapshot, _digest = read_continuation_binding(binding)
    paths = [Path(path) for path in analysis_paths(snapshot)]
    assert [p.name for p in paths] == ["000000000001.tiff", "000000000002.tiff", "000000000003.tiff"]
    for file, expected in zip(paths, samples, strict=True):
        np.testing.assert_array_equal(decode_samples(file), expected[..., 0])


def test_event_continuation_binding_preserves_selected_event_on_reopen(tmp_path):
    parent, child = tmp_path / "parent", tmp_path / "child"
    reference, first, second = _images()
    identity = _record(parent, (reference,), "reference")
    config = CaptureConfig.from_dict({
        "config_version": 1, "name": "Events", "cameras": [{"id": "sim:0"}],
        "phases": [{"mode": "manual", "frame_count": 1},
                   {"mode": "event", "fps": 4, "post_frames": 2, "event_count": 2}],
        "metadata": {"continuation_of": identity},
    })
    writer = ExperimentWriter(child, config)
    try:
        features = {name: {"value": value} for name, value in
                    (("width", 176), ("height", 144), ("offset_x", 0), ("offset_y", 0))}
        writer.register_camera(Device("sim:0", "sim", "Fixture", "Speckle camera", "same", "fixture"), features)
        for number, signal in enumerate(("event-one", "event-two"), 1):
            writer.event("signal_received", {"phase": 2, "id": signal, "source": "operator", "payload": {},
                         "received_utc_ns": 1_700_000_000_000_000_000 + number,
                         "received_monotonic_ns": 100 + number, "event_number": number}, event_id=signal)
        for sequence, (phase, signal, pixels) in enumerate(((1, None, first), (2, "event-one", first),
                                                            (2, "event-two", first), (2, "event-two", second)), 1):
            image = Image(pixels, "Mono8", 8, "mono", 1_700_000_000_000_000_100 + sequence, 200 + sequence)
            writer.commit_frames([writer.prepare_frame("sim:0", sequence, phase, "test", image, None,
                                                       signal, "post" if signal else None, 0)])
        writer.set_state("completed")
    finally:
        writer.close()

    selected = continuation_snapshot(child, parent, "sim:0", 1, event_context_id="event-two")
    assert [row["sequence"] for row in selected["frames"]] == [1, 3, 4]
    binding = prepare_continuation_binding(selected)
    reopened, _ = read_continuation_binding(binding)
    assert reopened["event"] == selected["event"]
    assert [row["frame_id"] for row in reopened["frames"]] == [row["frame_id"] for row in selected["frames"]]
    source = analysis_settings(reopened, {})["capture_source"]
    assert source["event"]["signal_id"] == "event-two"
    assert [row.get("event_position") for row in source["frames"]] == [None, "post", "post"]

    encoded = binding.read_text(encoding="utf-8")
    for key in ("signal_time", "event_position"):
        data = json.loads(encoded)
        if key == "signal_time":
            data["event"]["signal_utc_ns"] += 1
        else:
            data["frames"][1]["event_position"] = "pre"
        binding.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(CaptureError) as changed:
            read_continuation_binding(binding)
        assert changed.value.code == "CAPTURE_BINDING_INVALID"


@pytest.mark.parametrize(("color", "pixel_format"), [("rgba", "RGBa8"), ("bayer", "BayerRG8")])
def test_derivation_checks_the_same_encoded_bytes_it_decodes(tmp_path, monkeypatch, color, pixel_format):
    from straintrace_capture import image_inputs

    images = tmp_path / "capture" / "images"
    images.mkdir(parents=True)
    original, replacement = images / "original.tiff", images / "replacement.tiff"
    shape = (16, 16, 4) if color == "rgba" else (16, 16)
    assert cv2.imwrite(str(original), np.full(shape, 40, dtype=np.uint8))
    assert cv2.imwrite(str(replacement), np.full(shape, 140, dtype=np.uint8))
    snapshot = {"experiment_root": str(images.parent), "frames": [{
        "frame_id": "fixture/sim:0/1", "sequence": 1, "color_model": color, "pixel_format": pixel_format,
        "absolute_path": str(original), "sha256": sha256_file(original),
    }]}
    decode = image_inputs.decode_samples
    switched = False

    def change_before_decode(path, **options):
        nonlocal switched
        if path == original and not switched:
            switched = True
            original.write_bytes(replacement.read_bytes())
        return decode(path, **options)

    monkeypatch.setattr(image_inputs, "decode_samples", change_before_decode)
    with pytest.raises(CaptureError) as changed:
        derive_analysis_inputs(snapshot, tmp_path / "derived", debayer=True, drop_alpha=True)
    assert switched and changed.value.code == "CAPTURE_IMAGE_CHANGED"


def test_write_snapshot_does_not_leave_partial_json_when_serialization_fails(tmp_path):
    output = tmp_path / "snapshot.json"
    with pytest.raises(ValueError):
        write_snapshot({"value": float("nan")}, output)
    assert not output.exists() and not list(tmp_path.glob(".capture-report-*.part"))
    assert write_snapshot({"value": 3}, output) == output
    with pytest.raises(FileExistsError):
        write_snapshot({"value": 4}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 3}


def _live_writer(root, parent_id):
    plan = CaptureConfig.from_dict({
        "config_version": 1, "name": "Live continuation", "cameras": [{"id": "sim:0"}],
        "phases": [{"mode": "manual", "frame_count": 2}, {"mode": "single"}],
        "metadata": {"continuation_of": parent_id},
    })
    writer = ExperimentWriter(root, plan)
    features = {key: {"value": value} for key, value in (("width", 176), ("height", 144), ("offset_x", 0), ("offset_y", 0))}
    writer.register_camera(Device("sim:0", "sim", "Fixture", "Speckle camera", "same", "fixture"), features)
    writer.set_state("running")
    writer.event("phase_started", {"phase": 1, "camera_features": {"sim:0": features}})
    return writer, features


def _commit(writer, pixels, sequence, phase, received):
    image = Image(pixels, "Mono8", 8, "mono", 1_700_000_000_000_000_000 + received, received)
    writer.commit_frames([writer.prepare_frame("sim:0", sequence, phase, "test", image, None, None, None, 0)])


def test_live_reference_waits_before_first_test_and_follows_target_clock(tmp_path):
    ref, first, second = _images()
    prior, child = tmp_path / "parent", tmp_path / "child"
    parent = _record(prior, (ref, ref), "reference")
    writer, _ = _live_writer(child, parent)
    roi = {"field_roi": [32, 28, 96, 84], "field_step": 14}
    try:
        preview = continuation_snapshot(child, prior, "sim:0", 2, live=True)
        assert len(preview["frames"]) == 1
        binding = prepare_live_reference(preview)
        reopened, digest = read_live_reference(binding)
        assert read_live_reference(binding, expected_digest=digest)[0]["reference_frame_id"] == f"{parent}/sim:0/2"
        assert reopened["experiment_state"] == "running"
        with pytest.raises(CaptureError) as active:
            offline_from_live_reference(binding)
        assert active.value.code == "CONTINUATION_REFERENCE_ACTIVE"
        recording = LiveResultLog(child)
        with LiveAnalysis(child, reference_binding=binding, latest=False, **roi) as live:
            reference = live.poll()
            recording.append(reference, live, "ordered")
            assert reference["is_reference"] and reference["frame_id"] == preview["reference_frame_id"]
            assert reference["reference_experiment_id"] == reference["frame_experiment_id"] == parent
            assert reference["elapsed_receive_s"] is None and reference["processed"] == 0
            assert live.poll() is None
            _commit(writer, first, 1, 1, 12_000_000_000)
            _commit(writer, second, 2, 1, 12_250_000_000)
            one = live.poll()
            recording.append(one, live, "ordered")
            assert one["sequence"] == 1 and one["elapsed_receive_s"] == 0
            assert one["elapsed_receive_basis"] == "first_current_test_frame"
            assert one["frame_experiment_id"] == one["experiment_id"] != one["reference_experiment_id"]
            assert np.nanmedian(live.field["u"]) == pytest.approx(3, abs=.03)
            two = live.poll()
            recording.append(two, live, "ordered")
            assert two["sequence"] == 2 and two["elapsed_receive_s"] == .25
            assert np.nanmedian(live.field["u"]) == pytest.approx(4, abs=.03)
        with LiveAnalysis(child, reference_binding=binding, latest=True, **roi) as latest:
            assert latest.poll()["is_reference"]
            row = latest.poll()
            assert row["elapsed_receive_s"] == .25 and row["skipped_committed_frames"] == 1
        writer.set_state("completed")
    finally:
        writer.close()

    offline = offline_from_live_reference(binding, expected_digest=digest)
    assert recording.error is None and recording.test_frames == 2
    saved = [json.loads(line) for line in recording.path.read_text(encoding="utf-8").splitlines()]
    assert saved[0]["frame_experiment_id"] == parent and saved[0]["settings"]["external_reference"]["sequence"] == 2
    assert [row["elapsed_receive_s"] for row in saved] == [None, 0, .25]
    assert all(row["reference_sha256"] == saved[0]["image_sha256"] for row in saved)
    assert export_live_csv(recording.path, tmp_path / "field.csv")["frames"] == 3
    assert read_continuation_binding(offline)[0]["reference_frame_id"] == reference["frame_id"]
    config = write_snapshot(roi, tmp_path / "roi.json")
    result = subprocess.run([sys.executable, "-B", "-m", "straintrace_capture", "follow", str(child),
                             "--config", str(config), "--all-frames", "--reference-experiment", str(prior),
                             "--reference-sequence", "2"], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["frame_id"] for row in rows] == [reference["frame_id"], one["frame_id"], two["frame_id"]]
    assert [row["elapsed_receive_s"] for row in rows] == [None, 0, .25]


def test_live_checks_sensor_offset_again_when_a_later_phase_starts(tmp_path):
    ref, first, second = _images()
    prior, child = tmp_path / "parent", tmp_path / "child"
    parent = _record(prior, (ref,), "reference")
    writer, features = _live_writer(child, parent)
    try:
        with LiveAnalysis(child, reference_experiment=prior, reference_sequence=1,
                          field_roi=[32, 28, 96, 84], field_step=14, latest=False) as live:
            assert live.poll()["is_reference"]
            _commit(writer, first, 1, 1, 12_000_000_000)
            assert live.poll()["field_quality"]["scientific_ok"]
            changed = {**features, "offset_x": {"value": 12}}
            writer.event("phase_started", {"phase": 2, "camera_features": {"sim:0": changed}})
            _commit(writer, second, 2, 2, 12_250_000_000)
            with pytest.raises(CaptureError) as problem:
                live.poll()
            assert problem.value.code == "CONTINUATION_REFERENCE_GEOMETRY"
            assert live.processed == 1 and live.skipped == 0
        writer.set_state("completed")
    finally:
        writer.close()
