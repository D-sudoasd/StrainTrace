import csv
import json
from hashlib import sha256

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, ExperimentReader
from straintrace_capture.cameras.base import Device, Image, receipt_times
from straintrace_capture.cli import main
from straintrace_capture.pulse_audit import audit_external_pulses, pulse_audit_context
from straintrace_capture.pulse_binding import PULSE_BINDING_SCHEMA, PulseBinding
from straintrace_capture.pulse_timing import ExposureLimits, add_exposure_audit
from straintrace_capture.storage import ExperimentWriter
from straintrace_capture.trigger_review import counter_review_snapshot, read_counter_review

from test_capture_pulse_audit import CAMERAS, SOURCE, pulse_fixture, write_daq
from test_capture_pulse_timing import edges, write_edges


FIRST = 2**63 + 105


def binding_fixture(root, *, offsets=None, end=5):
    samples = offsets if offsets is not None else {CAMERAS[0]: [-1, 0, 2, 5, 7], CAMERAS[1]: [-1, 0, 1, 2, 2, 4, 5, 7]}
    config = CaptureConfig.from_dict({"config_version": 1, "name": "Unassociated hardware fixture",
                                     "cameras": [{"id": camera} for camera in CAMERAS],
                                     "phases": [{"mode": "hardware", "hardware_source": "Line1"}, {"mode": "single"}]})
    writer = ExperimentWriter(root, config)
    pair = {"schema": PULSE_BINDING_SCHEMA, "experiment_id": writer.experiment_id, "phase": 1, "source_id": SOURCE,
            "counter_feature": "ChunkCounterValue", "first_event_number": FIRST, "last_event_number": FIRST + end,
            "camera_frames": {key: [2, len(values) - 1] for key, values in samples.items()},
            "evidence": "Fixture pulse markers identify the exact images at both ends independently of arrival times."}
    counts = {key: 0 for key in CAMERAS}
    try:
        for camera in CAMERAS:
            writer.register_camera(Device(camera, "sim", "Fixture", camera, camera, "simulated"), {})
        for stage in (1, 2):
            writer.event("phase_started", {"phase": stage})
            frames = []
            for channel, (camera, values) in enumerate(samples.items()):
                for offset in (values if stage == 1 else [end]):
                    counts[camera] += 1
                    utc, monotonic = receipt_times()
                    metadata = {"ChunkCounterValue": 2**63 + 30 * (channel + 1) + offset} if offset is not None else {}
                    pattern = np.arange(120, dtype=np.uint8).reshape(10, 12) + counts[camera]
                    image = Image(pattern, "Mono8", 8, "mono", utc, monotonic, metadata=metadata)
                    frames.append(writer.prepare_frame(camera, counts[camera], stage, "test", image, None, None, None, 0))
            writer.commit_frames(frames)
            writer.event("phase_ended", {"phase": stage})
        writer.set_state("completed")
    finally:
        writer.close()
    return root, pair


def daq_file(path, numbers):
    return write_daq(path, "event_number,source_id,phase\n" + "".join(f"{number},{SOURCE},1\n" for number in numbers))


def test_unanchored_run_keeps_outside_images_and_all_identity_disagreements(tmp_path):
    root, pair = binding_fixture(tmp_path / "experiment")
    before = {path: sha256(path.read_bytes()).hexdigest() for path in root.rglob("*")
              if path.suffix == ".tiff" or path.name == "capture.sqlite3"}
    source = daq_file(tmp_path / "daq.csv", range(FIRST - 1, FIRST + 7))
    assert pulse_audit_context(root) == (pair["experiment_id"], {1: None}, CAMERAS)
    report = audit_external_pulses(root, source, phase=1, binding=PulseBinding.from_dict(pair))
    assert report["association"]["binding"] == pair
    assert report["summary"]["outside_anchor_frames"] == 4 and report["input"]["excluded_rows"] == {"outside_range": 2}
    assert not report["complete"] and report["summary"]["daq_unlisted_pulses"] == 0
    assert [row["status"] for row in report["pulses"]] == ["matched", "partial", "conflict", "missed", "partial", "matched"]
    row = report["pulses"][2]
    assert row["duplicate_cameras"] == [CAMERAS[1]] and row["capture_event_ids"] == []
    snapshot = counter_review_snapshot(report["experiment_id"], 1, SOURCE, row["event_number"], CAMERAS, row["frames"],
                                        "ChunkCounterValue", status="ambiguous")
    actual, frames, images, errors, _ = read_counter_review(root, snapshot)
    assert actual["status"] == "ambiguous" and len(frames[CAMERAS[1]]) == 2 and set(images) == {CAMERAS[0]} and not errors
    assert np.array_equal(images[CAMERAS[0]].pixels, np.arange(120, dtype=np.uint8).reshape(10, 12) + 3)
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items())
    with ExperimentReader(root) as reader:
        assert not reader.pairing()["groups"]
    (root / row["frames"][0]["path"]).write_bytes(b"changed fixture TIFF")
    _, _, images, errors, _ = read_counter_review(root, snapshot)
    assert not images and CAMERAS[0] in errors


def test_clipped_review_requires_both_anchor_csv_rows_and_can_join_timing(tmp_path):
    root, pair = binding_fixture(tmp_path / "experiment", offsets=dict.fromkeys(CAMERAS, [-1, 0, 1, 2, 3]), end=2)
    source = daq_file(tmp_path / "daq.csv", range(FIRST, FIRST + 3))
    binding = PulseBinding.from_dict(pair)
    report = audit_external_pulses(root, source, phase=1, binding=binding, first_event=FIRST + 1, last_event=FIRST + 1)
    assert report["complete"] and report["summary"]["capture_pulses_outside_range"] == 2
    assert report["input"]["anchor_csv_lines"] == {str(FIRST): 2, str(FIRST + 2): 4}
    trace = write_edges(tmp_path / "timing.csv", [*edges(FIRST, 1000), *edges(FIRST + 1, 2000), *edges(FIRST + 2, 3000)])
    timed = add_exposure_audit(report, trace, limits=ExposureLimits(200))
    assert timed["complete"] and timed["exposure_timing"]["summary"]["within_limits"] == 1
    daq_file(source, [FIRST, FIRST, FIRST + 1, FIRST + 2])
    with pytest.raises(CaptureError) as exc:
        audit_external_pulses(root, source, phase=1, binding=binding, first_event=FIRST + 1)
    assert exc.value.code == "PULSE_BINDING_DAQ"


def test_sparse_daq_file_reports_missing_id_intervals_without_expansion(tmp_path):
    span = 10**10
    root, pair = binding_fixture(tmp_path / "experiment", offsets=dict.fromkeys(CAMERAS, [-1, 0, span, span + 1]), end=span)
    source = daq_file(tmp_path / "daq.csv", (FIRST, FIRST + span))
    report = audit_external_pulses(root, source, phase=1, binding=PulseBinding.from_dict(pair))
    assert report["summary"]["matched"] == 2 and report["summary"]["daq_unlisted_pulses"] == span - 1
    assert not report["complete"] and report["association"]["daq_unlisted_ranges"] == [{"first_event": FIRST + 1, "last_event": FIRST + span - 1}]


@pytest.mark.parametrize(("case", "code"), [("reset", "PULSE_BINDING_COUNTER"), ("span", "PULSE_BINDING_COUNTER"),
                                           ("foreign", "PULSE_BINDING_SCOPE"), ("sequence", "PULSE_BINDING_FRAME"),
                                           ("digest", "PULSE_BINDING_FRAME"), ("range", "PULSE_BINDING_SCOPE"),
                                           ("cameras", "PULSE_BINDING_SCOPE")])
def test_fails_when_anchors_are_inconsistent_or_unverified(tmp_path, case, code):
    offsets = {CAMERAS[0]: [-1, 0, 2, 1, 5, 6], CAMERAS[1]: [-1, 0, 1, 5, 6]} if case == "reset" else None
    root, pair = binding_fixture(tmp_path / "experiment", offsets=offsets)
    kwargs = {}
    if case == "span":
        pair["last_event_number"] = FIRST + 4
    if case == "foreign":
        pair["experiment_id"] = "different-run"
    if case == "sequence":
        pair["camera_frames"][CAMERAS[0]] = [2, 6]
    if case == "digest":
        with ExperimentReader(root) as reader:
            frame = reader.frame_at_sequence(CAMERAS[0], 2)
        (root / frame["path"]).write_bytes(b"modified anchor")
    if case == "range":
        kwargs["last_event"] = FIRST + 6
    if case == "cameras":
        del pair["camera_frames"][CAMERAS[0]]
    with pytest.raises(CaptureError) as exc:
        audit_external_pulses(root, daq_file(tmp_path / "daq.csv", range(FIRST, FIRST + 6)), phase=1,
                             binding=PulseBinding.from_dict(pair), **kwargs)
    assert exc.value.code == code


def test_post_capture_binding_cannot_replace_saved_hardware_identities(tmp_path):
    root = pulse_fixture(tmp_path / "original", variations=False, start=FIRST)
    _, pair = binding_fixture(tmp_path / "different")
    with ExperimentReader(root) as reader:
        pair["experiment_id"] = reader.experiment["experiment_id"]
    source = daq_file(tmp_path / "daq.csv", (FIRST,))
    assert audit_external_pulses(root, source, phase=1)["complete"]
    with pytest.raises(CaptureError) as exc:
        audit_external_pulses(root, source, phase=1, binding=PulseBinding.from_dict(pair))
    assert exc.value.code == "PULSE_BINDING_SCOPE"


def test_cli_reuses_json_report_as_anchor_file_and_retains_csv_origin(tmp_path, capsys):
    root, pair = binding_fixture(tmp_path / "experiment", offsets={CAMERAS[0]: [-1, 0, None, 2, 3],
                                                                 CAMERAS[1]: [-1, 0, 1, 2, 3]}, end=2)
    source = daq_file(tmp_path / "daq.csv", range(FIRST, FIRST + 3))
    bind_path = tmp_path / "anchors.json"
    bind_path.write_text(json.dumps(pair), encoding="utf-8")
    report_path = tmp_path / "report.json"
    args = ["audit-pulses", str(root), "--source", str(source), "--phase", "1", "--anchor-file"]
    assert main([*args, str(bind_path), "--output", str(report_path)]) == 4
    assert json.loads(capsys.readouterr().out)["unassociated_frames"] == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["unassociated_frames"][0]["reason"] == "unusable_counter"
    table = tmp_path / "table.csv"
    assert main([*args, str(report_path), "--output", str(table)]) == 4
    with table.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["association_basis"] == "post_capture_counter_anchors" and rows[0]["event_number"] == str(FIRST)
