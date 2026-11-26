import csv
import json

import cv2
import numpy as np
import pytest

from ezdic_cli import main as analysis_cli
from straintrace_capture import CaptureConfig, CaptureError, ExperimentReader, SequenceSelection, export_sequence, verify_sequence
from straintrace_capture.cameras.base import Device, Image
from straintrace_capture.cli import main as capture_cli
from straintrace_capture.handoff import analysis_paths, analysis_snapshot
from straintrace_capture.storage import ExperimentWriter


FIRST = "81412f6c-5689-4c77-8624-12459ce73a01"
SECOND = "81412f6c-5689-4c77-8624-12459ce73a02"
REFERENCE_EVENT = "81412f6c-5689-4c77-8624-12459ce73a03"
CAMERA = "sim:0"
UTC = 1_795_570_200_123_456_789
MONO = 790_059_000_000


def event_experiment(root, *, with_reference=True, post_only=False, offset=None, reference_event=False,
                     sensor_offsets=None, changed_shape=False):
    phases = [
        {"mode": "single", "role": "reference" if with_reference else "test"},
        {"mode": "event", "fps": 4, "pre_frames": 0 if post_only else 1, "post_frames": 2, "event_count": 2},
        {"mode": "timed", "fps": 4, "frame_count": 1},
    ]
    if reference_event:
        phases[0] = {"mode": "event", "role": "reference", "fps": 4, "post_frames": 1}
    config = CaptureConfig.from_dict({"config_version": 1, "name": "事件 DIC", "cameras": [{"id": CAMERA}], "phases": phases})
    texture = cv2.GaussianBlur(np.random.default_rng(65).integers(20, 220, (144, 176), dtype=np.uint8), (5, 5), .85)
    records = [(1, "reference" if with_reference else "test", REFERENCE_EVENT if reference_event else None,
                "post" if reference_event else None, 0),
               (2, "test", FIRST, "pre", 5), (2, "test", FIRST, "post", 8),
               (2, "test", SECOND, "post" if post_only else "pre", 2),
               (2, "test", SECOND, "post" if post_only else "boundary", 3),
               (2, "test", SECOND, "post", 4), (3, "test", None, None, -7)]
    writer = ExperimentWriter(root, config)
    try:
        writer.register_camera(Device(CAMERA, "sim", "Simulated", "Sim", "0", "fixture"), {})
        offsets = sensor_offsets if sensor_offsets is not None else ({1: 0, 2: offset} if offset is not None else {})
        if offsets:
            for phase, phase_offset in offsets.items():
                writer.event("phase_started", {"phase": phase, "camera_features": {
                    CAMERA: {"offset_x": {"value": phase_offset}},
                }})
        for number, (context, tick) in enumerate(((FIRST, 25), (SECOND, 35 if post_only else 45)), 1):
            writer.event("signal_received", {"phase": 2, "id": context, "source": "operator", "payload": {},
                                             "received_utc_ns": UTC + tick, "received_monotonic_ns": MONO + tick,
                                             "event_number": number}, event_id=context)
        if reference_event:
            writer.event("signal_received", {"phase": 1, "id": REFERENCE_EVENT, "source": "operator", "payload": {},
                                             "received_utc_ns": UTC + 5, "received_monotonic_ns": MONO + 5,
                                             "event_number": 1}, event_id=REFERENCE_EVENT)
        frames = []
        for seq, (phase, role, event_id, position, dx) in enumerate(records, 1):
            pixels = cv2.warpAffine(texture, np.float32([[1, 0, dx], [0, 1, 0]]),
                                    (176, 144), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            if changed_shape and seq == len(records):
                pixels = pixels[:, :-1].copy()
            image = Image(pixels, "Mono8", 8, "mono", UTC + seq * 10, MONO + seq * 10)
            frames.append(writer.prepare_frame(CAMERA, seq, phase, role, image, None, event_id, position, 0))
        writer.commit_frames(frames)
        writer.set_state("completed")
    finally:
        writer.close()
    return root


def test_event_does_not_include_other_events_and_runs_the_known_displacement(tmp_path, capsys):
    source = event_experiment(tmp_path / "source")
    expected = analysis_snapshot(source, event_context_id=SECOND)
    assert expected["reference_selection"] == "capture_reference_role"
    assert [row["sequence"] for row in expected["frames"]] == [1, 4, 5, 6]
    assert expected["event"]["signal_utc_ns"] == UTC + 45
    local = analysis_snapshot(source, reference_sequence=4, event_context_id=SECOND)
    assert [row["sequence"] for row in local["frames"]] == [4, 5, 6]
    for options, error in (({"reference_sequence": 2}, "EVENT_REFERENCE_OUTSIDE_SCOPE"),
                           ({"reference_sequence": 6}, "TOO_FEW_FRAMES"), ({"expected_id": "other"}, "REVIEW_SCOPE_CHANGED")):
        with pytest.raises(CaptureError) as bad:
            analysis_snapshot(source, event_context_id=SECOND, **options)
        assert bad.value.code == error

    template = tmp_path / "template.json"
    template.write_text(json.dumps({"schema_version": 1, "analysis_mode": "fullfield",
                                    "output_dir": str(tmp_path / "results"), "field_roi": [32, 28, 96, 84],
                                    "solver": {"subset_size_px": 21, "step_px": 14}}), encoding="utf-8")
    snapshot, config = tmp_path / "snapshot.json", tmp_path / "analysis.json"
    assert capture_cli(["prepare-analysis", str(source), "--event-context", SECOND, "--reference-sequence", "4",
                        "--output", str(snapshot), "--template", str(template), "--run-config", str(config)]) == 0
    capsys.readouterr()
    assert analysis_cli(["run", "--config", str(config)]) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "results/run_manifest.json").read_text(encoding="utf-8"))
    provenance = manifest["config"]["capture_source"]
    assert provenance["event"] == local["event"]
    assert [row["event_position"] for row in provenance["frames"]] == ["pre", "boundary", "post"]
    assert [row["sha256"] for row in manifest["inputs"]] == [row["sha256"] for row in local["frames"]]
    results = []
    for path in (tmp_path / "results").rglob("*.npz"):
        with np.load(path) as details:
            if "u" in details:
                results.append(np.nanmedian(details["u"]))
    observed = {round(float(value), 2) for value in results if np.isfinite(value)}
    assert {1.0, 2.0} <= observed, observed


def test_event_export_preserves_reference_and_positions_in_a_standalone_sequence(tmp_path, capsys):
    source = event_experiment(tmp_path / "source")
    target = tmp_path / "sequence"
    assert capture_cli(["export-sequence", str(source), "--event-context", SECOND, "--output", str(target)]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["event"]["signal_id"] == SECOND and manifest["test_frames"] == 3
    assert manifest["reference_selection"] == "capture_reference_role"
    with (target / "frames.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["source_sequence"]) for row in rows] == [1, 4, 5, 6]
    assert [row["event_position"] for row in rows] == ["", "pre", "boundary", "post"]
    assert [row["event_context_id"] for row in rows] == ["", SECOND, SECOND, SECOND]
    with ExperimentReader(source, immutable=True) as original:
        for row in rows:
            frame = original.frame_at_sequence(CAMERA, int(row["source_sequence"]))
            assert (target / row["file"]).read_bytes() == original.image_path(frame).read_bytes()
    with pytest.raises(CaptureError) as invalid:
        export_sequence(source, tmp_path / "different", SequenceSelection(phase=3, event_context_id=SECOND))
    assert invalid.value.code == "EVENT_IMAGES_SCOPE"
    source.rename(tmp_path / "archived")
    assert verify_sequence(target)["ok"]


def test_event_baseline_fallback_and_geometry_guard(tmp_path):
    source = event_experiment(tmp_path / "local", with_reference=False)
    assert analysis_snapshot(source, event_context_id=SECOND)["reference_selection"] == "first_event_pre_frame"
    source = event_experiment(tmp_path / "post-only", with_reference=False, post_only=True)
    with pytest.raises(CaptureError) as choose:
        analysis_snapshot(source, event_context_id=SECOND)
    assert choose.value.code == "EVENT_REFERENCE_REQUIRED"
    explicit = analysis_snapshot(source, reference_sequence=4, event_context_id=SECOND)
    assert [row["sequence"] for row in explicit["frames"]] == [4, 5, 6]
    source = event_experiment(tmp_path / "offset", offset=8)
    with pytest.raises(CaptureError) as changed:
        analysis_snapshot(source, event_context_id=SECOND)
    assert changed.value.code == "SEQUENCE_SENSOR_CHANGED"


def test_regular_handoff_rejects_a_changed_recorded_sensor_window(tmp_path):
    source = event_experiment(tmp_path / "offset", offset=8)
    with pytest.raises(CaptureError) as changed:
        analysis_paths(analysis_snapshot(source))
    assert changed.value.code == "SEQUENCE_SENSOR_CHANGED"

    # An absent reference readback must not mask a mismatch between two later phases.
    source = event_experiment(tmp_path / "unknown-reference", sensor_offsets={2: 8, 3: 12})
    with pytest.raises(CaptureError) as changed:
        analysis_paths(analysis_snapshot(source))
    assert changed.value.code == "SEQUENCE_SENSOR_CHANGED"

    source = event_experiment(tmp_path / "shape", changed_shape=True)
    with pytest.raises(CaptureError) as changed:
        analysis_paths(analysis_snapshot(source))
    assert changed.value.code == "FRAME_GEOMETRY_CHANGED"


def test_reference_from_an_earlier_reference_event_keeps_its_own_provenance(tmp_path):
    source = event_experiment(tmp_path / "source", reference_event=True)
    target = tmp_path / "images"
    report = export_sequence(source, target, SequenceSelection(event_context_id=SECOND))
    assert report["reference_sequence"] == 1 and verify_sequence(target)["ok"]
    with (target / "frames.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["source_role"] == "reference" and rows[0]["event_context_id"] == REFERENCE_EVENT
    assert [row["event_context_id"] for row in rows[1:]] == [SECOND] * 3
