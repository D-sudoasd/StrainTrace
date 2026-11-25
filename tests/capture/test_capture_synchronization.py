import json
from dataclasses import replace

import numpy as np
import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureSession, ExperimentReader
from straintrace_capture.cameras.base import ClockReading, Device
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.image_inputs import decode_samples
from straintrace_capture.synchronization import ClockAnchor, LoadAlignment, ThresholdTrigger, capture_axis, link_detector
from straintrace_capture.trigger_review import read_trigger_review


class PulseCamera(SimulatedCamera):
    def __init__(self, identity):
        super().__init__(0 if identity.endswith("left") else 1)
        self.device = Device(identity, "pulse", "Test fixture", "Scripted pulse source", identity, "simulated")
        self.counter_values = iter([41, 42, 45] if identity.endswith("left") else [80, 82, 84])

    def start(self, trigger="free", source=None):
        assert (trigger, source) == ("hardware", "Line1")
        super().start()

    def read(self, timeout_s):
        image = super().read(timeout_s)
        if image is None:
            return None
        counter = next(self.counter_values)
        start = ClockReading(10100, "ticks", "sampled_exposure_start", self.device.id, "fixture_ticks",
                             uncertainty_ns=10, tick_frequency_hz=1000)
        end = replace(start, value=10300, source="sampled_exposure_end")
        return replace(image, exposure_start=start, exposure_end=end, metadata={"ChunkCounterValue": counter})


def _pulse_capture(root, factory=PulseCamera):
    names = ("pulse:left", "pulse:right")
    config = CaptureConfig.from_dict({
        "config_version": 1, "name": "Common-pulse fixture",
        "cameras": [{"id": camera, "settings": {"width": 96, "height": 80}} for camera in names],
        "phases": [{
            "mode": "hardware", "hardware_source": "Line1", "frame_count": 3,
            "hardware_pairing": {"source_id": "DAQ-A", "counter_feature": "ChunkCounterValue",
                                 "first_counter": dict(zip(names, (41, 80))), "first_event_number": 100},
        }],
    })
    registry = CameraRegistry(custom={"pulse": (lambda: [], factory)})
    capture = CaptureSession(config, root, registry=registry)
    capture.start()
    assert capture.wait(5)["state"] == "completed"
    return capture.output


def test_hardware_association_needs_explicit_id_and_preserves_missing_pulses(tmp_path):
    root = _pulse_capture(tmp_path / "fixture")
    with ExperimentReader(root) as reader:
        pairing = reader.pairing()
        indexed = {group["external_event_number"]: group for group in pairing["groups"]}
        assert pairing["hardware_counter_gap_ranges"] == [{"phase": 1, "first_event_number": 103, "last_event_number": 103}]
        assert indexed[100]["status"] == "associated"
        assert indexed[101]["missing_cameras"] == ["pulse:right"]
        assert indexed[102]["missing_cameras"] == ["pulse:left"]
        assert indexed[104]["status"] == "associated"
        verification = reader.verify()
        assert verification["committed_frames"] == 6 and not verification["complete"]
        snapshot = verification["hardware_groups"]
        assert snapshot["phase"] == 1 and not snapshot["accepting"]
        assert [row["number"] for row in snapshot["groups"]] == [100, 104, 102, 101]
        observed = {row["number"]: row for row in snapshot["groups"]}
        assert observed[100]["saved_cameras"] == 2 and not observed[100]["advanced"]
        assert observed[101]["advanced"] == ["pulse:right"] and observed[101]["saved_cameras"] == 1
        assert observed[102]["advanced"] == ["pulse:left"] and observed[102]["saved_cameras"] == 1
        frame = reader.frames(camera_id="pulse:left", limit=1)[0]
    source = tmp_path / "detector.csv"
    source.write_text("event_number,detector_frame\n100,cbf_01\n101,cbf_02\n103,cbf_03\n", encoding="utf-8")
    linked = link_detector(root, source, trigger_source="DAQ-A")["links"]
    assert {row["external_event_number"]: row["association"] for row in linked} == {
        100: "identity_match", 101: "partial_capture", 102: "capture_only",
        103: "detector_only", 104: "capture_only",
    }
    assert capture_axis(frame)["timestamp_unix_ns"] is None
    assert frame["timestamps"]["received_utc"]["source"] == "host_receive"
    assert frame["timestamps"]["exposure_midpoint"]["status"] == "derived"
    anchor = ClockAnchor.from_dict({
        "clock_id": "pulse:left", "unit": "ticks", "value_at_anchor": 10000, "unix_ns_at_anchor": 1800000000000000000,
        "units_per_second": 1000, "valid_min": 10000, "valid_max": 11000, "uncertainty_ns": 100,
        "evidence": "fixture independently specifies 1-ms ticks and the exact UTC anchor",
    })
    mechanics = tmp_path / "force.csv"
    mechanics.write_text("timestamp_unix_s,force_N,displacement_um\n1800000000,100,0\n1800000001,300,500\n", encoding="utf-8")
    align = LoadAlignment(mechanics, time_column="timestamp_unix_s", force_column="force_N",
                          anchors=(anchor,), mode="window_mean", displacement_column="displacement_um",
                          displacement_unit="um")
    result = align.at(frame)
    assert result["timestamp_unix_ns"] == 1800000000200000000
    assert result["time_uncertainty_ns"] == 110
    assert result["force_N"] == pytest.approx(140, abs=0.001)
    assert result["displacement_mm"] == pytest.approx(.1, abs=.001)
    assert result["force_support"]["status"] == "EXPOSURE_WINDOW_MEAN"
    align.max_gap_s = .5
    assert align.at(frame)["force_N"] is None


def test_trigger_review_pages_use_ledger_ids_and_leave_missing_images_empty(tmp_path):
    root = _pulse_capture(tmp_path / "paired")
    with ExperimentReader(root) as reader:
        equivalent = {row["event_id"]: row for row in reader.pairing()["groups"]}
        page = reader.trigger_group_page()
        assert len(page) == 4
        for row in page:
            assert all(row[key] == value for key, value in equivalent[row["event_id"]].items())
        recent = reader.trigger_group_page(limit=2)
        past = reader.trigger_group_page(limit=2, before=recent[-1]["ordinal"])
        assert recent + past == page
        assert reader.trigger_group_page(limit=2, after=past[0]["ordinal"], newest_first=False) == recent[::-1]
        gaps = reader.trigger_group_page(problems_only=True)
        assert {row["external_event_number"] for row in gaps} == {101, 102}
        target = gaps[-1]["event_id"]
        assert reader.trigger_group_page(through_event=target)[0]["event_id"] == target
        group, frames = reader.trigger_group(target)
        assert [row["trigger_event_id"] for row in frames] == [target]
        assert group["missing_cameras"] == ["pulse:left" if frames[0]["camera_id"] == "pulse:right" else "pulse:right"]
        intact = next(row["event_id"] for row in page if row["external_event_number"] == 100)
        displayed, records, images, errors, span = read_trigger_review(root, intact)
        assert displayed["status"] == "associated" and not errors
        assert set(images) == {"pulse:left", "pulse:right"}
        assert not np.array_equal(images["pulse:left"].pixels, images["pulse:right"].pixels)
        stamps = [rows[0]["timestamps"]["received_monotonic"]["value"] for rows in records.values()]
        assert span == max(stamps) - min(stamps)
        for camera, image in images.items():
            np.testing.assert_array_equal(image.pixels, decode_samples(reader.image_path(records[camera][0])), strict=True)

    group, _, images, errors, span = read_trigger_review(root, target)
    assert group["status"] == "incomplete" and not errors
    assert set(images) == set(group["expected_cameras"]) - set(group["missing_cameras"])
    assert span is None


def test_trigger_review_preserves_ambiguity_and_reports_a_corrupt_tiff(tmp_path):
    class RepeatCounter(PulseCamera):
        def __init__(self, identity):
            super().__init__(identity)
            if identity.endswith("left"):
                self.counter_values = iter([41, 41, 45])

    root = _pulse_capture(tmp_path / "duplicate", RepeatCounter)
    with ExperimentReader(root) as reader:
        conflicts = reader.trigger_group_page(problems_only=True)
        event = next(row for row in conflicts if row["external_event_number"] == 100)
        group, frames, images, errors, span = read_trigger_review(root, event["event_id"])
        assert group["status"] == "ambiguous" and group["ambiguous_cameras"] == ["pulse:left"]
        assert len(frames["pulse:left"]) == 2 and set(images) == {"pulse:right"} and span is None and not errors
        path = reader.image_path(frames["pulse:right"][0])
        original = path.read_bytes()
    path.write_bytes(original + b"bad")
    group, _, images, errors, _ = read_trigger_review(root, event["event_id"])
    assert group["status"] == "ambiguous" and not images
    assert "CAPTURE_IMAGE_CHANGED" in errors["pulse:right"]
    path.write_bytes(original)


def test_load_measurement_threshold_triggers_only_crossings():
    signals = []
    def request(source, payload):
        signals.append({"source": source, **payload})
        return "fixture-event"
    trigger = ThresholdTrigger(request, threshold=100, direction="rising", quantity="force", unit="N", source="DAQ")
    assert trigger.observe(120) is None
    assert trigger.observe(90) is None
    assert trigger.observe(float("nan")) is None
    assert trigger.observe(130) is None
    assert trigger.observe(80) is None
    clock = {"value": 89, "unit": "ticks", "clock_id": "daq"}
    assert trigger.observe(100, source_time=clock) == "fixture-event"
    assert trigger.observe(130) is None
    assert json.loads(json.dumps(signals))[0]["source_time"] == clock
