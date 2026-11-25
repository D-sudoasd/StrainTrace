import json
from dataclasses import replace

import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureSession, ExperimentReader
from straintrace_capture.cameras.base import ClockReading, Device
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.synchronization import ClockAnchor, LoadAlignment, ThresholdTrigger, capture_axis, link_detector


class PulseCamera(SimulatedCamera):
    def __init__(self, identity):
        super().__init__(0)
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


def test_hardware_association_needs_explicit_id_and_preserves_missing_pulses(tmp_path):
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
    registry = CameraRegistry(custom={"pulse": (lambda: [], PulseCamera)})
    capture = CaptureSession(config, tmp_path / "fixture", registry=registry)
    capture.start()
    assert capture.wait(5)["state"] == "completed"
    with ExperimentReader(capture.output) as reader:
        pairing = reader.pairing()
        indexed = {group["external_event_number"]: group for group in pairing["groups"]}
        assert pairing["hardware_counter_gap_ranges"] == [{"phase": 1, "first_event_number": 103, "last_event_number": 103}]
        assert indexed[100]["status"] == "associated"
        assert indexed[101]["missing_cameras"] == ["pulse:right"]
        assert indexed[102]["missing_cameras"] == ["pulse:left"]
        assert indexed[104]["status"] == "associated"
        assert reader.verify()["committed_frames"] == 6
        assert not reader.verify()["complete"]
        frame = reader.frames(camera_id="pulse:left", limit=1)[0]
    source = tmp_path / "detector.csv"
    source.write_text("event_number,detector_frame\n100,cbf_01\n101,cbf_02\n103,cbf_03\n", encoding="utf-8")
    linked = link_detector(capture.output, source, trigger_source="DAQ-A")["links"]
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
