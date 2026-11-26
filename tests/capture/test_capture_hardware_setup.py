from types import SimpleNamespace

import numpy as np
import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader, inspect_plan
from straintrace_capture.cameras._genapi import GenApi
from straintrace_capture.cameras.base import Camera, Device, Feature, Image, TriggerSupport, receipt_times
from straintrace_capture.hardware_counters import summarize_hardware_counters
from straintrace_capture.presentation import format_readiness, format_trial
from straintrace_capture.pulse_audit import audit_external_pulses
from straintrace_capture.trial import prepare_trial, trial_report

from test_capture_hardware_readiness import Node


LEFT, RIGHT = "pylon:wiring-fixture", "gentl:wiring-fixture"
PORTS = {LEFT: ("Line1", "Line2"), RIGHT: ("CxpTrigger0", "CXPTrigger1")}
ANCHORS = {LEFT: 2**63 + 7, RIGHT: 2**63 + 33}
EVENT = 2**63 + 98


class WiringFixture(Camera):
    def __init__(self, identity, counters, starts):
        backend = identity.partition(":")[0]
        self.device = Device(identity, backend, "Fixture", f"硬件试验 {backend}", backend, "simulated")
        self.counters, self.starts = counters, starts
        self.port = None

    def open(self):
        pass

    def close(self):
        self.stop()

    def features(self):
        return {"trigger_source": Feature("trigger_source", self.port, choices=PORTS[self.device.id]),
                "trigger_activation": Feature("trigger_activation", "RisingEdge")}

    def set_feature(self, name, value):
        raise AssertionError((name, value))

    def trigger_support(self):
        return TriggerSupport(False, True, "fixture", physical_inputs=PORTS[self.device.id])

    def start(self, trigger="free", source=None):
        assert trigger == "hardware" and source in PORTS[self.device.id]
        self.starts.append((self.device.id, source))
        self.port = source
        self.samples = iter(enumerate(self.counters[self.device.id], 1))

    def stop(self):
        pass

    def read(self, _timeout_s):
        if (sample := next(self.samples, None)) is None:
            return None
        sequence, value = sample
        utc, mono = receipt_times()
        image = np.arange(192, dtype=np.uint8).reshape(12, 16) + sequence
        return Image(image, "Mono8", 8, "mono", utc, mono, sequence, sequence,
                     metadata={"ChunkCounterValue": value})


def wiring_fixture():
    samples = {camera: [base + i for i in range(3)] for camera, base in ANCHORS.items()}
    starts = []
    devices = {camera: WiringFixture(camera, samples, starts) for camera in PORTS}
    registry = CameraRegistry(custom={camera.partition(":")[0]:
                                      (lambda value=device: [value.device], devices.__getitem__)
                                      for camera, device in devices.items()})
    raw = {"config_version": 1, "name": "Shared pulse fixture", "cameras": [{"id": camera} for camera in devices],
           "phases": [{"mode": "hardware", "hardware_source": {key: choices[0] for key, choices in PORTS.items()},
                       "frame_count": 3, "hardware_pairing": {"source_id": "DAQ-A", "counter_feature": "ChunkCounterValue",
                                                               "first_counter": ANCHORS, "first_event_number": EVENT}}]}
    return raw, registry, devices, samples, starts


@pytest.mark.parametrize("pylon", [False, True])
def test_input_catalog_is_read_only_and_only_applies_to_current_framestart(pylon):
    values = {"TriggerSelector": "AcquisitionStart", "TriggerMode": "Off", "TriggerSource": "Software"}
    writes = []

    def write(name, value):
        writes.append((name, value))
        values[name] = value

    choices = {"TriggerSelector": ("FrameStart", "AcquisitionStart"), "TriggerMode": ("Off", "On"),
               "TriggerSource": ("Software", "Line1", "CC1", "CxpTrigger0", "CXPTrigger1", "Counter1Active")}
    table = {name: Node(lambda key=name: values[key], lambda value, key=name: write(key, value), items)
             for name, items in choices.items()}
    table["TriggerSoftware"] = Node(lambda: None)
    node_map = SimpleNamespace(GetNode=table.get, get_node=table.get,
                               GetNodes=lambda: [SimpleNamespace(GetName=lambda key=name: key) for name in table],
                               nodes=[SimpleNamespace(node=SimpleNamespace(name=name)) for name in table])
    readable, writable = lambda node: node is not None, lambda node: node is not None and node.write is not None
    api = (GenApi(node_map, pylon_api=SimpleNamespace(IsReadable=readable, IsWritable=writable)) if pylon else
           GenApi(node_map, genicam_api=SimpleNamespace(is_readable=readable, is_writable=writable)))
    assert api.trigger_support().physical_inputs is None and not writes
    values["TriggerSelector"] = "FrameStart"
    assert api.trigger_support().physical_inputs == ("Line1", "CC1", "CxpTrigger0", "CXPTrigger1")
    values["TriggerMode"] = "On"
    assert api.trigger_support().physical_inputs is None
    assert not writes
    for source in ("CxpTrigger0", "CXPTrigger1"):
        api.trigger_setup("hardware", source)
        assert values["TriggerSource"] == source and values["TriggerMode"] == "On"
    with pytest.raises(CaptureError) as exc:
        api.trigger_setup("hardware", "Software")
    assert exc.value.code == "TRIGGER_SOURCE_UNSUPPORTED"


def test_preflight_reports_the_incompatible_camera_and_resolved_inputs(tmp_path):
    raw, _, devices, _, _ = wiring_fixture()
    support = {camera: device.trigger_support() for camera, device in devices.items()}
    report = inspect_plan(CaptureConfig.from_dict(raw), tmp_path / "run", trigger_support=support)
    assert report["ready"] and report["phases"][0]["hardware_inputs"] == {LEFT: "Line1", RIGHT: "CxpTrigger0"}
    raw["phases"][0]["hardware_source"] = "Line1"
    report = inspect_plan(CaptureConfig.from_dict(raw), tmp_path / "run", trigger_support=support)
    issue, = (item for item in report["issues"] if item["severity"] == "block")
    assert issue == {"code": "HARDWARE_INPUT_UNAVAILABLE", "severity": "block", "phase": 1, "camera": RIGHT,
                     "input": "Line1", "available": list(PORTS[RIGHT])}
    assert "当前目录 CxpTrigger0、CXPTrigger1" in format_readiness(report)
    raw["phases"][0]["hardware_source"] = "Software"
    report = inspect_plan(CaptureConfig.from_dict(raw), tmp_path / "run")
    assert {item["camera"] for item in report["issues"] if item["code"] == "HARDWARE_INPUT_INVALID"} == {LEFT, RIGHT}
    assert not (tmp_path / "run").exists()


def test_trial_observes_actual_counters_then_formal_plan_matches_exact_daq_ids(tmp_path):
    raw, registry, _, samples, starts = wiring_fixture()
    config = CaptureConfig.from_dict(raw)
    trial_plan, trial_path = prepare_trial(config, tmp_path / "formal", phase_number=1)
    assert trial_plan.phases[0].hardware_pairing is None
    samples[LEFT] = [2**63 + 100, 2**63 + 101, 2**63 + 103]
    samples[RIGHT] = [51, 51, 50]
    trial = CaptureSession(trial_plan, trial_path, registry=registry)
    trial.start()
    assert trial.wait(8)["state"] == "completed"
    report = trial_report(trial_path)
    assert report["observed"] and report["verification"]["committed_frames"] == 6
    left = report["cameras"][LEFT]["hardware_counters"]["ChunkCounterValue"]
    assert left == {"frames": 3, "first": {"sequence": 1, "value": 2**63 + 100},
                    "last": {"sequence": 3, "value": 2**63 + 103}, "duplicates": 0, "decreases": 0, "skipped_values": 1}
    right = report["cameras"][RIGHT]["hardware_counters"]["ChunkCounterValue"]
    assert (right["duplicates"], right["decreases"]) == (1, 1)
    text = format_trial(report)
    assert "触发沿 RisingEdge" in text and "跳过 1 个计数值" in text and "正式实验的已知共享脉冲" in text
    with ExperimentReader(trial_path) as reader:
        assert not reader.pairing()["groups"]

    samples.update({key: [value + i for i in range(3)] for key, value in ANCHORS.items()})
    formal = CaptureSession(config, tmp_path / "formal", registry=registry)
    formal.start()
    assert formal.wait(8)["state"] == "completed"
    csv_path = tmp_path / "pulses.csv"
    csv_path.write_text("event_number\n" + "".join(f"{EVENT + offset}\n" for offset in range(3)), encoding="utf-8")
    audit = audit_external_pulses(formal.output, csv_path, phase=1)
    assert audit["complete"] and audit["summary"]["matched"] == 3
    assert sorted(starts) == sorted([(LEFT, "Line1"), (RIGHT, "CxpTrigger0")] * 2)


def test_counter_summary_preserves_gaps_without_joining_across_missing_samples():
    rows = [{"sequence": i, "backend_metadata": {"ChunkFrameID": value}} for i, value in enumerate((7, True, 30, 34, -1), 1)]
    feature = summarize_hardware_counters(rows)["ChunkFrameID"]
    assert feature["frames"] == 3 and feature["first"]["value"] == 7 and feature["last"]["value"] == 34
    assert feature["skipped_values"] == 3 and not feature["duplicates"] and not feature["decreases"]
