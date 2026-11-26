import json
import queue
import subprocess
import sys
from threading import Event
import time
from types import SimpleNamespace

import numpy as np
import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader
from straintrace_capture.cameras._genapi import GenApi
from straintrace_capture.cameras.base import Camera, Device, Image, TriggerSupport, TriggerWaitSample, receipt_times
from straintrace_capture.config import Phase
from straintrace_capture.control import CaptureControl, send_command
from straintrace_capture.hardware_readiness import WAIT_MAX_AGE_MS, hardware_readiness, wait_for_hardware
from straintrace_capture.presentation import format_integrity, hardware_trigger_guidance, phase_display, wait_output_help
from straintrace_capture.storage import ExperimentWriter
from straintrace_capture.trial_batch import run_trial_batch


class Node:
    def __init__(self, read, write=None, choices=()):
        self.read, self.write = read, write
        self.symbolics = self.Symbolics = choices
        self.fresh_reads = []

    @property
    def value(self):
        return self.read()

    @value.setter
    def value(self, data):
        self.write(data)

    Value = value

    def get_value(self, verify, ignore_cache):
        self.fresh_reads.append((verify, ignore_cache))
        return self.read() if ignore_cache else None

    GetValue = get_value


class Lines:
    def __init__(self, *, pylon=False, invert=False, unreadable=False):
        self.selected = "Line1"
        self.wait = False
        self.writes = []
        self.table = {
            "LineSelector": Node(lambda: self.selected, self.select, ("Line1", "Line2", "Line3")),
            "LineMode": Node(lambda: "Input" if self.selected == "Line1" else "Output"),
            "LineSource": Node(lambda: "FrameTriggerWait" if self.selected == "Line3" else "ExposureActive"),
            "LineStatus": Node(lambda: self.wait if self.selected == "Line3" else True),
            "LineInverter": Node(lambda: invert),
        }
        self.status = self.table["LineStatus"]
        if unreadable:
            self.status.read = lambda: None
        node_map = SimpleNamespace(GetNode=self.table.get, get_node=self.table.get)
        readable = lambda node: node is not None
        writable = lambda node: bool(node and node.write)
        self.api = (GenApi(node_map, pylon_api=SimpleNamespace(IsReadable=readable, IsWritable=writable)) if pylon else
                    GenApi(node_map, genicam_api=SimpleNamespace(is_readable=readable, is_writable=writable)))

    def select(self, line):
        self.writes.append(line)
        self.selected = line


@pytest.mark.parametrize("pylon", [False, True])
def test_reads_fresh_wait_output_without_reconfiguring_io(pylon):
    lines = Lines(pylon=pylon)
    with lines.api.trigger_wait_monitor() as sample:
        assert lines.selected == "Line3"
        assert sample() == TriggerWaitSample("Line3", False)
        lines.wait = True
        assert sample() == TriggerWaitSample("Line3", True)
    assert lines.selected == "Line1" and lines.writes == ["Line2", "Line3", "Line1"]
    assert lines.status.fresh_reads == [(False, True), (False, True)]


@pytest.mark.parametrize(("settings", "reason"), [({"invert": True}, "output_polarity_unconfirmed"),
                                                  ({"unreadable": True}, "line_status_unreadable")])
def test_uncertain_line_polarity_or_readback_cannot_report_ready(settings, reason):
    lines = Lines(**settings)
    lines.wait = True
    with lines.api.trigger_wait_monitor() as sample:
        assert sample() == TriggerWaitSample("Line3", reason=reason)
    assert lines.selected == "Line1"


def test_startup_state_separates_stale_signals_receipts_and_stop():
    now = 2_000_000_000
    recent = TriggerWaitSample("Line3", True), now - 10_000, .001
    older = TriggerWaitSample("Line3", True), now - (WAIT_MAX_AGE_MS + 1) * 1_000_000, .001
    samples = {"a": recent, "b": older}
    report = hardware_readiness(samples, {"a": 0, "b": 0}, active=True, now_ns=now)
    assert not report["ready"] and report["cameras"]["b"]["state"] == "stale"
    assert "1/2 台相机" in phase_display("hardware", {"event_window": None, "received_by_camera": {"a": 0, "b": 0},
                                                    "hardware_readiness": report})[1]
    assert "回读已过时" in wait_output_help(report, {"a": "左相机", "b": "右相机"})
    assert hardware_readiness({"a": recent}, {"a": 1}, active=True, now_ns=now)["state"] == "receiving"
    assert hardware_readiness({"a": recent}, {"a": 0}, active=False, now_ns=now)["state"] == "ending"

    progress = {"event_window": None, "received_by_camera": {"a": 0, "b": 0}}
    assert hardware_trigger_guidance(progress)[0] == "等待硬件触发"
    progress["hardware_readiness"] = report
    assert hardware_trigger_guidance(progress)[0] == "核对相机就绪状态"
    assert "1/2 台相机" in hardware_trigger_guidance(progress)[1]
    progress["received_by_camera"]["a"] = 1
    assert hardware_trigger_guidance(progress)[0] == "等待其余相机首帧"
    progress["received_by_camera"]["b"] = 1
    assert hardware_trigger_guidance(progress)[0] == "硬件触发采集中"


class HardwareFixture(Camera):
    def __init__(self, identity, *, unreadable=False):
        self.device = Device(identity, "wired", "Fixture", identity.rsplit(":", 1)[1], identity, "fixture")
        self.lines = Lines(unreadable=unreadable)
        self.frames = queue.Queue()

    def open(self):
        pass

    def close(self):
        self.stop()

    def features(self):
        return {}

    def set_feature(self, name, value):
        raise AssertionError(name)

    def trigger_support(self):
        return TriggerSupport(False, True, "fixture")

    def trigger_wait_monitor(self):
        return self.lines.api.trigger_wait_monitor()

    def start(self, trigger="free", source=None):
        assert (trigger, source) == ("hardware", "Line1") and self.lines.selected == "Line3"

    def stop(self):
        pass

    def read(self, timeout_s):
        try:
            counter = self.frames.get(timeout=timeout_s)
        except queue.Empty:
            return None
        utc, monotonic = receipt_times()
        return Image(np.full((12, 16), counter, dtype=np.uint8), "Mono8", 8, "mono", utc, monotonic, counter, counter)


def hardware_fixture(*, unreadable=False):
    devices = {key: HardwareFixture(key, unreadable=unreadable and key.endswith("right")) for key in ("wired:left", "wired:right")}
    registry = CameraRegistry(custom={"wired": (lambda: [cam.device for cam in devices.values()], devices.__getitem__)})
    plan = CaptureConfig.from_dict({"config_version": 1, "name": "Hardware wait fixture", "cameras": [{"id": key} for key in devices],
                                    "phases": [{"mode": "hardware", "hardware_source": "Line1", "frame_count": 2,
                                                "monitor_trigger_wait": True}]})
    return plan, registry, devices


def wait_fixture(check, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result := check():
            return result
        time.sleep(.02)
    raise AssertionError("timed out waiting for hardware fixture")


def observed_state(session, camera, expected):
    progress = session.status()["phase_progress"]
    return progress and progress["hardware_readiness"]["cameras"][camera]["state"] == expected


def test_guided_hardware_stage_waits_for_physical_input_from_the_driver(tmp_path):
    plan, registry, devices = hardware_fixture()
    for camera in devices.values():
        camera.lines.wait = True
    triggered, reports = [], []

    def observe(live):
        if live is None:
            return
        assert live.action is None
        progress = live.status.get("phase_progress") or {}
        readiness = progress.get("hardware_readiness") or {}
        if readiness.get("ready") and not triggered:
            assert set(progress["hardware_inputs"].values()) == {"Line1"}
            triggered.append(live.session.output)
            for camera in devices.values():
                camera.frames.put(1)
                camera.frames.put(2)

    end = run_trial_batch(plan, tmp_path / "正式输出", (1,), 4, registry=registry, discovered_ids=set(devices), stop=Event(),
                          watch_quality=False, progress=lambda *_args: None, opened=lambda *_args: None,
                          completed=lambda _phase, _state, report: reports.append(report), guided=True, observe=observe)
    assert end.state == "completed", end
    assert len(triggered) == len(reports) == 1 and reports[0]["verification"]["committed_images"] == 4
    with ExperimentReader(triggered[0]) as reader:
        assert not reader.events(kind="signal_received")
        assert len(reader.events(kind="hardware_wait_observations")) == 1


def test_hardware_wait_cli_follows_both_cameras_and_closes_on_first_frame(tmp_path):
    plan, registry, devices = hardware_fixture()
    left, right = devices.values()
    left.lines.wait = True
    run = CaptureSession(plan, tmp_path / "run", registry=registry)
    run.start()
    try:
        with CaptureControl(run):
            wait_fixture(lambda: observed_state(run, "wired:right", "not_ready"))
            assert run.status()["phase_progress"]["hardware_inputs"] == dict.fromkeys(devices, "Line1")
            result = wait_for_hardware(run.output, 1, .55)
            assert not result["ok"] and result["error"]["code"] == "WAIT_HARDWARE_TIMEOUT"
            assert result["hardware_readiness"]["cameras"]["wired:left"]["state"] == "ready"
            with pytest.raises(CaptureError) as error:
                wait_for_hardware(run.output, 2, 3)
            assert error.value.code == "WAIT_HARDWARE_PHASE_INVALID"

            right.lines.wait = True
            cli = subprocess.run([sys.executable, "-m", "straintrace_capture", "wait-hardware", str(run.output),
                                  "--phase", "1", "--timeout", "4"], capture_output=True, text=True, timeout=10)
            report = json.loads(cli.stdout)
            assert cli.returncode == 0 and report["ok"] and report["hardware_readiness"]["ready"], cli.stderr
            assert set(report["hardware_readiness"]["cameras"]) == set(devices)

            left.frames.put(1)
            wait_fixture(lambda: observed_state(run, "wired:left", "received"))
            with pytest.raises(CaptureError) as error:
                wait_for_hardware(run.output, 1, 3)
            assert error.value.code == "WAIT_HARDWARE_FIRST_FRAME_RECEIVED"
            left_polls = len(left.lines.status.fresh_reads)
            for camera in devices.values():
                if camera is right:
                    camera.frames.put(1)
                camera.frames.put(2)
            assert run.wait(8)["state"] == "completed"
            assert len(left.lines.status.fresh_reads) == left_polls
    finally:
        if not run.finished:
            run.request_stop()
            run.wait(8)
    assert all(camera.lines.selected == "Line1" for camera in devices.values())
    assert not list((run.output / "control").iterdir())
    with ExperimentReader(run.output) as reader:
        report = reader.verify()
        assert report["complete"]
        assert "Line3 · 读到可接收触发 · 收图 2 张" in format_integrity(report)
        observation, = reader.events(kind="hardware_wait_observations")
        assert observation["phase"] == 1 and len(observation["cameras"]) == 2
        assert all(record["first_asserted_poll_monotonic_ns"] for record in observation["cameras"].values())
    with pytest.raises(CaptureError) as error:
        wait_for_hardware(run.output, 1, 3)
    assert error.value.code == "WAIT_HARDWARE_PHASE_ENDED"


def test_wait_hardware_requires_camera_evidence_after_phase_confirmation(tmp_path):
    initial, registry, devices = hardware_fixture()
    data = initial.to_dict()
    data["phases"][0]["start_confirmation"] = "核对触发源后启动相机。"
    for camera in devices.values():
        camera.lines.wait = True
    run = CaptureSession(CaptureConfig.from_dict(data), tmp_path / "confirmation", registry=registry)
    run.start()
    try:
        with CaptureControl(run):
            def waiting():
                gate = run.status().get("phase_confirmation")
                return gate if gate and gate["state"] == "waiting" else None

            gate = wait_fixture(waiting)
            result = wait_for_hardware(run.output, 1, .35)
            assert not result["ok"] and result["error"]["code"] == "WAIT_HARDWARE_TIMEOUT"
            assert result["hardware_readiness"] is None
            assert all(not camera.lines.status.fresh_reads for camera in devices.values())

            run.request_phase_confirmation(1, confirmation_id=gate["confirmation_id"])
            assert wait_for_hardware(run.output, 1, 4)["ok"]
            for camera in devices.values():
                camera.frames.put(1)
                camera.frames.put(2)
            assert run.wait(8)["state"] == "completed"
    finally:
        if not run.finished:
            run.request_stop()
            run.wait(8)
    with ExperimentReader(run.output) as reader:
        assert reader.verify()["committed_frames"] == 4


def test_wait_monitor_observes_unavailable_output_without_blocking_capture(tmp_path):
    plan, registry, devices = hardware_fixture(unreadable=True)
    for camera in devices.values():
        camera.lines.wait = True
    run = CaptureSession(plan, tmp_path / "unverified", registry=registry)
    run.start()
    try:
        with CaptureControl(run):
            wait_fixture(lambda: observed_state(run, "wired:right", "unavailable"))
            result = wait_for_hardware(run.output, 1, .55)
            assert not result["ok"] and result["hardware_readiness"]["state"] == "unverified"
            assert result["hardware_readiness"]["cameras"]["wired:right"]["reason"] == "line_status_unreadable"
            for camera in devices.values():
                camera.frames.put(1)
                camera.frames.put(2)
            assert run.wait(8)["state"] == "completed"
    finally:
        if not run.finished:
            run.request_stop()
            run.wait(8)
    assert not list((run.output / "control").iterdir())


def test_monitor_trigger_wait_is_only_a_hardware_phase_option():
    with pytest.raises(CaptureError, match="applies only to hardware"):
        Phase.from_dict({"mode": "single", "monitor_trigger_wait": True})
    assert "monitor_trigger_wait" not in Phase.from_dict({"mode": "hardware", "hardware_source": "Line1"}).to_dict()


def test_ready_reply_can_expire_while_on_the_operator_channel(tmp_path, monkeypatch):
    from straintrace_capture import control
    from straintrace_capture import hardware_readiness as module

    plan, _, devices = hardware_fixture()
    root = tmp_path / "waiting"
    owner = ExperimentWriter(root, plan)
    owner.close()

    class Clock:
        current = 50.

        def perf_counter(self):
            return self.current

        def sleep(self, seconds):
            self.current += seconds

    clock = Clock()

    def delayed_status(_root, command, *, timeout):
        assert command == "status" and timeout > 0
        clock.current += .08
        return {"ok": True, "status": {"experiment_id": owner.experiment_id, "state": "running", "phase": 1,
                "phase_progress": {"hardware_readiness": {"ready": True, "state": "ready",
                "cameras": {camera: {"state": "ready", "reason": None, "age_ms": 700.} for camera in devices}}}}}

    monkeypatch.setattr(module, "time", clock)
    monkeypatch.setattr(control, "send_command", delayed_status)
    reply = wait_for_hardware(root, 1, .125)
    assert not reply["ok"] and reply["error"]["code"] == "WAIT_HARDWARE_TIMEOUT"


def test_status_timeout_cancels_a_claimed_poll(tmp_path):
    plan, _, _ = hardware_fixture()
    root = tmp_path / "control-timeout"
    owner = ExperimentWriter(root, plan)
    owner.close()
    entered, release = Event(), Event()

    def delayed_status():
        entered.set()
        assert release.wait(3)
        return {"experiment_id": owner.experiment_id}

    session = SimpleNamespace(output=root, status=delayed_status)
    with CaptureControl(session):
        try:
            with pytest.raises(CaptureError) as error:
                send_command(root, "status", timeout=.5)
            assert entered.is_set() and error.value.code == "CONTROL_TIMEOUT"
        finally:
            release.set()
    assert not list((root / "control").iterdir())
