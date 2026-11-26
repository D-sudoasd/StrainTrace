import queue
import threading
import time

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureSession, ExperimentReader
from straintrace_capture.cameras.base import TriggerSupport
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.config import HardwarePairing
from straintrace_capture.control import CaptureControl, send_command
from straintrace_capture.live_pulses import LivePulseGroups, RECENT_PULSE_GROUPS
from straintrace_capture.monitor_gui import camera_monitor_rows
from straintrace_capture.pulse_monitor_gui import pulse_notice
from straintrace_capture.storage import ExperimentWriter
from straintrace_capture.write_timing import LiveWriteTiming


def _until(check, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := check():
            return value
        time.sleep(.012)
    raise AssertionError("timed out waiting for capture monitoring evidence")


def _plan(*phases):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "Recording monitor",
        "cameras": [{"id": f"sim:{index}", "settings": {"width": 120, "height": 88, "pixel_format": "Mono12"}}
                    for index in range(2)],
        "phases": list(phases),
    })


def _write_observation(monitor, second, *, pending, queued, prebuffer=0, running=True):
    args = {"pending": pending, "buffered": pending + prebuffer, "queued": queued}
    if running:
        monitor.observe(second, **args)
    return monitor.status(running=running, failed=False, **args)


def _pulse_groups(first=9007199254740993):
    return LivePulseGroups(2, HardwarePairing.from_dict({
        "source_id": "DAQ", "counter_feature": "ChunkCounterValue", "first_event_number": first,
        "first_counter": {"left": 100, "right": 800},
    }))


def test_live_pulses_separate_waiting_advanced_duplicate_and_unidentified_frames():
    monitor = _pulse_groups()
    first = monitor.first
    monitor.receive("left", first)
    view = monitor.snapshot(accepting=True)
    assert view["groups"][0]["number"] == first and view["groups"][0]["received_cameras"] == 1
    assert view["groups"][0]["saved_cameras"] == 0 and pulse_notice(view) is None

    monitor.receive("right", first + 1)
    monitor.commit("left", first)
    state = monitor.snapshot(accepting=True)
    assert state["groups"][0]["advanced"] == ["right"] and state["groups"][0]["saved_cameras"] == 1
    assert pulse_notice(state)[0] == "脉冲组需复核"

    monitor.receive("right", first)
    monitor.receive("right", first)
    monitor.receive("left", None)
    state = monitor.snapshot(accepting=True)
    assert not state["groups"][0]["advanced"] and state["groups"][0]["duplicates"] == ["right"]
    assert state["groups"][0]["cameras"]["right"] == {"received": 2, "saved": 0}
    assert state["cameras"]["right"]["backtracks"] == 1
    assert state["cameras"]["left"]["unidentified"] == 1
    assert pulse_notice(state)[0] == "核对脉冲编号"
    assert pulse_notice(monitor.snapshot(accepting=False)) is None


def test_live_pulses_keep_configured_start_and_bound_recent_numbers():
    monitor = _pulse_groups(first=0)
    for number in range(2, RECENT_PULSE_GROUPS + 8):
        monitor.receive("left", number)
    monitor.receive("right", 2)
    monitor.receive("right", 0)
    monitor.commit("right", 0)
    view = monitor.snapshot(accepting=False)
    first, *recent = view["groups"]
    assert first["number"] == 0 and first["saved_cameras"] == 1
    assert [row["number"] for row in recent] == list(range(RECENT_PULSE_GROUPS + 7, 7, -1))
    assert view["cameras"]["right"]["highest_number"] == 2
    assert view["cameras"]["right"]["latest_number"] == 0


def test_live_write_trend_waits_for_two_windows_and_uses_the_first_limit():
    mib = 2**20
    monitor = LiveWriteTiming(16, 96 * mib, 40)
    monitor.begin(1, "timed", 0.0)
    _write_observation(monitor, 0, pending=0, queued=0)
    for second in range(1, 5):
        monitor.accept(8 * mib, 24)
        monitor.commit(4 * mib, 12)
        observation = _write_observation(monitor, second, pending=4 * mib * second, queued=3 * second)
        if second == 2:
            assert observation["state"] == "growing" and observation["estimate"] is None
    assert observation["state"] == "sustained"
    assert len(observation["windows"]) == 2
    assert [window["committed_raw_bytes_per_s"] for window in observation["windows"]] == [4 * mib, 4 * mib]
    assert observation["estimate"] == {"seconds": (40 - 12) / 3, "resource": "batches", "growth_per_s": 3}
    alert, = monitor.alerts()
    assert alert["phase"] == 1 and alert["phase_elapsed_s"] == 4

    draining = _write_observation(monitor, 4.1, pending=16 * mib, queued=12, running=False)
    assert draining["state"] == "draining" and not draining["windows"] and draining["estimate"] is None
    monitor.begin(2, "continuous", 4.1)
    first = _write_observation(monitor, 4.1, pending=16 * mib, queued=12)
    assert first["state"] == "sampling" and first["estimate"] is None
    for delta in (1, 2):
        monitor.accept(4 * mib, 12)
        monitor.commit(8 * mib, 24)
        recovered = _write_observation(monitor, 4.1 + delta, pending=(16 - 4 * delta) * mib, queued=12 - 3 * delta)
    assert recovered["state"] == "recovering" and recovered["estimate"] is None
    assert monitor.alerts() == [alert]


def test_live_write_tracks_persistent_batch_growth_and_separates_event_prebuffer():
    mib = 2**20
    monitor = LiveWriteTiming(16, 100 * mib, 30)
    monitor.begin(1, "hardware", 0.)
    _write_observation(monitor, 0, pending=mib, queued=1)
    for second in range(1, 5):
        monitor.accept(6 * mib, 30)
        monitor.commit(6 * mib, 10)
        view = _write_observation(monitor, second, pending=mib, queued=1 + 2 * second)
    assert view["state"] == "queue_growth" and view["estimate"]["resource"] == "batches"
    assert all(row["pending_raw_bytes_growth"] == 0 for row in view["windows"])
    assert monitor.alerts()[0]["kind"] == "queue_growth"
    monitor.begin(2, "event", 6.0)
    prebuffer = _write_observation(monitor, 10, pending=0, queued=0, prebuffer=90 * mib)
    assert prebuffer["state"] == "prebuffer" and prebuffer["prebuffer_raw_bytes"] == 90 * mib
    assert len(monitor.alerts()) == 1
    near = _write_observation(monitor, 11, pending=mib, queued=1, prebuffer=90 * mib)
    assert near["state"] == "near_limit" and near["estimate"] is None and not near["windows"]
    assert monitor.alerts()[-1]["phase"] == 2


class _PulseCamera(SimulatedCamera):
    def __init__(self, index):
        super().__init__(index)
        self.pulses = queue.Queue()
        self.wired = False

    def trigger_support(self):
        return TriggerSupport(software=True, hardware=True, basis="fixture", physical_inputs=("Line1",))

    def start(self, trigger="free", source=None):
        self.wired = trigger == "hardware"
        if self.wired:
            assert source == "Line1"
        super().start("software" if self.wired else trigger)

    def read(self, timeout_s):
        if self.wired:
            try:
                self.pulses.get(timeout=timeout_s)
            except queue.Empty:
                return None
            self.trigger()
        return super().read(timeout_s)


def _camera_fixture():
    cameras = {camera.device.id: camera for camera in (_PulseCamera(0), _PulseCamera(1))}
    return cameras, CameraRegistry(custom={"sim": (lambda: [item.device for item in cameras.values()], cameras.__getitem__)})


def test_pending_write_includes_inflight_and_does_not_credit_previous_phase(tmp_path, monkeypatch):
    cameras, registry = _camera_fixture()
    config = _plan({"mode": "single", "role": "reference"}, {"mode": "hardware", "duration_s": 5, "hardware_source": "Line1"})
    entered, resume = threading.Event(), threading.Event()
    commit = ExperimentWriter.commit_frames

    def gated(writer, prepared):
        if prepared[0].record["phase"] == 1:
            entered.set()
            assert resume.wait(5)
        return commit(writer, prepared)

    monkeypatch.setattr(ExperimentWriter, "commit_frames", gated)
    session = CaptureSession(config, tmp_path / "run", registry=registry)
    session.start()
    left, right = "sim:0", "sim:1"
    try:
        assert entered.wait(4)
        running = _until(lambda: _phase_status(session, 2))
        assert running["queue_size"] == 0
        assert running["phase_progress"]["received_by_camera"] == {left: 0, right: 0}
        assert all(session.latest_sample(key)[0] == 1 for key in (left, right))
        assert all(row.pending == 1 and row.saved == 0 and row.caption == "等触发" and not row.attention
                   for row in camera_monitor_rows(running).values())
        resume.set()
        _until(lambda: session.status()["cameras"][left]["saved"] == 1)
        current = session.status()
        assert all(row["saved"] == 0 and row["last_receipt_age_s"] is None
                   for row in current["phase_progress"]["camera_receipts"].values())
        assert all(row["pending_write"] == 0 for row in current["cameras"].values())
        cameras[left].pulses.put(None)
        _until(lambda: session.status()["cameras"][left]["saved"] == 2)
        current = session.status()["phase_progress"]
        assert current["received_by_camera"] == {left: 1, right: 0}
        assert current["camera_receipts"][left]["saved"] == 1
        assert current["camera_receipts"][right]["saved"] == 0
        assert session.latest_sample(left)[0] == 2 and session.latest_sample(right)[0] == 1
    finally:
        resume.set()
        session.request_stop()
        result = session.wait(6)
    assert all(row["pending_write"] == 0 for row in result["cameras"].values())
    with ExperimentReader(session.output) as reader:
        assert reader.verify()["committed_frames"] == 3


def _phase_status(session, index):
    status = session.status()
    return status if status["phase"] == index and status["phase_progress"] else None


def test_capture_monitor_distinguishes_hardware_wait_asymmetry_and_limit():
    config = _plan({"mode": "hardware", "duration_s": 8, "hardware_source": "Line1"})
    status = CaptureSession(config, "unused-capture-monitor-path").status()
    left, right = "sim:0", "sim:1"
    status.update(state="running", mode="hardware", phase=1, phase_progress={
        "elapsed_s": 3., "accepting_images": True, "planned_fps": None,
        "frame_limit_per_camera": 5, "event_window": None, "manual_signal_id": None,
        "received_by_camera": {left: 0, right: 0}, "camera_receipts": {
            key: {"saved": 0, "last_receipt_age_s": None, "host_receive_fps": None} for key in (left, right)
        },
    })
    rows = camera_monitor_rows(status)
    assert all(row.caption == "等触发" and not row.attention for row in rows.values())
    progress = status["phase_progress"]
    progress["hardware_inputs"] = {left: "Line1", right: "Line2"}
    progress["hardware_readiness"] = {"cameras": {
        left: {"state": "ready", "line": "Line3"},
        right: {"state": "not_ready", "line": "Line4"},
    }}
    rows = camera_monitor_rows(status)
    assert rows[right].caption == "未就绪" and rows[right].attention
    assert rows[right].hardware_input == "Line2" and "等待输出 Line4" in rows[right].wait_output
    assert rows[left].caption == "等触发" and not rows[left].attention
    progress.pop("hardware_readiness")
    progress["received_by_camera"][left] = 4
    progress["camera_receipts"][left].update(saved=3, last_receipt_age_s=.1, host_receive_fps=12.)
    status["cameras"][left]["pending_write"] = 1
    rows = camera_monitor_rows(status)
    assert (rows[left].caption, rows[left].pending, rows[left].fps) == ("收图中", 1, 12.)
    assert rows[right].attention and rows[right].caption == "无新图" and not rows[right].evidence
    assert "漏拍需用触发记录确认" in rows[right].guidance
    progress["received_by_camera"][right] = 5
    assert camera_monitor_rows(status)[right].caption == "已收齐"
    progress["received_by_camera"][right] = 0
    progress["camera_receipts"][left]["last_receipt_age_s"] = 2.
    assert all(row.caption == "等触发" for row in camera_monitor_rows(status).values())
    status["cameras"][left]["app_unsaved"] = 1
    assert camera_monitor_rows(status)[left].evidence == ("未保存 1",)


def test_monitor_shows_saving_between_phases_when_ledger_registration_is_pending():
    status = CaptureSession(_plan({"mode": "single"}), "unused-capture-monitor-path").status()
    status["state"] = "running"
    status["cameras"]["sim:0"]["pending_write"] = 1

    rows = camera_monitor_rows(status)
    assert all(row.caption == "保存中" and "台账登记" in row.guidance for row in rows.values())
    assert rows["sim:0"].pending == 1 and status["buffered_raw_bytes"] == 0

    status["cameras"]["sim:0"]["pending_write"] = 0
    assert all(row.caption == "准备中" for row in camera_monitor_rows(status).values())


def test_event_prebuffer_is_visible_but_not_pending_write_in_live_status(tmp_path):
    config = _plan({"mode": "event", "fps": 26, "pre_frames": 4, "post_frames": 3, "duration_s": 7})
    session = CaptureSession(config, tmp_path / "event")
    session.start()
    left = "sim:0"
    try:
        with CaptureControl(session):
            _until(lambda: (status := _phase_status(session, 1)) and
                   status["phase_progress"]["event_window"]["state"] == "waiting")
            live = send_command(session.output, "status")["status"]
            assert live["buffered_raw_bytes"] > 0
            rows = camera_monitor_rows(live)
            assert all(row.pending == row.saved == 0 and row.received >= 4 and row.fps > 0 for row in rows.values())
            assert all(row.caption == "预缓存" for row in rows.values())
            session.request_trigger(require_prebuffer=True)
            result = session.wait(6)
    finally:
        if not session.finished:
            session.request_stop()
            session.wait(5)
    assert result["state"] == "completed" and result["cameras"][left]["saved"] >= 7
    assert all(row["pending_write"] == 0 for row in result["cameras"].values())
