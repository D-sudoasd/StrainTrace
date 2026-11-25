import gc
import threading
import time
import weakref
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureError
from straintrace_capture.cameras import gentl, pylon
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.preview import PreviewSession


def _until(test, *, timeout=5):
    deadline = time.monotonic() + timeout
    while not test():
        assert time.monotonic() < deadline, "preview did not reach the expected state"
        time.sleep(.006)


class _Rig:
    def __init__(self):
        self.link = threading.Event()
        self.link.set()
        self.opens = Counter()
        self.closes = Counter()
        self.scope = "serial_number"
        self.drop_after = None
        self.failure_code = "CAMERA_DISCONNECTED"
        self.replacement = None
        self.finish_close = None

    def camera(self, identity):
        rig = self

        class Camera(SimulatedCamera):
            def open(self):
                rig.opens[identity] += 1
                self.visit = rig.opens[identity]
                self.reads = 0
                if identity == "sim:0":
                    if not rig.link.is_set():
                        raise CaptureError("CAMERA_ID_UNRESOLVED", "fixture cable unplugged")
                    if rig.scope == "host_index":
                        self.device = replace(self.device, identity_scope="host_index", serial=None)
                    if self.visit > 1 and rig.replacement == "identity":
                        self.device = replace(self.device, model="different camera")
                super().open()

            def close(self):
                if identity == "sim:0" and rig.finish_close is not None:
                    rig.finish_close.wait(3)
                rig.closes[identity] += 1
                super().close()

            def set_feature(self, name, value):
                if identity == "sim:0" and self.visit > 1 and name == "width" and rig.replacement == "readback":
                    value += 8
                super().set_feature(name, value)

            def read(self, timeout_s):
                if identity == "sim:0" and (not rig.link.is_set() or self.reads == rig.drop_after):
                    raise CaptureError(rig.failure_code, "fixture connection failed")
                frame = super().read(timeout_s)
                if frame is not None:
                    self.reads += 1
                    if identity == "sim:0" and self.visit > 1 and rig.replacement == "layout":
                        frame = replace(frame, pixels=frame.pixels[:, :-1])
                return frame

        return Camera(int(identity[-1]))

    def preview(self, ids=("sim:0", "sim:1")):
        plan = CaptureConfig.from_dict({
            "config_version": 1, "name": "Recover preview",
            "cameras": [{"id": identity, "settings": {"width": 96, "height": 64, "pixel_format": "Mono12",
                                                      "exposure_us": 1700., "fps": 25.}} for identity in ids],
            "phases": [{"mode": "single", "role": "reference"}],
        })
        registry = CameraRegistry(custom={"sim": (lambda: [], self.camera)})
        return PreviewSession(plan.cameras, registry)


def _state(preview, identity="sim:0"):
    return preview.status()[identity]["state"]


def test_auto_reconnect_preserves_healthy_peer_and_replaces_stale_evidence(monkeypatch):
    monkeypatch.setattr("straintrace_capture.preview._RETRY_DELAYS_S", (.4, .4))
    rig = _Rig()
    preview = rig.preview()
    preview.start()
    preview.start()
    try:
        _until(lambda: preview.evidence_revision is not None)
        revision = preview.evidence_revision
        original = preview.latest_image("sim:0")
        peer_before = preview.latest_image("sim:1").received_monotonic_ns
        rig.link.clear()
        _until(lambda: _state(preview) == "retry_wait")
        assert preview.evidence_revision is None
        assert preview.latest_image("sim:0") is None
        assert preview.parameters("sim:0") is preview.trigger_support("sim:0") is None
        assert preview.settings_check(preview.cameras[0]) is None
        assert rig.closes["sim:0"] == 1
        _until(lambda: preview.latest_image("sim:1").received_monotonic_ns > peer_before)
        rig.link.set()
        _until(lambda: preview.evidence_revision is not None)
        assert preview.evidence_revision > revision
        fresh = preview.latest_image("sim:0")
        assert fresh.received_monotonic_ns > original.received_monotonic_ns
        assert fresh.pixels.shape == original.pixels.shape == (64, 96)
        assert preview.parameters("sim:0")[1]["exposure_us"].value == 1700.
        assert rig.opens["sim:0"] >= 2
        assert (rig.opens["sim:1"], rig.closes["sim:1"]) == (1, 0)
    finally:
        preview.stop()
        assert preview.wait(3)


def test_flapping_camera_stops_automatic_attempts_and_accepts_one_manual_retry(monkeypatch):
    monkeypatch.setattr("straintrace_capture.preview._RETRY_DELAYS_S", (.02, .03))
    rig = _Rig()
    rig.drop_after = 2
    preview = rig.preview(("sim:0",))
    preview.start()
    try:
        _until(lambda: _state(preview) == "failed")
        assert rig.opens["sim:0"] == rig.closes["sim:0"] == 3
        rig.drop_after = None
        preview.retry("sim:0")
        with pytest.raises(CaptureError, match="cannot retry"):
            preview.retry("sim:0")
        _until(lambda: _state(preview) == "preview")
        assert rig.opens["sim:0"] == 4
    finally:
        preview.stop()
        assert preview.wait(3)


def test_stable_stream_releases_a_new_automatic_retry_allowance(monkeypatch):
    monkeypatch.setattr("straintrace_capture.preview._RETRY_DELAYS_S", (.18,))
    monkeypatch.setattr("straintrace_capture.preview._STEADY_STREAM_NS", 90_000_000)
    rig = _Rig()
    preview = rig.preview(("sim:0",))
    preview.start()
    try:
        for expected_opens in (2, 3):
            _until(lambda: _state(preview) == "preview")
            initial = preview.latest_image("sim:0").received_monotonic_ns
            _until(lambda: preview.latest_image("sim:0").received_monotonic_ns - initial >= 100_000_000)
            rig.link.clear()
            _until(lambda: _state(preview) == "retry_wait")
            assert preview.status()["sim:0"]["retry_number"] == 1
            rig.link.set()
            _until(lambda: _state(preview) == "preview")
            assert rig.opens["sim:0"] == expected_opens
    finally:
        preview.stop()
        assert preview.wait(3)


def test_camera_unavailable_on_first_open_can_be_retried_by_identity():
    rig = _Rig()
    rig.link.clear()
    preview = rig.preview()
    preview.start()
    try:
        _until(lambda: _state(preview) == "failed" and _state(preview, "sim:1") == "preview")
        assert preview.status()["sim:0"]["error_code"] == "CAMERA_ID_UNRESOLVED"
        rig.link.set()
        preview.retry("sim:0")
        _until(lambda: preview.evidence_revision is not None)
        assert rig.opens == {"sim:0": 2, "sim:1": 1}
    finally:
        preview.stop()
        assert preview.wait(3)


def test_index_only_camera_waits_for_operator_after_disconnect():
    rig = _Rig()
    rig.scope = "host_index"
    preview = rig.preview(("sim:0",))
    preview.start()
    try:
        _until(lambda: _state(preview) == "preview")
        rig.link.clear()
        _until(lambda: _state(preview) == "failed")
        assert preview.status()["sim:0"]["identity_scope"] == "host_index"
        assert rig.opens["sim:0"] == 1
        rig.link.set()
        preview.retry("sim:0")
        _until(lambda: _state(preview) == "preview")
    finally:
        preview.stop()
        assert preview.wait(3)


@pytest.mark.parametrize(("replacement", "code"), [
    ("identity", "PREVIEW_IDENTITY_CHANGED"), ("layout", "PREVIEW_GEOMETRY_CHANGED"),
    ("readback", "FEATURE_READBACK_MISMATCH"),
])
def test_reconnection_rejects_changed_identity_layout_or_readback(monkeypatch, replacement, code):
    monkeypatch.setattr("straintrace_capture.preview._RETRY_DELAYS_S", (.5,))
    rig = _Rig()
    preview = rig.preview(("sim:0",))
    preview.start()
    try:
        _until(lambda: _state(preview) == "preview")
        config = preview.cameras
        rig.link.clear()
        _until(lambda: _state(preview) == "retry_wait")
        rig.replacement = replacement
        rig.link.set()
        preview.retry("sim:0")
        _until(lambda: _state(preview) == "failed")
        assert preview.status()["sim:0"]["error_code"] == code
        assert preview.cameras == config
        assert preview.latest_image("sim:0") is None
        assert preview.evidence_revision is None
    finally:
        preview.stop()
        assert preview.wait(3)


@pytest.mark.parametrize(("code", "expected"), [
    ("CAMERA_DISCONNECTED", "retry_wait"), ("UVC_DRIVER_TIMEOUT", "failed"),
])
def test_stop_cancels_backoff_and_unreleased_driver_cannot_reopen(monkeypatch, code, expected):
    monkeypatch.setattr("straintrace_capture.preview._RETRY_DELAYS_S", (6.,))
    rig = _Rig()
    rig.failure_code = code
    preview = rig.preview(("sim:0",))
    preview.start()
    try:
        _until(lambda: _state(preview) == "preview")
        rig.link.clear()
        _until(lambda: _state(preview) == expected)
        if expected == "failed":
            assert not preview.status()["sim:0"]["retryable"]
            with pytest.raises(CaptureError, match="cannot retry"):
                preview.retry("sim:0")
    finally:
        preview.stop()
        assert preview.wait(1)
    assert rig.opens["sim:0"] == rig.closes["sim:0"] == 1
    assert _state(preview) == "disconnected"


def test_fault_invalidates_images_before_releasing_the_old_driver(monkeypatch):
    monkeypatch.setattr("straintrace_capture.preview._RETRY_DELAYS_S", (4.,))
    rig = _Rig()
    rig.finish_close = threading.Event()
    preview = rig.preview()
    preview.start()
    try:
        _until(lambda: preview.evidence_revision is not None)
        rig.link.clear()
        _until(lambda: _state(preview) == "releasing")
        assert rig.closes["sim:0"] == 0
        assert preview.latest_image("sim:0") is None
        assert preview.evidence_revision is None
        with pytest.raises(CaptureError, match="cannot retry"):
            preview.retry("sim:0")
        rig.finish_close.set()
        _until(lambda: _state(preview) == "retry_wait")
        assert rig.closes["sim:0"] == 1
    finally:
        rig.finish_close.set()
        preview.stop()
        assert preview.wait(3)


def test_waiting_for_user_does_not_retain_closed_camera_or_frame():
    rig = _Rig()
    rig.scope = "host_index"
    create = rig.camera
    handles = []

    def track(identity):
        camera = create(identity)
        handles.append(weakref.ref(camera))
        return camera

    rig.camera = track
    preview = rig.preview(("sim:0",))
    preview.start()
    try:
        _until(lambda: _state(preview) == "preview")
        image = weakref.ref(preview.latest_image("sim:0"))
        rig.link.clear()
        _until(lambda: _state(preview) == "failed")
        assert not preview.stopped
        _until(lambda: handles[0]() is None and image() is None)
    finally:
        preview.stop()
        assert preview.wait(3)


def test_failed_adjustment_future_is_released_while_waiting_for_retry(monkeypatch):
    entered = threading.Event()
    finish = threading.Event()
    write = SimulatedCamera.set_feature

    def paused_write(camera, name, value):
        if name == "width" and value == 104:
            entered.set()
            assert finish.wait(3)
        write(camera, name, value)

    monkeypatch.setattr(SimulatedCamera, "set_feature", paused_write)
    rig = _Rig()
    preview = rig.preview(("sim:0",))
    preview.start()
    try:
        _until(lambda: _state(preview) == "preview")
        future = preview.reconfigure("sim:0", {**preview.cameras[0].settings, "width": 104})
        assert entered.wait(2)
        rig.link.clear()
        finish.set()
        with pytest.raises(CaptureError, match="fixture connection failed"):
            future.result(3)
        reference = weakref.ref(future)
        del future
        gc.collect()
        _until(lambda: reference() is None)
        assert _state(preview) == "failed"
    finally:
        finish.set()
        preview.stop()
        assert preview.wait(3)


def test_gentl_refresh_keeps_another_viky_camera_streaming():
    sdk = pytest.importorskip("genicam")
    pytest.importorskip("harvesters.core")
    cti = Path(sdk.__file__).with_name("viky.cti")
    if not cti.is_file():
        pytest.skip("Viky emulator is unavailable")
    registry = CameraRegistry(cti_files=(str(cti),))
    devices = registry.discover(("gentl",))["devices"]
    assert len(devices) >= 2
    left, right = (device["id"] for device in devices[:2])
    with registry.camera(left) as peer:
        peer.start()
        before = peer.read(2.)
        assert before is not None
        with registry.camera(right) as original:
            original.start()
            prior = original.read(2.)
            assert prior is not None
        with registry.camera(right, refresh=True) as reconnected:
            reconnected.start()
            fresh = reconnected.read(2.)
            assert fresh is not None and fresh.received_monotonic_ns > prior.received_monotonic_ns
            after = peer.read(2.)
            assert after is not None and after.received_monotonic_ns > before.received_monotonic_ns
        assert peer.read(2.) is not None


def test_gentl_refresh_resolves_reappearing_serial_after_index_changes(monkeypatch):
    harvesters = pytest.importorskip("harvesters.core")
    devices = [("left-port", "101"), ("right-port", "102")]

    class Info:
        def __init__(self, *, module, parent):
            self.property_dict = {"vendor": "Fixture", "model": "Camera", "serial_number": module[1], "id_": module[0]}
            self.parent = parent

    class Interface:
        def update_device_info_list(self, _timeout):
            self.device_info_list = list(devices)

    class Acquirer:
        def __init__(self, port):
            self.port = port
            self.closed = False

        def destroy(self):
            self.closed = True

    class Client:
        timeout_for_update = 0

        def __init__(self):
            self.interface = Interface()
            self.device_info_list = []
            self.reset_calls = 0

        def create(self, index):
            return Acquirer(self.device_info_list[index].property_dict["id_"])

        def reset(self):
            self.reset_calls += 1

    def load(client, _files):
        client.interface.update_device_info_list(0)
        client.device_info_list[:] = [Info(module=item, parent=client.interface) for item in client.interface.device_info_list]

    monkeypatch.setattr(gentl, "_sdk", lambda: (Client, object()))
    monkeypatch.setattr(gentl, "_load", load)
    monkeypatch.setattr(harvesters, "DeviceInfo", Info)
    group = gentl.GenTLGroup(())
    left, right = "gentl:Fixture:101", "gentl:Fixture:102"
    _, peer, _ = group.open(left)
    client = group._client
    try:
        _, old, _ = group.open(right)
        group.close(right)
        assert old.closed
        devices[:] = [("left-port", "101")]
        with pytest.raises(CaptureError, match="found 0"):
            group.open(right, refresh=True)
        devices[:] = [("new-right-port", "102"), ("left-port", "101")]
        _, restored, _ = group.open(right, refresh=True)
        assert restored.port == "new-right-port"
        assert not peer.closed and client.reset_calls == 0
    finally:
        group.close(right)
        group.close(left)
    assert peer.closed and client.reset_calls == 1


def test_pylon_releases_removed_device_after_sdk_reports_grab_error(monkeypatch):
    class VendorError(RuntimeError):
        pass

    class Instant:
        removed = False
        destroyed = False

        def IsCameraDeviceRemoved(self):
            return self.removed

        def RetrieveResult(self, *_args):
            self.removed = True
            raise VendorError("physical connection lost")

        def DestroyDevice(self):
            self.destroyed = True

    monkeypatch.setattr(pylon, "_sdk",
                        lambda: (SimpleNamespace(TimeoutHandling_Return=0), SimpleNamespace(GenericException=VendorError)))
    camera = pylon.PylonCamera("pylon:123")
    camera.device = replace(SimulatedCamera(0).device, id="pylon:123")
    camera._instant = driver = Instant()
    with pytest.raises(CaptureError, match="was removed") as raised:
        camera.read(.1)
    assert raised.value.code == "CAMERA_DISCONNECTED"
    camera.close()
    assert driver.destroyed and camera._instant is None
