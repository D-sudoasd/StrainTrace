import threading
import time

import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureSession, ExperimentReader, inspect_plan
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.config import CaptureError
from straintrace_capture.preview import PreviewSession, image_quality


def _plan(cameras):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "Preview adjustments", "cameras": cameras,
        "phases": [{"mode": "single", "role": "reference"}],
    })


def _wait_for_preview(preview, condition):
    deadline = time.monotonic() + 6
    while not condition():
        if time.monotonic() >= deadline or preview.stopped:
            pytest.fail(f"preview did not become ready: {preview.status()}")
        time.sleep(.02)


def test_preview_adjusts_one_camera_and_passes_new_geometry_to_capture(tmp_path):
    initial = {"width": 160, "height": 120, "pixel_format": "Mono12", "fps": 24., "exposure_us": 4000.}
    plan = _plan([{"id": key, "settings": initial} for key in ("sim:0", "sim:1")])
    preview = PreviewSession(plan.cameras, CameraRegistry())
    preview.start()
    try:
        _wait_for_preview(preview, lambda: all(preview.parameters(key) for key in ("sim:0", "sim:1")))
        left = preview.latest_image("sim:0")
        peer = preview.latest_image("sim:1")
        _, features, planned = preview.parameters("sim:0")
        assert features["exposure_us"].writable and features["exposure_us"].value == 4000
        assert not features["motion_px"].writable
        assert planned == initial
        assert preview.settings_check(plan.cameras[0]).checked == tuple(initial)
        adjusted, frame = preview.reconfigure("sim:0", {**planned, "exposure_us": 1200.}).result(timeout=6)
        assert adjusted.settings["exposure_us"] == 1200
        assert frame.received_monotonic_ns > left.received_monotonic_ns
        assert image_quality(frame)["bright_fraction"] < image_quality(left)["bright_fraction"]
        assert preview.status()["sim:0"]["features"]["exposure_us"] == 1200

        configured, frame = preview.reconfigure("sim:0", {**adjusted.settings, "width": 192}).result(timeout=6)
        assert frame.pixels.shape == (120, 192)
        _wait_for_preview(preview, lambda: preview.latest_image("sim:1").received_monotonic_ns > peer.received_monotonic_ns)
        assert preview.parameters("sim:1")[2] == initial
        assert preview.cameras == (configured, plan.cameras[1])
        assert preview.settings_check(plan.cameras[0]) is None
        assert preview.settings_check(configured).deferred == ()

        capture_plan = _plan([{"id": item.id, "settings": dict(item.settings)} for item in preview.cameras])
        output = tmp_path / "capture"
        report = inspect_plan(capture_plan, output, preview_images={"sim:0": frame})
        assert (report["cameras"]["sim:0"]["basis"], report["cameras"]["sim:0"]["frame_bytes"]) == ("preview", 46080)
    finally:
        preview.stop()
        assert preview.wait(6)

    session = CaptureSession(capture_plan, output)
    session.start()
    assert session.wait(6)["state"] == "completed"
    with ExperimentReader(output) as reader:
        observed = {row["device"]["id"]: row["features"] for row in reader.cameras()}
        assert [observed[key]["exposure_us"]["value"] for key in ("sim:0", "sim:1")] == [1200, 4000]
        assert [observed[key]["width"]["value"] for key in ("sim:0", "sim:1")] == [192, 160]
        assert reader.verify()["complete"]


class _RoundingCamera(SimulatedCamera):
    def set_feature(self, name, value):
        super().set_feature(name, 32. if name == "fps" and value == 33. else value)


def test_preview_preflight_requires_matching_readback_for_each_setting():
    camera = _RoundingCamera(0)
    plan = _plan([{"id": camera.device.id, "settings": {"width": 128, "fps": 33.}}])
    registry = CameraRegistry(custom={"sim": (lambda: [camera.device], lambda _: camera)})
    preview = PreviewSession(plan.cameras, registry)
    preview.start()
    try:
        _wait_for_preview(preview, lambda: preview.settings_check(plan.cameras[0]))
        check = preview.settings_check(plan.cameras[0])
        assert check.basis == "preview_readback"
        assert check.checked == ("width",) and check.deferred == ("fps",)
    finally:
        preview.stop()
        assert preview.wait(6)


def test_preview_rolls_back_partial_writes_if_readback_differs():
    camera = _RoundingCamera(0)
    plan = _plan([{"id": camera.device.id, "settings": {"width": 128, "height": 96}}])
    registry = CameraRegistry(custom={"sim": (lambda: [camera.device], lambda _: camera)})
    preview = PreviewSession(plan.cameras, registry)
    preview.start()
    try:
        _wait_for_preview(preview, lambda: preview.parameters(camera.device.id))
        old = preview.latest_image(camera.device.id)
        draft = {**plan.cameras[0].settings, "exposure_us": 800., "fps": 33.}
        with pytest.raises(CaptureError, match=r"fps: requested 33.0, read back 32.0") as exc:
            preview.reconfigure(camera.device.id, draft).result(timeout=6)
        assert exc.value.code == "PREVIEW_ADJUST_RESTORED"
        assert preview.cameras == plan.cameras
        _, features, _ = preview.parameters(camera.device.id)
        assert features["exposure_us"].value == 2000 and features["fps"].value == 25
        assert preview.latest_image(camera.device.id).received_monotonic_ns > old.received_monotonic_ns
        assert preview.status()[camera.device.id]["error"] is None
    finally:
        preview.stop()
        assert preview.wait(6)


def test_stopping_preview_mid_adjustment_restores_settings_and_settles_request():
    writing, release = threading.Event(), threading.Event()

    class PausedCamera(SimulatedCamera):
        def set_feature(self, name, value):
            super().set_feature(name, value)
            if name == "exposure_us" and value == 800.:
                writing.set()
                assert release.wait(5)

    camera, peer = PausedCamera(0), SimulatedCamera(1)
    devices = {item.device.id: item for item in (camera, peer)}
    plan = _plan([{"id": key, "settings": {"width": 128, "height": 96}} for key in devices])
    registry = CameraRegistry(custom={"sim": (lambda: [item.device for item in devices.values()], devices.__getitem__)})
    preview = PreviewSession(plan.cameras, registry)
    preview.start()
    try:
        _wait_for_preview(preview, lambda: all(preview.parameters(key) for key in devices))
        request = preview.reconfigure(camera.device.id, {**plan.cameras[0].settings, "exposure_us": 800.})
        assert writing.wait(4)
        stamp = preview.latest_image(peer.device.id).received_monotonic_ns
        _wait_for_preview(preview, lambda: preview.latest_image(peer.device.id).received_monotonic_ns > stamp)
        assert preview.status()[camera.device.id]["state"] == "adjusting" and not request.done()
        with pytest.raises(CaptureError) as pending:
            preview.reconfigure(camera.device.id, plan.cameras[0].settings)
        assert pending.value.code == "PREVIEW_NOT_READY"
        preview.stop()
        release.set()
        with pytest.raises(CaptureError) as stopped:
            request.result(timeout=6)
        assert stopped.value.code == "PREVIEW_STOPPED"
    finally:
        release.set()
        preview.stop()
        assert preview.wait(6)
    assert preview.latest_image(camera.device.id) is None
    assert camera.features()["exposure_us"].value == 2000
    assert preview.cameras == plan.cameras
