import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureError
from straintrace_capture.cameras.base import Feature, Image
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.exposure import (
    ExposureMeasurement, confirm_exposure_improvement, exposure_unavailable, recommend_exposure, sample_exposure,
)
from straintrace_capture.preview import PreviewSession, _image_layout


def reading(highlight, clipping=0., spread=0.):
    return ExposureMeasurement(highlight, clipping, spread, 0.)


@pytest.fixture
def features():
    return {
        "exposure_us": Feature("exposure_us", 2000.0, 50., 40_000., 50., unit="us"),
        "fps": Feature("fps", 100., unit="Hz"),
        "exposure_auto": Feature("exposure_auto", "Off"),
        "gain_auto": Feature("gain_auto", "Off"),
    }


def test_suggestion_uses_raw_highlights_camera_steps_and_fastest_rate(features):
    result = recommend_exposure(reading(.34), features, (15., 300.), operator_limit=2741.)
    assert result.maximum_us == 2741 and result.proposed_us == 2700
    assert recommend_exposure(reading(.34), features, (15., 300.)).proposed_us == 3300
    assert recommend_exposure(reading(1., .06), features).proposed_us == 1450
    assert recommend_exposure(reading(1., .17), features).proposed_us == 1000
    assert recommend_exposure(reading(.82), features).proposed_us is None
    assert recommend_exposure(reading(.34, spread=.12), features).proposed_us is None
    assert recommend_exposure(reading(.02), features).proposed_us is None
    assert recommend_exposure(reading(.34), features, operator_limit=1950).proposed_us is None
    with pytest.raises(CaptureError, match="below device minimum"):
        recommend_exposure(reading(.34), features, operator_limit=30)


def test_guard_manual_units_and_inactive_device_fps(features):
    features["fps"] = Feature("fps", 10_000., unit="Hz")
    features["fps_enable"] = Feature("fps_enable", False)
    assert recommend_exposure(reading(.34), features).proposed_us == 4000
    features["gain_auto"] = Feature("gain_auto", "Continuous")
    assert "自动曝光" in exposure_unavailable(features)
    features.pop("gain_auto")
    features["exposure_us"] = replace(features["exposure_us"], unit=None)
    assert "微秒" in exposure_unavailable(features)
    with pytest.raises(CaptureError) as missing:
        recommend_exposure(reading(.34), features)
    assert missing.value.code == "EXPOSURE_UNAVAILABLE"


def test_measurement_counts_distinct_raw_images_and_excludes_alpha():
    raw = np.empty((24, 24, 4), np.uint16)
    raw[...] = [682, 1024, 2048, 4095]
    raw[:12, :12, 1] = 4095
    first = Image(raw, "RGBa12", 12, "rgba", 1, 1)
    frames = [first, first, *(replace(first, received_monotonic_ns=n) for n in range(2, 7))]
    incoming = iter(frames)
    result, last = sample_exposure(lambda _wait: next(incoming), after=1, layout=_image_layout(first),
                                   roi=(12, 12, 12, 12))
    assert result.highlight == pytest.approx(2048 / 4095)
    assert result.near_saturation == 0 and result.stable and last.received_monotonic_ns == 6
    incoming = iter(frames)
    result, _ = sample_exposure(lambda _wait: next(incoming), after=1, layout=_image_layout(first))
    assert result.highlight == 1 and result.near_saturation == .25
    with pytest.raises(CaptureError) as short:
        sample_exposure(lambda _wait: replace(first, received_monotonic_ns=2), after=1, layout=_image_layout(first),
                        roi=(1, 1, 7, 7))
    assert short.value.code == "EXPOSURE_REGION_TOO_SMALL"


def test_verification_requires_stable_improvement_without_new_clipping():
    confirm_exposure_improvement(reading(.34), reading(.65), increased=True)
    confirm_exposure_improvement(reading(1., .27), reading(.83, .002), increased=False)
    for before, after, increased in (
        (reading(.34), reading(.35), True), (reading(.34), reading(1., .1), True),
        (reading(1., .2), reading(.12), False), (reading(.34), reading(.65, spread=.1), True),
        (reading(.50, spread=.07), reading(.61, spread=.06), True),
        (reading(.9922, .01156), reading(.9765, .00453), False),
        (reading(.9882, .00766), reading(.9765, .00453), True),
    ):
        with pytest.raises(CaptureError, match="亮部"):
            confirm_exposure_improvement(before, after, increased=increased)


def test_preview_trial_adopts_improvement_and_rejects_a_failed_or_stale_trial():
    plan = CaptureConfig.from_dict({
        "config_version": 1, "name": "exposure trial",
        "cameras": [{"id": f"sim:{index}", "settings": {
            "width": 160, "height": 120, "pixel_format": "Mono12", "exposure_us": 3600. if not index else 1900.,
            "motion_px": 0., "strain_per_frame": 0.,
        }} for index in range(2)],
        "phases": [{"mode": "single"}],
    })
    preview = PreviewSession(plan.cameras, CameraRegistry())
    preview.start()
    try:
        deadline = time.monotonic() + 6
        while not (binding := preview.camera_binding("sim:0")) or preview.camera_binding("sim:1") is None:
            assert time.monotonic() < deadline, preview.status()
            time.sleep(.025)
        peer = preview.latest_image("sim:1")
        _, observed, settings = preview.parameters("sim:0")
        first = preview.latest_image("sim:0")
        layout = _image_layout(first)
        before, _ = sample_exposure(lambda _wait: preview.latest_image("sim:0"), layout=layout,
                                    after=first.received_monotonic_ns)
        proposal = recommend_exposure(before, observed).proposed_us
        assert before.near_saturation > .05 and proposal is not None and proposal < 3600
        values = {**settings, "exposure_us": proposal}

        def verify(first, read):
            result, last = sample_exposure(read, after=first.received_monotonic_ns, layout=layout)
            confirm_exposure_improvement(before, result, increased=False)
            assert result.near_saturation < before.near_saturation
            return last

        camera, image = preview.reconfigure("sim:0", values, expected_binding=binding, verify=verify).result(timeout=8)
        assert camera.settings["exposure_us"] == proposal
        assert image.received_monotonic_ns > first.received_monotonic_ns
        assert preview.latest_image("sim:1").received_monotonic_ns > peer.received_monotonic_ns
        assert preview.parameters("sim:1")[1]["exposure_us"].value == 1900
        with pytest.raises(CaptureError) as changed:
            preview.reconfigure("sim:0", settings, expected_binding=binding, verify=verify)
        assert changed.value.code == "EXPOSURE_SCOPE_CHANGED"

        def reject(_first, _read):
            confirm_exposure_improvement(reading(.5), reading(.5), increased=True)

        with pytest.raises(CaptureError) as failed:
            preview.reconfigure("sim:0", settings, expected_binding=preview.camera_binding("sim:0"), verify=reject).result(timeout=8)
        assert failed.value.code == "PREVIEW_ADJUST_RESTORED"
        assert preview.parameters("sim:0")[1]["exposure_us"].value == proposal
        assert preview.cameras[0] == camera
    finally:
        preview.stop()
        assert preview.wait(7)


def test_stopping_preview_during_exposure_verification_restores_readback():
    camera = SimulatedCamera(0)
    plan = CaptureConfig.from_dict({
        "config_version": 1, "name": "cancel exposure trial",
        "cameras": [{"id": camera.device.id, "settings": {"width": 96, "height": 64, "exposure_us": 2200.}}],
        "phases": [{"mode": "single"}],
    })
    registry = CameraRegistry(custom={"sim": (lambda: [camera.device], lambda _: camera)})
    preview = PreviewSession(plan.cameras, registry)
    reached, release = threading.Event(), threading.Event()
    preview.start()
    try:
        deadline = time.monotonic() + 5
        while (binding := preview.camera_binding(camera.device.id)) is None:
            assert time.monotonic() < deadline, preview.status()
            time.sleep(.025)

        def verify(_first, read):
            reached.set()
            assert release.wait(5)
            return read(.2)

        values = {**plan.cameras[0].settings, "exposure_us": 1200.}
        result = preview.reconfigure(camera.device.id, values, expected_binding=binding, verify=verify)
        assert reached.wait(5)
        assert camera.features()["exposure_us"].value == 1200
        preview.stop()
        release.set()
        with pytest.raises(CaptureError) as stopped:
            result.result(timeout=6)
        assert stopped.value.code == "PREVIEW_STOPPED"
        assert camera.features()["exposure_us"].value == 2200
    finally:
        release.set()
        preview.stop()
        assert preview.wait(6)
