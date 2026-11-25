import time
from types import SimpleNamespace

import numpy as np
import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureSession, ExperimentReader, inspect_plan
from straintrace_capture.cameras import pylon
from straintrace_capture.cameras.base import Device, Feature, TriggerSupport
from straintrace_capture.image_inputs import bayer_luminance, decode_samples
from straintrace_capture.preview import PreviewSession, preview_pixels
from straintrace_capture.readiness import probe_camera_setup


CAMERA_ID = "pylon:fixture"
FORMATS = ("Mono8", "Mono10p", "Mono12Packed", "Mono12packed", "BayerBG12packed", "Mono14p")
DEVICE = Device(CAMERA_ID, "pylon", "Basler", "Fixture", "fixture", "test_fixture")


class FeatureNodes:
    def __init__(self, width, height, choices=FORMATS):
        self.width, self.height = width, height
        self.choices = choices
        self.format = choices[0]

    def public_features(self):
        return {
            "pixel_format": Feature("pixel_format", self.format, choices=self.choices),
            "width": Feature("width", self.width),
            "height": Feature("height", self.height),
        }

    def get(self, name):
        return self.format if name == "PixelFormat" else None

    def feature(self, name):
        return self.public_features().get(name)

    def write(self, name, value):
        self.public_features()[name].validate(value)
        if name == "pixel_format":
            self.format = value

    def configure(self, settings):
        for name, value in settings.items():
            self.write(name, value)

    def trigger_setup(self, trigger, source):
        assert trigger == "free" and source is None

    def trigger_support(self):
        return TriggerSupport(False, False, "fixture")


class PixelsResult:
    def __init__(self, pixels, pixel_type, counter, reported_shape):
        self.pixels = pixels.copy()
        self.reported_shape = reported_shape
        self.pixel_type, self.counter = pixel_type, counter

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # The driver reuses its buffer after the result's lifetime.
        self.pixels.fill(0)

    def IsValid(self):
        return True

    def GrabSucceeded(self):
        return True

    def GetArray(self):
        return self.pixels

    def GetPixelType(self):
        return self.pixel_type

    def GetHeight(self):
        return self.reported_shape[0]

    def GetWidth(self):
        return self.reported_shape[1]

    def GetBlockID(self):
        return self.counter

    def GetTimeStamp(self):
        return 0

    def GetImageNumber(self):
        return self.counter

    def GetNumberOfSkippedImages(self):
        return 0


class FixtureCamera:
    def __init__(self, result_factory):
        self.result_factory = result_factory
        self.counter, self.grabbing = 0, False

    def IsCameraDeviceRemoved(self):
        return False

    def IsGrabbing(self):
        return self.grabbing

    def StartGrabbing(self, strategy):
        self.grabbing = True

    def StopGrabbing(self):
        self.grabbing = False

    def Close(self):
        assert not self.grabbing

    def RetrieveResult(self, timeout, handling):
        assert self.grabbing
        self.counter += 1
        time.sleep(.002)
        return self.result_factory(self.counter)


def _registry(monkeypatch, pixels, pixel_type, *, mismatch=False, truncated=False):
    api = SimpleNamespace(GenericException=RuntimeError)
    sdk = SimpleNamespace(
        GrabStrategy_OneByOne=0, TimeoutHandling_Return=0,
        PixelType_Mono10p=0x010A0046, PixelType_Mono12packed=0x010C0006, PixelType_BayerBG12Packed=0x010C002D,
    )
    monkeypatch.setattr(pylon, "_sdk", lambda: (sdk, api))

    def open_fixture(camera):
        camera.device = DEVICE
        camera._nodes = FeatureNodes(pixels.shape[1], pixels.shape[0])
        camera._instant = FixtureCamera(lambda counter: PixelsResult(
            pixels[:, :-1] if truncated else pixels, 0x01100007 if mismatch else pixel_type, counter, pixels.shape,
        ))

    monkeypatch.setattr(pylon.PylonCamera, "open", open_fixture)
    return CameraRegistry(custom={"pylon": (lambda: [DEVICE], pylon.PylonCamera)})


def _plan(pixel_format, shape):
    height, width = shape
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "Packed sample fixture",
        "cameras": [{"id": CAMERA_ID, "settings": {"pixel_format": pixel_format, "width": width, "height": height}}],
        "phases": [{"mode": "continuous", "frame_count": 2}],
    })


@pytest.mark.parametrize(("pixel_format", "pixel_type", "bits", "color"), [
    ("Mono10p", 0x010A0046, 10, "mono"),
    ("Mono12Packed", 0x010C0006, 12, "mono"),
    ("Mono12packed", 0x010C0006, 12, "mono"),
    ("BayerBG12packed", 0x010C002D, 12, "bayer"),
])
def test_packed_basler_report_preview_and_saved_samples(tmp_path, monkeypatch, pixel_format, pixel_type, bits, color):
    samples = np.arange(63, dtype=np.uint16).reshape(7, 9) * 13
    samples[0, 0] = (1 << bits) - 1
    registry = _registry(monkeypatch, samples, pixel_type)
    config = _plan(pixel_format, samples.shape)
    with registry.camera(CAMERA_ID) as camera:
        assert camera.features()["pixel_format"].choices == FORMATS[:-1]
    triggers, setup = probe_camera_setup(config, registry)
    root = tmp_path / "packed"
    report = inspect_plan(config, root, trigger_support=triggers, camera_settings=setup)
    assert report["ready"] and not root.exists()
    assert report["phases"][0]["raw_payload_bytes"] == 2 * samples.nbytes

    preview = PreviewSession(config.cameras, registry)
    preview.start()
    try:
        limit = time.monotonic() + 3
        while (image := preview.latest_image(CAMERA_ID)) is None and time.monotonic() < limit:
            assert preview.status()[CAMERA_ID]["state"] != "failed", preview.status()
            time.sleep(.005)
        assert image is not None and image.pixel_format == pixel_format
        np.testing.assert_array_equal(image.pixels, samples, strict=True)
        assert preview_pixels(image)[0, 0] == 255
    finally:
        preview.stop()
        assert preview.wait(4)

    session = CaptureSession(config, root, registry=registry)
    session.start()
    assert session.wait(6)["state"] == "completed"
    with ExperimentReader(root) as reader:
        assert reader.verify()["complete"]
        rows = reader.frames()
        assert [row["native_frame_id"] for row in rows] == [1, 2]
        for row in rows:
            assert (row["pixel_format"], row["significant_bits"], row["color_model"]) == (pixel_format, bits, color)
            path = reader.image_path(row)
            np.testing.assert_array_equal(decode_samples(path), samples, strict=True)
            if color == "bayer":
                assert bayer_luminance(path, pixel_format).dtype == np.uint16


@pytest.mark.parametrize(("failure", "code"), [
    ("mismatch", "PYLON_PIXEL_FORMAT_MISMATCH"),
    ("truncated", "PYLON_BAD_PAYLOAD"),
])
def test_invalid_packed_basler_result_cannot_enter_ledger(tmp_path, monkeypatch, failure, code):
    samples = np.zeros((7, 9), dtype=np.uint16)
    registry = _registry(monkeypatch, samples, 0x010A0046, **{failure: True})
    session = CaptureSession(_plan("Mono10p", samples.shape), tmp_path / "invalid", registry=registry)
    session.start()
    status = session.wait(6)
    assert status["state"] == "failed" and status["error"]["code"] == code
    with ExperimentReader(session.output) as reader:
        assert not reader.frames()
