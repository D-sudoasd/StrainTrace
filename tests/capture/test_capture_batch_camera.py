from dataclasses import replace

import tifffile

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader
from straintrace_capture.batch_camera import shared_features, verify_batch_settings
from straintrace_capture.cameras.base import Feature
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.readiness import probe_camera_setup


class RecordingCamera(SimulatedCamera):
    def __init__(self, index):
        super().__init__(index)
        self.writes = []

    def set_feature(self, name, value):
        self.writes.append((name, value))
        super().set_feature(name, value)


def setup_batch(*cameras, settings=()):
    config = CaptureConfig.from_dict({
        "config_version": 1, "name": "Batch settings",
        "cameras": [{"id": camera.device.id, "settings": settings[index] if settings else {}}
                    for index, camera in enumerate(cameras)],
        "phases": [{"mode": "single", "role": "reference"}],
    })
    devices = {camera.device.id: camera for camera in cameras}
    registry = CameraRegistry(custom={"sim": (lambda: [camera.device for camera in cameras], devices.__getitem__)})
    _, catalog = probe_camera_setup(config, registry, triggers=False, include_readback=True)
    return config, catalog, registry


def test_batch_intersects_capabilities_and_respects_auto_modes_and_units():
    class RestrictedCamera(RecordingCamera):
        def features(self):
            features = super().features()
            features["exposure_auto"] = Feature("exposure_auto", "Continuous", choices=("Off", "Continuous"))
            features["gain_db"] = replace(features["gain_db"], unit="device_units")
            features["pixel_format"] = replace(features["pixel_format"], choices=("Mono8", "Mono12", "BGR8"))
            features["width"] = replace(features["width"], maximum=640)
            return features

    config, catalog, _ = setup_batch(RecordingCamera(0), RestrictedCamera(1))
    shared = shared_features(config, catalog)
    assert shared["pixel_format"].choices == ("Mono8", "Mono12")
    assert shared["width"].minimum == 32 and shared["width"].maximum == 640
    assert "关闭" in shared["exposure_us"].blockers["sim:1"]
    assert "单位" in shared["gain_db"].blockers["sim:1"]


def test_batch_restores_distinct_camera_values_then_new_plan_captures_expected_frames(tmp_path):
    left, right = RecordingCamera(0), RecordingCamera(1)
    right.set_features({"exposure_us": 1700.0, "pixel_format": "Mono14"})
    snapshots = [camera.features() for camera in (left, right)]
    config, catalog, registry = setup_batch(left, right, settings=(
        {"offset_x": 20, "gain_db": 2.0}, {"offset_x": 34},
    ))
    changes = {"exposure_us": 1250.0, "pixel_format": "Mono12", "width": 96, "height": 64}
    batch = verify_batch_settings(config, changes, catalog, registry)
    assert [row.state for row in batch.rows] == ["verified", "verified"]
    assert [row.observed for row in batch.rows] == [changes, changes]
    assert [row.before["pixel_format"] for row in batch.rows] == ["Mono8", "Mono14"]
    assert [camera.features() for camera in (left, right)] == snapshots
    assert batch.plan.cameras[0].settings == {**changes, "offset_x": 20, "gain_db": 2.0}
    assert batch.plan.cameras[1].settings == {**changes, "offset_x": 34}
    assert "width" not in config.cameras[0].settings

    capture = CaptureSession(batch.plan, tmp_path / "experiment", registry=registry)
    capture.start()
    assert capture.wait(10)["state"] == "completed"
    with ExperimentReader(capture.output) as reader:
        frames = reader.frames()
        assert {frame["camera_id"] for frame in frames} == {"sim:0", "sim:1"}
        for frame in frames:
            pixels = tifffile.imread(reader.image_path(frame))
            assert pixels.shape == (64, 96) and pixels.dtype.name == "uint16"
            assert frame["pixel_format"] == "Mono12" and frame["significant_bits"] == 12


def test_batch_blocks_all_writes_if_one_camera_rejects_an_increment():
    class FineOffsets(RecordingCamera):
        def features(self):
            features = super().features()
            features["width"] = replace(features["width"], increment=1)
            return features

    left, right = FineOffsets(0), RecordingCamera(1)
    config, catalog, registry = setup_batch(left, right)
    result = verify_batch_settings(config, {"width": 95}, catalog, registry)
    assert result.plan is None
    assert [row.state for row in result.rows] == ["skipped", "blocked"]
    assert result.rows[1].issues[0]["reason"] == "FEATURE_INCREMENT"
    assert not left.writes and not right.writes


def test_batch_reports_quantized_readback_and_restores_a_failed_trial():
    class RoundsExposure(RecordingCamera):
        def set_feature(self, name, value):
            if name == "exposure_us" and value == 1250.0:
                value = 1200.0
            super().set_feature(name, value)

    left, right = RoundsExposure(0), RecordingCamera(1)
    config, catalog, registry = setup_batch(left, right)
    result = verify_batch_settings(config, {"exposure_us": 1250.0}, catalog, registry)
    assert result.plan is None
    assert [row.state for row in result.rows] == ["rejected", "verified"]
    assert result.rows[0].observed["exposure_us"] == 1200.0
    assert result.rows[0].requested["exposure_us"] == 1250.0
    assert left.features()["exposure_us"].value == right.features()["exposure_us"].value == 2000.0


def test_batch_stops_later_writes_when_one_camera_cannot_be_restored():
    class CannotRestore(RecordingCamera):
        def set_features(self, changes):
            if changes.get("width") == 640:
                raise CaptureError("FEATURE_WRITE_FAILED", "injected restore failure")
            super().set_features(changes)

    left, right = CannotRestore(0), RecordingCamera(1)
    config, catalog, registry = setup_batch(left, right)
    result = verify_batch_settings(config, {"width": 96}, catalog, registry)
    assert result.plan is None
    assert [row.state for row in result.rows] == ["restore_failed", "skipped"]
    assert left.features()["width"].value == 96 and not right.writes
    assert "restore failure" in result.rows[0].detail


def test_batch_restores_parameters_written_before_a_device_error():
    class PartialWrite(RecordingCamera):
        def set_feature(self, name, value):
            if name == "height" and value == 64:
                raise CaptureError("FEATURE_WRITE_FAILED", "height rejected after width")
            super().set_feature(name, value)

    left, right = PartialWrite(0), RecordingCamera(1)
    config, catalog, registry = setup_batch(left, right)
    result = verify_batch_settings(config, {"width": 96, "height": 64}, catalog, registry)
    assert result.plan is None
    assert [row.state for row in result.rows] == ["rejected", "verified"]
    assert result.rows[0].observed == {"width": 96, "height": 480}
    assert "height rejected" in result.rows[0].detail
    assert left.writes == [("width", 96), ("width", 640)]
    assert left.features()["width"].value == right.features()["width"].value == 640


def test_batch_rejects_a_replaced_camera_without_touching_the_others():
    left, right = RecordingCamera(0), RecordingCamera(1)
    config, catalog, registry = setup_batch(left, right)
    right.device = replace(right.device, serial="replacement")
    result = verify_batch_settings(config, {"width": 96}, catalog, registry)
    assert result.plan is None
    assert [row.state for row in result.rows] == ["skipped", "blocked"]
    assert "身份已变化" in result.rows[1].detail
    assert not left.writes and not right.writes
