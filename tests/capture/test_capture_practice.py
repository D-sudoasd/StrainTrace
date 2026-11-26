import pytest

from straintrace_capture import CaptureError, CaptureSession, ExperimentReader
from straintrace_capture.cameras import CameraRegistry
from straintrace_capture.handoff import launch_capture
from straintrace_capture.practice import PracticeCameraRegistry, practice_config


def test_practice_only_discovers_and_opens_simulated_cameras(monkeypatch):
    original = CameraRegistry._provider
    called = []

    def provider(registry, name):
        called.append(name)
        assert name == "sim"
        return original(registry, name)

    monkeypatch.setattr(CameraRegistry, "_provider", provider)
    registry = PracticeCameraRegistry()
    report = registry.discover()
    assert set(report["providers"]) == {"sim"}
    assert {device["id"] for device in report["devices"]} == {"sim:0", "sim:1"}
    assert registry.discover(("pylon", "gentl")) == {"devices": [], "providers": {}}
    with pytest.raises(CaptureError) as error:
        registry.camera("pylon:test")
    assert error.value.code == "PRACTICE_CAMERA_REQUIRED"
    assert registry.camera("sim:0").device.backend == "sim"
    assert called == ["sim", "sim"]


def test_practice_uses_the_normal_reference_and_test_image_ledger(tmp_path):
    output = tmp_path / "模拟练习"
    capture = CaptureSession(practice_config(), output, registry=PracticeCameraRegistry())
    capture.start()
    assert capture.wait(15)["state"] == "completed"
    with ExperimentReader(output) as reader:
        rows = reader.frames()
        assert [row["role"] for row in rows] == ["reference"] + ["test"] * 12
        assert {row["significant_bits"] for row in rows} == {12}
        report = reader.verify()
        assert report["complete"] and report["file_hashes_checked"] and report["committed_images"] == 13


def test_practice_launch_uses_the_capture_entrypoint_without_an_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr("straintrace_capture.handoff._workbench_command", lambda *_: ["capture"])
    monkeypatch.setattr("straintrace_capture.handoff._launch_workbench", tuple)
    assert launch_capture(practice=True) == ("capture", "--practice")
    assert launch_capture(tmp_path) == ("capture", "--experiment", str(tmp_path.resolve()))
    with pytest.raises(ValueError):
        launch_capture(tmp_path, practice=True)
