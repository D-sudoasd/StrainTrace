import json
from contextlib import contextmanager

from straintrace_capture import CaptureConfig, CaptureError, inspect_plan
from straintrace_capture.cameras.base import Feature, TriggerSupport
from straintrace_capture.cameras.settings_probe import check_camera_settings
from straintrace_capture.cli import main
from straintrace_capture.presentation import format_readiness
from straintrace_capture.readiness import probe_camera_setup, probe_preflight_cameras


def _plan(identity, settings):
    return CaptureConfig.from_dict({
        "config_version": 1,
        "name": "settings preflight",
        "cameras": [{"id": identity, "settings": settings}],
        "phases": [{"mode": "continuous", "frame_count": 1}],
    })


def test_device_probe_shares_one_connection_and_reports_current_constraints(tmp_path):
    opened = []

    class Camera:
        def features(self):
            assert opened == ["gentl:viky"]
            return {
                "width": Feature("width", 400, 100, 800, 1, unit="px"),
                "height": Feature("height", 400, 100, 800, 2, unit="px"),
            }

        def trigger_support(self):
            assert opened == ["gentl:viky"]
            return TriggerSupport(False, False, "device_nodes")

    class Registry:
        @contextmanager
        def camera(self, identity):
            opened.append(identity)
            yield Camera()
            opened.append("closed")

    config = _plan("gentl:viky", {"width": 96, "height": 301, "unknown": 1})
    triggers, settings = probe_camera_setup(config, Registry())
    assert opened == ["gentl:viky", "closed"]
    assert triggers["gentl:viky"].software is False
    output = tmp_path / "formal"
    report = inspect_plan(config, output, camera_settings=settings, trigger_support=triggers)
    assert not report["ready"] and not output.exists()
    assert report["camera_settings"]["gentl:viky"]["basis"] == "device_features"
    assert [(item["feature"], item["reason"]) for item in report["issues"] if item["code"] == "CAMERA_SETTING_INVALID"] == [
        ("width", "FEATURE_RANGE"), ("height", "FEATURE_INCREMENT"), ("unknown", "FEATURE_UNAVAILABLE"),
    ]
    assert "当前范围 100–800 像素" in format_readiness(report)
    assert "步长 2 像素" in format_readiness(report)


def test_cli_declares_a_failed_exact_connection_before_trial_output(tmp_path, capsys, monkeypatch):
    identity = "uvc:1"
    attempts = []

    class Registry:
        def discover(self, backends):
            return {"devices": [{"id": "uvc:2"}], "providers": {"uvc": {"state": "ready", "count": 1}}}

        @contextmanager
        def camera(self, selected):
            attempts.append(selected)
            if selected == identity:
                raise CaptureError("CAMERA_ID_UNRESOLVED", "fixture camera unplugged")
            yield

    monkeypatch.setattr("straintrace_capture.cli.CameraRegistry", lambda **_kwargs: Registry())
    config = _plan(identity, {})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    flags = ["--config", str(path), "--output", str(tmp_path / "formal")]
    assert main(["preflight", *flags, "--discover"]) == 0
    discovery = json.loads(capsys.readouterr().out)
    assert not attempts and any(row["code"] == "CAMERAS_NOT_DISCOVERED" for row in discovery["issues"])
    assert main(["preflight", *flags, "--discover", "--probe-cameras"]) == 3
    report = json.loads(capsys.readouterr().out)
    assert [(row["code"], row["camera"], row["reason"]) for row in report["issues"] if row["severity"] == "block"] == [
        ("CAMERA_CONNECTION_CHECK_FAILED", identity, "CAMERA_ID_UNRESOLVED"),
    ]
    assert report["camera_connections"][identity]["state"] == "open_failed"
    assert main(["trial", *flags]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["camera_connections"][identity]["state"] == "open_failed"
    assert attempts == [identity, identity] and list(tmp_path.iterdir()) == [path]


def test_cli_keeps_an_openable_alias_when_inventory_and_readback_are_incomplete(tmp_path, capsys, monkeypatch):
    identity = "uvc:1"
    actions = []

    class Device:
        def features(self):
            actions.append("read")
            raise OSError("fixture feature catalog unavailable")

    class Registry:
        def discover(self, backends):
            return {"devices": [{"id": "uvc:dshow:stable-path"}], "providers": {}}

        @contextmanager
        def camera(self, selected):
            actions.append(selected)
            try:
                yield Device()
            finally:
                actions.append("closed")

    monkeypatch.setattr("straintrace_capture.cli.CameraRegistry", lambda **_kwargs: Registry())
    config = _plan(identity, {"width": 320})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    output = tmp_path / "formal"
    assert main(["preflight", "--config", str(path), "--output", str(output), "--discover", "--probe-cameras"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] and not output.exists() and actions == [identity, "read", "closed"]
    assert report["camera_connections"][identity]["state"] == "available"
    issues = {row["code"] for row in report["issues"]}
    assert "CAMERA_SETTINGS_PROBE_FAILED" in issues and "CAMERAS_NOT_DISCOVERED" not in issues


def test_preflight_reports_a_connection_release_failure(tmp_path):
    identity = "uvc:0"

    class Registry:
        @contextmanager
        def camera(self, selected):
            assert selected == identity
            yield
            raise OSError("fixture could not close the connection")

    config = _plan(identity, {})
    probe = probe_preflight_cameras(config, Registry(), triggers=False)
    report = inspect_plan(config, tmp_path / "formal", camera_connections=probe.connections)
    assert not report["ready"] and report["camera_connections"][identity]["state"] == "close_failed"
    issue, = [row for row in report["issues"] if row["severity"] == "block"]
    assert issue["camera"] == identity and issue["reason"] == "CAMERA_CLOSE_FAILED"
    assert "关闭这台相机的连接时出错" in format_readiness(report)


def test_checks_defer_limits_that_depend_on_preceding_writes():
    features = {
        "width": Feature("width", 320, 100, 400, 2, unit="px"),
        "offset_x": Feature("offset_x", 80, 0, 180, 2, unit="px"),
        "exposure_auto": Feature("exposure_auto", "Continuous", choices=("Off", "Continuous")),
        "exposure_us": Feature("exposure_us", 1200.0, 50.0, 3000.0, writable=False),
    }
    checked = check_camera_settings("pylon", {
        "width": 460, "offset_x": 0, "exposure_auto": "Off", "exposure_us": 4000.0,
    }, features)
    assert not checked.invalid
    assert checked.deferred == ("exposure_auto", "width", "offset_x", "exposure_us")
    auto_only = check_camera_settings("pylon", {"exposure_auto": "Off", "exposure_us": 4000.0}, features)
    assert auto_only.checked == ("exposure_auto",)
    assert auto_only.deferred == ("exposure_us",)
    invalid_type = check_camera_settings("pylon", {"width": "460", "offset_x": 0}, features)
    assert [issue["reason"] for issue in invalid_type.invalid] == ["FEATURE_TYPE"]
    assert check_camera_settings("pylon", {"width": 460}, features).invalid[0]["reason"] == "FEATURE_RANGE"


def test_simulated_setting_blockers_prevent_cli_run_and_trial_creation(tmp_path, capsys):
    config = _plan("sim:0", {"width": 4000, "offset_x": 100})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    for command in ("run", "trial"):
        output = tmp_path / command
        assert main([command, "--config", str(path), "--output", str(output)]) == 3
        response = json.loads(capsys.readouterr().out)
        assert not response["ready"]
        issue, = (row for row in response["issues"] if row["code"] == "CAMERA_SETTING_INVALID")
        assert issue["feature"] == "offset_x" and issue["maximum"] == 96
        assert not output.exists()
    assert list(tmp_path.iterdir()) == [path]
