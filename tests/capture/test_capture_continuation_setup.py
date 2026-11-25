import json
import time
from contextlib import closing
from dataclasses import asdict, replace

import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader, inspect_plan
from straintrace_capture.cameras.base import Feature
from straintrace_capture.cameras.settings_probe import SettingsCheck
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.continuation import CONTINUATION_REFERENCE, continuation_plan, inspect_continuation
from straintrace_capture.presentation import format_integrity, format_readiness
from straintrace_capture.readiness import probe_camera_setup
from straintrace_capture.storage import ExperimentWriter, sha256_file
from straintrace_capture.trial import prepare_trial


def _plan(*, cameras=None, phases=None):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "repeat setup", "segment_frames": 3,
        "cameras": cameras or [{"id": "sim:0", "settings": {"width": 160, "height": 120}}],
        "phases": phases or [{"mode": "single", "role": "reference"}, {"mode": "single", "role": "test"}],
    })


def _registry(exposure, pixel_format, writes):
    class ChangedDefaults(SimulatedCamera):
        def __init__(self):
            super().__init__(0)
            self._values.update(exposure_us=exposure, pixel_format=pixel_format)

        def set_feature(self, name, value):
            writes.append((name, value))
            super().set_feature(name, value)

    return CameraRegistry(custom={"sim": (lambda: [ChangedDefaults().device], lambda _: ChangedDefaults())})


def _record(config, path, registry):
    session = CaptureSession(config, path, registry=registry)
    session.start()
    status = session.wait(8)
    assert status["state"] == "completed", status
    with ExperimentReader(path) as reader:
        report = reader.verify()
        assert report["complete"], report
        return reader.experiment["experiment_id"], report


def test_reports_default_changes_and_compares_the_new_phase_readbacks(tmp_path):
    original_path = tmp_path / "原实验"
    old_id, _ = _record(_plan(), original_path, _registry(1200., "Mono12", []))
    raw_before = {path.relative_to(original_path): sha256_file(path)
                  for path in original_path.rglob("*") if path.suffix in (".sqlite3", ".tiff")}
    plan = continuation_plan(original_path, expected_experiment_id=old_id)
    assert "exposure_us" not in plan.cameras[0].settings
    reference = plan.metadata[CONTINUATION_REFERENCE]
    assert reference["initial"]["sim:0"]["exposure_us"]["value"] == 1200

    writes = []
    registry = _registry(3600., "Mono8", writes)
    support, settings = probe_camera_setup(plan, registry)
    report = inspect_plan(plan, tmp_path / "下一轮", trigger_support=support, camera_settings=settings)
    assert not writes and not (tmp_path / "下一轮").exists()
    assert report["ready"] and any(row["code"] == "CONTINUATION_SETUP_CHANGED" for row in report["issues"])
    setup = report["continuation"]
    rows = {row["field"]: row for row in setup["readbacks"]["sim:0"]["fields"]}
    assert rows["width"]["status"] == "will_apply"
    assert rows["exposure_us"]["recorded"] == 1200 and rows["exposure_us"]["current"] == 3600
    assert rows["exposure_us"]["status"] == "readback_changed"
    assert rows["pixel_format"]["status"] == "readback_changed"
    assert "曝光时间  原记录 1200 µs · 当前 3600 µs" in format_readiness(report)
    proposed = setup["suggested_settings"]["sim:0"]
    assert proposed["exposure_us"] == 1200 and proposed["pixel_format"] == "Mono12"
    assert "width" not in proposed

    _, changed = _record(plan, tmp_path / "保留当前参数", registry)
    checks = changed["continuation_checks"]
    assert len(checks) == 2
    for phase in checks:
        assert phase["source_experiment_id"] == old_id
        assert {item["field"] for item in phase["fields"]} == {"exposure_us", "pixel_format"}
    assert "原记录 1200 µs → 本次 3600 µs" in format_integrity(changed)

    draft = plan.to_dict()
    draft["cameras"][0]["settings"].update(proposed)
    corrected = CaptureConfig.from_dict(draft)
    _, evidence = _record(corrected, tmp_path / "采用原记录", registry)
    assert all(row["matched_fields"] and not row["fields"] for row in evidence["continuation_checks"])
    with ExperimentReader(tmp_path / "采用原记录") as reader:
        assert {row["pixel_format"] for row in reader.frames()} == {"Mono12"}
    assert {path: sha256_file(original_path / path) for path in raw_before} == raw_before
    trial, _ = prepare_trial(corrected, tmp_path / "计划", phase_number=2, seconds=2)
    assert CONTINUATION_REFERENCE not in trial.metadata


def test_keeping_explicit_changes_and_not_reusing_dynamic_or_unreadable_values(tmp_path):
    previous = tmp_path / "legacy"
    camera = SimulatedCamera(0)
    source_settings = {"width": Feature("width", 160, 32, 4096, unit="px"),
                       "height": Feature("height", 120, 32, 3072, unit="px"),
                       "exposure_auto": Feature("exposure_auto", "Continuous"),
                       "exposure_us": Feature("exposure_us", 1200., 1., 4000., unit="us"),
                       "gain_db": Feature("gain_db", 3., 0., 20., unit="dB")}
    with closing(ExperimentWriter(previous, _plan())) as writer:
        writer.register_camera(camera.device, {name: asdict(value) for name, value in source_settings.items()})
        for number, gain in ((1, 3.), (2, 6.)):
            phase = {**source_settings, "gain_db": Feature("gain_db", gain, unit="dB")}
            writer.event("phase_started", {"phase": number,
                                           "camera_features": {camera.device.id: {name: asdict(value)
                                                                                for name, value in phase.items()}}})
        writer.set_state("stopped")
    available = {**source_settings, "gain_db": Feature("gain_db", 3., unit=None)}
    evidence = {camera.device.id: SettingsCheck("device_features", readback=available, device=camera.device)}
    baseline = inspect_plan(continuation_plan(previous), tmp_path / "baseline", camera_settings=evidence)["continuation"]
    rows = {item["field"]: item for item in baseline["readbacks"]["sim:0"]["fields"]}
    assert "exposure_us" not in rows
    assert rows["gain_db"]["status"] == "unverified"
    assert baseline["suggested_settings"] == {"sim:0": {"exposure_auto": "Continuous"}}
    for device, identity in ((replace(camera.device, serial="different"), "changed"),
                             (replace(camera.device, identity_scope="enumeration_index"), "unverified")):
        candidate = inspect_plan(
            continuation_plan(previous), tmp_path / "identity", camera_settings={
                camera.device.id: SettingsCheck("device_features", readback=available, device=device),
            },
        )["continuation"]
        assert candidate["readbacks"][camera.device.id]["identity"] == identity
        assert not candidate["suggested_settings"]

    draft = _plan(cameras=[{"id": "sim:0", "settings": {"width": 192}}, {"id": "sim:1"}],
                  phases=[{"mode": "single", "role": "reference"}, {"mode": "continuous", "frame_count": 2}])
    plan = continuation_plan(previous, draft)
    report = inspect_plan(plan, tmp_path / "adjusted", camera_settings=evidence)["continuation"]
    assert report["added_cameras"] == ["sim:1"]
    assert report["phase_changes"][0]["phase"] == 2 and "mode" in report["phase_changes"][0]["fields"]
    assert not report["suggested_settings"]
    assert any(row["field"] == "width" and row["status"] == "plan_changed"
               for row in report["readbacks"]["sim:0"]["fields"])


def test_cli_can_check_a_continuation_without_creating_an_experiment(tmp_path, capsys):
    from straintrace_capture.cli import main

    source = tmp_path / "source"
    identity, _ = _record(_plan(), source, _registry(1200., "Mono12", []))
    config = tmp_path / "draft.json"
    config.write_text(json.dumps(_plan().to_dict()), encoding="utf-8")
    output = tmp_path / "target"
    args = ["preflight", "--config", str(config), "--output", str(output), "--continue-from", str(source), "--probe-cameras"]
    assert main(args) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["continuation"]["source_experiment_id"] == identity
    assert observed["continuation"]["suggested_settings"]["sim:0"]["exposure_us"] == 1200
    assert not output.exists()


def _wait_until(predicate):
    deadline = time.monotonic() + 6
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.01)


def test_continue_from_interrupted_stage_keeps_source_phase_ids_and_file_hashes(tmp_path, capsys):
    from straintrace_capture.cli import main

    path = tmp_path / "原实验"
    source = CaptureSession(_plan(phases=[
        {"mode": "single", "role": "reference"}, {"mode": "manual", "frame_count": 3},
        {"mode": "single", "role": "test"},
    ]), path, registry=_registry(1200., "Mono12", []))
    source.start()
    try:
        _wait_until(lambda: source.status()["mode"] == "manual")
        source.request_trigger(expected_phase=2)
        _wait_until(lambda: source.status()["cameras"]["sim:0"]["saved"] == 2)
    finally:
        source.request_stop()
    assert source.wait(8)["state"] == "stopped"
    report = inspect_continuation(path)
    assert [row["state"] for row in report["stages"]] == ["ended", "interrupted", "not_started"]
    assert [row["saved_by_camera"] for row in report["stages"]] == [{"sim:0": 1}, {"sim:0": 1}, {"sim:0": 0}]
    assert report["has_suggested_restart"] and report["suggested_phase"] == 2
    originals = {file: sha256_file(file) for file in path.rglob("*") if file.suffix in (".sqlite3", ".tiff")}
    assert len(originals) == 3

    assert main(["prepare-continuation", str(path), "--inspect"]) == 0
    assert json.loads(capsys.readouterr().out) == report
    assert main(["prepare-continuation", str(path), "--start-phase", "2"]) == 0
    config = CaptureConfig.from_dict(json.loads(capsys.readouterr().out))
    assert config == continuation_plan(path, start_phase=2, expected_experiment_id=report["experiment_id"])
    assert continuation_plan(path, config) == config
    assert [phase.mode for phase in config.phases] == ["manual", "single"]
    assert config.phases[0].frame_count == 3
    assert config.metadata[CONTINUATION_REFERENCE]["source_start_phase"] == 2
    assert [phase["mode"] for phase in config.metadata[CONTINUATION_REFERENCE]["phase_plans"]] == ["manual", "single"]
    target = tmp_path / "第二轮"
    preflight = inspect_plan(config, target)
    assert not target.exists() and not preflight["continuation"]["phase_changes"]
    assert "本轮阶段 1 对照原阶段 2" in format_readiness(preflight)

    next_run = CaptureSession(config, target, registry=_registry(1200., "Mono12", []))
    next_run.start()
    try:
        _wait_until(lambda: next_run.status()["mode"] == "manual")
        for index in range(1, 4):
            next_run.request_trigger(expected_phase=1)
            _wait_until(lambda: next_run.status()["cameras"]["sim:0"]["saved"] >= index)
        assert next_run.wait(8)["state"] == "completed"
    finally:
        if not next_run.finished:
            next_run.request_stop()
            next_run.wait(8)
    with ExperimentReader(target) as reader:
        verification = reader.verify()
        assert verification["complete"] and verification["committed_frames"] == 4
        assert reader.experiment["config"]["metadata"]["continuation_of"] == report["experiment_id"]
        assert [(row["phase"], row["source_phase"]) for row in verification["continuation_checks"]] == [(1, 2), (2, 3)]
        assert "阶段 1（原阶段 2）" in format_integrity(verification)
    assert {file: sha256_file(file) for file in originals} == originals


def test_continuation_checks_identity_terminal_state_and_new_hardware_anchors(tmp_path):
    source = tmp_path / "硬件实验"
    plan = _plan(phases=[
        {"mode": "single", "role": "reference"},
        {"mode": "hardware", "frame_count": 2, "hardware_source": "Line1",
         "hardware_pairing": {"source_id": "ttl", "counter_feature": "ChunkFramecounter",
                              "first_counter": {"sim:0": 620}, "first_event_number": 101}},
    ])
    with closing(ExperimentWriter(source, plan)) as writer:
        writer.set_state("running")
        with pytest.raises(CaptureError, match="结束采集"):
            inspect_continuation(source)
        writer.set_state("interrupted")
    report = inspect_continuation(source)
    assert report["stages"][1]["hardware_pairing"]
    assert report["suggested_phase"] == 1
    prepared = continuation_plan(source, start_phase=2)
    assert prepared.phases[0].hardware_pairing is None
    assert prepared.phases[0].hardware_source == "Line1"
    with pytest.raises(CaptureError, match="标识已变化"):
        continuation_plan(source, start_phase=2, expected_experiment_id="replaced")
    with pytest.raises(CaptureError, match="起点"):
        continuation_plan(source, start_phase=3)


def test_arming_failure_is_attributed_to_the_phase_being_opened(tmp_path):
    class FailsOnSecondStart(SimulatedCamera):
        starts = 0

        def start(self, trigger="free", source=None):
            self.starts += 1
            if self.starts == 2:
                raise CaptureError("CAMERA_DISCONNECTED", "fixture disconnected during second start")
            return super().start(trigger, source)

    previous = tmp_path / "disconnected"
    device = FailsOnSecondStart(0)
    registry = CameraRegistry(custom={"sim": (lambda: [device.device], lambda _: device)})
    session = CaptureSession(_plan(), previous, registry=registry)
    session.start()
    assert session.wait(8)["state"] == "failed"
    report = inspect_continuation(previous)
    first, second = report["stages"]
    assert (first["state"], first["saved_frames"], first["issues"]) == ("ended", 1, {})
    assert (second["state"], second["saved_frames"], second["issues"]) == ("needs_review", 0, {"error": 1})
    assert not second.get("phase_started") and report["suggested_phase"] == 2
    with ExperimentReader(previous) as reader:
        error, = reader.events(kind="error")
        assert error["phase"] == 2
