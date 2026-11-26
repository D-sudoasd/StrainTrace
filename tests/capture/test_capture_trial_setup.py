from dataclasses import replace

from straintrace_capture import CaptureConfig, CaptureSession, inspect_plan, prepare_trial, trial_report
from straintrace_capture.cameras.base import Feature
from straintrace_capture.cameras.settings_probe import readbacks_affected_by_pending_writes
from straintrace_capture.presentation import format_readiness
from straintrace_capture.readiness import probe_camera_setup
from straintrace_capture.trial_progress import inspect_trial_progress, summarize_trial_progress
from straintrace_capture.trial_reference import archive_trial_reference

from test_capture_continuation_setup import _registry


def _plan(*, parameters=None, two_phases=False):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "采前回读", "cameras": [{"id": "sim:0", "settings": {
            "width": 96, "height": 80, "pixel_format": "Mono12", **(parameters or {}),
        }}],
        "phases": ([{"mode": "single", "role": "reference"}] if two_phases else []) + [{"mode": "single"}],
    })


def _trial(config, formal, exposure, phase=1):
    prepared, output = prepare_trial(config, formal, phase_number=phase)
    session = CaptureSession(prepared, output, registry=_registry(exposure, "Mono12", []))
    session.start()
    assert session.wait(5)["state"] == "completed"
    return trial_report(output)


def _preflight(config, formal, reports, *, evidence=None, exposure=2000.0):
    if evidence is None:
        writes = []
        _, evidence = probe_camera_setup(config, _registry(exposure, "Mono12", writes), include_readback=True)
        assert writes == []
    return inspect_plan(config, formal, trial_reports=reports, camera_settings=evidence)


def _warning(report):
    return any(item["code"] == "TRIAL_CAMERA_READBACK_CHANGED" for item in report["issues"])


def test_readonly_preflight_detects_default_drift_and_preserves_the_confirmed_evidence(tmp_path):
    config = _plan()
    formal = tmp_path / "正式采集"
    trial = _trial(config, formal, 1200.0)
    assert trial["observed"] and "exposure_us" not in config.cameras[0].settings
    before = inspect_plan(config, formal, trial_reports=[trial])
    assert not _warning(before) and before["trial_setting_suggestions"] == {}
    assert "尚未读取相机" in format_readiness(before)
    report = _preflight(config, formal, [trial])
    row, = report["trial_assessments"]
    assert report["ready"] and row["status"] == "matched" and _warning(report)
    progress = summarize_trial_progress(config.phases, report["trial_assessments"], issues=report["issues"])
    stage, = progress.phases
    assert progress.checked_plan and not progress.blockers and progress.recommended == 1
    assert stage.status == "review" and stage.issues[0]["code"] == "TRIAL_CAMERA_READBACK_CHANGED"
    assert report["phases"][0]["trial_geometry_cameras"] == []
    assert report["trial_setting_suggestions"] == {"sim:0": {"exposure_us": 1200.0}}
    evidence = row["camera_readbacks"]["sim:0"]
    assert evidence["basis"] == "device_features" and evidence["identity"] == "matched"
    changed, = [item for item in evidence["fields"] if item["status"] == "changed"]
    assert changed == {"field": "exposure_us", "recorded": 1200.0, "current": 2000.0,
                       "unit": "us", "current_unit": "us", "status": "changed"}
    assert "曝光时间  试采 1200 µs → 当前 2000 µs" in "\n".join(format_readiness(report).splitlines()[:5])
    stage, = archive_trial_reference(report, [trial], readback_warnings_confirmed=True)["stages"]
    assert stage["readback_warnings_confirmed"] and stage["preflight_camera_readbacks"]["sim:0"] == evidence
    assert not formal.exists()


def test_trial_progress_recommends_experimental_phases_and_identifies_stale_evidence(tmp_path):
    plan = _plan(two_phases=True)
    output = tmp_path / "正式采集"
    initial = inspect_trial_progress(plan, output, ())
    assert not initial.checked_plan and initial.recommended == 2
    assert [row.status for row in initial.phases] == ["missing", "missing"]
    reference_phase = plan.phases[0]
    optional = summarize_trial_progress((reference_phase, replace(reference_phase, role="calibration")), ())
    assert optional.recommended is None and len(optional.phases) == 2

    trial = _trial(plan, output, 1200.0, phase=2)
    observed = inspect_trial_progress(plan, output, (trial,))
    assert observed.recommended is None and observed.phases[0].status == "missing"
    assert observed.phases[1].status == "matched" and observed.phases[1].record["experiment_id"] == trial["experiment_id"]

    reference_trial = _trial(plan, output, 1200.0, phase=1)
    compared = inspect_trial_progress(plan, output, (trial, reference_trial))
    reference_warning = {"phase": 1, "code": "TRIAL_CAMERA_READBACK_CHANGED", "severity": "warning"}
    warning = summarize_trial_progress(plan.phases, [row.record for row in compared.phases], issues=[reference_warning])
    assert warning.recommended == 1 and warning.phases[0].status == "review"

    edited = plan.to_dict()
    edited["phases"][1] = {"mode": "manual", "frame_count": 2}
    changed = inspect_trial_progress(CaptureConfig.from_dict(edited), output, (trial,))
    assert changed.recommended == 2 and changed.phases[1].status == "changed"
    assert changed.phases[1].record["changes"] == ["phase"]

    alerted = {**observed.phases[1].record, "image_checks": {"sim:0": {"flagged": 1}}}
    issues = [{"phase": 1, "code": "CAMERA_TRIGGER_UNSUPPORTED", "severity": "block"}]
    checked = summarize_trial_progress(plan.phases, (alerted,), issues=issues)
    assert checked.checked_plan and checked.blockers == 1 and checked.recommended == 2
    assert checked.phases[0].issues == tuple(issues)
    assert checked.phases[1].status == "review" and checked.phases[1].image_alerts == 1

    changed_preview = {**alerted, "status": "changed", "changes": ["preview_geometry"]}
    checked = summarize_trial_progress(plan.phases, (changed_preview,), issues=[])
    assert checked.phases[1].status == "changed" and checked.recommended == 2
    assert not output.exists()


def test_explicit_pending_parameters_and_linked_genapi_limits_do_not_report_drift(tmp_path):
    config = _plan(parameters={"exposure_us": 1200.0})
    formal = tmp_path / "planned"
    trial = _trial(config, formal, 1200.0)
    report = _preflight(config, formal, [trial])
    fields = report["trial_assessments"][0]["camera_readbacks"]["sim:0"]["fields"]
    assert any(item["field"] == "exposure_us" and item["status"] == "pending" for item in fields)
    assert any("计划应用后再核对" in line and "曝光时间" in line for line in format_readiness(report).splitlines())
    assert not _warning(report) and report["trial_setting_suggestions"] == {}
    pending = readbacks_affected_by_pending_writes("pylon", {"width": 160}, {"width": Feature("width", 96)})
    assert {"exposure_us", "fps", "width", "offset_x"} <= pending


def test_proposals_require_compatible_units_range_device_and_agreement_across_phases(tmp_path):
    config = _plan(two_phases=True)
    formal = tmp_path / "multi"
    first = _trial(config, formal, 1200.0)
    second = _trial(config, formal, 1400.0, phase=2)
    report = _preflight(config, formal, [first, second])
    assert _warning(report) and report["trial_setting_suggestions"] == {}
    _, evidence = probe_camera_setup(config, _registry(2000.0, "Mono12", []), include_readback=True)
    camera = evidence["sim:0"]
    exposure = camera.readback["exposure_us"]
    alternatives = [
        replace(camera, readback={**camera.readback, "exposure_us": replace(exposure, minimum=1500)}),
        replace(camera, readback={**camera.readback, "exposure_us": replace(exposure, unit=None)}),
        replace(camera, readback={**camera.readback, "exposure_auto": Feature("exposure_auto", "Continuous")}),
        replace(camera, device=replace(camera.device, serial="replacement")),
    ]
    for candidate in alternatives:
        assert _preflight(config, formal, [first], evidence={"sim:0": candidate})["trial_setting_suggestions"] == {}
    with_preview = replace(camera, basis="preview_readback")
    report = _preflight(config, formal, [first], evidence={"sim:0": with_preview})
    assert "预览回读" in format_readiness(report) and _warning(report)
    assert report["trial_setting_suggestions"] == {"sim:0": {"exposure_us": 1200.0}}
