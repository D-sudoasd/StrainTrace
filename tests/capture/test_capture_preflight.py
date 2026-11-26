import os
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader, inspect_plan, prepare_trial
from straintrace_capture.cameras._genapi import GenApi
from straintrace_capture.cameras.base import Image, TriggerSupport
from straintrace_capture.continuation import continuation_plan
from straintrace_capture.output import inspect_output
from straintrace_capture.presentation import format_readiness
from straintrace_capture.preview import image_quality
from straintrace_capture.readiness import apply_phase_suggestion, probe_trigger_support


def plan(phases, cameras=None, **options):
    return CaptureConfig.from_dict({
        "config_version": 1,
        "name": "preflight test",
        "cameras": cameras or [{"id": "sim:0", "settings": {"width": 64, "height": 48, "pixel_format": "Mono12"}}],
        "phases": phases,
        **options,
    })


def codes(report, *, severity=None):
    return {issue["code"] for issue in report["issues"] if severity is None or issue["severity"] == severity}


def test_preflight_binds_preview_bytes_and_minimum_phase_limits_without_writes(tmp_path):
    config = plan([
        {"mode": "single", "role": "reference"},
        {"mode": "timed", "duration_s": .15, "fps": 20, "frame_count": 8},
    ], cameras=[{"id": "uvc:stable-id", "settings": {}}], min_free_bytes=0)
    sample = Image(np.zeros((40, 60, 3), dtype=np.uint8), "BGR8", 8, "bgr", 1, 1)
    output = tmp_path / "new" / "experiment"
    report = inspect_plan(config, output, preview_images={"uvc:stable-id": sample}, discovered_ids={"uvc:stable-id"})
    assert report["ready"]
    assert [(issue["code"], issue["severity"], issue["phase"]) for issue in report["issues"]] == [
        ("DRIVER_HOST_FRAME_SELECTION", "info", 1), ("DRIVER_HOST_FRAME_SELECTION", "info", 2),
    ]
    assert report["trigger_support"]["uvc:stable-id"]["software_method"] == "host_next_receipt"
    assert "曝光同步需独立核实" in format_readiness(report)
    assert report["cameras"]["uvc:stable-id"]["basis"] == "preview"
    assert report["phases"][1]["frame_limit_per_camera"] == 3
    assert report["phases"][1]["scheduled_raw_bytes_per_s"] == 20 * sample.pixels.nbytes
    assert report["raw_payload_bytes"] == 4 * sample.pixels.nbytes
    assert report["volume_probe_path"] == str(tmp_path.resolve())
    assert not (tmp_path / "new").exists()


def test_preflight_reports_known_blockers_separately_from_unknown_geometry(tmp_path):
    config = plan([{"mode": "hardware", "hardware_source": "Line1", "frame_count": 1}],
                  cameras=[{"id": "uvc:input", "settings": {}}],
                  cti_files=[str(tmp_path / "missing.cti")], min_free_bytes=10**30)
    report = inspect_plan(config, tmp_path, discovered_ids=set())
    assert not report["ready"]
    assert codes(report, severity="block") == {
        "OUTPUT_EXISTS", "DISK_RESERVE_UNAVAILABLE", "DEPENDENCY_FILE_MISSING", "HARDWARE_TRIGGER_UNSUPPORTED",
    }
    assert "FRAME_GEOMETRY_UNAVAILABLE" in codes(report, severity="warning")
    assert "CAMERAS_NOT_DISCOVERED" in codes(report, severity="warning")
    assert report["raw_payload_bytes"] is None


def test_a_dangling_output_link_does_not_redirect_capture(tmp_path):
    output, target = tmp_path / "used-link", tmp_path / "missing-target"
    if sys.platform == "win32":
        target.mkdir()
        created = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "New-Item -ItemType Junction -Path $env:STRAINTRACE_TEST_LINK -Target $env:STRAINTRACE_TEST_TARGET -ErrorAction Stop"],
            env={**os.environ, "STRAINTRACE_TEST_LINK": str(output), "STRAINTRACE_TEST_TARGET": str(target)},
            capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, timeout=15,
        )
        assert created.returncode == 0, created.stderr
        target.rmdir()
    else:
        output.symlink_to(target, target_is_directory=True)
    config = plan([{"mode": "single"}], min_free_bytes=0)
    report = inspect_plan(config, output)
    assert codes(report, severity="block") == {"OUTPUT_EXISTS"}
    assert report["output"] == str(output)
    session = CaptureSession(config, output)
    with pytest.raises(CaptureError) as occupied:
        session.start()
    assert occupied.value.code == "OUTPUT_EXISTS"
    assert output.lstat() and not target.exists()


def test_output_suggestion_stays_with_the_plan_and_skips_existing_paths(tmp_path):
    output = tmp_path / "试样 A03"
    second = tmp_path / "试样 A03-run-02"
    output.mkdir()
    second.write_text("existing output", encoding="utf-8")
    inspection = inspect_output(output, suggest=True)
    assert inspection.suggestion == tmp_path / "试样 A03-run-03"
    report = inspect_plan(plan([{"mode": "single"}], min_free_bytes=0), output)
    assert report["output_suggestion"] == str(inspection.suggestion)
    assert not inspection.suggestion.exists()
    assert inspect_output(second, suggest=True).suggestion == inspection.suggestion
    assert inspect_output(tmp_path.anchor, suggest=True).suggestion is None
    assert inspect_output(output, repeat_source=output, suggest=True).suggestion is None

    invalid_parent = inspect_output(second / "new", suggest=True)
    assert [row["code"] for row in invalid_parent.issues] == ["OUTPUT_PARENT_NOT_DIRECTORY"]
    assert invalid_parent.suggestion is None and second.read_text(encoding="utf-8") == "existing output"


def test_trigger_catalog_blocks_triggered_phases_and_only_suggests_verified_independent_single_frames(tmp_path):
    left, right = "gentl:free-run", "gentl:other"
    config = plan([
        {"mode": "single", "role": "reference", "frame_timeout_s": 1.5},
        {"mode": "single", "role": "calibration"},
        {"mode": "timed", "fps": 5, "frame_count": 10},
        {"mode": "manual"},
        {"mode": "event", "fps": 5, "pre_frames": 2, "post_frames": 2},
        {"mode": "hardware", "hardware_source": "Line1", "frame_count": 1},
        {"mode": "continuous", "frame_count": 2},
    ], cameras=[{"id": key} for key in (left, right)])
    out = tmp_path / "trigger-check"
    image = Image(np.zeros((40, 64), dtype=np.uint8), "Mono8", 8, "mono", 1, 1)
    supports = {left: TriggerSupport(False, False, "device_nodes", ("TriggerMode",)),
                right: TriggerSupport(basis="device_nodes")}
    offline = inspect_plan(config, out, trigger_support=supports, preview_images={left: image})
    assert not offline["ready"] and not offline["phase_suggestions"]
    assert offline["issues"][0]["offer_preview"]
    report = inspect_plan(config, out, trigger_support=supports, preview_images={left: image, right: image})
    blocked = {item["phase"]: (item["code"], item["cameras"]) for item in report["issues"] if item["severity"] == "block"}
    assert blocked == {**{index: ("SOFTWARE_TRIGGER_UNSUPPORTED", [left]) for index in range(1, 6)},
                       6: ("HARDWARE_TRIGGER_UNSUPPORTED", [left])}
    proposal, *host = report["phase_suggestions"]
    assert proposal == {"phase": 1, "replacement": {"mode": "continuous", "role": "reference", "frame_count": 1,
                                                    "frame_timeout_s": 1.5},
                        "basis": "preview", "pairing": "independent_receipts"}
    assert [item["phase"] for item in host] == [3, 4, 5]
    for item in host:
        old = config.phases[item["phase"] - 1].to_dict()
        assert item["replacement"] == {**old, "software_trigger_fallback": True}
        assert item["pairing"] == "host_receipt"
    patched = plan([host[0]["replacement"]], cameras=[{"id": key} for key in (left, right)])
    ready = inspect_plan(patched, out, trigger_support=supports)
    assert ready["ready"] and "HOST_FRAME_SELECTION" in codes(ready, severity="warning")
    assert "阶段 3 可改为连续采集" in format_readiness(report, phase_offset=2, camera_labels={left: "Viky 0"})
    assert "双相机标定需要同一触发" in format_readiness(report)
    assert config.phases[0].mode == "single" and not out.exists()


def test_trigger_probe_reads_only_the_catalog_and_keeps_failed_reads_unknown(tmp_path):
    class Catalog:
        def __init__(self, names):
            self.nodes = [SimpleNamespace(node=SimpleNamespace(name=name)) for name in names]

    visited = []

    @contextmanager
    def open_device(identity):
        visited.append(("open", identity))
        try:
            if identity == "probe:busy":
                raise CaptureError("CAMERA_ALREADY_OPEN", "camera in use")
            catalog = Catalog(("TriggerMode", "TriggerSource", "TriggerSoftware")) if identity == "probe:present" else Catalog(("Width",))
            wrapper = GenApi(catalog, genicam_api=object())
            yield SimpleNamespace(trigger_support=wrapper.trigger_support)
        finally:
            visited.append(("closed", identity))

    registry = CameraRegistry(custom={"probe": (lambda: [], open_device)})
    identities = ("probe:absent", "probe:present", "probe:busy", "uvc:known-interface")
    config = plan([{"mode": "single"}], cameras=[{"id": name} for name in identities])
    supports = probe_trigger_support(config, registry)
    assert visited == [(action, name) for name in identities[:3] for action in ("open", "closed")]
    assert supports["probe:absent"].software is False and supports["probe:absent"].hardware is False
    assert supports["probe:present"].software is None and supports["probe:present"].error is None
    assert supports["probe:busy"].software is None and supports["probe:busy"].error == "camera in use"
    assert supports["uvc:known-interface"].software is True and supports["uvc:known-interface"].hardware is False
    report = inspect_plan(config, tmp_path / "read-only", trigger_support=supports)
    issue, = (issue for issue in report["issues"] if issue["code"] == "SOFTWARE_TRIGGER_UNSUPPORTED")
    assert issue["cameras"] == ["probe:absent"]
    assert "TRIGGER_PROBE_FAILED" in codes(report, severity="warning")
    assert not (tmp_path / "read-only").exists()


def test_apply_host_suggestion_uses_formal_timing_instead_of_trial_time_limit(tmp_path):
    identity = "gentl:free-run"
    config = plan([{"mode": "single", "role": "reference"}, {"mode": "timed", "fps": 5, "frame_count": 30}],
                  cameras=[{"id": identity}])
    trial, out = prepare_trial(config, tmp_path / "formal", phase_number=2, seconds=2)
    observed = Image(np.zeros((40, 64), dtype=np.uint8), "Mono8", 8, "mono", 1, 1)
    report = inspect_plan(trial, out, preview_images={identity: observed},
                          trigger_support={identity: TriggerSupport(False, False, "device_nodes")})
    suggestion, = report["phase_suggestions"]
    assert suggestion["replacement"]["duration_s"] == 2
    applied = apply_phase_suggestion(config.phases[1], trial.phases[0], suggestion["replacement"])
    assert applied == {**config.phases[1].to_dict(), "software_trigger_fallback": True}
    assert not out.exists()


def test_preflight_counts_the_event_boundary_and_explains_a_buffer_or_exposure_conflict(tmp_path):
    phases = [{"mode": "event", "pre_frames": 3, "post_frames": 4, "fps": 100}]
    cameras = [{"id": "sim:0", "settings": {"width": 64, "height": 48, "pixel_format": "Mono12",
                                          "exposure_us": 20_000}}]
    frame_bytes = 64 * 48 * 2
    config = plan(phases, cameras, max_buffer_bytes=3 * frame_bytes)
    report = inspect_plan(config, tmp_path / "event")
    assert report["phases"][0]["frame_limit_per_camera"] == 8
    assert report["phases"][0]["one_batch_raw_bytes"] == 4 * frame_bytes
    assert codes(report, severity="block") == {"RAW_BUFFER_TOO_SMALL"}
    assert "EXPOSURE_EXCEEDS_INTERVAL" in codes(report, severity="warning")
    contents = format_readiness(report, phase_offset=2, camera_labels={"sim:0": "模拟散斑 0"})
    assert "预计需要 24.0 KiB · 缓冲上限 18.0 KiB" in contents
    assert "阶段 3 · 模拟散斑 0\n  曝光 20000 微秒 · 拍摄间隔 10000 微秒" in contents


def test_continuous_budget_uses_the_settings_for_each_camera(tmp_path):
    camera_settings = {"width": 64, "height": 48, "pixel_format": "Mono12"}
    cameras = [
        {"id": "sim:0", "settings": {**camera_settings, "fps": 12}},
        {"id": "sim:1", "settings": {**camera_settings, "fps": 7}},
    ]
    frame_bytes = 64 * 48 * 2
    config = plan([{"mode": "continuous", "duration_s": 2}], cameras, max_buffer_bytes=frame_bytes)
    report = inspect_plan(config, tmp_path / "continuous")
    phase, = report["phases"]
    assert phase["frame_limit_per_camera"] is None
    assert phase["frames_at_configured_rate"] == 38
    assert phase["configured_stream_raw_bytes_per_s"] == 19 * frame_bytes
    assert phase["one_batch_raw_bytes"] == frame_bytes
    assert report["raw_payload_bytes"] == 38 * frame_bytes
    assert "FRAME_LIMIT_UNAVAILABLE" not in codes(report) and report["ready"]


def test_indexed_browsing_and_linked_continuation_keep_original_run(tmp_path):
    config = plan([{"mode": "single", "role": "reference"}, {"mode": "timed", "fps": 10, "frame_count": 3}])
    original = CaptureSession(config, tmp_path / "original")
    parent_id = original.start()
    assert original.wait(10)["state"] == "completed"
    assert original.finished
    with ExperimentReader(original.output) as reader:
        newest, = reader.frames(camera_id="sim:0", newest_first=True, limit=1)
        previous, = reader.frames(camera_id="sim:0", before=newest["ordinal"], newest_first=True, limit=1)
        following, = reader.frames(camera_id="sim:0", after=previous["ordinal"], limit=1)
        assert following == newest
        assert previous["sequence"] + 1 == newest["sequence"]
        evidence = reader.verify()
    linked = continuation_plan(original.output)
    assert linked.metadata["continuation_of"] == parent_id
    sequel = CaptureSession(linked, tmp_path / "sequel")
    assert sequel.start() != parent_id
    assert sequel.wait(10)["state"] == "completed"
    with ExperimentReader(original.output) as reader:
        assert reader.verify() == evidence
    with ExperimentReader(sequel.output) as reader:
        assert reader.experiment["config"]["metadata"]["continuation_of"] == parent_id


def test_continuation_refuses_an_active_ledger(tmp_path):
    session = CaptureSession(plan([{"mode": "manual"}]), tmp_path / "armed")
    session.start()
    try:
        with pytest.raises(CaptureError) as error:
            continuation_plan(session.output)
        assert error.value.code == "CONTINUATION_ACTIVE"
    finally:
        session.request_stop()
        assert session.wait(10)["state"] == "stopped"


def test_status_reads_the_disk_observation_from_the_writer(tmp_path, monkeypatch):
    session = CaptureSession(plan([{"mode": "single"}]), tmp_path / "recorded")
    session.start()
    final = session.wait(10)
    assert final["state"] == "completed" and final["disk_free_bytes"] > 0
    assert final["buffered_raw_bytes"] == 0
    assert final["raw_buffer_high_water_bytes"] == 64 * 48 * 2

    def unexpected_probe(path):
        pytest.fail(f"status made a filesystem query at {path}")

    monkeypatch.setattr("straintrace_capture.storage.shutil.disk_usage", unexpected_probe)
    for _ in range(3):
        assert session.status()["disk_free_bytes"] == final["disk_free_bytes"]


def test_color_inspection_converts_only_sampled_pixels_and_inspection_region(monkeypatch):
    raw = np.random.default_rng(41).integers(0, 256, size=(1010, 1005, 3), dtype=np.uint8)
    image = Image(raw, "BGR8", 8, "bgr", 1, 1)
    roi = (20, 45, 64, 40)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    mono = Image(gray, "Mono8", 8, "mono", 1, 1)
    whole, local = image_quality(mono), image_quality(mono, roi)
    convert = cv2.cvtColor
    shapes = []

    def record_shape(samples, code):
        shapes.append(samples.shape)
        return convert(samples, code)

    monkeypatch.setattr(cv2, "cvtColor", record_shape)
    for actual, expected in ((image_quality(image), whole), (image_quality(image, roi), local)):
        # Color clipping uses the three channels, while luminance metrics match grayscale.
        assert actual.pop("bright_fraction") > expected.pop("bright_fraction")
        assert actual == expected
    assert shapes == [raw[::2, ::2].shape, (roi[3], roi[2], 3)]
