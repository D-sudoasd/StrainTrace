from contextlib import closing
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, ExperimentReader, inspect_plan, prepare_trial, trial_report
from straintrace_capture.cameras.base import Feature, Image
from straintrace_capture.cameras.settings_probe import SettingsCheck
from straintrace_capture.cameras.simulated import simulated_device
from straintrace_capture.gui import CaptureWorkbench
from straintrace_capture.presentation import format_readiness, format_trial
from straintrace_capture.storage import ExperimentWriter, sha256_file
from straintrace_capture.trial import refresh_trial_image_regions
from straintrace_capture.trial_quality import open_trial_image, read_trial_quality_region, rescan_trial_images
from straintrace_capture.trial_reference import archive_trial_reference, referenced_trial_report


def trial_plan(count, cameras, settings=None):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "trial image checks",
        "cameras": [{"id": simulated_device(index).id, "settings": settings or {}} for index in range(cameras)],
        "phases": [{"mode": "continuous", "frame_count": count}],
    })


def make_trial(parent, count, *, clip=(), dark=(), resized=(), cameras=1, config=None, phase_number=None, offset=None):
    identities = [simulated_device(index) for index in range(cameras)]
    config = config or trial_plan(count, cameras)
    plan, root = prepare_trial(config, parent / "formal", seconds=5, phase_number=phase_number)
    generator = np.random.default_rng(476)
    base = generator.integers(540, 3500, size=(48, 64), dtype=np.uint16)
    with closing(ExperimentWriter(root, plan)) as writer:
        for device in identities:
            writer.register_camera(device, {})
        if offset is not None:
            writer.event("phase_started", {"phase": 1, "camera_features": {"sim:0": {"offset_x": {"value": offset}}}})
        for sequence in range(1, count + 1):
            for index, device in enumerate(identities):
                if index == 0:
                    pixels = base.copy()
                    if sequence in clip:
                        pixels[:, :20] = 4095
                    if sequence in dark:
                        pixels[:, :20] = 0
                    if sequence in resized:
                        pixels = pixels[:, :58]
                    fmt, bits, space = "Mono12", 12, "mono"
                else:
                    pixels = generator.integers(35, 220, size=(48, 64, 4), dtype=np.uint8)
                    pixels[..., 3] = 255
                    fmt, bits, space = "RGBa8", 8, "rgba"
                tick = sequence * 100_000_000 + index * 30_000
                img = Image(pixels, fmt, bits, space, 1795600000000000000 + tick, tick)
                frame = writer.prepare_frame(device.id, sequence, 1, "test", img, None, None, None, 0)
                writer.commit_frames([frame])
        writer.set_state("completed", {})
    return root


def test_trial_reports_exact_image_locations_and_rechecks_the_same_region(tmp_path):
    root = make_trial(tmp_path, 5, clip=(2,), dark=(4,), cameras=2)
    original = {path: sha256_file(path) for path in root.rglob("*.tiff")}
    report = trial_report(root)
    assert report["observed"] and len(original) == 10
    left, right = (report["cameras"][key]["image_check"] for key in ("sim:0", "sim:1"))
    assert (left["mode"], left["checked"], left["flagged"]) == ("all", 5, 2)
    assert right["checked"] == 5 and right["flagged"] == 0
    assert [row["sequence"] for row in left["rows"] if row["flags"]] == [2, 4]
    first_issue = left["rows"][1]
    assert first_issue["flags"] == ["TRIAL_IMAGE_NEAR_SATURATION"]
    assert first_issue["metrics"]["near_saturation_fraction"] == 20 / 64
    assert left["rows"][3]["flags"] == ["TRIAL_IMAGE_NEAR_BLACK"]
    assert "全查 5/5 张 · 已核验 5 张 · 提醒 2 张" in format_trial(report)

    raw = open_trial_image(root, report["experiment_id"], "sim:0", first_issue)
    assert raw.pixels.dtype == np.uint16 and np.all(raw.pixels[:, :20] == 4095)
    local = rescan_trial_images(root, report["experiment_id"], "sim:0", roi=(32, 8, 24, 32),
                               region_layout=first_issue["layout"])
    assert local["roi"] == [32, 8, 24, 32] and local["checked"] == 5 and local["flagged"] == 0
    assert all(row["metrics"]["texture_measured"] for row in local["rows"])
    assert {path: sha256_file(path) for path in original} == original
    assert "image_check" not in trial_report(root, inspect_images=False)["cameras"]["sim:0"]


def test_sampling_states_coverage_and_an_explicit_scan_can_find_an_unsampled_frame(tmp_path):
    root = make_trial(tmp_path, 105, clip=(12,))
    report = trial_report(root)
    quality = report["cameras"]["sim:0"]["image_check"]
    numbers = [row["sequence"] for row in quality["rows"]]
    assert quality["mode"] == "sampled" and len(set(numbers)) == quality["checked"] == 96
    assert quality["available"] == 105 and numbers[0] == 1 and numbers[-1] == 105
    assert 12 not in numbers and quality["flagged"] == 0
    steps = []
    all_frames = rescan_trial_images(root, report["experiment_id"], "sim:0", all_frames=True,
                                    progress=lambda count, total: steps.append((count, total)))
    assert all_frames["mode"] == "all" and all_frames["checked"] == 105
    assert quality["series_sha256"] == all_frames["series_sha256"]
    assert [row["sequence"] for row in all_frames["rows"] if row["flags"]] == [12]
    assert steps[0] == (1, 105) and steps[-1] == (105, 105)
    with pytest.raises(CaptureError) as caught:
        rescan_trial_images(root, report["experiment_id"], "sim:0", cancelled=lambda: True)
    assert caught.value.code == "TRIAL_IMAGE_CHECK_CANCELLED"


def test_preflight_preserves_coverage_and_warns_only_for_the_matching_trial(tmp_path):
    root = make_trial(tmp_path, 105, clip=(12,), cameras=2)
    config = trial_plan(105, 2)
    formal = tmp_path / "formal"
    trial = trial_report(root)
    first = inspect_plan(config, formal, trial_reports=[trial])
    scope = first["trial_assessments"][0]["image_checks"]["sim:0"]
    assert (scope["mode"], scope["flagged"], scope["checked"]) == ("sampled", 0, 96)
    assert not any(row["code"] == "TRIAL_IMAGES_NEED_REVIEW" for row in first["issues"])

    trial["cameras"]["sim:0"]["image_check"] = rescan_trial_images(root, trial["experiment_id"], "sim:0", all_frames=True)
    checked = inspect_plan(config, formal, trial_reports=[trial])
    warning, = [row for row in checked["issues"] if row["code"] == "TRIAL_IMAGES_NEED_REVIEW"]
    assert checked["ready"] and warning["phase"] == 1 and warning["flagged_images"] == 1
    assert "阶段 1 · 1 张原图有提醒" in format_readiness(checked).splitlines()[:3]
    assert "全查 105/105 张 · 已核验 105 张 · 提醒 1 张" in format_readiness(checked)
    assert "均匀抽查 96/105 张 · 已核验 96 张 · 提醒 0 张" in format_readiness(checked)
    assert "rows" not in checked["trial_assessments"][0]["image_checks"]["sim:0"]

    updated = inspect_plan(trial_plan(105, 2, {"exposure_us": 1200}), formal, trial_reports=[trial])
    assert updated["trial_assessments"][0]["changes"] == ["camera_settings"]
    assert "主图像检查 · 历史试采" in format_readiness(updated)
    assert "曝光时间  由设备决定 → 1200 µs" in format_readiness(updated)
    assert not any(row["code"] == "TRIAL_IMAGES_NEED_REVIEW" for row in updated["issues"])
    assert not formal.exists()


def test_image_inspection_refuses_changed_bytes_or_an_incompatible_region(tmp_path):
    root = make_trial(tmp_path, 4, resized=(4,))
    report = trial_report(root)
    identity = report["experiment_id"]
    first, _, _, last = report["cameras"]["sim:0"]["image_check"]["rows"]
    region = rescan_trial_images(root, identity, "sim:0", roi=(4, 4, 16, 16), region_layout=first["layout"])
    assert region["checked"] == 3 and region["rows"][-1]["metrics"] is None
    assert region["rows"][-1]["flags"] == ["TRIAL_IMAGE_REGION_CHANGED"]
    assert open_trial_image(root, identity, "sim:0", last).pixels.shape == (48, 58)

    with ExperimentReader(root, immutable=True) as reader:
        path = reader.image_path(reader.frame_at_sequence("sim:0", 1))
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(CaptureError) as changed:
        open_trial_image(root, identity, "sim:0", first)
    assert changed.value.code == "CAPTURE_IMAGE_CHANGED"
    with pytest.raises(CaptureError) as changed_region:
        read_trial_quality_region(root, identity, "sim:0", first, (4, 4, 16, 16), {})
    assert changed_region.value.code == "CAPTURE_IMAGE_CHANGED"
    separate = trial_report(root)
    assert not separate["observed"] and not separate["verification"]["ok"]
    image_check = separate["cameras"]["sim:0"]["image_check"]
    assert image_check["checked"] == 3 and image_check["rows"][0]["flags"] == ["CAPTURE_IMAGE_CHANGED"]
    with pytest.raises(CaptureError) as moved:
        open_trial_image(root, "another experiment", "sim:0", first)
    assert moved.value.code == "FRAME_REVIEW_CHANGED"


def test_region_from_verified_trial_frame_uses_actual_geometry_and_readbacks(tmp_path):
    root = make_trial(tmp_path, 2, offset=4)
    report = trial_report(root)
    source = report["cameras"]["sim:0"]["image_check"]["rows"][0]
    bounds = (6, 5, 41, 32)

    region = read_trial_quality_region(root, report["experiment_id"], "sim:0", source, bounds,
                                       {"width": 64, "pixel_format": "Mono12", "offset_x": 4, "exposure_us": 900})
    assert region.bounds == bounds
    assert region.image_size_px == (64, 48)
    assert region.pixel_format == "Mono12" and region.significant_bits == 12
    assert region.sensor_window == {"offset_x": 4}

    for settings, code in (
        ({"width": 56}, "TRIAL_REGION_IMAGE_LAYOUT"),
        ({"pixel_format": "Mono8"}, "TRIAL_REGION_IMAGE_LAYOUT"),
        ({"offset_x": 8}, "TRIAL_REGION_SENSOR_WINDOW"),
        ({"binning_x": 2}, "TRIAL_REGION_SENSOR_WINDOW"),
    ):
        with pytest.raises(CaptureError) as changed:
            read_trial_quality_region(root, report["experiment_id"], "sim:0", source, bounds, settings)
        assert changed.value.code == code

    with pytest.raises(CaptureError) as outside:
        read_trial_quality_region(root, report["experiment_id"], "sim:0", source, (45, 5, 41, 32), {})
    assert outside.value.code == "TRIAL_REGION_BOUNDS"


def test_region_from_trial_without_readback_requires_same_requested_sensor_settings(tmp_path):
    config = trial_plan(2, 1, {"offset_x": 4, "binning_x": 2})
    root = make_trial(tmp_path, 2, config=config)
    report = trial_report(root)
    source = report["cameras"]["sim:0"]["image_check"]["rows"][0]
    args = (root, report["experiment_id"], "sim:0", source, (6, 5, 41, 32))
    assert read_trial_quality_region(*args, config.cameras[0].settings).sensor_window == {}
    with pytest.raises(CaptureError) as removed:
        read_trial_quality_region(*args, {"binning_x": 2})
    assert removed.value.code == "TRIAL_REGION_SENSOR_WINDOW"


def plan_with_region(count, bounds, *, sensor_window=None):
    plan = trial_plan(count, 1).to_dict()
    region = {"bounds": bounds, "image_size_px": [64, 48], "pixel_format": "Mono12", "significant_bits": 12,
              "sensor_window": sensor_window or {}}
    return CaptureConfig.from_dict({**plan, "quality_regions": {"sim:0": region}})


def test_current_inspection_box_drives_preflight_and_can_be_rechecked_from_original_tiffs(tmp_path):
    formal = tmp_path / "formal"
    initial = plan_with_region(6, [30, 8, 18, 25])
    root = make_trial(tmp_path, 6, clip=(2,), dark=(4,), config=initial)
    report = trial_report(root)
    original = {path: sha256_file(path) for path in root.rglob("*.tiff")}
    whole = report["cameras"]["sim:0"]["image_check"]
    regional = whole["region_check"]["check"]
    assert whole["flagged"] == 2 and regional["flagged"] == 0
    assert [row["sha256"] for row in whole["rows"]] == [row["sha256"] for row in regional["rows"]]
    clear = inspect_plan(initial, formal, trial_reports=[report])
    assert clear["trial_assessments"][0]["region_checks"]["sim:0"]["used_for_review"]
    assert not any(row["code"] == "TRIAL_IMAGES_NEED_REVIEW" for row in clear["issues"])
    assert "用于采前亮度复核" in format_readiness(clear)

    revised = plan_with_region(6, [5, 8, 12, 25])
    stale = inspect_plan(revised, formal, trial_reports=[report])
    assert stale["trial_assessments"][0]["region_checks"]["sim:0"]["fallback"] == "plan_changed"
    reread = refresh_trial_image_regions(revised, formal, (report,))
    current = inspect_plan(revised, formal, trial_reports=reread)
    warn, = [item for item in current["issues"] if item["code"] == "TRIAL_IMAGES_NEED_REVIEW"]
    assert warn["flagged_images"] == 2
    check = reread[0]["cameras"]["sim:0"]["image_check"]["region_check"]["check"]
    assert check["roi"] == [5, 8, 12, 25]
    assert [row["sequence"] for row in check["rows"] if row["flags"]] == [2, 4]
    snapshot = archive_trial_reference(current, reread, image_warnings_confirmed=True)["stages"][0]
    assert snapshot["region_checks"]["sim:0"]["used_for_review"] and snapshot["image_warnings_confirmed"]
    reopened = referenced_trial_report(formal, snapshot)
    assert reopened["cameras"]["sim:0"]["image_check"]["region_check"]["check"] == check
    assert not formal.exists() and {path: sha256_file(path) for path in original} == original


def test_region_review_falls_back_when_camera_moves_or_an_unsampled_tiff_changes(tmp_path):
    formal = tmp_path / "formal"
    config = plan_with_region(105, [29, 7, 20, 22], sensor_window={"offset_x": 0})
    root = make_trial(tmp_path, 105, dark=(3,), config=config, offset=0)
    report = trial_report(root)
    original = report["cameras"]["sim:0"]["image_check"]
    assert 12 not in {row["sequence"] for row in original["rows"]} and original["region_check"]["fallback"] is None
    expanded = trial_report(root, full_image_cameras={"sim:0"})
    reread_geometry = deepcopy(report)
    reread_geometry["cameras"]["sim:0"]["image_check"]["region_check"].update(fallback="sensor_window", check=None)
    workbench = SimpleNamespace(_trial_results={1: expanded}, _invalidate_preflight=lambda: None,
                                plan_summary=SimpleNamespace(set=lambda _text: None))
    CaptureWorkbench._remember_trial(workbench, reread_geometry)
    latest = workbench._trial_results[1]["cameras"]["sim:0"]["image_check"]
    assert latest["mode"] == "sampled" and latest["region_check"]["fallback"] == "sensor_window"

    preview = open_trial_image(root, report["experiment_id"], "sim:0", original["rows"][0])
    moved = inspect_plan(config, formal, trial_reports=[report], preview_images={"sim:0": preview},
                         camera_settings={"sim:0": SettingsCheck("preview_readback", readback={
                             "offset_x": Feature("offset_x", 6, 0, 256, 2),
                         })})
    assert moved["trial_assessments"][0]["region_checks"]["sim:0"]["fallback"] == "sensor_window"
    assert any(row["code"] == "TRIAL_IMAGES_NEED_REVIEW" for row in moved["issues"])

    with ExperimentReader(root) as reader:
        path = reader.image_path(reader.frame_at_sequence("sim:0", 12))
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    reread = refresh_trial_image_regions(config, formal, (report,))
    actual = inspect_plan(config, formal, trial_reports=reread)
    assert not reread[0]["observed"] and actual["trial_assessments"][0]["region_checks"]["sim:0"]["fallback"] == "integrity"
    assert {"TRIAL_NEEDS_REVIEW", "TRIAL_IMAGES_NEED_REVIEW"} <= {row["code"] for row in actual["issues"]}
