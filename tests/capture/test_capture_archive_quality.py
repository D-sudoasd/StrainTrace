from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import csv
import io
import json
from shutil import copyfile, copytree
import sqlite3
import threading
import zipfile

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, ExperimentReader
from straintrace_capture.archive_quality import (
    ArchiveQualityReport, archive_quality_options, open_archive_comparison, scan_archive_quality, scan_archive_quality_batch,
)
from straintrace_capture.archive_quality_export import QUALITY_REPORT_SCHEMA, export_archive_quality
from straintrace_capture.cameras.base import Image
from straintrace_capture.cameras.simulated import simulated_device
from straintrace_capture.preview_image import preview_original, preview_region_mismatch
from straintrace_capture.storage import DATABASE, ExperimentWriter, json_text, sha256_file
from straintrace_capture.trial_quality import open_quality_frame


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    root = tmp_path_factory.mktemp("archived-image-quality") / "正式采集"
    bounds = [16, 8, 32, 32]
    plan = CaptureConfig.from_dict({
        "config_version": 1, "name": "Archived main image quality",
        "cameras": [{"id": f"sim:{index}"} for index in range(2)],
        "phases": [{"mode": "single", "role": "reference"},
                   {"mode": "continuous", "frame_count": 105},
                   {"mode": "continuous", "frame_count": 3}],
        "quality_regions": {"sim:0": {"bounds": bounds, "image_size_px": [64, 48], "pixel_format": "Mono12",
                                       "significant_bits": 12, "sensor_window": {"offset_x": 0}}},
    })
    shape = (48, 64)
    with closing(ExperimentWriter(root, plan)) as writer:
        for index in range(2):
            writer.register_camera(simulated_device(index), {})
        for phase, count, index_start in ((1, 1, 0), (2, 105, 1), (3, 3, 106)):
            writer.event("phase_started", {"phase": phase, "camera_features": {
                "sim:0": {"offset_x": {"value": 4 if phase == 3 else 0}, "offset_y": {"value": 0}},
            }})
            for subindex in range(count):
                sequence = (index_start + subindex) * 2 + 1
                for camera in ("sim:0", "sim:1"):
                    pixels = np.full(shape, 2048, dtype=np.uint16)
                    if camera == "sim:0" and phase == 2:
                        pixels[:, :16] = 0
                        if subindex == 11:
                            pixels[8:40, 16:48] = 4095
                    if camera == "sim:0" and phase == 3 and subindex == 1:
                        pixels = pixels[:, :58]
                    stamp = sequence * 100_000_000
                    image = Image(pixels, "Mono12", 12, "mono", 1795620000000000000 + stamp, stamp)
                    role = "reference" if phase == 1 else "test"
                    writer.commit_frames([writer.prepare_frame(camera, sequence, phase, role, image, None, None, None, 0)])
        writer.set_state("completed", {})
    with ExperimentReader(root) as reader:
        return root, reader.experiment["experiment_id"]


def test_sample_spans_the_selected_camera_and_phase_and_full_scan_finds_brief_clipping(archive):
    root, identity = archive
    options = archive_quality_options(root, identity)
    assert options["counts"]["sim:0", "2"] == 105 and options["test_phases"] == ("2", "3")
    original = {path: sha256_file(path) for path in root.rglob("*.tiff")}
    with closing(scan_archive_quality(root, identity, "sim:0", 2, use_region=True)) as sample:
        assert (sample.available, sample.checked, sample.verified, sample.flagged) == (105, 96, 96, 0)
        rows = sample.page(0, 96)
        sequences = [row["sequence"] for row in rows]
        assert sequences[0] == 3 and sequences[-1] == 211 and 25 not in sequences
        assert all(row["roi"] == [16, 8, 32, 32] for row in rows)
    with closing(scan_archive_quality(root, identity, "sim:0", 2, use_region=True, all_frames=True)) as full:
        assert (full.checked, full.verified, full.flagged) == (105, 105, 1)
        row, = full.page(0, 40, alerts_only=True)
        assert row["sequence"] == 25 and row["flags"] == ["TRIAL_IMAGE_NEAR_SATURATION"]
        assert row["metrics"]["near_saturation_fraction"] == 1
        original_image = open_quality_frame(root, identity, "sim:0", row)
        assert np.all(original_image.pixels[8:40, 16:48] == 4095)
    with closing(scan_archive_quality(root, identity, "sim:0", 2, use_region=False)) as whole:
        assert whole.flagged == 96 and whole.page(0, 1)[0]["metrics"]["dark_fraction"] == .25
    assert {path: sha256_file(path) for path in original} == original


def test_archived_comparison_uses_the_selected_frame_and_its_phase_sensor_readback(archive):
    root, identity = archive
    with closing(scan_archive_quality(root, identity, "sim:0", 2, use_region=True, all_frames=True)) as report:
        clipped, = report.page(0, 5, alerts_only=True)
    source = open_archive_comparison(root, identity, "sim:0", 2, clipped)
    assert (source.experiment_id, source.phase, source.sequence, source.roi) == (identity, 2, 25, (16, 8, 32, 32))
    assert source.original.path == root / clipped["path"]
    assert source.sha256 == sha256_file(source.original.path)
    assert np.all(source.original.image.pixels[8:40, 16:48] == 4095)
    image, camera = source.original.image, source.original.device
    assert preview_region_mismatch(source.original, preview_original(camera, image, {"offset_x": 0, "offset_y": 0})) is None
    shifted = preview_original(camera, image, {"offset_x": 4, "offset_y": 0})
    assert preview_region_mismatch(source.original, shifted) == "sensor_window"

    with closing(scan_archive_quality(root, identity, "sim:0", 3, use_region=True)) as report:
        first, = report.page(0, 1)
    changed = open_archive_comparison(root, identity, "sim:0", 3, first)
    assert changed.roi is None and changed.original.sensor_window == {"offset_x": 4, "offset_y": 0}


def test_archived_comparison_rechecks_the_experiment_frame_and_tiff(archive, tmp_path):
    original, identity = archive
    root = tmp_path / "待复核"
    copytree(original, root)
    with closing(scan_archive_quality(root, identity, "sim:0", 1, use_region=True)) as report:
        selected, = report.page(0, 1)
    for experiment_id, camera, phase in (("wrong", "sim:0", 1), (identity, "sim:1", 1), (identity, "sim:0", 2)):
        with pytest.raises(CaptureError) as changed:
            open_archive_comparison(root, experiment_id, camera, phase, selected)
        assert changed.value.code == "FRAME_REVIEW_CHANGED"
    tiff = root / selected["path"]
    content = tiff.read_bytes()
    tiff.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(CaptureError) as corrupted:
        open_archive_comparison(root, identity, "sim:0", 1, selected)
    assert corrupted.value.code == "CAPTURE_IMAGE_CHANGED"


def test_changed_layout_and_sensor_geometry_fall_back_and_hash_failures_stay_visible(archive):
    root, identity = archive
    with closing(scan_archive_quality(root, identity, "sim:0", 3, use_region=True)) as report:
        assert report.checked == report.verified == report.fallbacks == report.flagged == 3
        first, second, third = report.page(0, 40, alerts_only=True)
        assert [row["fallback"] for row in (first, second, third)] == ["sensor_window", "image_layout", "sensor_window"]
        assert first["roi"] is None and first["flags"] == ["ARCHIVE_QUALITY_SENSOR_CHANGED"]
    with ExperimentReader(root) as reader:
        path = reader.image_path(reader.frame_at_sequence("sim:0", first["sequence"]))
    original = path.read_bytes()
    try:
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with closing(scan_archive_quality(root, identity, "sim:0", 3, use_region=False)) as changed:
            row, = changed.page(0, 40, alerts_only=True)
            assert changed.verified == 2 and row["flags"] == ["CAPTURE_IMAGE_CHANGED"] and row["metrics"] is None
    finally:
        path.write_bytes(original)


def test_cancellation_returns_the_checked_prefix_and_rejects_another_experiment(archive):
    root, identity = archive
    progress = []
    with closing(scan_archive_quality(
        root, identity, "sim:1", 2, use_region=False, all_frames=True,
        progress=lambda done, total: progress.append((done, total)),
        cancelled=lambda: bool(progress and progress[-1][0] >= 7),
    )) as report:
        assert report.stopped and report.available == report.selected == 105
        assert report.checked == report.verified == 7 and report.flagged == 0
        assert [row["sequence"] for row in report.page(4, 8)] == [11, 13, 15]
        assert progress[0] == (0, 105) and progress[-1] == (7, 105)
    with pytest.raises(CaptureError) as stopped:
        scan_archive_quality(root, "another experiment", "sim:1", 2, use_region=False)
    assert stopped.value.code == "FRAME_REVIEW_CHANGED"


def test_large_result_pages_keep_row_payloads_in_scratch_storage(tmp_path):
    with closing(ArchiveQualityReport(5000, 5000, scratch=tmp_path)) as report:
        for i in range(5000):
            report.append({"sequence": i * 2 + 1, "metrics": {"sample": "元" * 150}, "fallback": None,
                           "flags": ["TRIAL_IMAGE_NEAR_BLACK"] if i % 9 == 0 else []})
        assert report._rows._rolled
        assert report.checked == report.verified == 5000
        assert [row["sequence"] for row in report.page(4600, 3)] == [9201, 9203, 9205]
        assert [row["sequence"] for row in report.page(501, 3, alerts_only=True)] == [9019, 9037, 9055]
    assert not list(tmp_path.iterdir())


def test_report_can_append_after_paging_without_overwriting_rows():
    with closing(ArchiveQualityReport(3, 3)) as report:
        for sequence in (1, 2):
            report.append({"sequence": sequence, "metrics": {}, "fallback": None, "flags": []})
        assert report.page(0, 1)[0]["sequence"] == 1
        report.append({"sequence": 3, "metrics": {}, "fallback": None, "flags": []})
        assert [row["sequence"] for row in report.page(0, 3)] == [1, 2, 3]


def test_whole_experiment_reuses_one_result_store_and_preserves_per_scope_coverage(archive, tmp_path):
    root, identity = archive
    with closing(scan_archive_quality_batch(root, identity, scratch=tmp_path)) as sampled:
        assert len(sampled.options["scopes"]) == len(sampled.reports) == 6
        assert not sampled.errors and not sampled.stopped
        assert sum(report.available for report in sampled.reports.values()) == 218
        assert sum(report.checked for report in sampled.reports.values()) == 200
        assert sampled.reports["sim:0", "2"].first_alert is None
        assert sampled.reports["sim:0", "3"].flagged == 3
        assert all(report._rows is sampled._rows for report in sampled.reports.values())
    with closing(scan_archive_quality_batch(root, identity, all_frames=True, scratch=tmp_path)) as full:
        assert sum(report.checked for report in full.reports.values()) == 218
        clipped = full.reports["sim:0", "2"]
        assert clipped.first_alert == 25 and clipped.flagged == 1
        outside = full.reports["sim:1", "2"]
        assert outside.page(0, 1)[0]["roi"] is None and outside.flagged == 0
        assert clipped.page(0, 1, alerts_only=True)[0]["sequence"] == 25
        clipped.close()
        assert outside.page(104, 1)[0]["sequence"] == 211
    assert not list(tmp_path.iterdir())


def test_stopping_a_whole_experiment_check_retains_completed_groups_and_current_prefix(archive):
    root, identity = archive
    stop = threading.Event()

    def progress(number, total, camera, phase, done, selected):
        if (number, total, camera, phase, done, selected) == (3, 6, "sim:0", "2", 7, 105):
            stop.set()

    with closing(scan_archive_quality_batch(
        root, identity, all_frames=True, cancelled=stop.is_set, progress=progress,
    )) as batch:
        assert batch.stopped and not batch.errors
        assert tuple(batch.reports) == (("sim:0", "1"), ("sim:1", "1"), ("sim:0", "2"))
        assert batch.reports["sim:0", "2"].checked == 7 and batch.reports["sim:0", "2"].stopped
        assert batch.options["counts"]["sim:1", "2"] == 105


def test_whole_experiment_read_uses_one_snapshot_when_a_later_group_changes(archive, tmp_path):
    root, identity = archive
    copy = tmp_path / "测试台账"
    copytree(root, copy)
    with ExperimentReader(copy) as reader:
        row = reader.frame_at_sequence("sim:1", 217)
        content = reader.image_path(row)
    path = content.with_name("later.tiff")
    copyfile(content, path)
    relative = path.relative_to(copy).as_posix()
    changed = {**row, "frame_id": f"{identity}/sim:1/219", "sequence": 219, "path": relative}
    changed.pop("ordinal")
    inserted = False

    def progress(number, _total, _camera, _phase, done, _selected):
        nonlocal inserted
        if number == 1 and done == 1:
            with closing(sqlite3.connect(copy / DATABASE)) as writer, writer:
                writer.execute("INSERT INTO frames(camera_id,sequence,path,record) VALUES (?,?,?,?)",
                               ("sim:1", 219, relative, json_text(changed)))
            inserted = True

    with closing(scan_archive_quality_batch(copy, identity, all_frames=True, progress=progress)) as batch:
        assert inserted and not batch.errors
        assert batch.options["counts"]["sim:1", "3"] == 3
        assert batch.reports["sim:1", "3"].checked == 3
    with closing(scan_archive_quality(copy, identity, "sim:1", 3, use_region=False)) as fresh:
        assert fresh.checked == fresh.verified == 4


def test_whole_experiment_distinguishes_empty_groups_from_a_failed_group(archive, tmp_path):
    root, identity = archive
    copy = tmp_path / "试验级错误"
    copytree(root, copy)
    with closing(sqlite3.connect(copy / DATABASE)) as writer, writer:
        writer.execute("DELETE FROM frames WHERE camera_id='sim:1' AND json_extract(record,'$.phase')=3")
        writer.execute("UPDATE frames SET record=json_set(record,'$.frame_id','inconsistent') "
                       "WHERE camera_id='sim:0' AND sequence=213")
    with closing(scan_archive_quality_batch(copy, identity, all_frames=True)) as batch:
        assert ("sim:1", "3") in batch.options["scopes"] and ("sim:1", "3") not in batch.options["counts"]
        assert ("sim:1", "3") not in batch.errors and ("sim:1", "3") not in batch.reports
        assert batch.errors["sim:0", "3"].code == "INVALID_LEDGER"
        assert batch.reports["sim:1", "2"].verified == 105


def _contents(path):
    with zipfile.ZipFile(path) as package:
        assert package.testzip() is None
        manifest = json.loads(package.read("report.json"))
        coverage = list(csv.DictReader(io.StringIO(package.read("coverage.csv").decode("utf-8-sig"), newline="")))
        frames = list(csv.DictReader(io.StringIO(package.read("frames.csv").decode("utf-8-sig"), newline="")))
    return manifest, coverage, frames


def test_export_keeps_the_checked_frame_identity_bounds_and_scope_evidence(archive, tmp_path):
    root, identity = archive
    output = tmp_path / "整场报告.zip"
    with closing(scan_archive_quality_batch(root, identity, all_frames=True)) as batch:
        receipt = export_archive_quality(batch, output)
    manifest, coverage, frames = _contents(output)
    assert receipt["verified"] == len(frames) == 218 and receipt["scopes"] == len(coverage) == 6
    assert receipt["alerts"] == 4
    assert manifest["experiment_id"] == identity and manifest["schema"] == QUALITY_REPORT_SCHEMA
    assert manifest["mode"] == "all_frames" and not manifest["stopped"]
    clipped, = [row for row in frames if row["相机标识"] == "sim:0" and row["帧号"] == "25"]
    assert clipped["近饱和比例"] == "1.0" and clipped["检查范围"] == "计划检查框"
    assert clipped["实际框横坐标"] == "16" and clipped["实际框宽"] == "32"
    assert clipped["像素采样步长"] == "1" and clipped["主机收图 UTC"].endswith("Z")
    assert clipped["台账 SHA-256"] == sha256_file(root / clipped["原图相对路径"])
    shifted = next(row for row in frames if row["相机标识"] == "sim:0" and row["阶段"] == "3")
    assert shifted["检查范围"] == "全图" and not shifted["实际框宽"]
    assert shifted["改查全图原因"] == "传感器范围变化"
    group = next(row for row in coverage if row["相机标识"] == "sim:0" and row["阶段"] == "2")
    assert group["首条提醒帧"] == "25" and group["预定检查帧数"] == "105"


def test_stopped_report_and_cancelled_write_preserve_distinct_coverage(archive, tmp_path):
    root, identity = archive
    stop_scan = threading.Event()

    def progress(_number, _total, camera, phase, done, _selected):
        if (camera, phase, done) == ("sim:0", "2", 5):
            stop_scan.set()

    output = tmp_path / "部分报告.zip"
    cancelled_output = tmp_path / "取消.zip"
    with closing(scan_archive_quality_batch(
        root, identity, all_frames=True, cancelled=stop_scan.is_set, progress=progress,
    )) as batch:
        export_archive_quality(batch, output)
        with pytest.raises(CaptureError) as duplicate:
            export_archive_quality(batch, output)
        assert duplicate.value.code == "QUALITY_REPORT_EXISTS"
        with pytest.raises(CaptureError) as inside:
            export_archive_quality(batch, root / "不写入原目录.zip")
        assert inside.value.code == "QUALITY_REPORT_INSIDE_ARCHIVE"
        stop_export = threading.Event()
        with pytest.raises(CaptureError) as cancelled:
            export_archive_quality(batch, cancelled_output, cancelled=stop_export.is_set,
                                   progress=lambda done, _total: stop_export.set() if done else None)
        assert cancelled.value.code == "QUALITY_REPORT_CANCELLED"
    manifest, coverage, frames = _contents(output)
    assert manifest["stopped"] and len(frames) == 7 and manifest["totals"]["unchecked_groups"] == 3
    assert any(row["状态"] == "已停止" and row["已查帧数"] == "5" for row in coverage)
    assert any(row["状态"] == "未检查" and row["已查帧数"] == "0" and not row["已核对帧数"] for row in coverage)
    assert list(tmp_path.iterdir()) == [output]


def test_report_retention_allows_details_and_owner_close_while_writing(archive, tmp_path):
    root, identity = archive
    batch = scan_archive_quality_batch(root, identity)
    started, resume = threading.Event(), threading.Event()
    batch.retain()

    def hold(done, _total):
        if done == 0:
            started.set()
            assert resume.wait(5)

    def write():
        try:
            return export_archive_quality(batch, tmp_path / "report.zip", progress=hold)
        finally:
            batch.release()

    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(write)
        try:
            assert started.wait(5)
            assert batch.reports["sim:0", "2"].page(0, 1)[0]["sequence"] == 3
            batch.close()
            assert not batch._rows.closed
        finally:
            resume.set()
        assert task.result(timeout=8)["checked"] == 200
    assert batch._rows.closed


def test_cli_reports_raw_integrity_failure_but_saves_evidence(archive, tmp_path, capsys):
    from straintrace_capture.cli import main

    root, identity = archive
    copy = tmp_path / "损坏测试"
    copytree(root, copy)
    with ExperimentReader(copy) as reader:
        victim = reader.image_path(reader.frame_at_sequence("sim:0", 25))
    raw = victim.read_bytes()
    victim.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    output = tmp_path / "校验报告.zip"
    args = ("export-quality-report", str(copy), "--all-frames", "--output", str(output))
    assert main([*args, "--expect-experiment-id", "another-experiment"]) == 3
    assert not output.exists()
    capsys.readouterr()
    assert main([*args, "--expect-experiment-id", identity]) == 4
    result = json.loads(capsys.readouterr().out)
    assert result["checked"] == 218 and result["verified"] == 217
    _, _, frames = _contents(output)
    row, = [row for row in frames if row["核对状态"] == "待复核"]
    assert row["帧号"] == "25" and row["提醒代码"] == "CAPTURE_IMAGE_CHANGED"
    assert not row["近饱和比例"] and not row["实际框宽"] and not row["检查范围"]
