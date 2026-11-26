from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, ExperimentReader, recover_experiment
from straintrace_capture.capture_review import inspect_capture_review
from straintrace_capture.cameras.base import Image, receipt_times
from straintrace_capture.cameras.simulated import simulated_device
from straintrace_capture.continuation import inspect_continuation
from straintrace_capture.presentation import format_integrity, integrity_issues
from straintrace_capture.recovery import inspect_recovery
from straintrace_capture.storage import FRAME_WRITE_JOURNAL, ExperimentWriter


def _record(root, summary):
    plan = CaptureConfig.from_dict({
        "config_version": 1, "name": "采集结果", "cameras": [{"id": "sim:0"}, {"id": "sim:1"}],
        "phases": [{"mode": "single"}],
    })
    with closing(ExperimentWriter(root, plan)) as writer:
        for number in range(2):
            writer.register_camera(simulated_device(number), {})
        utc, monotonic = receipt_times()
        image = Image(np.zeros((6, 8), dtype=np.uint8), "Mono8", 8, "mono", utc, monotonic)
        multipart = replace(image, auxiliary_images=(image,))
        frame = writer.prepare_frame("sim:0", 1, 1, "test", multipart, None, None, None, 0)
        writer.commit_frames([frame])
        writer.set_state("completed", summary)
        return writer.experiment_id, root / frame.record["path"]


def test_review_counts_ledger_frames_and_parts_without_claiming_file_integrity(tmp_path):
    root = tmp_path / "多部件"
    identity, path = _record(root, {
        "cameras": {"sim:0": {"saved": 12, "native_counter_gaps": 2, "reported_driver_skips": 1}},
        "scheduled_misses": 3, "short_pre_windows": 1,
    })
    path.write_bytes(b"damaged fixture")
    result = inspect_capture_review(root, expected_id=identity)
    assert (result.frames, result.images) == (1, 2)
    assert result.cameras["sim:1"] == {"frames": 0, "images": 0}
    assert result.runtime_recorded and result.has_incidents
    assert result.camera_losses == {"sim:0": {"native_counter_gaps": 2, "reported_driver_skips": 1}}
    with ExperimentReader(root) as reader:
        report = reader.verify()
    assert not report["ok"]
    assert report["issues"] == [{"code": "FILE_SIZE_MISMATCH", "path": path.relative_to(root).as_posix()}]
    text = format_integrity(report)
    assert "定时采集请求漏发 3 次" in text and "事件前缓存窗口不足 1 次" in text
    assert "相机计数缺号 2 个 · 驱动报告跳过 1 帧" in text


def test_review_preserves_unknown_runtime_coverage_and_checks_source_identity(tmp_path):
    root = tmp_path / "旧台账"
    identity, _ = _record(root, None)
    result = inspect_capture_review(root, expected_id=identity)
    assert result.frames == 1
    assert not result.runtime_recorded and not result.has_incidents
    assert not result.write_records_need_review and not result.recovery_recorded
    with pytest.raises(CaptureError) as mismatch:
        inspect_capture_review(root, expected_id="another-experiment")
    assert mismatch.value.code == "REVIEW_SCOPE_CHANGED"


def test_review_and_continuation_detect_pending_records_without_reading_tiffs(tmp_path, monkeypatch):
    root = tmp_path / "pending"
    plan = CaptureConfig.from_dict({
        "config_version": 1, "name": "采集写入中断", "cameras": [{"id": "sim:0"}],
        "phases": [{"mode": "continuous", "frame_count": 2}],
    })
    with closing(ExperimentWriter(root, plan)) as writer:
        identity = writer.experiment_id
        writer.register_camera(simulated_device(0), {})
        image = Image(np.arange(48, dtype=np.uint8).reshape(6, 8), "Mono8", 8, "mono", *receipt_times())
        frames = [writer.prepare_frame("sim:0", number, 1, "test", image, None, None, None, 0)
                  for number in (1, 2)]
        writer.commit_frames(frames[:1])
        writer.set_state("failed", {"error": {"code": "WRITE_FAILED", "message": "fixture failure"}})
    evidence = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    def reject_image_hash(_path):
        raise AssertionError("the review must read write records without hashing TIFF files")

    with monkeypatch.context() as patches:
        patches.setattr("straintrace_capture.recovery.sha256_file", reject_image_hash)
        review = inspect_capture_review(root, expected_id=identity)
        assert review.frames == 1 and review.write_records_need_review and not review.recovery_recorded
        assert inspect_continuation(root, expected_experiment_id=identity)["write_records_need_review"]
    assert {path: path.read_bytes() for path in evidence} == evidence

    restored = recover_experiment(root, expected_experiment_id=identity)
    assert restored["ok"] and restored["committed_frames"] == 2
    review = inspect_capture_review(root, expected_id=identity)
    assert review.frames == 2 and not review.write_records_need_review and review.recovery_recorded
    assert not inspect_continuation(root, expected_experiment_id=identity)["write_records_need_review"]
    for frame in frames:
        path = root / frame.record["path"]
        assert path.read_bytes() == evidence[path]

    journal = root / FRAME_WRITE_JOURNAL
    journal.write_bytes(journal.read_bytes() + b"torn descriptor")
    assert inspect_capture_review(root, expected_id=identity).write_records_need_review
    assessed = recover_experiment(root, expected_experiment_id=identity)
    assert not assessed["issues"] and not assessed["ok"]
    assert [row["code"] for row in integrity_issues(assessed)] == ["FRAME_JOURNAL_ENTRY_INVALID"]

    journal.unlink()
    assert inspect_capture_review(root, expected_id=identity).write_records_need_review
    checked = inspect_recovery(root, expected_experiment_id=identity)
    assert checked["recovery"]["journal"] == "missing"
    assert [row["code"] for row in integrity_issues(checked)] == ["FRAME_JOURNAL_MISSING"]
