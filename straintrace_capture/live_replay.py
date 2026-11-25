"""Recalculate a saved live measurement over an ordered snapshot of committed images."""

from __future__ import annotations

import hashlib
import os
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import CaptureError
from .image_inputs import frame_part_selector, select_frame_image
from .live import LiveAnalysis
from .live_history import HistoryIndex
from .live_results import LiveResultLog, live_results_directory
from .load_reconcile import prepare_load_reconciliation
from .storage import ExperimentReader, _publish_path


def _source_tests(reader, index, records, *, part, after, snapshot, check):
    for archived in records:
        check()
        sequence = archived["sequence"]
        row = reader.frame_at_sequence(index.camera, sequence)
        if (row is None or row["role"] != "test" or sequence <= after or row["ordinal"] > snapshot
                or row["frame_id"] != archived["frame_id"]):
            raise CaptureError("LIVE_REPLAY_FRAME_CHANGED", f"原记录的实验图序号 {sequence} 无法在台账中核对。")
        asset = select_frame_image(row, part)
        if (asset["sha256"] != archived["image_sha256"] or frame_part_selector(asset) != archived["image_part"]):
            raise CaptureError("LIVE_REPLAY_FRAME_CHANGED", f"原记录的实验图序号 {sequence} 已发生变化。")
        yield archived


def _settings(first):
    setup = first.get("settings")
    if not isinstance(setup, dict) or not (setup.get("gauges") or setup.get("field_roi")):
        raise CaptureError("LIVE_REPLAY_SETUP", "找不到原记录的标距或全场测量区。")
    if (first["frame_id"] != first["reference_frame_id"] or first["image_sha256"] != first["reference_sha256"]
            or setup.get("image_part") != first["image_part"]):
        raise CaptureError("LIVE_REPLAY_REFERENCE_CHANGED", "原记录中的参考图标识或校验值不一致。")
    return setup


def replay_live_results(index: HistoryIndex, *, images_only=False, load_table=None, cancelled=None, progress=None):
    """Reuse the indexed settings and publish a new log only after processing the admitted frames."""
    if images_only and load_table is not None:
        raise CaptureError("LIVE_REPLAY_SETUP", "请选择只重算图像，或选择一份载荷表参与重算。")

    def check():
        if cancelled is not None and cancelled():
            raise CaptureError("LIVE_REPLAY_CANCELLED", "已取消重算。")

    check()
    reference_id, reference_hash, experiment_id, camera = index.identity
    after = 0 if index.borrowed_reference else index.reference_sequence
    digest = hashlib.sha256()
    with ExperimentReader(index.experiment) as reader, closing(
        index.verified_records(on_line=digest.update, check=check)
    ) as records:
        if reader.experiment["experiment_id"] != experiment_id:
            raise CaptureError("FRAME_REVIEW_CHANGED", "工作台中的实验目录已变化。")
        total, snapshot = reader.test_sequence_extent(camera, after=after)
        if total == 0:
            raise CaptureError("LIVE_REPLAY_NO_TEST", "参考图之后还没有已保存的实验图。")
        if progress is not None:
            progress(0, total)
        first = next(records)
        setup = _settings(first)
        part = setup.get("image_part")
        prior_tests = _source_tests(reader, index, records, part=part, after=after, snapshot=snapshot, check=check)
        alignment = None
        if not images_only and (setup.get("mechanics") is not None or load_table is not None):
            alignment = prepare_load_reconciliation(first, prior_tests, load_table)
        else:
            for _ in prior_tests:
                pass

    check()
    replay = {
        "schema": "straintrace.live-replay/v1", "source_name": index.path.name,
        "source_sha256": digest.hexdigest(), "source_bytes": index.pages[-1].end,
        "source_test_frames": index.frames - 1, "source_skipped_test_frames": index.skipped,
        "source_reference_binding_sha256": setup.get("reference_binding_sha256"),
        "snapshot_ordinal": snapshot, "test_frames": total,
        "load_policy": "verified_table" if alignment else "images_only" if images_only else "not_configured",
        "load_sha256": alignment.provenance["sha256"] if alignment else None,
    }
    directory = live_results_directory(index.experiment)
    directory.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=".replay-", dir=directory) as staging, LiveAnalysis(
        index.experiment, camera, reference_sequence=index.reference_sequence, image_part=part,
        gauges=setup.get("gauges", ()), field_roi=setup.get("field_roi"), field_step=setup.get("field_step_px", 12),
        field_subset=setup.get("field_subset_px", 21), latest=False, load_alignment=alignment,
        debayer_luminance=setup.get("debayer_luminance", False), refresh_load=False,
        reference_experiment=index.reference_root if index.borrowed_reference else None,
    ) as analyzer:
        log = LiveResultLog(index.experiment, directory=Path(staging), replay=replay)

        def append(row):
            check()
            if (mechanics_error := (row.get("mechanics") or {}).get("error")):
                raise CaptureError("LIVE_REPLAY_LOAD_ERROR", f"序号 {row['sequence']} 的载荷对齐失败。{mechanics_error}")
            log.append(row, analyzer, "ordered")
            if log.error is not None:
                raise log.error

        reference = analyzer.poll()
        if (reference is None or reference["frame_id"] != reference_id or reference["image_sha256"] != reference_hash
                or analyzer.reader.experiment["experiment_id"] != experiment_id):
            raise CaptureError("LIVE_REPLAY_REFERENCE_CHANGED", "台账中的参考图与原记录无法一致核对。")
        for saved, actual in zip(setup.get("gauges", ()), analyzer.groups, strict=True):
            if saved.get("actual_mode", actual["actual_mode"]) != actual["actual_mode"]:
                raise CaptureError("LIVE_REPLAY_SETUP", f"当前版本解析的标距方向与原记录不同。{actual['name']}")
        append(reference)
        cursor = after
        while batch := analyzer.reader.test_sequence_page(camera, after=cursor, snapshot=snapshot):
            for frame in batch:
                check()
                if frame["frame_id"] != f"{experiment_id}/{camera}/{frame['sequence']}":
                    raise CaptureError("LIVE_REPLAY_FRAME_CHANGED", f"实验图序号 {frame['sequence']} 的归属不一致。")
                append(analyzer.analyze_committed_test(frame))
                if progress is not None:
                    progress(log.test_frames, total)
            cursor = batch[-1]["sequence"]
        if log.test_frames != total:
            raise CaptureError("LIVE_REPLAY_INCOMPLETE", "重算期间的实验图数量与开始时不同。请重新读取台账。")
        with log.path.open("r+b") as stream:
            os.fsync(stream.fileno())
        with ExperimentReader(index.experiment) as reader:
            if reader.experiment["experiment_id"] != experiment_id:
                raise CaptureError("FRAME_REVIEW_CHANGED", "工作台中的实验目录已变化。")
        check()
        target = live_results_directory(index.experiment) / f"{log.path.stem}-replay.jsonl"
        _publish_path(log.path, target, write_through=True)
    return {**replay, "record": str(target), "experiment_id": experiment_id, "camera_id": camera,
            "reference_frame_id": reference_id, "reference_sha256": reference_hash}
