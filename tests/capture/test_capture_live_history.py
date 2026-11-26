import csv
import json
import shutil
from pathlib import Path
from threading import Event

import pytest

from straintrace_capture import CaptureError, ExperimentReader
from straintrace_capture.live import LiveAnalysis
from straintrace_capture.live_curve import read_curve_frame
from straintrace_capture.live_history import export_history_csv, read_history_index
from straintrace_capture.live_results import RECORD_SCHEMA, LiveResultLog, export_live_csv
from straintrace_capture.storage import sha256_file

from test_capture_continuation_analysis import _images, _record
from test_capture_mechanics_setup import capture_with_known_receipts


def _recorded_rows(experiment, *, field=False):
    with ExperimentReader(experiment) as reader:
        run = reader.experiment["experiment_id"]
        base = reader.frames()[0]
    source = {
        "schema": "straintrace.live-analysis/v1", "provisional": True, "record_schema": RECORD_SCHEMA,
        "experiment_id": run, "frame_experiment_id": run, "reference_experiment_id": run,
        "reference_frame_id": base["frame_id"], "reference_sha256": base["sha256"],
        "camera_id": "sim:0", "image_part": None, "image_path": base["path"],
        "image_sha256": base["sha256"], "timestamps": base["timestamps"], "follow_mode": "ordered",
        "elapsed_receive_basis": "reference_frame",
    }

    def row(index, seq, *, missing=False):
        gauges = [{"name": "G01", "engineering_strain": .001 * seq, "strain_valid": not missing,
                   "quality_status": "REJECTED" if missing else "ACCEPTED_HARD"}] if not field else []
        quality = {"strain_valid_fraction": .5 if missing else .95, "scientific_ok": not missing} if field else None
        return {**source, "frame_id": f"{run}/sim:0/{seq}", "sequence": seq, "elapsed_receive_s": seq / 8,
                "analyzed_test_frames": index, "skipped_committed_frames": max(0, seq - index - 1),
                "role": "test" if index else "reference", "gauges": gauges, "field_quality": quality}
    return row


def _write(path, rows, *, append=False, tail=""):
    with Path(path).open("a" if append else "w", encoding="utf-8", newline="") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.write(tail)


def _csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def test_saved_history_seeks_early_and_late_frames_and_jumps_over_clean_pages(tmp_path):
    experiment = capture_with_known_receipts(tmp_path / "run")
    record = tmp_path / "archive.jsonl"
    row = _recorded_rows(experiment)
    sequences = [number for number in range(1, 1151) if number != 744]
    _write(record, (row(i, seq, missing=seq in (813, 1102)) for i, seq in enumerate(sequences)),
           tail='{"incomplete":')
    original = sha256_file(record)
    history = read_history_index(experiment, record)
    assert history.frames == 1149 and history.skipped == 1 and history.incomplete_tail
    assert [page.count for page in history.pages] == [500, 500, 149]
    assert [page.issues for page in history.pages] == [(0,), (1,), (1,)]
    early = history.find_sequence(37)
    assert early[0] == 0 and early[2].frame.sequence == 37
    late = history.find_sequence(1110)
    assert late[0] == 2 and late[2].frame.sequence == 1110
    assert history.find_sequence(744) is None
    jumped = history.next_issue(0, "strain-time", after_sequence=37)
    assert jumped[0] == 1 and jumped[2].frame.sequence == 813 and jumped[2].strain_pct is None
    assert history.next_issue(1, "strain-time", after_sequence=813)[2].frame.sequence == 1102
    assert sha256_file(record) == original


def test_history_only_reads_its_verified_prefix_until_refresh(tmp_path):
    experiment = capture_with_known_receipts(tmp_path / "run")
    record = tmp_path / "growing.jsonl"
    row = _recorded_rows(experiment)
    _write(record, [row(0, 1), row(1, 2), row(2, 3)])
    history = read_history_index(experiment, record, page_size=2)
    _write(record, [row(3, 4)], append=True)
    assert [point.frame.sequence for point in history.read_page(1)] == [3]
    assert history.find_sequence(4) is None
    refreshed = read_history_index(experiment, record, page_size=2)
    assert refreshed.find_sequence(4)[2].frame.sequence == 4
    assert refreshed.frames == 4

    body = record.read_bytes()
    record.write_bytes(body.replace(b'"engineering_strain": 0.003', b'"engineering_strain": 0.004', 1))
    assert record.read_bytes() != body
    with pytest.raises(CaptureError) as changed:
        refreshed.read_page(1)
    assert changed.value.code == "LIVE_RECORD_CHANGED"


def test_history_checks_experiment_identity_and_shows_field_quality(tmp_path):
    experiment = capture_with_known_receipts(tmp_path / "run")
    other = capture_with_known_receipts(tmp_path / "other")
    record = tmp_path / "field.jsonl"
    row = _recorded_rows(experiment, field=True)
    _write(record, [row(0, 1), row(1, 2, missing=True), row(2, 3)])
    with pytest.raises(CaptureError) as mismatch:
        read_history_index(other, record)
    assert mismatch.value.code == "LIVE_HISTORY_OTHER_EXPERIMENT"
    history = read_history_index(experiment, record)
    assert history.modes == ("field-time",)
    _, _, issue = history.next_issue(0, "field-time")
    assert issue.frame.sequence == 2 and issue.field_valid_pct == 50 and issue.field_ok is False
    assert issue.values("field-time")[1] == 50


def test_moved_history_resolves_the_borrowed_reference_and_the_current_frame(tmp_path):
    ref, first, _ = _images()
    trials = tmp_path / "trials"
    parent, current = trials / "parent", trials / "current"
    parent_id = _record(parent, (ref,), "reference")
    current_id = _record(current, (first,), "test", parent=parent_id)
    journal = LiveResultLog(current)
    with LiveAnalysis(current, gauges=[{"name": "G01", "roi1": [40, 22, 25, 25], "roi2": [40, 85, 25, 25]}],
                      latest=False, reference_experiment=parent, reference_sequence=1) as follower:
        while result := follower.poll():
            journal.append(result, follower, "ordered")
    moved = tmp_path / "moved"
    shutil.copytree(trials, moved)
    index = read_history_index(moved / "current", moved / journal.path.relative_to(trials), page_size=1)
    old = index.read_page(0)[0]
    present = index.find_sequence(1)[2]
    assert index.borrowed_reference
    assert old.frame.experiment == moved / "parent" and old.frame.frame_id.startswith(parent_id)
    assert present.frame.experiment == moved / "current" and present.frame.frame_id.startswith(current_id)
    assert read_curve_frame(old.frame).pixels.shape == read_curve_frame(present.frame).pixels.shape == ref.shape
    output = tmp_path / "续采选定帧.csv"
    export_history_csv(index, output, sequence_range=(1, 1))
    rows = _csv_rows(output)
    assert [(row["帧类型"], row["原图实验标识"], row["原图序号"]) for row in rows] == [
        ("参考图", parent_id, "1"), ("实验图", current_id, "1"),
    ]


def test_export_history_stays_on_the_viewed_prefix_and_preserves_reference_in_ranges(tmp_path):
    experiment = capture_with_known_receipts(tmp_path / "run")
    record = tmp_path / "growing.jsonl"
    row = _recorded_rows(experiment)
    _write(record, [row(i, seq) for i, seq in enumerate((1, 2, 4, 5))])
    index = read_history_index(experiment, record, page_size=2)
    size = record.stat().st_size
    _write(record, [row(4, 7)], append=True, tail='{"unfinished":')
    digest = sha256_file(record)
    frozen, selected, refreshed = (tmp_path / name for name in ("整份.csv", "部分.csv", "新增.csv"))
    result = export_history_csv(index, frozen)
    assert result["source_frames"] == 4 and result["source_snapshot_bytes"] == size
    assert [row["原图序号"] for row in _csv_rows(frozen)] == ["1", "2", "4", "5"]
    result = export_history_csv(index, selected, sequence_range=(3, 8))
    assert result["test_frames"] == 2
    assert [row["原图序号"] for row in _csv_rows(selected)] == ["1", "4", "5"]
    result = export_live_csv(record, refreshed, sequence_range=(4, 7))
    assert result["test_frames"] == 3 and result["incomplete_tail"]
    assert [row["原图序号"] for row in _csv_rows(refreshed)] == ["1", "4", "5", "7"]
    assert sha256_file(record) == digest


def test_export_history_rejects_an_empty_range_and_preserves_previous_csv_on_failure(tmp_path):
    experiment = capture_with_known_receipts(tmp_path / "run")
    record = tmp_path / "chosen.jsonl"
    row = _recorded_rows(experiment)
    _write(record, [row(i, seq) for i, seq in enumerate((1, 2, *range(4, 41)))])
    index = read_history_index(experiment, record, page_size=2)
    output = tmp_path / "已有结果.csv"
    output.write_bytes(b"prior valid CSV")

    with pytest.raises(CaptureError) as empty:
        export_history_csv(index, output, sequence_range=(3, 3))
    assert empty.value.code == "LIVE_CSV_EMPTY_RANGE"
    stop = Event()
    with pytest.raises(CaptureError) as cancelled:
        export_history_csv(index, output, cancelled=stop.is_set, progress=lambda _count: stop.set())
    assert cancelled.value.code == "LIVE_CSV_CANCELLED"

    def change_unread_page(count):
        if count == 1:
            raw = record.read_bytes()
            updated = raw.replace(b'"engineering_strain": 0.04,', b'"engineering_strain": 0.05,')
            assert updated != raw
            record.write_bytes(updated)

    with pytest.raises(CaptureError) as changed:
        export_history_csv(index, output, progress=change_unread_page)
    assert changed.value.code == "LIVE_RECORD_CHANGED"
    assert output.read_bytes() == b"prior valid CSV"
    assert not list(tmp_path.glob(".live-results-*.part"))
