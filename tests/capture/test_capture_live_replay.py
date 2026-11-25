from hashlib import sha256
from threading import Event

import pytest

from straintrace_capture import CaptureError, ExperimentReader
from straintrace_capture.live import LiveAnalysis
from straintrace_capture.live_history import read_history_index
from straintrace_capture.live_replay import replay_live_results
from straintrace_capture.live_results import LiveResultLog, export_live_csv
from straintrace_capture.mechanics_setup import inspect_load_csv, table_alignment
from straintrace_capture.storage import DATABASE, sha256_file

from test_capture_continuation_analysis import _commit, _images, _live_writer, _record
from test_capture_mechanics_setup import _options, capture_with_known_receipts


_GAUGES = [{"name": "G01", "roi1": (45, 25, 25, 25), "roi2": (45, 83, 25, 25), "strain_mode": "y"}]


def _latest(root, **settings):
    log = LiveResultLog(root)
    with LiveAnalysis(root, **settings) as live:
        while row := live.poll():
            log.append(row, live, "latest")
    assert log.error is None
    return log.path


def _assert_error(code, operation):
    with pytest.raises(CaptureError) as failed:
        operation()
    assert failed.value.code == code


def test_replay_restores_skipped_images_and_pins_the_proven_load_table(tmp_path):
    root = capture_with_known_receipts(tmp_path / "run")
    source = tmp_path / "原载荷.csv"
    original = "Time (s),Load (kN),Extension (mm)\n0,0,0\n0.5,0.3,0.2\n"
    source.write_text(original, encoding="utf-8")
    info = inspect_load_csv(source)
    load = table_alignment(info, root, "sim:0", **_options(info, sync_frame="2", sync_time="0.5"))
    path = _latest(root, gauges=_GAUGES, load_alignment=load)
    prefix = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"partiallyWritten":')
    index = read_history_index(root, path, page_size=1)
    assert index.frames == 2 and index.skipped == 2 and index.incomplete_tail
    preserved = {file: sha256_file(file) for file in (path, source, root / DATABASE, *root.rglob("*.tiff"))}
    updated = tmp_path / "同轮载荷.csv"
    expanded = original + "1.5,0.5,0.4\n5.0,0.7,0.8\n"
    updated.write_text(expanded, encoding="utf-8")
    expected_load_hash = sha256_file(updated)

    def change_after_first(done, _total):
        if done == 1:
            updated.write_text(expanded.replace("0.3", "0.9"), encoding="utf-8")

    result = replay_live_results(index, load_table=updated, progress=change_after_first)
    assert result["source_sha256"] == sha256(prefix).hexdigest()
    assert result["source_bytes"] == len(prefix)
    assert result["load_sha256"] == expected_load_hash
    assert (result["source_test_frames"], result["source_skipped_test_frames"], result["test_frames"]) == (1, 2, 3)
    history = read_history_index(root, result["record"])
    assert history.replay and history.skipped == 0 and history.frames == 4
    rows = list(history.verified_records())
    assert [row["sequence"] for row in rows] == [1, 2, 3, 4]
    assert [row["mechanics"]["force_N"] for row in rows] == [0, 300, 400, 700]
    assert [row["mechanics"]["displacement_mm"] for row in rows] == [0, .2, pytest.approx(.3), .8]
    assert all(row["mechanics"]["mechanics_source"]["sha256"] == expected_load_hash for row in rows)
    assert all(row["gauges"][0]["engineering_strain"] == pytest.approx(0, abs=1e-3) for row in rows)
    assert rows[0]["settings"]["gauges"] == next(index.verified_records())["settings"]["gauges"]
    assert rows[0]["settings"]["replay"]["source_sha256"] == result["source_sha256"]
    exported = tmp_path / "重算曲线.csv"
    csv_result = export_live_csv(result["record"], exported)
    assert csv_result["test_frames"] == 3
    assert all(line.startswith("按序重算,") for line in exported.read_text(encoding="utf-8-sig").splitlines()[1:])
    _assert_error("LOAD_ARCHIVE_PREFIX_CHANGED", lambda: replay_live_results(index, load_table=updated))
    image_history = read_history_index(root, replay_live_results(index, images_only=True)["record"])
    assert image_history.load_source is None
    assert all("mechanics" not in row for row in image_history.verified_records())
    assert all(sha256_file(file) == fingerprint for file, fingerprint in preserved.items())


def test_linked_replay_orders_committed_backfills_and_excludes_new_admissions(tmp_path):
    parent, child = tmp_path / "parent", tmp_path / "child"
    pixels, moved, _ = _images()
    parent_id = _record(parent, (pixels,), "reference")
    writer, _ = _live_writer(child, parent_id)
    try:
        _commit(writer, moved, 1, 1, 100_000_000)
        _commit(writer, moved, 4, 1, 400_000_000)
        _commit(writer, moved, 3, 1, 300_000_000)
        path = _latest(child, reference_experiment=parent, reference_sequence=1,
                       field_roi=(36, 30, 74, 65), field_step=16, field_subset=15)
        index = read_history_index(child, path)

        def append_when_pinned(done, total):
            if done == 0:
                assert total == 3
                _commit(writer, moved, 2, 1, 200_000_000)
                _commit(writer, moved, 5, 1, 500_000_000)

        receipt = replay_live_results(index, progress=append_when_pinned)
        recorded = read_history_index(child, receipt["record"])
        rows = tuple(recorded.verified_records())
        assert recorded.borrowed_reference and rows[0]["frame_experiment_id"] == parent_id
        assert [row["sequence"] for row in rows[1:]] == [1, 3, 4]
        assert [row["elapsed_receive_s"] for row in rows[1:]] == [0, .2, .3]
        assert all(row["field_quality"] for row in rows)
        assert rows[0]["settings"]["field_step_px"] == 16
        second = read_history_index(child, replay_live_results(index)["record"])
        assert [row["sequence"] for row in second.verified_records()] == [1, 1, 2, 3, 4, 5]
    finally:
        writer.close()


def test_replay_cancellation_and_invalid_source_leave_no_published_record(tmp_path):
    root = capture_with_known_receipts(tmp_path / "run")
    path = _latest(root, gauges=_GAUGES)
    index = read_history_index(root, path)
    stop = Event()

    def cancel_after_image(done, _total):
        if done == 1:
            stop.set()

    _assert_error("LIVE_REPLAY_CANCELLED", lambda: replay_live_results(
        index, cancelled=stop.is_set, progress=cancel_after_image,
    ))
    saved = path.read_bytes()
    path.write_bytes(saved.replace(b'"follow_mode":"latest"', b'"follow_mode":"ordered"', 1))
    _assert_error("LIVE_RECORD_CHANGED", lambda: replay_live_results(index))
    path.write_bytes(saved)
    with ExperimentReader(root) as reader:
        skipped = reader.image_path(reader.frame_at_sequence("sim:0", 3))
    skipped.write_bytes(b"changed TIFF")
    _assert_error("CAPTURE_IMAGE_CHANGED", lambda: replay_live_results(index))
    assert set(path.parent.iterdir()) == {path}
    assert path.read_bytes() == saved
