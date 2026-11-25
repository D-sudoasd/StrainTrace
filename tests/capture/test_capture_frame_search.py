import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, ExperimentReader
from straintrace_capture.cameras.base import Device, Image
from straintrace_capture.cli import main
from straintrace_capture.frame_search import FrameLocation, format_receipt_utc, parse_receipt_utc, parse_sequence_range
from straintrace_capture.image_inputs import recorded_image, verified_image_path
from straintrace_capture.storage import DATABASE, ExperimentWriter, TriggerEvent


_EPOCH = parse_receipt_utc("2026-11-25T09:30:00.123456780Z")


def _save(writer, camera, sequence, phase, role, time_ns, *, trigger=None, position=None, auxiliary=False):
    pixels = np.full((10, 12), sequence, dtype=np.uint8)
    image = Image(pixels, "Mono8", 8, "mono", time_ns, 1000 + sequence)
    if auxiliary:
        extra = replace(image, pixels=pixels[:, :6].copy())
        image = replace(image, auxiliary_images=(extra,))
    prepared = writer.prepare_frame(camera, sequence, phase, role, image, trigger, None, position, 0)
    writer.commit_frames([prepared])


@pytest.fixture
def open_capture(tmp_path):
    plan = CaptureConfig.from_dict({"config_version": 1, "name": "Frame search", "cameras": [{"id": "sim:0"}, {"id": "sim:1"}],
                                    "phases": [{"mode": "single", "role": "reference"}, {"mode": "manual"},
                                               {"mode": "manual", "role": "calibration"}]})
    root = tmp_path / "段落采集"
    writer = ExperimentWriter(root, plan)
    try:
        for name in ("sim:0", "sim:1"):
            writer.register_camera(Device(name, "sim", "fixture", name, name, "fixture"), {})
        writer.set_state("running")
        trigger = TriggerEvent("trigger-2", "software_trigger", {"expected_cameras": ["sim:0", "sim:1"], "phase": 2})
        for seq in (1, 3, 5, 9, 11, 13):
            phase = 1 if seq == 1 else 3 if seq == 13 else 2
            for name in ("sim:0", "sim:1"):
                _save(writer, name, seq, phase, plan.phases[phase - 1].role, _EPOCH + seq,
                      trigger=trigger if seq == 5 else None, position="post" if seq == 5 else None, auxiliary=seq == 5)
        yield root, writer
    finally:
        writer.set_state("completed")
        writer.close()


def _sequences(page):
    return [row["sequence"] for row in page["frames"]]


def _locate(page, index=0):
    row = page["frames"][index]
    return FrameLocation(page["experiment_id"], page["camera_id"], row["sequence"], row["ordinal"], row["frame_id"], row["sha256"])


def test_read_metadata_by_phase_role_camera_and_exact_host_time(open_capture):
    root, _ = open_capture
    with ExperimentReader(root) as reader:
        page = reader.frame_page(camera_id="sim:1", phase=2, role="test", first_sequence=4, last_sequence=9,
                                 received_from_ns=_EPOCH + 5, received_to_ns=_EPOCH + 8)
        assert _sequences(page) == [5]
        row = page["frames"][0]
        assert row["images"] == 2 and row["trigger_event_id"] == "trigger-2" and row["event_position"] == "post"
        assert row["received_utc_ns"] == _EPOCH + 5 and page["camera_id"] == "sim:1"
        assert _sequences(reader.frame_page(camera_id="sim:0", role="reference")) == [1]
        assert _sequences(reader.frame_page(camera_id="sim:0", role="calibration")) == [13]
        empty = reader.frame_page(camera_id="sim:0", first_sequence=10, last_sequence=10)
        assert empty["frames"] == [] and empty["older_before"] is None and empty["newer_after"] is None
        exact = _locate(page).read(reader)
        assert exact["camera_id"] == "sim:1" and exact["sequence"] == 5
        assert verified_image_path(reader, exact).is_file()


def test_keyset_pages_keep_stable_results_while_new_frames_and_gaps_are_written(open_capture):
    root, writer = open_capture
    with ExperimentReader(root) as reader:
        current = reader.frame_page(camera_id="sim:0", role="test", limit=2)
        assert _sequences(current) == [11, 9] and current["newer_after"] is None
        snapshot = current["snapshot_ordinal"]
    _save(writer, "sim:0", 14, 2, "test", _EPOCH + 14)
    _save(writer, "sim:0", 2, 2, "test", _EPOCH + 2)
    with ExperimentReader(root) as reader:
        older = reader.frame_page(camera_id="sim:0", role="test", limit=2, snapshot=snapshot, before=current["older_before"])
        assert _sequences(older) == [5, 3] and older["older_before"] is None
        newer = reader.frame_page(camera_id="sim:0", role="test", limit=2, snapshot=snapshot, after=older["newer_after"])
        assert newer == current
        refreshed = reader.frame_page(camera_id="sim:0", role="test", limit=6)
        assert _sequences(refreshed) == [2, 14, 11, 9, 5, 3]
        assert refreshed["snapshot_ordinal"] > snapshot


def test_frame_location_refuses_moved_scope_modified_record_and_tampered_tiff(open_capture):
    root, _ = open_capture
    with ExperimentReader(root) as reader:
        query = reader.frame_page(camera_id="sim:0", first_sequence=5, last_sequence=5)
        loc = _locate(query)
        for alternative in (replace(loc, experiment_id="another"), replace(loc, ordinal=loc.ordinal + 1),
                            replace(loc, sha256="a" * 64)):
            with pytest.raises(CaptureError) as error:
                alternative.read(reader)
            assert error.value.code == "FRAME_REVIEW_CHANGED"
        verified = loc.read(reader)
        path = verified_image_path(reader, verified)
        data = path.read_bytes()
        assert np.all(recorded_image(path, verified, verify=True).pixels == 5)
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        with pytest.raises(CaptureError) as corrupted:
            recorded_image(path, loc.read(reader), verify=True)
        assert corrupted.value.code == "CAPTURE_IMAGE_CHANGED"


def test_cli_filters_exact_offsets_fences_and_keeps_original_frames_shape(open_capture, capsys):
    root, _ = open_capture
    code = main(["find-frames", str(root), "--camera", "sim:1", "--sequence", "3-11", "--received-from",
                 "2026-11-25T17:30:00.123456785+08:00", "--received-to", "2026-11-25T09:30:00.123456789Z", "--limit", "1"])
    assert code == 0
    page = json.loads(capsys.readouterr().out)
    assert _sequences(page) == [9] and page["older_before"] == page["frames"][0]["ordinal"]
    codes = main(["find-frames", str(root), "--camera", "sim:1", "--sequence", "3-11", "--before-ordinal",
                  str(page["older_before"]), "--snapshot-ordinal", str(page["snapshot_ordinal"]),
                  "--expect-experiment-id", page["experiment_id"], "--limit", "2", "--received-from",
                  "2026-11-25T17:30:00.123456785+08:00", "--received-to", "2026-11-25T09:30:00.123456789Z"])
    assert codes == 0 and _sequences(json.loads(capsys.readouterr().out)) == [5]
    assert main(["frames", str(root), "--camera", "sim:1", "--limit", "1"]) == 0
    original = json.loads(capsys.readouterr().out)[0]
    assert "timestamps" in original and original["sequence"] == 1


@pytest.mark.parametrize("options,code", [
    ({"snapshot": 100_000}, "FRAME_REVIEW_CHANGED"), ({"expected_experiment_id": "moved"}, "FRAME_REVIEW_CHANGED"),
    ({"phase": 4}, "INVALID_FRAME_QUERY"), ({"role": "invalid"}, "INVALID_FRAME_QUERY"),
    ({"limit": 257}, "INVALID_FRAME_QUERY"), ({"first_sequence": 2**63}, "INVALID_FRAME_QUERY"),
    ({"before": 5, "after": 2}, "INVALID_FRAME_QUERY"), ({"received_from_ns": 7, "received_to_ns": 5}, "INVALID_FRAME_QUERY"),
])
def test_invalid_catalog_inputs_report_typed_errors(open_capture, options, code):
    root, _ = open_capture
    with ExperimentReader(root) as reader, pytest.raises(CaptureError) as error:
        reader.frame_page(camera_id="sim:0", **options)
    assert error.value.code == code


def test_missing_or_noninteger_receipt_does_not_match_a_time_filter(open_capture):
    root, _ = open_capture
    with closing(sqlite3.connect(root / DATABASE)) as db:
        db.execute("""UPDATE frames SET record=json_set(record, '$.timestamps.received_utc.value', 'unavailable')
                      WHERE camera_id=? AND sequence=?""", ("sim:0", 3))
        db.commit()
    with ExperimentReader(root) as reader:
        all_rows = reader.frame_page(camera_id="sim:0", first_sequence=3, last_sequence=3)
        assert all_rows["frames"][0]["received_utc_ns"] is None
        timed = reader.frame_page(camera_id="sim:0", first_sequence=3, last_sequence=3, received_from_ns=0)
        assert timed["frames"] == []


def test_timestamp_int_math_keeps_nanoseconds_and_handles_negative_epoch():
    assert parse_receipt_utc("1970-01-01T00:00:00.000000001Z") == 1
    assert parse_receipt_utc("1969-12-31T23:59:59.999999999Z") == -1
    assert format_receipt_utc(-1) == "1969-12-31 23:59:59.999999999"
    assert parse_receipt_utc("2026-11-25 17:30:00.123456789+08:00") == _EPOCH + 9
    for invalid in ("2026-11-25T09:30:00", "2026-11-25T09:30:00.1234567890Z", "2026-13-25T09:30:00Z"):
        with pytest.raises(CaptureError):
            parse_receipt_utc(invalid)
    assert parse_sequence_range("42") == (42, 42)
    assert parse_sequence_range(" 40–80 ") == (40, 80)
    with pytest.raises(CaptureError):
        parse_sequence_range("4096-2")
