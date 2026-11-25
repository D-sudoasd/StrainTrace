import csv
import json
import subprocess
import sys

import pytest

from straintrace_capture import ExperimentReader, live_results
from straintrace_capture.config import CaptureError
from straintrace_capture.live import LiveAnalysis
from straintrace_capture.live_history import export_history_csv, read_history_index
from straintrace_capture.live_results import LiveResultLog, export_live_csv
from straintrace_capture.mechanics_setup import inspect_load_csv, table_alignment
from straintrace_capture.storage import DATABASE, sha256_file

from test_capture_mechanics_setup import _options, capture_with_known_receipts
from test_capture_continuation_analysis import _images, _record


_GAUGES = [{"name": "G01", "roi1": (45, 25, 25, 25), "roi2": (45, 83, 25, 25), "strain_mode": "y"}]


def _live_log(experiment, source, **sync):
    info = inspect_load_csv(source)
    channel = table_alignment(info, experiment, "sim:0", **_options(info, **sync))
    log = LiveResultLog(experiment)
    with LiveAnalysis(experiment, gauges=_GAUGES, latest=False, load_alignment=channel) as follower:
        while row := follower.poll():
            log.append(row, follower, "ordered")
    assert log.error is None
    return log.path


def _table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def archived_relative(tmp_path):
    experiment = capture_with_known_receipts(tmp_path / "run")
    source = tmp_path / "力学.csv"
    source.write_text("Time (s),Load (kN),Extension (mm)\n0,0,0\n0.5,0.3,0.2\n", encoding="utf-8")
    log = _live_log(experiment, source, sync_frame="2", sync_time="0.5")
    return experiment, source, log


def test_load_backfill_reuses_the_saved_clock_and_protects_original_frames(tmp_path, archived_relative):
    experiment, source, log = archived_relative
    snapshots = {file: sha256_file(file) for file in (log, source, experiment / DATABASE, *experiment.rglob("*.tiff"))}
    # Another export of the same machine data can reformat the existing samples.
    final_load = tmp_path / "试验机最终导出.tsv"
    final_load.write_text("Time (s)\tLoad (kN)\tExtension (mm)\n0.0\t-0.000\t0.000\n"
                          "0.500\t0.300\t0.2\n1.5\t0.5\t0.4\n", encoding="utf-8")
    initial, result = tmp_path / "实时.csv", tmp_path / "补齐.csv"
    export_live_csv(log, initial)
    summary = export_live_csv(log, result, fill_load=True, load_table=final_load)
    assert summary["filled_frames"] == summary["filled_force_frames"] == summary["filled_displacement_frames"] == 1
    assert summary["missing_force_frames"] == 1 and summary["load_table_sha256"] == sha256_file(final_load)
    before, after = _table(initial), _table(result)
    for key in ("工程应变 (%)", "原图 SHA-256", "参考图标识"):
        assert [row[key] for row in before] == [row[key] for row in after]
    assert [float(row["载荷 (N)"]) if row["载荷 (N)"] else None for row in after] == [0, 300, 400, None]
    assert [row["载荷值来源"] for row in after] == ["实时匹配", "实时匹配", "后到采样", ""]
    filled = after[2]
    assert filled["实时载荷对应状态"] == "OUTSIDE_TIME_RANGE"
    assert filled["实时载荷表 SHA-256"] == snapshots[source] and filled["补齐载荷表 SHA-256"] == sha256_file(final_load)
    support = json.loads(filled["补齐采样依据"])["force_N"]
    assert (support["status"], support["indices"], support["weights"]) == ("LINEAR_TIME_INTERPOLATION", [1, 2], [.5, .5])
    assert after[3]["补齐核对载荷状态"] == "OUTSIDE_TIME_RANGE" and not after[3]["补齐采样依据"]
    assert all(sha256_file(file) == fingerprint for file, fingerprint in snapshots.items())

    archive = read_history_index(experiment, log, page_size=2)
    section = tmp_path / "缺测帧.csv"
    scoped = export_history_csv(archive, section, sequence_range=(3, 3), fill_load=True, load_table=final_load)
    selected = _table(section)
    assert scoped["test_frames"] == scoped["filled_force_frames"] == 1
    assert [row["原图序号"] for row in selected] == ["1", "3"]
    assert selected[1] == after[2]

    result_hash = sha256_file(result)
    final_load.write_text(final_load.read_text(encoding="utf-8").replace("0.300", "0.600"), encoding="utf-8")
    with pytest.raises(CaptureError) as rejected:
        export_live_csv(log, result, fill_load=True, load_table=final_load)
    assert rejected.value.code == "LOAD_ARCHIVE_PREFIX_CHANGED" and sha256_file(result) == result_hash
    for args, path in (({}, source), ({"fill_load": True, "load_table": source}, source)):
        with pytest.raises(CaptureError) as rejected:
            export_live_csv(log, path, **args)
        assert rejected.value.code == "LIVE_CSV_OVERWRITES_LOAD"


def test_legacy_raw_prefix_supports_appending_to_utf16_without_final_newline(tmp_path):
    experiment = capture_with_known_receipts(tmp_path / "run")
    source = tmp_path / "legacy.csv"
    source.write_text("Time (s),Load (N)\n0,0\n0.5,300", encoding="utf-16")
    log = _live_log(experiment, source, sync_frame="2", sync_time="0.5")
    old_rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    old_rows[0]["settings"]["mechanics"].pop("sample_prefix")
    for row in old_rows:
        row["mechanics"]["mechanics_source"].pop("sample_prefix")
    old_log = tmp_path / "legacy.jsonl"
    old_log.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in old_rows), encoding="utf-8")
    with source.open("ab") as stream:
        stream.write("\n1.5,500\n".encode("utf-16-le"))
    result = tmp_path / "late.csv"
    receipt = export_live_csv(old_log, result, fill_load=True)
    assert receipt["filled_force_frames"] == 1 and float(_table(result)[2]["载荷 (N)"]) == 400


def test_absolute_header_only_can_be_completed_from_the_original_source(tmp_path):
    experiment = capture_with_known_receipts(tmp_path / "run")
    source = tmp_path / "late-source.csv"
    header = "timestamp_unix_ms,force_N\n"
    source.write_text(header, encoding="utf-8")
    log = _live_log(experiment, source)
    source.write_text(header + "1800000000000,100\n1800000001000,200\n", encoding="utf-8")
    result = tmp_path / "late.csv"
    summary = export_live_csv(log, result, fill_load=True)
    assert summary["filled_force_frames"] == 3 and summary["missing_force_frames"] == 1
    assert _table(result)[1]["实时载荷对应状态"] == "LOAD_SOURCE_EMPTY"
    assert float(_table(result)[1]["载荷 (N)"]) == 150


def test_cli_exports_a_late_load_and_reports_coverage(tmp_path, archived_relative):
    _, source, log = archived_relative
    with source.open("a", encoding="utf-8") as stream:
        stream.write("1.5,0.5,0.4\n")
    output = tmp_path / "cli.csv"
    run = subprocess.run([sys.executable, "-B", "-m", "straintrace_capture", "export-live-results", str(log),
                          "--fill-load", "--output", str(output)], capture_output=True, text=True, timeout=25)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["filled_force_frames"] == 1
    assert float(_table(output)[2]["载荷 (N)"]) == 400


def test_recent_observed_load_prefix_is_verified_as_well_as_initial_prefix(tmp_path):
    experiment = capture_with_known_receipts(tmp_path / "run")
    source = tmp_path / "load.csv"
    first_samples = "Time (s),Load (N)\n0,0\n0.5,300\n"
    source.write_text(first_samples, encoding="utf-8")
    info = inspect_load_csv(source)
    channel = table_alignment(info, experiment, "sim:0", **_options(info, sync_frame="2", sync_time="0.5"))
    log = LiveResultLog(experiment)
    with LiveAnalysis(experiment, gauges=_GAUGES, latest=False, load_alignment=channel) as follower:
        for index in range(4):
            if index == 2:
                source.write_text(first_samples + "1.5,500\n", encoding="utf-8")
            log.append(follower.poll(), follower, "ordered")
    assert log.error is None
    records = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    assert [row["mechanics"]["mechanics_source"]["sample_prefix"]["rows"] for row in records] == [2, 2, 3, 3]
    source.write_text(first_samples + "1.5,700\n5,900\n", encoding="utf-8")
    result = tmp_path / "out.csv"
    with pytest.raises(CaptureError) as rejected:
        export_live_csv(log.path, result, fill_load=True)
    assert rejected.value.code == "LOAD_ARCHIVE_PREFIX_CHANGED" and not result.exists()


def test_archive_modified_between_validation_and_export_cannot_be_published(tmp_path, archived_relative, monkeypatch):
    _, source, log = archived_relative
    with source.open("a", encoding="utf-8") as stream:
        stream.write("1.5,0.5,0.4\n")
    real_prepare = live_results.prepare_load_reconciliation

    def change_after_validation(*args, **kwargs):
        result = real_prepare(*args, **kwargs)
        body = log.read_bytes()
        original_hash = json.loads(body.splitlines()[0])["image_sha256"].encode("ascii")
        alternate = b"0" * len(original_hash)
        assert original_hash != alternate
        log.write_bytes(body.replace(b'"image_sha256":"' + original_hash, b'"image_sha256":"' + alternate, 1))
        return result

    monkeypatch.setattr(live_results, "prepare_load_reconciliation", change_after_validation)
    result = tmp_path / "saved.csv"
    result.write_bytes(b"previous export")
    with pytest.raises(CaptureError) as rejected:
        export_live_csv(log, result, fill_load=True)
    assert rejected.value.code == "LIVE_RECORD_CHANGED" and result.read_bytes() == b"previous export"


def test_late_current_run_load_does_not_attach_to_a_borrowed_reference(tmp_path):
    ref, first, second = _images()
    parent, experiment = tmp_path / "parent", tmp_path / "continuation"
    parent_id = _record(parent, (ref,), "reference")
    current_id = _record(experiment, (first, second), "test", parent=parent_id)
    source = tmp_path / "absolute.csv"
    header = "timestamp_unix_s,force_N\n"
    source.write_text(header, encoding="utf-8")
    info = inspect_load_csv(source)
    channel = table_alignment(info, experiment, "sim:0", **_options(info, max_gap_s=30.0))
    log = LiveResultLog(experiment)
    with LiveAnalysis(experiment, gauges=_GAUGES, latest=False, load_alignment=channel,
                      reference_experiment=parent, reference_sequence=1) as follower:
        while row := follower.poll():
            log.append(row, follower, "ordered")
    assert log.error is None
    stamps = []
    for root in (parent, experiment):
        with ExperimentReader(root) as reader:
            stamps.extend(row["timestamps"]["received_utc"]["value"] // 10**9 for row in reader.frames())
    source.write_text(header + f"{min(stamps)-1},100\n{max(stamps)+2},500\n", encoding="utf-8")
    result = tmp_path / "late-continuation.csv"
    summary = export_live_csv(log.path, result, fill_load=True)
    rows = _table(result)
    assert [row["原图实验标识"] for row in rows] == [parent_id, current_id, current_id]
    assert not rows[0]["载荷 (N)"] and not rows[0]["补齐核对载荷状态"]
    assert all(row["载荷值来源"] == "后到采样" for row in rows[1:])
    assert summary["filled_force_frames"] == 2 and summary["missing_force_frames"] == 0
