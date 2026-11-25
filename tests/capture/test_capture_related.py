from contextlib import closing
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, ExperimentReader, inspect_plan, inspect_related_experiments, prepare_repeat, prepare_trial
from straintrace_capture.calibration import CheckerboardTarget, prepare_checkerboard_capture
from straintrace_capture.cameras.base import Device, Image, receipt_times
from straintrace_capture.cli import main
from straintrace_capture.continuation import continuation_plan
from straintrace_capture.repeat import validate_repeat_destination
from straintrace_capture.storage import DATABASE, ExperimentWriter


def _plan(name="同名实验", **metadata):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": name,
        "cameras": [{"id": "sim:0", "settings": {"width": 64, "height": 48, "pixel_format": "Mono8"}}],
        "phases": [{"mode": "single", "role": "reference"}, {"mode": "timed", "fps": 12, "frame_count": 3}],
        "metadata": metadata,
    })


def _record(directory, config=None, *, count=1, multipart=False, state="completed"):
    writer = ExperimentWriter(directory, config or _plan())
    try:
        camera = "sim:0"
        writer.register_camera(Device(camera, "sim", "StrainTrace", "Simulated", "SIM-0000", "simulated"), {})
        for sequence in range(1, count + 1):
            utc, monotonic = receipt_times()
            image = Image(np.full((48, 64), sequence, dtype=np.uint8), "Mono8", 8, "mono", utc, monotonic)
            if multipart:
                image = replace(image, auxiliary_images=(image,))
            frame = writer.prepare_frame(camera, sequence, 1 if sequence == 1 else 2,
                                         "reference" if sequence == 1 else "test", image, None, None, None, 0)
            writer.commit_frames((frame,))
        writer.set_state(state)
        return writer.experiment_id
    finally:
        writer.close()


def _replace_record(path, edit):
    with closing(sqlite3.connect(path / DATABASE)) as db, db:
        record = json.loads(db.execute("SELECT record FROM experiment").fetchone()[0])
        edit(record)
        db.execute("UPDATE experiment SET record=? WHERE singleton=1", (json.dumps(record),))


def _rows(report):
    return {Path(row["directory"]): row for row in report["runs"]}


def _fingerprint(project):
    return {path: sha256(path.read_bytes()).hexdigest() for path in project.rglob("*")
            if path.is_file() and (path.name == DATABASE or path.suffix == ".tiff")}


def test_repeat_loads_all_phases_and_preserves_separate_provenance(tmp_path, capsys):
    source = tmp_path / "原批次" / "来源"
    destination = tmp_path / "复拍批次" / "第二组"
    original = _plan("试样 A", procedure="tension", continuation_of="unrelated-prior").to_dict()
    original["phases"].append({
        "mode": "hardware", "hardware_source": "Line1", "frame_count": 3, "monitor_trigger_wait": True,
        "hardware_pairing": {"source_id": "DAQ", "counter_feature": "ChunkCounterValue", "first_counter": {"sim:0": 10}},
    })
    original["durability"] = "write_through"
    source_id = _record(source, CaptureConfig.from_dict(original), count=2)
    before = _fingerprint(source)
    repeat = prepare_repeat(source, name="试样 B", expected_experiment_id=source_id)
    assert repeat.config.name == "试样 B" and repeat.config.durability == "write_through"
    assert [phase.mode for phase in repeat.config.phases] == ["single", "timed", "hardware"]
    assert repeat.cleared_anchors == (3,) and repeat.config.phases[2].hardware_source == "Line1"
    assert repeat.config.phases[2].hardware_pairing is None and repeat.config.phases[2].monitor_trigger_wait
    assert repeat.config.metadata["procedure"] == "tension"
    assert "continuation_of" not in repeat.config.metadata and not destination.exists()
    assert repeat.config.metadata["repeat_of"] == source_id
    assert main(["prepare-repeat", str(source), "--name", "试样 B", "--output", str(destination),
                 "--expect-experiment-id", source_id]) == 0
    assert CaptureConfig.from_dict(json.loads(capsys.readouterr().out)) == repeat.config
    nested = source / "images" / "新实验"
    with pytest.raises(CaptureError) as bad_destination:
        validate_repeat_destination(source, nested)
    assert bad_destination.value.code == "OUTPUT_INSIDE_REPEAT_SOURCE"
    assert any(row["code"] == "OUTPUT_INSIDE_REPEAT_SOURCE" for row in inspect_plan(repeat.config, nested)["issues"])
    with pytest.raises(CaptureError) as nested_writer:
        ExperimentWriter(nested, repeat.config)
    assert nested_writer.value.code == "OUTPUT_INSIDE_REPEAT_SOURCE" and not nested.exists()

    repeated_id = _record(destination, repeat.config)
    sibling = destination.parent / "错误路径线索"
    _record(sibling, _plan(continuation_of=source_id, continuation_parent={"directory": str(tmp_path / "missing")}))
    with ExperimentReader(destination) as reader:
        recorded = reader.experiment["config"]["metadata"]
        assert recorded["repeat_parent"]["relative_directory"] == "../../原批次/来源"
        assert reader.experiment["experiment_id"] == repeated_id != source_id
    related = _rows(inspect_related_experiments(destination))
    assert related[source]["relation"] == "repeat_source" and related[destination]["repeat_of"] == source_id
    assert _rows(inspect_related_experiments(source, search_dirs=(destination.parent,)))[destination]["relation"] == "repeat"
    assert "repeat_of" not in continuation_plan(destination).metadata
    assert _fingerprint(source) == before


def test_repeat_checks_the_source_identity_and_rejects_trial_ledgers(tmp_path):
    source = tmp_path / "formal"
    _record(source)
    with pytest.raises(CaptureError) as changed:
        prepare_repeat(source, expected_experiment_id="wrong-id")
    assert changed.value.code == "REPEAT_SCOPE_CHANGED"
    trial, location = prepare_trial(_plan(), tmp_path / "candidate", phase_number=2)
    _record(location, trial)
    with pytest.raises(CaptureError) as wrong_source:
        prepare_repeat(location)
    assert wrong_source.value.code == "REPEAT_SOURCE_UNSUPPORTED"



def test_parent_child_ancestor_and_shared_parent_keep_distinct_ledger_counts(tmp_path):
    original, child, fork, later = (tmp_path / folder for folder in ("原实验", "续采", "并行续采", "再次续采"))
    identity = _record(original, count=2, multipart=True, state="interrupted")
    child_id = _record(child, continuation_plan(original), count=3, state="stopped")
    fork_id = _record(fork, continuation_plan(original), count=4)
    later_id = _record(later, continuation_plan(child), count=1)
    decoy = tmp_path / "同名无关实验"
    _record(decoy, count=6)
    _replace_record(decoy, lambda row: row["config"].update(metadata=[]))
    hashes = _fingerprint(tmp_path)

    result = inspect_related_experiments(child, expected_experiment_id=child_id)
    assert result["issues"] == []
    by_path = _rows(result)
    assert {row["experiment_id"] for row in by_path.values()} == {identity, child_id, fork_id, later_id}
    assert by_path[original]["relation"] == "parent" and by_path[original]["state"] == "interrupted"
    assert by_path[original]["cameras"] == {"sim:0": {"frames": 2, "images": 4}}
    assert (by_path[original]["committed_frames"], by_path[original]["committed_images"]) == (2, 4)
    assert [by_path[path]["relation"] for path in (child, fork, later)] == ["current", "branch", "continuation"]
    assert _rows(inspect_related_experiments(later))[original]["relation"] == "ancestor"
    assert _rows(inspect_related_experiments(original))[later]["relation"] == "continuation"
    assert _fingerprint(tmp_path) == hashes


def test_writer_records_actual_relative_output_and_finds_the_parent_after_project_copy_then_move(tmp_path):
    project = tmp_path / "初始目录"
    original = project / "第一批" / "原实验"
    child = project / "第二批" / "正式实验"
    original_id = _record(original)
    plan = continuation_plan(original)
    assert plan.metadata["continuation_parent"] == {"directory": str(original)}
    child_id = _record(child, plan, count=3)
    with ExperimentReader(child) as reader:
        metadata = reader.experiment["config"]["metadata"]
        assert metadata["continuation_of"] == original_id
        assert metadata["continuation_parent"]["relative_directory"] == "../../第一批/原实验"
    assert "relative_directory" not in plan.metadata["continuation_parent"]

    moved = tmp_path / "移动后"
    shutil.copytree(project, moved)
    copied_child, copied_parent = moved / child.relative_to(project), moved / original.relative_to(project)
    result = inspect_related_experiments(copied_child, expected_experiment_id=child_id)
    assert {path for path, row in _rows(result).items() if row["relation"] == "parent"} == {original, copied_parent}
    issue, = result["issues"]
    assert issue["code"] == "ID_MULTIPLE_DIRECTORIES"
    assert set(map(Path, issue["directories"])) == {original, copied_parent}

    project.rename(tmp_path / "原目录已停用")
    moved_report = inspect_related_experiments(copied_child)
    assert moved_report["issues"] == []
    assert _rows(moved_report)[copied_parent]["experiment_id"] == original_id
    following = continuation_plan(copied_child)
    assert following.metadata["continuation_parent"] == {"directory": str(copied_child)}


def test_replaced_hint_is_excluded_and_moved_parent_can_be_found_explicitly_or_from_history(tmp_path, capsys):
    root = tmp_path / "设备数据"
    original, child, storage = root / "原实验", tmp_path / "采集现场" / "续采", tmp_path / "归档"
    root.mkdir()
    storage.mkdir()
    original_id = _record(original)
    child_id = _record(child, continuation_plan(original))
    moved = original.rename(storage / "重新命名后的实验")
    wrong_id = _record(original)
    default = inspect_related_experiments(child)
    assert len(default["runs"]) == 1
    missing, = default["issues"]
    assert missing == {"code": "PARENT_UNAVAILABLE", "experiment_id": original_id, "child_ids": [child_id],
                       "mismatched_directories": [str(original)]}
    assert len(inspect_related_experiments(child, search_dirs=[wrong := original])["runs"]) == 1
    recent = inspect_related_experiments(child, recent_paths=[wrong, moved])
    assert _rows(recent)[moved]["experiment_id"] == original_id and not recent["issues"]

    before = _fingerprint(tmp_path)
    assert main(["related", str(child), "--search", str(storage)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "straintrace.capture-related/v1"
    assert {row["experiment_id"] for row in payload["runs"]} == {original_id, child_id}
    assert wrong_id not in {row["experiment_id"] for row in payload["runs"]}
    assert _fingerprint(tmp_path) == before
    with pytest.raises(CaptureError, match="标识已变化"):
        inspect_related_experiments(original, expected_experiment_id=original_id)


def test_old_child_resolves_by_id_and_search_is_limited_to_the_selected_levels(tmp_path):
    home = tmp_path / "实验"
    moved = tmp_path / "归档" / "2026" / "外移的原实验"
    original_id = _record(moved)
    child = home / "旧格式续采"
    _record(child, _plan(continuation_of=original_id))
    selected = inspect_related_experiments(child, search_dirs=[tmp_path / "归档"])
    assert len(selected["runs"]) == 1 and selected["issues"][0]["code"] == "PARENT_UNAVAILABLE"
    found = inspect_related_experiments(child, search_dirs=[moved])
    assert _rows(found)[moved]["relation"] == "parent"
    assert found["issues"] == []

    unknown = tmp_path / "不存在"
    report = inspect_related_experiments(child, search_dirs=[unknown])
    assert {"code": "SEARCH_UNAVAILABLE", "directory": str(unknown)} in report["issues"]


def test_conflicting_same_id_directories_never_bridge_two_parents(tmp_path):
    first, other, child, copy = (tmp_path / folder for folder in ("原实验", "无关实验", "续采", "续采副本"))
    first_id, other_id = _record(first), _record(other)
    child_id = _record(child, continuation_plan(first))
    shutil.copytree(child, copy)
    _replace_record(copy, lambda row: row["config"]["metadata"].update(continuation_of=other_id))

    report = inspect_related_experiments(child)
    assert {row["relation"] for row in report["runs"]} == {"current", "current_copy"}
    assert {row["experiment_id"] for row in report["runs"]} == {child_id}
    assert {item["code"] for item in report["issues"]} == {"ID_MULTIPLE_DIRECTORIES", "PARENT_CONFLICT"}
    conflict = next(item for item in report["issues"] if item["code"] == "PARENT_CONFLICT")
    assert set(conflict["parent_ids"]) == {first_id, other_id}


def test_cyclic_ancestry_is_reported_without_inventing_an_order(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first_id = _record(first)
    second_id = _record(second, continuation_plan(first))
    _replace_record(first, lambda row: row["config"]["metadata"].update(continuation_of=second_id))

    result = inspect_related_experiments(second)
    assert _rows(result)[first]["relation"] == "linked"
    assert result["issues"] == [{"code": "LINEAGE_CYCLE", "experiment_ids": sorted([first_id, second_id])}]


def test_independent_trial_and_calibration_omit_both_link_fields(tmp_path):
    source = _plan(continuation_of="old", continuation_parent={"directory": str(tmp_path / "other")}, user_tag="keep")
    formal = tmp_path / "planned"
    trial, _ = prepare_trial(source, formal)
    calibration, _ = prepare_checkerboard_capture(source, formal, CheckerboardTarget(("sim:0",), 7, 6, 2.0))
    for derived in (trial, calibration):
        assert {"continuation_of", "continuation_parent"}.isdisjoint(derived.metadata)
        assert derived.metadata["user_tag"] == "keep"
    assert source.metadata["continuation_of"] == "old" and "continuation_parent" in source.metadata


@pytest.mark.skipif(os.name != "nt", reason="Windows UNC paths require Windows path semantics")
def test_untrusted_unc_locator_does_not_touch_a_network_path(tmp_path, monkeypatch):
    child = tmp_path / "child"
    path = r"\\example.invalid\capture\old"
    _record(child, _plan(continuation_of="unavailable", continuation_parent={"directory": path}))
    lstat = Path.lstat

    def checked_lstat(self, *args, **kwargs):
        assert not str(self).startswith(r"\\example.invalid")
        return lstat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", checked_lstat)
    result = inspect_related_experiments(child)
    assert len(result["runs"]) == 1 and result["issues"][0]["code"] == "PARENT_UNAVAILABLE"
