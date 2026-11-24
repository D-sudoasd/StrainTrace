"""Manifest integrity and exact-file transaction regression tests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import ezdic_core as core
import ezdic_cli


@pytest.fixture(autouse=True)
def _isolated_operation_state(tmp_path, monkeypatch):
    monkeypatch.setenv("EZDIC_STATE_DIR", str(tmp_path / "operation-state"))


def _tx(root: Path, run_id: str, filename: str, text: str, *, relative_dir: str = "core"):
    tx = core.RunTransaction(root, config={"mode": "test"}, mode="test", run_id=run_id)
    tx.create_staging()
    target = tx.stage_path(Path(relative_dir) / filename)
    target.write_text(text, encoding="utf-8")
    tx.seal(status="completed", scientific_ok=True, required_outputs=[f"{relative_dir}/{filename}"])
    return tx.commit()


def test_transaction_archives_exact_previous_outputs_and_preserves_user_file(tmp_path: Path) -> None:
    root = tmp_path / "out"
    user = root / "core" / "strain_user_notes.txt"
    user.parent.mkdir(parents=True)
    user.write_text("user-owned", encoding="utf-8")
    manifest = _tx(root, "first", "result.txt", "generated-1")
    second = _tx(root, "second", "result.txt", "generated-2")
    assert manifest == root / "run_manifest.json"
    assert second == root / "run_manifest.json"
    assert (root / "core" / "result.txt").read_text(encoding="utf-8") == "generated-2"
    assert user.read_text(encoding="utf-8") == "user-owned"
    archived = root / "_previous_runs" / "second" / "core" / "result.txt"
    assert archived.read_text(encoding="utf-8") == "generated-1"
    assert core.verify_run_manifest(root / "run_manifest.json")["ok"] is True


@pytest.mark.parametrize(
    "run_id",
    [
        "foo/../../evil",
        r"C:\evil",
        r"\\server\share",
        ".",
        "..",
        "CON",
        "rυn",  # Unicode confusable, not an ASCII token.
        "run\ncontrol",
        "x" * 65,
    ],
)
def test_invalid_run_id_fails_before_any_output_path_is_created(tmp_path: Path, run_id: str) -> None:
    root = tmp_path / "invalid-run-id"
    with pytest.raises(core.CoreError) as error:
        core.RunTransaction(root, config={"mode": "test"}, mode="test", run_id=run_id)
    assert error.value.code == "INVALID_RUN_ID"
    assert not root.exists()
    assert not (tmp_path / "operation-state").exists()


def test_valid_deterministic_run_id_is_used_as_a_contained_staging_token(tmp_path: Path) -> None:
    root = tmp_path / "valid-run-id"
    tx = core.RunTransaction(root, config={"mode": "test"}, mode="test", run_id="run_20260830_01")
    staging = tx.create_staging()
    assert tx.run_id == "run_20260830_01"
    assert staging == root / ".staging_run_20260830_01"
    assert staging.is_dir()
    assert staging.resolve().is_relative_to(root.resolve())


def test_run_id_is_revalidated_at_defensive_path_boundaries(tmp_path: Path) -> None:
    tx = core.RunTransaction(tmp_path / "defensive", config={"mode": "test"}, mode="test", run_id="safe")
    tx.run_id = "../escaped"
    with pytest.raises(core.CoreError) as error:
        tx.create_staging()
    assert error.value.code == "INVALID_RUN_ID"
    assert not (tmp_path / "escaped").exists()


def test_successive_output_set_archives_ledger_owned_files_omitted_by_new_run(tmp_path: Path) -> None:
    root = tmp_path / "successive"
    _tx(root, "first", "strain_G01.txt", "first-generated")
    _tx(root, "second", "qc_summary.txt", "second-generated", relative_dir="qc")

    current_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert not (root / "core" / "strain_G01.txt").exists()
    assert (root / "qc" / "qc_summary.txt").read_text(encoding="utf-8") == "second-generated"
    assert (root / "_previous_runs" / "second" / "core" / "strain_G01.txt").read_text(encoding="utf-8") == "first-generated"
    assert "core/strain_G01.txt" not in current_manifest.get("preserved_output_paths", [])
    assert core.verify_run_manifest(root / "run_manifest.json")["ok"] is True

    # The omitted path is available for a later regeneration and is archived
    # again as part of the exact current-set transition.
    _tx(root, "third", "strain_G01.txt", "third-generated")
    assert (root / "core" / "strain_G01.txt").read_text(encoding="utf-8") == "third-generated"
    assert (root / "_previous_runs" / "third" / "qc" / "qc_summary.txt").read_text(encoding="utf-8") == "second-generated"


def test_case_variant_nonexistent_roots_share_windows_operation_lock(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows case-insensitive lock-key contract")
    upper = tmp_path / "OutNew"
    lower = tmp_path / "outnew"
    assert core._operation_key(upper) == core._operation_key(lower)
    ready = tmp_path / "ready-case.txt"
    script = """
from pathlib import Path
import sys
import time
import ezdic_core as core
with core._operation_os_lock(Path(sys.argv[1])):
    Path(sys.argv[2]).write_text('ready', encoding='utf-8')
    time.sleep(1.2)
"""
    child_env = dict(os.environ)
    child_env["EZDIC_STATE_DIR"] = str(tmp_path / "operation-state")
    child = subprocess.Popen([sys.executable, "-c", script, str(upper), str(ready)], env=child_env)
    try:
        deadline = time.monotonic() + 6.0
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        started = time.monotonic()
        with core._operation_os_lock(lower):
            elapsed = time.monotonic() - started
        assert elapsed >= 0.5
    finally:
        child.wait(timeout=5)


def test_manifest_detects_output_config_and_code_tamper(tmp_path: Path) -> None:
    output_root = tmp_path / "output-tamper"
    output_manifest = _tx(output_root, "output-tamper", "result.txt", "generated")
    output = output_root / "core" / "result.txt"
    output.write_text("changed", encoding="utf-8")
    verification = core.verify_run_manifest(output_manifest)
    assert verification["ok"] is False
    assert any(error["code"] == "FILE_IDENTITY_MISMATCH" for error in verification["errors"])

    config_root = tmp_path / "config-tamper"
    config_manifest = _tx(config_root, "config-tamper", "result.txt", "generated")
    config_payload = json.loads(config_manifest.read_text(encoding="utf-8"))
    config_payload["config"]["mode"] = "tampered"
    config_payload["manifest_hash"] = core.canonical_json_hash(
        {key: value for key, value in config_payload.items() if key != "manifest_hash"}
    )
    config_manifest.write_text(json.dumps(config_payload), encoding="utf-8")
    verification = core.verify_run_manifest(config_manifest)
    assert verification["ok"] is False
    assert any(error["code"] == "CONFIG_HASH_MISMATCH" for error in verification["errors"])
    assert not any(error["code"] == "MANIFEST_HASH_MISMATCH" for error in verification["errors"])

    code_root = tmp_path / "code-tamper"
    code_manifest = _tx(code_root, "code-tamper", "result.txt", "generated")
    code_payload = json.loads(code_manifest.read_text(encoding="utf-8"))
    code_payload["code_files"][0]["sha256"] = "0" * 64
    code_payload["manifest_hash"] = core.canonical_json_hash(
        {key: value for key, value in code_payload.items() if key != "manifest_hash"}
    )
    code_manifest.write_text(json.dumps(code_payload), encoding="utf-8")
    verification = core.verify_run_manifest(code_manifest)
    assert verification["ok"] is False
    assert any(error["code"] == "CODE_FINGERPRINT_MISMATCH" for error in verification["errors"])
    assert not any(error["code"] == "MANIFEST_HASH_MISMATCH" for error in verification["errors"])


def test_manifest_missing_required_field_and_unexpected_owned_pattern_fail(tmp_path: Path) -> None:
    root = tmp_path / "out"
    manifest_path = _tx(root, "required", "result.txt", "generated")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("code_fingerprint")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = core.verify_run_manifest(manifest_path, verify_code=False)
    assert verification["ok"] is False
    assert any(error["code"] == "MANIFEST_REQUIRED_FIELD_MISSING" for error in verification["errors"])

    # Restore a valid manifest and add a file matching an explicit ezDIC output
    # pattern; arbitrary notes.txt remains user-owned and is not flagged.
    root2 = tmp_path / "out-rogue"
    manifest_path = _tx(root2, "required-2", "result.txt", "generated")
    rogue = root2 / "core" / "strain_rogue.txt"
    rogue.write_text("rogue", encoding="utf-8")
    verification = core.verify_run_manifest(manifest_path)
    assert verification["ok"] is False
    assert any(error["code"] == "UNEXPECTED_OUTPUT" for error in verification["errors"])


def test_manifest_rejects_absolute_and_escaping_output_paths(tmp_path: Path) -> None:
    root = tmp_path / "out-escape"
    manifest_path = _tx(root, "escape", "result.txt", "generated")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["outputs"][0]["path"] = "../escaped.txt"
    payload["owned_output_paths"] = ["../escaped.txt"]
    payload["required_output_paths"] = ["../escaped.txt"]
    payload["manifest_hash"] = core.canonical_json_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = core.verify_run_manifest(manifest_path)
    assert verification["ok"] is False
    assert any(error["code"] == "MANIFEST_OUTPUT_OUTSIDE_ROOT" for error in verification["errors"])

    root_absolute = tmp_path / "out-absolute"
    manifest_absolute = _tx(root_absolute, "absolute", "result.txt", "generated")
    payload = json.loads(manifest_absolute.read_text(encoding="utf-8"))
    payload["outputs"][0]["path"] = str((tmp_path / "outside.txt").resolve())
    payload["owned_output_paths"] = [payload["outputs"][0]["path"]]
    payload["required_output_paths"] = [payload["outputs"][0]["path"]]
    payload["manifest_hash"] = core.canonical_json_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    manifest_absolute.write_text(json.dumps(payload), encoding="utf-8")
    verification = core.verify_run_manifest(manifest_absolute)
    assert verification["ok"] is False
    assert any(error["code"] == "MANIFEST_OUTPUT_ABSOLUTE" for error in verification["errors"])


def test_untrusted_manifest_ownership_is_not_used_for_archive(tmp_path: Path) -> None:
    root = tmp_path / "out"
    manifest_path = _tx(root, "trusted", "result.txt", "generated")
    user = root / "core" / "strain_user_notes.txt"
    user.write_bytes(b"user-owned")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["owned_output_paths"].append("core/strain_user_notes.txt")
    payload["manifest_hash"] = core.canonical_json_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    # The recomputed but malformed ownership ledger is rejected by verify and
    # must never authorize moving the user file in a subsequent transaction.
    verification = core.verify_run_manifest(manifest_path)
    assert verification["ok"] is False
    tx = core.RunTransaction(root, config={"mode": "collision"}, mode="test", run_id="untrusted")
    assert all(relative != "core/strain_user_notes.txt" for relative, _ in tx._previous_owned)
    assert user.read_bytes() == b"user-owned"


def test_empty_owned_inventory_cannot_validate_nonempty_outputs(tmp_path: Path) -> None:
    root = tmp_path / "out"
    manifest_path = _tx(root, "owned-empty", "result.txt", "generated")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["owned_output_paths"] = []
    payload["manifest_hash"] = core.canonical_json_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = core.verify_run_manifest(manifest_path)
    assert verification["ok"] is False
    assert any(error["code"] == "MANIFEST_OWNERSHIP_MISMATCH" for error in verification["errors"])


def test_internal_gui_run_tokens_do_not_change_canonical_config_hash() -> None:
    base = {"analysis_mode": "extensometer", "output_dir": "out", "roi_groups": []}
    first = dict(base, _run_token=1, _code_paths=["ezdic_core.py"], _legacy_direct_processing=False)
    second = dict(base, _run_token=99, _code_paths=["dic_virtual_extensometer_gui_v7_multi_roi_range.py"], _legacy_direct_processing=True)
    assert core.canonical_json_hash(core._canonical_settings(first)) == core.canonical_json_hash(core._canonical_settings(second))


def test_canonical_config_override_matches_cli_normalized_config_without_changing_runtime() -> None:
    config = {
        "schema_version": 1,
        "analysis_mode": "fullfield",
        "image_paths": ["frame_1.png", "frame_02.png"],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": "results",
        "field_roi": [10, 10, 80, 80],
    }
    normalized = ezdic_cli.normalize_config(config)
    runtime = {"output_dir": "C:/safe/results", "field_roi": (10, 10, 80, 80), "_canonical_config": normalized}
    snapshot = core._canonical_settings(runtime)
    assert core.canonical_json_hash(snapshot) == core.canonical_json_hash(normalized)
    assert runtime["output_dir"] == "C:/safe/results"
    assert runtime["field_roi"] == (10, 10, 80, 80)


def test_frozen_sources_code_paths_do_not_invent_archive_core_path(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "_MEI_frozen"
    sources = bundle / "sources"
    sources.mkdir(parents=True)
    core_source = sources / "ezdic_core.py"
    cli_source = sources / "ezdic_cli.py"
    schema_source = sources / "run_config_v1.json"
    core_source.write_text("# frozen core\n", encoding="utf-8")
    cli_source.write_text("# frozen cli\n", encoding="utf-8")
    schema_source.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "out"
    output_root.mkdir()
    output = output_root / "core.txt"
    output.write_text("output\n", encoding="utf-8")
    monkeypatch.setattr(core.sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(core, "__file__", str(bundle / "ezdic_core.py"))
    settings = {"_code_paths": ["sources/ezdic_core.py", "sources/ezdic_cli.py", "sources/run_config_v1.json"]}
    paths = core._code_paths_for_settings(settings)
    assert [path.relative_to(bundle).as_posix() for path in paths] == [
        "sources/ezdic_core.py", "sources/ezdic_cli.py", "sources/run_config_v1.json"
    ]
    manifest = core.build_run_manifest(
        config={"mode": "frozen-test"},
        outputs=[output],
        output_root=output_root,
        code_paths=paths,
    )
    manifest_path = output_root / "run_manifest.json"
    core.write_run_manifest(manifest, manifest_path)
    verification = core.verify_run_manifest(manifest_path)
    assert verification["ok"] is True
    assert all(entry["path"].startswith("sources/") for entry in manifest["code_files"])


def test_frozen_default_code_path_missing_is_structured(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "_MEI_missing"
    bundle.mkdir()
    monkeypatch.setattr(core.sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(core, "__file__", str(bundle / "ezdic_core.py"))
    with pytest.raises(core.CoreError) as error:
        core._code_paths_for_settings({})
    assert error.value.code == "CODE_FILE_MISSING"


def test_frozen_gui_archive_paths_resolve_to_real_bundle_sources(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "_MEI_gui"
    sources = bundle / "sources"
    schemas = bundle / "schemas"
    sources.mkdir(parents=True)
    schemas.mkdir()
    for filename in ("ezdic_core.py", core.GUI_SOURCE_FILENAME):
        (sources / filename).write_text(f"# {filename}\n", encoding="utf-8")
    (schemas / "run_config_v1.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(core.sys, "_MEIPASS", str(bundle), raising=False)

    paths = core.resolve_code_paths(
        paths=[
            str(bundle / "archive.pyz" / "ezdic_core.py"),
            str(bundle / "archive.pyz" / core.GUI_SOURCE_FILENAME),
        ],
        include_gui=True,
    )
    assert [path.relative_to(bundle).as_posix() for path in paths] == [
        "sources/ezdic_core.py",
        f"sources/{core.GUI_SOURCE_FILENAME}",
        "schemas/run_config_v1.json",
    ]
    assert all(path.is_file() for path in paths)


def test_cooperating_processes_share_operation_lock(tmp_path: Path) -> None:
    root = tmp_path / "out"
    ready = tmp_path / "ready.txt"
    script = """
from pathlib import Path
import sys
import time
import ezdic_core as core
root = Path(sys.argv[1])
ready = Path(sys.argv[2])
with core._operation_os_lock(root):
    ready.write_text('ready', encoding='utf-8')
    time.sleep(1.2)
"""
    child_env = dict(os.environ)
    child_env["EZDIC_STATE_DIR"] = str(tmp_path / "operation-state")
    child = subprocess.Popen([sys.executable, "-c", script, str(root), str(ready)], env=child_env)
    try:
        deadline = time.monotonic() + 6.0
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        started = time.monotonic()
        with core._operation_os_lock(root):
            elapsed = time.monotonic() - started
        assert elapsed >= 0.5
    finally:
        child.wait(timeout=5)


def test_commit_move_failure_restores_current_and_retains_failed_stage(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "out"
    user = root / "core" / "strain_user_notes.txt"
    user.parent.mkdir(parents=True)
    user.write_bytes(b"user-owned")
    _tx(root, "baseline", "result.txt", "baseline")
    tx = core.RunTransaction(root, config={"mode": "test"}, mode="test", run_id="failure")
    tx.create_staging()
    target = tx.stage_path(Path("core") / "result.txt")
    target.write_text("new", encoding="utf-8")
    tx.seal(status="completed", scientific_ok=True, required_outputs=["core/result.txt"])
    real_link = core.os.link
    stage_to_root = 0

    def fail_publish(source, destination, *args, **kwargs):
        nonlocal stage_to_root
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path == tx.stage_root / "run_manifest.json" and destination_path == root / "run_manifest.json":
            stage_to_root += 1
            raise OSError("forced commit move")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(core.os, "link", fail_publish)
    with pytest.raises(RuntimeError, match="事务提交失败"):
        tx.commit()
    assert stage_to_root == 1
    assert (root / "core" / "result.txt").read_text(encoding="utf-8") == "baseline"
    assert (root / "run_manifest.json").is_file()
    assert (root / "core" / "result.txt").read_text(encoding="utf-8") == "baseline"
    assert list((root / "_failed_runs").rglob("core/result.txt"))
    failed_manifests = list((root / "_failed_runs").rglob("run_manifest.json"))
    assert failed_manifests and core.verify_run_manifest(failed_manifests[0])["ok"] is True
    assert user.read_bytes() == b"user-owned"
    assert not list(root.glob(".staging_*"))


def test_unowned_destination_collision_fails_closed_and_keeps_user_bytes(tmp_path: Path) -> None:
    root = tmp_path / "out"
    user = root / "core" / "result.txt"
    user.parent.mkdir(parents=True)
    user.write_bytes(b"user-owned exact collision")
    tx = core.RunTransaction(root, config={"mode": "collision"}, mode="test", run_id="collision")
    tx.create_staging()
    tx.stage_path("core/result.txt").write_text("must not overwrite", encoding="utf-8")
    tx.seal(status="completed", scientific_ok=True, required_outputs=["core/result.txt"])
    with pytest.raises(RuntimeError, match="OUTPUT_COLLISION|事务提交失败"):
        tx.commit()
    assert user.read_bytes() == b"user-owned exact collision"
    assert not (root / "run_manifest.json").exists()
    assert list((root / "_failed_runs").rglob("core/result.txt"))


def test_atomic_publish_race_never_overwrites_new_user_file(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "out"
    _tx(root, "baseline", "result.txt", "baseline")
    tx = core.RunTransaction(root, config={"mode": "race"}, mode="test", run_id="race")
    tx.create_staging()
    tx.stage_path("core/new-result.txt").write_text("generated", encoding="utf-8")
    tx.seal(status="completed", scientific_ok=True, required_outputs=["core/new-result.txt"])
    real_link = core.os.link
    raced = root / "core" / "new-result.txt"
    injected = {"done": False}

    def create_user_file_before_link(source, destination, *args, **kwargs):
        if Path(destination) == raced and not injected["done"]:
            injected["done"] = True
            Path(destination).write_bytes(b"concurrent-user-byte")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(core.os, "link", create_user_file_before_link)
    with pytest.raises(RuntimeError, match="事务提交失败"):
        tx.commit()
    assert raced.read_bytes() == b"concurrent-user-byte"
    assert (root / "core" / "result.txt").read_text(encoding="utf-8") == "baseline"
    assert (root / "run_manifest.json").is_file()
    assert list((root / "_failed_runs").rglob("core/new-result.txt"))


@pytest.mark.skipif(os.name != "nt", reason="junction replacement requires Windows directory sharing semantics")
def test_junction_swap_during_publish_cannot_create_outside_file(tmp_path: Path, monkeypatch) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    probe_target = tmp_path / "junction-probe-target"
    probe_target.mkdir()
    probe_link = tmp_path / "junction-probe-link"
    probe = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(probe_link), str(probe_target)],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0 or not probe_link.is_dir():
        pytest.skip("Windows junction creation is unavailable: " + (probe.stderr or probe.stdout))
    probe_link.rmdir()

    root = tmp_path / "out"
    destination_parent = root / "core"
    destination_parent.mkdir(parents=True)
    tx = core.RunTransaction(root, config={"mode": "junction-swap"}, mode="test", run_id="junction-swap")
    tx.create_staging()
    tx.stage_path("core/result.txt").write_text("generated", encoding="utf-8")
    tx.seal(status="completed", scientific_ok=True, required_outputs=["core/result.txt"])
    real_link = core.os.link
    attack = {}

    def swap_destination_parent(source, destination, *args, **kwargs):
        if Path(destination) == root / "core" / "result.txt" and not attack:
            attack["rmdir"] = subprocess.run(
                ["cmd", "/c", "rmdir", "/q", str(destination_parent)],
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
            attack["mklink"] = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(destination_parent), str(outside)],
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
            if attack["rmdir"].returncode == 0 and attack["mklink"].returncode == 0:
                attack["real_link_called"] = True
                real_link(source, destination, *args, **kwargs)
            raise OSError("forced junction swap attempt")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(core.os, "link", swap_destination_parent)
    with pytest.raises(RuntimeError, match="事务提交失败"):
        tx.commit()

    assert attack["rmdir"].returncode != 0
    assert attack["mklink"].returncode != 0
    assert not (outside / "result.txt").exists()
    assert destination_parent.is_dir()
    assert not (root / "run_manifest.json").exists()
    failed_manifests = list((root / "_failed_runs").rglob("run_manifest.json"))
    failed_outputs = list((root / "_failed_runs").rglob("core/result.txt"))
    assert failed_outputs
    assert failed_manifests and core.verify_run_manifest(failed_manifests[0])["ok"] is True
    failure_records = list((root / "_failed_runs").rglob("failure.json"))
    assert failure_records
    record = json.loads(failure_records[0].read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["retention_warnings"] == []
    assert not list(root.glob(".staging_*"))


def test_failed_evidence_retention_does_not_copy_over_a_destination(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "out"
    tx = core.RunTransaction(root, config={"mode": "retention"}, mode="test", run_id="retention")
    tx.create_staging()
    source = tx.stage_path("core/result.txt")
    source.write_text("private staging", encoding="utf-8")
    real_move = core._move_exact

    def fail_retention_move(_source, _destination):
        raise OSError("forced retention move failure")

    monkeypatch.setattr(core, "_move_exact", fail_retention_move)
    monkeypatch.setattr(core.shutil, "copy2", lambda *_args, **_kwargs: pytest.fail("copy2 replacement is forbidden"))
    failed_dir = tx.abort(RuntimeError("original failure"))
    monkeypatch.setattr(core, "_move_exact", real_move)
    assert failed_dir is not None
    assert source.read_text(encoding="utf-8") == "private staging"
    record = json.loads((failed_dir / "failure.json").read_text(encoding="utf-8"))
    assert record["retention_warnings"]
    assert not (root / "core" / "result.txt").exists()


def test_archive_move_failure_restores_current_and_retains_failed_stage(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "out"
    _tx(root, "baseline", "result.txt", "baseline")
    user = root / "core" / "strain_user_notes.txt"
    user.write_bytes(b"user-owned")
    tx = core.RunTransaction(root, config={"mode": "archive-failure"}, mode="test", run_id="archive-failure")
    tx.create_staging()
    tx.stage_path("core/result.txt").write_text("new", encoding="utf-8")
    tx.seal(status="completed", scientific_ok=True, required_outputs=["core/result.txt"])
    real_move = core.shutil.move

    def fail_archive(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path == root / "core" / "result.txt" and str(destination_path).startswith(str(root / "_previous_runs")):
            raise OSError("forced archive move")
        return real_move(source, destination)

    monkeypatch.setattr(core.shutil, "move", fail_archive)
    with pytest.raises(RuntimeError, match="事务提交失败"):
        tx.commit()
    assert (root / "core" / "result.txt").read_text(encoding="utf-8") == "baseline"
    assert user.read_bytes() == b"user-owned"
    assert list((root / "_failed_runs").rglob("core/result.txt"))
    assert not list(root.glob(".staging_*"))


def test_input_change_after_seal_is_failed_with_current_rollback_and_evidence(tmp_path: Path) -> None:
    root = tmp_path / "out"
    _tx(root, "baseline", "result.txt", "baseline")
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"before")
    tx = core.RunTransaction(
        root,
        config={"mode": "input-change"},
        mode="test",
        input_identities=core.ordered_input_manifest([input_path]),
        run_id="input-change-commit",
    )
    tx.create_staging()
    tx.stage_path("core/result.txt").write_text("new", encoding="utf-8")
    tx.seal(status="completed", scientific_ok=True, required_outputs=["core/result.txt"])
    input_path.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="INPUT_CHANGED_DURING_RUN|事务提交失败"):
        tx.commit()
    assert (root / "core" / "result.txt").read_text(encoding="utf-8") == "baseline"
    failed_manifest = list((root / "_failed_runs").rglob("run_manifest.json"))
    assert failed_manifest
    failed_verification = core.verify_run_manifest(failed_manifest[0])
    assert failed_verification["ok"] is False
    assert any(error["code"] == "FILE_IDENTITY_MISMATCH" for error in failed_verification["errors"])


def test_post_publication_manifest_verification_failure_rolls_back(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "out"
    _tx(root, "baseline", "result.txt", "baseline")
    tx = core.RunTransaction(root, config={"mode": "verify-failure"}, mode="test", run_id="verify-failure")
    tx.create_staging()
    tx.stage_path("core/result.txt").write_text("new", encoding="utf-8")
    tx.seal(status="completed", scientific_ok=True, required_outputs=["core/result.txt"])
    real_verify = core.verify_run_manifest
    monkeypatch.setattr(core, "verify_run_manifest", lambda _path: {"ok": False, "code": "INJECTED_VERIFY_FAILURE", "errors": [{"code": "INJECTED_VERIFY_FAILURE"}]})
    with pytest.raises(RuntimeError, match="事务提交失败"):
        tx.commit()
    monkeypatch.setattr(core, "verify_run_manifest", real_verify)
    assert (root / "core" / "result.txt").read_text(encoding="utf-8") == "baseline"
    failed_manifest = list((root / "_failed_runs").rglob("run_manifest.json"))
    assert failed_manifest and core.verify_run_manifest(failed_manifest[0])["ok"] is True
    assert not list(root.glob(".staging_*"))


def test_seal_rejects_input_change_and_missing_required_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"before")
    tx = core.RunTransaction(tmp_path / "out", config={"mode": "test"}, mode="test", input_identities=core.ordered_input_manifest([input_path]), run_id="input-change")
    tx.create_staging()
    input_path.write_bytes(b"after!")
    tx.stage_path("core/result.txt").write_text("result", encoding="utf-8")
    with pytest.raises(core.CoreError, match="INPUT_CHANGED_DURING_RUN"):
        tx.seal(status="completed", scientific_ok=True, required_outputs=["core/result.txt"])

    tx2 = core.RunTransaction(tmp_path / "out-missing", config={"mode": "test"}, mode="test", run_id="missing")
    tx2.create_staging()
    with pytest.raises(core.CoreError, match="OUTPUT_MISSING"):
        tx2.seal(status="completed", scientific_ok=True, required_outputs=["core/missing.txt"])


def test_verify_manifest_v2_rejects_empty_code_files(tmp_path: Path) -> None:
    root = tmp_path / "out"
    manifest_path = _tx(root, "baseline", "result.txt", "baseline")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["code_files"] = []
    payload["code_fingerprint"] = core.code_fingerprint()
    payload["manifest_hash"] = core.canonical_json_hash({k: v for k, v in payload.items() if k != "manifest_hash"})
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = core.verify_run_manifest(manifest_path, verify_code=True)
    assert verification["ok"] is False
    assert any(err["code"] == "CODE_FILE_SET_INVALID" for err in verification["errors"])


def test_commit_preserves_core_error_code_cleans_archive_and_avoids_duplicate_abort(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "out"
    _tx(root, "baseline", "result.txt", "baseline")
    tx = core.RunTransaction(root, config={"mode": "verify-failure"}, mode="test", run_id="preserve-code")
    tx.create_staging()
    tx.stage_path("core/result.txt").write_text("new", encoding="utf-8")
    tx.seal(status="completed", scientific_ok=True, required_outputs=["core/result.txt"])
    real_verify = core.verify_run_manifest
    monkeypatch.setattr(
        core,
        "verify_run_manifest",
        lambda _path, **kwargs: real_verify(_path, **kwargs)
        if kwargs.get("verify_code") is False
        else {"ok": False, "code": "INJECTED", "errors": [{"code": "INJECTED"}]},
    )
    with pytest.raises(core.CoreError, match="事务提交失败") as exc:
        tx.commit()
    monkeypatch.setattr(core, "verify_run_manifest", real_verify)
    assert exc.value.code == "MANIFEST_VERIFY_FAILED"
    assert not (root / "_previous_runs" / "preserve-code").exists()
    first_failed = tx.abort(exc.value)
    assert first_failed is not None
    assert len(list((root / "_failed_runs").iterdir())) == 1
