from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from straintrace_campaign import delivery


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_keystone_source(root: Path) -> Path:
    version = root / "keystone" / "20261015_v001"
    catalog: list[dict[str, object]] = []
    source_files: list[Path] = []
    fixture_rows = {
        "coupon-dic": [
            {"frame": 1, "original_frame_index": 1, "native_cbf_frame": 1, "scan_id": 44, "scan_id_is_global_fio": True, "stress_MPa": 7.123456789012345, "engineering_strain": 0.0, "timestamp_unix_s": 1777550465.572986, "estimated_timestamp_unix_s": None},
            {"frame": 2, "original_frame_index": 2, "native_cbf_frame": 3, "scan_id": 44, "scan_id_is_global_fio": True, "stress_MPa": None, "engineering_strain": None, "timestamp_unix_s": None, "estimated_timestamp_unix_s": 1777550466.572986},
        ],
        "coupon-no-dic": [
            {"frame": 1, "original_frame_index": 1, "native_cbf_frame": 8, "scan_id": 157, "scan_id_is_global_fio": True, "stress_MPa": 10.271179164299999, "engineering_strain": None, "timestamp_unix_s": None, "estimated_timestamp_unix_s": 1778000001.0},
        ],
    }
    for acquisition_id, rows in fixture_rows.items():
        rel = f"5_Grip_5mm/condition/{acquisition_id}"
        sample_dir = version / "samples" / rel
        sample = {
            "schema_version": "desy-frame-mechanics-sample-1.0",
            "acquisition_id": acquisition_id,
            "sample_name": acquisition_id,
            "delivery_relative_path": rel,
            "physical_specimen_id": None,
            "dimensions": {
                "width_mm_recorded": 2.0,
                "thickness_mm_recorded": 1.5,
                "accepted_area_mm2": 3.0,
                "accepted_area_status": "CONFIRMED_RECORDED_DIMENSIONS",
                "accepted_area_basis": "recorded width * thickness",
                "alternate_area_mm2": 3.1,
                "alternate_area_source": "conflicting historical record",
                "area_conflict": True,
            },
            "dic": {"present_in_indexed_sources": acquisition_id == "coupon-dic", "reference_frame": 1},
        }
        sources = {"schema_version": "desy-frame-mechanics-sources-1.0", "acquisition_id": acquisition_id, "sources": [{"role": "accepted baseline", "path": "private/source.csv", "sha256": "abc", "verification_status": "UPSTREAM_HASH_PIN_NOT_REHASHED"}]}
        _write_json(sample_dir / "sample.json", sample)
        _write_json(sample_dir / "sources.json", sources)
        source_files.extend([sample_dir / "sample.json", sample_dir / "sources.json"])
        with (sample_dir / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        source_files.append(sample_dir / "frames.jsonl")
        with (sample_dir / "frames.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            fields = list(rows[0])
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: ("null" if key == "stress_MPa" and value is None else "" if value is None else "true" if value is True else "false" if value is False else format(value, ".17g") if isinstance(value, float) else str(value)) for key, value in row.items()})
        source_files.append(sample_dir / "frames.csv")
        origin = sample_dir / "frame_stress_strain.txt"
        origin.write_text("frame\tstress_MPa\n1\t7.123456789012345\n", encoding="utf-8")
        source_files.append(origin)
        catalog.append({
            "acquisition_id": acquisition_id,
            "sample_name": acquisition_id,
            "delivery_relative_path": rel,
            "DIC_present": acquisition_id == "coupon-dic",
            "frames": len(rows),
            "stress_valid": sum(row["stress_MPa"] is not None for row in rows),
            "strain_valid": sum(row["engineering_strain"] is not None for row in rows),
            "valid_pairs": 1 if acquisition_id == "coupon-dic" else 0,
            "sample_json": f"samples/{rel}/sample.json",
            "frames_csv": f"samples/{rel}/frames.csv",
            "frames_jsonl": f"samples/{rel}/frames.jsonl",
            "sources_json": f"samples/{rel}/sources.json",
            "origin_txt": f"samples/{rel}/frame_stress_strain.txt",
        })
    _write_json(version / "catalog.json", catalog)
    with (version / "catalog.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(catalog[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(catalog)
    _write_json(version / "VALIDATION.json", {"status": "PASS", "passed": True, "scope_acquisitions_expected": 2, "scope_acquisitions_written": 2, "frames_written": 3})
    checksums = {path.relative_to(version).as_posix(): {"sha256": _sha(path), "bytes": path.stat().st_size} for path in [*source_files, version / "catalog.json", version / "catalog.csv", version / "VALIDATION.json"]}
    _write_json(version / "CHECKSUMS.json", checksums)
    return version


def _project_with_import(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    source = _write_keystone_source(tmp_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_json(project_dir / "project.json", {"project_id": "fixture", "name": "fixture", "acquisitions": []})
    acquisitions = delivery.import_keystone(project_dir, source)
    return project_dir, acquisitions


def _campaign(project_dir: Path, acquisitions: list[dict[str, object]]) -> dict[str, object]:
    return {"schema_version": 1, "project_id": "fixture", "name": "fixture", "created_at": "2026-10-15T12:00:00Z", "metadata": {"output_version": "fixture_v001"}, "roots": {}, "acquisitions": acquisitions}


def test_import_preserves_ids_frames_exact_rows_and_source_metadata(tmp_path: Path) -> None:
    project_dir, acquisitions = _project_with_import(tmp_path)
    assert [item["acquisition_id"] for item in acquisitions] == ["coupon-dic", "coupon-no-dic"]
    dic = acquisitions[0]
    assert dic["result"]["status"] == "IMPORTED_ACCEPTED"
    assert dic["image_frames"] == []
    assert dic["measurement_frames"][1]["frame"] == 2
    assert dic["measurement_frames"][1]["native_cbf_frame"] == 3
    assert dic["dimensions"]["area_mm2"] == 3.0
    assert dic["dimensions"]["alternate_area_mm2"] == 3.1
    import_root = project_dir / dic["result"]["rows_path"]
    source_root = tmp_path / "keystone" / "20261015_v001"
    assert (import_root.parent / "frames.jsonl").read_bytes() == (source_root / "samples" / "5_Grip_5mm/condition/coupon-dic/frames.jsonl").read_bytes()
    original_sample = json.loads((import_root.parent / "sample.json").read_text(encoding="utf-8"))
    original_sources = json.loads((import_root.parent / "sources.json").read_text(encoding="utf-8"))
    assert original_sample["acquisition_id"] == dic["acquisition_id"]
    assert original_sources["sources"][0]["verification_status"] == "UPSTREAM_HASH_PIN_NOT_REHASHED"
    assert dic["result"]["provenance"]["numerics_rerun"] is False
    assert dic["metadata"]["keystone_import"]["historical_source_claims"][0]["rehashed_during_import"] is False
    assert not any(path.suffix.lower() in {".tif", ".tiff", ".cbf"} for path in project_dir.rglob("*"))


def test_import_reuses_only_complete_matching_staging_after_project_save_failure(tmp_path: Path) -> None:
    project_dir, first = _project_with_import(tmp_path)
    source = tmp_path / "keystone" / "20261015_v001"
    manifest_path = project_dir / "imports/keystone/20261015_v001/IMPORT_MANIFEST.json"
    before = manifest_path.read_bytes()
    retry = delivery.import_keystone(project_dir, source)
    assert retry == first
    assert manifest_path.read_bytes() == before
    imported_file = project_dir / "imports/keystone/20261015_v001/coupon-dic/frames.jsonl"
    imported_file.unlink()
    with pytest.raises(ValueError, match="checksum failed|incomplete"):
        delivery.import_keystone(project_dir, source)


def test_keystone_current_supports_version_relative_path(tmp_path: Path) -> None:
    version = _write_keystone_source(tmp_path)
    source_root = version.parent
    _write_json(source_root / "CURRENT.json", {"version_relative_path": version.name})
    assert delivery._resolve_keystone_version(source_root) == version.resolve()


def test_export_verifies_masks_precision_origin_files_and_zip(tmp_path: Path) -> None:
    project_dir, acquisitions = _project_with_import(tmp_path)
    receipt = delivery.export_project(project_dir, _campaign(project_dir, acquisitions), tmp_path / "out")
    assert receipt["verification"]["valid"]
    assert receipt["archive_verification"]["valid"]
    assert delivery.verify_package(receipt["archive_path"])["valid"]
    version = Path(receipt["version_path"])
    dic_dir = version / "samples/5_Grip_5mm/condition/coupon-dic"
    no_dic_dir = version / "samples/5_Grip_5mm/condition/coupon-no-dic"
    with (dic_dir / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[1]["stress_MPa"] == "null"
    assert csv_rows[1]["engineering_strain"] == ""
    json_rows = [json.loads(line) for line in (dic_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert json_rows[0]["stress_MPa"] == 7.123456789012345
    assert json_rows[1]["stress_MPa"] is None
    assert json_rows[1]["engineering_strain"] is None
    assert (dic_dir / "frame_stress_strain.txt").read_text(encoding="utf-8-sig").splitlines()[0].split("\t") == ["frame", "stress_MPa", "engineering_strain"]
    assert (no_dic_dir / "frame_stress.txt").read_text(encoding="utf-8-sig").splitlines()[0].split("\t") == ["frame", "stress_MPa"]
    current = json.loads((tmp_path / "out/CURRENT.json").read_text(encoding="utf-8"))
    assert current["version_path"] == "versions/fixture_v001"
    assert current["catalog_json"] == "versions/fixture_v001/catalog.json"


def test_export_copies_linked_endpoint_qc_roi_and_fullfield_artifacts(tmp_path: Path) -> None:
    project_dir = tmp_path / "campaign"
    project_dir.mkdir()
    sample_id = "analysis-sample"
    rel = "5_Grip_5mm/condition/analysis-sample"
    run_root = project_dir / "stages" / sample_id / "run-abc"
    rows_path = run_root / "rows.jsonl"
    rows_path.parent.mkdir(parents=True)
    row = {"frame": 1, "native_cbf_frame": 10, "stress_MPa": 12.5, "engineering_strain": 0.002, "G01_engineering_strain": 0.002, "quality_status": "ACCEPTED"}
    rows_path.write_text(json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8")
    endpoint = run_root / "endpoint_trajectories.csv"
    endpoint.write_text("frame,gauge,x1,y1,x2,y2\n1,G01,10,20,30,20\n", encoding="utf-8")
    qc = run_root / "tracking_qc.csv"
    qc.write_text("frame,gauge,status,corr,fb_error\n1,G01,ACCEPTED,0.98,0.02\n", encoding="utf-8")
    field_output = run_root / "fullfield" / "fields.npz"
    field_output.parent.mkdir()
    field_output.write_bytes(b"synthetic full-field artifact\x00\x01")
    preview = project_dir / "proposals" / sample_id / "reference_preview.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"synthetic ROI preview")
    raw_image = project_dir / "inputs" / "raw.tif"
    raw_image.parent.mkdir()
    raw_image.write_bytes(b"raw pixels must stay external to package")
    rel_rows = rows_path.relative_to(project_dir).as_posix()
    stage_artifacts = [rel_rows, endpoint.relative_to(project_dir).as_posix(), qc.relative_to(project_dir).as_posix(), field_output.relative_to(project_dir).as_posix()]
    checksums = {path: _sha(project_dir / path) for path in stage_artifacts}
    acquisition = {
        "acquisition_id": sample_id,
        "name": sample_id,
        "relative_path": rel,
        "source_kind": "generic",
        "physical_specimen_id": None,
        "dimensions": {"area_mm2": 2.0, "status": "CONFIRMED", "source": "fixture"},
        "analysis_mode": "fullfield",
        "image_frames": [{"frame": 1, "path": str(raw_image)}],
        "measurement_frames": [{"frame": 1, "native_cbf_frame": 10}],
        "roi": {"preview_paths": [preview.relative_to(project_dir).as_posix()]},
        "approval": None,
        "analysis_config": {},
        "result": {"status": "COMPLETED", "fingerprint": "abc123456789", "rows_path": rel_rows, "artifacts": stage_artifacts, "checksums": checksums, "provenance": {}, "summary": {}},
        "alignment": None,
        "metadata": {},
    }
    project = {"project_id": "artifact-test", "name": "artifact-test", "created_at": "2026-10-15T00:00:00Z", "metadata": {"output_version": "artifact_v001"}, "acquisitions": [acquisition]}
    receipt = delivery.export_project(project_dir, project, tmp_path / "artifact-out")
    version = Path(receipt["version_path"])
    sample_dir = version / "samples" / Path(rel)
    sources = json.loads((sample_dir / "sources.json").read_text(encoding="utf-8"))
    by_source = {item["source_project_relative_path"]: item for item in sources["analysis_artifacts"]}
    for source in (endpoint, qc, field_output, preview):
        source_rel = source.relative_to(project_dir).as_posix()
        record = by_source[source_rel]
        packaged = version / Path(record["package_path"])
        assert packaged.read_bytes() == source.read_bytes()
        assert record["sha256"] == _sha(packaged)
    assert not list(version.rglob("*.tif"))
    assert len(list(sample_dir.glob("frames.jsonl"))) == 1
    assert delivery.verify_package(version)["valid"]


def test_verifier_rejects_csv_mask_change_and_wrong_catalog_link(tmp_path: Path) -> None:
    project_dir, acquisitions = _project_with_import(tmp_path)
    receipt = delivery.export_project(project_dir, _campaign(project_dir, acquisitions), tmp_path / "out")
    version = Path(receipt["version_path"])
    dic_dir = version / "samples/5_Grip_5mm/condition/coupon-dic"
    csv_path = dic_dir / "frames.csv"
    content = csv_path.read_text(encoding="utf-8-sig").replace("null", "", 1)
    csv_path.write_text(content, encoding="utf-8-sig")
    assert not delivery.verify_package(version)["valid"]

    # Restore accepted CSV from the campaign import and change the catalog link
    # while updating hashes, so the verifier must reject path traversal itself.
    source_csv = project_dir / acquisitions[0]["result"]["rows_path"]
    source_csv = source_csv.parent / "frames.csv"
    csv_path.write_bytes(source_csv.read_bytes())
    catalog_path = version / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog[0]["package_path"] = "../../outside"
    _write_json(catalog_path, catalog)
    checksum_files = {p.relative_to(version).as_posix(): {"sha256": _sha(p), "bytes": p.stat().st_size} for p in version.rglob("*") if p.is_file() and p.name != "CHECKSUMS.json"}
    _write_json(version / "CHECKSUMS.json", {"schema_version": "straintrace-checksums-1", "files": checksum_files})
    report = delivery.verify_package(version)
    assert not report["valid"]
    assert any("Unsafe relative artifact path" in error for error in report["errors"])

    root_current = tmp_path / "out/CURRENT.json"
    current = json.loads(root_current.read_text(encoding="utf-8"))
    current["catalog_json"] = "../../outside/catalog.json"
    _write_json(root_current, current)
    root_report = delivery.verify_package(tmp_path / "out")
    assert not root_report["valid"]
    assert any("catalog links" in error for error in root_report["errors"])


def test_publish_interruption_keeps_previous_current_and_resume_promotes_complete_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir, acquisitions = _project_with_import(tmp_path)
    published = tmp_path / "remote"
    first_project = _campaign(project_dir, acquisitions)
    first_project["metadata"]["output_version"] = "first"
    first = delivery.export_project(project_dir, first_project, tmp_path / "local1", publish_to=published)
    previous_current = (published / "CURRENT.json").read_bytes()
    assert first["publish"]["verification"]["valid"]

    second_project = _campaign(project_dir, acquisitions)
    second_project["metadata"]["output_version"] = "second"
    second = delivery.export_project(project_dir, second_project, tmp_path / "local2")

    original_copy = delivery._copy_file_within_worker
    calls = 0

    def interrupted_copy(src: Path, dst: Path, digest: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_copy(src, dst, digest)
            raise TimeoutError("simulated child interruption")
        original_copy(src, dst, digest)

    monkeypatch.setattr(delivery, "_copy_file_within_worker", interrupted_copy)
    with pytest.raises(TimeoutError, match="simulated child interruption"):
        delivery._worker_publish(Path(second["version_path"]), published, "second")
    assert (published / "CURRENT.json").read_bytes() == previous_current
    assert delivery.verify_package(published)["valid"]
    assert (published / "versions/.second.staging").is_dir()
    monkeypatch.setattr(delivery, "_copy_file_within_worker", original_copy)
    resumed = delivery._worker_publish(Path(second["version_path"]), published, "second")
    assert resumed["verification"]["valid"]
    assert json.loads((published / "CURRENT.json").read_text(encoding="utf-8"))["version_id"] == "second"


def test_publish_lock_serializes_writers_and_releases_os_lock_without_signals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "publish-root"
    root.mkdir()
    lock_path = root / "PUBLISH_LOCK.json"
    monkeypatch.setattr(delivery.os, "kill", lambda *_args: pytest.fail("PID liveness must not signal any process"))
    with delivery._publish_lock(root) as active:
        assert active["state"] == "ACTIVE"
        with pytest.raises(RuntimeError, match="locked by another writer"):
            with delivery._publish_lock(root):
                pass
    released = json.loads(lock_path.read_text(encoding="utf-8"))
    assert released["state"] == "RELEASED"
    assert released["token"] == active["token"]

    # A persistent owner record alone is not a held lock: the OS advisory lock
    # was released, so a new writer safely acquires and replaces the record.
    lock_path.write_text(json.dumps({"host": "another-host", "pid": 1, "state": "ACTIVE"}), encoding="utf-8")
    with delivery._publish_lock(root) as active:
        assert active["host"] == delivery.socket.gethostname()
        assert active["state"] == "ACTIVE"


def test_io_worker_enforces_subprocess_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout_run(*args: object, **kwargs: object) -> None:
        raise delivery.subprocess.TimeoutExpired(cmd="worker", timeout=0.1)

    monkeypatch.setattr(delivery.subprocess, "run", timeout_run)
    with pytest.raises(TimeoutError, match="Bounded zip worker exceeded"):
        delivery._run_io_worker("zip", ["source", "target"], 0.1)


def test_frozen_gui_worker_uses_sibling_console_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gui_exe = tmp_path / "ezDIC.exe"
    cli_exe = tmp_path / "ezDIC-cli.exe"
    gui_exe.touch()
    cli_exe.touch()
    monkeypatch.setattr(delivery.sys, "frozen", True, raising=False)
    monkeypatch.setattr(delivery.sys, "executable", str(gui_exe))
    monkeypatch.setattr(delivery.sys, "stdout", None)
    command = delivery._worker_command(["publish", "src", "dest", "v1"])
    assert command[:2] == [str(cli_exe), "--campaign-io-worker"]
    cli_exe.unlink()
    with pytest.raises(RuntimeError, match="sibling ezDIC-cli.exe was not found"):
        delivery._worker_command(["publish", "src", "dest", "v1"])


def test_failed_publish_keeps_complete_local_archive_and_failure_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir, acquisitions = _project_with_import(tmp_path)

    def fail_publish(*args: object, **kwargs: object) -> dict[str, object]:
        raise TimeoutError("destination unavailable")

    monkeypatch.setattr(delivery, "_publish_version", fail_publish)
    output = tmp_path / "out"
    with pytest.raises(TimeoutError, match="destination unavailable"):
        delivery.export_project(project_dir, _campaign(project_dir, acquisitions), output, publish_to=tmp_path / "unavailable")
    receipt = json.loads((output / "PUBLICATION_fixture_v001.json").read_text(encoding="utf-8"))
    assert receipt["local_package_verified"] is True
    assert receipt["local_archive_sha256"]
    assert Path(receipt["local_archive_path"]).is_file()
    assert delivery.verify_package(output)["valid"]


def test_keystone_import_rejects_wrong_checksum_and_duplicate_frames(tmp_path: Path) -> None:
    source = _write_keystone_source(tmp_path)
    frame_file = source / "samples/5_Grip_5mm/condition/coupon-dic/frames.jsonl"
    frame_file.write_text(frame_file.read_text(encoding="utf-8") + frame_file.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    # Checksum validation catches source edits before any project import is staged.
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_json(project_dir / "project.json", {"project_id": "fixture", "acquisitions": []})
    with pytest.raises(ValueError, match="checksum mismatch"):
        delivery.import_keystone(project_dir, source)
    assert not (project_dir / "imports/keystone").exists()
