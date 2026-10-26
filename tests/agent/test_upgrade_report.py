"""Offline report acceptance: source fidelity, gaps, geometry and portability."""
import csv
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from straintrace_agent.reporting import build_report, verify_report
from straintrace_agent.api import _hash_json


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seal_snapshots(run, config, plan):
    plan["configuration_sha256"] = _hash_json(config)
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _hash_json(plan)
    source = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    source["configuration_sha256"] = _hash_json(config)
    source["plan_sha256"] = plan["plan_sha256"]
    _json(run / "manifest.json", source)


def _resign_package_files(package):
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    for relative in manifest["files"]:
        path = package / relative
        manifest["files"][relative] = {"sha256": _digest(path), "bytes": path.stat().st_size}
    _json(package / "manifest.json", manifest)


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / "项目" / "runs" / "run-例"
    (run / "checkpoints").mkdir(parents=True)
    raw = tmp_path / "原始数据"
    raw.mkdir()
    reference = raw / "图像_1.png"
    Image.fromarray(np.arange(80 * 96, dtype=np.uint8).reshape(80, 96)).save(reference)
    mask = raw / "mask.png"
    valid_mask = np.ones((80, 96), dtype=np.uint8) * 255
    valid_mask[20:30, 40:50] = 0
    Image.fromarray(valid_mask).save(mask)
    config = {
        "project_id": '中文试样 <script>alert("x")</script> & "报告"',
        "image_dir": str(raw), "mask_file": str(mask), "mask_sha256": _digest(mask),
        "calibration": {"mm_per_px": .1, "origin_px": [10, 20], "origin_mm": [10, -2], "axis_sign": [1, -1]},
        "regions": [
            {"name": "中心 <R1>", "bounds": [20, 50, 20, 50], "coordinate_system": "image"},
            {"name": "边缘 & R2", "bounds": [10, 11, -3, -2], "coordinate_system": "calibrated"},
        ],
        "gauges": [
            {"name": "标距 <A>", "axis": "x", "a_roi_px": [20, 25, 10, 10], "b_roi_px": [60, 25, 10, 10]},
            {"name": '标距 "B"', "axis": "y", "a_point_px": [45, 25], "b_point_px": [45, 55]},
        ],
        "exclusions_px": [[40, 20, 10, 10]],
        "loads": {"path": "load.csv", "clock_confirmed": True, "area_mm2": 2.0},
    }
    plan = {
        "reference_native_frame": 1, "reference_path": str(reference),
        "analysis_rectangle_candidate_px": [10, 10, 70, 50],
        "selected_parameters": {"subset_size_px": 21, "step_px": 5, "strain_window_poi": 5, "solver": "icgn"},
        "gauge_reference_geometry": [
            {"name": "标距 <A>", "axis": "x", "a": {"reference_xy": [25, 30], "poi_indices": [0]},
             "b": {"reference_xy": [65, 30], "poi_indices": [1]}, "initial_length_px": 40,
             "reference_status": "resolved", "endpoint_policy": "fixed_reference_poi_set_all_pois_must_be_valid"},
            {"name": '标距 "B"', "axis": "y", "a": {"reference_xy": [45, 25], "poi_indices": [2]},
             "b": {"reference_xy": [45, 55], "poi_indices": [3]}, "initial_length_px": 30,
             "reference_status": "resolved", "endpoint_policy": "fixed_reference_poi_set_all_pois_must_be_valid"},
        ],
    }
    summary = []
    for native in (1, 2, 3, 7):
        failed = native == 3
        summary.append({"native_frame": native, "time_s": (native - 1) * .5,
                        "status": "failed" if failed else "complete", "quality_status": "fail" if failed else "pass",
                        "force_N": "NaN" if failed else "20.123456789123456", "stress_MPa": "" if failed else "10.061728394561728",
                        "machine_engineering_strain": "" if failed else "0.0042",
                        "failure_code": "READ_ERROR" if failed else "", "failure_reason": "无法读取 <坏帧> & 原因" if failed else ""})
    _csv(run / "summary.csv", summary)
    gauges = []
    for native in (1, 2, 7):
        for index, name in enumerate(("标距 <A>", '标距 "B"')):
            gauges.append({"native_frame": native, "time_s": (native - 1) * .5, "gauge_name": name,
                           "axis": "x" if index == 0 else "y", "status": "valid", "initial_length_px": 40 if index == 0 else 30,
                           "engineering_strain": str((native - 1) * .01 * (index + 1)), "true_strain": str(np.log1p((native - 1) * .01 * (index + 1)))})
    _csv(run / "gauges.csv", gauges)
    regions = []
    for native in (1, 2, 3, 7):
        for region in config["regions"]:
            for component in ("Exx", "exx"):
                regions.append({"native_frame": native, "time_s": (native - 1) * .5, "region_name": region["name"],
                                "component": component, "unit": "dimensionless", "status": "failed" if native == 3 else "valid",
                                "mean": "" if native == 3 else "0.012345678901234567", "sd_sample": "" if native == 3 else "0.003",
                                "n_poi_total": "4", "n_poi_valid": "0" if native == 3 else "3", "valid_fraction": "0" if native == 3 else ".75"})
    _csv(run / "regions.csv", regions)
    receipts = {}
    for native in (1, 2, 7):
        checkpoint = run / "checkpoints" / f"frame_native_{native:08d}.npz"
        np.savez_compressed(checkpoint, x=[25., 65., 45., 45.], y=[30., 30., 25., 55.],
                            u=[0., np.nan, .12, .24], valid=[True, False, True, True],
                            Exx=[0., np.nan, .01, .02], strain_valid=[True, False, True, True])
        sidecar = checkpoint.with_suffix(".json")
        _json(sidecar, {"native_frame": native, "status": "complete", "checkpoint": checkpoint.name, "checkpoint_sha256": _digest(checkpoint)})
        receipts[sidecar.name] = {"sha256": _digest(sidecar), "checkpoint_sha256": _digest(checkpoint), "native_frame": native, "status": "complete"}
    failure = run / "checkpoints/frame_native_00000003.json"
    _json(failure, {"native_frame": 3, "status": "failed", "error": {"code": "READ_ERROR"}})
    receipts[failure.name] = {"sha256": _digest(failure), "checkpoint_sha256": None, "native_frame": 3, "status": "failed"}
    source = {"schema": "straintrace.agent-manifest/v1", "project_id": config["project_id"], "run_id": run.name,
              "run_status": "partial", "run_fingerprint": "fixture-run", "source_images": [
                  {"native_frame": native, "time_s": (native - 1) * .5, "path": str(reference if native == 1 else raw / f"image_{native}.png"),
                   "sha256": _digest(reference) if native == 1 else "0" * 64} for native in (1, 2, 3, 4, 7)],
              "artifacts": {name: _digest(run / name) for name in ("summary.csv", "gauges.csv", "regions.csv")},
              "checkpoint_receipts": receipts, "environment": {"python": "test fixture", "straintrace": "0.1.4"}}
    _json(run / "manifest.json", source)
    _json(run / "quality.json", {"numerical_quality_passed": False, "scientific_acceptance": "not_assessed", "failed_frame_count": 1})
    _seal_snapshots(run, config, plan)
    return run, config, plan


def test_report_preserves_csv_precision_gaps_chinese_and_geometry(saved_run):
    run, config, plan = saved_run
    receipt = build_report(run, config=config, plan=plan)
    assert receipt["verification"]["valid"], receipt
    data = json.loads(Path(receipt["report_data"]).read_text(encoding="utf-8"))
    html = Path(receipt["report_html"]).read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "中心 &lt;R1&gt;" in html
    assert "0.012345678901234567" in html and "20.123456789123456" in html
    assert data["tables"]["summary"]["rows"][0]["force_N"] == "20.123456789123456"
    assert [row["native_frame"] for row in data["frames"]] == [1, 2, 3, 4, 7]
    assert data["frames"][2]["status"] == "failed" and data["frames"][3]["status"] == "missing"
    force = next(curve for curve in data["curves"] if curve["y_column"] == "force_N")
    assert [point["value"] for point in force["series"][0]["points"]][2:4] == [None, None]
    gauge = next(curve for curve in data["curves"] if curve["table"] == "gauges" and curve["y_column"] == "engineering_strain")
    assert len(gauge["series"]) == 2
    assert [point["source_row_index"] for point in gauge["series"][0]["points"]] == [0, 2, None, None, 4]
    roi_curves = [curve for curve in data["curves"] if curve["table"] == "regions"]
    assert len(roi_curves) == 2 and all(len(curve["series"]) == 2 for curve in roi_curves)
    calibrated = next(rect for rect in data["geometry"]["rectangles"] if rect["name"] == "边缘 & R2")
    assert calibrated["xywh"] == [10., 20., 10., 10.]
    assert data["geometry"]["image_shape_yx"] == [80, 96]
    assert data["geometry"]["mask_preview"] == "assets/mask.png"
    assert "不是标准误或实验不确定度" in html
    assert "<h1>StrainTrace 应变分析报告</h1>" in html
    assert "图像序列：原始数据" in html
    assert 'class="geometry-label"' in html and 'fill-opacity="0.96"' in html
    assert not list((run / "report/data/checkpoints").glob("*.npz"))
    assert all(record["sha256"] for record in data["checkpoint_files"])


def test_curve_does_not_bridge_missing_or_nonconsecutive_frames(saved_run):
    from straintrace_agent.reporting import _curve_svg
    run, config, plan = saved_run
    receipt = build_report(run, config=config, plan=plan)
    data = json.loads(Path(receipt["report_data"]).read_text(encoding="utf-8"))
    force = next(curve for curve in data["curves"] if curve["y_column"] == "force_N")
    svg = _curve_svg(force)
    # Only frames 1 -> 2 form a measured contiguous segment. Frames 3/4 are
    # missing, and 7 must not be connected to the last valid frame 2.
    assert svg.count('fill="none" stroke="#2563eb"') == 1
    assert svg.count('data-value=') == 3
    stress = next(curve for curve in data["curves"] if curve.get("x_table") == "gauges")
    point = stress["series"][1]["points"][-1]
    assert point["x_value"] == .12
    assert point["value"] == float(data["tables"]["summary"]["rows"][-1]["stress_MPa"])


def test_full_package_is_portable_and_verifiable_without_installed_package(saved_run, tmp_path):
    run, config, plan = saved_run
    original = {path.name: path.read_bytes() for path in (run / "checkpoints").iterdir()}
    receipt = build_report(run, config=config, plan=plan, include_checkpoints=True)
    package = tmp_path / "离线交付"
    shutil.copytree(run / "report", package)
    assert {path.name: path.read_bytes() for path in (run / "checkpoints").iterdir()} == original
    assert {path.name: path.read_bytes() for path in (package / "data/checkpoints").iterdir()} == original
    with np.load(package / "data/checkpoints/frame_native_00000002.npz", allow_pickle=False) as arrays:
        assert np.isnan(arrays["u"][1]) and arrays["valid"].tolist() == [True, False, True, True]
    shutil.rmtree(run)
    shutil.rmtree(Path(config["image_dir"]))
    result = subprocess.run([sys.executable, "-I", str(package / "verify_report.py"), str(package)],
                            cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    checked = json.loads(result.stdout)
    assert checked["valid"] and checked["include_checkpoints"]
    assert checked["external_sources_checked"] is False
    assert receipt["verification"]["scientific_acceptance"] == "not_assessed"


def test_report_copies_only_manifest_registered_checkpoint_files(saved_run, tmp_path):
    run, config, plan = saved_run
    registered = {path.name for path in (run / "checkpoints").iterdir()}
    (run / "checkpoints/private.json").write_text('{"private": true}', encoding="utf-8")
    (run / "checkpoints/private.npz").write_bytes(b"unregistered arrays")
    private = tmp_path / "external-private.json"
    private.write_text('{"external_private": true}', encoding="utf-8")
    # Windows may require Developer Mode for file symlinks. The directory
    # junction test below exercises real reparse points without that privilege.
    try:
        (run / "checkpoints/external-private.json").symlink_to(private)
    except OSError:
        pass
    receipt = build_report(run, config=config, plan=plan, include_checkpoints=True)
    data = json.loads(Path(receipt["report_data"]).read_text(encoding="utf-8"))
    assert {path.name for path in (run / "report/data/checkpoints").iterdir()} == registered
    assert {Path(row["source_relative_path"]).name for row in data["checkpoint_files"]} == registered
    assert (run / "report/data/checkpoints/frame_native_00000003.json").is_file()
    assert not (run / "report/data/checkpoints/frame_native_00000003.npz").exists()
    assert private.read_text(encoding="utf-8") == '{"external_private": true}'


@pytest.mark.parametrize("linked_part", ["directory", "sidecar", "npz"])
def test_report_rejects_registered_checkpoint_links(saved_run, tmp_path, linked_part):
    run, config, plan = saved_run
    if linked_part == "directory":
        checkpoint_dir = run / "checkpoints"
        external = tmp_path / "external-checkpoints"
        checkpoint_dir.rename(external)
        if os.name == "nt":
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(checkpoint_dir), str(external)],
                                    capture_output=True, text=True, check=False)
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            checkpoint_dir.symlink_to(external, target_is_directory=True)
    else:
        suffix = ".json" if linked_part == "sidecar" else ".npz"
        path = run / "checkpoints" / ("frame_native_00000001" + suffix)
        external = tmp_path / ("external-private" + suffix)
        path.rename(external)
        try:
            path.symlink_to(external)
        except OSError as exc:
            pytest.skip(f"File symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="Symlink or reparse point"):
        build_report(run, config=config, plan=plan, include_checkpoints=True)
    assert not (run / "report").exists()


@pytest.mark.parametrize("missing", ["sha256", "checkpoint_sha256"])
def test_report_rejects_incomplete_source_checkpoint_receipts(saved_run, missing):
    run, config, plan = saved_run
    path = run / "manifest.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    source["checkpoint_receipts"]["frame_native_00000001.json"][missing] = None
    _json(path, source)
    with pytest.raises(ValueError, match="lacks.*SHA-256"):
        build_report(run, config=config, plan=plan, include_checkpoints=True)


def test_report_rejects_missing_registered_checkpoint_in_solver_run(saved_run):
    run, config, plan = saved_run
    (run / "checkpoints/frame_native_00000001.npz").unlink()
    with pytest.raises(ValueError, match="Registered checkpoint is missing"):
        build_report(run, config=config, plan=plan)


@pytest.mark.parametrize("include_checkpoints", [False, True])
def test_derived_report_preserves_external_checkpoint_receipts_without_copying(saved_run, include_checkpoints):
    run, config, plan = saved_run
    extraction = run / "extractions/derived"
    extraction.mkdir(parents=True)
    for name in ("summary.csv", "gauges.csv", "regions.csv", "quality.json", "manifest.json"):
        shutil.copyfile(run / name, extraction / name)
    original_plan_sha = plan["plan_sha256"]
    plan.update(source_plan_sha256=original_plan_sha, analysis_kind="saved_field_postprocessing")
    source = json.loads((extraction / "manifest.json").read_text(encoding="utf-8"))
    source["postprocessing"] = {"schema": "straintrace.reprocess/v1", "source_run": run.name,
                                "source_plan_sha256": original_plan_sha, "solver_recomputed": False,
                                "checkpoint_receipts": source["checkpoint_receipts"]}
    _json(extraction / "manifest.json", source)
    _seal_snapshots(extraction, config, plan)
    receipt = build_report(extraction, config=config, plan=plan, include_checkpoints=include_checkpoints)
    data = json.loads(Path(receipt["report_data"]).read_text(encoding="utf-8"))
    assert receipt["verification"]["valid"]
    assert data["source_manifest"]["checkpoint_receipts"] == source["checkpoint_receipts"]
    assert len(data["checkpoint_files"]) == 7
    assert all(row["availability"] == "external" and row["bytes"] is None and not row["included"]
               for row in data["checkpoint_files"])
    assert not (extraction / "report/data/checkpoints").exists()
    assert data["included_array_count"] == 0
    if include_checkpoints:
        assert any("没有可复制的 NPZ" in warning for warning in data["warnings"])


def test_verifier_binds_resigned_checkpoint_records_to_source_receipts(saved_run):
    from straintrace_agent.reporting import _html
    run, config, plan = saved_run
    build_report(run, config=config, plan=plan, include_checkpoints=True)
    package = run / "report"
    data_path = package / "data/report_data.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    record = next(row for row in data["checkpoint_files"] if row["source_relative_path"].endswith(".npz"))
    path = package / record["package_path"]
    with path.open("ab") as stream:
        stream.write(b"changed checkpoint")
    record.update(sha256=_digest(path), bytes=path.stat().st_size)
    _json(data_path, data)
    (package / "index.html").write_text(_html(data), encoding="utf-8")
    _resign_package_files(package)
    checked = verify_report(package)
    assert not checked["valid"]
    assert f"Checkpoint source receipt mismatch: {record['source_relative_path']}" in checked["errors"]
    result = subprocess.run([sys.executable, "-I", str(package / "verify_report.py"), str(package)],
                            capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 1
    assert "Checkpoint source receipt mismatch" in result.stdout


def test_verifier_hashes_checkpoint_arrays_once_per_call_and_rechecks_tampering(saved_run, monkeypatch):
    import straintrace_agent.reporting as reporting
    run, config, plan = saved_run
    (run / "plots").mkdir()
    field_map = run / "plots/frame_native_00000001_Exx.png"
    Image.new("RGB", (32, 24), "white").save(field_map)
    source = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    source["artifacts"][field_map.relative_to(run).as_posix()] = _digest(field_map)
    _json(run / "manifest.json", source)
    build_report(run, config=config, plan=plan, include_checkpoints=True)
    package = run / "report"
    original_sha256 = reporting._sha256
    reads = {}

    def counting_sha256(path):
        if path.suffix == ".npz" or path.parent.name == "plots":
            reads[path.name] = reads.get(path.name, 0) + 1
        return original_sha256(path)

    monkeypatch.setattr(reporting, "_sha256", counting_sha256)
    assert verify_report(package)["valid"]
    assert reads and set(reads.values()) == {1}
    assert "00_frame_native_00000001_Exx.png" in reads
    path = next((package / "data/checkpoints").glob("*.npz"))
    # Same byte count and restored timestamp must not permit cross-call reuse.
    timestamp = path.stat()
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    os.utime(path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
    reads.clear()
    checked = verify_report(package)
    assert not checked["valid"]
    assert set(reads.values()) == {1}
    assert any("Checkpoint source checksum mismatch" in error for error in checked["errors"])


def test_tampering_and_resigned_csv_disagreement_are_detected(saved_run):
    run, config, plan = saved_run
    build_report(run, config=config, plan=plan)
    package = run / "report"
    path = package / "data/gauges.csv"
    path.write_text(path.read_text(encoding="utf-8-sig").replace("0.06", "0.99"), encoding="utf-8-sig")
    assert verify_report(package)["valid"] is False
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["data/gauges.csv"] = {"sha256": _digest(path), "bytes": path.stat().st_size}
    _json(package / "manifest.json", manifest)
    checked = verify_report(package)
    assert checked["valid"] is False
    assert "CSV/report-data mismatch: gauges" in checked["errors"]


def test_failed_rebuild_preserves_prior_package_and_unowned_paths(saved_run, tmp_path):
    run, config, plan = saved_run
    build_report(run, config=config, plan=plan)
    before = (run / "report/index.html").read_bytes()
    (run / "summary.csv").write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with source manifest"):
        build_report(run, config=config, plan=plan)
    assert (run / "report/index.html").read_bytes() == before
    assert verify_report(run / "report")["valid"]
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="not owned"):
        build_report(run, config=config, plan=plan, output_dir=foreign)
    assert (foreign / "keep.txt").read_text() == "user data"
    with pytest.raises(ValueError, match="ancestor"):
        build_report(run, config=config, plan=plan, output_dir=run.parent)


def test_rebuild_is_deterministic_and_independent_of_outer_report_hashes(saved_run):
    run, config, plan = saved_run
    build_report(run, config=config, plan=plan, include_checkpoints=True)
    first = {file.relative_to(run / "report").as_posix(): _digest(file) for file in (run / "report").rglob("*") if file.is_file()}
    source = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    source["artifacts"]["report/index.html"] = "outer-report-hash"
    _json(run / "manifest.json", source)
    build_report(run, config=config, plan=plan, include_checkpoints=True)
    second = {file.relative_to(run / "report").as_posix(): _digest(file) for file in (run / "report").rglob("*") if file.is_file()}
    assert first == second


def test_missing_reference_retains_spatial_geometry_and_reports_unavailable_preview(saved_run):
    run, config, plan = saved_run
    Path(plan["reference_path"]).unlink()
    receipt = build_report(run, config=config, plan=plan)
    data = json.loads(Path(receipt["report_data"]).read_text(encoding="utf-8"))
    assert data["warnings"] and "preview" not in data["geometry"]
    assert len(data["geometry"]["gauges"]) == 2 and data["geometry"]["rectangles"]
    assert verify_report(run / "report")["valid"]


def test_independent_verifier_rejects_unsafe_manifest_paths(saved_run):
    run, config, plan = saved_run
    build_report(run, config=config, plan=plan)
    manifest_path = run / "report/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["../outside.txt"] = {"bytes": 1, "sha256": "0" * 64}
    _json(manifest_path, manifest)
    checked = verify_report(run / "report")
    assert checked["valid"] is False
    assert any("Unsafe package path" in error for error in checked["errors"])


def test_frame_index_mode_never_labels_unknown_time_as_seconds(saved_run):
    from straintrace_agent.reporting import _curve_svg
    run, config, plan = saved_run
    config["time_basis"] = "frame_index"
    config["loads"] = None
    source = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for frame in source["source_images"]:
        frame["time_s"] = None
    for name in ("summary", "gauges", "regions"):
        with (run / f"{name}.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["time_s"] = ""
            for key in ("force_N", "stress_MPa", "machine_engineering_strain"):
                if key in row:
                    row[key] = ""
        _csv(run / f"{name}.csv", rows)
        source["artifacts"][f"{name}.csv"] = _digest(run / f"{name}.csv")
    _json(run / "manifest.json", source)
    _seal_snapshots(run, config, plan)
    receipt = build_report(run, config=config, plan=plan)
    data = json.loads(Path(receipt["report_data"]).read_text(encoding="utf-8"))
    gauge = next(curve for curve in data["curves"] if curve["table"] == "gauges" and curve["y_column"] == "engineering_strain")
    assert gauge["x_label"] == "native frame (index; not time)"
    assert [point["x_value"] for point in gauge["series"][0]["points"]] == [1, 2, 3, 4, 7]
    assert all(point["time_s"] is None for point in gauge["series"][0]["points"])
    assert _curve_svg(gauge).count('data-value=') == 6
    html = Path(receipt["report_html"]).read_text(encoding="utf-8")
    assert 'time (s)' not in html
    assert "帧号不代表秒数" in html
    assert html.index("逐标距工程应变</h3>") < html.index('class="unavailable-channels"')
    assert "未提供或无有效数据的通道" in html and "未提供试验机载荷或位移数据" in html
    assert "<h3>载荷</h3>" not in html
    assert verify_report(run / "report")["valid"]


def test_saved_field_gallery_is_bounded_hash_checked_and_keeps_index(saved_run):
    run, config, plan = saved_run
    (run / "plots").mkdir()
    source = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for native in range(1, 7):
        for component in ("Exx", "Eyy", "Exy"):
            path = run / "plots" / f"frame_native_{native:08d}_{component}.png"
            Image.new("RGB", (32, 24), "white").save(path)
            source["artifacts"][path.relative_to(run).as_posix()] = _digest(path)
    unknown = run / "plots" / "user_saved_unknown.png"
    Image.new("RGB", (32, 24), "blue").save(unknown)
    source["artifacts"][unknown.relative_to(run).as_posix()] = _digest(unknown)
    _json(run / "manifest.json", source)
    receipt = build_report(run, config=config, plan=plan)
    data = json.loads(Path(receipt["report_data"]).read_text(encoding="utf-8"))
    gallery = data["field_maps"]
    assert gallery["available_count"] == 18
    assert len(gallery["selected"]) == 12 and len(gallery["all_saved_pngs"]) == 19
    assert [entry["native_frame"] for entry in gallery["selected"][:3]] == [1, 4, 6]
    for entry in gallery["selected"]:
        assert (run / "report" / entry["package_path"]).read_bytes() == (run / entry["source_relative_path"]).read_bytes()
        assert "dimensionless" in entry["caption"] and "图像基底" in entry["caption"]
    assert gallery["all_saved_pngs"][-1]["component"] is None
    (run / gallery["selected"][0]["source_relative_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="field map disagrees"):
        build_report(run, config=config, plan=plan)
    assert verify_report(run / "report")["valid"]


def test_large_run_has_bounded_html_preview_and_complete_source_data(saved_run):
    run, config, plan = saved_run
    source = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summary = [{"native_frame": native, "time_s": (native - 1) * .5, "status": "failed" if native == 2500 else "complete",
                "quality_status": "fail" if native == 2500 else "pass", "force_N": "" if native == 2500 else "15.125",
                "failure_reason": "late failure" if native == 2500 else ""} for native in range(1, 3001)]
    _csv(run / "summary.csv", summary)
    source["source_images"] = [{"native_frame": row["native_frame"], "time_s": row["time_s"], "path": "external.png", "sha256": "0" * 64} for row in summary]
    source["artifacts"]["summary.csv"] = _digest(run / "summary.csv")
    _json(run / "manifest.json", source)
    receipt = build_report(run, config=config, plan=plan)
    data = json.loads(Path(receipt["report_data"]).read_text(encoding="utf-8"))
    markup = Path(receipt["report_html"]).read_text(encoding="utf-8")
    assert len(data["tables"]["summary"]["rows"]) == 3000 and len(data["frames"]) == 3000
    assert data["frames"][2499]["failure_reason"] == "late failure"
    assert "前 100 / 3000 行" in markup and "后续帧未显示" in markup
    assert markup.count('<tr data-source-row=') < 400
    assert markup.count('<circle ') < 2000
    assert len(markup) < 500_000
    with (run / "report/data/frame_status.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3000
    assert verify_report(run / "report")["valid"]


def test_report_rejects_csv_time_disagreeing_with_hashed_source_inventory(saved_run):
    run, config, plan = saved_run
    path = run / "gauges.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[2]["time_s"] = "9.9"
    _csv(path, rows)
    source = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    source["artifacts"]["gauges.csv"] = _digest(path)
    _json(run / "manifest.json", source)
    with pytest.raises(ValueError, match="time mismatch at frame 2"):
        build_report(run, config=config, plan=plan)


@pytest.mark.parametrize("change", ["calibration", "stale_plan", "resealed_plan"])
def test_report_rejects_snapshots_not_bound_to_solver_manifest(saved_run, change):
    run, config, plan = saved_run
    build_report(run, config=config, plan=plan, include_checkpoints=True)
    before = (run / "report/index.html").read_bytes()
    config, plan = copy.deepcopy(config), copy.deepcopy(plan)
    if change == "calibration":
        config["calibration"]["mm_per_px"] = 100.0
        message = "Configuration checksum does not match source manifest"
    else:
        plan["selected_parameters"]["subset_size_px"] = 23
        message = "Plan checksum does not match its contents"
        if change == "resealed_plan":
            plan.pop("plan_sha256")
            plan["plan_sha256"] = _hash_json(plan)
            message = "Plan checksum does not match source manifest"
    with pytest.raises(ValueError, match=message):
        build_report(run, config=config, plan=plan)
    assert (run / "report/index.html").read_bytes() == before
    assert verify_report(run / "report")["valid"]


@pytest.mark.parametrize("snapshot", ["project", "plan"])
def test_independent_verifier_rejects_mismatched_snapshots_even_after_package_resigning(saved_run, snapshot):
    run, config, plan = saved_run
    build_report(run, config=config, plan=plan)
    package = run / "report"
    path = package / "provenance" / f"{snapshot}.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    if snapshot == "project":
        changed["calibration"]["mm_per_px"] = 100.0
    else:
        changed["selected_parameters"]["subset_size_px"] = 23
    _json(path, changed)
    _resign_package_files(package)
    checked = verify_report(package)
    assert not checked["valid"]
    assert any("checksum does not match" in error for error in checked["errors"])
    result = subprocess.run([sys.executable, "-I", str(package / "verify_report.py"), str(package)],
                            capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 1
    assert "checksum does not match" in result.stdout


def test_derived_plan_binding_and_attention_status_preserve_source_dic_gate(saved_run):
    run, config, plan = saved_run
    original_sha = plan["plan_sha256"]
    original_config_sha = _hash_json(config)
    config["regions"][0]["bounds"] = [21, 49, 21, 49]
    plan["source_plan_sha256"] = original_sha
    plan["analysis_kind"] = "saved_field_postprocessing"
    _seal_snapshots(run, config, plan)
    source = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    status = {"schema": "straintrace.postprocessing-quality/v1", "status": "attention_required",
              "region_measurements": {"total": 16, "valid": 0, "unavailable": 16},
              "gauge_measurements": {"total": 6, "valid": 3, "unavailable": 3},
              "source_dic_numerical_quality_passed": True, "source_run_id": run.name}
    source.update(source_configuration_sha256=original_config_sha, postprocessing={"source_plan_sha256": original_sha}, postprocessing_quality=status)
    _json(run / "manifest.json", source)
    _json(run / "quality.json", {"numerical_quality_passed": True, "postprocessing_quality": status, "scientific_acceptance": "not_assessed"})
    receipt = build_report(run, config=config, plan=plan)
    assert receipt["verification"]["valid"], receipt
    data = json.loads(Path(receipt["report_data"]).read_text(encoding="utf-8"))
    html = Path(receipt["report_html"]).read_text(encoding="utf-8")
    assert data["quality"]["numerical_quality_passed"] is True
    assert data["quality"]["postprocessing_quality"] == status
    assert "数值质量 通过" in html and "本次后处理：attention_required" in html
    assert "逐帧 ROI/分量测量行" in html and "<td>16</td>" in html
