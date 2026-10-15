from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np
import pytest

import ezdic_core as core
from straintrace_campaign import numerics, roi
from straintrace_campaign.delivery import _write_frames_csv
from straintrace_campaign.models import CampaignError
from straintrace_campaign.storage import fingerprint_data, read_jsonl, sha256_file


def _save(path: Path, image: np.ndarray) -> None:
    core.write_image_checked(path, image)


def _campaign(tmp_path: Path, *, mode="extensometer"):
    reference = core.generate_synthetic_speckle(256, 256, seed=29)
    angle = np.deg2rad(1.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = core.warp_image_deformation_gradient(reference, rotation)
    stretched = core.warp_image_deformation_gradient(reference, np.array([[1.04, 0.0], [0.0, 1.0]]))
    paths = [tmp_path / "ref.png", tmp_path / "rotated.png", tmp_path / "stretched.png"]
    for path, image in zip(paths, (reference, rotated, stretched)):
        _save(path, image)
    acquisition = {
        "acquisition_id": "campaign-test",
        "name": "synthetic test",
        "analysis_mode": mode,
        "image_frames": [
            {"frame": 10, "native_frame": 110, "path": str(paths[0]), "time_s": 0.0},
            {"frame": 12, "native_frame": 113, "path": str(paths[1]), "time_s": 2.0},
            {"frame": 15, "native_frame": 117, "path": str(paths[2]), "time_s": 5.0},
        ],
        "analysis_config": ({"tracking": {"search_radius_px": 18, "enable_adaptive": False, "use_prev_frame_template": False}} if mode == "extensometer" else {}),
        "roi": None,
        "approval": None,
    }
    proposal = roi.propose(acquisition, tmp_path, free_region=[20, 20, 216, 216])
    acquisition["roi"] = proposal
    acquisition["approval"] = roi.approve(acquisition, "synthetic-reviewer", specimen_confirmed=True)
    return acquisition, reference, paths


def test_fixed_reference_tracking_retains_gaps_and_recovers_rotation_and_extension(tmp_path):
    acquisition, _reference, _paths = _campaign(tmp_path)
    assert len(acquisition["roi"]["groups"]) >= 3
    result = numerics.analyze(acquisition, tmp_path)
    rows = read_jsonl(tmp_path / result["rows_path"])

    reference_sha256 = sha256_file(_paths[0])
    image_dependencies = numerics._fingerprint_dependencies(acquisition)["image_frames"]
    assert result["provenance"]["source_images_fingerprint"] == numerics._source_images_fingerprint(image_dependencies)
    assert result["provenance"]["reference_sha256"] == reference_sha256
    assert result["provenance"]["reference_fingerprint"] == fingerprint_data("roi_reference_v1", {
        "acquisition_id": acquisition["acquisition_id"],
        "reference_frame": 10,
        "native_frame": 110,
        "reference_sha256": reference_sha256,
    })
    assert [row["frame"] for row in rows] == [10, 12, 15]
    assert [row["native_dic_frame"] for row in rows] == [110, 113, 117]
    assert [row["dic_sequence_index"] for row in rows] == [0, 1, 2]
    assert result["summary"]["frame_count"] == 3
    assert rows[0]["engineering_strain"] == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["engineering_strain"] == pytest.approx(0.0, abs=0.006)
    assert rows[2]["engineering_strain"] == pytest.approx(0.04, abs=0.012)
    true_gauge_values = [gauge["true_strain"] for gauge in rows[2]["gauge_results"] if gauge["accepted"]]
    assert rows[2]["true_strain"] == pytest.approx(np.mean(true_gauge_values), abs=1e-12)
    assert rows[2]["valid_gauge_count"] >= 3
    assert rows[2]["engineering_strain_sd"] >= 0
    assert all(gauge["template_policy"] == "fixed_reference" for gauge in rows[2]["gauge_results"] if gauge["accepted"])
    for gauge in rows[2]["gauge_results"]:
        prefix = gauge["name"]
        assert rows[2][f"{prefix}_engineering_strain"] == gauge["engineering_strain"]
        assert rows[2][f"{prefix}_true_strain"] == gauge["true_strain"]
        assert rows[2][f"{prefix}_status"] == gauge["quality_status"]
        assert rows[2][f"{prefix}_corr_score_roi1"] == gauge["correlation_roi1"]
        assert rows[2][f"{prefix}_fb_error_roi1_px"] == gauge["fb_error_roi1_px"]
    assert any(artifact.endswith("endpoint_trajectories.csv") for artifact in result["artifacts"])
    assert any(artifact.endswith("tracking_qc.csv") for artifact in result["artifacts"])
    for filename in ("endpoint_trajectories.csv", "tracking_qc.csv"):
        path = tmp_path / Path(result["rows_path"]).parent / filename
        assert path.is_file() and path.stat().st_size > 0
    endpoint_file = tmp_path / Path(result["rows_path"]).parent / "endpoint_trajectories.csv"
    with endpoint_file.open(encoding="utf-8-sig", newline="") as stream:
        endpoint_rows = list(csv.DictReader(stream))
    assert len(endpoint_rows) == sum(len(row["gauge_results"]) for row in rows)
    for artifact in result["artifacts"]:
        assert result["checksums"][artifact] == sha256_file(tmp_path / artifact)
    exported_csv = tmp_path / "package_frames.csv"
    _write_frames_csv(exported_csv, rows)
    accepted_prefix = next(gauge["name"] for gauge in rows[2]["gauge_results"] if gauge["accepted"])
    with exported_csv.open(encoding="utf-8-sig", newline="") as stream:
        exported = csv.DictReader(stream)
        assert {
            f"{accepted_prefix}_engineering_strain", f"{accepted_prefix}_true_strain", f"{accepted_prefix}_status",
            f"{accepted_prefix}_corr_score_roi1", f"{accepted_prefix}_fb_error_roi1_px",
        }.issubset(exported.fieldnames)
        exported_rows = list(exported)
    assert exported_rows[2][f"{accepted_prefix}_engineering_strain"] != ""
    assert result["rows_path"].endswith("rows.jsonl")
    raw = (tmp_path / result["rows_path"]).read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw


def test_cancelled_tracking_never_writes_a_complete_rows_file(tmp_path):
    acquisition, _reference, _paths = _campaign(tmp_path)
    events = []

    def progress(event):
        events.append(event)

    with pytest.raises(CampaignError, match="cancelled"):
        numerics.analyze(acquisition, tmp_path, progress=progress, cancel=lambda: bool(events))
    stage_root = tmp_path / "stages" / acquisition["acquisition_id"]
    assert stage_root.exists()
    assert not list(stage_root.rglob("rows.jsonl"))


def test_tracking_failure_stays_missing_and_does_not_fill_a_cycle_frame(tmp_path):
    acquisition, _reference, paths = _campaign(tmp_path)
    _save(paths[1], np.full((256, 256), 90, dtype=np.uint8))
    result = numerics.analyze(acquisition, tmp_path)
    rows = read_jsonl(tmp_path / result["rows_path"])

    assert [row["frame"] for row in rows] == [10, 12, 15]
    failed = rows[1]
    assert failed["valid_gauge_count"] == 0
    assert failed["engineering_strain"] is None
    assert failed["strain_status"] == "NO_VALID_GAUGES"
    assert all(gauge["accepted"] is False for gauge in failed["gauge_results"])


def test_analysis_fingerprint_changes_with_source_bytes_and_review_geometry(tmp_path):
    acquisition, _reference, paths = _campaign(tmp_path)
    initial = numerics.fingerprint(acquisition)
    acquisition["approval"]["reviewer"] = "another reviewer"
    acquisition["approval"]["approved_at"] = "2099-01-01T00:00:00Z"
    acquisition["roi"]["diagnostics"]["review_note"] = "non-numerical review text"
    acquisition["roi"]["preview_paths"] = ["changed-preview.png"]
    acquisition["roi"]["groups"][0].setdefault("texture_diagnostics", []).append({"review_note": "diagnostic only"})
    assert numerics.fingerprint(acquisition) == initial
    image_dependencies = numerics._fingerprint_dependencies(acquisition)["image_frames"]
    source_images = numerics._source_images_fingerprint(image_dependencies)
    original_free_region = list(acquisition["roi"]["free_region"])
    acquisition["roi"]["free_region"][0] += 1
    assert numerics._source_images_fingerprint(numerics._fingerprint_dependencies(acquisition)["image_frames"]) == source_images
    acquisition["roi"]["free_region"] = original_free_region
    acquisition["image_frames"][0]["timestamp_unix_s"] = 1_800_000_000.0
    assert numerics.fingerprint(acquisition) == initial  # time alignment is a separate stage
    source_images_before = numerics._source_images_fingerprint(numerics._fingerprint_dependencies(acquisition)["image_frames"])
    changed = core.warp_image_translation(core.read_gray_image(paths[-1]), 1.0, 0.0)
    _save(paths[-1], changed)
    after_image_change = numerics.fingerprint(acquisition)
    assert after_image_change != initial
    assert numerics._source_images_fingerprint(numerics._fingerprint_dependencies(acquisition)["image_frames"]) != source_images_before
    acquisition["roi"]["groups"][0]["roi1"][0] += 1
    acquisition["roi"]["proposal_hash"] = roi.proposal_hash(acquisition["roi"])
    assert numerics.fingerprint(acquisition) != after_image_change


def test_fullfield_mode_keeps_component_results_separate_from_extensometer_average(tmp_path, monkeypatch):
    acquisition, _reference, _paths = _campaign(tmp_path, mode="fullfield")
    acquisition["roi"]["groups"] = []
    acquisition["roi"]["proposal_hash"] = roi.proposal_hash(acquisition["roi"])
    acquisition["approval"] = roi.approve(acquisition, "synthetic-reviewer", specimen_confirmed=True)
    seen = {}

    def fake_core(settings, progress_callback=None):
        seen.update(settings)
        out_dir = Path(settings["output_dir"]) / "run-test"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact = out_dir / "Exx_map.txt"
        artifact.write_text("field artifact\n", encoding="utf-8")
        if progress_callback:
            progress_callback({"fraction": 1.0, "frame_global_1based": 2, "status": "scientific_valid"})
        return {
            "status": "completed", "scientific_ok": True, "outputs": [str(artifact)],
            "manifest_path": None,
            "frames": [{"frame_global_1based": 2, "status": "scientific_valid", "scientific_valid": True, "artifact_stem": "frame_0002"},
                       {"frame_global_1based": 3, "status": "scientific_valid", "scientific_valid": True, "artifact_stem": "frame_0003"}],
            "json_summary": {"scientific_ok": True},
        }

    monkeypatch.setattr(core, "run_fullfield_sequence", fake_core)
    result = numerics.analyze(acquisition, tmp_path)
    rows = read_jsonl(tmp_path / result["rows_path"])

    assert seen["analysis_mode"] == "fullfield"
    assert list(seen["field_roi"]) == acquisition["roi"]["field_roi"]
    assert len(rows) == 3
    assert all(row["engineering_strain"] is None and row["true_strain"] is None for row in rows)
    assert all(row["strain_status"] == "FIELD_ONLY_NO_ENGINEERING_AGGREGATE" for row in rows)
    assert result["summary"]["valid_mean_frame_count"] == 2
    assert any(path.endswith("Exx_map.txt") for path in result["artifacts"])
