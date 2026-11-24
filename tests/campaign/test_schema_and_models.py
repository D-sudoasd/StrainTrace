from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from straintrace_campaign import models, storage
from straintrace_campaign.models import CampaignError

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "campaign_config_v1.json"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _valid_project(tmp_path: Path) -> dict:
    img = tmp_path / "dic_000001.tif"
    img.write_bytes(b"img-bytes")
    rows = tmp_path / "stages" / "acq-1" / "analysis" / "rows.jsonl"
    rows.parent.mkdir(parents=True, exist_ok=True)
    rows.write_text('{"frame":1}\n', encoding="utf-8")
    rel_rows = rows.relative_to(tmp_path).as_posix()
    digest = storage.sha256_file(rows)
    img_sha = storage.sha256_file(img)
    roi_doc = {
        "free_region": [0, 0, 100, 60],
        "field_roi": [0, 0, 100, 60],
        "exclusions": [[0, 0, 5, 5]],
        "groups": [{"name": "G01", "roi1": [10, 10, 15, 15], "roi2": [60, 10, 15, 15]}],
        "reference_frame": 1,
        "reference_path": str(img.resolve()),
        "reference_sha256": img_sha,
        "diagnostics": {},
        "preview_paths": [],
    }
    roi_doc["proposal_hash"] = storage.proposal_hash(roi_doc)
    return {
        "schema_version": 1,
        "kind": "straintrace_campaign",
        "project_id": "proj-001",
        "name": "Schema Test",
        "roots": {"raw": str(tmp_path.resolve())},
        "acquisitions": [
            {
                "acquisition_id": "acq-1",
                "relative_path": "batch_a",
                "name": "batch_a",
                "source_kind": "generic_image_sequence",
                "physical_specimen_id": None,
                "dimensions": {
                    "area_mm2": 3.0,
                    "width_mm": 2.0,
                    "thickness_mm": 1.5,
                    "status": "confirmed",
                    "source": "micrometer log",
                },
                "analysis_mode": "extensometer",
                "image_frames": [
                    {
                        "frame": 1,
                        "native_frame": 1,
                        "path": str(img.resolve()),
                        "timestamp_unix_s": 1_700_000_000.0,
                        "timestamp_source": "recorded",
                        "timestamp_status": "RECORDED_ABSOLUTE_EPOCH",
                    }
                ],
                "measurement_frames": [
                    {
                        "frame": 1,
                        "native_cbf_frame": 1,
                        "scan_id": 12,
                        "scan_id_is_global_fio": True,
                        "timestamp_unix_s": 1_700_000_000.0,
                    }
                ],
                "roi": roi_doc,
                "approval": {
                    "proposal_hash": roi_doc["proposal_hash"],
                    "reference_sha256": img_sha,
                    "reviewer": "operator",
                    "approved_at": "2026-10-15T12:05:00Z",
                    "specimen_confirmed": True,
                    "review_status": "APPROVED",
                },
                "analysis_config": {},
                "result": {
                    "status": "COMPLETED",
                    "fingerprint": "a" * 64,
                    "rows_path": rel_rows,
                    "artifacts": [rel_rows],
                    "checksums": {rel_rows: digest},
                    "provenance": {},
                    "summary": {},
                },
                "alignment": None,
                "metadata": {},
            }
        ],
        "metadata": {},
    }


def test_campaign_schema_and_models_accept_valid_project(tmp_path: Path) -> None:
    project = _valid_project(tmp_path)
    jsonschema.Draft202012Validator(_schema()).validate(project)
    models.validate_project(project)


@pytest.mark.parametrize("status", ["confirmed", "verified_recorded_dimensions", "verified_recorded_area"])
@pytest.mark.parametrize(
    "dims_patch",
    [
        {"area_mm2": 3.0, "width_mm": None, "thickness_mm": None, "source": None},
        {"area_mm2": None, "width_mm": 2.0, "thickness_mm": 1.5, "source": ""},
    ],
)
def test_confirmed_dimensions_require_source_in_both_schema_and_models(
    tmp_path: Path, status: str, dims_patch: dict
) -> None:
    project = _valid_project(tmp_path)
    project["acquisitions"][0]["dimensions"] = {**dims_patch, "status": status}
    validator = jsonschema.Draft202012Validator(_schema())
    assert list(validator.iter_errors(project)), "JSON schema must reject confirmed dimensions without source"
    with pytest.raises(CampaignError, match="source"):
        models.validate_project(project)


def test_stage_record_rejects_empty_artifacts_or_missing_checksums(tmp_path: Path) -> None:
    validator = jsonschema.Draft202012Validator(_schema())

    empty_artifacts = _valid_project(tmp_path)
    empty_artifacts["acquisitions"][0]["result"]["artifacts"] = []
    assert list(validator.iter_errors(empty_artifacts))
    with pytest.raises(CampaignError):
        models.validate_project(empty_artifacts)

    missing_checksum = _valid_project(tmp_path)
    missing_checksum["acquisitions"][0]["result"]["artifacts"].append("stages/acq-1/analysis/extra.csv")
    with pytest.raises(CampaignError) as exc:
        models.validate_project(missing_checksum)
    assert exc.value.code == "PROJECT_STAGE_CHECKSUM_MISSING"

    conflicting_slashes = _valid_project(tmp_path)
    rel = conflicting_slashes["acquisitions"][0]["result"]["rows_path"]
    conflicting_slashes["acquisitions"][0]["result"]["checksums"][rel.replace("/", "\\")] = "b" * 64
    with pytest.raises(CampaignError):
        models.validate_project(conflicting_slashes)


def test_models_validate_roi_approval_and_relative_paths(tmp_path: Path) -> None:
    with pytest.raises(CampaignError):
        models.validate_relative_path("\\leading_backslash", "path")
    with pytest.raises(CampaignError):
        models.validate_relative_path("bad\x00name", "path")

    bad_approval = _valid_project(tmp_path)
    bad_approval["acquisitions"][0]["approval"]["extra_field"] = 1
    with pytest.raises(CampaignError) as exc:
        models.validate_project(bad_approval)
    assert exc.value.code == "PROJECT_APPROVAL_INVALID"

    bad_roi = _valid_project(tmp_path)
    bad_roi["acquisitions"][0]["roi"]["free_region"] = [0, 0, 0, 10]
    with pytest.raises(CampaignError):
        models.validate_project(bad_roi)

    bad_scan_id = _valid_project(tmp_path)
    bad_scan_id["acquisitions"][0]["measurement_frames"][0]["scan_id"] = True
    with pytest.raises(CampaignError):
        models.validate_project(bad_scan_id)


def test_storage_finalize_normalizes_slashes_and_reusable_verifies_all_artifacts(tmp_path: Path) -> None:
    rows = tmp_path / "stages" / "acq-1" / "analysis" / "rows.jsonl"
    sidecar = rows.parent / "tracking_qc.csv"
    rows.parent.mkdir(parents=True, exist_ok=True)
    rows.write_text('{"frame":1}\n', encoding="utf-8")
    sidecar.write_text("frame,gauge\n1,G01\n", encoding="utf-8")
    rel_rows_win = str(rows.relative_to(tmp_path)).replace("/", "\\")
    rel_sidecar_win = str(sidecar.relative_to(tmp_path)).replace("/", "\\")

    record = storage.finalize_stage_record(
        tmp_path,
        {
            "status": "COMPLETED",
            "fingerprint": "c" * 64,
            "rows_path": rel_rows_win,
            "artifacts": [rel_rows_win, rel_sidecar_win],
            "checksums": {rel_rows_win: storage.sha256_file(rows), rel_sidecar_win: storage.sha256_file(sidecar)},
            "provenance": {},
            "summary": {},
        },
    )
    assert "\\" not in record["rows_path"]
    assert all("\\" not in item for item in record["artifacts"])
    assert all("\\" not in key for key in record["checksums"])
    assert storage.stage_record_reusable(tmp_path, record, "c" * 64)

    # Corrupting a sidecar artifact invalidates stage reuse even when rows.jsonl is intact.
    sidecar.write_text("tampered\n", encoding="utf-8")
    assert not storage.stage_record_reusable(tmp_path, copy.deepcopy(record), "c" * 64)


def test_artifact_path_normalizes_backslashes_before_path_join(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    real_path = storage.Path

    def spy_path(value: object) -> Path:
        if isinstance(value, str) and value.startswith("versions"):
            seen.append(value)
        return real_path(value)  # type: ignore[arg-type]

    monkeypatch.setattr(storage, "Path", spy_path)
    resolved = storage.artifact_path(tmp_path, "versions\\v001\\catalog.json")
    assert seen == ["versions/v001/catalog.json"]
    assert resolved == (tmp_path / "versions" / "v001" / "catalog.json").resolve()


def test_headless_package_import_does_not_pull_gui_or_vision_deps() -> None:
    script = (
        "import json, sys; "
        "import straintrace_campaign; "
        "import straintrace_campaign._time_io; "
        "import straintrace_campaign.delivery; "
        "blocked = [m for m in ('cv2', 'numpy', 'pandas', 'tkinter', 'PIL', 'ezdic_core') if m in sys.modules]; "
        "print(json.dumps(blocked))"
    )
    proc = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert json.loads(proc.stdout) == []
