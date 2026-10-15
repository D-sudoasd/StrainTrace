from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ezdic_core as core
from straintrace_campaign import roi
from straintrace_campaign.models import CampaignError
from straintrace_campaign.storage import read_jsonl


def _save(path: Path, image: np.ndarray) -> None:
    core.write_image_checked(path, image)


def _vertical_specimen(tmp_path: Path):
    height, width = 320, 240
    rng = np.random.default_rng(4)
    image = np.full((height, width), 12, dtype=np.uint8)
    body = np.clip(165 + rng.normal(0, 35, (280, 82)), 0, 255).astype(np.uint8)
    image[20:300, 79:161] = body
    path = tmp_path / "vertical_ref.png"
    _save(path, image)
    acq = {
        "acquisition_id": "vertical-test",
        "analysis_mode": "extensometer",
        "image_frames": [{"frame": 7, "native_frame": 107, "path": str(path)}],
        "roi": None,
    }
    return acq, image, path


def test_propose_detects_vertical_specimen_and_writes_three_review_previews(tmp_path):
    acquisition, _image, _path = _vertical_specimen(tmp_path)
    proposal = roi.propose(acquisition, tmp_path)

    assert proposal["free_region_status"] == "AUTO_CANDIDATE_REQUIRES_HUMAN_REVIEW"
    orientation = proposal["diagnostics"]["orientation"]["axis"]
    assert abs(orientation[1]) > abs(orientation[0])
    assert 0 < len(proposal["groups"]) <= 5
    assert len(proposal["preview_paths"]) == 3  # one input frame is shown in all three review positions
    assert (tmp_path / proposal["preview_paths"][0]).is_file()
    assert proposal["proposal_hash"] == roi.proposal_hash(proposal)
    assert all(group["actual_mode"] == "distance" for group in proposal["groups"])


def test_manual_free_region_respects_exclusions_and_approval_is_hash_bound(tmp_path):
    acquisition, _image, path = _vertical_specimen(tmp_path)
    acquisition["roi"] = {"exclusions": [[108, 130, 28, 54]]}
    proposal = roi.propose(acquisition, tmp_path, free_region=[65, 5, 110, 310])
    assert proposal["diagnostics"]["manual_free_region_override"] is True
    for group in proposal["groups"]:
        for key in ("roi1", "roi2"):
            x, y, w, h = group[key]
            ex, ey, ew, eh = proposal["exclusions"][0]
            assert x + w <= ex or ex + ew <= x or y + h <= ey or ey + eh <= y

    acquisition["roi"] = proposal
    approval = roi.approve(acquisition, "reviewer-1", specimen_confirmed=True)
    acquisition["approval"] = approval
    roi.validate_approval(acquisition)

    edited = {**proposal, "groups": [dict(proposal["groups"][0], roi1=[0, 0, 15, 15]), *proposal["groups"][1:]]}
    acquisition["roi"] = edited
    with pytest.raises(core.CoreError, match="ROI_APPROVAL_STALE"):
        roi.validate_approval(acquisition)

    # Reference pixels are also part of the approval identity.
    acquisition["roi"] = proposal
    changed = np.zeros((320, 240), dtype=np.uint8)
    _save(path, changed)
    with pytest.raises(core.CoreError, match="ROI_REFERENCE_CHANGED"):
        roi.validate_approval(acquisition)


def test_weak_texture_does_not_create_automatically_approved_gauges(tmp_path):
    flat = np.full((128, 128), 80, dtype=np.uint8)
    path = tmp_path / "flat.png"
    _save(path, flat)
    acquisition = {
        "acquisition_id": "flat-test", "analysis_mode": "extensometer",
        "image_frames": [{"frame": 1, "native_frame": 1, "path": str(path)}], "roi": None,
    }
    proposal = roi.propose(acquisition, tmp_path)
    assert proposal["free_region_status"] == "UNRESOLVED_MANUAL_REVIEW_REQUIRED"
    assert proposal["groups"] == []
    acquisition["roi"] = proposal
    with pytest.raises(core.CoreError, match="ROI_REVIEW_REQUIRED"):
        roi.approve(acquisition, "reviewer", specimen_confirmed=True)


def test_exclusion_geometry_is_rejected_if_user_edits_a_gauge_into_the_mask(tmp_path):
    acquisition, _image, _path = _vertical_specimen(tmp_path)
    acquisition["roi"] = {"exclusions": [[100, 100, 30, 30]]}
    proposal = roi.propose(acquisition, tmp_path, free_region=[65, 5, 110, 310])
    if not proposal["groups"]:
        pytest.skip("synthetic body did not yield an accepted texture window")
    broken = {**proposal, "groups": [dict(proposal["groups"][0], roi1=[105, 105, 15, 15]), *proposal["groups"][1:]]}
    broken["proposal_hash"] = roi.proposal_hash(broken)
    acquisition["roi"] = broken
    acquisition["approval"] = {
        "proposal_hash": broken["proposal_hash"], "reference_sha256": broken["reference_sha256"],
        "reviewer": "reviewer", "approved_at": "2026-01-01T00:00:00Z", "specimen_confirmed": True,
    }
    with pytest.raises(core.CoreError, match="ROI_OVERLAPS_EXCLUSION"):
        roi.validate_approval(acquisition)
