"""Archive the trial considered at preflight and locate its original images."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from .config import CaptureError, QualityRegion
from .related import experiment_hint_paths
from .storage import TRIAL_REFERENCE_SCHEMA
from .trial import trial_report


def trial_locator(report, formal):
    source = {"directory": report["output"]}
    try:
        source["relative_directory"] = Path(os.path.relpath(report["output"], formal)).as_posix()
    except (OSError, ValueError):
        pass
    regions = {camera: {"region": pair["region"]} for camera, row in report["cameras"].items()
               if (pair := (row.get("image_check") or {}).get("region_check")) is not None}
    return {
        "phase": report["trial"]["phase_number"], "experiment_id": report["experiment_id"], "source": source,
        "image_checks": {camera: {"mode": check["mode"], "series_sha256": check["series_sha256"]}
                         for camera, row in report["cameras"].items() if (check := row.get("image_check")) is not None},
        **({"region_checks": regions} if regions else {}),
    }


def archive_trial_reference(preflight, reports, *, image_warnings_confirmed=False, readback_warnings_confirmed=False):
    """Keep the actual preflight comparison, independent of subsequent trial reads."""
    sources = {(report["experiment_id"], report["output"]): report for report in reports}
    warnings = {issue["phase"] for issue in preflight["issues"] if issue["code"] == "TRIAL_IMAGES_NEED_REVIEW"}
    readback_warnings = {issue["phase"] for issue in preflight["issues"] if issue["code"] == "TRIAL_CAMERA_READBACK_CHANGED"}
    stages = []
    for assessment in preflight["trial_assessments"]:
        if not assessment.get("latest_match") or assessment["status"] != "matched":
            continue
        trial = sources[(assessment["experiment_id"], assessment["output"])]
        locator = trial_locator(trial, preflight["output"])
        checks = {key: {**check, "series_sha256": trial["cameras"][key]["image_check"]["series_sha256"]}
                  for key, check in assessment.get("image_checks", {}).items()}
        stages.append({
            **locator,
            "created_utc_ns": trial["created_utc_ns"], "duration_limit_s": assessment["duration_limit_s"],
            "state": trial["verification"]["state"], "committed_images": trial["verification"]["committed_images"],
            "observed": assessment["observed"], "sustained_write_lag": assessment["sustained_write_lag"],
            "image_checks": checks,
            **({"region_checks": regional} if (regional := assessment.get("region_checks")) else {}),
            "image_warnings_confirmed": image_warnings_confirmed and assessment["phase"] in warnings,
            "preflight_camera_readbacks": assessment.get("camera_readbacks", {}),
            "readback_warnings_confirmed": readback_warnings_confirmed and assessment["phase"] in readback_warnings,
            "readbacks": {key: camera.get("setup_readback", {}) for key, camera in trial["cameras"].items()},
            "device_identities": {key: camera.get("device_identity") for key, camera in trial["cameras"].items()},
        })
    return {"schema": TRIAL_REFERENCE_SCHEMA, "stages": stages} if stages else None


def referenced_trial_report(formal, stage, *, directory=None, cancelled=None):
    """Reopen the matching trial and reproduce its recorded image-check coverage."""
    checks = stage["image_checks"]
    full = {key for key, check in checks.items() if check["mode"] == "all"}
    regions = {key: QualityRegion.from_dict(check["region"]) for key, check in stage.get("region_checks", {}).items()}

    def load(folder):
        report = trial_report(folder, expected_experiment_id=stage["experiment_id"],
                              full_image_cameras=full, image_quality_regions=regions, cancelled=cancelled)
        if report["trial"]["phase_number"] != stage["phase"] or any(
            report["cameras"].get(key, {}).get("image_check", {}).get("series_sha256") != check["series_sha256"]
            for key, check in checks.items()
        ):
            raise CaptureError("TRIAL_REFERENCE_CHANGED", str(folder))
        return report

    if directory is not None:
        return load(directory)
    for folder in dict.fromkeys(experiment_hint_paths(Path(formal), stage["source"])):
        try:
            return load(folder)
        except CaptureError as exc:
            if exc.code not in {"TRIAL_REFERENCE_MISMATCH", "TRIAL_REQUIRED", "UNSUPPORTED_SCHEMA", "INCOMPLETE_EXPERIMENT"}:
                raise
        except (OSError, sqlite3.Error):
            continue
    raise CaptureError("TRIAL_REFERENCE_UNAVAILABLE", stage["experiment_id"])
