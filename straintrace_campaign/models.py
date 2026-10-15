"""Shared data validation primitives for campaign projects.

The campaign package deliberately keeps its persistence boundary independent
of GUI, image, and data-frame libraries. Values crossing this boundary must be
ordinary finite JSON values so a project can be inspected and resumed later.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import PureWindowsPath
from typing import Any, Mapping


class CampaignError(ValueError):
    """An expected, machine-readable campaign failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)

    def as_dict(self) -> dict[str, str]:
        return {"error_code": self.code, "message": self.message}


_PROJECT_KEYS = {
    "schema_version", "kind", "project_id", "name", "roots",
    "acquisitions", "metadata",
}
_ACQUISITION_KEYS = {
    "acquisition_id", "name", "relative_path", "source_kind",
    "physical_specimen_id", "dimensions", "analysis_mode", "image_frames",
    "measurement_frames", "roi", "approval", "analysis_config", "result",
    "alignment", "metadata",
}
_DIMENSION_KEYS = {"area_mm2", "width_mm", "thickness_mm", "status", "source"}


def _bad(code: str, message: str) -> None:
    raise CampaignError(code, message)


def validate_json_value(value: Any, path: str = "$") -> None:
    """Reject values that cannot be represented as strict, finite JSON."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _bad("PROJECT_NONFINITE_NUMBER", f"non-finite number at {path}")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            validate_json_value(item, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _bad("PROJECT_INVALID_JSON_KEY", f"object key at {path} must be a string")
            validate_json_value(item, f"{path}.{key}")
        return
    _bad("PROJECT_INVALID_JSON_VALUE", f"unsupported value {type(value).__name__} at {path}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _bad("PROJECT_SCHEMA_ERROR", f"{label} must be an object")
    return value


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        _bad("PROJECT_SCHEMA_ERROR", f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:  # bool is intentionally rejected.
        _bad("PROJECT_SCHEMA_ERROR", f"{label} must be an integer >= {minimum}")
    return value


def _number_or_null(value: Any, label: str, *, positive: bool = False) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _bad("PROJECT_SCHEMA_ERROR", f"{label} must be a finite number or null")
    if not math.isfinite(float(value)) or (positive and value <= 0):
        _bad("PROJECT_SCHEMA_ERROR", f"{label} must be finite" + (" and > 0" if positive else ""))


def is_absolute_path(value: str) -> bool:
    return os.path.isabs(value) or PureWindowsPath(value).is_absolute()


def validate_relative_path(value: Any, label: str) -> str:
    rel = _string(value, label)
    # Interpret both separators on every platform; a saved project may travel.
    parts = re.split(r"[\\/]", rel)
    windows_path = PureWindowsPath(rel)
    if is_absolute_path(rel) or windows_path.drive or any(part == ".." for part in parts):
        _bad("PROJECT_PATH_ESCAPE", f"{label} must be a safe relative path")
    return rel


def validate_stage_record(value: Any, label: str) -> None:
    if value is None:
        return
    record = _object(value, label)
    required = {"status", "fingerprint", "rows_path", "artifacts", "checksums", "provenance", "summary"}
    missing = sorted(required - set(record))
    if missing:
        _bad("PROJECT_STAGE_SCHEMA_ERROR", f"{label} is missing {missing[0]}")
    _string(record["status"], f"{label}.status")
    _string(record["fingerprint"], f"{label}.fingerprint")
    validate_relative_path(record["rows_path"], f"{label}.rows_path")
    if not isinstance(record["artifacts"], list):
        _bad("PROJECT_STAGE_SCHEMA_ERROR", f"{label}.artifacts must be an array")
    for i, item in enumerate(record["artifacts"]):
        validate_relative_path(item, f"{label}.artifacts[{i}]")
    checksums = _object(record["checksums"], f"{label}.checksums")
    for rel, digest in checksums.items():
        validate_relative_path(rel, f"{label}.checksums key")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            _bad("PROJECT_STAGE_SCHEMA_ERROR", f"{label} contains an invalid SHA-256")
    _object(record["provenance"], f"{label}.provenance")
    _object(record["summary"], f"{label}.summary")
    validate_json_value(record, label)


def validate_acquisition(value: Any) -> dict[str, Any]:
    acq = _object(value, "acquisition")
    unknown = sorted(set(acq) - _ACQUISITION_KEYS)
    if unknown:
        _bad("PROJECT_UNKNOWN_FIELD", f"acquisition has unknown field {unknown[0]!r}; put source annotations in metadata")
    required = _ACQUISITION_KEYS
    missing = sorted(required - set(acq))
    if missing:
        _bad("PROJECT_SCHEMA_ERROR", f"acquisition is missing {missing[0]}")
    _string(acq["acquisition_id"], "acquisition.acquisition_id")
    _string(acq["name"], "acquisition.name")
    validate_relative_path(acq["relative_path"], "acquisition.relative_path")
    _string(acq["source_kind"], "acquisition.source_kind")
    specimen_id = acq["physical_specimen_id"]
    if specimen_id is not None:
        _string(specimen_id, "acquisition.physical_specimen_id")
    if acq["analysis_mode"] not in {"extensometer", "fullfield"}:
        _bad("PROJECT_SCHEMA_ERROR", "acquisition.analysis_mode must be extensometer or fullfield")

    dims = _object(acq["dimensions"], "acquisition.dimensions")
    unknown_dims = sorted(set(dims) - _DIMENSION_KEYS)
    if unknown_dims:
        _bad("PROJECT_UNKNOWN_FIELD", f"dimensions has unknown field {unknown_dims[0]!r}")
    if set(dims) != _DIMENSION_KEYS:
        _bad("PROJECT_SCHEMA_ERROR", "dimensions must contain area_mm2, width_mm, thickness_mm, status, and source")
    for key in ("area_mm2", "width_mm", "thickness_mm"):
        _number_or_null(dims[key], f"dimensions.{key}", positive=True)
    _string(dims["status"], "dimensions.status")
    if dims["source"] is not None:
        _string(dims["source"], "dimensions.source")
    if dims["status"].lower() == "confirmed" and dims["area_mm2"] is not None and not dims["source"]:
        _bad("DIMENSION_SOURCE_REQUIRED", "a confirmed area requires a non-empty source")

    for list_key in ("image_frames", "measurement_frames"):
        frames = acq[list_key]
        if not isinstance(frames, list):
            _bad("PROJECT_SCHEMA_ERROR", f"acquisition.{list_key} must be an array")
        seen: set[int] = set()
        for i, raw in enumerate(frames):
            frame = _object(raw, f"{list_key}[{i}]")
            frame_no_key = "frame"
            n = _integer(frame.get(frame_no_key), f"{list_key}[{i}].frame", minimum=0)
            if n in seen:
                _bad("PROJECT_DUPLICATE_FRAME", f"duplicate {list_key} frame {n}")
            seen.add(n)
            if list_key == "image_frames":
                _integer(frame.get("native_frame"), f"image_frames[{i}].native_frame", minimum=0)
                path = _string(frame.get("path"), f"image_frames[{i}].path")
                if not is_absolute_path(path):
                    _bad("PROJECT_IMAGE_PATH_NOT_ABSOLUTE", "image frame paths must be absolute")
            else:
                _integer(frame.get("native_cbf_frame"), f"measurement_frames[{i}].native_cbf_frame", minimum=0)
                if "scan_id_is_global_fio" not in frame or type(frame["scan_id_is_global_fio"]) is not bool:
                    _bad("PROJECT_SCHEMA_ERROR", f"measurement_frames[{i}].scan_id_is_global_fio must be boolean")
                if "scan_id" not in frame:
                    _bad("PROJECT_SCHEMA_ERROR", f"measurement_frames[{i}] must preserve scan_id explicitly")
            for tkey in ("timestamp_unix_s", "estimated_timestamp_unix_s", "time_s", "exposure_start_unix_s", "exposure_end_unix_s"):
                if tkey in frame:
                    _number_or_null(frame[tkey], f"{list_key}[{i}].{tkey}")
            validate_json_value(frame, f"{list_key}[{i}]")

    if acq["roi"] is not None:
        roi = _object(acq["roi"], "acquisition.roi")
        _string(roi.get("proposal_hash"), "roi.proposal_hash")
        _string(roi.get("reference_sha256"), "roi.reference_sha256")
        if not isinstance(roi.get("free_region"), list) or len(roi["free_region"]) != 4:
            _bad("PROJECT_ROI_SCHEMA_ERROR", "roi.free_region must be [x,y,width,height]")
        for i, coord in enumerate(roi["free_region"]):
            _integer(coord, f"roi.free_region[{i}]", minimum=0)
        if not isinstance(roi.get("groups"), list):
            _bad("PROJECT_ROI_SCHEMA_ERROR", "roi.groups must be an array")
        validate_json_value(roi, "roi")
    if acq["approval"] is not None:
        approval = _object(acq["approval"], "acquisition.approval")
        _string(approval.get("reviewer"), "approval.reviewer")
        _string(approval.get("approved_at"), "approval.approved_at")
        if approval.get("specimen_confirmed") is not True:
            _bad("PROJECT_APPROVAL_INVALID", "approval requires explicit specimen_confirmed=true")
        if acq["roi"] is None or approval.get("proposal_hash") != acq["roi"].get("proposal_hash"):
            _bad("PROJECT_APPROVAL_STALE", "approval does not bind the current ROI proposal")
        if approval.get("reference_sha256") != acq["roi"].get("reference_sha256"):
            _bad("PROJECT_APPROVAL_STALE", "approval does not bind the current reference image")
        validate_json_value(approval, "approval")

    for key in ("analysis_config", "metadata"):
        _object(acq[key], f"acquisition.{key}")
    validate_stage_record(acq["result"], "acquisition.result")
    validate_stage_record(acq["alignment"], "acquisition.alignment")
    validate_json_value(acq, "acquisition")
    return acq


def validate_project(value: Any) -> dict[str, Any]:
    project = _object(value, "project")
    unknown = sorted(set(project) - _PROJECT_KEYS)
    if unknown:
        _bad("PROJECT_UNKNOWN_FIELD", f"project has unknown field {unknown[0]!r}")
    missing = sorted(_PROJECT_KEYS - set(project))
    if missing:
        _bad("PROJECT_SCHEMA_ERROR", f"project is missing {missing[0]}")
    if type(project["schema_version"]) is not int or project["schema_version"] != 1:
        _bad("PROJECT_SCHEMA_VERSION_UNSUPPORTED", "only campaign schema_version 1 is supported")
    if project["kind"] != "straintrace_campaign":
        _bad("PROJECT_KIND_INVALID", "kind must be straintrace_campaign")
    _string(project["project_id"], "project.project_id")
    _string(project["name"], "project.name")
    roots = _object(project["roots"], "project.roots")
    for alias, path in roots.items():
        _string(alias, "root alias")
        _string(path, f"roots.{alias}")
        if not is_absolute_path(path):
            _bad("PROJECT_ROOT_NOT_ABSOLUTE", f"root {alias!r} must be absolute")
    if not isinstance(project["acquisitions"], list):
        _bad("PROJECT_SCHEMA_ERROR", "project.acquisitions must be an array")
    ids: set[str] = set()
    for item in project["acquisitions"]:
        acq = validate_acquisition(item)
        aid = acq["acquisition_id"]
        if aid in ids:
            _bad("PROJECT_DUPLICATE_ACQUISITION_ID", f"duplicate acquisition_id {aid!r}")
        ids.add(aid)
    _object(project["metadata"], "project.metadata")
    validate_json_value(project)
    return project
