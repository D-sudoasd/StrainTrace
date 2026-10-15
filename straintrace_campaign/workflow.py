"""Shared campaign orchestration used by the CLI and desktop workbench."""

from __future__ import annotations

import copy
import csv
import importlib
import io
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .models import CampaignError, is_absolute_path, validate_acquisition, validate_project, validate_relative_path
from .storage import (
    acquire_project_lock,
    artifact_path,
    atomic_write_bytes,
    atomic_write_json,
    finalize_stage_record,
    fingerprint_data,
    project_dir as _project_dir,
    proposal_hash,
    read_jsonl,
    relative_artifact,
    sha256_file,
    stage_output_dir,
    stage_record_reusable,
    strict_json_load,
)


_ACQUISITION_NAMESPACE = uuid.UUID("82b93936-c077-4f93-aeb7-2c34fbd42550")
_MUTABLE_FIELDS = {"dimensions", "physical_specimen_id", "analysis_mode", "analysis_config", "metadata"}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _module(name: str) -> Any:
    try:
        return importlib.import_module(f".{name}", package=__package__)
    except ImportError as exc:
        raise CampaignError("CAMPAIGN_COMPONENT_UNAVAILABLE", f"campaign component {name!r} is unavailable: {exc}") from exc


def _external_call(component: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except CampaignError:
        raise
    except Exception as exc:
        stable_code = getattr(exc, "code", None)
        code = str(stable_code) if stable_code else f"{component.upper()}_FAILED"
        raise CampaignError(code, str(exc)) from exc


def _clone(value: Any) -> Any:
    return copy.deepcopy(value)


def _find_acquisition(project: dict[str, Any], acquisition_id: str) -> dict[str, Any]:
    matches = [a for a in project["acquisitions"] if a["acquisition_id"] == acquisition_id]
    if not matches:
        raise CampaignError("ACQUISITION_NOT_FOUND", f"acquisition_id {acquisition_id!r} was not found")
    return matches[0]


def _read_project(path: str | os.PathLike[str]) -> dict[str, Any]:
    root = _project_dir(path)
    project_file = root / "project.json"
    if not project_file.is_file():
        raise CampaignError("PROJECT_NOT_FOUND", f"project.json was not found in {root}")
    return validate_project(strict_json_load(project_file))


def _write_project(root: Path, project: dict[str, Any]) -> None:
    validate_project(project)
    atomic_write_json(root / "project.json", project, pretty=True)


def _normalize_root(alias: str, raw_path: str | os.PathLike[str]) -> str:
    if not isinstance(alias, str) or not alias.strip():
        raise CampaignError("ROOT_ALIAS_INVALID", "root aliases must be non-empty strings")
    value = os.fspath(raw_path)
    if not isinstance(value, str) or not value.strip() or not is_absolute_path(value):
        raise CampaignError("ROOT_PATH_NOT_ABSOLUTE", f"root {alias!r} must be an absolute path")
    return str(Path(value).expanduser().resolve())


def init_project(
    path: str | os.PathLike[str], name: str, roots: Mapping[str, str | os.PathLike[str]]
) -> dict[str, Any]:
    """Create a new campaign project without touching any source files."""
    if not isinstance(name, str) or not name.strip():
        raise CampaignError("PROJECT_NAME_REQUIRED", "project name must be non-empty")
    if not isinstance(roots, Mapping):
        raise CampaignError("PROJECT_ROOTS_INVALID", "roots must map aliases to absolute paths")
    root = _project_dir(path)
    root.mkdir(parents=True, exist_ok=True)
    project_file = root / "project.json"
    with acquire_project_lock(root):
        if project_file.exists():
            raise CampaignError("PROJECT_ALREADY_EXISTS", f"project already exists: {project_file}")
        normalized_roots = {alias: _normalize_root(alias, value) for alias, value in roots.items()}
        project = {
            "schema_version": 1,
            "kind": "straintrace_campaign",
            "project_id": str(uuid.uuid4()),
            "name": name.strip(),
            "roots": normalized_roots,
            "acquisitions": [],
            "metadata": {"created_at_utc": _now_utc()},
        }
        _write_project(root, project)
    return project


def load_project(path: str | os.PathLike[str]) -> dict[str, Any]:
    return _read_project(path)


def _default_dimensions(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise CampaignError("ACQUISITION_DIMENSIONS_INVALID", "dimensions must be an object")
    dimensions = {
        "area_mm2": value.get("area_mm2"),
        "width_mm": value.get("width_mm"),
        "thickness_mm": value.get("thickness_mm"),
        "status": value.get("status", "unknown"),
        "source": value.get("source"),
    }
    return dimensions


def _normalize_discovered(
    raw: Mapping[str, Any], *, root_alias: str, root: Path, adapter: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CampaignError("DISCOVERY_RECORD_INVALID", "adapter acquisition records must be objects")
    rel_raw = raw.get("relative_path")
    if not isinstance(rel_raw, str) or not rel_raw:
        # Allow common adapter shape that reports an absolute acquisition path.
        candidate = raw.get("path") or raw.get("source_path")
        if not isinstance(candidate, str):
            raise CampaignError("DISCOVERY_RECORD_INVALID", "adapter record needs relative_path or absolute path")
        source = Path(candidate).resolve()
        try:
            rel_raw = source.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise CampaignError("DISCOVERY_OUTSIDE_ROOT", f"discovered path lies outside root {root}: {source}") from exc
    rel = validate_relative_path(rel_raw, "discovered relative_path")
    normalized_rel = "/".join(part for part in rel.replace("\\", "/").split("/") if part not in ("", "."))
    canonical_root = root.resolve().as_posix().casefold()
    identity_seed = f"{root_alias.casefold()}\n{canonical_root}\n{normalized_rel.casefold()}"
    acquisition_id = str(uuid.uuid5(_ACQUISITION_NAMESPACE, identity_seed))
    known = {
        "name", "relative_path", "source_kind", "physical_specimen_id", "dimensions", "analysis_mode",
        "image_frames", "measurement_frames", "roi", "approval", "analysis_config", "result", "alignment", "metadata",
        "path", "source_path", "acquisition_id",
    }
    metadata = dict(raw.get("metadata") or {})
    dimension_extras = {k: _clone(v) for k, v in dict(raw.get("dimensions") or {}).items() if k not in {"area_mm2", "width_mm", "thickness_mm", "status", "source"}}
    if dimension_extras:
        metadata.setdefault("adapter_dimension_source_fields", dimension_extras)
    extras = {k: _clone(v) for k, v in raw.items() if k not in known}
    if extras:
        metadata.setdefault("adapter_source_fields", extras)
    metadata["discovery_root_alias"] = root_alias
    metadata["discovery_root_path"] = canonical_root
    metadata["discovery_adapter"] = adapter
    acq = {
        "acquisition_id": acquisition_id,
        "name": raw.get("name") or Path(normalized_rel).name,
        "relative_path": normalized_rel,
        "source_kind": raw.get("source_kind") or adapter,
        "physical_specimen_id": raw.get("physical_specimen_id"),
        "dimensions": _default_dimensions(raw.get("dimensions")),
        "analysis_mode": raw.get("analysis_mode", "extensometer"),
        "image_frames": _clone(raw.get("image_frames") or []),
        "measurement_frames": _clone(raw.get("measurement_frames") or []),
        "roi": _clone(raw.get("roi")),
        "approval": _clone(raw.get("approval")),
        "analysis_config": _clone(raw.get("analysis_config") or {}),
        "result": _clone(raw.get("result")),
        "alignment": _clone(raw.get("alignment")),
        "metadata": metadata,
    }
    return validate_acquisition(acq)


def scan(
    path: str | os.PathLike[str],
    root_alias: str,
    adapter: str = "generic",
    pattern: str | None = None,
    exclude: tuple[str, ...] | list[str] = (),
) -> list[dict[str, Any]]:
    """Discover acquisitions beneath a configured root and add new identities."""
    if adapter not in {"generic", "desy", "desy_delun", "desy_camera"}:
        raise CampaignError("ADAPTER_UNSUPPORTED", f"unsupported campaign adapter: {adapter}")
    root_project = _project_dir(path)
    with acquire_project_lock(root_project):
        project = _read_project(root_project)
        if root_alias not in project["roots"]:
            raise CampaignError("ROOT_ALIAS_NOT_FOUND", f"root alias {root_alias!r} is not configured")
        root = Path(project["roots"][root_alias])
        if not root.is_dir():
            raise CampaignError("ROOT_NOT_FOUND", f"configured root directory does not exist: {root}")
        adapter_module = _module("adapters")
        discovered = _external_call("discovery", adapter_module.discover,
            root,
            adapter=adapter,
            pattern=pattern,
            exclude=_scan_exclusions(root_project, root, exclude, project.get("metadata", {}).get("output_dirs", [])),
        )
        if not isinstance(discovered, list):
            raise CampaignError("DISCOVERY_RESULT_INVALID", "adapter.discover must return a list")
        existing_keys = {
            (item.get("metadata", {}).get("discovery_root_alias"), item["relative_path"].casefold()): item
            for item in project["acquisitions"]
        }
        added: list[dict[str, Any]] = []
        updated: list[dict[str, Any]] = []
        for raw in discovered:
            acq = _normalize_discovered(raw, root_alias=root_alias, root=root, adapter=adapter)
            key = (root_alias, acq["relative_path"].casefold())
            current = existing_keys.get(key)
            if current is None:
                exact_paths = [item for item in project["acquisitions"] if item["relative_path"].casefold() == acq["relative_path"].casefold()]
                imported_matches = [item for item in exact_paths if item["source_kind"] == "imported_keystone"]
                if len(imported_matches) == 1:
                    current = imported_matches[0]
                elif len(imported_matches) > 1:
                    raise CampaignError("ACQUISITION_IDENTITY_AMBIGUOUS", f"relative path {acq['relative_path']!r} matches multiple imported acquisitions")
            if current is not None:
                if current["source_kind"] == "imported_keystone":
                    # Exact relative-path matches may attach newly discovered
                    # image sources to an imported acquisition. Accepted rows,
                    # measurement identity, dimensions, and acquisition ID stay
                    # untouched.
                    changed = False
                    new_images = _clone(acq["image_frames"])
                    if new_images and _frame_inventory_hash(current["image_frames"]) != _frame_inventory_hash(new_images):
                        image_identity_changed = _image_identity_hash(current["image_frames"]) != _image_identity_hash(new_images)
                        _archive_alignment(current, reason="imported_image_inventory_changed")
                        if image_identity_changed:
                            if current.get("roi") is not None:
                                current.setdefault("metadata", {}).setdefault("stale_roi_proposals", []).append(_clone(current["roi"]))
                            current["roi"] = None
                            current["approval"] = None
                        current["image_frames"] = new_images
                        current.setdefault("metadata", {})["discovery_root_alias"] = root_alias
                        current["metadata"]["discovery_root_path"] = root.resolve().as_posix().casefold()
                        current["metadata"]["discovery_adapter"] = adapter
                        if image_identity_changed:
                            current["metadata"]["workflow_reanalysis_required"] = True
                        current["metadata"]["last_scan_inventory_change"] = {
                            "image_frames_changed": image_identity_changed,
                            "image_frame_metadata_changed": True,
                            "measurement_frames_changed": False,
                            "observed_at_utc": _now_utc(),
                        }
                        changed = True
                    new_measurements = _clone(acq["measurement_frames"])
                    if new_measurements:
                        if current["measurement_frames"] and _frame_inventory_hash(current["measurement_frames"]) != _frame_inventory_hash(new_measurements):
                            candidates = current.setdefault("metadata", {}).setdefault("measurement_frame_axis_candidates", [])
                            candidate_fingerprint = _frame_inventory_hash(new_measurements)
                            if not any(item.get("fingerprint") == candidate_fingerprint for item in candidates):
                                candidates.append({
                                    "status": "UNCONFIRMED_SOURCE_AXIS",
                                    "fingerprint": candidate_fingerprint,
                                    "root_alias": root_alias,
                                    "adapter": adapter,
                                    "observed_at_utc": _now_utc(),
                                    "frames": new_measurements,
                                })
                                current.setdefault("metadata", {}).setdefault("source_conflicts", []).append({
                                    "kind": "measurement_frame_axis_conflict", "root_alias": root_alias,
                                    "candidate_fingerprint": candidate_fingerprint,
                                    "observed_at_utc": _now_utc(), "discovery_adapter": adapter,
                                })
                                changed = True
                        if not current["measurement_frames"]:
                            _archive_alignment(current, reason="measurement_axis_added")
                            current["measurement_frames"] = new_measurements
                            current.setdefault("metadata", {}).setdefault("last_scan_inventory_change", {}).update({"measurement_frames_changed": True})
                            changed = True
                    if changed:
                        current.setdefault("metadata", {})["discovery_root_alias"] = root_alias
                        current["metadata"]["discovery_root_path"] = root.resolve().as_posix().casefold()
                        current["metadata"]["discovery_adapter"] = adapter
                        validate_acquisition(current)
                        updated.append(current)
                    existing_keys[(root_alias, current["relative_path"].casefold())] = current
                    continue
                if current["acquisition_id"] != acq["acquisition_id"]:
                    raise CampaignError("ACQUISITION_IDENTITY_CONFLICT", f"discovery identity changed for {acq['relative_path']}")
                image_inventory_changed = _frame_inventory_hash(current["image_frames"]) != _frame_inventory_hash(acq["image_frames"])
                image_changed = _image_identity_hash(current["image_frames"]) != _image_identity_hash(acq["image_frames"])
                measurement_changed = _frame_inventory_hash(current["measurement_frames"]) != _frame_inventory_hash(acq["measurement_frames"])
                if image_inventory_changed or measurement_changed:
                    if image_inventory_changed:
                        if image_changed and current.get("roi") is not None:
                            current.setdefault("metadata", {}).setdefault("stale_roi_proposals", []).append(_clone(current["roi"]))
                        if image_changed:
                            current["roi"] = None
                            current["approval"] = None
                            if current.get("result") is not None and current["result"]["status"] != "IMPORTED_ACCEPTED":
                                current.setdefault("metadata", {})["workflow_result_stale"] = True
                        _archive_alignment(current, reason="image_inventory_changed")
                        current["image_frames"] = _clone(acq["image_frames"])
                    if measurement_changed:
                        _archive_alignment(current, reason="measurement_axis_changed")
                        current["measurement_frames"] = _clone(acq["measurement_frames"])
                    if image_changed or measurement_changed:
                        current.setdefault("metadata", {})["discovery_root_alias"] = root_alias
                        current["metadata"]["discovery_root_path"] = root.resolve().as_posix().casefold()
                        current["metadata"]["discovery_adapter"] = adapter
                        current["metadata"]["last_scan_inventory_change"] = {
                            "image_frames_changed": image_changed,
                            "image_frame_metadata_changed": image_inventory_changed and not image_changed,
                            "measurement_frames_changed": measurement_changed,
                            "observed_at_utc": _now_utc(),
                        }
                    validate_acquisition(current)
                    if image_inventory_changed or measurement_changed:
                        updated.append(current)
                continue
            if any(old["acquisition_id"] == acq["acquisition_id"] for old in project["acquisitions"]):
                raise CampaignError("ACQUISITION_IDENTITY_CONFLICT", f"stable identity collision for {acq['relative_path']}")
            project["acquisitions"].append(acq)
            existing_keys[key] = acq
            added.append(acq)
        if added or updated:
            project.setdefault("metadata", {}).setdefault("scan_history", []).append(
                {"root_alias": root_alias, "adapter": adapter, "pattern": pattern, "exclude": list(exclude or ()),
                 "added_count": len(added), "updated_count": len(updated), "completed_at_utc": _now_utc()}
            )
            _write_project(root_project, project)
        return _clone(added + updated)


def _scan_exclusions(
    project_root: Path,
    input_root: Path,
    exclude: tuple[str, ...] | list[str],
    output_dirs: list[str] | tuple[str, ...] = (),
) -> tuple[str, ...]:
    values = [str(value) for value in (exclude or ())]
    try:
        relative = project_root.resolve().relative_to(input_root.resolve())
    except ValueError:
        relative = None
    if relative is not None:
        if relative.parts:
            values.append(relative.as_posix().rstrip("/") + "/**")
        else:
            values.extend(("stages", "proposals", "imports", "exports", "output"))
    for raw in output_dirs or ():
        try:
            relative_output = Path(raw).expanduser().resolve().relative_to(input_root.resolve())
        except (OSError, ValueError):
            continue
        if relative_output.parts:
            values.append(relative_output.as_posix().rstrip("/") + "/**")
    return tuple(dict.fromkeys(values))


def _frame_inventory_hash(frames: Any) -> str:
    if not isinstance(frames, list):
        return fingerprint_data("frame_inventory", frames)
    canonical = sorted(
        (_clone(frame) for frame in frames),
        key=lambda item: (item.get("frame", -1), item.get("native_frame", item.get("native_cbf_frame", -1))),
    )
    return fingerprint_data("frame_inventory", canonical)


def _image_identity_hash(frames: Any) -> str:
    if not isinstance(frames, list):
        return fingerprint_data("image_identity", frames)
    keys = ("frame", "native_frame", "path", "content_sha256", "sha256", "frame_namespace", "identity_status", "duplicate_source_paths")
    values = [{key: frame.get(key) for key in keys if key in frame} for frame in frames]
    return _frame_inventory_hash(values)


def import_keystone(path: str | os.PathLike[str], source: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Import accepted campaign data, preserving every supplied acquisition ID."""
    root = _project_dir(path)
    source_path = str(Path(source).expanduser().resolve())
    with acquire_project_lock(root):
        project = _read_project(root)
        version_id, version_digest = _keystone_source_info(source_path)
        prior_imports = project.get("metadata", {}).get("imports", [])
        for prior in prior_imports:
            if prior.get("kind") != "keystone" or prior.get("source_version_id") != version_id:
                continue
            if prior.get("source_version_sha256") != version_digest:
                raise CampaignError("IMPORT_SOURCE_VERSION_CHANGED", f"Keystone version {version_id!r} changed after import; accepted rows remain immutable")
            return _clone([
                item for item in project["acquisitions"]
                if item.get("metadata", {}).get("keystone_import", {}).get("source_version_id") == version_id
            ])
        delivery = _module("delivery")
        imported = _external_call("keystone_import", delivery.import_keystone, root, source_path)
        if not isinstance(imported, list):
            raise CampaignError("IMPORT_RESULT_INVALID", "delivery.import_keystone must return a list")
        known_by_id = {a["acquisition_id"]: a for a in project["acquisitions"]}
        appended: list[dict[str, Any]] = []
        for raw in imported:
            raw_acq = _clone(dict(raw))
            extra_dims = {k: _clone(v) for k, v in dict(raw_acq.get("dimensions") or {}).items() if k not in {"area_mm2", "width_mm", "thickness_mm", "status", "source"}}
            if extra_dims:
                raw_acq.setdefault("metadata", {}).setdefault("keystone_import", {})["dimension_source_fields"] = extra_dims
                raw_acq["dimensions"] = {k: v for k, v in raw_acq["dimensions"].items() if k in {"area_mm2", "width_mm", "thickness_mm", "status", "source"}}
            acq = validate_acquisition(raw_acq)
            imported_digest = fingerprint_data("keystone_acquisition", acq)
            # Imported acquisition IDs are source identities and are never replaced.
            if acq["acquisition_id"] in known_by_id:
                old = known_by_id[acq["acquisition_id"]]
                previous_digest = old.get("metadata", {}).get("imported_keystone_digest")
                if previous_digest == imported_digest:
                    continue
                raise CampaignError("IMPORTED_IDENTITY_CONFLICT", f"imported acquisition {acq['acquisition_id']} conflicts with the project")
            acq.setdefault("metadata", {})["imported_keystone_digest"] = imported_digest
            if acq["result"] is not None:
                record = dict(acq["result"])
                if record.get("status") != "IMPORTED_ACCEPTED":
                    record["provenance"] = dict(record.get("provenance") or {})
                    record["provenance"].setdefault("import_source", source_path)
                    record["provenance"]["imported_status_before_tag"] = record.get("status")
                    record["status"] = "IMPORTED_ACCEPTED"
                finalized = finalize_stage_record(root, record)
                rows = read_jsonl(artifact_path(root, finalized["rows_path"]))
                _validate_imported_rows(rows, acq)
                acq["result"] = finalized
            project["acquisitions"].append(acq)
            known_by_id[acq["acquisition_id"]] = acq
            appended.append(acq)
        if appended:
            project.setdefault("metadata", {}).setdefault("imports", []).append(
                {"kind": "keystone", "source": source_path, "source_version_id": version_id,
                 "source_version_sha256": version_digest, "acquisition_count": len(appended), "imported_at_utc": _now_utc()}
            )
            _write_project(root, project)
        return _clone(appended)


def _keystone_source_info(source: str | os.PathLike[str]) -> tuple[str, str]:
    """Resolve the immutable version identity used by delivery.import_keystone."""
    candidate = Path(source).expanduser().resolve()
    pointer_root: Path | None = None
    if candidate.is_file() and candidate.name.lower() == "current.json":
        pointer_root = candidate.parent.resolve()
        current = strict_json_load(candidate)
        relative = validate_relative_path(current.get("version_relative_path", current.get("version_path")), "CURRENT.version path")
        version = (pointer_root / Path(relative)).resolve()
    elif candidate.is_dir() and (candidate / "current.json").is_file():
        pointer_root = candidate.resolve()
        current = strict_json_load(candidate / "current.json")
        relative = validate_relative_path(current.get("version_relative_path", current.get("version_path")), "CURRENT.version path")
        version = (pointer_root / Path(relative)).resolve()
    else:
        version = candidate
    if pointer_root is not None:
        try:
            version.relative_to(pointer_root)
        except ValueError as exc:
            raise CampaignError("IMPORT_SOURCE_INVALID", "CURRENT.version_path escapes its source root") from exc
    if not version.is_dir() or not (version / "catalog.json").is_file():
        raise CampaignError("IMPORT_SOURCE_INVALID", f"not a Keystone version directory: {version}")
    checksum_file = version / "CHECKSUMS.json"
    pin = checksum_file if checksum_file.is_file() else version / "catalog.json"
    return version.name, sha256_file(pin)


def _validate_imported_rows(rows: list[dict[str, Any]], acq: Mapping[str, Any]) -> None:
    seen: set[tuple[str, Any]] = set()
    for i, row in enumerate(rows):
        if "frame" not in row:
            raise CampaignError("IMPORTED_ROWS_INVALID", f"imported row {i} has no explicit frame number")
        frame = row["frame"]
        if type(frame) is not int:
            raise CampaignError("IMPORTED_ROWS_INVALID", f"imported row {i} frame must be an integer")
        key = ("frame", frame)
        if key in seen:
            raise CampaignError("IMPORTED_ROWS_DUPLICATE_FRAME", f"duplicate imported frame {frame}")
        seen.add(key)


def compute_roi_proposal_hash(proposal: Mapping[str, Any]) -> str:
    try:
        roi = _module("roi")
        helper = getattr(roi, "proposal_hash", None)
        if callable(helper):
            return str(helper(proposal))
    except CampaignError:
        pass
    return proposal_hash(proposal)


def _reference_frame(acq: Mapping[str, Any], proposal: Mapping[str, Any]) -> Mapping[str, Any]:
    frame_number = proposal.get("reference_frame")
    if type(frame_number) is not int:
        raise CampaignError("ROI_REFERENCE_FRAME_INVALID", "ROI proposal reference_frame must be an integer")
    matching = [f for f in acq["image_frames"] if f.get("frame") == frame_number]
    if not matching:
        matching = [f for f in acq["image_frames"] if f.get("native_frame") == frame_number]
    if len(matching) != 1:
        raise CampaignError("ROI_REFERENCE_FRAME_INVALID", f"reference frame {frame_number} is absent or ambiguous")
    return matching[0]


def _persist_roi(project: dict[str, Any], acq: dict[str, Any], proposal: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if not isinstance(proposal, Mapping):
        raise CampaignError("ROI_PROPOSAL_INVALID", "ROI proposal must be an object")
    candidate = _clone(dict(proposal))
    candidate["proposal_hash"] = compute_roi_proposal_hash(candidate)
    reference = _reference_frame(acq, candidate)
    ref_path = candidate.get("reference_path")
    if not isinstance(ref_path, str) or Path(ref_path).resolve() != Path(reference["path"]).resolve():
        raise CampaignError("ROI_REFERENCE_PATH_MISMATCH", "proposal reference_path must identify its explicit image frame")
    image_path = Path(reference["path"])
    if not image_path.is_file():
        raise CampaignError("ROI_REFERENCE_IMAGE_MISSING", f"reference image does not exist: {image_path}")
    actual_sha = sha256_file(image_path)
    if candidate.get("reference_sha256") != actual_sha:
        raise CampaignError("ROI_REFERENCE_CHANGED", "reference image SHA-256 differs from ROI proposal")
    if "free_region" not in candidate or not isinstance(candidate["free_region"], list) or len(candidate["free_region"]) != 4:
        raise CampaignError("ROI_PROPOSAL_INVALID", "proposal needs a free_region [x,y,width,height]")
    for idx, number in enumerate(candidate["free_region"]):
        if type(number) is not int or number < 0 or (idx >= 2 and number <= 0):
            raise CampaignError("ROI_PROPOSAL_INVALID", "free_region must be integer [x,y,width,height] with positive size")
    if not isinstance(candidate.get("groups"), list):
        raise CampaignError("ROI_PROPOSAL_INVALID", "proposal groups must be an array")
    roi_module = _module("roi")
    validate_proposal = getattr(roi_module, "validate_proposal", None)
    if callable(validate_proposal):
        _external_call("roi", validate_proposal, acq, candidate, root)
    old = acq.get("roi")
    old_geometry = _roi_tracking_geometry_for(acq, old)
    new_geometry = _roi_tracking_geometry_for(acq, candidate)
    tracking_geometry_changed = old_geometry != new_geometry
    if old is None or old.get("proposal_hash") != candidate["proposal_hash"]:
        if tracking_geometry_changed:
            _capture_roi_sensitivity_baseline(acq, old_geometry, root)
            _archive_alignment(acq, reason="roi_proposal_changed")
        result = acq.get("result") or {}
        if result.get("status") == "IMPORTED_ACCEPTED":
            if tracking_geometry_changed:
                acq.setdefault("metadata", {})["workflow_reanalysis_required"] = True
        elif result and tracking_geometry_changed:
            acq.setdefault("metadata", {})["workflow_result_stale"] = True
    acq["roi"] = candidate
    acq["approval"] = None
    validate_acquisition(acq)
    return candidate


def _capture_roi_sensitivity_baseline(
    acq: dict[str, Any], geometry: Mapping[str, Any] | None, root: Path
) -> None:
    """Pin the first completed generic result across one or more ROI retries."""
    result = acq.get("result")
    if not isinstance(result, Mapping) or result.get("status") == "IMPORTED_ACCEPTED":
        return
    if result.get("status") not in {"COMPLETE", "COMPLETED"}:
        return
    metadata = acq.setdefault("metadata", {})
    existing = metadata.get("pending_roi_sensitivity")
    if isinstance(existing, Mapping) and isinstance(existing.get("baseline_result"), Mapping):
        return
    record = {
        "baseline_result": _clone(dict(result)),
        "baseline_roi_geometry": _clone(dict(geometry)) if isinstance(geometry, Mapping) else None,
        "baseline_reference_fingerprint": _reference_fingerprint_from_provenance(acq, result),
        "captured_at_utc": _now_utc(),
    }
    # A missing fingerprint is deliberate evidence. It must not be reconstructed
    # from the currently edited ROI because that would relabel old rows.
    metadata["pending_roi_sensitivity"] = record


def _reference_fingerprint_from_provenance(
    acq: Mapping[str, Any], result: Mapping[str, Any]
) -> str | None:
    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    fingerprint = provenance.get("reference_fingerprint")
    if isinstance(fingerprint, str) and len(fingerprint) == 64 and all(ch in "0123456789abcdef" for ch in fingerprint):
        return fingerprint
    # Older current-code records may carry all inputs in provenance without the
    # precomputed key. Derive only from those recorded fields, never from today's ROI.
    sha = provenance.get("reference_sha256")
    frame = provenance.get("reference_frame")
    native = provenance.get("reference_native_dic_frame")
    if (
        isinstance(sha, str) and len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha)
        and type(frame) is int and type(native) is int
    ):
        return fingerprint_data("roi_reference_v1", {
            "acquisition_id": acq.get("acquisition_id"),
            "reference_frame": frame,
            "native_frame": native,
            "reference_sha256": sha,
        })
    return None


def _archive_result(acq: dict[str, Any], *, reason: str) -> None:
    record = acq.get("result")
    if record is None:
        return
    metadata = acq.setdefault("metadata", {})
    history = metadata.setdefault("workflow_stage_history", [])
    history.append({"stage": "result", "reason": reason, "archived_at_utc": _now_utc(), "record": _clone(record)})
    if record.get("status") == "IMPORTED_ACCEPTED":
        metadata.setdefault("imported_accepted_results", []).append(_clone(record))


def _archive_alignment(acq: dict[str, Any], *, reason: str) -> None:
    record = acq.get("alignment")
    if record is None:
        return
    metadata = acq.setdefault("metadata", {})
    metadata.setdefault("workflow_stage_history", []).append({
        "stage": "alignment", "reason": reason, "archived_at_utc": _now_utc(), "record": _clone(record),
    })
    metadata["alignment_recalculation_required"] = True
    acq["alignment"] = None


def update_roi(path: str | os.PathLike[str], acquisition_id: str, proposal: Mapping[str, Any]) -> dict[str, Any]:
    root = _project_dir(path)
    with acquire_project_lock(root):
        project = _read_project(root)
        acq = _find_acquisition(project, acquisition_id)
        saved = _persist_roi(project, acq, proposal, root)
        _write_project(root, project)
        return _clone(saved)


def propose_rois(
    path: str | os.PathLike[str], acquisition_id: str, free_region: list[int] | None = None
) -> dict[str, Any]:
    root = _project_dir(path)
    with acquire_project_lock(root):
        project = _read_project(root)
        acq = _find_acquisition(project, acquisition_id)
        roi_module = _module("roi")
        proposal = _external_call("roi", roi_module.propose, acq, root, free_region=free_region)
        saved = _persist_roi(project, acq, proposal, root)
        _write_project(root, project)
        return _clone(saved)


def approve_rois(
    path: str | os.PathLike[str],
    acquisition_id: str,
    reviewer: str,
    specimen_confirmed: bool,
) -> dict[str, Any]:
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise CampaignError("ROI_REVIEWER_REQUIRED", "reviewer must be a non-empty name or identifier")
    if specimen_confirmed is not True:
        raise CampaignError("GATE_SPECIMEN_UNCONFIRMED", "approval requires explicit confirmation that the reviewed region belongs to the specimen")
    root = _project_dir(path)
    with acquire_project_lock(root):
        project = _read_project(root)
        acq = _find_acquisition(project, acquisition_id)
        if acq.get("roi") is None:
            raise CampaignError("ROI_PROPOSAL_REQUIRED", "propose and review the ROI before approving it")
        reference = _reference_frame(acq, acq["roi"])
        if not Path(reference["path"]).is_file() or sha256_file(reference["path"]) != acq["roi"].get("reference_sha256"):
            raise CampaignError("ROI_REFERENCE_CHANGED", "ROI reference image is missing or changed; propose and review again")
        roi_module = _module("roi")
        approval = _external_call("roi", roi_module.approve, acq, reviewer.strip(), specimen_confirmed=True)
        if not isinstance(approval, Mapping):
            raise CampaignError("ROI_APPROVAL_INVALID", "roi.approve must return an approval record")
        approval = dict(approval)
        approval.setdefault("proposal_hash", acq["roi"]["proposal_hash"])
        approval.setdefault("reference_sha256", acq["roi"]["reference_sha256"])
        approval.setdefault("reviewer", reviewer.strip())
        approval.setdefault("approved_at", _now_utc())
        approval["specimen_confirmed"] = True
        acq["approval"] = approval
        _external_call("roi", roi_module.validate_approval, acq)
        validate_acquisition(acq)
        _write_project(root, project)
        return _clone(approval)


def update_acquisition(
    path: str | os.PathLike[str], acquisition_id: str, patch: Mapping[str, Any]
) -> dict[str, Any]:
    """Update explicit identity/geometry/configuration fields safely.

    Source identity, frame mappings, and result records cannot be patched. An
    imported accepted result is copied to project history before a changed
    analysis configuration makes it inapplicable to a new run.
    """
    if not isinstance(patch, Mapping):
        raise CampaignError("ACQUISITION_PATCH_INVALID", "patch must be an object")
    unknown = sorted(set(patch) - _MUTABLE_FIELDS)
    if unknown:
        raise CampaignError("ACQUISITION_FIELD_IMMUTABLE", f"field {unknown[0]!r} cannot be updated through this API")
    root = _project_dir(path)
    with acquire_project_lock(root):
        project = _read_project(root)
        acq = _find_acquisition(project, acquisition_id)
        analysis_changed = False
        if "dimensions" in patch:
            dimensions = _default_dimensions(patch["dimensions"])
            # A candidate size may be recorded, but only confirmed dimensions
            # with an explicit locator/source can be used as a formal area.
            if dimensions["status"].lower() == "confirmed" and dimensions["area_mm2"] is not None and not dimensions["source"]:
                raise CampaignError("DIMENSION_SOURCE_REQUIRED", "confirmed area requires a source or record locator")
            if dimensions != acq["dimensions"]:
                _archive_alignment(acq, reason="dimensions_changed")
                acq.setdefault("metadata", {})["alignment_recalculation_required"] = True
            acq["dimensions"] = dimensions
        if "physical_specimen_id" in patch:
            specimen = patch["physical_specimen_id"]
            if specimen is not None and (not isinstance(specimen, str) or not specimen.strip()):
                raise CampaignError("ACQUISITION_PATCH_INVALID", "physical_specimen_id must be null or a non-empty string")
            acq["physical_specimen_id"] = specimen
        if "analysis_mode" in patch:
            mode = patch["analysis_mode"]
            if mode not in {"extensometer", "fullfield"}:
                raise CampaignError("ACQUISITION_PATCH_INVALID", "analysis_mode must be extensometer or fullfield")
            analysis_changed |= mode != acq["analysis_mode"]
            acq["analysis_mode"] = mode
        if "analysis_config" in patch:
            config = patch["analysis_config"]
            if not isinstance(config, Mapping):
                raise CampaignError("ACQUISITION_PATCH_INVALID", "analysis_config must be an object")
            analysis_changed |= dict(config) != acq["analysis_config"]
            acq["analysis_config"] = _clone(dict(config))
        if "metadata" in patch:
            metadata = patch["metadata"]
            if not isinstance(metadata, Mapping):
                raise CampaignError("ACQUISITION_PATCH_INVALID", "metadata patch must be an object")
            if set(metadata) - {"user_annotations", "dimension_candidates", "identity_candidates"}:
                raise CampaignError("ACQUISITION_FIELD_IMMUTABLE", "only user_annotations, dimension_candidates, and identity_candidates may be changed; source and workflow metadata are protected")
            if "user_annotations" in metadata:
                annotations = metadata["user_annotations"]
                if not isinstance(annotations, Mapping):
                    raise CampaignError("ACQUISITION_PATCH_INVALID", "metadata.user_annotations must be an object")
                acq["metadata"].setdefault("user_annotations", {}).update(_clone(dict(annotations)))
            for candidate_key in ("dimension_candidates", "identity_candidates"):
                if candidate_key in metadata:
                    candidates = metadata[candidate_key]
                    if not isinstance(candidates, list) or any(not isinstance(item, Mapping) for item in candidates):
                        raise CampaignError("ACQUISITION_PATCH_INVALID", f"metadata.{candidate_key} must be an array of records")
                    acq["metadata"][candidate_key] = _clone([dict(item) for item in candidates])
        if analysis_changed and (acq.get("result") or {}).get("status") == "IMPORTED_ACCEPTED":
            acq.setdefault("metadata", {})["workflow_reanalysis_required"] = True
        elif analysis_changed and acq.get("result") is not None:
            acq.setdefault("metadata", {})["workflow_result_stale"] = True
        if analysis_changed:
            _archive_alignment(acq, reason="analysis_configuration_changed")
        validate_acquisition(acq)
        _write_project(root, project)
        return _clone(acq)


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancel)


def _check_cancel(cancel: Any) -> None:
    if _cancelled(cancel):
        raise CampaignError("CAMPAIGN_CANCELLED", "campaign operation was cancelled")


def _notify(progress: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    if progress is not None:
        progress(event)


def _expected_analysis_fingerprint(numerics: Any, acq: Mapping[str, Any]) -> str:
    fn = getattr(numerics, "fingerprint", None)
    if not callable(fn):
        raise CampaignError("NUMERICS_INTERFACE_INVALID", "numerics.fingerprint(acquisition) is required")
    module_fingerprint = fn(acq)
    return fingerprint_data("analysis", {
        "numerics_fingerprint": module_fingerprint,
        "numerics_code_sha256": _module_code_sha256(numerics),
        "analysis_mode": acq["analysis_mode"],
        "analysis_config": acq["analysis_config"],
        "roi_tracking_geometry": _roi_tracking_geometry(acq),
    })


def _module_code_sha256(module: Any) -> str:
    module_name = getattr(module, "__name__", None)
    if not isinstance(module_name, str) or not module_name:
        spec = getattr(module, "__spec__", None)
        module_name = getattr(spec, "name", None)
    if not isinstance(module_name, str) or not module_name:
        for attr in ("fingerprint", "analyze", "align"):
            function = getattr(module, attr, None)
            function_module = getattr(function, "__module__", None)
            if isinstance(function_module, str) and function_module:
                module_name = function_module
                break

    relative_source = Path(*module_name.split(".")).with_suffix(".py") if isinstance(module_name, str) and module_name else None
    executable_dir = Path(sys.executable).resolve().parent
    frozen = bool(getattr(sys, "frozen", False) or getattr(sys, "_MEIPASS", None))
    candidate_paths: list[Path] = []
    if frozen and relative_source is not None:
        bundle_roots = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bundle_roots.append(Path(meipass) / "sources")
        bundle_roots.extend((executable_dir / "_internal" / "sources", executable_dir / "sources"))
        candidate_paths.extend(base / relative_source for base in bundle_roots)

    if not frozen:
        # Editable installs and synthetic tests can identify the defining
        # module through __file__ or a callable's source location.
        source = getattr(module, "__file__", None)
        if isinstance(source, str) and Path(source).suffix.lower() == ".py":
            candidate_paths.append(Path(source))
        if relative_source is not None:
            candidate_paths.append(Path(__file__).resolve().parent.parent / relative_source)
        for attr in ("fingerprint", "analyze", "align"):
            function = getattr(module, attr, None)
            code = getattr(function, "__code__", None)
            filename = getattr(code, "co_filename", None)
            if isinstance(filename, str) and Path(filename).suffix.lower() == ".py":
                candidate_paths.append(Path(filename))

    seen: set[str] = set()
    for candidate in candidate_paths:
        try:
            canonical = candidate.resolve()
            key = str(canonical).casefold()
            if key in seen:
                continue
            seen.add(key)
            if canonical.is_file():
                return sha256_file(canonical)
        except (OSError, TypeError, ValueError):
            continue
    raise CampaignError(
        "CAMPAIGN_CODE_FINGERPRINT_UNAVAILABLE",
        f"cannot resolve an auditable source copy for campaign module {getattr(module, '__name__', type(module).__name__)}",
    )


def _roi_tracking_geometry(acq: Mapping[str, Any]) -> dict[str, Any] | None:
    return _roi_tracking_geometry_for(acq, acq.get("roi"))


def _roi_tracking_geometry_for(
    acq: Mapping[str, Any], proposal: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if not isinstance(proposal, Mapping):
        return None
    reference = _reference_frame(acq, proposal)
    groups = proposal.get("groups", [])
    normalized_groups = []
    if isinstance(groups, list):
        for group in groups:
            if isinstance(group, Mapping):
                normalized_groups.append({
                    key: _clone(group[key])
                    for key in ("name", "roi1", "roi2", "actual_mode", "selected_mode", "role")
                    if key in group
                })
            else:
                normalized_groups.append(_clone(group))
    return {
        "reference_frame": proposal.get("reference_frame"),
        "reference_native_dic_frame": reference.get("native_frame"),
        "reference_sha256": proposal.get("reference_sha256"),
        "image_shape": proposal.get("image_shape"),
        "free_region": proposal.get("free_region"),
        "field_roi": proposal.get("field_roi"),
        "exclusions": proposal.get("exclusions", []),
        "groups": normalized_groups,
    }


def _roi_reference_fingerprint(acq: Mapping[str, Any], proposal: Mapping[str, Any] | None) -> str | None:
    if not isinstance(proposal, Mapping):
        return None
    reference = _reference_frame(acq, proposal)
    return fingerprint_data("roi_reference_v1", {
        "acquisition_id": acq.get("acquisition_id"),
        "reference_frame": proposal.get("reference_frame"),
        "native_frame": reference.get("native_frame"),
        "reference_sha256": proposal.get("reference_sha256"),
    })


_SENSITIVITY_CSV_FIELDS = [
    "frame", "old_native_dic_frame", "new_native_dic_frame", "join_status",
    "old_valid_gauge_count", "new_valid_gauge_count",
    "old_strain_status", "new_strain_status",
    "old_engineering_strain", "new_engineering_strain", "delta_engineering_strain",
    "old_true_strain", "new_true_strain", "delta_true_strain",
    "comparison_status",
]


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _explicit_row_index(rows: list[dict[str, Any]]) -> tuple[dict[tuple[int, int], dict[str, Any]], set[tuple[int, int]], int]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates: set[tuple[int, int]] = set()
    malformed = 0
    for row in rows:
        frame = row.get("frame")
        native = row.get("native_dic_frame")
        if type(frame) is not int or type(native) is not int:
            malformed += 1
            continue
        key = (frame, native)
        if key in indexed:
            indexed.pop(key, None)
            duplicates.add(key)
        elif key not in duplicates:
            indexed[key] = row
    return indexed, duplicates, malformed


def _row_strain(row: Mapping[str, Any] | None, field: str) -> float | None:
    if row is None:
        return None
    count = row.get("valid_gauge_count")
    status = row.get("strain_status")
    if type(count) is int and count < 3:
        return None
    if status is not None and status != "TRACKED_MEAN":
        return None
    return _finite_float(row.get(field))


def _delta_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_signed": None, "mean_absolute": None, "median_absolute": None, "max_absolute": None, "rmse": None}
    absolute = sorted(abs(value) for value in values)
    middle = len(absolute) // 2
    median_absolute = absolute[middle] if len(absolute) % 2 else (absolute[middle - 1] + absolute[middle]) / 2
    return {
        "count": len(values),
        "mean_signed": sum(values) / len(values),
        "mean_absolute": sum(abs(value) for value in values) / len(values),
        "median_absolute": median_absolute,
        "max_absolute": max(absolute),
        "rmse": (sum(value * value for value in values) / len(values)) ** 0.5,
    }


def _comparison_context_status(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any],
    baseline_reference: str | None, candidate_reference: str | None,
) -> str:
    if baseline_reference is None:
        return "BASELINE_REFERENCE_PROVENANCE_MISSING"
    if candidate_reference is None:
        return "CANDIDATE_REFERENCE_PROVENANCE_MISSING"
    if baseline_reference != candidate_reference:
        return "REFERENCE_MISMATCH"
    old_provenance = baseline.get("provenance")
    new_provenance = candidate.get("provenance")
    if not isinstance(old_provenance, Mapping) or not isinstance(new_provenance, Mapping):
        return "ANALYSIS_CONTEXT_PROVENANCE_MISSING"
    old_images = old_provenance.get("source_images_fingerprint")
    new_images = new_provenance.get("source_images_fingerprint")
    if not all(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value) for value in (old_images, new_images)):
        return "SOURCE_IMAGES_PROVENANCE_MISSING"
    if old_images != new_images:
        return "INPUT_IMAGES_MISMATCH"
    for field in ("analysis_mode", "analysis_config", "engine_code_sha256"):
        if field not in old_provenance or field not in new_provenance:
            return "ANALYSIS_CONTEXT_PROVENANCE_MISSING"
        if old_provenance[field] != new_provenance[field]:
            return "ANALYSIS_CONTEXT_MISMATCH"
    return "COMPARABLE"


def _read_result_rows(root: Path, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rel = result.get("rows_path")
    if not isinstance(rel, str):
        raise CampaignError("ROI_SENSITIVITY_ROWS_INVALID", "result has no rows_path")
    target = artifact_path(root, rel)
    if not target.is_file():
        raise CampaignError("ROI_SENSITIVITY_ROWS_INVALID", f"result rows are missing: {rel}")
    expected = (result.get("checksums") or {}).get(rel)
    if not isinstance(expected, str) or sha256_file(target) != expected:
        raise CampaignError("ROI_SENSITIVITY_ROWS_INVALID", f"result rows checksum is absent or invalid: {rel}")
    return read_jsonl(target)


def _write_roi_sensitivity_comparison(
    root: Path,
    acq: Mapping[str, Any],
    pending: Mapping[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline = pending.get("baseline_result")
    if not isinstance(baseline, Mapping) or baseline.get("status") == "IMPORTED_ACCEPTED":
        raise CampaignError("ROI_SENSITIVITY_BASELINE_INVALID", "sensitivity baseline must be a generic campaign result")
    baseline_reference = pending.get("baseline_reference_fingerprint")
    candidate_reference = _reference_fingerprint_from_provenance(acq, candidate)
    comparison_status = _comparison_context_status(baseline, candidate, baseline_reference, candidate_reference)

    baseline_rows = _read_result_rows(root, baseline)
    candidate_rows = _read_result_rows(root, candidate)
    old_index, old_duplicates, old_malformed = _explicit_row_index(baseline_rows)
    new_index, new_duplicates, new_malformed = _explicit_row_index(candidate_rows)
    duplicate_keys = old_duplicates | new_duplicates
    if old_malformed or new_malformed or duplicate_keys:
        comparison_status = "FRAME_ID_JOIN_INVALID"

    union = sorted(set(old_index) | set(new_index) | duplicate_keys)
    frames: list[dict[str, Any]] = []
    engineering_deltas: list[float] = []
    true_deltas: list[float] = []
    overlap = 0
    for frame_id, native_id in union:
        key = (frame_id, native_id)
        old_row = old_index.get(key)
        new_row = new_index.get(key)
        if key in duplicate_keys:
            join_status = "DUPLICATE_IDENTITY"
        elif old_row is not None and new_row is not None:
            join_status = "MATCHED"
            overlap += 1
        elif old_row is not None:
            join_status = "BASELINE_ONLY"
        else:
            join_status = "CANDIDATE_ONLY"

        old_eng = _row_strain(old_row, "engineering_strain")
        new_eng = _row_strain(new_row, "engineering_strain")
        old_true = _row_strain(old_row, "true_strain")
        new_true = _row_strain(new_row, "true_strain")
        can_compare = comparison_status == "COMPARABLE" and join_status == "MATCHED"
        delta_eng = new_eng - old_eng if can_compare and old_eng is not None and new_eng is not None else None
        delta_true = new_true - old_true if can_compare and old_true is not None and new_true is not None else None
        if delta_eng is not None:
            engineering_deltas.append(delta_eng)
        if delta_true is not None:
            true_deltas.append(delta_true)
        row_comparison = comparison_status if comparison_status != "COMPARABLE" else ("COMPARED" if can_compare else "INSUFFICIENT_VALID_OVERLAP")
        frames.append({
            "frame": frame_id,
            "old_native_dic_frame": native_id if old_row is not None else None,
            "new_native_dic_frame": native_id if new_row is not None else None,
            "join_status": join_status,
            "old_valid_gauge_count": old_row.get("valid_gauge_count") if old_row is not None else None,
            "new_valid_gauge_count": new_row.get("valid_gauge_count") if new_row is not None else None,
            "old_strain_status": old_row.get("strain_status") if old_row is not None else None,
            "new_strain_status": new_row.get("strain_status") if new_row is not None else None,
            "old_engineering_strain": old_eng,
            "new_engineering_strain": new_eng,
            "delta_engineering_strain": delta_eng,
            "old_true_strain": old_true,
            "new_true_strain": new_true,
            "delta_true_strain": delta_true,
            "comparison_status": row_comparison,
        })

    payload = {
        "schema_version": 1,
        "kind": "straintrace_roi_sensitivity_comparison",
        "acquisition_id": acq.get("acquisition_id"),
        "created_at_utc": _now_utc(),
        "comparison_status": comparison_status,
        "join_key": ["frame", "native_dic_frame"],
        "baseline": {
            "fingerprint": baseline.get("fingerprint"),
            "rows_path": baseline.get("rows_path"),
            "reference_fingerprint": baseline_reference,
            "source_images_fingerprint": (baseline.get("provenance") or {}).get("source_images_fingerprint"),
            "roi_geometry": pending.get("baseline_roi_geometry"),
        },
        "candidate": {
            "fingerprint": candidate.get("fingerprint"),
            "rows_path": candidate.get("rows_path"),
            "reference_fingerprint": candidate_reference,
            "source_images_fingerprint": (candidate.get("provenance") or {}).get("source_images_fingerprint"),
            "roi_geometry": _roi_tracking_geometry(acq),
        },
        "join": {
            "baseline_frame_count": len(old_index),
            "candidate_frame_count": len(new_index),
            "matched_frame_count": overlap,
            "union_frame_count": len(union),
            "baseline_only_count": len(set(old_index) - set(new_index)),
            "candidate_only_count": len(set(new_index) - set(old_index)),
            "duplicate_identity_count": len(duplicate_keys),
            "baseline_malformed_identity_count": old_malformed,
            "candidate_malformed_identity_count": new_malformed,
            "valid_engineering_delta_count": len(engineering_deltas),
            "valid_true_strain_delta_count": len(true_deltas),
        },
        "differences": {
            "engineering_strain": _delta_summary(engineering_deltas),
            "true_strain": _delta_summary(true_deltas),
        },
        "acceptance_rule": "Comparison only; both source stages retain their original quality gate of at least three valid gauges.",
        "frames": frames,
    }
    fingerprint = fingerprint_data("roi_sensitivity", {
        "acquisition_id": acq.get("acquisition_id"),
        "baseline_fingerprint": baseline.get("fingerprint"),
        "candidate_fingerprint": candidate.get("fingerprint"),
        "baseline_reference_fingerprint": baseline_reference,
        "candidate_reference_fingerprint": candidate_reference,
        "baseline_geometry": pending.get("baseline_roi_geometry"),
        "candidate_geometry": payload["candidate"]["roi_geometry"],
    })
    output_dir = stage_output_dir(root, str(acq["acquisition_id"]), "roi-sensitivity", fingerprint)
    json_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"
    atomic_write_json(json_path, payload)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_SENSITIVITY_CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(frames)
    atomic_write_bytes(csv_path, stream.getvalue().encode("utf-8"))
    json_rel = relative_artifact(root, json_path)
    csv_rel = relative_artifact(root, csv_path)
    for rel in (json_rel, csv_rel):
        candidate.setdefault("artifacts", []).append(rel)
        candidate.setdefault("checksums", {})[rel] = sha256_file(artifact_path(root, rel))
    report = {
        "status": comparison_status,
        "fingerprint": fingerprint,
        "json_path": json_rel,
        "csv_path": csv_rel,
        "summary": _clone(payload["join"]),
    }
    return report


def _expected_alignment_fingerprint(
    aligner: Any,
    acq: Mapping[str, Any],
    load: Path,
    options: Mapping[str, Any],
) -> str:
    source_fingerprint = sha256_file(load)
    module_fingerprint = None
    fingerprint_fn = getattr(aligner, "fingerprint", None)
    if callable(fingerprint_fn):
        module_fingerprint = fingerprint_fn(acq, load, dict(options))
    return fingerprint_data("alignment", {
        "aligner_fingerprint": module_fingerprint,
        "alignment_code_sha256": _module_code_sha256(aligner),
        "load_sha256": source_fingerprint,
        "result_fingerprint": (acq.get("result") or {}).get("fingerprint"),
        "measurement_frames": acq["measurement_frames"],
        "image_frames": acq["image_frames"],
        "dimensions": acq["dimensions"],
        "options": dict(options),
    })


def _alignment_options_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    provenance = record.get("provenance") or {}
    workflow_options = provenance.get("workflow_alignment_options")
    if isinstance(workflow_options, Mapping):
        return dict(workflow_options)
    return {
        "mode": provenance.get("mode", "instantaneous"),
        "max_gap_s": provenance.get("max_gap_s", 2.5),
        "force_column": provenance.get("load_force_column", "force_N"),
        "dic_max_gap_s": provenance.get("dic_max_gap_s"),
        "force_unit": provenance.get("load_force_unit_override"),
        "measurement_time_field": provenance.get("measurement_time_field", "timestamp_unix_s"),
        "load_time_column": provenance.get("load_time_column", "timestamp_unix_s"),
    }


def _validate_export_freshness(root: Path, project: dict[str, Any]) -> None:
    for acq in project["acquisitions"]:
        metadata = acq.get("metadata", {})
        result = acq.get("result")
        if result is not None:
            if result.get("status") == "IMPORTED_ACCEPTED":
                if not stage_record_reusable(root, result, result["fingerprint"]):
                    raise CampaignError("EXPORT_IMPORTED_RESULT_INVALID", f"imported accepted rows or checksums are invalid for {acq['acquisition_id']}")
            else:
                if metadata.get("workflow_result_stale"):
                    raise CampaignError("EXPORT_STALE_RESULT", f"tracking output is stale for {acq['acquisition_id']}; approve and rerun tracking")
                _ensure_approval(acq)
                numerics = _module("numerics")
                expected = _expected_analysis_fingerprint(numerics, acq)
                if result.get("fingerprint") != expected or not stage_record_reusable(root, result, expected):
                    raise CampaignError("EXPORT_STALE_RESULT", f"tracking fingerprint or artifacts are stale for {acq['acquisition_id']}; rerun tracking")
        alignment = acq.get("alignment")
        if alignment is None:
            continue
        if alignment.get("status") == "IMPORTED_ACCEPTED":
            if not stage_record_reusable(root, alignment, alignment["fingerprint"]):
                raise CampaignError("EXPORT_IMPORTED_ALIGNMENT_INVALID", f"imported alignment artifacts are invalid for {acq['acquisition_id']}")
            continue
        provenance = alignment.get("provenance") or {}
        load_path = provenance.get("load_source_path")
        if not isinstance(load_path, str) or not Path(load_path).is_file():
            raise CampaignError("EXPORT_ALIGNMENT_SOURCE_MISSING", f"alignment source is missing for {acq['acquisition_id']}")
        aligner = _module("alignment")
        options = _alignment_options_from_record(alignment)
        expected = _expected_alignment_fingerprint(aligner, acq, Path(load_path), options)
        if alignment.get("fingerprint") != expected or not stage_record_reusable(root, alignment, expected):
            raise CampaignError("EXPORT_STALE_ALIGNMENT", f"alignment is stale for {acq['acquisition_id']}; realign before export")


def _ensure_approval(acq: dict[str, Any]) -> None:
    if acq.get("roi") is None or acq.get("approval") is None:
        raise CampaignError("ROI_APPROVAL_REQUIRED", "run requires an approved, current ROI proposal")
    roi = _module("roi")
    _external_call("roi", roi.validate_approval, acq)


def _run_one(
    root: Path,
    project: dict[str, Any],
    acq: dict[str, Any],
    *,
    progress: Callable[[dict[str, Any]], None] | None,
    cancel: Any,
) -> dict[str, Any]:
    _check_cancel(cancel)
    numerics = _module("numerics")
    imported = acq.get("result")
    if imported is not None and imported.get("status") == "IMPORTED_ACCEPTED" and not acq.get("metadata", {}).get("workflow_reanalysis_required"):
        if stage_record_reusable(root, imported, imported["fingerprint"]):
            _notify(progress, {"event": "stage_reused", "stage": "tracking", "acquisition_id": acq["acquisition_id"], "status": "IMPORTED_ACCEPTED"})
            return _clone(imported)
    _ensure_approval(acq)
    expected = _expected_analysis_fingerprint(numerics, acq)
    current = acq.get("result")
    if stage_record_reusable(root, current, expected):
        metadata = acq.setdefault("metadata", {})
        changed = False
        if metadata.pop("workflow_result_stale", None) is not None:
            changed = True
        pending = metadata.get("pending_roi_sensitivity")
        baseline_pointer = pending.get("baseline_result") if isinstance(pending, Mapping) else None
        if isinstance(pending, Mapping) and isinstance(baseline_pointer, Mapping) and baseline_pointer.get("fingerprint") == current.get("fingerprint"):
            if pending.get("baseline_roi_geometry") == _roi_tracking_geometry(acq):
                metadata.pop("pending_roi_sensitivity", None)
                changed = True
        if changed:
            _write_project(root, project)
        _notify(progress, {"event": "stage_reused", "stage": "tracking", "acquisition_id": acq["acquisition_id"], "fingerprint": expected})
        return _clone(current)
    _notify(progress, {"event": "stage_started", "stage": "tracking", "acquisition_id": acq["acquisition_id"], "fingerprint": expected})

    def stage_progress(event: Any) -> None:
        if isinstance(event, Mapping):
            payload = dict(event)
        else:
            payload = {"message": str(event)}
        payload.update({"event": "progress", "stage": "tracking", "acquisition_id": acq["acquisition_id"]})
        _notify(progress, payload)

    result = _external_call("numerics", numerics.analyze, acq, root, progress=stage_progress, cancel=cancel)
    _check_cancel(cancel)
    if not isinstance(result, Mapping):
        raise CampaignError("NUMERICS_RESULT_INVALID", "numerics.analyze must return a stage record")
    finalized = finalize_stage_record(root, result, fingerprint=expected)
    pending = acq.get("metadata", {}).get("pending_roi_sensitivity")
    if isinstance(pending, Mapping):
        if finalized.get("status") not in {"COMPLETE", "COMPLETED"}:
            raise CampaignError("ROI_SENSITIVITY_CANDIDATE_INCOMPLETE", "previous completed tracking result was retained because the ROI retry did not complete")
        comparison = _write_roi_sensitivity_comparison(root, acq, pending, finalized)
        acq.setdefault("metadata", {}).setdefault("roi_sensitivity_comparisons", []).append(comparison)
        acq["metadata"].pop("pending_roi_sensitivity", None)
    if acq.get("alignment") is not None:
        _archive_alignment(acq, reason="tracking_result_changed")
    if acq.get("result") is not None:
        reason = "campaign_reanalysis_completed" if acq["result"].get("status") == "IMPORTED_ACCEPTED" else "tracking_result_replaced"
        _archive_result(acq, reason=reason)
    acq["result"] = finalized
    acq.get("metadata", {}).pop("workflow_reanalysis_required", None)
    acq.get("metadata", {}).pop("workflow_result_stale", None)
    validate_acquisition(acq)
    _write_project(root, project)
    _notify(progress, {"event": "stage_completed", "stage": "tracking", "acquisition_id": acq["acquisition_id"], "fingerprint": expected})
    return _clone(finalized)


def run(
    path: str | os.PathLike[str],
    acquisition_id: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancel: Any = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Track one approved acquisition or return per-acquisition batch outcomes."""
    root = _project_dir(path)
    with acquire_project_lock(root):
        project = _read_project(root)
        if acquisition_id is None:
            acquisitions = project["acquisitions"]
            outcomes: list[dict[str, Any]] = []
            for acq in acquisitions:
                _check_cancel(cancel)
                try:
                    stage = _run_one(root, project, acq, progress=progress, cancel=cancel)
                    outcomes.append({"acquisition_id": acq["acquisition_id"], "status": stage["status"], "stage": stage})
                except CampaignError as exc:
                    if exc.code == "ROI_APPROVAL_REQUIRED":
                        outcomes.append({"acquisition_id": acq["acquisition_id"], "status": "BLOCKED", "error": exc.as_dict()})
                        _notify(progress, {"event": "stage_blocked", "stage": "tracking", "acquisition_id": acq["acquisition_id"], **exc.as_dict()})
                        continue
                    raise
            # Persist in case the operation updated only metadata/history or an
            # imported record was moved before a new stage started.
            _write_project(root, project)
            return outcomes
        acq = _find_acquisition(project, acquisition_id)
        result = _run_one(root, project, acq, progress=progress, cancel=cancel)
        # _run_one writes on new output; a cache hit still needs no project edit.
        return result


def align(
    path: str | os.PathLike[str],
    acquisition_id: str,
    load_path: str | os.PathLike[str],
    **kwargs: Any,
) -> dict[str, Any]:
    root = _project_dir(path)
    load = Path(load_path).expanduser().resolve()
    if not load.is_file():
        raise CampaignError("LOAD_SOURCE_NOT_FOUND", f"load source file does not exist: {load}")
    kwargs = dict(kwargs)
    legacy_time_field = kwargs.pop("time_column", None)
    if "measurement_time_field" in kwargs and legacy_time_field is not None:
        raise CampaignError("ALIGNMENT_OPTION_INVALID", "provide either measurement_time_field or legacy time_column, not both")
    kwargs.setdefault("measurement_time_field", legacy_time_field or "timestamp_unix_s")
    kwargs.setdefault("load_time_column", "timestamp_unix_s")
    allowed = {"mode", "max_gap_s", "force_column", "dic_max_gap_s", "force_unit", "measurement_time_field", "load_time_column"}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise CampaignError("ALIGNMENT_OPTION_INVALID", f"unsupported alignment option {unknown[0]!r}")
    with acquire_project_lock(root):
        project = _read_project(root)
        acq = _find_acquisition(project, acquisition_id)
        aligner = _module("alignment")
        source_fingerprint = sha256_file(load)
        expected = _expected_alignment_fingerprint(aligner, acq, load, kwargs)
        current = acq.get("alignment")
        if stage_record_reusable(root, current, expected):
            return _clone(current)
        raw = _external_call("alignment", aligner.align, acq, root, load, **kwargs)
        if sha256_file(load) != source_fingerprint:
            raise CampaignError("INPUT_CHANGED_DURING_ALIGNMENT", "load source changed while alignment was running")
        if not isinstance(raw, Mapping):
            raise CampaignError("ALIGNMENT_RESULT_INVALID", "alignment.align must return a stage record")
        record = finalize_stage_record(root, raw, fingerprint=expected)
        if current is not None:
            _archive_alignment(acq, reason="alignment_fingerprint_changed")
        record.setdefault("provenance", {}).update({
            "load_source_path": str(load),
            "load_source_sha256": source_fingerprint,
            "workflow_alignment_options": dict(kwargs),
        })
        acq["alignment"] = record
        acq.get("metadata", {}).pop("alignment_recalculation_required", None)
        validate_acquisition(acq)
        _write_project(root, project)
        return _clone(record)


def export(
    path: str | os.PathLike[str],
    output: str | os.PathLike[str],
    publish_to: str | os.PathLike[str] | None = None,
    timeout_s: float = 60,
) -> dict[str, Any]:
    root = _project_dir(path)
    if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise CampaignError("EXPORT_TIMEOUT_INVALID", "timeout_s must be a positive number")
    output_path = Path(output).expanduser().resolve()
    publish_path = Path(publish_to).expanduser().resolve() if publish_to is not None else None
    with acquire_project_lock(root):
        project = _read_project(root)
        _validate_export_freshness(root, project)
        output_dirs = project.setdefault("metadata", {}).setdefault("output_dirs", [])
        for output_dir in (str(output_path), str(publish_path) if publish_path is not None else None):
            if output_dir and output_dir not in output_dirs:
                output_dirs.append(output_dir)
        _write_project(root, project)
        delivery = _module("delivery")
        receipt = _external_call("delivery", delivery.export_project, root, project, output_path, publish_to=publish_path, timeout_s=float(timeout_s))
        if not isinstance(receipt, Mapping):
            raise CampaignError("EXPORT_RECEIPT_INVALID", "delivery.export_project must return a receipt object")
        return dict(receipt)


def verify(path: str | os.PathLike[str]) -> dict[str, Any]:
    value = Path(path).expanduser()
    if value.is_file() or not (value / "project.json").is_file():
        delivery = _module("delivery")
        report = _external_call("delivery", delivery.verify_package, value)
        if not isinstance(report, Mapping):
            raise CampaignError("VERIFY_REPORT_INVALID", "delivery.verify_package must return a report object")
        return dict(report)
    root = _project_dir(value)
    project = _read_project(root)
    failures: list[dict[str, str]] = []
    checked = 0
    for acq in project["acquisitions"]:
        for stage_name in ("result", "alignment"):
            record = acq.get(stage_name)
            if record is None:
                continue
            checked += 1
            for rel, expected in record["checksums"].items():
                target = artifact_path(root, rel)
                if not target.is_file():
                    failures.append({"acquisition_id": acq["acquisition_id"], "stage": stage_name, "artifact": rel, "reason": "missing"})
                elif sha256_file(target) != expected:
                    failures.append({"acquisition_id": acq["acquisition_id"], "stage": stage_name, "artifact": rel, "reason": "sha256_mismatch"})
    try:
        _validate_export_freshness(root, project)
    except CampaignError as exc:
        failures.append({"acquisition_id": "", "stage": "project_freshness", "artifact": "", "reason": exc.code})
    return {
        "valid": not failures,
        "project_id": project["project_id"],
        "acquisition_count": len(project["acquisitions"]),
        "stage_record_count": checked,
        "failures": failures,
    }
