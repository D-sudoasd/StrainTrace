"""Campaign package import, export, verification, and atomic publication.

This module deliberately has no image or numerical-engine dependencies.  It
copies accepted tables and provenance documents, and never copies raw images.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import socket
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


_PACKAGE_SCHEMA = "straintrace-campaign-package-1"
_CURRENT_SCHEMA = "straintrace-campaign-current-1"
_CATALOG_LINK_SCHEMA = "straintrace-campaign-catalog-link-1"
_IMPORT_SCHEMA = "straintrace-keystone-import-1"
_RAW_IMAGE_SUFFIXES = {".tif", ".tiff", ".cbf", ".edf", ".mar3450", ".raw"}
_CHECKSUMS = "CHECKSUMS.json"
_VALIDATION = "VALIDATION.json"
_CURRENT = "CURRENT.json"
_IMPORT_FILES = ("sample.json", "sources.json", "frames.csv", "frames.jsonl", "frame_stress_strain.txt", "frame_stress.txt")


def _json_bytes(value: Any, *, indent: int | None = 2) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=indent) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(_json_safe(value)))


def _json_safe(value: Any) -> Any:
    """Convert common numeric scalars to strict JSON values; NaN means missing."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # Accept NumPy scalar values without importing NumPy in the headless module.
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is forbidden: {value}")


def _read_json(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_reject_json_constant)
    _assert_finite(value, str(path))
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=_reject_json_constant)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            _assert_finite(row, f"{path}:{line_number}")
            rows.append(row)
    return rows


def _assert_finite(value: Any, where: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where}: non-finite numeric value")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{where}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{where}[{index}]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_file(root: Path, value: str | os.PathLike[str]) -> Path:
    """Resolve an artifact path and reject absolute paths and traversal."""
    raw = str(value).replace("\\", "/")
    rel = PurePosixPath(raw)
    if rel.is_absolute() or not rel.parts or any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError(f"Unsafe relative artifact path: {value!r}")
    target = root.joinpath(*rel.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes its root: {value!r}")
    return target


def _relative_acquisition_path(value: Any, fallback: str) -> str:
    if value in (None, ""):
        value = fallback
    raw = str(value).replace("\\", "/").strip("/")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe acquisition relative_path: {value!r}")
    return path.as_posix()


def _resolve_keystone_version(source: str | os.PathLike[str]) -> Path:
    path = Path(source).expanduser().resolve()
    if path.is_file() and path.name.lower() == _CURRENT.lower():
        root = path.parent
        current = _read_json(path)
        relative = current.get("version_path") or current.get("version_relative_path")
        if not isinstance(relative, str):
            raise ValueError("CURRENT.json must identify version_path")
        version = _relative_file(root, relative)
        if not version.is_dir():
            raise FileNotFoundError(f"Keystone current version is missing: {version}")
        return version
    if path.is_dir() and (path / "catalog.json").is_file() and (path / "samples").is_dir():
        return path
    if path.is_dir() and (path / _CURRENT).is_file():
        return _resolve_keystone_version(path / _CURRENT)
    raise ValueError(f"Expected a Keystone version directory or root with {_CURRENT}: {path}")


def _resolve_package(path: str | os.PathLike[str]) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file() and candidate.suffix.lower() == ".zip":
        temp = tempfile.TemporaryDirectory(prefix="straintrace-verify-")
        try:
            with zipfile.ZipFile(candidate) as archive:
                bad_member = archive.testzip()
                if bad_member:
                    raise ValueError(f"ZIP CRC failed for {bad_member}")
                root = Path(temp.name)
                for item in archive.infolist():
                    member = PurePosixPath(item.filename)
                    if member.is_absolute() or any(part == ".." for part in member.parts):
                        raise ValueError(f"Unsafe ZIP member path: {item.filename}")
                archive.extractall(root)
            # Archives emitted here contain package files at their root. Be
            # tolerant of one enclosing directory for hand-carried packages.
            if not (root / "catalog.json").is_file():
                children = [entry for entry in root.iterdir() if entry.is_dir()]
                if len(children) == 1 and (children[0] / "catalog.json").is_file():
                    root = children[0]
            return root, temp
        except Exception:
            temp.cleanup()
            raise
    if candidate.is_dir() and (candidate / _CURRENT).is_file():
        current = _read_json(candidate / _CURRENT)
        version = _relative_file(candidate, current.get("version_path", ""))
        return version, None
    return candidate, None


def _load_catalog(version: Path) -> list[dict[str, Any]]:
    catalog = _read_json(version / "catalog.json")
    if not isinstance(catalog, list) or any(not isinstance(row, dict) for row in catalog):
        raise ValueError("catalog.json must contain a list of acquisition records")
    return catalog


def _verify_keystone_source(version: Path) -> dict[str, Any]:
    """Verify the DESY Keystone source's own checksum layout before importing."""
    errors: list[str] = []
    checksum_path = version / _CHECKSUMS
    validation_path = version / "VALIDATION.json"
    if not checksum_path.is_file() or not validation_path.is_file():
        return {"valid": False, "errors": ["Keystone source is missing CHECKSUMS.json or VALIDATION.json"]}
    manifest = _read_json(checksum_path)
    validation = _read_json(validation_path)
    if not isinstance(manifest, dict):
        return {"valid": False, "errors": ["Keystone CHECKSUMS.json must be an object"]}
    for relative, claim in manifest.items():
        try:
            target = _relative_file(version, relative)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not target.is_file():
            errors.append(f"Keystone checksum target missing: {relative}")
            continue
        expected = claim.get("sha256") if isinstance(claim, dict) else claim
        if expected and _sha256(target) != expected:
            errors.append(f"Keystone checksum mismatch: {relative}")
    actual_files = {path.relative_to(version).as_posix() for path in version.rglob("*") if path.is_file() and path.name != _CHECKSUMS}
    unlisted = actual_files - set(manifest)
    # BUILD_RECEIPT pins the checksum manifest itself and is intentionally
    # outside that manifest to avoid a checksum cycle.
    unexpected = sorted(unlisted - {"BUILD_RECEIPT.json"})
    missing = sorted(set(manifest) - actual_files)
    if unexpected or missing:
        errors.append(f"Keystone checksum inventory mismatch (unlisted={unexpected[:5]}, missing={missing[:5]})")
    if not (validation.get("passed") is True or validation.get("status") == "PASS"):
        errors.append("Keystone VALIDATION.json is not PASS")
    catalog = _load_catalog(version)
    expected_count = validation.get("scope_acquisitions_expected")
    written_count = validation.get("scope_acquisitions_written")
    if expected_count is not None and int(expected_count) != len(catalog):
        errors.append("Keystone catalog does not match the validated acquisition count")
    if written_count is not None and int(written_count) != len(catalog):
        errors.append("Keystone catalog does not match scope_acquisitions_written")
    return {
        "valid": not errors,
        "errors": errors,
        "acquisition_count": len(catalog),
        "frame_count": validation.get("frames_written"),
        "checksummed_file_count": len(manifest),
        "claims_rehashed": True,
    }


def _project_import_dir(project_dir: Path, version_id: str) -> Path:
    safe_version = Path(version_id).name
    if safe_version != version_id or safe_version in ("", ".", ".."):
        safe_version = hashlib.sha256(version_id.encode("utf-8")).hexdigest()[:16]
    return project_dir / "imports" / "keystone" / safe_version


def _keystone_version_digest(version: Path) -> str:
    pinned = version / _CHECKSUMS
    return _sha256(pinned if pinned.is_file() else version / "catalog.json")


def _validate_existing_import(
    import_dir: Path,
    catalog: list[dict[str, Any]],
    source_version_id: str,
    source_version_sha256: str,
) -> dict[str, Any]:
    manifest_path = import_dir / "IMPORT_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(f"Existing Keystone import is incomplete and will not be overwritten: {import_dir}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != _IMPORT_SCHEMA:
        raise ValueError(f"Existing Keystone import has an unsupported manifest: {manifest_path}")
    if manifest.get("source_version_id") != source_version_id or manifest.get("source_version_sha256") != source_version_sha256:
        raise FileExistsError(f"A different Keystone content version already occupies {import_dir}; it will not be overwritten")
    records = manifest.get("acquisitions")
    if not isinstance(records, list) or len(records) != len(catalog):
        raise ValueError(f"Existing Keystone import has an incomplete acquisition list: {manifest_path}")
    expected_ids = {str(row.get("acquisition_id")) for row in catalog}
    actual_ids = {str(row.get("acquisition_id")) for row in records if isinstance(row, dict)}
    if actual_ids != expected_ids or len(actual_ids) != len(records):
        raise ValueError(f"Existing Keystone import acquisition identities differ from the source: {manifest_path}")
    expected_files: set[str] = {"IMPORT_MANIFEST.json"}
    frame_total = 0
    for record in records:
        acquisition_id = str(record["acquisition_id"])
        files = record.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError(f"Existing Keystone import has no file inventory for {acquisition_id}")
        acquisition_dir = _relative_file(import_dir, acquisition_id)
        for filename, digest in files.items():
            target = _relative_file(acquisition_dir, filename)
            relative = target.relative_to(import_dir).as_posix()
            expected_files.add(relative)
            if not target.is_file() or _sha256(target) != digest:
                raise ValueError(f"Existing Keystone import checksum failed; refusing to overwrite: {relative}")
        rows_path = acquisition_dir / "frames.jsonl"
        rows = _read_jsonl(rows_path)
        _validate_unique_frames(rows, f"Existing imported acquisition {acquisition_id}")
        if len(rows) != int(record.get("frame_count", -1)):
            raise ValueError(f"Existing imported frame count failed for {acquisition_id}")
        frame_total += len(rows)
    actual_files = {path.relative_to(import_dir).as_posix() for path in import_dir.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(f"Existing Keystone import file inventory differs: {import_dir}")
    if int(manifest.get("acquisition_count", -1)) != len(catalog) or int(manifest.get("frame_count", -1)) != frame_total:
        raise ValueError(f"Existing Keystone import aggregate counts failed: {manifest_path}")
    return manifest


def import_keystone(project_dir: str | os.PathLike[str], source: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Import accepted Keystone tables and source documents without raw images.

    Imported frame rows remain byte-preserved in their JSONL artifact. Their
    stage status is explicitly ``IMPORTED_ACCEPTED``; this function does not
    run tracking, recalculate stress, or reinterpret timestamps.
    """
    project_root = Path(project_dir).expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"Campaign project directory does not exist: {project_root}")
    version = _resolve_keystone_version(source)
    report = _verify_keystone_source(version)
    if not report["valid"]:
        raise ValueError("Keystone source package failed verification: " + "; ".join(report["errors"]))

    catalog = _load_catalog(version)
    source_version_id = version.name
    source_version_sha256 = _keystone_version_digest(version)
    final_import_dir = _project_import_dir(project_root, source_version_id)
    existing_import = None
    if final_import_dir.exists():
        existing_import = _validate_existing_import(final_import_dir, catalog, source_version_id, source_version_sha256)

    known_ids: set[str] = set()
    project_file = project_root / "project.json"
    if project_file.is_file():
        existing = _read_json(project_file)
        known_ids = {str(item.get("acquisition_id")) for item in existing.get("acquisitions", []) if item.get("acquisition_id")}

    stage_parent = final_import_dir.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    staging = stage_parent / f".{final_import_dir.name}.staging-{uuid.uuid4().hex}"
    imported: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    try:
        staging.mkdir()
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        for item in catalog:
            acquisition_id = str(item.get("acquisition_id") or "")
            if not acquisition_id or acquisition_id in seen_ids or acquisition_id in known_ids:
                raise ValueError(f"Missing or duplicate acquisition_id in Keystone import: {acquisition_id!r}")
            seen_ids.add(acquisition_id)
            relative_path = _relative_acquisition_path(item.get("delivery_relative_path"), acquisition_id)
            if relative_path.casefold() in seen_paths:
                raise ValueError(f"Duplicate delivery_relative_path in Keystone catalog: {relative_path}")
            seen_paths.add(relative_path.casefold())
            sample_rel = item.get("sample_json") or f"samples/{relative_path}/sample.json"
            source_rel = item.get("sources_json") or f"samples/{relative_path}/sources.json"
            frame_rel = item.get("frames_jsonl") or f"samples/{relative_path}/frames.jsonl"
            csv_rel = item.get("frames_csv") or f"samples/{relative_path}/frames.csv"
            source_sample = _relative_file(version, sample_rel)
            source_sources = _relative_file(version, source_rel)
            source_jsonl = _relative_file(version, frame_rel)
            source_csv = _relative_file(version, csv_rel)
            for required in (source_sample, source_sources, source_jsonl, source_csv):
                if not required.is_file():
                    raise FileNotFoundError(f"Required Keystone metadata/table missing: {required}")

            original_sample = _read_json(source_sample)
            original_sources = _read_json(source_sources)
            if str(original_sample.get("acquisition_id")) != acquisition_id:
                raise ValueError(f"Keystone sample identity mismatch for {acquisition_id}")
            rows = _read_jsonl(source_jsonl)
            _validate_unique_frames(rows, f"Keystone acquisition {acquisition_id}")
            table_errors: list[str] = []
            _verify_frame_csv(source_csv, rows, table_errors, acquisition_id)
            if table_errors:
                raise ValueError("Keystone CSV/JSONL mismatch: " + "; ".join(table_errors))
            expected_count = item.get("frames")
            if expected_count is not None and len(rows) != int(expected_count):
                raise ValueError(f"Keystone frame count mismatch for {acquisition_id}: catalog={expected_count}, rows={len(rows)}")

            imported_rel = PurePosixPath("imports", "keystone", final_import_dir.name, acquisition_id)
            dest = staging / acquisition_id
            dest.mkdir(parents=True)
            copied: dict[str, str] = {}
            for filename, source_file in (("sample.json", source_sample), ("sources.json", source_sources), ("frames.jsonl", source_jsonl), ("frames.csv", source_csv)):
                shutil.copyfile(source_file, dest / filename)
                copied[filename] = _sha256(dest / filename)
            origin_rel = item.get("origin_txt")
            if origin_rel:
                origin = _relative_file(version, origin_rel)
                if origin.is_file():
                    shutil.copyfile(origin, dest / origin.name)
                    copied[origin.name] = _sha256(dest / origin.name)
            else:
                for filename in ("frame_stress_strain.txt", "frame_stress.txt"):
                    candidate = source_jsonl.parent / filename
                    if candidate.is_file():
                        shutil.copyfile(candidate, dest / filename)
                        copied[filename] = _sha256(dest / filename)
                        break

            dimensions = original_sample.get("dimensions", {})
            dic = original_sample.get("dic", {})
            width = dimensions.get("width_mm_recorded")
            thickness = dimensions.get("thickness_mm_recorded")
            area = dimensions.get("accepted_area_mm2")
            area_status = dimensions.get("accepted_area_status") or dimensions.get("dimension_match_status") or "UNCONFIRMED"
            metadata = {
                "keystone_import": {
                    "schema_version": _IMPORT_SCHEMA,
                    "source_version_id": source_version_id,
                    "source_relative_path": str(item.get("delivery_relative_path", relative_path)),
                    "source_sample_sha256": _sha256(source_sample),
                    "source_sources_sha256": _sha256(source_sources),
                    "source_frames_jsonl_sha256": _sha256(source_jsonl),
                    "source_frames_csv_sha256": _sha256(source_csv),
                    "preserved_files": {name: f"{imported_rel.as_posix()}/{name}" for name in copied},
                    "dic_present": bool(item.get("DIC_present", dic.get("present_in_indexed_sources", False))),
                    "historical_source_claims": _historical_source_claims(original_sources),
                },
                "original_sample_json_path": f"{imported_rel.as_posix()}/sample.json",
                "original_sources_json_path": f"{imported_rel.as_posix()}/sources.json",
            }
            measurement_frames = [_measurement_frame(row) for row in rows]
            result_path = f"{imported_rel.as_posix()}/frames.jsonl"
            imported_acquisition = {
                "acquisition_id": acquisition_id,
                "name": str(item.get("sample_name") or original_sample.get("sample_name") or acquisition_id),
                "relative_path": relative_path,
                "source_kind": "imported_keystone",
                "physical_specimen_id": original_sample.get("physical_specimen_id"),
                "dimensions": {
                    "area_mm2": area,
                    "width_mm": width,
                    "thickness_mm": thickness,
                    "status": area_status,
                    "source": dimensions.get("dimension_source"),
                    "area_basis": dimensions.get("accepted_area_basis"),
                    "area_conflict": bool(dimensions.get("area_conflict", False)),
                    "alternate_area_mm2": dimensions.get("alternate_area_mm2"),
                    "alternate_area_source": dimensions.get("alternate_area_source"),
                },
                "analysis_mode": "extensometer",
                # Images remain at their source location and are never copied.
                "image_frames": [],
                "measurement_frames": measurement_frames,
                "roi": None,
                "approval": None,
                "analysis_config": {},
                "result": {
                    "status": "IMPORTED_ACCEPTED",
                    "fingerprint": _sha256(source_jsonl),
                    "rows_path": result_path,
                    "artifacts": [f"{imported_rel.as_posix()}/{name}" for name in copied],
                    "checksums": {f"{imported_rel.as_posix()}/{name}": digest for name, digest in copied.items()},
                    "provenance": {
                        "origin": "IMPORTED_ACCEPTED",
                        "numerics_rerun": False,
                        "source_package": str(version),
                        "source_manifest_claims_are_historical_unless_rehashed": True,
                    },
                    "summary": {
                        "frame_count": len(rows),
                        "stress_valid": int(item.get("stress_valid", 0)),
                        "strain_valid": int(item.get("strain_valid", 0)),
                        "valid_pairs": int(item.get("valid_pairs", 0)),
                    },
                },
                "alignment": None,
                "metadata": metadata,
            }
            imported.append(imported_acquisition)
            manifest_rows.append({"acquisition_id": acquisition_id, "relative_path": relative_path, "files": copied, "frame_count": len(rows)})

        if len(imported) != len(catalog):
            raise ValueError("Keystone catalog import was incomplete")
        _write_json(staging / "IMPORT_MANIFEST.json", {
            "schema_version": _IMPORT_SCHEMA,
            "source_version_id": source_version_id,
            "source_version_sha256": source_version_sha256,
            "source_package_path": str(version),
            "source_verification": report,
            "acquisition_count": len(imported),
            "frame_count": sum(row["frame_count"] for row in manifest_rows),
            "acquisitions": manifest_rows,
        })
        final_import_dir.parent.mkdir(parents=True, exist_ok=True)
        if existing_import is not None:
            new_manifest = _read_json(staging / "IMPORT_MANIFEST.json")
            if (
                new_manifest.get("source_version_sha256") != existing_import.get("source_version_sha256")
                or new_manifest.get("acquisition_count") != existing_import.get("acquisition_count")
                or new_manifest.get("frame_count") != existing_import.get("frame_count")
                or new_manifest.get("acquisitions") != existing_import.get("acquisitions")
            ):
                raise FileExistsError(f"Existing import does not match this exact source content: {final_import_dir}")
            shutil.rmtree(staging)
        else:
            os.replace(staging, final_import_dir)
        # All returned artifact paths now refer to the committed import root.
        return imported
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _historical_source_claims(sources: Any) -> list[dict[str, Any]]:
    if not isinstance(sources, dict):
        return []
    claims = []
    for item in sources.get("sources", []):
        if not isinstance(item, dict):
            continue
        claims.append({
            "role": item.get("role"),
            "path": item.get("path"),
            "sha256": item.get("sha256"),
            "verification_status_at_source": item.get("verification_status"),
            "rehashed_during_import": False,
        })
    return claims


def _measurement_frame(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        # Keep the aligned-axis identity and the already accepted mechanical
        # values together. Alignment can add new fields, but must never lose or
        # silently replace these imported values.
        "frame", "original_frame_index", "native_cbf_frame", "native_dic_frame", "scan_id",
        "scan_id_is_global_fio", "frame_namespace", "timestamp_unix_s", "timestamp_utc",
        "timestamp_china", "timestamp_status", "timestamp_source", "time_s", "time_source",
        "estimated_timestamp_unix_s", "estimated_timestamp_utc", "estimated_timestamp_status",
        "estimated_timestamp_source", "dic_timestamp_unix_s", "dic_timestamp_status",
        "dic_timestamp_source", "dic_left_frame", "dic_left_filename", "dic_left_timestamp_unix_s",
        "dic_left_time_source", "dic_right_frame", "dic_right_filename", "dic_right_timestamp_unix_s",
        "dic_right_time_source", "dic_interpolation_alpha", "dic_timing_basis", "timezone_basis",
        "exposure_start_unix_s", "exposure_end_unix_s", "force_N", "force_unit", "force_status",
        "force_source", "force_definition", "force_timing_basis", "force_interpolation_status",
        "force_recovery_method", "area_mm2", "area_basis", "area_status", "area_conflict_flag",
        "area_alternate_mm2", "area_alternate_source", "stress_MPa", "stress_status",
        "stress_missing_reason", "engineering_strain", "reprocessed_source_strain_value",
        "true_strain", "engineering_strain_sd", "true_strain_sd", "valid_gauge_count",
        "source_valid_gauge_count", "reprocessed_valid_gauge_count", "reference_frame",
        "strain_status", "strain_missing_reason", "strain_source", "quality_status",
        "dic_time_support_status", "source_row_present", "source_record_key",
    )
    return {key: _json_safe(row[key]) for key in keys if key in row}


def _validate_unique_frames(rows: list[Mapping[str, Any]], where: str) -> None:
    seen: set[Any] = set()
    for index, row in enumerate(rows, 1):
        frame = row.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError(f"{where}: row {index} has missing/non-integer analysis frame")
        if frame in seen:
            raise ValueError(f"{where}: duplicate analysis frame {frame}")
        seen.add(frame)


def _load_stage_rows(project_root: Path, acquisition: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Path, str]:
    stage = acquisition.get("alignment") or acquisition.get("result")
    if not isinstance(stage, Mapping):
        raise ValueError(f"Acquisition {acquisition.get('acquisition_id')} has no result rows")
    rows_path = stage.get("rows_path")
    if not rows_path:
        raise ValueError(f"Acquisition {acquisition.get('acquisition_id')} result has no rows_path")
    path = _relative_file(project_root, rows_path)
    if not path.is_file():
        raise FileNotFoundError(f"Campaign result rows are missing: {rows_path}")
    rows = _read_jsonl(path)
    _validate_unique_frames(rows, f"Acquisition {acquisition.get('acquisition_id')}")
    return rows, path, str(stage.get("status", "UNKNOWN"))


def _csv_text(value: Any, field: str) -> str:
    value = _json_safe(value)
    if value is None:
        lower = field.lower()
        if field == "stress_MPa" or ("stress" in lower and "mpa" in lower and "candidate" not in lower):
            return "null"
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return str(value)


def _write_frames_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot export an acquisition with no frame rows")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_text(row.get(key), key) for key in fields})


def _write_frames_jsonl(path: Path, rows: list[dict[str, Any]], original_path: Path | None = None, preserve_bytes: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if preserve_bytes and original_path is not None:
        shutil.copyfile(original_path, path)
        return
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False, separators=(",", ":")))
            handle.write("\n")


def _has_dic(acquisition: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> bool:
    imported = acquisition.get("metadata", {}).get("keystone_import", {}) if isinstance(acquisition.get("metadata"), Mapping) else {}
    if isinstance(imported, Mapping) and "dic_present" in imported:
        return bool(imported["dic_present"])
    if acquisition.get("image_frames"):
        return True
    if any(row.get(key) is not None for row in rows for key in ("engineering_strain", "engineering_strain_mean", "dic_engineering_strain")):
        return True
    return False


def _origin_fields(rows: list[Mapping[str, Any]], has_dic: bool) -> tuple[list[str], list[list[Any]]]:
    strain_key = next((key for key in ("engineering_strain", "engineering_strain_mean", "dic_engineering_strain") if any(key in row for row in rows)), None)
    fields = ["frame", "stress_MPa"] + (["engineering_strain"] if has_dic else [])
    values: list[list[Any]] = []
    for row in rows:
        strain = row.get(strain_key) if strain_key else None
        values.append([row.get("frame"), row.get("stress_MPa"), strain] if has_dic else [row.get("frame"), row.get("stress_MPa")])
    return fields, values


def _write_origin(path: Path, rows: list[Mapping[str, Any]], has_dic: bool) -> None:
    fields, values = _origin_fields(rows, has_dic)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        for row in values:
            writer.writerow([_csv_text(value, field) for value, field in zip(row, fields)])


def _verify_origin(path: Path, rows: list[Mapping[str, Any]], has_dic: bool, acquisition_id: str) -> list[str]:
    errors: list[str] = []
    expected_fields, expected_values = _origin_fields(rows, has_dic)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, [])
            values = list(reader)
        if header != expected_fields:
            errors.append(f"{acquisition_id}: Origin header does not match DIC presence")
            return errors
        if len(values) != len(expected_values):
            errors.append(f"{acquisition_id}: Origin row count mismatch")
            return errors
        for index, (actual, expected) in enumerate(zip(values, expected_values), 1):
            expected_tokens = [_csv_text(value, field) for value, field in zip(expected, expected_fields)]
            if actual != expected_tokens:
                errors.append(f"{acquisition_id}: Origin value mismatch at frame row {index}")
                return errors
    except Exception as exc:
        errors.append(f"{acquisition_id}: Origin file could not be read: {exc}")
    return errors


def _normalized_external_path(value: Any) -> str:
    return str(value).replace("\\", "/").rstrip("/").casefold()


def _raw_image_path_set(acquisition: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for frame in acquisition.get("image_frames", []) or []:
        if isinstance(frame, Mapping):
            for key in ("path", "image_path", "source_path"):
                if frame.get(key):
                    values.add(_normalized_external_path(frame[key]))
    for frame in acquisition.get("measurement_frames", []) or []:
        if isinstance(frame, Mapping):
            for key in ("path", "cbf_source_file", "image_path", "source_image_path"):
                if frame.get(key):
                    values.add(_normalized_external_path(frame[key]))
    return values


def _copy_analysis_artifacts(
    project_root: Path,
    version_dir: Path,
    sample_dir: Path,
    relative_sample_dir: str,
    acquisition: Mapping[str, Any],
    primary_files: Sequence[Path],
) -> list[dict[str, Any]]:
    """Copy only artifacts linked from the current result/alignment/ROI records."""
    raw_paths = _raw_image_path_set(acquisition)
    primary_hashes = {_sha256(path) for path in primary_files if path.is_file()}
    copied_by_source: dict[str, dict[str, Any]] = {}

    def copy_one(stage_name: str, value: Any, *, expected_hash: str | None = None) -> None:
        if not isinstance(value, (str, os.PathLike)):
            raise ValueError(f"{stage_name} artifact path must be a relative path: {value!r}")
        source_rel = str(value).replace("\\", "/")
        source = _relative_file(project_root, source_rel)
        normalized = _normalized_external_path(source)
        if normalized in raw_paths or source.suffix.lower() in _RAW_IMAGE_SUFFIXES:
            return
        if not source.is_file():
            raise FileNotFoundError(f"Linked {stage_name} artifact is missing: {source_rel}")
        digest = _sha256(source)
        if expected_hash is not None and expected_hash != digest:
            raise ValueError(f"Linked {stage_name} artifact checksum mismatch: {source_rel}")
        # The main frame tables and Origin export already have canonical copies.
        if digest in primary_hashes:
            return
        existing = copied_by_source.get(normalized)
        if existing is not None:
            if stage_name not in existing["stages"]:
                existing["stages"].append(stage_name)
            return
        destination_rel = f"{relative_sample_dir}/analysis_artifacts/{stage_name}/{source_rel}"
        destination = _relative_file(version_dir, destination_rel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied_by_source[normalized] = {
            "stages": [stage_name],
            "source_project_relative_path": source_rel,
            "package_path": destination.relative_to(version_dir).as_posix(),
            "sha256": digest,
            "bytes": destination.stat().st_size,
        }

    for stage_name in ("result", "alignment"):
        stage = acquisition.get(stage_name)
        if not isinstance(stage, Mapping):
            continue
        status = str(stage.get("status", "")).upper()
        if status not in {"COMPLETE", "COMPLETED", "IMPORTED_ACCEPTED"}:
            continue
        # Imported accepted JSONL and metadata are already preserved in their
        # canonical locations; its artifact list names those same files.
        if stage_name == "result" and status == "IMPORTED_ACCEPTED":
            continue
        artifacts = stage.get("artifacts", [])
        if isinstance(artifacts, Mapping):
            artifacts = list(artifacts.values())
        if not isinstance(artifacts, list):
            raise ValueError(f"{stage_name}.artifacts must be an array")
        row_path = str(stage.get("rows_path", "")).replace("\\", "/")
        checksums = stage.get("checksums", {}) if isinstance(stage.get("checksums"), Mapping) else {}
        for artifact in artifacts:
            if not isinstance(artifact, (str, os.PathLike)):
                raise ValueError(f"{stage_name}.artifacts contains a non-path value")
            artifact_rel = str(artifact).replace("\\", "/")
            if artifact_rel == row_path:
                continue
            copy_one(stage_name, artifact_rel, expected_hash=checksums.get(artifact_rel))

    roi = acquisition.get("roi")
    if isinstance(roi, Mapping):
        previews = roi.get("preview_paths", [])
        if not isinstance(previews, list):
            raise ValueError("roi.preview_paths must be an array")
        for preview in previews:
            copy_one("roi", preview)
    return sorted(copied_by_source.values(), key=lambda record: record["package_path"])


def _stage_name(project: Mapping[str, Any]) -> str:
    metadata = project.get("metadata", {}) if isinstance(project.get("metadata"), Mapping) else {}
    explicit = metadata.get("output_version") or metadata.get("version_id")
    if explicit:
        value = str(explicit)
        if Path(value).name != value or value in ("", ".", ".."):
            raise ValueError(f"Unsafe campaign output version: {value!r}")
        return value
    created = project.get("created_at") or metadata.get("created_at")
    if isinstance(created, str):
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _root_files(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "versions").mkdir(exist_ok=True)
    if not (root / "README.md").exists():
        (root / "README.md").write_text(
            "# StrainTrace Campaign Results\n\n"
            "Read `CURRENT.json` to find the verified current version. Each version is immutable after promotion.\n",
            encoding="utf-8",
        )
    if not (root / "AI_README.md").exists():
        (root / "AI_README.md").write_text(
            "# AI reading guide\n\n"
            "Resolve `CURRENT.json.version_path` first. Preserve frame identities and null/blank semantics. "
            "Do not infer physical specimen identity, times, area, or missing measurements.\n",
            encoding="utf-8",
        )
    if not (root / "catalog.json").exists():
        _write_json(root / "catalog.json", {
            "schema_version": _CATALOG_LINK_SCHEMA,
            "kind": "version_catalog_pointer",
            "current_pointer": _CURRENT,
            "version_path_field": "version_path",
            "version_catalog_json": "{version_path}/catalog.json",
            "version_catalog_csv": "{version_path}/catalog.csv",
        })
    if not (root / "catalog.csv").exists():
        with (root / "catalog.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["kind", "pointer", "json_template", "csv_template"])
            writer.writerow(["current_version", _CURRENT, "{version_path}/catalog.json", "{version_path}/catalog.csv"])


def _root_catalog_errors(root: Path) -> list[str]:
    errors: list[str] = []
    expected_json = {
        "schema_version": _CATALOG_LINK_SCHEMA,
        "kind": "version_catalog_pointer",
        "current_pointer": _CURRENT,
        "version_path_field": "version_path",
        "version_catalog_json": "{version_path}/catalog.json",
        "version_catalog_csv": "{version_path}/catalog.csv",
    }
    try:
        if _read_json(root / "catalog.json") != expected_json:
            errors.append("root catalog.json is not the canonical CURRENT-relative catalog pointer")
    except Exception as exc:
        errors.append(f"root catalog.json could not be read: {exc}")
    try:
        with (root / "catalog.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        expected_csv = [
            ["kind", "pointer", "json_template", "csv_template"],
            ["current_version", _CURRENT, "{version_path}/catalog.json", "{version_path}/catalog.csv"],
        ]
        if rows != expected_csv:
            errors.append("root catalog.csv does not link to the canonical CURRENT-relative catalogs")
    except Exception as exc:
        errors.append(f"root catalog.csv could not be read: {exc}")
    return errors


def _build_version(project_root: Path, project: Mapping[str, Any], version_dir: Path, version_id: str) -> dict[str, Any]:
    acquisitions = project.get("acquisitions")
    if not isinstance(acquisitions, list) or not acquisitions:
        raise ValueError("Cannot export an empty campaign")
    output_samples = version_dir / "samples"
    output_samples.mkdir(parents=True)
    catalog: list[dict[str, Any]] = []
    acquisition_ids: set[str] = set()
    relative_paths: set[str] = set()
    frame_total = 0
    for acquisition in acquisitions:
        if not isinstance(acquisition, Mapping):
            raise ValueError("Each campaign acquisition must be an object")
        acquisition_id = str(acquisition.get("acquisition_id") or "")
        if not acquisition_id or acquisition_id in acquisition_ids:
            raise ValueError(f"Missing or duplicate acquisition_id: {acquisition_id!r}")
        acquisition_ids.add(acquisition_id)
        rel = _relative_acquisition_path(acquisition.get("relative_path"), acquisition_id)
        if rel.casefold() in relative_paths:
            raise ValueError(f"Duplicate acquisition relative_path: {rel}")
        relative_paths.add(rel.casefold())
        rows, source_rows_path, stage_status = _load_stage_rows(project_root, acquisition)
        has_dic = _has_dic(acquisition, rows)
        sample_dir = _relative_file(output_samples, rel)
        sample_dir.mkdir(parents=True)
        imported_meta = acquisition.get("metadata", {}).get("keystone_import", {}) if isinstance(acquisition.get("metadata"), Mapping) else {}
        imported_dir: Path | None = None
        if isinstance(imported_meta, Mapping) and imported_meta.get("preserved_files"):
            jsonl_rel = acquisition.get("result", {}).get("rows_path") if isinstance(acquisition.get("result"), Mapping) else None
            if jsonl_rel and str(jsonl_rel).replace("\\", "/").endswith("/frames.jsonl"):
                imported_dir = _relative_file(project_root, jsonl_rel).parent

        rel_sample_dir = sample_dir.relative_to(version_dir).as_posix()
        first_recorded = next((row for row in rows if row.get("timestamp_unix_s") is not None), None)
        first_estimated = next((row for row in rows if row.get("estimated_timestamp_unix_s") is not None), None)
        sample_payload = {
            "schema_version": "straintrace-campaign-sample-1",
            "acquisition_id": acquisition_id,
            "name": acquisition.get("name"),
            "original_sample_name": acquisition.get("metadata", {}).get("original_sample_name", acquisition.get("name")) if isinstance(acquisition.get("metadata"), Mapping) else acquisition.get("name"),
            "relative_path": rel,
            "source_kind": acquisition.get("source_kind"),
            "physical_specimen_id": acquisition.get("physical_specimen_id"),
            "dimensions": acquisition.get("dimensions", {}),
            "analysis_mode": acquisition.get("analysis_mode"),
            "dic_present": has_dic,
            "acquisition_timing": {
                "first_recorded_timestamp_unix_s": first_recorded.get("timestamp_unix_s") if first_recorded else None,
                "first_recorded_timestamp_status": first_recorded.get("timestamp_status") if first_recorded else None,
                "first_recorded_timestamp_source": first_recorded.get("timestamp_source") if first_recorded else None,
                "first_estimated_timestamp_unix_s": first_estimated.get("estimated_timestamp_unix_s") if first_estimated else None,
                "first_estimated_timestamp_status": first_estimated.get("estimated_timestamp_status") if first_estimated else None,
                "first_estimated_timestamp_source": first_estimated.get("estimated_timestamp_source") if first_estimated else None,
            },
            "roi": acquisition.get("roi"),
            "approval": acquisition.get("approval"),
            "analysis_config": acquisition.get("analysis_config", {}),
            "result_status": stage_status,
            "metadata": acquisition.get("metadata", {}),
        }
        _write_json(sample_dir / "sample.json", sample_payload)
        _write_frames_csv(sample_dir / "frames.csv", rows)
        preserve_original = bool(
            isinstance(imported_meta, Mapping)
            and imported_meta.get("schema_version") == _IMPORT_SCHEMA
            and acquisition.get("alignment") is None
            and acquisition.get("result", {}).get("status") == "IMPORTED_ACCEPTED"
            and imported_dir is not None
            and (imported_dir / "frames.jsonl").is_file()
            and _sha256(imported_dir / "frames.jsonl") == _sha256(source_rows_path)
        )
        _write_frames_jsonl(sample_dir / "frames.jsonl", rows, source_rows_path, preserve_bytes=preserve_original)
        origin_filename = "frame_stress_strain.txt" if has_dic else "frame_stress.txt"
        _write_origin(sample_dir / origin_filename, rows, has_dic)
        copied_analysis_artifacts = _copy_analysis_artifacts(
            project_root,
            version_dir,
            sample_dir,
            rel_sample_dir,
            acquisition,
            [sample_dir / "frames.csv", sample_dir / "frames.jsonl", sample_dir / origin_filename],
        )

        source_payload: dict[str, Any] = {
            "schema_version": "straintrace-campaign-sources-1",
            "acquisition_id": acquisition_id,
            "result": _json_safe((acquisition.get("alignment") or acquisition.get("result") or {}).get("provenance", {})),
            "result_status": stage_status,
            "active_stage": "alignment" if acquisition.get("alignment") is not None else "result",
            "result_rows_source": str(source_rows_path.relative_to(project_root).as_posix()),
            "result_rows_source_path_basis": "campaign_project_root",
            "result_rows_in_package": f"{rel_sample_dir}/frames.jsonl",
            "result_rows_sha256": _sha256(source_rows_path),
            "package_rows_sha256": _sha256(sample_dir / "frames.jsonl"),
            "checksums": _json_safe((acquisition.get("alignment") or acquisition.get("result") or {}).get("checksums", {})),
            "analysis_result_provenance": _json_safe((acquisition.get("result") or {}).get("provenance", {})) if isinstance(acquisition.get("result"), Mapping) else {},
            "alignment_provenance": _json_safe((acquisition.get("alignment") or {}).get("provenance", {})) if isinstance(acquisition.get("alignment"), Mapping) else {},
            "analysis_artifacts": copied_analysis_artifacts,
            "sources": _json_safe(acquisition.get("metadata", {}).get("sources", [])) if isinstance(acquisition.get("metadata"), Mapping) else [],
        }
        if imported_dir is not None:
            for source_name in ("sample.json", "sources.json"):
                original = imported_dir / source_name
                if original.is_file():
                    dest = sample_dir / "keystone_import" / source_name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, dest)
                    source_payload.setdefault("preserved_keystone_metadata", {})[dest.relative_to(version_dir).as_posix()] = _sha256(dest)
            source_payload["historical_source_claims"] = _json_safe(imported_meta.get("historical_source_claims", []))
            source_payload["historical_claims_rehashed"] = False
        _write_json(sample_dir / "sources.json", source_payload)

        valid_stress = sum(row.get("stress_MPa") is not None for row in rows)
        strain_key = next((key for key in ("engineering_strain", "engineering_strain_mean", "dic_engineering_strain") if any(key in row for row in rows)), None)
        valid_strain = sum(row.get(strain_key) is not None for row in rows) if strain_key else 0
        catalog_row = {
            "acquisition_id": acquisition_id,
            "sample_name": acquisition.get("name"),
            "relative_path": rel,
            "package_path": rel_sample_dir,
            "sample_json": f"{rel_sample_dir}/sample.json",
            "frames_csv": f"{rel_sample_dir}/frames.csv",
            "frames_jsonl": f"{rel_sample_dir}/frames.jsonl",
            "sources_json": f"{rel_sample_dir}/sources.json",
            "origin_txt": f"{rel_sample_dir}/{origin_filename}",
            "frames": len(rows),
            "stress_valid": valid_stress,
            "strain_valid": valid_strain,
            "valid_pairs": sum(row.get("stress_MPa") is not None and (row.get(strain_key) is not None if strain_key else False) for row in rows),
            "DIC_present": has_dic,
            "result_status": stage_status,
        }
        catalog.append(catalog_row)
        frame_total += len(rows)

    _write_json(version_dir / "catalog.json", catalog)
    with (version_dir / "catalog.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(catalog[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: _csv_text(item.get(key), key) for key in fieldnames} for item in catalog)
    _write_json(version_dir / "schema.json", {
        "schema_version": _PACKAGE_SCHEMA,
        "units": {"force_N": "N", "stress_MPa": "MPa", "engineering_strain": "dimensionless", "true_strain": "dimensionless", "area_mm2": "mm^2"},
        "missing_values": {"csv_stress": "literal null", "csv_strain": "empty field", "json_numeric": "null; NaN and Infinity forbidden"},
        "frame_identity": "frame is the analysis/source index; native_cbf_frame and scan_id namespace remain distinct",
        "timestamp_policy": "recorded and estimated timestamps remain separate; provenance states source and timezone",
    })
    _write_json(version_dir / "PACKAGE.json", {
        "schema_version": _PACKAGE_SCHEMA,
        "version_id": version_id,
        "project_id": project.get("project_id"),
        "project_name": project.get("name"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "acquisition_count": len(catalog),
        "frame_count": frame_total,
        "source_data_copied": "metadata and accepted result tables only; raw images are referenced, not copied",
    })
    return {"acquisition_count": len(catalog), "frame_count": frame_total, "catalog": catalog}


def _write_checksum_manifest(version_dir: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(version_dir.rglob("*")):
        if path.is_file() and path.name != _CHECKSUMS:
            checksums[path.relative_to(version_dir).as_posix()] = _sha256(path)
    _write_json(version_dir / _CHECKSUMS, {"schema_version": "straintrace-checksums-1", "files": checksums})
    return checksums


def verify_package(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify a package directory, package root, or emitted ZIP artifact."""
    errors: list[str] = []
    temp: tempfile.TemporaryDirectory[str] | None = None
    try:
        candidate = Path(path).expanduser().resolve()
        root_current: dict[str, Any] | None = None
        if candidate.is_dir() and (candidate / _CURRENT).is_file():
            root_current = _read_json(candidate / _CURRENT)
            expected_path = f"versions/{root_current.get('version_id')}"
            if root_current.get("version_path") != expected_path:
                errors.append("CURRENT.json version_path does not match the immutable versions/<version_id> layout")
            if root_current.get("catalog_json") != f"{expected_path}/catalog.json" or root_current.get("catalog_csv") != f"{expected_path}/catalog.csv":
                errors.append("CURRENT.json catalog links do not point inside its selected version")
            errors.extend(_root_catalog_errors(candidate))
        version, temp = _resolve_package(path)
        if not version.is_dir():
            raise FileNotFoundError(version)
        for required in ("PACKAGE.json", "catalog.json", "catalog.csv", "schema.json", "samples", _CHECKSUMS):
            if not (version / required).exists():
                errors.append(f"missing required package item: {required}")
        if errors:
            return {"valid": False, "path": str(path), "errors": errors}
        manifest = _read_json(version / _CHECKSUMS)
        checksums = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(checksums, dict):
            errors.append("CHECKSUMS.json has no files map")
            checksums = {}
        actual_files = {p.relative_to(version).as_posix() for p in version.rglob("*") if p.is_file() and p.name != _CHECKSUMS}
        if set(checksums) != actual_files:
            missing = sorted(actual_files - set(checksums))[:5]
            extra = sorted(set(checksums) - actual_files)[:5]
            errors.append(f"checksum inventory mismatch (unlisted={missing}, missing_files={extra})")
        for rel, expected in checksums.items():
            try:
                file = _relative_file(version, rel)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not file.is_file():
                errors.append(f"checksum target missing: {rel}")
            elif _sha256(file) != expected:
                errors.append(f"checksum mismatch: {rel}")

        catalog = _load_catalog(version)
        _verify_catalog_csv(version / "catalog.csv", catalog, errors)
        ids: set[str] = set()
        paths: set[str] = set()
        frame_total = 0
        acquisition_total = 0
        for item in catalog:
            acquisition_id = str(item.get("acquisition_id") or "")
            rel = str(item.get("relative_path") or "")
            if not acquisition_id or acquisition_id in ids:
                errors.append(f"missing or duplicate acquisition_id in catalog: {acquisition_id!r}")
            ids.add(acquisition_id)
            normalized = _relative_acquisition_path(rel, acquisition_id)
            if normalized.casefold() in paths:
                errors.append(f"duplicate acquisition path in catalog: {normalized}")
            paths.add(normalized.casefold())
            package_rel = str(item.get("package_path") or f"samples/{normalized}")
            try:
                sample_dir = _relative_file(version, package_rel)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            artifact_links = {
                "sample_json": "sample.json",
                "frames_csv": "frames.csv",
                "frames_jsonl": "frames.jsonl",
                "sources_json": "sources.json",
            }
            for link_field, filename in artifact_links.items():
                expected_link = f"{package_rel}/{filename}"
                link = item.get(link_field)
                if link != expected_link:
                    errors.append(f"{acquisition_id}: catalog {link_field} does not resolve to {expected_link}")
                else:
                    try:
                        if _relative_file(version, link) != sample_dir / filename:
                            errors.append(f"{acquisition_id}: catalog {link_field} resolves outside the sample directory")
                    except ValueError as exc:
                        errors.append(f"{acquisition_id}: {exc}")
            required = ["sample.json", "frames.csv", "frames.jsonl", "sources.json"]
            for filename in required:
                if not (sample_dir / filename).is_file():
                    errors.append(f"{acquisition_id}: missing {filename}")
            if errors and any(not (sample_dir / filename).is_file() for filename in required):
                continue
            try:
                sample = _read_json(sample_dir / "sample.json")
                if str(sample.get("acquisition_id")) != acquisition_id:
                    errors.append(f"{acquisition_id}: sample.json identity mismatch")
                sources = _read_json(sample_dir / "sources.json")
                if str(sources.get("acquisition_id")) != acquisition_id:
                    errors.append(f"{acquisition_id}: sources.json identity mismatch")
                packaged_rows_link = sources.get("result_rows_in_package")
                if packaged_rows_link != f"{package_rel}/frames.jsonl":
                    errors.append(f"{acquisition_id}: source rows link does not resolve to the packaged JSONL")
                elif _sha256(_relative_file(version, packaged_rows_link)) != sources.get("package_rows_sha256"):
                    errors.append(f"{acquisition_id}: packaged JSONL link/checksum mismatch")
                for link, digest in sources.get("preserved_keystone_metadata", {}).items():
                    linked_file = _relative_file(version, link)
                    if not linked_file.is_file() or _sha256(linked_file) != digest:
                        errors.append(f"{acquisition_id}: preserved source metadata link/checksum failed: {link}")
                seen_derived_links: set[str] = set()
                primary_names = {"sample.json", "frames.csv", "frames.jsonl", "sources.json", "frame_stress_strain.txt", "frame_stress.txt"}
                for artifact in sources.get("analysis_artifacts", []):
                    if not isinstance(artifact, dict):
                        errors.append(f"{acquisition_id}: analysis_artifacts contains a non-object record")
                        continue
                    package_link = artifact.get("package_path")
                    source_link = artifact.get("source_project_relative_path")
                    if not isinstance(package_link, str) or not isinstance(source_link, str):
                        errors.append(f"{acquisition_id}: derived artifact is missing its source/package link")
                        continue
                    source_parts = PurePosixPath(source_link.replace("\\", "/"))
                    if source_parts.is_absolute() or any(part in ("", ".", "..") for part in source_parts.parts):
                        errors.append(f"{acquisition_id}: unsafe source artifact link {source_link!r}")
                    try:
                        linked_file = _relative_file(version, package_link)
                        if linked_file.relative_to(sample_dir).as_posix() in primary_names:
                            errors.append(f"{acquisition_id}: derived artifact duplicates a canonical table")
                        if package_link in seen_derived_links:
                            errors.append(f"{acquisition_id}: duplicate derived artifact link {package_link}")
                        seen_derived_links.add(package_link)
                        if not linked_file.is_file() or _sha256(linked_file) != artifact.get("sha256"):
                            errors.append(f"{acquisition_id}: derived artifact link/checksum failed: {package_link}")
                        if int(artifact.get("bytes", -1)) != linked_file.stat().st_size:
                            errors.append(f"{acquisition_id}: derived artifact byte count mismatch: {package_link}")
                    except (ValueError, OSError) as exc:
                        errors.append(f"{acquisition_id}: invalid derived artifact link {package_link!r}: {exc}")
                rows = _read_jsonl(sample_dir / "frames.jsonl")
                _validate_unique_frames(rows, f"{acquisition_id}")
                _verify_frame_csv(sample_dir / "frames.csv", rows, errors, acquisition_id)
                if len(rows) != int(item.get("frames", len(rows))):
                    errors.append(f"{acquisition_id}: catalog frame count mismatch")
                if sample.get("relative_path") != normalized and sample.get("relative_path") != rel:
                    errors.append(f"{acquisition_id}: sample relative_path mismatch")
                is_dic = bool(item.get("DIC_present", sample.get("dic_present", False)))
                origin_name = "frame_stress_strain.txt" if is_dic else "frame_stress.txt"
                if not (sample_dir / origin_name).is_file():
                    errors.append(f"{acquisition_id}: expected Origin file is missing ({origin_name})")
                else:
                    errors.extend(_verify_origin(sample_dir / origin_name, rows, is_dic, acquisition_id))
                if item.get("origin_txt") != f"{package_rel}/{origin_name}":
                    errors.append(f"{acquisition_id}: catalog origin_txt does not link to the expected Origin file")
                frame_total += len(rows)
                acquisition_total += 1
            except Exception as exc:
                errors.append(f"{acquisition_id}: {type(exc).__name__}: {exc}")
        package = _read_json(version / "PACKAGE.json")
        if root_current is not None and package.get("version_id") != root_current.get("version_id"):
            errors.append("CURRENT.json version_id does not match PACKAGE.json")
        if int(package.get("acquisition_count", -1)) != acquisition_total:
            errors.append("PACKAGE.json acquisition_count mismatch")
        if int(package.get("frame_count", -1)) != frame_total:
            errors.append("PACKAGE.json frame_count mismatch")
        return {
            "valid": not errors,
            "path": str(path),
            "version_path": str(version),
            "acquisition_count": acquisition_total,
            "frame_count": frame_total,
            "checksum_file_count": len(checksums),
            "errors": errors,
        }
    except Exception as exc:
        return {"valid": False, "path": str(path), "errors": [f"{type(exc).__name__}: {exc}"]}
    finally:
        if temp is not None:
            temp.cleanup()


def _verify_catalog_csv(path: Path, rows: list[dict[str, Any]], errors: list[str]) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            actual = list(csv.DictReader(handle))
        if len(actual) != len(rows):
            errors.append("catalog.csv/catalog.json row count mismatch")
            return
        for index, (csv_row, json_row) in enumerate(zip(actual, rows), 1):
            for key, value in json_row.items():
                if csv_row.get(key) != _csv_text(value, key):
                    errors.append(f"catalog.csv row {index} field {key} differs from catalog.json")
                    return
    except Exception as exc:
        errors.append(f"catalog.csv could not be read: {exc}")


def _verify_frame_csv(path: Path, rows: list[dict[str, Any]], errors: list[str], acquisition_id: str) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            csv_rows = list(reader)
            fields = reader.fieldnames or []
        if len(csv_rows) != len(rows):
            errors.append(f"{acquisition_id}: frames.csv/JSONL row count mismatch")
            return
        expected_fields = {key for row in rows for key in row}
        if set(fields) != expected_fields:
            errors.append(f"{acquisition_id}: frames.csv/JSONL column inventory mismatch")
            return
        for index, (csv_row, json_row) in enumerate(zip(csv_rows, rows), 1):
            for field in fields:
                value = json_row.get(field)
                token = csv_row.get(field, "")
                expected = _csv_text(value, field)
                if value is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
                    try:
                        parsed = int(token) if isinstance(value, int) else float(token)
                    except (TypeError, ValueError):
                        errors.append(f"{acquisition_id}: frame row {index} field {field} is not numeric")
                        return
                    if parsed != value:
                        errors.append(f"{acquisition_id}: frame row {index} field {field} numeric round-trip mismatch")
                        return
                elif token != expected:
                    errors.append(f"{acquisition_id}: frame row {index} field {field} mask/value mismatch")
                    return
    except Exception as exc:
        errors.append(f"{acquisition_id}: frames.csv could not be verified: {exc}")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_current(root: Path, version_id: str, version_rel: str, version: Path) -> dict[str, Any]:
    payload = {
        "schema_version": _CURRENT_SCHEMA,
        "version_id": version_id,
        "version_path": version_rel,
        "catalog_json": f"{version_rel}/catalog.json",
        "catalog_csv": f"{version_rel}/catalog.csv",
        "validation_json": f"{version_rel}/{_VALIDATION}",
        "checksums_json": f"{version_rel}/{_CHECKSUMS}",
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_atomic(root / _CURRENT, _json_bytes(payload))
    return payload


def _zip_package(version: Path, target: Path, timeout_s: float) -> str:
    result = _run_io_worker("zip", [str(version), str(target)], timeout_s)
    return str(result["sha256"])


def _worker_command(arguments: list[str]) -> list[str]:
    """Select an importable Python worker or the frozen application's worker mode."""
    if getattr(sys, "frozen", False):
        app_exe = Path(sys.executable)
        console_exe = app_exe.with_name("ezDIC-cli.exe")
        if console_exe.is_file():
            return [str(console_exe), "--campaign-io-worker", *arguments]
        if sys.stdout is None:
            raise RuntimeError("Frozen GUI has no console stream and sibling ezDIC-cli.exe was not found")
        # A console frozen build can safely spawn itself when no sibling CLI
        # executable is bundled, while a windowed GUI must use the sibling.
        return [sys.executable, "--campaign-io-worker", *arguments]
    return [sys.executable, "-m", "straintrace_campaign.delivery", "--io-worker", *arguments]


def _run_io_worker(operation: str, arguments: list[str], timeout_s: float) -> dict[str, Any]:
    try:
        result = subprocess.run(
            _worker_command([operation, *arguments]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=max(0.1, float(timeout_s)),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Bounded {operation} worker exceeded {timeout_s}s") from exc
    if result.returncode != 0:
        raise RuntimeError(f"{operation} worker failed ({result.returncode}): {result.stderr[-2000:]}")
    try:
        response = json.loads(result.stdout, parse_constant=_reject_json_constant)
    except Exception as exc:
        raise RuntimeError(f"{operation} worker returned invalid receipt: {result.stdout[-1000:]!r}") from exc
    if not isinstance(response, dict) or response.get("ok") is not True:
        raise RuntimeError(f"{operation} worker returned a failure receipt: {response!r}")
    return response


def _copy_file_within_worker(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return
    if destination.exists():
        destination.unlink()
    shutil.copyfile(source, destination)
    if _sha256(destination) != expected_sha256:
        raise IOError(f"Copied file failed SHA-256 verification: {destination}")


@contextmanager
def _publish_lock(publish_root: Path):
    """Serialize writers per destination using OS-released advisory locks.

    ``PUBLISH_LOCK.json`` is a persistent owner receipt. The actual lock is an
    OS advisory lock, automatically released when a process exits or is killed;
    its prior owner text is therefore never used to decide whether to steal a
    lock. The file is intentionally retained to avoid delete/recreate races.
    """
    lock_path = publish_root / "PUBLISH_LOCK.json"
    handle = os.fdopen(os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600), "r+b", buffering=0)
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    handle.seek(0)
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Publish destination is locked by another writer: {lock_path}") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"Publish destination is locked by another writer: {lock_path}") from exc
        acquired = True
        host = socket.gethostname()
        token = uuid.uuid4().hex
        record = {
            "schema_version": "straintrace-publish-lock-1",
            "token": token,
            "host": host,
            "pid": os.getpid(),
            "state": "ACTIVE",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        def write_record(value: Mapping[str, Any]) -> None:
            handle.seek(0)
            handle.truncate()
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())

        write_record(record)
        try:
            yield record
        finally:
            released = dict(record)
            released["state"] = "RELEASED"
            released["released_at_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                write_record(released)
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                acquired = False
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def _worker_publish_locked(
    local_version: Path,
    publish_root: Path,
    version_id: str,
    expected: Mapping[str, str],
) -> dict[str, Any]:
    if Path(version_id).name != version_id or version_id in ("", ".", ".."):
        raise ValueError(f"Unsafe publish version id: {version_id!r}")
    versions = publish_root / "versions"
    versions.mkdir(exist_ok=True)
    final_dir = versions / version_id
    stage_dir = versions / f".{version_id}.staging"
    if final_dir.exists():
        final_report = verify_package(final_dir)
        if not final_report["valid"]:
            raise FileExistsError(f"Published version exists but is invalid: {final_dir}")
        for rel, digest in expected.items():
            target = final_dir.joinpath(*PurePosixPath(rel).parts)
            if not target.is_file() or _sha256(target) != digest:
                raise FileExistsError(f"Published immutable version differs at {rel}")
        promoted = final_dir
    else:
        stage_dir.mkdir(exist_ok=True)
        expected_paths = set(expected)
        # Remove leftovers from a previous interrupted attempt for another
        # package with the same version ID. This only touches our staging tree.
        for stale in sorted((item for item in stage_dir.rglob("*") if item.is_file()), reverse=True):
            rel = stale.relative_to(stage_dir).as_posix()
            if rel not in expected_paths:
                stale.unlink()
        for rel, digest in expected.items():
            source = local_version.joinpath(*PurePosixPath(rel).parts)
            target = stage_dir.joinpath(*PurePosixPath(rel).parts)
            _copy_file_within_worker(source, target, digest)
        stage_report = verify_package(stage_dir)
        if not stage_report["valid"]:
            raise ValueError("Staged publish package failed verification: " + "; ".join(stage_report["errors"]))
        os.replace(stage_dir, final_dir)
        promoted = final_dir
        final_report = verify_package(final_dir)
        if not final_report["valid"]:
            raise ValueError("Promoted package failed verification: " + "; ".join(final_report["errors"]))

    _root_files(publish_root)
    root_errors = _root_catalog_errors(publish_root)
    if root_errors:
        raise ValueError("Publish root index failed verification: " + "; ".join(root_errors))
    current = _atomic_current(publish_root, version_id, f"versions/{version_id}", promoted)
    return {
        "ok": True,
        "version_id": version_id,
        "version_path": str(promoted),
        "current": current,
        "verification": final_report,
        "file_count": len(expected),
    }


def _worker_publish(local_version: Path, publish_root: Path, version_id: str) -> dict[str, Any]:
    """Run every destination filesystem operation under one bounded child lock."""
    if Path(version_id).name != version_id or version_id in ("", ".", ".."):
        raise ValueError(f"Unsafe publish version id: {version_id!r}")
    local_report = verify_package(local_version)
    if not local_report["valid"]:
        raise ValueError("Local source package is invalid: " + "; ".join(local_report["errors"]))
    local_files = sorted(path for path in local_version.rglob("*") if path.is_file())
    expected = {path.relative_to(local_version).as_posix(): _sha256(path) for path in local_files}
    publish_root.mkdir(parents=True, exist_ok=True)
    with _publish_lock(publish_root):
        return _worker_publish_locked(local_version, publish_root, version_id, expected)


def _worker_zip(source_dir: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp.zip")
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(source_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(source_dir).as_posix())
        with zipfile.ZipFile(temp) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise ValueError(f"ZIP CRC verification failed: {bad_member}")
        report = verify_package(temp)
        if not report["valid"]:
            raise ValueError("ZIP package verification failed: " + "; ".join(report["errors"]))
        digest = _sha256(temp)
        os.replace(temp, target)
        report["path"] = str(target)
        return {"ok": True, "sha256": digest, "verification": report}
    finally:
        if temp.exists():
            temp.unlink()


def _worker_inspect(publish_root: Path) -> dict[str, Any]:
    report = verify_package(publish_root)
    current: dict[str, Any] | None = None
    current_path = publish_root / _CURRENT
    if current_path.is_file():
        try:
            current = _read_json(current_path)
        except Exception:
            current = None
    return {
        "ok": True,
        **report,
        "current": current,
        "current_version_id": current.get("version_id") if isinstance(current, dict) else None,
    }


def worker_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for bounded filesystem workers, including frozen builds.

    Frozen entrypoints call this with arguments following
    ``--campaign-io-worker``. Python subprocesses use ``-m`` and pass
    ``--io-worker``. Both spellings are accepted for direct tests and hosts.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("--campaign-io-worker", "--io-worker"):
        args.pop(0)
    if not args:
        print(json.dumps({"ok": False, "error": "missing worker operation"}, ensure_ascii=False))
        return 2
    operation, *rest = args
    try:
        if operation == "zip" and len(rest) == 2:
            response = _worker_zip(Path(rest[0]), Path(rest[1]))
        elif operation == "publish" and len(rest) == 3:
            response = _worker_publish(Path(rest[0]), Path(rest[1]), rest[2])
        elif operation == "inspect" and len(rest) == 1:
            response = _worker_inspect(Path(rest[0]))
        else:
            raise ValueError(f"Unsupported worker operation or arguments: {operation} {rest!r}")
        print(json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 1


def _publish_version(local_version: Path, publish_to: Path, version_id: str, timeout_s: float) -> dict[str, Any]:
    try:
        receipt = _run_io_worker("publish", [str(local_version), str(publish_to), version_id], timeout_s)
    except TimeoutError as timeout_error:
        # A worker timeout can race the final atomic CURRENT replacement. Probe
        # through another bounded child before reporting the outcome.
        try:
            current = _run_io_worker("inspect", [str(publish_to)], min(max(timeout_s, 0.1), 5.0))
        except Exception as inspect_error:
            raise TimeoutError(f"Publish worker timed out; destination state could not be read: {inspect_error}") from timeout_error
        if current.get("valid") and current.get("current_version_id") == version_id:
            return {"path": str(publish_to), "version_path": current["version_path"], "current": current.get("current"), "verification": current}
        raise TimeoutError(f"Publish worker timed out; previous current version remains {current.get('current_version_id')!r}") from timeout_error
    return {
        "path": str(publish_to),
        "version_path": receipt["version_path"],
        "current": receipt["current"],
        "verification": receipt["verification"],
        "file_count": receipt["file_count"],
    }


def export_project(
    project_dir: str | os.PathLike[str],
    project: Mapping[str, Any],
    output: str | os.PathLike[str],
    publish_to: str | os.PathLike[str] | None = None,
    timeout_s: float = 60,
) -> dict[str, Any]:
    """Build, verify, archive, and optionally publish one immutable version.

    ``output`` is the versioned local package root. If a publish destination is
    given, files are copied sequentially into a resumable staging directory;
    only a fully verified version can atomically replace its ``CURRENT.json``.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    project_root = Path(project_dir).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _root_files(output_root)
    version_id = _stage_name(project)
    versions = output_root / "versions"
    final_dir = versions / version_id
    if final_dir.exists():
        existing = verify_package(final_dir)
        if existing["valid"]:
            raise FileExistsError(f"Version is immutable and already exists: {final_dir}")
        raise FileExistsError(f"Version directory already exists but is invalid: {final_dir}")
    staging = versions / f".{version_id}.staging-{uuid.uuid4().hex}"
    archive_path = output_root / "archives" / f"{version_id}.zip"
    try:
        staging.mkdir()
        summary = _build_version(project_root, project, staging, version_id)
        validation = {
            "valid": True,
            "schema_version": "straintrace-package-validation-1",
            "version_id": version_id,
            "acquisition_count": summary["acquisition_count"],
            "frame_count": summary["frame_count"],
            "checks": ["strict_json_finite", "unique_acquisition_ids", "unique_frame_ids", "CSV_JSONL_value_and_mask_equivalence", "catalog_identity_and_counts"],
        }
        _write_json(staging / _VALIDATION, validation)
        _write_checksum_manifest(staging)
        validation_result = verify_package(staging)
        if not validation_result["valid"]:
            raise ValueError("Local staged package failed verification: " + "; ".join(validation_result["errors"]))
        # The zip is part of the deliverable receipt, not an input to CURRENT.
        zip_receipt = _run_io_worker("zip", [str(staging), str(archive_path)], timeout_s)
        archive_sha = str(zip_receipt["sha256"])
        zip_report = zip_receipt["verification"]
        os.replace(staging, final_dir)
        local_report = verify_package(final_dir)
        if not local_report["valid"]:
            raise ValueError("Final local version failed verification; CURRENT was not changed: " + "; ".join(local_report["errors"]))
        root_errors = _root_catalog_errors(output_root)
        if root_errors:
            raise ValueError("Local root index failed verification; CURRENT was not changed: " + "; ".join(root_errors))
        current = _atomic_current(output_root, version_id, f"versions/{version_id}", final_dir)
        publish_receipt = None
        if publish_to is not None:
            try:
                publish_receipt = _publish_version(final_dir, Path(publish_to).expanduser().resolve(), version_id, timeout_s)
            except Exception as exc:
                destination_state: dict[str, Any] | None = None
                try:
                    destination_state = _run_io_worker("inspect", [str(Path(publish_to).expanduser().resolve())], min(max(timeout_s, 0.1), 5.0))
                except Exception:
                    pass
                if destination_state and destination_state.get("valid") and destination_state.get("current_version_id") == version_id:
                    publication_status = "PUBLISHED_CURRENT_VERIFIED"
                elif destination_state and destination_state.get("valid"):
                    publication_status = "NOT_PROMOTED_PREVIOUS_CURRENT_VERIFIED"
                else:
                    publication_status = "PUBLICATION_STATE_UNCONFIRMED"
                failure = {
                    "schema_version": "straintrace-publication-failure-1",
                    "status": publication_status,
                    "version_id": version_id,
                    "publish_target": str(Path(publish_to).expanduser().resolve()),
                    "local_version_path": str(final_dir),
                    "local_archive_path": str(archive_path),
                    "local_archive_sha256": archive_sha,
                    "destination_current_version_id": destination_state.get("current_version_id") if destination_state else None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "local_package_verified": True,
                    "raw_images_copied": False,
                }
                _write_json(output_root / f"PUBLICATION_{version_id}.json", failure)
                raise
        return {
            "version_id": version_id,
            "version_path": str(final_dir),
            "archive_path": str(archive_path),
            "archive_sha256": archive_sha,
            "archive_verification": zip_report,
            "current": current,
            "verification": local_report,
            "publish": publish_receipt,
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(worker_main())
