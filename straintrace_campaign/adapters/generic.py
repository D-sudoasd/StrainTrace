"""Configurable discovery for image-sequence acquisition batches.

The filename counter is the frame identity.  Discovery reports gaps and
conflicts instead of replacing missing counters with row ordinals.
"""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_IMAGE_PATTERNS = ("**/*.tif", "**/*.tiff", "**/*.png", "**/*.jpg", "**/*.jpeg")
DEFAULT_FRAME_REGEX = r"(?P<frame>\d+)$"
_MISSING = {"", "na", "n/a", "nan", "none", "null", "<no-data>"}


def _config(pattern: Any) -> dict[str, Any]:
    if pattern is None:
        return {"image_patterns": list(DEFAULT_IMAGE_PATTERNS)}
    if isinstance(pattern, str):
        return {"image_patterns": [pattern]}
    if not isinstance(pattern, dict):
        raise TypeError("pattern must be a glob string or configuration mapping")
    config = dict(pattern)
    if "image_patterns" not in config:
        value = config.get("image_pattern", config.get("glob"))
        config["image_patterns"] = [value] if isinstance(value, str) else list(value or DEFAULT_IMAGE_PATTERNS)
    if isinstance(config["image_patterns"], str):
        config["image_patterns"] = [config["image_patterns"]]
    return config


def _is_excluded(path: Path, root: Path, patterns: Iterable[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    rel_casefold = rel.casefold()
    parts = {part.casefold() for part in path.relative_to(root).parts}
    for item in patterns:
        candidate = str(item).replace("\\", "/")
        if candidate.casefold() in parts:
            return True
        if fnmatch.fnmatchcase(rel, candidate) or fnmatch.fnmatchcase(rel_casefold, candidate.casefold()):
            return True
    return False


def _counter(path: Path, expression: re.Pattern[str]) -> int | None:
    match = expression.search(path.stem)
    if not match:
        return None
    value = match.groupdict().get("frame")
    if value is None:
        if match.lastindex == 1:
            value = match.group(1)
        elif match.lastindex:
            raise ValueError("frame_regex must use a named 'frame' group or exactly one capture group")
        else:
            value = match.group(0)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _range_list(frames: list[int]) -> list[list[int]]:
    if len(frames) < 2:
        return []
    result: list[list[int]] = []
    unique = sorted(set(frames))
    for left, right in zip(unique, unique[1:]):
        if right > left + 1:
            result.append([left + 1, right - 1])
    return result


def _parse_cell(value: str | None) -> Any:
    text = "" if value is None else str(value).strip()
    if text.casefold() in _MISSING:
        return None
    if text.casefold() in {"true", "yes", "y"}:
        return True
    if text.casefold() in {"false", "no", "n"}:
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def read_explicit_frame_mapping(
    csv_path: str | os.PathLike[str],
    *,
    columns: dict[str, str] | None = None,
    scan_id_is_global_fio: bool | None = None,
    scan_id_namespace: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read an explicit integration map without inferring counters by order.

    ``columns`` maps canonical fields (``frame``, ``native_cbf_frame``,
    ``native_dic_frame``, ``scan_id``, ``timestamp_unix_s``, ``time_s``,
    ``exposure_start_unix_s``, ``exposure_end_unix_s``, ``clock_id``) to CSV
    headers.  Only rows with an explicit integer ``frame`` are emitted.
    Duplicate analysis frame identifiers remain present and are marked as
    conflicts; no row is silently selected.
    """
    path = Path(csv_path)
    mapping = {
        "frame": "frame",
        "native_cbf_frame": "native_cbf_frame",
        "native_dic_frame": "native_dic_frame",
        "scan_id": "scan_id",
        "timestamp_unix_s": "timestamp_unix_s",
        "estimated_timestamp_unix_s": "estimated_timestamp_unix_s",
        "time_s": "time_s",
        "exposure_start_unix_s": "exposure_start_unix_s",
        "exposure_end_unix_s": "exposure_end_unix_s",
        "clock_id": "clock_id",
        "timestamp_status": "timestamp_status",
        "timestamp_source": "timestamp_source",
        "scan_id_is_global_fio": "scan_id_is_global_fio",
        "scan_id_namespace": "scan_id_namespace",
        "native_dic_frame": "native_dic_frame",
    }
    if columns:
        mapping.update({str(key): str(value) for key, value in columns.items()})
    frames: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for source_row, raw in enumerate(reader, start=2):
            converted = {key: _parse_cell(raw.get(header)) for key, header in mapping.items() if header in (reader.fieldnames or [])}
            # Keep noncanonical source columns losslessly for later provenance.
            for header in reader.fieldnames or []:
                if header not in mapping.values():
                    converted.setdefault(header, _parse_cell(raw.get(header)))
            try:
                raw_frame = converted.get("frame")
                frame = int(raw_frame)
                if isinstance(raw_frame, float) and raw_frame != frame:
                    raise ValueError("frame is not an integer")
            except (TypeError, ValueError):
                issues.append({"code": "MAPPING_FRAME_ID_MISSING", "source_row": source_row})
                continue
            if frame < 0:
                issues.append({"code": "MAPPING_FRAME_ID_INVALID", "source_row": source_row, "frame": frame})
                continue
            converted["frame"] = frame
            converted["source_mapping_row"] = source_row
            if "native_cbf_frame" in converted and converted["native_cbf_frame"] is not None:
                try:
                    raw_counter = converted["native_cbf_frame"]
                    converted["native_cbf_frame"] = int(raw_counter)
                    if isinstance(raw_counter, float) and raw_counter != converted["native_cbf_frame"]:
                        raise ValueError("native counter is not an integer")
                    if converted["native_cbf_frame"] < 0:
                        raise ValueError("native counter is negative")
                except (TypeError, ValueError):
                    issues.append({"code": "MAPPING_NATIVE_CBF_COUNTER_INVALID", "source_row": source_row})
                    continue
            if "native_dic_frame" in converted and converted["native_dic_frame"] is not None:
                try:
                    raw_counter = converted["native_dic_frame"]
                    converted["native_dic_frame"] = int(raw_counter)
                    if isinstance(raw_counter, float) and raw_counter != converted["native_dic_frame"]:
                        raise ValueError("native counter is not an integer")
                    if converted["native_dic_frame"] < 0:
                        raise ValueError("native counter is negative")
                except (TypeError, ValueError):
                    issues.append({"code": "MAPPING_NATIVE_DIC_COUNTER_INVALID", "source_row": source_row})
                    continue
            if converted.get("native_cbf_frame") is None:
                issues.append({"code": "MAPPING_NATIVE_CBF_COUNTER_MISSING", "source_row": source_row})
                continue
            if "scan_id" in converted and converted["scan_id"] is not None:
                if converted.get("scan_id_is_global_fio") is None:
                    converted["scan_id_is_global_fio"] = bool(scan_id_is_global_fio)
                if converted.get("scan_id_namespace") is None:
                    converted["scan_id_namespace"] = scan_id_namespace
            else:
                converted["scan_id"] = None
                converted["scan_id_is_global_fio"] = False
                converted["scan_id_namespace"] = None
            frames.append(converted)

    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in frames:
        by_frame.setdefault(row["frame"], []).append(row)
    duplicate_frames: set[int] = set()
    deduplicated: list[dict[str, Any]] = []
    for frame, rows in by_frame.items():
        if len(rows) == 1:
            deduplicated.append(rows[0])
        else:
            signatures = {
                tuple(sorted((key, repr(value)) for key, value in row.items() if key not in {"source_mapping_row", "identity_status"}))
                for row in rows
            }
            source_rows = [row["source_mapping_row"] for row in rows]
            if len(signatures) == 1:
                retained = dict(rows[0])
                retained["identity_status"] = "DUPLICATE_MAPPING_IDENTICAL_CONTENT_DEDUPED"
                retained["duplicate_source_mapping_rows"] = source_rows[1:]
                deduplicated.append(retained)
                issues.append({"code": "DUPLICATE_MAPPING_IDENTICAL_CONTENT_DEDUPED",
                               "frame": frame, "source_rows": source_rows})
            else:
                duplicate_frames.add(frame)
                issues.append({"code": "DUPLICATE_MAPPING_CONFLICT", "frame": frame,
                               "source_rows": source_rows, "rows": rows})
    by_native: dict[int, list[dict[str, Any]]] = {}
    for row in deduplicated:
        by_native.setdefault(int(row["native_cbf_frame"]), []).append(row)
    duplicate_native = {native for native, rows in by_native.items() if len(rows) > 1}
    for native in sorted(duplicate_native):
        rows = by_native[native]
        issues.append({
            "code": "DUPLICATE_NATIVE_CBF_COUNTER",
            "native_cbf_frame": native,
            "frames": [row["frame"] for row in rows],
            "rows": rows,
        })
    mapping_gaps = _range_list([row["frame"] for row in deduplicated])
    if mapping_gaps:
        issues.append({"code": "MAPPING_ANALYSIS_FRAME_GAPS", "missing_ranges": mapping_gaps})
    # Keep every disputed source row in the issue receipt, but do not emit an
    # ambiguous measurement axis that would violate the campaign frame key.
    return [
        row for row in deduplicated
        if row["frame"] not in duplicate_frames and row["native_cbf_frame"] not in duplicate_native
    ], issues


_FRAME_METADATA_FIELDS = (
    "filename",
    "frame",
    "native_frame",
    "timestamp_unix_s",
    "estimated_timestamp_unix_s",
    "time_s",
    "scan_id",
    "scan_id_is_global_fio",
    "scan_id_namespace",
    "clock_id",
    "timestamp_source",
    "timestamp_status",
    "exposure_start_unix_s",
    "exposure_end_unix_s",
)


def _read_frame_metadata(path: Path, columns: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mapping = {name: name for name in _FRAME_METADATA_FIELDS}
    if columns:
        mapping.update({str(key): str(value) for key, value in columns.items()})
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        for source_row, raw in enumerate(reader, start=2):
            row = {key: _parse_cell(raw.get(header)) for key, header in mapping.items() if header in fields}
            for header in fields:
                if header not in mapping.values():
                    row.setdefault(header, _parse_cell(raw.get(header)))
            row["source_metadata_row"] = source_row
            filename = row.get("filename")
            native_frame = row.get("native_frame")
            if filename is None and native_frame is None:
                issues.append({"code": "FRAME_METADATA_WITHOUT_IDENTITY", "source_row": source_row})
                continue
            if native_frame is not None:
                try:
                    row["native_frame"] = int(native_frame)
                    if row["native_frame"] < 0:
                        raise ValueError("native counter is negative")
                except (TypeError, ValueError):
                    issues.append({"code": "FRAME_METADATA_NATIVE_COUNTER_INVALID", "source_row": source_row})
                    continue
            if row.get("frame") is not None:
                try:
                    raw_frame = row["frame"]
                    row["frame"] = int(raw_frame)
                    if isinstance(raw_frame, float) and raw_frame != row["frame"]:
                        raise ValueError("frame is not an integer")
                    if row["frame"] < 0:
                        raise ValueError("frame is negative")
                except (TypeError, ValueError):
                    issues.append({"code": "FRAME_METADATA_ANALYSIS_COUNTER_INVALID", "source_row": source_row})
                    continue
            rows.append(row)
    return rows, issues


def _apply_frame_metadata(
    image_frames: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich images by explicit filename/native-counter keys only."""
    issues: list[dict[str, Any]] = []
    by_filename: dict[str, list[dict[str, Any]]] = {}
    by_native: dict[int, list[dict[str, Any]]] = {}
    for row in metadata_rows:
        filename = row.get("filename")
        if filename is not None:
            by_filename.setdefault(str(filename).replace("\\", "/").casefold(), []).append(row)
        if row.get("native_frame") is not None:
            by_native.setdefault(int(row["native_frame"]), []).append(row)

    for frame, rows in by_filename.items():
        if len(rows) > 1:
            issues.append({"code": "DUPLICATE_FRAME_METADATA_FILENAME", "filename": frame,
                           "source_rows": [row["source_metadata_row"] for row in rows]})
    for frame, rows in by_native.items():
        if len(rows) > 1:
            issues.append({"code": "DUPLICATE_FRAME_METADATA_NATIVE_COUNTER", "native_frame": frame,
                           "source_rows": [row["source_metadata_row"] for row in rows]})

    for image in image_frames:
        key = str(image["filename"]).replace("\\", "/").casefold()
        candidates = by_filename.get(key, []) if key in by_filename else by_native.get(int(image["native_frame"]), [])
        if len(candidates) != 1:
            if len(candidates) > 1:
                image["timestamp_status"] = "FRAME_METADATA_IDENTITY_CONFLICT"
                image["identity_status"] = "FRAME_METADATA_IDENTITY_CONFLICT"
            continue
        meta = candidates[0]
        declared_native = meta.get("native_frame")
        filename_native = int(image["native_frame"])
        if declared_native is not None and int(declared_native) != filename_native:
            image["filename_counter"] = filename_native
            image["native_frame"] = int(declared_native)
            image["frame_namespace"] = "frame_metadata.native_frame"
            image["identity_status"] = "EXPLICIT_METADATA_COUNTER_DIFFERS_FROM_FILENAME"
            issues.append({"code": "FILENAME_NATIVE_COUNTER_MISMATCH", "filename": image["filename"],
                           "filename_counter": filename_native, "metadata_native_frame": int(declared_native),
                           "metadata_source_row": meta["source_metadata_row"]})
        if meta.get("frame") is not None:
            image["frame"] = int(meta["frame"])
        for field in _FRAME_METADATA_FIELDS:
            if field in {"filename", "frame", "native_frame"}:
                continue
            if meta.get(field) is not None:
                image[field] = meta[field]
        image["frame_metadata_source_row"] = meta["source_metadata_row"]
    return issues


def discover(
    root: str | os.PathLike[str],
    pattern: Any = None,
    exclude: tuple[str, ...] | list[str] = (),
) -> list[dict[str, Any]]:
    """Discover image batches, retaining native counters and duplicate evidence.

    Configuration keys include ``image_patterns``, ``acquisition_parent_levels``,
    ``frame_regex``, ``source_kind``, ``measurement_map`` and
    ``measurement_columns``.  The default groups files in their immediate
    containing directory.  Use ``acquisition_parent_levels=1`` for layouts
    such as ``sample/DIC/image_000001.tif``.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)
    config = _config(pattern)
    excluded = tuple(str(item) for item in (*exclude, *config.get("exclude", ())))
    patterns = [str(item) for item in config["image_patterns"]]
    if not patterns:
        raise ValueError("At least one image pattern is required")
    levels = int(config.get("acquisition_parent_levels", 0))
    if levels < 0:
        raise ValueError("acquisition_parent_levels must be non-negative")
    try:
        frame_regex = re.compile(str(config.get("frame_regex", DEFAULT_FRAME_REGEX)))
    except re.error as exc:
        raise ValueError(f"Invalid frame_regex: {exc}") from exc

    files: set[Path] = set()
    for glob_pattern in patterns:
        files.update(path.resolve() for path in root_path.glob(glob_pattern) if path.is_file())
    files = {path for path in files if not _is_excluded(path, root_path, excluded)}

    groups: dict[Path, list[Path]] = {}
    for path in files:
        acquisition_dir = path.parent
        for _ in range(levels):
            if acquisition_dir == root_path or root_path not in acquisition_dir.parents:
                acquisition_dir = root_path
                break
            acquisition_dir = acquisition_dir.parent
        groups.setdefault(acquisition_dir, []).append(path)

    result: list[dict[str, Any]] = []
    for acquisition_dir, batch_files in sorted(groups.items(), key=lambda item: item[0].as_posix().casefold()):
        records: list[dict[str, Any]] = []
        unmapped: list[dict[str, Any]] = []
        by_counter: dict[int, list[dict[str, Any]]] = {}
        for path in sorted(batch_files, key=lambda item: item.as_posix().casefold()):
            counter = _counter(path, frame_regex)
            record = {
                "path": str(path),
                "source_relative_path": path.relative_to(root_path).as_posix(),
                "filename": path.name,
                "content_size_bytes": path.stat().st_size,
                "content_sha256": _sha256(path),
            }
            if counter is None:
                record["identity_status"] = "UNPARSED_NATIVE_COUNTER"
                unmapped.append(record)
                continue
            record.update({
                "frame": counter,
                "native_frame": counter,
                "frame_namespace": str(config.get("frame_namespace", "filename_counter")),
                "identity_status": "NATIVE_COUNTER_PARSED",
            })
            by_counter.setdefault(counter, []).append(record)

        issues: list[dict[str, Any]] = []
        image_frames: list[dict[str, Any]] = []
        for counter, candidates in sorted(by_counter.items()):
            signatures = {candidate["content_sha256"] for candidate in candidates}
            if len(candidates) == 1:
                image_frames.append(candidates[0])
            elif len(signatures) == 1:
                retained = candidates[0]
                retained["identity_status"] = "DUPLICATE_IDENTICAL_CONTENT_DEDUPED"
                retained["duplicate_source_paths"] = [candidate["source_relative_path"] for candidate in candidates[1:]]
                image_frames.append(retained)
                issues.append({
                    "code": "DUPLICATE_IDENTICAL_CONTENT",
                    "native_frame": counter,
                    "source_paths": [candidate["source_relative_path"] for candidate in candidates],
                    "content_sha256": retained["content_sha256"],
                })
            else:
                issues.append({
                    "code": "DUPLICATE_NATIVE_COUNTER_CONTENT_CONFLICT",
                    "native_frame": counter,
                    "source_paths": [candidate["source_relative_path"] for candidate in candidates],
                    "content_sha256": sorted(signatures),
                    "records": candidates,
                })
                for candidate in candidates:
                    candidate["identity_status"] = "NATIVE_COUNTER_CONTENT_CONFLICT"
                    unmapped.append(candidate)
        counters = sorted(by_counter)
        missing_ranges = _range_list(counters)
        if missing_ranges:
            issues.append({"code": "NATIVE_COUNTER_GAPS", "missing_ranges": missing_ranges})
        if unmapped:
            issues.append({"code": "UNPARSED_NATIVE_COUNTER_FILES", "count": len(unmapped)})

        try:
            relative = acquisition_dir.relative_to(root_path).as_posix()
        except ValueError:
            relative = "."
        name = acquisition_dir.name if relative != "." else root_path.name
        frame_metadata_path = config.get("frame_metadata", "frame_metadata.csv")
        metadata_issues: list[dict[str, Any]] = []
        if frame_metadata_path:
            metadata_path = Path(frame_metadata_path)
            if not metadata_path.is_absolute():
                metadata_path = acquisition_dir / metadata_path
            if metadata_path.is_file():
                metadata_rows, metadata_issues = _read_frame_metadata(
                    metadata_path, columns=config.get("frame_metadata_columns")
                )
                metadata_issues.extend(_apply_frame_metadata(image_frames, metadata_rows))

        conflicted = [frame for frame in image_frames if "CONFLICT" in str(frame.get("identity_status", ""))]
        if conflicted:
            metadata_issues.append({"code": "FRAME_METADATA_IDENTITY_CONFLICT", "records": conflicted})
            unmapped.extend(conflicted)
            image_frames = [frame for frame in image_frames if frame not in conflicted]

        by_analysis_frame: dict[int, list[dict[str, Any]]] = {}
        for frame in image_frames:
            by_analysis_frame.setdefault(int(frame["frame"]), []).append(frame)
        duplicate_analysis_frames = {key for key, rows in by_analysis_frame.items() if len(rows) > 1}
        for frame_no in sorted(duplicate_analysis_frames):
            records_for_frame = by_analysis_frame[frame_no]
            metadata_issues.append({"code": "DUPLICATE_ANALYSIS_FRAME_ID", "frame": frame_no,
                                    "records": records_for_frame})
            unmapped.extend(records_for_frame)
        if duplicate_analysis_frames:
            image_frames = [frame for frame in image_frames if int(frame["frame"]) not in duplicate_analysis_frames]
        for frame in image_frames:
            for field in (
                "timestamp_unix_s", "estimated_timestamp_unix_s", "time_s", "clock_id",
                "timestamp_source", "timestamp_status", "exposure_start_unix_s", "exposure_end_unix_s",
            ):
                frame.setdefault(field, None)
            if frame["timestamp_status"] is None and all(
                frame[field] is None for field in ("timestamp_unix_s", "estimated_timestamp_unix_s", "time_s")
            ):
                frame["timestamp_status"] = "NO_TIMESTAMP_SOURCE"

        acquisition: dict[str, Any] = {
            "name": name,
            "relative_path": relative if relative != "." else "_root_sequence",
            "source_kind": str(config.get("source_kind", "generic_image_sequence")),
            "physical_specimen_id": None,
            "dimensions": {
                "area_mm2": None,
                "width_mm": None,
                "thickness_mm": None,
                "status": "UNCONFIRMED",
                "source": None,
            },
            "analysis_mode": str(config.get("analysis_mode", "extensometer")),
            "image_frames": sorted(image_frames, key=lambda row: (row["frame"], row["path"].casefold())),
            "measurement_frames": [],
            "discovery_issues": [*issues, *metadata_issues],
            "unmapped_images": unmapped,
            "metadata": {
                "discovery_root_relative_path": relative,
                "native_frame_namespace": str(config.get("frame_namespace", "filename_counter")),
                "image_patterns": patterns,
                "missing_frame_ranges": missing_ranges,
            },
        }
        if frame_metadata_path:
            metadata_path = Path(frame_metadata_path)
            if not metadata_path.is_absolute():
                metadata_path = acquisition_dir / metadata_path
            if metadata_path.is_file():
                acquisition["metadata"]["frame_metadata_source"] = str(metadata_path.resolve())
        mapping_path = config.get("measurement_map", config.get("measurement_frames", "measurement_frames.csv"))
        if mapping_path:
            map_path = Path(mapping_path)
            if not map_path.is_absolute():
                map_path = acquisition_dir / map_path
            if map_path.is_file():
                measurement, map_issues = read_explicit_frame_mapping(
                    map_path,
                    columns=config.get("measurement_columns"),
                    scan_id_is_global_fio=config.get("scan_id_is_global_fio"),
                    scan_id_namespace=config.get("scan_id_namespace"),
                )
                acquisition["measurement_frames"] = measurement
                acquisition["discovery_issues"].extend(map_issues)
                acquisition["metadata"]["measurement_map_source"] = str(map_path.resolve())
        result.append(acquisition)
    return result


__all__ = ["discover", "read_explicit_frame_mapping"]
