"""Pure DESY FIO/load-log parsers and configurable Delun/camera discovery.

There are three distinct frame identities in common DESY exports: a Delun DIC
filename counter, a camera/CBF native counter, and a global FIO scan number.
This adapter records their namespaces separately and only joins them when an
explicit counter mapping or a unique colocated FIO file supports the join.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import math
import os
import re
from pathlib import Path
from typing import Any

from .generic import _is_excluded, _parse_cell, discover as generic_discover, read_explicit_frame_mapping


_FIO_COLUMN = re.compile(r"\s*Col\s+(\d+)\s+(\S+)(?:\s+\S+)?", re.IGNORECASE)
_FIO_SCAN = re.compile(r"(?:^|[/\\])eh3scan_(\d+)\.fio$", re.IGNORECASE)
_FIO_FOLDER = re.compile(r"^fio[_-]?(\d+)$", re.IGNORECASE)
_FIO_TOKEN = re.compile(r"<no-data>|None|[^\s]+", re.IGNORECASE)
_MISSING = {"", "nan", "none", "na", "n/a", "null", "<no-data>"}
_ABSOLUTE_EPOCH_MIN = 1_000_000_000.0


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None:
        return None
    rounded = round(number)
    return int(rounded) if abs(number - rounded) <= 1e-8 else None


def parse_time_value(value: Any) -> dict[str, Any]:
    """Classify a timestamp without turning a relative clock into Unix time."""
    number = _number(value)
    if number is not None:
        if number >= _ABSOLUTE_EPOCH_MIN:
            return {
                "timestamp_unix_s": number,
                "time_s": None,
                "timestamp_status": "RECORDED_ABSOLUTE_EPOCH",
                "timestamp_timezone": "UTC_EPOCH",
            }
        return {
            "timestamp_unix_s": None,
            "time_s": number,
            "timestamp_status": "RECORDED_RELATIVE_CLOCK",
            "timestamp_timezone": None,
        }
    text = "" if value is None else str(value).strip()
    if not text or text.casefold() in _MISSING:
        return {
            "timestamp_unix_s": None,
            "time_s": None,
            "timestamp_status": "MISSING_TIMESTAMP",
            "timestamp_timezone": None,
        }
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError:
        return {
            "timestamp_unix_s": None,
            "time_s": None,
            "timestamp_status": "UNPARSEABLE_TIMESTAMP",
            "timestamp_timezone": None,
            "timestamp_raw": text,
        }
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return {
            "timestamp_unix_s": None,
            "time_s": None,
            "timestamp_status": "NAIVE_DATETIME_TIMEZONE_UNKNOWN",
            "timestamp_timezone": None,
            "timestamp_raw": text,
        }
    return {
        "timestamp_unix_s": parsed.timestamp(),
        "time_s": None,
        "timestamp_status": "RECORDED_ABSOLUTE_TIMEZONE_AWARE",
        "timestamp_timezone": str(parsed.tzinfo),
    }


def parse_fio_text(text: str, source_path: str = "<memory>") -> dict[str, Any]:
    """Parse a SPEC/FIO ``%d`` data block into typed rows plus parser issues.

    The parser preserves source line numbers and reports malformed rows.  FIO
    detector counters and timestamps remain separate columns; no counter or
    time interval is inferred here.
    """
    lines = str(text).splitlines()
    data_markers = [index for index, line in enumerate(lines) if line.strip() == "%d"]
    scan_match = _FIO_SCAN.search(str(source_path).replace("\\", "/"))
    scan_id = int(scan_match.group(1)) if scan_match else None
    source_identity = {
        "scan_id": scan_id,
        "scan_id_namespace": "global_fio_scan_id" if scan_id is not None else None,
        "scan_id_is_global_fio": scan_id is not None,
    }
    if len(data_markers) != 1:
        return {**source_identity, "columns": [], "rows": [], "issues": [{
            "code": "FIO_DATA_MARKER_INVALID", "source_path": source_path,
            "count": len(data_markers),
        }]}
    cursor = data_markers[0] + 1
    columns: list[tuple[int, str]] = []
    while cursor < len(lines):
        match = _FIO_COLUMN.match(lines[cursor])
        if not match:
            if columns:
                break
            cursor += 1
            continue
        columns.append((int(match.group(1)), match.group(2)))
        cursor += 1
    if not columns:
        return {**source_identity, "columns": [], "rows": [], "issues": [{
            "code": "FIO_COLUMNS_MISSING", "source_path": source_path,
        }]}

    issues: list[dict[str, Any]] = []
    ids = [column_id for column_id, _ in columns]
    if ids != list(range(1, len(ids) + 1)):
        issues.append({"code": "FIO_COLUMN_NUMBER_SEQUENCE", "column_ids": ids})
    names = [name for _, name in columns]
    output_names: list[str] = []
    seen_names: dict[str, int] = {}
    if len(set(names)) != len(names):
        issues.append({"code": "FIO_DUPLICATE_COLUMN_NAMES", "column_names": names})
    for col_id, name in columns:
        seen_names[name] = seen_names.get(name, 0) + 1
        if seen_names[name] == 1:
            output_names.append(name)
        else:
            candidate = f"{name}__col{col_id}"
            while candidate in output_names:
                candidate = f"{candidate}_{seen_names[name]}"
            output_names.append(candidate)

    rows: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines[cursor:], start=cursor + 1):
        if not line.strip() or line.lstrip().startswith(("!", "#")):
            continue
        tokens = _FIO_TOKEN.findall(line)
        if len(tokens) != len(columns):
            issues.append({"code": "FIO_DATA_COLUMN_COUNT", "line": line_index,
                           "expected": len(columns), "found": len(tokens)})
            continue
        row: dict[str, Any] = {"source_line": line_index}
        for name, token in zip(output_names, tokens):
            if token.casefold() in _MISSING:
                row[name] = None
                continue
            value = _number(token)
            if value is not None:
                row[name] = value
                continue
            time_info = parse_time_value(token)
            if time_info["timestamp_status"] == "RECORDED_ABSOLUTE_TIMEZONE_AWARE":
                row[name] = token
            elif time_info["timestamp_status"] == "NAIVE_DATETIME_TIMEZONE_UNKNOWN":
                row[name] = token
                issues.append({"code": "FIO_NAIVE_TIMESTAMP_VALUE", "line": line_index,
                               "column": name, "value": token})
            else:
                row[name] = None
                issues.append({"code": "FIO_NONNUMERIC_VALUE", "line": line_index,
                               "column": name, "value": token})
        rows.append(row)
    return {**source_identity, "columns": output_names, "rows": rows, "issues": issues}


def parse_load_table_text(
    text: str,
    *,
    time_column: str,
    force_column: str,
    force_unit: str | None = None,
    delimiter: str | None = None,
) -> dict[str, Any]:
    """Parse a headered load CSV/log table with explicit time and force fields.

    Force is normalized to N only when the source column is explicitly named
    ``force_N`` or its declared unit is recognized.  Unknown units leave
    ``force_N`` null and retain the raw value/status for review.
    """
    content = str(text)
    if delimiter is None:
        try:
            dialect = csv.Sniffer().sniff(content[:8192], delimiters=",;\t")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
    reader = csv.DictReader(io.StringIO(content, newline=""), delimiter=delimiter)
    headers = reader.fieldnames or []
    issues: list[dict[str, Any]] = []
    if time_column not in headers:
        issues.append({"code": "LOAD_TIME_COLUMN_MISSING", "column": time_column})
    if force_column not in headers:
        issues.append({"code": "LOAD_FORCE_COLUMN_MISSING", "column": force_column})
    rows: list[dict[str, Any]] = []
    for source_row, raw in enumerate(reader, start=2):
        time_info = parse_time_value(raw.get(time_column)) if time_column in headers else parse_time_value(None)
        raw_force = _number(raw.get(force_column)) if force_column in headers else None
        unit_text = force_unit
        if not unit_text:
            for key in ("force_unit", "load_unit", "unit"):
                if raw.get(key):
                    unit_text = str(raw[key]).strip()
                    break
        if not unit_text and force_column == "force_N":
            unit_text = "N"
        normalized_unit = (unit_text or "").casefold().replace(" ", "")
        scale = {"n": 1.0, "newton": 1.0, "newtons": 1.0, "kn": 1000.0, "kilonewton": 1000.0, "kilonewtons": 1000.0}.get(normalized_unit)
        if raw_force is None:
            force_n = None
            force_status = "MISSING_OR_INVALID_FORCE"
        elif scale is None:
            force_n = None
            force_status = "UNKNOWN_FORCE_UNIT"
        else:
            force_n = raw_force * scale
            force_status = "RECORDED_FORCE_N" if scale == 1.0 else "CONVERTED_FORCE_TO_N"
        rows.append({
            "source_row": source_row,
            **time_info,
            "clock_id": raw.get("clock_id") or raw.get("timestamp_clock_id") or None,
            "force_raw": raw_force,
            "force_unit_source": unit_text,
            "force_N": force_n,
            "force_status": force_status,
            "source_values": {str(key): _parse_cell(value) for key, value in raw.items() if key is not None},
        })
    return {"headers": headers, "rows": rows, "issues": issues,
            "source_definition": {"time_column": time_column, "force_column": force_column,
                                  "force_unit": force_unit}}


def parse_cbf_header_text(text: str) -> dict[str, Any]:
    """Extract explicitly labeled CBF time/exposure fields from header text.

    Only labeled values are accepted.  In particular, no acquisition time is
    built from a counter or an assumed exposure interval.
    """
    found_timestamp: str | None = None
    found_exposure: tuple[str, str | None] | None = None
    timestamp_re = re.compile(
        r"^\s*#?\s*(?:timestamp|start_time|datetime|date_time)\s*(?:[:=]\s*|\s+)(.*?)\s*$",
        re.I,
    )
    exposure_re = re.compile(
        r"^\s*#?\s*(?:exposure_time|exposure_s)\s*(?:[:=]\s*|\s+)([+\-0-9.eE]+)(?:\s*([a-zA-Z]+))?\s*$",
        re.I,
    )
    for line in str(text).splitlines():
        if found_timestamp is None:
            match = timestamp_re.match(line)
            if match:
                found_timestamp = match.group(1).strip()
        if found_exposure is None:
            match = exposure_re.match(line)
            if match:
                found_exposure = (match.group(1).strip(), match.group(2).strip() if match.group(2) else None)
    result: dict[str, Any] = {"timestamp_source": "CBF_HEADER" if found_timestamp is not None else None}
    if found_timestamp is not None:
        result.update(parse_time_value(found_timestamp))
    else:
        result.update({"timestamp_unix_s": None, "time_s": None,
                       "timestamp_status": "NO_EXPLICIT_TIMESTAMP_IN_CBF_HEADER",
                       "timestamp_timezone": None})
    if found_exposure is not None:
        raw_exposure, raw_unit = found_exposure
        exposure = _number(raw_exposure)
        unit_scale = {
            None: 1.0,
            "s": 1.0,
            "sec": 1.0,
            "secs": 1.0,
            "second": 1.0,
            "seconds": 1.0,
            "ms": 1e-3,
            "msec": 1e-3,
            "millisecond": 1e-3,
            "milliseconds": 1e-3,
        }.get(raw_unit.casefold() if raw_unit else None)
        if unit_scale is None:
            result["exposure_s"] = None
            result["exposure_status"] = "INVALID_EXPOSURE_UNIT"
        elif exposure is None or exposure < 0:
            result["exposure_s"] = exposure
            result["exposure_status"] = "INVALID_EXPOSURE_VALUE"
        else:
            result["exposure_s"] = exposure * unit_scale
            result["exposure_status"] = "RECORDED_CBF_EXPOSURE"
    else:
        result["exposure_s"] = None
        result["exposure_status"] = "NO_EXPLICIT_EXPOSURE_IN_CBF_HEADER"
    return result


def classify_desy_path(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Classify the local file layout without claiming a physical sample ID."""
    source = Path(path)
    inspect = source
    if root is not None:
        try:
            inspect = source.expanduser().resolve().relative_to(Path(root).expanduser().resolve())
        except ValueError:
            inspect = source
    dir_parts = inspect.parent.parts
    parts = [part.casefold() for part in dir_parts]
    suffix = inspect.suffix.casefold()
    if "dic" in parts and suffix in {".tif", ".tiff"}:
        kind = "desy_delun_dic"
        counter_namespace = "delun_dic_filename_counter"
    elif suffix == ".cbf":
        kind = "desy_camera_cbf"
        counter_namespace = "camera_cbf_filename_counter"
    elif "dic_images" in parts and suffix in {".tif", ".tiff"}:
        kind = "desy_camera_dic"
        counter_namespace = "camera_tiff_filename_counter"
    elif any(part in {"1236b_1", "varex_1"} for part in parts) and suffix in {".tif", ".tiff"}:
        kind = "desy_camera_dic"
        counter_namespace = "camera_tiff_filename_counter"
    else:
        kind = "desy_other_image"
        counter_namespace = "filename_counter"

    candidate = None
    for part in dir_parts:
        match = _FIO_FOLDER.match(part)
        if match:
            candidate = int(match.group(1))
            break
    return {
        "source_kind": kind,
        "frame_namespace": counter_namespace,
        "scan_id_candidate": candidate,
        "scan_id_candidate_namespace": "camera_folder_label" if candidate is not None else None,
        "scan_id_is_global_fio": False,
    }


def _group_directory(path: Path, root: Path, source_kind: str, explicit_levels: int | None) -> Path:
    if explicit_levels is not None:
        current = path.parent
        for _ in range(max(0, explicit_levels)):
            if current == root or root not in current.parents:
                return root
            current = current.parent
        return current
    try:
        rel_parts = list(path.relative_to(root).parts)
    except ValueError:
        return root
    dir_parts = rel_parts[:-1]
    dir_lowered = [part.casefold() for part in dir_parts]
    if source_kind == "desy_delun_dic" and "dic" in dir_lowered:
        current = path.parent.parent if dir_lowered and dir_lowered[-1] == "dic" else path.parent
    elif source_kind == "desy_camera_dic" and "dic_images" in dir_lowered:
        index = dir_lowered.index("dic_images")
        # Preserve the FIO folder label as the acquisition directory; camera
        # counter folders below it are namespace labels, not acquisition IDs.
        if index + 1 < len(dir_parts):
            current = root.joinpath(*dir_parts[: index + 2])
        else:
            current = path.parent
    elif source_kind == "desy_camera_dic":
        markers = [index for index, part in enumerate(dir_lowered) if part in {"1236b_1", "varex_1"}]
        current = root.joinpath(*dir_parts[: markers[0]]) if markers else path.parent
    else:
        current = path.parent
    try:
        current.relative_to(root)
    except ValueError:
        return root
    return current


def _scan_from_fio(path: Path) -> int | None:
    match = _FIO_SCAN.search(path.as_posix())
    return int(match.group(1)) if match else None


def _linked_fio(acquisition_dir: Path) -> tuple[Path | None, int | None, str]:
    files = sorted(acquisition_dir.glob("*.fio"), key=lambda item: item.name.casefold())
    if len(files) != 1:
        status = "NO_COLOCATED_FIO" if not files else "MULTIPLE_COLOCATED_FIO_FILES"
        return None, None, status
    fio_path = files[0]
    scan_id = _scan_from_fio(fio_path)
    if scan_id is None:
        return fio_path, None, "FIO_FILENAME_HAS_NO_GLOBAL_SCAN_ID"
    return fio_path, scan_id, "UNIQUE_COLOCATED_GLOBAL_FIO"


def _attach_explicit_fio_clock(
    acquisition: dict[str, Any],
    fio_path: Path,
    scan_id: int | None,
    *,
    counter_column: str,
    timestamp_column: str,
) -> None:
    """Attach FIO times by a caller-selected native counter column."""
    parsed = parse_fio_text(fio_path.read_text(encoding="utf-8-sig", errors="replace"), str(fio_path))
    acquisition["discovery_issues"].extend(
        {**issue, "source": str(fio_path.resolve())} for issue in parsed["issues"]
    )
    if counter_column not in parsed["columns"] or timestamp_column not in parsed["columns"]:
        acquisition["discovery_issues"].append({
            "code": "FIO_CLOCK_COLUMNS_MISSING",
            "source": str(fio_path.resolve()),
            "counter_column": counter_column,
            "timestamp_column": timestamp_column,
            "available_columns": parsed["columns"],
        })
        return
    counters: dict[int, list[dict[str, Any]]] = {}
    clock_id = f"desy.fio.{scan_id if scan_id is not None else fio_path.stem}.{timestamp_column}"
    for row in parsed["rows"]:
        counter = _integer(row.get(counter_column))
        if counter is None:
            continue
        time = parse_time_value(row.get(timestamp_column))
        counters.setdefault(counter, []).append({
            **time,
            "fio_source_line": row["source_line"],
            "clock_id": clock_id,
            "timestamp_source": "FIO_LOG",
            "counter_column": counter_column,
            "counter_namespace": counter_column,
        })
    duplicates = {counter: rows for counter, rows in counters.items() if len(rows) > 1}
    for counter, rows in duplicates.items():
        acquisition["discovery_issues"].append({
            "code": "FIO_DUPLICATE_NATIVE_COUNTER",
            "source": str(fio_path.resolve()),
            "counter_column": counter_column,
            "native_counter": counter,
            "source_lines": [row["fio_source_line"] for row in rows],
        })
    mapped = 0
    for frame in acquisition.get("measurement_frames", []):
        native = _integer(frame.get("native_cbf_frame"))
        candidates = counters.get(native, []) if native is not None else []
        if len(candidates) != 1:
            continue
        source = candidates[0]
        for field in ("timestamp_unix_s", "time_s", "clock_id", "timestamp_source", "timestamp_status", "timestamp_timezone"):
            frame[field] = source.get(field)
        frame["timestamp_basis_namespace"] = source["counter_namespace"]
        frame["timestamp_source_line"] = source["fio_source_line"]
        mapped += 1
    acquisition["metadata"]["fio_clock_mapping"] = {
        "status": "COUNTER_MATCHED" if mapped else "NO_COUNTER_MATCH",
        "source_path": str(fio_path.resolve()),
        "counter_column": counter_column,
        "timestamp_column": timestamp_column,
        "counter_namespace": counter_column,
        "clock_id": clock_id,
        "matched_measurement_frames": mapped,
        "join_method": "EXACT_NATIVE_COUNTER_EQUALITY",
        "time_interval_assumption": None,
    }


def discover(
    root: str | os.PathLike[str],
    pattern: Any = None,
    exclude: tuple[str, ...] | list[str] = (),
) -> list[dict[str, Any]]:
    """Discover Delun DIC and DESY camera image axes beneath ``root``.

    ``pattern`` may be a glob or a mapping with ``image_patterns``,
    ``frame_regex``, ``acquisition_parent_levels``, ``frame_metadata`` and
    explicit ``measurement_map``/``measurement_columns`` options.  FIO folder
    labels stay candidates unless a unique colocated ``eh3scan_*.fio`` file
    establishes the global scan namespace.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)
    config = dict(pattern) if isinstance(pattern, dict) else {}
    if isinstance(pattern, str):
        image_patterns = [pattern]
    else:
        selected = config.get("image_patterns", config.get("image_pattern", config.get("glob")))
        if selected is None:
            image_patterns = ["**/*.tif", "**/*.tiff", "**/*.cbf"]
        elif isinstance(selected, str):
            image_patterns = [selected]
        else:
            image_patterns = [str(item) for item in selected]
    excluded = tuple(str(item) for item in (*exclude, *config.get("exclude", ())))
    explicit_levels = config.get("acquisition_parent_levels")
    if explicit_levels is not None:
        explicit_levels = int(explicit_levels)

    files: set[Path] = set()
    for image_pattern in image_patterns:
        files.update(path.resolve() for path in root_path.glob(image_pattern) if path.is_file())
    files = {path for path in files if not _is_excluded(path, root_path, excluded)}
    grouped: dict[tuple[Path, str], list[Path]] = {}
    kinds_per_dir: dict[Path, set[str]] = {}
    for path in files:
        classification = classify_desy_path(path, root=root_path)
        source_kind = classification["source_kind"]
        directory = _group_directory(path, root_path, source_kind, explicit_levels)
        grouped.setdefault((directory, source_kind), []).append(path)
        kinds_per_dir.setdefault(directory, set()).add(source_kind)

    result: list[dict[str, Any]] = []
    for (directory, source_kind), batch_files in sorted(
        grouped.items(), key=lambda item: (item[0][0].as_posix().casefold(), item[0][1])
    ):
        generic_config = {
            "explicit_files": batch_files,
            "image_patterns": image_patterns,
            "acquisition_parent_levels": 1_000_000,
            "frame_regex": config.get("frame_regex", r"(?P<frame>\d+)$"),
            "frame_namespace": {
                "desy_delun_dic": "delun_dic_filename_counter",
                "desy_camera_cbf": "camera_cbf_filename_counter",
                "desy_camera_dic": "camera_tiff_filename_counter",
            }.get(source_kind, "filename_counter"),
            "source_kind": source_kind,
            "analysis_mode": str(config.get("analysis_mode", "extensometer")),
            "exclude": (),
            "frame_metadata": config.get("frame_metadata", "frame_metadata.csv"),
            "frame_metadata_columns": config.get("frame_metadata_columns"),
            "measurement_map": None,
        }
        discovered = generic_discover(directory, pattern=generic_config)
        if not discovered:
            continue
        acquisition = discovered[0]
        base_relative = directory.relative_to(root_path).as_posix() if directory != root_path else "_root_sequence"
        disambiguate = len(kinds_per_dir.get(directory, ())) > 1
        relative = f"{base_relative}__{source_kind}" if disambiguate else base_relative
        base_name = directory.name if directory != root_path else root_path.name
        acquisition["name"] = f"{base_name} ({source_kind})" if disambiguate else base_name
        acquisition["relative_path"] = relative
        acquisition["source_kind"] = source_kind
        acquisition["metadata"].update({
            "adapter": "desy",
            "source_layout": source_kind,
            "image_root_relative_path": base_relative,
        })

        fio_path, fio_scan_id, fio_link_status = _linked_fio(directory)
        folder_scan_candidate = None
        rel_dir_parts = directory.relative_to(root_path).parts if directory != root_path else ()
        for part in rel_dir_parts:
            match = _FIO_FOLDER.match(part)
            if match:
                folder_scan_candidate = int(match.group(1))
                break
        if fio_path is None:
            acquisition["metadata"].update({
                "scan_id": None,
                "scan_id_candidate": folder_scan_candidate,
                "scan_id_candidate_namespace": "camera_folder_label" if folder_scan_candidate is not None else None,
                "scan_id_is_global_fio": False,
                "fio_association_status": fio_link_status,
            })
        else:
            acquisition["metadata"].update({
                "scan_id": fio_scan_id,
                "scan_id_candidate": folder_scan_candidate,
                "scan_id_candidate_namespace": "camera_folder_label" if folder_scan_candidate is not None else None,
                "scan_id_namespace": "global_fio_scan_id" if fio_scan_id is not None else None,
                "scan_id_is_global_fio": fio_scan_id is not None,
                "fio_association_status": fio_link_status,
                "fio_source_path": str(fio_path.resolve()),
            })

        # An explicit measurement map takes precedence over filename-derived CBF rows
        # without ever appending duplicate analysis frames.
        if "measurement_map" in config:
            map_setting = config.get("measurement_map")
        elif "measurement_frames" in config:
            map_setting = config.get("measurement_frames")
        else:
            map_setting = "measurement_frames.csv"
        loaded_explicit_map = False
        if map_setting:
            map_path = Path(map_setting)
            if not map_path.is_absolute():
                map_path = directory / map_path
            if map_path.is_file():
                mapped, map_issues = read_explicit_frame_mapping(
                    map_path,
                    columns=config.get("measurement_columns"),
                    scan_id_is_global_fio=config.get("scan_id_is_global_fio"),
                    scan_id_namespace=config.get("scan_id_namespace"),
                )
                acquisition["measurement_frames"] = mapped
                acquisition["discovery_issues"].extend(map_issues)
                acquisition["metadata"]["measurement_map_source"] = str(map_path.resolve())
                loaded_explicit_map = True

        if source_kind == "desy_camera_cbf" and not loaded_explicit_map:
            has_fio_scan = fio_path is not None and fio_scan_id is not None
            derived_frames: list[dict[str, Any]] = []
            for frame in acquisition["image_frames"]:
                entry: dict[str, Any] = {
                    "frame": frame["frame"],
                    "native_cbf_frame": frame["native_frame"],
                    "scan_id": fio_scan_id if has_fio_scan else None,
                    "scan_id_is_global_fio": has_fio_scan,
                    "scan_id_namespace": "global_fio_scan_id" if has_fio_scan else None,
                    "counter_namespace": "camera_cbf_filename_counter",
                }
                for field in (
                    "timestamp_unix_s",
                    "estimated_timestamp_unix_s",
                    "time_s",
                    "clock_id",
                    "timestamp_source",
                    "timestamp_status",
                    "exposure_start_unix_s",
                    "exposure_end_unix_s",
                ):
                    if frame.get(field) is not None:
                        entry[field] = frame[field]
                derived_frames.append(entry)
            acquisition["measurement_frames"] = derived_frames

        if source_kind == "desy_camera_cbf" and fio_path is not None:
            counter_column = config.get("fio_counter_column")
            if counter_column:
                _attach_explicit_fio_clock(
                    acquisition,
                    fio_path,
                    fio_scan_id,
                    counter_column=str(counter_column),
                    timestamp_column=str(config.get("fio_timestamp_column", "timestamp")),
                )
        result.append(acquisition)
    return result


__all__ = [
    "classify_desy_path",
    "discover",
    "parse_cbf_header_text",
    "parse_fio_text",
    "parse_load_table_text",
    "parse_time_value",
]
