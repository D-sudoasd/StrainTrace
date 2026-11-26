"""Shared strict-JSON, hashing, atomic I/O, and safe-path utilities for straintrace_agent."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
import uuid
from typing import Any, Callable, Iterable, Mapping, Sequence


_REPLACE_RETRY_DELAYS_S = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
_DIR_REPLACE_RETRY_DELAYS_S = (0.01, 0.02, 0.05, 0.1)


def _api_attr(name: str, default: Any) -> Any:
    """Resolve an attribute through ``straintrace_agent.api`` when loaded.

    This preserves monkeypatch hooks on ``straintrace_agent.api`` while keeping
    internal submodules decoupled from top-level facade imports.
    """
    mod = sys.modules.get("straintrace_agent.api")
    return getattr(mod, name, default) if mod is not None else default


class AgentError(RuntimeError):
    """Stable, machine-readable workflow error."""

    def __init__(self, code: str, message: str, *, path: str = "$", details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"error_code": self.code, "message": self.message, "path": self.path}
        if self.details is not None:
            result["details"] = _json_safe(self.details)
        return result


class _RunCancelled(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    # Support NumPy scalars/arrays lazily without importing numpy at module load.
    np_mod = sys.modules.get("numpy")
    if np_mod is not None:
        if isinstance(value, np_mod.generic):
            return _json_safe(value.item())
        if isinstance(value, np_mod.ndarray):
            return _json_safe(value.tolist())
    raise AgentError("JSON_TYPE_ERROR", f"unsupported JSON value: {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            _json_safe(value), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AgentError("JSON_VALUE_ERROR", str(exc)) from exc


def _sha256_raw(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    try:
        return _sha256_raw(path)
    except OSError as exc:
        raise AgentError("INPUT_READ_ERROR", f"cannot read {path}: {exc}", path=str(path)) from exc


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    """Canonical JSON SHA-256 matching reporting's standalone verifier."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _replace_with_retry(
    source: Path, destination: Path, *, delays: Sequence[float] | None = None,
) -> None:
    """Replace a closed temporary file or directory, retrying bounded PermissionErrors."""
    retry_delays = _api_attr("_REPLACE_RETRY_DELAYS_S", _REPLACE_RETRY_DELAYS_S) if delays is None else delays
    for attempt in range(len(retry_delays) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == len(retry_delays):
                raise
            time.sleep(retry_delays[attempt])


def _unlink_with_retry(path: Path, *, delays: Sequence[float] | None = None) -> None:
    """Remove a file, retrying only bounded Windows sharing violations."""
    retry_delays = _api_attr("_REPLACE_RETRY_DELAYS_S", _REPLACE_RETRY_DELAYS_S) if delays is None else delays
    for attempt in range(len(retry_delays) + 1):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == len(retry_delays):
                raise
            time.sleep(retry_delays[attempt])


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _canonical(value) + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite value {token}")),
        )
    except FileNotFoundError as exc:
        raise AgentError("PROJECT_FILE_MISSING", f"required project file is missing: {path.name}", path=str(path)) from exc
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AgentError("PROJECT_JSON_INVALID", f"cannot read {path.name}: {exc}", path=str(path)) from exc
    if not isinstance(raw, dict):
        raise AgentError("PROJECT_JSON_INVALID", f"{path.name} must contain a JSON object", path=str(path))
    return raw


def _csv_atomic(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                safe = {
                    key: (
                        json.dumps(_json_safe(row[key]), ensure_ascii=False, allow_nan=False)
                        if isinstance(row.get(key), (list, dict, tuple))
                        else ""
                        if row.get(key) is None or (isinstance(row.get(key), float) and not math.isfinite(row[key]))
                        else row.get(key)
                    )
                    for key in columns
                }
                writer.writerow(safe)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        attributes = path.lstat()
    except FileNotFoundError:
        return False
    return bool(
        stat.S_ISLNK(attributes.st_mode)
        or getattr(attributes, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _relative_under_root(root: Path, candidate: Path) -> str | None:
    resolved_root = root.resolve()
    candidate_abs = candidate.expanduser().absolute()
    if candidate_abs.is_relative_to(resolved_root):
        rel = candidate_abs.relative_to(resolved_root)
        return rel.as_posix() if rel.parts else None
    parts: list[str] = []
    current = candidate_abs
    while True:
        try:
            if current.resolve() == resolved_root:
                return PurePosixPath(*reversed(parts)).as_posix() if parts else None
        except OSError:
            pass
        if current.parent == current:
            return None
        parts.append(current.name)
        current = current.parent


def _safe_file(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if not relative or not rel.parts or rel.is_absolute() or ".." in rel.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe package path: {relative!r}")
    resolved_root = root.resolve()
    current = resolved_root
    for segment in rel.parts:
        current = current / segment
        if _is_reparse_or_symlink(current):
            raise ValueError(f"Symlink or reparse point in package path: {relative!r}")
    if not current.resolve().is_relative_to(resolved_root):
        raise ValueError(f"Path escapes package: {relative!r}")
    return current


def _relocate_paths(value: Any, source_prefix: str | Path, target_prefix: str | Path) -> Any:
    """Rewrite staged directory prefixes inside nested dicts/lists after atomic commit."""
    src = str(source_prefix)
    dst = str(target_prefix)
    if isinstance(value, str) and value.startswith(src):
        return dst + value[len(src):]
    if isinstance(value, dict):
        return {key: _relocate_paths(item, src, dst) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocate_paths(item, src, dst) for item in value]
    return value


def _project_root(project: str | os.PathLike[str]) -> Path:
    return Path(project).expanduser().resolve(strict=False)


def _current_pointer(root: Path) -> dict[str, Any]:
    return _read_json(root / "CURRENT.json")


def _plan_path(root: Path, pointer: Mapping[str, Any]) -> Path:
    relative = pointer.get("current_plan")
    if not relative:
        raise AgentError("PLAN_REQUIRED", "run plan_project before running DIC")
    try:
        path = _safe_file(root, str(relative))
    except ValueError as exc:
        raise AgentError("PLAN_PATH_UNSAFE", str(exc), path="$.current_plan") from exc
    if not path.is_file():
        raise AgentError("PLAN_MISSING", f"current analysis plan is missing: {path}", path=str(path))
    return path


def _load_run_pointer(root: Path) -> tuple[dict[str, Any], Path]:
    pointer = _current_pointer(root)
    relative = pointer.get("current_run")
    if not relative:
        raise AgentError("RUN_REQUIRED", "project does not have a current run")
    try:
        run_root = _safe_file(root, str(relative))
    except ValueError as exc:
        raise AgentError("RUN_PATH_UNSAFE", str(exc), path="$.current_run") from exc
    if not run_root.is_dir():
        raise AgentError("RUN_MISSING", f"current run directory is missing: {run_root}", path=str(run_root))
    return pointer, run_root


def _write_progress(
    run_root: Path, event: Mapping[str, Any],
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    record = {"time_unix_s": time.time(), **_json_safe(dict(event))}
    path = run_root / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
    _atomic_json(run_root / "progress.json", record)
    if callback is not None:
        callback(record)


def _component_file_slug(component: str) -> str:
    """Return a stable filename token that stays unique on case-insensitive filesystems.

    Reuse the core's established component suffixes for its nine field
    components (notably distinguishing Green-Lagrange ``Exx`` from
    infinitesimal ``exx``). Other supported field-helper components receive a
    readable, case-fold-safe slug with a short hash of the original spelling.
    """
    import ezdic_core as core

    if not isinstance(component, str) or not component:
        raise AgentError("COMPONENT_INVALID", "component must be a non-empty string", path="$.component")
    suffixes = getattr(core, "DIC_COMPONENT_FILE_SUFFIXES", {})
    if component in suffixes:
        return str(suffixes[component])
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", component).strip("_").lower() or "component"
    digest = hashlib.sha256(component.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"
