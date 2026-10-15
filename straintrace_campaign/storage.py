"""Atomic, strict-JSON storage and reusable-stage helpers."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .models import CampaignError, validate_json_value, validate_relative_path, validate_stage_record


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    validate_json_value(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CampaignError("PROJECT_JSON_ENCODE_ERROR", str(exc)) from exc
    return (text + "\n").encode("utf-8")


def strict_json_load(path: str | os.PathLike[str]) -> Any:
    """Load JSON while rejecting non-standard NaN/Infinity constants."""
    def reject_constant(token: str) -> None:
        raise CampaignError("PROJECT_NONFINITE_NUMBER", f"non-standard JSON constant {token}")

    try:
        with Path(path).open("r", encoding="utf-8-sig") as stream:
            value = json.load(stream, parse_constant=reject_constant)
    except CampaignError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError("PROJECT_READ_ERROR", f"cannot read JSON {path}: {exc}") from exc
    validate_json_value(value)
    return value


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    except OSError as exc:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise CampaignError("PROJECT_WRITE_ERROR", f"cannot atomically write {target}: {exc}") from exc
    return target


def atomic_write_json(path: str | os.PathLike[str], value: Any, *, pretty: bool = True) -> Path:
    return atomic_write_bytes(path, _json_bytes(value, pretty=pretty))


def write_jsonl_atomic(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    try:
        with temp.open("xb") as stream:
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    raise CampaignError("JSONL_ROW_INVALID", f"row {index} must be an object")
                stream.write(_json_bytes(dict(row), pretty=False))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    except CampaignError:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    except OSError as exc:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise CampaignError("JSONL_WRITE_ERROR", f"cannot atomically write {target}: {exc}") from exc
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8-sig") as stream:
            for line_no, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(
                        line,
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            CampaignError("PROJECT_NONFINITE_NUMBER", f"non-standard JSON constant {token}")
                        ),
                    )
                except CampaignError:
                    raise
                except json.JSONDecodeError as exc:
                    raise CampaignError("JSONL_READ_ERROR", f"invalid JSONL at line {line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise CampaignError("JSONL_ROW_INVALID", f"line {line_no} must contain a JSON object")
                validate_json_value(row, f"line[{line_no}]")
                rows.append(row)
    except CampaignError:
        raise
    except OSError as exc:
        raise CampaignError("JSONL_READ_ERROR", f"cannot read {path}: {exc}") from exc
    return rows


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while True:
                block = stream.read(chunk_size)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise CampaignError("SOURCE_READ_ERROR", f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def fingerprint_data(stage: str, dependencies: Any) -> str:
    payload = {"campaign_stage": str(stage), "dependencies": dependencies}
    return hashlib.sha256(_json_bytes(payload, pretty=False)).hexdigest()


def proposal_hash(proposal: Mapping[str, Any]) -> str:
    """Canonical proposal hash shared by ROI generation, editing, and approval."""
    payload = dict(proposal)
    payload.pop("proposal_hash", None)
    return fingerprint_data("roi_proposal_v1", payload)


def project_dir(path: str | os.PathLike[str]) -> Path:
    """Accept a project directory or its project.json file."""
    value = Path(path).expanduser()
    if value.name.lower() == "project.json":
        value = value.parent
    return value.resolve()


def project_json_path(path: str | os.PathLike[str]) -> Path:
    return project_dir(path) / "project.json"


def artifact_path(project_root: str | os.PathLike[str], relative: str) -> Path:
    rel = validate_relative_path(relative, "artifact path")
    base = project_dir(project_root)
    target = (base / Path(rel)).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise CampaignError("PROJECT_PATH_ESCAPE", f"artifact escapes project directory: {relative}") from exc
    return target


def relative_artifact(project_root: str | os.PathLike[str], path: str | os.PathLike[str]) -> str:
    base = project_dir(project_root)
    target = Path(path).resolve()
    try:
        return target.relative_to(base).as_posix()
    except ValueError as exc:
        raise CampaignError("PROJECT_ARTIFACT_OUTSIDE_ROOT", f"artifact is outside project directory: {target}") from exc


def stage_output_dir(
    project_root: str | os.PathLike[str], acquisition_id: str, stage: str, fingerprint: str
) -> Path:
    """Return a deterministic, isolated output directory for a stage attempt."""
    for value, label in ((acquisition_id, "acquisition_id"), (stage, "stage"), (fingerprint, "fingerprint")):
        if not value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in value):
            raise CampaignError("STAGE_PATH_INVALID", f"unsafe {label} for stage output directory")
    if len(fingerprint) < 12:
        raise CampaignError("STAGE_FINGERPRINT_INVALID", "stage fingerprint must contain at least 12 characters")
    path = project_dir(project_root) / "stages" / acquisition_id / f"{stage}-{fingerprint}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def finalize_stage_record(
    project_root: str | os.PathLike[str], record: Mapping[str, Any], *, fingerprint: str | None = None
) -> dict[str, Any]:
    """Normalize a worker record and verify every declared artifact checksum."""
    normalized = dict(record)
    if fingerprint is not None:
        normalized["fingerprint"] = fingerprint
    if "status" not in normalized:
        normalized["status"] = "COMPLETE"
    if "provenance" not in normalized:
        normalized["provenance"] = {}
    if "summary" not in normalized:
        normalized["summary"] = {}
    if "rows_path" not in normalized:
        raise CampaignError("STAGE_RECORD_INVALID", "worker stage record must provide rows_path")
    rows_rel = validate_relative_path(normalized["rows_path"], "stage.rows_path")
    artifacts = normalized.get("artifacts", [])
    if isinstance(artifacts, Mapping):
        artifacts = list(artifacts.values())
    if not isinstance(artifacts, list):
        raise CampaignError("STAGE_RECORD_INVALID", "stage artifacts must be an array of relative file paths")
    clean_artifacts: list[str] = []
    for artifact in [rows_rel, *artifacts]:
        rel = validate_relative_path(artifact, "stage.artifact")
        if rel not in clean_artifacts:
            clean_artifacts.append(rel)
    supplied = normalized.get("checksums", {})
    if not isinstance(supplied, Mapping):
        raise CampaignError("STAGE_RECORD_INVALID", "stage checksums must be an object")
    checksums: dict[str, str] = {}
    for rel in clean_artifacts:
        path = artifact_path(project_root, rel)
        if not path.is_file():
            raise CampaignError("STAGE_ARTIFACT_MISSING", f"stage artifact does not exist: {rel}")
        digest = sha256_file(path)
        claimed = supplied.get(rel)
        if claimed is not None and claimed != digest:
            raise CampaignError("STAGE_CHECKSUM_MISMATCH", f"worker checksum does not match artifact: {rel}")
        checksums[rel] = digest
    normalized["rows_path"] = rows_rel
    normalized["artifacts"] = clean_artifacts
    normalized["checksums"] = checksums
    if not normalized.get("fingerprint"):
        raise CampaignError("STAGE_RECORD_INVALID", "stage record must provide a fingerprint")
    normalized["fingerprint"] = str(normalized["fingerprint"])
    validate_stage_record(normalized, "stage")
    return normalized


def stage_record_reusable(
    project_root: str | os.PathLike[str], record: Mapping[str, Any] | None, expected_fingerprint: str
) -> bool:
    if not isinstance(record, Mapping):
        return False
    try:
        validate_stage_record(dict(record), "stage")
        if record.get("status") == "IMPORTED_ACCEPTED":
            # Imported results are immutable accepted evidence, not a cache hit
            # produced by the current numerical implementation.
            for rel, expected in record["checksums"].items():
                path = artifact_path(project_root, rel)
                if not path.is_file() or sha256_file(path) != expected:
                    return False
            return bool(record["artifacts"])
        if record.get("fingerprint") != expected_fingerprint:
            return False
        for rel, expected in record["checksums"].items():
            path = artifact_path(project_root, rel)
            if not path.is_file() or sha256_file(path) != expected:
                return False
        return bool(record["artifacts"])
    except CampaignError:
        return False


@contextmanager
def acquire_project_lock(project_root: str | os.PathLike[str]) -> Iterator[Path]:
    """Acquire a fail-fast, process-exclusive writer lock for a project.

    A crash intentionally leaves a lock receipt for explicit recovery. This
    function never guesses that a lock is stale and never removes another
    process's lock.
    """
    root = project_dir(project_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".campaign.lock"
    token = secrets.token_hex(16)
    payload = json.dumps(
        {"pid": os.getpid(), "token": token, "created_unix_s": time.time()},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise CampaignError("PROJECT_LOCKED", f"another writer holds {lock_path}; inspect the lock receipt before recovery") from exc
    except OSError as exc:
        raise CampaignError("PROJECT_LOCK_ERROR", f"cannot create project lock: {exc}") from exc
    try:
        created_stat = os.fstat(fd)
        created_identity = (created_stat.st_dev, created_stat.st_ino, created_stat.st_ctime_ns)
    except OSError:
        created_identity = None
    initialized = False
    try:
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            initialized = True
        except OSError as exc:
            raise CampaignError("PROJECT_LOCK_ERROR", f"cannot initialize project lock: {exc}") from exc
        yield lock_path
    finally:
        # Normal release requires our unguessable receipt token. If setup
        # itself failed, remove only the exact file created by our O_EXCL open;
        # never guess whether a pre-existing lock left by a crashed process is
        # safe to clear.
        try:
            current_stat = os.stat(lock_path)
            same_created_file = False
            if created_identity is not None:
                dev, ino, ctime_ns = created_identity
                same_created_file = current_stat.st_dev == dev and (
                    current_stat.st_ino == ino if ino else current_stat.st_ctime_ns == ctime_ns
                )
            if initialized:
                current = strict_json_load(lock_path)
                if current.get("token") == token:
                    lock_path.unlink(missing_ok=True)
            elif same_created_file:
                lock_path.unlink(missing_ok=True)
        except (CampaignError, OSError, AttributeError):
            pass
