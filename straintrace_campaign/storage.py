"""Atomic, strict-JSON storage and reusable-stage helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .models import CampaignError, validate_json_value, validate_relative_path, validate_stage_record


CAMPAIGN_SOURCE_FILES: tuple[str, ...] = (
    "straintrace_campaign/__init__.py",
    "straintrace_campaign/__main__.py",
    "straintrace_campaign/_time_io.py",
    "straintrace_campaign/adapters/__init__.py",
    "straintrace_campaign/adapters/desy.py",
    "straintrace_campaign/adapters/generic.py",
    "straintrace_campaign/alignment.py",
    "straintrace_campaign/cli.py",
    "straintrace_campaign/delivery.py",
    "straintrace_campaign/gui.py",
    "straintrace_campaign/models.py",
    "straintrace_campaign/numerics.py",
    "straintrace_campaign/roi.py",
    "straintrace_campaign/storage.py",
    "straintrace_campaign/workflow.py",
)


def json_safe(value: Any, *, coerce_unknown: bool = False) -> Any:
    """Convert numeric/NumPy/path scalars to strict JSON values without importing NumPy."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item, coerce_unknown=coerce_unknown) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item, coerce_unknown=coerce_unknown) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return json_safe(tolist(), coerce_unknown=coerce_unknown)
    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        return json_safe(item_fn(), coerce_unknown=coerce_unknown)
    if isinstance(value, os.PathLike):
        return str(value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    if coerce_unknown:
        return str(value)
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


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


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    """Serialize a strict JSON value using canonical sorted-key formatting."""
    return _json_bytes(value, pretty=pretty)


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 hex digest of compact canonical JSON bytes."""
    return hashlib.sha256(
        json.dumps(
            json_safe(value, coerce_unknown=True),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def strict_json_loads(text: str) -> Any:
    """Parse a JSON string while rejecting non-standard NaN/Infinity constants."""
    def reject_constant(token: str) -> None:
        raise CampaignError("PROJECT_NONFINITE_NUMBER", f"non-standard JSON constant {token}")

    try:
        value = json.loads(text, parse_constant=reject_constant)
    except CampaignError:
        raise
    except json.JSONDecodeError as exc:
        raise CampaignError("PROJECT_READ_ERROR", f"cannot parse JSON: {exc}") from exc
    validate_json_value(value)
    return value


def strict_json_load(path: str | os.PathLike[str]) -> Any:
    """Load JSON while rejecting non-standard NaN/Infinity constants."""
    try:
        raw = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise CampaignError("PROJECT_READ_ERROR", f"cannot read JSON {path}: {exc}") from exc
    try:
        return strict_json_loads(raw)
    except CampaignError as exc:
        if exc.code == "PROJECT_READ_ERROR":
            raise CampaignError("PROJECT_READ_ERROR", f"cannot read JSON {path}: {exc.message}") from exc
        raise


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 6, delay_s: float = 0.02) -> None:
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt + 1 >= attempts:
                raise
            time.sleep(delay_s * (attempt + 1))


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temp, target)
    except OSError as exc:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise CampaignError("PROJECT_WRITE_ERROR", f"cannot atomically write {target}: {exc}") from exc
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
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
        _replace_with_retry(temp, target)
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


def roi_tracking_geometry(
    acquisition: Mapping[str, Any], proposal: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    """Return only the reviewed geometry that affects numerical tracking.

    Shared by ``workflow`` and ``numerics`` so cache keys and ROI sensitivity
    comparisons stay strictly identical across stages.
    """
    active = acquisition.get("roi") if proposal is None else proposal
    if not isinstance(active, Mapping):
        return None
    reference_id = active.get("reference_frame")
    if type(reference_id) is not int:
        raise CampaignError("ROI_REFERENCE_FRAME_INVALID", "ROI proposal reference_frame must be an integer")
    frames = acquisition.get("image_frames")
    reference_matches = [
        item for item in frames or []
        if isinstance(item, Mapping) and item.get("frame") == reference_id
    ]
    if len(reference_matches) != 1:
        raise CampaignError("ROI_REFERENCE_FRAME_INVALID", f"reference frame {reference_id!r} is absent or ambiguous")
    reference = reference_matches[0]
    raw_groups = active.get("groups", [])
    groups: list[Any] = []
    if isinstance(raw_groups, list):
        for group in raw_groups:
            if isinstance(group, Mapping):
                groups.append({
                    key: json_safe(group[key], coerce_unknown=True)
                    for key in ("name", "roi1", "roi2", "actual_mode", "selected_mode", "role")
                    if key in group
                })
            else:
                groups.append(json_safe(group, coerce_unknown=True))
    return {
        "reference_frame": active.get("reference_frame"),
        "reference_native_dic_frame": reference.get("native_frame"),
        "reference_sha256": active.get("reference_sha256"),
        "image_shape": json_safe(active.get("image_shape"), coerce_unknown=True),
        "free_region": json_safe(active.get("free_region"), coerce_unknown=True),
        "field_roi": json_safe(active.get("field_roi"), coerce_unknown=True),
        "exclusions": json_safe(active.get("exclusions", []), coerce_unknown=True),
        "groups": groups,
    }


def resolve_source_file(
    relative_to_bundle: str | None = None,
    *,
    module: Any = None,
    module_file: str | None = None,
    error_code: str = "CAMPAIGN_CODE_FINGERPRINT_UNAVAILABLE",
) -> Path:
    """Resolve an auditable Python source file in editable installs and frozen bundles."""
    rel_path: Path | None = Path(relative_to_bundle) if relative_to_bundle else None
    if rel_path is None and module is not None:
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
        if isinstance(module_name, str) and module_name:
            rel_path = Path(*module_name.split(".")).with_suffix(".py")

    frozen = bool(getattr(sys, "frozen", False) or getattr(sys, "_MEIPASS", None))
    candidates: list[Path] = []
    if frozen and rel_path is not None:
        bundle_roots: list[Path] = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bundle_roots.append(Path(meipass) / "sources")
        if getattr(sys, "executable", None):
            exe_dir = Path(sys.executable).resolve().parent
            bundle_roots.extend((exe_dir / "_internal" / "sources", exe_dir / "sources"))
        candidates.extend(base / rel_path for base in bundle_roots)

    if not frozen:
        raw_file = module_file or getattr(module, "__file__", None)
        if isinstance(raw_file, (str, os.PathLike)):
            source_path = Path(raw_file)
            if source_path.suffix.casefold() == ".py":
                candidates.append(source_path)
            else:
                candidates.append(source_path.with_suffix(".py"))
        if rel_path is not None:
            candidates.append(Path(__file__).resolve().parent.parent / rel_path)
        if module is not None:
            for attr in ("fingerprint", "analyze", "align"):
                function = getattr(module, attr, None)
                code = getattr(function, "__code__", None)
                filename = getattr(code, "co_filename", None)
                if isinstance(filename, str) and Path(filename).suffix.casefold() == ".py":
                    candidates.append(Path(filename))

    seen: set[str] = set()
    for candidate in candidates:
        try:
            canonical = candidate.resolve()
            key = str(canonical).casefold()
            if key in seen:
                continue
            seen.add(key)
            if canonical.is_file() and canonical.suffix.casefold() == ".py":
                return canonical
        except (OSError, TypeError, ValueError):
            continue
    label = relative_to_bundle or getattr(module, "__name__", type(module).__name__ if module is not None else "unknown")
    raise CampaignError(error_code, f"cannot resolve an auditable source copy for {label}")


def module_code_sha256(module: Any, *, relative_to_bundle: str | None = None) -> str:
    """Return the SHA-256 digest of a module's auditable Python source file."""
    return sha256_file(resolve_source_file(relative_to_bundle, module=module))


def campaign_code_fingerprints() -> dict[str, str]:
    """Hash every campaign Python module from editable or frozen bundle sources."""
    return {
        rel: sha256_file(resolve_source_file(rel, error_code="ENGINE_SOURCE_UNAVAILABLE"))
        for rel in CAMPAIGN_SOURCE_FILES
    }


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
    normalized = rel.replace("\\", "/")
    target = (base / Path(normalized)).resolve()
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
    rows_rel = validate_relative_path(normalized["rows_path"], "stage.rows_path").replace("\\", "/")
    artifacts = normalized.get("artifacts", [])
    if isinstance(artifacts, Mapping):
        artifacts = list(artifacts.values())
    if not isinstance(artifacts, list):
        raise CampaignError("STAGE_RECORD_INVALID", "stage artifacts must be an array of relative file paths")
    clean_artifacts: list[str] = []
    for artifact in [rows_rel, *artifacts]:
        rel = validate_relative_path(artifact, "stage.artifact").replace("\\", "/")
        if rel not in clean_artifacts:
            clean_artifacts.append(rel)
    supplied = normalized.get("checksums", {})
    if not isinstance(supplied, Mapping):
        raise CampaignError("STAGE_RECORD_INVALID", "stage checksums must be an object")
    supplied_norm: dict[str, Any] = {}
    for raw_rel, raw_digest in supplied.items():
        norm_rel = validate_relative_path(raw_rel, "stage.checksums key").replace("\\", "/")
        if norm_rel in supplied_norm and supplied_norm[norm_rel] != raw_digest:
            raise CampaignError("STAGE_CHECKSUM_MISMATCH", f"conflicting worker checksums for artifact: {norm_rel}")
        supplied_norm[norm_rel] = raw_digest
    checksums: dict[str, str] = {}
    for rel in clean_artifacts:
        path = artifact_path(project_root, rel)
        if not path.is_file():
            raise CampaignError("STAGE_ARTIFACT_MISSING", f"stage artifact does not exist: {rel}")
        digest = sha256_file(path)
        claimed = supplied_norm.get(rel)
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
        if record.get("status") != "IMPORTED_ACCEPTED" and record.get("fingerprint") != expected_fingerprint:
            return False
        checksums_by_norm = {
            validate_relative_path(rel, "stage.checksums key").replace("\\", "/"): expected
            for rel, expected in record["checksums"].items()
        }
        required_rels = [
            validate_relative_path(record["rows_path"], "stage.rows_path").replace("\\", "/"),
            *(validate_relative_path(item, "stage.artifact").replace("\\", "/") for item in record["artifacts"]),
            *checksums_by_norm.keys(),
        ]
        seen: set[str] = set()
        for rel in required_rels:
            if rel in seen:
                continue
            seen.add(rel)
            expected = checksums_by_norm.get(rel)
            if not expected:
                return False
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
    created_identity = None
    initialized = False
    fd_owned = True
    try:
        try:
            created_stat = os.fstat(fd)
            created_identity = (created_stat.st_dev, created_stat.st_ino, created_stat.st_ctime_ns)
        except OSError:
            created_identity = None
        try:
            fd_owned = False
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            initialized = True
        except OSError as exc:
            raise CampaignError("PROJECT_LOCK_ERROR", f"cannot initialize project lock: {exc}") from exc
        yield lock_path
    finally:
        if fd_owned:
            try:
                os.close(fd)
            except OSError:
                pass
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
