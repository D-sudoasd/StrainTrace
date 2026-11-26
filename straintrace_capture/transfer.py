"""Copy a closed Capture experiment as an independently verifiable directory."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from .config import CaptureError
from .storage import (
    DATABASE, FRAME_WRITE_JOURNAL, TERMINAL_STATES, ExperimentReader, WriterLease,
    _flush_directory, _windows_move, frame_image_records,
)

TRANSFER_MANIFEST = "capture-transfer.json"
_SCHEMA = "straintrace.capture-transfer/v1"
_STAGE_SCHEMA = "straintrace.capture-transfer-stage/v1"
_STATE = "transfer-state.json"
_BLOCK_BYTES = 4 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|(?:COM|LPT)[1-9¹²³])\Z", re.I)
_KINDS = {"image": "images", "recovery": "images", "calibration": "calibration", "attachment": "attachments"}
_SIDECARS = tuple(DATABASE + suffix for suffix in ("-wal", "-shm", "-journal"))


@dataclass(frozen=True, slots=True)
class TransferProgress:
    stage: str
    path: str = ""
    bytes_done: int = 0
    bytes_total: int = 0
    files_done: int = 0
    files_total: int = 0
    reused: int = 0


@dataclass(frozen=True, slots=True)
class _Asset:
    path: str
    kind: str
    size: int
    expected: str | None = None
    external: Path | None = None


def _relative(name):
    if not isinstance(name, str) or not name:
        raise CaptureError("TRANSFER_PATH", f"invalid relative path: {name!r}")
    parts = name.split("/")
    if any(not part or part in (".", "..") or part.endswith((".", " ")) or
           _RESERVED.fullmatch(part.partition(".")[0].rstrip()) or
           any(ord(char) < 32 or char in '<>:"\\|?*' for char in part) for part in parts):
        raise CaptureError("TRANSFER_PATH", f"unsafe or non-portable relative path: {name!r}")
    return parts


def _member(name, kind):
    parts = _relative(name)
    if kind in _KINDS and len(parts) > 1 and parts[0] == _KINDS[kind]:
        return parts
    if (kind, name) in (("ledger", DATABASE), ("journal", FRAME_WRITE_JOURNAL)):
        return parts
    raise CaptureError("TRANSFER_PATH", f"unexpected path for {kind}: {name}")


def _node(path, *, directory, missing=False, independent=False):
    try:
        result = path.lstat()
    except FileNotFoundError:
        if missing:
            return None
        raise CaptureError("TRANSFER_MISSING", f"required file or directory is missing: {path}") from None
    test = stat.S_ISDIR if directory else stat.S_ISREG
    if (not test(result.st_mode) or getattr(result, "st_reparse_tag", 0) == 0xA0000003 or
            (independent and not directory and result.st_nlink > 1)):
        raise CaptureError("TRANSFER_PATH", f"unexpected file type or filesystem link: {path}")
    return result


def _file(root, name, *, independent=False):
    *directories, leaf = _relative(name)
    for part in directories:
        root /= part
        _node(root, directory=True)
    path = root / leaf
    return path, _node(path, directory=False, independent=independent)


def _files(root, directory):
    start = root / directory
    if not _node(start, directory=True, missing=True):
        return
    pending = [start]
    while pending:
        folder = pending.pop()
        for item in sorted(folder.iterdir(), reverse=True):
            name = item.relative_to(root).as_posix()
            _relative(name)
            metadata = item.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                _node(item, directory=True)
                pending.append(item)
            else:
                yield name, _node(item, directory=False)


def _add(assets, item):
    key = item.path.casefold()
    if key in assets:
        if item == assets[key] and item.kind == "attachment":
            return
        raise CaptureError("TRANSFER_PATH", f"duplicate or case-colliding path: {item.path}")
    assets[key] = item


def _check_tree(names, leaves):
    directories = {}
    for name in names:
        prefix = name.rpartition("/")[0]
        while prefix:
            key = prefix.casefold()
            if key in leaves or directories.setdefault(key, prefix) != prefix:
                raise CaptureError("TRANSFER_PATH", f"file or directory names collide on a case-insensitive filesystem: {name}")
            prefix = prefix.rpartition("/")[0]


def _ledger_asset(source, name, kind, digest, size=None):
    _member(name, kind)
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise CaptureError("INVALID_LEDGER", f"missing or invalid stored digest: {name}")
    _, metadata = _file(source, name)
    if size is not None and (type(size) is not int or metadata.st_size != size):
        raise CaptureError("TRANSFER_CHANGED", f"file size differs from the ledger: {name}")
    return _Asset(name, kind, metadata.st_size, digest)


def _inventory(reader, source, attachments, abort):
    experiment = reader.experiment
    assets = {}
    after = count = images = 0
    while frames := reader.frames(after=after, limit=512):
        abort()
        count += len(frames)
        after = frames[-1]["ordinal"]
        for frame in frames:
            for asset in frame_image_records(frame):
                _add(assets, _ledger_asset(source, asset["path"], "image", asset["sha256"], asset["file_bytes"]))
                images += 1
    for attachment in experiment.get("calibrations", ()):
        _add(assets, _ledger_asset(source, attachment["path"], "calibration", attachment["sha256"]))
    for name, info in _files(source, "images"):
        abort()
        known = assets.get(name.casefold())
        if known is None:
            _add(assets, _Asset(name, "recovery", info.st_size))
        elif known.path != name:
            raise CaptureError("TRANSFER_PATH", f"image path spelling differs from the ledger: {name}")
    if info := _node(source / FRAME_WRITE_JOURNAL, directory=False, missing=True):
        _add(assets, _Asset(FRAME_WRITE_JOURNAL, "journal", info.st_size))
    for name, info in _files(source, "attachments"):
        abort()
        _add(assets, _Asset(name, "attachment", info.st_size))
    for path in attachments:
        abort()
        name = f"attachments/{path.name}"
        _member(name, "attachment")
        existing = assets.get(name.casefold())
        if existing and existing.path == name and path == source / name:
            continue
        _add(assets, _Asset(name, "attachment", _node(path, directory=False).st_size, external=path))
    _check_tree((asset.path for asset in assets.values()), assets)
    return tuple(assets.values()), count, images


def _layout(experiment, output):
    source = Path(experiment).resolve(strict=True)
    _node(source, directory=True)
    destination = Path(output).absolute()
    if len(_relative(destination.name)) != 1:
        raise CaptureError("TRANSFER_PATH", f"invalid target directory: {output}")
    parent = destination.parent.resolve(strict=True)
    _node(parent, directory=True)
    destination = parent / destination.name
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise CaptureError("TRANSFER_PATH", "source and target experiment directories must be separate")
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise CaptureError("OUTPUT_EXISTS", f"the target directory already exists: {destination}")
    key = hashlib.sha256(os.path.normcase(destination.name).encode("utf-8")).hexdigest()[:20]
    stage = parent / (".capture-copy-" + key)
    if source.is_relative_to(stage) or stage.is_relative_to(source):
        raise CaptureError("TRANSFER_PATH", "source experiment overlaps the temporary copy directory")
    return source, destination, stage


@contextmanager
def _experiment_lease(source):
    _node(source / DATABASE, directory=False)
    marker = source / ".capture-writer"
    existing = _node(marker, directory=False, missing=True)

    def check_source():
        # Recovery can create a writer marker on a legacy or transported ledger.
        if existing is None and _node(marker, directory=False, missing=True) is not None:
            raise CaptureError("TRANSFER_SCOPE", "a Capture writer opened this experiment during the operation")

    with WriterLease(source, create=False):
        yield check_source


def _terminal_record(reader, expected_id):
    experiment = reader.experiment
    if experiment["state"] not in TERMINAL_STATES:
        raise CaptureError("TRANSFER_ACTIVE", f"capture state is {experiment['state']}; finish or recover it first")
    if expected_id is not None and experiment["experiment_id"] != expected_id:
        raise CaptureError("TRANSFER_SCOPE", "the selected experiment identity changed")
    return experiment


@contextmanager
def _source_reader(source, expected_id):
    with _experiment_lease(source) as check_source:
        wal, _shm, journal = (_node(source / name, directory=False, missing=True) for name in _SIDECARS)
        # An idle, checkpointed run can be read without creating any SQLite sidecars in the source.
        immutable = not (wal and wal.st_size) and not (journal and journal.st_size)
        with ExperimentReader(source, immutable=immutable) as reader:
            yield reader, _terminal_record(reader, expected_id), check_source


def _descriptor(source, destination, experiment, attachments):
    return {"schema": _STAGE_SCHEMA, "source": str(source), "target": str(destination),
            "experiment_id": experiment["experiment_id"], "attachments": list(map(str, attachments))}


def _read_object(path):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise CaptureError("TRANSFER_MANIFEST", f"cannot read JSON: {path}") from exc
    if not isinstance(result, dict):
        raise CaptureError("TRANSFER_MANIFEST", f"expected a JSON object: {path}")
    return result


def _stage_state(stage, expected):
    marker = stage / _STATE
    if _node(marker, directory=False, missing=True, independent=True):
        if _read_object(marker) != expected:
            raise CaptureError("TRANSFER_SCOPE", f"existing partial copy uses a different source or attachment list: {stage}")
        return True
    if any(path.name not in (".capture-writer", _STATE + ".part") for path in stage.iterdir()):
        raise CaptureError("TRANSFER_SCOPE", f"unrecognized contents in the temporary copy directory: {stage}")
    return False


def _remove_temp(path):
    if _node(path, directory=False, missing=True, independent=True):
        path.unlink()


def _write_json(part, target, value):
    _remove_temp(part)
    with part.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, target)
    _flush_directory(target.parent)


def _folder(path):
    path.mkdir(exist_ok=True)
    _node(path, directory=True)
    return path


def _attachment_paths(paths):
    return tuple(sorted({Path(path).resolve(strict=True) for path in paths}, key=str))


def _totals(experiment, assets, frames, images, ledger_bytes):
    return {"experiment_id": experiment["experiment_id"], "state": experiment["state"], "frames": frames,
            "registered_images": images, "files": len(assets) + 1,
            "bytes": ledger_bytes + sum(asset.size for asset in assets),
            "calibration_files": sum(asset.kind == "calibration" for asset in assets),
            "recovery_files": sum(asset.kind == "recovery" for asset in assets),
            "attachment_files": sum(asset.kind == "attachment" for asset in assets)}


def inspect_transfer(experiment, output, *, attachments=(), expected_id=None):
    """Report copy scope and size without creating the destination or staging files."""
    source, destination, stage = _layout(experiment, output)
    extra = _attachment_paths(attachments)
    with _source_reader(source, expected_id) as (reader, record, check_source):
        assets, frames, images = _inventory(reader, source, extra, lambda: None)
        info = _totals(record, assets, frames, images, reader.database_bytes())
        resume = bool(_node(stage, directory=True, missing=True)) and _stage_state(
            stage, _descriptor(source, destination, record, extra))
        check_source()
    return {**info, "output": str(destination), "staging": str(stage), "can_resume": resume,
            "available_bytes": shutil.disk_usage(destination.parent).free}


def _fingerprint(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_bytes(path, size, abort, *, sink=None, advance=None):
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        initial = os.fstat(stream.fileno())
        if initial.st_size != size:
            raise CaptureError("TRANSFER_CHANGED", f"file size changed: {path}")
        while True:
            abort()
            block = stream.read(_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            if sink is not None and sink.write(block) != len(block):
                raise OSError(f"short write while copying {path}")
            total += len(block)
            if advance is not None:
                advance(total)
        if total != size or _fingerprint(os.fstat(stream.fileno())) != _fingerprint(initial):
            raise CaptureError("TRANSFER_CHANGED", f"file changed while reading: {path}")
    return digest.hexdigest()


def _check_digest(actual, expected, name):
    if expected is not None and actual != expected:
        raise CaptureError("TRANSFER_CHANGED", f"SHA-256 differs from the recorded value: {name}")


def _publish_folder(data, destination):
    # All paths are on the destination volume. The rename must fail if another writer created the target.
    if os.name == "nt":
        if not _windows_move()(str(data), str(destination), 0x8):  # MOVEFILE_WRITE_THROUGH
            code = ctypes.get_last_error()
            if code in (80, 183):  # ERROR_FILE_EXISTS, ERROR_ALREADY_EXISTS
                raise CaptureError("OUTPUT_EXISTS", f"the target directory already exists: {destination}")
            raise ctypes.WinError(code)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    source, target = os.fsencode(data), os.fsencode(destination)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        move = libc.renameat2
        move.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        result = move(-100, source, -100, target, 1)  # AT_FDCWD, RENAME_NOREPLACE
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        move = libc.renamex_np
        move.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        result = move(source, target, 4)  # RENAME_EXCL
    else:
        raise CaptureError("TRANSFER_PUBLISH", "atomic no-replace directory publication is unavailable")
    if result:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise CaptureError("OUTPUT_EXISTS", f"the target directory already exists: {destination}")
        raise OSError(code, os.strerror(code), str(destination))
    with suppress(OSError):
        _flush_directory(destination.parent)


def _copy_asset(asset, source, data, temporary, abort, advance, checking):
    origin = asset.external or _file(source, asset.path)[0]
    *directories, leaf = _member(asset.path, asset.kind)
    target = data
    for name in directories:
        target = _folder(target / name)
    target /= leaf
    previous = _node(target, directory=False, missing=True, independent=True)
    if previous is not None and previous.st_size == asset.size:
        checking()
        digest = _read_bytes(origin, asset.size, abort)
        _check_digest(digest, asset.expected, asset.path)
        if _read_bytes(target, asset.size, abort) == digest:
            return digest, True
    _remove_temp(temporary)
    with temporary.open("xb") as stream:
        digest = _read_bytes(origin, asset.size, abort, sink=stream, advance=advance)
        _check_digest(digest, asset.expected, asset.path)
        stream.flush()
        os.fsync(stream.fileno())
    checking()
    _check_digest(_read_bytes(temporary, asset.size, abort), digest, asset.path)
    os.replace(temporary, target)
    return digest, False


def _entry(asset, digest):
    return {"path": asset.path, "kind": asset.kind, "bytes": asset.size, "sha256": digest}


def copy_experiment(experiment, output, *, attachments=(), expected_id=None, cancelled=None, progress=None):
    """Publish a complete copy, reusing verified files when retrying the same operation."""
    def abort():
        if cancelled is not None and cancelled():
            raise CaptureError("TRANSFER_CANCELLED", "copy cancelled; verified staged files can be reused")

    def report(stage_name, path="", byte_offset=0):
        if progress is not None:
            progress(TransferProgress(stage_name, path, done_bytes + byte_offset, total_bytes,
                                      len(entries), total_files, reused))
        abort()

    source, destination, stage = _layout(experiment, output)
    extra = _attachment_paths(attachments)
    if any(path.is_relative_to(stage) for path in extra):
        raise CaptureError("TRANSFER_PATH", "attachments must be outside the staging directory")
    entries = []
    done_bytes = total_bytes = total_files = reused = 0
    report("preparing")
    with _source_reader(source, expected_id) as (reader, record, check_source):
        _folder(stage)
        _node(stage / ".capture-writer", directory=False, missing=True, independent=True)
        with WriterLease(stage) as lease:
            descriptor = _descriptor(source, destination, record, extra)
            if not _stage_state(stage, descriptor):
                _write_json(stage / (_STATE + ".part"), stage / _STATE, descriptor)
            data, scratch = _folder(stage / "experiment"), _folder(stage / "scratch")
            snapshot = scratch / DATABASE
            for name in (DATABASE, *_SIDECARS):
                _remove_temp(scratch / name)
            reader.backup_to(snapshot, progress=abort)
            with snapshot.open("r+b") as stream:
                os.fsync(stream.fileno())
            ledger = _Asset(DATABASE, "ledger", snapshot.stat().st_size)
            # The inventory comes from the exact database readers will see in the copy.
            with ExperimentReader(scratch) as frozen:
                assets, frames, images = _inventory(frozen, source, extra, abort)
                saved = frozen.experiment
            if (saved["experiment_id"], saved["state"]) != (record["experiment_id"], record["state"]):
                raise CaptureError("TRANSFER_SCOPE", "the source ledger changed before it was snapshotted")
            totals = _totals(saved, assets, frames, images, ledger.size)
            total_bytes, total_files = totals["bytes"], totals["files"]
            planned_sizes = {item.path: item.size for item in assets}
            missing_bytes = total_bytes - ledger.size
            for name, info in _files(stage, "experiment"):
                member = Path(name).relative_to("experiment").as_posix()
                if member in planned_sizes:
                    if planned_sizes[member] == info.st_size:
                        missing_bytes -= info.st_size
                elif member not in (DATABASE, TRANSFER_MANIFEST):
                    raise CaptureError("TRANSFER_SCOPE", f"unrecognized file in the partial copy: {name}")
            # Replace any previous staged database only after the new snapshot is checked.
            _node(data / DATABASE, directory=False, missing=True, independent=True)
            entries.append(_entry(ledger, _read_bytes(snapshot, ledger.size, abort)))
            os.replace(snapshot, data / DATABASE)
            done_bytes += ledger.size
            _remove_temp(scratch / "file.part")
            free = None
            with suppress(OSError):
                free = shutil.disk_usage(destination.parent).free
            if free is not None and missing_bytes > free:
                raise CaptureError("TRANSFER_SPACE", f"at least {missing_bytes} bytes needed for remaining files; {free} bytes free")
            for item in assets:
                report("copying", item.path)
                digest, kept = _copy_asset(
                    item, source, data, scratch / "file.part", abort,
                    lambda offset: report("copying", item.path, offset), lambda: report("checking", item.path),
                )
                entries.append(_entry(item, digest))
                reused += int(kept)
                done_bytes += item.size
            report("publishing")
            _node(data / TRANSFER_MANIFEST, directory=False, missing=True, independent=True)
            _write_json(scratch / "manifest.part", data / TRANSFER_MANIFEST,
                        {"schema": _SCHEMA, **totals, "entries": entries})
            if os.name != "nt":
                for folder, _children, _names in os.walk(data, topdown=False, followlinks=False):
                    abort()
                    _flush_directory(Path(folder))
            abort()
            check_source()
            _publish_folder(data, destination)
        # Cleanup is limited to files and empty directories owned by this operation.
        for path in (scratch / "file.part", scratch / "manifest.part", stage / _STATE, lease.path):
            with suppress(OSError):
                path.unlink(missing_ok=True)
        for folder in (scratch, stage):
            with suppress(OSError):
                folder.rmdir()
    return {**totals, "ok": True, "output": str(destination), "reused_files": reused}


def _manifest_entries(manifest):
    entries = manifest.get("entries")
    if manifest.get("schema") != _SCHEMA or not isinstance(entries, list):
        raise CaptureError("TRANSFER_MANIFEST", "unknown transfer manifest format")
    index = {}
    paths = set()
    for entry in entries:
        if (not isinstance(entry, dict) or not isinstance(entry.get("kind"), str) or
                type(entry.get("bytes")) is not int or entry["bytes"] < 0 or
                not isinstance(entry.get("sha256"), str) or not _DIGEST.fullmatch(entry["sha256"])):
            raise CaptureError("TRANSFER_MANIFEST", "invalid file entry in the transfer manifest")
        name = entry.get("path")
        _member(name, entry["kind"])
        folded = name.casefold()
        if folded in paths:
            raise CaptureError("TRANSFER_MANIFEST", f"duplicate path in the transfer manifest: {name}")
        paths.add(folded)
        index[name] = entry
    if DATABASE not in index:
        raise CaptureError("TRANSFER_MANIFEST", "the transfer manifest does not contain the ledger")
    _check_tree(index, paths)
    return index


def verify_transfer(experiment):
    """Verify the transport manifest, original file bytes, and scope against the copied ledger."""
    root = Path(experiment).resolve(strict=True)
    manifest = _read_object(_file(root, TRANSFER_MANIFEST)[0])
    entries = _manifest_entries(manifest)

    def verify_file(name):
        entry = entries[name]
        path, info = _file(root, name)
        if info.st_size != entry["bytes"]:
            raise CaptureError("TRANSFER_CHANGED", f"size differs from the transfer manifest: {name}")
        _check_digest(_read_bytes(path, info.st_size, lambda: None), entry["sha256"], name)

    with _experiment_lease(root) as check_source:
        for name in _SIDECARS:
            try:
                (root / name).lstat()
            except FileNotFoundError:
                continue
            raise CaptureError("TRANSFER_CHANGED", f"unexpected SQLite sidecar in a standalone copy: {name}")
        verify_file(DATABASE)
        with ExperimentReader(root, immutable=True) as reader:
            record = _terminal_record(reader, manifest.get("experiment_id"))
            assets, frames, images = _inventory(reader, root, (), lambda: None)
        totals = _totals(record, assets, frames, images, entries[DATABASE]["bytes"])
        if any(type(manifest.get(key)) is not type(value) or manifest[key] != value for key, value in totals.items()):
            raise CaptureError("TRANSFER_MANIFEST", "the manifest totals or experiment identity differ from the copied ledger")
        if set(entries) != {DATABASE, *(item.path for item in assets)}:
            raise CaptureError("TRANSFER_MANIFEST", "the manifest file list differs from the copied experiment")
        for item in assets:
            entry = entries[item.path]
            if item.kind != entry["kind"] or item.size != entry["bytes"]:
                raise CaptureError("TRANSFER_MANIFEST", f"file scope differs from the copied experiment: {item.path}")
            _check_digest(entry["sha256"], item.expected, item.path)
            verify_file(item.path)
        check_source()
    return {**totals, "ok": True, "output": str(root)}
