"""Find related Capture ledgers by experiment ID and recorded acquisition links."""

from __future__ import annotations

import os
import sqlite3
import stat
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import REPEAT_OF, REPEAT_PARENT, CaptureError
from .storage import DATABASE, ExperimentReader

_REPORT_SCHEMA = "straintrace.capture-related/v1"
_LEDGER_ERRORS = (CaptureError, OSError, sqlite3.Error, ValueError, KeyError, TypeError)


@dataclass(frozen=True)
class _Run:
    directory: Path
    identity: str
    parent_id: str | None
    link: str | None
    name: str
    state: str
    created_utc_ns: int | None
    hint: dict

    @classmethod
    def from_record(cls, directory, record):
        identity = record["experiment_id"]
        config = record["config"]
        if not (isinstance(identity, str) and identity.strip() and isinstance(config, dict)
                and isinstance(config.get("name"), str) and isinstance(record["state"], str)):
            raise CaptureError("RELATED_INVALID_EXPERIMENT", f"实验标识、名称或状态缺失：{directory}")
        metadata = config.get("metadata", {})
        if not isinstance(metadata, dict):
            raise CaptureError("RELATED_INVALID_EXPERIMENT", f"无法读取实验元数据：{directory}")
        parent_id, link, hint = None, None, None
        for key, locator, kind in (("continuation_of", "continuation_parent", "continuation"),
                                   (REPEAT_OF, REPEAT_PARENT, "repeat")):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                parent_id, link, hint = value, kind, metadata.get(locator)
                break
        created = record.get("created_utc_ns")
        return cls(directory, identity, parent_id, link,
                   config["name"], record["state"], created if type(created) is int and created >= 0 else None,
                   hint if isinstance(hint, dict) else {})


def _read_run(directory):
    with ExperimentReader(directory) as reader:
        return _Run.from_record(reader.root, reader.experiment)


def _is_redirect(info):
    return stat.S_ISLNK(info.st_mode) or getattr(info, "st_reparse_tag", 0) == getattr(
        stat, "IO_REPARSE_TAG_MOUNT_POINT", -1)


def _local_hint(directory):
    # A locator from a copied ledger must not initiate an SMB connection or
    # traverse a symlink or junction. Explicit search directories are separate.
    if os.name == "nt":
        if str(directory).startswith("\\\\"):
            return False
        import ctypes

        if ctypes.windll.kernel32.GetDriveTypeW(str(directory.anchor)) == 4:  # DRIVE_REMOTE
            return False
    try:
        return not any(_is_redirect(path.lstat()) for path in (*reversed(directory.parents), directory, directory / DATABASE))
    except OSError:
        return False


def experiment_hint_paths(root, hints):
    """Resolve stored relative and absolute experiment directories without contacting remote shares."""
    for key, relative in (("relative_directory", True), ("directory", False)):
        value = hints.get(key)
        if not isinstance(value, str) or not value or "\0" in value:
            continue
        hint = Path(value)
        if (relative and hint.anchor) or (not relative and not hint.is_absolute()):
            continue
        directory = Path(os.path.abspath(root / hint if relative else hint))
        if _local_hint(directory):
            yield directory


def _search_directory(directory):
    if _has_ledger(directory):
        yield directory
    with os.scandir(directory) as entries:
        for entry in entries:
            try:
                child = Path(entry.path)
                if (entry.is_dir(follow_symlinks=False) and not _is_redirect(entry.stat(follow_symlinks=False))
                        and _has_ledger(child)):
                    yield child
            except OSError:
                continue  # Neighbors can move while the directory is being scanned.


def _has_ledger(directory):
    try:
        info = (directory / DATABASE).lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(info.st_mode) and not _is_redirect(info)


def _parent_map(runs):
    options = defaultdict(set)
    for run in runs:
        options[run.identity].add((run.parent_id, run.link))
    return {identity: next(iter(parents))[0] for identity, parents in options.items() if len(parents) == 1}


def _reachable(root_id, edges):
    seen, pending = set(), [root_id]
    while pending:
        identity = pending.pop()
        if identity not in seen:
            seen.add(identity)
            pending.extend(edges.get(identity, ()))
    return seen


def _child_map(parents):
    children = defaultdict(set)
    for child, parent in parents.items():
        if parent is not None:
            children[parent].add(child)
    return children


def _connected(root_id, parents):
    neighbors = _child_map(parents)
    for child, parent in parents.items():
        if parent is not None:
            neighbors[child].add(parent)
    return _reachable(root_id, neighbors)


def _cycles(related, parents):
    checked = set()
    for identity in sorted(related):
        positions = {}
        while identity is not None and identity not in checked:
            if identity in positions:
                yield sorted(member for member, index in positions.items() if index >= positions[identity])
                break
            positions[identity] = len(positions)
            identity = parents.get(identity)
        checked.update(positions)


def _counts(run):
    with ExperimentReader(run.directory) as reader:
        record, counts = reader.committed_state()
        latest = _Run.from_record(reader.root, record)
    if (latest.identity, latest.parent_id, latest.link) != (run.identity, run.parent_id, run.link):
        raise CaptureError("RELATED_SCOPE_CHANGED", f"实验标识或关联记录已变化：{run.directory}")
    return latest, counts


def inspect_related_experiments(root, *, search_dirs=(), recent_paths=(), expected_experiment_id=None):
    """Read a separate summary for each related ledger without scanning its TIFF files.

    Inspect immediate sibling experiment folders, explicitly selected directories
    and optional recent experiment paths. Follow local parent location hints
    only when their actual experiment IDs match the recorded links.
    """
    current = _read_run(Path(root))
    if expected_experiment_id is not None and current.identity != expected_experiment_id:
        raise CaptureError("RELATED_SCOPE_CHANGED", "当前目录中的实验标识已变化。请重新打开实验。")

    known = {current.directory: current}
    cache = {}
    issues = []

    def probe(directory):
        if directory in known:
            return known[directory]
        if directory not in cache:
            try:
                cache[directory] = _read_run(directory) if _has_ledger(directory) else None
            except _LEDGER_ERRORS:
                cache[directory] = None
        return cache[directory]

    requested = tuple(dict.fromkeys(Path(os.path.abspath(Path(directory).expanduser())) for directory in search_dirs))
    for directory in dict.fromkeys((current.directory.parent, *requested)):
        try:
            for candidate in _search_directory(directory):
                run = probe(candidate)
                if run is not None:
                    known[run.directory] = run
                elif candidate == directory and directory in requested:
                    issues.append({"code": "SEARCH_UNAVAILABLE", "directory": str(directory)})
        except OSError:
            issues.append({"code": "SEARCH_UNAVAILABLE", "directory": str(directory)})
    for directory in recent_paths:
        if (run := probe(Path(directory))) is not None:
            known[run.directory] = run

    checked = set()
    wrong_hints = defaultdict(set)
    while True:
        parents = _parent_map(known.values())
        related = _connected(current.identity, parents)
        candidates = [run for run in known.values() if run.identity in related and run.directory not in checked]
        if not candidates:
            break
        for run in candidates:
            checked.add(run.directory)
            if not parents.get(run.identity):
                continue
            for path in experiment_hint_paths(run.directory, run.hint):
                if (prior := probe(path)) is None:
                    continue
                if prior.identity == run.parent_id:
                    known[prior.directory] = prior
                else:
                    wrong_hints[run.parent_id].add(str(path))

    summaries = {}
    for run in known.values():
        if run.identity not in related:
            continue
        try:
            summary, counts = _counts(run)
        except _LEDGER_ERRORS:
            if run.directory == current.directory:
                raise
            issues.append({"code": "EXPERIMENT_UNAVAILABLE", "directory": str(run.directory), "experiment_id": run.identity})
            continue
        summaries[summary.directory] = (summary, counts)

    groups = defaultdict(list)
    for run, _ in summaries.values():
        groups[run.identity].append(run)
    parents = _parent_map(run for run, _ in summaries.values())
    related = _connected(current.identity, parents)
    cycles = list(_cycles(related, parents))
    cyclic = {identity for cycle in cycles for identity in cycle}
    issues.extend({"code": "LINEAGE_CYCLE", "experiment_ids": cycle} for cycle in cycles)
    previous = _reachable(current.identity, {key: (value,) for key, value in parents.items() if value})
    following = _reachable(current.identity, _child_map(parents))
    missing = defaultdict(set)
    for identity in sorted(related.intersection(groups)):
        copies = groups[identity]
        if len(copies) > 1:
            paths = sorted(str(run.directory) for run in copies)
            issues.append({"code": "ID_MULTIPLE_DIRECTORIES", "experiment_id": identity, "directories": paths})
            if identity not in parents:
                issues.append({"code": "PARENT_CONFLICT", "experiment_id": identity, "directories": paths,
                               "parent_ids": sorted({run.parent_id for run in copies}, key=lambda parent: parent or "")})
        if (parent := parents.get(identity)) and parent not in groups:
            missing[parent].add(identity)
    issues.extend({"code": "PARENT_UNAVAILABLE", "experiment_id": parent, "child_ids": sorted(children),
                   "mismatched_directories": sorted(wrong_hints[parent])} for parent, children in sorted(missing.items()))

    rows = []
    for run, counts in sorted(summaries.values(), key=lambda item: (
        item[0].created_utc_ns or 0, item[0].identity, str(item[0].directory),
    )):
        if run.identity not in related:
            continue
        if run.directory == current.directory:
            relation = "current"
        elif run.identity == current.identity:
            relation = "current_copy"
        elif run.identity in cyclic:
            relation = "linked"
        elif run.identity in previous:
            if run.identity == current.parent_id:
                relation = "repeat_source" if current.link == "repeat" else "parent"
            else:
                relation = "ancestor"
        elif run.identity in following:
            relation = "repeat" if run.link == "repeat" else "continuation"
        else:
            relation = "branch"
        rows.append({
            "experiment_id": run.identity, "continuation_of": run.parent_id if run.link == "continuation" else None,
            **({REPEAT_OF: run.parent_id} if run.link == "repeat" else {}), "name": run.name,
            "directory": str(run.directory), "created_utc_ns": run.created_utc_ns, "state": run.state,
            "relation": relation, "directories_with_id": len(groups[run.identity]), "cameras": counts,
            "committed_frames": sum(count["frames"] for count in counts.values()),
            "committed_images": sum(count["images"] for count in counts.values()),
        })
    return {
        "schema": _REPORT_SCHEMA, "experiment_id": current.identity, "directory": str(current.directory),
        "search_directories": [str(directory) for directory in requested], "runs": rows, "issues": issues,
    }
