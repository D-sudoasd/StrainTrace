"""Inspect acquisition destinations without creating or following their final entry."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OutputInspection:
    path: Path
    volume_directory: Path | None
    issues: tuple[dict, ...]
    suggestion: Path | None = None


def resolve_output_path(output: str | Path) -> Path:
    requested = Path(output).expanduser()
    # Resolve the parent while keeping the leaf for the writer's exclusive mkdir.
    # A directory link at the requested name occupies that name even if its target is missing.
    return (requested.parent.resolve() / requested.name
            if requested.name not in ("", "..") else requested.resolve())


def output_ancestor(output: Path) -> Path:
    ancestor = output.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    return ancestor


def _occupied(path: Path) -> bool:
    try:
        path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def _available_sibling(path: Path) -> Path | None:
    base, marker, number = path.name.rpartition("-run-")
    if not (base and marker and 2 <= len(number) <= 6 and number.isascii() and number.isdecimal()):
        base, number = path.name, "1"
    start = int(number) + 1
    try:
        for index in range(start, start + 32):
            candidate = path.with_name(f"{base}-run-{index:02d}")
            if not _occupied(candidate):
                return candidate
    except OSError:
        return None
    return None


def inspect_output(output: str | Path, *, repeat_source: Path | None = None, suggest=False) -> OutputInspection:
    """Return path findings and optionally suggest an unused sibling."""
    root = Path(output)
    issues = []
    volume_directory = None

    def block(code, path, **details):
        issues.append({"code": code, "severity": "block", "path": str(path), **details})

    try:
        root = resolve_output_path(root)
        if repeat_source is not None and root.is_relative_to(repeat_source):
            block("OUTPUT_INSIDE_REPEAT_SOURCE", root)
        if _occupied(root):
            block("OUTPUT_EXISTS", root)
        ancestor = output_ancestor(root)
        if ancestor.is_dir():
            volume_directory = ancestor
        else:
            block("OUTPUT_PARENT_NOT_DIRECTORY", ancestor)
    except (OSError, RuntimeError, ValueError) as exc:
        block("OUTPUT_INSPECTION_FAILED", root, detail=str(exc))
    suggestion = None
    if suggest and root.name and len(issues) == 1 and issues[0]["code"] == "OUTPUT_EXISTS" and volume_directory == root.parent:
        suggestion = _available_sibling(root)
    return OutputInspection(root, volume_directory, tuple(issues), suggestion)
