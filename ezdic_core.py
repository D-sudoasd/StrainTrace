"""GUI-independent numerical and export engine for ezDIC.

The legacy Tk application imports and re-exports this module's pure helpers.
No Tk, GUI widget, desktop-shell, or message-box module is imported here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import uuid
import threading
import stat
from contextlib import contextmanager
from collections.abc import Iterable, Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

APP_NAME = "ezDIC"
APP_VERSION = "0.1.4"
ORIGIN_OPJU_FILENAME = "ezDIC_results.opju"
IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"})
IMAGE_EXTENSIONS = [
    "*.tif", "*.tiff", "*.TIF", "*.TIFF",
    "*.png", "*.jpg", "*.jpeg", "*.bmp",
    "*.PNG", "*.JPG", "*.JPEG", "*.BMP",
]
TRACKING_ACCEPT_MODE_LABELS = {
    "initial": "初始",
    "hard": "硬接受",
    "adaptive": "自适应接受",
    "rejected": "已拒绝",
}
ROI_ROLE_VALUES = {"none", "axial", "transverse"}
POISSON_MIN_ABS_AXIAL_ENGINEERING_STRAIN = 1e-6
ANALYSIS_MODE_EXTENSOMETER = "extensometer"
ANALYSIS_MODE_FULLFIELD = "fullfield"
DIC_SOLVER_ICGN = "IC-GN"
DIC_SOLVER_ICLM = "IC-LM"
DIC_SOLVERS = (DIC_SOLVER_ICGN, DIC_SOLVER_ICLM)
DIC_FIELD_COMPONENTS = ("u", "v", "zncc", "Exx", "Eyy", "Exy", "exx", "eyy", "exy")
DIC_COMPONENT_FILE_SUFFIXES = {
    component: f"{component}_infinitesimal" if component in ("exx", "eyy", "exy") else component
    for component in DIC_FIELD_COMPONENTS
}
DIC_COMPONENT_LABELS = {
    "u": "u (px)",
    "v": "v (px)",
    "zncc": "ZNCC",
    "Exx": "Exx (Green-Lagrange)",
    "Eyy": "Eyy (Green-Lagrange)",
    "Exy": "Exy (Green-Lagrange)",
    "exx": "exx (infinitesimal)",
    "eyy": "eyy (infinitesimal)",
    "exy": "exy (infinitesimal)",
}

NORMALIZATION_VERSION = "reference_percentile_v1"
DEFAULT_NORMALIZATION_LOWER_PERCENTILE = 1.0
DEFAULT_NORMALIZATION_UPPER_PERCENTILE = 99.0
# Texture decisions are part of the scientific provenance contract.  Keep the
# metric and decision versions explicit so a manifest can be replayed after a
# future implementation changes the feature set or thresholds.
TEXTURE_METRICS_VERSION = "structure_tensor_rank1_periodic_v2"
TEXTURE_DISCRIMINATOR_VERSION = "rank_one_periodic_v1"
TEXTURE_PREFLIGHT_VERSION = "texture_preflight_v2"
DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO = 0.02
DEFAULT_TEXTURE_MAX_DIRECTIONAL_COHERENCE = 0.85
DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE = 0.90
DEFAULT_TEXTURE_MAX_RANK_ONE_RATIO = DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO
GUI_SOURCE_FILENAME = "dic_virtual_extensometer_gui_v7_multi_roi_range.py"


class CoreError(RuntimeError):
    """Machine-readable error raised by the GUI-independent core.

    ``code`` is stable for callers and ``details`` contains structured,
    JSON-compatible context.  The human-readable message is retained through
    ``RuntimeError`` for the legacy GUI and command-line diagnostics.
    """

    def __init__(self, code, details=None):
        self.code = str(code)
        if details is None:
            details = {}
        elif not isinstance(details, Mapping):
            details = {"message": str(details)}
        self.details = dict(details)
        message = self.details.get("message") or self.details.get("reason") or self.code
        super().__init__(f"{self.code}: {message}")

    def as_dict(self):
        return {"code": self.code, "details": dict(self.details)}

PLOT_COLOR_CYCLE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
]

PLOT_EXPORT_FORMATS = ("png", "tiff", "pdf", "svg", "eps")
PLOT_EXPORT_PRESETS = {
    "single_column": {
        "figsize": (3.45, 2.55),
        "dpi": 600,
        "font_size": 8,
        "label_size": 8,
        "tick_size": 7,
        "legend_size": 7,
        "title_size": 8,
        "line_width": 1.15,
        "marker_size": 18,
        "axis_line_width": 0.8,
        "grid_alpha": 0.18,
        "colorbar_label_size": 8,
        "colorbar_tick_size": 7,
        "colorbar_fraction": 0.046,
        "colorbar_pad": 0.035,
        "constrained_layout": True,
    },
    "double_column": {
        "figsize": (7.1, 4.2),
        "dpi": 600,
        "font_size": 9,
        "label_size": 9,
        "tick_size": 8,
        "legend_size": 8,
        "title_size": 9,
        "line_width": 1.25,
        "marker_size": 20,
        "axis_line_width": 0.85,
        "grid_alpha": 0.18,
        "colorbar_label_size": 9,
        "colorbar_tick_size": 8,
        "colorbar_fraction": 0.042,
        "colorbar_pad": 0.03,
        "constrained_layout": True,
    },
    "presentation": {
        "figsize": (10.0, 5.8),
        "dpi": 300,
        "font_size": 13,
        "label_size": 13,
        "tick_size": 11,
        "legend_size": 11,
        "title_size": 14,
        "line_width": 2.0,
        "marker_size": 34,
        "axis_line_width": 1.1,
        "grid_alpha": 0.22,
        "colorbar_label_size": 13,
        "colorbar_tick_size": 11,
        "colorbar_fraction": 0.038,
        "colorbar_pad": 0.03,
        "constrained_layout": True,
    },
    "raw_inspection": {
        "figsize": (7.0, 4.6),
        "dpi": 300,
        "font_size": 10,
        "label_size": 10,
        "tick_size": 9,
        "legend_size": 9,
        "title_size": 11,
        "line_width": 1.2,
        "marker_size": 20,
        "axis_line_width": 0.9,
        "grid_alpha": 0.25,
        "colorbar_label_size": 10,
        "colorbar_tick_size": 9,
        "colorbar_fraction": 0.044,
        "colorbar_pad": 0.035,
        "constrained_layout": True,
    },
    "publication": {
        "figsize": (7.1, 4.2),
        "dpi": 600,
        "font_size": 9,
        "label_size": 9,
        "tick_size": 8,
        "legend_size": 8,
        "title_size": 9,
        "line_width": 1.25,
        "marker_size": 20,
        "axis_line_width": 0.85,
        "grid_alpha": 0.18,
        "colorbar_label_size": 9,
        "colorbar_tick_size": 8,
        "colorbar_fraction": 0.042,
        "colorbar_pad": 0.03,
        "constrained_layout": True,
    },
}

def format_tracking_status_line(frame_i, n, group_name, accept_mode, strain_text, score1, score2):
    accept_text = TRACKING_ACCEPT_MODE_LABELS.get(str(accept_mode), str(accept_mode))
    return (
        f"第 {frame_i}/{n} 帧，组 {group_name}：{accept_text}，"
        f"应变={strain_text}，相关=({score1:.3f}, {score2:.3f})"
    )



def write_image_checked(path, image):
    """Write an image through an encoded buffer, preserving Unicode paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower() or ".png"
    try:
        ok, encoded = cv2.imencode(extension, np.asarray(image))
        if not ok or encoded is None:
            raise RuntimeError("OpenCV 编码失败")
        encoded.tofile(str(path))
    except Exception as exc:
        raise RuntimeError(f"写入图像失败：{path}（{exc}）") from exc
    if not ok or not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"写入图像失败：{path}")
    return path


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def safe_name(s):
    s = str(s).strip()
    if not s:
        return "group"
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    return s[:80]


def _output_name_key(value):
    """Canonical key for a generated per-group filename on the host OS."""
    # Windows output paths are case-insensitive even before a directory/file
    # exists.  ``casefold`` also makes the check deterministic on a POSIX test
    # host simulating a Windows-produced manifest.
    return safe_name(value).casefold()


def _validate_group_output_names(groups):
    """Reject sanitization/truncation/case collisions before staging."""
    seen = {}
    collisions = []
    for index, group in enumerate(groups or []):
        raw_name = str(group.get("name", ""))
        sanitized = safe_name(raw_name)
        key = _output_name_key(raw_name)
        previous = seen.get(key)
        if previous is not None:
            collisions.append(
                {
                    "sanitized_name": sanitized,
                    "canonical_key": key,
                    "indices": [previous["index"], index],
                    "names": [previous["name"], raw_name],
                }
            )
        else:
            seen[key] = {"index": index, "name": raw_name}
    if collisions:
        raise CoreError(
            "GROUP_NAME_COLLISION",
            {
                "collisions": collisions,
                "message": "ROI group names collide after deterministic output-name sanitization/truncation",
            },
        )
    return {str(group.get("name", "")): safe_name(group.get("name", "")) for group in groups or []}


def collect_images(folder):
    # Suffix matching is case-insensitive on every platform.  Glob patterns
    # such as ``*.png`` miss ``.PNG`` on Linux; Windows glob happens to fold
    # case, so the two would otherwise disagree.
    try:
        folder_path = Path(folder)
    except TypeError:
        return []
    try:
        if not folder_path.is_dir():
            return []
        entries = list(folder_path.iterdir())
    except OSError:
        return []
    unique = {}
    for raw_path in entries:
        try:
            if not raw_path.is_file():
                continue
        except OSError:
            continue
        if raw_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        normalized = os.path.normcase(os.path.abspath(os.fspath(raw_path)))
        unique.setdefault(normalized, os.fspath(raw_path))
    # Natural-sort ties (frame_2 versus frame_02) need an explicit lexical
    # tie-breaker.  A set of raw spellings is not a stable order.
    return sorted(
        unique.values(),
        key=lambda value: (
            natural_sort_key(os.fspath(value)),
            os.path.normcase(os.path.abspath(os.fspath(value))),
        ),
    )


def image_sequence_fingerprint(paths):
    """Stable sequence identity based on normalized paths and file metadata."""
    fingerprint = []
    for raw_path in paths or []:
        path = Path(raw_path)
        try:
            stat = path.stat()
            fingerprint.append(
                (
                    os.path.normcase(os.path.abspath(str(path))),
                    int(stat.st_size),
                    int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
                )
            )
        except OSError:
            fingerprint.append((os.path.normcase(os.path.abspath(str(path))), None, None))
    return tuple(fingerprint)


def canonicalize_json(value):
    """Convert common scientific/Python values to deterministic JSON data."""
    if isinstance(value, Mapping):
        return {str(key): canonicalize_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonicalize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return canonicalize_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value):
            return {"__float__": "nan"}
        if np.isposinf(value):
            return {"__float__": "+inf"}
        if np.isneginf(value):
            return {"__float__": "-inf"}
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def canonical_json_bytes(value):
    return json.dumps(
        canonicalize_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_hash(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path, *, chunk_size=1024 * 1024):
    """Return the content digest of a file without relying on metadata."""
    path = Path(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(int(chunk_size))
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise CoreError("INPUT_FILE_ERROR", {"path": str(path), "message": str(exc)}) from exc
    return digest.hexdigest()


def file_identity(path):
    """Capture path, size, mtime and bytes for one ordered input."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError as exc:
        raise CoreError("INPUT_FILE_ERROR", {"path": str(path), "message": str(exc)}) from exc
    if not path.is_file():
        raise CoreError("INPUT_FILE_ERROR", {"path": str(path), "message": "input is not a file"})
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
        "sha256": sha256_file(path),
    }


def ordered_input_manifest(paths):
    """Return content-addressed identities in caller-supplied order."""
    return [file_identity(path) for path in list(paths or [])]


ordered_input_identities = ordered_input_manifest


def collect_environment():
    """Collect JSON-safe Python/platform and installed-package information."""
    packages = {}
    for name in ("numpy", "opencv-python", "opencv-contrib-python", "pandas", "Pillow", "matplotlib", "scipy"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
    }


def environment_info():
    return collect_environment()


def _runtime_code_root():
    """Return the source root used by both source and frozen execution."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root).resolve()
    return Path(__file__).resolve().parent


def _code_path_candidates(root, filename):
    """Return the only supported filesystem locations for one source file.

    PyInstaller executes Python modules from its archive, so ``module.__file__``
    is not a usable provenance path in a frozen process.  The release bundle
    deliberately carries auditable copies below ``_MEIPASS/sources`` while the
    schema is copied below ``_MEIPASS/schemas``.  Keep the legacy
    ``sources/schemas`` layout as a compatibility candidate for older bundles.
    """
    root = Path(root).resolve()
    filename = str(filename)
    if filename.casefold() == "run_config_v1.json":
        return (
            root / "schemas" / filename,
            root / "sources" / "schemas" / filename,
            root / "sources" / filename,
            root / filename,
        )
    if getattr(sys, "_MEIPASS", None):
        return (root / "sources" / filename, root / filename)
    return (root / filename, root / "sources" / filename)


def _resolve_one_code_path(value, *, root):
    """Resolve a requested code path to a real file inside ``root``.

    Frozen archive paths are intentionally mapped only by their known basename
    to the audited ``sources`` tree.  Any other missing or outside-root path is
    rejected so a manifest can never silently fall back to an archive member,
    a developer checkout, or an unrelated external file.
    """
    root = Path(root).resolve()
    raw = Path(value)
    direct_candidates = []
    if raw.is_absolute():
        direct_candidates.append(raw)
    else:
        direct_candidates.append(root / raw)
    for candidate in direct_candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise CoreError(
                    "CODE_FILE_OUTSIDE_ROOT",
                    {"path": str(resolved), "root": str(root), "message": "code file must be inside the runtime root"},
                ) from exc
            return resolved

    # ``Path.resolve`` can make a PyInstaller archive member look like a path
    # below ``_MEIPASS`` even though no filesystem file exists there.  Only the
    # known executable/schema basenames may be remapped to the portable copy.
    basename = raw.name.casefold()
    known = {
        "ezdic_core.py",
        "ezdic_cli.py",
        "ezdic_benchmark.py",
        GUI_SOURCE_FILENAME.casefold(),
        "run_config_v1.json",
    }
    if (basename == "run_config_v1.json" and not raw.is_absolute()) or getattr(sys, "_MEIPASS", None):
        if basename not in known:
            raise CoreError(
                "CODE_FILE_MISSING",
                {"path": str(raw), "root": str(root), "message": "requested code provenance file is missing"},
            )
        for candidate in _code_path_candidates(root, raw.name):
            if candidate.is_file():
                return candidate.resolve()
    raise CoreError(
        "CODE_FILE_MISSING",
        {"path": str(raw), "root": str(root), "message": "requested code provenance file is missing"},
    )


def resolve_code_paths(
    paths=None,
    *,
    include_cli=False,
    include_gui=False,
    include_benchmark=False,
    include_schema=True,
):
    """Resolve the portable source set for a source or frozen execution.

    ``paths`` is an optional caller-supplied set.  It is resolved through the
    same bundle-aware logic, which lets callers pass ``module.__file__`` in a
    source checkout and still work when that value is an archive-style path in
    a frozen process.  Optional executable modules are required when their
    corresponding flag is true; missing files fail closed with ``CoreError``.
    The core source is always included.
    """
    root = _runtime_code_root()
    selected = []
    if paths is not None:
        if not isinstance(paths, (list, tuple)):
            raise CoreError("CODE_FILE_SET_INVALID", {"message": "code paths must be a sequence"})
        selected.extend(paths)

    try:
        requested_names = {Path(value).name.casefold() for value in selected}
    except (TypeError, ValueError) as exc:
        raise CoreError("CODE_FILE_SET_INVALID", {"message": "code paths must contain path-like values"}) from exc
    required = []
    if paths is None:
        required.append("ezdic_core.py")
    elif "ezdic_core.py" not in requested_names:
        required.append("ezdic_core.py")
    if include_cli:
        required.append("ezdic_cli.py")
    if include_gui:
        required.append(GUI_SOURCE_FILENAME)
    if include_benchmark:
        required.append("ezdic_benchmark.py")
    if include_schema:
        required.append("run_config_v1.json")
    selected.extend(required)

    resolved = []
    seen = set()
    for value in selected:
        path = _resolve_one_code_path(value, root=root)
        relative = path.relative_to(root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        resolved.append(path)
    if not resolved:
        raise CoreError("CODE_FILE_SET_EMPTY", {"message": "code fingerprint requires at least one source file"})
    return resolved


# Stable public alias used by the CLI/GUI adapters.  Keep the underscored
# implementation below for compatibility with existing internal callers.
resolve_source_paths = resolve_code_paths
_resolve_code_paths = resolve_code_paths


def _default_code_paths():
    return resolve_code_paths(include_schema=False)


def _code_file_entries(paths=None, *, root=None):
    """Return a portable, explicitly bound source-file set.

    The old implementation hashed absolute path strings.  That made a valid
    bundle unverifiable after moving the checkout to another machine.  The
    manifest now records the relative file set separately and hashes the same
    set during verification.
    """
    root_path = Path(root).resolve() if root is not None else _runtime_code_root()
    selected = _default_code_paths() if paths is None else [Path(path) for path in paths]
    entries = []
    seen = set()
    for raw_path in selected:
        path = raw_path if raw_path.is_absolute() else root_path / raw_path
        path = path.resolve()
        if not path.is_file():
            raise CoreError("CODE_FILE_MISSING", {"path": str(path), "root": str(root_path), "message": "code fingerprint source file is missing"})
        try:
            relative = path.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise CoreError(
                "CODE_FILE_OUTSIDE_ROOT",
                {"path": str(path), "root": str(root_path), "message": "code file must be inside the declared root"},
            ) from exc
        if relative in seen:
            continue
        seen.add(relative)
        entries.append({"path": relative, "sha256": sha256_file(path)})
    if not entries:
        raise CoreError("CODE_FILE_SET_EMPTY", {"message": "code fingerprint requires at least one source file"})
    return entries


def code_fingerprint(paths=None, *, root=None):
    """Hash an explicit portable source-file set for a reproducible identity."""
    return canonical_json_hash(_code_file_entries(paths, root=root))


def _manifest_path_for_entry(path, *, output_root=None, staged_root=None):
    """Return ``(identity_path, portable_relative_path)`` for one output."""
    identity_path = Path(path)
    if staged_root is not None:
        stage = Path(staged_root).resolve()
        try:
            relative = identity_path.resolve().relative_to(stage).as_posix()
        except ValueError as exc:
            raise CoreError(
                "OUTPUT_OUTSIDE_STAGING",
                {"path": str(identity_path), "staging_root": str(stage), "message": "output is outside staging"},
            ) from exc
        return identity_path, relative
    if output_root is not None:
        root = Path(output_root).resolve()
        try:
            relative = identity_path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise CoreError(
                "OUTPUT_OUTSIDE_ROOT",
                {"path": str(identity_path), "output_root": str(root), "message": "output is outside output root"},
            ) from exc
        return identity_path, relative
    return identity_path, str(identity_path.resolve())


def _manifest_file_identity(path, portable_path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise CoreError("OUTPUT_MISSING", {"path": str(portable_path), "message": "required output is missing"})
    stat = path.stat()
    return {
        "path": str(portable_path).replace("\\", "/"),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
        "sha256": sha256_file(path),
    }


def build_run_manifest(
    *,
    config=None,
    input_paths=None,
    outputs=None,
    output_root=None,
    staged_root=None,
    required_outputs=None,
    code_paths=None,
    status="completed",
    scientific_ok=None,
    **extra,
):
    """Build a versioned provenance record for a headless or GUI run.

    ``outputs`` are mandatory inventory entries: a missing path raises instead
    of being silently omitted.  When ``output_root`` is supplied, paths are
    portable POSIX-relative paths, which is the format used by published run
    manifests.  ``staged_root`` lets a transaction hash staged bytes while
    recording their eventual published paths.
    """
    config_value = canonicalize_json(config or {})
    selected_code_paths = code_paths if code_paths is not None else _default_code_paths()
    code_root = _runtime_code_root()
    code_entries = _code_file_entries(selected_code_paths, root=code_root)
    manifest = {
        "manifest_version": 2,
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "config": config_value,
        "config_hash": canonical_json_hash(config_value),
        "inputs": ordered_input_manifest(input_paths or []),
        "environment": collect_environment(),
        "code_fingerprint": canonical_json_hash(code_entries),
        "code_files": code_entries,
        "code_root": "module_directory",
        "status": str(status),
        "scientific_ok": scientific_ok,
        "outputs": [],
    }
    if output_root is not None:
        manifest["output_root"] = "."
    output_paths = []
    for path in list(outputs or []):
        identity_path, portable_path = _manifest_path_for_entry(
            path,
            output_root=output_root,
            staged_root=staged_root,
        )
        manifest["outputs"].append(_manifest_file_identity(identity_path, portable_path))
        output_paths.append(str(portable_path).replace("\\", "/"))
    required = [str(value).replace("\\", "/") for value in list(required_outputs or output_paths)]
    if required:
        actual = set(output_paths)
        missing = [value for value in required if value not in actual]
        if missing:
            raise CoreError(
                "OUTPUT_MISSING",
                {"missing": missing, "message": "required output inventory is incomplete"},
            )
    manifest["owned_output_paths"] = sorted(set(output_paths))
    manifest["required_output_paths"] = required
    manifest.update(canonicalize_json(extra))
    manifest["manifest_hash"] = canonical_json_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    return manifest


def write_run_manifest(manifest, path):
    """Write a manifest atomically in the target directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonicalize_json(manifest)
    payload["manifest_hash"] = canonical_json_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
        os.replace(str(temporary), str(path))
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def verify_run_manifest(manifest_or_path, *, verify_code=True):
    """Recompute all listed identities and return a non-mutating result."""
    if isinstance(manifest_or_path, Mapping):
        manifest = dict(manifest_or_path)
        manifest_path = None
    else:
        manifest_path = Path(manifest_or_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"ok": False, "code": "MANIFEST_READ_ERROR", "errors": [str(exc)]}
    errors = []
    manifest_root = manifest_path.parent.resolve() if manifest_path is not None else Path.cwd().resolve()
    required_manifest_fields = (
        "manifest_version", "manifest_hash", "config_hash", "inputs", "outputs", "code_fingerprint",
        "required_output_paths", "owned_output_paths",
    )
    missing_manifest_fields = [field for field in required_manifest_fields if field not in manifest]
    if missing_manifest_fields:
        errors.append({"code": "MANIFEST_REQUIRED_FIELD_MISSING", "fields": missing_manifest_fields})
    if manifest.get("manifest_version", 0) >= 2 and "code_files" not in manifest:
        errors.append({"code": "MANIFEST_REQUIRED_FIELD_MISSING", "fields": ["code_files"]})
    expected_hash = manifest.get("manifest_hash")
    if expected_hash is not None:
        actual_hash = canonical_json_hash(
            {key: value for key, value in manifest.items() if key != "manifest_hash"}
        )
        if expected_hash != actual_hash:
            errors.append({"code": "MANIFEST_HASH_MISMATCH", "expected": expected_hash, "observed": actual_hash})
    expected_output_paths = set()
    for section in ("inputs", "outputs"):
        for entry in manifest.get(section, []) or []:
            raw_path = entry.get("path") if isinstance(entry, Mapping) else None
            if not raw_path:
                errors.append({"code": "MANIFEST_ENTRY_INVALID", "section": section, "entry": entry})
                continue
            path = Path(raw_path)
            if section == "outputs" and not path.is_absolute():
                output_root = Path(str(manifest.get("output_root", ".")))
                if output_root.is_absolute():
                    errors.append({"code": "MANIFEST_OUTPUT_ROOT_ABSOLUTE", "path": str(output_root)})
                    continue
                output_base = (manifest_root / output_root).resolve()
                resolved = (output_base / path).resolve()
                try:
                    resolved.relative_to(output_base)
                except ValueError:
                    errors.append({"code": "MANIFEST_OUTPUT_OUTSIDE_ROOT", "path": str(raw_path)})
                    continue
                path = resolved
                expected_output_paths.add(str(raw_path).replace("\\", "/"))
            elif section == "outputs":
                errors.append({"code": "MANIFEST_OUTPUT_ABSOLUTE", "path": str(raw_path)})
                continue
            try:
                observed = file_identity(path)
            except CoreError as exc:
                errors.append({"code": exc.code, "section": section, "path": str(path), "details": exc.details})
                continue
            for key in ("size", "mtime_ns", "sha256"):
                if entry.get(key) != observed.get(key):
                    errors.append(
                        {
                            "code": "FILE_IDENTITY_MISMATCH",
                            "section": section,
                            "path": str(path),
                            "field": key,
                            "expected": entry.get(key),
                            "observed": observed.get(key),
                        }
                    )
    listed_required = {str(value).replace("\\", "/") for value in manifest.get("required_output_paths", []) or []}
    if listed_required - expected_output_paths:
        errors.append({"code": "MANIFEST_REQUIRED_OUTPUT_MISSING", "paths": sorted(listed_required - expected_output_paths)})
    listed_owned = {str(value).replace("\\", "/") for value in manifest.get("owned_output_paths", []) or []}
    if "owned_output_paths" in manifest and listed_owned != expected_output_paths:
        errors.append(
            {
                "code": "MANIFEST_OWNERSHIP_MISMATCH",
                "owned_only": sorted(listed_owned - expected_output_paths),
                "unowned_outputs": sorted(expected_output_paths - listed_owned),
            }
        )
    expected_config_hash = manifest.get("config_hash")
    if expected_config_hash is not None:
        observed_config_hash = canonical_json_hash(manifest.get("config", {}))
        if expected_config_hash != observed_config_hash:
            errors.append(
                {
                    "code": "CONFIG_HASH_MISMATCH",
                    "expected": expected_config_hash,
                    "observed": observed_config_hash,
                }
            )
    if verify_code and manifest.get("code_fingerprint"):
        try:
            code_entries = manifest.get("code_files")
            if code_entries:
                module_root = _runtime_code_root()
                observed_entries = []
                for item in code_entries:
                    if not isinstance(item, Mapping) or not item.get("path"):
                        raise CoreError("CODE_FILE_SET_INVALID", {"entry": item})
                    code_path = (module_root / str(item["path"])).resolve()
                    try:
                        code_path.relative_to(module_root)
                    except ValueError as exc:
                        raise CoreError("CODE_FILE_OUTSIDE_ROOT", {"path": str(code_path)}) from exc
                    observed_sha256 = sha256_file(code_path)
                    if item.get("sha256") != observed_sha256:
                        errors.append(
                            {
                                "code": "CODE_FINGERPRINT_MISMATCH",
                                "path": str(item["path"]).replace("\\", "/"),
                                "expected": item.get("sha256"),
                                "observed": observed_sha256,
                            }
                        )
                    observed_entries.append({"path": str(item["path"]).replace("\\", "/"), "sha256": observed_sha256})
                observed_code = canonical_json_hash(observed_entries)
            else:
                # Version-1 manifests used the module's implicit single file.
                observed_code = code_fingerprint()
        except CoreError as exc:
            errors.append({"code": exc.code, "details": exc.details})
        else:
            if manifest["code_fingerprint"] != observed_code:
                errors.append({"code": "CODE_FINGERPRINT_MISMATCH", "expected": manifest["code_fingerprint"], "observed": observed_code})
    owned_output_paths = manifest.get("owned_output_paths") or sorted(expected_output_paths)
    if manifest_path is not None and owned_output_paths:
        preserved = {str(value).replace("\\", "/") for value in manifest.get("preserved_output_paths", []) or []}

        def is_explicit_ezdic_candidate(relative):
            if relative.startswith("core/"):
                return bool(re.fullmatch(r"core/(?:strain|engineering_strain|poisson_ratio)(?:_[\w.\-]+)?\.(?:txt|png)", relative, flags=re.IGNORECASE)) or relative == f"core/{ORIGIN_OPJU_FILENAME}"
            if relative == "qc/qc_summary.txt":
                return True
            if relative.startswith("optional/"):
                return bool(re.fullmatch(r"optional/(?:publication_figures|correlation_plots)/[\w.\-/]+\.(?:png|tiff|pdf|svg|eps)", relative, flags=re.IGNORECASE)) or bool(re.fullmatch(r"optional/full_csv/(?:strain_results_all_groups\.csv|per_group_results/strain_results_[\w.\-]+\.csv)", relative, flags=re.IGNORECASE)) or bool(re.fullmatch(r"optional/parameters/(?:tracking_parameters|acceptance_summary)\.txt", relative, flags=re.IGNORECASE)) or bool(re.fullmatch(r"optional/overlays/[\w.\-]+/tracked_\d{5}\.png", relative, flags=re.IGNORECASE))
            if relative.startswith("dic/"):
                return relative == "dic/specimen_mask.png" or bool(re.fullmatch(r"dic/frame_\d{4}(?:_(?:u|v|zncc|Exx|Eyy|Exy|exx|eyy|exy|exx_infinitesimal|eyy_infinitesimal|exy_infinitesimal|overlay)|_parameters)?\.(?:txt|csv|png|npz)", relative, flags=re.IGNORECASE))
            return False

        # Scan only explicit ezDIC output patterns.  Arbitrary files under a
        # shared output folder remain user-owned and are not verification
        # failures.  Runs snapshot such preserved paths in their manifest.
        for relative_root in ("core", "qc", "optional", "dic"):
            root = (manifest_root / relative_root).resolve()
            if not root.exists() or not root.is_dir():
                continue
            for candidate in root.rglob("*"):
                if not candidate.is_file():
                    continue
                relative = candidate.resolve().relative_to(manifest_root).as_posix()
                if relative not in expected_output_paths and relative not in preserved and is_explicit_ezdic_candidate(relative):
                    errors.append({"code": "UNEXPECTED_OUTPUT", "path": relative})
    result = {"ok": not errors, "code": "OK" if not errors else "MANIFEST_INVALID", "errors": errors}
    if manifest_path is not None:
        result["manifest"] = str(manifest_path)
    return result

def _validate_finite_image(image, *, path=None):
    """Return an image array or fail closed on non-finite samples."""
    try:
        arr = np.asarray(image)
    except Exception as exc:
        raise CoreError(
            "INVALID_IMAGE",
            {"path": str(path) if path is not None else None, "message": str(exc)},
        ) from exc
    if arr.ndim not in (2, 3) or arr.size == 0:
        raise CoreError(
            "INVALID_IMAGE",
            {
                "path": str(path) if path is not None else None,
                "shape": tuple(int(value) for value in arr.shape),
                "message": "image must be a non-empty 2-D or 3-D array",
            },
        )
    if arr.dtype.kind not in "biufc":
        raise CoreError(
            "INVALID_IMAGE",
            {
                "path": str(path) if path is not None else None,
                "dtype": str(arr.dtype),
                "message": "image samples must be numeric",
            },
        )
    try:
        finite = np.isfinite(arr)
    except TypeError as exc:
        raise CoreError(
            "INVALID_IMAGE",
            {"path": str(path) if path is not None else None, "dtype": str(arr.dtype), "message": str(exc)},
        ) from exc
    if not bool(finite.all()):
        count = int((~finite).sum())
        raise CoreError(
            "NONFINITE_IMAGE",
            {
                "path": str(path) if path is not None else None,
                "count": count,
                "shape": tuple(int(value) for value in arr.shape),
                "dtype": str(arr.dtype),
                "message": f"image contains {count} non-finite sample(s)",
            },
        )
    return arr


def _prepare_pillow_image(im):
    """Return a Pillow image whose array is grayscale or RGB/RGBA."""
    mode = im.mode
    if mode in ("L", "I", "F", "I;16", "I;16B", "I;16L", "RGB", "RGBA"):
        return im
    if mode == "1":
        return im.convert("L")
    if mode in ("P", "PA"):
        return im.convert("RGBA" if ("transparency" in im.info or mode == "PA") else "RGB")
    if mode == "LA":
        return im.convert("RGBA")
    if mode in ("CMYK", "YCbCr", "LAB", "HSV"):
        return im.convert("RGB")
    return im.convert("RGB")


def _color_to_gray(img, *, color_order, path=None):
    """Convert a decoded 2-D/3-D array to grayscale using the decoder's channel order."""
    img = _validate_finite_image(img, path=path)
    if img.ndim == 2:
        return img
    channels = int(img.shape[2])
    if channels == 1:
        return _validate_finite_image(img[:, :, 0], path=path)
    if channels not in (3, 4):
        raise CoreError(
            "INVALID_IMAGE",
            {
                "path": str(path) if path is not None else None,
                "shape": tuple(int(value) for value in img.shape),
                "message": "only 1-, 3-, or 4-channel images can be converted to grayscale",
            },
        )
    if color_order == "rgb":
        code = cv2.COLOR_RGBA2GRAY if channels == 4 else cv2.COLOR_RGB2GRAY
    elif color_order == "bgr":
        code = cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY
    else:
        raise CoreError(
            "INVALID_IMAGE",
            {
                "path": str(path) if path is not None else None,
                "message": f"unsupported colour order: {color_order}",
            },
        )
    try:
        gray = cv2.cvtColor(img, code)
    except cv2.error as exc:
        raise CoreError(
            "INVALID_IMAGE",
            {
                "path": str(path) if path is not None else None,
                "shape": tuple(int(value) for value in img.shape),
                "dtype": str(img.dtype),
                "message": f"colour conversion failed: {exc}",
            },
        ) from exc
    return _validate_finite_image(gray, path=path)


def read_gray_image(path):
    """
    稳健读取图像。
    使用 np.fromfile + cv2.imdecode 解决 Windows 中文路径问题；
    如果失败则用 Pillow 兜底读取。
    """
    path = str(path)
    img = None
    color_order = "bgr"
    errors = []

    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is not None:
                color_order = "bgr"
    except Exception as exc:
        errors.append(f"cv2.imdecode failed: {exc}")
        img = None

    if img is None:
        try:
            with Image.open(path) as im:
                if getattr(im, "n_frames", 1) > 1:
                    im.seek(0)
                img = np.array(_prepare_pillow_image(im))
                color_order = "rgb"
        except Exception as exc:
            errors.append(f"Pillow failed: {exc}")

    if img is None:
        detail = " | ".join(errors) if errors else "unknown error"
        raise CoreError(
            "INPUT_FILE_ERROR",
            {
                "path": path,
                "stage": "decode",
                "message": f"无法读取图片：{path}；原因：{detail}",
            },
        )

    # Validate the decoded samples before colour conversion.  OpenCV/Pillow
    # decoding can otherwise propagate NaN/Inf into a later uint8 cast.
    return _color_to_gray(img, color_order=color_order, path=path)


def _coerce_normalization_bounds(bounds):
    if isinstance(bounds, Mapping):
        nested = bounds.get("bounds")
        if isinstance(nested, Mapping):
            lo = nested.get("lo", nested.get("lower"))
            hi = nested.get("hi", nested.get("upper"))
        else:
            lo = bounds.get("lo", bounds.get("lower", bounds.get("lower_bound")))
            hi = bounds.get("hi", bounds.get("upper", bounds.get("upper_bound")))
    else:
        try:
            lo, hi = bounds
        except (TypeError, ValueError) as exc:
            raise CoreError("INVALID_NORMALIZATION_BOUNDS", {"message": "bounds must contain lo and hi"}) from exc
    try:
        lo = float(lo)
        hi = float(hi)
    except (TypeError, ValueError) as exc:
        raise CoreError("INVALID_NORMALIZATION_BOUNDS", {"message": "normalization bounds must be finite numbers"}) from exc
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        raise CoreError(
            "INVALID_NORMALIZATION_BOUNDS",
            {"lo": lo, "hi": hi, "message": "normalization bounds must be finite and ordered"},
        )
    if hi == lo:
        hi = lo + 1.0
    return lo, hi


def _normalize_with_bounds_array(image, lo, hi):
    arr = _validate_finite_image(image)
    try:
        arr_float = arr.astype(np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CoreError("INVALID_IMAGE", {"dtype": str(arr.dtype), "message": str(exc)}) from exc
    if not np.isfinite(arr_float).all():
        # A very large finite integer/float can overflow float32.  Do not
        # silently turn that conversion into a black/white pixel.
        count = int((~np.isfinite(arr_float)).sum())
        raise CoreError(
            "NONFINITE_IMAGE",
            {"count": count, "dtype": str(arr.dtype), "message": "float32 conversion produced non-finite samples"},
        )
    out = np.clip((arr_float - np.float32(lo)) / np.float32(hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _normalization_sample_metadata(image, lo, hi, *, policy=None, version=NORMALIZATION_VERSION):
    arr = _validate_finite_image(image)
    values = arr.astype(np.float64, copy=False)
    finite_count = int(values.size)
    low_count = int(np.count_nonzero(values < lo))
    high_count = int(np.count_nonzero(values > hi))
    return {
        "normalization_version": str(version),
        "policy": policy or "explicit_bounds",
        "input_dtype": str(arr.dtype),
        "shape": [int(value) for value in arr.shape],
        "bounds": {"lo": float(lo), "hi": float(hi)},
        "lo": float(lo),
        "hi": float(hi),
        "finite_count": finite_count,
        "clip_fraction_low": low_count / finite_count if finite_count else 0.0,
        "clip_fraction_high": high_count / finite_count if finite_count else 0.0,
        "clip_fraction": (low_count + high_count) / finite_count if finite_count else 0.0,
    }


def compute_reference_normalization(reference, *, lower_percentile=DEFAULT_NORMALIZATION_LOWER_PERCENTILE, upper_percentile=DEFAULT_NORMALIZATION_UPPER_PERCENTILE):
    """Compute one deterministic percentile policy from a reference image."""
    arr = _validate_finite_image(reference)
    try:
        lower_percentile = float(lower_percentile)
        upper_percentile = float(upper_percentile)
    except (TypeError, ValueError) as exc:
        raise CoreError("INVALID_NORMALIZATION_POLICY", {"message": "percentiles must be numeric"}) from exc
    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise CoreError(
            "INVALID_NORMALIZATION_POLICY",
            {"lower_percentile": lower_percentile, "upper_percentile": upper_percentile, "message": "percentiles must satisfy 0 <= lower < upper <= 100"},
        )
    values = arr.astype(np.float64, copy=False)
    lo, hi = np.percentile(values, [lower_percentile, upper_percentile])
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    metadata = _normalization_sample_metadata(
        arr,
        lo,
        hi,
        policy="reference_percentile",
    )
    metadata.update(
        {
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "reference_dtype": str(arr.dtype),
            "reference_shape": [int(value) for value in arr.shape],
            "reference_bounds": {"lo": lo, "hi": hi},
        }
    )
    return metadata


def compute_reference_normalization_bounds(reference, *, lower_percentile=DEFAULT_NORMALIZATION_LOWER_PERCENTILE, upper_percentile=DEFAULT_NORMALIZATION_UPPER_PERCENTILE):
    """Return the reference-derived ``{"lo", "hi"}`` bounds plus policy metadata."""
    return compute_reference_normalization(
        reference,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )


def reference_percentile_bounds(reference, *, lower_percentile=DEFAULT_NORMALIZATION_LOWER_PERCENTILE, upper_percentile=DEFAULT_NORMALIZATION_UPPER_PERCENTILE):
    metadata = compute_reference_normalization(
        reference,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    bounds = metadata["reference_bounds"]
    return float(bounds["lo"]), float(bounds["hi"])


def normalize_with_bounds(image, bounds, *, return_metadata=False, policy=None, version=NORMALIZATION_VERSION):
    """Normalize an image with precomputed bounds without recomputing per frame."""
    lo, hi = _coerce_normalization_bounds(bounds)
    out = _normalize_with_bounds_array(image, lo, hi)
    if not return_metadata:
        return out
    metadata = _normalization_sample_metadata(image, lo, hi, policy=policy, version=version)
    return out, metadata


def normalize_dic_float(image, bounds):
    """Reference-fixed intensity scaling without an 8-bit quantization step."""
    lo, hi = _coerce_normalization_bounds(bounds)
    arr = _validate_finite_image(image).astype(np.float64)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(float)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=float)
    return np.clip((arr-lo)*255/(hi-lo), 0, 255)


def normalize_sequence_frames(reference, frames, *, lower_percentile=DEFAULT_NORMALIZATION_LOWER_PERCENTILE, upper_percentile=DEFAULT_NORMALIZATION_UPPER_PERCENTILE):
    """Normalize a fixed-reference sequence with one reference-derived policy."""
    policy = compute_reference_normalization(
        reference,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    reference8, reference_meta = normalize_with_bounds(
        reference,
        policy,
        return_metadata=True,
        policy="reference_percentile",
    )
    normalized_frames = []
    frame_metadata = []
    for frame in frames:
        frame8, frame_meta = normalize_with_bounds(
            frame,
            policy,
            return_metadata=True,
            policy="reference_percentile",
        )
        normalized_frames.append(frame8)
        frame_metadata.append(frame_meta)
    return {
        "reference": reference8,
        "frames": normalized_frames,
        "metadata": {
            "normalization_version": NORMALIZATION_VERSION,
            "policy": "reference_percentile",
            "lower_percentile": float(lower_percentile),
            "upper_percentile": float(upper_percentile),
            "bounds": dict(policy["reference_bounds"]),
            "reference": reference_meta,
            "frames": frame_metadata,
        },
    }


# Synonyms keep the boundary discoverable for callers that use the shorter
# names while preserving one implementation.
compute_reference_bounds = reference_percentile_bounds
normalize_frame_with_bounds = normalize_with_bounds


def normalize_to_uint8(img, lo=None, hi=None):
    """Backward-compatible uint8 normalization with finite-input rejection."""
    if lo is None or hi is None:
        metadata = compute_reference_normalization(img)
        lo, hi = metadata["reference_bounds"]["lo"], metadata["reference_bounds"]["hi"]
    else:
        lo, hi = _coerce_normalization_bounds((lo, hi))
    return _normalize_with_bounds_array(img, lo, hi)


def get_display_image(img8, max_w=1120, max_h=720):
    arr = np.asarray(img8)
    if arr.ndim != 2:
        raise ValueError("get_display_image expects a 2-D grayscale image")
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(np.nan_to_num(arr, nan=0.0)), 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    if scale < 1:
        disp = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp = arr.copy()

    rgb = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
    return rgb, scale


def rect_normalize(x1, y1, x2, y2):
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def clamp_rect(rect, img_shape):
    x, y, w, h = rect
    H, W = img_shape[:2]

    x = max(0, min(int(round(x)), W - 1))
    y = max(0, min(int(round(y)), H - 1))
    w = max(1, min(int(round(w)), W - x))
    h = max(1, min(int(round(h)), H - y))
    return x, y, w, h


def rect_center(rect):
    x, y, w, h = rect
    return float(x + w / 2.0), float(y + h / 2.0)


def move_rect_center(rect, new_cx=None, new_cy=None, img_shape=None):
    x, y, w, h = rect
    cx, cy = rect_center(rect)
    if new_cx is None:
        new_cx = cx
    if new_cy is None:
        new_cy = cy

    new_x = int(round(new_cx - w / 2.0))
    new_y = int(round(new_cy - h / 2.0))
    out = (new_x, new_y, w, h)
    if img_shape is not None:
        out = clamp_rect(out, img_shape)
    return out


def center_distance(rect_a, rect_b):
    ax, ay = rect_center(rect_a)
    bx, by = rect_center(rect_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def roi_separation(rect1, rect2):
    x1, y1 = rect_center(rect1)
    x2, y2 = rect_center(rect2)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dist = math.sqrt(dx ** 2 + dy ** 2)
    return dx, dy, dist


def resolve_strain_mode(rect1, rect2, selected_mode):
    """
    自动判断应变方向：
    - 左右分开明显：x
    - 上下分开明显：y
    - 倾斜明显：distance
    """
    if selected_mode != "auto":
        return selected_mode

    dx, dy, dist = roi_separation(rect1, rect2)

    if dx >= 3.0 * max(dy, 1.0):
        return "x"
    if dy >= 3.0 * max(dx, 1.0):
        return "y"
    return "distance"


def length_between(rect1, rect2, mode="x"):
    x1, y1 = rect_center(rect1)
    x2, y2 = rect_center(rect2)

    if mode == "y":
        return abs(y2 - y1)
    if mode == "x":
        return abs(x2 - x1)
    if mode == "distance":
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    raise ValueError("应变方向只能是 x, y 或 distance。")


def extract_patch(img8, rect):
    x, y, w, h = clamp_rect(rect, img8.shape)
    return img8[y:y + h, x:x + w].astype(np.float32)


def extract_patch_subpixel(img8, rect):
    """Extract a registered ROI patch at a floating-point top-left position."""
    image = _as_float_image(img8)
    x, y, w, h = rect
    w, h = int(round(w)), int(round(h))
    if w <= 0 or h <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    center = (float(x) + (w - 1) / 2.0, float(y) + (h - 1) / 2.0)
    height, width = image.shape[:2]
    if (
        center[0] - (w - 1) / 2.0 < 0
        or center[1] - (h - 1) / 2.0 < 0
        or center[0] + (w - 1) / 2.0 >= width
        or center[1] + (h - 1) / 2.0 >= height
    ):
        return np.zeros((0, 0), dtype=np.float32)
    patch = cv2.getRectSubPix(image, (w, h), center)
    return np.asarray(patch, dtype=np.float32) if patch is not None else np.zeros((0, 0), dtype=np.float32)


def _periodicity_profile_metrics(patch):
    """Measure repeatability of the two axis-mean profiles.

    This is a deliberately conservative companion to the structure-tensor
    rank test.  It is recorded for auditability and catches an oscillatory
    profile even when a future tensor implementation is changed.  A periodic
    score is never used to exempt a rank-one patch from rejection.
    """
    best_score = 0.0
    best_period = None
    best_axis = None
    array = np.asarray(patch, dtype=np.float64)
    for axis, profile in (("x", np.mean(array, axis=0)), ("y", np.mean(array, axis=1))):
        profile = np.asarray(profile, dtype=np.float64).ravel()
        n = profile.size
        if n < 8 or not np.isfinite(profile).all():
            continue
        centered = profile - float(np.mean(profile))
        energy = float(np.dot(centered, centered))
        if energy <= 1e-12:
            continue
        # Compare only non-trivial lags.  Restricting the search to half the
        # profile prevents a single end-to-end correlation from being called a
        # repeat period and keeps the metric stable across ROI sizes.
        max_lag = max(2, n // 2)
        for lag in range(2, max_lag + 1):
            score = float(np.dot(centered[:-lag], centered[lag:]) / energy)
            if score > best_score:
                best_score = score
                best_period = int(lag)
                best_axis = axis
    return {
        "periodicity_score": float(np.clip(best_score, -1.0, 1.0)),
        "periodicity_period_px": best_period,
        "periodicity_axis": best_axis,
        "discriminator_version": TEXTURE_DISCRIMINATOR_VERSION,
    }


def roi_texture_metrics(img8, rect):
    patch = extract_patch(img8, rect)
    if patch.size == 0:
        return {
            "std_gray": 0.0,
            "contrast_p95_p5": 0.0,
            "low_frac": 1.0,
            "high_frac": 1.0,
            "structure_tensor_ratio": 0.0,
            "texture_rank_ratio": 0.0,
            "minor_to_major_gradient_eigenvalue_ratio": 0.0,
            "rank_one_ratio": 0.0,
            "rank_one_score": 1.0,
            "gradient_energy": 0.0,
            "directional_gradient_coherence": 0.0,
            "structure_tensor_eigenvalues": (0.0, 0.0),
            "metrics_version": TEXTURE_METRICS_VERSION,
            **_periodicity_profile_metrics(np.zeros((0, 0), dtype=np.float64)),
        }

    patch = _validate_finite_image(patch)
    p5, p95 = np.percentile(patch, [5, 95])
    patch_float = patch.astype(np.float64, copy=False)
    if patch_float.ndim == 2 and min(patch_float.shape) >= 2:
        gradient_y, gradient_x = np.gradient(patch_float)
        tensor_xx = float(np.mean(gradient_x * gradient_x))
        tensor_yy = float(np.mean(gradient_y * gradient_y))
        tensor_xy = float(np.mean(gradient_x * gradient_y))
        tensor = np.array([[tensor_xx, tensor_xy], [tensor_xy, tensor_yy]], dtype=float)
        eigenvalues = np.linalg.eigvalsh(tensor)
        minor = max(float(eigenvalues[0]), 0.0)
        major = max(float(eigenvalues[1]), 0.0)
        ratio = minor / major if major > 1e-12 else 0.0
        gradient_energy = tensor_xx + tensor_yy
        directional_coherence = (
            float(np.hypot(np.mean(gradient_x), np.mean(gradient_y)))
            / max(float(np.sqrt(gradient_energy)), 1e-12)
        )
    else:
        minor = major = ratio = gradient_energy = directional_coherence = 0.0
    metrics = {
        "std_gray": float(np.std(patch)),
        "contrast_p95_p5": float(p95 - p5),
        "low_frac": float(np.mean(patch <= 5)),
        "high_frac": float(np.mean(patch >= 250)),
        "structure_tensor_ratio": float(ratio),
        "texture_rank_ratio": float(ratio),
        "minor_to_major_gradient_eigenvalue_ratio": float(ratio),
        "gradient_energy": float(gradient_energy),
        "directional_gradient_coherence": float(np.clip(directional_coherence, 0.0, 1.0)),
        "structure_tensor_eigenvalues": (float(minor), float(major)),
        "metrics_version": TEXTURE_METRICS_VERSION,
    }
    metrics.update(_periodicity_profile_metrics(patch_float))
    # Explicitly retain both names in the manifest/row contract.  The ratio is
    # the measured rank-one discriminator; the score is an auxiliary periodic
    # diagnostic and is intentionally not allowed to override rank rejection.
    metrics["rank_one_ratio"] = float(ratio)
    metrics["rank_one_score"] = float(np.clip(1.0 - ratio, 0.0, 1.0))
    return metrics


def texture_failure_code(
    metrics,
    min_std,
    min_contrast,
    max_saturated_frac,
    min_structure_ratio=DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO,
    max_directional_coherence=DEFAULT_TEXTURE_MAX_DIRECTIONAL_COHERENCE,
    min_periodicity_score=DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE,
):
    """Return a stable rejection code, or ``None`` for acceptable texture."""
    try:
        min_std = float(min_std)
        min_contrast = float(min_contrast)
        max_saturated_frac = float(max_saturated_frac)
        min_structure_ratio = float(min_structure_ratio)
        max_directional_coherence = float(max_directional_coherence)
        min_periodicity_score = float(min_periodicity_score)
    except (TypeError, ValueError):
        return "INVALID_TEXTURE_THRESHOLDS"
    if not np.isfinite([min_std, min_contrast, max_saturated_frac, min_structure_ratio, max_directional_coherence, min_periodicity_score]).all():
        return "INVALID_TEXTURE_THRESHOLDS"
    if float(metrics.get("std_gray", 0.0)) < min_std or float(metrics.get("contrast_p95_p5", 0.0)) < min_contrast:
        return "LOW_TEXTURE"
    ratio = float(metrics.get("structure_tensor_ratio", metrics.get("texture_rank_ratio", 0.0)))
    periodicity_score = float(metrics.get("periodicity_score", 0.0))
    # A rank-one patch is ambiguous regardless of its local mean-gradient
    # coherence.  The former coherence exemption allowed a long-period stripe
    # (or a single monotonic segment of it) to reach the solver.  The explicit
    # versioned discriminator keeps the scientific gate fail-closed for both
    # multi-period and partial-period sinusoidal fixtures.
    if ratio < min_structure_ratio:
        return "AMBIGUOUS_TEXTURE"
    # Keep the periodic signal as a defensive second branch for near-rank-one
    # patches whose tensor ratio lands just above the configured floor.  The
    # rank floor is deliberately not relaxed; this branch only rejects.
    if periodicity_score >= min_periodicity_score and ratio <= max(min_structure_ratio * 5.0, 0.10):
        return "AMBIGUOUS_TEXTURE"
    if float(metrics.get("low_frac", 1.0)) > max_saturated_frac or float(metrics.get("high_frac", 1.0)) > max_saturated_frac:
        return "SATURATED_TEXTURE"
    return None


def texture_is_ok(
    metrics,
    min_std,
    min_contrast,
    max_saturated_frac,
    min_structure_ratio=DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO,
    max_directional_coherence=DEFAULT_TEXTURE_MAX_DIRECTIONAL_COHERENCE,
    min_periodicity_score=DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE,
    *,
    raise_on_failure=False,
    return_reason=False,
):
    """Validate texture while retaining the historical boolean API."""
    code = texture_failure_code(
        metrics,
        min_std,
        min_contrast,
        max_saturated_frac,
        min_structure_ratio,
        max_directional_coherence,
        min_periodicity_score,
    )
    if raise_on_failure and code is not None:
        raise CoreError(code, {"metrics": dict(metrics), "message": code})
    ok = code is None
    return (ok, code) if return_reason else ok


def require_texture(
    image,
    rect,
    min_std=8.0,
    min_contrast=25.0,
    max_saturated_frac=0.20,
    min_structure_ratio=DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO,
    max_directional_coherence=DEFAULT_TEXTURE_MAX_DIRECTIONAL_COHERENCE,
    min_periodicity_score=DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE,
):
    """Raise ``CoreError`` for texture that cannot support unique tracking."""
    metrics = roi_texture_metrics(image, rect)
    code = texture_failure_code(
        metrics,
        min_std,
        min_contrast,
        max_saturated_frac,
        min_structure_ratio,
        max_directional_coherence,
        min_periodicity_score,
    )
    if code is not None:
        raise CoreError(
            code,
            {
                "roi": tuple(int(round(value)) for value in rect),
                "metrics": metrics,
                "min_std": float(min_std),
                "min_contrast": float(min_contrast),
                "max_saturated_frac": float(max_saturated_frac),
                "min_structure_ratio": float(min_structure_ratio),
                "max_directional_coherence": float(max_directional_coherence),
                "min_periodicity_score": float(min_periodicity_score),
                "metrics_version": TEXTURE_METRICS_VERSION,
                "discriminator_version": TEXTURE_DISCRIMINATOR_VERSION,
                "preflight_version": TEXTURE_PREFLIGHT_VERSION,
                "message": code,
            },
        )
    return metrics


validate_roi_texture = require_texture


def has_nonzero_variance(values, *, min_std=1e-8):
    """Return whether an image/template patch has finite, usable contrast.

    OpenCV's ``TM_CCOEFF_NORMED`` is undefined for a constant template or
    candidate patch and can return a misleading score of 1.0.  Keep this
    check independent of OpenCV so all normalized-correlation entry points
    share the same scientific rejection rule.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    return bool(float(np.std(finite)) > float(min_std))


def subpixel_peak(corr, px, py):
    H, W = corr.shape
    dx = 0.0
    dy = 0.0

    if 1 <= px < W - 1:
        c1 = corr[py, px - 1]
        c2 = corr[py, px]
        c3 = corr[py, px + 1]
        denom = c1 - 2 * c2 + c3
        if abs(denom) > 1e-12:
            dx = 0.5 * (c1 - c3) / denom

    if 1 <= py < H - 1:
        c1 = corr[py - 1, px]
        c2 = corr[py, px]
        c3 = corr[py + 1, px]
        denom = c1 - 2 * c2 + c3
        if abs(denom) > 1e-12:
            dy = 0.5 * (c1 - c3) / denom

    return float(np.clip(dx, -0.5, 0.5)), float(np.clip(dy, -0.5, 0.5))


def _empty_template_match_result(last_rect, reason):
    return {
        "candidate_rect": tuple(last_rect),
        "rect": tuple(last_rect),
        "score": -1.0,
        "zncc": -1.0,
        "best_peak": -1.0,
        "second_peak": -1.0,
        "peak_margin": 0.0,
        "peak_ratio": 1.0,
        "best_to_second_peak_ratio": 1.0,
        "second_to_best_peak_ratio": 1.0,
        "peak_is_ambiguous": False,
        "reason": str(reason),
    }


def _match_template_candidate_diagnostic(img8, last_rect, template, search_radius):
    H, W = img8.shape[:2]
    x, y, w, h = last_rect
    x = float(x)
    y = float(y)
    w = int(round(w))
    h = int(round(h))

    sx1 = int(max(0, math.floor(x - search_radius)))
    sy1 = int(max(0, math.floor(y - search_radius)))
    sx2 = int(min(W, math.ceil(x + w + search_radius)))
    sy2 = int(min(H, math.ceil(y + h + search_radius)))

    search_img = img8[sy1:sy2, sx1:sx2].astype(np.float32)

    if search_img.shape[0] < h or search_img.shape[1] < w:
        return _empty_template_match_result(last_rect, "SEARCH_TOO_SMALL")

    if template.shape[0] != h or template.shape[1] != w:
        return _empty_template_match_result(last_rect, "TEMPLATE_SHAPE_MISMATCH")

    # TM_CCOEFF_NORMED is undefined for constant inputs.  OpenCV may report
    # a perfect score for these patches, so reject before calling it and also
    # validate the selected candidate window below.
    if not has_nonzero_variance(template) or not has_nonzero_variance(search_img):
        return _empty_template_match_result(last_rect, "ZERO_VARIANCE")

    corr = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
    finite_corr = np.isfinite(corr)
    if not finite_corr.any():
        return _empty_template_match_result(last_rect, "NONFINITE_CORRELATION")
    safe_corr = np.where(finite_corr, corr, -1.0).astype(np.float32, copy=False)
    _, max_val, _, max_loc = cv2.minMaxLoc(safe_corr)
    if not np.isfinite(max_val):
        return _empty_template_match_result(last_rect, "NONFINITE_PEAK")

    px, py = max_loc
    candidate_patch = search_img[py : py + h, px : px + w]
    if candidate_patch.shape != template.shape or not has_nonzero_variance(candidate_patch):
        return _empty_template_match_result(last_rect, "ZERO_VARIANCE_CANDIDATE")
    dx, dy = subpixel_peak(safe_corr, px, py)

    new_x = sx1 + px + dx
    new_y = sy1 + py + dy

    peak_diagnostics = _correlation_peak_diagnostics(safe_corr, max_loc)
    peak_is_ambiguous = bool(
        np.isfinite(peak_diagnostics["peak_margin"])
        and np.isfinite(peak_diagnostics["peak_ratio"])
        and peak_diagnostics["peak_margin"] < DEFAULT_PEAK_MARGIN_MIN
        and peak_diagnostics["peak_ratio"] < DEFAULT_PEAK_RATIO_MIN
    )
    return {
        "candidate_rect": (float(new_x), float(new_y), w, h),
        "rect": (float(new_x), float(new_y), w, h),
        "score": float(max_val),
        "zncc": float(max_val),
        **peak_diagnostics,
        "peak_is_ambiguous": peak_is_ambiguous,
        "reason": "peak_found",
    }


def match_template_candidate(img8, last_rect, template, search_radius, *, return_diagnostics=False):
    """Track a 1D ROI; default return remains the historical two-tuple."""
    result = _match_template_candidate_diagnostic(img8, last_rect, template, search_radius)
    if return_diagnostics:
        return result
    return result["candidate_rect"], result["score"]


def match_template_candidate_diagnostic(img8, last_rect, template, search_radius):
    return _match_template_candidate_diagnostic(img8, last_rect, template, search_radius)


_CANONICAL_MATCH_TEMPLATE_CANDIDATE = match_template_candidate


def update_template_from_rect(img8, rect, old_template, alpha):
    x, y, w, h = rect
    x = float(x)
    y = float(y)
    w = int(round(w))
    h = int(round(h))

    H, W = img8.shape[:2]
    center_x = x + (w - 1) / 2.0
    center_y = y + (h - 1) / 2.0
    if center_x - (w - 1) / 2.0 < 0 or center_y - (h - 1) / 2.0 < 0:
        return old_template
    if center_x + (w - 1) / 2.0 >= W or center_y + (h - 1) / 2.0 >= H:
        return old_template

    # ``getRectSubPix`` preserves the sub-pixel registration returned by the
    # correlation peak; integer slicing here would introduce template drift.
    patch = extract_patch_subpixel(img8, rect)
    if patch.size == 0:
        return old_template
    if patch.shape != old_template.shape:
        return old_template

    return (1.0 - alpha) * old_template + alpha * patch


def forward_backward_error(prev_img8, curr_img8, prev_rect, curr_candidate_rect, search_radius):
    """
    前后向一致性检查：
    curr 候选位置若真实，应能用当前 patch 反追踪回 prev_rect。
    """
    curr_patch = extract_patch_subpixel(curr_img8, curr_candidate_rect)
    back_rect, back_score = match_template_candidate(
        prev_img8,
        prev_rect,
        curr_patch,
        search_radius=search_radius,
    )
    if back_score < 0 or not np.isfinite(back_score):
        return float("inf"), float(back_score)
    err = center_distance(back_rect, prev_rect)
    return float(err), float(back_score)


_CANONICAL_FORWARD_BACKWARD_ERROR = forward_backward_error


def initialize_extensometer_group_state(
    first_img8,
    group,
    *,
    min_texture_std=8.0,
    min_texture_contrast=25.0,
    max_saturated_frac=0.20,
    min_structure_ratio=DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO,
    max_directional_coherence=DEFAULT_TEXTURE_MAX_DIRECTIONAL_COHERENCE,
    min_periodicity_score=DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE,
):
    """Create explicit mutable state for one two-ROI virtual extensometer."""
    image = _as_float_image(first_img8)
    try:
        rect1 = tuple(group["roi1"])
        rect2 = tuple(group["roi2"])
        actual_mode = str(group.get("actual_mode") or resolve_strain_mode(rect1, rect2, "auto"))
    except (KeyError, TypeError, ValueError) as exc:
        raise CoreError("INVALID_ROI_GROUP", {"message": str(exc)}) from exc
    if len(rect1) != 4 or len(rect2) != 4:
        raise CoreError("INVALID_ROI_GROUP", {"message": "each ROI must contain x, y, width, height"})
    if not rect_is_inside_image(rect1, image.shape) or not rect_is_inside_image(rect2, image.shape):
        raise CoreError("ROI_OUT_OF_BOUNDS", {"roi1": rect1, "roi2": rect2})
    texture1 = roi_texture_metrics(image, rect1)
    texture2 = roi_texture_metrics(image, rect2)
    texture_code1 = texture_failure_code(
        texture1,
        min_texture_std,
        min_texture_contrast,
        max_saturated_frac,
        min_structure_ratio,
        max_directional_coherence,
        min_periodicity_score,
    )
    texture_code2 = texture_failure_code(
        texture2,
        min_texture_std,
        min_texture_contrast,
        max_saturated_frac,
        min_structure_ratio,
        max_directional_coherence,
        min_periodicity_score,
    )
    ambiguous = [
        {"roi": "ROI1", "code": texture_code1, "metrics": texture1},
        {"roi": "ROI2", "code": texture_code2, "metrics": texture2},
    ]
    ambiguous = [item for item in ambiguous if item["code"] == "AMBIGUOUS_TEXTURE"]
    if ambiguous:
        raise CoreError(
            "AMBIGUOUS_TEXTURE",
            {"group": group.get("name", ""), "rois": ambiguous, "message": "periodic/near-1D ROI texture is not unique"},
        )
    L0 = length_between(rect1, rect2, actual_mode)
    return {
        "group": dict(group),
        "last_good_rect1": rect1,
        "last_good_rect2": rect2,
        "template1": extract_patch(image, rect1),
        "template2": extract_patch(image, rect2),
        "last_good_template1": extract_patch(image, rect1),
        "last_good_template2": extract_patch(image, rect2),
        "last_good_img8": image.copy(),
        "last_good_img": image.copy(),
        "L0": float(L0),
        "last_valid_strain": 0.0,
        "last_good_strain": 0.0,
        "consecutive_fail_count": 0,
        "texture_metrics1": texture1,
        "texture_metrics2": texture2,
        "texture_code1": texture_code1,
        "texture_code2": texture_code2,
        "texture_thresholds": {
            "min_texture_std": float(min_texture_std),
            "min_texture_contrast": float(min_texture_contrast),
            "max_saturated_frac": float(max_saturated_frac),
            "min_structure_ratio": float(min_structure_ratio),
            "max_directional_coherence": float(max_directional_coherence),
            "min_periodicity_score": float(min_periodicity_score),
            "metrics_version": TEXTURE_METRICS_VERSION,
            "discriminator_version": TEXTURE_DISCRIMINATOR_VERSION,
            "preflight_version": TEXTURE_PREFLIGHT_VERSION,
        },
        "template_policy": "fixed_reference",
    }


def _tracking_param(params, name, default):
    value = params.get(name, default) if isinstance(params, Mapping) else default
    return default if value is None else value


def _tracking_template_match(image, rect, template, search_radius):
    candidate = match_template_candidate
    adapter = sys.modules.get("dic_virtual_extensometer_gui_v7_multi_roi_range")
    if adapter is not None:
        adapter_candidate = getattr(adapter, "match_template_candidate", candidate)
        if callable(adapter_candidate) and adapter_candidate is not _CANONICAL_MATCH_TEMPLATE_CANDIDATE:
            candidate = adapter_candidate
    try:
        raw = candidate(image, rect, template, search_radius, return_diagnostics=True)
    except TypeError:
        raw = candidate(image, rect, template, search_radius)
    if isinstance(raw, Mapping):
        result = dict(raw)
        result.setdefault("candidate_rect", rect)
        result.setdefault("score", result.get("zncc", -1.0))
        result.setdefault("best_peak", result.get("score", -1.0))
        result.setdefault("second_peak", np.nan)
        result.setdefault("peak_margin", np.nan)
        result.setdefault("peak_ratio", np.nan)
        result.setdefault("best_to_second_peak_ratio", result.get("peak_ratio"))
        result.setdefault("second_to_best_peak_ratio", np.nan)
        result.setdefault("peak_is_ambiguous", False)
        return result
    try:
        candidate_rect, score = raw
    except (TypeError, ValueError):
        return _empty_template_match_result(rect, "INVALID_RETURN")
    return {
        **_empty_template_match_result(candidate_rect, "legacy_tuple"),
        "candidate_rect": candidate_rect,
        "score": float(score),
        "best_peak": float(score),
    }


def _tracking_forward_backward_error(prev_img8, curr_img8, prev_rect, candidate_rect, search_radius):
    if candidate_rect is None:
        return float("inf"), float("nan")
    candidate = forward_backward_error
    adapter = sys.modules.get("dic_virtual_extensometer_gui_v7_multi_roi_range")
    if adapter is not None:
        adapter_candidate = getattr(adapter, "forward_backward_error", candidate)
        if callable(adapter_candidate) and adapter_candidate is not _CANONICAL_FORWARD_BACKWARD_ERROR:
            candidate = adapter_candidate
    try:
        return candidate(prev_img8, curr_img8, prev_rect, candidate_rect, search_radius)
    except TypeError:
        # Defensive fallback for a legacy implementation that does not accept
        # the positional search-radius spelling.
        return candidate(prev_img8, curr_img8, prev_rect, candidate_rect, search_radius=search_radius)


def track_extensometer_group_frame(state, img8, frame_idx, filename, params):
    """Advance one ROI group state and return the legacy row/overlay pair."""
    image = _as_float_image(img8)
    group = state["group"]
    group_name = group.get("name", "")
    actual_mode = str(group.get("actual_mode") or resolve_strain_mode(group["roi1"], group["roi2"], "auto"))
    search_radius_base = int(_tracking_param(params, "search_radius_base", 180))
    hard_corr = float(_tracking_param(params, "hard_corr", 0.55))
    soft_corr = float(_tracking_param(params, "soft_corr", 0.35))
    enable_adaptive = bool(_tracking_param(params, "enable_adaptive", True))
    use_prev_frame_template = bool(_tracking_param(params, "use_prev_frame_template", False))
    template_alpha = float(_tracking_param(params, "template_alpha", 0.70))
    max_frame_jump = _tracking_param(params, "max_frame_jump", None)
    enable_fb_check = bool(_tracking_param(params, "enable_fb_check", True))
    fb_tolerance = float(_tracking_param(params, "fb_tolerance", _tracking_param(params, "fb_tolerance_px", 12.0)))
    pixel_size_mm = _tracking_param(params, "pixel_size_mm", None)
    peak_margin_min = float(_tracking_param(params, "peak_margin_min", DEFAULT_PEAK_MARGIN_MIN))
    peak_ratio_min = float(_tracking_param(params, "peak_ratio_min", DEFAULT_PEAK_RATIO_MIN))
    template_policy = "experimental_follow" if use_prev_frame_template else "fixed_reference"
    state["template_policy"] = template_policy
    last_rect1 = tuple(state["last_good_rect1"])
    last_rect2 = tuple(state["last_good_rect2"])
    template1 = state["template1"]
    template2 = state["template2"]
    L0 = float(state["L0"])
    last_valid_strain = state.get("last_valid_strain", np.nan)
    consecutive_fail_count = int(state.get("consecutive_fail_count", 0))
    candidate_rect1 = candidate_rect2 = None
    diag1 = _empty_template_match_result(last_rect1, "NOT_RUN")
    diag2 = _empty_template_match_result(last_rect2, "NOT_RUN")
    score1 = score2 = 1.0
    fb_err1 = fb_err2 = fb_score1 = fb_score2 = np.nan
    fb_ok = None
    search_radius_used = 0
    accepted = False
    accept_mode = "rejected"
    tracking_status_code = "REJECTED"
    reason = "rejected"
    used_rect1, used_rect2 = last_rect1, last_rect2
    L = strain = true_strain = np.nan
    strain_valid = False

    if frame_idx == 0:
        template_ok1 = has_nonzero_variance(template1)
        template_ok2 = has_nonzero_variance(template2)
        score1 = 1.0 if template_ok1 else -1.0
        score2 = 1.0 if template_ok2 else -1.0
        if template_ok1 and template_ok2 and np.isfinite(L0) and L0 > 0:
            accepted = True
            accept_mode = "initial"
            tracking_status_code = "INITIAL_ACCEPTED"
            reason = "initial frame"
            L = L0
            strain = 0.0
            true_strain = 0.0
            strain_valid = True
        else:
            if not (np.isfinite(L0) and L0 > 0):
                tracking_status_code = "NONFINITE_LENGTH"
                reason = "non-positive or non-finite gauge length; frame rejected"
            else:
                tracking_status_code = "ZERO_VARIANCE_TEMPLATE"
                reason = "zero-variance template; frame rejected"
            state["consecutive_fail_count"] = consecutive_fail_count + 1
    else:
        search_radius_used = int(search_radius_base * min(5, 1 + consecutive_fail_count))
        diag1 = _tracking_template_match(image, last_rect1, template1, search_radius_used)
        diag2 = _tracking_template_match(image, last_rect2, template2, search_radius_used)
        candidate_rect1, candidate_rect2 = diag1["candidate_rect"], diag2["candidate_rect"]
        score1, score2 = float(diag1["score"]), float(diag2["score"])
        ambiguous = bool(
            (
                np.isfinite(float(diag1.get("peak_margin", np.nan)))
                and np.isfinite(float(diag1.get("peak_ratio", np.nan)))
                and float(diag1.get("peak_margin")) < peak_margin_min
                and float(diag1.get("peak_ratio")) < peak_ratio_min
            )
            or (
                np.isfinite(float(diag2.get("peak_margin", np.nan)))
                and np.isfinite(float(diag2.get("peak_ratio", np.nan)))
                and float(diag2.get("peak_margin")) < peak_margin_min
                and float(diag2.get("peak_ratio")) < peak_ratio_min
            )
            or diag1.get("peak_is_ambiguous", False)
            or diag2.get("peak_is_ambiguous", False)
        )
        candidate_ok = score1 >= 0 and score2 >= 0 and candidate_rect1 is not None and candidate_rect2 is not None
        if candidate_ok:
            try:
                candidate_L = float(length_between(candidate_rect1, candidate_rect2, actual_mode))
            except (TypeError, ValueError):
                candidate_L = np.nan
        else:
            candidate_L = np.nan
        candidate_strain = (candidate_L - L0) / L0 if L0 > 0 and np.isfinite(candidate_L) else np.nan
        finite_strain = bool(np.isfinite(candidate_strain) and (1.0 + candidate_strain) > 0)
        jump_value = abs(candidate_strain - last_valid_strain) if finite_strain and np.isfinite(last_valid_strain) else np.inf
        ok_jump = finite_strain and (
            max_frame_jump in (None, "") or jump_value <= float(max_frame_jump)
        )
        ok_hard_corr = score1 >= hard_corr and score2 >= hard_corr and not ambiguous
        ok_soft_corr = score1 >= soft_corr and score2 >= soft_corr and not ambiguous
        ok_fb = True
        if enable_fb_check:
            fb_err1, fb_score1 = _tracking_forward_backward_error(
                state["last_good_img8"], image, last_rect1, candidate_rect1, search_radius_used
            )
            fb_err2, fb_score2 = _tracking_forward_backward_error(
                state["last_good_img8"], image, last_rect2, candidate_rect2, search_radius_used
            )
            fb_ok = bool(fb_err1 <= fb_tolerance and fb_err2 <= fb_tolerance)
            ok_fb = fb_ok
        accepted = bool(ok_hard_corr and ok_jump and ok_fb)
        if accepted:
            accept_mode = "hard"
            tracking_status_code = "ACCEPTED_HARD"
            reason = "hard corr + continuous strain" + (" + FB check" if enable_fb_check else " + FB disabled")
        elif enable_adaptive and bool(ok_soft_corr and ok_jump and ok_fb):
            accepted = True
            accept_mode = "adaptive"
            tracking_status_code = "ACCEPTED_ADAPTIVE"
            reason = "soft corr + continuous strain" + (" + FB check" if enable_fb_check else " + FB disabled")
        else:
            accepted = False
            accept_mode = "rejected"
            if ambiguous:
                tracking_status_code = "AMBIGUOUS_PEAK"
                reason = "repeated correlation peak; frame rejected"
            elif enable_fb_check and not ok_fb:
                tracking_status_code = "FB_FAILED"
                reason = f"FB fail: ROI1 {fb_err1:.2f}px, ROI2 {fb_err2:.2f}px > {fb_tolerance:.2f}px"
            elif not finite_strain:
                tracking_status_code = "NONFINITE_STRAIN"
                reason = "non-finite or non-positive gauge length"
            elif not ok_jump:
                tracking_status_code = "STRAIN_JUMP"
                reason = f"strain jump {jump_value:.4f} > {max_frame_jump}"
            elif not ok_hard_corr and not ok_soft_corr:
                tracking_status_code = "CORRELATION_BELOW_THRESHOLD"
                reason = f"correlation below soft threshold: ROI1 {score1:.3f}, ROI2 {score2:.3f}"
            else:
                tracking_status_code = "REJECTED"
                reason = "tracking rejected"

        if accepted:
            used_rect1, used_rect2 = candidate_rect1, candidate_rect2
            L, strain = candidate_L, candidate_strain
            true_strain = math.log1p(strain) if finite_strain else np.nan
            strain_valid = bool(np.isfinite(L) and np.isfinite(strain) and (1.0 + strain) > 0)
            state["last_good_rect1"] = used_rect1
            state["last_good_rect2"] = used_rect2
            state["last_valid_strain"] = strain
            state["last_good_strain"] = strain
            state["consecutive_fail_count"] = 0
            if use_prev_frame_template:
                state["template1"] = update_template_from_rect(image, used_rect1, template1, template_alpha)
                state["template2"] = update_template_from_rect(image, used_rect2, template2, template_alpha)
                state["last_good_template1"] = state["template1"]
                state["last_good_template2"] = state["template2"]
            state["last_good_img8"] = image.copy()
            state["last_good_img"] = state["last_good_img8"]
        else:
            # Rejected rows expose NaN for current measurements and leave every
            # last-good rectangle/template/image/strain value untouched.
            state["consecutive_fail_count"] = consecutive_fail_count + 1

    c1x, c1y = rect_center(used_rect1)
    c2x, c2y = rect_center(used_rect2)
    texture1 = state.get("texture_metrics1", {})
    texture2 = state.get("texture_metrics2", {})
    texture_thresholds = state.get("texture_thresholds", {})

    def eigen_component(metrics, index):
        values = metrics.get("structure_tensor_eigenvalues")
        try:
            return values[index]
        except (IndexError, KeyError, TypeError):
            return np.nan

    row = {
        "frame": frame_idx,
        "filename": filename,
        "group": group_name,
        "role": normalize_roi_role(group.get("role", "none")),
        "selected_mode": group.get("selected_mode", "auto"),
        "actual_mode": actual_mode,
        "accepted": bool(accepted),
        "strain_valid": bool(strain_valid),
        "accept_mode": accept_mode,
        "reason": reason,
        "tracking_status_code": tracking_status_code,
        "invalid_reason": "" if strain_valid else tracking_status_code,
        "consecutive_fail_count": state["consecutive_fail_count"],
        "corr_score_roi1": score1,
        "corr_score_roi2": score2,
        "best_peak_roi1": diag1.get("best_peak", score1),
        "second_peak_roi1": diag1.get("second_peak", np.nan),
        "peak_margin_roi1": diag1.get("peak_margin", np.nan),
        "best_to_second_peak_ratio_roi1": diag1.get("best_to_second_peak_ratio", np.nan),
        "second_to_best_peak_ratio_roi1": diag1.get("second_to_best_peak_ratio", np.nan),
        "best_peak_roi2": diag2.get("best_peak", score2),
        "second_peak_roi2": diag2.get("second_peak", np.nan),
        "peak_margin_roi2": diag2.get("peak_margin", np.nan),
        "best_to_second_peak_ratio_roi2": diag2.get("best_to_second_peak_ratio", np.nan),
        "second_to_best_peak_ratio_roi2": diag2.get("second_to_best_peak_ratio", np.nan),
        "peak_is_ambiguous_roi1": bool(diag1.get("peak_is_ambiguous", False)),
        "peak_is_ambiguous_roi2": bool(diag2.get("peak_is_ambiguous", False)),
        "fb_error_roi1_px": fb_err1,
        "fb_error_roi2_px": fb_err2,
        "fb_score_roi1": fb_score1,
        "fb_score_roi2": fb_score2,
        "fb_check_enabled": bool(enable_fb_check),
        "fb_status": (
            "not_applicable_initial"
            if frame_idx == 0
            else ("disabled" if not enable_fb_check else ("passed" if fb_ok else "failed"))
        ),
        "fb_ok": fb_ok,
        "search_radius_used_px": search_radius_used,
        "template_policy": template_policy,
        "peak_margin_min": peak_margin_min,
        "peak_ratio_min": peak_ratio_min,
        "texture_code_roi1": state.get("texture_code1"),
        "texture_code_roi2": state.get("texture_code2"),
        "texture_std_roi1": texture1.get("std_gray", np.nan),
        "texture_contrast_roi1": texture1.get("contrast_p95_p5", np.nan),
        "texture_low_frac_roi1": texture1.get("low_frac", np.nan),
        "texture_high_frac_roi1": texture1.get("high_frac", np.nan),
        "texture_structure_ratio_roi1": texture1.get("structure_tensor_ratio", np.nan),
        "texture_rank_ratio_roi1": texture1.get("rank_one_ratio", texture1.get("texture_rank_ratio", np.nan)),
        "texture_rank_score_roi1": texture1.get("rank_one_score", np.nan),
        "texture_gradient_energy_roi1": texture1.get("gradient_energy", np.nan),
        "texture_directional_coherence_roi1": texture1.get("directional_gradient_coherence", np.nan),
        "texture_periodicity_score_roi1": texture1.get("periodicity_score", np.nan),
        "texture_periodicity_period_px_roi1": texture1.get("periodicity_period_px", np.nan),
        "texture_metrics_version_roi1": texture1.get("metrics_version", TEXTURE_METRICS_VERSION),
        "texture_discriminator_version_roi1": texture1.get("discriminator_version", TEXTURE_DISCRIMINATOR_VERSION),
        "texture_structure_eigenvalue_minor_roi1": eigen_component(texture1, 0),
        "texture_structure_eigenvalue_major_roi1": eigen_component(texture1, 1),
        "texture_std_roi2": texture2.get("std_gray", np.nan),
        "texture_contrast_roi2": texture2.get("contrast_p95_p5", np.nan),
        "texture_low_frac_roi2": texture2.get("low_frac", np.nan),
        "texture_high_frac_roi2": texture2.get("high_frac", np.nan),
        "texture_structure_ratio_roi2": texture2.get("structure_tensor_ratio", np.nan),
        "texture_rank_ratio_roi2": texture2.get("rank_one_ratio", texture2.get("texture_rank_ratio", np.nan)),
        "texture_rank_score_roi2": texture2.get("rank_one_score", np.nan),
        "texture_gradient_energy_roi2": texture2.get("gradient_energy", np.nan),
        "texture_directional_coherence_roi2": texture2.get("directional_gradient_coherence", np.nan),
        "texture_periodicity_score_roi2": texture2.get("periodicity_score", np.nan),
        "texture_periodicity_period_px_roi2": texture2.get("periodicity_period_px", np.nan),
        "texture_metrics_version_roi2": texture2.get("metrics_version", TEXTURE_METRICS_VERSION),
        "texture_discriminator_version_roi2": texture2.get("discriminator_version", TEXTURE_DISCRIMINATOR_VERSION),
        "texture_structure_eigenvalue_minor_roi2": eigen_component(texture2, 0),
        "texture_structure_eigenvalue_major_roi2": eigen_component(texture2, 1),
        "texture_metrics_version": texture_thresholds.get("metrics_version", TEXTURE_METRICS_VERSION),
        "texture_discriminator_version": texture_thresholds.get("discriminator_version", TEXTURE_DISCRIMINATOR_VERSION),
        "texture_preflight_version": texture_thresholds.get("preflight_version", TEXTURE_PREFLIGHT_VERSION),
        "texture_min_std": texture_thresholds.get("min_texture_std", np.nan),
        "texture_min_contrast": texture_thresholds.get("min_texture_contrast", np.nan),
        "texture_max_saturated_frac": texture_thresholds.get("max_saturated_frac", np.nan),
        "texture_min_structure_ratio": texture_thresholds.get("min_structure_ratio", np.nan),
        "texture_max_directional_coherence": texture_thresholds.get("max_directional_coherence", np.nan),
        "texture_min_periodicity_score": texture_thresholds.get("min_periodicity_score", np.nan),
        "L0_px": L0,
        "used_roi1_x_px": used_rect1[0],
        "used_roi1_y_px": used_rect1[1],
        "used_roi1_w_px": used_rect1[2],
        "used_roi1_h_px": used_rect1[3],
        "used_roi1_center_x_px": c1x,
        "used_roi1_center_y_px": c1y,
        "used_roi2_x_px": used_rect2[0],
        "used_roi2_y_px": used_rect2[1],
        "used_roi2_w_px": used_rect2[2],
        "used_roi2_h_px": used_rect2[3],
        "used_roi2_center_x_px": c2x,
        "used_roi2_center_y_px": c2y,
        "length_px": L,
        "engineering_strain": strain,
        "true_strain": true_strain,
        "last_valid_engineering_strain": state["last_valid_strain"],
    }
    if candidate_rect1 is not None:
        row["candidate_roi1_center_x_px"], row["candidate_roi1_center_y_px"] = rect_center(candidate_rect1)
    else:
        row["candidate_roi1_center_x_px"] = row["candidate_roi1_center_y_px"] = np.nan
    if candidate_rect2 is not None:
        row["candidate_roi2_center_x_px"], row["candidate_roi2_center_y_px"] = rect_center(candidate_rect2)
    else:
        row["candidate_roi2_center_x_px"] = row["candidate_roi2_center_y_px"] = np.nan
    if pixel_size_mm not in (None, "") and np.isfinite(L):
        row["length_mm"] = L * float(pixel_size_mm)
        row["elongation_mm"] = (L - L0) * float(pixel_size_mm)
    overlay_info = {
        "used_rect1": used_rect1,
        "used_rect2": used_rect2,
        "candidate_rect1": candidate_rect1,
        "candidate_rect2": candidate_rect2,
        "strain": strain,
        "last_valid_strain": state["last_valid_strain"],
        "score1": score1,
        "score2": score2,
        "accepted": bool(accepted),
        "strain_valid": bool(strain_valid),
        "accept_mode": accept_mode,
        "reason": reason,
        "fb_err1": fb_err1,
        "fb_err2": fb_err2,
        "tracking_status_code": tracking_status_code,
    }
    return row, overlay_info


def draw_group_overlay(
    img8,
    group_name,
    actual_mode,
    used_rect1,
    used_rect2,
    candidate_rect1,
    candidate_rect2,
    frame_idx,
    strain,
    last_valid_strain,
    score1,
    score2,
    accepted,
    accept_mode,
    reason,
    fb_err1=None,
    fb_err2=None,
):
    rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    def draw_rect(rect, color, label, thickness=2):
        if rect is None:
            return
        x, y, w, h = rect
        x = int(round(x))
        y = int(round(y))
        w = int(round(w))
        h = int(round(h))

        cv2.rectangle(rgb, (x, y), (x + w, y + h), color, thickness)
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        cv2.circle(rgb, (cx, cy), 4, color, -1)
        cv2.putText(
            rgb,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # 候选框：橙/紫
    draw_rect(candidate_rect1, (0, 165, 255), "ROI1 candidate", 1)
    draw_rect(candidate_rect2, (255, 0, 255), "ROI2 candidate", 1)

    # 实际用于计算的框：红/蓝
    draw_rect(used_rect1, (0, 0, 255), "ROI1 used", 2)
    draw_rect(used_rect2, (255, 0, 0), "ROI2 used", 2)

    c1 = rect_center(used_rect1)
    c2 = rect_center(used_rect2)
    cv2.line(
        rgb,
        (int(round(c1[0])), int(round(c1[1]))),
        (int(round(c2[0])), int(round(c2[1]))),
        (0, 255, 255),
        2,
    )

    status = "ACCEPTED" if accepted else "REJECTED - ROI/TEMPLATE NOT UPDATED"
    if accepted and accept_mode:
        status += f" ({accept_mode})"

    strain_txt = f"Eng. strain: {strain:.6f}" if np.isfinite(strain) else "Eng. strain: NaN"
    last_txt = f"Last valid strain: {last_valid_strain:.6f}" if np.isfinite(last_valid_strain) else "Last valid strain: NaN"

    lines = [
        f"Group: {group_name} | mode: {actual_mode}",
        f"Frame: {frame_idx}",
        status,
        strain_txt,
        last_txt,
        f"Corr: ROI1={score1:.3f}, ROI2={score2:.3f}",
    ]

    if fb_err1 is not None and fb_err2 is not None:
        lines.append(f"FB error: ROI1={fb_err1:.2f}px, ROI2={fb_err2:.2f}px")

    lines.append(f"Reason: {reason}")

    y0 = 34
    for k, line in enumerate(lines):
        cv2.putText(
            rgb,
            line[:140],
            (20, y0 + 30 * k),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68 if k < 3 else 0.56,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return rgb

def generate_synthetic_speckle(height, width, *, seed=0, n_dots=None, sigma=1.8):
    """Smooth Gaussian-dot speckle used by tests and the GUI-independent DIC entry."""
    rng = np.random.default_rng(seed)
    height = int(height)
    width = int(width)
    if n_dots is None:
        n_dots = max(120, (height * width) // 28)
    img = np.full((height, width), 25.0, dtype=np.float32)
    ys = rng.uniform(2, max(3, height - 2), size=n_dots)
    xs = rng.uniform(2, max(3, width - 2), size=n_dots)
    amps = rng.uniform(90, 230, size=n_dots)
    rad = int(np.ceil(3.0 * float(sigma)))
    yy, xx = np.ogrid[-rad : rad + 1, -rad : rad + 1]
    blob = np.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) * float(sigma))).astype(np.float32)
    for x, y, amp in zip(xs, ys, amps):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = xi - rad, xi + rad + 1
        y0, y1 = yi - rad, yi + rad + 1
        gx0 = gy0 = 0
        gx1, gy1 = blob.shape[1], blob.shape[0]
        if x0 < 0:
            gx0 = -x0
            x0 = 0
        if y0 < 0:
            gy0 = -y0
            y0 = 0
        if x1 > width:
            gx1 -= x1 - width
            x1 = width
        if y1 > height:
            gy1 -= y1 - height
            y1 = height
        if x1 <= x0 or y1 <= y0:
            continue
        img[y0:y1, x0:x1] += amp * blob[gy0:gy1, gx0:gx1]
    return np.clip(img, 0, 255).astype(np.float32)


def _as_float_image(image):
    arr = _validate_finite_image(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.shape[2] >= 3 else arr[:, :, 0]
    arr = arr.astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        count = int((~np.isfinite(arr)).sum())
        raise CoreError("NONFINITE_IMAGE", {"count": count, "message": "float32 conversion produced non-finite samples"})
    return arr


def warp_image_translation(image, tx, ty):
    """Move material points by (tx, ty) px using bicubic sampling: def(X) = ref(X - t)."""
    img = _as_float_image(image)
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        img,
        xs - np.float32(tx),
        ys - np.float32(ty),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def warp_image_deformation_gradient(image, F, center=None):
    """Apply a uniform 2x2 deformation gradient about `center` (default image center)."""
    img = _as_float_image(image)
    F = np.asarray(F, dtype=np.float64).reshape(2, 2)
    Finv = np.linalg.inv(F)
    h, w = img.shape[:2]
    if center is None:
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    else:
        cx, cy = float(center[0]), float(center[1])
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx = xs - np.float32(cx)
    dy = ys - np.float32(cy)
    map_x = (np.float32(Finv[0, 0]) * dx + np.float32(Finv[0, 1]) * dy + np.float32(cx)).astype(np.float32)
    map_y = (np.float32(Finv[1, 0]) * dx + np.float32(Finv[1, 1]) * dy + np.float32(cy)).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def green_lagrange_from_F(F):
    """Oracle Green-Lagrange components from a 2x2 deformation gradient."""
    F = np.asarray(F, dtype=np.float64).reshape(2, 2)
    E = 0.5 * (F.T @ F - np.eye(2))
    return {"Exx": float(E[0, 0]), "Eyy": float(E[1, 1]), "Exy": float(E[0, 1])}


def _odd_subset_size(subset_size):
    size = int(subset_size)
    if size < 9:
        raise ValueError("subset_size must be >= 9.")
    if size % 2 == 0:
        size += 1
    return size


def build_poi_grid(roi, subset_size, step, image_shape):
    """POI centers inside `roi=(x,y,w,h)` that keep a full subset on the image."""
    x, y, w, h = [int(round(v)) for v in roi]
    subset_size = _odd_subset_size(subset_size)
    step = max(1, int(step))
    half = subset_size // 2
    H, W = image_shape[:2]
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, W)
    y1 = min(y + h, H)
    xs = np.arange(x0 + half, x1 - half, step, dtype=np.float64)
    ys = np.arange(y0 + half, y1 - half, step, dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def poi_grid_is_usable(X, Y, *, min_rows=3, min_cols=3):
    """Return whether a POI grid can support a 2-D affine strain fit."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
        return False
    rows, cols = X.shape
    return bool(rows >= int(min_rows) and cols >= int(min_cols) and X.size > 0 and Y.size > 0)


def build_specimen_mask(image, roi, *, texture_threshold=3.0):
    """Conservative texture-based proposal; users should inspect its boundary.

    Low-texture background and holes are excluded. No hole filling is applied:
    an excluded area must never become a measured surface by postprocessing.
    """
    gray = _as_float_image(image).astype(np.float64)
    mean = ndimage.uniform_filter(gray, size=9, mode="reflect")
    variance = np.maximum(0, ndimage.uniform_filter(gray*gray, size=9, mode="reflect")-mean*mean)
    mask = np.sqrt(variance) >= float(texture_threshold)
    roi_mask = np.zeros(gray.shape, dtype=bool)
    x, y, width, height = [int(round(value)) for value in roi]
    roi_mask[y:y+height, x:x+width] = True
    mask &= roi_mask
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    return mask


def _resolve_specimen_mask(reference, roi, settings):
    source = settings.get("mask") if isinstance(settings.get("mask"), Mapping) else {}
    mode = str(settings.get("mask_mode", source.get("mode", "none")))
    path = settings.get("mask_path", source.get("path"))
    threshold = float(settings.get("mask_texture_threshold", source.get("texture_threshold", 3.0)))
    exclusions = source.get("exclusions", [])
    if not np.isfinite(threshold) or threshold <= 0:
        raise CoreError("INVALID_MASK", {"message": "mask texture threshold must be finite and positive"})
    identity = None
    if mode == "none":
        if not exclusions:
            return None, {"mode": mode}, None
        mask = np.ones(reference.shape[:2], dtype=bool)
    elif mode == "auto":
        mask = build_specimen_mask(reference, roi, texture_threshold=threshold)
    elif mode == "file" and path:
        identity = file_identity(path)
        with Image.open(path) as raster:
            mask = np.asarray(raster.convert("L")) >= 128
        if file_identity(path) != identity:
            raise CoreError("INPUT_CHANGED", {"path": str(path)})
    else:
        raise CoreError("INVALID_MASK", {"message": "mask mode must be none, auto or file with a path"})
    if mask.shape != reference.shape[:2] or not mask.any():
        raise CoreError("INVALID_MASK", {"message": "mask must be nonempty and match the reference image dimensions"})
    for rect in exclusions:
        if not rect_is_inside_image(rect, reference.shape):
            raise CoreError("INVALID_MASK", {"message": "excluded rectangle must be inside the reference image"})
        x, y, width, height = [int(round(value)) for value in rect]
        mask[y:y+height,x:x+width] = False
    record = {"mode": mode, "texture_threshold": threshold, "included_pixels": int(mask.sum()),
              "sha256": hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
              "source_sha256": identity["sha256"] if identity else None,
              "reference_shape": list(mask.shape)}
    record["exclusions"] = [list(rect) for rect in exclusions]
    return mask, record, identity


def _displacement_outliers(X, Y, U, V, threshold_px=1.0):
    """Leave-one-out local quadratic prediction; never replace a measurement."""
    rejected = np.zeros(U.shape, dtype=bool)
    if not threshold_px:
        return rejected
    for i, j in zip(*np.where(np.isfinite(U)&np.isfinite(V))):
        i0, i1 = max(0, i-2), min(U.shape[0], i+3)
        j0, j1 = max(0, j-2), min(U.shape[1], j+3)
        valid = np.isfinite(U[i0:i1, j0:j1]) & np.isfinite(V[i0:i1, j0:j1])
        labels, _ = ndimage.label(valid)
        valid &= labels == labels[i-i0, j-j0]
        valid[i-i0, j-j0] = False
        if valid.sum() < 12:
            continue
        dx, dy = (X[i0:i1,j0:j1][valid]-X[i,j]), (Y[i0:i1,j0:j1][valid]-Y[i,j])
        scale = max(np.max(np.abs(dx)), np.max(np.abs(dy)), 1)
        dx, dy = dx/scale, dy/scale
        A = np.column_stack([np.ones(dx.size), dx, dy, dx*dx, dx*dy, dy*dy])
        if np.linalg.matrix_rank(A) < 6:
            continue
        data = np.column_stack([U[i0:i1,j0:j1][valid], V[i0:i1,j0:j1][valid]])
        fit = np.linalg.lstsq(A, data, rcond=None)[0]
        for _ in range(4):
            residual = np.linalg.norm(data-A@fit, axis=1)
            noise = max(.001, 1.4826*np.median(np.abs(residual-np.median(residual))))
            weights = np.minimum(1, 3*noise/np.maximum(residual, .000001))
            fit = np.linalg.lstsq(A*np.sqrt(weights[:,None]), data*np.sqrt(weights[:,None]), rcond=None)[0]
        prediction_error = np.linalg.norm(np.array([U[i,j],V[i,j]])-fit[0])
        rejected[i,j] = prediction_error > max(float(threshold_px), 6*noise, 3*np.median(residual))
    return rejected


def estimate_dic_initial_transform(reference, deformed, roi, specimen_mask=None):
    """SIFT + RANSAC affine initialization, never a displacement measurement."""
    height, width = reference.shape[:2]
    scale = min(1., 1600/max(height,width))
    shape = (max(1,round(width*scale)),max(1,round(height*scale)))
    f = cv2.resize(np.clip(reference,0,255).astype(np.uint8),shape)
    g = cv2.resize(np.clip(deformed,0,255).astype(np.uint8),shape)
    mask = np.zeros((height,width),dtype=np.uint8)
    x,y,w,h = [int(round(value)) for value in roi]
    mask[y:y+h,x:x+w] = 255
    if specimen_mask is not None:
        mask[~specimen_mask] = 0
    mask = cv2.resize(mask,shape,interpolation=cv2.INTER_NEAREST)
    detector = cv2.SIFT_create(nfeatures=2000)
    fk,fd = detector.detectAndCompute(f,mask)
    gk,gd = detector.detectAndCompute(g,None)
    if fd is None or gd is None or min(len(fk),len(gk)) < 12:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(fd,gd,k=2)
    matches = [pair[0] for pair in pairs if len(pair)==2 and pair[0].distance < .7*pair[1].distance]
    if len(matches) < 12:
        return None
    source = np.float32([fk[m.queryIdx].pt for m in matches])
    target = np.float32([gk[m.trainIdx].pt for m in matches])
    transform,inliers = cv2.estimateAffine2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=2000,confidence=.995,refineIters=10)
    if transform is None or inliers is None or np.count_nonzero(inliers) < 10 or inliers.mean() < .6:
        return None
    linear = transform[:,:2]
    if not np.isfinite(transform).all() or not .5 < np.linalg.det(linear) < 2 or np.linalg.cond(linear) > 2:
        return None
    error = np.linalg.norm(source@linear.T+transform[:,2]-target,axis=1)
    if np.median(error[inliers.ravel().astype(bool)]) > .8:
        return None
    transform[:,2] /= scale
    return {"matrix": transform, "matches": len(matches), "inliers": int(inliers.sum()),
            "median_reprojection_error_px": float(np.median(error[inliers.ravel().astype(bool)])/scale)}


def _affine_initial_parameters(transform, x, y):
    if transform is None:
        return np.zeros(6)
    matrix = transform["matrix"]
    u,v = matrix@np.array([x,y,1])-np.array([x,y])
    return np.array([u,matrix[0,0]-1,matrix[0,1],v,matrix[1,0],matrix[1,1]-1])


def _odd_window_size(window):
    size = max(3, int(window))
    if size % 2 == 0:
        size += 1
    return size


def rect_is_inside_image(rect, image_shape):
    """Return whether a positive rectangular ROI is fully inside an image."""
    if rect is None or image_shape is None:
        return False
    try:
        x, y, w, h = (int(round(value)) for value in rect)
        H, W = (int(value) for value in image_shape[:2])
    except (TypeError, ValueError, IndexError):
        return False
    return bool(x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= W and y + h <= H)


def validate_image_sequence_dimensions(paths):
    """Read a sequence and reject frames whose 2-D image dimensions differ."""
    paths = list(paths or [])
    if not paths:
        raise RuntimeError("全场 DIC 没有可读取的图像帧。")
    reference_shape = tuple(read_gray_image(paths[0]).shape[:2])
    for path in paths[1:]:
        shape = tuple(read_gray_image(path).shape[:2])
        if shape != reference_shape:
            raise RuntimeError(
                "全场 DIC 要求所有图像尺寸一致："
                f"参考帧 {reference_shape[1]}×{reference_shape[0]} px，"
                f"{Path(path).name} 为 {shape[1]}×{shape[0]} px。"
            )
    return reference_shape


def fullfield_field_has_finite_strain(field):
    """Check that a DIC frame contains at least one valid finite strain point."""
    valid = np.asarray(field.get("valid", []), dtype=bool).ravel()
    if valid.size == 0 or not valid.any():
        return False
    try:
        finite_strain = (
            np.isfinite(np.asarray(field["Exx"], dtype=float).ravel())
            & np.isfinite(np.asarray(field["Eyy"], dtype=float).ravel())
            & np.isfinite(np.asarray(field["Exy"], dtype=float).ravel())
        )
    except (KeyError, TypeError, ValueError):
        return False
    if finite_strain.size != valid.size:
        return False
    return bool(np.any(valid & finite_strain))


def _empty_integer_cc_diagnostic(reason, *, return_diagnostics):
    if return_diagnostics:
        return {
            "u": 0.0,
            "v": 0.0,
            "zncc": -1.0,
            "best_peak": -1.0,
            "second_peak": -1.0,
            "peak_margin": 0.0,
            "second_peak_margin": 0.0,
            "peak_ratio": 1.0,
            "best_to_second_peak_ratio": 1.0,
            "second_to_best_peak_ratio": 1.0,
            "peak_is_ambiguous": False,
            "search_level": 0,
            "stop_reason": str(reason),
            "reason": str(reason),
        }
    return 0.0, 0.0, -1.0


def _correlation_peak_diagnostics(corr, max_loc, *, exclusion_radius=None):
    safe_corr = np.asarray(corr, dtype=np.float32)
    px, py = (int(max_loc[0]), int(max_loc[1]))
    best_peak = float(safe_corr[py, px])
    if exclusion_radius is None:
        exclusion_radius = 2
    radius = max(1, int(exclusion_radius))
    candidates = safe_corr.copy()
    y0, y1 = max(0, py - radius), min(candidates.shape[0], py + radius + 1)
    x0, x1 = max(0, px - radius), min(candidates.shape[1], px + radius + 1)
    candidates[y0:y1, x0:x1] = -np.inf
    finite = np.isfinite(candidates)
    second_peak = float(np.max(candidates[finite])) if finite.any() else -1.0
    peak_margin = float(best_peak - second_peak)
    if second_peak > 1e-12:
        peak_ratio = float(best_peak / second_peak)
    elif best_peak > 0:
        peak_ratio = float("inf")
    else:
        peak_ratio = 1.0
    second_to_best = float(second_peak / best_peak) if abs(best_peak) > 1e-12 else float("inf")
    return {
        "best_peak": best_peak,
        "second_peak": second_peak,
        "peak_margin": peak_margin,
        "second_peak_margin": peak_margin,
        "peak_ratio": peak_ratio,
        "best_to_second_peak_ratio": peak_ratio,
        "second_to_best_peak_ratio": second_to_best,
        "second_peak_ratio": second_to_best,
        "second_peak_exclusion_radius": radius,
    }


def integer_cc_guess(
    reference,
    deformed,
    x,
    y,
    subset_size,
    search_radius,
    *,
    return_diagnostics=False,
    return_details=None,
    peak_exclusion_radius=None,
    initial_u=0.0,
    initial_v=0.0,
):
    """Integer template-match plus a sub-pixel peak and optional diagnostics.

    The historical call returns exactly ``(u, v, zncc)``.  New callers can
    request a dictionary with a second-peak measurement; the main solver uses
    that diagnostic path to reject repeated periodic peaks.
    """
    if return_details is not None:
        return_diagnostics = bool(return_details)
    reference = _as_float_image(reference)
    deformed = _as_float_image(deformed)
    if reference.shape != deformed.shape:
        raise ValueError(
            "reference and deformed image dimensions must match for full-field DIC."
        )
    subset_size = _odd_subset_size(subset_size)
    half = subset_size // 2
    H, W = reference.shape[:2]
    ix, iy = int(round(x)), int(round(y))
    if ix - half < 0 or iy - half < 0 or ix + half >= W or iy + half >= H:
        return _empty_integer_cc_diagnostic("OUT_OF_BOUNDS", return_diagnostics=return_diagnostics)
    tpl = reference[iy - half : iy + half + 1, ix - half : ix + half + 1]
    radius = max(1, int(search_radius))
    try:
        initial_u = float(initial_u)
        initial_v = float(initial_v)
    except (TypeError, ValueError) as exc:
        raise ValueError("initial displacement seed must be numeric") from exc
    sx1 = max(0, int(math.floor(ix - half + initial_u - radius)))
    sy1 = max(0, int(math.floor(iy - half + initial_v - radius)))
    sx2 = min(W, int(math.ceil(ix + half + 1 + initial_u + radius)))
    sy2 = min(H, int(math.ceil(iy + half + 1 + initial_v + radius)))
    search = deformed[sy1:sy2, sx1:sx2]
    if search.shape[0] < tpl.shape[0] or search.shape[1] < tpl.shape[1]:
        return _empty_integer_cc_diagnostic("SEARCH_TOO_SMALL", return_diagnostics=return_diagnostics)
    if not has_nonzero_variance(tpl) or not has_nonzero_variance(search):
        return _empty_integer_cc_diagnostic("ZERO_VARIANCE", return_diagnostics=return_diagnostics)
    corr = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
    finite_corr = np.isfinite(corr)
    if not finite_corr.any():
        return _empty_integer_cc_diagnostic("NONFINITE_CORRELATION", return_diagnostics=return_diagnostics)
    safe_corr = np.where(finite_corr, corr, -1.0).astype(np.float32, copy=False)
    _, max_val, _, max_loc = cv2.minMaxLoc(safe_corr)
    if not np.isfinite(max_val):
        return _empty_integer_cc_diagnostic("NONFINITE_PEAK", return_diagnostics=return_diagnostics)
    px, py = max_loc
    candidate_patch = search[py : py + tpl.shape[0], px : px + tpl.shape[1]]
    if candidate_patch.shape != tpl.shape or not has_nonzero_variance(candidate_patch):
        return _empty_integer_cc_diagnostic("ZERO_VARIANCE_CANDIDATE", return_diagnostics=return_diagnostics)
    dx, dy = subpixel_peak(safe_corr, px, py)
    u = (sx1 + px + dx) - (ix - half)
    v = (sy1 + py + dy) - (iy - half)
    peak_diagnostics = _correlation_peak_diagnostics(
        safe_corr,
        max_loc,
        exclusion_radius=peak_exclusion_radius,
    )
    peak_ambiguous = bool(
        np.isfinite(peak_diagnostics["peak_margin"])
        and np.isfinite(peak_diagnostics["peak_ratio"])
        and peak_diagnostics["peak_margin"] < DEFAULT_PEAK_MARGIN_MIN
        and peak_diagnostics["peak_ratio"] < DEFAULT_PEAK_RATIO_MIN
    )
    diagnostics = {
        "u": float(u),
        "v": float(v),
        "zncc": float(max_val),
        **peak_diagnostics,
        "peak_is_ambiguous": peak_ambiguous,
        "search_level": 0,
        "stop_reason": "peak_found",
        "reason": "peak_found",
    }
    if return_diagnostics:
        return diagnostics
    return float(u), float(v), float(max_val)


_CANONICAL_INTEGER_GUESS = integer_cc_guess


def _legacy_integer_guess(
    reference,
    deformed,
    x,
    y,
    subset_size,
    search_radius,
    *,
    initial_u=0.0,
    initial_v=0.0,
):
    """Call the canonical/legacy helper and normalize its return shape."""
    candidate = integer_cc_guess
    adapter = sys.modules.get("dic_virtual_extensometer_gui_v7_multi_roi_range")
    if adapter is not None:
        adapter_candidate = getattr(adapter, "integer_cc_guess", candidate)
        if callable(adapter_candidate) and adapter_candidate is not _CANONICAL_INTEGER_GUESS:
            candidate = adapter_candidate
    try:
        raw = candidate(
            reference,
            deformed,
            x,
            y,
            subset_size,
            search_radius,
            return_diagnostics=True,
            initial_u=initial_u,
            initial_v=initial_v,
        )
    except TypeError:
        # Existing integrations may expose only the original six-argument,
        # three-value helper.
        raw = candidate(reference, deformed, x, y, subset_size, search_radius)
    if isinstance(raw, Mapping):
        result = dict(raw)
        result.setdefault("u", 0.0)
        result.setdefault("v", 0.0)
        result.setdefault("zncc", result.get("score", -1.0))
        result.setdefault("best_peak", result.get("zncc", -1.0))
        result.setdefault("second_peak", np.nan)
        result.setdefault("peak_margin", np.nan)
        result.setdefault("second_peak_margin", result.get("peak_margin"))
        result.setdefault("peak_ratio", np.nan)
        result.setdefault("best_to_second_peak_ratio", result.get("peak_ratio"))
        result.setdefault("second_to_best_peak_ratio", np.nan)
        result.setdefault("second_peak_ratio", result.get("second_to_best_peak_ratio"))
        result.setdefault("peak_is_ambiguous", False)
        result.setdefault("search_level", 0)
        return result
    try:
        values = tuple(raw)
    except TypeError:
        values = ()
    if len(values) >= 3:
        u, v, score = values[:3]
        result = {
            "u": float(u),
            "v": float(v),
            "zncc": float(score),
            "best_peak": float(score),
            "second_peak": np.nan,
            "peak_margin": np.nan,
            "second_peak_margin": np.nan,
            "peak_ratio": np.nan,
            "best_to_second_peak_ratio": np.nan,
            "second_to_best_peak_ratio": np.nan,
            "second_peak_ratio": np.nan,
            "peak_is_ambiguous": False,
            "search_level": 0,
            "stop_reason": "legacy_tuple",
        }
        if len(values) >= 4 and isinstance(values[3], Mapping):
            result.update(values[3])
        return result
    return _empty_integer_cc_diagnostic("INVALID_RETURN", return_diagnostics=True)


def _compose_warp_inverse(p, dp):
    Mc = np.array(
        [[1.0 + p[1], p[2], p[0]], [p[4], 1.0 + p[5], p[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    Md = np.array(
        [[1.0 + dp[1], dp[2], dp[0]], [dp[4], 1.0 + dp[5], dp[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    try:
        Mn = Mc @ np.linalg.inv(Md)
    except np.linalg.LinAlgError:
        return None
    return np.array(
        [Mn[0, 2], Mn[0, 0] - 1.0, Mn[0, 1], Mn[1, 2], Mn[1, 0], Mn[1, 1] - 1.0],
        dtype=np.float64,
    )




def _prepare_dic_images(reference, deformed):
    """Cache interpolation coefficients and fourth-order reference gradients."""
    f = np.asarray(reference, dtype=np.float64)
    g = np.asarray(deformed, dtype=np.float64)
    fy, fx = np.gradient(f)
    fx[:, 2:-2] = (f[:, :-4] - 8*f[:, 1:-3] + 8*f[:, 3:-1] - f[:, 4:]) / 12
    fy[2:-2, :] = (f[:-4, :] - 8*f[1:-3, :] + 8*f[3:-1, :] - f[4:, :]) / 12
    return {"reference": f, "coefficients": ndimage.spline_filter(g, order=5, mode="mirror"),
            "fx": fx, "fy": fy, "shape": g.shape}


def _refine_subset_ic(reference, deformed, x, y, subset_size, p0=None,
                      method="GN", max_iter=25, tol=1e-3, _prepared=None):
    """Affine IC correlation with continuous quintic B-splines and ZNSSD.

    The normalized Jacobian projects out brightness and contrast. Both GN and
    LM evaluate a proposal before accepting it; rejected proposals never
    replace the current state. Coefficients are reusable across all subsets.
    """
    prepared = _prepared or _prepare_dic_images(_as_float_image(reference), _as_float_image(deformed))
    f_image, height, width = prepared["reference"], *prepared["shape"]
    half = _odd_subset_size(subset_size) // 2
    ix, iy = int(round(x)), int(round(y))
    if ix-half < 0 or iy-half < 0 or ix+half >= width or iy+half >= height:
        return None
    ys, xs = slice(iy-half, iy+half+1), slice(ix-half, ix+half+1)
    f = f_image[ys, xs].ravel()
    f = f - f.mean()
    f_norm = np.linalg.norm(f)
    if not np.isfinite(f_norm) or f_norm < 1e-8:
        return None
    fn = f / f_norm
    yy, xx = np.mgrid[-half:half+1, -half:half+1].astype(float)
    xf, yf = xx.ravel(), yy.ravel()
    fx, fy = prepared["fx"][ys, xs].ravel(), prepared["fy"][ys, xs].ravel()
    sd = np.column_stack([fx, fx*xf, fx*yf, fy, fy*xf, fy*yf]) / f_norm
    sd -= sd.mean(axis=0)
    sd -= fn[:, None] * (fn @ sd)[None, :]
    hess = sd.T @ sd
    condition = float(np.linalg.cond(hess))
    if not np.isfinite(condition) or np.linalg.matrix_rank(hess) < 6:
        return None
    p = np.zeros(6) if p0 is None else np.asarray(p0, dtype=float).reshape(6).copy()

    def evaluate(parameters):
        map_x = float(x) + parameters[0] + (1+parameters[1])*xx + parameters[2]*yy
        map_y = float(y) + parameters[3] + parameters[4]*xx + (1+parameters[5])*yy
        if map_x.min() < 2 or map_y.min() < 2 or map_x.max() > width-3 or map_y.max() > height-3:
            return None
        g = ndimage.map_coordinates(prepared["coefficients"], [map_y.ravel(), map_x.ravel()],
                                   order=5, mode="mirror", prefilter=False)
        g -= g.mean()
        norm = np.linalg.norm(g)
        if not np.isfinite(norm) or norm < 1e-8:
            return None
        residual = fn - g / norm
        return residual, float(residual @ residual)

    state = evaluate(p)
    if state is None:
        return None
    mu, converged, reason, increment = 1e-3, False, "max_iterations", np.nan
    iteration = 0
    use_forward = False
    for iteration in range(1, int(max_iter)+1):
        residual, cost = state
        if use_forward:
            # At a noisy optimum the fixed reference Jacobian can point uphill.
            # Recompute the normalized target Jacobian only for stalled subsets.
            mx = float(x)+p[0]+(1+p[1])*xf+p[2]*yf
            my = float(y)+p[3]+p[4]*xf+(1+p[5])*yf
            def sample(dx, dy):
                return ndimage.map_coordinates(prepared["coefficients"], [my+dy, mx+dx],
                                              order=5, mode="mirror", prefilter=False)
            g = sample(0, 0)
            g -= g.mean()
            g_norm = np.linalg.norm(g)
            normalized = g/g_norm
            gx = (sample(.01, 0)-sample(-.01, 0))/.02
            gy = (sample(0, .01)-sample(0, -.01))/.02
            sd = np.column_stack([gx, gx*xf, gx*yf, gy, gy*xf, gy*yf])/g_norm
            sd -= sd.mean(axis=0)
            sd -= normalized[:, None]*(normalized@sd)[None, :]
            hess = sd.T@sd
            current_condition = float(np.linalg.cond(hess))
            condition = max(condition, current_condition)
            if not np.isfinite(current_condition) or np.linalg.matrix_rank(hess) < 6:
                reason = "ill_conditioned_forward_hessian"
                break
        rhs = sd.T @ residual
        try:
            undamped = -np.linalg.solve(hess, rhs)
        except np.linalg.LinAlgError:
            reason = "hessian_solve_failure"
            break
        undamped_norm = float(np.linalg.norm(undamped*np.array([1,half,half,1,half,half])))
        if undamped_norm <= float(tol):
            reason = "converged_forward_affine_increment" if use_forward else "converged_all_affine_increment"
            converged, increment = True, undamped_norm
            break
        accepted = False
        for trial in range(12):
            try:
                dp = -np.linalg.solve(hess + (mu*np.diag(np.diag(hess)) if method == "LM" else 0), rhs)
            except np.linalg.LinAlgError:
                reason = "hessian_solve_failure"
                break
            if method != "LM":
                dp *= .5**trial
            increment = float(np.linalg.norm(dp * np.array([1, half, half, 1, half, half])))
            if not np.isfinite(increment):
                reason = "nonfinite_increment"
                break
            # A rejected step shrunk by damping/line search is not convergence
            # evidence: only the original proposed step qualifies.
            candidate = p-dp if use_forward else _compose_warp_inverse(p, dp)
            next_state = evaluate(candidate) if candidate is not None else None
            if next_state is not None and next_state[1] <= cost + 1e-14:
                p, state, accepted = candidate, next_state, True
                mu = max(mu / 3, 1e-10)
                break
            if method == "LM":
                mu *= 10
        if converged:
            break
        if not use_forward and (not accepted or trial >= 3):
            use_forward = True
            mu = 1e-3
            continue
        if not accepted:
            reason = "no_descent_step"
            break
    residual, cost = state
    return {"u": float(p[0]), "v": float(p[3]), "p": p, "zncc": float(1-cost/2),
            "iters": iteration, "iterations": iteration,
            "residual_rms": float(np.sqrt(cost/residual.size)),
            "hessian_condition_number": condition, "converged": converged,
            "stop_reason": reason, "increment_norm_px": increment}


def refine_subset_icgn(reference, deformed, x, y, subset_size, p0=None, max_iter=25, tol=1e-3, _prepared=None):
    return _refine_subset_ic(
        reference, deformed, x, y, subset_size, p0=p0, method="GN", max_iter=max_iter, tol=tol, _prepared=_prepared
    )


def refine_subset_iclm(reference, deformed, x, y, subset_size, p0=None, max_iter=25, tol=1e-3, _prepared=None):
    return _refine_subset_ic(
        reference, deformed, x, y, subset_size, p0=p0, method="LM", max_iter=max_iter, tol=tol, _prepared=_prepared
    )


def _nan_gaussian(arr, sigma):
    if sigma is None or float(sigma) <= 0:
        return arr
    values = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan)
    radius = max(1, int(np.ceil(3*float(sigma))))
    ny, nx = values.shape
    # Restrict each kernel to the center's connected neighborhood. A normalized
    # convolution over all finite values would mix opposite sides of a crack.
    for i, j in zip(*np.nonzero(finite)):
        i0, i1 = max(0, i-radius), min(ny, i+radius+1)
        j0, j1 = max(0, j-radius), min(nx, j+radius+1)
        labels, _ = ndimage.label(finite[i0:i1, j0:j1])
        support = labels == labels[i-i0, j-j0]
        yy, xx = np.mgrid[i0:i1, j0:j1]
        weights = np.exp(-((xx-j)**2+(yy-i)**2)/(2*float(sigma)**2))*support
        out[i, j] = np.sum(weights*np.nan_to_num(values[i0:i1,j0:j1]))/weights.sum()
    return out


def compute_strain_fields(
    X,
    Y,
    U,
    V,
    *,
    window=5,
    smooth_sigma=0.0,
    max_condition_number=1e12,
    degree=2,
    robust=True,
):
    """Local polynomial virtual strain gauge, without filling missing data.

    Coordinates are scaled before fitting. A quadratic fit avoids one-sided
    linear-fit bias near specimen boundaries. Tukey IRLS limits the influence
    of isolated mismatches; local connectivity prevents fits across gaps.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64).copy()
    V = np.asarray(V, dtype=np.float64).copy()
    if X.shape != Y.shape or U.shape != X.shape or V.shape != X.shape:
        raise ValueError("X, Y, U, and V must have identical grid shapes.")
    if degree not in (1, 2):
        raise ValueError("strain polynomial degree must be 1 or 2")
    if smooth_sigma and float(smooth_sigma) > 0:
        U = _nan_gaussian(U, smooth_sigma)
        V = _nan_gaussian(V, smooth_sigma)

    ny, nx = X.shape
    win = _odd_window_size(window)
    half = win // 2
    Exx = np.full((ny, nx), np.nan)
    Eyy = np.full((ny, nx), np.nan)
    Exy = np.full((ny, nx), np.nan)
    exx = np.full((ny, nx), np.nan)
    eyy = np.full((ny, nx), np.nan)
    exy = np.full((ny, nx), np.nan)
    dudx = np.full((ny, nx), np.nan)
    dudy = np.full((ny, nx), np.nan)
    dvdx = np.full((ny, nx), np.nan)
    dvdy = np.full((ny, nx), np.nan)
    strain_valid = np.zeros((ny, nx), dtype=bool)
    fit_condition_number = np.full((ny, nx), np.nan)
    fit_residual_rms = np.full((ny, nx), np.nan)
    fit_point_count = np.zeros((ny, nx), dtype=np.int32)
    strain_invalid_reason = np.full((ny, nx), "INVALID_CENTER", dtype=object)
    fit_effective_count = np.zeros((ny, nx), dtype=float)
    fit_degree = np.zeros((ny, nx), dtype=np.int32)
    gradient_standard_error = np.full((ny, nx, 4), np.nan)

    for i in range(ny):
        i0, i1 = max(0, i - half), min(ny, i + half + 1)
        for j in range(nx):
            j0, j1 = max(0, j - half), min(nx, j + half + 1)
            xs = X[i0:i1, j0:j1].ravel()
            ys = Y[i0:i1, j0:j1].ravel()
            us = U[i0:i1, j0:j1].ravel()
            vs = V[i0:i1, j0:j1].ravel()
            if not (np.isfinite(U[i, j]) and np.isfinite(V[i, j])):
                continue
            ok = np.isfinite(us) & np.isfinite(vs) & np.isfinite(xs) & np.isfinite(ys)
            connected, _ = ndimage.label(ok.reshape(i1-i0, j1-j0))
            center_label = connected[i-i0, j-j0]
            ok &= connected.ravel() == center_label
            count = int(ok.sum())
            fit_point_count[i, j] = count
            if count < 6:
                strain_invalid_reason[i, j] = "INSUFFICIENT_NEIGHBORS"
                continue
            scale = max(np.max(np.abs(xs[ok]-X[i, j])), np.max(np.abs(ys[ok]-Y[i, j])), 1.0)
            dx, dy = (xs[ok]-X[i, j])/scale, (ys[ok]-Y[i, j])/scale
            A = np.column_stack([np.ones(count), dx, dy])
            effective_degree = 1
            if degree == 2:
                quadratic = np.column_stack([A, dx*dx, dx*dy, dy*dy])
                if count >= 9 and np.linalg.matrix_rank(quadratic) == 6:
                    data_for_order = np.column_stack([us[ok], vs[ok]])
                    linear_fit = np.linalg.lstsq(A, data_for_order, rcond=None)[0]
                    quadratic_fit = np.linalg.lstsq(quadratic, data_for_order, rcond=None)[0]
                    linear_rss = float(np.sum((data_for_order-A@linear_fit)**2))
                    quadratic_rss = float(np.sum((data_for_order-quadratic@quadratic_fit)**2))
                    # Avoid paying quadratic boundary variance for a flat or
                    # affine noisy field. Record the order used at every POI.
                    curvature_f = max(0, linear_rss-quadratic_rss)/3 / max(quadratic_rss/(count-6), 1e-18)
                    if curvature_f > 6:
                        A, effective_degree = quadratic, 2
            rank = int(np.linalg.matrix_rank(A))
            if rank < A.shape[1]:
                strain_invalid_reason[i, j] = "RANK_DEFICIENT"
                continue
            try:
                condition = float(np.linalg.cond(A))
            except np.linalg.LinAlgError:
                strain_invalid_reason[i, j] = "FIT_CONDITION_FAILURE"
                continue
            fit_condition_number[i, j] = condition
            if not np.isfinite(condition) or condition > float(max_condition_number):
                strain_invalid_reason[i, j] = "ILL_CONDITIONED_FIT"
                continue
            try:
                data = np.column_stack([us[ok], vs[ok]])
                weights = np.ones(count)
                coefficients, *_ = np.linalg.lstsq(A, data, rcond=None)
                if robust:
                    for _ in range(5):
                        residual = data-A@coefficients
                        noise = max(1e-6, float(np.max(1.4826*np.median(np.abs(residual-np.median(residual, axis=0)),axis=0))))
                        radius = np.linalg.norm(residual, axis=1)/(8.0*noise)
                        proposed = np.square(np.clip(1-radius**2, 0, 1))
                        if np.sum(proposed > .05) < A.shape[1]+3:
                            break
                        root_weights = np.sqrt(proposed)
                        weighted = A*root_weights[:, None]
                        if np.linalg.matrix_rank(weighted) < A.shape[1]:
                            break
                        updated, *_ = np.linalg.lstsq(weighted, data*root_weights[:, None], rcond=None)
                        weights, coefficients = proposed, updated
                au, av = coefficients[:, 0], coefficients[:, 1]
            except np.linalg.LinAlgError:
                strain_invalid_reason[i, j] = "FIT_FAILURE"
                continue
            if not (np.isfinite(au).all() and np.isfinite(av).all()):
                strain_invalid_reason[i, j] = "NONFINITE_FIT"
                continue
            weighted_condition = float(np.linalg.cond(A*np.sqrt(weights[:,None])))
            fit_condition_number[i,j] = weighted_condition
            if not np.isfinite(weighted_condition) or weighted_condition > float(max_condition_number):
                strain_invalid_reason[i,j] = "ILL_CONDITIONED_FIT"
                continue
            residual_u = A @ au - us[ok]
            residual_v = A @ av - vs[ok]
            residual_rms = float(np.sqrt(np.mean(np.concatenate([residual_u, residual_v]) ** 2)))
            fit_residual_rms[i, j] = residual_rms
            du_dx, du_dy = float(au[1]/scale), float(au[2]/scale)
            dv_dx, dv_dy = float(av[1]/scale), float(av[2]/scale)
            fit_effective_count[i, j] = float(weights.sum())
            fit_degree[i, j] = effective_degree
            # Conditional fit standard errors are diagnostics, not calibrated
            # experimental uncertainties (overlapping subsets are correlated).
            dof = max(float(weights.sum())-A.shape[1], 1)
            covariance = np.linalg.pinv(A.T@(weights[:, None]*A))
            variance = np.sum(weights[:, None]*(data-A@coefficients)**2, axis=0)/dof
            gradient_standard_error[i, j] = np.sqrt(np.maximum(0, [covariance[1,1]*variance[0], covariance[2,2]*variance[0], covariance[1,1]*variance[1], covariance[2,2]*variance[1]]))/scale
            Fxx, Fxy, Fyx, Fyy = 1.0 + du_dx, du_dy, dv_dx, 1.0 + dv_dy
            values = (
                0.5 * (Fxx * Fxx + Fyx * Fyx - 1.0),
                0.5 * (Fxy * Fxy + Fyy * Fyy - 1.0),
                0.5 * (Fxx * Fxy + Fyx * Fyy),
                du_dx,
                dv_dy,
                0.5 * (du_dy + dv_dx),
            )
            if not np.isfinite(values).all():
                strain_invalid_reason[i, j] = "NONFINITE_STRAIN"
                continue
            dudx[i, j], dudy[i, j] = du_dx, du_dy
            dvdx[i, j], dvdy[i, j] = dv_dx, dv_dy
            Exx[i, j], Eyy[i, j], Exy[i, j], exx[i, j], eyy[i, j], exy[i, j] = values
            strain_valid[i, j] = True
            strain_invalid_reason[i, j] = ""

    return {
        "Exx": Exx,
        "Eyy": Eyy,
        "Exy": Exy,
        "exx": exx,
        "eyy": eyy,
        "exy": exy,
        "dudx": dudx,
        "dudy": dudy,
        "dvdx": dvdx,
        "dvdy": dvdy,
        "U": U,
        "V": V,
        "window": win,
        "strain_valid": strain_valid,
        "fit_condition_number": fit_condition_number,
        "strain_fit_condition_number": fit_condition_number,
        "fit_residual_rms": fit_residual_rms,
        "strain_fit_residual_rms": fit_residual_rms,
        "fit_point_count": fit_point_count,
        "fit_effective_count": fit_effective_count,
        "fit_degree": fit_degree,
        "gradient_standard_error": gradient_standard_error,
        "degree": int(degree),
        "robust": bool(robust),
        "strain_invalid_reason": strain_invalid_reason,
    }


_CANONICAL_REFINERS = {
    DIC_SOLVER_ICGN: refine_subset_icgn,
    DIC_SOLVER_ICLM: refine_subset_iclm,
}


def _legacy_refiner_for(solver_name):
    """Resolve a legacy adapter monkeypatch without importing the GUI."""
    default = _CANONICAL_REFINERS[solver_name]
    # A direct patch of the core remains authoritative.  This matters for
    # headless tests and for downstream callers that instrument the solver.
    current_core = refine_subset_icgn if solver_name == DIC_SOLVER_ICGN else refine_subset_iclm
    if current_core is not default:
        default = current_core
    adapter = sys.modules.get("dic_virtual_extensometer_gui_v7_multi_roi_range")
    if adapter is None:
        return default
    name = "refine_subset_icgn" if solver_name == DIC_SOLVER_ICGN else "refine_subset_iclm"
    candidate = getattr(adapter, name, default)
    if candidate is not _CANONICAL_REFINERS[solver_name] and callable(candidate):
        return candidate
    return default


DEFAULT_PEAK_MARGIN_MIN = 0.02
DEFAULT_PEAK_RATIO_MIN = 1.02
DEFAULT_MAX_HESSIAN_CONDITION_NUMBER = 1e12
DEFAULT_MIN_CORRELATION_VALID_FRACTION = 0.95
DEFAULT_MIN_STRAIN_VALID_FRACTION = 0.80


def _field_array(field, name, *, dtype=None, size=None, default=None):
    if name not in field:
        if size is None:
            return np.asarray([], dtype=dtype)
        return np.full(size, default, dtype=dtype)
    arr = np.asarray(field[name], dtype=dtype).ravel()
    if size is not None and arr.size != size:
        return np.full(size, default, dtype=dtype)
    return arr


def field_quality_summary(
    field,
    *,
    min_correlation_valid_fraction=DEFAULT_MIN_CORRELATION_VALID_FRACTION,
    min_strain_valid_fraction=DEFAULT_MIN_STRAIN_VALID_FRACTION,
    max_residual_rms=float("inf"),
):
    """Summarize correlation and strain gates without conflating them."""
    try:
        residual_threshold = float(max_residual_rms)
    except (TypeError, ValueError):
        residual_threshold = float("inf")
    valid = np.asarray(field.get("valid", []), dtype=bool).ravel()
    size = int(valid.size)
    eligible = _field_array(field, "eligible", dtype=bool, size=size, default=True)
    eligible_count = int(eligible.sum())
    strain_valid = _field_array(field, "strain_valid", dtype=bool, size=size, default=False)
    if eligible_count == 0:
        correlation_fraction = strain_fraction = 0.0
    else:
        correlation_fraction = float((valid & eligible).sum()/eligible_count)
        strain_fraction = float((strain_valid & eligible).sum()/eligible_count)
    reasons = _field_array(field, "invalid_reason", dtype=object, size=size, default="")
    peak_ambiguous = _field_array(field, "peak_is_ambiguous", dtype=bool, size=size, default=False)
    reason_ambiguous = np.asarray(
        ["AMBIGUOUS" in str(reason).upper() for reason in reasons],
        dtype=bool,
    )
    ambiguous_mask = peak_ambiguous | reason_ambiguous
    ambiguous_count = int(ambiguous_mask.sum())
    ambiguous_rejected_count = int((ambiguous_mask & ~valid).sum())
    ambiguous_accepted_count = int((ambiguous_mask & valid).sum())
    converged = _field_array(field, "converged", dtype=bool, size=size, default=False)
    convergence_known = _field_array(field, "convergence_known", dtype=bool, size=size, default=False)
    converged_count = int((converged & (valid | convergence_known)).sum())
    convergence_known_count = int(convergence_known.sum())
    convergence_fraction = (
        float(converged_count / convergence_known_count)
        if convergence_known_count
        else None
    )
    invalid_reason_histogram = {}
    for reason in reasons:
        key = str(reason)
        if key:
            invalid_reason_histogram[key] = invalid_reason_histogram.get(key, 0) + 1
    false_accept_mask = valid & ambiguous_mask
    false_accept_count = int(false_accept_mask.sum())
    false_reject_proxy_mask = eligible & (~valid) & (~ambiguous_mask) & np.asarray(
        [str(reason) not in {"", "INVALID_CORRELATION", "AMBIGUOUS_PEAK"} for reason in reasons],
        dtype=bool,
    )
    false_reject_proxy_count = int(false_reject_proxy_mask.sum())
    reasons_out = []
    if correlation_fraction < float(min_correlation_valid_fraction):
        reasons_out.append("correlation_valid_fraction_below_threshold")
    if strain_fraction < float(min_strain_valid_fraction):
        reasons_out.append("strain_valid_fraction_below_threshold")
    if ambiguous_accepted_count:
        reasons_out.append("ambiguous_peaks_accepted")
    return {
        "point_count": size,
        "eligible_point_count": eligible_count,
        "masked_point_count": size-eligible_count,
        "spatial_outlier_count": int(np.count_nonzero(field.get("spatial_outlier", []))),
        "correlation_valid_count": int(valid.sum()),
        "strain_valid_count": int(strain_valid.sum()),
        "correlation_valid_fraction": correlation_fraction,
        "strain_valid_fraction": strain_fraction,
        "converged_count": converged_count,
        "convergence_known_count": convergence_known_count,
        "converged_fraction": convergence_fraction,
        "invalid_reason_histogram": invalid_reason_histogram,
        "ambiguous_rejected_count": ambiguous_rejected_count,
        "ambiguous_accepted_count": ambiguous_accepted_count,
        "unsafe_accept_count": false_accept_count,
        "false_accept_count": false_accept_count,
        # Kept solely as a compatibility proxy; no ground truth is available
        # to call ordinary quality rejections true false-rejects.
        "false_reject_count": false_reject_proxy_count,
        "false_reject_proxy_count": false_reject_proxy_count,
        "false_reject_is_proxy": True,
        "ambiguous_count": ambiguous_count,
        "residual_threshold_enabled": bool(np.isfinite(residual_threshold)),
        "max_residual_rms": None if not np.isfinite(residual_threshold) else residual_threshold,
        "scientific_ok": bool(not reasons_out),
        "scientific_reasons": reasons_out,
        "thresholds": {
            "min_correlation_valid_fraction": float(min_correlation_valid_fraction),
            "min_strain_valid_fraction": float(min_strain_valid_fraction),
        },
    }


def _pyramid_subset_size(subset_size, scale, image_shape):
    # Keep enough texture support at the coarse levels.  A naively scaled
    # 21-pixel subset becomes 5 pixels at 1/4 scale and is too ambiguous for
    # the affine Hessian; an 11-pixel floor still fits edge-safe coarse ROIs
    # without synthesizing image data.
    requested = max(11, int(round(float(subset_size) * math.sqrt(float(scale)))))
    if requested % 2 == 0:
        requested += 1
    min_dimension = min(int(image_shape[0]), int(image_shape[1]))
    if requested > min_dimension:
        raise ValueError(
            f"pyramid level image {tuple(int(value) for value in image_shape[:2])} "
            f"cannot contain subset_size={requested}."
        )
    return requested


def _build_image_pyramid(image, levels, scale):
    """Build one fine-to-coarse float image pyramid for a frame pair."""
    fine = _as_float_image(image)
    pyramid = [fine]
    for level_index in range(1, int(levels)):
        previous = pyramid[-1]
        height = max(1, int(round(previous.shape[0] * float(scale))))
        width = max(1, int(round(previous.shape[1] * float(scale))))
        if height < 2 or width < 2:
            raise ValueError(
                f"pyramid level {level_index} is too small ({width}x{height}) for DIC."
            )
        pyramid.append(
            cv2.resize(previous, (width, height), interpolation=cv2.INTER_AREA).astype(
                np.float32,
                copy=False,
            )
        )
    return list(reversed(pyramid))


def _coerce_refine_result(result):
    if result is None:
        return None
    if isinstance(result, Mapping):
        output = dict(result)
    else:
        try:
            values = tuple(result)
        except TypeError:
            return None
        if len(values) < 3:
            return None
        output = {
            "u": values[0],
            "v": values[1],
            "zncc": values[2],
            "p": values[3] if len(values) >= 4 else [values[0], 0.0, 0.0, values[1], 0.0, 0.0],
        }
    try:
        output["u"] = float(output.get("u"))
        output["v"] = float(output.get("v"))
        output["zncc"] = float(output.get("zncc"))
        output["p"] = np.asarray(output.get("p"), dtype=float).reshape(6)
    except (TypeError, ValueError):
        return None
    output["finite"] = bool(
        np.isfinite(output["u"])
        and np.isfinite(output["v"])
        and np.isfinite(output["zncc"])
        and np.isfinite(output["p"]).all()
    )
    return output


def _serializable_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _displacement_support_is_inside(x, y, u, v, subset_size, image_shape):
    half = int(subset_size) // 2
    height, width = (int(value) for value in image_shape[:2])
    return bool(
        np.isfinite([x, y, u, v]).all()
        and x + u - half >= 0
        and y + v - half >= 0
        and x + u + half < width
        and y + v + half < height
    )


def _run_2d_dic_multiscale(
    reference,
    deformed,
    roi,
    *,
    subset_size,
    step,
    solver,
    search_radius,
    max_iter,
    conv_tol,
    zncc_min,
    strain_window,
    smooth_sigma,
    progress_callback,
    peak_margin_min,
    peak_ratio_min,
    reject_ambiguous_peaks,
    max_condition_number,
    reject_nonconverged,
    max_residual_rms,
    min_correlation_valid_fraction,
    min_strain_valid_fraction,
    pyramid_levels,
    pyramid_scale,
    specimen_mask=None,
    strain_degree=2,
    robust_strain=True,
    outlier_threshold_px=1.0,
    initialization="auto",
    _global_initial_transform=None,
):
    """Recover large translations coarse-to-fine, then run the canonical fine solver."""
    reference = _as_float_image(reference)
    deformed = _as_float_image(deformed)
    if reference.shape != deformed.shape:
        raise ValueError("reference and deformed image dimensions must match for full-field DIC.")
    if search_radius is None:
        search_radius = max(8, _odd_subset_size(subset_size) // 2)
    search_radius = int(search_radius)
    if search_radius < 1:
        raise ValueError("search_radius must be positive.")
    solver_name = str(solver).strip().upper().replace("_", "-")
    if solver_name not in DIC_SOLVERS:
        raise ValueError(f"solver must be {DIC_SOLVER_ICGN} or {DIC_SOLVER_ICLM}.")
    fine_subset = _odd_subset_size(subset_size)
    X, Y = build_poi_grid(roi, fine_subset, step, reference.shape)
    if X.size == 0:
        raise ValueError("pyramid DIC ROI has no usable points at the finest level.")
    reference_pyramid = _build_image_pyramid(reference, pyramid_levels, pyramid_scale)
    deformed_pyramid = _build_image_pyramid(deformed, pyramid_levels, pyramid_scale)
    if len(reference_pyramid) != len(deformed_pyramid):
        raise ValueError("reference/deformed pyramid level counts differ.")
    scales_x = [float(level.shape[1] / reference.shape[1]) for level in reference_pyramid]
    scales_y = [float(level.shape[0] / reference.shape[0]) for level in reference_pyramid]
    level_scales = [float((sx + sy) / 2.0) for sx, sy in zip(scales_x, scales_y)]
    level_subset_sizes = [
        _pyramid_subset_size(fine_subset, scale, level.shape)
        for scale, level in zip(level_scales, reference_pyramid)
    ]
    refine = _legacy_refiner_for(solver_name)
    initial_guesses = {}
    prepared_levels = [_prepare_dic_images(f, g) for f, g in zip(reference_pyramid, deformed_pyramid)] if refine is _CANONICAL_REFINERS[solver_name] else []
    pyramid_diagnostics = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_fine, y_fine = float(X[i, j]), float(Y[i, j])
            seed_p = _affine_initial_parameters(_global_initial_transform,x_fine,y_fine) if _global_initial_transform is not None else None
            seed_u = float(seed_p[0]*scales_x[0]) if seed_p is not None else 0.
            seed_v = float(seed_p[3]*scales_y[0]) if seed_p is not None else 0.
            if seed_p is not None:
                seed_p[0],seed_p[3] = seed_u,seed_v
            point_levels = []
            for level_index, (reference_level, deformed_level) in enumerate(
                zip(reference_pyramid, deformed_pyramid)
            ):
                sx, sy = scales_x[level_index], scales_y[level_index]
                x_level, y_level = x_fine * sx, y_fine * sy
                subset_level = level_subset_sizes[level_index]
                guess = _legacy_integer_guess(
                    reference_level,
                    deformed_level,
                    x_level,
                    y_level,
                    subset_level,
                    search_radius,
                    initial_u=seed_u,
                    initial_v=seed_v,
                )
                guess_u = float(guess.get("u", 0.0))
                guess_v = float(guess.get("v", 0.0))
                p0 = [guess_u, 0.0, 0.0, guess_v, 0.0, 0.0]
                if seed_p is not None and np.isfinite(seed_p).all():
                    p0 = seed_p.copy()
                    p0[0], p0[3] = guess_u, guess_v
                boundary_ok = _displacement_support_is_inside(
                    x_level,
                    y_level,
                    guess_u,
                    guess_v,
                    subset_level,
                    deformed_level.shape,
                )
                if boundary_ok:
                    try:
                        refined = _coerce_refine_result(
                            refine(
                                reference_level,
                                deformed_level,
                                x_level,
                                y_level,
                                subset_level,
                                p0=p0,
                                max_iter=max_iter,
                                tol=conv_tol,
                                **({"_prepared": prepared_levels[level_index]} if prepared_levels else {}),
                            )
                        )
                    except Exception:
                        refined = None
                else:
                    refined = None
                refined_finite = bool(refined is not None and refined.get("finite"))
                condition = refined.get("hessian_condition_number") if refined else None
                residual = refined.get("residual_rms") if refined else None
                converged = refined.get("converged") if refined else None
                layer_accepted = bool(
                    refined_finite
                    and float(refined["zncc"]) >= float(zncc_min)
                    and (
                        condition is None
                        or (np.isfinite(float(condition)) and float(condition) <= max_condition_number)
                    )
                )
                layer_record = {
                    "level_index": int(level_index),
                    "scale_x": float(sx),
                    "scale_y": float(sy),
                    "scale": float(level_scales[level_index]),
                    "pad": 0,
                    "x": float(x_level),
                    "y": float(y_level),
                    "subset_size": int(subset_level),
                    "seed_u": _serializable_float(seed_u),
                    "seed_v": _serializable_float(seed_v),
                    "guess_u": _serializable_float(guess_u),
                    "guess_v": _serializable_float(guess_v),
                    "guess_zncc": _serializable_float(guess.get("zncc")),
                    "best_peak": _serializable_float(guess.get("best_peak")),
                    "second_peak": _serializable_float(guess.get("second_peak")),
                    "peak_margin": _serializable_float(guess.get("peak_margin")),
                    "best_to_second_peak_ratio": _serializable_float(guess.get("best_to_second_peak_ratio")),
                    "second_to_best_peak_ratio": _serializable_float(guess.get("second_to_best_peak_ratio")),
                    "refine_u": _serializable_float(refined.get("u") if refined else None),
                    "refine_v": _serializable_float(refined.get("v") if refined else None),
                    "refine_zncc": _serializable_float(refined.get("zncc") if refined else None),
                    "residual_rms": _serializable_float(residual),
                    "hessian_condition_number": _serializable_float(condition),
                    "iterations": int(refined.get("iterations", refined.get("iters", 0))) if refined else 0,
                    "converged": None if converged is None else bool(converged),
                    "stop_reason": str(refined.get("stop_reason", "")) if refined else (
                        "boundary_support_failure" if not boundary_ok else "refine_failed"
                    ),
                    "boundary_support_ok": bool(boundary_ok),
                    "accepted": layer_accepted,
                }
                point_levels.append(layer_record)
                if not boundary_ok:
                    seed_p = None
                    break
                if refined_finite:
                    seed_u, seed_v = float(refined["u"]), float(refined["v"])
                    seed_p = np.asarray(refined["p"], dtype=float).copy()
                elif np.isfinite([guess_u, guess_v]).all():
                    seed_u, seed_v = guess_u, guess_v
                    seed_p = np.asarray(p0, dtype=float)
                if level_index + 1 < len(reference_pyramid):
                    seed_u *= scales_x[level_index + 1] / sx
                    seed_v *= scales_y[level_index + 1] / sy
                    if seed_p is not None:
                        seed_p[0] *= scales_x[level_index + 1] / sx
                        seed_p[3] *= scales_y[level_index + 1] / sy
            if seed_p is not None and np.isfinite(seed_p).all():
                initial_guesses[(i, j)] = [float(value) for value in seed_p]
            else:
                initial_guesses[(i, j)] = (float(seed_u), float(seed_v))
            pyramid_diagnostics.append(point_levels)
    field = run_2d_dic(
        reference,
        deformed,
        roi,
        subset_size=subset_size,
        step=step,
        solver=solver,
        search_radius=search_radius,
        max_iter=max_iter,
        conv_tol=conv_tol,
        zncc_min=zncc_min,
        strain_window=strain_window,
        smooth_sigma=smooth_sigma,
        progress_callback=progress_callback,
        peak_margin_min=peak_margin_min,
        peak_ratio_min=peak_ratio_min,
        reject_ambiguous_peaks=reject_ambiguous_peaks,
        max_condition_number=max_condition_number,
        reject_nonconverged=reject_nonconverged,
        max_residual_rms=max_residual_rms,
        min_correlation_valid_fraction=min_correlation_valid_fraction,
        min_strain_valid_fraction=min_strain_valid_fraction,
        pyramid_levels=1,
        pyramid_scale=pyramid_scale,
        _initial_guesses=initial_guesses,
        _pyramid_diagnostics=pyramid_diagnostics,
        _coordinate_offset=(0.0, 0.0),
        _output_roi=roi,
        specimen_mask=specimen_mask, strain_degree=strain_degree,
        robust_strain=robust_strain, outlier_threshold_px=outlier_threshold_px,
        initialization=initialization, _global_initial_transform=_global_initial_transform,
    )
    field["pyramid_levels_requested"] = int(pyramid_levels)
    field["pyramid_levels_used"] = int(pyramid_levels)
    field["pyramid_scale"] = float(pyramid_scale)
    field["pyramid_degradation_reason"] = None
    field["pyramid_level_diagnostics"] = pyramid_diagnostics
    field["quality_summary"] = field_quality_summary(
        field,
        min_correlation_valid_fraction=min_correlation_valid_fraction,
        min_strain_valid_fraction=min_strain_valid_fraction,
        max_residual_rms=max_residual_rms,
    )
    return field


def run_2d_dic(
    reference,
    deformed,
    roi,
    *,
    subset_size=21,
    step=5,
    solver=DIC_SOLVER_ICGN,
    search_radius=None,
    max_iter=25,
    conv_tol=1e-3,
    zncc_min=0.75,
    strain_window=5,
    smooth_sigma=0.0,
    progress_callback=None,
    peak_margin_min=DEFAULT_PEAK_MARGIN_MIN,
    peak_ratio_min=DEFAULT_PEAK_RATIO_MIN,
    reject_ambiguous_peaks=True,
    max_condition_number=DEFAULT_MAX_HESSIAN_CONDITION_NUMBER,
    reject_nonconverged=False,
    max_residual_rms=float("inf"),
    min_correlation_valid_fraction=DEFAULT_MIN_CORRELATION_VALID_FRACTION,
    min_strain_valid_fraction=DEFAULT_MIN_STRAIN_VALID_FRACTION,
    pyramid_levels=1,
    pyramid_scale=0.5,
    specimen_mask=None,
    strain_degree=2,
    robust_strain=True,
    outlier_threshold_px=1.0,
    initialization="auto",
    _global_initial_transform=None,
    _initial_guesses=None,
    _pyramid_diagnostics=None,
    _coordinate_offset=None,
    _output_roi=None,
):
    """
    Correlate a rectangular ROI with IC-GN or IC-LM.

    Failed or out-of-ROI points stay NaN (not interpolated). Strain is a windowed
    local fit of the displacement field, optionally Gaussian-smoothed first.
    """
    requested_subset_size = int(subset_size)
    requested_strain_window = int(strain_window)
    try:
        raw_pyramid_levels = float(pyramid_levels)
        pyramid_scale = float(pyramid_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("pyramid_levels must be an integer and pyramid_scale must be numeric.") from exc
    if not np.isfinite(raw_pyramid_levels) or raw_pyramid_levels != int(raw_pyramid_levels):
        raise ValueError("pyramid_levels must be an integer.")
    pyramid_levels = int(raw_pyramid_levels)
    if pyramid_levels < 1:
        raise ValueError("pyramid_levels must be >= 1.")
    if not (0.0 < pyramid_scale < 1.0):
        raise ValueError("pyramid_scale must satisfy 0 < pyramid_scale < 1.")
    reference = _as_float_image(reference)
    deformed = _as_float_image(deformed)
    if reference.shape != deformed.shape:
        raise ValueError(
            "reference and deformed image dimensions must match for full-field DIC."
        )
    if not rect_is_inside_image(roi, reference.shape):
        raise ValueError("roi must be a positive rectangle fully inside the reference image.")
    if strain_degree not in (1, 2) or not np.isfinite(outlier_threshold_px) or outlier_threshold_px < 0:
        raise ValueError("strain degree must be 1 or 2 and outlier threshold finite and nonnegative")
    if specimen_mask is not None:
        specimen_mask = np.asarray(specimen_mask, dtype=bool)
        if specimen_mask.shape != reference.shape:
            raise ValueError("specimen mask must match the reference image shape")
    if initialization not in ("auto","local"):
        raise ValueError("initialization must be auto or local")
    global_transform = _global_initial_transform
    if initialization == "auto" and global_transform is None and _initial_guesses is None:
        proposal = estimate_dic_initial_transform(reference,deformed,roi,specimen_mask)
        if proposal is not None:
            x,y,w,h = roi
            initial = _affine_initial_parameters(proposal,x+w/2,y+h/2)
            if np.linalg.norm(initial[[0,3]]) > max(2.,.6*float(search_radius or 10)) or np.linalg.norm(initial[[1,2,4,5]]) > .02:
                global_transform = proposal
    subset_size = _odd_subset_size(subset_size)
    step = max(1, int(step))
    solver_name = str(solver).strip().upper().replace("_", "-")
    if solver_name not in (DIC_SOLVER_ICGN, DIC_SOLVER_ICLM):
        raise ValueError(f"solver must be {DIC_SOLVER_ICGN} or {DIC_SOLVER_ICLM}.")
    try:
        peak_margin_min = float(peak_margin_min)
        peak_ratio_min = float(peak_ratio_min)
        max_condition_number = float(max_condition_number)
        max_residual_rms = float(max_residual_rms)
    except (TypeError, ValueError) as exc:
        raise ValueError("DIC quality thresholds must be numeric.") from exc
    if not np.isfinite([peak_margin_min, peak_ratio_min, max_condition_number]).all():
        raise ValueError("DIC peak/Hessian quality thresholds must be finite.")
    if np.isnan(max_residual_rms) or max_residual_rms <= 0:
        raise ValueError("max_residual_rms must be positive or infinity.")
    if pyramid_levels > 1 and _initial_guesses is None:
        return _run_2d_dic_multiscale(
            reference,
            deformed,
            roi,
            subset_size=subset_size,
            step=step,
            solver=solver,
            search_radius=search_radius,
            max_iter=max_iter,
            conv_tol=conv_tol,
            zncc_min=zncc_min,
            strain_window=strain_window,
            smooth_sigma=smooth_sigma,
            progress_callback=progress_callback,
            peak_margin_min=peak_margin_min,
            peak_ratio_min=peak_ratio_min,
            reject_ambiguous_peaks=reject_ambiguous_peaks,
            max_condition_number=max_condition_number,
            reject_nonconverged=reject_nonconverged,
            max_residual_rms=max_residual_rms,
            min_correlation_valid_fraction=min_correlation_valid_fraction,
            min_strain_valid_fraction=min_strain_valid_fraction,
            pyramid_levels=pyramid_levels,
            pyramid_scale=pyramid_scale,
            specimen_mask=specimen_mask, strain_degree=strain_degree,
            robust_strain=robust_strain, outlier_threshold_px=outlier_threshold_px,
            initialization=initialization, _global_initial_transform=global_transform,
        )
    refine = _legacy_refiner_for(solver_name)
    prepared_images = _prepare_dic_images(reference, deformed) if refine is _CANONICAL_REFINERS[solver_name] else None
    if search_radius is None:
        search_radius = max(8, subset_size // 2)
    effective_search_radius = int(search_radius)
    if global_transform is not None:
        effective_search_radius = min(effective_search_radius, max(3,int(math.ceil(3*global_transform["median_reprojection_error_px"]))))

    grid_X, grid_Y = build_poi_grid(roi, subset_size, step, reference.shape)
    if _coordinate_offset is None:
        coordinate_offset = (0.0, 0.0)
    else:
        try:
            coordinate_offset = (float(_coordinate_offset[0]), float(_coordinate_offset[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("_coordinate_offset must contain two numeric values.") from exc
    X = grid_X - coordinate_offset[0]
    Y = grid_Y - coordinate_offset[1]
    ny, nx = X.shape
    eligible = np.ones((ny, nx), dtype=bool)
    if specimen_mask is not None:
        supported = ndimage.minimum_filter(specimen_mask.astype(np.uint8), size=subset_size, mode="constant", cval=0).astype(bool)
        eligible = supported[np.rint(grid_Y).astype(int), np.rint(grid_X).astype(int)]
    U = np.full((ny, nx), np.nan, dtype=np.float64)
    V = np.full((ny, nx), np.nan, dtype=np.float64)
    Z = np.full((ny, nx), np.nan, dtype=np.float64)
    valid = np.zeros((ny, nx), dtype=bool)
    P = np.full((ny, nx, 6), np.nan, dtype=np.float64)
    peak_best = np.full((ny, nx), np.nan, dtype=np.float64)
    peak_second = np.full((ny, nx), np.nan, dtype=np.float64)
    peak_margin = np.full((ny, nx), np.nan, dtype=np.float64)
    peak_ratio = np.full((ny, nx), np.nan, dtype=np.float64)
    peak_is_ambiguous = np.zeros((ny, nx), dtype=bool)
    residual_rms = np.full((ny, nx), np.nan, dtype=np.float64)
    hessian_condition_number = np.full((ny, nx), np.nan, dtype=np.float64)
    iterations = np.zeros((ny, nx), dtype=np.int32)
    converged = np.zeros((ny, nx), dtype=bool)
    convergence_known = np.zeros((ny, nx), dtype=bool)
    increment_norm_px = np.full((ny, nx), np.nan, dtype=np.float64)
    invalid_reason = np.full((ny, nx), "", dtype=object)
    stop_reason = np.full((ny, nx), "", dtype=object)
    total = max(ny * nx, 1)

    for i in range(ny):
        for j in range(nx):
            if not eligible[i, j]:
                invalid_reason[i, j] = "MASK_EXCLUDED"
                if progress_callback is not None:
                    progress_callback(i*nx+j+1, total)
                continue
            px = float(grid_X[i, j])
            py = float(grid_Y[i, j])
            seed = _affine_initial_parameters(global_transform,px,py) if global_transform is not None else (0.0,0.0)
            if _initial_guesses is not None:
                seed = _initial_guesses.get((i, j), seed)
            try:
                seed_values = np.asarray(seed, dtype=float).ravel()
                if seed_values.size >= 6 and np.isfinite(seed_values[:6]).all():
                    seed_p = seed_values[:6].copy()
                    seed_u, seed_v = float(seed_p[0]), float(seed_p[3])
                else:
                    seed_p = None
                    seed_u, seed_v = float(seed_values[0]), float(seed_values[1])
            except (TypeError, ValueError, IndexError):
                seed_p = None
                seed_u = seed_v = 0.0
            guess = _legacy_integer_guess(
                reference,
                deformed,
                px,
                py,
                subset_size,
                effective_search_radius,
                initial_u=seed_u,
                initial_v=seed_v,
            )
            u0 = float(guess.get("u", 0.0))
            v0 = float(guess.get("v", 0.0))
            cc = float(guess.get("zncc", -1.0))
            if np.isfinite([u0, v0, cc]).all() and not _displacement_support_is_inside(
                px,
                py,
                u0,
                v0,
                subset_size,
                deformed.shape,
            ):
                invalid_reason[i, j] = "BOUNDARY_SUPPORT"
                stop_reason[i, j] = "boundary_support_failure"
                if progress_callback is not None:
                    progress_callback(i * nx + j + 1, total)
                continue
            for target, key in (
                (peak_best, "best_peak"),
                (peak_second, "second_peak"),
                (peak_margin, "peak_margin"),
                (peak_ratio, "peak_ratio"),
            ):
                value = guess.get(key)
                if value is not None:
                    try:
                        target[i, j] = float(value)
                    except (TypeError, ValueError):
                        pass
            margin = peak_margin[i, j]
            ratio = peak_ratio[i, j]
            has_peak_diagnostics = np.isfinite(margin) and np.isfinite(ratio)
            ambiguous = bool(
                has_peak_diagnostics
                and (margin < peak_margin_min and ratio < peak_ratio_min)
            )
            peak_is_ambiguous[i, j] = ambiguous
            if ambiguous and reject_ambiguous_peaks:
                invalid_reason[i, j] = "AMBIGUOUS_PEAK"
                stop_reason[i, j] = "ambiguous_peak"
                if progress_callback is not None:
                    progress_callback(i * nx + j + 1, total)
                continue
            p0 = [u0, 0.0, 0.0, v0, 0.0, 0.0]
            if seed_p is not None:
                p0 = seed_p.copy()
                p0[0], p0[3] = u0, v0
            try:
                result = refine(
                    reference,
                    deformed,
                    px,
                    py,
                    subset_size,
                    p0=p0,
                    max_iter=max_iter,
                    tol=conv_tol,
                    **({"_prepared": prepared_images} if prepared_images is not None else {}),
                )
            except Exception:
                result = None
                invalid_reason[i, j] = "REFINE_EXCEPTION"
            if result is None:
                invalid_reason[i, j] = invalid_reason[i, j] or "REFINE_FAILED"
                stop_reason[i, j] = stop_reason[i, j] or "refine_failed"
                if progress_callback is not None:
                    progress_callback(i * nx + j + 1, total)
                continue
            try:
                result = dict(result)
            except (TypeError, ValueError):
                try:
                    values = tuple(result)
                except TypeError:
                    values = ()
                if len(values) >= 3:
                    result = {
                        "u": values[0],
                        "v": values[1],
                        "zncc": values[2],
                        "p": values[3] if len(values) >= 4 else [values[0], 0.0, 0.0, values[1], 0.0, 0.0],
                    }
                else:
                    result = {}
            try:
                result_p = np.asarray(result.get("p"), dtype=float).reshape(6)
                result_u = float(result.get("u"))
                result_v = float(result.get("v"))
                result_zncc = float(result.get("zncc"))
                result_finite = (
                    np.isfinite(result_u)
                    and np.isfinite(result_v)
                    and np.isfinite(result_zncc)
                    and np.isfinite(result_p).all()
                )
            except (TypeError, ValueError):
                result_p = None
                result_u = result_v = result_zncc = np.nan
                result_finite = False
            if "residual_rms" in result:
                try:
                    residual_rms[i, j] = float(result["residual_rms"])
                except (TypeError, ValueError):
                    pass
            condition_value = result.get("hessian_condition_number", result.get("condition_number"))
            if condition_value is not None:
                try:
                    hessian_condition_number[i, j] = float(condition_value)
                except (TypeError, ValueError):
                    pass
            iteration_value = result.get("iterations", result.get("iters"))
            if iteration_value is not None:
                try:
                    iterations[i, j] = max(0, int(iteration_value))
                except (TypeError, ValueError):
                    pass
            if "converged" in result and result.get("converged") is not None:
                convergence_known[i, j] = True
                converged[i, j] = bool(result.get("converged"))
            if result.get("increment_norm_px") is not None:
                try:
                    increment_norm_px[i, j] = float(result["increment_norm_px"])
                except (TypeError, ValueError):
                    pass
            stop_reason[i, j] = str(result.get("stop_reason", ""))
            if not result_finite:
                invalid_reason[i, j] = invalid_reason[i, j] or "NONFINITE_RESULT"
            elif result_zncc < float(zncc_min):
                invalid_reason[i, j] = invalid_reason[i, j] or "ZNCC_BELOW_THRESHOLD"
            elif np.isfinite(hessian_condition_number[i, j]) and hessian_condition_number[i, j] > max_condition_number:
                invalid_reason[i, j] = invalid_reason[i, j] or "ILL_CONDITIONED_HESSIAN"
            elif np.isfinite(max_residual_rms) and (
                not np.isfinite(residual_rms[i, j]) or residual_rms[i, j] > max_residual_rms
            ):
                invalid_reason[i, j] = invalid_reason[i, j] or "RESIDUAL_ABOVE_THRESHOLD"
            elif reject_nonconverged and convergence_known[i, j] and not converged[i, j]:
                invalid_reason[i, j] = invalid_reason[i, j] or "NOT_CONVERGED"
            if not invalid_reason[i, j]:
                U[i, j] = result_u
                V[i, j] = result_v
                Z[i, j] = result_zncc
                valid[i, j] = True
                P[i, j, :] = result_p
            if progress_callback is not None:
                progress_callback(i * nx + j + 1, total)

    raw_U, raw_V = U.copy(), V.copy()
    spatial_outlier = _displacement_outliers(X, Y, U, V, outlier_threshold_px)
    U[spatial_outlier] = V[spatial_outlier] = np.nan
    valid[spatial_outlier] = False
    invalid_reason[spatial_outlier] = "SPATIAL_OUTLIER"
    strains = compute_strain_fields(
        X,
        Y,
        U,
        V,
        window=strain_window,
        smooth_sigma=smooth_sigma,
        max_condition_number=max_condition_number,
        degree=strain_degree, robust=robust_strain,
    )
    strain_valid = np.asarray(strains["strain_valid"], dtype=bool) & valid
    field = {
        "x": X.ravel(),
        "y": Y.ravel(),
        # ``u``/``v`` retain the historical optional-smoothed semantics;
        # raw solver outputs are exposed separately for auditability.
        "u": strains["U"].ravel(),
        "v": strains["V"].ravel(),
        "u_raw": raw_U.ravel(),
        "v_raw": raw_V.ravel(),
        "eligible": eligible.ravel(),
        "spatial_outlier": spatial_outlier.ravel(),
        "specimen_mask": specimen_mask,
        "zncc": Z.ravel(),
        "valid": valid.ravel(),
        "strain_valid": strain_valid.ravel(),
        "Exx": strains["Exx"].ravel(),
        "Eyy": strains["Eyy"].ravel(),
        "Exy": strains["Exy"].ravel(),
        "exx": strains["exx"].ravel(),
        "eyy": strains["eyy"].ravel(),
        "exy": strains["exy"].ravel(),
        "X": X,
        "Y": Y,
        "U": strains["U"],
        "V": strains["V"],
        "P": P,
        "peak_best": peak_best.ravel(),
        "best_peak": peak_best.ravel(),
        "peak_second": peak_second.ravel(),
        "second_peak": peak_second.ravel(),
        "peak_margin": peak_margin.ravel(),
        "second_peak_margin": peak_margin.ravel(),
        "peak_ratio": peak_ratio.ravel(),
        "best_to_second_peak_ratio": peak_ratio.ravel(),
        "second_to_best_peak_ratio": np.divide(
            peak_second.ravel(),
            peak_best.ravel(),
            out=np.full(peak_best.size, np.nan, dtype=float),
            where=np.abs(peak_best.ravel()) > 1e-12,
        ),
        "second_peak_ratio": np.divide(
            peak_second.ravel(),
            peak_best.ravel(),
            out=np.full(peak_best.size, np.nan, dtype=float),
            where=np.abs(peak_best.ravel()) > 1e-12,
        ),
        "peak_is_ambiguous": peak_is_ambiguous.ravel(),
        "residual_rms": residual_rms.ravel(),
        "hessian_condition_number": hessian_condition_number.ravel(),
        "condition_number": hessian_condition_number.ravel(),
        "iterations": iterations.ravel(),
        "converged": converged.ravel(),
        "convergence_known": convergence_known.ravel(),
        "increment_norm_px": increment_norm_px.ravel(),
        "stop_reason": stop_reason.ravel(),
        "invalid_reason": invalid_reason.ravel(),
        "strain_invalid_reason": np.asarray(strains["strain_invalid_reason"], dtype=object).ravel(),
        "strain_fit_condition_number": np.asarray(strains["fit_condition_number"], dtype=float).ravel(),
        "strain_fit_residual_rms": np.asarray(strains["fit_residual_rms"], dtype=float).ravel(),
        "fit_point_count": np.asarray(strains["fit_point_count"], dtype=int).ravel(),
        "strain_fit_effective_count": strains["fit_effective_count"].ravel(),
        "strain_fit_degree": strains["fit_degree"].ravel(),
        "dudx_standard_error": strains["gradient_standard_error"][:,:,0].ravel(),
        "dudy_standard_error": strains["gradient_standard_error"][:,:,1].ravel(),
        "dvdx_standard_error": strains["gradient_standard_error"][:,:,2].ravel(),
        "dvdy_standard_error": strains["gradient_standard_error"][:,:,3].ravel(),
        "subset_size": subset_size,
        "step": step,
        "solver": solver_name,
        "roi": tuple(int(round(v)) for v in (_output_roi if _output_roi is not None else roi)),
        "zncc_min": float(zncc_min),
        "smooth_sigma": float(smooth_sigma or 0.0),
        "interpolation": "quintic_bspline_v1",
        "strain_degree": int(strain_degree),
        "robust_strain": bool(robust_strain),
        "outlier_threshold_px": float(outlier_threshold_px),
        "strain_gauge_span_px": (int(strains["window"])-1)*step,
        "initialization": initialization,
        "effective_search_radius_px": effective_search_radius,
        "initial_transform": {key: (value.tolist() if isinstance(value,np.ndarray) else value) for key,value in global_transform.items()} if global_transform is not None else None,
        "requested_subset_size": requested_subset_size,
        "requested_strain_window": requested_strain_window,
        "strain_window": int(strains["window"]),
        "effective_subset_size": subset_size,
        "effective_strain_window": int(strains["window"]),
        "peak_margin_min": peak_margin_min,
        "peak_ratio_min": peak_ratio_min,
        "reject_ambiguous_peaks": bool(reject_ambiguous_peaks),
        "max_condition_number": max_condition_number,
        "reject_nonconverged": bool(reject_nonconverged),
        "max_residual_rms": max_residual_rms,
        "min_correlation_valid_fraction": float(min_correlation_valid_fraction),
        "min_strain_valid_fraction": float(min_strain_valid_fraction),
        "pyramid_levels_requested": int(pyramid_levels),
        "pyramid_levels_used": int(pyramid_levels),
        "pyramid_scale": float(pyramid_scale),
        "pyramid_degradation_reason": None,
        "pyramid_level_diagnostics": _pyramid_diagnostics or [],
        "Exx_grid": strains["Exx"],
        "Eyy_grid": strains["Eyy"],
        "Exy_grid": strains["Exy"],
        "exx_grid": strains["exx"],
        "eyy_grid": strains["eyy"],
        "exy_grid": strains["exy"],
        "strain_valid_grid": strain_valid,
        "strain_type": "green_lagrange",
        "strain_convention": "tensor Exy; dimensionless",
    }
    if _pyramid_diagnostics is None:
        field["pyramid_level_diagnostics"] = [
            {
                "level_index": 0,
                "scale": 1.0,
                "subset_size": int(subset_size),
                "accepted": bool(valid.ravel()[index]),
                "best_peak": None if not np.isfinite(peak_best.ravel()[index]) else float(peak_best.ravel()[index]),
                "second_peak": None if not np.isfinite(peak_second.ravel()[index]) else float(peak_second.ravel()[index]),
                "peak_margin": None if not np.isfinite(peak_margin.ravel()[index]) else float(peak_margin.ravel()[index]),
                "best_to_second_peak_ratio": None if not np.isfinite(peak_ratio.ravel()[index]) else float(peak_ratio.ravel()[index]),
            }
            for index in range(valid.size)
        ]
    field["quality_summary"] = field_quality_summary(
        field,
        min_correlation_valid_fraction=min_correlation_valid_fraction,
        min_strain_valid_fraction=min_strain_valid_fraction,
        max_residual_rms=max_residual_rms,
    )
    return field


def run_2d_dic_sequence(reference, deformed_frames, roi, **kwargs):
    """Correlate each deformed frame to the same reference (sequence batch)."""
    return [run_2d_dic(reference, frame, roi, **kwargs) for frame in deformed_frames]

def dic_field_to_dataframe(field, *, include_quality=True):
    """Tabular field export with legacy columns first and audit columns last."""
    table = {
        "x": np.asarray(field["x"], dtype=float),
        "y": np.asarray(field["y"], dtype=float),
        "u": np.asarray(field["u"], dtype=float),
        "v": np.asarray(field["v"], dtype=float),
        "zncc": np.asarray(field["zncc"], dtype=float),
        "valid": np.asarray(field["valid"], dtype=bool),
        "Exx": np.asarray(field["Exx"], dtype=float).ravel(),
        "Eyy": np.asarray(field["Eyy"], dtype=float).ravel(),
        "Exy": np.asarray(field["Exy"], dtype=float).ravel(),
        "exx": np.asarray(field["exx"], dtype=float).ravel(),
        "eyy": np.asarray(field["eyy"], dtype=float).ravel(),
        "exy": np.asarray(field["exy"], dtype=float).ravel(),
    }
    if include_quality:
        quality_columns = (
            ("u_raw", float),
            ("v_raw", float),
            ("best_peak", float),
            ("second_peak", float),
            ("peak_margin", float),
            ("second_peak_margin", float),
            ("peak_ratio", float),
            ("best_to_second_peak_ratio", float),
            ("second_to_best_peak_ratio", float),
            ("second_peak_ratio", float),
            ("peak_is_ambiguous", bool),
            ("residual_rms", float),
            ("hessian_condition_number", float),
            ("iterations", int),
            ("converged", bool),
            ("convergence_known", bool),
            ("increment_norm_px", float),
            ("stop_reason", object),
            ("invalid_reason", object),
            ("strain_valid", bool),
            ("strain_fit_condition_number", float),
            ("strain_fit_residual_rms", float),
            ("fit_point_count", int),
            ("strain_invalid_reason", object),
            ("eligible", bool),
            ("spatial_outlier", bool),
            ("strain_fit_degree", int),
            ("strain_fit_effective_count", float),
            ("dudx_standard_error", float),
            ("dudy_standard_error", float),
            ("dvdx_standard_error", float),
            ("dvdy_standard_error", float),
        )
        for name, dtype in quality_columns:
            if name in field:
                values = np.asarray(field[name], dtype=dtype).ravel()
                if values.size == len(table["x"]):
                    table[name] = values
    return pd.DataFrame(table)


def write_dic_field_txt(field, path):
    table = dic_field_to_dataframe(field)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(table.columns)
    lines = ["\t".join(columns)]
    for row in table.itertuples(index=False):
        cells = []
        for col, value in zip(columns, row):
            if col in {"valid", "strain_valid", "peak_is_ambiguous", "converged", "convergence_known", "eligible", "spatial_outlier"}:
                cells.append("1" if bool(value) else "0")
            elif col in {"stop_reason", "invalid_reason", "strain_invalid_reason"}:
                cells.append("" if value is None else str(value))
            elif pd.isna(value) or not np.isfinite(float(value)):
                cells.append("NaN")
            else:
                cells.append(f"{float(value):.8f}")
        lines.append("\t".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_dic_field_parameters(field, path):
    """Write human-readable provenance and solver parameters for one field."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = dict(field.get("provenance") or {})
    defaults = {
        "analysis_mode": ANALYSIS_MODE_FULLFIELD,
        "roi": field.get("roi"),
        "subset_size_px": field.get("subset_size"),
        "effective_subset_size_px": field.get("effective_subset_size", field.get("subset_size")),
        "requested_subset_size": field.get("requested_subset_size", field.get("subset_size")),
        "step_px": field.get("step"),
        "solver": field.get("solver"),
        "zncc_min": field.get("zncc_min"),
        "strain_window": field.get("strain_window"),
        "effective_strain_window": field.get("effective_strain_window", field.get("strain_window")),
        "requested_strain_window": field.get("requested_strain_window", field.get("strain_window")),
        "smooth_sigma": field.get("smooth_sigma"),
        "interpolation": field.get("interpolation"),
        "strain_degree": field.get("strain_degree"),
        "robust_strain": field.get("robust_strain"),
        "strain_gauge_span_px": field.get("strain_gauge_span_px"),
        "outlier_threshold_px": field.get("outlier_threshold_px"),
        "initialization": field.get("initialization"),
        "initial_transform": field.get("initial_transform"),
        "effective_search_radius_px": field.get("effective_search_radius_px"),
        "mask": field.get("mask_provenance"),
        "display": field.get("display_options", {}),
        "peak_margin_min": field.get("peak_margin_min"),
        "peak_ratio_min": field.get("peak_ratio_min"),
        "max_condition_number": field.get("max_condition_number"),
        "reject_nonconverged": field.get("reject_nonconverged"),
        "max_residual_rms": field.get("max_residual_rms"),
        "strain_type": field.get("strain_type", "green_lagrange"),
        "strain_convention": field.get("strain_convention", "tensor Exy; dimensionless"),
    }
    if defaults["max_residual_rms"] is not None:
        try:
            if not np.isfinite(float(defaults["max_residual_rms"])):
                defaults["max_residual_rms"] = None
        except (TypeError, ValueError):
            defaults["max_residual_rms"] = None
    for key, value in defaults.items():
        provenance.setdefault(key, value)
    if "max_residual_rms" in provenance:
        try:
            if provenance["max_residual_rms"] is not None and not np.isfinite(float(provenance["max_residual_rms"])):
                provenance["max_residual_rms"] = None
        except (TypeError, ValueError):
            provenance["max_residual_rms"] = None
    quality_summary = field.get("quality_summary")
    if quality_summary is not None:
        provenance.setdefault("computed", True)
        provenance.setdefault("scientific_ok", quality_summary.get("scientific_ok"))
        provenance.setdefault("correlation_valid_fraction", quality_summary.get("correlation_valid_fraction"))
        provenance.setdefault("strain_valid_fraction", quality_summary.get("strain_valid_fraction"))
        provenance.setdefault("quality_ambiguous_count", quality_summary.get("ambiguous_count"))
        provenance.setdefault("quality_scientific_reasons", quality_summary.get("scientific_reasons"))

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ezDIC full-field DIC provenance and parameters\n")
        handle.write("===============================================\n")
        for key, value in provenance.items():
            if isinstance(value, (list, tuple)):
                value = ",".join(str(item) for item in value)
            if key == "max_residual_rms" and value is None:
                value = "null"
            handle.write(f"{key} = {value}\n")
    return path


def style_dic_colorbar(cbar, preset, label):
    cbar.set_label(label, fontsize=preset["colorbar_label_size"])
    cbar.ax.tick_params(labelsize=preset["colorbar_tick_size"])
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=5, min_n_ticks=3)
    cbar.update_ticks()
    return cbar


def add_dic_colorbar(fig, ax, mappable, label, preset_name="publication"):
    """Attach a publication-styled colorbar; used by exports and the in-app field viewer."""
    preset = get_plot_preset(preset_name)
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        fraction=preset["colorbar_fraction"],
        pad=preset["colorbar_pad"],
    )
    return style_dic_colorbar(cbar, preset, label)


def dic_color_limits(field, component, options=None):
    """One explicit scale for maps and overlays; no percentile clipping."""
    options = dict(field.get("display_options") or {}) | dict(options or {})
    values = np.asarray(field[component], dtype=float)
    scale = 100.0 if options.get("percent", False) and component not in ("u", "v", "zncc") else 1.0
    finite = values[np.isfinite(values)]*scale
    mode = options.get("color_mode", "range")
    if mode == "manual":
        lo, hi = float(options["vmin"]), float(options["vmax"])
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError("color minimum and maximum must be finite and increasing")
    elif component == "zncc" and mode in ("range", "symmetric"):
        lo, hi = 0., 1.
    elif mode == "symmetric":
        hi = max(float(np.max(np.abs(finite))) if finite.size else 0, 1e-9)
        lo = -hi
    elif mode == "range":
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
        if hi-lo < 1e-12:
            padding = max(abs(lo)*.01, 1e-9)
            lo, hi = lo-padding, hi+padding
    else:
        raise ValueError("color mode must be range, symmetric or manual")
    return lo, hi, scale


def _dic_display_mesh(field, component, options=None):
    """Structured cells only: gaps, holes and folded cells have no triangles."""
    if component not in DIC_FIELD_COMPONENTS:
        raise ValueError(f"unknown DIC component: {component}")
    options = dict(field.get("display_options") or {}) | dict(options or {})
    background = options.get("background", "none")
    if background in ("reference", "deformed"):
        options["coordinate_frame"] = background
    X = np.asarray(field["X"], dtype=float)
    Y = np.asarray(field["Y"], dtype=float)
    values = np.asarray(field[component], dtype=float).reshape(X.shape)
    valid = np.isfinite(values) & np.asarray(field.get("valid", np.ones(X.size)), dtype=bool).reshape(X.shape)
    if component not in ("u", "v", "zncc"):
        valid &= np.asarray(field.get("strain_valid", np.ones(X.size)), dtype=bool).reshape(X.shape)
    coordinate = options.get("coordinate_frame", "reference")
    if coordinate == "deformed":
        X = X+np.asarray(field["u"]).reshape(X.shape)
        Y = Y+np.asarray(field["v"]).reshape(Y.shape)
    elif coordinate != "reference":
        raise ValueError("coordinate frame must be reference or deformed")
    valid &= np.isfinite(X)&np.isfinite(Y)
    triangles = []
    ny, nx = X.shape
    specimen = field.get("specimen_mask")
    if specimen is not None:
        specimen = np.asarray(specimen,dtype=bool)
    for i in range(ny-1):
        for j in range(nx-1):
            if valid[i:i+2,j:j+2].all():
                if specimen is not None:
                    rx = np.asarray(field["X"])[i:i+2,j:j+2]
                    ry = np.asarray(field["Y"])[i:i+2,j:j+2]
                    region = specimen[max(0,int(np.ceil(ry.min()))):int(np.floor(ry.max()))+1,
                                      max(0,int(np.ceil(rx.min()))):int(np.floor(rx.max()))+1]
                    if region.size == 0 or not region.all():
                        continue
                a, b, c, d = i*nx+j, i*nx+j+1, (i+1)*nx+j, (i+1)*nx+j+1
                for tri in ((a,b,d), (a,d,c)):
                    tx, ty = X.ravel()[list(tri)], Y.ravel()[list(tri)]
                    if (tx[1]-tx[0])*(ty[2]-ty[0])-(tx[2]-tx[0])*(ty[1]-ty[0]) > 1e-8:
                        triangles.append(tri)
    mesh = mtri.Triangulation(np.nan_to_num(X.ravel()), np.nan_to_num(Y.ravel()), np.asarray(triangles, dtype=int)) if triangles else None
    return X, Y, values, valid, mesh, options


def render_dic_field_on_axes(ax, field, component="u", *, cmap=None, options=None):
    """Continuous piecewise-linear contours, with original POIs unchanged."""
    X, Y, values, valid, tri, options = _dic_display_mesh(field, component, options)
    lo, hi, scale = dic_color_limits(field, component, options)
    cmap = cmap or options.get("cmap") or ("viridis" if component == "zncc" else "RdBu_r")
    image = field.get(f"{options.get('background', 'none')}_image")
    alpha = float(options.get("alpha", .65)) if image is not None else 1.
    if image is not None:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, origin="upper")
    if tri is not None and options.get("style", "contour") == "contour":
        minimum,maximum = float(np.min(values[valid]*scale)),float(np.max(values[valid]*scale))
        extension = "both" if minimum < lo and maximum > hi else ("min" if minimum < lo else ("max" if maximum > hi else "neither"))
        mesh = ax.tricontourf(tri, np.nan_to_num(values.ravel())*scale, levels=np.linspace(lo,hi,129),
                             cmap=cmap, vmin=lo, vmax=hi, extend=extension, alpha=alpha)
    else:
        mesh = ax.scatter(X[valid], Y[valid], c=values[valid]*scale, cmap=cmap, vmin=lo, vmax=hi, s=12, marker="s")
    ax.set_aspect("equal")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    return mesh


def plot_dic_field_map(field, path, component="u", title=None, preset_name="publication"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax, preset = create_plot_figure(preset_name)
    span_x, span_y = np.ptp(field["X"]), np.ptp(field["Y"])
    if span_y > 1.8*max(span_x,1):
        fig.set_size_inches(4.2,6.4)
    mesh = render_dic_field_on_axes(ax, field, component)
    label = DIC_COMPONENT_LABELS.get(component, component)
    if (field.get("display_options") or {}).get("percent") and component not in ("u", "v", "zncc"):
        label += " (%)"
    add_dic_colorbar(fig, ax, mesh, label, preset_name=preset_name)
    if title is None:
        title = label
    style_publication_axes(ax, preset, "x (px)", "y (px)", title, show_legend=False)
    ax.grid(False)
    ax.set_aspect("equal")
    save_plot_figure(fig, path, preset_name)
    return path


def overlay_dic_field_on_image(image, field, component="u", *, alpha=0.55, cmap=None, options=None):
    """Blend a DIC colormap onto the specimen image (uint8 RGB)."""
    gray = _as_float_image(image)
    h, w = gray.shape[:2]
    base = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    settings = {"coordinate_frame": "deformed"} | dict(field.get("display_options") or {}) | dict(options or {})
    alpha = float(settings.get("alpha", alpha))
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("overlay opacity must be between zero and one")
    X, Y, values, valid, tri, settings = _dic_display_mesh(field, component, settings)
    if tri is None:
        return base
    x0, x1 = max(0,int(np.floor(X[valid].min()))), min(w,int(np.ceil(X[valid].max()))+1)
    y0, y1 = max(0,int(np.floor(Y[valid].min()))), min(h,int(np.ceil(Y[valid].max()))+1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    interpolator = mtri.LinearTriInterpolator(tri, np.nan_to_num(values.ravel()))
    pixels = interpolator(xx, yy)
    mask = ~np.ma.getmaskarray(pixels)
    vmin, vmax, scale = dic_color_limits(field, component, settings)
    cmap = cmap or settings.get("cmap") or ("viridis" if component == "zncc" else "RdBu_r")
    color = plt.get_cmap(cmap)(Normalize(vmin, vmax, clip=True)(pixels.filled(0)*scale))[:,:,:3]
    overlay = np.clip(color*255, 0, 255).astype(np.uint8)
    out = base.copy()
    region = out[y0:y1,x0:x1]
    region[mask] = (
        (1.0 - alpha) * region[mask].astype(np.float32) + alpha * overlay[mask].astype(np.float32)
    ).astype(np.uint8)
    return out


def export_dic_field_outputs(field, output_dir, *, stem="dic_field", preset_name="publication"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = write_dic_field_txt(field, output_dir / f"{stem}.txt")
    plot_paths = []
    for component in DIC_FIELD_COMPONENTS:
        plot_paths.append(
            plot_dic_field_map(
                field,
                output_dir / f"{stem}_{DIC_COMPONENT_FILE_SUFFIXES[component]}.png",
                component=component,
                preset_name=preset_name,
            )
        )
    csv_path = output_dir / f"{stem}.csv"
    dic_field_to_dataframe(field).to_csv(csv_path, index=False)
    parameters_path = write_dic_field_parameters(field, output_dir / f"{stem}_parameters.txt")
    arrays = {key: np.asarray(value) for key, value in field.items()
              if isinstance(value, np.ndarray) and value.dtype.kind in "biuf" and not key.endswith("_image")}
    parameters = {key: field.get(key) for key in ("subset_size", "step", "strain_window", "smooth_sigma", "strain_degree",
                  "robust_strain", "outlier_threshold_px", "initialization", "initial_transform", "effective_search_radius_px")}
    arrays["metadata_json"] = np.asarray(json.dumps({"roi": field.get("roi"), "parameters": parameters, "provenance": field.get("provenance", {}),
                       "interpolation": field.get("interpolation"), "strain_gauge_span_px": field.get("strain_gauge_span_px"),
                       "mask": field.get("mask_provenance"), "display": field.get("display_options", {})}))
    array_path = output_dir / f"{stem}.npz"
    np.savez_compressed(array_path, **arrays)
    return {"txt": table_path, "csv": csv_path, "plots": plot_paths, "parameters": parameters_path, "arrays": array_path}

def _frame_column(df):
    if "frame_global_1based" in df.columns:
        return "frame_global_1based"
    if "frame_local_1based" in df.columns:
        return "frame_local_1based"
    return "frame"


def _format_origin_float(value):
    if pd.isna(value) or not np.isfinite(float(value)):
        return "NaN"
    return f"{float(value):.8f}"


def _engineering_to_true(value):
    if pd.isna(value) or not np.isfinite(float(value)) or (1.0 + float(value)) <= 0:
        return np.nan
    return math.log1p(float(value))


def _format_origin_value(column, value):
    if column.startswith("ValidGroupCount_"):
        return "NaN" if pd.isna(value) else str(int(value))
    return _format_origin_float(value)


def normalize_roi_role(role):
    role = str(role or "none").strip()
    if role not in ROI_ROLE_VALUES:
        return "none"
    return role


def poisson_roles_are_configured(groups):
    return any(normalize_roi_role(g.get("role", "none")) != "none" for g in groups)


def get_poisson_role_groups(groups):
    axial = [g for g in groups if normalize_roi_role(g.get("role", "none")) == "axial"]
    transverse = [g for g in groups if normalize_roi_role(g.get("role", "none")) == "transverse"]
    return axial, transverse


def _validate_single_actual_mode(groups, role_label):
    modes = sorted({str(g.get("actual_mode", "unknown") or "unknown").strip() for g in groups})
    if len(modes) > 1:
        raise RuntimeError(f"{role_label} ROI 组的 actual_mode 必须一致；当前为 {', '.join(modes)}。")


def validate_poisson_role_groups(groups):
    if not poisson_roles_are_configured(groups):
        return False

    axial, transverse = get_poisson_role_groups(groups)
    errors = []
    if len(axial) < 1:
        errors.append(f"拉伸方向 ROI 组必须至少有 1 个；当前为 {len(axial)} 个。")
    if len(transverse) < 1:
        errors.append(f"横向方向 ROI 组必须至少有 1 个；当前为 {len(transverse)} 个。")
    if errors:
        raise RuntimeError("\n".join(errors))
    axial_names = {g.get("name") for g in axial}
    transverse_names = {g.get("name") for g in transverse}
    if axial_names & transverse_names:
        raise RuntimeError("同一个 ROI 组不能同时作为拉伸方向和横向方向。")
    _validate_single_actual_mode(axial, "拉伸方向")
    _validate_single_actual_mode(transverse, "横向方向")
    return True


def _strain_valid_mask(gdf):
    """Resolve scientific strain validity, with an explicit legacy fallback."""
    if "strain_valid" in gdf.columns:
        values = gdf["strain_valid"]
        if values.dtype == bool:
            return values.fillna(False).astype(bool).reset_index(drop=True)
        lowered = values.astype(str).str.strip().str.lower()
        return lowered.isin({"1", "true", "yes", "y", "on"}).reset_index(drop=True)
    if "accepted" in gdf.columns:
        return gdf["accepted"].fillna(False).astype(bool).reset_index(drop=True)
    return pd.Series([True] * len(gdf), dtype=bool).reset_index(drop=True)


def build_core_strain_table(gdf):
    """
    Build the minimal Origin-friendly strain table:
    Frame, EngineeringStrain, TrueStrain.
    True strain is recomputed from engineering strain to keep export logic explicit.
    """
    frame_col = _frame_column(gdf)
    frames = pd.to_numeric(gdf[frame_col], errors="coerce").reset_index(drop=True)
    eng = pd.to_numeric(gdf["engineering_strain"], errors="coerce")
    valid = _strain_valid_mask(gdf)
    eng = eng.reset_index(drop=True).where(valid, np.nan)

    true_values = [_engineering_to_true(value) for value in eng]

    out = pd.DataFrame(
        {
            "Frame": frames.astype("Int64"),
            "EngineeringStrain": eng.astype(float),
            "TrueStrain": np.array(true_values, dtype=float),
        }
    )
    return out.reset_index(drop=True)


def write_origin_txt(gdf, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = build_core_strain_table(gdf)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("Frame\tEngineeringStrain\tTrueStrain\n")
        for _, row in table.iterrows():
            frame = "NaN" if pd.isna(row["Frame"]) else str(int(row["Frame"]))
            f.write(
                f"{frame}\t"
                f"{_format_origin_float(row['EngineeringStrain'])}\t"
                f"{_format_origin_float(row['TrueStrain'])}\n"
            )


def _group_engineering_strain_table(df, group_name, output_column):
    gdf = df[df["group"] == group_name].copy()
    if gdf.empty:
        return pd.DataFrame(columns=["Frame", output_column])

    table = build_core_strain_table(gdf)
    strain = table["EngineeringStrain"].astype(float)
    strain = strain.where(_strain_valid_mask(gdf), np.nan)

    return pd.DataFrame(
        {
            "Frame": table["Frame"],
            output_column: strain,
        }
    ).reset_index(drop=True)


def _frame_table(df):
    frame_col = _frame_column(df)
    frames = pd.to_numeric(df[frame_col], errors="coerce").dropna().drop_duplicates().sort_values()
    return pd.DataFrame({"Frame": frames.astype("Int64")}).reset_index(drop=True)


def _mean_group_key(group):
    role = normalize_roi_role(group.get("role", "none"))
    actual_mode = str(group.get("actual_mode", "unknown") or "unknown").strip()
    return safe_name(f"{role}_{actual_mode}")


def _mean_group_specs(groups):
    specs = {}
    for group in groups or []:
        key = _mean_group_key(group)
        if key not in specs:
            specs[key] = {"key": key, "groups": []}
        specs[key]["groups"].append(group)
    return list(specs.values())


def _mean_engineering_strain_table_for_groups(df, groups, output_column):
    out = _frame_table(df)
    if out.empty:
        return pd.DataFrame(columns=["Frame", output_column])

    strain_cols = []
    merged = out.copy()
    for idx, group in enumerate(groups, start=1):
        col = f"__strain_{idx}"
        group_table = _group_engineering_strain_table(df, group.get("name"), col)
        merged = pd.merge(merged, group_table, on="Frame", how="left")
        strain_cols.append(col)

    if not strain_cols:
        merged[output_column] = np.nan
    else:
        strains = merged[strain_cols].apply(pd.to_numeric, errors="coerce")
        counts = strains.notna().sum(axis=1)
        merged[output_column] = strains.mean(axis=1, skipna=True).where(counts > 0, np.nan)

    return merged[["Frame", output_column]].reset_index(drop=True)


def build_mean_strain_table(df, groups):
    out = _frame_table(df)
    if out.empty or not groups:
        return out

    for spec in _mean_group_specs(groups):
        key = spec["key"]
        merged = out.copy()
        strain_cols = []
        for idx, group in enumerate(spec["groups"], start=1):
            col = f"__{key}_{idx}"
            group_table = _group_engineering_strain_table(df, group.get("name"), col)
            merged = pd.merge(merged, group_table, on="Frame", how="left")
            strain_cols.append(col)

        strains = merged[strain_cols].apply(pd.to_numeric, errors="coerce")
        counts = strains.notna().sum(axis=1)
        mean = strains.mean(axis=1, skipna=True).where(counts > 0, np.nan)
        std = strains.std(axis=1, skipna=True, ddof=1).where(counts >= 2, np.nan)
        sem = (std / np.sqrt(counts.astype(float))).where(counts >= 2, np.nan)

        out[f"MeanEngineeringStrain_{key}"] = mean
        out[f"MeanTrueStrain_{key}"] = mean.apply(_engineering_to_true)
        out[f"StdEngineeringStrain_{key}"] = std
        out[f"SemEngineeringStrain_{key}"] = sem
        out[f"ValidGroupCount_{key}"] = counts.astype(int)

    return out.reset_index(drop=True)


def build_poisson_ratio_table(df, groups, min_abs_axial=POISSON_MIN_ABS_AXIAL_ENGINEERING_STRAIN):
    if not validate_poisson_role_groups(groups):
        raise RuntimeError("请先设置 1 个拉伸方向 ROI 组和 1 个横向方向 ROI 组。")
    axial_groups, transverse_groups = get_poisson_role_groups(groups)

    axial = _mean_engineering_strain_table_for_groups(df, axial_groups, "AxialEngineeringStrain")
    transverse = _mean_engineering_strain_table_for_groups(df, transverse_groups, "TransverseEngineeringStrain")
    merged = pd.merge(axial, transverse, on="Frame", how="outer").sort_values("Frame").reset_index(drop=True)

    axial_strain = pd.to_numeric(merged["AxialEngineeringStrain"], errors="coerce").astype(float)
    transverse_strain = pd.to_numeric(merged["TransverseEngineeringStrain"], errors="coerce").astype(float)
    valid = (
        axial_strain.notna()
        & transverse_strain.notna()
        & np.isfinite(axial_strain)
        & np.isfinite(transverse_strain)
        & (axial_strain.abs() >= float(min_abs_axial))
    )
    poisson = pd.Series(np.nan, index=merged.index, dtype=float)
    poisson.loc[valid] = -transverse_strain.loc[valid] / axial_strain.loc[valid]
    merged["PoissonRatio"] = poisson

    return merged[
        ["Frame", "AxialEngineeringStrain", "TransverseEngineeringStrain", "PoissonRatio"]
    ].reset_index(drop=True)


def build_all_groups_strain_table(df, groups=None):
    tables = []
    for gname, gdf in df.groupby("group", sort=False):
        sg = safe_name(gname)
        table = build_core_strain_table(gdf).rename(
            columns={
                "EngineeringStrain": f"EngineeringStrain_{sg}",
                "TrueStrain": f"TrueStrain_{sg}",
            }
        )
        tables.append(table)

    if not tables:
        return pd.DataFrame(columns=["Frame"])

    merged = tables[0]
    for table in tables[1:]:
        merged = pd.merge(merged, table, on="Frame", how="outer")
    merged = merged.sort_values("Frame").reset_index(drop=True)

    if groups is not None:
        mean_table = build_mean_strain_table(df, groups)
        if len(mean_table.columns) > 1:
            merged = pd.merge(merged, mean_table, on="Frame", how="outer")
            merged = merged.sort_values("Frame").reset_index(drop=True)

    if groups is not None and poisson_roles_are_configured(groups):
        poisson = build_poisson_ratio_table(df, groups)
        merged = pd.merge(merged, poisson, on="Frame", how="outer")
        merged = merged.sort_values("Frame").reset_index(drop=True)

    return merged


def write_all_groups_origin_txt(df, path, groups=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = build_all_groups_strain_table(df, groups)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(table.columns) + "\n")
        for _, row in table.iterrows():
            values = []
            for col in table.columns:
                if col == "Frame":
                    values.append("NaN" if pd.isna(row[col]) else str(int(row[col])))
                else:
                    values.append(_format_origin_value(col, row[col]))
            f.write("\t".join(values) + "\n")


def write_mean_groups_origin_txt(df, groups, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = build_mean_strain_table(df, groups)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(table.columns) + "\n")
        for _, row in table.iterrows():
            values = []
            for col in table.columns:
                if col == "Frame":
                    values.append("NaN" if pd.isna(row[col]) else str(int(row[col])))
                else:
                    values.append(_format_origin_value(col, row[col]))
            f.write("\t".join(values) + "\n")


def write_poisson_ratio_txt(df, groups, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = build_poisson_ratio_table(df, groups)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("Frame\tAxialEngineeringStrain\tTransverseEngineeringStrain\tPoissonRatio\n")
        for _, row in table.iterrows():
            frame = "NaN" if pd.isna(row["Frame"]) else str(int(row["Frame"]))
            f.write(
                f"{frame}\t"
                f"{_format_origin_float(row['AxialEngineeringStrain'])}\t"
                f"{_format_origin_float(row['TransverseEngineeringStrain'])}\t"
                f"{_format_origin_float(row['PoissonRatio'])}\n"
            )


def build_origin_project_tables(df, groups):
    groups = list(groups or [])
    tables = []

    for group in groups:
        gname = group.get("name")
        sg = safe_name(gname)
        gdf = df[df["group"] == gname].copy()
        tables.append((f"strain_{sg}", build_core_strain_table(gdf)))

    tables.append(("strain_all_groups", build_all_groups_strain_table(df, groups)))

    mean_table = build_mean_strain_table(df, groups)
    if len(mean_table.columns) > 1:
        tables.append(("strain_mean_groups", mean_table))

    if poisson_roles_are_configured(groups):
        tables.append(("poisson_ratio", build_poisson_ratio_table(df, groups)))

    return tables


def _load_originpro_module():
    try:
        import originpro as op
    except ImportError as exc:
        raise RuntimeError(
            "无法导入 originpro。请在 Windows + OriginPro 2021+ 环境中安装 originpro Python 包后再导出 OPJU。"
        ) from exc
    return op


def write_origin_opju_project(df, groups, path, origin_module=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    op = origin_module if origin_module is not None else _load_originpro_module()

    try:
        op.new(asksave=True)
        for table_name, table in build_origin_project_tables(df, groups):
            worksheet = op.new_sheet("w", lname=table_name)
            if worksheet is None:
                raise RuntimeError(f"无法创建 Origin worksheet：{table_name}")
            worksheet.from_df(table)

        if not op.save(str(path)):
            raise RuntimeError(f"保存 Origin OPJU 项目失败：{path}")
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("保存 Origin OPJU 项目失败") or message.startswith("无法"):
            raise
        raise RuntimeError(f"生成 Origin OPJU 项目失败：{exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"生成 Origin OPJU 项目失败：{exc}") from exc

    return path

def get_plot_preset(name="publication"):
    preset = PLOT_EXPORT_PRESETS.get(name, PLOT_EXPORT_PRESETS["publication"])
    return dict(preset)


def create_plot_figure(preset_name="publication"):
    preset = get_plot_preset(preset_name)
    fig, ax = plt.subplots(
        figsize=preset["figsize"],
        constrained_layout=preset["constrained_layout"],
    )
    fig.patch.set_facecolor("white")
    ax.set_prop_cycle(color=PLOT_COLOR_CYCLE)
    return fig, ax, preset


def style_publication_axes(ax, preset, xlabel, ylabel, title=None, show_legend=True):
    ax.set_xlabel(xlabel, fontsize=preset["label_size"])
    ax.set_ylabel(ylabel, fontsize=preset["label_size"])
    if title:
        ax.set_title(title, fontsize=preset["title_size"], pad=8)
    ax.tick_params(axis="both", labelsize=preset["tick_size"], width=preset["axis_line_width"], length=3.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_linewidth(preset["axis_line_width"])
    ax.grid(True, color="#b8c2cc", alpha=preset["grid_alpha"], linewidth=0.55)
    ax.margins(x=0.02, y=0.08)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ncol = 2 if len(handles) > 6 else 1
            ax.legend(
                loc="best",
                frameon=False,
                fontsize=preset["legend_size"],
                handlelength=1.6,
                borderaxespad=0.3,
                ncol=ncol,
            )


def save_plot_figure(fig, path, preset_name="publication"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preset = get_plot_preset(preset_name)
    fig.savefig(path, dpi=preset["dpi"], bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def publication_figure_paths(folder, stem):
    folder = Path(folder)
    return [folder / f"{stem}.{ext}" for ext in PLOT_EXPORT_FORMATS]


def plot_engineering_strain(gdf, path, title, preset_name="publication"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = build_core_strain_table(gdf)
    frame = table["Frame"].astype(float)
    strain = table["EngineeringStrain"].astype(float)
    accepted = gdf["accepted"].astype(bool).reset_index(drop=True) if "accepted" in gdf.columns else strain.notna()
    accept_mode = gdf["accept_mode"].astype(str).reset_index(drop=True) if "accept_mode" in gdf.columns else pd.Series([""] * len(gdf))

    rejected_mask = (~accepted) | strain.isna()
    adaptive_mask = accept_mode.eq("adaptive") & (~rejected_mask) & strain.notna()
    normal_mask = (~rejected_mask) & (~adaptive_mask) & strain.notna()

    fig, ax, preset = create_plot_figure(preset_name)
    ax.plot(frame, strain, color=PLOT_COLOR_CYCLE[0], linewidth=preset["line_width"], alpha=0.72)
    ax.scatter(frame[normal_mask], strain[normal_mask], color=PLOT_COLOR_CYCLE[0], s=preset["marker_size"], label="Accepted")

    if adaptive_mask.any():
        ax.scatter(frame[adaptive_mask], strain[adaptive_mask], color="#E69F00", s=preset["marker_size"] * 1.35, label="Adaptive")

    if rejected_mask.any():
        finite_strain = strain[np.isfinite(strain)]
        if len(finite_strain) > 0:
            ymin = float(finite_strain.min())
            ymax = float(finite_strain.max())
            span = ymax - ymin if ymax > ymin else max(abs(ymax), 1e-6)
            rejected_y = np.full(int(rejected_mask.sum()), ymin - 0.08 * span)
        else:
            rejected_y = np.zeros(int(rejected_mask.sum()))
        ax.scatter(frame[rejected_mask], rejected_y, color="#CC3311", marker="x", s=preset["marker_size"] * 1.75, label="Rejected/NaN")

    style_publication_axes(ax, preset, "Frame", "Engineering strain (-)", title=title)
    save_plot_figure(fig, path, preset_name)


def plot_all_groups_engineering_strain(df, groups, path, preset_name="publication"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax, preset = create_plot_figure(preset_name)
    for group in groups:
        gname = group["name"]
        gdf = df[df["group"] == gname]
        if gdf.empty:
            continue
        ax.plot(
            gdf["frame_global_1based"],
            gdf["engineering_strain"],
            linewidth=max(preset["line_width"] * 0.8, 0.8),
            alpha=0.56,
            label=gname,
        )

    mean_table = build_mean_strain_table(df, groups)
    if not mean_table.empty:
        frame = mean_table["Frame"].astype(float)
        for col in [c for c in mean_table.columns if c.startswith("MeanEngineeringStrain_")]:
            key = col.replace("MeanEngineeringStrain_", "", 1)
            mean = pd.to_numeric(mean_table[col], errors="coerce").astype(float)
            std_col = f"StdEngineeringStrain_{key}"
            line = ax.plot(frame, mean, linewidth=preset["line_width"] * 1.8, label=f"Mean {key}")[0]
            if std_col in mean_table.columns:
                std = pd.to_numeric(mean_table[std_col], errors="coerce").astype(float)
                finite = mean.notna() & std.notna() & np.isfinite(mean) & np.isfinite(std)
                if finite.any():
                    ax.fill_between(
                        frame[finite],
                        mean[finite] - std[finite],
                        mean[finite] + std[finite],
                        color=line.get_color(),
                        alpha=0.13,
                        linewidth=0,
                    )

    style_publication_axes(ax, preset, "Frame", "Engineering strain (-)", title="Engineering strain - all ROI groups")
    save_plot_figure(fig, path, preset_name)


def plot_poisson_ratio(df, groups, path, preset_name="publication"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = build_poisson_ratio_table(df, groups)
    frame = table["Frame"].astype(float)
    ratio = table["PoissonRatio"].astype(float)
    valid = ratio.notna() & np.isfinite(ratio)

    fig, ax, preset = create_plot_figure(preset_name)
    ax.plot(frame, ratio, color="#009E73", linewidth=preset["line_width"], alpha=0.78)
    if valid.any():
        ax.scatter(frame[valid], ratio[valid], color="#009E73", s=preset["marker_size"], label="Valid")
    invalid = ~valid
    if invalid.any():
        finite_ratio = ratio[valid]
        if len(finite_ratio) > 0:
            ymin = float(finite_ratio.min())
            ymax = float(finite_ratio.max())
            span = max(ymax - ymin, 1e-6)
            invalid_y = np.full(int(invalid.sum()), ymin - 0.08 * span)
        else:
            invalid_y = np.zeros(int(invalid.sum()))
        ax.scatter(frame[invalid], invalid_y, color="#CC3311", marker="x", s=preset["marker_size"] * 1.75, label="NaN")

    style_publication_axes(ax, preset, "Frame", "Poisson ratio (-)", title="Poisson ratio")
    save_plot_figure(fig, path, preset_name)


def plot_correlation_scores(gdf, path, group_name, hard_corr, soft_corr, preset_name="publication"):
    fig, ax, preset = create_plot_figure(preset_name)
    frame = gdf["frame_global_1based"]
    ax.plot(frame, gdf["corr_score_roi1"], marker="o", markersize=4, linewidth=preset["line_width"], label="ROI 1")
    ax.plot(frame, gdf["corr_score_roi2"], marker="s", markersize=4, linewidth=preset["line_width"], label="ROI 2")
    ax.axhline(hard_corr, color="#CC3311", linestyle="--", linewidth=preset["line_width"] * 0.9, label="strict threshold")
    ax.axhline(soft_corr, color="#E69F00", linestyle=":", linewidth=preset["line_width"] * 0.9, label="weak threshold")
    ax.set_ylim(-0.05, 1.05)
    style_publication_axes(ax, preset, "Frame", "Normalized correlation score (-)", title=f"Correlation scores - {group_name}")
    save_plot_figure(fig, path, preset_name)


def build_qc_summary(df):
    levels = {"Good": 0, "Warning": 1, "Poor": 2}
    groups = {}

    for gname, sub in df.groupby("group", sort=False):
        frames = int(len(sub))
        accepted = sub["accepted"].astype(bool) if "accepted" in sub.columns else pd.Series([True] * frames, index=sub.index)
        eng = pd.to_numeric(sub["engineering_strain"], errors="coerce")
        strain_valid = _strain_valid_mask(sub)
        rejected_mask = pd.Series(
            (~strain_valid.to_numpy()) | eng.isna().to_numpy(),
            index=sub.index,
        )
        adaptive_mask = sub["accept_mode"].astype(str).eq("adaptive") if "accept_mode" in sub.columns else pd.Series([False] * frames, index=sub.index)

        rejected_frames = int(rejected_mask.sum())
        valid_frames = int(frames - rejected_frames)
        adaptive_frames = int((adaptive_mask & (~rejected_mask)).sum())

        corr1 = pd.to_numeric(sub["corr_score_roi1"], errors="coerce") if "corr_score_roi1" in sub.columns else pd.Series(dtype=float)
        corr2 = pd.to_numeric(sub["corr_score_roi2"], errors="coerce") if "corr_score_roi2" in sub.columns else pd.Series(dtype=float)
        mean_corr1 = float(corr1.mean()) if len(corr1.dropna()) else np.nan
        mean_corr2 = float(corr2.mean()) if len(corr2.dropna()) else np.nan

        valid_eng = eng.dropna()
        max_abs_strain = float(valid_eng.abs().max()) if len(valid_eng) else np.nan
        jumps = valid_eng.diff().abs().dropna()
        max_jump = float(jumps.max()) if len(jumps) else 0.0

        frame_col = _frame_column(sub)
        rejected_frame_list = [
            int(x) for x in pd.to_numeric(sub.loc[rejected_mask, frame_col], errors="coerce").dropna().tolist()
        ]

        rejected_ratio = rejected_frames / frames if frames else 0.0
        finite_means = [value for value in [mean_corr1, mean_corr2] if np.isfinite(value)]
        min_mean_corr = min(finite_means) if finite_means else np.nan
        if rejected_ratio > 0.05:
            qc_level = "Poor"
        elif rejected_frames > 0 or adaptive_frames > 0 or (np.isfinite(min_mean_corr) and min_mean_corr < 0.80):
            qc_level = "Warning"
        else:
            qc_level = "Good"

        groups[str(gname)] = {
            "frames": frames,
            "valid_frames": valid_frames,
            "strain_valid_frames": int(strain_valid.sum()),
            "rejected_frames": rejected_frames,
            "adaptive_accepted_frames": adaptive_frames,
            "mean_corr_roi1": mean_corr1,
            "mean_corr_roi2": mean_corr2,
            "max_abs_engineering_strain": max_abs_strain,
            "max_frame_strain_jump": max_jump,
            "rejected_frame_list": rejected_frame_list,
            "qc_level": qc_level,
        }

    overall_level = "Good"
    for item in groups.values():
        if levels[item["qc_level"]] > levels[overall_level]:
            overall_level = item["qc_level"]

    return {
        "overall": {
            "qc_level": overall_level,
            "groups": len(groups),
            "rejected_frames": int(sum(item["rejected_frames"] for item in groups.values())),
            "adaptive_accepted_frames": int(sum(item["adaptive_accepted_frames"] for item in groups.values())),
        },
        "groups": groups,
    }


def _format_qc_number(value, digits=3):
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return "NaN"
    return f"{float(value):.{digits}f}"


def write_qc_summary(summary, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("ezDIC QC Summary\n")
        f.write("================\n")
        f.write(f"Overall QC level: {summary['overall']['qc_level']}\n")
        f.write(f"Groups: {summary['overall']['groups']}\n")
        f.write(f"Rejected frames: {summary['overall']['rejected_frames']}\n")
        f.write(f"Adaptive accepted frames: {summary['overall']['adaptive_accepted_frames']}\n")

        for gname, item in summary["groups"].items():
            rejected_list = ", ".join(str(x) for x in item["rejected_frame_list"]) or "None"
            f.write(f"\n[{gname}]\n")
            f.write(f"Frames: {item['frames']}\n")
            f.write(f"Valid frames: {item['valid_frames']}\n")
            f.write(f"Rejected frames: {item['rejected_frames']}\n")
            f.write(f"Adaptive accepted frames: {item['adaptive_accepted_frames']}\n")
            f.write(f"Mean corr ROI1: {_format_qc_number(item['mean_corr_roi1'], 3)}\n")
            f.write(f"Mean corr ROI2: {_format_qc_number(item['mean_corr_roi2'], 3)}\n")
            f.write(f"Max abs engineering strain: {_format_qc_number(item['max_abs_engineering_strain'], 4)}\n")
            f.write(f"Max frame strain jump: {_format_qc_number(item['max_frame_strain_jump'], 4)}\n")
            f.write(f"Rejected frame list: {rejected_list}\n")
            f.write(f"QC level: {item['qc_level']}\n")

# ---------------------------------------------------------------------------
# Headless sequence orchestration and transactional publication
# ---------------------------------------------------------------------------

RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_FAILED_DIRNAME = "_failed_runs"
RUN_PREVIOUS_DIRNAME = "_previous_runs"
RUN_STAGING_PREFIX = ".staging"
RUN_ID_MAX_LENGTH = 64
_RUN_ID_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,62}[A-Za-z0-9])?\Z", flags=re.ASCII)
_WINDOWS_RESERVED_RUN_IDS = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_RUN_TRANSACTION_LOCKS = {}
_RUN_TRANSACTION_LOCKS_GUARD = threading.Lock()


def _validate_run_id(run_id):
    """Validate a run identifier before it can participate in any path."""
    if not isinstance(run_id, str):
        raise CoreError(
            "INVALID_RUN_ID",
            {"run_id": str(run_id), "reason": "type", "message": "run_id must be an ASCII string token"},
        )
    if len(run_id) > RUN_ID_MAX_LENGTH:
        raise CoreError(
            "INVALID_RUN_ID",
            {"run_id": run_id[:80], "reason": "length", "max_length": RUN_ID_MAX_LENGTH, "message": "run_id is too long"},
        )
    try:
        run_id.encode("ascii")
    except UnicodeEncodeError as exc:
        raise CoreError(
            "INVALID_RUN_ID",
            {"run_id": run_id, "reason": "non_ascii", "message": "run_id must contain ASCII characters only"},
        ) from exc
    if not _RUN_ID_RE.fullmatch(run_id):
        raise CoreError(
            "INVALID_RUN_ID",
            {
                "run_id": run_id,
                "reason": "unsafe_token",
                "message": "run_id may contain only ASCII letters, digits, underscore and hyphen",
            },
        )
    if run_id.casefold() in _WINDOWS_RESERVED_RUN_IDS:
        raise CoreError(
            "INVALID_RUN_ID",
            {"run_id": run_id, "reason": "reserved_name", "message": "run_id is a reserved Windows device name"},
        )
    return run_id


def _operation_state_dir():
    """Return the external operation-ledger/lock directory."""
    override = os.environ.get("EZDIC_STATE_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / APP_NAME / "state"
        return Path.home() / "AppData" / "Local" / APP_NAME / "state"
    base = os.environ.get("XDG_STATE_HOME")
    if base:
        return Path(base) / APP_NAME / "state"
    return Path.home() / ".local" / "state" / APP_NAME


def _operation_key(output_root):
    # ``Path.resolve`` does not canonicalize the case of a final component that
    # has not been created yet.  Windows nevertheless treats ``OutNew`` and
    # ``outnew`` as the same output root, so normalize with the host's path
    # semantics before hashing the cross-process lock key.
    resolved = os.path.abspath(os.fspath(Path(output_root).expanduser()))
    resolved = os.path.realpath(resolved)
    resolved = os.path.normcase(resolved)
    if os.name == "nt":
        resolved = resolved.casefold()
    resolved = resolved.replace("\\", "/")
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def _operation_ledger_path(output_root):
    return _operation_state_dir() / f"{_operation_key(output_root)}.json"


def _operation_lock_path(output_root):
    return _operation_state_dir() / f"{_operation_key(output_root)}.lock"


def _assert_no_reparse_components(path, *, allow_missing_leaf=True):
    """Reject symlink/reparse components before an operation touches them."""
    candidate = Path(os.path.abspath(os.fspath(path)))
    current = Path(candidate.anchor) if candidate.anchor else Path.cwd()
    parts = candidate.parts
    if parts and candidate.anchor == parts[0]:
        parts = parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = os.lstat(str(current))
        except FileNotFoundError:
            if allow_missing_leaf:
                return
            raise CoreError("PATH_COMPONENT_MISSING", {"path": str(current)})
        except OSError as exc:
            raise CoreError("PATH_COMPONENT_ERROR", {"path": str(current), "message": str(exc)}) from exc
        attributes = int(getattr(info, "st_file_attributes", 0))
        reparse_attribute = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if stat.S_ISLNK(info.st_mode) or (attributes & reparse_attribute):
            raise CoreError("REPARSE_PATH_REJECTED", {"path": str(current), "message": "symlink/junction/reparse component is not allowed"})
        if index == len(parts) - 1 and allow_missing_leaf and not stat.S_ISDIR(info.st_mode):
            return


@contextmanager
def _operation_os_lock(output_root):
    """Acquire an OS-level cooperative lock for one output root."""
    state_dir = _operation_state_dir()
    _assert_no_reparse_components(state_dir, allow_missing_leaf=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_components(state_dir, allow_missing_leaf=False)
    lock_path = _operation_lock_path(output_root)
    _assert_no_reparse_components(lock_path, allow_missing_leaf=True)
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield lock_path
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _write_operation_ledger(output_root, manifest, mode):
    state_dir = _operation_state_dir()
    _assert_no_reparse_components(state_dir, allow_missing_leaf=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_components(state_dir, allow_missing_leaf=False)
    ledger_path = _operation_ledger_path(output_root)
    _assert_no_reparse_components(ledger_path, allow_missing_leaf=True)
    payload = {
        "ledger_version": 1,
        "output_root": str(Path(output_root).resolve()),
        "manifest_hash": manifest.get("manifest_hash"),
        "mode": mode,
        "owned_output_paths": sorted(str(value).replace("\\", "/") for value in manifest.get("owned_output_paths", [])),
    }
    payload["ledger_hash"] = canonical_json_hash(payload)
    temporary = ledger_path.with_name(f".{ledger_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
        os.replace(str(temporary), str(ledger_path))
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return ledger_path


def _read_operation_ledger(output_root):
    path = _operation_ledger_path(output_root)
    try:
        _assert_no_reparse_components(path, allow_missing_leaf=True)
    except CoreError:
        return None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    expected_hash = payload.get("ledger_hash")
    actual_hash = canonical_json_hash({key: value for key, value in payload.items() if key != "ledger_hash"})
    if not expected_hash or expected_hash != actual_hash:
        return None
    return dict(payload)


def _portable_relative(path, root):
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError as exc:
        raise CoreError(
            "PATH_OUTSIDE_ROOT",
            {"path": str(path), "root": str(root), "message": "path must remain inside the selected root"},
        ) from exc


def _assert_contained_path(path, root, *, code="PATH_OUTSIDE_ROOT"):
    """Reject symlink/junction-resolved paths outside a transaction root."""
    candidate = Path(path).resolve(strict=False)
    base = Path(root).resolve(strict=False)
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise CoreError(code, {"path": str(path), "resolved": str(candidate), "root": str(base)}) from exc
    return candidate


def _iter_files(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted((path for path in root.rglob("*") if path.is_file()), key=lambda item: item.as_posix())


def _is_ezdic_output_pattern(relative, mode=None):
    """Return whether a relative path is an explicit generated-output name."""
    relative = str(relative).replace("\\", "/")
    if mode in (None, ANALYSIS_MODE_EXTENSOMETER):
        if relative == "qc/qc_summary.txt":
            return True
        if relative.startswith("core/"):
            return bool(re.fullmatch(r"core/(?:strain|engineering_strain|poisson_ratio)(?:_[\w.\-]+)?\.(?:txt|png)", relative, flags=re.IGNORECASE)) or relative == f"core/{ORIGIN_OPJU_FILENAME}"
        if relative.startswith("optional/"):
            return bool(re.fullmatch(r"optional/(?:publication_figures|correlation_plots)/[\w.\-/]+\.(?:png|tiff|pdf|svg|eps)", relative, flags=re.IGNORECASE)) or bool(re.fullmatch(r"optional/full_csv/(?:strain_results_all_groups\.csv|per_group_results/strain_results_[\w.\-]+\.csv)", relative, flags=re.IGNORECASE)) or bool(re.fullmatch(r"optional/parameters/(?:tracking_parameters|acceptance_summary)\.txt", relative, flags=re.IGNORECASE)) or bool(re.fullmatch(r"optional/overlays/[\w.\-]+/tracked_\d{5}\.png", relative, flags=re.IGNORECASE))
    if mode in (None, ANALYSIS_MODE_FULLFIELD) and relative.startswith("dic/"):
        return relative == "dic/specimen_mask.png" or bool(re.fullmatch(r"dic/frame_\d{4}(?:_(?:u|v|zncc|Exx|Eyy|Exy|exx|eyy|exy|exx_infinitesimal|eyy_infinitesimal|exy_infinitesimal|overlay)|_parameters)?\.(?:txt|csv|png|npz)", relative, flags=re.IGNORECASE))
    return False


def _legacy_owned_output_paths(output_root):
    """Return conservative, known legacy output paths when no ledger exists."""
    root = Path(output_root)
    result = []
    exact_names = {
        "qc/qc_summary.txt",
        "core/strain_all_groups.txt",
        "core/strain_mean_groups.txt",
        "core/poisson_ratio.txt",
        f"core/{ORIGIN_OPJU_FILENAME}",
        "core/engineering_strain_all_groups.png",
        "core/poisson_ratio.png",
        "optional/full_csv/strain_results_all_groups.csv",
        "optional/parameters/tracking_parameters.txt",
        "optional/parameters/acceptance_summary.txt",
    }
    for relative in exact_names:
        path = root / Path(relative)
        if path.is_file():
            result.append((relative, path))
    # Do not guess per-group or per-frame ownership without a manifest.  A
    # collision is safer than moving a similarly named user file; successful
    # v0.2 runs always create a manifest for the next transaction to trust.
    return sorted({relative: path for relative, path in result}.items(), key=lambda item: item[0])


def _current_owned_output_paths(output_root, *, mode=None, allowed_paths=None):
    """Read the current run's exact ownership ledger, safely falling back."""
    root = Path(output_root)
    manifest_path = root / RUN_MANIFEST_FILENAME
    if not manifest_path.is_file():
        # Without a valid ownership ledger, do not infer that legacy-looking
        # files belong to ezDIC.  A destination collision will fail closed and
        # preserve the user's bytes.
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, Mapping):
        return []
    ledger = _read_operation_ledger(root)
    if not isinstance(ledger, Mapping):
        return []
    if ledger.get("output_root") != str(root.resolve()):
        return []
    if ledger.get("manifest_hash") != payload.get("manifest_hash"):
        return []
    # Ownership is established by the valid previous ledger, not by the mode
    # of the next run.  Switching between 1-D and full-field export policies
    # must still archive every path from the previous current bundle.
    if not isinstance(ledger.get("owned_output_paths"), list):
        return []
    # Verify the actual current bundle before granting it authority to archive
    # any file.  In particular, ``owned_output_paths`` must exactly match the
    # verified output inventory; an extra user path in a recomputed ledger is
    # never silently moved.
    verification = verify_run_manifest(manifest_path, verify_code=False)
    if not verification.get("ok"):
        return []
    output_entries = payload.get("outputs")
    owned_paths = payload.get("owned_output_paths")
    if not isinstance(output_entries, list) or not isinstance(owned_paths, list):
        return []
    output_paths = {str(entry.get("path")).replace("\\", "/") for entry in output_entries if isinstance(entry, Mapping) and entry.get("path")}
    owned_set = {str(value).replace("\\", "/") for value in owned_paths}
    if owned_set != output_paths:
        return []
    if {str(value).replace("\\", "/") for value in ledger.get("owned_output_paths", [])} != owned_set:
        return []
    owned = []
    # ``allowed_paths`` is retained as a compatibility argument for callers
    # that only need a subset for inspection.  Transaction commit intentionally
    # passes no filter: omitted files are stale generated outputs and must be
    # archived under the external operation lock.
    allowed = None if allowed_paths is None else {str(value).replace("\\", "/") for value in allowed_paths}
    for relative in sorted(output_paths):
        if allowed is not None and relative not in allowed:
            continue
        candidate = Path(relative)
        if candidate.is_absolute():
            return []
        candidate = root / candidate
        try:
            if _portable_relative(candidate, root) != relative:
                return []
        except CoreError:
            return []
        if not candidate.is_file():
            return []
        owned.append((relative, candidate))
    owned.append((RUN_MANIFEST_FILENAME, manifest_path))
    return owned


def _preserved_unowned_output_paths(output_root, owned=None):
    """Snapshot current user files so verify never mistakes them for outputs."""
    root = Path(output_root)
    owned_rel = {str(relative).replace("\\", "/") for relative, _ in (owned or _current_owned_output_paths(root))}
    preserved = []
    for path in _iter_files(root):
        relative = _portable_relative(path, root)
        if relative == RUN_MANIFEST_FILENAME or relative in owned_rel:
            continue
        if relative.startswith(f"{RUN_PREVIOUS_DIRNAME}/") or relative.startswith(f"{RUN_FAILED_DIRNAME}/") or relative.startswith(f"{RUN_STAGING_PREFIX}_"):
            continue
        preserved.append(relative)
    return sorted(preserved)


def _windows_normalize_handle_path(path):
    """Normalize a Win32 final path for a lexical identity comparison."""
    value = str(path)
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    value = os.path.normcase(os.path.abspath(value))
    if len(value) > 3:
        value = value.rstrip("\\/")
    return value


def _windows_existing_directory_components(path):
    """Return existing lexical directory components from root to ``path``."""
    current = Path(os.path.abspath(os.fspath(path)))
    components = []
    while True:
        try:
            info = os.lstat(str(current))
        except FileNotFoundError:
            parent = current.parent
            if parent == current:
                break
            current = parent
            continue
        except OSError as exc:
            raise CoreError("WINDOWS_DIRECTORY_GUARD_FAILED", {"path": str(current), "message": str(exc)}) from exc
        if not stat.S_ISDIR(info.st_mode):
            break
        components.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return list(reversed(components))


class _WindowsDirectoryGuards:
    """Keep every existing move-path directory open without DELETE sharing."""

    def __init__(self):
        self._records = {}
        self._kernel32 = None
        self._ctypes = None
        self._wintypes = None
        self._file_info_type = None

    def _load_api(self):
        if self._kernel32 is not None:
            return
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateFileW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.c_void_p,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            kernel32.CreateFileW.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            kernel32.GetFinalPathNameByHandleW.argtypes = [
                wintypes.HANDLE,
                wintypes.LPWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
            ]
            kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD

            class _ByHandleFileInformation(ctypes.Structure):
                _fields_ = [
                    ("dwFileAttributes", wintypes.DWORD),
                    ("ftCreationTime", wintypes.FILETIME),
                    ("ftLastAccessTime", wintypes.FILETIME),
                    ("ftLastWriteTime", wintypes.FILETIME),
                    ("dwVolumeSerialNumber", wintypes.DWORD),
                    ("nFileSizeHigh", wintypes.DWORD),
                    ("nFileSizeLow", wintypes.DWORD),
                    ("nNumberOfLinks", wintypes.DWORD),
                    ("nFileIndexHigh", wintypes.DWORD),
                    ("nFileIndexLow", wintypes.DWORD),
                ]

            kernel32.GetFileInformationByHandle.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_ByHandleFileInformation),
            ]
            kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
        except Exception as exc:
            raise CoreError(
                "WINDOWS_DIRECTORY_GUARD_UNAVAILABLE",
                {"message": str(exc)},
            ) from exc
        self._kernel32 = kernel32
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._file_info_type = _ByHandleFileInformation

    @staticmethod
    def _key(path):
        return _windows_normalize_handle_path(path)

    def _final_path(self, handle, path):
        self._load_api()
        size = 512
        while True:
            buffer = self._ctypes.create_unicode_buffer(size)
            length = self._kernel32.GetFinalPathNameByHandleW(handle, buffer, size, 0)
            if not length:
                error = self._ctypes.get_last_error()
                raise CoreError(
                    "WINDOWS_DIRECTORY_GUARD_FAILED",
                    {"path": str(path), "message": self._ctypes.WinError(error).strerror},
                )
            if length < size - 1:
                return buffer.value
            size *= 2
            if size > 32768:
                raise CoreError(
                    "WINDOWS_DIRECTORY_GUARD_FAILED",
                    {"path": str(path), "message": "directory handle final path is too long"},
                )

    def _handle_attributes(self, handle, path):
        self._load_api()
        info = self._file_info_type()
        if not self._kernel32.GetFileInformationByHandle(handle, self._ctypes.byref(info)):
            error = self._ctypes.get_last_error()
            raise CoreError(
                "WINDOWS_DIRECTORY_GUARD_FAILED",
                {"path": str(path), "message": self._ctypes.WinError(error).strerror},
            )
        return int(info.dwFileAttributes)

    def _validate_record(self, record):
        final_path = self._final_path(record["handle"], record["path"])
        attributes = self._handle_attributes(record["handle"], record["path"])
        expected = self._key(record["path"])
        observed = self._key(final_path)
        if observed != expected:
            raise CoreError(
                "WINDOWS_DIRECTORY_HANDLE_MISMATCH",
                {
                    "path": str(record["path"]),
                    "expected": expected,
                    "observed": observed,
                    "message": "directory handle final path changed or resolved elsewhere",
                },
            )
        if not attributes & 0x00000010 or attributes & 0x00000400:
            raise CoreError(
                "REPARSE_PATH_REJECTED",
                {
                    "path": str(record["path"]),
                    "attributes": attributes,
                    "message": "directory handle is not a plain directory",
                },
            )
        record["final_path"] = final_path
        record["attributes"] = attributes

    def add_paths(self, paths):
        self._load_api()
        for path in paths:
            for component in _windows_existing_directory_components(path):
                key = self._key(component)
                if key in self._records:
                    continue
                handle = self._kernel32.CreateFileW(
                    str(component),
                    # Attribute-only directory handles do not reliably block
                    # RemoveDirectory on Windows; LIST_DIRECTORY does.
                    0x00000001,  # FILE_LIST_DIRECTORY
                    0x00000001 | 0x00000002,  # FILE_SHARE_READ | FILE_SHARE_WRITE; no DELETE
                    None,
                    3,  # OPEN_EXISTING
                    0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
                    None,
                )
                invalid = self._ctypes.c_void_p(-1).value
                if not handle or handle == invalid:
                    error = self._ctypes.get_last_error()
                    raise CoreError(
                        "WINDOWS_DIRECTORY_GUARD_FAILED",
                        {"path": str(component), "message": self._ctypes.WinError(error).strerror},
                    )
                record = {"path": component, "handle": handle}
                try:
                    self._validate_record(record)
                except Exception:
                    self._kernel32.CloseHandle(handle)
                    raise
                self._records[key] = record

    def validate(self):
        for record in self._records.values():
            self._validate_record(record)

    def close(self):
        if self._kernel32 is None:
            return
        for record in reversed(list(self._records.values())):
            try:
                self._kernel32.CloseHandle(record["handle"])
            except Exception:
                pass
        self._records.clear()


@contextmanager
def _move_directory_guards(source, destination, root=None):
    """Guard move-path directories on Windows; retain the old path on POSIX."""
    if os.name != "nt":
        yield None
        return
    guards = _WindowsDirectoryGuards()
    try:
        paths = [Path(source).parent, Path(destination).parent]
        if root is not None:
            paths.append(Path(root))
        guards.add_paths(paths)
        guards.validate()
        yield guards
    finally:
        guards.close()


def _move_exact(source, destination, *, root=None):
    """Move one file without replacing an existing destination.

    Staging is created below the selected output root, so a hard-link plus
    unlink is an atomic same-volume no-replace publication primitive on the
    supported local filesystems.  Falling back to ``shutil.move`` here would
    reintroduce the exists()/move() TOCTOU window.
    """
    source = Path(source)
    destination = Path(destination)
    if root is not None:
        _assert_contained_path(source, root, code="PATH_OUTSIDE_ROOT")
        _assert_contained_path(destination.parent, root, code="PATH_OUTSIDE_ROOT")
    _assert_no_reparse_components(source, allow_missing_leaf=False)
    _assert_no_reparse_components(destination.parent, allow_missing_leaf=True)
    with _move_directory_guards(source, destination, root=root) as guards:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if guards is not None:
            # A destination parent may have been created above the initially
            # existing component set.  Its stable ancestor handles cover that
            # mkdir; add and validate the newly existing components before the
            # no-replace publication primitive starts.
            _assert_no_reparse_components(destination.parent, allow_missing_leaf=False)
            if root is not None:
                _assert_contained_path(destination.parent, root, code="PATH_OUTSIDE_ROOT")
            guards.add_paths((destination.parent,))
            guards.validate()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"destination already exists: {destination}")
        try:
            os.link(str(source), str(destination), follow_symlinks=False)
        except FileExistsError:
            raise
        except OSError as exc:
            raise CoreError(
                "ATOMIC_PUBLISH_UNAVAILABLE",
                {"source": str(source), "destination": str(destination), "message": str(exc)},
            ) from exc
        try:
            if root is not None:
                _assert_contained_path(source, root, code="PATH_OUTSIDE_ROOT")
                _assert_contained_path(destination, root, code="PATH_OUTSIDE_ROOT")
            _assert_no_reparse_components(source, allow_missing_leaf=False)
            if guards is not None:
                guards.validate()
            source.unlink()
        except Exception:
            # Do not replace or delete an unexpected destination.  The duplicate
            # link is left for the transaction failure evidence path to handle.
            raise


def _transaction_lock(output_root):
    key = str(Path(output_root).resolve()).casefold()
    with _RUN_TRANSACTION_LOCKS_GUARD:
        lock = _RUN_TRANSACTION_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _RUN_TRANSACTION_LOCKS[key] = lock
    return lock


def _write_json_atomic(path, payload):
    """Atomically retain a JSON-safe failure record in private evidence."""
    return write_run_manifest(canonicalize_json(payload), path)


def _rollback_moves(moves, *, root=None):
    errors = []
    for source, destination in reversed(list(moves)):
        source = Path(source)
        destination = Path(destination)
        if not destination.exists():
            continue
        try:
            _move_exact(destination, source, root=root)
        except Exception as exc:
            errors.append(f"{destination} -> {source}: {exc}")
    return errors


def _remove_empty_directories(root):
    root = Path(root)
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


class RunTransaction:
    """Publish exact generated files with recoverable failure evidence."""

    def __init__(self, output_root, *, config=None, input_identities=None, mode=None, run_id=None):
        if run_id is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_id = f"{stamp}_{uuid.uuid4().hex[:10]}"
        # Validate before constructing any path that incorporates run_id.  The
        # output root itself is independent, but keeping this first also makes
        # invalid requests side-effect free for a not-yet-created root.
        self.run_id = _validate_run_id(run_id)
        requested_root = Path(output_root).expanduser().absolute()
        _assert_no_reparse_components(requested_root, allow_missing_leaf=True)
        self.requested_output_root = requested_root
        self.output_root = requested_root.resolve()
        _assert_no_reparse_components(self.output_root, allow_missing_leaf=True)
        self.config = canonicalize_json(config or {})
        self.input_identities = list(input_identities or [])
        self.mode = mode
        self.staging_dir = None
        self._sealed_manifest = None
        self._sealed_manifest_path = None
        self._failure_record = None
        self._previous_owned = _current_owned_output_paths(self.output_root, mode=self.mode)
        self.preserved_output_paths = _preserved_unowned_output_paths(self.output_root, self._previous_owned)

    @property
    def stage_root(self):
        if self.staging_dir is None:
            raise RuntimeError("run transaction staging has not been created")
        return self.staging_dir

    def create_staging(self):
        self.run_id = _validate_run_id(self.run_id)
        _assert_no_reparse_components(self.output_root, allow_missing_leaf=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        _assert_no_reparse_components(self.output_root, allow_missing_leaf=False)
        candidate = self.output_root / f"{RUN_STAGING_PREFIX}_{self.run_id}"
        _assert_contained_path(candidate, self.output_root, code="INVALID_RUN_ID")
        suffix = 1
        while candidate.exists():
            candidate = self.output_root / f"{RUN_STAGING_PREFIX}_{self.run_id}_{suffix:02d}"
            _assert_contained_path(candidate, self.output_root, code="INVALID_RUN_ID")
            suffix += 1
        _assert_no_reparse_components(candidate, allow_missing_leaf=True)
        candidate.mkdir()
        _assert_no_reparse_components(candidate, allow_missing_leaf=False)
        self.staging_dir = candidate
        return candidate

    def stage_path(self, relative):
        if self.staging_dir is None:
            self.create_staging()
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise CoreError("INVALID_STAGE_PATH", {"path": str(relative)})
        target = self.stage_root / relative
        _assert_contained_path(target.parent, self.stage_root, code="INVALID_STAGE_PATH")
        _assert_no_reparse_components(target.parent, allow_missing_leaf=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        _assert_no_reparse_components(target.parent, allow_missing_leaf=False)
        return target

    def _retain_failure(self, exc):
        self.run_id = _validate_run_id(self.run_id)
        if self.staging_dir is None or not self.staging_dir.exists():
            return None
        failed_root = self.output_root / RUN_FAILED_DIRNAME
        try:
            _assert_contained_path(failed_root, self.output_root, code="FAILED_OUTPUT_OUTSIDE_ROOT")
            failed_root.mkdir(parents=True, exist_ok=True)
            candidate = failed_root / self.run_id
            _assert_contained_path(candidate, failed_root, code="INVALID_RUN_ID")
            suffix = 1
            while candidate.exists():
                candidate = failed_root / f"{self.run_id}_{suffix:02d}"
                _assert_contained_path(candidate, failed_root, code="INVALID_RUN_ID")
                suffix += 1
            candidate.mkdir()
        except Exception:
            # Retention is best effort; never mask the solver/commit exception.
            return None
        try:
            retention_warnings = []
            for source in _iter_files(self.staging_dir):
                destination = candidate / source.relative_to(self.staging_dir)
                try:
                    _move_exact(source, destination, root=self.output_root)
                except Exception:
                    # Never copy over a destination after a no-replace failure:
                    # it may be a user hard-link/symlink or a concurrent file.
                    # Leave this source in the private staging tree and record
                    # the retention warning below.
                    retention_warnings.append({"path": str(source), "message": "could not move failed evidence without replacement"})
                    continue
            for directory in sorted((p for p in self.staging_dir.rglob("*") if p.is_dir()), reverse=True):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            _remove_empty_directories(self.staging_dir)
        except Exception:
            pass
        record = {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "retained_at": datetime.now().isoformat(timespec="seconds"),
            "retention_warnings": retention_warnings if "retention_warnings" in locals() else [{"message": "failed evidence transfer aborted before enumeration"}],
            "staging_path": str(self.staging_dir) if self.staging_dir.exists() else None,
        }
        try:
            _write_json_atomic(candidate / "failure.json", record)
        except Exception:
            pass
        self._failure_record = candidate
        return candidate

    def seal(self, *, status, scientific_ok, manifest_extra=None, required_outputs=None, code_paths=None):
        self.run_id = _validate_run_id(self.run_id)
        if self.staging_dir is None:
            raise RuntimeError("cannot seal a transaction without staging")
        expected_inputs = list(self.input_identities)
        current_inputs = ordered_input_manifest([entry.get("path") for entry in expected_inputs])
        if current_inputs != expected_inputs:
            raise CoreError(
                "INPUT_CHANGED_DURING_RUN",
                {"expected": expected_inputs, "observed": current_inputs, "message": "an input changed after preflight"},
            )
        files = _iter_files(self.staging_dir)
        if any(path.name == RUN_MANIFEST_FILENAME for path in files):
            raise CoreError("MANIFEST_ORDER_ERROR", {"message": "manifest must be written last"})
        extra = dict(manifest_extra or {})
        extra.setdefault("mode", self.mode)
        extra.setdefault("run_id", self.run_id)
        extra.setdefault(
            "preserved_output_paths",
            # Only paths that were never authorized by the previous valid
            # ledger are preserved in the new current bundle.  Ledger-owned
            # paths omitted by this run are archived during ``commit``.
            sorted(set(self.preserved_output_paths)),
        )
        extra.setdefault("transaction", {"staging": True, "run_id": self.run_id})
        manifest = build_run_manifest(
            config=self.config,
            input_paths=[entry.get("path") for entry in self.input_identities],
            outputs=files,
            staged_root=self.staging_dir,
            required_outputs=required_outputs,
            code_paths=code_paths,
            status=status,
            scientific_ok=scientific_ok,
            **extra,
        )
        manifest_path = self.stage_root / RUN_MANIFEST_FILENAME
        write_run_manifest(manifest, manifest_path)
        self._sealed_manifest = manifest
        self._sealed_manifest_path = manifest_path
        return manifest_path, manifest

    def commit(self):
        self.run_id = _validate_run_id(self.run_id)
        if self.staging_dir is None or self._sealed_manifest_path is None:
            raise RuntimeError("transaction must be sealed before commit")
        archived_moves = []
        published_moves = []
        os_lock = _operation_os_lock(self.output_root)
        try:
            os_lock.__enter__()
        except Exception as exc:
            failed = self._retain_failure(exc)
            detail = f"事务锁定失败：{exc}"
            if failed is not None:
                detail += f"；失败证据保留于 {failed}"
            raise RuntimeError(detail) from exc
        lock = _transaction_lock(self.output_root)
        with lock:
            try:
                current_inputs = ordered_input_manifest([entry.get("path") for entry in self.input_identities])
                if current_inputs != self.input_identities:
                    raise CoreError("INPUT_CHANGED_DURING_RUN", {"message": "input changed after manifest sealing"})
                # Resolve ownership again, but never trust an altered ledger.
                # The complete previous inventory is authoritative: an output
                # omitted by the new stage is stale generated data and must be
                # moved to ``_previous_runs`` before publishing the new set.
                previous = _current_owned_output_paths(self.output_root, mode=self.mode)
                snapshot_previous = list(self._previous_owned)
                if {(a, str(b)) for a, b in previous} != {(a, str(b)) for a, b in snapshot_previous}:
                    raise CoreError("OWNERSHIP_CHANGED_DURING_RUN", {"message": "current owned output set changed during the run"})
                archive_dir = self.output_root / RUN_PREVIOUS_DIRNAME / self.run_id
                _assert_contained_path(archive_dir, self.output_root, code="INVALID_RUN_ID")
                if previous:
                    _assert_contained_path(archive_dir, self.output_root, code="ARCHIVE_OUTSIDE_ROOT")
                    archive_dir.mkdir(parents=True, exist_ok=False)
                    for relative, source in previous:
                        destination = archive_dir / relative
                        _assert_contained_path(destination.parent, self.output_root, code="ARCHIVE_OUTSIDE_ROOT")
                        _move_exact(source, destination, root=self.output_root)
                        archived_moves.append((source, destination))
                for source in _iter_files(self.staging_dir):
                    relative = source.relative_to(self.staging_dir)
                    destination = self.output_root / relative
                    _assert_contained_path(destination.parent, self.output_root, code="OUTPUT_OUTSIDE_ROOT")
                    if destination.exists() or destination.is_symlink():
                        raise CoreError("OUTPUT_COLLISION", {"path": relative.as_posix(), "message": "current path is not owned by ezDIC"})
                    _move_exact(source, destination, root=self.output_root)
                    published_moves.append((source, destination))
                _remove_empty_directories(self.staging_dir)
                manifest_path = self.output_root / RUN_MANIFEST_FILENAME
                verification = verify_run_manifest(manifest_path)
                if not verification.get("ok"):
                    raise CoreError("MANIFEST_VERIFY_FAILED", {"verification": verification})
                _write_operation_ledger(self.output_root, self._sealed_manifest, self.mode)
                try:
                    os_lock.__exit__(None, None, None)
                except Exception:
                    # Publication and ledger sealing have already succeeded;
                    # the context manager closes its handle in finally, so an
                    # unlock warning must not roll back a verified run.
                    pass
                return manifest_path
            except Exception as exc:
                rollback_errors = _rollback_moves(published_moves, root=self.output_root)
                rollback_errors.extend(_rollback_moves(archived_moves, root=self.output_root))
                failed = self._retain_failure(exc)
                try:
                    os_lock.__exit__(type(exc), exc, exc.__traceback__)
                except Exception:
                    pass
                message = f"事务提交失败：{exc}"
                if rollback_errors:
                    message += "；回滚失败：" + " | ".join(rollback_errors)
                if failed is not None:
                    message += f"；失败证据保留于 {failed}"
                raise RuntimeError(message) from exc

    def abort(self, exc):
        return self._retain_failure(exc)


def _canonical_settings(settings):
    if not isinstance(settings, Mapping):
        raise CoreError("INVALID_SETTINGS", {"message": "settings must be a mapping"})
    internal = {"_run_token", "_code_paths", "_legacy_direct_processing", "_gui_adapter", "_cli_adapter"}
    canonical_override = settings.get("_canonical_config")
    if canonical_override is not None:
        if not isinstance(canonical_override, Mapping):
            raise CoreError("INVALID_CANONICAL_CONFIG", {"message": "_canonical_config must be a mapping"})
        return canonicalize_json({key: value for key, value in canonical_override.items() if key not in internal and key != "_canonical_config"})
    return canonicalize_json({key: value for key, value in settings.items() if key not in internal and key != "_canonical_config"})


def _adapter_callable(name, fallback, *, allow_adapter=False):
    if not allow_adapter:
        return fallback
    adapter = sys.modules.get("dic_virtual_extensometer_gui_v7_multi_roi_range")
    candidate = getattr(adapter, name, None) if adapter is not None else None
    return candidate if callable(candidate) and candidate is not fallback else fallback


def _notify_progress(callback, fraction, **details):
    if callback is None:
        return
    payload = {"fraction": float(np.clip(fraction, 0.0, 1.0)), **details}
    try:
        callback(payload)
    except TypeError:
        try:
            callback(float(payload["fraction"]), payload)
        except TypeError:
            callback(float(payload["fraction"]))


def _resolve_sequence_inputs(settings, *, require_two=False):
    if "image_paths" in settings and settings.get("image_paths") is not None:
        raw_paths = list(settings.get("image_paths") or [])
    elif settings.get("image_folder"):
        raw_paths = collect_images(settings.get("image_folder"))
    else:
        raw_paths = []
    if not raw_paths:
        raise CoreError("NO_INPUT_IMAGES", {"message": "no image files were selected"})
    paths = [Path(path).expanduser().resolve() for path in raw_paths]
    if len({os.path.normcase(str(path)) for path in paths}) != len(paths):
        raise CoreError("DUPLICATE_INPUT_IMAGES", {"message": "input image paths must be unique"})
    if "start_idx" in settings:
        start_idx = int(settings["start_idx"])
        end_idx = int(settings.get("end_idx", len(paths) - 1))
        start_frame = start_idx + 1
        end_frame = end_idx + 1
    else:
        start_frame = int(settings.get("start_frame_1based", 1))
        end_frame = int(settings.get("end_frame_1based", len(paths)))
        start_idx = start_frame - 1
        end_idx = end_frame - 1
    if start_idx < 0 or end_idx < start_idx or end_idx >= len(paths):
        raise CoreError("INVALID_FRAME_RANGE", {"start_frame_1based": start_frame, "end_frame_1based": end_frame, "count": len(paths)})
    selected_indices = list(range(start_idx, end_idx + 1))
    if require_two and len(selected_indices) < 2:
        raise CoreError("INSUFFICIENT_FRAMES", {"message": "at least a reference and one deformed frame are required"})
    reference_frame = int(settings.get("reference_frame_1based", start_frame))
    if reference_frame < start_frame or reference_frame > end_frame:
        raise CoreError("INVALID_REFERENCE_FRAME", {"reference_frame_1based": reference_frame, "start_frame_1based": start_frame, "end_frame_1based": end_frame})
    reference_idx = reference_frame - 1
    input_paths = [paths[index] for index in selected_indices]
    identities = ordered_input_manifest(input_paths)
    decoded = {}
    reference_shape = None
    for index in selected_indices:
        image = read_gray_image(paths[index])
        decoded[index] = image
        shape = tuple(int(value) for value in image.shape[:2])
        if reference_shape is None:
            reference_shape = shape
        elif shape != reference_shape:
            raise CoreError("IMAGE_DIMENSION_MISMATCH", {"reference_shape": list(reference_shape), "path": str(paths[index]), "shape": list(shape), "message": "all selected images must have identical dimensions"})
    after_identities = ordered_input_manifest(input_paths)
    if [entry.get("sha256") for entry in identities] != [entry.get("sha256") for entry in after_identities]:
        raise CoreError("INPUT_CHANGED_DURING_PREFLIGHT", {"message": "input bytes changed while they were being read"})
    processing_indices = [reference_idx] + [index for index in selected_indices if index != reference_idx]
    return {
        "paths": paths,
        "selected_indices": selected_indices,
        "processing_indices": processing_indices,
        "reference_idx": reference_idx,
        "reference_frame_1based": reference_frame,
        "start_frame_1based": start_frame,
        "end_frame_1based": end_frame,
        "input_paths": input_paths,
        "input_identities": identities,
        "decoded": decoded,
        "shape": reference_shape,
    }


def _normalization_from_settings(reference, settings):
    source = settings.get("normalization") if isinstance(settings.get("normalization"), Mapping) else {}
    if source.get("clip", True) is not True:
        raise CoreError(
            "UNSUPPORTED_NORMALIZATION_CLIP",
            {"clip": source.get("clip"), "message": "ezDIC normalization currently requires clip=true"},
        )
    policy = str(source.get("policy", "reference_percentile"))
    if policy == "fixed_bounds":
        bounds = source.get("bounds")
        if bounds is None and "lo" in source and "hi" in source:
            bounds = (source["lo"], source["hi"])
        if bounds is None:
            raise CoreError("INVALID_NORMALIZATION_POLICY", {"message": "fixed_bounds requires bounds.lo and bounds.hi"})
        lo, hi = _coerce_normalization_bounds(bounds)
        metadata = _normalization_sample_metadata(reference, lo, hi, policy="fixed_bounds")
        metadata.update({"normalization_version": NORMALIZATION_VERSION, "policy": policy, "reference_bounds": {"lo": lo, "hi": hi}, "clip": bool(source.get("clip", True))})
        return metadata
    if policy != "reference_percentile":
        raise CoreError("INVALID_NORMALIZATION_POLICY", {"policy": policy, "message": "unknown normalization policy"})
    try:
        lower = float(source.get("lower_percentile", DEFAULT_NORMALIZATION_LOWER_PERCENTILE))
        upper = float(source.get("upper_percentile", DEFAULT_NORMALIZATION_UPPER_PERCENTILE))
    except (TypeError, ValueError) as exc:
        raise CoreError("INVALID_NORMALIZATION_POLICY", {"message": "percentiles must be numeric"}) from exc
    metadata = compute_reference_normalization(reference, lower_percentile=lower, upper_percentile=upper)
    metadata["clip"] = bool(source.get("clip", True))
    return metadata


def _prepare_roi_groups(settings, reference8):
    groups = list(settings.get("roi_groups") or [])
    if not groups:
        raise CoreError("NO_ROI_GROUPS", {"message": "at least one ROI pair is required"})
    prepared = []
    for index, raw in enumerate(groups):
        if not isinstance(raw, Mapping):
            raise CoreError("INVALID_ROI_GROUP", {"index": index, "message": "ROI group must be an object"})
        group = dict(raw)
        group["name"] = str(group.get("name") or f"ROI{index + 1}")
        try:
            group["roi1"] = tuple(float(value) for value in group["roi1"])
            group["roi2"] = tuple(float(value) for value in group["roi2"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreError("INVALID_ROI_GROUP", {"index": index, "message": str(exc)}) from exc
        selected = group.get("selected_mode", group.get("strain_mode", "auto"))
        group["selected_mode"] = str(selected)
        group["actual_mode"] = str(group.get("actual_mode") or resolve_strain_mode(group["roi1"], group["roi2"], selected))
        if not rect_is_inside_image(group["roi1"], reference8.shape) or not rect_is_inside_image(group["roi2"], reference8.shape):
            raise CoreError("ROI_OUT_OF_BOUNDS", {"group": group["name"], "roi1": group["roi1"], "roi2": group["roi2"]})
        prepared.append(group)
    if len({group["name"] for group in prepared}) != len(prepared):
        raise CoreError("DUPLICATE_ROI_GROUP", {"message": "ROI group names must be unique"})
    _validate_group_output_names(prepared)
    return prepared


def _setting_value(settings, name, *, section=None, aliases=(), default=None):
    if name in settings:
        return settings[name]
    for alias in aliases:
        if alias in settings:
            return settings[alias]
    nested = settings.get(section) if section else None
    if isinstance(nested, Mapping) and name in nested:
        return nested[name]
    return default


def _validate_transaction_policy(settings):
    """The core always runs with the auditable transaction contract enabled."""
    if not isinstance(settings, Mapping):
        raise CoreError("INVALID_SETTINGS", {"message": "settings must be a mapping"})
    sources = [settings]
    canonical = settings.get("_canonical_config") if isinstance(settings, Mapping) else None
    if isinstance(canonical, Mapping):
        sources.append(canonical)
    for source in sources:
        transaction = source.get("transaction") if isinstance(source.get("transaction"), Mapping) else {}
        flat = {
            "enabled": source.get("transaction_enabled", True),
            "archive_previous": source.get("archive_previous", True),
            "retain_failed_staging": source.get("retain_failed_staging", True),
        }
        for key in tuple(flat):
            if key in transaction:
                flat[key] = transaction[key]
        invalid = [key for key, value in flat.items() if value is not True]
        if invalid:
            raise CoreError(
                "INVALID_TRANSACTION_POLICY",
                {"fields": invalid, "message": "ezDIC core requires enabled/archive_previous/retain_failed_staging=true"},
            )


def _validate_fullfield_export_policy(settings):
    """Reject an unimplemented silent parameter-export opt-out."""
    export = settings.get("export") if isinstance(settings.get("export"), Mapping) else {}
    if "write_parameters" in export and export["write_parameters"] is not True:
        raise CoreError(
            "INVALID_EXPORT_POLICY",
            {"field": "export.write_parameters", "message": "fullfield parameter provenance is mandatory when exporting fields"},
        )
    # GUI's historical flat checkbox is retained as a compatibility control;
    # the adapter still records/writes the mandatory parameter sidecar.  Direct
    # callers cannot silently request a missing provenance artifact.
    if not settings.get("_gui_adapter", False) and "export_parameters" in settings and settings["export_parameters"] is not True:
        raise CoreError(
            "INVALID_EXPORT_POLICY",
            {"field": "export_parameters", "message": "fullfield parameter provenance is mandatory"},
        )


def _code_paths_for_settings(settings):
    raw_paths = settings.get("_code_paths", settings.get("code_paths"))
    if raw_paths is not None:
        if not isinstance(raw_paths, (list, tuple)):
            raise CoreError("CODE_FILE_SET_INVALID", {"message": "_code_paths must be a sequence of paths"})
        try:
            names = {Path(value).name.casefold() for value in raw_paths}
        except (TypeError, ValueError) as exc:
            raise CoreError("CODE_FILE_SET_INVALID", {"message": "_code_paths must contain path-like values"}) from exc
        has_schema = "run_config_v1.json" in names
        include_cli = bool(settings.get("_cli_adapter", False)) or "ezdic_cli.py" in names
        include_gui = bool(settings.get("_gui_adapter", False)) or GUI_SOURCE_FILENAME.casefold() in names
        return resolve_code_paths(
            paths=raw_paths,
            include_cli=include_cli,
            include_gui=include_gui,
            include_schema=not has_schema,
        )
    # Direct core callers need the schema in their provenance when it is
    # available, while the low-level default used by standalone manifest
    # helpers intentionally remains core-only for backwards compatibility.
    return resolve_code_paths(include_schema=True)

def _extensometer_options(settings):
    nested = settings.get("export") if isinstance(settings.get("export"), Mapping) else {}
    flat = "export_origin_txt" in settings or "export_engineering_png" in settings
    defaults = {
        "origin_txt": True if flat else False,
        "origin_opju": False,
        "engineering_png": True if flat else False,
        "publication_figures": False,
        "qc_summary": True,
        "full_csv": False if flat else True,
        "corr_plot": False,
        "overlays": False,
        "parameters": False,
    }
    mapping = {
        "origin_txt": ("export_origin_txt", "write_origin_txt"),
        "origin_opju": ("export_origin_opju", "write_origin_opju"),
        "engineering_png": ("export_engineering_png", "write_engineering_png"),
        "publication_figures": ("export_publication_figures", "write_publication_figures"),
        "qc_summary": ("export_qc_summary", "write_qc"),
        "full_csv": ("export_full_csv", "write_full_csv"),
        "corr_plot": ("export_corr_plot", "write_corr_plot", "write_correlation_plots"),
        "overlays": ("export_overlays", "write_overlays"),
        "parameters": ("export_parameters", "write_parameters"),
    }
    result = {}
    for key, aliases in mapping.items():
        value = None
        for alias in aliases:
            if alias in settings:
                value = settings[alias]
                break
        if value is None:
            for alias in aliases:
                if alias in nested:
                    value = nested[alias]
                    break
        result[key] = bool(defaults[key] if value is None else value)
    result["origin_opju_required"] = bool(
        settings.get(
            "origin_opju_required",
            nested.get("origin_opju_required", nested.get("required_for_scientific_gate", False)),
        )
    )
    return result


def _write_tracking_parameters(settings, groups, frame_count, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ezDIC virtual extensometer parameters",
        "======================================",
        f"analysis_mode = {ANALYSIS_MODE_EXTENSOMETER}",
        f"number_of_images_in_analysis_range = {int(frame_count)}",
        f"start_frame_1based = {settings.get('start_frame_1based', int(settings.get('start_idx', 0)) + 1)}",
        f"end_frame_1based = {settings.get('end_frame_1based', int(settings.get('end_idx', frame_count - 1)) + 1)}",
        f"reference_frame_1based = {settings.get('reference_frame_1based', settings.get('start_frame_1based', 1))}",
    ]
    for key in (
        "search_radius_base", "hard_corr", "soft_corr", "enable_adaptive",
        "use_prev_frame_template", "template_alpha", "max_frame_jump",
        "enable_fb_check", "fb_tolerance", "pixel_size_mm", "peak_margin_min", "peak_ratio_min",
    ):
        if key in settings:
            lines.append(f"{key} = {settings[key]}")
    lines.extend(["", "Groups:"])
    for group in groups:
        lines.append(
            f"{group['name']}: role={normalize_roi_role(group.get('role', 'none'))}, "
            f"selected={group.get('selected_mode', 'auto')}, actual={group.get('actual_mode')}, "
            f"roi1={group['roi1']}, roi2={group['roi2']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _export_extensometer_artifacts(df, groups, summary, settings, stage_root, *, input_count, staged_overlays=None):
    # This check is intentionally repeated at the export boundary: callers
    # can construct a pre-normalized group list directly, and no per-group
    # artifact may be opened until its sanitized path is known to be unique.
    _validate_group_output_names(groups)
    options = _extensometer_options(settings)
    use_adapter_hooks = bool(settings.get("_gui_adapter", False))

    def adapter(name, fallback):
        return _adapter_callable(name, fallback, allow_adapter=use_adapter_hooks)

    root = Path(stage_root)
    core_dir = root / "core"
    qc_dir = root / "qc"
    optional_dir = root / "optional"
    written = []
    optional_failures = []
    if options["origin_txt"] or options["engineering_png"] or options["origin_opju"]:
        core_dir.mkdir(parents=True, exist_ok=True)
    if options["publication_figures"]:
        (optional_dir / "publication_figures").mkdir(parents=True, exist_ok=True)
    for group in groups:
        name = group["name"]
        safe = safe_name(name)
        gdf = df[df["group"] == name].copy()
        if options["origin_txt"]:
            path = core_dir / f"strain_{safe}.txt"
            adapter("write_origin_txt", write_origin_txt)(gdf, path)
            written.append(path)
        if options["engineering_png"]:
            path = core_dir / f"engineering_strain_{safe}.png"
            adapter("plot_engineering_strain", plot_engineering_strain)(gdf, path, f"Engineering strain - {name}")
            written.append(path)
        if options["publication_figures"]:
            for path in publication_figure_paths(optional_dir / "publication_figures", f"engineering_strain_{safe}"):
                adapter("plot_engineering_strain", plot_engineering_strain)(gdf, path, f"Engineering strain - {name}", preset_name="publication")
                written.append(path)
    if options["origin_txt"]:
        path = core_dir / "strain_all_groups.txt"
        adapter("write_all_groups_origin_txt", write_all_groups_origin_txt)(df, path, groups)
        written.append(path)
        path = core_dir / "strain_mean_groups.txt"
        adapter("write_mean_groups_origin_txt", write_mean_groups_origin_txt)(df, groups, path)
        written.append(path)
        if poisson_roles_are_configured(groups):
            path = core_dir / "poisson_ratio.txt"
            adapter("write_poisson_ratio_txt", write_poisson_ratio_txt)(df, groups, path)
            written.append(path)
    if options["engineering_png"]:
        path = core_dir / "engineering_strain_all_groups.png"
        adapter("plot_all_groups_engineering_strain", plot_all_groups_engineering_strain)(df, groups, path)
        written.append(path)
        if poisson_roles_are_configured(groups):
            path = core_dir / "poisson_ratio.png"
            adapter("plot_poisson_ratio", plot_poisson_ratio)(df, groups, path)
            written.append(path)
    if options["publication_figures"]:
        for path in publication_figure_paths(optional_dir / "publication_figures", "engineering_strain_all_groups"):
            adapter("plot_all_groups_engineering_strain", plot_all_groups_engineering_strain)(df, groups, path, preset_name="publication")
            written.append(path)
        if poisson_roles_are_configured(groups):
            for path in publication_figure_paths(optional_dir / "publication_figures", "poisson_ratio"):
                adapter("plot_poisson_ratio", plot_poisson_ratio)(df, groups, path, preset_name="publication")
                written.append(path)
    if options["origin_opju"]:
        path = core_dir / ORIGIN_OPJU_FILENAME
        try:
            adapter("write_origin_opju_project", write_origin_opju_project)(df, groups, path)
            written.append(path)
        except Exception as exc:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            optional_failures.append({"artifact": _portable_relative(path, root), "code": "OPTIONAL_ORIGIN_EXPORT_FAILED", "message": str(exc), "requested": True, "required_for_scientific_gate": bool(options["origin_opju_required"])})
    if options["qc_summary"]:
        path = qc_dir / "qc_summary.txt"
        adapter("write_qc_summary", write_qc_summary)(summary, path)
        written.append(path)
    if options["full_csv"]:
        full_csv_dir = optional_dir / "full_csv"
        full_csv_dir.mkdir(parents=True, exist_ok=True)
        path = full_csv_dir / "strain_results_all_groups.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
        per_group = full_csv_dir / "per_group_results"
        per_group.mkdir(parents=True, exist_ok=True)
        for group in groups:
            path = per_group / f"strain_results_{safe_name(group['name'])}.csv"
            df[df["group"] == group["name"]].to_csv(path, index=False, encoding="utf-8-sig")
            written.append(path)
    if options["corr_plot"]:
        corr_dir = optional_dir / "correlation_plots"
        corr_dir.mkdir(parents=True, exist_ok=True)
        hard_corr = float(_setting_value(settings, "hard_corr", section="tracking", default=0.55))
        soft_corr = float(_setting_value(settings, "soft_corr", section="tracking", default=0.35))
        for group in groups:
            name = group["name"]
            path = corr_dir / f"correlation_scores_{safe_name(name)}.png"
            adapter("plot_correlation_scores", plot_correlation_scores)(df[df["group"] == name].copy(), path, name, hard_corr, soft_corr, preset_name="raw_inspection")
            written.append(path)
            if options["publication_figures"]:
                for public_path in publication_figure_paths(optional_dir / "publication_figures", f"correlation_scores_{safe_name(name)}"):
                    adapter("plot_correlation_scores", plot_correlation_scores)(df[df["group"] == name].copy(), public_path, name, hard_corr, soft_corr, preset_name="publication")
                    written.append(public_path)
    if options["parameters"]:
        parameter_dir = optional_dir / "parameters"
        path = _write_tracking_parameters(settings, groups, input_count, parameter_dir / "tracking_parameters.txt")
        written.append(path)
        path = parameter_dir / "acceptance_summary.txt"
        path.write_text(
            "Acceptance summary by group\n---------------------------\n" + "\n".join(
                f"\n[{name}]\n" + str(df[df["group"] == name]["accept_mode"].value_counts(dropna=False))
                for name in [group["name"] for group in groups]
            ) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        written.append(path)
    if options["overlays"] and staged_overlays:
        every = max(1, int(_setting_value(settings, "overlay_every", section="tracking", default=5)))
        for group in groups:
            name = group["name"]
            gdir = optional_dir / "overlays" / safe_name(name)
            gdir.mkdir(parents=True, exist_ok=True)
            for frame_number, image, overlay_info in staged_overlays.get(name, []):
                if frame_number % every != 0 and frame_number != input_count - 1 and overlay_info.get("accepted", False) and overlay_info.get("accept_mode") != "adaptive":
                    continue
                overlay = draw_group_overlay(
                    image, name, group.get("actual_mode", "auto"),
                    overlay_info["used_rect1"], overlay_info["used_rect2"],
                    overlay_info.get("candidate_rect1"), overlay_info.get("candidate_rect2"),
                    frame_number, overlay_info.get("strain", np.nan), overlay_info.get("last_valid_strain", np.nan),
                    overlay_info.get("score1", np.nan), overlay_info.get("score2", np.nan),
                    overlay_info.get("accepted", False), overlay_info.get("accept_mode", "rejected"),
                    overlay_info.get("reason", ""), fb_err1=overlay_info.get("fb_err1"), fb_err2=overlay_info.get("fb_err2"),
                )
                path = gdir / f"tracked_{frame_number:05d}.png"
                write_image_checked(path, overlay)
                written.append(path)
    missing = [str(path) for path in written if not Path(path).is_file()]
    if missing:
        raise CoreError("OUTPUT_MISSING", {"missing": missing, "message": "exporter reported an output that is not present"})
    return written, optional_failures, options


def _build_1d_gate(df, groups, settings, optional_failures=None):
    min_frames = int(_setting_value(settings, "min_valid_frames", section="quality", aliases=("min_strain_points",), default=1))
    min_ratio = float(_setting_value(settings, "min_strain_valid_ratio", section="quality", aliases=("min_strain_valid_fraction",), default=0.0))
    reference_frame = int(settings.get("reference_frame_1based", settings.get("start_frame_1based", int(settings.get("start_idx", 0)) + 1)))
    details = {}
    reasons = []
    for group in groups:
        # The reference row is a registration baseline, not an independent
        # deformation measurement.  It must never satisfy a minimum-valid
        # deformation-frame gate by itself.
        sub = df[(df["group"] == group["name"]) & (df["frame_global_1based"] != reference_frame)]
        accepted = sub.get("accepted", pd.Series(dtype=bool)).fillna(False).astype(bool)
        strain_valid = _strain_valid_mask(sub)
        count = len(sub)
        valid_count = int(accepted.sum())
        strain_count = int(strain_valid.sum())
        ratio = float(strain_count / count) if count else 0.0
        passed = bool(valid_count >= min_frames and strain_count >= min_frames and ratio >= min_ratio)
        if not passed:
            reasons.append(f"{group['name']}:validity_gate")
        details[group["name"]] = {"frames": count, "valid_frames": valid_count, "strain_valid_frames": strain_count, "strain_valid_ratio": ratio, "passed": passed}
    required_optional_failures = [item for item in (optional_failures or []) if item.get("required_for_scientific_gate")]
    if required_optional_failures:
        reasons.append("required_optional_export_failed")
    return {"scientific_ok": not reasons, "reasons": reasons, "groups": details, "required_optional_failures": required_optional_failures, "thresholds": {"min_valid_frames": min_frames, "min_strain_valid_ratio": min_ratio}}


def _texture_preflight_record(groups, texture_thresholds, *, states=None, field_roi=None, field_metrics=None, field_code=None):
    """Build the immutable texture evidence carried by a run manifest."""
    thresholds = dict(texture_thresholds or {})
    record = {
        "version": TEXTURE_PREFLIGHT_VERSION,
        "metrics_version": TEXTURE_METRICS_VERSION,
        "discriminator_version": TEXTURE_DISCRIMINATOR_VERSION,
        "effective_thresholds": {
            key: thresholds[key]
            for key in (
                "min_texture_std",
                "min_texture_contrast",
                "max_saturated_frac",
                "min_structure_ratio",
                "max_directional_coherence",
                "min_periodicity_score",
            )
            if key in thresholds
        },
        # Keep flat keys for consumers of the v0.2 manifest while the nested
        # object is the canonical replay record.
        **thresholds,
    }
    if groups is not None:
        record["groups"] = {}
        for index, group in enumerate(groups):
            name = str(group.get("name", ""))
            state = states[index] if states is not None and index < len(states) else None
            if state is None:
                state = group.get("_texture_state") if isinstance(group, Mapping) else None
            metrics = group.get("_texture_metrics") if isinstance(group, Mapping) else None
            if state is not None:
                metrics = {
                    "roi1": state.get("texture_metrics1", {}),
                    "roi2": state.get("texture_metrics2", {}),
                }
            if metrics is None:
                metrics = {"roi1": {}, "roi2": {}}
            group_record = dict(metrics)
            # Coordinates and decision codes make the measured values
            # unambiguous when a group is edited/replayed later; keep the
            # metric dictionaries at ``roi1``/``roi2`` for compatibility.
            group_record.update(
                {
                    "roi1_rect": group.get("roi1"),
                    "roi2_rect": group.get("roi2"),
                    "roi1_code": state.get("texture_code1") if state is not None else None,
                    "roi2_code": state.get("texture_code2") if state is not None else None,
                }
            )
            record["groups"][name] = canonicalize_json(group_record)
    if field_metrics is not None:
        record["field_roi"] = {
            "roi": canonicalize_json(field_roi),
            "metrics": canonicalize_json(field_metrics),
            "code": field_code,
        }
    return canonicalize_json(record)


def run_extensometer_sequence(settings, progress_callback=None):
    """Run a fixed-reference 1-D virtual extensometer without Tk or dialogs."""
    _validate_transaction_policy(settings)
    snapshot = _canonical_settings(settings)
    inputs = _resolve_sequence_inputs(settings, require_two=False)
    if inputs["reference_idx"] != inputs["selected_indices"][0]:
        raise CoreError("UNSUPPORTED_REFERENCE_ORDER", {"reference_frame_1based": inputs["reference_frame_1based"], "start_frame_1based": inputs["start_frame_1based"], "message": "1-D tracking currently requires reference_frame_1based == start_frame_1based"})
    reference_raw = inputs["decoded"][inputs["reference_idx"]]
    normalization = _normalization_from_settings(reference_raw, settings)
    reference8 = normalize_with_bounds(reference_raw, normalization)
    groups = _prepare_roi_groups(settings, reference8)
    min_structure_ratio = float(_setting_value(settings, "min_structure_ratio", section="texture", default=DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO))
    max_directional_coherence = float(_setting_value(settings, "max_directional_coherence", section="texture", default=DEFAULT_TEXTURE_MAX_DIRECTIONAL_COHERENCE))
    min_periodicity_score = float(_setting_value(settings, "min_periodicity_score", section="texture", default=DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE))
    texture_settings = {
        "preflight_version": TEXTURE_PREFLIGHT_VERSION,
        "metrics_version": TEXTURE_METRICS_VERSION,
        "discriminator_version": TEXTURE_DISCRIMINATOR_VERSION,
        "min_texture_std": float(_setting_value(settings, "min_texture_std", section="texture", default=8.0)),
        "min_texture_contrast": float(_setting_value(settings, "min_texture_contrast", section="texture", default=25.0)),
        "max_saturated_frac": float(_setting_value(settings, "max_saturated_frac", section="texture", default=0.20)),
        "min_structure_ratio": min_structure_ratio,
        "max_directional_coherence": max_directional_coherence,
        "min_periodicity_score": min_periodicity_score,
    }
    states = [
        initialize_extensometer_group_state(
            reference8,
            group,
            min_texture_std=texture_settings["min_texture_std"],
            min_texture_contrast=texture_settings["min_texture_contrast"],
            max_saturated_frac=texture_settings["max_saturated_frac"],
            min_structure_ratio=texture_settings["min_structure_ratio"],
            max_directional_coherence=texture_settings["max_directional_coherence"],
            min_periodicity_score=texture_settings["min_periodicity_score"],
        )
        for group in groups
    ]
    texture_preflight = _texture_preflight_record(groups, texture_settings, states=states)
    transaction = RunTransaction(settings.get("output_dir"), config=snapshot, input_identities=inputs["input_identities"], mode=ANALYSIS_MODE_EXTENSOMETER)
    transaction.create_staging()
    params = {
        "search_radius_base": int(_setting_value(settings, "search_radius_base", section="tracking", aliases=("search_radius_px",), default=180)),
        "hard_corr": float(settings.get("hard_corr", (settings.get("quality") or {}).get("zncc_min", 0.55))),
        "soft_corr": float(_setting_value(settings, "soft_corr", section="tracking", default=0.35)),
        "enable_adaptive": bool(_setting_value(settings, "enable_adaptive", section="tracking", default=True)),
        "use_prev_frame_template": bool(_setting_value(settings, "use_prev_frame_template", section="tracking", default=False)),
        "template_alpha": float(_setting_value(settings, "template_alpha", section="tracking", default=0.70)),
        "max_frame_jump": _setting_value(settings, "max_frame_jump", section="tracking", default=None),
        "enable_fb_check": bool(_setting_value(settings, "enable_fb_check", section="quality", default=True)),
        "fb_tolerance": float(_setting_value(settings, "fb_tolerance", section="quality", aliases=("fb_tolerance_px",), default=12.0)),
        "pixel_size_mm": _setting_value(settings, "pixel_size_mm", section="tracking", default=None),
        "peak_margin_min": float(_setting_value(settings, "peak_margin_min", section="quality", aliases=("second_peak_margin_min",), default=DEFAULT_PEAK_MARGIN_MIN)),
        "peak_ratio_min": float(_setting_value(settings, "peak_ratio_min", section="quality", default=DEFAULT_PEAK_RATIO_MIN)),
    }
    all_rows = []
    staged_overlays = {group["name"]: [] for group in groups}
    try:
        for local_index, global_index in enumerate(inputs["processing_indices"]):
            image8 = normalize_with_bounds(inputs["decoded"][global_index], normalization)
            filename = Path(inputs["paths"][global_index]).name
            for state in states:
                row, overlay_info = track_extensometer_group_frame(state, image8, local_index, filename, params)
                row["frame_local_1based"] = local_index + 1
                row["frame_global_1based"] = global_index + 1
                all_rows.append(row)
                staged_overlays[state["group"]["name"]].append((local_index, image8, overlay_info))
            _notify_progress(progress_callback, (local_index + 1) / max(len(inputs["processing_indices"]), 1), frame_global_1based=global_index + 1, mode=ANALYSIS_MODE_EXTENSOMETER)
        df = pd.DataFrame(all_rows)
        summary = build_qc_summary(df)
        written, optional_failures, options = _export_extensometer_artifacts(df, groups, summary, settings, transaction.stage_root, input_count=len(inputs["processing_indices"]), staged_overlays=staged_overlays)
        gate = _build_1d_gate(df, groups, settings, optional_failures)
        status = "completed" if gate["scientific_ok"] and not optional_failures else "completed_with_warnings"
        invalid_rows = [
            {"frame_global_1based": int(row["frame_global_1based"]), "group": row["group"], "reason": row.get("invalid_reason", "")}
            for row in all_rows if not bool(row.get("strain_valid", False))
        ]
        manifest_extra = {
            "analysis_mode": ANALYSIS_MODE_EXTENSOMETER,
            "input_metadata": [
                {"path": identity["path"], "shape": [int(value) for value in inputs["decoded"][index].shape], "dtype": str(inputs["decoded"][index].dtype)}
                for identity, index in zip(inputs["input_identities"], inputs["selected_indices"])
            ],
            "processing_order_frame_1based": [index + 1 for index in inputs["processing_indices"]],
            "reference_frame_1based": inputs["reference_frame_1based"],
            "reference_filename": Path(inputs["paths"][inputs["reference_idx"]]).name,
            "normalization": normalization,
            "texture_preflight": texture_preflight,
            "scientific_gate": gate,
            "invalid_frames": invalid_rows,
            "warnings": (["invalid_tracking_rows_present"] if invalid_rows else []) + [item["message"] for item in optional_failures],
            "optional_failures": optional_failures,
            "output_policy": options,
            "summary": summary,
        }
        required_outputs = [_portable_relative(path, transaction.stage_root) for path in written]
        if not gate["scientific_ok"]:
            manifest_extra["warnings"].append("scientific_gate_failed")
            _, failed_manifest = transaction.seal(status="scientific_gate_failed", scientific_ok=False, manifest_extra=manifest_extra, required_outputs=required_outputs, code_paths=_code_paths_for_settings(settings))
            failed_dir = transaction.abort(CoreError("SCIENTIFIC_GATE_FAILED", {"gate": gate, "message": "scientific validity gate did not pass"}))
            manifest_path = (failed_dir / RUN_MANIFEST_FILENAME) if failed_dir is not None else failed_manifest
            failed_verification = verify_run_manifest(manifest_path)
            return {
                "status": "scientific_gate_failed",
                "scientific_ok": False,
                "integrity_ok": bool(failed_verification.get("ok")),
                "manifest_path": str(manifest_path),
                "outputs": [],
                "warnings": manifest_extra["warnings"],
                "errors": gate["reasons"],
                "dataframe": df,
                "summary": summary,
                "json_summary": {"status": "scientific_gate_failed", "scientific_ok": False, "integrity_ok": bool(failed_verification.get("ok")), "mode": ANALYSIS_MODE_EXTENSOMETER, "manifest_path": str(manifest_path), "gate": gate},
                "manifest": json.loads(Path(manifest_path).read_text(encoding="utf-8")) if Path(manifest_path).is_file() else None,
            }
        _, manifest = transaction.seal(status=status, scientific_ok=True, manifest_extra=manifest_extra, required_outputs=required_outputs, code_paths=_code_paths_for_settings(settings))
        manifest_path = transaction.commit()
        return {
            "status": status,
            "scientific_ok": bool(gate["scientific_ok"]),
            "integrity_ok": True,
            "manifest_path": str(manifest_path),
            "outputs": [str(transaction.output_root / entry["path"]) for entry in manifest.get("outputs", [])],
            "warnings": manifest_extra["warnings"],
            "errors": [],
            "dataframe": df,
            "summary": summary,
            "json_summary": {"status": status, "scientific_ok": bool(gate["scientific_ok"]), "mode": ANALYSIS_MODE_EXTENSOMETER, "manifest_path": str(manifest_path), "gate": gate},
            "manifest": json.loads(Path(manifest_path).read_text(encoding="utf-8")),
        }
    except Exception as exc:
        transaction.abort(exc)
        raise


def _fullfield_settings(settings):
    solver = settings.get("solver") if isinstance(settings.get("solver"), Mapping) else {}
    quality = settings.get("quality") if isinstance(settings.get("quality"), Mapping) else {}
    pyramid = settings.get("pyramid") if isinstance(settings.get("pyramid"), Mapping) else {}
    ratio_max = quality.get("second_peak_ratio_max")
    if ratio_max is not None:
        try:
            ratio_min = 1.0 / float(ratio_max) if float(ratio_max) > 0 else float(np.finfo(float).max)
        except (TypeError, ValueError, ZeroDivisionError):
            ratio_min = DEFAULT_PEAK_RATIO_MIN
    else:
        ratio_min = DEFAULT_PEAK_RATIO_MIN
    return {
        "subset_size": int(_setting_value(settings, "subset_size", aliases=("dic_subset_size",), section="solver", default=solver.get("subset_size_px", 21))),
        "step": int(_setting_value(settings, "step", aliases=("dic_step",), section="solver", default=solver.get("step_px", 5))),
        "strain_window": int(_setting_value(settings, "strain_window", aliases=("dic_strain_window",), section="solver", default=solver.get("strain_window_px", 5))),
        "smooth_sigma": float(_setting_value(settings, "smooth_sigma", aliases=("dic_smooth_sigma",), section="solver", default=solver.get("smooth_sigma_poi", 0.0))),
        "strain_degree": int(_setting_value(settings, "strain_degree", section="solver", default=2)),
        "robust_strain": bool(_setting_value(settings, "robust_strain", section="solver", default=True)),
        "outlier_threshold_px": float(_setting_value(settings, "outlier_threshold_px", section="solver", default=1.0)),
        "initialization": str(_setting_value(settings, "initialization", section="solver", default="auto")),
        "search_radius": int(_setting_value(settings, "search_radius", aliases=("dic_search_radius",), section="solver", default=solver.get("search_radius_px", 20))),
        "solver": str(_setting_value(settings, "solver_name", aliases=("dic_solver",), section="solver", default=solver.get("name", DIC_SOLVER_ICGN))),
        "max_iter": int(_setting_value(settings, "max_iter", section="solver", default=solver.get("max_iterations", 25))),
        "conv_tol": float(_setting_value(settings, "conv_tol", section="solver", default=solver.get("tolerance", 1e-3))),
        "zncc_min": float(_setting_value(settings, "zncc_min", aliases=("dic_zncc_min",), section="quality", default=quality.get("zncc_min", 0.75))),
        "peak_margin_min": float(_setting_value(settings, "peak_margin_min", section="quality", aliases=("second_peak_margin_min",), default=quality.get("second_peak_margin_min", DEFAULT_PEAK_MARGIN_MIN))),
        "peak_ratio_min": float(_setting_value(settings, "peak_ratio_min", section="quality", default=ratio_min)),
        "max_condition_number": float(_setting_value(settings, "max_condition_number", section="quality", default=quality.get("max_condition_number", DEFAULT_MAX_HESSIAN_CONDITION_NUMBER))),
        "max_residual_rms": float(_setting_value(settings, "max_residual_rms", section="quality", default=quality.get("max_residual_rms", float("inf")))),
        "reject_nonconverged": bool(_setting_value(settings, "reject_nonconverged", section="quality", default=False)),
        "pyramid_levels": int(_setting_value(settings, "pyramid_levels", aliases=("levels", "dic_pyramid_levels"), section="pyramid", default=pyramid.get("levels", 1))),
        "pyramid_scale": float(_setting_value(settings, "pyramid_scale", aliases=("scale", "dic_pyramid_scale"), section="pyramid", default=pyramid.get("scale", 0.5))),
        "min_correlation_valid_fraction": float(_setting_value(settings, "min_correlation_valid_fraction", section="quality", default=DEFAULT_MIN_CORRELATION_VALID_FRACTION)),
        "min_strain_valid_fraction": float(_setting_value(settings, "min_strain_valid_fraction", aliases=("min_strain_valid_ratio",), section="quality", default=DEFAULT_MIN_STRAIN_VALID_FRACTION)),
    }


def _fullfield_export_options(settings):
    source = settings.get("export") if isinstance(settings.get("export"), Mapping) else {}
    requested_parameters = bool(settings.get("export_parameters", source.get("write_parameters", True)))
    return {
        "overlays": bool(settings.get("export_overlays", source.get("write_overlays", False))),
        "parameters": True,
        "parameters_requested": requested_parameters,
        "parameters_mandatory": True,
    }


def run_fullfield_sequence(settings, progress_callback=None):
    """Run fixed-reference local-subset 2-D DIC without Tk or dialogs."""
    _validate_transaction_policy(settings)
    _validate_fullfield_export_policy(settings)
    snapshot = _canonical_settings(settings)
    inputs = _resolve_sequence_inputs(settings, require_two=True)
    reference_raw = inputs["decoded"][inputs["reference_idx"]]
    normalization = _normalization_from_settings(reference_raw, settings)
    reference8 = normalize_with_bounds(reference_raw, normalization)
    reference_float = normalize_dic_float(reference_raw, normalization)
    roi = settings.get("field_roi")
    if roi is None:
        raise CoreError("NO_FIELD_ROI", {"message": "fullfield mode requires field_roi"})
    try:
        roi = tuple(float(value) for value in roi)
    except (TypeError, ValueError) as exc:
        raise CoreError("INVALID_FIELD_ROI", {"message": str(exc)}) from exc
    if not rect_is_inside_image(roi, reference8.shape):
        raise CoreError("ROI_OUT_OF_BOUNDS", {"field_roi": roi})
    specimen_mask, mask_record, mask_identity = _resolve_specimen_mask(reference8, roi, settings)
    input_identities = inputs["input_identities"] + ([mask_identity] if mask_identity is not None else [])
    min_structure_ratio = float(_setting_value(settings, "min_structure_ratio", section="texture", default=DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO))
    max_directional_coherence = float(_setting_value(settings, "max_directional_coherence", section="texture", default=DEFAULT_TEXTURE_MAX_DIRECTIONAL_COHERENCE))
    min_periodicity_score = float(_setting_value(settings, "min_periodicity_score", section="texture", default=DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE))
    texture_settings = {
        "preflight_version": TEXTURE_PREFLIGHT_VERSION,
        "metrics_version": TEXTURE_METRICS_VERSION,
        "discriminator_version": TEXTURE_DISCRIMINATOR_VERSION,
        "min_texture_std": float(_setting_value(settings, "min_texture_std", section="texture", default=8.0)),
        "min_texture_contrast": float(_setting_value(settings, "min_texture_contrast", section="texture", default=25.0)),
        "max_saturated_frac": float(_setting_value(settings, "max_saturated_frac", section="texture", default=0.20)),
        "min_structure_ratio": min_structure_ratio,
        "max_directional_coherence": max_directional_coherence,
        "min_periodicity_score": min_periodicity_score,
    }
    field_texture = roi_texture_metrics(reference8, roi)
    field_texture_code = texture_failure_code(
        field_texture,
        texture_settings["min_texture_std"],
        texture_settings["min_texture_contrast"],
        texture_settings["max_saturated_frac"],
        texture_settings["min_structure_ratio"],
        texture_settings["max_directional_coherence"],
        texture_settings["min_periodicity_score"],
    )
    if field_texture_code is not None and (
        not settings.get("_allow_low_texture_preflight", False)
        or field_texture_code not in {"LOW_TEXTURE", "SATURATED_TEXTURE"}
    ):
        raise CoreError(
            field_texture_code,
            {
                "roi": roi,
                "metrics": field_texture,
                "texture_preflight": _texture_preflight_record(
                    None,
                    texture_settings,
                    field_roi=roi,
                    field_metrics=field_texture,
                    field_code=field_texture_code,
                ),
                "message": "field ROI texture does not meet the configured texture contract",
            },
        )
    solver = _fullfield_settings(settings)
    if solver["subset_size"] < 9 or solver["step"] < 1 or solver["strain_window"] < 3 or solver["search_radius"] < 1:
        raise CoreError("INVALID_DIC_SETTINGS", {"settings": solver})
    X, Y = build_poi_grid(roi, _odd_subset_size(solver["subset_size"]), solver["step"], reference8.shape)
    if not poi_grid_is_usable(X, Y, min_rows=3, min_cols=3):
        raise CoreError("UNUSABLE_POI_GRID", {"message": "current field ROI must produce at least a 3x3 POI grid"})
    transaction = RunTransaction(settings.get("output_dir"), config=snapshot, input_identities=input_identities, mode=ANALYSIS_MODE_FULLFIELD)
    transaction.create_staging()
    output_options = _fullfield_export_options(settings)
    fields = []
    frame_records = []
    last_field = None
    last_image = reference8
    try:
        use_adapter_hooks = bool(settings.get("_gui_adapter", False))
        if specimen_mask is not None:
            write_image_checked(transaction.stage_root/"dic"/"specimen_mask.png", specimen_mask.astype(np.uint8)*255)
        run_dic = _adapter_callable("run_2d_dic", run_2d_dic, allow_adapter=use_adapter_hooks)
        export_dic = _adapter_callable("export_dic_field_outputs", export_dic_field_outputs, allow_adapter=use_adapter_hooks)
        for local_index, global_index in enumerate(inputs["processing_indices"]):
            if global_index == inputs["reference_idx"]:
                continue
            image8 = normalize_with_bounds(inputs["decoded"][global_index], normalization)
            image_float = normalize_dic_float(inputs["decoded"][global_index], normalization)
            try:
                field = run_dic(
                    reference_float, image_float, roi,
                    subset_size=solver["subset_size"], step=solver["step"], solver=solver["solver"],
                    search_radius=solver["search_radius"], max_iter=solver["max_iter"], conv_tol=solver["conv_tol"],
                    zncc_min=solver["zncc_min"], strain_window=solver["strain_window"], smooth_sigma=solver["smooth_sigma"],
                    peak_margin_min=solver["peak_margin_min"], peak_ratio_min=solver["peak_ratio_min"],
                    max_condition_number=solver["max_condition_number"], reject_nonconverged=solver["reject_nonconverged"],
                    max_residual_rms=solver["max_residual_rms"],
                    min_correlation_valid_fraction=solver["min_correlation_valid_fraction"],
                    min_strain_valid_fraction=solver["min_strain_valid_fraction"],
                    pyramid_levels=solver["pyramid_levels"], pyramid_scale=solver["pyramid_scale"],
                    specimen_mask=specimen_mask, strain_degree=solver["strain_degree"],
                    robust_strain=solver["robust_strain"], outlier_threshold_px=solver["outlier_threshold_px"],
                    initialization=solver["initialization"],
                )
            except Exception as exc:
                raise RuntimeError(f"fullfield solver failed at frame {global_index + 1}: {exc}") from exc
            field = dict(field)
            field["mask_provenance"] = mask_record
            field["reference_image"], field["deformed_image"] = reference8, image8
            field["display_options"] = dict(settings.get("display") or {})
            field["provenance"] = {
                "analysis_mode": ANALYSIS_MODE_FULLFIELD,
                "reference_frame_1based": inputs["reference_frame_1based"],
                "reference_filename": Path(inputs["paths"][inputs["reference_idx"]]).name,
                "frame_global_1based": global_index + 1,
                "frame_filename": Path(inputs["paths"][global_index]).name,
                "field_roi": tuple(int(round(value)) for value in roi),
                "normalization": normalization,
                "correlation_intensity": "float64 reference-fixed scaling; display uses uint8",
                "mask": mask_record,
            }
            field["reference_frame_1based"] = inputs["reference_frame_1based"]
            field["reference_filename"] = Path(inputs["paths"][inputs["reference_idx"]]).name
            field["frame_global_1based"] = global_index + 1
            field["frame_filename"] = Path(inputs["paths"][global_index]).name
            if "strain_valid" not in field:
                try:
                    valid_values = np.asarray(field.get("valid", []), dtype=bool).ravel()
                    finite_values = (
                        np.isfinite(np.asarray(field["Exx"], dtype=float).ravel())
                        & np.isfinite(np.asarray(field["Eyy"], dtype=float).ravel())
                        & np.isfinite(np.asarray(field["Exy"], dtype=float).ravel())
                    )
                    if valid_values.size == finite_values.size:
                        field["strain_valid"] = valid_values & finite_values
                except (KeyError, TypeError, ValueError):
                    pass
            quality_summary = field.get("quality_summary") or field_quality_summary(
                field,
                min_correlation_valid_fraction=solver["min_correlation_valid_fraction"],
                min_strain_valid_fraction=solver["min_strain_valid_fraction"],
                max_residual_rms=solver["max_residual_rms"],
            )
            field["quality_summary"] = quality_summary
            frame_record = {
                "frame_global_1based": global_index + 1,
                "filename": Path(inputs["paths"][global_index]).name,
                "correlation_valid_fraction": quality_summary.get("correlation_valid_fraction"),
                "strain_valid_fraction": quality_summary.get("strain_valid_fraction"),
                "scientific_ok": bool(quality_summary.get("scientific_ok", False)),
                "invalid_reason_histogram": quality_summary.get("invalid_reason_histogram", {}),
            }
            computed = fullfield_field_has_finite_strain(field)
            scientific_valid = bool(
                computed
                and quality_summary.get("scientific_ok", False)
                and float(quality_summary.get("correlation_valid_fraction", 0.0)) >= solver["min_correlation_valid_fraction"]
                and float(quality_summary.get("strain_valid_fraction", 0.0)) >= solver["min_strain_valid_fraction"]
            )
            if computed:
                stem = f"frame_{global_index + 1:04d}"
                exported = export_dic(field, transaction.stage_root / "dic", stem=stem)
                if not use_adapter_hooks:
                    expected = [
                        transaction.stage_root / "dic" / f"{stem}.txt",
                        transaction.stage_root / "dic" / f"{stem}.csv",
                        transaction.stage_root / "dic" / f"{stem}.npz",
                        transaction.stage_root / "dic" / f"{stem}_parameters.txt",
                        *[transaction.stage_root / "dic" / f"{stem}_{DIC_COMPONENT_FILE_SUFFIXES[component]}.png" for component in DIC_FIELD_COMPONENTS],
                    ]
                    missing = [str(path) for path in expected if not path.is_file()]
                    if missing:
                        raise CoreError("OUTPUT_MISSING", {"missing": missing, "message": "fullfield exporter omitted a requested artifact"})
                elif isinstance(exported, Mapping):
                    reported = []
                    for value in exported.values():
                        reported.extend(value if isinstance(value, (list, tuple)) else [value])
                    missing = [str(path) for path in reported if path is not None and not Path(path).is_file()]
                    if missing:
                        raise CoreError("OUTPUT_MISSING", {"missing": missing, "message": "fullfield exporter reported a missing artifact"})
                if output_options["overlays"]:
                    overlay = overlay_dic_field_on_image(image8, field, component="Exx")
                    write_image_checked(transaction.stage_root / "dic" / f"{stem}_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                fields.append(field)
                last_field = field
                last_image = image8
                frame_record["artifact_stem"] = stem
                frame_record["computed"] = True
                frame_record["scientific_valid"] = scientific_valid
                frame_record["status"] = "scientific_valid" if scientific_valid else "computed_partial"
            else:
                frame_record["computed"] = False
                frame_record["scientific_valid"] = False
                frame_record["status"] = "invalid"
            frame_records.append(frame_record)
            _notify_progress(progress_callback, (local_index + 1) / max(len(inputs["processing_indices"]), 1), frame_global_1based=global_index + 1, mode=ANALYSIS_MODE_FULLFIELD, status=frame_record["status"])
        quality_frames = [item for item in frame_records if item.get("status") == "scientific_valid"]
        min_valid_frames = int(_setting_value(settings, "min_valid_frames", section="quality", aliases=("min_strain_points",), default=1))
        min_frame_ratio = float(_setting_value(settings, "min_valid_frame_ratio", section="quality", default=0.0))
        correlations = [float(item["correlation_valid_fraction"]) for item in quality_frames if item.get("correlation_valid_fraction") is not None and np.isfinite(item["correlation_valid_fraction"])]
        strains = [float(item["strain_valid_fraction"]) for item in quality_frames if item.get("strain_valid_fraction") is not None and np.isfinite(item["strain_valid_fraction"])]
        gate_reasons = []
        if len(quality_frames) < min_valid_frames:
            gate_reasons.append("valid_frame_count_below_threshold")
        if len(quality_frames) / max(len(frame_records), 1) < min_frame_ratio:
            gate_reasons.append("valid_frame_ratio_below_threshold")
        if correlations and min(correlations) < solver["min_correlation_valid_fraction"]:
            gate_reasons.append("correlation_valid_fraction_below_threshold")
        if strains and min(strains) < solver["min_strain_valid_fraction"]:
            gate_reasons.append("strain_valid_fraction_below_threshold")
        if not quality_frames:
            gate_reasons.append("no_valid_field_frames")
        gate = {
            "scientific_ok": not gate_reasons,
            "reasons": gate_reasons,
            "valid_frame_count": len(quality_frames),
            "frame_count": len(frame_records),
            "thresholds": {
                "min_valid_frames": min_valid_frames,
                "min_valid_frame_ratio": min_frame_ratio,
                "min_correlation_valid_fraction": solver["min_correlation_valid_fraction"],
                "min_strain_valid_fraction": solver["min_strain_valid_fraction"],
            },
        }
        status = "completed" if gate["scientific_ok"] else "scientific_gate_failed"
        solver_provenance = dict(solver)
        if not np.isfinite(float(solver_provenance["max_residual_rms"])):
            solver_provenance["max_residual_rms"] = None
        manifest_extra = {
            "analysis_mode": ANALYSIS_MODE_FULLFIELD,
            "input_metadata": [
                {"path": identity["path"], "shape": [int(value) for value in inputs["decoded"][index].shape], "dtype": str(inputs["decoded"][index].dtype)}
                for identity, index in zip(inputs["input_identities"], inputs["selected_indices"])
            ],
            "processing_order_frame_1based": [index + 1 for index in inputs["processing_indices"]],
            "reference_frame_1based": inputs["reference_frame_1based"],
            "reference_filename": Path(inputs["paths"][inputs["reference_idx"]]).name,
            "field_roi": [float(value) for value in roi],
            "normalization": normalization,
            "texture_preflight": _texture_preflight_record(
                None,
                texture_settings,
                field_roi=roi,
                field_metrics=field_texture,
                field_code=field_texture_code,
            ),
            "solver": solver_provenance,
            "frames": frame_records,
            "invalid_frames": [item for item in frame_records if item.get("status") != "scientific_valid"],
            "scientific_gate": gate,
            "warnings": ["invalid_field_frames_present"] if any(item.get("status") != "scientific_valid" for item in frame_records) else [],
            "optional_failures": [],
            "output_policy": output_options,
        }
        required_outputs = [_portable_relative(path, transaction.stage_root) for path in _iter_files(transaction.stage_root)]
        if not gate["scientific_ok"]:
            _, failed_manifest = transaction.seal(status=status, scientific_ok=False, manifest_extra=manifest_extra, required_outputs=required_outputs, code_paths=_code_paths_for_settings(settings))
            failed_dir = transaction.abort(CoreError("SCIENTIFIC_GATE_FAILED", {"gate": gate, "message": "fullfield scientific validity gate did not pass"}))
            manifest_path = (failed_dir / RUN_MANIFEST_FILENAME) if failed_dir is not None else failed_manifest
            failed_verification = verify_run_manifest(manifest_path)
            return {
                "status": status,
                "scientific_ok": False,
                "integrity_ok": bool(failed_verification.get("ok")),
                "manifest_path": str(manifest_path),
                "outputs": [],
                "warnings": manifest_extra["warnings"],
                "errors": gate["reasons"],
                "fields": fields,
                "last_field": last_field,
                "last_image": last_image,
                "frames": frame_records,
                "json_summary": {"status": status, "scientific_ok": False, "integrity_ok": bool(failed_verification.get("ok")), "mode": ANALYSIS_MODE_FULLFIELD, "manifest_path": str(manifest_path), "gate": gate},
                "manifest": json.loads(Path(manifest_path).read_text(encoding="utf-8")) if Path(manifest_path).is_file() else None,
            }
        _, manifest = transaction.seal(status=status, scientific_ok=True, manifest_extra=manifest_extra, required_outputs=required_outputs, code_paths=_code_paths_for_settings(settings))
        manifest_path = transaction.commit()
        return {
            "status": status,
            "scientific_ok": bool(gate["scientific_ok"]),
            "integrity_ok": True,
            "manifest_path": str(manifest_path),
            "outputs": [str(transaction.output_root / entry["path"]) for entry in manifest.get("outputs", [])],
            "warnings": manifest_extra["warnings"],
            "errors": [],
            "fields": fields,
            "last_field": last_field,
            "last_image": last_image,
            "frames": frame_records,
            "json_summary": {"status": status, "scientific_ok": bool(gate["scientific_ok"]), "mode": ANALYSIS_MODE_FULLFIELD, "manifest_path": str(manifest_path), "gate": gate},
            "manifest": json.loads(Path(manifest_path).read_text(encoding="utf-8")),
        }
    except Exception as exc:
        transaction.abort(exc)
        raise


__all__ = [
    "CoreError",
    "NORMALIZATION_VERSION",
    "RUN_ID_MAX_LENGTH",
    "TEXTURE_METRICS_VERSION",
    "TEXTURE_DISCRIMINATOR_VERSION",
    "TEXTURE_PREFLIGHT_VERSION",
    "DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO",
    "DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE",
    "run_extensometer_sequence",
    "run_fullfield_sequence",
    "run_2d_dic",
    "run_2d_dic_sequence",
    "compute_strain_fields",
    "dic_field_to_dataframe",
    "export_dic_field_outputs",
    "build_core_strain_table",
    "write_origin_txt",
    "read_gray_image",
    "normalize_to_uint8",
    "extract_patch_subpixel",
    "integer_cc_guess",
    "match_template_candidate",
    "match_template_candidate_diagnostic",
    "update_template_from_rect",
    "forward_backward_error",
    "initialize_extensometer_group_state",
    "track_extensometer_group_frame",
    "refine_subset_icgn",
    "refine_subset_iclm",
    "generate_synthetic_speckle",
    "warp_image_translation",
    "warp_image_deformation_gradient",
    "green_lagrange_from_F",
    "compute_reference_normalization",
    "compute_reference_normalization_bounds",
    "reference_percentile_bounds",
    "normalize_with_bounds",
    "normalize_sequence_frames",
    "sha256_file",
    "file_identity",
    "ordered_input_manifest",
    "ordered_input_identities",
    "canonicalize_json",
    "canonical_json_bytes",
    "canonical_json_hash",
    "collect_environment",
    "environment_info",
    "resolve_code_paths",
    "resolve_source_paths",
    "code_fingerprint",
    "build_run_manifest",
    "write_run_manifest",
    "verify_run_manifest",
    "RunTransaction",
    "roi_texture_metrics",
    "texture_failure_code",
    "texture_is_ok",
    "require_texture",
    "validate_roi_texture",
    "fullfield_field_has_finite_strain",
    "field_quality_summary",
    "DEFAULT_PEAK_MARGIN_MIN",
    "DEFAULT_PEAK_RATIO_MIN",
    "DEFAULT_MAX_HESSIAN_CONDITION_NUMBER",
    "DEFAULT_MIN_CORRELATION_VALID_FRACTION",
    "DEFAULT_MIN_STRAIN_VALID_FRACTION",
]
