"""Source image inventory, frame maps, reference selection, mask loading, and code provenance."""
from __future__ import annotations

import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._config import _positive_number
from ._io import (
    AgentError,
    _api_attr,
    _atomic_json,
    _hash_json,
    _json_safe,
    _read_json,
    _sha256,
)


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
_FRAME_TOKEN = re.compile(r"\d+")
_REQUIRED_AGENT_MODULES = {
    "__init__.py",
    "__main__.py",
    "_config.py",
    "_inventory.py",
    "_io.py",
    "_locks.py",
    "_outputs.py",
    "_verify.py",
    "api.py",
    "cli.py",
    "discovery.py",
    "fields.py",
    "gui.py",
    "mechanics.py",
    "regions.py",
    "reporting.py",
}


def _code_fingerprint() -> dict[str, str]:
    import ezdic_core

    sha256_fn = _api_attr("_sha256", _sha256)
    frozen_root = getattr(sys, "_MEIPASS", None)
    runtime_root = Path(frozen_root).resolve() if frozen_root else Path(__file__).resolve().parent.parent
    try:
        core_sources = ezdic_core.resolve_source_paths(
            include_cli=True, include_gui=False, include_benchmark=False, include_schema=False,
        )
    except Exception as exc:
        raise AgentError("CODE_PROVENANCE_UNAVAILABLE", f"cannot resolve core/CLI source files: {exc}") from exc
    package_root = runtime_root / "sources" / "straintrace_agent" if frozen_root else Path(__file__).resolve().parent
    if not package_root.is_dir():
        raise AgentError("CODE_PROVENANCE_UNAVAILABLE", f"agent source directory is missing: {package_root}")
    package_files = sorted(path for path in package_root.rglob("*.py") if path.is_file())
    missing = sorted(_REQUIRED_AGENT_MODULES - {path.name for path in package_files})
    if missing:
        raise AgentError("CODE_PROVENANCE_UNAVAILABLE", f"agent source file(s) are missing: {', '.join(missing)}")
    result = {}
    for path in core_sources:
        result[path.name] = sha256_fn(Path(path))
    for path in package_files:
        relative = path.relative_to(package_root).as_posix()
        result[f"straintrace_agent/{relative}"] = sha256_fn(path)
    return result


def _explicit_frame_map(path: Path, image_root: Path, image_paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    try:
        table = pd.read_csv(path)
    except Exception as exc:
        raise AgentError("FRAME_MAP_READ_ERROR", f"cannot read frame map: {exc}", path=str(path)) from exc
    names = {str(column).strip().lower(): str(column) for column in table.columns}
    file_col = next((names[key] for key in ("path", "filename", "file") if key in names), None)
    frame_col = names.get("native_frame") or names.get("frame")
    time_col = names.get("time_s") or names.get("timestamp_s")
    if file_col is None or frame_col is None:
        raise AgentError("FRAME_MAP_COLUMNS", "frame map requires path/filename and native_frame columns", path=str(path))
    known = {item.resolve(strict=False) for item in image_paths}
    by_basename: dict[str, list[Path]] = {}
    for image_path in known:
        by_basename.setdefault(image_path.name.casefold(), []).append(image_path)
    by_key: dict[str, dict[str, Any]] = {}
    for row_index, row in table.iterrows():
        raw_path = str(row[file_col]).strip()
        if not raw_path:
            raise AgentError("FRAME_MAP_VALUE", f"empty path at row {row_index + 2}", path=str(path))
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = image_root / candidate
        candidate = candidate.resolve(strict=False)
        if candidate not in known:
            # A basename is accepted only if unique in the image directory.
            # Never reinterpret an explicit (possibly mistyped) directory as
            # a different source image with the same basename.
            if (len(PurePosixPath(raw_path).parts) != 1
                    or len(PureWindowsPath(raw_path).parts) != 1
                    or PurePosixPath(raw_path).is_absolute()
                    or PureWindowsPath(raw_path).is_absolute()):
                raise AgentError("FRAME_MAP_PATH_UNKNOWN", f"frame map row does not identify one source image: {raw_path}", path=str(path))
            matches = by_basename.get(Path(raw_path).name.casefold(), [])
            if len(matches) != 1:
                raise AgentError("FRAME_MAP_PATH_UNKNOWN", f"frame map row does not identify one source image: {raw_path}", path=str(path))
            candidate = matches[0]
        try:
            native_value = float(row[frame_col])
            if not math.isfinite(native_value) or native_value != int(native_value) or native_value < 0:
                raise ValueError
            native = int(native_value)
            time_value = None
            if time_col is not None and pd.notna(row[time_col]):
                time_value = float(row[time_col])
                if not math.isfinite(time_value):
                    raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise AgentError("FRAME_MAP_VALUE", f"invalid native_frame or time at row {row_index + 2}", path=str(path)) from exc
        key = str(candidate).casefold()
        if key in by_key:
            raise AgentError("FRAME_MAP_DUPLICATE_IMAGE", f"image appears more than once in frame map: {candidate}", path=str(path))
        by_key[key] = {"native_frame": native, "time_s": time_value}
    if set(by_key) != {str(item.resolve(strict=False)).casefold() for item in image_paths}:
        raise AgentError("FRAME_MAP_INCOMPLETE", "frame map must cover every source image exactly once", path=str(path))
    return by_key


def parse_frame_map(path: str | os.PathLike[str], image_root: str | os.PathLike[str], image_paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Parse and validate an explicit filename-to-native-frame CSV mapping."""
    frame_map_fn = _api_attr("_explicit_frame_map", _explicit_frame_map)
    return frame_map_fn(Path(path), Path(image_root), image_paths)


def _native_frame_from_name(path: Path) -> int:
    tokens = _FRAME_TOKEN.findall(path.stem)
    if len(tokens) != 1:
        raise AgentError(
            "FRAME_ID_AMBIGUOUS",
            f"expected one numeric frame token in {path.name}; provide an explicit frame map",
            path=str(path),
            details={"numeric_tokens": tokens},
        )
    return int(tokens[0])


def parse_native_frame(path: str | os.PathLike[str]) -> int:
    """Extract the unique integer native frame number from an image filename."""
    return _native_frame_from_name(Path(path))


def resolve_time_basis(config: Mapping[str, Any]) -> str:
    """Determine ``physical_time`` vs ``frame_index`` from explicit timing inputs."""
    if config.get("frame_interval_s") is not None or config.get("loads") or config.get("loads_csv"):
        return "physical_time"
    if not config.get("frame_map_csv"):
        return "frame_index"
    from .discovery import source_files

    image_root = Path(str(config["image_dir"])).expanduser().resolve()
    mask = Path(str(config["mask_file"])).expanduser().resolve() if config.get("mask_file") else None
    paths = [
        path.resolve() for path in source_files(image_root)
        if path.suffix.lower() in IMAGE_SUFFIXES and path.resolve() != mask
    ]
    mapping = parse_frame_map(Path(str(config["frame_map_csv"])).expanduser().resolve(), image_root, paths)
    return "physical_time" if any(row["time_s"] is not None for row in mapping.values()) else "frame_index"


def _scan_inventory(
    config: Mapping[str, Any], *, read_dimensions: bool = True,
    cached_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    from .discovery import source_files

    sha256_fn = _api_attr("_sha256", _sha256)
    frame_map_fn = _api_attr("_explicit_frame_map", _explicit_frame_map)
    image_root = Path(str(config["image_dir"])).expanduser().resolve(strict=False)
    if not image_root.is_dir():
        raise AgentError("IMAGE_DIRECTORY_MISSING", f"image directory does not exist: {image_root}", path="$.image_dir")
    mask_path = Path(str(config["mask_file"])).resolve(strict=False) if config.get("mask_file") else None
    paths = sorted(
        (p.resolve(strict=False) for p in source_files(image_root) if p.suffix.lower() in IMAGE_SUFFIXES and (mask_path is None or p.resolve(strict=False) != mask_path)),
        key=lambda p: str(p).casefold(),
    )
    if not paths:
        raise AgentError("NO_IMAGES", f"no supported DIC images found in {image_root}", path="$.image_dir")
    explicit: dict[str, dict[str, Any]] | None = None
    if config.get("frame_map_csv"):
        map_path = Path(str(config["frame_map_csv"])).expanduser().resolve(strict=False)
        explicit = frame_map_fn(map_path, image_root, paths)
    cached_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for cached in cached_records or ():
        if not isinstance(cached, Mapping):
            continue
        cached_path = cached.get("path")
        cached_sha = cached.get("sha256")
        if isinstance(cached_path, str) and isinstance(cached_sha, str) and (cached.get("shape_yx") or cached.get("status") == "unreadable"):
            key = (str(Path(cached_path).resolve(strict=False)).casefold(), cached_sha)
            cached_by_identity[key] = cached
    records: list[dict[str, Any]] = []
    native_values: set[int] = set()
    times_explicit = explicit is not None and any(value.get("time_s") is not None for value in explicit.values())
    for path in paths:
        if explicit is None:
            native = _native_frame_from_name(path)
            time_value = None
        else:
            item = explicit[str(path).casefold()]
            native = item["native_frame"]
            time_value = item.get("time_s")
        if native in native_values:
            raise AgentError("FRAME_ID_DUPLICATE", f"duplicate native frame number: {native}", path=str(path))
        native_values.add(native)
        file_sha = sha256_fn(path)
        row: dict[str, Any] = {
            "native_frame": native,
            "path": str(path),
            "relative_path": path.relative_to(image_root).as_posix(),
            "sha256": file_sha,
            "time_s": time_value,
            "status": "pending",
        }
        if read_dimensions:
            cached = cached_by_identity.get((str(path).casefold(), file_sha))
            if cached is not None and cached.get("shape_yx"):
                row["shape_yx"] = [int(cached["shape_yx"][0]), int(cached["shape_yx"][1])]
                if cached.get("dtype") is not None:
                    row["dtype"] = str(cached["dtype"])
            elif cached is not None and cached.get("status") == "unreadable":
                row["status"] = "unreadable"
                row["read_error"] = str(cached.get("read_error") or "unreadable")
            else:
                try:
                    import ezdic_core as core
                    image = core.read_gray_image(path)
                    row["shape_yx"] = [int(image.shape[0]), int(image.shape[1])]
                    row["dtype"] = str(image.dtype)
                except Exception as exc:
                    row["status"] = "unreadable"
                    row["read_error"] = f"{type(exc).__name__}: {exc}"
        records.append(row)
    records.sort(key=lambda item: int(item["native_frame"]))
    if len(native_values) != len(records):
        raise AgentError("FRAME_ID_DUPLICATE", "native frame identity is not unique")
    native_order = [int(item["native_frame"]) for item in records]
    if any(right <= left for left, right in zip(native_order, native_order[1:])):
        raise AgentError("FRAME_ID_ORDER", "native frame numbers must be strictly increasing")
    if times_explicit:
        if config.get("time_basis") == "frame_index":
            raise AgentError("CONFIG_TIME_BASIS_CONFLICT", "frame_index cannot discard explicitly recorded frame times", path="$.time_basis")
        times = [item.get("time_s") for item in records]
        if any(value is None for value in times) or any(float(b) <= float(a) for a, b in zip(times, times[1:])):
            raise AgentError("FRAME_TIME_ORDER", "explicit frame times must be complete and strictly increasing")
    elif config.get("time_basis") != "frame_index":
        interval = config.get("frame_interval_s")
        if interval is None:
            raise AgentError("FRAME_TIME_REQUIRED", "provide a positive frame_interval_s or explicit frame-map time_s values", path="$.frame_interval_s")
        interval = _positive_number(interval, "frame_interval_s")
        first = native_order[0]
        for row in records:
            row["time_s"] = round((int(row["native_frame"]) - first) * interval, 12)
    reference_native = config.get("reference_native_frame")
    reference_row = next((row for row in records if row.get("native_frame") == reference_native and row.get("shape_yx")), None) if reference_native is not None else None
    reference_shape = tuple(reference_row["shape_yx"]) if reference_row else next((tuple(row["shape_yx"]) for row in records if row.get("shape_yx")), None)
    if reference_shape is not None:
        for row in records:
            if row.get("shape_yx") and tuple(row["shape_yx"]) != reference_shape:
                row["status"] = "dimension_mismatch"
    if config.get("reference_native_frame") is not None and int(config["reference_native_frame"]) not in native_values:
        raise AgentError("REFERENCE_FRAME_UNKNOWN", "reference_native_frame is not present in the image inventory", path="$.reference_native_frame")
    return records


def _inventory_fresh(root: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    cached_frames: list[Mapping[str, Any]] | None = None
    inventory_path = root / "inventory.json"
    if inventory_path.is_file():
        try:
            stored = _read_json(inventory_path)
            frames = stored.get("frames")
            if isinstance(frames, list):
                cached_frames = frames
        except AgentError:
            cached_frames = None
    current = _scan_inventory(config, read_dimensions=True, cached_records=cached_frames)
    return current, _inventory_fingerprint(config, current)


def _inventory_fingerprint(config: Mapping[str, Any], current: Sequence[Mapping[str, Any]]) -> str:
    """Bind an already hashed image inventory to its auxiliary source files."""
    sha256_fn = _api_attr("_sha256", _sha256)
    auxiliary: dict[str, str] = {}
    for key in ("frame_map_csv", "mask_file"):
        if config.get(key):
            path = Path(str(config[key]))
            if not path.is_file():
                raise AgentError("INPUT_FILE_MISSING", f"configured input is missing: {path}", path=f"$.{key}")
            auxiliary[key] = sha256_fn(path)
    loads = config.get("loads")
    if loads:
        load_path = Path(str(loads["path"]))
        if not load_path.is_file():
            raise AgentError("LOAD_FILE_MISSING", f"load data file is missing: {load_path}", path="$.loads.path")
        current_load_sha = sha256_fn(load_path)
    else:
        current_load_sha = None
    return _hash_json({
        "frames": [{key: item.get(key) for key in ("native_frame", "path", "relative_path", "sha256", "time_s")} for item in current],
        "loads_sha256": current_load_sha,
        "auxiliary_sha256": auxiliary,
    })


def _record_inventory(root: Path, records: Sequence[Mapping[str, Any]], fingerprint: str) -> None:
    inventory_path = root / "inventory.json"
    if inventory_path.exists():
        previous = _read_json(inventory_path)
        previous_fp = previous.get("input_fingerprint") or previous.get("inventory_sha256")
        if previous_fp != fingerprint:
            archive = root / "inventories" / f"inventory_{str(previous_fp)[:16]}.json"
            if not archive.exists():
                _atomic_json(archive, previous)
    _atomic_json(
        inventory_path,
        {
            "schema": "straintrace.agent-inventory/v1",
            "frames": _json_safe(list(records)),
            "input_fingerprint": fingerprint,
            "inventory_sha256": _hash_json(records),
            "updated_unix_s": time.time(),
        },
    )


def _load_image(path: str | os.PathLike[str]) -> np.ndarray:
    import ezdic_core as core

    try:
        image = np.asarray(core.read_gray_image(path))
    except Exception as exc:
        raise AgentError("IMAGE_DECODE_FAILED", f"cannot decode image {path}: {exc}", path=str(path)) from exc
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        raise AgentError("IMAGE_INVALID", f"decoded image is empty or non-finite: {path}", path=str(path))
    return image


def _load_specimen_mask(config: Mapping[str, Any], image_shape: Sequence[int]) -> np.ndarray | None:
    sha256_fn = _api_attr("_sha256", _sha256)
    mask_path = config.get("mask_file")
    if not mask_path:
        if not config.get("exclusions_px"):
            return None
        mask = np.ones(tuple(int(v) for v in image_shape[:2]), dtype=bool)
    else:
        import ezdic_core as core

        if sha256_fn(Path(str(mask_path))) != config.get("mask_sha256"):
            raise AgentError("MASK_CHANGED", "mask file changed since project initialization", path="$.mask_file")
        try:
            mask_image = core.read_gray_image(mask_path)
        except Exception as exc:
            raise AgentError("MASK_UNREADABLE", f"cannot read specimen mask: {exc}", path=str(mask_path)) from exc
        if tuple(mask_image.shape[:2]) != tuple(image_shape[:2]):
            raise AgentError("MASK_DIMENSION_MISMATCH", "mask dimensions must match source images", path=str(mask_path))
        mask = np.asarray(mask_image) > 0
    for rect in config.get("exclusions_px") or []:
        x, y, width, height = (int(value) for value in rect)
        if x + width > mask.shape[1] or y + height > mask.shape[0]:
            raise AgentError("EXCLUSION_OUTSIDE_IMAGE", f"exclusion rectangle outside image: {rect}")
        mask[y:y + height, x:x + width] = False
    return mask


def _texture_roi(image: np.ndarray, subset_size: int) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    import cv2

    gray = np.asarray(image, dtype=np.float64)
    if min(gray.shape) < subset_size + 2:
        raise AgentError("IMAGE_TOO_SMALL", f"image is too small for subset_size={subset_size}")
    scale = min(1.0, 1024.0 / max(gray.shape))
    if scale < 1.0:
        analysis = cv2.resize(
            gray,
            (max(1, int(round(gray.shape[1] * scale))), max(1, int(round(gray.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        analysis = gray
    centered = analysis - float(np.mean(analysis))
    window = max(9, min(31, (subset_size // 2) * 2 + 1))
    mean = cv2.boxFilter(centered, ddepth=cv2.CV_64F, ksize=(window, window), normalize=True)
    mean_square = cv2.boxFilter(centered * centered, ddepth=cv2.CV_64F, ksize=(window, window), normalize=True)
    local_std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    q50, q80, q95 = (float(value) for value in np.quantile(local_std, [0.50, 0.80, 0.95]))
    if not math.isfinite(q95) or q95 <= max(1e-6, q50 * 1.01):
        raise AgentError("TEXTURE_INSUFFICIENT", "no usable spatial texture variation was detected")
    threshold = max(q50 + 0.25 * (q80 - q50), q95 * 0.18)
    active = local_std >= threshold
    fraction = float(active.mean())
    if fraction < 0.03:
        raise AgentError("TEXTURE_INSUFFICIENT", f"only {fraction:.3%} of the image passes the texture screen")
    if fraction >= 0.72:
        x0, y0, x1, y1 = 0, 0, analysis.shape[1], analysis.shape[0]
        method = "full_view_texture_coverage"
    else:
        yy, xx = np.nonzero(active)
        pad = max(0, int(round((subset_size // 2) * scale)))
        x0 = max(0, int(xx.min()) - pad)
        y0 = max(0, int(yy.min()) - pad)
        x1 = min(analysis.shape[1], int(xx.max()) + pad + 1)
        y1 = min(analysis.shape[0], int(yy.max()) + pad + 1)
        method = "local_texture_quantile_bounding_box"
    if scale < 1.0:
        left = int(math.floor(x0 / scale))
        top = int(math.floor(y0 / scale))
        right = min(gray.shape[1], int(math.ceil(x1 / scale)))
        bottom = min(gray.shape[0], int(math.ceil(y1 / scale)))
    else:
        left, top, right, bottom = x0, y0, x1, y1
    roi = (left, top, right - left, bottom - top)
    return roi, {
        "method": method, "window_px": window, "downsample_scale": scale,
        "threshold_local_std": threshold, "texture_fraction": fraction,
        "local_std_quantiles": {"q50": q50, "q80": q80, "q95": q95},
    }


def _safe_roi_for_image(roi: Sequence[int], image_shape: Sequence[int], *, search_radius_px: int) -> tuple[int, int, int, int]:
    """Keep translation search plus spline support inside image bounds."""
    x, y, width, height = (int(value) for value in roi)
    margin = int(search_radius_px) + 3
    x0, y0 = max(x, margin), max(y, margin)
    x1, y1 = min(x + width, int(image_shape[1]) - margin), min(y + height, int(image_shape[0]) - margin)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _reference_record(config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    requested = config.get("reference_native_frame")
    if requested is None:
        requested = records[0]["native_frame"]
    found = next((row for row in records if int(row["native_frame"]) == int(requested)), None)
    if found is None:
        raise AgentError("REFERENCE_FRAME_UNKNOWN", f"native reference frame {requested} is absent")
    return found


def _automatic_reference(
    config: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *,
    allow_untextured_fallback: bool = False,
) -> tuple[Mapping[str, Any], np.ndarray, dict[str, Any]]:
    load_fn = _api_attr("_load_image", _load_image)
    texture_fn = _api_attr("_texture_roi", _texture_roi)
    allow_fallback = bool(config.get("roi_override_px") or allow_untextured_fallback)
    if config.get("reference_native_frame") is not None:
        record = _reference_record(config, records)
        image = load_fn(record["path"])
        try:
            roi, texture = texture_fn(image, 21)
            return record, image, {"selection": "user_selected", "detected_roi_px": list(roi), **texture}
        except AgentError as exc:
            if not allow_fallback:
                raise
            method = "user_roi_override_texture_not_assessed" if config.get("roi_override_px") else "preview_texture_not_assessed"
            return record, image, {"selection": "user_selected", "method": method, "assessment_error": exc.as_dict()}
    failures = []
    first = records[0]
    first_readable: tuple[Mapping[str, Any], np.ndarray] | None = None
    for record in records[: min(len(records), 12)]:
        try:
            image = load_fn(record["path"])
        except AgentError as exc:
            failures.append({"native_frame": int(record["native_frame"]), "code": exc.code, "message": exc.message})
            continue
        if first_readable is None:
            first_readable = (record, image)
        try:
            roi, texture = texture_fn(image, 21)
            return record, image, {
                "selection": "first_readable_textured_frame",
                "initial_inventory_frame": int(first["native_frame"]),
                "detected_roi_px": list(roi),
                **texture,
            }
        except AgentError as exc:
            failures.append({"native_frame": int(record["native_frame"]), "code": exc.code, "message": exc.message})
    if allow_fallback and first_readable is not None:
        record, image = first_readable
        method = "user_roi_override_texture_not_assessed" if config.get("roi_override_px") else "preview_texture_not_assessed"
        return record, image, {
            "selection": "first_readable_frame_fallback",
            "initial_inventory_frame": int(first["native_frame"]),
            "method": method,
            "assessment_errors": failures,
        }
    raise AgentError("REFERENCE_FRAME_UNAVAILABLE", "no readable textured reference found among the first 12 frames", details=failures)


def load_reference_frame(
    config: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *,
    plan: Mapping[str, Any] | None = None,
    allow_untextured_fallback: bool = True,
) -> tuple[Mapping[str, Any], np.ndarray]:
    """Load the active reference frame record and grayscale image for UI/preview callers."""
    if plan is None:
        auto_ref_fn = _api_attr("_automatic_reference", _automatic_reference)
        try:
            record, gray, _selection = auto_ref_fn(config, records, allow_untextured_fallback=allow_untextured_fallback)
        except TypeError:
            record, gray, _selection = auto_ref_fn(config, records)
        return record, gray
    native = plan.get("reference_native_frame")
    if native is None:
        raise ValueError("当前计划未指定参考原生帧；请重新生成计划。")
    ref_record_fn = _api_attr("_reference_record", _reference_record)
    load_fn = _api_attr("_load_image", _load_image)
    record = ref_record_fn({"reference_native_frame": native}, records)
    return record, load_fn(record["path"])


def _pilot_frame(
    records: Sequence[Mapping[str, Any]], reference: Mapping[str, Any], *,
    reference_shape: Sequence[int] | None = None,
) -> tuple[Mapping[str, Any] | None, str | None]:
    load_fn = _api_attr("_load_image", _load_image)
    expected_shape = tuple(int(v) for v in (reference_shape or reference.get("shape_yx") or ()))
    for row in records:
        if int(row["native_frame"]) == int(reference["native_frame"]):
            continue
        if row.get("status") == "unreadable":
            continue
        if expected_shape and row.get("shape_yx") and tuple(int(v) for v in row["shape_yx"]) != expected_shape:
            continue
        try:
            image = load_fn(row["path"])
            if expected_shape and tuple(image.shape) != expected_shape:
                continue
            return row, None
        except AgentError:
            continue
    return None, "no readable non-reference frame is available for bounded parameter trials"
