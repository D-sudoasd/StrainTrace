"""Shared provenance, I/O, and error-statistics helpers for benchmark scripts."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_core_module(path: str | Path, module_name: str = "archived_ezdic_core") -> Any:
    """Dynamically load a standalone frozen ``ezdic_core.py`` file."""
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"cannot load core module: {target}")
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load core module: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def green_lagrange_components(deformation_gradient: Any) -> tuple[float, float, float]:
    """Return ``(Exx, Eyy, Exy)`` from a 2x2 deformation gradient tensor."""
    import numpy as np

    F = np.asarray(deformation_gradient, dtype=float).reshape(2, 2)
    E = 0.5 * (F.T @ F - np.eye(2))
    return float(E[0, 0]), float(E[1, 1]), float(E[0, 1])


def set_deterministic_rng(seed: int = 0) -> None:
    """Seed OpenCV's global RNG for reproducible SIFT/RANSAC execution."""
    import cv2

    cv2.setRNGSeed(int(seed))


def signed_error_stats(values: Any) -> dict[str, Any]:
    """Summary statistics for finite signed errors (matching necking reports)."""
    import numpy as np

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not data.size:
        return {"count": 0, "bias": None, "rmse": None, "p95_abs": None, "max_abs": None}
    return {
        "count": int(data.size),
        "bias": float(np.mean(data)),
        "rmse": float(np.sqrt(np.mean(data**2))),
        "p95_abs": float(np.percentile(np.abs(data), 95)),
        "max_abs": float(np.max(np.abs(data))),
    }


def distribution_stats(values: Any) -> dict[str, Any]:
    """Distribution quantiles for finite scalar diagnostics."""
    import numpy as np

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not data.size:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(data.size),
        "min": float(np.min(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
    }


def compact_vector_stats(values: Any) -> dict[str, Any]:
    """Compact count/RMSE/max-abs summary for paired study comparisons."""
    import numpy as np

    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    return {
        "count": int(finite.size),
        "rmse": float(np.sqrt(np.mean(finite**2))) if finite.size else None,
        "max_abs": float(np.max(np.abs(finite))) if finite.size else None,
    }


def threshold_vector_stats(values: Any, *, cutoff: float = 0.5) -> dict[str, Any]:
    """Vector displacement error stats with descriptive cutoff count."""
    import numpy as np

    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    if not finite.size:
        return {"count": 0, "rmse_px": None, "max_px": None, "over_0_5_px_count": 0}
    return {
        "count": int(finite.size),
        "rmse_px": float(np.sqrt(np.mean(finite**2))),
        "max_px": float(finite.max()),
        "over_0_5_px_count": int((finite > float(cutoff)).sum()),
    }


def runtime_environment(*, include_opencv: bool = True) -> dict[str, str]:
    import numpy as np
    import scipy

    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if include_opencv:
        import cv2

        env["opencv"] = cv2.__version__
    return env


def stream_supports_utf8(stream: Any) -> bool:
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    return str(encoding).lower().replace("_", "-") in {"utf-8", "utf8", "utf-8-sig"}


def write_json_line(stream: Any, value: Any) -> None:
    ensure_ascii = not stream_supports_utf8(stream)
    stream.write(
        json.dumps(value, ensure_ascii=ensure_ascii, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )


def write_json_lf(
    path: str | Path,
    payload: Any,
    *,
    exclusive: bool = False,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    allow_nan: bool = False,
    atomic: bool = False,
) -> Path:
    """Write UTF-8 JSON with LF line endings (optionally exclusive or atomic)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if indent is None:
        text = json.dumps(
            payload,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            separators=(",", ":"),
            allow_nan=allow_nan,
        ) + "\n"
    else:
        text = json.dumps(
            payload,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            indent=indent,
            allow_nan=allow_nan,
        ) + "\n"
    if exclusive:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return target
    if atomic:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            os.replace(tmp_path, target)
        except BaseException:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return target
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


def write_csv_lf(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
    *,
    exclusive: bool = False,
    extrasaction: str = "raise",
    atomic: bool = False,
) -> Path:
    """Write UTF-8 CSV with LF line endings (optionally exclusive or atomic)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    if exclusive:
        with target.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction=extrasaction, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return target
    if atomic:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction=extrasaction, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp_path, target)
        except BaseException:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return target
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction=extrasaction, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return target


__all__ = [
    "compact_vector_stats",
    "distribution_stats",
    "green_lagrange_components",
    "load_core_module",
    "runtime_environment",
    "set_deterministic_rng",
    "sha256_bytes",
    "sha256_file",
    "signed_error_stats",
    "stream_supports_utf8",
    "threshold_vector_stats",
    "write_csv_lf",
    "write_json_lf",
    "write_json_line",
]
