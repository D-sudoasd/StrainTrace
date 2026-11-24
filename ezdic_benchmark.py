"""Compatibility facade for the canonical :mod:`benchmarks` benchmark.

The release and CLI use ``benchmarks.run_benchmark`` directly.  This module is
kept only for historical imports and frozen-entrypoint metadata; it contains
no independent case definitions or PASS/FAIL accounting.
"""

from __future__ import annotations

import sys
import copy
from pathlib import Path
from typing import Any

from benchmarks.run_benchmark import (
    BenchmarkConfigError,
    BenchmarkError,
    BenchmarkGateError,
    BenchmarkIOError,
    REPORT_VERSION,
    _ambiguity_result_passes,
    _code_provenance,
    _machine_ambiguity_code,
    _module_hash,
    run_benchmark as _canonical_run_benchmark,
)
from benchmarks.synthetic_cases import (
    CASE_DOCUMENT_VERSION,
    IMAGE_SHAPE,
    LOCKED_CASE_DOCUMENT,
    LOCKED_COORDINATES,
    ROI,
    STEP,
    STRAIN_WINDOW,
    SUBSET_SIZE,
    default_cases_path,
    locked_case_hash,
    load_locked_cases,
)


LOCKED_CASES_VERSION = CASE_DOCUMENT_VERSION
LOCKED_CASES = copy.deepcopy(LOCKED_CASE_DOCUMENT["cases"])
LOCKED_THRESHOLDS = copy.deepcopy(LOCKED_CASE_DOCUMENT["thresholds"])
LOCKED_GEOMETRY = {
    "image_shape": list(IMAGE_SHAPE),
    "roi": list(ROI),
    "subset_size": SUBSET_SIZE,
    "step": STEP,
    "strain_window": STRAIN_WINDOW,
    "locked_poi_count": len(LOCKED_COORDINATES),
    "locked_coordinates": [list(point) for point in LOCKED_COORDINATES],
}
locked_cases_hash = locked_case_hash


def validate_cases_path(path: str | Path) -> None:
    """Validate a caller-provided case document against the canonical lock."""

    try:
        load_locked_cases(path)
    except ValueError as exc:
        raise BenchmarkConfigError(str(exc)) from exc


def locked_cases_document() -> dict[str, Any]:
    """Return a copy of the canonical migrated case document."""

    return copy.deepcopy(LOCKED_CASE_DOCUMENT)


def run_benchmark(
    output_dir: str | Path | None = None,
    *,
    cases_path: str | Path | None = None,
    core_module: Any | None = None,
) -> dict[str, Any]:
    """Delegate historical calls to the canonical benchmark implementation."""

    return _canonical_run_benchmark(cases_path=cases_path, output_dir=output_dir, core_module=core_module)


def _source_hash() -> str:
    return _module_hash(sys.modules.get(__name__), "ezdic_benchmark")


def _core_provenance(core: Any | None) -> dict[str, Any]:
    try:
        info = _code_provenance(core, default_cases_path())
    except Exception:
        return {"module": "ezdic_core", "source_sha256": "unavailable", "version": "unavailable"}
    version = getattr(core, "APP_VERSION", None) or getattr(core, "__version__", None) or getattr(core, "VERSION", None) or "unknown"
    return {"module": "ezdic_core", "source_sha256": info.get("core_source_sha256", "unavailable"), "version": str(version)}


def _green_lagrange(F: Any, np: Any) -> tuple[float, float, float]:
    tensor = 0.5 * (np.asarray(F, dtype=float).T @ np.asarray(F, dtype=float) - np.eye(2))
    return float(tensor[0, 0]), float(tensor[1, 1]), float(tensor[0, 1])


__all__ = [
    "BenchmarkCapabilityError",
    "BenchmarkConfigError",
    "BenchmarkError",
    "BenchmarkGateError",
    "BenchmarkIOError",
    "BenchmarkResultError",
    "LOCKED_CASES",
    "LOCKED_CASES_VERSION",
    "LOCKED_COORDINATES",
    "LOCKED_GEOMETRY",
    "LOCKED_THRESHOLDS",
    "REPORT_VERSION",
    "_ambiguity_result_passes",
    "_machine_ambiguity_code",
    "locked_cases_document",
    "locked_cases_hash",
    "locked_case_hash",
    "validate_cases_path",
    "run_benchmark",
]


class BenchmarkCapabilityError(BenchmarkError):
    """Historical name retained for callers of the old facade."""

    code = "CAPABILITY_NOT_AVAILABLE"
    exit_code = 4


BenchmarkResultError = BenchmarkGateError
