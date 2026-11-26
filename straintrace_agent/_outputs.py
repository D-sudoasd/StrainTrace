"""Checkpoints, derived deliverables, single-pass plotting, and atomic publication."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import shutil
import uuid
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ._config import QUALITY_GATE_SCOPE, _solver_parameters_match
from ._io import (
    AgentError,
    _api_attr,
    _atomic_bytes,
    _atomic_json,
    _component_file_slug,
    _csv_atomic,
    _hash_json,
    _json_safe,
    _read_json,
    _relocate_paths,
    _replace_with_retry,
    _safe_file,
    _sha256,
    _unlink_with_retry,
)
from .mechanics import _interpolate_bounded_detail


_GAUGE_COLUMNS = [
    "frame_position_1based", "native_frame", "time_s", "gauge_name", "axis", "geometry_status", "status",
    "initial_length_px", "initial_length_mm", "current_length_px", "extension_px", "current_length_mm", "extension_mm",
    "a_expected_poi_count", "b_expected_poi_count", "a_valid_poi_count", "b_valid_poi_count", "endpoint_policy",
    "engineering_strain", "true_strain", "calibration_status", "measurement_kind", "loading_axis", "min_zncc", "reasons",
]
_REGION_COLUMNS = [
    "frame_position_1based", "native_frame", "time_s", "frame_status", "region_name", "component", "coordinate_system",
    "coordinate_unit", "bounds", "unit", "n_poi_total", "n_poi_valid", "valid_fraction", "n_displacement_valid",
    "n_strain_valid", "displacement_valid_fraction", "strain_valid_fraction", "min_valid_pois", "min_valid_fraction", "min_zncc",
    "mean", "median", "sd_sample", "min", "max", "status", "reasons", "source", "interpolation", "calibration_status",
    "quality_ok", "sd_interpretation", "field_support_bounds", "coverage_interpretation",
]
_PLOT_COMPONENTS = ("u", "v", "zncc", "Exx", "Eyy", "Exy", "exx", "eyy", "exy")


def _run_fingerprint(config: Mapping[str, Any], plan: Mapping[str, Any], input_sha: str, code: Mapping[str, str]) -> str:
    return _hash_json({"config": config, "plan_sha256": plan.get("plan_sha256"), "input_fingerprint": input_sha, "code_sha256": code})


def _frame_key(native_frame: int) -> str:
    return f"frame_native_{native_frame:08d}"


def _displacement_reference(
    records: Sequence[Mapping[str, Any]], plan: Mapping[str, Any], config: Mapping[str, Any],
    mechanics: Mapping[str, Any] | None, mechanics_status: Mapping[str, Any],
) -> tuple[float | None, dict[str, Any] | None]:
    if mechanics is None or mechanics_status.get("status") != "aligned" or "displacement_time_s" not in mechanics:
        return None, None
    reference_record = next((record for record in records if int(record["native_frame"]) == int(plan["reference_native_frame"])), None)
    if reference_record is None:
        return None, None
    alignment = _interpolate_bounded_detail(
        float(reference_record["time_s"]), mechanics["displacement_time_s"], mechanics["displacement_mm"],
        float(config["loads"]["max_gap_s"]), mechanics.get("displacement_source_rows"),
    )
    return alignment.get("value"), alignment


def _serialize_field(path: Path, field: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    sha256_fn = _api_attr("_sha256", _sha256)
    arrays: dict[str, np.ndarray] = {}
    for key, value in field.items():
        if isinstance(value, np.ndarray):
            if key == "specimen_mask":
                continue
            array = np.asarray(value)
            if array.dtype.kind == "O":
                array = array.astype(str)
            if array.dtype.kind not in "biufSU":
                continue
            arrays[str(key)] = array
    if not arrays:
        raise AgentError("CHECKPOINT_EMPTY", "DIC solver returned no persistable arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    metadata_path = path.with_suffix(".json")
    record = {
        "schema": "straintrace.agent-checkpoint/v1",
        **_json_safe(dict(metadata)),
        "checkpoint": path.name,
        "checkpoint_sha256": sha256_fn(path),
        "array_names": sorted(arrays),
    }
    _atomic_json(metadata_path, record)
    return {"path": path, "metadata_path": metadata_path, "sha256": record["checkpoint_sha256"], "metadata": record}


def _load_field(checkpoint: Path, metadata_path: Path | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sha256_fn = _api_attr("_sha256", _sha256)
    if metadata_path is None:
        metadata_path = checkpoint.with_suffix(".json")
    metadata = _read_json(metadata_path)
    actual_sha = sha256_fn(checkpoint)
    if actual_sha != metadata.get("checkpoint_sha256"):
        raise AgentError("CHECKPOINT_CHECKSUM_MISMATCH", f"checkpoint checksum mismatch: {checkpoint}", path=str(checkpoint))
    try:
        with np.load(checkpoint, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except Exception as exc:
        raise AgentError("CHECKPOINT_INVALID", f"cannot load safe NPZ checkpoint: {exc}", path=str(checkpoint)) from exc
    return arrays, metadata


def _checkpoint_valid(
    run_root: Path, record: Mapping[str, Any], frame_position: int,
    reference_native_frame: int, run_fingerprint: str, input_sha: str,
    code_sha: Mapping[str, str], expected_parameters: Mapping[str, Any],
    *, prior_receipts: Mapping[str, Any] | None = None, allow_unregistered: bool = True,
) -> tuple[Path, dict[str, Any]] | None:
    native_frame = int(record["native_frame"])
    checkpoint = run_root / "checkpoints" / f"{_frame_key(native_frame)}.npz"
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.is_file():
        return None
    prior_receipt = prior_receipts.get(metadata_path.name) if prior_receipts is not None else None
    if prior_receipt is None:
        if not allow_unregistered:
            return None
    elif not isinstance(prior_receipt, Mapping) or _api_attr("_sha256", _sha256)(metadata_path) != prior_receipt.get("sha256"):
        # A sidecar also contains result rows and gauge measurements; checking
        # only the NPZ hash and frame identity would silently bless edits.
        return None

    def matches_expected_identity(metadata: Mapping[str, Any]) -> bool:
        if metadata_path.name != f"{_frame_key(native_frame)}.json":
            return False
        if (metadata.get("native_frame") != native_frame
                or metadata.get("frame_position") != frame_position
                or metadata.get("source_path") != record.get("path")
                or metadata.get("source_sha256") != record.get("sha256")
                or metadata.get("reference_native_frame") != reference_native_frame
                or metadata.get("run_fingerprint") != run_fingerprint
                or metadata.get("input_fingerprint") != input_sha
                or metadata.get("code_sha256") != code_sha
                or not _solver_parameters_match(metadata.get("parameters"), expected_parameters)):
            return False
        expected_time = record.get("time_s")
        if expected_time is None:
            if metadata.get("time_s") is not None:
                return False
        else:
            try:
                actual_time = float(metadata.get("time_s"))
                if not math.isfinite(actual_time) or not math.isclose(actual_time, float(expected_time), rel_tol=0.0, abs_tol=1e-12):
                    return False
            except (TypeError, ValueError):
                return False
        summary = metadata.get("summary_row")
        if not isinstance(summary, Mapping):
            return False
        if (summary.get("native_frame") != native_frame
                or summary.get("frame_position_1based") != frame_position
                or summary.get("reference_native_frame") != reference_native_frame
                or summary.get("status") != metadata.get("status")):
            return False
        summary_time = summary.get("time_s")
        if expected_time is None:
            if summary_time is not None:
                return False
        else:
            try:
                if not math.isclose(float(summary_time), float(expected_time), rel_tol=0.0, abs_tol=1e-12):
                    return False
            except (TypeError, ValueError):
                return False
        if metadata.get("status") == "failed":
            return metadata.get("checkpoint") is None and summary.get("checkpoint") is None
        return (metadata.get("status") in {"complete", "quality_failed"}
                and metadata.get("checkpoint") == checkpoint.name
                and summary.get("checkpoint") == f"checkpoints/{checkpoint.name}")

    try:
        metadata = _read_json(metadata_path)
        if not matches_expected_identity(metadata):
            return None
        if metadata.get("status") == "failed":
            if prior_receipt is not None and prior_receipt.get("checkpoint_sha256") is not None:
                return None
            if metadata.get("retryable") is True:
                return None
            return metadata_path, metadata
        if not checkpoint.is_file():
            return None
        load_field_fn = _api_attr("_load_field", _load_field)
        arrays, metadata = load_field_fn(checkpoint, metadata_path)
        if prior_receipt is not None and prior_receipt.get("checkpoint_sha256") != metadata.get("checkpoint_sha256"):
            return None
        if not matches_expected_identity(metadata) or not arrays:
            return None
        return checkpoint, metadata
    except AgentError:
        return None


def _frame_failure(
    run_root: Path, record: Mapping[str, Any], run_fingerprint: str, input_sha: str,
    code_sha: Mapping[str, str], error: AgentError, *,
    frame_position: int | None = None, reference_native_frame: int | None = None,
    retryable: bool = False, parameters: Mapping[str, Any] | None = None,
    mechanics_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    native = int(record["native_frame"])
    path = run_root / "checkpoints" / f"{_frame_key(native)}.json"
    row = {
        "frame_position_1based": frame_position, "native_frame": native, "time_s": record["time_s"],
        "reference_native_frame": reference_native_frame, "status": "failed", "quality_status": "unavailable",
        "scientific_ok": False, "failure_code": error.code, "failure_reason": error.message, "checkpoint": None,
        "force_N": None, "stress_MPa": None, "mechanics_status": "not_evaluated", **dict(mechanics_row or {}),
    }
    payload = {
        "schema": "straintrace.agent-checkpoint/v1", "native_frame": native,
        "frame_position": frame_position, "time_s": record.get("time_s"),
        "reference_native_frame": reference_native_frame,
        "source_path": record["path"], "source_sha256": record["sha256"],
        "run_fingerprint": run_fingerprint, "input_fingerprint": input_sha,
        "code_sha256": code_sha, "status": "failed", "checkpoint": None,
        "parameters": dict(parameters or {}),
        "retryable": bool(retryable), "failure_class": "runtime" if retryable else "deterministic",
        "error": error.as_dict(), "summary_row": row,
    }
    _unlink_with_retry(path.with_suffix(".npz"))
    _atomic_json(path, payload)
    return row


def _iter_checkpoint_metadata(run_root: Path) -> Iterable[dict[str, Any]]:
    for metadata_path in sorted((run_root / "checkpoints").glob("frame_native_*.json")):
        yield _read_json(metadata_path)


def _iter_checkpoint_rows(run_root: Path, *, validate_fingerprint: str | None = None) -> Iterable[tuple[dict[str, np.ndarray], dict[str, Any]]]:
    load_field_fn = _api_attr("_load_field", _load_field)
    for meta in _iter_checkpoint_metadata(run_root):
        if meta.get("status") == "failed":
            yield {}, meta
            continue
        if validate_fingerprint and meta.get("run_fingerprint") != validate_fingerprint:
            raise AgentError("CHECKPOINT_STALE", f"checkpoint belongs to another run: {meta.get('checkpoint')}", path=str(meta.get("checkpoint")))
        checkpoint = run_root / "checkpoints" / str(meta.get("checkpoint"))
        metadata_path = checkpoint.with_suffix(".json")
        arrays, meta = load_field_fn(checkpoint, metadata_path)
        yield arrays, meta


def _field_from_arrays(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> dict[str, Any]:
    field: dict[str, Any] = {key: value for key, value in arrays.items()}
    field["quality_summary"] = dict(metadata.get("quality") or {})
    field["roi"] = tuple(metadata.get("roi_candidate_px") or ())
    field["subset_size"] = (metadata.get("parameters") or {}).get("subset_size_px")
    field["step"] = (metadata.get("parameters") or {}).get("step_px")
    field["strain_window"] = (metadata.get("parameters") or {}).get("strain_window_poi")
    field["provenance"] = {
        "native_frame": metadata.get("native_frame"),
        "reference_native_frame": metadata.get("reference_native_frame"),
        "time_s": metadata.get("time_s"),
    }
    return field


def _region_rows(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    from .regions import summarize_regions

    frame = {
        "frame_position_1based": metadata.get("frame_position"),
        "native_frame": metadata.get("native_frame"),
        "time_s": metadata.get("time_s"),
        "frame_status": metadata.get("status"),
    }
    if arrays:
        field = _field_from_arrays(arrays, metadata)
        field["calibration"] = config.get("calibration")
        rows = summarize_regions(field, config.get("regions") or [], quality_ok=metadata.get("status") == "complete")
    else:
        rows = [
            {
                "region_name": region["name"], "bounds": region["bounds"], "component": component,
                "coordinate_system": region["coordinate_system"],
                "coordinate_unit": "px" if region["coordinate_system"] == "image" else "mm",
                "unit": "dimensionless", "status": "unavailable", "reasons": ["frame_failed"],
                "source": "original_poi", "quality_ok": False, "interpolation": "none",
                "n_poi_total": 0, "n_poi_valid": 0,
            }
            for region in config.get("regions") or [] for component in ("Exx", "Eyy", "Exy")
        ]
    return [{**frame, **row} for row in rows]


def _write_region_outputs(run_root: Path, config: Mapping[str, Any], plan: Mapping[str, Any], run_fingerprint: str) -> None:
    rows = (
        row for arrays, metadata in _iter_checkpoint_rows(run_root, validate_fingerprint=run_fingerprint)
        for row in _region_rows(arrays, metadata, config)
    ) if config.get("regions") else iter(())
    _csv_atomic(run_root / "regions.csv", rows, _REGION_COLUMNS)
    _atomic_json(
        run_root / "geometry.json",
        {
            "schema": "straintrace.geometry/v1", "regions": config.get("regions", []),
            "gauges": config.get("gauges", []), "gauge_reference_geometry": plan.get("gauge_reference_geometry", []),
            "analysis_rectangle_candidate_px": plan.get("analysis_rectangle_candidate_px"),
            "reference_native_frame": plan.get("reference_native_frame"), "reference_path": plan.get("reference_path"),
            "mask_file": config.get("mask_file"), "calibration": config.get("calibration"),
            "specimen_geometry_confirmed": False, "loading_axis": None,
        },
    )


def _refresh_manifest_artifacts(run_root: Path, *, include_report: bool = True) -> None:
    sha256_fn = _api_attr("_sha256", _sha256)
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _read_json(manifest_path)
    files = [run_root / name for name in ("summary.csv", "quality.json", "gauges.csv", "regions.csv", "geometry.json", "report.md")]
    subdirs = ("plots", "fields_csv", "extractions", "report") if include_report else ("plots", "fields_csv", "extractions")
    for subdir in subdirs:
        directory = run_root / subdir
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    manifest["artifacts"] = {path.relative_to(run_root).as_posix(): sha256_fn(path) for path in sorted(set(files)) if path.is_file()}
    _atomic_json(manifest_path, manifest)


def _remove_owned_report_dir(run_root: Path) -> None:
    """Remove a pre-existing owned report directory when report build is skipped."""
    from . import reporting

    resolved_root = run_root.resolve()
    try:
        target = _safe_file(resolved_root, "report")
    except ValueError:
        return
    if not target.exists() or not target.is_dir():
        return
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        schema = _read_json(manifest_path).get("schema")
    except AgentError:
        return
    if schema != reporting.PACKAGE_SCHEMA:
        return
    shutil.rmtree(target)


def _build_report_transaction(
    run_root: Path, config: Mapping[str, Any], plan: Mapping[str, Any], *, include_checkpoints: bool,
) -> dict[str, Any]:
    """Commit the report and its outer hashes together, retaining rollback data."""
    from . import reporting

    refresh_fn = _api_attr("_refresh_manifest_artifacts", _refresh_manifest_artifacts)
    run_root = run_root.resolve()
    target = _safe_file(run_root, "report")
    if target.exists() and _read_json(target / "manifest.json").get("schema") != reporting.PACKAGE_SCHEMA:
        raise AgentError("REPORT_OUTPUT_UNOWNED", "refusing to replace an unrecognized report directory", path=str(target))
    candidate = run_root / f".report-candidate-{uuid.uuid4().hex}"
    backup = run_root / f".report-backup-{uuid.uuid4().hex}"
    manifest_path = run_root / "manifest.json"
    previous_manifest = manifest_path.read_bytes()
    committed = False
    failure: BaseException | None = None
    try:
        result = reporting.build_report(
            run_root, config=config, plan=plan, regions_csv=run_root / "regions.csv",
            output_dir=candidate, include_checkpoints=include_checkpoints,
        )
        if target.exists():
            _replace_with_retry(target, backup)
        try:
            _replace_with_retry(candidate, target)
            try:
                refresh_fn(run_root)
            except BaseException:
                _replace_with_retry(target, candidate)
                if backup.exists():
                    _replace_with_retry(backup, target)
                _atomic_bytes(manifest_path, previous_manifest)
                raise
        except BaseException:
            if backup.exists() and not target.exists():
                _replace_with_retry(backup, target)
            raise
        committed = True
        return _relocate_paths(result, candidate, target)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        # These unique staging paths are owned by this operation. Validate the
        # complete path before recursive cleanup; retain a failed rollback.
        for path in (candidate, backup) if committed else (candidate,):
            try:
                safe_path = _safe_file(run_root, path.name)
                if safe_path.exists():
                    shutil.rmtree(safe_path)
            except OSError as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"Report transaction cleanup failed for {path}: {cleanup_error}")


@contextmanager
def _extraction_transaction(run_root: Path):
    """Publish a complete extraction and its hashes, or retain the old files."""
    refresh_fn = _api_attr("_refresh_manifest_artifacts", _refresh_manifest_artifacts)
    staging_root = run_root / "_postprocess_staging" / f"extract-{uuid.uuid4().hex}"
    staged = staging_root / "new"
    staged.mkdir(parents=True)
    previous_manifest = (run_root / "manifest.json").read_bytes()
    destination_existed = (run_root / "extractions").exists()
    replaced = []
    cleanup = True
    try:
        yield staged
        for source in sorted(staged.iterdir()):
            target = _safe_file(run_root, f"extractions/{source.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = staging_root / "previous" / source.name
            if target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                _replace_with_retry(target, backup)
            replaced.append((target, backup))
            _replace_with_retry(source, target)
        refresh_fn(run_root)
        manifest = _read_json(run_root / "manifest.json")
        stale = manifest.get("stale_derived_artifacts", [])
        if stale:
            replaced_paths = {target.relative_to(run_root).as_posix() for target, _ in replaced}
            manifest["stale_derived_artifacts"] = [path for path in stale if path not in replaced_paths]
            _atomic_json(run_root / "manifest.json", manifest)
    except BaseException:
        cleanup = False
        for target, backup in reversed(replaced):
            target.unlink(missing_ok=True)
            if backup.exists():
                _replace_with_retry(backup, target)
        if replaced:
            _atomic_bytes(run_root / "manifest.json", previous_manifest)
        destination = run_root / "extractions"
        if not destination_existed and destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        cleanup = True
        raise
    finally:
        if cleanup:
            shutil.rmtree(staging_root)


def _plot_selection(run_root: Path, frame_metas: Sequence[Mapping[str, Any]], plot_every: int) -> list[Mapping[str, Any]]:
    successes = [meta for meta in frame_metas if meta.get("status") in {"complete", "quality_failed"} and meta.get("checkpoint")]
    if not successes:
        return []
    if plot_every > 0:
        selected = successes[::plot_every]
        if successes[-1] not in selected:
            selected.append(successes[-1])
        if len(selected) > 24:
            indices = np.linspace(0, len(selected) - 1, 24).round().astype(int)
            selected = [selected[index] for index in sorted(set(indices.tolist()))]
        return selected
    if len(successes) <= 3:
        return successes
    by_native = {int(meta["native_frame"]): meta for meta in successes}
    selected = [successes[0], successes[len(successes) // 2], successes[-1]]
    strain_peak = max(successes, key=lambda meta: (meta.get("summary_row") or {}).get("max_abs_local_exx") or -1.0)
    load_peak = max(successes, key=lambda meta: abs((meta.get("summary_row") or {}).get("stress_MPa") or 0.0))
    selected.extend((strain_peak, load_peak))
    return list({int(meta["native_frame"]): by_native[int(meta["native_frame"])] for meta in selected}.values())


def _plot_run_locked(
    root: Path, run_root: Path, config: Mapping[str, Any], plan: Mapping[str, Any], *,
    plot_every: int, run_fingerprint: str,
) -> dict[str, Any]:
    """Generate consistent-scale field maps with single-pass streaming NPZ loads."""
    from . import fields as field_tools

    load_field_fn = _api_attr("_load_field", _load_field)
    all_metas = list(_iter_checkpoint_metadata(run_root))
    selected = _plot_selection(run_root, all_metas, plot_every)
    plots_root = run_root / "plots"
    plots_root.mkdir(exist_ok=True)
    generated: list[str] = []
    issues: list[dict[str, Any]] = []

    # Pass 1: stream every valid checkpoint ONCE to accumulate min/max limits for all 9 components.
    lows: dict[str, float] = {comp: math.inf for comp in _PLOT_COMPONENTS}
    highs: dict[str, float] = {comp: -math.inf for comp in _PLOT_COMPONENTS}
    component_errors: dict[str, str] = {}
    for metadata in all_metas:
        if metadata.get("status") == "failed" or not metadata.get("checkpoint"):
            continue
        arrays, loaded = load_field_fn(run_root / "checkpoints" / str(metadata["checkpoint"]))
        field = _field_from_arrays(arrays, loaded)
        for component in _PLOT_COMPONENTS:
            if component in component_errors:
                continue
            try:
                values = field_tools.component_values(field, component)
                valid = field_tools._component_valid(field, component, values)
                finite = values[valid]
                if finite.size:
                    lows[component] = min(lows[component], float(np.min(finite)))
                    highs[component] = max(highs[component], float(np.max(finite)))
            except ValueError as exc:
                component_errors[component] = str(exc)

    limits_by_component: dict[str, tuple[float, float]] = {}
    for component in _PLOT_COMPONENTS:
        if component in component_errors:
            issues.append({"component": component, "reason": component_errors[component]})
            continue
        low, high = lows[component], highs[component]
        if not (math.isfinite(low) and math.isfinite(high)):
            issues.append({"component": component, "reason": "no valid finite values are available for color limits"})
            continue
        if component not in {"u", "v", "zncc"}:
            extent = max(abs(low), abs(high))
            low, high = -extent, extent
        if low == high:
            delta = max(abs(low) * 1e-6, 1e-12)
            low, high = low - delta, high + delta
        limits_by_component[component] = (float(low), float(high))

    # Pass 2: stream each selected frame ONCE and render all eligible components for it.
    if limits_by_component and selected:
        import matplotlib.pyplot as plt

        for meta in selected:
            if meta.get("status") == "failed" or not meta.get("checkpoint"):
                continue
            checkpoint = run_root / "checkpoints" / str(meta["checkpoint"])
            arrays, metadata = load_field_fn(checkpoint)
            field = _field_from_arrays(arrays, metadata)
            name = _frame_key(int(meta["native_frame"]))
            title = f"native frame {meta['native_frame']} · {meta.get('status')}"
            for component, color_limits in limits_by_component.items():
                slug = _component_file_slug(component)
                path = plots_root / f"{name}_{slug}.png"
                temp_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
                fig = None
                try:
                    fig, _ax, _artist = field_tools.plot_field_map(
                        field, component, color_limits=color_limits, title=title,
                    )
                    fig.savefig(temp_path, dpi=220, bbox_inches="tight", format="png")
                    _replace_with_retry(temp_path, path)
                    generated.append(path.relative_to(run_root).as_posix())
                except ValueError as exc:
                    issues.append({"native_frame": meta.get("native_frame"), "component": component, "reason": str(exc)})
                finally:
                    if fig is not None:
                        plt.close(fig)
                    try:
                        temp_path.unlink()
                    except FileNotFoundError:
                        pass

    generated_names = {Path(rel).name for rel in generated}
    for existing_plot in plots_root.glob("frame_native_*.png"):
        if existing_plot.name not in generated_names:
            _unlink_with_retry(existing_plot)
    _atomic_json(plots_root / "plot_issues.json", issues)
    history_plots = _plot_standard_histories(run_root)
    _atomic_json(plots_root / "history_summary.json", history_plots)
    return {
        "plot_count": len(generated), "paths": generated,
        "selected_native_frames": [meta.get("native_frame") for meta in selected],
        "plot_issues": issues, "plot_selection_cap": 24, "history_plots": history_plots,
    }


def _plot_standard_histories(run_root: Path) -> dict[str, Any]:
    """Emit sparse instrument and virtual-gauge curves without connecting gaps."""
    table_path = run_root / "summary.csv"
    try:
        table = pd.read_csv(table_path)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"summary_unreadable: {exc}", "plots": []}
    plots_dir = run_root / "plots"
    plots_dir.mkdir(exist_ok=True)

    def values(column: str) -> np.ndarray:
        if column not in table:
            return np.full(len(table), np.nan, dtype=float)
        return pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)

    frame_axis = not np.isfinite(values("time_s")).any()
    history_column = "native_frame" if frame_axis else "time_s"
    history_label = "Native frame" if frame_axis else "Time (s)"

    specifications: list[dict[str, Any]] = []
    for plot_name, x_column, y_column, x_label, y_label, label, source, alignment_column, unavailable_reason in (
        ("time_force.png", "time_s", "force_N", "Time (s)", "Force (N)", "Force", "force_N", "force_alignment_kind", "requires a configured force channel, confirmed clock, and in-support source samples"),
        ("time_machine_displacement.png", "time_s", "machine_displacement_mm", "Time (s)", "Machine displacement from reference (mm)", "Machine displacement", "machine_displacement_mm", "displacement_alignment_kind", "requires a configured displacement channel, confirmed clock, and source support at the frame and reference times"),
        ("time_machine_engineering_strain.png", "time_s", "machine_engineering_strain", "Time (s)", "Machine engineering strain", "Machine engineering strain", "machine_engineering_strain", "displacement_alignment_kind", "requires machine displacement and a user-supplied machine gauge length"),
        ("time_stress.png", "time_s", "stress_MPa", "Time (s)", "Stress (MPa)", "Stress", "stress_MPa", "force_alignment_kind", "requires confirmed force, user-supplied area_mm2, and in-support source samples"),
    ):
        alignment_values = table[alignment_column].astype(str).tolist() if alignment_column in table else []
        source_values = table["machine_displacement_source"].dropna().astype(str).unique().tolist() if y_column.startswith("machine_") and "machine_displacement_source" in table else []
        series_label = f"{label} ({', '.join(source_values)})" if source_values and y_column.startswith("machine_") else label
        specifications.append({
            "plot": plot_name, "x_label": x_label, "y_label": y_label,
            "series": [{"label": series_label, "x": values(x_column), "y": values(y_column), "source": source,
                        "alignment_column": alignment_column, "alignment_values": alignment_values}],
            "unavailable_reason": unavailable_reason,
        })

    gauge_series = []
    for axis in ("x", "y", "distance"):
        column = f"dic_virtual_gauge_engineering_strain_{axis}"
        gauge_series.append({
            "label": f"DIC virtual-gauge engineering strain ({axis})", "x": values(history_column),
            "y": values(column), "source": column, "alignment_column": "quality_status",
            "alignment_values": table["quality_status"].astype(str).tolist() if "quality_status" in table else [],
        })
    specifications.append({
        "plot": ("frame" if frame_axis else "time") + "_dic_virtual_gauge_engineering_strain.png", "x_label": history_label,
        "y_label": "DIC virtual-gauge engineering strain", "series": gauge_series,
        "unavailable_reason": "requires configured fixed-reference DIC virtual-gauge endpoints; no full-field mean is substituted",
    })
    stress_gauge_series = []
    for axis in ("x", "y", "distance"):
        column = f"dic_virtual_gauge_engineering_strain_{axis}"
        stress_gauge_series.append({
            "label": f"DIC virtual-gauge strain ({axis})", "x": values(column),
            "y": values("stress_MPa"), "source": column, "alignment_column": "force_alignment_kind",
            "alignment_values": table["force_alignment_kind"].astype(str).tolist() if "force_alignment_kind" in table else [],
        })
    specifications.append({
        "plot": "stress_vs_dic_virtual_gauge_engineering_strain.png",
        "x_label": "DIC virtual-gauge engineering strain", "y_label": "Stress (MPa)", "series": stress_gauge_series,
        "unavailable_reason": "requires both force/area stress and valid fixed-reference DIC virtual-gauge engineering strain",
    })

    plot_summaries = []
    import matplotlib.pyplot as plt

    for specification in specifications:
        valid_series = []
        summary_series = []
        for series in specification["series"]:
            x = np.asarray(series["x"], dtype=float)
            y = np.asarray(series["y"], dtype=float)
            count = min(x.size, y.size)
            valid = np.isfinite(x[:count]) & np.isfinite(y[:count])
            alignment_values = series.get("alignment_values") or []
            alignment_counts: dict[str, int] = {}
            if alignment_values:
                for item in set(alignment_values):
                    alignment_counts[item] = alignment_values.count(item)
            summary_series.append({
                "label": series["label"], "source_column": series["source"],
                "finite_sample_count": int(valid.sum()), "alignment_counts": alignment_counts,
            })
            if valid.any():
                valid_series.append((series["label"], x[:count][valid], y[:count][valid]))
        path = plots_dir / specification["plot"]
        if not valid_series:
            # A previous export may have had supported measurements. Never
            # leave its old curve beside a newly unavailable result.
            _unlink_with_retry(path)
            plot_summaries.append({"plot": specification["plot"], "status": "unavailable", "reason": specification.get("unavailable_reason", "no finite aligned samples"), "series": summary_series})
            continue
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for label, x, y in valid_series:
            ax.scatter(x, y, s=22, alpha=0.85, label=label, edgecolors="none")
        ax.set_xlabel(specification["x_label"])
        ax.set_ylabel(specification["y_label"])
        ax.grid(True, alpha=0.25)
        if len(valid_series) > 1:
            ax.legend(frameon=False)
        temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
        try:
            fig.savefig(temp, dpi=220, bbox_inches="tight", format="png")
            _replace_with_retry(temp, path)
        finally:
            plt.close(fig)
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        plot_summaries.append({
            "plot": specification["plot"], "status": "generated",
            "path": path.relative_to(run_root).as_posix(),
            "point_count": sum(len(x) for _, x, _ in valid_series),
            "point_rendering": "scatter_no_interframe_lines", "series": summary_series,
        })
    return {"status": "complete", "plots": plot_summaries}


def _update_report_visualizations(run_root: Path, history_summary: Mapping[str, Any] | None, *, plot_count: int | None = None) -> None:
    report_path = run_root / "report.md"
    if not report_path.is_file():
        return
    lines = ["## Standard visualizations", ""]
    if plot_count is not None:
        lines.append(f"- Field maps: {plot_count} PNG file(s) generated for selected native frames.")
    if not history_summary or history_summary.get("status") != "complete":
        reason = (history_summary or {}).get("reason") or "plot generation was not requested"
        lines.append(f"- Time-history, displacement/gauge and stress–DIC curves: unavailable ({reason}).")
    else:
        for item in history_summary.get("plots", []):
            if item.get("status") == "generated":
                lines.append(f"- `{item['plot']}`: generated with {item.get('point_count', 0)} scatter point(s); gaps are not connected.")
            else:
                lines.append(f"- `{item['plot']}`: unavailable ({item.get('reason', 'no finite aligned samples')}).")
    block = "\n".join(lines) + "\n\n"
    text = report_path.read_text(encoding="utf-8")
    marker = "## Standard visualizations"
    start = text.find(marker)
    if start >= 0:
        end = text.find("## Selected solver parameters", start)
        if end >= 0:
            text = text[:start] + text[end:]
    insertion = text.find("## Selected solver parameters")
    if insertion >= 0:
        text = text[:insertion] + block + text[insertion:]
    else:
        text = text.rstrip() + "\n\n" + block
    _atomic_bytes(report_path, text.encode("utf-8"))


def _build_outputs(
    root: Path, run_root: Path, config: Mapping[str, Any], plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
    mechanics_status: Mapping[str, Any], run_fingerprint: str, input_sha: str,
    code_sha: Mapping[str, str], *, field_csv: bool, plots: bool,
    include_checkpoints: bool = False, build_report: bool = True, invalidate_extractions: bool = False,
) -> dict[str, Any]:
    sha256_fn = _api_attr("_sha256", _sha256)
    refresh_fn = _api_attr("_refresh_manifest_artifacts", _refresh_manifest_artifacts)
    run_state = _read_json(run_root / "run_state.json")
    prior_manifest_path = run_root / "manifest.json"
    prior_manifest = _read_json(prior_manifest_path) if prior_manifest_path.is_file() else {}
    stale_artifacts = set(prior_manifest.get("stale_derived_artifacts", []))
    if invalidate_extractions:
        stale_artifacts.update(
            path for path in (prior_manifest.get("artifacts") or {})
            if isinstance(path, str) and path.startswith("extractions/")
        )
    expected_native_frames = [int(record["native_frame"]) for record in records]
    computed_rows = [row for row in rows if row.get("status") != "pending"]
    try:
        row_native_frames = [int(row["native_frame"]) for row in computed_rows]
    except (KeyError, TypeError, ValueError):
        row_native_frames = []
    native_frames_match_expected = row_native_frames == expected_native_frames
    run_status = run_state.get("status", "unknown")
    expected_frame_count = len(records)
    completed_frame_count = len(computed_rows)
    summary_columns = [
        "frame_position_1based", "native_frame", "time_s", "reference_native_frame", "status", "quality_status", "scientific_ok",
        "eligible_point_count", "correlation_valid_fraction", "strain_valid_fraction", "max_abs_local_exx",
        "dic_virtual_gauge_engineering_strain", "dic_virtual_gauge_engineering_strain_sd", "dic_virtual_gauge_true_strain",
        "dic_virtual_gauge_valid_count", "dic_virtual_gauge_total_count",
        "dic_virtual_gauge_engineering_strain_x", "dic_virtual_gauge_engineering_strain_sd_x", "dic_virtual_gauge_valid_count_x", "dic_virtual_gauge_true_strain_x",
        "dic_virtual_gauge_engineering_strain_y", "dic_virtual_gauge_engineering_strain_sd_y", "dic_virtual_gauge_valid_count_y", "dic_virtual_gauge_true_strain_y",
        "dic_virtual_gauge_engineering_strain_distance", "dic_virtual_gauge_engineering_strain_sd_distance", "dic_virtual_gauge_valid_count_distance", "dic_virtual_gauge_true_strain_distance",
        "engineering_strain",
        "force_N", "stress_MPa", "mechanics_status", "force_alignment_kind",
        "force_bracket_left_time_s", "force_bracket_right_time_s", "force_bracket_left_source_row", "force_bracket_right_source_row",
        "machine_displacement_mm", "machine_engineering_strain", "machine_true_strain", "machine_displacement_source",
        "displacement_alignment_kind", "displacement_bracket_left_time_s", "displacement_bracket_right_time_s",
        "displacement_bracket_left_source_row", "displacement_bracket_right_source_row",
        "displacement_reference_alignment_kind", "displacement_reference_bracket_left_time_s", "displacement_reference_bracket_right_time_s",
        "displacement_reference_bracket_left_source_row", "displacement_reference_bracket_right_source_row",
        "failure_code", "failure_reason", "checkpoint",
    ]
    _csv_atomic(run_root / "summary.csv", rows, summary_columns)
    numerical_quality_passed = (
        run_status == "complete"
        and expected_frame_count > 0
        and completed_frame_count == expected_frame_count
        and native_frames_match_expected
        and all(row.get("status") == "complete" and row.get("scientific_ok") is True for row in computed_rows)
    )
    quality = {
        "schema": "straintrace.agent-quality/v1", "scientific_ok": numerical_quality_passed,
        "numerical_quality_passed": numerical_quality_passed,
        "run_status": run_status,
        "expected_frame_count": expected_frame_count,
        "completed_frame_count": completed_frame_count,
        "native_frames_match_expected": native_frames_match_expected,
        "scientific_ok_scope": QUALITY_GATE_SCOPE,
        "scientific_acceptance": "not_assessed", "experimental_uncertainty_calibrated": False,
        "quality_gate_scope": QUALITY_GATE_SCOPE,
        "frame_count": completed_frame_count, "passed_frame_count": sum(row.get("scientific_ok") is True for row in computed_rows),
        "failed_frame_count": sum(row.get("scientific_ok") is not True for row in computed_rows),
        "threshold_policy": "existing ezdic_core field_quality_summary thresholds; never relaxed by Agent",
        "aggregate_engineering_strain": None,
        "mechanics_alignment": _json_safe(dict(mechanics_status)),
        "frames": [],
    }
    for metadata in _iter_checkpoint_metadata(run_root):
        quality["frames"].append({
            "native_frame": metadata.get("native_frame"), "status": metadata.get("status"),
            "quality": metadata.get("quality"), "failure": metadata.get("error"),
        })
    _atomic_json(run_root / "quality.json", quality)
    gauge_csv = run_root / "gauges.csv"
    gauge_tmp = gauge_csv.with_name(f".{gauge_csv.name}.{uuid.uuid4().hex}.tmp")
    try:
        with gauge_tmp.open("w", encoding="utf-8-sig", newline="") as stream:
            gauge_writer = csv.DictWriter(stream, fieldnames=_GAUGE_COLUMNS, extrasaction="ignore")
            gauge_writer.writeheader()
            for metadata in _iter_checkpoint_metadata(run_root):
                summary = metadata.get("summary_row") or {}
                for gauge_row in metadata.get("gauge_measurements") or []:
                    gauge_writer.writerow({
                        "frame_position_1based": summary.get("frame_position_1based"),
                        "native_frame": metadata.get("native_frame"),
                        "time_s": metadata.get("time_s"),
                        **gauge_row,
                    })
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(gauge_tmp, gauge_csv)
    finally:
        try:
            gauge_tmp.unlink()
        except FileNotFoundError:
            pass
    fields_dir = run_root / "fields_csv"
    # Once field CSVs have been requested, keep them consistent when a later
    # run recomputes or fails a frame, even if this invocation did not request
    # an additional export.
    refresh_existing_fields = fields_dir.is_dir() and any(
        (fields_dir / f"{_frame_key(int(record['native_frame']))}.csv").is_file()
        for record in records
    )
    if field_csv or refresh_existing_fields:
        import ezdic_core as core

        fields_dir.mkdir(exist_ok=True)
        generated_fields: set[str] = set()
        for arrays, metadata in _iter_checkpoint_rows(run_root, validate_fingerprint=run_fingerprint):
            if not arrays:
                continue
            field = _field_from_arrays(arrays, metadata)
            path = fields_dir / f"{_frame_key(int(metadata['native_frame']))}.csv"
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                core.dic_field_to_dataframe(field).to_csv(temporary, index=False)
                _replace_with_retry(temporary, path)
                generated_fields.add(path.name)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        for record in records:
            expected = fields_dir / f"{_frame_key(int(record['native_frame']))}.csv"
            if expected.name not in generated_fields:
                _unlink_with_retry(expected)
    plot_result = None
    history_summary = None
    if plots:
        plot_result = _plot_run_locked(root, run_root, config, plan, plot_every=int(config.get("plot_every", 0)), run_fingerprint=run_fingerprint)
        history_summary = plot_result.get("history_plots")
    elif (run_root / "plots" / "history_summary.json").is_file():
        history_summary = _read_json(run_root / "plots" / "history_summary.json")
    report_lines = [
        "# StrainTrace Agent analysis report", "",
        f"- Project ID: `{config['project_id']}`", f"- Run: `{run_root.name}`",
        f"- Reference native frame: `{plan['reference_native_frame']}`",
        f"- Analysis rectangle candidate (px): `{plan['analysis_rectangle_candidate_px']}`",
        "- The rectangle is a texture-based analysis candidate; physical specimen geometry is not confirmed.",
        f"- Run execution status: `{run_status}`",
        f"- Frame coverage: `{completed_frame_count}/{expected_frame_count}` expected native frames; identity matches: `{native_frames_match_expected}`.",
        f"- Core-quality frames among completed rows: `{quality['passed_frame_count']}/{completed_frame_count}`",
        f"- Core numerical quality gates: `{'PASS' if quality['numerical_quality_passed'] else 'NOT PASSED'}`",
        "- Scientific acceptance: not assessed; a numerical quality-gate pass is not experimental validation.",
        f"- Load alignment: `{mechanics_status.get('status')}`",
        "- DIC virtual-gauge engineering strain is emitted only for configured valid endpoints; machine displacement strain is reported separately with its channel source.",
        "- Stress is force_N / user-confirmed area_mm2. Missing clock alignment or area leaves dependent outputs unavailable.",
        "- Machine-channel `alignment_kind` distinguishes exact samples from bounded linear interpolation; bracket times/source rows are in `summary.csv`, and the source CSV hash is in `manifest.json`.",
        "- Local strain fit span is `(strain_window_poi - 1) * step_px`; full support span including subset is `subset_size_px + (strain_window_poi - 1) * step_px`.",
        "- This is in-plane 2D DIC. Out-of-plane motion correction and experimental uncertainty calibration are not provided.",
        "- Invalid POIs and unsupported regions remain NaN; no interpolation is performed by this workflow.", "",
        "## Selected solver parameters", "",
        "```json", json.dumps(plan["selected_parameters"], indent=2, ensure_ascii=False, allow_nan=False), "```", "",
        "See `summary.csv`, `quality.json`, per-frame checkpoint sidecars, and `manifest.json` for machine-readable evidence.",
    ]
    _atomic_bytes(run_root / "report.md", ("\n".join(report_lines) + "\n").encode("utf-8"))
    _update_report_visualizations(run_root, history_summary, plot_count=plot_result.get("plot_count") if plot_result else None)
    artifact_paths = [run_root / name for name in ("summary.csv", "quality.json", "report.md")]
    artifact_paths.append(gauge_csv)
    artifact_paths.extend(path for path in sorted((run_root / "plots").rglob("*")) if path.is_file())
    artifact_paths.extend(sorted((run_root / "extractions").rglob("*")) if (run_root / "extractions").exists() else [])
    if field_csv:
        artifact_paths.extend(sorted((run_root / "fields_csv").glob("*.csv")))
    manifest = {
        "schema": "straintrace.agent-manifest/v1", "project_id": config["project_id"],
        "run_id": run_root.name, "run_fingerprint": run_fingerprint,
        "run_status": run_status,
        "expected_frame_count": expected_frame_count,
        "completed_frame_count": completed_frame_count,
        "native_frames_match_expected": native_frames_match_expected,
        "input_fingerprint": input_sha, "configuration_sha256": _hash_json(config),
        "plan_sha256": plan.get("plan_sha256"), "code_sha256": code_sha,
        "source_images": [{"native_frame": row["native_frame"], "path": row["path"], "sha256": row["sha256"], "time_s": row["time_s"]} for row in records],
        "load_file_sha256": sha256_fn(Path(config["loads"]["path"])) if config.get("loads") else None,
        "mask_sha256": config.get("mask_sha256"),
        "time_basis": config.get("time_basis", "physical_time"),
        "environment": __import__("ezdic_core").collect_environment(),
        "artifacts": {path.relative_to(run_root).as_posix(): sha256_fn(path) for path in artifact_paths if path.is_file()},
        "checkpoint_receipts": {
            metadata_path.name: {
                "sha256": sha256_fn(metadata_path),
                "checkpoint_sha256": _read_json(metadata_path).get("checkpoint_sha256"),
                "native_frame": _read_json(metadata_path).get("native_frame"),
                "status": _read_json(metadata_path).get("status"),
            }
            for metadata_path in sorted((run_root / "checkpoints").glob("frame_native_*.json"))
        },
        "checkpoint_count": sum(1 for metadata in _iter_checkpoint_metadata(run_root) if metadata.get("checkpoint_sha256") is not None),
        "checkpoint_failures": sum(1 for path in (run_root / "checkpoints").glob("frame_native_*.json") if _read_json(path).get("status") == "failed"),
        "scientific_ok": quality["scientific_ok"], "numerical_quality_passed": quality["numerical_quality_passed"],
        "stale_derived_artifacts": sorted(stale_artifacts),
        "quality_gate_scope": QUALITY_GATE_SCOPE, "scientific_acceptance": "not_assessed",
        "experimental_uncertainty_calibrated": False,
        "units": {"x_px": "px", "u_px": "px", "strain": "dimensionless", "force_N": "N", "stress_MPa": "MPa"},
    }
    _atomic_json(run_root / "manifest.json", manifest)
    _write_region_outputs(run_root, config, plan, run_fingerprint)
    portable_report = None
    if build_report:
        refresh_fn(run_root)
        portable_report = _build_report_transaction(run_root, config, plan, include_checkpoints=include_checkpoints)
    else:
        _remove_owned_report_dir(run_root)
        try:
            refresh_fn(run_root, include_report=False)
        except TypeError:
            refresh_fn(run_root)
    artifacts: dict[str, Any] = {
        "summary": str(run_root / "summary.csv"),
        "gauges": str(gauge_csv),
        "regions": str(run_root / "regions.csv"),
        "geometry": str(run_root / "geometry.json"),
        "quality": str(run_root / "quality.json"),
        "manifest": str(run_root / "manifest.json"),
        "report": str(run_root / "report.md"),
        "report_html": portable_report["report_html"] if portable_report else None,
        "plots": str(run_root / "plots"),
    }
    result: dict[str, Any] = {"quality": quality, "artifacts": artifacts, "stale_derived_artifacts": sorted(stale_artifacts)}
    if portable_report is not None:
        result["portable_report"] = portable_report
    return result
