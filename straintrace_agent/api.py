"""Streaming, provenance-bound 2D DIC automation API.

The package calls the established ezdic_core solver one frame at a time. It
keeps source images read-only and stores numeric/string NPZ checkpoints without
pickle so a stopped run can resume without retaining the full image sequence.

Implementation helpers are decomposed across cohesive internal modules
(``_io``, ``_locks``, ``_config``, ``_inventory``, ``_outputs``, ``_verify``)
while this module remains the stable public orchestration facade.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import inspect
import multiprocessing
import os
from pathlib import Path
import re
import time
import uuid
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ._config import (
    PROJECT_SCHEMA as PROJECT_SCHEMA,
    QUALITY_GATE_SCOPE as QUALITY_GATE_SCOPE,
    RUN_SCHEMA as RUN_SCHEMA,
    _CORE_SOLVER_PARAMETER_KEYS as _CORE_SOLVER_PARAMETER_KEYS,
    _config_from_arguments as _config_from_arguments,
    _config_init_options as _config_init_options,
    _ensure_known_fields as _ensure_known_fields,
    _positive_number as _positive_number,
    _read_config as _read_config,
    _solver_parameters_match as _solver_parameters_match,
    _validate_gauge as _validate_gauge,
    _validate_plan_checksum as _validate_plan_checksum,
    _validate_project_config as _validate_project_config,
    _validate_rect as _validate_rect,
    read_project_config as read_project_config,
)
from ._inventory import (
    IMAGE_SUFFIXES as IMAGE_SUFFIXES,
    _FRAME_TOKEN as _FRAME_TOKEN,
    _automatic_reference as _automatic_reference,
    _code_fingerprint as _code_fingerprint,
    _explicit_frame_map as _explicit_frame_map,
    _inventory_fingerprint as _inventory_fingerprint,
    _inventory_fresh as _inventory_fresh,
    _load_image as _load_image,
    _load_specimen_mask as _load_specimen_mask,
    _native_frame_from_name as _native_frame_from_name,
    _pilot_frame as _pilot_frame,
    _record_inventory as _record_inventory,
    _reference_record as _reference_record,
    _safe_roi_for_image as _safe_roi_for_image,
    _scan_inventory as _scan_inventory,
    _texture_roi as _texture_roi,
    load_reference_frame as load_reference_frame,
    parse_frame_map as parse_frame_map,
    parse_native_frame as parse_native_frame,
    resolve_time_basis as resolve_time_basis,
)
from ._io import (
    AgentError as AgentError,
    _REPLACE_RETRY_DELAYS_S as _REPLACE_RETRY_DELAYS_S,
    _RunCancelled as _RunCancelled,
    _atomic_bytes as _atomic_bytes,
    _atomic_json as _atomic_json,
    _canonical as _canonical,
    _component_file_slug as _component_file_slug,
    _csv_atomic as _csv_atomic,
    _current_pointer as _current_pointer,
    _hash_json as _hash_json,
    _json_safe as _json_safe,
    _load_run_pointer as _load_run_pointer,
    _plan_path as _plan_path,
    _project_root as _project_root,
    _read_json as _read_json,
    _relocate_paths as _relocate_paths,
    _replace_with_retry as _replace_with_retry,
    _sha256 as _sha256,
    _unlink_with_retry as _unlink_with_retry,
    _write_progress as _write_progress,
)
from ._locks import (
    _pid_state as _pid_state,
    _process_start_token as _process_start_token,
    _project_lock as _project_lock,
    _read_lock_details as _read_lock_details,
    recover_lock as recover_lock,
)
from ._outputs import (
    _GAUGE_COLUMNS as _GAUGE_COLUMNS,
    _PLOT_COMPONENTS as _PLOT_COMPONENTS,
    _REGION_COLUMNS as _REGION_COLUMNS,
    _build_outputs as _build_outputs,
    _build_report_transaction as _build_report_transaction,
    _checkpoint_valid as _checkpoint_valid,
    _displacement_reference as _displacement_reference,
    _extraction_transaction as _extraction_transaction,
    _field_from_arrays as _field_from_arrays,
    _frame_failure as _frame_failure,
    _frame_key as _frame_key,
    _iter_checkpoint_metadata as _iter_checkpoint_metadata,
    _iter_checkpoint_rows as _iter_checkpoint_rows,
    _load_field as _load_field,
    _plot_run_locked as _plot_run_locked,
    _plot_selection as _plot_selection,
    _plot_standard_histories as _plot_standard_histories,
    _refresh_manifest_artifacts as _refresh_manifest_artifacts,
    _region_rows as _region_rows,
    _run_fingerprint as _run_fingerprint,
    _serialize_field as _serialize_field,
    _update_report_visualizations as _update_report_visualizations,
    _write_region_outputs as _write_region_outputs,
)
from ._verify import (
    _require_current_run_binding as _require_current_run_binding,
    _require_recorded_run_integrity as _require_recorded_run_integrity,
    _require_retained_artifacts_integrity as _require_retained_artifacts_integrity,
    verify as verify,
)
from .mechanics import (
    _auto_gauge_candidates as _auto_gauge_candidates,
    _gauge_group_summary as _gauge_group_summary,
    _interpolate_bounded_detail as _interpolate_bounded_detail,
    _load_mechanics as _load_mechanics,
    _measure_gauges as _measure_gauges,
    _mechanics_at as _mechanics_at,
    _resolve_gauge_geometry as _resolve_gauge_geometry,
)


def init_project(
    project: str | os.PathLike[str], image_dir: str | os.PathLike[str], *,
    frame_interval_s: float | None = None, frame_map_csv: str | os.PathLike[str] | None = None,
    reference_native_frame: int | None = None, loads_csv: str | os.PathLike[str] | None = None,
    load_time_column: str | None = None, load_force_column: str | None = None,
    load_time_unit: str | None = None, force_unit: str | None = None,
    displacement_column: str | None = None, displacement_unit: str | None = None,
    displacement_source: str | None = None, machine_gauge_length_mm: float | None = None,
    clock_confirmed: bool = False, load_time_offset_s: float | None = None,
    max_gap_s: float | None = None, area_mm2: float | None = None,
    calibration: Mapping[str, Any] | None = None,
    regions: Sequence[Mapping[str, Any]] | None = None,
    paths: Sequence[Mapping[str, Any]] | None = None,
    gauges: Sequence[Mapping[str, Any]] | None = None,
    mask_file: str | os.PathLike[str] | None = None,
    roi_override_px: Sequence[int] | None = None,
    exclusions_px: Sequence[Sequence[int]] | None = None,
    step_hint_px: int | None = None,
    plot_every: int = 0,
    time_basis: str = "physical_time",
) -> dict[str, Any]:
    """Create a new sample project and hash every source image once."""
    if not isinstance(project, (str, os.PathLike)) or not str(project).strip():
        raise AgentError("CONFIG_TYPE_ERROR", "project must be a non-empty path string", path="$.project")
    if not isinstance(image_dir, (str, os.PathLike)) or not str(image_dir).strip():
        raise AgentError("CONFIG_TYPE_ERROR", "image_dir must be a non-empty path string", path="$.image_dir")
    root = _project_root(project)
    if root.exists() and any(root.iterdir()):
        raise AgentError("PROJECT_ALREADY_EXISTS", f"refusing to overwrite a nonempty project: {root}", path=str(root))
    image_root = Path(image_dir).expanduser().resolve(strict=False)
    try:
        root.relative_to(image_root)
    except ValueError:
        pass
    else:
        raise AgentError("PROJECT_INSIDE_IMAGE_DIRECTORY", "project output must be outside the raw image directory", path=str(root))
    config = _config_from_arguments(
        image_dir=image_root, frame_interval_s=frame_interval_s, frame_map_csv=frame_map_csv,
        reference_native_frame=reference_native_frame, loads_csv=loads_csv,
        load_time_column=load_time_column, load_force_column=load_force_column,
        load_time_unit=load_time_unit, force_unit=force_unit,
        displacement_column=displacement_column, displacement_unit=displacement_unit,
        displacement_source=displacement_source, machine_gauge_length_mm=machine_gauge_length_mm,
        clock_confirmed=clock_confirmed,
        load_time_offset_s=load_time_offset_s, max_gap_s=max_gap_s, area_mm2=area_mm2,
        calibration=calibration, regions=regions, paths=paths, gauges=gauges,
        mask_file=mask_file, roi_override_px=roi_override_px, exclusions_px=exclusions_px,
        step_hint_px=step_hint_px,
        plot_every=plot_every,
        time_basis=time_basis,
    )
    config = _validate_project_config(config, root)
    records = _scan_inventory(config)
    input_fingerprint = _inventory_fingerprint(config, records)
    root.mkdir(parents=True, exist_ok=True)
    with _project_lock(root, purpose="init"):
        if any(entry.name != ".agent.lock" and not entry.name.startswith(".agent.lock.") for entry in root.iterdir()):
            raise AgentError("PROJECT_ALREADY_EXISTS", f"refusing to overwrite a nonempty project: {root}", path=str(root))
        (root / "plans").mkdir(exist_ok=True)
        (root / "runs").mkdir(exist_ok=True)
        _atomic_json(root / "project.json", config)
        _record_inventory(root, records, input_fingerprint)
        _atomic_json(root / "CURRENT.json", {"schema": PROJECT_SCHEMA, "project_id": config["project_id"], "current_plan": None, "current_run": None})
    return {"project": str(root), "project_id": config["project_id"], "frame_count": len(records), "readable_count": sum(row["status"] == "pending" for row in records), "inventory_sha256": _hash_json(records)}


def inspect_project(project: str | os.PathLike[str]) -> dict[str, Any]:
    """Read-only summary of recorded inputs and analysis availability."""
    root = _project_root(project)
    config = _read_config(root)
    inventory = _read_json(root / "inventory.json")
    pointer = _read_json(root / "CURRENT.json")
    frames = inventory.get("frames", [])
    return {
        "project": str(root), "project_id": config.get("project_id"),
        "frame_count": len(frames), "readable_count": sum(row.get("status") == "pending" for row in frames),
        "unreadable_frames": [row.get("native_frame") for row in frames if row.get("status") != "pending"],
        "native_frames": [row.get("native_frame") for row in frames],
        "time_range_s": [frames[0].get("time_s"), frames[-1].get("time_s")] if frames else None,
        "image_shape_yx": next((row.get("shape_yx") for row in frames if row.get("shape_yx")), None),
        "frame_interval_s": config.get("frame_interval_s"),
        "time_basis": config.get("time_basis", "physical_time"),
        "load_data_configured": config.get("loads") is not None,
        "load_clock_confirmed": bool((config.get("loads") or {}).get("clock_confirmed", False)),
        "stress_available_in_principle": bool((config.get("loads") or {}).get("clock_confirmed") and (config.get("loads") or {}).get("area_mm2")),
        "physical_calibration_available": config.get("calibration") is not None,
        "analysis_rectangle_status": "candidate_not_physical_specimen_confirmation",
        "current_plan": pointer.get("current_plan"), "current_run": pointer.get("current_run"),
    }


def plan_project(project: str | os.PathLike[str], *, progress: Callable[[dict[str, Any]], None] | None = None, force: bool = False) -> dict[str, Any]:
    """Propose a texture ROI and choose among a small, fixed solver trial set."""
    root = _project_root(project)
    config = _read_config(root)
    with _project_lock(root, purpose="plan"):
        records, input_sha = _inventory_fresh(root, config)
        reference_record, reference, reference_selection = _automatic_reference(config, records)
        detected_roi_list = reference_selection.pop("detected_roi_px", None)
        texture = {k: v for k, v in reference_selection.items() if k not in {"selection", "initial_inventory_frame"}}
        if config.get("roi_override_px"):
            roi = tuple(config["roi_override_px"])
            texture = {**texture, "selected_by": "explicit_user_roi_override"}
        else:
            roi = tuple(detected_roi_list)
        x, y, width, height = roi
        specimen_mask = _load_specimen_mask(config, reference.shape)
        if specimen_mask is not None and not config.get("roi_override_px"):
            supported_rows = np.flatnonzero(np.any(specimen_mask, axis=1))
            supported_cols = np.flatnonzero(np.any(specimen_mask, axis=0))
            if supported_cols.size and supported_rows.size:
                mx0, my0 = int(supported_cols[0]), int(supported_rows[0])
                mx1, my1 = int(supported_cols[-1]) + 1, int(supported_rows[-1]) + 1
                x0, y0 = max(x, mx0), max(y, my0)
                x1, y1 = min(x + width, mx1), min(y + height, my1)
                if x1 > x0 and y1 > y0:
                    roi = (x0, y0, x1 - x0, y1 - y0)
                    x, y, width, height = roi
                    texture = {**texture, "selected_by": "intersection_of_texture_and_user_mask_bounds"}
        if x + width > reference.shape[1] or y + height > reference.shape[0]:
            raise AgentError("ROI_OUTSIDE_IMAGE", f"analysis rectangle {roi} exceeds reference image shape {reference.shape}")
        if width < 35 or height < 35:
            raise AgentError("ROI_TOO_SMALL", f"texture analysis candidate {roi} cannot support a 3x3 POI field")
        pilot_record, pilot_problem = _pilot_frame(records, reference_record, reference_shape=reference.shape)
        candidates: list[dict[str, Any]] = []
        pilot_roi: tuple[int, int, int, int] | None = None
        pilot_rois_by_subset: dict[int, tuple[int, int, int, int]] = {}
        import ezdic_core as core

        if pilot_record is not None:
            try:
                pilot = _load_image(pilot_record["path"])
                pilot_w = min(width, 160)
                pilot_h = min(height, 160)
                px = x + max(0, (width - pilot_w) // 2)
                py = y + max(0, (height - pilot_h) // 2)
                base_pilot_roi = (px, py, pilot_w, pilot_h)
                pilot_step = int(config.get("step_hint_px") or 8)
                candidate_specs: list[tuple[int, int, tuple[int, int, int, int]]] = []
                for size in (21, 31):
                    sr = max(8, size // 2)
                    cropped = _safe_roi_for_image(base_pilot_roi, reference.shape, search_radius_px=sr)
                    if cropped[2] >= size + 2 * pilot_step and cropped[3] >= size + 2 * pilot_step:
                        candidate_specs.append((size, sr, cropped))
                if not candidate_specs:
                    sr = max(8, 15 // 2)
                    cropped = _safe_roi_for_image(base_pilot_roi, reference.shape, search_radius_px=sr)
                    if cropped[2] >= 15 + 2 * pilot_step and cropped[3] >= 15 + 2 * pilot_step:
                        candidate_specs.append((15, sr, cropped))
                for subset, search_radius, cropped_roi in candidate_specs[:2]:
                    pilot_roi = cropped_roi
                    pilot_rois_by_subset[int(subset)] = cropped_roi
                    try:
                        field = core.run_2d_dic(
                            reference, pilot, cropped_roi, subset_size=subset, step=pilot_step,
                            solver=core.DIC_SOLVER_ICGN, strain_window=5,
                            search_radius=search_radius,
                            specimen_mask=specimen_mask,
                            reject_nonconverged=True,
                        )
                        quality = field.get("quality_summary") or core.field_quality_summary(field)
                        candidates.append({
                            "subset_size_px": int(subset), "step_px": pilot_step, "strain_window_poi": 5,
                            "solver": core.DIC_SOLVER_ICGN,
                            "reject_nonconverged": True,
                            "scientific_ok": bool(quality.get("scientific_ok", False)),
                            "correlation_valid_fraction": float(quality.get("correlation_valid_fraction", 0.0)),
                            "strain_valid_fraction": float(quality.get("strain_valid_fraction", 0.0)),
                            "quality_reasons": list(quality.get("scientific_reasons") or []),
                            "thresholds": quality.get("thresholds"),
                        })
                    except Exception as exc:
                        candidates.append({
                            "subset_size_px": int(subset), "step_px": pilot_step, "strain_window_poi": 5,
                            "solver": core.DIC_SOLVER_ICGN, "reject_nonconverged": True, "scientific_ok": False,
                            "correlation_valid_fraction": 0.0, "strain_valid_fraction": 0.0,
                            "quality_reasons": ["pilot_error"], "error": f"{type(exc).__name__}: {exc}",
                        })
                pilot_problem = None if candidates else "no candidate subset fits a 3x3 pilot grid"
            except AgentError as exc:
                pilot_problem = exc.message
        if candidates:
            chosen = max(candidates, key=lambda item: (bool(item["scientific_ok"]), min(item["correlation_valid_fraction"], item["strain_valid_fraction"]), -item["subset_size_px"]))
            pilot_roi = pilot_rois_by_subset.get(int(chosen["subset_size_px"]), pilot_roi)
            plan_status = "ready" if chosen["scientific_ok"] else "review_required"
        else:
            fallback_step = int(config.get("step_hint_px") or 8)
            fallback_subset = 21
            if min(width, height) < fallback_subset + 2 * fallback_step:
                fallback_subset = 15
                fallback_step = max(1, min(fallback_step, (min(width, height) - fallback_subset) // 2))
            chosen = {"subset_size_px": fallback_subset, "step_px": fallback_step, "strain_window_poi": 5, "solver": core.DIC_SOLVER_ICGN}
            plan_status = "review_required"
        chosen["search_radius_px"] = max(8, int(chosen["subset_size_px"]) // 2)
        chosen["reject_nonconverged"] = True
        roi_before_support_crop = roi
        roi = _safe_roi_for_image(roi, reference.shape, search_radius_px=int(chosen["search_radius_px"]))
        x, y, width, height = roi
        if width <= 0 or height <= 0:
            raise AgentError("ROI_TOO_SMALL", f"analysis rectangle {roi_before_support_crop} leaves no interior support after border margin")
        if width < int(chosen["subset_size_px"]) + 2 * int(chosen["step_px"]) or height < int(chosen["subset_size_px"]) + 2 * int(chosen["step_px"]):
            plan_status = "review_required"
        gauge_reference_geometry = _resolve_gauge_geometry(
            config.get("gauges") or [], roi, reference.shape,
            int(chosen["subset_size_px"]), int(chosen["step_px"]), specimen_mask,
            calibration=config.get("calibration"),
        )
        if any(item.get("reference_status") != "resolved" for item in gauge_reference_geometry):
            plan_status = "review_required"
        chosen_subset = int(chosen["subset_size_px"])
        chosen_step = int(chosen["step_px"])
        plan = {
            "schema": "straintrace.agent-plan/v1", "plan_id": uuid.uuid4().hex,
            "created_unix_s": time.time(), "project_id": config["project_id"], "status": plan_status,
            "input_fingerprint": input_sha, "configuration_sha256": _hash_json(config), "code_sha256": _code_fingerprint(),
            "reference_native_frame": int(reference_record["native_frame"]),
            "reference_path": reference_record["path"],
            "reference_selection": reference_selection,
            "analysis_rectangle_candidate_px": list(roi), "texture_roi_bbox_px": list(roi_before_support_crop),
            "analysis_rectangle_status": "candidate_not_physical_specimen_confirmation",
            "specimen_geometry_confirmed": False, "roi_detection": texture,
            "mask_file": config.get("mask_file"), "mask_sha256": config.get("mask_sha256"),
            "exclusions_px": config.get("exclusions_px", []),
            "pilot_native_frame": int(pilot_record["native_frame"]) if pilot_record else None,
            "pilot_rectangle_px": list(pilot_roi) if pilot_roi is not None else None,
            "pilot_issue": pilot_problem, "candidate_trials": candidates,
            "automatic_gauge_candidates": _auto_gauge_candidates(roi, chosen_subset, chosen_step),
            "gauge_reference_geometry": gauge_reference_geometry,
            "selected_parameters": chosen,
            "parameter_decision": "bounded central-pilot comparison; existing core thresholds unchanged; no threshold relaxation",
            "strain_support_span_px": int((int(chosen["strain_window_poi"]) - 1) * chosen_step),
            "subset_support_span_px": int(chosen_subset + (int(chosen["strain_window_poi"]) - 1) * chosen_step),
            "correlation_search_radius_px": int(chosen["search_radius_px"]),
            "environment": core.collect_environment(),
        }
        from .regions import recommend_regions

        plan["region_recommendations"] = recommend_regions(reference, chosen_subset, chosen_step, specimen_mask=specimen_mask, analysis_roi=roi)
        plan_sha = _hash_json(plan)
        plan["plan_sha256"] = plan_sha
        plans_root = root / "plans"
        plans_root.mkdir(exist_ok=True)
        existing_versions = [
            int(match.group(1))
            for existing_plan in plans_root.glob("plan_v*.json")
            if (match := re.fullmatch(r"plan_v(\d+)\.json", existing_plan.name)) is not None
        ]
        version = (max(existing_versions) if existing_versions else 0) + 1
        plan_file = plans_root / f"plan_v{version:03d}.json"
        _atomic_json(plan_file, plan)
        pointer = _current_pointer(root)
        _record_inventory(root, records, input_sha)
        pointer["current_plan"] = plan_file.relative_to(root).as_posix()
        pointer["current_run"] = None
        _atomic_json(root / "CURRENT.json", pointer)
        if progress:
            progress({"event": "plan_complete", "status": plan_status, "candidate_count": len(candidates), "plan": str(plan_file)})
        return plan


def run(
    project: str | os.PathLike[str], *, progress: Callable[[dict[str, Any]], None] | None = None,
    make_plots: bool = True, field_csv: bool = False, build_report: bool = True,
    report_include_checkpoints: bool = False, build_report_on_cancel: bool | None = None,
) -> dict[str, Any]:
    """Stream fixed-reference frame calculations and rebuild deliverables."""
    cancel_build_report = build_report if build_report_on_cancel is None else build_report_on_cancel
    root = _project_root(project)
    config = _read_config(root)
    with _project_lock(root, purpose="run"):
        records, input_sha = _inventory_fresh(root, config)
        pointer = _current_pointer(root)
        plan_path = _plan_path(root, pointer)
        plan = _read_json(plan_path)
        _validate_plan_checksum(plan)
        code_sha = _code_fingerprint()
        if plan.get("input_fingerprint") != input_sha:
            raise AgentError("PLAN_STALE_INPUT", "source image, load, frame-map, or mask identity changed; create a fresh plan")
        if plan.get("configuration_sha256") != _hash_json(config):
            raise AgentError("PLAN_STALE_CONFIG", "project parameters changed after planning; create a fresh plan")
        if plan.get("code_sha256") != code_sha:
            raise AgentError("PLAN_STALE_CODE", "analysis code changed after planning; create a fresh plan")
        import ezdic_core as core

        current_environment = core.collect_environment()
        if plan.get("environment") != current_environment:
            raise AgentError("PLAN_STALE_ENVIRONMENT", "runtime packages or platform changed after planning; create a fresh plan")
        if plan.get("status") != "ready":
            raise AgentError(
                "PLAN_REVIEW_REQUIRED",
                "saved plan is not computationally ready; inspect its ROI, pilot, and parameter candidates before running",
                details={
                    "plan_status": plan.get("status"),
                    "pilot_issue": plan.get("pilot_issue"),
                    "analysis_rectangle_candidate_px": plan.get("analysis_rectangle_candidate_px"),
                    "candidate_trials": plan.get("candidate_trials", []),
                },
            )
        parameters = dict(plan["selected_parameters"])
        if parameters.get("reject_nonconverged") is not True:
            raise AgentError("PLAN_SOLVER_POLICY_INVALID", "create a fresh plan with explicit nonconverged-point rejection")
        _record_inventory(root, records, input_sha)
        reference_record = next((row for row in records if int(row["native_frame"]) == int(plan["reference_native_frame"])), None)
        if reference_record is None:
            raise AgentError("REFERENCE_FRAME_UNKNOWN", f"native reference frame {plan['reference_native_frame']} is absent")
        reference_sha_before = _sha256(Path(reference_record["path"]))
        if reference_sha_before != reference_record["sha256"]:
            raise AgentError("INPUT_CHANGED_DURING_RUN", "reference image changed after inventory", path=str(reference_record["path"]))
        try:
            reference = _load_image(reference_record["path"])
        except AgentError as exc:
            raise AgentError("REFERENCE_FRAME_UNREADABLE", exc.message, path=str(reference_record["path"])) from exc
        roi = tuple(int(value) for value in plan["analysis_rectangle_candidate_px"])
        specimen_mask = _load_specimen_mask(config, reference.shape)
        mechanics, mechanics_status = _load_mechanics(config)
        displacement_reference, displacement_reference_alignment = _displacement_reference(
            records, plan, config, mechanics, mechanics_status,
        )
        run_fingerprint = _run_fingerprint(config, plan, input_sha, code_sha)
        run_id = f"run-{run_fingerprint[:20]}"
        run_root = root / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "checkpoints").mkdir(exist_ok=True)
        (run_root / "plots").mkdir(exist_ok=True)
        manifest_path = run_root / "manifest.json"
        prior_manifest = _read_json(manifest_path) if manifest_path.is_file() else None
        state_path = run_root / "run_state.json"
        prior_state = _read_json(state_path) if state_path.is_file() else {}
        prior_receipts = None
        if prior_manifest is not None:
            if prior_manifest.get("run_fingerprint") != run_fingerprint:
                raise AgentError("RUN_STALE_BINDING", "existing run manifest does not match the requested run")
            prior_receipts = prior_manifest.get("checkpoint_receipts")
            if not isinstance(prior_receipts, Mapping):
                raise AgentError("RUN_MANIFEST_INVALID", "existing run manifest has invalid checkpoint receipts", path=str(manifest_path))
            regenerated_fields = {
                f"fields_csv/{_frame_key(int(record['native_frame']))}.csv" for record in records
            } if field_csv else set()
            _require_retained_artifacts_integrity(run_root, prior_manifest, regenerated_paths=regenerated_fields)
        # A process can stop after writing a new sidecar and before publishing
        # the next manifest. Accept unregistered frames only in that case.
        allow_unregistered = prior_manifest is None or prior_state.get("status") in {"running", "failed"}
        previous_pointer = dict(pointer)
        pointer["current_run"] = run_root.relative_to(root).as_posix()
        _atomic_json(root / "CURRENT.json", pointer)
        cancel_file = root / "CANCEL.request"
        try:
            cancel_file.unlink()
        except FileNotFoundError:
            pass
        _atomic_json(run_root / "run_state.json", {
            "schema": RUN_SCHEMA, "status": "running", "run_id": run_id,
            "run_fingerprint": run_fingerprint, "input_fingerprint": input_sha,
            "code_sha256": code_sha, "started_unix_s": time.time(), "completed_frames": 0,
        })
        rows: list[dict[str, Any]] = []
        recomputed_frames: set[int] = set()

        def finish_cancelled(remaining_frames: int) -> dict[str, Any]:
            _atomic_json(run_root / "run_state.json", {
                "schema": RUN_SCHEMA, "status": "cancelled", "run_id": run_id,
                "run_fingerprint": run_fingerprint, "input_fingerprint": input_sha,
                "code_sha256": code_sha, "completed_frames": len(rows), "updated_unix_s": time.time(),
            })
            _write_progress(run_root, {
                "event": "run_cancelled", "completed_frames": len(rows), "remaining_frames": remaining_frames,
            }, progress)
            result = {
                "project": str(root), "run_id": run_id, "status": "cancelled",
                "scientific_ok": False, "numerical_quality_passed": False, "scientific_acceptance": "not_assessed",
                "completed_frames": len(rows), "total_frames": len(records), "run_dir": str(run_root),
            }
            if prior_manifest is not None:
                try:
                    _require_recorded_run_integrity(run_root, prior_manifest)
                except AgentError as exc:
                    # Unprocessed or newly replaced sidecars cannot be signed
                    # against an older manifest. Preserve the old evidence and
                    # let a later resume validate/recompute individual frames.
                    return {**result, "outputs_updated": False, "output_issue": exc.as_dict()}
            _build_outputs(
                root, run_root, config, plan, records, rows, mechanics_status,
                run_fingerprint, input_sha, code_sha, field_csv=False, plots=False,
                include_checkpoints=report_include_checkpoints, build_report=cancel_build_report,
            )
            return {**result, "outputs_updated": True}

        try:
            _write_progress(run_root, {
                "event": "run_started", "run_id": run_id, "frame_count": len(records),
                "reference_native_frame": int(reference_record["native_frame"]), "run_fingerprint": run_fingerprint,
            }, progress)
            for position, record in enumerate(records, 1):
                native = int(record["native_frame"])
                if cancel_file.exists():
                    return finish_cancelled(len(records) - position + 1)
                existing = _checkpoint_valid(
                    run_root, record, position, int(reference_record["native_frame"]),
                    run_fingerprint, input_sha, code_sha, parameters,
                    prior_receipts=prior_receipts, allow_unregistered=allow_unregistered,
                )
                if existing is not None:
                    meta = existing[1]
                    row = dict(meta.get("summary_row") or {})
                    if row:
                        row["frame_position_1based"] = position
                        rows.append(row)
                        _write_progress(run_root, {"event": "frame_reused", "native_frame": native, "position": position, "total": len(records)}, progress)
                        continue
                recomputed_frames.add(native)
                _write_progress(run_root, {"event": "frame_started", "native_frame": native, "position": position, "total": len(records)}, progress)
                try:
                    if record.get("status") == "unreadable":
                        raise AgentError("FRAME_UNREADABLE", record.get("read_error") or "frame is not readable", path=record["path"])
                    if _sha256(Path(record["path"])) != record["sha256"]:
                        raise AgentError("INPUT_CHANGED_DURING_RUN", "source image changed after inventory", path=record["path"])
                    if record.get("shape_yx") and tuple(int(v) for v in record["shape_yx"]) != tuple(reference.shape):
                        raise AgentError("FRAME_DIMENSION_MISMATCH", f"shape {tuple(record['shape_yx'])} differs from reference {reference.shape}", path=record["path"])
                    deformed = _load_image(record["path"])
                    if tuple(deformed.shape) != tuple(reference.shape):
                        raise AgentError("FRAME_DIMENSION_MISMATCH", f"shape {deformed.shape} differs from reference {reference.shape}", path=record["path"])
                    last_progress = 0

                    def dic_progress(done: int, total: int) -> None:
                        nonlocal last_progress
                        if cancel_file.exists():
                            raise _RunCancelled("cancel requested during DIC frame")
                        interval = max(1, int(total) // 20)
                        if done == total or done - last_progress >= interval:
                            last_progress = done
                            _write_progress(run_root, {
                                "event": "dic_progress", "native_frame": native, "position": position,
                                "total_frames": len(records), "points_done": int(done), "points_total": int(total),
                                "fraction": float(done / total) if total else 1.0,
                            }, progress)

                    field = core.run_2d_dic(
                        reference, deformed, roi,
                        subset_size=int(parameters["subset_size_px"]), step=int(parameters["step_px"]),
                        solver=str(parameters["solver"]), strain_window=int(parameters["strain_window_poi"]),
                        search_radius=int(parameters["search_radius_px"]),
                        reject_nonconverged=parameters["reject_nonconverged"],
                        specimen_mask=specimen_mask,
                        progress_callback=dic_progress,
                    )
                    quality = field.get("quality_summary") or core.field_quality_summary(field)
                    field["provenance"] = {
                        "project_id": config["project_id"], "run_id": run_id,
                        "reference_frame_1based": records.index(reference_record) + 1,
                        "reference_native_frame": int(reference_record["native_frame"]),
                        "reference_filename": Path(reference_record["path"]).name,
                        "frame_global_1based": position, "native_frame": native,
                        "frame_filename": Path(record["path"]).name,
                        "time_s": record.get("time_s"), "roi_candidate_px": list(roi),
                        "analysis_rectangle_status": "candidate_not_physical_specimen_confirmation",
                        "input_sha256": record["sha256"],
                    }
                    strain_valid = np.asarray(field.get("strain_valid", []), dtype=bool)
                    exx = np.asarray(field.get("exx", []), dtype=float)
                    finite_exx = exx[np.isfinite(exx) & strain_valid] if exx.size == strain_valid.size else np.asarray([], dtype=float)
                    mechanics_row = _mechanics_at(
                        record.get("time_s"), mechanics, config,
                        displacement_reference_mm=displacement_reference,
                        displacement_reference_alignment=displacement_reference_alignment,
                    )
                    quality_ok = bool(quality.get("scientific_ok", False)) and bool(core.fullfield_field_has_finite_strain(field))
                    status_text = "complete" if quality_ok else "quality_failed"
                    gauge_rows = _measure_gauges(field, plan.get("gauge_reference_geometry") or [], quality_ok=quality_ok)
                    gauge_summary = _gauge_group_summary(gauge_rows)
                    summary_row = {
                        "frame_position_1based": position, "native_frame": native, "time_s": record.get("time_s"),
                        "reference_native_frame": int(reference_record["native_frame"]),
                        "status": status_text, "quality_status": "pass" if quality_ok else "fail",
                        "scientific_ok": quality_ok,
                        "eligible_point_count": int(quality.get("eligible_point_count", 0)),
                        "correlation_valid_fraction": float(quality.get("correlation_valid_fraction", 0.0)),
                        "strain_valid_fraction": float(quality.get("strain_valid_fraction", 0.0)),
                        "max_abs_local_exx": float(np.max(np.abs(finite_exx))) if quality_ok and finite_exx.size else None,
                        **gauge_summary,
                        "engineering_strain": gauge_summary["dic_virtual_gauge_engineering_strain"],
                        **mechanics_row,
                        "failure_code": None, "failure_reason": None,
                        "checkpoint": f"checkpoints/{_frame_key(native)}.npz",
                    }
                    checkpoint = run_root / "checkpoints" / f"{_frame_key(native)}.npz"
                    meta = {
                        "native_frame": native, "frame_position": position, "time_s": record.get("time_s"),
                        "source_path": record["path"], "source_sha256": record["sha256"],
                        "reference_native_frame": int(reference_record["native_frame"]), "run_fingerprint": run_fingerprint,
                        "input_fingerprint": input_sha, "code_sha256": code_sha, "status": status_text,
                        "quality": quality, "parameters": parameters, "roi_candidate_px": list(roi),
                        "gauge_measurements": gauge_rows,
                        "summary_row": summary_row,
                    }
                    _serialize_field(checkpoint, field, meta)
                    rows.append(summary_row)
                    _write_progress(run_root, {"event": "frame_complete", "native_frame": native, "position": position, "status": status_text, "quality": quality}, progress)
                except _RunCancelled:
                    return finish_cancelled(len(records) - position + 1)
                except AgentError as exc:
                    mechanics_row = _mechanics_at(
                        record.get("time_s"), mechanics, config,
                        displacement_reference_mm=displacement_reference,
                        displacement_reference_alignment=displacement_reference_alignment,
                    )
                    row = _frame_failure(
                        run_root, record, run_fingerprint, input_sha, code_sha, exc,
                        frame_position=position, reference_native_frame=int(reference_record["native_frame"]),
                        parameters=parameters, mechanics_row=mechanics_row,
                    )
                    rows.append(row)
                    _write_progress(run_root, {"event": "frame_failed", "native_frame": native, "position": position, "error": exc.as_dict()}, progress)
                except Exception as exc:
                    error = AgentError("DIC_FRAME_ERROR", f"{type(exc).__name__}: {exc}", path=record["path"])
                    mechanics_row = _mechanics_at(
                        record.get("time_s"), mechanics, config,
                        displacement_reference_mm=displacement_reference,
                        displacement_reference_alignment=displacement_reference_alignment,
                    )
                    row = _frame_failure(
                        run_root, record, run_fingerprint, input_sha, code_sha, error,
                        frame_position=position, reference_native_frame=int(reference_record["native_frame"]),
                        retryable=True, parameters=parameters, mechanics_row=mechanics_row,
                    )
                    rows.append(row)
                    _write_progress(run_root, {"event": "frame_failed", "native_frame": native, "position": position, "error": error.as_dict()}, progress)
            if cancel_file.exists():
                return finish_cancelled(0)
            if _sha256(Path(reference_record["path"])) != reference_sha_before:
                raise AgentError("INPUT_CHANGED_DURING_RUN", "reference image changed during analysis")
            failed = sum(row.get("status") not in {"complete"} for row in rows)
            passed = sum(row.get("status") == "complete" for row in rows)
            run_status = "complete" if failed == 0 and passed > 0 else "partial" if passed > 0 else "failed"
            run_state = {
                "schema": RUN_SCHEMA, "status": run_status, "run_id": run_id,
                "run_fingerprint": run_fingerprint, "input_fingerprint": input_sha,
                "code_sha256": code_sha, "completed_frames": len(rows),
                "passed_frames": passed, "failed_frames": failed, "updated_unix_s": time.time(),
            }
            _atomic_json(run_root / "run_state.json", run_state)
            sources_changed = False
            if prior_manifest is not None:
                for native in recomputed_frames:
                    name = f"{_frame_key(native)}.json"
                    previous_receipt = prior_receipts.get(name)
                    path = run_root / "checkpoints" / name
                    if not isinstance(previous_receipt, Mapping) or not path.is_file() or _sha256(path) != previous_receipt.get("sha256"):
                        sources_changed = True
                        break
            had_plots = any(
                isinstance(name, str) and name.startswith("plots/")
                for name in (prior_manifest.get("artifacts") or {})
            ) if prior_manifest else False
            output = _build_outputs(
                root, run_root, config, plan, records, rows, mechanics_status,
                run_fingerprint, input_sha, code_sha,
                field_csv=field_csv, plots=make_plots or (sources_changed and had_plots),
                include_checkpoints=report_include_checkpoints,
                build_report=build_report,
                invalidate_extractions=sources_changed,
            )
            _write_progress(run_root, {
                "event": "run_finished", "status": run_status, "passed_frames": passed, "failed_frames": failed,
                "scientific_ok": output["quality"]["scientific_ok"],
                "numerical_quality_passed": output["quality"]["numerical_quality_passed"],
            }, progress)
            return {
                "project": str(root), "run_id": run_id, "status": run_status,
                "scientific_ok": output["quality"]["scientific_ok"],
                "numerical_quality_passed": output["quality"]["numerical_quality_passed"],
                "scientific_acceptance": "not_assessed", "experimental_uncertainty_calibrated": False,
                "passed_frames": passed, "failed_frames": failed, "total_frames": len(records),
                "run_dir": str(run_root), "artifacts": output["artifacts"],
                "stale_derived_artifacts": output.get("stale_derived_artifacts", []),
            }
        except BaseException as exc:
            error_info = exc.as_dict() if isinstance(exc, AgentError) else {"error_code": type(exc).__name__, "message": str(exc)}
            try:
                _atomic_json(run_root / "run_state.json", {
                    "schema": RUN_SCHEMA, "status": "failed", "run_id": run_id,
                    "run_fingerprint": run_fingerprint, "input_fingerprint": input_sha,
                    "code_sha256": code_sha, "completed_frames": len(rows),
                    "error": error_info, "updated_unix_s": time.time(),
                })
                if not (run_root / "manifest.json").is_file() and previous_pointer.get("current_run") != pointer.get("current_run"):
                    _atomic_json(root / "CURRENT.json", previous_pointer)
            except Exception:
                pass
            raise


def plot(project: str | os.PathLike[str], *, plot_every: int | None = None) -> dict[str, Any]:
    root = _project_root(project)
    with _project_lock(root, purpose="plot"):
        _, run_root = _load_run_pointer(root)
        config = _read_config(root)
        pointer = _current_pointer(root)
        plan = _read_json(_plan_path(root, pointer))
        run_state = _read_json(run_root / "run_state.json")
        _require_current_run_binding(root, config, run_root, plan, run_state)
        count = int(config.get("plot_every", 0) if plot_every is None else plot_every)
        if count < 0:
            raise AgentError("CONFIG_VALUE_ERROR", "plot_every must be >= 0", path="$.plot_every")
        result = _plot_run_locked(root, run_root, config, plan, plot_every=count, run_fingerprint=str(run_state["run_fingerprint"]))
        _update_report_visualizations(run_root, result.get("history_plots"), plot_count=result.get("plot_count"))
        _refresh_manifest_artifacts(run_root)
        report_manifest = run_root / "report" / "manifest.json"
        if report_manifest.is_file():
            try:
                include_cp = bool(_read_json(report_manifest).get("include_checkpoints", False))
            except AgentError:
                include_cp = False
            _build_report_transaction(run_root, config, plan, include_checkpoints=include_cp)
        return {"run_id": run_root.name, "status": "complete", **result}


def export(project: str | os.PathLike[str], *, field_csv: bool = False) -> dict[str, Any]:
    root = _project_root(project)
    config = _read_config(root)
    with _project_lock(root, purpose="export"):
        pointer, run_root = _load_run_pointer(root)
        plan = _read_json(_plan_path(root, pointer))
        run_state = _read_json(run_root / "run_state.json")
        records, input_sha, current_code = _require_current_run_binding(root, config, run_root, plan, run_state)
        rows = []
        mechanics, mechanics_status = _load_mechanics(config)
        displacement_reference, displacement_reference_alignment = _displacement_reference(
            records, plan, config, mechanics, mechanics_status,
        )
        by_native = {int(meta["native_frame"]): meta for meta in _iter_checkpoint_metadata(run_root)}
        for index, record in enumerate(records, 1):
            native = int(record["native_frame"])
            meta = by_native.get(native)
            if meta is None:
                rows.append({
                    "frame_position_1based": index, "native_frame": native, "time_s": record.get("time_s"),
                    "reference_native_frame": int(plan["reference_native_frame"]), "status": "pending",
                    "quality_status": "unavailable", "scientific_ok": False, "engineering_strain": None,
                    **_mechanics_at(
                        record.get("time_s"), mechanics, config,
                        displacement_reference_mm=displacement_reference,
                        displacement_reference_alignment=displacement_reference_alignment,
                    ),
                })
                continue
            row = dict(meta.get("summary_row") or {})
            if row.get("status") == "failed":
                rows.append(row)
                continue
            row.update(_mechanics_at(
                record.get("time_s"), mechanics, config,
                displacement_reference_mm=displacement_reference,
                displacement_reference_alignment=displacement_reference_alignment,
            ))
            row["engineering_strain"] = row.get("dic_virtual_gauge_engineering_strain")
            rows.append(row)
        output = _build_outputs(
            root, run_root, config, plan, records, rows, mechanics_status,
            str(run_state["run_fingerprint"]), input_sha, current_code,
            field_csv=field_csv, plots=False, include_checkpoints=True, build_report=True,
        )
        return {
            "run_id": run_root.name, "status": run_state.get("status"),
            "scientific_ok": output["quality"]["scientific_ok"],
            "numerical_quality_passed": output["quality"]["numerical_quality_passed"],
            "scientific_acceptance": "not_assessed", "experimental_uncertainty_calibrated": False,
            "artifacts": output["artifacts"],
        }


def extract(
    project: str | os.PathLike[str], *, component: str = "exx",
    roi: Sequence[float] | None = None, path_points: Sequence[Sequence[float]] | None = None,
    max_distance: float | None = None, coordinate_system: str = "image",
) -> dict[str, Any]:
    """Extract configured spatial queries using the optional field helpers."""
    try:
        from . import fields
    except ImportError as exc:
        raise AgentError("FIELD_API_UNAVAILABLE", "spatial field extraction support is not installed") from exc
    if roi is None and path_points is None:
        raise AgentError("CONFIG_EXTRACTION_QUERY_REQUIRED", "provide an ROI or path points for extraction")
    valid_components = set(fields._STRAIN_COMPONENTS) | {"u", "v", "u_raw", "v_raw", "u_mm", "v_mm", "zncc"}
    if not isinstance(component, str) or component not in valid_components:
        raise AgentError("COMPONENT_INVALID", f"unknown or unsupported field component: {component!r}", path="$.component")
    if coordinate_system not in {"image", "calibrated"}:
        raise AgentError("CONFIG_COORDINATE_SYSTEM", "coordinate_system must be image or calibrated", path="$.coordinate_system")
    if roi is not None:
        try:
            roi = fields._validate_roi(roi)
        except ValueError as exc:
            raise AgentError("CONFIG_EXTRACTION_ROI", str(exc), path="$.roi") from exc
    if path_points is not None:
        if max_distance is None:
            raise AgentError("PATH_TOLERANCE_REQUIRED", "max_distance is required for path extraction", path="$.max_distance")
        max_distance = _positive_number(max_distance, "max_distance", allow_zero=True)
        try:
            points = np.asarray(path_points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
                raise ValueError("path_points must be a nonempty finite N by 2 array")
        except (TypeError, ValueError) as exc:
            raise AgentError("CONFIG_EXTRACTION_PATH", str(exc), path="$.path_points") from exc
        path_points = points.tolist()
    root = _project_root(project)
    config = _read_config(root)
    if (coordinate_system == "calibrated" or component in {"u_mm", "v_mm"}) and not config.get("calibration"):
        raise AgentError("CONFIG_CALIBRATION_REQUIRED", "calibrated extraction requires an explicit project calibration", path="$.calibration")
    with _project_lock(root, purpose="extract"):
        pointer, run_root = _load_run_pointer(root)
        plan = _read_json(_plan_path(root, pointer))
        state = _read_json(run_root / "run_state.json")
        records, _, _ = _require_current_run_binding(root, config, run_root, plan, state, allow_stale_derived=True)
        output: dict[str, Any] = {"run_id": run_root.name, "component": component, "coordinate_system": coordinate_system}
        checkpoint_root = run_root / "checkpoints"
        metadata_by_native = {int(meta["native_frame"]): meta for meta in _iter_checkpoint_metadata(run_root)}

        def load_field(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
            meta = metadata_by_native.get(int(record["native_frame"]))
            if not meta or meta.get("status") == "failed" or not meta.get("checkpoint"):
                return None, meta
            arrays, loaded = _load_field(checkpoint_root / str(meta["checkpoint"]))
            field = _field_from_arrays(arrays, loaded)
            if config.get("calibration"):
                field.update(fields.calibrate_field(field, config["calibration"]))
            return field, loaded

        with _extraction_transaction(run_root) as dest:
            if roi is not None:
                output["rois"] = []

                def roi_rows():
                    for record in records:
                        field, meta = load_field(record)
                        native = int(record["native_frame"])
                        if field is None:
                            continue
                        summary = fields.summarize_roi(field, component, roi, coordinate_system=coordinate_system)
                        output["rois"].append({"native_frame": native, "time_s": meta.get("time_s"), **_json_safe(summary)})
                        for row in fields.iter_roi_rows(field, component, roi, coordinate_system=coordinate_system):
                            yield {"native_frame": native, "time_s": meta.get("time_s"), **row}

                component_slug = _component_file_slug(component)
                roi_csv = fields.write_rows_csv(
                    roi_rows(), dest / f"roi_{component_slug}.csv",
                    fieldnames=["native_frame", "time_s", "poi_index", "x", "y", "coordinate_system", "component", "value", "value_unit", "status", "interpolation"],
                )
                _atomic_json(dest / f"roi_{component_slug}_summary.json", output["rois"])
                output["roi_csv"] = str(roi_csv)
                output["roi_summary"] = str(dest / f"roi_{component_slug}_summary.json")
            if path_points is not None:
                def sample_rows():
                    for record in records:
                        field, meta = load_field(record)
                        native = int(record["native_frame"])
                        if field is None:
                            for index, point in enumerate(path_points):
                                yield {
                                    "native_frame": native, "time_s": record.get("time_s"), "run_status": "unavailable",
                                    "path_index": index, "target_x": point[0], "target_y": point[1],
                                    "poi_index": None, "poi_x": None, "poi_y": None, "distance": None,
                                    "coordinate_system": coordinate_system, "interpolation": "none",
                                    component: None, "status": "frame_failed",
                                }
                            continue
                        for row in fields.sample_path(field, path_points, [component], max_distance, coordinate_system=coordinate_system):
                            yield {"native_frame": native, "time_s": meta.get("time_s"), "run_status": meta.get("status"), **row}

                component_slug = _component_file_slug(component)
                path_csv = fields.write_rows_csv(sample_rows(), dest / f"path_{component_slug}.csv")
                output["paths_csv"] = str(path_csv)
                frame_axis = config.get("time_basis") == "frame_index"
                times = [record["native_frame"] if frame_axis else record.get("time_s") for record in records]

                def plot_fields():
                    for record in records:
                        field, _meta = load_field(record)
                        yield field

                finite_steps = np.diff([float(value) for value in times]) if len(times) > 1 else np.asarray([])
                max_time_gap = float(np.median(finite_steps) * 1.5) if finite_steps.size and not frame_axis else None
                figure_path = dest / f"position_{'frame' if frame_axis else 'time'}_{component_slug}.png"
                fig, _ax, _artist, plot_summary = fields.plot_position_time(
                    plot_fields(), times, path_points, component, max_distance,
                    coordinate_system=coordinate_system, max_time_gap_s=max_time_gap,
                    axis_unit="frame" if frame_axis else "s", axis_label="Native frame" if frame_axis else "Time (s)",
                )
                import matplotlib.pyplot as plt

                try:
                    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
                finally:
                    plt.close(fig)
                output["position_time_plot"] = str(figure_path)
                output["position_time_summary"] = plot_summary
        final_destination = run_root / "extractions"
        for key, value in list(output.items()):
            if isinstance(value, str) and value.startswith(str(dest) + os.sep):
                output[key] = str(final_destination / Path(value).relative_to(dest))
        return output


def _current_plan_if_fresh(root: Path, config: Mapping[str, Any]) -> dict[str, Any] | None:
    pointer = _current_pointer(root)
    if not pointer.get("current_plan"):
        return None
    plan = _read_json(_plan_path(root, pointer))
    _validate_plan_checksum(plan)
    _, input_sha = _inventory_fresh(root, config)
    import ezdic_core as core

    return plan if (
        plan.get("configuration_sha256") == _hash_json(config)
        and plan.get("input_fingerprint") == input_sha
        and plan.get("code_sha256") == _code_fingerprint()
        and plan.get("environment") == core.collect_environment()
    ) else None


def analyze(project: str | os.PathLike[str], *, progress: Callable[[dict[str, Any]], None] | None = None, field_csv: bool = False, make_plots: bool = True) -> dict[str, Any]:
    root = _project_root(project)
    config = _read_config(root)
    plan = _current_plan_if_fresh(root, config)
    if plan is None:
        plan = plan_project(root, progress=progress)
    elif progress:
        progress({"event": "plan_reused", "plan_id": plan.get("plan_id"), "status": plan.get("status")})
    result = run(root, progress=progress, field_csv=field_csv, make_plots=make_plots)
    return {"plan": plan, "run": result}


def cancel(project: str | os.PathLike[str]) -> dict[str, Any]:
    root = _project_root(project)
    if not root.is_dir():
        raise AgentError("PROJECT_NOT_FOUND", f"project does not exist: {root}", path=str(root))
    _atomic_json(root / "CANCEL.request", {"pid": os.getpid(), "requested_unix_s": time.time()})
    return {"project": str(root), "cancel_requested": True}


def status(project: str | os.PathLike[str]) -> dict[str, Any]:
    root = _project_root(project)
    config = _read_config(root)
    pointer = _current_pointer(root)
    lock_path = root / ".agent.lock"
    locked = False
    lock: dict[str, Any] | None = None
    if lock_path.exists():
        locked = True
        try:
            lock = _read_json(lock_path)
        except AgentError as exc:
            if exc.code == "PROJECT_FILE_MISSING":
                locked = False
                lock = None
            else:
                lock = {"corrupt": True, "error": exc.as_dict()}
    run_state = None
    progress_state = None
    if pointer.get("current_run"):
        run_root = root / str(pointer["current_run"])
        state_path = run_root / "run_state.json"
        if state_path.exists():
            run_state = _read_json(state_path)
        progress_path = run_root / "progress.json"
        if progress_path.exists():
            progress_state = _read_json(progress_path)
    return {
        "project": str(root), "project_id": config.get("project_id"),
        "locked": locked, "lock_owner": lock,
        "cancel_requested": (root / "CANCEL.request").exists(),
        "current_plan": pointer.get("current_plan"), "current_run": pointer.get("current_run"),
        "run": run_state, "progress": progress_state,
    }


def _batch_one(project_path: str, field_csv: bool, make_plots: bool) -> dict[str, Any]:
    try:
        result = analyze(project_path, field_csv=field_csv, make_plots=make_plots)
        run_result = result.get("run", {})
        scientific_pass = run_result.get("scientific_ok") is True and run_result.get("status") == "complete"
        numerical_status = "pass" if scientific_pass else "attention_required"
        return {
            "project": project_path, "execution_status": "complete",
            "scientific_status": numerical_status, "numerical_quality_status": numerical_status,
            "scientific_acceptance": "not_assessed", "result": result,
        }
    except AgentError as exc:
        return {
            "project": project_path, "execution_status": "failed",
            "scientific_status": "not_evaluated", "numerical_quality_status": "not_evaluated",
            "scientific_acceptance": "not_assessed", "error": exc.as_dict(),
        }
    except Exception as exc:
        return {
            "project": project_path, "execution_status": "failed",
            "scientific_status": "not_evaluated", "numerical_quality_status": "not_evaluated",
            "scientific_acceptance": "not_assessed",
            "error": {"error_code": "BATCH_RUNTIME_ERROR", "message": f"{type(exc).__name__}: {exc}", "path": "$"},
        }


def batch(projects: Sequence[str | os.PathLike[str]], *, max_workers: int = 2, field_csv: bool = False, make_plots: bool = True) -> dict[str, Any]:
    """Run independent sample projects with bounded cross-project concurrency."""
    paths = [str(_project_root(item)) for item in projects]
    if not paths:
        raise AgentError("BATCH_EMPTY", "batch requires at least one project", path="$.projects")
    if len(set(path.casefold() for path in paths)) != len(paths):
        raise AgentError("BATCH_DUPLICATE_PROJECT", "each project may appear only once in a batch", path="$.projects")
    if type(max_workers) is not int or max_workers < 1 or max_workers > 8:
        raise AgentError("BATCH_WORKERS", "max_workers must be in [1, 8]", path="$.max_workers")
    results = []
    with ProcessPoolExecutor(max_workers=min(max_workers, len(paths)), mp_context=multiprocessing.get_context("spawn")) as executor:
        future_map = {executor.submit(_batch_one, path, field_csv, make_plots): path for path in paths}
        for future in as_completed(future_map):
            project_path = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "project": project_path, "execution_status": "failed",
                    "scientific_status": "not_evaluated", "numerical_quality_status": "not_evaluated",
                    "scientific_acceptance": "not_assessed",
                    "error": {"error_code": "BATCH_WORKER_FAILED", "message": f"{type(exc).__name__}: {exc}", "path": project_path},
                })
    results.sort(key=lambda item: paths.index(item["project"]))
    completed = sum(item["execution_status"] == "complete" for item in results)
    passed = sum(item["scientific_status"] == "pass" for item in results)
    attention = sum(item["scientific_status"] == "attention_required" for item in results)
    failed = len(results) - completed
    return {
        "status": "complete" if failed == 0 else "partial",
        "scientific_status": "pass" if completed == len(results) and passed == len(results) else "attention_required",
        "numerical_quality_status": "pass" if completed == len(results) and passed == len(results) else "attention_required",
        "scientific_acceptance": "not_assessed",
        "experimental_uncertainty_calibrated": False,
        "project_count": len(results), "execution_completed": completed, "execution_failed": failed,
        "scientific_passed": passed, "attention_required": attention, "results": results,
    }


def reprocess(
    project: str | os.PathLike[str], *,
    regions: Sequence[Mapping[str, Any]] | None = None,
    gauges: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract new regions/gauges from saved original POIs without rerunning DIC."""
    root = _project_root(project)
    with _project_lock(root, purpose="reprocess"):
        checked = verify(root)
        if not checked.get("valid"):
            raise AgentError("REPROCESS_SOURCE_INVALID", "current run verification failed", details=checked.get("issues"))
        config = _read_config(root)
        candidate = {
            **config,
            "regions": config["regions"] if regions is None else list(regions),
            "gauges": config["gauges"] if gauges is None else list(gauges),
        }
        derived_config = _validate_project_config(candidate, root)
        pointer, run_root = _load_run_pointer(root)
        plan = _read_json(_plan_path(root, pointer))
        source_manifest = _read_json(run_root / "manifest.json")
        reference = _load_image(plan["reference_path"])
        parameters = plan["selected_parameters"]
        geometry = _resolve_gauge_geometry(
            derived_config["gauges"], plan["analysis_rectangle_candidate_px"], reference.shape,
            parameters["subset_size_px"], parameters["step_px"], _load_specimen_mask(config, reference.shape),
            calibration=config.get("calibration"),
        )
        derived_plan = {
            **plan, "gauge_reference_geometry": geometry, "source_plan_sha256": plan["plan_sha256"],
            "configuration_sha256": _hash_json(derived_config), "analysis_kind": "saved_field_postprocessing",
        }
        derived_plan.pop("plan_sha256")
        derived_plan["plan_sha256"] = _hash_json(derived_plan)
        receipt = {
            "schema": "straintrace.reprocess/v1", "source_run": run_root.name,
            "run_fingerprint": source_manifest["run_fingerprint"], "source_plan_sha256": plan["plan_sha256"],
            "regions": derived_config["regions"], "gauges": derived_config["gauges"],
            "checkpoint_receipts": source_manifest["checkpoint_receipts"], "solver_recomputed": False,
        }
        destination = run_root / "extractions" / f"reprocess-{_hash_json(receipt)[:20]}"
        if destination.exists():
            saved = _read_json(destination / "reprocess.json")
            if saved != receipt:
                raise AgentError("REPROCESS_IDENTITY_CONFLICT", "existing extraction has a different identity")
            from .reporting import verify_report

            report_check = verify_report(destination / "report")
            if not report_check.get("valid"):
                raise AgentError("REPROCESS_ARTIFACT_INVALID", "saved extraction report failed verification", details=report_check)
            saved_result = _read_json(destination / "result.json")
            measurement_status = saved_result["status"]
            return {
                "project": str(root), "run_id": run_root.name,
                "status": "reused" if measurement_status == "complete" else measurement_status,
                "output_dir": str(destination), "solver_recomputed": False,
                "postprocessing_quality": saved_result.get("postprocessing_quality"),
                "report": {"report_html": str(destination / "report/index.html"), "verification": report_check},
                "artifacts": {name: str(destination / name) for name in ("summary.csv", "gauges.csv", "regions.csv", "geometry.json", "reprocess.json")},
            }
        extraction = run_root / "_postprocess_staging" / uuid.uuid4().hex
        extraction.mkdir(parents=True)
        with (run_root / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            summary_columns = list(reader.fieldnames or [])
            summaries = {int(row["native_frame"]): row for row in reader}
        gauge_rows, region_rows = [], []
        for arrays, metadata in _iter_checkpoint_rows(run_root, validate_fingerprint=source_manifest["run_fingerprint"]):
            frame = {"frame_position_1based": metadata.get("frame_position"), "native_frame": metadata["native_frame"], "time_s": metadata.get("time_s")}
            field = _field_from_arrays(arrays, metadata) if arrays else {}
            measured = _measure_gauges(field, geometry, quality_ok=metadata.get("status") == "complete")
            gauge_rows.extend({**frame, **row} for row in measured)
            region_rows.extend(_region_rows(arrays, metadata, derived_config))
            summary = summaries[int(metadata["native_frame"])]
            summary.update(_gauge_group_summary(measured))
            summary["engineering_strain"] = summary["dic_virtual_gauge_engineering_strain"]
        _csv_atomic(extraction / "summary.csv", summaries.values(), summary_columns)
        _csv_atomic(extraction / "gauges.csv", gauge_rows, _GAUGE_COLUMNS)
        _csv_atomic(extraction / "regions.csv", region_rows, _REGION_COLUMNS)
        _atomic_json(extraction / "reprocess.json", receipt)
        _atomic_json(extraction / "geometry.json", {"regions": derived_config["regions"], "gauges": derived_config["gauges"], "gauge_reference_geometry": geometry})
        complete = source_manifest.get("numerical_quality_passed") is True and all(row.get("status") == "valid" for row in (*gauge_rows, *region_rows))

        def measurement_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
            valid = sum(row.get("status") == "valid" for row in rows)
            return {"total": len(rows), "valid": valid, "unavailable": len(rows) - valid}

        postprocessing_quality = {
            "schema": "straintrace.postprocessing-quality/v1",
            "status": "complete" if complete else "attention_required",
            "region_measurements": measurement_counts(region_rows),
            "gauge_measurements": measurement_counts(gauge_rows),
            "source_dic_numerical_quality_passed": source_manifest.get("numerical_quality_passed") is True,
            "source_run_id": run_root.name,
        }
        derived_quality = {**_read_json(run_root / "quality.json"), "postprocessing_quality": postprocessing_quality}
        _atomic_json(extraction / "quality.json", derived_quality)
        _atomic_json(extraction / "result.json", {"status": postprocessing_quality["status"], "solver_recomputed": False, "postprocessing_quality": postprocessing_quality})
        manifest = {
            **source_manifest,
            "artifacts": {path.name: _sha256(path) for path in extraction.iterdir() if path.is_file()},
            "postprocessing": receipt, "configuration_sha256": _hash_json(derived_config),
            "postprocessing_quality": postprocessing_quality,
            "source_configuration_sha256": source_manifest["configuration_sha256"],
            "plan_sha256": derived_plan["plan_sha256"],
        }
        _atomic_json(extraction / "manifest.json", manifest)
        from .reporting import build_report

        portable_report = build_report(extraction, config=derived_config, plan=derived_plan, regions_csv=extraction / "regions.csv")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _replace_with_retry(extraction, destination)
        portable_report = _relocate_paths(portable_report, extraction, destination)
        _refresh_manifest_artifacts(run_root)
        return {
            "project": str(root), "run_id": run_root.name,
            "status": "complete" if complete else "attention_required",
            "output_dir": str(destination), "solver_recomputed": False, "report": portable_report,
            "postprocessing_quality": postprocessing_quality,
            "artifacts": {name: str(destination / name) for name in ("summary.csv", "gauges.csv", "regions.csv", "geometry.json", "reprocess.json")},
        }


def update_project(project: str | os.PathLike[str], changes: Mapping[str, Any]) -> dict[str, Any]:
    """Validate top-level replacements, archive the old state and invalidate plans."""
    root = _project_root(project)
    if not isinstance(changes, Mapping):
        raise AgentError("CONFIG_TYPE_ERROR", "changes must be an object", path="$.changes")
    immutable = {"schema", "project_id", "image_dir", "mask_sha256"}
    if immutable.intersection(changes):
        raise AgentError("CONFIG_IMMUTABLE_FIELD", f"cannot update {sorted(immutable.intersection(changes))[0]}", path="$.changes")
    with _project_lock(root, purpose="update"):
        previous = _read_config(root)
        candidate = {**previous, **dict(changes)}
        if "mask_file" in changes:
            raw_mask = candidate.get("mask_file")
            if raw_mask is None:
                candidate["mask_file"] = None
                candidate["mask_sha256"] = None
            else:
                if not isinstance(raw_mask, (str, os.PathLike)) or not str(raw_mask).strip():
                    raise AgentError("CONFIG_TYPE_ERROR", "mask_file must be a path string or null", path="$.mask_file")
                resolved_mask = Path(raw_mask).expanduser().resolve(strict=False)
                if not resolved_mask.is_file():
                    raise AgentError("MASK_FILE_MISSING", f"mask image is missing: {resolved_mask}", path="$.mask_file")
                candidate["mask_file"] = str(resolved_mask)
                candidate["mask_sha256"] = _sha256(resolved_mask)
        validated = _validate_project_config(candidate, root)
        if _hash_json(validated) == _hash_json(previous):
            return {"project": str(root), "status": "unchanged", "configuration_sha256": _hash_json(previous)}
        records = _scan_inventory(validated)
        fingerprint = _inventory_fingerprint(validated, records)
        pointer = _current_pointer(root)
        revision = root / "configuration_history" / f"{time.time_ns()}-{_hash_json(previous)[:12]}.json"
        _atomic_json(revision, {
            "schema": "straintrace.configuration-revision/v1", "config": previous,
            "pointer": pointer, "replacement_sha256": _hash_json(validated),
        })
        _atomic_json(root / "project.json", validated)
        _record_inventory(root, records, fingerprint)
        pointer.update(current_plan=None, current_run=None)
        _atomic_json(root / "CURRENT.json", pointer)
        return {
            "project": str(root), "status": "updated", "configuration_sha256": _hash_json(validated),
            "previous_configuration": str(revision), "next_action": "analyze", "previous_runs_preserved": True,
        }


def discover_experiment(folder: str | os.PathLike[str], *, objective: str = "full_field", options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from .discovery import discover_experiment as discover

    return discover(folder, objective=objective, options=options)


def auto_analyze(
    folder: str | os.PathLike[str], project: str | os.PathLike[str], *,
    objective: str = "full_field", options: Mapping[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    field_csv: bool = False, make_plots: bool = True,
) -> dict[str, Any]:
    """Discover one sequence and execute the established reproducible workflow."""
    root = _project_root(project)
    existing = _read_config(root) if (root / "project.json").is_file() else None
    if options is not None and not isinstance(options, Mapping):
        raise AgentError("CONFIG_TYPE_ERROR", "options must be an object", path="$.options")
    effective_options = {**(_config_init_options(existing) if existing else {}), **dict(options or {})}
    discovery = discover_experiment(folder, objective=objective, options=effective_options)
    result: dict[str, Any] = {
        "project": str(root), "discovery": discovery, "status": discovery["status"],
        "scientific_acceptance": "not_assessed",
    }
    if discovery["status"] != "ready":
        return result
    if existing:
        binding = inspect.signature(init_project).bind(root, **discovery["init_options"])
        binding.apply_defaults()
        args = dict(binding.arguments)
        args.pop("project")
        candidate = _config_from_arguments(**args)
        candidate["project_id"] = existing["project_id"]
        if _hash_json(candidate) != _hash_json(existing):
            result.update(
                status="needs_input",
                required_inputs=[{"field": "project", "reason": "Supplied options change the existing configuration. Apply update_project explicitly or select a new project."}],
            )
            return result
    else:
        result["initialization"] = init_project(root, **discovery["init_options"])
    with _project_lock(root, purpose="auto_discovery"):
        _atomic_json(root / "discovery.json", discovery)
    config = _read_config(root)
    try:
        plan = _current_plan_if_fresh(root, config) or plan_project(root, progress=progress)
    except AgentError as exc:
        if exc.code not in {
            "TEXTURE_INSUFFICIENT", "ROI_TOO_SMALL", "IMAGE_TOO_SMALL", "ROI_OUTSIDE_IMAGE",
            "REFERENCE_FRAME_UNREADABLE", "REFERENCE_FRAME_UNAVAILABLE", "IMAGE_DECODE_FAILED", "IMAGE_INVALID",
        }:
            raise
        result.update(status="review_required", error=exc.as_dict())
        return result
    result["plan"] = plan
    if plan.get("status") != "ready":
        result["status"] = "review_required"
        return result
    # Build the portable report once (with checkpoints) at the report boundary
    # below instead of building an intermediate checkpoint-free report in run().
    # Cancellation still produces a partial report directly inside run().
    result["run"] = run(
        root, progress=progress, field_csv=field_csv, make_plots=make_plots,
        build_report=False, build_report_on_cancel=True,
    )
    if result["run"].get("status") == "cancelled":
        result["status"] = "cancelled"
        run_root = Path(result["run"]["run_dir"])
        report_path = run_root / "report" / "index.html"
        result["report"] = {
            "report_html": str(report_path) if report_path.is_file() else None,
            "partial": True, "updated": result["run"].get("outputs_updated", True),
        }
        return result
    expected_run = str(result["run"]["run_id"])
    if result["run"].get("stale_derived_artifacts"):
        with _project_lock(root, purpose="auto_finalize"):
            _require_expected_run(root, expected_run)
            result["verification"] = verify(root)
            result["status"] = "attention_required"
            result["report"] = {"report_html": None, "reason": "Existing extractions must be regenerated from the changed checkpoints."}
            _atomic_json(root / "automation.json", {
                "schema": "straintrace.automation/v1", "objective": discovery["objective"],
                "status": result["status"], "run_id": expected_run,
                "verification_valid": result["verification"].get("valid"), "scientific_acceptance": "not_assessed",
            })
        return result
    result["report"] = report(root, include_checkpoints=True, expected_run_id=expected_run)
    if isinstance(result["run"].get("artifacts"), dict):
        result["run"]["artifacts"]["report_html"] = result["report"]["report_html"]
    # A writer can change CURRENT between the independently locked steps.
    # Bind verification, goal checks and the final receipt to this same run.
    with _project_lock(root, purpose="auto_finalize"):
        _require_expected_run(root, expected_run)
        result["verification"] = verify(root)
        complete = (
            result["run"].get("status") == "complete"
            and result["run"].get("numerical_quality_passed") is True
            and result["verification"].get("valid") is True
        )
        result["status"] = "complete" if complete else "attention_required"
        if config.get("regions"):
            region_table = pd.read_csv(Path(result["run"]["run_dir"]) / "regions.csv")
            if region_table.empty or not region_table["status"].eq("valid").all():
                result.update(status="attention_required", region_issue="One or more requested regions lack valid measurement support.")
        if discovery["objective"] in {"virtual_gauge", "stress_strain"}:
            run_root = Path(result["run"]["run_dir"])
            table = pd.read_csv(run_root / "summary.csv")
            gauge_table = pd.read_csv(run_root / "gauges.csv")
            expected = {(int(frame), gauge["name"]) for frame in table["native_frame"] for gauge in config["gauges"]}
            actual = list(zip(gauge_table["native_frame"].astype(int), gauge_table["gauge_name"]))
            available = (
                len(actual) == len(expected) and set(actual) == expected and gauge_table["status"].eq("valid").all()
                and np.isfinite(pd.to_numeric(gauge_table["engineering_strain"], errors="coerce")).all()
            )
            if discovery["objective"] == "stress_strain":
                available = available and "stress_MPa" in table and np.isfinite(pd.to_numeric(table["stress_MPa"], errors="coerce")).all()
            if not available:
                result.update(status="attention_required", objective_issue="One or more frames lack the requested supported gauge or stress measurement.")
        _atomic_json(root / "automation.json", {
            "schema": "straintrace.automation/v1", "objective": discovery["objective"],
            "status": result["status"], "run_id": expected_run,
            "verification_valid": result["verification"].get("valid"), "scientific_acceptance": "not_assessed",
        })
    return result


def _require_expected_run(root: Path, expected_run_id: str) -> None:
    _, run_root = _load_run_pointer(root)
    if run_root.name != expected_run_id:
        raise AgentError(
            "AUTOMATION_STALE_RUN",
            "the current project run changed during automatic analysis; retry with the intended configuration",
            details={"expected_run_id": expected_run_id, "current_run_id": run_root.name},
        )


def report(project: str | os.PathLike[str], *, include_checkpoints: bool = True, expected_run_id: str | None = None) -> dict[str, Any]:
    """Rebuild the portable report package from a verified current run."""
    root = _project_root(project)
    with _project_lock(root, purpose="report"):
        if expected_run_id is not None:
            _require_expected_run(root, expected_run_id)
        checked = verify(root)
        # A killed report commit can leave only its regenerated output hashes
        # stale. Rebuild those outputs after every underlying source still
        # passes verification; never waive a source or checkpoint failure.
        issues = checked.get("issues") or []
        report_only = bool(issues) and all(
            issue.get("code") in {"ARTIFACT_MISSING", "ARTIFACT_CHECKSUM_MISMATCH"}
            and isinstance(issue.get("path"), str) and issue["path"].startswith("report/")
            for issue in issues
        )
        if not checked.get("valid") and not report_only:
            raise AgentError("REPORT_SOURCE_INVALID", "current run verification failed", details=checked.get("issues"))
        config = _read_config(root)
        pointer, run_root = _load_run_pointer(root)
        plan = _read_json(_plan_path(root, pointer))
        result = _build_report_transaction(run_root, config, plan, include_checkpoints=include_checkpoints)
        return {"run_id": run_root.name, **result}
