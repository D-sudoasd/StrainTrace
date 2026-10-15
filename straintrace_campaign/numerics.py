"""Campaign orchestration over the existing ezdic_core numerical engine."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

import ezdic_core as core
from .models import CampaignError
from . import storage
from . import roi as roi_module


_NUMERICS_FINGERPRINT_VERSION = "campaign_numerics_v1"
_MIN_HARD_CORR = 0.55
_MIN_SOFT_CORR = 0.35
_MAX_FB_TOLERANCE_PX = 12.0


def _as_json(value: Any) -> Any:
    """Convert core diagnostics to strict JSON values; NaN/Inf become null."""
    if isinstance(value, Mapping):
        return {str(key): _as_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_as_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _as_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, os.PathLike):
        return str(value)
    return str(value)


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancel)


def _check_cancel(cancel: Any) -> None:
    if _cancelled(cancel):
        raise CampaignError("CAMPAIGN_CANCELLED", "campaign operation was cancelled")


def _notify(progress, fraction: float, **details) -> None:
    if progress is not None:
        progress({"fraction": float(min(max(fraction, 0.0), 1.0)), **details})


def _source_file(module_file: str | None, relative_to_bundle: str) -> Path:
    """Resolve executable source in editable installs and frozen bundles."""
    if module_file:
        candidate = Path(module_file).resolve()
        if candidate.is_file():
            return candidate
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidate = Path(bundle_root) / "sources" / relative_to_bundle
        if candidate.is_file():
            return candidate.resolve()
    raise CampaignError("ENGINE_SOURCE_UNAVAILABLE", f"cannot fingerprint executable source: {relative_to_bundle}")


def _ordered_frames(acquisition: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames = acquisition.get("image_frames")
    if not isinstance(frames, list) or not frames:
        raise CampaignError("NO_INPUT_IMAGES", "acquisition.image_frames is empty")
    normalized = []
    seen = set()
    for item in frames:
        if not isinstance(item, Mapping):
            raise CampaignError("IMAGE_FRAME_INVALID", "image frame record must be an object")
        frame = item.get("frame")
        native = item.get("native_frame")
        path = item.get("path")
        if type(frame) is not int or type(native) is not int or not isinstance(path, str) or not path:
            raise CampaignError("IMAGE_FRAME_INVALID", "image frame needs integer frame/native_frame and absolute path")
        if frame in seen:
            raise CampaignError("DUPLICATE_IMAGE_FRAME", f"image frame identity {frame} is duplicated")
        seen.add(frame)
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise CampaignError("IMAGE_FRAME_MISSING", f"image frame {frame} is missing: {resolved}")
        normalized.append({**dict(item), "frame": frame, "native_frame": native, "path": str(resolved)})
    return sorted(normalized, key=lambda item: (item["frame"], item["native_frame"], item["path"].casefold()))


def _code_fingerprints() -> dict[str, str]:
    campaign_code = _source_file(__file__, "straintrace_campaign/numerics.py")
    roi_code = _source_file(roi_module.__file__, "straintrace_campaign/roi.py")
    engine_code = _source_file(getattr(core, "__file__", None), "ezdic_core.py")
    return {
        "straintrace_campaign/numerics.py": storage.sha256_file(campaign_code),
        "straintrace_campaign/roi.py": storage.sha256_file(roi_code),
        "ezdic_core.py": storage.sha256_file(engine_code),
    }


def _roi_tracking_geometry(acquisition: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return only the reviewed geometry that changes numerical tracking.

    Reviewer identity, approval time, proposal diagnostics, and preview paths
    describe the review workflow, not the numerical inputs. Keeping this subset
    aligned with ``workflow._roi_tracking_geometry`` avoids needless analysis
    cache invalidation while approval is still validated before every run.
    """
    proposal = acquisition.get("roi")
    if not isinstance(proposal, Mapping):
        return None
    frames = acquisition.get("image_frames")
    reference_id = proposal.get("reference_frame")
    reference_matches = [
        item for item in frames or []
        if isinstance(item, Mapping) and item.get("frame") == reference_id
    ]
    if len(reference_matches) != 1:
        raise CampaignError("ROI_REFERENCE_FRAME_INVALID", f"reference frame {reference_id!r} is absent or ambiguous")
    reference = reference_matches[0]
    raw_groups = proposal.get("groups", [])
    groups = []
    if isinstance(raw_groups, list):
        for group in raw_groups:
            if isinstance(group, Mapping):
                groups.append({
                    key: _as_json(group[key])
                    for key in ("name", "roi1", "roi2", "actual_mode", "selected_mode", "role")
                    if key in group
                })
            else:
                groups.append(_as_json(group))
    return {
        "reference_frame": proposal.get("reference_frame"),
        "reference_native_dic_frame": reference.get("native_frame"),
        "reference_sha256": proposal.get("reference_sha256"),
        "image_shape": _as_json(proposal.get("image_shape")),
        "free_region": _as_json(proposal.get("free_region")),
        "field_roi": _as_json(proposal.get("field_roi")),
        "exclusions": _as_json(proposal.get("exclusions", [])),
        "groups": groups,
    }


def _fingerprint_dependencies(acquisition: Mapping[str, Any]) -> dict[str, Any]:
    frames = _ordered_frames(acquisition)
    image_dependencies = []
    for item in frames:
        path = Path(item["path"])
        image_dependencies.append({
            "frame": item["frame"],
            "native_frame": item["native_frame"],
            "path": str(path),
            "sha256": storage.sha256_file(path),
            "counter_namespace": _as_json(item.get("counter_namespace")),
            "duplicate_identity": _as_json(item.get("duplicate_identity", item.get("duplicate_of"))),
        })
    return {
        "version": _NUMERICS_FINGERPRINT_VERSION,
        "acquisition_id": acquisition.get("acquisition_id"),
        "analysis_mode": acquisition.get("analysis_mode"),
        "analysis_config": _as_json(acquisition.get("analysis_config", {})),
        "roi_tracking_geometry": _roi_tracking_geometry(acquisition),
        "image_frames": image_dependencies,
        "engine_code_sha256": _code_fingerprints(),
    }


def _source_images_fingerprint(image_dependencies: list[Mapping[str, Any]]) -> str:
    """Fingerprint only ordered source image identities and their byte hashes."""
    return storage.fingerprint_data("campaign_source_images_v1", image_dependencies)


def fingerprint(acquisition: Mapping[str, Any]) -> str:
    """Hash image bytes/identity, analysis settings, tracking geometry, and engine source."""
    if not isinstance(acquisition, Mapping):
        raise CampaignError("ACQUISITION_INVALID", "acquisition must be an object")
    return storage.fingerprint_data(_NUMERICS_FINGERPRINT_VERSION, _fingerprint_dependencies(acquisition))


def _reference_index(frames: list[dict[str, Any]], proposal: Mapping[str, Any]) -> int:
    reference = proposal.get("reference_frame")
    matches = [index for index, frame in enumerate(frames) if frame["frame"] == reference]
    if len(matches) != 1:
        raise CampaignError("ROI_REFERENCE_FRAME_INVALID", f"reference frame {reference!r} is absent or ambiguous")
    return matches[0]


def _normalization(raw: np.ndarray, analysis_config: Mapping[str, Any]) -> dict[str, Any]:
    configured = analysis_config.get("normalization")
    settings = {"normalization": dict(configured)} if isinstance(configured, Mapping) else {}
    try:
        if settings.get("normalization", {}).get("policy") == "fixed_bounds":
            bounds = settings["normalization"].get("bounds")
            if not isinstance(bounds, Mapping) or "lo" not in bounds or "hi" not in bounds:
                raise CampaignError("NORMALIZATION_CONFIG_INVALID", "fixed_bounds normalization requires bounds.lo and bounds.hi")
            lo, hi = float(bounds["lo"]), float(bounds["hi"])
            if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                raise CampaignError("NORMALIZATION_CONFIG_INVALID", "normalization bounds must be finite with hi > lo")
            return {"lo": lo, "hi": hi, "policy": "fixed_bounds"}
        lower = float(settings.get("normalization", {}).get("lower_percentile", core.DEFAULT_NORMALIZATION_LOWER_PERCENTILE))
        upper = float(settings.get("normalization", {}).get("upper_percentile", core.DEFAULT_NORMALIZATION_UPPER_PERCENTILE))
        record = core.compute_reference_normalization(raw, lower_percentile=lower, upper_percentile=upper)
        return {"lo": record["reference_bounds"]["lo"], "hi": record["reference_bounds"]["hi"], "policy": "reference_percentile", "lower_percentile": lower, "upper_percentile": upper}
    except CampaignError:
        raise
    except Exception as exc:
        raise CampaignError("NORMALIZATION_CONFIG_INVALID", str(exc)) from exc


def _tracking_params(config: Mapping[str, Any]) -> dict[str, Any]:
    tracking = config.get("tracking") if isinstance(config.get("tracking"), Mapping) else {}
    quality = config.get("quality") if isinstance(config.get("quality"), Mapping) else {}
    search_radius = tracking.get("search_radius_px", 180)
    if type(search_radius) is not int or search_radius < 1:
        raise CampaignError("TRACKING_CONFIG_INVALID", "search_radius_px must be a positive integer")
    for section, keys in ((tracking, ("hard_corr", "soft_corr", "template_alpha", "max_frame_jump_px", "pixel_size_mm")), (quality, ("fb_tolerance_px", "peak_margin_min", "best_to_second_peak_ratio_min"))):
        for key in keys:
            value = section.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise CampaignError("TRACKING_CONFIG_INVALID", f"{key} must be a number or null")
    for key in ("enable_adaptive", "use_prev_frame_template"):
        if key in tracking and type(tracking[key]) is not bool:
            raise CampaignError("TRACKING_CONFIG_INVALID", f"{key} must be a boolean")
    if "enable_fb_check" in quality and type(quality["enable_fb_check"]) is not bool:
        raise CampaignError("TRACKING_CONFIG_INVALID", "enable_fb_check must be a boolean")
    # Conservative campaign defaults are fixed-reference and hard-only. A
    # caller can request stricter thresholds; adaptive acceptance remains an
    # explicit opt-in and is never silently enabled for campaigns.
    params = {
        "search_radius_base": search_radius,
        "hard_corr": float(tracking.get("hard_corr", 0.55)),
        "soft_corr": float(tracking.get("soft_corr", 0.35)),
        "enable_adaptive": bool(tracking.get("enable_adaptive", False)),
        "use_prev_frame_template": bool(tracking.get("use_prev_frame_template", False)),
        "template_alpha": float(tracking.get("template_alpha", 0.70)),
        "max_frame_jump": tracking.get("max_frame_jump_px"),
        "enable_fb_check": bool(quality.get("enable_fb_check", True)),
        "fb_tolerance": float(quality.get("fb_tolerance_px", 12.0)),
        "pixel_size_mm": tracking.get("pixel_size_mm"),
        "peak_margin_min": float(quality.get("peak_margin_min", core.DEFAULT_PEAK_MARGIN_MIN)),
        "peak_ratio_min": float(quality.get("best_to_second_peak_ratio_min", core.DEFAULT_PEAK_RATIO_MIN)),
    }
    numeric_values = [params[key] for key in ("hard_corr", "soft_corr", "fb_tolerance", "peak_margin_min", "peak_ratio_min", "template_alpha")]
    if not np.isfinite(numeric_values).all():
        raise CampaignError("TRACKING_CONFIG_INVALID", "tracking and quality thresholds must be finite numbers")
    if not (0.0 <= params["soft_corr"] <= 1.0 and 0.0 <= params["hard_corr"] <= 1.0):
        raise CampaignError("TRACKING_CONFIG_INVALID", "correlation thresholds must be in [0, 1]")
    if params["peak_margin_min"] < 0 or params["peak_ratio_min"] < 1 or not 0 <= params["template_alpha"] <= 1:
        raise CampaignError("TRACKING_CONFIG_INVALID", "peak and template settings are outside their valid ranges")
    if params["hard_corr"] < params["soft_corr"]:
        raise CampaignError("TRACKING_CONFIG_INVALID", "hard_corr must be >= soft_corr")
    if params["search_radius_base"] < 1 or params["fb_tolerance"] < 0:
        raise CampaignError("TRACKING_CONFIG_INVALID", "search radius must be positive and FB tolerance non-negative")
    if params["max_frame_jump"] is not None and (not math.isfinite(float(params["max_frame_jump"])) or float(params["max_frame_jump"]) <= 0):
        raise CampaignError("TRACKING_CONFIG_INVALID", "max_frame_jump_px must be positive and finite")
    if params["pixel_size_mm"] is not None and (not math.isfinite(float(params["pixel_size_mm"])) or float(params["pixel_size_mm"]) <= 0):
        raise CampaignError("TRACKING_CONFIG_INVALID", "pixel_size_mm must be positive and finite")
    relaxations = []
    if params["hard_corr"] < _MIN_HARD_CORR or params["soft_corr"] < _MIN_SOFT_CORR:
        relaxations.append("correlation_threshold_below_core_default")
    if params["enable_adaptive"]:
        relaxations.append("adaptive_acceptance_enabled")
    if params["use_prev_frame_template"]:
        relaxations.append("previous_frame_template_enabled")
    if tracking.get("template_policy") not in (None, "fixed_reference"):
        relaxations.append("non_fixed_template_policy_requested")
    if not params["enable_fb_check"] or params["fb_tolerance"] > _MAX_FB_TOLERANCE_PX:
        relaxations.append("forward_backward_check_disabled_or_relaxed")
    if params["peak_margin_min"] < core.DEFAULT_PEAK_MARGIN_MIN or params["peak_ratio_min"] < core.DEFAULT_PEAK_RATIO_MIN:
        relaxations.append("correlation_peak_ambiguity_threshold_relaxed")
    if relaxations:
        raise CampaignError("CAMPAIGN_QUALITY_RELAXATION_FORBIDDEN", ", ".join(relaxations))
    return params


def _gauge_record(group: Mapping[str, Any], tracking_row: Mapping[str, Any] | None, *, failure_reason: str | None = None) -> dict[str, Any]:
    if tracking_row is None:
        return {
            "name": group.get("name"), "accepted": False, "quality_status": "DIC_FRAME_UNAVAILABLE",
            "reason": failure_reason or "DIC_FRAME_UNAVAILABLE", "engineering_strain": None, "true_strain": None,
            "length_px": None, "reference_length_px": None, "endpoint1_x_px": None, "endpoint1_y_px": None,
            "endpoint2_x_px": None, "endpoint2_y_px": None,
        }
    accepted = bool(tracking_row.get("strain_valid", False)) and bool(tracking_row.get("accepted", False))

    def number(key):
        value = tracking_row.get(key)
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def endpoint(candidate_x, candidate_y, used_x, used_y):
        if accepted:
            cx, cy = number(candidate_x), number(candidate_y)
            if cx is not None and cy is not None:
                return cx, cy
        return number(used_x), number(used_y)

    p1 = endpoint("candidate_roi1_center_x_px", "candidate_roi1_center_y_px", "used_roi1_center_x_px", "used_roi1_center_y_px")
    p2 = endpoint("candidate_roi2_center_x_px", "candidate_roi2_center_y_px", "used_roi2_center_x_px", "used_roi2_center_y_px")
    return {
        "name": group.get("name"),
        "accepted": accepted,
        "quality_status": tracking_row.get("tracking_status_code"),
        "reason": tracking_row.get("reason") or tracking_row.get("invalid_reason"),
        "accept_mode": tracking_row.get("accept_mode"),
        "template_policy": tracking_row.get("template_policy"),
        "engineering_strain": number("engineering_strain") if accepted else None,
        "true_strain": number("true_strain") if accepted else None,
        "length_px": number("length_px") if accepted else None,
        "reference_length_px": number("L0_px"),
        "endpoint1_x_px": p1[0], "endpoint1_y_px": p1[1],
        "endpoint2_x_px": p2[0], "endpoint2_y_px": p2[1],
        "candidate_endpoint1_x_px": number("candidate_roi1_center_x_px"),
        "candidate_endpoint1_y_px": number("candidate_roi1_center_y_px"),
        "candidate_endpoint2_x_px": number("candidate_roi2_center_x_px"),
        "candidate_endpoint2_y_px": number("candidate_roi2_center_y_px"),
        "correlation_roi1": number("corr_score_roi1"), "correlation_roi2": number("corr_score_roi2"),
        "peak_ambiguous_roi1": bool(tracking_row.get("peak_is_ambiguous_roi1", False)),
        "peak_ambiguous_roi2": bool(tracking_row.get("peak_is_ambiguous_roi2", False)),
        "fb_error_roi1_px": number("fb_error_roi1_px"), "fb_error_roi2_px": number("fb_error_roi2_px"),
        "fb_status": tracking_row.get("fb_status"),
        "tracking_diagnostics": _as_json(dict(tracking_row)),
    }


def _flat_gauge_columns(gauges: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Expose stable G01…G05 scalar columns while retaining nested diagnostics."""
    flattened: dict[str, Any] = {}
    for index, gauge in enumerate(gauges, start=1):
        name = str(gauge.get("name") or "")
        prefix = name if re.fullmatch(r"G\d{2}", name) else f"G{index:02d}"
        mapping = {
            "accepted": "accepted",
            "engineering_strain": "engineering_strain",
            "true_strain": "true_strain",
            "status": "quality_status",
            "reason": "reason",
            "corr_score_roi1": "correlation_roi1",
            "corr_score_roi2": "correlation_roi2",
            "fb_error_roi1_px": "fb_error_roi1_px",
            "fb_error_roi2_px": "fb_error_roi2_px",
            "fb_status": "fb_status",
            "endpoint1_x_px": "endpoint1_x_px",
            "endpoint1_y_px": "endpoint1_y_px",
            "endpoint2_x_px": "endpoint2_x_px",
            "endpoint2_y_px": "endpoint2_y_px",
            "candidate_endpoint1_x_px": "candidate_endpoint1_x_px",
            "candidate_endpoint1_y_px": "candidate_endpoint1_y_px",
            "candidate_endpoint2_x_px": "candidate_endpoint2_x_px",
            "candidate_endpoint2_y_px": "candidate_endpoint2_y_px",
        }
        for output_suffix, source_key in mapping.items():
            flattened[f"{prefix}_{output_suffix}"] = _as_json(gauge.get(source_key))
    return flattened


def _csv_cell(value: Any) -> str:
    clean = _as_json(value)
    if clean is None:
        return ""
    if isinstance(clean, bool):
        return "true" if clean else "false"
    if isinstance(clean, (dict, list)):
        return json.dumps(clean, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return str(clean)


def _csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_cell(row.get(key)) for key in fields})
    return buffer.getvalue().encode("utf-8-sig")


def _tracking_sidecars(rows: list[dict[str, Any]]) -> dict[str, bytes]:
    endpoints = []
    quality = []
    for frame_row in rows:
        for gauge in frame_row.get("gauge_results", []):
            base = {
                "frame": frame_row["frame"],
                "native_dic_frame": frame_row["native_dic_frame"],
                "dic_sequence_index": frame_row["dic_sequence_index"],
                "dic_path": frame_row["dic_path"],
                "reference_frame": frame_row["reference_frame"],
                "gauge": gauge.get("name"),
                "accepted": gauge.get("accepted"),
                "status": gauge.get("quality_status"),
                "reason": gauge.get("reason"),
            }
            endpoints.append({
                **base,
                "engineering_strain": gauge.get("engineering_strain"),
                "true_strain": gauge.get("true_strain"),
                "reference_length_px": gauge.get("reference_length_px"),
                "length_px": gauge.get("length_px"),
                "endpoint1_x_px": gauge.get("endpoint1_x_px"),
                "endpoint1_y_px": gauge.get("endpoint1_y_px"),
                "endpoint2_x_px": gauge.get("endpoint2_x_px"),
                "endpoint2_y_px": gauge.get("endpoint2_y_px"),
                "candidate_endpoint1_x_px": gauge.get("candidate_endpoint1_x_px"),
                "candidate_endpoint1_y_px": gauge.get("candidate_endpoint1_y_px"),
                "candidate_endpoint2_x_px": gauge.get("candidate_endpoint2_x_px"),
                "candidate_endpoint2_y_px": gauge.get("candidate_endpoint2_y_px"),
            })
            quality.append({
                **base,
                "correlation_roi1": gauge.get("correlation_roi1"),
                "correlation_roi2": gauge.get("correlation_roi2"),
                "peak_ambiguous_roi1": gauge.get("peak_ambiguous_roi1"),
                "peak_ambiguous_roi2": gauge.get("peak_ambiguous_roi2"),
                "fb_error_roi1_px": gauge.get("fb_error_roi1_px"),
                "fb_error_roi2_px": gauge.get("fb_error_roi2_px"),
                "fb_status": gauge.get("fb_status"),
                "accept_mode": gauge.get("accept_mode"),
                "template_policy": gauge.get("template_policy"),
                "tracking_diagnostics": gauge.get("tracking_diagnostics"),
            })
    endpoint_fields = [
        "frame", "native_dic_frame", "dic_sequence_index", "dic_path", "reference_frame", "gauge",
        "accepted", "status", "reason", "engineering_strain", "true_strain", "reference_length_px",
        "length_px", "endpoint1_x_px", "endpoint1_y_px", "endpoint2_x_px", "endpoint2_y_px",
        "candidate_endpoint1_x_px", "candidate_endpoint1_y_px", "candidate_endpoint2_x_px", "candidate_endpoint2_y_px",
    ]
    quality_fields = [
        "frame", "native_dic_frame", "dic_sequence_index", "dic_path", "reference_frame", "gauge",
        "accepted", "status", "reason", "correlation_roi1", "correlation_roi2", "peak_ambiguous_roi1",
        "peak_ambiguous_roi2", "fb_error_roi1_px", "fb_error_roi2_px", "fb_status", "accept_mode",
        "template_policy", "tracking_diagnostics",
    ]
    return {
        "endpoint_trajectories.csv": _csv_bytes(endpoints, endpoint_fields),
        "tracking_qc.csv": _csv_bytes(quality, quality_fields),
    }


def _extensometer_rows(acquisition, frames, output_dir, progress, cancel):
    proposal = acquisition["roi"]
    reference_index = _reference_index(frames, proposal)
    reference_frame = frames[reference_index]
    try:
        reference_raw = core.read_gray_image(reference_frame["path"])
    except Exception as exc:
        raise CampaignError("REFERENCE_IMAGE_UNAVAILABLE", str(exc)) from exc
    norm = _normalization(reference_raw, acquisition.get("analysis_config", {}))
    reference8 = core.normalize_with_bounds(reference_raw, norm)
    groups = proposal.get("groups")
    if not isinstance(groups, list) or not groups:
        raise CampaignError("ROI_GROUPS_UNAVAILABLE", "approved extensometer ROI contains no gauge groups")
    params = _tracking_params(acquisition.get("analysis_config", {}))
    texture = acquisition.get("analysis_config", {}).get("texture", {})
    texture = texture if isinstance(texture, Mapping) else {}
    texture_keys = ("min_texture_std", "min_texture_contrast", "max_saturated_frac", "min_structure_ratio", "max_directional_coherence", "min_periodicity_score")
    if any(key in texture and (isinstance(texture[key], bool) or not isinstance(texture[key], (int, float))) for key in texture_keys):
        raise CampaignError("TEXTURE_CONFIG_INVALID", "texture thresholds must be numeric")
    try:
        texture_params = {
            "min_texture_std": float(texture.get("min_texture_std", 8.0)),
            "min_texture_contrast": float(texture.get("min_texture_contrast", 25.0)),
            "max_saturated_frac": float(texture.get("max_saturated_frac", 0.20)),
            "min_structure_ratio": float(texture.get("min_structure_ratio", core.DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO)),
            "max_directional_coherence": float(texture.get("max_directional_coherence", core.DEFAULT_TEXTURE_MAX_DIRECTIONAL_COHERENCE)),
            "min_periodicity_score": float(texture.get("min_periodicity_score", core.DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE)),
        }
    except (TypeError, ValueError) as exc:
        raise CampaignError("TEXTURE_CONFIG_INVALID", "texture thresholds must be numeric") from exc
    if not np.isfinite(list(texture_params.values())).all():
        raise CampaignError("TEXTURE_CONFIG_INVALID", "texture thresholds must be finite")
    if (
        texture_params["min_texture_std"] < 8.0
        or texture_params["min_texture_contrast"] < 25.0
        or texture_params["max_saturated_frac"] > 0.20
        or texture_params["min_structure_ratio"] < core.DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO
        or texture_params["max_directional_coherence"] > core.DEFAULT_TEXTURE_MAX_DIRECTIONAL_COHERENCE
        or texture_params["min_periodicity_score"] < core.DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE
    ):
        raise CampaignError("CAMPAIGN_QUALITY_RELAXATION_FORBIDDEN", "texture preflight settings may only retain or strengthen core defaults")
    states = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise CampaignError("ROI_GROUP_INVALID", "gauge group must be an object")
        core_group = {**dict(group), "actual_mode": group.get("actual_mode", "distance")}
        try:
            state = core.initialize_extensometer_group_state(reference8, core_group, **texture_params)
        except core.CoreError as exc:
            raise CampaignError(exc.code, str(exc)) from exc
        if state.get("template_policy") != "fixed_reference":
            raise CampaignError("FIXED_REFERENCE_REQUIRED", "campaign analysis requires fixed-reference templates")
        states.append(state)
    rows = []
    reference_source_id = int(reference_frame["frame"])
    height, width = reference_raw.shape[:2]
    tracking_step = 0
    for sequence_index, frame in enumerate(frames):
        _check_cancel(cancel)
        frame_id = int(frame["frame"])
        native_id = int(frame["native_frame"])
        if sequence_index < reference_index:
            gauge_rows = [_gauge_record(group, None, failure_reason="FRAME_BEFORE_REFERENCE_FRAME") for group in groups]
            row = {
                "frame": frame_id,
                "native_dic_frame": native_id,
                "dic_sequence_index": sequence_index,
                "dic_path": str(frame["path"]),
                "reference_frame": int(reference_frame["frame"]),
                "reference_native_dic_frame": int(reference_frame["native_frame"]),
                "force_N": None, "area_mm2": None, "stress_MPa": None,
                "engineering_strain": None, "engineering_strain_sd": None,
                "true_strain": None, "true_strain_sd": None,
                "valid_gauge_count": 0, "requested_gauge_count": len(groups),
                "strain_status": "FRAME_BEFORE_REFERENCE_FRAME",
                "quality_status": "NOT_TRACKED_BEFORE_REFERENCE",
                "reasons": ["FRAME_BEFORE_REFERENCE_FRAME"],
                "gauge_results": gauge_rows,
                **_flat_gauge_columns(gauge_rows),
            }
            rows.append(_as_json(row))
            _notify(progress, (sequence_index + 1) / len(frames), frame=frame_id, native_dic_frame=native_id, status="FRAME_BEFORE_REFERENCE_FRAME", valid_gauge_count=0)
            continue
        image8 = None
        read_failure = None
        try:
            raw = core.read_gray_image(frame["path"])
            if tuple(raw.shape[:2]) != (height, width):
                raise CampaignError("IMAGE_DIMENSION_MISMATCH", f"frame {frame_id} has shape {raw.shape}, expected {(height, width)}")
            image8 = core.normalize_with_bounds(raw, norm)
        except Exception as exc:
            read_failure = str(exc)
        gauge_rows = []
        if image8 is not None:
            for group, state in zip(groups, states):
                try:
                    raw_row, _overlay = core.track_extensometer_group_frame(
                        state, image8, tracking_step, Path(frame["path"]).name, params
                    )
                    gauge_rows.append(_gauge_record(group, raw_row))
                except Exception as exc:
                    gauge_rows.append(_gauge_record(group, None, failure_reason=str(exc)))
        else:
            gauge_rows = [_gauge_record(group, None, failure_reason=read_failure or "DIC_FRAME_UNAVAILABLE") for group in groups]
        tracking_step += 1
        accepted = [gauge for gauge in gauge_rows if gauge["accepted"] and gauge["engineering_strain"] is not None and gauge["true_strain"] is not None]
        valid_count = len(accepted)
        if valid_count >= 3:
            engineering = np.asarray([gauge["engineering_strain"] for gauge in accepted], dtype=float)
            true = np.asarray([gauge["true_strain"] for gauge in accepted], dtype=float)
            eng_mean, eng_sd = float(np.mean(engineering)), float(np.std(engineering, ddof=1))
            true_mean, true_sd = float(np.mean(true)), float(np.std(true, ddof=1))
            strain_status = "TRACKED_MEAN"
        else:
            eng_mean = eng_sd = true_mean = true_sd = None
            strain_status = "INSUFFICIENT_VALID_GAUGES" if valid_count else ("DIC_FRAME_UNAVAILABLE" if read_failure else "NO_VALID_GAUGES")
        row = {
            "frame": frame_id,
            "native_dic_frame": native_id,
            "dic_sequence_index": sequence_index,
            "dic_path": str(frame["path"]),
            "reference_frame": reference_source_id,
            "reference_native_dic_frame": int(reference_frame["native_frame"]),
            "force_N": None,
            "area_mm2": None,
            "stress_MPa": None,
            "engineering_strain": eng_mean,
            "engineering_strain_sd": eng_sd,
            "true_strain": true_mean,
            "true_strain_sd": true_sd,
            "valid_gauge_count": valid_count,
            "requested_gauge_count": len(groups),
            "strain_status": strain_status,
            "quality_status": "FRAME_READ_FAILED" if read_failure else ("ACCEPTED_GAUGES_PRESENT" if valid_count else "ALL_GAUGES_REJECTED"),
            "reasons": [gauge["reason"] for gauge in gauge_rows if gauge.get("reason")],
            "gauge_results": gauge_rows,
            **_flat_gauge_columns(gauge_rows),
        }
        if read_failure:
            row["dic_failure_reason"] = read_failure
        rows.append(_as_json(row))
        _notify(progress, (sequence_index + 1) / len(frames), frame=frame_id, native_dic_frame=native_id, status=strain_status, valid_gauge_count=valid_count)
    return rows, reference_frame, norm


def _fullfield_settings(acquisition, frames, reference_index, output_dir, project_root):
    proposal = acquisition["roi"]
    config = acquisition.get("analysis_config", {})
    if not isinstance(config, Mapping):
        raise CampaignError("ANALYSIS_CONFIG_INVALID", "analysis_config must be an object")
    # Use the same strict v1 configuration validator and adapter as ezdic-cli;
    # campaign metadata cannot silently bypass the established solver contract.
    raw = dict(config)
    approved_field_roi = proposal.get("field_roi", proposal.get("free_region"))
    if "field_roi" in raw and (not isinstance(raw["field_roi"], (list, tuple)) or list(raw["field_roi"]) != list(approved_field_roi)):
        raise CampaignError("FULLFIELD_ROI_CONFIG_CONFLICT", "analysis_config.field_roi must match the human-approved ROI geometry")
    raw.update({
        "schema_version": 1,
        "analysis_mode": "fullfield",
        "image_paths": [frame["path"] for frame in frames],
        "start_frame_1based": 1,
        "end_frame_1based": len(frames),
        "reference_frame_1based": reference_index + 1,
        "output_dir": str(output_dir),
        "field_roi": approved_field_roi,
    })
    try:
        import ezdic_cli

        normalized = ezdic_cli.normalize_config(raw, base_dir=Path(project_root).resolve())
        return ezdic_cli.build_core_settings(normalized, core)
    except Exception as exc:
        raise CampaignError(getattr(exc, "code", "FULLFIELD_CONFIG_INVALID"), str(exc)) from exc


def _fullfield_rows(acquisition, frames, reference_index, result, output_dir, progress, cancel):
    _check_cancel(cancel)
    frame_by_position = {index + 1: frame for index, frame in enumerate(frames)}
    core_records = result.get("frames") or []
    by_position = {int(item["frame_global_1based"]): item for item in core_records if item.get("frame_global_1based") is not None}
    reference = frames[reference_index]
    rows = []
    for position, frame in enumerate(frames, start=1):
        _check_cancel(cancel)
        if position == reference_index + 1:
            core_record = None
            quality = "REFERENCE_FRAME"
            strain_status = "FIELD_ONLY_NO_ENGINEERING_AGGREGATE"
        else:
            core_record = by_position.get(position)
            quality = (core_record or {}).get("status", "CORE_FRAME_RECORD_MISSING")
            strain_status = "FIELD_ONLY_NO_ENGINEERING_AGGREGATE"
        row = {
            "frame": int(frame["frame"]),
            "native_dic_frame": int(frame["native_frame"]),
            "dic_sequence_index": position - 1,
            "dic_path": frame["path"],
            "reference_frame": int(reference["frame"]),
            "reference_native_dic_frame": int(reference["native_frame"]),
            "force_N": None,
            "area_mm2": None,
            "stress_MPa": None,
            "engineering_strain": None,
            "engineering_strain_sd": None,
            "true_strain": None,
            "true_strain_sd": None,
            "valid_gauge_count": 0,
            "requested_gauge_count": 0,
            "strain_status": strain_status,
            "quality_status": quality,
            "field_quality": _as_json(core_record),
            "field_artifact_stem": (core_record or {}).get("artifact_stem"),
            "analysis_mode": "fullfield",
        }
        rows.append(_as_json(row))
        _notify(progress, position / len(frames), frame=int(frame["frame"]), native_dic_frame=int(frame["native_frame"]), status=quality)
    return rows, reference


def analyze(acquisition: Mapping[str, Any], project_dir: str | os.PathLike[str], progress=None, cancel=None) -> dict[str, Any]:
    """Analyze an approved acquisition with ezdic_core and atomically record full rows."""
    if not isinstance(acquisition, Mapping):
        raise CampaignError("ACQUISITION_INVALID", "acquisition must be an object")
    _check_cancel(cancel)
    try:
        roi_module.validate_approval(acquisition)
    except core.CoreError as exc:
        raise CampaignError(exc.code, str(exc)) from exc
    frames = _ordered_frames(acquisition)
    mode = acquisition.get("analysis_mode")
    if mode not in {"extensometer", "fullfield"}:
        raise CampaignError("ANALYSIS_MODE_INVALID", f"unsupported analysis mode {mode!r}")
    fingerprint_dependencies = _fingerprint_dependencies(acquisition)
    input_fingerprint = storage.fingerprint_data(_NUMERICS_FINGERPRINT_VERSION, fingerprint_dependencies)
    source_images_fingerprint = _source_images_fingerprint(fingerprint_dependencies["image_frames"])
    stage_dir = storage.stage_output_dir(project_dir, str(acquisition["acquisition_id"]), "analysis", input_fingerprint)
    _notify(progress, 0.0, status="preflight", acquisition_id=acquisition.get("acquisition_id"), fingerprint=input_fingerprint)
    core_result = None
    reference_frame = None
    normalization = None
    generated_stage_files: list[Path] = []
    try:
        if mode == "extensometer":
            rows, reference_frame, normalization = _extensometer_rows(acquisition, frames, stage_dir, progress, cancel)
            engine_status = "COMPLETED"
            engine_summary = {}
            engine_artifacts = []
        else:
            if len(frames) < 2:
                raise CampaignError("INSUFFICIENT_FRAMES", "fullfield mode requires a reference and at least one deformed image")
            core_output = stage_dir / "core_output"
            core_output.mkdir(parents=True, exist_ok=True)
            reference_index = _reference_index(frames, acquisition["roi"])
            settings = _fullfield_settings(acquisition, frames, reference_index, core_output, project_dir)

            def core_progress(event):
                _check_cancel(cancel)
                if isinstance(event, Mapping):
                    _notify(progress, event.get("fraction", 0.0), **{key: value for key, value in event.items() if key != "fraction"})

            try:
                core_result = core.run_fullfield_sequence(settings, progress_callback=core_progress)
            except CampaignError:
                raise
            except Exception as exc:
                raise CampaignError(getattr(exc, "code", "FULLFIELD_ANALYSIS_FAILED"), str(exc)) from exc
            _check_cancel(cancel)
            rows, reference_frame = _fullfield_rows(acquisition, frames, reference_index, core_result, core_output, progress, cancel)
            engine_status = str(core_result.get("status", "UNKNOWN"))
            engine_summary = _as_json(core_result.get("json_summary", {}))
            engine_artifacts = []
            reported_outputs = list(core_result.get("outputs", []) or [])
            manifest_value = core_result.get("manifest_path")
            manifest_path = Path(str(manifest_value)).resolve() if manifest_value else None
            manifest_data = core_result.get("manifest")
            if manifest_path is not None and isinstance(manifest_data, Mapping):
                for output_record in manifest_data.get("outputs", []) or []:
                    if isinstance(output_record, Mapping) and output_record.get("path"):
                        reported_outputs.append(str(manifest_path.parent / str(output_record["path"])))
            for value in reported_outputs:
                path = Path(value).resolve()
                if path.is_file():
                    engine_artifacts.append(storage.relative_artifact(project_dir, path))
            if manifest_path is not None and manifest_path.is_file():
                engine_artifacts.append(storage.relative_artifact(project_dir, manifest_path))
        _check_cancel(cancel)
        # Rehash to detect source or executable edits during a long run.
        final_fingerprint = fingerprint(acquisition)
        if final_fingerprint != input_fingerprint:
            raise CampaignError("INPUT_CHANGED_DURING_ANALYSIS", "an image, ROI, configuration, or numerical source changed during analysis")
        rows_path = stage_dir / "rows.jsonl"
        storage.write_jsonl_atomic(rows_path, rows)
        generated_stage_files.append(rows_path)
        sidecar_paths = []
        if mode == "extensometer":
            for filename, payload in _tracking_sidecars(rows).items():
                _check_cancel(cancel)
                target = stage_dir / filename
                storage.atomic_write_bytes(target, payload)
                generated_stage_files.append(target)
                sidecar_paths.append(target)
        _check_cancel(cancel)
        rows_relative = storage.relative_artifact(project_dir, rows_path)
        sidecar_relatives = [storage.relative_artifact(project_dir, path) for path in sidecar_paths]
        artifacts = list(dict.fromkeys([rows_relative, *sidecar_relatives, *engine_artifacts]))
        checksums = {relative: storage.sha256_file(storage.artifact_path(project_dir, relative)) for relative in artifacts}
        valid_frames = sum(row.get("valid_gauge_count", 0) >= 3 for row in rows) if mode == "extensometer" else sum((row.get("field_quality") or {}).get("scientific_valid") is True for row in rows)
        summary = {
            "analysis_mode": mode,
            "frame_count": len(rows),
            "valid_mean_frame_count": int(valid_frames),
            "failed_frame_count": int(len(rows) - valid_frames),
            "reference_frame": int(reference_frame["frame"]) if reference_frame else None,
            "reference_native_dic_frame": int(reference_frame["native_frame"]) if reference_frame else None,
            "engine_status": engine_status,
            "scientific_ok": bool(valid_frames > 0) if mode == "extensometer" else bool((engine_summary or {}).get("scientific_ok")),
            "engine_summary": engine_summary,
            "endpoint_row_count": sum(len(row.get("gauge_results", [])) for row in rows),
            "tracking_qc_row_count": sum(len(row.get("gauge_results", [])) for row in rows),
            "scientific_mean_rule": "at least 3 accepted extensometer gauges; sample SD uses ddof=1" if mode == "extensometer" else "fullfield maps only; no engineering-strain aggregate",
        }
        reference_image = next((
            item for item in fingerprint_dependencies["image_frames"]
            if reference_frame is not None
            and item["frame"] == reference_frame["frame"]
            and item["native_frame"] == reference_frame["native_frame"]
        ), None)
        if reference_frame is not None and reference_image is None:
            raise CampaignError("ROI_REFERENCE_FRAME_INVALID", "reference image is absent from numerical fingerprint inputs")
        reference_sha256 = reference_image["sha256"] if reference_image else None
        reference_fingerprint = storage.fingerprint_data("roi_reference_v1", {
            "acquisition_id": acquisition.get("acquisition_id"),
            "reference_frame": summary["reference_frame"],
            "native_frame": reference_frame.get("native_frame") if reference_frame else None,
            "reference_sha256": reference_sha256,
        }) if reference_frame else None
        provenance = {
            "engine": "ezdic_core",
            "engine_version": core.APP_VERSION,
            "analysis_mode": mode,
            "input_fingerprint": input_fingerprint,
            "source_images_fingerprint": source_images_fingerprint,
            "engine_code_sha256": _code_fingerprints(),
            "analysis_config": _as_json(acquisition.get("analysis_config", {})),
            "roi_proposal_hash": acquisition.get("roi", {}).get("proposal_hash"),
            "approval": _as_json(acquisition.get("approval")),
            "reference_frame": summary["reference_frame"],
            "reference_native_dic_frame": summary["reference_native_dic_frame"],
            "reference_sha256": reference_sha256,
            "reference_fingerprint": reference_fingerprint,
            "normalization": _as_json(normalization) if normalization is not None else None,
            "frame_identity_policy": "rows retain explicit image_frames.frame and native_frame; gaps are not filled",
            "tracking_template_policy": "fixed_reference" if mode == "extensometer" else "fixed_reference_field",
            "tracking_sidecar_artifacts": sidecar_relatives if mode == "extensometer" else [],
        }
        _notify(progress, 1.0, status="complete", acquisition_id=acquisition.get("acquisition_id"), frame_count=len(rows))
        _check_cancel(cancel)
        return {
            "status": "COMPLETED" if engine_status in {"COMPLETED", "completed", "completed_with_warnings"} else engine_status.upper(),
            "fingerprint": input_fingerprint,
            "rows_path": rows_relative,
            "artifacts": artifacts,
            "checksums": checksums,
            "provenance": provenance,
            "summary": summary,
        }
    except CampaignError:
        for path in generated_stage_files:
            path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        for path in generated_stage_files:
            path.unlink(missing_ok=True)
        raise CampaignError("CAMPAIGN_ANALYSIS_FAILED", str(exc)) from exc
