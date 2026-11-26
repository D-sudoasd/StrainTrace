"""Shared run provenance binding and read-only integrity verification."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ._config import (
    QUALITY_GATE_SCOPE,
    _read_config,
    _solver_parameters_match,
    _validate_plan_checksum,
)
from ._inventory import _code_fingerprint, _inventory_fresh
from ._io import (
    AgentError,
    _api_attr,
    _current_pointer,
    _hash_json,
    _project_root,
    _read_json,
    _safe_file,
    _sha256,
)
from ._outputs import _load_field, _run_fingerprint


_REQUIRED_RUN_ARTIFACTS = frozenset({
    "summary.csv", "quality.json", "gauges.csv", "regions.csv", "geometry.json", "report.md",
})


def _artifact_registration_issues(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return [{"code": "MANIFEST_ARTIFACTS_INVALID"}]
    return [
        {"code": "ARTIFACT_UNREGISTERED", "path": name}
        for name in sorted(_REQUIRED_RUN_ARTIFACTS - set(artifacts))
    ]


def _stale_derived_artifact_issues(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    stale = manifest.get("stale_derived_artifacts", [])
    if not isinstance(stale, list) or any(not isinstance(path, str) or not path.startswith("extractions/") for path in stale):
        return [{"code": "STALE_DERIVED_ARTIFACTS_INVALID"}]
    return [{"code": "DERIVED_ARTIFACT_STALE", "path": path} for path in stale]


def _artifact_file_issues(run_root: Path, artifacts: Mapping[str, Any], *, prefixes: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    sha256_fn = _api_attr("_sha256", _sha256)
    issues: list[dict[str, Any]] = []
    for relative, expected in artifacts.items():
        if prefixes and (not isinstance(relative, str) or not relative.startswith(prefixes)):
            continue
        try:
            if not isinstance(relative, str):
                raise ValueError("artifact path must be a string")
            path = _safe_file(run_root, relative)
        except ValueError as exc:
            issues.append({"code": "ARTIFACT_PATH_UNSAFE", "path": str(relative), "message": str(exc)})
            continue
        if not path.is_file():
            issues.append({"code": "ARTIFACT_MISSING", "path": relative})
        elif sha256_fn(path) != expected:
            issues.append({"code": "ARTIFACT_CHECKSUM_MISMATCH", "path": relative})
    return issues


_RETAINED_PREFIXES = ("plots/", "fields_csv/", "extractions/")


def _unregistered_retained_artifact_issues(run_root: Path, artifacts: Mapping[str, Any]) -> list[dict[str, Any]]:
    issues = []
    for subdir in _RETAINED_PREFIXES:
        try:
            directory = _safe_file(run_root, subdir.rstrip("/"))
        except ValueError as exc:
            issues.append({"code": "ARTIFACT_PATH_UNSAFE", "path": subdir, "message": str(exc)})
            continue
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(run_root).as_posix()
                    if relative not in artifacts:
                        issues.append({"code": "ARTIFACT_UNREGISTERED", "path": relative})
    return issues


def _require_retained_artifacts_integrity(
    run_root: Path, manifest: Mapping[str, Any], *, regenerated_paths: set[str] | None = None,
) -> None:
    """Do not bless changed optional products which a rerun may retain."""
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        issues = [{"code": "MANIFEST_ARTIFACTS_INVALID"}]
    else:
        issues = _artifact_file_issues(run_root, artifacts, prefixes=_RETAINED_PREFIXES)
        issues.extend(_unregistered_retained_artifact_issues(run_root, artifacts))
        if regenerated_paths:
            # These exact owned names are replaced or removed by the full
            # field-CSV export. Other user/derived files remain protected.
            issues = [
                issue for issue in issues
                if not (
                    issue.get("code") in {"ARTIFACT_MISSING", "ARTIFACT_CHECKSUM_MISMATCH", "ARTIFACT_UNREGISTERED"}
                    and issue.get("path") in regenerated_paths
                )
            ]
    if issues:
        raise AgentError(
            "RUN_ARTIFACT_INVALID",
            "retained derived files are missing, changed, or unregistered; restore the recorded files before rerunning",
            details=issues,
        )


def _require_recorded_run_integrity(
    run_root: Path, manifest: Mapping[str, Any], *, allow_stale_derived: bool = False,
) -> None:
    """Check saved bytes before a derived operation can refresh their hashes.

    Only recorded outputs are required, so a cancelled run can still export
    its completed frames. Missing frames remain an execution/quality issue.
    """
    sha256_fn = _api_attr("_sha256", _sha256)
    issues = _artifact_registration_issues(manifest)
    stale_issues = _stale_derived_artifact_issues(manifest)
    issues.extend(issue for issue in stale_issues if not allow_stale_derived or issue["code"] != "DERIVED_ARTIFACT_STALE")
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, Mapping):
        issues.extend(_artifact_file_issues(run_root, artifacts))
        issues.extend(_unregistered_retained_artifact_issues(run_root, artifacts))
    receipts = manifest.get("checkpoint_receipts")
    if not isinstance(receipts, Mapping):
        issues.append({"code": "MANIFEST_RECEIPTS_INVALID"})
        receipts = {}
    recorded_names = set(receipts)
    actual_names = {path.name for path in (run_root / "checkpoints").glob("frame_native_*.json")}
    for name in sorted(actual_names - recorded_names):
        issues.append({"code": "UNREGISTERED_FRAME_RECEIPT", "path": name})
    expected_checkpoints = {
        str(Path(name).with_suffix(".npz")) for name, receipt in receipts.items()
        if isinstance(receipt, Mapping) and receipt.get("checkpoint_sha256") is not None
    }
    actual_checkpoints = {path.name for path in (run_root / "checkpoints").glob("frame_native_*.npz")}
    for name in sorted(actual_checkpoints - expected_checkpoints):
        issues.append({"code": "UNREGISTERED_FRAME_CHECKPOINT", "path": name})
    for name, receipt in receipts.items():
        if not isinstance(receipt, Mapping):
            issues.append({"code": "RECEIPT_RECORD_INVALID", "path": str(name)})
            continue
        try:
            path = _safe_file(run_root, f"checkpoints/{name}")
        except ValueError as exc:
            issues.append({"code": "RECEIPT_PATH_UNSAFE", "path": str(name), "message": str(exc)})
            continue
        if not path.is_file():
            issues.append({"code": "FRAME_RECEIPT_MISSING", "path": name})
            continue
        if sha256_fn(path) != receipt.get("sha256"):
            issues.append({"code": "RECEIPT_CHECKSUM_MISMATCH", "path": name})
            continue
        if receipt.get("checkpoint_sha256") is not None:
            checkpoint = path.with_suffix(".npz")
            checkpoint = _safe_file(run_root, checkpoint.relative_to(run_root.resolve()).as_posix())
            if not checkpoint.is_file():
                issues.append({"code": "CHECKPOINT_MISSING", "path": checkpoint.name})
            elif sha256_fn(checkpoint) != receipt["checkpoint_sha256"]:
                issues.append({"code": "CHECKPOINT_CHECKSUM_MISMATCH", "path": checkpoint.name})
    if issues:
        raise AgentError(
            "RUN_ARTIFACT_INVALID",
            "saved run files changed or are missing; rerun analyze to regenerate outputs from verified checkpoints before plotting, extracting or exporting",
            details=issues,
        )


def _require_current_run_binding(
    root: Path, config: Mapping[str, Any], run_root: Path,
    plan: Mapping[str, Any], run_state: Mapping[str, Any],
    *, allow_stale_derived: bool = False,
) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    """Reject derived-output writes when their run no longer binds current inputs."""
    code_fp_fn = _api_attr("_code_fingerprint", _code_fingerprint)
    plan_copy = dict(plan)
    saved_plan_sha = _validate_plan_checksum(plan_copy)
    records, input_sha = _inventory_fresh(root, config)
    if plan.get("input_fingerprint") != input_sha or run_state.get("input_fingerprint") != input_sha:
        raise AgentError("RUN_STALE_INPUT", "raw images, load data, frame map, or mask changed since this run")
    config_sha = _hash_json(config)
    if plan.get("configuration_sha256") != config_sha:
        raise AgentError("RUN_STALE_CONFIG", "project configuration changed since this run")
    if run_root.joinpath("manifest.json").is_file():
        manifest = _read_json(run_root / "manifest.json")
    else:
        raise AgentError("RUN_MANIFEST_MISSING", "the current run has no manifest", path=str(run_root / "manifest.json"))
    if manifest.get("configuration_sha256") != config_sha:
        raise AgentError("RUN_STALE_CONFIG", "run manifest does not match the current project configuration")
    if manifest.get("plan_sha256") != saved_plan_sha:
        raise AgentError("RUN_STALE_PLAN", "run manifest does not match the saved analysis plan")
    if manifest.get("input_fingerprint") != input_sha:
        raise AgentError("RUN_STALE_INPUT", "run manifest does not match current source inputs")
    current_code = code_fp_fn()
    if plan.get("code_sha256") != current_code or run_state.get("code_sha256") != current_code or manifest.get("code_sha256") != current_code:
        raise AgentError("RUN_STALE_CODE", "analysis code changed since this run")
    import ezdic_core as core

    current_environment = core.collect_environment()
    if plan.get("environment") != current_environment or manifest.get("environment") != current_environment:
        raise AgentError("RUN_STALE_ENVIRONMENT", "runtime packages or platform changed since this run")
    expected_fingerprint = _run_fingerprint(config, plan, input_sha, current_code)
    if (run_state.get("run_fingerprint") != expected_fingerprint
            or manifest.get("run_fingerprint") != expected_fingerprint
            or run_root.name != str(run_state.get("run_id"))):
        raise AgentError("RUN_STALE_BINDING", "run identity does not match current inputs, configuration, and code")
    _require_recorded_run_integrity(run_root, manifest, allow_stale_derived=allow_stale_derived)
    return records, input_sha, current_code


def verify(project: str | os.PathLike[str]) -> dict[str, Any]:
    """Read-only integrity check for source inventory, current plan/run, and outputs."""
    sha256_fn = _api_attr("_sha256", _sha256)
    code_fp_fn = _api_attr("_code_fingerprint", _code_fingerprint)
    load_field_fn = _api_attr("_load_field", _load_field)
    root = _project_root(project)
    config = _read_config(root)
    pointer = _current_pointer(root)
    records, input_sha = _inventory_fresh(root, config)
    config_sha = _hash_json(config)
    current_code = code_fp_fn()
    import ezdic_core as core

    current_environment = core.collect_environment()
    issues: list[dict[str, Any]] = []
    try:
        stored_inventory = _read_json(root / "inventory.json")
        if stored_inventory.get("input_fingerprint") != input_sha:
            issues.append({"code": "PROJECT_INVENTORY_STALE"})
    except AgentError as exc:
        issues.append({"code": exc.code, "path": "inventory.json", "message": exc.message})
    plan: dict[str, Any] | None = None
    if pointer.get("current_plan"):
        try:
            plan_path = _safe_file(root, str(pointer["current_plan"]))
            plan = _read_json(plan_path)
        except (ValueError, AgentError) as exc:
            issues.append({"code": "PLAN_INVALID", "path": str(pointer["current_plan"]), "message": str(exc)})
            plan = None
        if plan is not None:
            expected = plan.pop("plan_sha256", None)
            if expected != _hash_json(plan):
                issues.append({"code": "PLAN_CHECKSUM_MISMATCH"})
            plan["plan_sha256"] = expected
            if plan.get("configuration_sha256") != config_sha:
                issues.append({"code": "PLAN_STALE_CONFIG"})
            if plan.get("input_fingerprint") != input_sha:
                issues.append({"code": "PLAN_STALE_INPUT"})
            if plan.get("code_sha256") != current_code:
                issues.append({"code": "PLAN_STALE_CODE"})
    else:
        issues.append({"code": "PLAN_MISSING"})
    if not pointer.get("current_run"):
        issues.append({"code": "RUN_MISSING"})
        return {"project": str(root), "valid": False, "scientific_ok": False, "numerical_quality_passed": False, "issues": issues, "input_fingerprint": input_sha}
    try:
        run_root = _safe_file(root, str(pointer["current_run"]))
    except ValueError as exc:
        issues.append({"code": "RUN_PATH_UNSAFE", "path": str(pointer["current_run"]), "message": str(exc)})
        return {"project": str(root), "valid": False, "scientific_ok": False, "numerical_quality_passed": False, "issues": issues, "input_fingerprint": input_sha}
    if not run_root.is_dir():
        issues.append({"code": "RUN_MISSING", "path": str(pointer["current_run"])})
        return {"project": str(root), "valid": False, "scientific_ok": False, "numerical_quality_passed": False, "issues": issues, "input_fingerprint": input_sha}
    try:
        state = _read_json(run_root / "run_state.json")
    except AgentError as exc:
        issues.append({"code": "RUN_STATE_INVALID", "error_code": exc.code, "path": "run_state.json", "message": exc.message})
        state = {}
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        issues.append({"code": "MANIFEST_MISSING"})
        manifest: dict[str, Any] = {}
    else:
        try:
            manifest = _read_json(manifest_path)
        except AgentError as exc:
            issues.append({"code": "MANIFEST_INVALID", "error_code": exc.code, "path": "manifest.json", "message": exc.message})
            manifest = {}
    expected_fingerprint = _run_fingerprint(config, plan or {}, input_sha, current_code)
    if state.get("run_fingerprint") != expected_fingerprint:
        issues.append({"code": "RUN_FINGERPRINT_STALE"})
    if manifest and (manifest.get("run_fingerprint") != expected_fingerprint or manifest.get("run_fingerprint") != state.get("run_fingerprint")):
        issues.append({"code": "MANIFEST_RUN_FINGERPRINT_MISMATCH"})
    if manifest and manifest.get("configuration_sha256") != config_sha:
        issues.append({"code": "CONFIG_FINGERPRINT_MISMATCH"})
    if manifest and plan is not None and manifest.get("plan_sha256") != plan.get("plan_sha256"):
        issues.append({"code": "PLAN_FINGERPRINT_MISMATCH"})
    if str(state.get("run_id")) != run_root.name or (manifest and str(manifest.get("run_id")) != run_root.name):
        issues.append({"code": "RUN_ID_MISMATCH"})
    if state and state.get("input_fingerprint") != input_sha:
        issues.append({"code": "RUN_STATE_INPUT_MISMATCH"})
    if state and state.get("code_sha256") != current_code:
        issues.append({"code": "RUN_STATE_CODE_MISMATCH"})
    if manifest.get("run_status") != state.get("status"):
        issues.append({"code": "RUN_STATUS_MISMATCH"})
    if manifest.get("expected_frame_count") != len(records):
        issues.append({"code": "EXPECTED_FRAME_COUNT_MISMATCH", "expected": len(records), "actual": manifest.get("expected_frame_count")})
    if manifest.get("input_fingerprint") != input_sha:
        issues.append({"code": "INPUT_FINGERPRINT_MISMATCH"})
    if manifest.get("code_sha256") != current_code:
        issues.append({"code": "CODE_FINGERPRINT_MISMATCH"})
    if manifest.get("environment") != current_environment or (plan or {}).get("environment") != current_environment:
        issues.append({"code": "ENVIRONMENT_MISMATCH"})
    issues.extend(_artifact_registration_issues(manifest))
    issues.extend(_stale_derived_artifact_issues(manifest))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        artifacts = {}
    issues.extend(_artifact_file_issues(run_root, artifacts))
    issues.extend(_unregistered_retained_artifact_issues(run_root, artifacts))
    expected_frames = {int(row["native_frame"]): row for row in records}
    manifest_receipts = manifest.get("checkpoint_receipts")
    if manifest_receipts is not None and not isinstance(manifest_receipts, Mapping):
        issues.append({"code": "MANIFEST_RECEIPTS_INVALID"})
        manifest_receipts = {}
    manifest_receipts = manifest_receipts or {}
    seen: dict[int, Path] = {}
    checkpoint_count = 0
    checkpoint_dir = run_root / "checkpoints"
    receipt_paths = sorted(checkpoint_dir.glob("frame_native_*.json")) if checkpoint_dir.is_dir() else []
    actual_receipt_names = {path.name for path in receipt_paths}
    for name in sorted(set(manifest_receipts) - actual_receipt_names):
        issues.append({"code": "REGISTERED_RECEIPT_MISSING", "path": name})
    expected_checkpoints = {
        str(Path(name).with_suffix(".npz"))
        for name, receipt in manifest_receipts.items()
        if isinstance(receipt, Mapping) and receipt.get("checkpoint_sha256") is not None
    }
    actual_checkpoints = {path.name for path in checkpoint_dir.glob("frame_native_*.npz")} if checkpoint_dir.is_dir() else set()
    for name in sorted(actual_checkpoints - expected_checkpoints):
        issues.append({"code": "UNREGISTERED_FRAME_CHECKPOINT", "path": name})
    for metadata_path in receipt_paths:
        try:
            safe_meta = _safe_file(run_root, f"checkpoints/{metadata_path.name}")
            metadata = _read_json(safe_meta)
        except (ValueError, AgentError) as exc:
            issues.append({"code": "RECEIPT_INVALID", "path": metadata_path.name, "message": str(exc)})
            continue
        try:
            native = int(metadata["native_frame"])
        except (KeyError, TypeError, ValueError):
            issues.append({"code": "RECEIPT_NATIVE_FRAME_INVALID", "path": metadata_path.name})
            continue
        if native not in expected_frames:
            issues.append({"code": "UNEXPECTED_FRAME_RECEIPT", "native_frame": native})
        if native in seen:
            issues.append({"code": "DUPLICATE_FRAME_RECEIPT", "native_frame": native})
        seen[native] = metadata_path
        expected = expected_frames.get(native)
        if expected and (metadata.get("source_sha256") != expected.get("sha256") or metadata.get("source_path") != expected.get("path")):
            issues.append({"code": "RECEIPT_SOURCE_MISMATCH", "native_frame": native})
        if metadata.get("run_fingerprint") != state.get("run_fingerprint") or metadata.get("input_fingerprint") != input_sha or metadata.get("code_sha256") != current_code:
            issues.append({"code": "RECEIPT_STALE", "native_frame": native})
        if not _solver_parameters_match(metadata.get("parameters"), (plan or {}).get("selected_parameters")):
            issues.append({"code": "CHECKPOINT_PARAMETER_MISMATCH", "native_frame": native})
        receipt_record = manifest_receipts.get(metadata_path.name)
        if not isinstance(receipt_record, Mapping) or receipt_record.get("sha256") != sha256_fn(safe_meta):
            issues.append({"code": "RECEIPT_CHECKSUM_MISMATCH", "native_frame": native})
        if metadata.get("status") == "failed":
            if not isinstance(metadata.get("error"), Mapping):
                issues.append({"code": "FAILED_RECEIPT_REASON_MISSING", "native_frame": native})
            if metadata.get("checkpoint") is not None:
                issues.append({"code": "FAILED_RECEIPT_HAS_NPZ", "native_frame": native})
            continue
        checkpoint = metadata_path.with_suffix(".npz")
        try:
            safe_checkpoint = _safe_file(run_root, f"checkpoints/{checkpoint.name}")
            arrays, loaded = load_field_fn(safe_checkpoint, safe_meta)
            if not arrays:
                issues.append({"code": "CHECKPOINT_EMPTY", "path": checkpoint.name})
            if loaded.get("checkpoint_sha256") != (receipt_record if isinstance(receipt_record, Mapping) else {}).get("checkpoint_sha256"):
                issues.append({"code": "CHECKPOINT_MANIFEST_CHECKSUM_MISMATCH", "native_frame": native})
            if loaded.get("input_fingerprint") != input_sha or loaded.get("code_sha256") != current_code:
                issues.append({"code": "CHECKPOINT_STALE", "path": checkpoint.name})
            checkpoint_count += 1
        except (OSError, ValueError, AgentError) as exc:
            issues.append({"code": exc.code if isinstance(exc, AgentError) else ("CHECKPOINT_PATH_UNSAFE" if isinstance(exc, ValueError) else "CHECKPOINT_MISSING"), "path": checkpoint.name})
    for native in sorted(set(expected_frames) - set(seen)):
        issues.append({"code": "FRAME_RECEIPT_MISSING", "native_frame": native})
    if len(receipt_paths) != len(records):
        issues.append({"code": "FRAME_RECEIPT_COUNT_MISMATCH", "expected": len(records), "actual": len(receipt_paths)})
    if checkpoint_count != manifest.get("checkpoint_count"):
        issues.append({"code": "CHECKPOINT_COUNT_MISMATCH", "expected": manifest.get("checkpoint_count"), "actual": checkpoint_count})
    summary_native_matches_expected = False
    summary_quality_passed = False
    try:
        summary_table = pd.read_csv(run_root / "summary.csv")
        if len(summary_table) != len(records):
            issues.append({"code": "SUMMARY_FRAME_COUNT_MISMATCH", "expected": len(records), "actual": len(summary_table)})
        if "native_frame" in summary_table:
            summary_native_frames = [int(value) for value in summary_table["native_frame"].tolist()]
            summary_native_matches_expected = summary_native_frames == [int(row["native_frame"]) for row in records]
        if not summary_native_matches_expected:
            issues.append({"code": "SUMMARY_FRAME_IDENTITY_MISMATCH"})
        if "status" in summary_table and "scientific_ok" in summary_table:
            summary_quality_passed = bool(
                len(summary_table)
                and summary_table["status"].eq("complete").all()
                and summary_table["scientific_ok"].eq(True).all()
            )
        computed_summary_count = int(summary_table["status"].ne("pending").sum()) if "status" in summary_table else len(summary_table)
        if manifest.get("completed_frame_count") != computed_summary_count:
            issues.append({"code": "COMPLETED_FRAME_COUNT_MISMATCH", "expected": computed_summary_count, "actual": manifest.get("completed_frame_count")})
    except Exception as exc:
        issues.append({"code": "SUMMARY_INVALID", "message": str(exc)})
    numerical_quality_passed = (
        not issues
        and state.get("status") == "complete"
        and manifest.get("run_status") == "complete"
        and manifest.get("expected_frame_count") == len(records)
        and manifest.get("completed_frame_count") == len(records)
        and manifest.get("native_frames_match_expected") is True
        and summary_native_matches_expected
        and summary_quality_passed
        and manifest.get("numerical_quality_passed", manifest.get("scientific_ok") is True) is True
    )
    return {
        "project": str(root), "valid": not issues, "scientific_ok": numerical_quality_passed,
        "numerical_quality_passed": numerical_quality_passed,
        "quality_gate_scope": manifest.get("quality_gate_scope", QUALITY_GATE_SCOPE),
        "scientific_acceptance": manifest.get("scientific_acceptance", "not_assessed"),
        "experimental_uncertainty_calibrated": manifest.get("experimental_uncertainty_calibrated", False),
        "issues": issues, "input_fingerprint": input_sha, "checkpoint_count": checkpoint_count,
        "receipt_count": len(receipt_paths), "expected_frame_count": len(records), "run_status": state.get("status"),
    }
