"""Bind an explicitly selected reference from an earlier linked experiment."""

from __future__ import annotations

import json
import os
import sqlite3
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .calibration import calibration_geometry, phase_sensor_geometry
from .cameras.base import Device
from .config import CaptureError
from .continuation import camera_identity
from .event_images import event_scope, event_sequence
from .handoff import ANALYSIS_SCHEMA, CAPTURE_BINDING, analysis_source, derive_analysis_inputs, select_camera, write_snapshot
from .image_inputs import recorded_image, require_part_geometry, select_frame_image, validate_analysis_part, verified_image_path
from .related import inspect_related_experiments
from .storage import TERMINAL_STATES, ExperimentReader, sha256_file


BINDING_SCHEMA = "straintrace.capture-continuation-analysis/v1"
BINDING_PREFIX = "continued-"
LIVE_REFERENCE_SCHEMA = "straintrace.capture-live-reference/v1"
LIVE_REFERENCE_PREFIX = "live-reference-"


def reference_experiments(root, *, search_dirs=(), recent_paths=(), expected_id=None):
    report = inspect_related_experiments(root, search_dirs=search_dirs, recent_paths=recent_paths,
                                         expected_experiment_id=expected_id)
    return [row for row in reversed(report["runs"]) if row["relation"] in ("parent", "ancestor")]


def _camera(reader, camera_id):
    return next((row for row in reader.cameras() if row["device"]["id"] == camera_id), None)


def _value_map(camera):
    return {name: feature["value"] for name, feature in camera["features"].items() if feature.get("readable", True)}


def _scope(target, source, camera_id, *, live=False):
    current, prior = target.experiment, source.experiment
    if prior["state"] not in TERMINAL_STATES or (not live and current["state"] not in TERMINAL_STATES):
        raise CaptureError("CONTINUATION_REFERENCE_ACTIVE", "请先结束两轮采集。")
    parent = current["config"]["metadata"].get("continuation_of")
    if not parent or current["experiment_id"] == prior["experiment_id"]:
        raise CaptureError("CONTINUATION_REFERENCE_SOURCE", "请选择台账关联的上一轮或更早的实验。")
    if parent != prior["experiment_id"]:
        candidates = reference_experiments(target.root, search_dirs=(source.root,), expected_id=current["experiment_id"])
        if not any(row["experiment_id"] == prior["experiment_id"] and Path(row["directory"]) == source.root
                   for row in candidates):
            raise CaptureError("CONTINUATION_REFERENCE_SOURCE", "请选择台账关联的上一轮或更早的实验。")
    old, new = _camera(source, camera_id), _camera(target, camera_id)
    if old is None or new is None:
        raise CaptureError("CONTINUATION_REFERENCE_CAMERA", "两轮实验均需登记本次选择的相机。")
    identity = camera_identity({camera_id: old["device"]}, camera_id, Device(**new["device"]))
    if identity not in ("matched", "path_matched") or old["device"]["model"] != new["device"]["model"]:
        raise CaptureError("CONTINUATION_REFERENCE_CAMERA", "相机序列号或设备路径无法一致核对。")
    return current, prior, old, new, identity


def recorded_reference_choices(root, source, camera_id, *, live=False):
    with ExperimentReader(root) as target_reader, ExperimentReader(source) as source_reader:
        _scope(target_reader, source_reader, camera_id, live=live)
        page = source_reader.frame_page(camera_id=camera_id, role="reference", limit=128)
        return [(row["sequence"], row["phase"]) for row in page["frames"]]


def require_continuation_geometry(reference, geometry, row, features):
    require_part_geometry(reference, row)
    observed = calibration_geometry(row["shape"], features, row.get("backend_metadata") or {})
    if observed != geometry:
        raise CaptureError("CONTINUATION_REFERENCE_GEOMETRY", f"本轮序号 {row['sequence']} 的成像区域或采样方式不同。")


def continuation_snapshot(root, source, camera_id, reference_sequence, *, image_part=None, live=False, event_context_id=None):
    """Keep the prior reference first, followed by the new experiment's test sequence."""
    validate_analysis_part(image_part)
    if live and event_context_id is not None:
        raise CaptureError("EVENT_IMAGES_ACTIVE", "按事件取图适用于已经完成写入的离线分析。")
    if type(reference_sequence) is not int or not 1 <= reference_sequence < 2**63:
        raise CaptureError("CONTINUATION_REFERENCE_SEQUENCE", "请输入原实验登记的参考图序号。")
    with ExperimentReader(root) as target, ExperimentReader(source) as prior:
        camera_id = select_camera(target, camera_id)
        experiment, origin, old_camera, camera, basis = _scope(target, prior, camera_id, live=live)
        reference = prior.frame_at_sequence(camera_id, reference_sequence)
        if reference is None or reference["role"] != "reference":
            raise CaptureError("CONTINUATION_REFERENCE_SEQUENCE", "请选择原实验中用途标为参考帧的图像。")
        reference = select_frame_image(reference, image_part)
        ref_path = verified_image_path(prior, reference)
        old_geometry = phase_sensor_geometry(prior, camera_id).get(str(reference["phase"]), _value_map(old_camera))
        reference_geometry = calibration_geometry(reference["shape"], old_geometry, reference.get("backend_metadata") or {})
        phases = phase_sensor_geometry(target, camera_id)
        features = _value_map(camera)
        frames = [{**reference, "absolute_path": str(ref_path), "analysis_frame": 1,
                   "source_experiment_id": origin["experiment_id"], "analysis_sensor_geometry": old_geometry}]
        event = None
        if event_context_id is not None:
            event = event_scope(target, experiment, event_context_id)
            tests = list(event_sequence(target, camera_id, event))
        elif live:
            first, _ = target.next_test_frame(camera_id, after=0, latest=False)
            tests = [first] if first is not None else []
        else:
            tests = sorted((row for row in target.frames(camera_id=camera_id) if row["role"] == "test"),
                           key=lambda row: row["sequence"])
        for number, row in enumerate(tests, 2):
            row = select_frame_image(row, image_part)
            phase = phases.get(str(row["phase"]), features)
            require_continuation_geometry(reference, reference_geometry, row, phase)
            absolute = target.image_path(row)
            if not absolute.is_file():
                raise CaptureError("FRAME_UNAVAILABLE", row["path"])
            frames.append({**row, "absolute_path": str(absolute), "analysis_frame": number})
        if not tests and not live:
            raise CaptureError("TOO_FEW_FRAMES", "本轮实验需要至少一张已保存的实验图。")
        try:
            relative = os.path.relpath(prior.root, target.root)
        except ValueError:
            relative = None
        return {
            "schema": ANALYSIS_SCHEMA, "experiment_id": experiment["experiment_id"], "experiment_root": str(target.root),
            "experiment_state": experiment["state"], "camera_id": camera_id, "image_part": image_part,
            **({"event": event} if event is not None else {}),
            "reference_selection": "continuation_reference", "reference_frame_id": reference["frame_id"],
            "external_reference": {"experiment_id": origin["experiment_id"], "name": origin["config"]["name"],
                                   "directory": str(prior.root), "relative_directory": relative,
                                   "sequence": reference_sequence, "identity_basis": basis},
            "metadata": experiment["config"]["metadata"], "calibrations": experiment.get("calibrations", []),
            "camera_features": features, "phase_sensor_geometry": phases, "frames": frames,
        }


def preview_continuation(snapshot):
    return [recorded_image(Path(row["absolute_path"]), row, verify=True) for row in snapshot["frames"][:2]]


def _identities(snapshot):
    return [(row["frame_id"], row["raw_sha256"]) for row in analysis_source(snapshot)["frames"]]


def _analysis_inputs_folder(experiment):
    root = Path(experiment) / "analysis-inputs"
    if root.resolve() != root:
        raise CaptureError("UNSAFE_ANALYSIS_OUTPUT", "分析输入必须写在本实验的 analysis-inputs 内。")
    root.mkdir(exist_ok=True)
    return root


def prepare_continuation_binding(preview):
    """Recheck the chosen inputs and publish the independent Analysis binding."""
    source = preview["external_reference"]
    snapshot = continuation_snapshot(preview["experiment_root"], source["directory"], preview["camera_id"],
                                     source["sequence"], image_part=preview["image_part"])
    if (snapshot["experiment_id"] != preview["experiment_id"] or snapshot["external_reference"] != source
            or _identities(snapshot) != _identities(preview)):
        raise CaptureError("CONTINUATION_REFERENCE_CHANGED", "实验或原图记录已变化，请重新预览后选择。")
    first_test = snapshot["frames"][1]
    if sha256_file(Path(first_test["absolute_path"])) != first_test["sha256"]:
        raise CaptureError("CAPTURE_IMAGE_CHANGED", first_test["frame_id"])
    root = _analysis_inputs_folder(snapshot["experiment_root"])
    target = root / (BINDING_PREFIX + uuid4().hex[:12])
    with TemporaryDirectory(prefix=".linking-", dir=root) as work:
        stage = Path(work) / "inputs"
        converted = derive_analysis_inputs(snapshot, stage, debayer=True, drop_alpha=True)
        stage.mkdir(exist_ok=True)
        data = {**analysis_source(converted), "schema": BINDING_SCHEMA, "reference": source,
                "image_part": snapshot["image_part"]}
        for receipt, row in zip(data["frames"], converted["frames"], strict=True):
            if "analysis_path" in row:
                receipt["derived_name"] = Path(row["analysis_path"]).name
        write_snapshot(data, stage / CAPTURE_BINDING)
        stage.rename(target)
    return target / CAPTURE_BINDING


def _reference_directory(root, reference):
    paths = []
    if relative := reference["relative_directory"]:
        paths.append(root / relative)
    paths.append(Path(reference["directory"]))
    for path in dict.fromkeys(path.resolve() for path in paths):
        if not path.is_dir():
            continue
        try:
            with ExperimentReader(path) as reader:
                if reader.experiment["experiment_id"] == reference["experiment_id"]:
                    return path
        except (CaptureError, OSError, sqlite3.Error):
            continue
    raise CaptureError("CONTINUATION_REFERENCE_SOURCE", "原参考图所在实验目录已变化。可回到 Capture 重新选择目录。")


def _binding_location(path, prefix):
    binding = Path(path).expanduser().resolve()
    directory = binding.parent
    root = directory.parent.parent
    if binding.name != CAPTURE_BINDING or directory.parent != root / "analysis-inputs" or not directory.name.startswith(prefix):
        raise CaptureError("CAPTURE_BINDING_INVALID", "请选择本实验保存在 analysis-inputs 中的采集绑定。")
    return binding, root


def read_continuation_binding(path):
    binding, root = _binding_location(path, BINDING_PREFIX)
    directory = binding.parent
    try:
        encoded = binding.read_bytes()
        data = json.loads(encoded)
        if data["schema"] != BINDING_SCHEMA:
            raise ValueError("invalid continuation analysis schema")
        source = _reference_directory(root, data["reference"])
        snapshot = continuation_snapshot(root, source, data["camera_id"], data["reference"]["sequence"],
                                         image_part=data["image_part"])
        if data["experiment_id"] != snapshot["experiment_id"]:
            raise ValueError("capture binding identifies another experiment")
        for receipt, row in zip(data["frames"], snapshot["frames"], strict=True):
            if ((receipt["frame_id"], receipt["raw_sha256"], receipt.get("experiment_id")) !=
                    (row["frame_id"], row["sha256"], row.get("source_experiment_id"))):
                raise ValueError("capture binding differs from committed frame identity")
            derived = row["color_model"] in ("bayer", "rgba", "bgra")
            if derived:
                name = f"{row['analysis_frame']:012d}.tiff"
                actual = (directory / name).resolve()
                if receipt["derived_name"] != name or actual.parent != directory:
                    raise ValueError("invalid derived frame path")
                if sha256_file(actual) != receipt["input_sha256"]:
                    raise CaptureError("CAPTURE_IMAGE_CHANGED", name)
                row.update(analysis_path=str(actual), analysis_sha256=receipt["input_sha256"])
            elif "derived_name" in receipt or receipt["input_sha256"] != row["sha256"]:
                raise ValueError("invalid raw frame binding")
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise CaptureError("CAPTURE_BINDING_INVALID", str(exc)) from exc
    return snapshot, sha256(encoded).hexdigest()


def prepare_live_reference(preview):
    """Publish one reference identity independently of the growing test series."""
    source = preview["external_reference"]
    snapshot = continuation_snapshot(preview["experiment_root"], source["directory"], preview["camera_id"],
                                     source["sequence"], image_part=preview["image_part"], live=True)
    identities = _identities(preview)
    if (snapshot["experiment_id"] != preview["experiment_id"] or snapshot["external_reference"] != source
            or _identities(snapshot)[:len(identities)] != identities):
        raise CaptureError("CONTINUATION_REFERENCE_CHANGED", "实验或原图记录已变化，请重新预览后选择。")
    data = {
        "schema": LIVE_REFERENCE_SCHEMA, "experiment_id": snapshot["experiment_id"],
        "camera_id": snapshot["camera_id"], "image_part": snapshot["image_part"], "source": source,
        "frame_id": snapshot["reference_frame_id"], "sha256": snapshot["frames"][0]["sha256"],
    }
    root = _analysis_inputs_folder(snapshot["experiment_root"])
    target = root / (LIVE_REFERENCE_PREFIX + uuid4().hex[:12])
    with TemporaryDirectory(prefix=".linking-", dir=root) as work:
        stage = Path(work) / "input"
        stage.mkdir()
        write_snapshot(data, stage / CAPTURE_BINDING)
        stage.rename(target)
    return target / CAPTURE_BINDING


def live_reference_root(path):
    return _binding_location(path, LIVE_REFERENCE_PREFIX)[1]


def read_live_reference(path, *, expected_digest=None):
    binding, root = _binding_location(path, LIVE_REFERENCE_PREFIX)
    try:
        encoded = binding.read_bytes()
        digest = sha256(encoded).hexdigest()
        if expected_digest is not None and digest != expected_digest:
            raise CaptureError("CONTINUATION_REFERENCE_CHANGED", "实时参考图绑定已变化，请重新打开。")
        data = json.loads(encoded)
        if data["schema"] != LIVE_REFERENCE_SCHEMA:
            raise ValueError("invalid live reference binding schema")
        source = _reference_directory(root, data["source"])
        snapshot = continuation_snapshot(root, source, data["camera_id"], data["source"]["sequence"],
                                         image_part=data["image_part"], live=True)
        if (snapshot["experiment_id"], snapshot["reference_frame_id"], snapshot["frames"][0]["sha256"]) != (
            data["experiment_id"], data["frame_id"], data["sha256"],
        ):
            raise CaptureError("CONTINUATION_REFERENCE_CHANGED", "实验或参考图的台账标识已变化。")
    except (TypeError, KeyError, ValueError) as exc:
        raise CaptureError("CAPTURE_BINDING_INVALID", str(exc)) from exc
    return snapshot, digest


def offline_from_live_reference(path, *, expected_digest=None):
    preview, _ = read_live_reference(path, expected_digest=expected_digest)
    source = preview["external_reference"]
    snapshot = continuation_snapshot(preview["experiment_root"], source["directory"], preview["camera_id"],
                                     source["sequence"], image_part=preview["image_part"])
    if (snapshot["experiment_id"] != preview["experiment_id"] or _identities(snapshot)[:1] != _identities(preview)[:1]):
        raise CaptureError("CONTINUATION_REFERENCE_CHANGED", "所选实时参考图已变化，请重新选择。")
    return prepare_continuation_binding(snapshot)
