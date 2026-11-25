"""An Analysis input is a snapshot of committed ledger identities."""

from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .config import CaptureError
from .event_images import event_scope, event_sequence, select_event_reference
from .image_inputs import frame_part_selector, require_part_geometry, select_frame_image, validate_analysis_part
from .storage import ExperimentReader, sha256_file


CAPTURE_BINDING = "capture_binding.json"
LENS_PROFILE = "lens_profile.json"
ANALYSIS_SCHEMA = "straintrace.capture-analysis-input/v1"


def select_camera(reader: ExperimentReader, camera_id: str | None) -> str:
    cameras = [entry["device"]["id"] for entry in reader.cameras()]
    if camera_id is None and len(cameras) == 1:
        return cameras[0]
    if camera_id not in cameras:
        raise CaptureError("CHOOSE_CAMERA", f"select one registered camera: {', '.join(cameras) or '(none yet)'}")
    return camera_id


def select_reference(reader: ExperimentReader, camera_id: str, sequence: int | None) -> dict:
    candidates = reader.reference_frames(camera_id, sequence=sequence)
    if len(candidates) > 1:
        raise CaptureError("CHOOSE_REFERENCE", "multiple reference images exist; select a reference sequence")
    if not candidates:
        raise CaptureError("REFERENCE_UNAVAILABLE", f"no suitable committed reference image for {camera_id}")
    return candidates[0]


def analysis_snapshot(root: str | Path, camera_id: str | None = None, reference_sequence: int | None = None,
                      *, image_part: str | None = None, event_context_id: str | None = None,
                      expected_id: str | None = None) -> dict:
    """Resolve one camera, image part and reference from committed receipts."""
    from .calibration import calibration_geometry, phase_sensor_geometry

    validate_analysis_part(image_part)
    with ExperimentReader(root) as reader:
        experiment = reader.experiment
        if expected_id is not None and experiment["experiment_id"] != expected_id:
            raise CaptureError("REVIEW_SCOPE_CHANGED", "实验标识已变化，请重新选择采集记录。")
        camera_id = select_camera(reader, camera_id)
        camera = next(item for item in reader.cameras() if item["device"]["id"] == camera_id)
        sensor = phase_sensor_geometry(reader, camera_id)
        event = None
        if event_context_id is not None:
            event = event_scope(reader, experiment, event_context_id)
            reference, selection = select_event_reference(reader, camera_id, event, reference_sequence)
            chosen = [reference, *event_sequence(reader, camera_id, event, after=reference["sequence"])]
            if len(chosen) < 2:
                raise CaptureError("TOO_FEW_FRAMES", "所选参考图后没有可分析的同事件图像。")
        else:
            reference = select_reference(reader, camera_id, reference_sequence)
            selection = ("explicit_sequence" if reference_sequence is not None else
                         "capture_reference_role" if reference["role"] == "reference" else "first_committed_test_frame")
            rows = reader.frames(camera_id=camera_id)
            chosen = [reference] + [r for r in rows if r["role"] == "test" and r["sequence"] > reference["sequence"]]
        if image_part is not None:
            chosen = [select_frame_image(row, image_part) for row in chosen]
        if image_part is not None or event is not None:
            for row in chosen[1:]:
                require_part_geometry(chosen[0], row)
        if event is not None:
            geometry = {}
            for row in chosen:
                phase = str(row["phase"])
                if phase in geometry:
                    continue
                layout = calibration_geometry(row["shape"], sensor.get(phase, {}), row["backend_metadata"])
                geometry[phase] = layout["sensor_geometry"]
            reference_geometry = geometry[str(reference["phase"])]
            event_geometry = geometry[str(event["phase"])]
            if any(reference_geometry[key] != event_geometry[key] for key in reference_geometry.keys() & event_geometry.keys()):
                raise CaptureError("SEQUENCE_SENSOR_CHANGED", "参考图和本次事件的相机采样区域或配置不同。")
        inputs = [{**row, "absolute_path": str(reader.image_path(row)), "analysis_frame": number}
                  for number, row in enumerate(chosen, 1)]
        for frame in inputs:
            if not Path(frame["absolute_path"]).is_file():
                raise CaptureError("FRAME_UNAVAILABLE", f"committed frame is missing: {frame['frame_id']}")
        return {
            "schema": ANALYSIS_SCHEMA,
            "experiment_id": experiment["experiment_id"], "experiment_root": str(reader.root),
            "experiment_state": experiment["state"], "camera_id": camera_id,
            **({"image_part": image_part} if image_part is not None else {}),
            **({"event": event} if event is not None else {}),
            "reference_selection": selection, "reference_frame_id": reference["frame_id"],
            "metadata": experiment["config"]["metadata"], "calibrations": experiment.get("calibrations", []),
            "camera_features": {key: item["value"] for key, item in camera["features"].items()},
            "phase_sensor_geometry": sensor,
            "frames": inputs,
        }


def analysis_settings(snapshot: dict, template: dict) -> dict:
    """Bind a stock ezDIC config to the committed frame order."""
    if len(snapshot["frames"]) < 2:
        raise CaptureError("TOO_FEW_FRAMES", "Analysis needs a reference and at least one test image")
    data = dict(template)
    data.pop("image_folder", None)
    data["image_paths"] = analysis_paths(snapshot)
    data["capture_source"] = analysis_source(snapshot)
    data.update(start_frame_1based=1, reference_frame_1based=1, end_frame_1based=len(snapshot["frames"]))
    if "field_roi_reference_frame_1based" in data:
        data["field_roi_reference_frame_1based"] = 1
    return data


def analysis_source(snapshot: dict) -> dict:
    parts = {frame_part_selector(row) for row in snapshot["frames"]}
    event = snapshot.get("event")
    return {"experiment_id": snapshot["experiment_id"], "camera_id": snapshot["camera_id"],
            "image_part": next(iter(parts)) if len(parts) == 1 else None,
            **({"event": event} if event is not None else {}),
            "frames": [{"frame_id": row["frame_id"], "raw_sha256": row["sha256"],
                        "input_sha256": row.get("analysis_sha256", row["sha256"]),
                        **({"event_position": row["event_position"]}
                           if event is not None and row.get("event_context_id") == event["signal_id"] else {}),
                        **({"experiment_id": row["source_experiment_id"]} if "source_experiment_id" in row else {})}
                       for row in snapshot["frames"]]}


def analysis_paths(snapshot: dict) -> list[str]:
    if any(frame["color_model"] == "bayer" and not frame.get("analysis_path") for frame in snapshot["frames"]):
        raise CaptureError("BAYER_DERIVATION_REQUIRED", "prepare Bayer luminance in a derived folder before offline analysis")
    if any(frame["color_model"] in ("rgba", "bgra") and not frame.get("analysis_path") for frame in snapshot["frames"]):
        raise CaptureError("ALPHA_DERIVATION_REQUIRED", "prepare luminance from the camera colors without alpha before offline analysis")
    return [row.get("analysis_path", row["absolute_path"]) for row in snapshot["frames"]]


def _lens_model(snapshot, content):
    from .calibration import calibration_geometry, parse_profile, require_matching_calibration

    model = parse_profile(content)["cameras"].get(snapshot["camera_id"])
    if not model:
        raise CaptureError("CALIBRATION_CAMERA_UNBOUND", "the lens profile does not identify this camera")
    phase_geometry = snapshot.get("phase_sensor_geometry", {})
    for row in snapshot["frames"]:
        features = row.get("analysis_sensor_geometry", phase_geometry.get(str(row["phase"]), snapshot["camera_features"]))
        try:
            geometry = calibration_geometry(row["shape"], features, row.get("backend_metadata") or {})
            require_matching_calibration(model, geometry)
        except CaptureError as exc:
            raise CaptureError(exc.code, f"{row['frame_id']}: {exc}") from exc
    return model


def derive_analysis_inputs(snapshot: dict, output: str | Path, *, debayer: bool = False, drop_alpha: bool = False,
                           lens_profile: str | Path | None = None) -> dict:
    """Save requested image transforms in a separate, source-hashed TIFF layer."""
    model = digest = None
    if lens_profile:
        content = Path(lens_profile).read_bytes()
        model, digest = _lens_model(snapshot, content), sha256(content).hexdigest()
    return _derive(snapshot, output, debayer=debayer, drop_alpha=drop_alpha, model=model, profile_hash=digest)


def _derive(snapshot, output, *, debayer, drop_alpha, model, profile_hash):
    import cv2

    from .calibration import undistort
    from .image_inputs import bayer_luminance, decode_samples

    has_bayer = any(frame["color_model"] == "bayer" for frame in snapshot["frames"])
    has_alpha = any(frame["color_model"] in ("rgba", "bgra") for frame in snapshot["frames"])
    if not (model or (debayer and has_bayer) or (drop_alpha and has_alpha)):
        return snapshot
    if model and has_bayer and not debayer:
        raise CaptureError("BAYER_DERIVATION_REQUIRED", "apply Bayer luminance conversion before optical correction")
    if model and has_alpha and not drop_alpha:
        raise CaptureError("ALPHA_DERIVATION_REQUIRED", "request luminance without alpha before optical correction")
    root = Path(output).resolve()
    raw_roots = [Path(snapshot["experiment_root"]) / "images"]
    if reference := snapshot.get("external_reference"):
        raw_roots.append(Path(reference["directory"]) / "images")
    if any(root.is_relative_to(raw) for raw in raw_roots):
        raise CaptureError("UNSAFE_ANALYSIS_OUTPUT", "choose a derived folder outside the original images tree")
    root.mkdir(parents=True, exist_ok=False)
    converted = []
    for row in snapshot["frames"]:
        color = row["color_model"]
        is_bayer = color == "bayer" and debayer
        is_alpha = color in ("rgba", "bgra") and drop_alpha
        if not (is_bayer or is_alpha or model):
            converted.append(row)
            continue
        source = Path(row["absolute_path"])
        if sha256_file(source) != row["sha256"]:
            raise CaptureError("CAPTURE_IMAGE_CHANGED", f"source checksum mismatch for {row['frame_id']}")
        transforms = []
        if is_bayer:
            pixels = bayer_luminance(source, row["pixel_format"])
            transforms.append({"method": "opencv_bayer_to_luminance", "pixel_format": row["pixel_format"]})
        else:
            pixels = decode_samples(source, color_model=color)
        if is_alpha:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2GRAY)
            transforms.append({"method": "opencv_bgra_to_luminance", "alpha_policy": "exclude_unassociated_alpha"})
        if model:
            pixels = undistort(pixels, model)
            transforms.append({"method": "opencv_pinhole_undistort", "calibration_sha256": profile_hash})
        index = row["analysis_frame"] if snapshot.get("external_reference") else row["sequence"]
        path = root / f"{index:012d}.tiff"
        valid, buffer = cv2.imencode(".tiff", pixels, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        if not valid:
            raise CaptureError("DERIVED_TIFF_FAILED", f"cannot encode luminance for {row['frame_id']}")
        with path.open("xb") as stream:
            stream.write(buffer)
        converted.append({
            **row, "analysis_path": str(path), "analysis_sha256": sha256_file(path),
            "analysis_transforms": transforms, "opencv_version": cv2.__version__,
        })
    return {**snapshot, "frames": converted}


def derive_bayer_inputs(snapshot: dict, output: str | Path) -> dict:
    return derive_analysis_inputs(snapshot, output, debayer=True)


def corrected_reference_choices(root, camera_id, *, image_part=None):
    """List the saved reference frames that have an experimental frame after them."""
    with ExperimentReader(root) as reader:
        rows = reader.frames(camera_id=camera_id)
        references = [row for row in rows if row["role"] == "reference"]
        if not references:
            references = reader.reference_frames(camera_id)
        last_test = next((row["sequence"] for row in reversed(rows) if row["role"] == "test"), 0)
        options = [row for row in references if row["sequence"] < last_test]
    if not options:
        raise CaptureError("TOO_FEW_FRAMES", "校正后分析需要一张参考图和其后的至少一张实验图。")
    if image_part is not None:
        validate_analysis_part(image_part)
        return [select_frame_image(row, image_part) for row in options]
    return options


def _select_lens(snapshot, explicit, loaded_profiles):
    from .calibration import archived_profile_path, profile_bytes

    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        content = profile_bytes(path)
        return _lens_model(snapshot, content), content, {"source": "selected_file", "name": path.name}

    attachments = [record for record in snapshot["calibrations"] if snapshot["camera_id"] in record["camera_ids"]]
    first_problem = None
    candidates = [("experiment_archive", item) for item in reversed(attachments)]
    candidates.extend(("loaded_profile", path) for path in reversed(loaded_profiles))
    for source, candidate in candidates:
        try:
            if source == "experiment_archive":
                path = archived_profile_path(snapshot["experiment_root"], candidate)
                content = profile_bytes(path, expected_hash=candidate["sha256"])
            else:
                path = Path(candidate).expanduser().resolve()
                content = profile_bytes(path)
            return _lens_model(snapshot, content), content, {"source": source, "name": path.name}
        except (CaptureError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            if first_problem is None:
                first_problem = exc
    if first_problem is not None:
        raise first_problem
    raise CaptureError("CALIBRATION_UNAVAILABLE", "请归档、载入或选择这台相机的镜头标定文件。")


def prepare_corrected_analysis(root, camera_id, reference_sequence, *, lens_profile=None, loaded_profiles=(), image_part=None):
    """Publish the corrected TIFFs, exact lens profile and frame binding together."""
    snapshot = analysis_snapshot(root, camera_id, reference_sequence, image_part=image_part)
    if len(snapshot["frames"]) < 2:
        raise CaptureError("TOO_FEW_FRAMES", "校正后分析需要一张参考图和其后的至少一张实验图。")
    model, content, provenance = _select_lens(snapshot, lens_profile, loaded_profiles)
    digest = sha256(content).hexdigest()
    experiment = Path(snapshot["experiment_root"])
    parent = experiment / "analysis-inputs"
    if parent.resolve() != parent:
        raise CaptureError("UNSAFE_ANALYSIS_OUTPUT", "校正图像目录必须位于本实验的 analysis-inputs 文件夹内。")
    parent.mkdir(exist_ok=True)
    target = parent / f"corrected-{uuid4().hex[:12]}"
    with TemporaryDirectory(prefix=".correcting-", dir=parent) as work:
        stage = Path(work) / "inputs"
        binding = _derive(snapshot, stage, debayer=True, drop_alpha=True, model=model, profile_hash=digest)
        binding["lens_calibration"] = {"path": LENS_PROFILE, "sha256": digest, **provenance}
        for row in binding["frames"]:
            row["analysis_path"] = str(target / Path(row["analysis_path"]).name)
        (stage / LENS_PROFILE).write_bytes(content)
        write_snapshot(binding, stage / CAPTURE_BINDING)
        stage.rename(target)
    return target / CAPTURE_BINDING


def read_corrected_analysis(path):
    """Reopen the prepared series and validate the binding and corrected bytes."""
    from .calibration import phase_sensor_geometry, profile_bytes

    binding = Path(path).expanduser().resolve()
    directory = binding.parent
    root = directory.parent.parent
    if directory.parent != root / "analysis-inputs" or not directory.name.startswith("corrected-"):
        raise CaptureError("CAPTURE_BINDING_INVALID", "请打开实验 analysis-inputs 中校正序列目录里的采集绑定文件。")
    try:
        encoded = binding.read_bytes()
        snapshot = json.loads(encoded)
        image_part = snapshot.get("image_part")
        validate_analysis_part(image_part)
        lens = snapshot["lens_calibration"]
        frames = snapshot["frames"]
        if (snapshot["schema"] != ANALYSIS_SCHEMA or lens["path"] != LENS_PROFILE
                or len(frames) < 2 or snapshot["reference_frame_id"] != frames[0]["frame_id"]):
            raise ValueError("unexpected capture binding structure")
        lens_path = (directory / LENS_PROFILE).resolve()
        if lens_path.parent != directory:
            raise ValueError("lens profile path leaves the corrected dataset")
        content = profile_bytes(lens_path, expected_hash=lens["sha256"])
        with ExperimentReader(root) as reader:
            if snapshot["experiment_id"] != reader.experiment["experiment_id"]:
                raise ValueError("capture binding identifies a different experiment")
            recorded = reader.frames(camera_id=snapshot["camera_id"])
            start, stop = frames[0]["sequence"], frames[-1]["sequence"]
            reference, = reader.reference_frames(snapshot["camera_id"], sequence=start)
            selected = [reference, *(row for row in recorded if start < row["sequence"] <= stop and row["role"] == "test")]
            keys = ("frame_id", "camera_id", "sequence", "phase", "role", "path", "sha256", "shape", "dtype", "color_model",
                    "pixel_format", "significant_bits", "backend_metadata")
            for row, expected in zip(frames, selected, strict=True):
                expected = select_frame_image(expected, image_part)
                if any(row.get(key) != expected.get(key) for key in keys):
                    raise ValueError(f"capture binding differs from the ledger at frame {expected['frame_id']}")
                corrected = (directory / f"{expected['sequence']:012d}.tiff").resolve()
                if corrected.parent != directory:
                    raise ValueError("derived image path leaves the corrected dataset")
                if sha256_file(corrected) != row["analysis_sha256"]:
                    raise CaptureError("CORRECTED_IMAGE_CHANGED", f"checksum mismatch for {corrected.name}")
                if {"method": "opencv_pinhole_undistort", "calibration_sha256": lens["sha256"]} not in row["analysis_transforms"]:
                    raise ValueError(f"missing lens correction record for {corrected.name}")
                row.update(absolute_path=str(reader.image_path(expected)), analysis_path=str(corrected))
            snapshot["phase_sensor_geometry"] = phase_sensor_geometry(reader, snapshot["camera_id"])
            camera = next(camera for camera in reader.cameras() if camera["device"]["id"] == snapshot["camera_id"])
            snapshot["camera_features"] = {name: item["value"] for name, item in camera["features"].items()}
        _lens_model(snapshot, content)
    except (KeyError, ValueError, TypeError, IndexError, StopIteration) as exc:
        raise CaptureError("CAPTURE_BINDING_INVALID", str(exc)) from exc
    snapshot["experiment_root"] = str(root)
    return snapshot, sha256(encoded).hexdigest()


def write_snapshot(snapshot: dict, output: str | Path) -> Path:
    path = Path(output).expanduser().resolve()
    with path.open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return path


def _workbench_command(executable: str, module: str) -> list[str]:
    if getattr(sys, "frozen", False):
        folder = Path(sys.executable).resolve().parent
        return [str(folder / executable)]
    return [sys.executable, "-m", module]


def _launch_workbench(command: list[str]) -> subprocess.Popen:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    # An absolute source-script launch adds its directory to the parent's
    # import path only. The sibling process must also be able to resolve -m.
    directory = None if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    return subprocess.Popen(command, creationflags=flags, cwd=directory)


def launch_analysis(root: str | Path, *, camera_id: str | None = None, live: bool = False,
                    reference_sequence: int | None = None, image_part: str | None = None,
                    event_context_id: str | None = None, expected_id: str | None = None) -> subprocess.Popen:
    command = _workbench_command("ezDIC.exe", "dic_virtual_extensometer_gui_v7_multi_roi_range")
    command += ["--capture", str(Path(root).resolve())]
    if camera_id is not None:
        command += ["--camera", camera_id]
    if reference_sequence is not None:
        command += ["--reference-sequence", str(reference_sequence)]
    if image_part is not None:
        command += ["--image-part", image_part]
    if event_context_id is not None:
        command += ["--event-context", event_context_id]
    if expected_id is not None:
        command += ["--expect-experiment-id", expected_id]
    if live:
        command += ["--live"]
    return _launch_workbench(command)


def launch_corrected_analysis(binding: str | Path) -> subprocess.Popen:
    command = _workbench_command("ezDIC.exe", "dic_virtual_extensometer_gui_v7_multi_roi_range")
    return _launch_workbench([*command, "--capture-binding", str(Path(binding).resolve())])


def launch_live_reference(binding: str | Path) -> subprocess.Popen:
    command = _workbench_command("ezDIC.exe", "dic_virtual_extensometer_gui_v7_multi_roi_range")
    return _launch_workbench([*command, "--live-reference-binding", str(Path(binding).resolve())])


def launch_capture(root: str | Path | None = None, *, practice: bool = False) -> subprocess.Popen:
    command = _workbench_command("StrainTrace-Capture.exe", "straintrace_capture.gui")
    if practice:
        if root is not None:
            raise ValueError("a new simulated exercise cannot open an existing experiment on startup")
        command.append("--practice")
    elif root is not None:
        command += ["--experiment", str(Path(root).resolve())]
    return _launch_workbench(command)
