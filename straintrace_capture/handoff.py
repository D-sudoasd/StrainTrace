"""An Analysis input is a snapshot of committed ledger identities."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from subprocess import Popen

from .config import CaptureError
from .storage import ExperimentReader, sha256_file


def select_camera(reader: ExperimentReader, camera_id: str | None) -> str:
    cameras = [entry["device"]["id"] for entry in reader.cameras()]
    if camera_id is None and len(cameras) == 1:
        return cameras[0]
    if camera_id not in cameras:
        raise CaptureError("CHOOSE_CAMERA", f"select one registered camera: {', '.join(cameras) or '(none yet)'}")
    return camera_id


def analysis_snapshot(root: str | Path, camera_id: str | None = None, reference_sequence: int | None = None) -> dict:
    """Resolve exactly one camera and reference, excluding calibration files."""
    with ExperimentReader(root) as reader:
        camera_id = select_camera(reader, camera_id)
        experiment = reader.experiment
        camera = next(item for item in reader.cameras() if item["device"]["id"] == camera_id)
        rows = reader.frames(camera_id=camera_id)
        candidates = reader.reference_frames(camera_id, sequence=reference_sequence)
        if len(candidates) > 1:
            raise CaptureError("CHOOSE_REFERENCE", "multiple reference images exist; select a reference sequence")
        if not candidates:
            raise CaptureError("REFERENCE_UNAVAILABLE", f"no suitable committed reference image for {camera_id}")
        reference = candidates[0]
        selection = ("explicit_sequence" if reference_sequence is not None else
                     "capture_reference_role" if reference["role"] == "reference" else "first_committed_test_frame")
        chosen = [reference] + [r for r in rows if r["role"] == "test" and r["sequence"] > reference["sequence"]]
        inputs = [{**row, "absolute_path": str(reader.image_path(row)), "analysis_frame": number}
                  for number, row in enumerate(chosen, 1)]
        for frame in inputs:
            if not Path(frame["absolute_path"]).is_file():
                raise CaptureError("FRAME_UNAVAILABLE", f"committed frame is missing: {frame['frame_id']}")
        return {
            "schema": "straintrace.capture-analysis-input/v1",
            "experiment_id": experiment["experiment_id"], "experiment_root": str(reader.root),
            "experiment_state": experiment["state"], "camera_id": camera_id,
            "reference_selection": selection, "reference_frame_id": reference["frame_id"],
            "metadata": experiment["config"]["metadata"], "calibrations": experiment.get("calibrations", []),
            "camera_features": {key: item["value"] for key, item in camera["features"].items()},
            "frames": inputs,
        }


def analysis_settings(snapshot: dict, template: dict) -> dict:
    """Bind a stock ezDIC config to the committed frame order."""
    if len(snapshot["frames"]) < 2:
        raise CaptureError("TOO_FEW_FRAMES", "Analysis needs a reference and at least one test image")
    data = dict(template)
    data.pop("image_folder", None)
    data["image_paths"] = analysis_paths(snapshot)
    data.update(start_frame_1based=1, reference_frame_1based=1, end_frame_1based=len(snapshot["frames"]))
    if "field_roi_reference_frame_1based" in data:
        data["field_roi_reference_frame_1based"] = 1
    return data


def analysis_paths(snapshot: dict) -> list[str]:
    if any(frame["color_model"] == "bayer" and not frame.get("analysis_path") for frame in snapshot["frames"]):
        raise CaptureError("BAYER_DERIVATION_REQUIRED", "prepare Bayer luminance in a derived folder before offline analysis")
    if any(frame["color_model"] in ("rgba", "bgra") and not frame.get("analysis_path") for frame in snapshot["frames"]):
        raise CaptureError("ALPHA_DERIVATION_REQUIRED", "prepare luminance from the camera colors without alpha before offline analysis")
    return [row.get("analysis_path", row["absolute_path"]) for row in snapshot["frames"]]


def derive_analysis_inputs(snapshot: dict, output: str | Path, *, debayer: bool = False, drop_alpha: bool = False,
                           lens_profile: str | Path | None = None) -> dict:
    """Save requested image transforms in a separate, source-hashed TIFF layer."""
    import cv2

    from .calibration import parse_profile, undistort
    from .image_inputs import bayer_luminance, decode_samples

    has_bayer = any(frame["color_model"] == "bayer" for frame in snapshot["frames"])
    has_alpha = any(frame["color_model"] in ("rgba", "bgra") for frame in snapshot["frames"])
    if not (lens_profile or (debayer and has_bayer) or (drop_alpha and has_alpha)):
        return snapshot
    model = profile_hash = None
    if lens_profile:
        from hashlib import sha256

        content = Path(lens_profile).read_bytes()
        profile_hash = sha256(content).hexdigest()
        profile = parse_profile(content)
        if has_bayer and not debayer:
            raise CaptureError("BAYER_DERIVATION_REQUIRED", "apply Bayer luminance conversion before optical correction")
        if has_alpha and not drop_alpha:
            raise CaptureError("ALPHA_DERIVATION_REQUIRED", "request luminance without alpha before optical correction")
        model = profile["cameras"].get(snapshot["camera_id"])
        if not model:
            raise CaptureError("CALIBRATION_CAMERA_UNBOUND", "the lens profile does not identify this camera")
        for key, expected in model.get("sensor_geometry", {}).items():
            if expected is not None and snapshot["camera_features"].get(key) != expected:
                raise CaptureError("CALIBRATION_GEOMETRY_MISMATCH", f"lens profile expects {key}={expected}")
    root = Path(output).resolve()
    raw_root = Path(snapshot["experiment_root"]) / "images"
    if root.is_relative_to(raw_root):
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
        path = root / f"{row['sequence']:012d}.tiff"
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


def launch_analysis(root: str | Path, *, camera_id: str | None = None, live: bool = False,
                    reference_sequence: int | None = None) -> Popen:
    command = _workbench_command("ezDIC.exe", "dic_virtual_extensometer_gui_v7_multi_roi_range")
    command += ["--capture", str(Path(root).resolve())]
    if camera_id is not None:
        command += ["--camera", camera_id]
    if reference_sequence is not None:
        command += ["--reference-sequence", str(reference_sequence)]
    if live:
        command += ["--live"]
    return Popen(command)


def launch_capture(root: str | Path | None = None) -> Popen:
    command = _workbench_command("StrainTrace-Capture.exe", "straintrace_capture.gui")
    if root is not None:
        command += ["--experiment", str(Path(root).resolve())]
    return Popen(command)
