import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, CaptureSession, ExperimentReader
from straintrace_capture.calibration import CALIBRATION_SCHEMA, undistort
from straintrace_capture.cameras._gentl_multipart import IMAGE_PART
from straintrace_capture.cameras.base import Device, Image, receipt_times
from straintrace_capture.handoff import (
    analysis_paths, analysis_settings, analysis_snapshot, corrected_reference_choices, prepare_corrected_analysis,
    read_corrected_analysis, write_snapshot,
)
from straintrace_capture.image_inputs import decode_samples
from straintrace_capture.storage import ExperimentWriter

from test_capture_lifecycle import config


PART_CAMERA = "gentl:part-analysis"
PRIMARY_PART = "2/16/0/Mono12"
ANALYSIS_PART = "6/0/0/Mono12"


def multipart_analysis_experiment(root, *, change=None):
    settings = CaptureConfig.from_dict({"config_version": 1, "name": "Multipart DIC fixture",
                                       "cameras": [{"id": PART_CAMERA}], "phases": [
                                           {"mode": "single", "role": "reference"}, {"mode": "single"}]})
    rng = np.random.default_rng(15941)
    primary, reference = [cv2.GaussianBlur(rng.integers(120, 3850, (144, 176), dtype=np.uint16), (5, 5), .85)
                          for _ in range(2)]
    deformed = cv2.warpAffine(reference, np.float32([[1, 0, 4], [0, 1, -2]]),
                             (176, 144), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    writer = ExperimentWriter(root, settings)
    try:
        writer.register_camera(Device(PART_CAMERA, "gentl", "Fixture", "Multipart DIC", "DIC", "fixture"), {})
        prepared = []
        for sequence, pixels in enumerate((reference, deformed), 1):
            stamp, monotonic = receipt_times()
            main = Image(primary, "Mono12", 12, "mono", stamp, monotonic, sequence, sequence,
                         metadata={IMAGE_PART: {"selector": PRIMARY_PART}})
            aux = replace(main, pixels=pixels[:, :-1].copy() if sequence == 2 and change == "geometry" else pixels,
                          metadata={IMAGE_PART: {"selector": ANALYSIS_PART, "part_index": 2 - sequence}})
            images = () if sequence == 2 and change == "missing" else (aux,)
            prepared.append(writer.prepare_frame(PART_CAMERA, sequence, sequence, "reference" if sequence == 1 else "test",
                                                 replace(main, auxiliary_images=images), None, None, None, 0))
        writer.commit_frames(prepared)
        writer.set_state("completed")
    finally:
        writer.close()
    return primary, reference, deformed


def test_live_and_offline_analysis_use_exact_same_saved_part(tmp_path):
    from straintrace_capture.live import LiveAnalysis

    root = tmp_path / "experiment"
    primary, reference, deformed = multipart_analysis_experiment(root)
    ordinary = analysis_snapshot(root)
    binding = analysis_snapshot(root, image_part=ANALYSIS_PART)
    assert binding["image_part"] == ANALYSIS_PART
    assert [row["frame_id"] for row in binding["frames"]] == [row["frame_id"] for row in ordinary["frames"]]
    assert [row["timestamps"] for row in binding["frames"]] == [row["timestamps"] for row in ordinary["frames"]]
    assert all("-part-02.tiff" in name for name in analysis_paths(binding))
    for path, samples in zip(analysis_paths(binding), (reference, deformed), strict=True):
        np.testing.assert_array_equal(decode_samples(Path(path)), samples, strict=True)
    np.testing.assert_array_equal(decode_samples(Path(analysis_paths(ordinary)[1])), primary, strict=True)
    with LiveAnalysis(root, image_part=ANALYSIS_PART, field_roi=(32, 28, 96, 84), field_step=14, field_subset=21) as live:
        assert live.poll()["frame_id"] == binding["reference_frame_id"]
        row = live.poll()
        assert row["image_part"] == ANALYSIS_PART and row["image_sha256"] == binding["frames"][1]["sha256"]
        assert row["image_path"] == binding["frames"][1]["path"]
        assert row["field_quality"]["scientific_ok"]
        assert np.nanmedian(live.field["u"]) == pytest.approx(4, abs=.03)
        assert np.nanmedian(live.field["v"]) == pytest.approx(-2, abs=.03)

    template = {"schema_version": 1, "analysis_mode": "fullfield", "output_dir": str(tmp_path / "results"),
                "field_roi": [32, 28, 96, 84], "solver": {"subset_size_px": 21, "step_px": 14}}
    source = write_snapshot(template, tmp_path / "template.json")
    command = [sys.executable, "-m", "straintrace_capture", "prepare-analysis", str(root), "--image-part", ANALYSIS_PART,
               "--template", str(source), "--run-config", str(tmp_path / "analysis.json"), "--output", str(tmp_path / "binding.json")]
    gathered = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert gathered.returncode == 0, gathered.stdout + gathered.stderr
    run = [sys.executable, "-m", "ezdic_cli", "run", "--config", str(tmp_path / "analysis.json")]
    report = subprocess.run(run, capture_output=True, text=True, timeout=45)
    assert report.returncode == 0, report.stdout + report.stderr
    manifest = json.loads((tmp_path / "results/run_manifest.json").read_text(encoding="utf-8"))
    assert [row["sha256"] for row in manifest["inputs"]] == [row["sha256"] for row in binding["frames"]]
    assert manifest["config"]["capture_source"]["image_part"] == ANALYSIS_PART
    target = Path(analysis_paths(binding)[1])
    contents = target.read_bytes()
    target.write_bytes(contents[:-1] + bytes([contents[-1] ^ 1]))
    denied = subprocess.run(run, capture_output=True, timeout=45)
    assert denied.returncode != 0
    assert json.loads(denied.stderr)["error_code"] == "CAPTURE_IMAGE_CHANGED"
    with LiveAnalysis(root, image_part=ANALYSIS_PART, field_roi=(32, 28, 96, 84), field_step=14, field_subset=21) as live:
        assert live.poll()["sequence"] == 1
        with pytest.raises(CaptureError) as err:
            live.poll()
        assert err.value.code == "CAPTURE_IMAGE_CHANGED"


@pytest.mark.parametrize(("change", "code"), [("missing", "ANALYSIS_PART_MISSING"), ("geometry", "FRAME_GEOMETRY_CHANGED")])
def test_analysis_never_falls_back_to_another_saved_part(tmp_path, change, code):
    from straintrace_capture.live import LiveAnalysis

    root = tmp_path / "experiment"
    multipart_analysis_experiment(root, change=change)
    with pytest.raises(CaptureError) as error:
        analysis_snapshot(root, image_part=ANALYSIS_PART)
    assert error.value.code == code
    assert len(analysis_snapshot(root)["frames"]) == 2
    with LiveAnalysis(root, image_part=ANALYSIS_PART, field_roi=(32, 28, 96, 84), field_step=14, field_subset=21) as live:
        assert live.poll()["sequence"] == 1
        with pytest.raises(CaptureError) as error:
            live.poll()
        assert error.value.code == code


def test_corrected_part_requires_matching_lens_and_reopens_exact_channel(tmp_path):
    root = tmp_path / "experiment"
    _, reference, _ = multipart_analysis_experiment(root)
    basic = {"schema": CALIBRATION_SCHEMA, "capture_experiment_id": "lens-part-test"}
    model = {"image_size_px": [176, 144], "camera_matrix": [[201, 0, 87.5], [0, 198, 71.5], [0, 0, 1]],
             "distortion_coefficients": [.10, 0, 0, 0, 0], "sensor_geometry": {}}
    wrong = write_snapshot({**basic, "cameras": {PART_CAMERA: {**model, IMAGE_PART: PRIMARY_PART}}}, tmp_path / "wrong.json")
    match = {**model, IMAGE_PART: ANALYSIS_PART}
    right = write_snapshot({**basic, "cameras": {PART_CAMERA: match}}, tmp_path / "right.json")
    with pytest.raises(CaptureError) as error:
        prepare_corrected_analysis(root, PART_CAMERA, 1, lens_profile=wrong, image_part=ANALYSIS_PART)
    assert error.value.code == "CALIBRATION_PART_MISMATCH"
    assert not (root / "analysis-inputs").exists()
    corrected = prepare_corrected_analysis(root, PART_CAMERA, 1, lens_profile=right, image_part=ANALYSIS_PART)
    data, _ = read_corrected_analysis(corrected)
    assert data["image_part"] == ANALYSIS_PART
    assert all(row["absolute_path"].endswith("-part-02.tiff") for row in data["frames"])
    np.testing.assert_array_equal(decode_samples(Path(analysis_paths(data)[0])), undistort(reference, match), strict=True)
    data["image_part"] = PRIMARY_PART
    corrected.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CaptureError) as error:
        read_corrected_analysis(corrected)
    assert error.value.code == "CAPTURE_BINDING_INVALID"


@pytest.mark.parametrize("mode", ["extensometer", "fullfield"])
def test_committed_capture_files_feed_stock_analysis_cli(tmp_path, mode):
    capture = CaptureSession(config([
        {"mode": "single", "role": "calibration"},
        {"mode": "single", "role": "reference"},
        {"mode": "timed", "fps": 8, "frame_count": 2},
    ], **{"sim:0": {"pixel_format": "Mono12", "motion_px": 0.02}}), tmp_path / "raw")
    capture.start()
    assert capture.wait(6)["state"] == "completed"
    binding = analysis_snapshot(capture.output, "sim:0")
    assert [frame["sequence"] for frame in binding["frames"]] == [2, 3, 4]
    template = {
        "schema_version": 1, "analysis_mode": mode,
        "start_frame_1based": 1, "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "results"),
        "export": {"write_manifest": True, "write_qc": True, "write_full_csv": True},
        "transaction": {"enabled": True, "archive_previous": True, "retain_failed_staging": True},
    }
    if mode == "extensometer":
        template["roi_groups"] = [{
            "name": "G01", "roi1": [41, 20, 27, 27], "roi2": [41, 73, 27, 27],
            "strain_mode": "y", "role": "none",
        }]
    else:
        template["field_roi"] = [24, 24, 80, 72]
        template["solver"] = {"name": "IC-GN", "subset_size_px": 19, "step_px": 12, "strain_window_px": 5,
                              "search_radius_px": 9}
    bound = analysis_settings(binding, template)
    path = write_snapshot(bound, tmp_path / "analysis.json")
    completed = subprocess.run([sys.executable, "-m", "ezdic_cli", "run", "--config", str(path)],
                               text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["scientific_ok"] and summary["manifest_verified"]
    manifest = json.loads((tmp_path / "results/run_manifest.json").read_text(encoding="utf-8"))
    assert [row["sha256"] for row in manifest["inputs"]] == [row["sha256"] for row in binding["frames"]]


def test_live_fullfield_carries_quality_and_frame_identity(tmp_path):
    from straintrace_capture.live import LiveAnalysis

    capture = CaptureSession(config([
        {"mode": "single", "role": "reference"}, {"mode": "single"},
    ], **{"sim:0": {"motion_px": 0., "strain_per_frame": 0.}}), tmp_path / "static")
    capture.start()
    assert capture.wait(6)["state"] == "completed"
    with LiveAnalysis(capture.output, field_roi=(24, 24, 80, 72), field_step=12, field_subset=19) as live:
        assert live.poll()["sequence"] == 1
        row = live.poll()
        assert row["provisional"] and row["field_quality"]["scientific_ok"]
        assert row["sequence"] == 2 and row["frame_id"].endswith("/sim:0/2")
        assert row["field_quality"]["correlation_valid_fraction"] == 1.
        assert live.poll() is None


def test_corrected_desktop_binding_round_trip_and_analysis_provenance(tmp_path):
    import ezdic_core as core

    camera = "sim:0"
    model = {
        "image_size_px": [160, 128], "camera_matrix": [[170, 0, 79.5], [0, 165, 63.5], [0, 0, 1]],
        "distortion_coefficients": [.12, 0, 0, 0, 0], "sensor_geometry": {},
    }
    profile = {"schema": CALIBRATION_SCHEMA, "capture_experiment_id": "simulated-lens", "cameras": {camera: model}}
    valid = write_snapshot(profile, tmp_path / "matching.json")
    incompatible = write_snapshot({**profile, "cameras": {camera: {**model, "image_size_px": [320, 240]}}},
                                  tmp_path / "incompatible.json")
    capture = CaptureSession(CaptureConfig.from_dict({
        "config_version": 1, "name": "Corrected handoff fixture",
        "cameras": [{"id": camera, "settings": {"width": 160, "height": 128, "motion_px": 0., "strain_per_frame": 0.}}],
        "phases": [{"mode": "single", "role": "reference"}, {"mode": "single", "role": "reference"},
                   {"mode": "continuous", "frame_count": 2}, {"mode": "single", "role": "reference"}],
        "calibration_files": [str(valid), str(incompatible)],
    }), tmp_path / "experiment")
    capture.start()
    assert capture.wait(6)["state"] == "completed"
    assert [row["sequence"] for row in corrected_reference_choices(capture.output, camera)] == [1, 2]
    with pytest.raises(CaptureError) as wrong:
        prepare_corrected_analysis(capture.output, camera, 2, lens_profile=incompatible)
    assert wrong.value.code == "CALIBRATION_GEOMETRY_MISMATCH"
    assert not (capture.output / "analysis-inputs").exists()

    path = prepare_corrected_analysis(capture.output, camera, 2)
    assert (path.parent / "lens_profile.json").read_bytes() == valid.read_bytes()
    binding, digest = read_corrected_analysis(path)
    assert [row["sequence"] for row in binding["frames"]] == [2, 3, 4]
    assert binding["lens_calibration"]["source"] == "experiment_archive"
    for row in binding["frames"]:
        raw = decode_samples(Path(row["absolute_path"]))
        np.testing.assert_array_equal(decode_samples(Path(row["analysis_path"])), undistort(raw, model), strict=True)

    # A copied experiment rebases the ledger and all derived paths.
    relocated = tmp_path / "relocated"
    shutil.copytree(capture.output, relocated)
    relocated_binding = relocated / path.relative_to(capture.output)
    rebased, rebased_digest = read_corrected_analysis(relocated_binding)
    assert rebased_digest == digest and all(Path(p).is_relative_to(relocated) for p in analysis_paths(rebased))

    settings = {
        "image_paths": analysis_paths(binding), "output_dir": tmp_path / "results",
        "roi_groups": [{"name": "G1", "roi1": [42, 23, 26, 26], "roi2": [42, 74, 26, 26],
                        "strain_mode": "y", "role": "none"}],
        "_corrected_input_hashes": [row["analysis_sha256"] for row in binding["frames"]],
        "capture_binding": {
            "path": str(path), "sha256": digest, "experiment_id": binding["experiment_id"], "camera_id": camera,
            "reference_frame_id": binding["reference_frame_id"], "calibration_path": str(path.parent / "lens_profile.json"),
            "calibration_sha256": binding["lens_calibration"]["sha256"],
        },
    }
    result = core.run_extensometer_sequence(settings)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["config"]["capture_binding"] == settings["capture_binding"]
    assert "_corrected_input_hashes" not in manifest["config"]
    assert [row["sha256"] for row in manifest["inputs"]] == settings["_corrected_input_hashes"]
    assert core.verify_run_manifest(result["manifest_path"])["ok"]

    derived = Path(binding["frames"][-1]["analysis_path"])
    original = derived.read_bytes()
    derived.write_bytes(original + b"changed")
    with pytest.raises(CaptureError) as altered:
        read_corrected_analysis(path)
    assert altered.value.code == "CORRECTED_IMAGE_CHANGED"
    with pytest.raises(core.CoreError) as altered_input:
        core.run_extensometer_sequence(settings)
    assert altered_input.value.code == "CORRECTED_INPUT_CHANGED"
    derived.write_bytes(original)

    last_raw = Path(binding["frames"][-1]["absolute_path"])
    original = last_raw.read_bytes()
    last_raw.write_bytes(original + b"changed")
    with pytest.raises(CaptureError) as altered_raw:
        prepare_corrected_analysis(capture.output, camera, 2)
    assert altered_raw.value.code == "CAPTURE_IMAGE_CHANGED"
    assert list(path.parent.parent.iterdir()) == [path.parent]
    last_raw.write_bytes(original)
    with ExperimentReader(capture.output) as reader:
        assert reader.verify()["complete"]
