import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureSession, ExperimentReader, prepare_repeat
from straintrace_capture.config import CaptureError
from straintrace_capture.calibration import (
    CheckerboardScan, CheckerboardTarget, calibrate, calibration_geometry, checkerboard_target, matching_calibration,
    prepare_checkerboard_capture, read_profile, undistort,
)
from straintrace_capture.cameras._gentl_multipart import IMAGE_PART
from straintrace_capture.cameras.base import Camera, Device, Feature, Image, receipt_times
from straintrace_capture.handoff import analysis_paths, analysis_snapshot, derive_analysis_inputs, derive_bayer_inputs
from straintrace_capture.image_inputs import decode_samples, read_analysis_image
from straintrace_capture.storage import ExperimentWriter


def checkerboards():
    width, height, cols, rows, side = 640, 480, 9, 6, 20
    k = np.float64([[720, 0, 319.5], [0, 715, 239.5], [0, 0, 1]])
    outer = np.float64([[-side, -side, 0], [cols * side, -side, 0],
                        [cols * side, rows * side, 0], [-side, rows * side, 0]])
    checks = (np.indices((rows + 1, cols + 1)).sum(axis=0) % 2 * 224 + 15).astype(np.uint8)
    tile = cv2.resize(checks, None, fx=40, fy=40, interpolation=cv2.INTER_NEAREST)
    rc = np.float32([[0., 0.], [tile.shape[1] - 1., 0],
                     [tile.shape[1] - 1., tile.shape[0] - 1.], [0, tile.shape[0] - 1.]])
    result = {}
    for name, baseline in (("board:left", np.array([0., 0., 0.])), ("board:right", np.array([-52., 0., 0.]))):
        result[name] = []
        for index in range(8):
            rv = np.float64([-.28 + index * .085, (-1)**index * (.10 + index * .02), -.1 + index * .03])
            tv = np.float64([-65 + index % 3 * 15, -50 + index % 4 * 12, 420 + index % 5 * 23]) + baseline
            xy, _ = cv2.projectPoints(outer, rv, tv, k, None)
            perspective = cv2.getPerspectiveTransform(rc, xy.reshape(4, 2).astype(np.float32))
            result[name].append(cv2.warpPerspective(tile, perspective, (width, height), borderValue=200))
    return result


def test_fit_lens_from_an_additional_saved_part(tmp_path):
    identity, part = "gentl:board", "4/0/0/Mono8"
    config = CaptureConfig.from_dict({"config_version": 1, "name": "Multipart checkerboard",
                                     "cameras": [{"id": identity}], "phases": [{"mode": "manual", "role": "calibration"}]})
    root = tmp_path / "checkerboards"
    writer = ExperimentWriter(root, config)
    try:
        writer.register_camera(Device(identity, "gentl", "Fixture", "Multipart checkerboard", "BOARD", "fixture"), {})
        rows = []
        for sequence, pixels in enumerate(checkerboards()["board:left"], 1):
            now, monotonic = receipt_times()
            main = Image(np.zeros_like(pixels), "Mono8", 8, "mono", now, monotonic,
                         metadata={IMAGE_PART: {"selector": "1/0/0/Mono8"}})
            sample = replace(main, pixels=pixels, metadata={IMAGE_PART: {"selector": part}})
            rows.append(writer.prepare_frame(identity, sequence, 1, "calibration", replace(main, auxiliary_images=(sample,)),
                                             None, None, None, 0))
        writer.commit_frames(rows)
    finally:
        writer.close()
    fitted = calibrate(root, [identity], inner_columns=9, inner_rows=6, square_mm=20, image_parts={identity: part})
    model = fitted["cameras"][identity]
    assert model[IMAGE_PART] == part and model["reprojection_rms_px"] < .6
    assert len(model["views"]) >= 6 and all(row["path"].endswith("-part-02.tiff") for row in model["views"])


class BoardCamera(Camera):
    def __init__(self, identity, frames):
        self.device = Device(identity, "board", "Test fixture", "Projected checkerboard", identity, "simulated")
        self.images = iter(frames)

    def open(self):
        self.sequence = 0

    def close(self):
        self.stop()

    def features(self):
        return {}

    def set_feature(self, name, value):
        raise AssertionError(name)

    def start(self, trigger="free", source=None):
        assert trigger == "software"
        self.armed = False

    def stop(self):
        self.armed = False

    def trigger(self):
        self.armed = True

    def read(self, timeout_s):
        assert self.armed
        self.armed = False
        self.sequence += 1
        return Image(next(self.images), "Mono8", 8, "mono", *receipt_times(), self.sequence, self.sequence)


def test_static_checkerboards_recover_calibrated_intrinsics_baseline_and_profile_binding(tmp_path):
    boards = checkerboards()
    registry = CameraRegistry(custom={"board": (lambda: [], lambda identity: BoardCamera(identity, boards[identity]))})
    configuration = {
        "config_version": 1, "name": "Known-geometry checkerboard",
        "cameras": [{"id": name} for name in boards],
        "phases": [{"mode": "timed", "role": "calibration", "fps": 6, "frame_count": 8}],
    }
    session = CaptureSession(CaptureConfig.from_dict(configuration), tmp_path / "board-capture", registry=registry)
    session.start()
    assert session.wait(10)["state"] == "completed"
    mono = calibrate(session.output, ["board:left"], inner_columns=9, inner_rows=6, square_mm=20)
    left = mono["cameras"]["board:left"]
    assert left["camera_matrix"][0][0] == pytest.approx(720, abs=22)
    assert left["reprojection_rms_px"] < .5
    stereo = calibrate(session.output, list(boards), inner_columns=9, inner_rows=6, square_mm=20)
    assert np.linalg.norm(stereo["stereo"]["translation_left_to_right_mm"]) == pytest.approx(52, abs=4)
    assert len(stereo["stereo"]["paired_trigger_events"]) >= 6
    corrected = undistort(boards["board:left"][0], stereo["cameras"]["board:left"])
    assert corrected.shape == (480, 640)
    profile = tmp_path / "calibration.json"
    profile.write_text(json.dumps(stereo), encoding="utf-8")
    configuration["phases"] = [{"mode": "single", "role": "reference"}, {"mode": "single"}]
    configuration["calibration_files"] = [str(profile)]
    sequel = CaptureSession(CaptureConfig.from_dict(configuration), tmp_path / "bound", registry=registry)
    sequel.start()
    assert sequel.wait(6)["state"] == "completed"
    with ExperimentReader(sequel.output) as reader:
        binding, = reader.experiment["calibrations"]
        assert (sequel.output / binding["path"]).read_bytes() == profile.read_bytes()
        assert reader.verify()["complete"]
    derived = derive_analysis_inputs(analysis_snapshot(sequel.output, "board:left"), tmp_path / "corrected", lens_profile=profile)
    np.testing.assert_array_equal(decode_samples(Path(derived["frames"][0]["analysis_path"])), corrected, strict=True)
    archived = sequel.output / binding["path"]
    assert prepare_repeat(sequel.output).config.calibration_files == (str(archived.resolve()),)
    archived.write_bytes(b"altered archive")
    with pytest.raises(CaptureError) as invalid_repeat:
        prepare_repeat(sequel.output)
    assert invalid_repeat.value.code == "REPEAT_CALIBRATION_INVALID"


class MultipartBoardCamera(BoardCamera):
    def __init__(self, identity, images, parts, sampling):
        super().__init__(identity, images)
        self.parts = iter(parts)
        self.sampling = iter(sampling)
        self.binning = 1

    def start(self, trigger="free", source=None):
        super().start(trigger, source)
        self.binning = next(self.sampling)

    def features(self):
        # These nodes describe a different component than the captured part.
        return {name: Feature(name, value) for name, value in {
            "width": 1280, "height": 1024, "offset_x": 80, "offset_y": 60, "binning_x": self.binning,
        }.items()}

    def read(self, timeout_s):
        frame = super().read(timeout_s)
        return replace(frame, metadata={IMAGE_PART: {"selector": next(self.parts), "part_index": self.sequence % 2}})


def test_calibration_checks_the_delivered_part_and_phase_before_export(tmp_path):
    identity = "board:left"
    boards = checkerboards()[identity]
    calibrated, shifted = "2/16/8/Mono8", "2/32/8/Mono8"

    def run(name, phases, parts, sampling, calibration_files=()):
        registry = CameraRegistry(custom={"board": (lambda: [], lambda camera: MultipartBoardCamera(
            camera, boards, parts, sampling,
        ))})
        config = CaptureConfig.from_dict({
            "config_version": 1, "name": name, "cameras": [{"id": identity}], "phases": phases,
            "calibration_files": list(calibration_files),
        })
        session = CaptureSession(config, tmp_path / name, registry=registry)
        session.start()
        assert session.wait(8)["state"] == "completed"
        return session.output

    board_phases = [{"mode": "timed", "role": "calibration", "fps": 12, "frame_count": len(boards)}]
    target = {"inner_columns": 9, "inner_rows": 6, "square_mm": 20}
    board_root = run("boards", board_phases, [calibrated] * len(boards), [1])
    profile = calibrate(board_root, [identity], **target)
    model = profile["cameras"][identity]
    assert model[IMAGE_PART] == calibrated
    assert model["sensor_geometry"] == {"width": 640, "height": 480, "offset_x": 16, "offset_y": 8, "binning_x": 1}

    changed = run("mixed-boards", board_phases, [calibrated] * (len(boards) - 1) + [shifted], [1])
    with pytest.raises(CaptureError) as mismatch:
        calibrate(changed, [identity], **target)
    assert mismatch.value.code == "CALIBRATION_GEOMETRY_CHANGED"

    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(CaptureError) as changed_profile:
        read_profile(path, expected_hash="0" * 64)
    assert changed_profile.value.code == "CALIBRATION_HASH_MISMATCH"
    phases = [{"mode": "single", "role": "reference"}, {"mode": "single"}]
    for name, parts, sampling, error in (
        ("matching", [calibrated] * 2, [1, 1], None),
        ("shifted", [calibrated, shifted], [1, 1], "CALIBRATION_PART_MISMATCH"),
        ("other-sensor", [calibrated, "9/16/8/Mono8"], [1, 1], "CALIBRATION_PART_MISMATCH"),
        ("sampled", [calibrated] * 2, [1, 2], "CALIBRATION_GEOMETRY_MISMATCH"),
    ):
        root = run(name, phases, parts, sampling, [str(path)])
        snapshot = analysis_snapshot(root)
        assert [snapshot["phase_sensor_geometry"][str(phase)]["binning_x"] for phase in (1, 2)] == sampling
        output = tmp_path / f"{name}-derived"
        if error:
            with pytest.raises(CaptureError) as incompatible:
                derive_analysis_inputs(snapshot, output, lens_profile=path)
            assert incompatible.value.code == error and snapshot["frames"][1]["frame_id"] in str(incompatible.value)
            assert not output.exists()
            continue
        derived = derive_analysis_inputs(snapshot, output, lens_profile=path)
        with ExperimentReader(root) as reader:
            assert reader.verify()["complete"]
        for row, original in zip(derived["frames"], boards[:2], strict=True):
            corrected = undistort(original, model)
            np.testing.assert_array_equal(decode_samples(Path(row["analysis_path"])), corrected, strict=True)
        geometry = calibration_geometry(boards[0].shape, {"binning_x": 1}, snapshot["frames"][0]["backend_metadata"])
        assert matching_calibration([model, {**model, IMAGE_PART: shifted}], geometry) is model
        legacy_model = {key: value for key, value in model.items() if key != IMAGE_PART}
        path.write_text(json.dumps({**profile, "cameras": {identity: legacy_model}}), encoding="utf-8")
        with pytest.raises(CaptureError) as unverified:
            derive_analysis_inputs(snapshot, tmp_path / "legacy-derived", lens_profile=path)
        assert unverified.value.code == "CALIBRATION_PART_UNVERIFIED"
        assert not (tmp_path / "legacy-derived").exists()
        path.write_text(json.dumps(profile), encoding="utf-8")


def test_guided_checkerboard_progress_counts_new_committed_poses_and_usable_pairs(tmp_path):
    boards = checkerboards()
    left, right = tuple(boards)
    images = {
        left: [boards[left][0]] * 2 + [boards[left][1], *boards[left][1:]],
        right: [boards[right][0]] * 2 + [np.zeros_like(boards[right][0]), *boards[right][1:]],
    }
    registry = CameraRegistry(custom={"board": (lambda: [], lambda identity: BoardCamera(identity, images[identity]))})
    plan = CaptureConfig.from_dict({
        "config_version": 1, "name": "Tension", "cameras": [{"id": key} for key in (left, right, "sim:0")],
        "phases": [{"mode": "single", "role": "reference"}, {"mode": "timed", "fps": 10, "frame_count": 12}],
        "calibration_files": ["existing.json"], "metadata": {"continuation_of": "earlier-experiment"},
        "durability": "write_through",
    })
    with pytest.raises(CaptureError, match="one odd"):
        CheckerboardTarget((left, right), 7, 5, 20)
    target = CheckerboardTarget((right, left), 9, 6, 20)
    formal_output = tmp_path / "tensile"
    config, output = prepare_checkerboard_capture(plan, formal_output, target)
    assert output.parent == formal_output.parent and output.name.startswith("calibration-")
    assert [camera.id for camera in config.cameras] == [right, left]
    assert config.durability == plan.durability and config.calibration_files == ()
    assert config.metadata.keys() == {"capture_checkerboard"}
    assert plan.calibration_files == ("existing.json",) and len(plan.phases) == 2
    assert not formal_output.exists() and checkerboard_target(plan.to_dict()) is None
    assert checkerboard_target(config.to_dict()) == target

    session = CaptureSession(config, output, registry=registry)
    scan = CheckerboardScan(output, target)

    def wait_for(check):
        until = time.monotonic() + 8
        while not check(session.status()):
            if session.finished or time.monotonic() > until:
                pytest.fail(f"checkerboard capture did not reach the expected state: {session.status()}")
            time.sleep(.02)

    session.start()
    try:
        wait_for(lambda status: status["phase_progress"] and status["phase_progress"]["trigger_available"])
        assert scan.scan()["latest"] == "waiting"
        for index in range(len(images[left])):
            session.request_trigger(expected_phase=1)
            wait_for(lambda status: all(row["saved"] >= index + 1 for row in status["cameras"].values()))
            report = scan.scan()
            if index in (0, 1, 2):
                assert report["latest"] == ("new_pose", "similar_pose", "unpaired")[index]
                assert report["distinct_poses"] == 1
            if index == 2:
                assert report["usable_pairs"] == 2
                assert report["cameras"][left]["detected"] == 3
                assert report["cameras"][right]["overlay"] is None
            assert not session.finished
    finally:
        session.request_stop()
        session.wait(8)

    assert report["usable_pairs"] == 9 and report["distinct_poses"] >= 6
    assert report["cameras"][left]["corner_span_fraction_xy"][0] > .25
    with ExperimentReader(output) as reader:
        assert len(reader.frames()) == 20
        assert not reader.verify()["issues"]
        assert checkerboard_target(reader.experiment["config"]) == target
        latest_right, = reader.frames(camera_id=right, limit=1, newest_first=True)
        stamps = latest_right["timestamps"]
        overlay = report["cameras"][right]["overlay"]
        assert overlay[:2] == (stamps["received_utc"]["value"], stamps["received_monotonic"]["value"])
        assert len(overlay[2]) == 54
        image_path = reader.image_path(latest_right)
    profile = calibrate(output, target.camera_ids, **target.parameters())
    assert profile["stereo"]["left_camera"] == right
    assert np.linalg.norm(profile["stereo"]["translation_left_to_right_mm"]) == pytest.approx(52, abs=4)
    assert len(profile["stereo"]["paired_trigger_events"]) == report["usable_pairs"]

    image_path.write_bytes(b"corrupt TIFF fixture")
    reopened = CheckerboardScan(output, target).scan()
    assert reopened["read_errors"] == 1 and reopened["usable_pairs"] == 8
    with pytest.raises(CaptureError, match="checksum mismatch"):
        calibrate(output, target.camera_ids, **target.parameters())


def test_bayer_analysis_derives_luminance_without_overwriting_raw_samples(tmp_path):
    class BayerCamera(BoardCamera):
        def read(self, timeout_s):
            return replace(super().read(timeout_s), pixel_format="BayerRG12", significant_bits=12, color_model="bayer")

    mosaic = np.zeros((80, 96), dtype=np.uint16)
    mosaic[::2, ::2] = 1000
    mosaic[1::2, ::2] = mosaic[::2, 1::2] = 2000
    mosaic[1::2, 1::2] = 3000
    registry = CameraRegistry(custom={"board": (lambda: [], lambda identity: BayerCamera(identity, [mosaic, mosaic]))})
    configuration = CaptureConfig.from_dict({
        "config_version": 1, "name": "Known Bayer mosaic", "cameras": [{"id": "board:left"}],
        "phases": [{"mode": "single", "role": "reference"}, {"mode": "single"}],
    })
    capture = CaptureSession(configuration, tmp_path / "mosaic", registry=registry)
    capture.start()
    assert capture.wait(6)["state"] == "completed"
    snapshot = analysis_snapshot(capture.output)
    with pytest.raises(CaptureError, match="Bayer luminance"):
        analysis_paths(snapshot)
    derived = derive_bayer_inputs(snapshot, tmp_path / "luminance")
    original, luminance = snapshot["frames"][0], derived["frames"][0]
    np.testing.assert_array_equal(decode_samples(capture.output / original["path"]), mosaic)
    pixels = decode_samples(Path(luminance["analysis_path"]))
    assert pixels.dtype == np.uint16
    assert int(pixels[10, 10]) == pytest.approx(1815, abs=3)
    assert luminance["sha256"] == original["sha256"] != luminance["analysis_sha256"]


def test_rgba_tiff_keeps_unassociated_samples_and_alpha_does_not_dim_dic_input(tmp_path):
    from PIL import Image as PillowImage

    class AlphaCamera(BoardCamera):
        def read(self, timeout_s):
            assert self.armed
            self.armed = False
            return Image(next(self.images), "RGBa8", 8, "rgba", *receipt_times())

    pixels = np.zeros((40, 64, 4), dtype=np.uint8)
    pixels[..., :3] = (90, 132, 20)
    pixels[..., 3] = np.arange(64) * 4
    registry = CameraRegistry(custom={"board": (lambda: [], lambda identity: AlphaCamera(identity, [pixels]))})
    config = CaptureConfig.from_dict({
        "config_version": 1, "name": "Unassociated alpha fixture", "cameras": [{"id": "board:left"}],
        "phases": [{"mode": "single", "role": "reference"}],
    })
    capture = CaptureSession(config, tmp_path / "rgba", registry=registry)
    capture.start()
    assert capture.wait(6)["state"] == "completed"
    binding = analysis_snapshot(capture.output)
    frame = binding["frames"][0]
    raw = Path(frame["absolute_path"])
    with PillowImage.open(raw) as tiff:
        assert tiff.tag_v2[338] == (2,)
        np.testing.assert_array_equal(np.asarray(tiff), pixels)
    np.testing.assert_array_equal(read_analysis_image(raw, frame), np.full((40, 64), 107, dtype=np.uint8))
    with pytest.raises(CaptureError, match="without alpha"):
        analysis_paths(binding)
    prepared = derive_analysis_inputs(binding, tmp_path / "luminance", drop_alpha=True)
    np.testing.assert_array_equal(decode_samples(Path(analysis_paths(prepared)[0])), read_analysis_image(raw, frame))
