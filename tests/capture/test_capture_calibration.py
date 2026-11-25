import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureSession, ExperimentReader
from straintrace_capture.config import CaptureError
from straintrace_capture.calibration import calibrate, undistort
from straintrace_capture.cameras.base import Camera, Device, Image, receipt_times
from straintrace_capture.handoff import analysis_paths, analysis_snapshot, derive_analysis_inputs, derive_bayer_inputs
from straintrace_capture.image_inputs import decode_samples, read_analysis_image


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
    configuration["phases"] = [{"mode": "single", "role": "reference"}]
    configuration["calibration_files"] = [str(profile)]
    sequel = CaptureSession(CaptureConfig.from_dict(configuration), tmp_path / "bound", registry=registry)
    sequel.start()
    assert sequel.wait(6)["state"] == "completed"
    with ExperimentReader(sequel.output) as reader:
        binding, = reader.experiment["calibrations"]
        assert (sequel.output / binding["path"]).read_bytes() == profile.read_bytes()
        assert reader.verify()["complete"]


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
