import cv2
import numpy as np

from straintrace_capture import CaptureConfig, CaptureSession, ExperimentReader


def test_simulated_reference_and_sequence_are_committed_at_original_bit_depth(tmp_path):
    config = CaptureConfig.from_dict({
        "config_version": 1,
        "name": "Tensile coupon",
        "cameras": [{"id": "sim:0", "settings": {
            "width": 96, "height": 80, "pixel_format": "Mono12",
        }}],
        "phases": [
            {"mode": "single", "role": "reference"},
            {"mode": "timed", "fps": 20, "frame_count": 3},
        ],
    })
    root = tmp_path / "试验-01"
    capture = CaptureSession(config, root)
    capture.start()
    assert capture.wait(timeout=10)["state"] == "completed"
    with ExperimentReader(root) as experiment:
        rows = experiment.frames(camera_id="sim:0")
        assert [row["sequence"] for row in rows] == [1, 2, 3, 4]
        assert [row["role"] for row in rows] == ["reference", "test", "test", "test"]
        assert len({row["frame_id"] for row in rows}) == 4
        assert {row["significant_bits"] for row in rows} == {12}
        assert all(row["timestamps"]["received_utc"]["source"] == "host_receive" for row in rows)
        assert all(row["timestamps"]["exposure_start"] is None for row in rows)
        assert experiment.verify()["ok"]
        last = rows[-1]
        image = cv2.imdecode(np.fromfile(root / last["path"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert image.dtype == np.uint16
        assert image.shape == (80, 96)
        assert 255 < int(image.max()) <= 4095
        np.testing.assert_array_equal(image, capture.latest_image("sim:0").pixels)


def test_write_through_segments_keep_native_file_identity(tmp_path):
    run = CaptureSession(CaptureConfig.from_dict({
        "config_version": 1, "name": "Segment rehearsal",
        "cameras": [{"id": "sim:0", "settings": {"width": 96, "height": 80}}],
        "phases": [{"mode": "continuous", "frame_count": 3}],
        "segment_frames": 2, "durability": "write_through",
    }), tmp_path / "segments")
    run.start()
    assert run.wait(6)["state"] == "completed"
    with ExperimentReader(run.output) as reader:
        paths = [reader.image_path(row) for row in reader.frames()]
        assert paths[0].parent == paths[1].parent
        assert paths[2].parent != paths[0].parent
        assert [int(path.stem) for path in paths] == [1, 2, 3]
        assert reader.verify()["complete"]
