import hashlib
import threading

import numpy as np
import pytest
from PIL import Image as PillowImage

from straintrace_capture import (
    CaptureConfig,
    CaptureSession,
    ExperimentReader,
    session,
    storage,
)
from straintrace_capture.cameras.base import Image, receipt_times
from straintrace_capture.image_inputs import decode_samples
from straintrace_capture.storage import _stage_tiff


@pytest.mark.parametrize(
    ("model", "fmt", "bits", "shape"),
    [
        ("mono", "Mono8", 8, (11, 17)),
        ("mono", "Mono16", 16, (641, 511)),
        ("bayer", "BayerRG12", 12, (129, 511)),
        ("rgb", "RGB8", 8, (127, 199, 3)),
        ("bgr", "BGR16", 16, (113, 263, 3)),
        ("rgba", "RGBa8", 8, (301, 239, 4)),
        ("bgra", "BGRa8", 8, (39, 27, 4)),
    ],
)
def test_capture_tiff_retains_samples_and_journal_digest(tmp_path, model, fmt, bits, shape):
    rng = np.random.default_rng(61)
    dtype = np.uint8 if bits <= 8 else np.uint16
    pixels = rng.integers(0, 2**bits, shape, dtype=dtype)[::-1]
    if model in ("rgba", "bgra"):
        # Nonzero RGB values must survive even when the alpha sample is zero.
        pixels[0, 0] = (3, 121, 232, 0)
    image = Image(pixels, fmt, bits, model, *receipt_times())
    target = tmp_path / "samples.tiff"
    descriptors = []
    partial = _stage_tiff(target, image, before_write=lambda size, digest: descriptors.append((size, digest)))
    data = partial.read_bytes()
    assert descriptors == [(len(data), hashlib.sha256(data).hexdigest())]
    assert not target.exists()

    expected = pixels
    if model in ("rgb", "rgba"):
        expected = pixels[..., [2, 1, 0, 3] if model == "rgba" else [2, 1, 0]]
    np.testing.assert_array_equal(decode_samples(partial, color_model=model), expected, strict=True)

    with PillowImage.open(partial) as decoded:
        assert decoded.size == (shape[1], shape[0])
        assert decoded.tag_v2[259] == 1
        offsets, lengths = decoded.tag_v2[273], decoded.tag_v2[279]
        assert len(offsets) == len(lengths) and sum(lengths) == pixels.nbytes
        assert all(offset % 2 == 0 for offset in offsets)
        assert all(start + length == next_start for start, length, next_start in zip(offsets, lengths, offsets[1:]))
        if model in ("rgba", "bgra"):
            assert decoded.tag_v2[338] == (2,)


def _large_frame_session(tmp_path, mode):
    config = CaptureConfig.from_dict({
        "config_version": 1, "name": "Large TIFF checksum", "durability": mode,
        "cameras": [{"id": "sim:0", "settings": {"width": 1600, "height": 800, "pixel_format": "Mono12"}}],
        "phases": [{"mode": "continuous", "frame_count": 4}],
    })
    return CaptureSession(config, tmp_path / "capture")


@pytest.mark.parametrize("mode", ["buffered", "write_through"])
def test_large_tiff_preparation_uses_one_checksum_per_recorded_frame(tmp_path, monkeypatch, mode):
    prepare = storage.prepare_tiff
    threads = []

    def record_preparation(image):
        threads.append(threading.current_thread().name)
        return prepare(image)

    monkeypatch.setattr(session, "prepare_tiff", record_preparation)
    monkeypatch.setattr(storage, "prepare_tiff", record_preparation)
    capture = _large_frame_session(tmp_path, mode)
    capture.start()
    assert capture.wait(8)["state"] == "completed"
    with ExperimentReader(capture.output) as reader:
        checked = reader.verify()
    assert checked["complete"] and checked["committed_frames"] == len(threads) == 4
    prefix = "tiff-checksum" if mode == "buffered" else "capture-disk"
    assert all(thread.startswith(prefix) for thread in threads)


def test_checksum_worker_failure_terminates_with_an_unsaved_frame(tmp_path, monkeypatch):
    def fail_preparation(_image):
        raise OSError("simulated checksum worker failure")

    monkeypatch.setattr(session, "prepare_tiff", fail_preparation)
    capture = _large_frame_session(tmp_path, "buffered")
    capture.start()
    status = capture.wait(8)
    assert status["state"] == "failed" and status["error"]["code"] == "WRITE_FAILED"
    assert status["cameras"]["sim:0"]["app_unsaved"] >= 1
    with ExperimentReader(capture.output) as reader:
        checked = reader.verify()
    assert checked["committed_frames"] == 0
