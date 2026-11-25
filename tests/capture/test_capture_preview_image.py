from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import tifffile

from straintrace_capture.cameras._gentl_multipart import IMAGE_PART
from straintrace_capture.cameras.base import Image
from straintrace_capture.cameras.simulated import simulated_device
from straintrace_capture.config import CaptureError
from straintrace_capture.preview_image import (
    PREVIEW_IMAGE_SCHEMA, preview_original, preview_region_mismatch, read_preview_image, save_preview_image,
)
from straintrace_capture.tiff_samples import sample_parts


@pytest.mark.parametrize(("model", "format_name", "bits", "shape"), [
    ("mono", "Mono12", 12, (257, 299)),
    ("bayer", "BayerRG12", 12, (257, 299)),
    ("bgr", "BGR8", 8, (15, 23, 3)),
    ("bgra", "RGBa8", 8, (421, 209, 4)),
])
def test_preview_tiff_preserves_raw_samples_and_receipt(tmp_path, model, format_name, bits, shape):
    dtype = np.uint16 if bits > 8 else np.uint8
    pixels = np.random.default_rng(25).integers(0, 1 << bits, size=shape, dtype=dtype)
    original = Image(pixels, format_name, bits, model, 1795631201001234567, 167, native_frame_id="931",
                     metadata={IMAGE_PART: {"selector": "intensity", "offset_x": 14, "offset_y": 5}})
    if model == "bgr":
        pixels = pixels[..., [2, 1, 0]]
    elif model == "bgra":
        pixels = pixels[..., [2, 1, 0, 3]]
    camera = replace(simulated_device(1), model="模拟相机")
    target = tmp_path / "预览原图.TIF"

    assert save_preview_image(target, camera, original) == target
    with tifffile.TiffFile(target) as tiff:
        page, = tiff.pages
        assert np.array_equal(page.asarray(), pixels)
        assert page.bitspersample == np.dtype(dtype).itemsize * 8
        assert page.compression.name == "NONE"
        record = json.loads(page.description)
    assert record["schema"] == PREVIEW_IMAGE_SCHEMA
    assert record["camera"]["id"] == "sim:1" and record["camera"]["model"] == "模拟相机"
    assert record["significant_bits"] == bits and record["shape"] == list(shape)
    assert record["pixel_format"] == format_name and record["received_utc_ns"] == original.received_utc_ns
    assert record["image_part"] == original.metadata[IMAGE_PART] and record["native_frame_id"] == "931"
    loaded = read_preview_image(target)
    assert loaded.path == target and loaded.device == camera and loaded.image.received_utc_ns == original.received_utc_ns
    assert loaded.sensor_window == {"offset_x": 14, "offset_y": 5}
    assert np.array_equal(loaded.image.pixels, original.pixels)
    assert list(tmp_path.iterdir()) == [target]


def test_preview_save_never_replaces_a_file_appearing_during_write(tmp_path, monkeypatch):
    from straintrace_capture import preview_image

    image = Image(np.ones((5, 7), dtype=np.uint8), "Mono8", 8, "mono", 100, 1)
    saved = tmp_path / "other.tiff"
    publish = preview_image._publish_path

    def race(partial, finished, *, write_through):
        assert partial.is_file() and finished == saved
        finished.write_bytes(b"external file")
        publish(partial, finished, write_through=write_through)

    monkeypatch.setattr(preview_image, "_publish_path", race)
    for _ in range(2):
        with pytest.raises(CaptureError) as error:
            save_preview_image(saved, simulated_device(0), image)
        assert error.value.code == "PREVIEW_SAVE_EXISTS"
        assert saved.read_bytes() == b"external file" and list(tmp_path.iterdir()) == [saved]
    with pytest.raises(CaptureError) as bad_name:
        save_preview_image(tmp_path / "原图.png", simulated_device(0), image)
    assert bad_name.value.code == "PREVIEW_SAVE_SUFFIX" and list(tmp_path.iterdir()) == [saved]


def test_preview_comparison_binds_region_to_camera_layout_and_reported_sensor_window(tmp_path):
    device = simulated_device(1)
    image = Image(np.zeros((48, 64), dtype=np.uint16), "Mono12", 12, "mono", 100, 1)
    readback = {"offset_x": 4, "offset_y": 8, "binning_x": 2}
    path = save_preview_image(tmp_path / "原图.tiff", device, image, readback=readback)
    first = read_preview_image(path)
    assert first.sensor_window == readback
    assert preview_region_mismatch(first, preview_original(device, image, readback)) is None
    assert preview_region_mismatch(first, preview_original(simulated_device(0), image, readback)) == "camera"
    assert preview_region_mismatch(first, preview_original(replace(device, serial="different"), image, readback)) == "camera"
    assert preview_region_mismatch(first, preview_original(device, replace(image, significant_bits=10), readback)) == "layout"
    assert preview_region_mismatch(first, preview_original(device, image, {**readback, "offset_x": 6})) == "sensor_window"
    assert preview_region_mismatch(first, preview_original(device, image, {"offset_x": 4, "offset_y": 8})) == "sensor_unknown"


def test_previous_preview_tiff_without_sensor_metadata_opens_for_whole_image_review(tmp_path):
    image = Image(np.full((17, 19), 1100, dtype=np.uint16), "Mono12", 12, "mono", 190_000, 1)
    device = simulated_device(0)
    record = {"schema": PREVIEW_IMAGE_SCHEMA, "camera": asdict(device), "pixel_format": "Mono12", "significant_bits": 12,
              "color_model": "mono", "shape": [17, 19], "received_utc_ns": 190_000, "time_source": "host_receive",
              "native_frame_id": None, "image_part": None}
    source = tmp_path / "早期预览.tiff"
    source.write_bytes(b"".join(sample_parts(image, description=json.dumps(record))))
    previous = read_preview_image(source)
    assert np.array_equal(previous.image.pixels, image.pixels) and previous.sensor_window == {}
    current = preview_original(device, image, {"offset_x": 0, "offset_y": 0})
    assert preview_region_mismatch(previous, current) == "sensor_unknown"

    source.write_bytes(b"".join(sample_parts(image, description=json.dumps({**record, "shape": [32, 19]}))))
    with pytest.raises(CaptureError) as changed:
        read_preview_image(source)
    assert changed.value.code == "PREVIEW_ORIGINAL_LAYOUT"
    tifffile.imwrite(source, image.pixels)
    with pytest.raises(CaptureError) as unrelated:
        read_preview_image(source)
    assert unrelated.value.code == "PREVIEW_ORIGINAL_METADATA"
