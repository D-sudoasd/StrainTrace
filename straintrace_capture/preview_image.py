"""Standalone preview TIFFs and the evidence for comparing their raw samples."""

from dataclasses import asdict, dataclass
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
from PIL import Image as PillowImage

from .cameras._gentl_multipart import IMAGE_PART
from .cameras.base import Device, Image
from .config import CaptureError, SENSOR_WINDOW_FIELDS
from .continuation import camera_identity
from .image_inputs import decode_samples
from .image_quality_watch import sensor_window_values
from .output import resolve_output_path
from .storage import _publish_path
from .tiff_samples import sample_parts


PREVIEW_IMAGE_SCHEMA = "straintrace.capture.preview-image/v1"
_TIFF_COLOR = {"bgr": "rgb", "bgra": "rgba"}
_DECODED_COLOR = {"rgb": "bgr", "rgba": "bgra"}


@dataclass(frozen=True, slots=True)
class PreviewOriginal:
    device: Device
    image: Image
    sensor_window: dict[str, int]
    path: Path | None = None


def preview_original(device, image, values, *, path=None):
    return PreviewOriginal(device, image, sensor_window_values(image.metadata, values), path)


def save_preview_image(target: Path, device: Device, image: Image, *, readback=None) -> Path:
    target = resolve_output_path(target)
    if target.suffix.lower() not in {".tif", ".tiff"}:
        raise CaptureError("PREVIEW_SAVE_SUFFIX", str(target))
    if os.path.lexists(target):
        raise CaptureError("PREVIEW_SAVE_EXISTS", str(target))

    description = json.dumps({
        "schema": PREVIEW_IMAGE_SCHEMA,
        "camera": asdict(device),
        "pixel_format": image.pixel_format,
        "significant_bits": image.significant_bits,
        "color_model": _TIFF_COLOR.get(image.color_model, image.color_model),
        "shape": list(image.pixels.shape),
        "dtype": image.pixels.dtype.name,
        "sensor_window": sensor_window_values(image.metadata, readback or {}),
        "received_utc_ns": image.received_utc_ns,
        "time_source": "host_receive",
        "native_frame_id": image.native_frame_id,
        "image_part": image.metadata.get(IMAGE_PART),
    }, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    chunks = sample_parts(image, description=description)
    partial = None
    try:
        with NamedTemporaryFile(dir=target.parent, prefix=".capture-preview-", suffix=".part", delete=False) as stream:
            partial = Path(stream.name)
            for chunk in chunks:
                if stream.write(chunk) != len(chunk):
                    raise OSError(f"incomplete TIFF write at {partial}")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            _publish_path(partial, target, write_through=True)
        except FileExistsError as exc:
            raise CaptureError("PREVIEW_SAVE_EXISTS", str(target)) from exc
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    return target


def read_preview_image(path: Path) -> PreviewOriginal:
    """Decode metadata and raw samples from the same file contents."""
    encoded = path.read_bytes()
    try:
        with PillowImage.open(BytesIO(encoded)) as tiff:
            if tiff.format != "TIFF":
                raise CaptureError("PREVIEW_ORIGINAL_METADATA", str(path))
            record = json.loads(tiff.tag_v2.get(270, "null"))
            if not isinstance(record, dict) or record.get("schema") != PREVIEW_IMAGE_SCHEMA:
                raise CaptureError("PREVIEW_ORIGINAL_METADATA", str(path))
            if tiff.n_frames != 1 or tiff.tag_v2.get(259, 1) != 1:
                raise CaptureError("PREVIEW_ORIGINAL_LAYOUT", str(path))
    except (OSError, ValueError, TypeError, PillowImage.DecompressionBombError) as exc:
        raise CaptureError("PREVIEW_ORIGINAL_METADATA", str(path)) from exc

    try:
        device = Device(**record["camera"])
        color, pixel_format = record["color_model"], record["pixel_format"]
        received, bits = record["received_utc_ns"], record["significant_bits"]
        sensor, part = record.get("sensor_window", {}), record.get("image_part")
        if (not all(isinstance(getattr(device, field), str) and getattr(device, field)
                    for field in ("id", "backend", "model", "identity_scope"))
                or device.serial is not None and not isinstance(device.serial, str)
                or not isinstance(pixel_format, str) or not pixel_format
                or color not in ("mono", "bayer", "rgb", "rgba")
                or type(received) is not int or type(bits) is not int
                or not isinstance(sensor, dict) or any(
                    name not in SENSOR_WINDOW_FIELDS or type(value) is not int
                    or not int(not name.startswith("offset_")) <= value <= 1e12 for name, value in sensor.items()
                )
                or (part is not None and (not isinstance(part, dict) or not isinstance(part.get("selector"), str)))):
            raise ValueError("invalid preview image record")
    except (KeyError, TypeError, ValueError) as exc:
        raise CaptureError("PREVIEW_ORIGINAL_METADATA", str(path)) from exc

    try:
        pixels = decode_samples(path, color_model=color, encoded=encoded)
        if (record.get("shape") != list(pixels.shape)
                or record.get("dtype", pixels.dtype.name) != pixels.dtype.name):
            raise ValueError("TIFF samples disagree with the recorded image geometry")
        image = Image(pixels, pixel_format, bits, _DECODED_COLOR.get(color, color), received, 0,
                      native_frame_id=record.get("native_frame_id"), metadata={IMAGE_PART: part} if part else {})
    except (OSError, ValueError, cv2.error, CaptureError) as exc:
        raise CaptureError("PREVIEW_ORIGINAL_LAYOUT", str(path)) from exc
    return PreviewOriginal(device, image, sensor, path)


def preview_region_mismatch(first: PreviewOriginal, second: PreviewOriginal) -> str | None:
    """Check whether the same pixel rectangle can be used in both images."""
    left, right = first.device, second.device
    if left.id != right.id:
        return "camera"
    identity = camera_identity({left.id: asdict(left)}, left.id, right)
    if identity == "changed":
        return "camera"
    if identity not in ("matched", "path_matched"):
        return "identity_unknown"
    a, b = first.image, second.image
    if (a.pixels.shape != b.pixels.shape or a.pixels.dtype != b.pixels.dtype
            or a.pixel_format != b.pixel_format or a.significant_bits != b.significant_bits
            or _TIFF_COLOR.get(a.color_model, a.color_model) != _TIFF_COLOR.get(b.color_model, b.color_model)
            or (a.metadata.get(IMAGE_PART) or {}).get("selector") != (b.metadata.get(IMAGE_PART) or {}).get("selector")):
        return "layout"
    x, y = first.sensor_window, second.sensor_window
    if any(x[name] != y[name] for name in x.keys() & y.keys()):
        return "sensor_window"
    if x.keys() != y.keys() or not {"offset_x", "offset_y"} <= x.keys():
        return "sensor_unknown"
    return None
