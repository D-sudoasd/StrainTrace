"""Numerical input conversion that keeps recorded sensor samples intact."""

from __future__ import annotations

import re
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

from .cameras._gentl_multipart import IMAGE_PART, parse_image_part
from .cameras.base import Image
from .config import CaptureError
from .storage import frame_image_records, sha256_file


def frame_part_selector(frame: dict) -> str | None:
    part = (frame.get("backend_metadata") or {}).get(IMAGE_PART)
    return part.get("selector") if isinstance(part, dict) else None


def analysis_part_options(frame: dict) -> dict[str, tuple[str | None, str]]:
    choices = {}
    for index, asset in enumerate(frame_image_records(frame)):
        selector = frame_part_selector(asset) if index else None
        if index and not selector:
            continue
        label = f"部件 {index + 1} · {selector}" if index else f"主图像 · {asset['pixel_format']}"
        height, width = asset["shape"][:2]
        choices[label] = (selector, f"{width} × {height} 像素 · {asset['significant_bits']} 位有效数据")
    return choices


def validate_analysis_part(selector: str | None) -> None:
    if selector is None:
        return
    try:
        if isinstance(parse_image_part(selector), tuple):
            return
    except CaptureError:
        pass
    raise CaptureError("ANALYSIS_PART_SELECTOR", "请输入完整的图像部件编号，例如 2/16/0/Mono8。")


def select_frame_image(frame: dict, part: str | None) -> dict:
    """Use one TIFF from this receipt while retaining the parent frame's identity and clocks."""
    if part is None:
        return frame
    choices = [asset for asset in frame_image_records(frame) if frame_part_selector(asset) == part]
    if len(choices) != 1:
        code = "ANALYSIS_PART_AMBIGUOUS" if choices else "ANALYSIS_PART_MISSING"
        raise CaptureError(code, f"{frame['frame_id']} · {part}")
    asset = choices[0]
    selected = {**frame, **asset, "backend_metadata": {**(frame.get("backend_metadata") or {}), **asset["backend_metadata"]}}
    selected.pop("auxiliary_images", None)
    return selected


def require_part_geometry(reference: dict, sample: dict) -> None:
    keys = ("shape", "dtype", "pixel_format", "color_model", "significant_bits")
    if frame_part_selector(reference) != frame_part_selector(sample) or any(reference[key] != sample[key] for key in keys):
        raise CaptureError("FRAME_GEOMETRY_CHANGED", f"{sample['frame_id']} · {frame_part_selector(sample)}")


def verified_image_path(reader, frame: dict) -> Path:
    path = reader.image_path(frame)
    try:
        if sha256_file(path) == frame["sha256"]:
            return path
    except FileNotFoundError as exc:
        raise CaptureError("FRAME_UNAVAILABLE", frame["path"]) from exc
    raise CaptureError("CAPTURE_IMAGE_CHANGED", frame["path"])


def _verified_bytes(path: Path, digest: str, encoded: bytes | None = None) -> bytes:
    if encoded is None:
        try:
            encoded = path.read_bytes()
        except FileNotFoundError as exc:
            raise CaptureError("FRAME_UNAVAILABLE", str(path)) from exc
    if sha256(encoded).hexdigest() != digest:
        raise CaptureError("CAPTURE_IMAGE_CHANGED", str(path))
    return encoded


def decode_samples(path: Path, *, color_model: str | None = None, expected_sha256: str | None = None,
                   encoded: bytes | None = None) -> np.ndarray:
    if expected_sha256 is not None:
        encoded = _verified_bytes(path, expected_sha256, encoded)
    if color_model in ("rgba", "bgra"):
        from PIL import Image as PillowImage

        # libtiff's RGBA shortcut premultiplies unassociated alpha. Read the
        # true samples through Pillow and return OpenCV's BGRA channel order.
        with PillowImage.open(path if encoded is None else BytesIO(encoded)) as image:
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGRA)
    raw = np.fromfile(path, dtype=np.uint8) if encoded is None else np.frombuffer(encoded, dtype=np.uint8)
    data = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if data is None:
        raise CaptureError("FRAME_UNREADABLE", f"cannot decode {path}")
    return data


def _recorded_digest(path: Path, row: dict) -> str:
    digest = row.get("sha256")
    if not isinstance(digest, str):
        raise CaptureError("INVALID_LEDGER", f"no image digest recorded for {path}")
    return digest


def recorded_image(path: Path, row: dict, *, verify=False) -> Image:
    color = row["color_model"]
    digest = _recorded_digest(path, row) if verify else None
    pixels = decode_samples(path, color_model=color, expected_sha256=digest)
    color = {"rgb": "bgr", "rgba": "bgra"}.get(color, color)
    stamps = row["timestamps"]
    return Image(pixels, row["pixel_format"], row["significant_bits"], color,
                 stamps["received_utc"]["value"], stamps["received_monotonic"]["value"],
                 metadata=row.get("backend_metadata") or {})


def bayer_luminance(path: Path, pixel_format: str, *, expected_sha256: str | None = None) -> np.ndarray:
    match = re.fullmatch(r"Bayer(RG|GR|BG|GB)(8|10|12|14|16)(?:p|[Pp]acked)?", pixel_format)
    if match is None:
        raise CaptureError("BAYER_FORMAT_UNKNOWN", f"no mosaic pattern for {pixel_format!r}")
    pattern = {"RG": "RGGB", "GR": "GRBG", "BG": "BGGR", "GB": "GBRG"}[match[1]]
    code = getattr(cv2, f"COLOR_Bayer{pattern}2GRAY")
    return cv2.cvtColor(decode_samples(path, expected_sha256=expected_sha256), code)


def read_analysis_image(path: Path, frame: dict, *, debayer: bool = False, verify: bool = False) -> np.ndarray:
    color = frame["color_model"]
    digest = _recorded_digest(path, frame) if verify else None
    if color == "bayer":
        if not debayer:
            raise CaptureError("BAYER_DERIVATION_REQUIRED", "enable Bayer luminance conversion or use a monochrome capture format")
        return bayer_luminance(path, frame["pixel_format"], expected_sha256=digest)
    if color in ("rgba", "bgra"):
        return cv2.cvtColor(decode_samples(path, color_model=color, expected_sha256=digest), cv2.COLOR_BGRA2GRAY)
    import ezdic_core as core

    encoded = _verified_bytes(path, digest) if verify else None
    return core.read_gray_image(path, encoded=encoded)
