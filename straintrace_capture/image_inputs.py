"""Numerical input conversion that keeps recorded sensor samples intact."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .config import CaptureError


def decode_samples(path: Path, *, color_model: str | None = None) -> np.ndarray:
    if color_model in ("rgba", "bgra"):
        from PIL import Image as PillowImage

        # libtiff's RGBA shortcut premultiplies unassociated alpha. Read the
        # true samples through Pillow and return OpenCV's BGRA channel order.
        with PillowImage.open(path) as image:
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGRA)
    data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise CaptureError("FRAME_UNREADABLE", f"cannot decode {path}")
    return data


def bayer_luminance(path: Path, pixel_format: str) -> np.ndarray:
    match = re.fullmatch(r"Bayer(RG|GR|BG|GB)(8|10|12|14|16)(?:p|Packed)?", pixel_format)
    if match is None:
        raise CaptureError("BAYER_FORMAT_UNKNOWN", f"no mosaic pattern for {pixel_format!r}")
    pattern = {"RG": "RGGB", "GR": "GRBG", "BG": "BGGR", "GB": "GBRG"}[match[1]]
    code = getattr(cv2, f"COLOR_Bayer{pattern}2GRAY")
    return cv2.cvtColor(decode_samples(path), code)


def read_analysis_image(path: Path, frame: dict, *, debayer: bool = False) -> np.ndarray:
    color = frame["color_model"]
    if color == "bayer":
        if not debayer:
            raise CaptureError("BAYER_DERIVATION_REQUIRED", "enable Bayer luminance conversion or use a monochrome capture format")
        return bayer_luminance(path, frame["pixel_format"])
    if color in ("rgba", "bgra"):
        return cv2.cvtColor(decode_samples(path, color_model=color), cv2.COLOR_BGRA2GRAY)
    import ezdic_core as core

    return core.read_gray_image(path)
