"""Locate the packed image bytes before passing them to the PFNC decoder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import gcd

import numpy as np

from ..config import CaptureError
from ._genapi import image_layout, is_packed_mono_or_bayer


@dataclass(frozen=True)
class PackedFormat:
    name: str
    code: int
    wire_bits: int
    group_pixels: int
    expand: Callable[[np.ndarray], np.ndarray]


def packed_format(name: str | None) -> PackedFormat | None:
    if not isinstance(name, str) or not is_packed_mono_or_bayer(name):
        return None
    from harvesters.util.pfnc import Dictionary, dict_by_names

    legacy = name.endswith(("Packed", "packed"))
    symbolic = name[:-6] + "Packed" if legacy else name
    code = dict_by_names.get(symbolic)
    if code is None:
        return None
    bits, _ = image_layout(symbolic)
    # Legacy GigE Vision Mono10Packed and Bayer10Packed use three bytes for two samples.
    wire_bits = 12 if legacy else bits
    return PackedFormat(name, code, wire_bits, 8 // gcd(wire_bits, 8), Dictionary.get_proxy(symbolic=symbolic).expand)


def packed_samples(
    layout: PackedFormat, data: bytes, *, width: int, height: int, offset: int, filled: int,
    padding_x: int, padding_y: int, includes_chunks: bool | None,
) -> np.ndarray:
    geometry = f"{layout.name} {width} × {height}"
    if (not all(type(value) is int for value in (width, height, offset, filled, padding_x, padding_y))
        or width <= 0 or height <= 0 or min(offset, padding_x, padding_y) < 0 or not offset <= filled <= len(data)):
        raise CaptureError("GENTL_BAD_PAYLOAD", f"invalid packed image bounds for {geometry}")

    line_bits = width * layout.wire_bits
    row_bytes = (line_bits + 7) // 8
    frame_bytes = (line_bits * height + 7) // 8
    image_bytes = filled - offset - padding_y
    if image_bytes < frame_bytes:
        raise CaptureError("GENTL_BAD_PAYLOAD", f"truncated packed image for {geometry}")

    # PFNC permits both packing across the image and restarting at each row.
    # Explicit row padding or byte-aligned rows makes the layout determinate.
    row_footprint = (row_bytes + padding_x) * height
    if padding_x:
        if image_bytes < row_footprint:
            raise CaptureError("GENTL_BAD_PAYLOAD", f"truncated padded rows for {geometry}")
        rows, samples, data_bytes, stride = height, width, row_bytes, row_bytes + padding_x
    elif height == 1 or line_bits % 8 == 0:
        rows, samples, data_bytes, stride = 1, width * height, frame_bytes, frame_bytes
    elif includes_chunks is False and frame_bytes != row_footprint and image_bytes == frame_bytes:
        rows, samples, data_bytes, stride = 1, width * height, frame_bytes, frame_bytes
    elif includes_chunks is False and frame_bytes != row_footprint and image_bytes == row_footprint:
        rows, samples, data_bytes, stride = height, width, row_bytes, row_bytes
    else:
        raise CaptureError("GENTL_PACKING_AMBIGUOUS", f"{geometry}: use a width divisible by {layout.group_pixels} or an unpacked format")

    source = np.ndarray((rows, data_bytes), dtype=np.uint8, buffer=data, offset=offset, strides=(stride, 1))
    group = layout.group_pixels
    expanded_width = ((samples + group - 1) // group) * group
    if expanded_width != samples:
        # The decoder accepts whole groups. Zero-fill the unused end of each group after checking the actual extent.
        aligned = np.zeros((rows, expanded_width * layout.wire_bits // 8), dtype=np.uint8)
        aligned[:, :data_bytes] = source
        source = aligned
    try:
        expanded = layout.expand(source.ravel())
    except ValueError as exc:
        raise CaptureError("GENTL_BAD_PAYLOAD", f"could not decode packed samples for {geometry}") from exc
    if expanded.dtype != np.uint16 or expanded.size != rows * expanded_width:
        raise CaptureError("GENTL_BAD_PAYLOAD", f"invalid decoded samples for {geometry}")
    return np.ascontiguousarray(expanded.reshape(rows, expanded_width)[:, :samples]).reshape(height, width)
