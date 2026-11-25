"""Uncompressed TIFF for the sensor sample layouts accepted by Capture."""

import hashlib
import struct
import sys
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .cameras.base import Image
from .config import CaptureError

_ENDIAN, _MARKER = ("<", b"II") if sys.byteorder == "little" else (">", b"MM")
_ASCII, _SHORT, _LONG, _RATIONAL = 2, 3, 4, 5
_STRIP_BYTES = 64 * 1024
_COALESCED_TIFF_LIMIT = 256 * 1024


@dataclass(frozen=True, slots=True)
class TiffPayload:
    size: int
    sha256: str
    chunks: tuple[bytes | memoryview, ...]


def _entry(tag: int, field_type: int, count: int, value: int) -> bytes:
    entry = struct.pack(f"{_ENDIAN}HHI", tag, field_type, count)
    if field_type == _SHORT and count == 1:
        return entry + struct.pack(f"{_ENDIAN}Hxx", value)
    return entry + struct.pack(f"{_ENDIAN}I", value)


def _build_header(width: int, height: int, channels: int, bits: int, description: bytes | None = None) -> bytes:
    row_bytes = width * channels * (bits // 8)
    rows_per_strip = max(1, _STRIP_BYTES // row_bytes)
    if row_bytes % 2:
        # Keep adjacent strip starts word-aligned without padding each strip.
        rows_per_strip = max(2, rows_per_strip - rows_per_strip % 2)
    rows_per_strip = min(height, rows_per_strip)
    strips = (height + rows_per_strip - 1) // rows_per_strip

    # One IFD, followed by its out-of-line field values and adjacent image strips.
    entries = 13 + (channels == 4) + (description is not None)
    x_resolution = 8 + 2 + 12 * entries + 4
    y_resolution = x_resolution + 8
    bits_offset = y_resolution + 8
    description_offset = bits_offset + (2 * channels if channels > 1 else 0)
    description_size = len(description) if description is not None else 0
    description_storage = description_size + description_size % 2 if description_size > 4 else 0
    offsets_table = description_offset + description_storage
    lengths_table = offsets_table + 4 * strips
    pixels_offset = lengths_table + 4 * strips if strips > 1 else offsets_table
    pixel_bytes = height * row_bytes
    if pixels_offset + pixel_bytes > 0xFFFFFFFF:
        raise CaptureError("ENCODE_FAILED", "image exceeds the classic TIFF file size limit")

    directory = [
        _entry(256, _LONG, 1, width),                 # ImageWidth
        _entry(257, _LONG, 1, height),                # ImageLength
        _entry(258, _SHORT, channels, bits if channels == 1 else bits_offset),
        _entry(259, _SHORT, 1, 1),                    # Compression: none
        _entry(262, _SHORT, 1, 1 if channels == 1 else 2),  # BlackIsZero / RGB
    ]
    if description is not None:
        if description_storage:
            directory.append(_entry(270, _ASCII, description_size, description_offset))
        else:
            directory.append(struct.pack(f"{_ENDIAN}HHI", 270, _ASCII, description_size) + description.ljust(4, b"\0"))
    directory += [
        _entry(273, _LONG, strips, offsets_table if strips > 1 else pixels_offset),
        _entry(277, _SHORT, 1, channels),              # SamplesPerPixel
        _entry(278, _LONG, 1, rows_per_strip),        # RowsPerStrip
        _entry(279, _LONG, strips, lengths_table if strips > 1 else pixel_bytes),
        _entry(282, _RATIONAL, 1, x_resolution),
        _entry(283, _RATIONAL, 1, y_resolution),
        _entry(284, _SHORT, 1, 1),                    # Chunky samples
        _entry(296, _SHORT, 1, 1),                    # ResolutionUnit: none
    ]
    if channels == 4:
        directory.append(_entry(338, _SHORT, 1, 2))   # Unassociated alpha

    tiff_header = struct.pack(f"{_ENDIAN}2sHIH", _MARKER, 42, 8, entries)
    # No next IFD. Give both axes unitless resolution rather than assigning a physical scale.
    values = struct.pack(f"{_ENDIAN}IIIII", 0, 1, 1, 1, 1)
    if channels > 1:
        values += struct.pack(f"{_ENDIAN}{channels}H", *([bits] * channels))
    if description_storage:
        values += description.ljust(description_storage, b"\0")
    if strips > 1:
        strip_bytes = rows_per_strip * row_bytes
        offsets = range(pixels_offset, pixels_offset + pixel_bytes, strip_bytes)
        last_length = pixel_bytes - strip_bytes * (strips - 1)
        values += struct.pack(f"{_ENDIAN}{strips}I", *offsets)
        values += struct.pack(f"{_ENDIAN}{strips}I", *([strip_bytes] * (strips - 1)), last_length)
    return tiff_header + b"".join(directory) + values


_header = lru_cache(maxsize=64)(_build_header)


def sample_parts(image: Image, *, description: str | None = None) -> tuple[bytes, memoryview]:
    pixels = image.pixels
    height, width = pixels.shape[:2]
    channels = pixels.shape[2] if pixels.ndim == 3 else 1
    bits = pixels.dtype.itemsize * 8
    if channels == 4 and bits != 8:
        raise CaptureError("PIXEL_FORMAT_UNSUPPORTED", "four-channel TIFF requires 8-bit samples")
    header = (_header(width, height, channels, bits) if description is None else
              _build_header(width, height, channels, bits, description.encode("ascii") + b"\0"))
    if image.color_model == "bgr":
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    elif image.color_model == "bgra":
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGBA)
    return header, memoryview(np.ascontiguousarray(pixels)).cast("B")


def prepare_tiff(image: Image) -> TiffPayload:
    header, samples = sample_parts(image)
    size = len(header) + samples.nbytes
    # Use one file write for small frames. Large frames can be written from the sensor buffer.
    chunks = (header + samples,) if size <= _COALESCED_TIFF_LIMIT else (header, samples)
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return TiffPayload(size, digest.hexdigest(), chunks)
