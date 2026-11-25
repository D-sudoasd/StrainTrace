"""Identify and decode PFNC 2D images from a GenTL multipart buffer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..config import CaptureError
from ._genapi import image_layout, optional
from ._gentl_packed import PackedFormat, packed_format, packed_samples


IMAGE_PART = "gentl_image_part"
SAVE_ALL_2D = "gentl_save_all_2d"
AUTO_PART = "auto"


def parse_image_part(value):
    """Parse auto, an unsigned source ID, or source/x-offset/y-offset/PFNC name."""
    def number(token):
        return (int(token) if len(token) <= 20 and token.isascii() and token.isdecimal()
                and int(token) < 2**64 else None)

    if isinstance(value, str):
        if value == AUTO_PART:
            return None
        if (source := number(value)) is not None:
            return source
        fields = value.split("/")
        if len(fields) == 4 and all(token == "-" or number(token) is not None for token in fields[:3]):
            image_layout(fields[3])
            return (*map(number, fields[:3]), fields[3])
    raise CaptureError("GENTL_PART_SELECTOR_INVALID", "请输入 auto、源编号（如 2）或完整编号（如 2/0/0/Mono8）。")


def _id(value):
    return value if type(value) is int and 0 <= value < 2**64 else None


@dataclass(frozen=True)
class ImagePartOption:
    """A selectable descriptor with no reference to a returned SDK buffer."""

    selector: str
    width: int
    height: int


class ImagePartSelectionError(CaptureError):
    def __init__(self, code, message, options):
        super().__init__(code, message)
        self.options = options


@dataclass(frozen=True)
class ImagePart:
    part: object
    index: int
    code: int
    name: str
    bits: int
    color: str
    packed: PackedFormat | None
    source: int | None
    x: int | None
    y: int | None
    width: int | None
    height: int | None

    @property
    def key(self):
        return self.source, self.x, self.y, self.name

    @property
    def binding(self):
        return (*self.key, self.code, self.width, self.height)

    @property
    def selector(self):
        return "/".join(str(value) if value is not None else "-" for value in self.key)

    def metadata(self, byte_order):
        return {"selector": self.selector, "part_index": self.index, "source_id": self.source,
                "offset_x": self.x, "offset_y": self.y, "pixel_format_code": self.code, "byte_order": byte_order}


def select_image_part(buffer, selector, bound):
    from genicam.gentl import PARTDATATYPE_IDS as PartType
    from genicam.gentl import PIXELFORMAT_NAMESPACE_IDS as Namespace
    from harvesters.util.pfnc import dict_by_ints

    candidates = []
    namespaces = (None, Namespace.PIXELFORMAT_NAMESPACE_UNKNOWN, Namespace.PIXELFORMAT_NAMESPACE_GEV,
                  Namespace.PIXELFORMAT_NAMESPACE_PFNC_32BIT)
    for index, part in enumerate(buffer.parts):
        if optional(lambda: part.data_type) != PartType.PART_DATATYPE_2D_IMAGE:
            continue
        code = optional(lambda: part.data_format)
        fmt = dict_by_ints.get(code)
        if fmt is None or optional(lambda: part.data_format_namespace) not in namespaces:
            continue
        try:
            bits, color = image_layout(fmt)
        except CaptureError:
            continue
        packed = packed_format(fmt)
        storage_bits = (8 if bits <= 8 else 16) * (1 if color in ("mono", "bayer") else len(color))
        if packed is None and ((code >> 16) & 0xff) != storage_bits:
            continue
        candidates.append(ImagePart(
            part, index, code, fmt, bits, color, packed, _id(optional(lambda: part.source_id)),
            _id(optional(lambda: part.x_offset)), _id(optional(lambda: part.y_offset)),
            optional(lambda: part.width), optional(lambda: part.delivered_image_height) or optional(lambda: part.height),
        ))
    if bound is not None:
        eligible = [part for part in candidates if part.binding == bound]
    elif isinstance(selector, int):
        eligible = [part for part in candidates if part.source == selector]
    elif selector is not None:
        eligible = [part for part in candidates if part.key == selector]
    else:
        eligible = candidates
    counts = Counter(part.key for part in candidates)
    options = tuple(sorted(
        (ImagePartOption(part.selector, part.width, part.height) for part in candidates
         if counts[part.key] == 1 and type(part.width) is int and type(part.height) is int and min(part.width, part.height) > 0),
        key=lambda option: option.selector,
    ))
    if len(eligible) == 1:
        return eligible[0], options, candidates
    report = "\n".join(f"{part.selector} · {part.width} × {part.height} 像素" for part in candidates)
    if not eligible:
        raise ImagePartSelectionError("GENTL_PART_MISSING", report or "此缓冲区未报告可读取的 PFNC 二维图像部件。", options)
    raise ImagePartSelectionError("GENTL_PART_AMBIGUOUS", "部件编号（源/横向偏移/纵向偏移/格式）\n" + report, options)


def part_pixels(data, selected: ImagePart, filled: int | None, byte_order):
    from genicam.gentl import PIXELENDIANNESS_IDS as Endian

    part = selected.part
    width, height = selected.width, selected.height
    offset, length = optional(lambda: part.data_offset), optional(lambda: part.data_size)
    x_padding = optional(lambda: part.x_padding, 0)
    context = f"{selected.selector} ({width} × {height})"
    if (not all(type(v) is int for v in (width, height, offset, length, filled, x_padding))
        or min(width, height, length) <= 0 or min(offset, x_padding) < 0 or not offset + length <= filled <= len(data)):
        raise CaptureError("GENTL_BAD_PAYLOAD", f"invalid image part extent: {context}")

    packed = selected.packed
    storage_bytes = 1 if selected.bits <= 8 else 2
    channels = 1 if selected.color in ("mono", "bayer") else len(selected.color)
    pixel_bits = packed.wire_bits if packed else channels * storage_bytes * 8
    row_bytes = (width * pixel_bits + 7) // 8
    if x_padding:
        # GenTL documents part padding in pixels, while bindings and producers also report bytes.
        # The part extent must imply the same whole-row stride as one documented interpretation.
        stride, remainder = divmod(length, height)
        if remainder or stride not in (row_bytes + x_padding, ((width + x_padding) * pixel_bits + 7) // 8):
            raise CaptureError("GENTL_BAD_PAYLOAD", f"inconsistent image part row padding: {context}")
        pad_bytes = stride - row_bytes
    else:
        stride, pad_bytes = row_bytes, 0

    if packed is not None:
        pixels = packed_samples(packed, data, width=width, height=height, offset=offset, filled=offset + length,
                                padding_x=pad_bytes, padding_y=0, includes_chunks=False)
        return pixels, "pfnc_packed"

    if length != height * stride:
        raise CaptureError("GENTL_BAD_PAYLOAD", f"unexpected unpacked image part layout: {context}")
    # Keep the Harvesters native default for older producers lacking byte-order metadata.
    dtype = np.dtype(np.uint8 if storage_bytes == 1 else np.uint16)
    order = {Endian.PIXELENDIANNESS_LITTLE: ("<", "little"), Endian.PIXELENDIANNESS_BIG: (">", "big")}
    marker, basis = order.get(byte_order, ("=", "harvesters_native_default")) if storage_bytes == 2 else ("=", "one_byte")
    rows = np.ndarray((height, width * channels), dtype=dtype.newbyteorder(marker), buffer=data,
                      offset=offset, strides=(stride, storage_bytes))
    shape = (height, width) if channels == 1 else (height, width, channels)
    return rows.astype(dtype, copy=True, order="C").reshape(shape), basis
