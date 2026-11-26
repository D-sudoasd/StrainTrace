import numpy as np
import pytest

from straintrace_capture import CaptureError
from straintrace_capture.cameras._gentl_packed import PackedFormat, packed_samples


def pack(samples, bits):
    positions = np.arange(bits, dtype=np.uint16)
    return np.packbits(((samples.reshape(-1, 1) >> positions) & 1).ravel(), bitorder="little").tobytes()


def layout_for(bits):
    weights = 1 << np.arange(bits, dtype=np.uint16)

    def expand(source):
        # This decoder only accepts whole groups, like the producer-independent PFNC decoder.
        return np.unpackbits(source, bitorder="little").reshape(-1, bits) @ weights

    return PackedFormat(f"Mono{bits}p", 0, bits, 4 if bits == 10 else 2, expand)


@pytest.mark.parametrize(("line_packed", "pad_x", "has_chunks"), [
    (False, 0, False), (True, 0, False), (True, 3, True),
])
def test_odd_width_groups_use_only_their_reported_pixels(line_packed, pad_x, has_chunks):
    samples = np.arange(1, 22, dtype=np.uint16).reshape(3, 7) * 191
    packed = (b"".join(pack(row, 12) + b"\xaa" * pad_x for row in samples)
              if line_packed else pack(samples, 12))
    prefix, padding_y = b"prefix", 2
    body = prefix + packed + b"\xbb" * padding_y + (b"chunk data" if has_chunks else b"")
    decoded = packed_samples(layout_for(12), body + b"unused capacity", width=7, height=3,
                             offset=len(prefix), filled=len(body), padding_x=pad_x, padding_y=padding_y,
                             includes_chunks=has_chunks)
    np.testing.assert_array_equal(decoded, samples, strict=True)
    assert decoded.flags.c_contiguous


def test_10_bit_line_and_image_layouts_can_have_the_same_byte_count():
    body = pack(np.arange(6, dtype=np.uint16), 10)
    assert len(body) == 8
    with pytest.raises(CaptureError) as error:
        packed_samples(layout_for(10), body, width=3, height=2, offset=0, filled=len(body),
                       padding_x=0, padding_y=0, includes_chunks=False)
    assert error.value.code == "GENTL_PACKING_AMBIGUOUS"


def test_chunk_bytes_cannot_be_used_to_infer_a_packing_boundary():
    # This is also the byte count of two independently packed rows of three samples.
    frame_with_chunk = pack(np.arange(6, dtype=np.uint16), 12) + b"\x00"
    with pytest.raises(CaptureError) as error:
        packed_samples(layout_for(12), frame_with_chunk, width=3, height=2, offset=0, filled=len(frame_with_chunk),
                       padding_x=0, padding_y=0, includes_chunks=True)
    assert error.value.code == "GENTL_PACKING_AMBIGUOUS"


def test_allocated_but_unfilled_bytes_do_not_complete_a_truncated_row():
    body = bytes(16)
    with pytest.raises(CaptureError) as error:
        packed_samples(layout_for(12), body, width=3, height=2, offset=0, filled=11,
                       padding_x=1, padding_y=0, includes_chunks=False)
    assert error.value.code == "GENTL_BAD_PAYLOAD"
