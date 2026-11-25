import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader, inspect_plan
from straintrace_capture.cameras._gentl_multipart import IMAGE_PART, SAVE_ALL_2D, parse_image_part
from straintrace_capture.cameras.base import Device, Feature, TriggerSupport
from straintrace_capture.cameras.gentl import GenTLCamera
from straintrace_capture.image_inputs import decode_samples
from straintrace_capture.preview import PART_REQUIRED, PreviewSession
from straintrace_capture.readiness import probe_camera_setup
from straintrace_capture.storage import frame_image_records
from straintrace_capture.trial import prepare_trial, trial_report


DEVICE = Device("gentl:parts", "gentl", "Fixture", "Multipart camera", "parts", "fixture")
MONO8, MONO12P, MONO16 = 0x01080001, 0x010C0047, 0x01100007


@pytest.mark.parametrize("value", [3, "", "-1", "2/3/Mono8", "18446744073709551616"])
def test_image_part_selector_rejects_invalid_settings(value):
    with pytest.raises(CaptureError) as error:
        parse_image_part(value)
    assert error.value.code == "GENTL_PART_SELECTOR_INVALID"


def test_image_part_selector_allows_zero_and_unreported_offsets():
    assert parse_image_part("auto") is None
    assert parse_image_part("0") == 0
    assert parse_image_part("2/0/-/Mono8") == (2, 0, None, "Mono8")


@pytest.fixture
def sdk():
    api = pytest.importorskip("genicam.gentl")
    pytest.importorskip("harvesters.core")
    return api


class Nodes:
    def __init__(self, pixel_format):
        self.pixel_format = pixel_format

    def public_features(self):
        return {"width": Feature("width", 6), "height": Feature("height", 5),
                "pixel_format": Feature("pixel_format", self.pixel_format)}

    def get(self, name):
        return self.pixel_format if name == "PixelFormat" else None

    def feature(self, name):
        return self.public_features().get(name)

    def configure(self, settings):
        assert IMAGE_PART not in settings and SAVE_ALL_2D not in settings

    def trigger_setup(self, trigger, source):
        assert trigger == "free" and source is None

    def trigger_support(self):
        return TriggerSupport(False, False, "fixture")


class Stream:
    def __init__(self, frames):
        self.frames = frames
        self.reads = self.returned = 0
        self.outstanding = None

    def start(self):
        pass

    def stop(self):
        assert self.outstanding is None

    def try_fetch(self, *, timeout, is_raw):
        assert is_raw and self.outstanding is None
        fields = self.frames[min(self.reads, len(self.frames) - 1)]
        self.reads += 1
        self.outstanding = SimpleNamespace(**fields, frame_id=self.reads, parent=self)
        time.sleep(.002)
        return self.outstanding

    def queue_buffer(self, buffer):
        assert buffer is self.outstanding
        self.outstanding = None
        self.returned += 1


def payload(sdk, parts, *, endian=None):
    raw = bytearray(b"prefix")
    descriptors = []
    for data, source, x, fmt, kind, width, height, pad in parts:
        descriptors.append(SimpleNamespace(
            data_type=kind, data_format=fmt, data_format_namespace=sdk.PIXELFORMAT_NAMESPACE_IDS.PIXELFORMAT_NAMESPACE_PFNC_32BIT,
            data_offset=len(raw), data_size=len(data), source_id=source, x_offset=x, y_offset=0,
            width=width, height=height + 1, delivered_image_height=height, x_padding=pad,
        ))
        raw.extend(data + b"gap")
    return {"parts": descriptors, "raw_buffer": bytes(raw), "size_filled": len(raw), "data_size": len(raw),
            "payload_type": sdk.PAYLOADTYPE_INFO_IDS.PAYLOAD_TYPE_MULTI_PART, "pixel_endianness": endian,
            "timestamp_ns": 1000, "is_complete": lambda: True}


def camera_registry(monkeypatch, frames, *, pixel_format="Mono8"):
    streams = []

    def opened(camera):
        camera.device, camera._nodes, camera._acquirer = DEVICE, Nodes(pixel_format), Stream(frames)
        streams.append(camera._acquirer)

    monkeypatch.setattr(GenTLCamera, "open", opened)
    registry = CameraRegistry(custom={"gentl": (lambda: [DEVICE], lambda identity: GenTLCamera(
        identity, SimpleNamespace(close=lambda _identity: None)))})
    return registry, streams


def capture_plan(part=None, *, phases=None, save_all=False):
    settings = {"width": 6, "height": 5, "pixel_format": "Mono8"}
    if part is not None:
        settings[IMAGE_PART] = part
    if save_all:
        settings[SAVE_ALL_2D] = True
    return CaptureConfig.from_dict({"config_version": 1, "name": "GenTL multipart fixture",
                                    "cameras": [{"id": DEVICE.id, "settings": settings}],
                                    "phases": phases or [{"mode": "continuous", "frame_count": 2}]})


def test_auto_uses_2d_part_and_keeps_its_identity_across_phases(tmp_path, monkeypatch, sdk):
    kinds = sdk.PARTDATATYPE_IDS
    expected = np.arange(12, dtype=np.uint8).reshape(3, 4)
    image = (expected.tobytes(), 0, 2, MONO8, kinds.PART_DATATYPE_2D_IMAGE, 4, 3, 0)
    confidence = (bytes([239] * 12), 0, 2, MONO8, kinds.PART_DATATYPE_CONFIDENCE_MAP, 4, 3, 0)
    configured = CaptureConfig.from_dict({
        "config_version": 1, "name": "Multipart auto selection", "cameras": [{"id": DEVICE.id}],
        "phases": [{"mode": "continuous", "role": "reference", "frame_count": 1},
                   {"mode": "continuous", "frame_count": 1}],
    })
    registry, streams = camera_registry(monkeypatch, [payload(sdk, (confidence, image)), payload(sdk, (image, confidence))],
                                        pixel_format="Coord3D_ABC16")
    session = CaptureSession(configured, tmp_path / "auto", registry=registry)
    session.start()
    assert session.wait(8)["state"] == "completed"
    assert streams[-1].reads == streams[-1].returned == 2
    with ExperimentReader(session.output) as reader:
        assert reader.verify()["complete"]
        rows = reader.frames()
        assert [row["phase"] for row in rows] == [1, 2]
        assert [row["backend_metadata"][IMAGE_PART]["part_index"] for row in rows] == [1, 0]
        for row in rows:
            assert row["backend_metadata"][IMAGE_PART]["selector"] == "0/2/0/Mono8"
            np.testing.assert_array_equal(decode_samples(reader.image_path(row)), expected, strict=True)


def test_explicit_selection_is_deferred_until_preview_and_keeps_actual_roi(tmp_path, monkeypatch, sdk):
    kind = sdk.PARTDATATYPE_IDS.PART_DATATYPE_2D_IMAGE
    a = (bytes([1] * 12), 2, 0, MONO8, kind, 4, 3, 0)
    b = (bytes([7] * 15), 2, 4, MONO8, kind, 5, 3, 0)
    registry, streams = camera_registry(monkeypatch, [payload(sdk, (a, b))])
    plan = capture_plan("2/0/0/Mono8")
    _, checked = probe_camera_setup(plan, registry)
    assert IMAGE_PART in checked[DEVICE.id].deferred
    assert inspect_plan(plan, tmp_path / "estimate")["phases"][0]["raw_payload_bytes"] is None
    preview = PreviewSession(plan.cameras, registry)
    preview.start()
    try:
        deadline = time.monotonic() + 3
        while preview.latest_image(DEVICE.id) is None:
            assert time.monotonic() < deadline and preview.status()[DEVICE.id]["state"] != "failed", preview.status()
            time.sleep(.006)
        changed = {**plan.cameras[0].settings, IMAGE_PART: "2/4/0/Mono8"}
        configured, frame = preview.reconfigure(DEVICE.id, changed).result(timeout=6)
        assert configured.settings[IMAGE_PART] == "2/4/0/Mono8" and frame.pixels.shape == (3, 5)
        assert np.all(frame.pixels == 7) and preview.status()[DEVICE.id]["features"]["offset_x"] == 4
        report = inspect_plan(capture_plan(changed[IMAGE_PART]), tmp_path / "recheck", preview_images={DEVICE.id: frame})
        assert report["phases"][0]["raw_payload_bytes"] == 30
        with pytest.raises(CaptureError) as error:
            preview.reconfigure(DEVICE.id, {name: value for name, value in changed.items() if name != IMAGE_PART}).result(6)
        assert error.value.code == "PREVIEW_ADJUST_RESTORED"
        assert preview.latest_image(DEVICE.id).metadata[IMAGE_PART]["selector"] == "2/4/0/Mono8"
    finally:
        preview.stop()
        assert preview.wait(5)
    assert all(stream.reads == stream.returned for stream in streams)


def wait_for_part_selection(preview):
    deadline = time.monotonic() + 3
    while True:
        status = preview.status()[DEVICE.id]
        if status["state"] == PART_REQUIRED:
            return status["image_parts"]
        assert time.monotonic() < deadline and status["state"] != "failed", status
        time.sleep(.005)


def test_preview_waits_for_choice_without_reopening_and_records_it(tmp_path, monkeypatch, sdk):
    kind = sdk.PARTDATATYPE_IDS.PART_DATATYPE_2D_IMAGE
    a = (bytes([3] * 12), 2, 0, MONO8, kind, 4, 3, 0)
    b = (bytes([7] * 15), 2, 4, MONO8, kind, 5, 3, 0)
    confidence = (bytes(12), 9, 0, MONO8, sdk.PARTDATATYPE_IDS.PART_DATATYPE_CONFIDENCE_MAP, 4, 3, 0)
    registry, streams = camera_registry(monkeypatch, [payload(sdk, (b, confidence, a))])
    plan = capture_plan()
    preview = PreviewSession(plan.cameras, registry)
    preview.start()
    try:
        choices = wait_for_part_selection(preview)
        assert [(part.selector, part.width, part.height) for part in choices] == [
            ("2/0/0/Mono8", 4, 3), ("2/4/0/Mono8", 5, 3),
        ]
        assert preview.latest_image(DEVICE.id) is None and len(streams) == 1
        assert not preview.wait(.03) and streams[0].reads == streams[0].returned == 1
        with pytest.raises(CaptureError) as invalid:
            preview.reconfigure(DEVICE.id, {**plan.cameras[0].settings, IMAGE_PART: "9/0/0/Mono8"})
        assert invalid.value.code == "PREVIEW_PART_REQUIRED"
        selected = choices[1].selector
        config, image = preview.reconfigure(DEVICE.id, {**plan.cameras[0].settings, IMAGE_PART: selected}).result(5)
        assert image.pixels.shape == (3, 5) and np.all(image.pixels == 7)
        assert len(streams) == 1 and preview.status()[DEVICE.id]["image_part"] == selected
        assert preview.cameras == (config,) and config.settings[IMAGE_PART] == selected
    finally:
        preview.stop()
        assert preview.wait(5)
    session = CaptureSession(replace(plan, cameras=preview.cameras), tmp_path / "picked", registry=registry)
    session.start()
    assert session.wait(8)["state"] == "completed"
    with ExperimentReader(session.output) as reader:
        assert reader.verify()["complete"]
        for row in reader.frames():
            assert row["backend_metadata"][IMAGE_PART]["selector"] == selected
            np.testing.assert_array_equal(decode_samples(reader.image_path(row)), image.pixels, strict=True)
    assert all(stream.reads == stream.returned for stream in streams)


def test_failed_first_selection_refreshes_choices_and_leaves_plan_unchanged(monkeypatch, sdk):
    kind = sdk.PARTDATATYPE_IDS.PART_DATATYPE_2D_IMAGE
    a = (bytes([3] * 12), 1, 0, MONO8, kind, 4, 3, 0)
    b = (bytes([7] * 12), 2, 0, MONO8, kind, 4, 3, 0)
    registry, streams = camera_registry(monkeypatch, [payload(sdk, (a, b)), payload(sdk, (a,))])
    plan = capture_plan()
    preview = PreviewSession(plan.cameras, registry)
    preview.start()
    try:
        choices = wait_for_part_selection(preview)
        with pytest.raises(CaptureError) as rejection:
            preview.reconfigure(DEVICE.id, {**plan.cameras[0].settings, IMAGE_PART: choices[1].selector}).result(5)
        assert rejection.value.code == "PREVIEW_PART_ADJUST_RESTORED"
        assert preview.cameras == plan.cameras and preview.latest_image(DEVICE.id) is None
        assert wait_for_part_selection(preview) == (choices[0],)
        _, image = preview.reconfigure(DEVICE.id, {**plan.cameras[0].settings, IMAGE_PART: choices[0].selector}).result(5)
        assert np.all(image.pixels == 3) and len(streams) == 1
    finally:
        preview.stop()
        assert preview.wait(5)
    assert streams[0].reads == streams[0].returned


def test_preview_requires_confirmation_if_selected_geometry_changes(monkeypatch, sdk):
    kind = sdk.PARTDATATYPE_IDS.PART_DATATYPE_2D_IMAGE
    old = (bytes(12), 2, 0, MONO8, kind, 4, 3, 0)
    new = (bytes([8] * 15), 2, 0, MONO8, kind, 5, 3, 0)
    registry, streams = camera_registry(monkeypatch, [payload(sdk, (old,)), payload(sdk, (new,))])
    plan = capture_plan("2/0/0/Mono8")
    preview = PreviewSession(plan.cameras, registry)
    preview.start()
    try:
        choices = wait_for_part_selection(preview)
        assert len(choices) == 1 and choices[0].width == 5
        assert preview.latest_image(DEVICE.id) is None and preview.cameras == plan.cameras
        config_settings = plan.cameras[0].settings
        config, image = preview.reconfigure(DEVICE.id, config_settings).result(5)
        assert config.settings == config_settings
        assert image.pixels.shape == (3, 5) and np.all(image.pixels == 8)
    finally:
        preview.stop()
        assert preview.wait(5)
    assert streams[0].reads == streams[0].returned


def test_disconnect_releases_camera_waiting_for_a_part(monkeypatch, sdk):
    kind = sdk.PARTDATATYPE_IDS.PART_DATATYPE_2D_IMAGE
    a = (bytes(12), 1, 0, MONO8, kind, 4, 3, 0)
    b = (bytes(12), 2, 0, MONO8, kind, 4, 3, 0)
    registry, streams = camera_registry(monkeypatch, [payload(sdk, (a, b))])
    preview = PreviewSession(capture_plan().cameras, registry)
    preview.start()
    try:
        assert len(wait_for_part_selection(preview)) == 2
    finally:
        preview.stop()
        assert preview.wait(3)
    assert preview.status()[DEVICE.id]["state"] == "disconnected"
    assert streams[0].reads == streams[0].returned == 1


def test_ambiguous_or_replaced_images_fail_without_silently_switching(monkeypatch, sdk):
    kind = sdk.PARTDATATYPE_IDS.PART_DATATYPE_2D_IMAGE
    a = (bytes(12), 0, 0, MONO8, kind, 4, 3, 0)
    b = (bytes(12), 1, 0, MONO8, kind, 4, 3, 0)
    for frames, expected in (((payload(sdk, (a, b)),), "GENTL_PART_AMBIGUOUS"),
                             ((payload(sdk, (a,)), payload(sdk, (b,))), "GENTL_PART_MISSING")):
        registry, streams = camera_registry(monkeypatch, frames)
        with registry.camera(DEVICE.id) as camera:
            camera.start()
            if len(frames) == 2:
                assert camera.read(.2).metadata[IMAGE_PART]["source_id"] == 0
            with pytest.raises(CaptureError) as error:
                camera.read(.2)
            assert error.value.code == expected
            assert streams[-1].reads == streams[-1].returned


@pytest.mark.parametrize("fmt", [MONO12P, MONO16])
def test_packed_and_big_endian_parts_use_their_selected_extent(monkeypatch, sdk, fmt):
    kind = sdk.PARTDATATYPE_IDS.PART_DATATYPE_2D_IMAGE
    values = np.array([[0, 4095, 0x123], [0, 4095, 0x123]], dtype=np.uint16)
    if fmt == MONO12P:
        body = (bytes((0, 0xF0, 0xFF, 0x23, 1, 0xAA))) * 2
    else:
        body = b"".join(row.astype(">u2").tobytes() + b"\xaa\xaa" for row in values)
    fields = payload(sdk, ((body, 3, 0, fmt, kind, 3, 2, 1),), endian=sdk.PIXELENDIANNESS_IDS.PIXELENDIANNESS_BIG)
    registry, streams = camera_registry(monkeypatch, [fields])
    with registry.camera(DEVICE.id) as camera:
        camera.start()
        frame = camera.read(.2)
        np.testing.assert_array_equal(frame.pixels, values, strict=True)
        assert streams[-1].returned == 1


def test_duplicate_selection_and_truncated_part_return_the_buffer(monkeypatch, sdk):
    kind = sdk.PARTDATATYPE_IDS.PART_DATATYPE_2D_IMAGE
    chosen = (bytes(12), 2, 0, MONO8, kind, 4, 3, 0)
    ambiguous = payload(sdk, (chosen, chosen))
    truncated = payload(sdk, (chosen,))
    truncated["parts"][0].data_size = len(truncated["raw_buffer"])
    for fields, expected in ((ambiguous, "GENTL_PART_AMBIGUOUS"), (truncated, "GENTL_BAD_PAYLOAD")):
        registry, streams = camera_registry(monkeypatch, [fields])
        with registry.camera(DEVICE.id) as camera:
            camera.set_feature(IMAGE_PART, "2/0/0/Mono8")
            camera.start()
            with pytest.raises(CaptureError) as error:
                camera.read(.2)
            assert error.value.code == expected and streams[-1].returned == 1


@pytest.mark.parametrize("durability", ["buffered", "write_through"])
def test_records_all_requested_parts_and_estimates_decoded_bytes(tmp_path, monkeypatch, sdk, durability):
    kind = sdk.PARTDATATYPE_IDS.PART_DATATYPE_2D_IMAGE
    mono8 = np.arange(12, dtype=np.uint8).reshape(3, 4)
    mono12 = np.array([[0, 4095, 0x123], [0, 4095, 0x123]], dtype=np.uint16)
    primary = (mono8.tobytes(), 2, 4, MONO8, kind, 4, 3, 0)
    additional = (bytes((0, 0xF0, 0xFF, 0x23, 1, 0xAA)) * 2, 6, 0, MONO12P, kind, 3, 2, 1)
    confidence = (bytes(12), 8, 0, MONO8, sdk.PARTDATATYPE_IDS.PART_DATATYPE_CONFIDENCE_MAP, 4, 3, 0)
    registry, streams = camera_registry(monkeypatch, [payload(sdk, (primary, additional, confidence)),
                                                    payload(sdk, (confidence, additional, primary))])
    primary_selector = "2/4/0/Mono8"
    plan = capture_plan(primary_selector)
    preview = PreviewSession(plan.cameras, registry)
    preview.start()
    try:
        deadline = time.monotonic() + 3
        while preview.latest_image(DEVICE.id) is None:
            assert time.monotonic() < deadline and preview.status()[DEVICE.id]["state"] != "failed", preview.status()
            time.sleep(.006)
        assert not preview.latest_image(DEVICE.id).auxiliary_images
        selected, frame = preview.reconfigure(DEVICE.id, {**plan.cameras[0].settings, SAVE_ALL_2D: True}).result(5)
        assert frame.raw_bytes == mono8.nbytes + mono12.nbytes
        assert frame.auxiliary_images[0].native_frame_id == frame.native_frame_id
        assert preview.status()[DEVICE.id]["archived_parts"] == 2
        _, frame = preview.reconfigure(DEVICE.id, plan.cameras[0].settings).result(5)
        assert frame.raw_bytes == mono8.nbytes and not frame.auxiliary_images
    finally:
        preview.stop()
        assert preview.wait(5)

    plan = replace(plan, cameras=(selected,), durability=durability, segment_frames=1)
    _, settings = probe_camera_setup(plan, registry)
    assert SAVE_ALL_2D in settings[DEVICE.id].deferred
    plan_report = inspect_plan(plan, tmp_path / "planned")
    assert plan_report["raw_payload_bytes"] is None
    trial, output = prepare_trial(plan, tmp_path / "planned")
    session = CaptureSession(trial, output, registry=registry)
    session.start()
    status = session.wait(8)
    assert status["state"] == "completed"
    assert all(stream.reads == stream.returned for stream in streams)
    with ExperimentReader(output) as reader:
        verification = reader.verify()
        assert verification["complete"] and verification["committed_frames"] == 2 and verification["committed_images"] == 4
        rows = reader.frames()
        assert [row["backend_metadata"][IMAGE_PART]["part_index"] for row in rows] == [0, 2]
        for row in rows:
            first, second = frame_image_records(row)
            assert [part["backend_metadata"][IMAGE_PART]["selector"] for part in (first, second)] == [primary_selector, "6/0/0/Mono12p"]
            assert reader.image_path(second).name.endswith("-part-02.tiff")
            np.testing.assert_array_equal(decode_samples(reader.image_path(first)), mono8, strict=True)
            np.testing.assert_array_equal(decode_samples(reader.image_path(second)), mono12, strict=True)
        assert status["cameras"][DEVICE.id]["saved_bytes"] == sum(
            part["file_bytes"] for row in rows for part in frame_image_records(row))
        path = reader.image_path(second)
        damaged = bytearray(path.read_bytes())
    report = trial_report(output)
    assert report["cameras"][DEVICE.id]["geometry"]["frame_bytes"] == mono8.nbytes + mono12.nbytes
    cached = inspect_plan(plan, tmp_path / "planned", trial_reports=(report,))
    assert cached["raw_payload_bytes"] == 2 * (mono8.nbytes + mono12.nbytes)
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    with ExperimentReader(output) as reader:
        assert reader.verify()["issues"] == [{"code": "FILE_HASH_MISMATCH", "path": second["path"]}]


@pytest.mark.parametrize("failure", ["budget", "missing", "duplicate", "truncated"])
def test_multipart_save_does_not_commit_incomplete_frames(tmp_path, monkeypatch, sdk, failure):
    kind = sdk.PARTDATATYPE_IDS.PART_DATATYPE_2D_IMAGE
    primary = (bytes(12), 1, 0, MONO8, kind, 4, 3, 0)
    secondary = (bytes(12), 2, 0, MONO8, kind, 4, 3, 0)
    valid = payload(sdk, (primary, secondary))
    bad = valid
    if failure == "missing":
        bad = payload(sdk, (primary,))
    elif failure == "duplicate":
        bad = payload(sdk, (primary, secondary, secondary))
    elif failure == "truncated":
        bad = payload(sdk, (primary, secondary))
        bad["parts"][1].data_size += 1
    registry, streams = camera_registry(monkeypatch, [valid, bad])
    plan = capture_plan("1/0/0/Mono8", save_all=True)
    if failure == "budget":
        plan = replace(plan, max_buffer_bytes=20)
    session = CaptureSession(plan, tmp_path / "experiment", registry=registry)
    session.start()
    status = session.wait(8)
    expected = {"budget": "RAW_BUFFER_FULL", "missing": "GENTL_PART_SET_CHANGED",
                "duplicate": "GENTL_PART_SET_AMBIGUOUS", "truncated": "GENTL_BAD_PAYLOAD"}
    assert status["state"] == "failed" and status["error"]["code"] == expected[failure]
    assert streams[-1].reads == streams[-1].returned
    with ExperimentReader(session.output) as reader:
        assert reader.verify()["committed_images"] == (0 if failure == "budget" else 2)
