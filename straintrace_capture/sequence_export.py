"""Export byte-identical TIFFs for image-based analysis outside Capture."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile, TemporaryDirectory

from .calibration import calibration_geometry, phase_sensor_geometry
from .config import CaptureError
from .event_images import event_scope, event_sequence, select_event_reference, validate_event_id
from .frame_search import format_receipt_utc
from .handoff import select_camera, select_reference
from .image_inputs import frame_part_selector, require_part_geometry, select_frame_image, validate_analysis_part
from .storage import _flush_directory
from .transfer import (
    TransferProgress, _DIGEST, _check_digest, _file, _layout, _ledger_asset, _node, _publish_folder,
    _read_bytes, _read_object, _source_reader, _write_json,
)

SEQUENCE_MANIFEST = "capture-sequence.json"
SEQUENCE_INDEX = "frames.csv"
_SCHEMA = "straintrace.capture-image-sequence/v1"
_FIELDS = (
    "file", "index", "sequence_role", "source_role", "source_sequence", "source_phase", "frame_id",
    "source_path", "trigger_event_id", "received_utc_ns", "received_utc_iso", "received_monotonic_ns",
    "clock_readings_json", "bytes", "sha256",
)
_EVENT_FIELDS = ("event_context_id", "event_position")


@dataclass(frozen=True, slots=True)
class SequenceSelection:
    camera_id: str | None = None
    phase: int | None = None
    reference_sequence: int | None = None
    first_sequence: int | None = None
    last_sequence: int | None = None
    image_part: str | None = None
    event_context_id: str | None = None

    def __post_init__(self):
        for value in (self.phase, self.reference_sequence, self.first_sequence, self.last_sequence):
            if value is not None and (type(value) is not int or not 1 <= value < (1 << 63)):
                raise CaptureError("INVALID_FRAME_QUERY", "阶段和帧序号必须为有效的正整数。")
        if self.first_sequence is not None and self.last_sequence is not None and self.first_sequence > self.last_sequence:
            raise CaptureError("INVALID_FRAME_QUERY", "起始帧序号大于结束帧序号。")
        validate_analysis_part(self.image_part)
        if self.event_context_id is not None:
            validate_event_id(self.event_context_id)


def _image_name(index):
    return f"images/frame_{index:012d}.tif"


def _check_cancel(cancelled):
    if cancelled and cancelled():
        raise CaptureError("SEQUENCE_CANCELLED", "image sequence export cancelled")


def _frames(reader, reference, selection, abort, event=None):
    yield reference
    camera = reference["camera_id"]
    cursor = max(reference["sequence"], (selection.first_sequence or 1) - 1)
    if event is not None:
        for frame in event_sequence(reader, camera, event, after=cursor, through=selection.last_sequence):
            abort()
            yield select_frame_image(frame, selection.image_part)
        return
    while batch := reader.test_sequence_page(camera, after=cursor, through=selection.last_sequence,
                                             phase=selection.phase):
        for frame in batch:
            abort()
            yield select_frame_image(frame, selection.image_part)
        cursor = batch[-1]["sequence"]


def _scope(reader, experiment, source, destination, selection, abort):
    camera = select_camera(reader, selection.camera_id)
    if selection.phase is not None:
        phases = experiment["config"]["phases"]
        if selection.phase > len(phases) or phases[selection.phase - 1]["role"] != "test":
            raise CaptureError("SEQUENCE_PHASE", f"第 {selection.phase} 阶段的用途不是实验图。")
    event = None
    if selection.event_context_id is not None:
        event = event_scope(reader, experiment, selection.event_context_id)
        if selection.phase is not None and selection.phase != event["phase"]:
            raise CaptureError("EVENT_IMAGES_SCOPE", "所选阶段与事件所属阶段不同。")
        reference, reference_basis = select_event_reference(reader, camera, event, selection.reference_sequence)
    else:
        reference = select_reference(reader, camera, selection.reference_sequence)
        if reference["role"] == "test" and selection.reference_sequence is None:
            reference = reader.test_sequence_page(camera, after=0, limit=1)[0]
    reference = select_frame_image(reference, selection.image_part)
    sensor = phase_sensor_geometry(reader, camera)
    geometry_by_phase = {}
    known = {}
    count = image_bytes = 0
    for frame in _frames(reader, reference, selection, abort, event):
        abort()
        require_part_geometry(reference, frame)
        phase = frame["phase"]
        if str(phase) not in geometry_by_phase:
            geometry = calibration_geometry(frame["shape"], sensor.get(str(phase), {}), frame["backend_metadata"])
            for feature, value in geometry["sensor_geometry"].items():
                previous_value, previous_phase = known.setdefault(feature, (value, phase))
                if previous_value != value:
                    raise CaptureError("SEQUENCE_SENSOR_CHANGED",
                                       f"阶段 {previous_phase} 的 {feature}={previous_value}，阶段 {phase} 为 {value}。")
            geometry_by_phase[str(phase)] = geometry["sensor_geometry"]
        asset = _ledger_asset(source, frame["path"], "image", frame["sha256"], frame["file_bytes"])
        count += 1
        image_bytes += asset.size
    if count < 2:
        raise CaptureError("SEQUENCE_EMPTY", f"no matching test frames after source sequence {reference['sequence']}")
    result = {
        "schema": _SCHEMA, "experiment_id": experiment["experiment_id"], "experiment_state": experiment["state"],
        "camera_id": camera, "frames": count, "test_frames": count - 1, "image_bytes": image_bytes,
        "reference_sequence": reference["sequence"], "reference_frame_id": reference["frame_id"],
        "reference_phase": reference["phase"], "reference_source_role": reference["role"],
        "filter": {"phase": event["phase"] if event is not None else selection.phase, "first_sequence": selection.first_sequence,
                   "last_sequence": selection.last_sequence},
        "sample_layout": {key: reference[key] for key in ("shape", "dtype", "pixel_format", "color_model", "significant_bits")},
        "image_part": frame_part_selector(reference), "sensor_geometry_by_phase": geometry_by_phase,
        "included_phases": sorted(map(int, geometry_by_phase)),
    }
    if event is not None:
        result.update(event=event, reference_selection=reference_basis)
    return reference, result, shutil.disk_usage(destination.parent).free


def inspect_sequence(experiment, output, selection=SequenceSelection(), *, expected_id=None, cancelled=None):
    """Check the selected source images and estimate destination space without writing."""
    source, destination, _ = _layout(experiment, output)
    with _source_reader(source, expected_id) as (reader, record, check_source):
        _, scope, available = _scope(reader, record, source, destination, selection, lambda: _check_cancel(cancelled))
        check_source()
    return {**scope, "output": str(destination), "available_bytes": available}


def _index_row(frame, index, size, digest):
    stamps = frame["timestamps"]
    utc = stamps["received_utc"]["value"]
    human_time = format_receipt_utc(utc)
    return (
        _image_name(index), index, "test" if index else "reference", frame["role"], frame["sequence"],
        frame["phase"], frame["frame_id"], frame["path"], frame["trigger_event_id"],
        utc, human_time.replace(" ", "T") + "Z" if human_time != "—" else None,
        stamps["received_monotonic"]["value"], json.dumps(stamps, ensure_ascii=False, separators=(",", ":")),
        size, digest,
    )


def _copy_images(stage, source, frames, scope, abort, report):
    (stage / "images").mkdir()
    index_path = stage / SEQUENCE_INDEX
    done = 0
    with index_path.open("x", encoding="utf-8-sig", newline="") as stream:
        index_csv = csv.writer(stream)
        event = scope.get("event")
        index_csv.writerow((*_FIELDS, *_EVENT_FIELDS) if event is not None else _FIELDS)
        for index, frame in enumerate(frames):
            abort()
            asset = _ledger_asset(source, frame["path"], "image", frame["sha256"], frame["file_bytes"])
            original, _ = _file(source, asset.path)
            name = _image_name(index)

            def advance(copied):
                report(TransferProgress("copying", name, done + copied, scope["image_bytes"], index, scope["frames"]))

            with (stage / name).open("xb") as sink:
                digest = _read_bytes(original, asset.size, abort, sink=sink, advance=advance)
                _check_digest(digest, asset.expected, asset.path)
                sink.flush()
                os.fsync(sink.fileno())
            done += asset.size
            row = _index_row(frame, index, asset.size, digest)
            if event is not None:
                row += tuple(frame.get(key) for key in _EVENT_FIELDS)
            index_csv.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    _flush_directory(stage / "images")
    size = index_path.stat().st_size
    return {"file": SEQUENCE_INDEX, "bytes": size, "sha256": _read_bytes(index_path, size, abort)}


def export_sequence(experiment, output, selection=SequenceSelection(), *, expected_id=None, cancelled=None, progress=None):
    """Publish a checked standalone sequence under a fresh destination name."""
    def abort():
        _check_cancel(cancelled)

    def report(update):
        if progress:
            progress(update)

    abort()
    report(TransferProgress("preparing"))
    source, destination, _ = _layout(experiment, output)
    with _source_reader(source, expected_id) as (reader, record, check_source):
        reference, scope, available = _scope(reader, record, source, destination, selection, abort)
        if available < scope["image_bytes"]:
            raise CaptureError("TRANSFER_SPACE", f"{scope['image_bytes']} bytes of images, {available} bytes available")
        with TemporaryDirectory(prefix=".capture-sequence-", dir=destination.parent) as temp:
            stage = Path(temp) / "data"
            stage.mkdir()
            frames = _frames(reader, reference, selection, abort, scope.get("event"))
            manifest = {**scope, "index": _copy_images(stage, source, frames, scope, abort, report)}
            _write_json(stage / (SEQUENCE_MANIFEST + ".part"), stage / SEQUENCE_MANIFEST, manifest)
            _verify(stage, manifest, abort, report)
            report(TransferProgress("publishing", files_done=scope["frames"], files_total=scope["frames"]))
            abort()
            check_source()
            _publish_folder(stage, destination)
    return {**manifest, "output": str(destination), "ok": True}


def _valid_index_info(info):
    return (isinstance(info, dict) and info.get("file") == SEQUENCE_INDEX
            and type(info.get("bytes")) is int and info["bytes"] > 0
            and isinstance(info.get("sha256"), str) and _DIGEST.fullmatch(info["sha256"]))


def _verify_rows(root, manifest, rows, abort, report):
    count = done = 0
    previous_sequence = 0
    for index, row in enumerate(rows):
        abort()
        name = _image_name(index)
        source_sequence = int(row["source_sequence"])
        role = "reference" if index == 0 else "test"
        size = int(row["bytes"])
        if (None in row or None in row.values() or row["file"] != name or row["index"] != str(index)
                or row["sequence_role"] != role or source_sequence <= previous_sequence
                or row["source_role"] not in (("reference", "test") if index == 0 else ("test",))
                or row["frame_id"] != f"{manifest['experiment_id']}/{manifest['camera_id']}/{source_sequence}"
                or size <= 0 or not _DIGEST.fullmatch(row["sha256"])):
            raise CaptureError("SEQUENCE_INDEX", f"invalid image index entry at row {index + 2}")
        if not index and (source_sequence, row["frame_id"]) != (manifest.get("reference_sequence"),
                                                                manifest.get("reference_frame_id")):
            raise CaptureError("SEQUENCE_INDEX", "the reference in the image index differs from the manifest")
        if event := manifest.get("event"):
            context, position = (row[key] for key in _EVENT_FIELDS)
            positions = ("pre", "boundary", "post")
            if index or context == event["signal_id"]:
                if (context != event["signal_id"] or position not in positions
                        or int(row["source_phase"]) != event["phase"]):
                    raise CaptureError("SEQUENCE_INDEX", f"image index row {index + 2} does not match the selected event")
            elif row["source_role"] != "reference" or (context and position not in positions) or (position and not context):
                raise CaptureError("SEQUENCE_INDEX", "a baseline from outside this event must be a recorded reference image")
        path, meta = _file(root, name, independent=True)
        if meta.st_size != size:
            raise CaptureError("TRANSFER_CHANGED", f"size differs from the image index: {name}")

        def advance(checked):
            report(TransferProgress("checking", name, done + checked, manifest["image_bytes"], index,
                                    manifest["frames"]))

        _check_digest(_read_bytes(path, size, abort, advance=advance), row["sha256"], name)
        done += size
        count += 1
        previous_sequence = source_sequence
    return count, done


def _verify(root, manifest, abort, report):
    index = manifest.get("index")
    count, size = manifest.get("frames"), manifest.get("image_bytes")
    if (manifest.get("schema") != _SCHEMA or not _valid_index_info(index)
            or not all(isinstance(manifest.get(key), str) and manifest[key] for key in ("experiment_id", "camera_id"))
            or type(count) is not int or count < 2 or manifest.get("test_frames") != count - 1
            or type(size) is not int or size <= 0):
        raise CaptureError("SEQUENCE_INDEX", "invalid or unsupported image sequence manifest")
    event = manifest.get("event")
    if "event" in manifest and (
            not isinstance(event, dict) or not isinstance(event.get("signal_id"), str) or not event["signal_id"]
            or any(type(event.get(key)) is not int or event[key] < 1 for key in ("phase", "event_number", "signal_utc_ns"))):
        raise CaptureError("SEQUENCE_INDEX", "invalid event scope in the image sequence manifest")
    index_path, meta = _file(root, SEQUENCE_INDEX, independent=True)
    if meta.st_size != index["bytes"]:
        raise CaptureError("TRANSFER_CHANGED", f"size differs from the manifest: {SEQUENCE_INDEX}")
    with SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b") as spool:
        _check_digest(_read_bytes(index_path, index["bytes"], abort, sink=spool), index["sha256"], SEQUENCE_INDEX)
        spool.seek(0)
        with io.TextIOWrapper(spool, encoding="utf-8-sig", newline="") as stream:
            try:
                rows = csv.DictReader(stream, strict=True)
                if rows.fieldnames != list((*_FIELDS, *_EVENT_FIELDS) if event is not None else _FIELDS):
                    raise CaptureError("SEQUENCE_INDEX", "unexpected column names in the image index")
                observed = _verify_rows(root, manifest, rows, abort, report)
            except (csv.Error, ValueError, TypeError) as exc:
                raise CaptureError("SEQUENCE_INDEX", "invalid value in the image index") from exc
    with os.scandir(root / "images") as entries:
        num_files = 0
        for _ in entries:
            abort()
            num_files += 1
    if observed != (count, size) or num_files != count:
        raise CaptureError("SEQUENCE_INDEX", "image count or total byte size differs from the manifest")


def verify_sequence(folder):
    """Verify a received sequence without needing access to its source experiment."""
    root = Path(folder).resolve(strict=True)
    _node(root, directory=True)
    manifest = _read_object(_file(root, SEQUENCE_MANIFEST, independent=True)[0])
    _verify(root, manifest, lambda: None, lambda _update: None)
    return {**manifest, "output": str(root), "ok": True}
