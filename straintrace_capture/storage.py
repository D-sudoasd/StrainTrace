"""Lossless image publication and a versioned, independently readable ledger."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from .cameras.base import HOST_NEXT_RECEIPT, ClockReading, Device, Image
from .config import REPEAT_OF, REPEAT_PARENT, CaptureConfig, CaptureError, inside_repeat_source
from .output import resolve_output_path
from .tiff_samples import TiffPayload, prepare_tiff


EXPERIMENT_SCHEMA = "straintrace.capture/v1"
TRIAL_REFERENCE_SCHEMA = "straintrace.capture.trial-reference/v1"
TERMINAL_STATES = frozenset({"completed", "stopped", "failed", "interrupted"})
DATABASE = "capture.sqlite3"
FRAME_WRITE_JOURNAL = "pending-frames.log"
FRAME_WRITE_SCHEMA = "straintrace.capture.frame-write-journal/v1"
_TRIGGER_EVENTS = "e.kind IN ('software_trigger', 'hardware_trigger')"
_RETAINED_TRIGGER = """(json_extract(e.record, '$.retention') IS NOT 'rolling_prebuffer'
    OR EXISTS (SELECT 1 FROM frames f WHERE f.event_id = e.event_id))"""
_TRIGGER_PROBLEM = """(EXISTS (
    SELECT 1 FROM json_each(e.record, '$.expected_cameras') c
    WHERE (SELECT COUNT(*) FROM frames f WHERE f.event_id = e.event_id AND f.camera_id = c.value) != 1
) OR EXISTS (
    SELECT 1 FROM frames f WHERE f.event_id = e.event_id
    AND NOT EXISTS (SELECT 1 FROM json_each(e.record, '$.expected_cameras') c WHERE c.value = f.camera_id)
))"""
_FRAME_RECEIPT_UTC = "json_extract(record, '$.timestamps.received_utc.value')"
_SQLITE_INTEGER_MAX = (1 << 63) - 1
_FRAME_SEQUENCE_WINDOW = 4096
_FRAME_JOURNAL_CHECKPOINT_ENTRIES = 1024
CAMERA_LOSS_COUNTERS = ("app_unsaved", "native_counter_gaps", "native_counter_duplicates",
                        "hardware_identity_missing", "reported_driver_skips", "transport_errors")

_DDL = """
    CREATE TABLE experiment (singleton INTEGER PRIMARY KEY CHECK(singleton=1), record TEXT NOT NULL);
    CREATE TABLE cameras (camera_id TEXT PRIMARY KEY, record TEXT NOT NULL);
    CREATE TABLE events (ordinal INTEGER PRIMARY KEY, event_id TEXT UNIQUE, kind TEXT NOT NULL, record TEXT NOT NULL);
    CREATE TABLE frames (
        ordinal INTEGER PRIMARY KEY,
        camera_id TEXT NOT NULL REFERENCES cameras(camera_id),
        sequence INTEGER NOT NULL,
        event_id TEXT REFERENCES events(event_id),
        path TEXT NOT NULL UNIQUE,
        record TEXT NOT NULL,
        UNIQUE (camera_id, sequence)
    );
    CREATE INDEX frame_camera_ordinal ON frames(camera_id, ordinal);
    CREATE INDEX frame_event ON frames(event_id);
"""
_EVENT_CONTEXT = "json_extract(record, '$.event_context_id')"
_HAS_EVENT_CONTEXT = "json_type(record, '$.event_context_id')='text'"
_EVENT_LINKS_DDL = """
    CREATE TABLE event_frame_links (
        event_id TEXT NOT NULL REFERENCES events(event_id),
        camera_id TEXT NOT NULL REFERENCES cameras(camera_id),
        sequence INTEGER NOT NULL,
        position TEXT NOT NULL CHECK(position IN ('pre', 'boundary', 'post')),
        PRIMARY KEY (event_id, camera_id, sequence)
    ) WITHOUT ROWID;
"""
_EVENT_FRAMES = """(
    SELECT f.ordinal, l.camera_id, l.sequence, f.event_id, f.record, l.position AS member_position
    FROM event_frame_links AS l JOIN frames AS f USING (camera_id, sequence)
    WHERE l.event_id=? AND l.camera_id=?
)"""


@dataclass(frozen=True)
class TriggerEvent:
    id: str
    kind: str
    details: dict


@dataclass(frozen=True)
class PreparedFrame:
    record: dict
    trigger: TriggerEvent | None


class PartialFrameCommitError(CaptureError):
    def __init__(self, committed_count: int, cause: OSError | CaptureError):
        code = cause.code if isinstance(cause, CaptureError) else "WRITE_FAILED"
        super().__init__(code, str(cause))
        self.committed_count = committed_count


def _trigger_group(event, received, state):
    mapping = defaultdict(list)
    for row in received:
        mapping[row["camera_id"]].append(row["frame_id"])
    expected = set(event["expected_cameras"])
    missing = sorted(expected - set(mapping))
    unexpected = sorted(set(mapping) - expected)
    ambiguous = sorted(camera for camera, frames in mapping.items() if len(frames) != 1)
    if unexpected or ambiguous:
        status = "ambiguous"
    elif missing:
        status = "incomplete" if state in TERMINAL_STATES else "pending_or_incomplete"
    else:
        status = "associated"
    basis, synchronization = event["pairing_basis"], event["synchronization"]
    methods = event.get("camera_methods", {})
    if basis == "common_capture_request" and synchronization == "unverified" and HOST_NEXT_RECEIPT in methods.values():
        synchronization = "host_receipt_only"
    return {
        "event_id": event["event_id"], "phase": event["phase"], "status": status,
        "basis": basis, "synchronization": synchronization,
        "camera_methods": methods,
        "external_event_number": event.get("external_event_number"), "external_source_id": event.get("external_source_id"),
        "missing_cameras": missing, "unexpected_cameras": unexpected, "ambiguous_cameras": ambiguous,
        "frames_by_camera": dict(mapping),
    }


def _review_group(event, received, state):
    return {
        **_trigger_group(event, received, state), "ordinal": event["ordinal"],
        "expected_cameras": event["expected_cameras"], "issued_utc_ns": event.get("issued_utc_ns"),
    }


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def image_review_indices(count: int, largest_file: int, *, all_frames=False):
    """Use the first and last saved frame numbers and spread the sample between them."""
    if all_frames or count <= 2:
        return range(count)
    size = min(count, 96, max(2, (64 * 1024**2) // max(1, largest_file)))
    return tuple(index * (count - 1) // (size - 1) for index in range(size))


def reported_camera_losses(cameras: dict) -> dict:
    losses = {camera: {key: stats[key] for key in CAMERA_LOSS_COUNTERS if stats.get(key)}
              for camera, stats in cameras.items()}
    return {camera: counts for camera, counts in losses.items() if counts}


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class WriterLease:
    """An OS-owned lease. A stale marker file never prevents an explicit recovery."""

    def __init__(self, root: Path, *, create=True):
        self.path = root / ".capture-writer"
        self._create = create
        self._stream = None

    def acquire(self) -> None:
        try:
            stream = self.path.open("a+b" if self._create else "rb")
        except FileNotFoundError:
            if self._create:
                raise
            return
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise CaptureError("WRITER_ACTIVE", f"capture writer lease could not be acquired: {self.path}") from exc
        self._stream = stream

    def release(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> WriterLease:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _connection(root: Path, mode: str, *, database=DATABASE, immutable=False) -> sqlite3.Connection:
    url = (root / database).resolve().as_uri()
    query = f"?mode={mode}" + ("&immutable=1" if immutable else "")
    return sqlite3.connect(url + query, uri=True, timeout=5, check_same_thread=False)


def _clock(value: ClockReading | None) -> dict | None:
    return asdict(value) if value else None


def _midpoint(image: Image) -> dict | None:
    start, end = image.exposure_start, image.exposure_end
    if not start or not end or start.value > end.value:
        return None
    if (start.clock_id, start.unit, start.reference, start.tick_frequency_hz) != (
        end.clock_id, end.unit, end.reference, end.tick_frequency_hz
    ):
        return None
    uncertainty = max(start.uncertainty_ns, end.uncertainty_ns) if (
        start.uncertainty_ns is not None and end.uncertainty_ns is not None
    ) else None
    return _clock(ClockReading(
        (start.value + end.value) // 2, start.unit, "exposure_window_midpoint",
        start.clock_id, start.reference, start.sync_status if start.sync_status == end.sync_status else "unverified",
        uncertainty, start.tick_frequency_hz, "derived",
    ))


def _flush_directory(directory: Path) -> None:
    if os.name != "nt":
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

@cache
def _windows_move():
    import ctypes

    move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
    move.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong)
    move.restype = ctypes.c_int
    return move


def _publish_path(partial: Path, target: Path, *, write_through: bool) -> None:
    if os.name == "nt" and write_through:
        import ctypes

        move = _windows_move()
        if not move(str(partial), str(target), 0x8):  # MOVEFILE_WRITE_THROUGH, without replace
            raise ctypes.WinError(ctypes.get_last_error())
    elif os.name == "nt":
        os.rename(partial, target)
    else:
        os.link(partial, target)
        partial.unlink()
        if write_through:
            _flush_directory(target.parent)


def _publish_staged_tiff(partial: Path, target: Path, *, write_through: bool) -> None:
    if write_through:
        with partial.open("r+b", buffering=0) as stream:
            os.fsync(stream.fileno())
    _publish_path(partial, target, write_through=write_through)


def _stage_tiff(target: Path, image: Image, *, before_write: Callable[[int, str], None] | None = None,
                payload: TiffPayload | None = None) -> Path:
    if target.exists():
        raise CaptureError("OUTPUT_EXISTS", f"refusing to replace {target}")
    payload = payload if payload is not None else prepare_tiff(image)
    partial = target.with_suffix(".tiff.part")
    with partial.open("xb") as stream:
        if before_write is not None:
            before_write(payload.size, payload.sha256)
        for chunk in payload.chunks:
            if stream.write(chunk) != len(chunk):
                raise OSError(f"incomplete TIFF write at {partial}")
    return partial


def _frame_target(folder: Path, sequence: int, segment_frames: int) -> Path:
    if segment_frames:
        folder /= f"segment-{(sequence - 1) // segment_frames + 1:07d}"
    return folder / f"{sequence:012d}.tiff"


def _part_target(primary: Path, index: int) -> Path:
    return primary.with_name(f"{primary.stem}-part-{index:02d}.tiff")


def frame_image_records(frame: dict) -> tuple[dict, ...]:
    """Return the primary TIFF descriptor, then any TIFFs from the same receipt."""
    return (frame, *frame.get("auxiliary_images", ()))


def _image_record(path: str, image: Image) -> dict:
    return {"path": path, "shape": list(image.pixels.shape), "dtype": image.pixels.dtype.name,
            "significant_bits": image.significant_bits, "pixel_format": image.pixel_format,
            "color_model": image.color_model, "backend_metadata": image.metadata}


def _insert_event(db, kind, details, *, event_id=None, once=False):
    record = {"kind": kind, "utc_ns": time.time_ns(), "event_id": event_id, **details}
    statement = "INSERT INTO events(event_id, kind, record) VALUES (?, ?, ?)"
    if once:
        statement += " ON CONFLICT(event_id) DO NOTHING"
    db.execute(statement, (event_id, kind, json_text(record)))


def _insert_frames(db, prepared):
    seen_events = set()
    for entry in prepared:
        if entry.trigger and entry.trigger.id not in seen_events:
            event = entry.trigger
            _insert_event(db, event.kind, event.details, event_id=event.id, once=True)
            seen_events.add(event.id)
        record = entry.record
        db.execute("INSERT INTO frames(camera_id, sequence, event_id, path, record) VALUES (?, ?, ?, ?, ?)",
                   (record["camera_id"], record["sequence"], record["trigger_event_id"], record["path"], json_text(record)))
        if memberships := record.get("event_memberships"):
            _insert_event_frame_links(db, ((context, record["camera_id"], record["sequence"], position)
                                           for context, position in memberships.items()))


def _insert_event_frame_links(db, links):
    db.executemany("INSERT INTO event_frame_links(event_id, camera_id, sequence, position) VALUES (?, ?, ?, ?)", links)


def _recorded_config(config: CaptureConfig, root: Path) -> dict:
    data = config.to_dict()
    metadata = data["metadata"]
    for parent_id, locator in (("continuation_of", "continuation_parent"), (REPEAT_OF, REPEAT_PARENT)):
        hint = metadata.get(locator)
        directory = hint.get("directory") if isinstance(hint, dict) else None
        if not (metadata.get(parent_id) and isinstance(directory, str) and Path(directory).is_absolute()):
            continue
        # Use the actual output directory, which may have changed since preflight.
        hint = {key: value for key, value in hint.items() if key != "relative_directory"}
        try:
            hint["relative_directory"] = Path(os.path.relpath(directory, root.resolve())).as_posix()
        except (OSError, ValueError):
            pass  # Separate drives or shares may have only an absolute locator.
        metadata[locator] = hint
    return data


class ExperimentWriter:
    def __init__(self, root: Path, config: CaptureConfig, *, trial_reference=None):
        root = resolve_output_path(root)
        if inside_repeat_source(config, root):
            raise CaptureError("OUTPUT_INSIDE_REPEAT_SOURCE", "请选择来源实验目录之外的新输出位置。")
        calibrated = self._load_calibrations(config)
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise CaptureError("OUTPUT_EXISTS", f"choose a new experiment directory: {root}") from exc
        self.root = root
        self._write_through = config.durability == "write_through"
        self._segment_frames = config.segment_frames
        try:
            self.disk_free_bytes: int | None = shutil.disk_usage(root).free
        except OSError:
            self.disk_free_bytes = None
        self._lock = threading.Lock()
        self._lease = WriterLease(root)
        self._lease.acquire()
        self._db = None
        self._frame_journal = None
        self._journal_lock = threading.Lock()
        self._journal_entries = 0
        self._journal_committed = 0
        try:
            staged = root / f".{DATABASE}.initializing"
            self._db = _connection(root, "rwc", database=staged.name)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(_DDL)
            if any(phase.overlap_events for phase in config.phases):
                self._db.executescript(_EVENT_LINKS_DDL)
            if any(phase.repeats_events for phase in config.phases):
                self._db.execute(f"CREATE INDEX frame_event_window ON frames(camera_id, {_EVENT_CONTEXT}, ordinal) "
                                 f"WHERE {_HAS_EVENT_CONTEXT}")
            self.experiment_id = str(uuid4())
            self._record = {
                "schema": EXPERIMENT_SCHEMA,
                "experiment_id": self.experiment_id,
                "created_utc_ns": time.time_ns(),
                "state": "opening",
                "config": _recorded_config(config, root),
                "calibrations": self._save_calibrations(calibrated),
                "frame_write_journal": {"schema": FRAME_WRITE_SCHEMA, "path": FRAME_WRITE_JOURNAL},
                "summary": None,
                "clocks": {
                    "host_utc": {
                        "clock_id": f"{self.experiment_id}:host_utc",
                        "reference": "unix_epoch", "unit": "ns", "sync_status": "unverified",
                        "implementation": time.get_clock_info("time").implementation,
                        "nominal_resolution_ns": round(1e9 * time.get_clock_info("time").resolution),
                    },
                    "host_monotonic": {
                        "clock_id": f"{self.experiment_id}:host_monotonic",
                        "reference": "host_performance_counter_origin", "unit": "ns", "sync_status": "local",
                        "implementation": time.get_clock_info("perf_counter").implementation,
                        "nominal_resolution_ns": round(1e9 * time.get_clock_info("perf_counter").resolution),
                    },
                },
            }
            if trial_reference:
                self._record["trial_reference"] = trial_reference
            with self._db:
                self._db.execute("INSERT INTO experiment VALUES (1, ?)", (json_text(self._record),))
            # Checkpoint the private database before publishing its main file.
            # Readers must be able to see the committed experiment on first open.
            if self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
                raise CaptureError("LEDGER_CHECKPOINT_FAILED", f"cannot checkpoint initial experiment ledger in {root}")
            self._db.close()
            self._db = None
            self._frame_journal = (root / FRAME_WRITE_JOURNAL).open("xb", buffering=0)
            _publish_path(staged, root / DATABASE, write_through=self._write_through)
            self._db = _connection(root, "rw")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._paths: dict[str, Path] = {}
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _load_calibrations(config):
        from .calibration import parse_profile

        records = {}
        camera_ids = {device.id for device in config.cameras}
        for path in config.calibration_files:
            data = Path(path).read_bytes()
            profile = parse_profile(data)
            using = camera_ids.intersection(profile["cameras"])
            if not using:
                raise CaptureError("CALIBRATION_CAMERA_UNBOUND", f"no configured camera matches {path}")
            records[hashlib.sha256(data).hexdigest()] = data, profile, sorted(using)
        return records

    def _save_calibrations(self, records):
        if not records:
            return []
        folder = self.root / "calibration"
        folder.mkdir()
        attached = []
        for digest, (data, profile, cameras) in records.items():
            file = folder / f"{digest}.json"
            with file.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            attached.append({
                "path": file.relative_to(self.root).as_posix(), "sha256": digest, "camera_ids": cameras,
                "source_experiment_id": profile["capture_experiment_id"],
            })
        _flush_directory(folder)
        _flush_directory(self.root)
        return attached

    def register_camera(self, device: Device, features: dict) -> None:
        slug = hashlib.sha256(device.id.encode("utf-8")).hexdigest()[:24]
        folder = self.root / "images" / slug
        folder.mkdir(parents=True)
        _flush_directory(folder.parent)
        _flush_directory(self.root)
        self._paths[device.id] = folder
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO cameras VALUES (?, ?)",
                (device.id, json_text({"device": asdict(device), "features": features, "path": f"images/{slug}"})),
            )

    def set_state(self, state: str, summary: dict | None = None) -> None:
        with self._lock, self._db:
            self._record.update(state=state, summary=summary)
            if state in TERMINAL_STATES:
                self._record["ended_utc_ns"] = time.time_ns()
            self._db.execute("UPDATE experiment SET record=? WHERE singleton=1", (json_text(self._record),))

    def committed_frame_totals(self) -> dict[str, tuple[int, int]]:
        """Read authoritative counts and TIFF bytes when a writer thread fails."""
        with self._lock:
            rows = self._db.execute("""
                SELECT camera_id, COUNT(*), SUM(
                    json_extract(record, '$.file_bytes') + COALESCE(
                        (SELECT SUM(json_extract(value, '$.file_bytes'))
                         FROM json_each(record, '$.auxiliary_images')), 0))
                FROM frames GROUP BY camera_id
            """).fetchall()
        return {camera: (count, size) for camera, count, size in rows}

    def event(self, kind: str, details: dict, *, event_id: str | None = None, once: bool = False) -> None:
        with self._lock, self._db:
            _insert_event(self._db, kind, details, event_id=event_id, once=once)

    def record_capture_events(self, events, *, links=()) -> None:
        """Commit signal and window records with their additional frame memberships."""
        with self._lock, self._db:
            for kind, details, identity in events:
                _insert_event(self._db, kind, details, event_id=identity)
            if links:
                _insert_event_frame_links(self._db, links)

    def record_markers(self, markers) -> None:
        from .markers import MARKER_EVENT

        with self._lock, self._db:
            for identity, record in markers:
                _insert_event(self._db, MARKER_EVENT, record, event_id=identity)

    def _checkpoint_frame_journal(self):
        # The staging thread calls this under the journal lock. Entries for a
        # published TIFF remain recoverable until SQLite has committed its row.
        if self._journal_entries and self._journal_committed == self._journal_entries:
            self._frame_journal.seek(0)
            self._frame_journal.truncate()
            self._journal_entries = 0
            self._journal_committed = 0

    def frame_journal_needs_checkpoint(self) -> bool:
        with self._journal_lock:
            return self._journal_entries >= _FRAME_JOURNAL_CHECKPOINT_ENTRIES

    def _journal_frame(self, prepared: PreparedFrame):
        descriptor = json_text({
            "experiment_id": self.experiment_id, "frame": prepared.record,
            "trigger": asdict(prepared.trigger) if prepared.trigger is not None else None,
        }).encode("utf-8")
        envelope = hashlib.sha256(descriptor).hexdigest().encode("ascii") + b" " + descriptor + b"\n"
        with self._journal_lock:
            self._checkpoint_frame_journal()
            # Count a failed append as pending too. A later commit of other rows
            # must retain evidence from this append.
            self._journal_entries += 1
            if self._frame_journal.write(envelope) != len(envelope):
                raise CaptureError("FRAME_JOURNAL_WRITE_FAILED", "could not write the complete frame descriptor")

    def prepare_frame(
        self, camera_id: str, sequence: int, phase: int, role: str, image: Image,
        trigger: TriggerEvent | None, event_context_id: str | None, event_position: str | None, min_free_bytes: int,
        *, payload: TiffPayload | None = None, event_memberships: tuple[tuple[str, str], ...] = (),
    ) -> PreparedFrame:
        try:
            self.disk_free_bytes = shutil.disk_usage(self.root).free
        except OSError:
            self.disk_free_bytes = None
            raise
        if self.disk_free_bytes < min_free_bytes + image.raw_bytes:
            raise CaptureError("DISK_RESERVE", f"disk reserve reached at {self.root}")
        folder = self._paths[camera_id]
        target = _frame_target(folder, sequence, self._segment_frames)
        if self._segment_frames and not target.parent.is_dir():
            target.parent.mkdir()
            if self._write_through:
                _flush_directory(folder)
        path = target.relative_to(self.root).as_posix()
        utc = self._record["clocks"]["host_utc"]
        monotonic = self._record["clocks"]["host_monotonic"]
        record = {
            "frame_id": f"{self.experiment_id}/{camera_id}/{sequence}",
            "camera_id": camera_id,
            "sequence": sequence,
            "native_frame_id": image.native_frame_id,
            "native_counter": image.native_counter,
            "phase": phase,
            "role": role,
            **_image_record(path, image),
            "trigger_event_id": trigger.id if trigger else None,
            "event_context_id": event_context_id,
            "event_position": event_position,
            "event_position_basis": "host_receipt_and_request_order" if event_position else None,
            "timestamps": {
                "received_utc": {"value": image.received_utc_ns, "source": "host_receive",
                                 "status": "recorded", "uncertainty_ns": None, **utc},
                "received_monotonic": {"value": image.received_monotonic_ns, "source": "host_receive",
                                       "status": "recorded", "uncertainty_ns": None, **monotonic},
                "device": _clock(image.device_time),
                "exposure_start": _clock(image.exposure_start),
                "exposure_end": _clock(image.exposure_end),
                "exposure_midpoint": _midpoint(image),
            },
        }
        if event_memberships:
            record["event_memberships"] = dict(event_memberships)
        prepared = PreparedFrame(record, trigger)
        parts = []
        if image.auxiliary_images:
            record["auxiliary_images"] = []
            for index, part in enumerate(image.auxiliary_images, 2):
                location = _part_target(target, index)
                tiff = prepare_tiff(part)
                part_record = _image_record(location.relative_to(self.root).as_posix(), part)
                part_record.update(file_bytes=tiff.size, sha256=tiff.sha256)
                record["auxiliary_images"].append(part_record)
                parts.append((location, part, tiff))

        def journal(size, digest):
            record.update(file_bytes=size, sha256=digest)
            self._journal_frame(prepared)

        options = {} if payload is None else {"payload": payload}
        _stage_tiff(target, image, before_write=journal, **options)
        for location, part, tiff in parts:
            _stage_tiff(location, part, payload=tiff)
        if not self._write_through:
            self._publish_frame(prepared)
        return prepared

    def _publish_frame(self, frame: PreparedFrame) -> None:
        for asset in frame_image_records(frame.record):
            target = self.root / asset["path"]
            partial = target.with_suffix(".tiff.part")
            if self._write_through:
                _publish_staged_tiff(partial, target, write_through=True)
            else:
                _publish_path(partial, target, write_through=False)

    def commit_frames(self, prepared: list[PreparedFrame]) -> None:
        if not prepared:
            return
        if self._write_through:
            # The frame descriptors must survive before any durable TIFF is
            # published. One journal sync covers the entire staged batch.
            try:
                os.fsync(self._frame_journal.fileno())
            except OSError as exc:
                raise CaptureError("FRAME_JOURNAL_WRITE_FAILED", str(exc)) from exc
            for index, frame in enumerate(prepared):
                try:
                    self._publish_frame(frame)
                except (OSError, CaptureError) as exc:
                    self._commit_published(prepared[:index])
                    raise PartialFrameCommitError(index, exc) from exc
        self._commit_published(prepared)

    def _commit_published(self, prepared: list[PreparedFrame]) -> None:
        if not prepared:
            return
        with self._lock, self._db:
            _insert_frames(self._db, prepared)
        with self._journal_lock:
            self._journal_committed += len(prepared)

    def close(self) -> None:
        try:
            if self._db is not None:
                self._db.close()
                self._db = None
        finally:
            try:
                if self._frame_journal is not None:
                    try:
                        self._checkpoint_frame_journal()
                    except OSError:
                        # The committed ledger remains authoritative if cleanup fails.
                        pass
                    finally:
                        self._frame_journal.close()
                        self._frame_journal = None
            finally:
                self._lease.release()


class ExperimentReader:
    """Read only committed entries. No camera drivers or Analysis imports."""

    def __init__(self, root: str | Path, *, immutable=False):
        self.root = Path(root).resolve()
        self._db = None
        try:
            self._db = _connection(self.root, "ro", immutable=immutable)
            self._db.execute("PRAGMA query_only=ON")
            if self.experiment["schema"] != EXPERIMENT_SCHEMA:
                raise CaptureError("UNSUPPORTED_SCHEMA", f"unsupported Capture schema in {self.root}")
        except BaseException:
            self.close()
            raise

    @property
    def experiment(self) -> dict:
        try:
            row = self._db.execute("SELECT record FROM experiment WHERE singleton=1").fetchone()
            if row is None:
                raise CaptureError("INCOMPLETE_EXPERIMENT", f"no experiment record in {self.root}")
            return json.loads(row[0])
        except (sqlite3.Error, ValueError) as exc:
            raise CaptureError("INVALID_LEDGER", f"cannot read experiment ledger in {self.root}: {exc}") from exc

    def cameras(self) -> list[dict]:
        return [json.loads(record) for (record,) in self._db.execute("SELECT record FROM cameras ORDER BY rowid")]

    def database_bytes(self) -> int:
        """Estimate the size of an independent snapshot, including committed WAL pages."""
        return self._db.execute("PRAGMA page_count").fetchone()[0] * self._db.execute("PRAGMA page_size").fetchone()[0]

    def backup_to(self, path: Path, *, progress: Callable[[], None]) -> None:
        """Write an independent SQLite file at a new path without checkpointing the source."""
        with path.open("xb"):
            pass
        target = sqlite3.connect(path)
        try:
            self._db.backup(target, pages=256, progress=lambda *_args: progress())
            if target.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
                raise CaptureError("INVALID_LEDGER", "could not prepare a standalone Capture ledger")
            if target.execute("PRAGMA quick_check(1)").fetchall() != [("ok",)]:
                raise CaptureError("INVALID_LEDGER", "the Capture ledger snapshot failed SQLite quick_check")
        finally:
            target.close()

    def committed_state(self) -> tuple[dict, dict]:
        """Read the experiment and per-camera ledger counts in one snapshot."""
        with self._db:
            self._db.execute("BEGIN")
            experiment = self.experiment
            counts = {camera["id"]: {"frames": 0, "images": 0} for camera in experiment["config"]["cameras"]}
            counts.update({camera: {"frames": frames, "images": images} for camera, frames, images in self._db.execute(
                """SELECT camera_id, COUNT(*), SUM(1 + COALESCE(json_array_length(record, '$.auxiliary_images'), 0))
                   FROM frames GROUP BY camera_id""",
            )})
            return experiment, counts

    def frames(self, *, camera_id: str | None = None, after: int = 0, before: int | None = None,
               limit: int | None = None, newest_first: bool = False) -> list[dict]:
        conditions = ["ordinal>?"]
        args = [after]
        if camera_id is not None:
            conditions.append("camera_id=?")
            args.append(camera_id)
        if before is not None:
            conditions.append("ordinal<?")
            args.append(before)
        order = "DESC" if newest_first else "ASC"
        query = f"SELECT ordinal,record FROM frames WHERE {' AND '.join(conditions)} ORDER BY ordinal {order} LIMIT ?"
        args.append(limit if limit is not None else -1)
        return [{"ordinal": ordinal, **json.loads(record)} for ordinal, record in self._db.execute(query, args)]

    def frame_at_sequence(self, camera_id: str, sequence: int) -> dict | None:
        row = self._db.execute("SELECT ordinal,event_id,record FROM frames WHERE camera_id=? AND sequence=?",
                               (camera_id, sequence)).fetchone()
        if row is None:
            return None
        ordinal, event, record = row
        frame = {**json.loads(record), "ordinal": ordinal}
        identity = f"{self.experiment['experiment_id']}/{camera_id}/{sequence}"
        if (frame.get("frame_id"), frame.get("camera_id"), frame.get("sequence"), frame.get("trigger_event_id")) != (
            identity, camera_id, sequence, event
        ):
            raise CaptureError("INVALID_LEDGER", f"inconsistent saved frame identity for {camera_id}/{sequence}")
        return frame

    @contextmanager
    def read_snapshot(self):
        """Keep related reads in the same SQLite transaction."""
        if self._db.in_transaction:
            yield
        else:
            with self._db:
                self._db.execute("BEGIN")
                yield

    @contextmanager
    def phase_image_review(self, camera_id, phase, *, expected_id, all_frames=False, cancelled=None):
        """Read one phase in a stable ledger snapshot, lazily decoding full-scan rows."""
        with self.read_snapshot():
            experiment = self.experiment
            if experiment["experiment_id"] != expected_id:
                raise CaptureError("FRAME_REVIEW_CHANGED", "the selected directory now contains a different experiment")
            if experiment["state"] not in TERMINAL_STATES:
                raise CaptureError("ARCHIVE_QUALITY_ACTIVE", "the record is still open for acquisition or recovery")
            plan = experiment["config"]
            if (camera_id not in {camera["id"] for camera in plan["cameras"]}
                    or type(phase) is not int or not 1 <= phase <= len(plan["phases"])):
                raise CaptureError("INVALID_FRAME_QUERY", f"{camera_id} / phase {phase}")
            where = "camera_id=? AND json_extract(record, '$.phase')=?"
            args = (camera_id, phase)
            count, largest = self._db.execute(
                f"SELECT COUNT(*), MAX(json_extract(record, '$.file_bytes')) FROM frames WHERE {where}", args,
            ).fetchone()
            indices = image_review_indices(count, largest or 0, all_frames=all_frames)
            if len(indices) == count:
                cursor = self._db.execute(f"SELECT ordinal,sequence,record FROM frames WHERE {where} ORDER BY sequence", args)
            else:
                positions = set(indices)
                ordinals = []
                with closing(self._db.execute(f"SELECT ordinal FROM frames WHERE {where} ORDER BY sequence", args)) as ordered:
                    for index, (ordinal,) in enumerate(ordered):
                        if cancelled and index % 256 == 0 and cancelled():
                            break
                        if index in positions:
                            ordinals.append(ordinal)
                placeholders = ",".join("?" for _ in ordinals)
                cursor = self._db.execute(f"SELECT ordinal,sequence,record FROM frames WHERE ordinal IN "
                                          f"({placeholders}) ORDER BY sequence", ordinals)

            def rows():
                for ordinal, sequence, text in cursor:
                    row = json.loads(text)
                    identity = f"{expected_id}/{camera_id}/{sequence}"
                    if (row.get("frame_id"), row.get("camera_id"), row.get("sequence"), row.get("phase")) != (
                        identity, camera_id, sequence, phase
                    ):
                        raise CaptureError("INVALID_LEDGER", f"inconsistent saved frame identity: {identity}")
                    yield {**row, "ordinal": ordinal}

            try:
                yield experiment, count, len(indices), rows()
            finally:
                cursor.close()

    def frame_page(self, *, camera_id=None, phase=None, role=None, first_sequence=None, last_sequence=None,
                   received_from_ns=None, received_to_ns=None, before=None, after=None, snapshot=None,
                   expected_experiment_id=None, event_context_id=None, limit=48):
        """Read compact frame metadata through an inclusive high-water mark in ledger order."""
        numeric = ((phase, 1), (first_sequence, 1), (last_sequence, 1), (before, 1), (after, 1), (snapshot, 0),
                   (received_from_ns, -_SQLITE_INTEGER_MAX - 1), (received_to_ns, -_SQLITE_INTEGER_MAX - 1))
        if (type(limit) is not int or not 1 <= limit <= 256 or (before is not None and after is not None)
                or any(value is not None and (type(value) is not int or not lo <= value <= _SQLITE_INTEGER_MAX)
                       for value, lo in numeric)):
            raise CaptureError("INVALID_FRAME_QUERY", "invalid page size, cursor, phase, sequence or time bound")
        if role is not None and role not in ("reference", "test", "calibration"):
            raise CaptureError("INVALID_FRAME_QUERY", f"unknown frame role: {role!r}")
        if event_context_id is not None and (not isinstance(event_context_id, str) or not event_context_id.strip()):
            raise CaptureError("INVALID_FRAME_QUERY", "event context ID must be nonempty text")
        for start, stop in ((first_sequence, last_sequence), (received_from_ns, received_to_ns)):
            if start is not None and stop is not None and start > stop:
                raise CaptureError("INVALID_FRAME_QUERY", "the start of a filter is greater than its end")

        with self._db:
            self._db.execute("BEGIN")
            experiment = self.experiment
            experiment_id = experiment["experiment_id"]
            if expected_experiment_id is not None and experiment_id != expected_experiment_id:
                raise CaptureError("FRAME_REVIEW_CHANGED", "the selected directory now contains a different experiment")
            cameras = tuple(camera["id"] for camera in experiment["config"]["cameras"])
            if camera_id is None and len(cameras) == 1:
                camera_id = cameras[0]
            if camera_id not in cameras:
                raise CaptureError("FRAME_REVIEW_CAMERA", "choose one camera recorded in the experiment plan")
            if phase is not None and phase > len(experiment["config"]["phases"]):
                raise CaptureError("INVALID_FRAME_QUERY", f"there is no phase {phase} in this experiment")
            head = self._db.execute("SELECT MAX(ordinal) FROM frames WHERE camera_id=?", (camera_id,)).fetchone()[0] or 0
            if snapshot is not None and snapshot > head:
                raise CaptureError("FRAME_REVIEW_CHANGED", "the ledger has fewer saved records than the selected snapshot")
            snapshot = head if snapshot is None else snapshot
            linked = self._has_shared_event_frames(event_context_id)
            source = "frames"
            source_args = ()
            conditions = ["camera_id=?", "ordinal<=?"]
            args = [camera_id, snapshot]
            if first_sequence is not None and first_sequence == last_sequence:
                conditions.append("sequence=?")
                args.append(first_sequence)
            elif (not linked and first_sequence is not None and last_sequence is not None and
                  last_sequence - first_sequence < _FRAME_SEQUENCE_WINDOW):
                # Seek with the camera/sequence uniqueness index before filtering and sorting a small window.
                # The inclusive sequence span bounds this subquery even in a long experiment with gaps.
                source = """(SELECT ordinal,sequence,event_id,record FROM frames
                             WHERE camera_id=? AND sequence BETWEEN ? AND ? ORDER BY sequence LIMIT ?)"""
                source_args = (camera_id, first_sequence, last_sequence, last_sequence - first_sequence + 1)
                conditions = ["ordinal<=?"]
                args = [snapshot]
            else:
                for clause, value in (("sequence>=?", first_sequence), ("sequence<=?", last_sequence)):
                    if value is not None:
                        conditions.append(clause)
                        args.append(value)
            for column, value in (("json_extract(record, '$.phase')=?", phase), ("json_extract(record, '$.role')=?", role),
                                  (f"{_FRAME_RECEIPT_UTC}>=?", received_from_ns), (f"{_FRAME_RECEIPT_UTC}<=?", received_to_ns)):
                if value is not None:
                    conditions.append(column)
                    args.append(value)
            if received_from_ns is not None or received_to_ns is not None:
                conditions.append("json_type(record, '$.timestamps.received_utc.value')='integer'")
            if linked:
                source, source_args = _EVENT_FRAMES, (event_context_id, camera_id)
            elif event_context_id is not None:
                conditions.extend((_HAS_EVENT_CONTEXT, f"{_EVENT_CONTEXT}=?"))
                args.append(event_context_id)
            scope = f"FROM {source} WHERE " + " AND ".join(conditions)
            params = (*source_args, *args)
            bound = ""
            cursor = ()
            if before is not None or after is not None:
                bound, cursor = (" AND ordinal < ?", (before,)) if before is not None else (" AND ordinal > ?", (after,))
            order = "ASC" if after is not None else "DESC"
            position_column = "member_position" if linked else "NULL"
            rows = self._db.execute(
                f"SELECT ordinal,sequence,event_id,record,{position_column} {scope}{bound} ORDER BY ordinal {order} LIMIT ?",
                (*params, *cursor, limit + 1),
            ).fetchall()
            more = len(rows) > limit
            rows = rows[:limit]
            if after is not None:
                rows.reverse()
            older = newer = None
            if rows:
                first, last = rows[0][0], rows[-1][0]
                if after is not None:
                    newer = first if more else None
                    if self._db.execute(f"SELECT 1 {scope} AND ordinal < ? LIMIT 1", (*params, last)).fetchone():
                        older = last
                else:
                    older = last if more else None
                    if before is not None and self._db.execute(
                            f"SELECT 1 {scope} AND ordinal > ? LIMIT 1", (*params, first)).fetchone():
                        newer = first
        results = []
        for ordinal, sequence, event_id, text, member_position in rows:
            record = json.loads(text)
            stamps = record.get("timestamps") or {}
            receive = (stamps.get("received_utc") or {}).get("value")
            results.append({
                "ordinal": ordinal, "sequence": sequence, "phase": record["phase"], "role": record["role"],
                "received_utc_ns": receive if type(receive) is int else None, "trigger_event_id": event_id,
                "event_position": member_position or record.get("event_position"), "pixel_format": record["pixel_format"],
                "images": len(frame_image_records(record)), "frame_id": record["frame_id"], "sha256": record["sha256"],
            })
        return {"experiment_id": experiment_id, "camera_id": camera_id, "snapshot_ordinal": snapshot,
                "older_before": older, "newer_after": newer, "frames": results}

    def reference_frames(self, camera_id: str, *, sequence: int | None = None, before: int | None = None) -> list[dict]:
        if sequence is not None:
            sql = "SELECT ordinal,record FROM frames WHERE camera_id=? AND sequence=? AND json_extract(record,'$.role')!='calibration'"
            rows = self._db.execute(sql, (camera_id, sequence))
        elif before is not None:
            rows = self._db.execute(
                "SELECT ordinal,record FROM frames WHERE camera_id=? AND sequence<? "
                "AND json_extract(record,'$.role')='reference' ORDER BY sequence", (camera_id, before),
            )
        else:
            sql = "SELECT ordinal,record FROM frames WHERE camera_id=? AND json_extract(record,'$.role')=? ORDER BY ordinal LIMIT ?"
            rows = list(self._db.execute(sql, (camera_id, "reference", 2)))
            if not rows and not any(phase["role"] == "reference" for phase in self.experiment["config"]["phases"]):
                rows = self._db.execute(sql, (camera_id, "test", 1))
        return [{"ordinal": ordinal, **json.loads(record)} for ordinal, record in rows]

    def next_test_frame(self, camera_id: str, *, after: int, latest: bool = True) -> tuple[dict | None, int]:
        where = "camera_id=? AND ordinal>? AND json_extract(record,'$.role')='test'"
        order = "DESC" if latest else "ASC"
        row = self._db.execute(f"SELECT ordinal,record FROM frames WHERE {where} ORDER BY ordinal {order} LIMIT 1",
                               (camera_id, after)).fetchone()
        if row is None:
            return None, 0
        ordinal, record = row
        omitted = self._db.execute(f"SELECT COUNT(*) FROM frames WHERE {where} AND ordinal<?",
                                   (camera_id, after, ordinal)).fetchone()[0] if latest else 0
        return {"ordinal": ordinal, **json.loads(record)}, omitted

    def committed_test_progress(self, camera_id: str, *, after: int) -> tuple[str, int, int]:
        """Read the run state and unseen test receipts from one ledger snapshot."""
        with self._db:
            self._db.execute("BEGIN")
            state = self.experiment["state"]
            count, last = self._db.execute(
                "SELECT COUNT(CASE WHEN json_extract(record,'$.role')='test' THEN 1 END),MAX(ordinal) "
                "FROM frames WHERE camera_id=? AND ordinal>?", (camera_id, after),
            ).fetchone()
        return state, count, last if last is not None else after

    def test_sequence_extent(self, camera_id: str, *, after: int) -> tuple[int, int]:
        """Count test frames and pin their inclusive ledger high-water mark in one read."""
        count, high_water = self._db.execute(
            "SELECT COUNT(*),MAX(ordinal) FROM frames "
            "WHERE camera_id=? AND sequence>? AND json_extract(record,'$.role')='test'", (camera_id, after),
        ).fetchone()
        return count, high_water or 0

    def test_sequence_page(self, camera_id: str, *, after: int, through: int | None = None,
                           phase: int | None = None, snapshot: int | None = None,
                           event_context_id: str | None = None, limit: int = 512) -> list[dict]:
        """Read experiment images in camera sequence order, including backfilled frames."""
        conditions = ["camera_id=?", "sequence>?", "json_extract(record,'$.role')='test'"]
        parameters = [camera_id, after]
        if through is not None:
            conditions.append("sequence<=?")
            parameters.append(through)
        if phase is not None:
            conditions.append("json_extract(record,'$.phase')=?")
            parameters.append(phase)
        if snapshot is not None:
            conditions.append("ordinal<=?")
            parameters.append(snapshot)
        linked = self._has_shared_event_frames(event_context_id)
        source = "frames"
        source_args = ()
        position_column = "NULL"
        if linked:
            source, source_args = _EVENT_FRAMES, (event_context_id, camera_id)
            position_column = "member_position"
        elif event_context_id is not None:
            conditions.extend((_HAS_EVENT_CONTEXT, f"{_EVENT_CONTEXT}=?"))
            parameters.append(event_context_id)
        where = " AND ".join(conditions)
        rows = self._db.execute(
            f"SELECT ordinal,sequence,record,{position_column} FROM {source} WHERE {where} ORDER BY sequence LIMIT ?",
            (*source_args, *parameters, limit),
        )
        frames = []
        for ordinal, sequence, text, member_position in rows:
            frame = json.loads(text)
            if frame["sequence"] != sequence or frame["camera_id"] != camera_id:
                raise CaptureError("INVALID_LEDGER", f"inconsistent saved frame identity for {camera_id}/{sequence}")
            if linked:
                frame.update(event_context_id=event_context_id, event_position=member_position)
            frames.append({**frame, "ordinal": ordinal})
        return frames

    def timeline(self, *, camera_id=None, phase=None, before=None, after=None, snapshot=None,
                 expected_experiment_id=None, limit=48, markers_only=False, event_windows_only=False,
                 incidents_only=False) -> dict:
        from .timeline import read_timeline

        with self._db:
            self._db.execute("BEGIN")
            return read_timeline(self._db, self.experiment, camera_id=camera_id, phase=phase,
                                 before=before, after=after, snapshot=snapshot,
                                 expected_experiment_id=expected_experiment_id, limit=limit, markers_only=markers_only,
                                 event_windows_only=event_windows_only, incidents_only=incidents_only)

    def has_timeline_incidents(self) -> bool:
        from .timeline import has_timeline_incidents

        return has_timeline_incidents(self._db)

    def marker_context(self, marker_id, *, expected_experiment_id=None) -> dict:
        from .markers import read_marker_context

        with self._db:
            self._db.execute("BEGIN")
            experiment = self.experiment
            if expected_experiment_id is not None and expected_experiment_id != experiment["experiment_id"]:
                raise CaptureError("FRAME_REVIEW_CHANGED", "the selected directory now contains a different experiment")
            return read_marker_context(self._db, experiment, marker_id, self.frame_at_sequence)

    def phase_evidence(self, event_kinds: tuple[str, ...]) -> tuple[dict, list, list]:
        """Read per-camera phase counts and event summaries in one ledger snapshot."""
        phase = "json_extract(record, '$.phase')"
        stopped = "json_extract(record, '$.stop_requested')"
        with self.read_snapshot():
            experiment = self.experiment
            frames = self._db.execute(
                f"SELECT {phase} AS phase_number,camera_id,COUNT(*) FROM frames GROUP BY phase_number,camera_id",
            ).fetchall()
            events = self._db.execute(
                f"SELECT {phase} AS phase_number,kind,MAX({stopped}),COUNT(*) FROM events "
                f"WHERE kind IN ({','.join('?' for _ in event_kinds)}) GROUP BY phase_number,kind", event_kinds,
            ).fetchall()
        return experiment, frames, events

    def events(self, *, after: int = 0, kind: str | None = None) -> list[dict]:
        if kind is None:
            rows = self._db.execute("SELECT ordinal,record FROM events WHERE ordinal>? ORDER BY ordinal", (after,))
        else:
            rows = self._db.execute(
                "SELECT ordinal,record FROM events WHERE ordinal>? AND kind=? ORDER BY ordinal", (after, kind),
            )
        return [{"ordinal": ordinal, **json.loads(record)} for ordinal, record in rows]

    def received_signal(self, signal_id: str) -> dict | None:
        row = self._db.execute("SELECT record FROM events WHERE event_id=? AND kind='signal_received'", (signal_id,)).fetchone()
        return json.loads(row[0]) if row is not None else None

    def _has_shared_event_frames(self, signal_id):
        signal = self.received_signal(signal_id) if signal_id is not None else None
        return signal is not None and signal.get("overlap") is True

    def trial_comparison(self) -> dict | None:
        reference = self.experiment.get("trial_reference")
        if not isinstance(reference, dict) or reference.get("schema") != TRIAL_REFERENCE_SCHEMA:
            return None
        stages = reference.get("stages")
        if not isinstance(stages, list) or not all(
            isinstance(row, dict) and type(row.get("phase")) is int and isinstance(row.get("experiment_id"), str)
            for row in stages
        ):
            raise CaptureError("INVALID_LEDGER", "invalid trial references")
        comparisons = {(event["phase"], check["trial_experiment_id"]): check for event in self.events(kind="phase_started")
                       if (check := event.get("trial_setup_check")) is not None}
        return {**reference, "stages": [
            {**row, "setup_check": comparisons.get((row["phase"], row["experiment_id"]))} for row in stages
        ]}

    def image_path(self, row: dict) -> Path:
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root / "images"):
            raise CaptureError("UNSAFE_IMAGE_PATH", f"frame path leaves the experiment image tree: {row['path']}")
        return path

    def _frames_for_triggers(self, events):
        groups = defaultdict(list)
        ids = [event["event_id"] for event in events]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            query = f"SELECT ordinal,event_id,record FROM frames WHERE event_id IN ({placeholders}) ORDER BY ordinal"
            for ordinal, event_id, record in self._db.execute(query, ids):
                groups[event_id].append({"ordinal": ordinal, **json.loads(record)})
        return groups

    def trigger_group_page(self, *, limit=24, before=None, after=None, newest_first=True,
                           problems_only=False, through_event=None):
        """Read one page with indexed frame lookups and a consistent ledger snapshot."""
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("request 1–256 trigger groups per page")
        with self._db:
            self._db.execute("BEGIN")
            state = self.experiment["state"]
            conditions = [_TRIGGER_EVENTS, _RETAINED_TRIGGER]
            args = []
            if through_event is not None:
                anchor = self._db.execute(
                    f"SELECT e.ordinal FROM events e WHERE e.event_id=? AND {_TRIGGER_EVENTS} AND {_RETAINED_TRIGGER}",
                    (through_event,),
                ).fetchone()
                if anchor is None:
                    raise CaptureError("TRIGGER_GROUP_UNAVAILABLE", f"no recorded trigger group {through_event}")
                conditions.append("e.ordinal <= ?")
                args.append(anchor[0])
            elif before is not None:
                conditions.append("e.ordinal < ?")
                args.append(before)
            if after is not None:
                conditions.append("e.ordinal > ?")
                args.append(after)
            if problems_only:
                conditions.append(_TRIGGER_PROBLEM)
            order = "DESC" if newest_first else "ASC"
            query = f"SELECT e.ordinal,e.record FROM events e WHERE {' AND '.join(conditions)} ORDER BY e.ordinal {order} LIMIT ?"
            events = [{"ordinal": ordinal, **json.loads(record)} for ordinal, record in self._db.execute(query, (*args, limit))]
            frames = self._frames_for_triggers(events)
            return [_review_group(event, frames[event["event_id"]], state) for event in events]

    def trigger_group(self, event_id):
        """Resolve the exact committed frames of one recorded trigger."""
        with self._db:
            self._db.execute("BEGIN")
            state = self.experiment["state"]
            query = f"SELECT e.ordinal,e.record FROM events e WHERE e.event_id=? AND {_TRIGGER_EVENTS} AND {_RETAINED_TRIGGER}"
            row = self._db.execute(query, (event_id,)).fetchone()
            if row is None:
                raise CaptureError("TRIGGER_GROUP_UNAVAILABLE", f"no recorded trigger group {event_id}")
            event = {"ordinal": row[0], **json.loads(row[1])}
            frames = self._frames_for_triggers([event])[event_id]
            return _review_group(event, frames, state), frames

    def hardware_phase_evidence(self, phase: int, counter_feature: str, *, include_hash=False):
        """Read hardware event identities and compact frame references in one snapshot."""
        with self._db:
            self._db.execute("BEGIN")
            experiment = self.experiment
            events = []
            for kind, event_id, record in self._db.execute(
                """SELECT kind,event_id,record FROM events
                   WHERE kind IN ('hardware_trigger','phase_started','phase_ended')
                   AND json_extract(record, '$.phase')=? ORDER BY ordinal""", (phase,)):
                event = json.loads(record)
                consistent = (event.get("kind"), event.get("event_id")) == (kind, event_id)
                events.append({**event, "kind": kind, "event_id": event_id, "ledger_consistent": consistent})
            frames = []
            for camera_id, sequence, event_id, record in self._db.execute(
                """SELECT f.camera_id,f.sequence,f.event_id,f.record FROM frames f
                   WHERE json_extract(f.record, '$.phase')=? OR f.event_id IN (
                       SELECT e.event_id FROM events e WHERE e.kind='hardware_trigger'
                       AND json_extract(e.record, '$.phase')=?) ORDER BY f.ordinal""", (phase, phase)):
                row = json.loads(record)
                metadata = row.get("backend_metadata")
                expected_id = f"{experiment['experiment_id']}/{camera_id}/{sequence}"
                frames.append({"camera_id": camera_id, "sequence": sequence, "event_id": event_id,
                               "frame_id": row.get("frame_id"), "phase": row.get("phase"), "path": row.get("path"),
                               "counter": metadata.get(counter_feature) if isinstance(metadata, dict) else None,
                               "ledger_consistent": (row.get("frame_id"), row.get("camera_id"), row.get("sequence"), row.get("trigger_event_id"))
                               == (expected_id, camera_id, sequence, event_id)})
                if include_hash:
                    frames[-1]["sha256"] = row.get("sha256")
        return experiment, events, frames

    def pairing(self) -> dict:
        experiment = self.experiment
        state = experiment["state"]
        frames = self.frames()
        by_event = defaultdict(list)
        observed_hardware = defaultdict(list)
        unassociated = defaultdict(int)
        for row in frames:
            if row["trigger_event_id"]:
                by_event[row["trigger_event_id"]].append(row)
            else:
                unassociated[row["camera_id"]] += 1
        groups = []
        for event in self.events():
            if event["kind"] not in ("software_trigger", "hardware_trigger"):
                continue
            received = by_event.pop(event["event_id"], [])
            if not received and event.get("retention") == "rolling_prebuffer":
                continue
            if event["kind"] == "hardware_trigger":
                observed_hardware[event["phase"]].append(event["external_event_number"])
            groups.append(_trigger_group(event, received, state))
        gaps = []
        for phase, observed in observed_hardware.items():
            configuration = self.experiment["config"]["phases"][phase - 1]["hardware_pairing"]
            left = configuration["first_event_number"]
            for value in sorted(set(observed)):
                if left < value:
                    gaps.append({"phase": phase, "first_event_number": left, "last_event_number": value - 1})
                left = value + 1
        return {"groups": groups, "unassociated_frames": dict(unassociated),
                "unrecognized_events": sorted(by_event), "hardware_counter_gap_ranges": gaps}

    def verify(self, *, hash_files: bool = True) -> dict:
        experiment = self.experiment
        state = experiment["state"]
        rows = self.frames()
        expected = set()
        issues = []
        for frame in rows:
            for asset in frame_image_records(frame):
                name = asset["path"]
                expected.add(name)
                try:
                    path = self.image_path(asset)
                    if path.stat().st_size != asset["file_bytes"]:
                        issues.append({"code": "FILE_SIZE_MISMATCH", "path": name})
                    elif hash_files and sha256_file(path) != asset["sha256"]:
                        issues.append({"code": "FILE_HASH_MISMATCH", "path": name})
                except FileNotFoundError:
                    issues.append({"code": "MISSING_FILE", "path": name})
                except CaptureError as exc:
                    issues.append({"code": exc.code, "path": name})
                except OSError as exc:
                    issues.append({"code": "UNREADABLE_FILE", "path": name, "message": str(exc)})
        for attachment in experiment.get("calibrations", []):
            path = self.root / attachment["path"]
            if not path.resolve().is_relative_to(self.root / "calibration") or not path.is_file():
                issues.append({"code": "CALIBRATION_FILE_MISSING", "path": attachment["path"]})
            elif hash_files and sha256_file(path) != attachment["sha256"]:
                issues.append({"code": "CALIBRATION_HASH_MISMATCH", "path": attachment["path"]})
        uncommitted = sorted(
            path.relative_to(self.root).as_posix()
            for path in (self.root / "images").rglob("*")
            if path.is_file() and path.relative_to(self.root).as_posix() not in expected
        )
        terminal = state in TERMINAL_STATES
        phases = experiment["config"]["phases"]
        shared_phases = {index for index, phase in enumerate(phases, start=1) if phase.get("overlap_events")}
        repeated_phases = shared_phases | {index for index, phase in enumerate(phases, start=1)
                                          if phase["mode"] == "event" and phase.get("event_count", 1) != 1}
        if terminal:
            issues.extend({"code": "UNCOMMITTED_FILE", "path": path} for path in uncommitted)
            pairing = self.pairing()
            issues.extend({"code": "TRIGGER_PAIRING_" + group["status"].upper(), "event_id": group["event_id"]}
                          for group in pairing["groups"] if group["status"] != "associated")
            issues.extend({"code": "TRIGGER_EVENT_UNKNOWN", "event_id": event_id}
                          for event_id in pairing["unrecognized_events"])
            issues.extend({"code": "SHARED_HARDWARE_COUNTER_GAP", **gap}
                          for gap in pairing["hardware_counter_gap_ranges"])
            if repeated_phases:
                from .event_integrity import event_window_issues

                shared_links = {}
                if shared_phases:
                    if self._db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='event_frame_links'").fetchone():
                        for context, camera, sequence, position, ordinal in self._db.execute("""
                            SELECT l.event_id, l.camera_id, l.sequence, l.position, f.ordinal
                            FROM event_frame_links AS l LEFT JOIN frames AS f USING (camera_id, sequence)
                        """):
                            if ordinal is None:
                                issues.append({"code": "EVENT_FRAME_LINK_MISSING_IMAGE", "event_id": context,
                                               "camera_id": camera, "sequence": sequence})
                            else:
                                shared_links.setdefault(ordinal, {})[context] = position
                    else:
                        issues.append({"code": "EVENT_FRAME_LINK_TABLE_MISSING"})
                issues.extend(event_window_issues(rows, self.events(kind="signal_received"), self.events(kind="event_window"),
                                                  repeated_phases, [item["id"] for item in experiment["config"]["cameras"]],
                                                  shared_phases=shared_phases, shared_links=shared_links))
        summary = experiment.get("summary") or {}
        losses = reported_camera_losses(summary.get("cameras", {}))
        misses = summary.get("scheduled_misses", 0)
        pre_short = summary.get("short_pre_windows", 0)
        write_buffer = {key: summary.get(key) for key in (
            "queue_capacity", "queue_high_water", "raw_buffer_limit_bytes", "raw_buffer_high_water_bytes",
        )}
        report = {
            "ok": not issues, "complete": state == "completed" and not issues and not losses and not misses and not pre_short,
            "state": state, "committed_frames": len(rows), "committed_images": len(expected), "issues": issues,
            "uncommitted_candidates": uncommitted if not terminal else [],
            "file_hashes_checked": hash_files, "reported_camera_losses": losses, "scheduled_misses": misses,
            "short_pre_windows": pre_short,
            "write_buffer": write_buffer,
            "capture_error": summary.get("error"),
            "runtime_summary_available": {"cameras", "scheduled_misses", "short_pre_windows"} <= summary.keys(),
            "hardware_wait_observations": self.events(kind="hardware_wait_observations"),
        }
        if alerts := summary.get("write_alerts"):
            report["write_alerts"] = alerts
        if repeated_phases:
            report["event_phases"] = self.events(kind="event_phase_summary")
        if pulses := summary.get("hardware_groups"):
            report["hardware_groups"] = pulses
        if experiment["config"].get("metadata", {}).get("continuation_reference"):
            report["continuation_checks"] = [row["continuation_check"] for row in self.events(kind="phase_started")
                                             if "continuation_check" in row]
        if experiment.get("trial_reference") and (comparison := self.trial_comparison()) is not None:
            report["trial_comparison"] = comparison
        if any(phase.get("start_confirmation") for phase in experiment["config"]["phases"]):
            report["phase_confirmation_events"] = [{"ordinal": ordinal, **json.loads(record)}
                                                  for ordinal, record in self._db.execute("""
                SELECT ordinal, record FROM events
                WHERE kind IN ('phase_confirmation_waiting', 'phase_confirmed', 'phase_confirmation_cancelled')
                    OR (kind = 'phase_started' AND json_type(record, '$.confirmation_id') = 'text')
                ORDER BY ordinal
            """)]
        return report

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> ExperimentReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def recover_experiment(root: str | Path, *, expected_experiment_id: str | None = None) -> dict:
    """Recover verified frames and finalize an interrupted ledger."""
    from .recovery import recover

    return recover(root, expected_experiment_id=expected_experiment_id)
