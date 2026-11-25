"""Lossless image publication and a versioned, independently readable ledger."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sqlite3
import threading
import time
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from .cameras.base import ClockReading, Device, Image
from .config import CaptureConfig, CaptureError


EXPERIMENT_SCHEMA = "straintrace.capture/v1"
TERMINAL_STATES = frozenset({"completed", "stopped", "failed", "interrupted"})
DATABASE = "capture.sqlite3"

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


@dataclass(frozen=True)
class TriggerEvent:
    id: str
    kind: str
    details: dict


@dataclass(frozen=True)
class PreparedFrame:
    record: dict
    trigger: TriggerEvent | None


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class WriterLease:
    """An OS-owned lease. A stale marker file never prevents an explicit recovery."""

    def __init__(self, root: Path):
        self.path = root / ".capture-writer"
        self._stream = None

    def acquire(self) -> None:
        stream = self.path.open("a+b")
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


def _connection(root: Path, mode: str) -> sqlite3.Connection:
    url = (root / DATABASE).resolve().as_uri()
    return sqlite3.connect(f"{url}?mode={mode}", uri=True, timeout=5, check_same_thread=False)


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


def _publish_tiff(target: Path, image: Image, *, write_through: bool) -> tuple[int, str]:
    # OpenCV's TIFF encoder takes BGR arrays, while the file stores RGB samples.
    pixels = image.pixels
    if image.color_model in ("rgba", "bgra"):
        from PIL import Image as PillowImage

        if pixels.dtype.itemsize != 1:
            raise CaptureError("PIXEL_FORMAT_UNSUPPORTED", "four-channel TIFF requires 8-bit samples")
        rgba = cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGBA) if image.color_model == "bgra" else pixels
        with io.BytesIO() as payload:
            PillowImage.fromarray(rgba).save(payload, format="TIFF", compression="raw")
            data = payload.getvalue()
    else:
        if image.color_model == "rgb":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        encoded, data = cv2.imencode(".tiff", pixels, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        if not encoded:
            raise CaptureError("ENCODE_FAILED", f"cannot encode {target.name} as lossless TIFF")
    if target.exists():
        raise CaptureError("OUTPUT_EXISTS", f"refusing to replace {target}")
    partial = target.with_suffix(".tiff.part")
    with partial.open("xb") as stream:
        stream.write(data)
        if write_through:
            stream.flush()
            os.fsync(stream.fileno())
    _publish_path(partial, target, write_through=write_through)
    return memoryview(data).nbytes, hashlib.sha256(data).hexdigest()


class ExperimentWriter:
    def __init__(self, root: Path, config: CaptureConfig):
        calibrated = self._load_calibrations(config)
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise CaptureError("OUTPUT_EXISTS", f"choose a new experiment directory: {root}") from exc
        self.root = root
        self._write_through = config.durability == "write_through"
        self._segment_frames = config.segment_frames
        self._lock = threading.Lock()
        self._lease = WriterLease(root)
        self._lease.acquire()
        self._db = None
        try:
            self._db = _connection(root, "rwc")
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(_DDL)
            self.experiment_id = str(uuid4())
            self._record = {
                "schema": EXPERIMENT_SCHEMA,
                "experiment_id": self.experiment_id,
                "created_utc_ns": time.time_ns(),
                "state": "opening",
                "config": config.to_dict(),
                "calibrations": self._save_calibrations(calibrated),
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
            with self._db:
                self._db.execute("INSERT INTO experiment VALUES (1, ?)", (json_text(self._record),))
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

    def event(self, kind: str, details: dict, *, event_id: str | None = None, once: bool = False) -> None:
        with self._lock, self._db:
            self._insert_event(kind, details, event_id=event_id, once=once)

    def _insert_event(self, kind: str, details: dict, *, event_id: str | None, once: bool):
        record = {"kind": kind, "utc_ns": time.time_ns(), "event_id": event_id, **details}
        statement = "INSERT INTO events(event_id, kind, record) VALUES (?, ?, ?)"
        if once:
            statement += " ON CONFLICT(event_id) DO NOTHING"
        self._db.execute(statement, (event_id, kind, json_text(record)))

    def prepare_frame(
        self, camera_id: str, sequence: int, phase: int, role: str, image: Image,
        trigger: TriggerEvent | None, event_context_id: str | None, event_position: str | None, min_free_bytes: int,
    ) -> PreparedFrame:
        if shutil.disk_usage(self.root).free < min_free_bytes + image.pixels.nbytes:
            raise CaptureError("DISK_RESERVE", f"disk reserve reached at {self.root}")
        folder = self._paths[camera_id]
        if self._segment_frames:
            segment = folder / f"segment-{(sequence - 1) // self._segment_frames + 1:07d}"
            if not segment.is_dir():
                segment.mkdir()
                if self._write_through:
                    _flush_directory(folder)
            folder = segment
        target = folder / f"{sequence:012d}.tiff"
        size, digest = _publish_tiff(target, image, write_through=self._write_through)
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
            "path": path,
            "file_bytes": size,
            "sha256": digest,
            "shape": list(image.pixels.shape),
            "dtype": image.pixels.dtype.name,
            "significant_bits": image.significant_bits,
            "pixel_format": image.pixel_format,
            "color_model": image.color_model,
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
            "backend_metadata": image.metadata,
        }
        return PreparedFrame(record, trigger)

    def commit_frames(self, prepared: list[PreparedFrame]) -> None:
        with self._lock, self._db:
            seen_events = set()
            for entry in prepared:
                if entry.trigger and entry.trigger.id not in seen_events:
                    event = entry.trigger
                    self._insert_event(event.kind, event.details, event_id=event.id, once=True)
                    seen_events.add(event.id)
                record = entry.record
                self._db.execute("INSERT INTO frames(camera_id, sequence, event_id, path, record) VALUES (?, ?, ?, ?, ?)",
                                 (record["camera_id"], record["sequence"], record["trigger_event_id"], record["path"],
                                  json_text(record)))

    def close(self) -> None:
        try:
            if self._db is not None:
                self._db.close()
                self._db = None
        finally:
            self._lease.release()


class ExperimentReader:
    """Read only committed entries. No camera drivers or Analysis imports."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._db = None
        try:
            self._db = _connection(self.root, "ro")
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

    def frames(self, *, camera_id: str | None = None, after: int = 0, limit: int | None = None) -> list[dict]:
        if camera_id is None:
            query = "SELECT ordinal,record FROM frames WHERE ordinal>? ORDER BY ordinal LIMIT ?"
            args = (after, limit if limit is not None else -1)
        else:
            query = "SELECT ordinal,record FROM frames WHERE camera_id=? AND ordinal>? ORDER BY ordinal LIMIT ?"
            args = (camera_id, after, limit if limit is not None else -1)
        return [{"ordinal": ordinal, **json.loads(record)} for ordinal, record in self._db.execute(query, args)]

    def reference_frames(self, camera_id: str, *, sequence: int | None = None) -> list[dict]:
        if sequence is not None:
            sql = "SELECT ordinal,record FROM frames WHERE camera_id=? AND sequence=? AND json_extract(record,'$.role')!='calibration'"
            rows = self._db.execute(sql, (camera_id, sequence))
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

    def events(self, *, after: int = 0) -> list[dict]:
        rows = self._db.execute("SELECT ordinal,record FROM events WHERE ordinal>? ORDER BY ordinal", (after,))
        return [{"ordinal": ordinal, **json.loads(record)} for ordinal, record in rows]

    def image_path(self, row: dict) -> Path:
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root / "images"):
            raise CaptureError("UNSAFE_IMAGE_PATH", f"frame path leaves the experiment image tree: {row['path']}")
        return path

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
            mapping = defaultdict(list)
            for row in received:
                mapping[row["camera_id"]].append(row["frame_id"])
            expected = set(event["expected_cameras"])
            if event["kind"] == "hardware_trigger":
                observed_hardware[event["phase"]].append(event["external_event_number"])
            missing = sorted(expected - set(mapping))
            unexpected = sorted(set(mapping) - expected)
            ambiguous = sorted(camera for camera, ids in mapping.items() if len(ids) != 1)
            status = "associated"
            if unexpected or ambiguous:
                status = "ambiguous"
            elif missing:
                status = "incomplete" if state in TERMINAL_STATES else "pending_or_incomplete"
            groups.append({
                "event_id": event["event_id"], "phase": event["phase"], "status": status,
                "basis": event["pairing_basis"], "synchronization": event["synchronization"],
                "external_event_number": event.get("external_event_number"),
                "external_source_id": event.get("external_source_id"),
                "missing_cameras": missing, "unexpected_cameras": unexpected,
                "ambiguous_cameras": ambiguous, "frames_by_camera": dict(mapping),
            })
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
        state = self.experiment["state"]
        rows = self.frames()
        expected = {row["path"] for row in rows}
        issues = []
        for row in rows:
            path = self.image_path(row)
            try:
                if path.stat().st_size != row["file_bytes"]:
                    issues.append({"code": "FILE_SIZE_MISMATCH", "path": row["path"]})
                elif hash_files and sha256_file(path) != row["sha256"]:
                    issues.append({"code": "FILE_HASH_MISMATCH", "path": row["path"]})
            except FileNotFoundError:
                issues.append({"code": "MISSING_FILE", "path": row["path"]})
            except OSError as exc:
                issues.append({"code": "UNREADABLE_FILE", "path": row["path"], "message": str(exc)})
        for attachment in self.experiment.get("calibrations", []):
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
        if terminal:
            issues.extend({"code": "UNCOMMITTED_FILE", "path": path} for path in uncommitted)
            pairing = self.pairing()
            issues.extend({"code": "TRIGGER_PAIRING_" + group["status"].upper(), "event_id": group["event_id"]}
                          for group in pairing["groups"] if group["status"] != "associated")
            issues.extend({"code": "TRIGGER_EVENT_UNKNOWN", "event_id": event_id}
                          for event_id in pairing["unrecognized_events"])
            issues.extend({"code": "SHARED_HARDWARE_COUNTER_GAP", **gap}
                          for gap in pairing["hardware_counter_gap_ranges"])
        summary = self.experiment.get("summary") or {}
        losses = {
            camera: {key: stats[key] for key in ("app_unsaved", "native_counter_gaps", "native_counter_duplicates",
                                                "hardware_identity_missing", "reported_driver_skips", "transport_errors")
                     if stats.get(key)}
            for camera, stats in summary.get("cameras", {}).items()
        }
        losses = {camera: counts for camera, counts in losses.items() if counts}
        misses = summary.get("scheduled_misses", 0)
        pre_short = summary.get("short_pre_windows", 0)
        return {
            "ok": not issues, "complete": state == "completed" and not issues and not losses and not misses and not pre_short,
            "state": state, "committed_frames": len(rows), "issues": issues,
            "uncommitted_candidates": uncommitted if not terminal else [],
            "file_hashes_checked": hash_files, "reported_camera_losses": losses, "scheduled_misses": misses,
            "short_pre_windows": pre_short,
        }

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> ExperimentReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def recover_experiment(root: str | Path) -> dict:
    """Finalize an interrupted ledger without modifying or inventing images."""
    root = Path(root).resolve()
    with WriterLease(root), ExperimentReader(root) as reader:
        record = reader.experiment
        if record["state"] not in TERMINAL_STATES:
            verification = reader.verify()
            prior = record["state"]
            record.update(state="interrupted", recovered_utc_ns=time.time_ns())
            with closing(_connection(root, "rw")) as db, db:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("UPDATE experiment SET record=? WHERE singleton=1", (json_text(record),))
                db.execute("INSERT INTO events(kind, record) VALUES (?, ?)", ("recovery", json_text({
                    "kind": "recovery", "event_id": None, "previous_state": prior,
                    "utc_ns": time.time_ns(), "observed": verification,
                })))
        return reader.verify()
