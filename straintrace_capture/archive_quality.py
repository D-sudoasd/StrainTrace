"""Cancellable quality review of the main TIFFs from an archived capture phase."""

from array import array
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from tempfile import SpooledTemporaryFile
from threading import Lock
import time

from .cameras.base import Device
from .config import CaptureError, QualityRegion
from .image_quality_watch import quality_region_layout_mismatch
from .preview_image import PreviewOriginal, preview_original
from .storage import ExperimentReader, TERMINAL_STATES, json_text
from .trial_quality import (
    _IMAGE_ERRORS, _failed_row, _measured_row, _position, _read_image, _verified_quality_frame, trial_sensor_values,
)


def _scratch_rows(scratch):
    return SpooledTemporaryFile(max_size=2 * 1024**2, dir=scratch)


class ArchiveQualityReport:
    """Keep row payloads on scratch storage after a small, bounded in-memory spool."""

    def __init__(self, available, selected, *, scratch=None, shared_rows=None, row_lock=None):
        self.available = available
        self.selected = selected
        self.verified = 0
        self.fallbacks = 0
        self.stopped = False
        self.first_alert = None
        self._owns_rows = shared_rows is None
        self._rows = _scratch_rows(scratch) if self._owns_rows else shared_rows
        self._row_lock = Lock() if row_lock is None else row_lock
        self._offsets = array("Q")
        self._alerts = array("Q")

    @property
    def checked(self):
        return len(self._offsets)

    @property
    def flagged(self):
        return len(self._alerts)

    def append(self, row):
        offset = self._rows.tell()
        self._rows.write(json_text(row).encode("utf-8") + b"\n")
        self._offsets.append(offset)
        if row["flags"]:
            self._alerts.append(offset)
            if self.first_alert is None:
                self.first_alert = row["sequence"]
        self.verified += row["metrics"] is not None
        self.fallbacks += row["fallback"] is not None

    def page(self, start, count, *, alerts_only=False):
        offsets = self._alerts if alerts_only else self._offsets
        rows = []
        with self._row_lock:
            for offset in offsets[start:start + count]:
                self._rows.seek(offset)
                rows.append(json.loads(self._rows.readline()))
        return rows

    def close(self):
        if self._owns_rows:
            with self._row_lock:
                self._rows.close()


def archive_quality_options(root: Path, experiment_id: str):
    with ExperimentReader(root) as reader:
        return _options(reader, experiment_id)


@dataclass(frozen=True, slots=True)
class ArchiveComparison:
    original: PreviewOriginal
    experiment_id: str
    experiment_name: str
    phase: int
    sequence: int
    sha256: str
    roi: tuple[int, int, int, int] | None


def open_archive_comparison(root: Path, experiment_id, camera_id, phase, expected):
    """Bind the selected TIFF and its inspection coordinates to current ledger evidence."""
    with ExperimentReader(root) as reader:
        experiment = reader.experiment
        if experiment["experiment_id"] != experiment_id:
            raise CaptureError("FRAME_REVIEW_CHANGED", str(reader.root))
        if experiment["state"] not in TERMINAL_STATES:
            raise CaptureError("ARCHIVE_QUALITY_ACTIVE", str(reader.root))
        row, image = _verified_quality_frame(reader, camera_id, expected)
        if row["phase"] != phase:
            raise CaptureError("FRAME_REVIEW_CHANGED", row["frame_id"])
        saved = next((item["device"] for item in reader.cameras() if item["device"]["id"] == camera_id), {})
        try:
            device = Device(**{"identity_scope": "unverified", **saved})
        except TypeError:
            device = Device(camera_id, camera_id.split(":", 1)[0], "", camera_id, None, "unknown", "unverified")
        started = next((item for item in reader.events(kind="phase_started") if item.get("phase") == phase), {})
        source = preview_original(device, image, trial_sensor_values(started, camera_id), path=reader.image_path(row))
        return ArchiveComparison(source, experiment_id, experiment["config"]["name"], phase, row["sequence"],
                                 row["sha256"], tuple(expected["roi"]) if expected["roi"] is not None else None)


def _options(reader, experiment_id):
    experiment, counts, _events = reader.phase_evidence(("phase_started",))
    if experiment["experiment_id"] != experiment_id:
        raise CaptureError("FRAME_REVIEW_CHANGED", str(reader.root))
    if experiment["state"] not in TERMINAL_STATES:
        raise CaptureError("ARCHIVE_QUALITY_ACTIVE", str(reader.root))
    plan = experiment["config"]
    return {
        "experiment_id": experiment["experiment_id"],
        "experiment_name": plan["name"],
        "counts": {(camera, str(phase)): count for phase, camera, count in counts},
        "regions": plan.get("quality_regions") or {},
        "test_phases": tuple(str(index) for index, phase in enumerate(plan["phases"], 1) if phase.get("role", "test") == "test"),
        "scopes": tuple((camera["id"], str(phase))
                        for phase in range(1, len(plan["phases"]) + 1) for camera in plan["cameras"]),
    }


def scan_archive_quality(root, experiment_id, camera, phase, *, use_region, all_frames=False,
                         cancelled=None, progress=None, scratch=None):
    """Verify and measure selected raw TIFFs while keeping the source ledger read-only."""
    with ExperimentReader(root) as reader:
        return _scan_scope(reader, experiment_id, camera, phase, use_region=use_region, all_frames=all_frames,
                           cancelled=cancelled, progress=progress, scratch=scratch)


def _scan_scope(reader, experiment_id, camera, phase, *, use_region, all_frames, cancelled, progress,
                scratch=None, shared_rows=None, row_lock=None, starts=None):
    result = None
    try:
        with reader.phase_image_review(
            camera, phase, expected_id=experiment_id, all_frames=all_frames, cancelled=cancelled,
        ) as (experiment, count, selected, rows):
            result = ArchiveQualityReport(count, selected, scratch=scratch, shared_rows=shared_rows, row_lock=row_lock)
            saved = (experiment["config"].get("quality_regions") or {}).get(camera) if use_region else None
            region = QualityRegion.from_dict(saved) if saved is not None else None
            started = {}
            if region is not None:
                if starts is None:
                    started = next((event for event in reader.events(kind="phase_started") if event.get("phase") == phase), {})
                else:
                    started = starts.get(phase, {})
            values = trial_sensor_values(started, camera)
            if progress:
                progress(0, selected)
            for row in rows:
                if cancelled and cancelled():
                    break
                item = _position(row)
                fallback = quality_region_layout_mismatch(
                    region, row["shape"], row["pixel_format"], row["significant_bits"], row.get("backend_metadata") or {}, values,
                ) if region else None
                roi = region.bounds if region is not None and fallback is None else None
                try:
                    checked = _measured_row(item, _read_image(reader, row), roi, measure_texture=False)
                except _IMAGE_ERRORS as exc:
                    checked = _failed_row(item, exc)
                if fallback is not None:
                    checked["flags"].append("ARCHIVE_QUALITY_SENSOR_CHANGED" if fallback == "sensor_window"
                                            else "ARCHIVE_QUALITY_LAYOUT_CHANGED")
                result.append({**checked, "path": row["path"], "roi": list(roi) if roi is not None else None,
                               "fallback": fallback})
                if progress:
                    progress(result.checked, selected)
            result.stopped = result.checked < selected and bool(cancelled and cancelled())
            return result
    except BaseException:
        if result is not None:
            result.close()
        raise


class ArchiveQualityBatch:
    """Per-scope reports from a single ledger snapshot with one scratch stream."""

    def __init__(self, root, options, all_frames, scratch):
        self.root = root
        self.options = options
        self.all_frames = all_frames
        self.reports = {}
        self.errors = {}
        self.stopped = False
        self.started_utc_ns = time.time_ns()
        self.finished_utc_ns = None
        self._rows = _scratch_rows(scratch)
        self._row_lock = Lock()
        self._retained = 0
        self._closing = False

    def coverage(self, scope):
        camera, phase = scope
        count = self.options["counts"].get(scope, 0)
        report = self.reports.get(scope)
        error = self.errors.get(scope)
        if report is not None:
            state = "stopped" if report.stopped else "full" if report.checked == count else "sampled"
        else:
            state = "failed" if error is not None else "unchecked" if count else "no_frames"
        region = self.options["regions"].get(camera)
        return {
            "camera": camera, "phase": int(phase), "status": state,
            "available": count, "selected": report.selected if report else None,
            "checked": report.checked if report else 0, "verified": report.verified if report else None,
            "alerts": report.flagged if report else None, "first_alert_sequence": report.first_alert if report else None,
            "whole_frame_fallbacks": report.fallbacks if report else None,
            "requested_region": region, "error_code": getattr(error, "code", "CAPTURE_ERROR") if error else None,
        }

    def retain(self):
        """Keep the completed rows available to an export while its window closes."""
        with self._row_lock:
            if self._closing:
                raise CaptureError("QUALITY_REPORT_CLOSED", str(self.root))
            self._retained += 1

    def release(self):
        with self._row_lock:
            self._retained -= 1
            if self._closing and not self._retained:
                self._rows.close()

    def close(self):
        with self._row_lock:
            self._closing = True
            if not self._retained:
                self._rows.close()


def scan_archive_quality_batch(root, experiment_id, *, all_frames=False, cancelled=None, progress=None, scratch=None):
    """Check every recorded stage and camera using the same committed ledger snapshot."""
    with ExperimentReader(root) as reader, reader.read_snapshot():
        options = _options(reader, experiment_id)
        batch = ArchiveQualityBatch(reader.root, options, all_frames, scratch)
        try:
            jobs = [scope for scope in options["scopes"] if options["counts"].get(scope)]
            starts = {}
            for event in reader.events(kind="phase_started"):
                starts.setdefault(event.get("phase"), event)

            def advance(done, selected):
                if progress is not None:
                    progress(index + 1, len(jobs), camera, phase, done, selected)

            for index, (camera, phase) in enumerate(jobs):
                if cancelled and cancelled():
                    batch.stopped = True
                    break
                scope = camera, phase
                try:
                    batch.reports[scope] = _scan_scope(
                        reader, experiment_id, camera, int(phase), use_region=camera in options["regions"],
                        all_frames=all_frames, cancelled=cancelled, progress=advance, shared_rows=batch._rows,
                        row_lock=batch._row_lock, starts=starts,
                    )
                except (CaptureError, sqlite3.Error, OSError) as exc:
                    batch.errors[scope] = exc
                if scope in batch.reports and batch.reports[scope].stopped:
                    batch.stopped = True
                    break
            batch.finished_utc_ns = time.time_ns()
            return batch
        except BaseException:
            batch.close()
            raise
