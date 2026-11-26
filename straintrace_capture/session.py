"""Capture controller with bounded raw storage and a replaceable preview slot."""

from __future__ import annotations

import copy
import math
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from .cameras import Camera, CameraRegistry
from .cameras.base import HOST_NEXT_RECEIPT, Image, TriggerWaitSample, receipt_times
from .cameras.host_sampler import HostFrameSampler
from .config import EVENT_CONCURRENT_LIMIT, CaptureConfig, CaptureError, Phase, nonempty_text
from .continuation import compare_continuation_phase, compare_phase_readbacks
from .csv_trigger import CSV_TRIGGER_EVENT, CsvTriggerWatch
from .hardware_readiness import WAIT_MAX_AGE_MS, WAIT_POLL_S, hardware_readiness
from .live_pulses import LivePulseGroups
from .markers import marker_label
from .output import resolve_output_path
from .storage import ExperimentWriter, PartialFrameCommitError, TriggerEvent, frame_image_records, json_text
from .tiff_samples import TiffPayload, prepare_tiff
from .write_timing import LiveWriteTiming, TrialWriteTiming


WRITE_BATCH_FRAME_LIMIT = 16
_BUFFERED_LEDGER_BATCH_LIMIT = 8
_WRITE_BATCH_WAIT_S = .075
_BUFFERED_BATCH_WAIT_S = .045
_SMALL_BUFFERED_BATCH_WAIT_S = .020
_SMALL_BUFFERED_FRAME_BYTES = 1 << 20
_TIFF_PREFETCH_FRAME_BYTES = 2 << 20
_TIFF_PREFETCH_LIMIT = 8


@dataclass(frozen=True)
class _Pending:
    camera_id: str
    sequence: int
    phase: int
    role: str
    image: Image
    trigger: TriggerEvent | None
    context_id: str | None = None
    position: str | None = None
    tiff: Future[TiffPayload] | None = None
    event_memberships: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _WriteReceipt:
    camera_id: str
    sequence: int
    phase: int
    received_utc_ns: int
    raw_bytes: int
    trigger: TriggerEvent | None


@dataclass(frozen=True)
class _EventWrite:
    records: tuple[tuple[str, dict, str | None], ...]
    links: tuple[tuple[str, str, int, str], ...]


@dataclass
class _CameraStats:
    received: int = 0
    saved: int = 0
    pending_write: int = 0
    saved_bytes: int = 0
    intentionally_skipped: int = 0
    host_unselected_frames: int = 0
    app_unsaved: int = 0
    native_counter_gaps: int = 0
    native_counter_duplicates: int = 0
    native_counter_resets: int = 0
    hardware_identity_missing: int = 0
    reported_driver_skips: int = 0
    transport_errors: int = 0


@dataclass
class _EventWindow:
    pre_required_rounds: int
    post_required_rounds: int
    number: int = 1
    pre_receipts_ns: deque[int] = field(default_factory=deque)
    signal: dict | None = None
    pre_retained_rounds: int | None = None
    boundary_rounds: int | None = None
    post_received_rounds: int = 0
    recorded: bool = False

    @property
    def signal_id(self) -> str | None:
        return self.signal["id"] if self.signal is not None else None

    def prebuffer_ready(self, *, before_ns: int | None = None) -> bool:
        required = self.pre_required_rounds
        if len(self.pre_receipts_ns) < required:
            return False
        # Each software round finishes before the next starts. The last receipt
        # in a round has to precede the signal for the whole round to count.
        return not required or before_ns is None or self.pre_receipts_ns[required - 1] < before_ns

    def counts(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "event_number": self.number,
            "pre_required_rounds": self.pre_required_rounds,
            "pre_retained_rounds": self.pre_retained_rounds,
            "boundary_rounds": self.boundary_rounds,
            "post_required_rounds": self.post_required_rounds,
            "post_received_rounds": self.post_received_rounds,
        }

    def status(self) -> dict:
        if self.recorded:
            stage = "complete"
        elif self.pre_retained_rounds is not None:
            stage = "recording_post"
        elif self.signal_id is not None:
            stage = "trigger_queued"
        else:
            stage = "waiting" if self.prebuffer_ready() else "buffering"
        return {
            **self.counts(), "state": stage,
            "pre_buffered_rounds": min(len(self.pre_receipts_ns), self.pre_required_rounds),
        }


@dataclass
class _EventRound:
    pending: list[_Pending]
    identities: tuple[tuple[str, int], ...]
    issued_ns: int
    received_ns: int
    complete: bool

    @classmethod
    def capture(cls, items, expected_cameras):
        return cls(items, tuple((item.camera_id, item.sequence) for item in items),
                   items[0].trigger.details["issued_monotonic_ns"],
                   max(item.image.received_monotonic_ns for item in items), len(items) == expected_cameras)


@dataclass
class _OverlapEvents:
    recent: deque[_EventRound] = field(default_factory=deque)
    collecting: list[_EventWindow] = field(default_factory=list)
    accepted: int = 0
    recorded: int = 0
    closing: bool = False


@dataclass
class _RunningPhase:
    plan: Phase
    started: float
    deadline: float
    received_at_start: dict[str, int]
    event: _EventWindow | None
    host_sampling_cameras: tuple[str, ...] = ()
    quality_sensor_values: dict[str, dict] = field(default_factory=dict)
    manual_signal_id: str | None = None
    wait_samples: dict[str, tuple[TriggerWaitSample, int, float]] = field(default_factory=dict)
    first_wait_asserted: dict[str, int] = field(default_factory=dict)
    first_receipt_ns: dict[str, int] = field(default_factory=dict)
    saved_by_camera: dict[str, int] = field(default_factory=dict)
    paused_ns: int = 0
    csv_trigger: CsvTriggerWatch | None = None
    completed_events: int = 0
    busy_event_signals: int = 0
    early_event_signals: int = 0
    overlap: _OverlapEvents | None = None

    def event_progress(self):
        window = self.event
        if window is None:
            return None
        status = {**window.status(), "completed_events": self.completed_events, "event_count": self.plan.event_count,
                  "busy_signals": self.busy_event_signals, "early_signals": self.early_event_signals}
        if self.overlap is not None:
            status.update(overlap=True, active_events=self.overlap.accepted - self.overlap.recorded,
                          accepted_events=self.overlap.accepted,
                          event_target_reached=bool(self.plan.event_count and self.overlap.accepted >= self.plan.event_count))
        return status


@dataclass
class _PhaseConfirmation:
    phase: int
    confirmation_id: str
    instruction: str
    waiting_since: tuple[int, int] | None = None
    request: dict | None = None

    def record(self):
        return {"phase": self.phase, "confirmation_id": self.confirmation_id, "instruction": self.instruction}

    def status(self, stopping):
        state = ("stopping" if stopping else "arming" if self.request else
                 "waiting" if self.waiting_since else "saving_previous")
        result = {**self.record(), "state": state}
        if self.waiting_since:
            result["waiting_since_utc_ns"], result["waiting_since_monotonic_ns"] = self.waiting_since
        return result


@dataclass
class _PhasePause:
    phase: int
    pause_id: str
    requested: dict
    paused_at: tuple[int, int] | None = None
    resume_request: dict | None = None

    def record(self) -> dict:
        return {"phase": self.phase, "pause_id": self.pause_id, **self.requested}

    def status(self, stopping) -> dict:
        state = ("stopping" if stopping else "resuming" if self.resume_request else
                 "paused" if self.paused_at else "requested")
        record = {**self.record(), "state": state}
        if self.paused_at:
            record["paused_utc_ns"], record["paused_monotonic_ns"] = self.paused_at
        return record


class CaptureSession:
    """Start a new experiment, monitor it and request authorized camera actions."""

    def __init__(self, config: CaptureConfig, output: str | Path, *, registry: CameraRegistry | None = None,
                 trial_reference: dict | None = None, watch_quality: bool = False):
        self.config = CaptureConfig.from_dict(config.to_dict())
        self.output = resolve_output_path(output)
        self.registry = registry if registry is not None else CameraRegistry(cti_files=self.config.cti_files)
        self._trial_reference = copy.deepcopy(trial_reference)
        self._trial_stages = {row["phase"]: row for row in self._trial_reference["stages"]} if self._trial_reference else {}
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._producers_done = threading.Event()
        self._writer_ready = threading.Event()
        self._mutex = threading.RLock()
        self._phase_changed = threading.Condition(self._mutex)
        self._frames: queue.Queue[tuple[_Pending, ...]] = queue.Queue(maxsize=self.config.effective_queue_capacity)
        self._event_writes: queue.Queue[_EventWrite] = queue.Queue(
            maxsize=2 * (self._frames.maxsize + EVENT_CONCURRENT_LIMIT),
        )
        self._event_ledger_drained = threading.Event()
        self._event_ledger_drained.set()
        self._event_ledger_failed = False
        self._tiff_workers: ThreadPoolExecutor | None = None
        self._tiff_slots = threading.BoundedSemaphore(_TIFF_PREFETCH_LIMIT)
        self._signals: queue.Queue[dict] = queue.Queue(maxsize=EVENT_CONCURRENT_LIMIT)
        self._markers: queue.Queue[tuple[str, dict, Future[str]]] = queue.Queue(maxsize=32)
        self._quality_watch = None
        self._quality_notices = None
        self._quality_admitted: dict[str, int] = {}
        if watch_quality:
            from .image_quality_watch import ImageQualityWatch

            self._quality_watch = ImageQualityWatch()
            self._quality_notices = queue.Queue(maxsize=32)
        self._latest: dict[str, tuple[int, Image]] = {}
        self._stats = {camera.id: _CameraStats() for camera in self.config.cameras}
        self._counter: dict[str, int] = {}
        self._queue_bytes = 0
        self._queue_high_water = 0
        self._prebuffer_bytes = 0
        self._raw_high_water = 0
        self._schedule_misses = 0
        self._short_pre_windows = 0
        self._state = "idle"
        self._phase: int | None = None
        self._mode: str | None = None
        self._running_phase: _RunningPhase | None = None
        self._phase_confirmation: _PhaseConfirmation | None = None
        self._phase_pause: _PhasePause | None = None
        self._pulse_groups: LivePulseGroups | None = None
        self._started: float | None = None
        self._ended: float | None = None
        self._error: dict | None = None
        self._writer_failed = False
        self._writer_crashed = False
        self._store: ExperimentWriter | None = None
        self._live_write = LiveWriteTiming(max(WRITE_BATCH_FRAME_LIMIT, len(self.config.cameras)),
                                          self.config.max_buffer_bytes, self._frames.maxsize)
        self._trial_timing = None
        if len(self.config.phases) == 1 and "capture_trial" in self.config.metadata:
            self._trial_timing = TrialWriteTiming(max(WRITE_BATCH_FRAME_LIMIT, len(self.config.cameras)))
        self._write_timing: dict | None = None
        self._csv_triggers: list[dict] = []
        self._event_phases: list[dict] = []

    def start(self, *, stop_event: threading.Event | None = None) -> str:
        """Start the writer and cameras, with an optional stop signal from the caller."""
        with self._mutex:
            if self._state != "idle":
                raise CaptureError("ALREADY_STARTED", "a CaptureSession can be started only once")
            if stop_event is not None:
                self._stop = stop_event
            writer_options = {"trial_reference": self._trial_reference} if self._trial_reference else {}
            self._store = ExperimentWriter(self.output, self.config, **writer_options)
            self._state = "opening"
            self._started = time.perf_counter()
            threading.Thread(target=self._run, name="capture-controller").start()
            return self._store.experiment_id

    def request_stop(self) -> None:
        with self._mutex:
            if self._state in ("opening", "running"):
                self._stop.set()
                self._phase_changed.notify_all()

    @property
    def finished(self) -> bool:
        """Whether the writer has saved its terminal state and released its lease."""
        return self._finished.is_set()

    def wait(self, timeout: float | None = None) -> dict:
        if self._started is None:
            raise CaptureError("NOT_STARTED", "start the capture before waiting")
        if not self._finished.wait(timeout):
            raise TimeoutError("capture is still running")
        return self.status()

    def request_trigger(self, source: str = "operator", *, payload: dict | None = None,
                        expected_phase: int | None = None, expected_window: int | None = None,
                        require_prebuffer: bool = False) -> str:
        """Queue an incoming signal, optionally binding it to a phase or a full pre-window."""
        return self._submit_trigger(source, payload=payload, expected_phase=expected_phase,
                                    expected_window=expected_window, require_prebuffer=require_prebuffer)

    def _submit_trigger(self, source, *, payload=None, expected_phase=None, expected_window=None,
                        require_prebuffer=False, from_csv=False):
        source = nonempty_text(source, "event source")
        if payload is not None and not isinstance(payload, dict):
            raise CaptureError("SIGNAL_PAYLOAD_INVALID", "trigger metadata must be a JSON object")
        if expected_phase is not None and (type(expected_phase) is not int or expected_phase < 1):
            raise CaptureError("SIGNAL_PHASE_INVALID", "expected phase must be a positive integer")
        if expected_window is not None and (type(expected_window) is not int or expected_window < 1):
            raise CaptureError("SIGNAL_WINDOW_INVALID", "expected event window must be a positive integer")
        if type(require_prebuffer) is not bool:
            raise CaptureError("SIGNAL_PREBUFFER_INVALID", "require_prebuffer must be boolean")
        utc, monotonic = receipt_times()
        signal = {
            "id": str(uuid4()), "source": source, "payload": dict(payload or {}),
            "received_utc_ns": utc, "received_monotonic_ns": monotonic,
            **({"origin": "instrument_csv"} if from_csv else {}),
        }
        signal = copy.deepcopy(signal)
        try:
            json_text(signal)
        except (TypeError, ValueError) as exc:
            raise CaptureError("SIGNAL_PAYLOAD_INVALID", f"trigger metadata must contain finite JSON values: {exc}") from exc
        with self._mutex:
            if expected_phase is not None and self._phase != expected_phase:
                raise CaptureError("SIGNAL_PHASE_CHANGED", f"expected phase {expected_phase}, active phase is {self._phase}")
            running = self._running_phase
            window = running.event if running is not None else None
            if expected_window is not None and (window is None or window.number != expected_window):
                raise CaptureError("SIGNAL_WINDOW_CHANGED", "the requested event window is no longer active")
            if unavailable := self._trigger_unavailable(time.perf_counter()):
                if window is not None and running.plan.repeats_events:
                    if unavailable[0] == "EVENT_ALREADY_RECEIVED":
                        running.busy_event_signals += 1
                    elif unavailable[0] == "PREBUFFER_NOT_READY":
                        running.early_event_signals += 1
                raise CaptureError(*unavailable)
            if require_prebuffer or running.plan.repeats_events:
                if window is None:
                    raise CaptureError("PREBUFFER_NOT_APPLICABLE", "the active phase has no event prebuffer")
                if not window.prebuffer_ready(before_ns=monotonic):
                    if running.plan.repeats_events:
                        running.early_event_signals += 1
                    raise CaptureError("PREBUFFER_NOT_READY", "the event prebuffer was not full when the signal arrived")
            if window is not None:
                signal["event_number"] = window.number
                if running.overlap is not None:
                    signal["overlap"] = True
            try:
                self._signals.put_nowait(signal)
            except queue.Full as exc:
                raise CaptureError("SIGNAL_QUEUE_FULL", "the camera trigger queue is full") from exc
            if window is not None:
                if running.overlap is None:
                    window.signal = signal
                else:
                    running.overlap.accepted += 1
                    window.number += 1
        return signal["id"]

    def request_phase_confirmation(self, expected_phase: int, *, confirmation_id: str | None = None,
                                   source: str = "operator") -> str:
        """Release the named phase only while its confirmation is pending."""
        if type(expected_phase) is not int or expected_phase < 1:
            raise CaptureError("PHASE_CONFIRMATION_INVALID", "expected phase must be a positive integer")
        if confirmation_id is not None and (not isinstance(confirmation_id, str) or not confirmation_id.strip()):
            raise CaptureError("PHASE_CONFIRMATION_INVALID", "confirmation_id must be a nonempty string")
        source = nonempty_text(source, "confirmation source")
        with self._phase_changed:
            gate = self._phase_confirmation
            if gate is None or self._stop.is_set() or self._error is not None:
                raise CaptureError("PHASE_CONFIRMATION_NOT_PENDING", "no phase confirmation is pending")
            if expected_phase != gate.phase or (confirmation_id is not None and confirmation_id != gate.confirmation_id):
                raise CaptureError("PHASE_CONFIRMATION_CHANGED", f"the pending confirmation is for phase {gate.phase}")
            if gate.waiting_since is None:
                raise CaptureError("PHASE_CONFIRMATION_SAVING", "the preceding frames are still being committed")
            if gate.request is not None:
                raise CaptureError("PHASE_CONFIRMATION_REQUESTED", "confirmation was already requested")
            utc, monotonic = receipt_times()
            gate.request = {"source": source, "requested_utc_ns": utc, "requested_monotonic_ns": monotonic}
            self._phase_changed.notify_all()
            return gate.confirmation_id

    def request_phase_pause(self, expected_phase: int, *, source: str = "operator") -> str:
        """Suspend a software-request phase after the current camera round."""
        source = nonempty_text(source, "pause source")
        with self._phase_changed:
            self._check_pause_phase(expected_phase)
            if unavailable := self._pause_unavailable(time.perf_counter()):
                raise CaptureError(*unavailable)
            utc, mono = receipt_times()
            gate = _PhasePause(expected_phase, str(uuid4()), {
                "source": source, "requested_utc_ns": utc, "requested_monotonic_ns": mono,
            })
            self._phase_pause = gate
            self._phase_changed.notify_all()
            return gate.pause_id

    def request_phase_resume(self, expected_phase: int, pause_id: str, *, source: str = "operator") -> str:
        """Release the exact pause identified in the live phase status."""
        pause_id = nonempty_text(pause_id, "pause ID")
        source = nonempty_text(source, "resume source")
        with self._phase_changed:
            self._check_pause_phase(expected_phase)
            gate = self._phase_pause
            if gate is None or self._stop.is_set() or self._error is not None:
                raise CaptureError("PHASE_PAUSE_NOT_ACTIVE", "no phase pause is active")
            if gate.pause_id != pause_id:
                raise CaptureError("PHASE_PAUSE_CHANGED", "the phase has a different pause ID")
            if gate.paused_at is None or gate.resume_request is not None:
                raise CaptureError("PHASE_PAUSE_TRANSITION", "the phase is pausing or resuming")
            utc, mono = receipt_times()
            gate.resume_request = {"source": source, "requested_utc_ns": utc, "requested_monotonic_ns": mono}
            self._phase_changed.notify_all()
            return gate.pause_id

    def _check_pause_phase(self, expected_phase):
        if type(expected_phase) is not int or expected_phase < 1:
            raise CaptureError("PHASE_PAUSE_INVALID", "expected phase must be a positive integer")
        if self._phase != expected_phase:
            raise CaptureError("PHASE_PAUSE_CHANGED", f"expected phase {expected_phase}, active phase is {self._phase}")

    def _pause_unavailable(self, now):
        running = self._running_phase
        if (self._state != "running" or self._stop.is_set() or self._error is not None
                or running is None or now >= running.deadline):
            return "PHASE_PAUSE_NOT_ACTIVE", "the phase is not accepting new pause requests"
        if running.plan.mode not in {"timed", "manual"} or self._trial_timing is not None:
            return "PHASE_PAUSE_UNSUPPORTED", "pause requires a formal timed or manual phase"
        if self._phase_pause is not None:
            return "PHASE_PAUSE_TRANSITION", "a phase pause is already pending or active"
        if not self._signals.empty():
            return "PHASE_PAUSE_SIGNALS_QUEUED", "accepted manual trigger requests must finish before pausing"
        return None

    def _can_mark(self):
        return self._state == "running" and not self._stop.is_set() and self._error is None

    @property
    def marker_available(self):
        with self._mutex:
            return self._can_mark()

    def request_marker(self, label: str = "", *, source: str = "operator") -> tuple[str, Future[str]]:
        """Queue an observation and return its ID and eventual ledger receipt."""
        utc, monotonic = receipt_times()
        label = marker_label(label)
        source = nonempty_text(source, "marker source")
        identity = str(uuid4())
        receipt = Future()
        with self._mutex:
            if not self._can_mark():
                raise CaptureError("MARKER_NOT_RECORDING", "the experiment is not accepting markers")
            record = {
                "label": label, "source": source,
                "phase": self._phase if self._running_phase is not None else None,
                "marked_utc_ns": utc, "marked_monotonic_ns": monotonic,
                "monotonic_clock_id": f"{self._store.experiment_id}:host_monotonic",
            }
            receipt.set_running_or_notify_cancel()
            try:
                self._markers.put_nowait((identity, record, receipt))
            except queue.Full as exc:
                raise CaptureError("MARKER_QUEUE_FULL", "wait for the pending markers to be saved") from exc
            self._writer_ready.set()
        return identity, receipt

    def _trigger_unavailable(self, now: float) -> tuple[str, str] | None:
        phase = self._running_phase
        if (self._stop.is_set() or self._error is not None or self._state != "running"
            or self._mode not in ("manual", "event") or phase is None):
            return "NOT_ARMED", "the active phase does not accept external or manual signals"
        if self._phase_pause is not None:
            return "PHASE_PAUSE_TRANSITION", "resume the phase before submitting a trigger"
        if now >= phase.deadline:
            return "SIGNAL_PHASE_EXPIRED", "the active phase has reached its time limit"
        if phase.overlap is not None:
            overlap = phase.overlap
            if overlap.closing or (phase.plan.event_count and overlap.accepted >= phase.plan.event_count):
                return "EVENT_TARGET_REACHED", "the event phase has accepted all planned triggers"
            if overlap.accepted - overlap.recorded >= self._signals.maxsize:
                return "EVENT_ALREADY_RECEIVED", "the maximum number of concurrent event windows is active"
        if phase.event is not None and phase.event.signal_id is not None:
            return "EVENT_ALREADY_RECEIVED", "a signal is already queued or being recorded for this event phase"
        if phase.plan.repeats_events and not phase.event.prebuffer_ready():
            return "PREBUFFER_NOT_READY", "wait for fresh pre-event images for the next event"
        if self._signals.full():
            return "SIGNAL_QUEUE_FULL", "the camera trigger queue is full"
        return None

    def latest_image(self, camera_id: str) -> Image | None:
        sample = self.latest_sample(camera_id)
        return sample[1] if sample else None

    def latest_sample(self, camera_id: str) -> tuple[int, Image] | None:
        """Return the displayed image and its capture phase from the same receipt."""
        with self._mutex:
            return self._latest.get(camera_id)

    def status(self) -> dict:
        with self._mutex:
            now = time.perf_counter()
            elapsed = (self._ended or now) - self._started if self._started is not None else 0
            per_camera = {}
            for camera_id, stats in self._stats.items():
                per_camera[camera_id] = {
                    **asdict(stats),
                    "mean_receive_fps": stats.received / elapsed if elapsed else 0,
                    "mean_save_fps": stats.saved / elapsed if elapsed else 0,
                }
            progress = None
            pause = self._phase_pause
            if running := self._running_phase:
                received = {camera: stats.received - running.received_at_start[camera]
                            for camera, stats in self._stats.items()}
                now_ns = time.perf_counter_ns()
                accepting = self._state == "running" and not self._stop.is_set() and now < running.deadline and pause is None
                phase_now = pause.paused_at[1] / 1e9 if pause and pause.paused_at else now
                progress = {
                    "elapsed_s": max(0., phase_now - running.started - running.paused_ns / 1e9),
                    "duration_s": running.plan.duration_s,
                    "frame_limit_per_camera": running.plan.frame_count,
                    "planned_fps": running.plan.fps,
                    "accepting_images": accepting,
                    "pause_available": self._pause_unavailable(now) is None,
                    "received_by_camera": received,
                    "camera_receipts": self._phase_camera_receipts(running, received, now_ns),
                    "trigger_available": self._trigger_unavailable(now) is None,
                    "queued_signals": self._signals.qsize(),
                    "manual_signal_id": running.manual_signal_id,
                    "host_sampling_cameras": list(running.host_sampling_cameras),
                    "event_window": running.event_progress(),
                }
                if running.plan.mode == "hardware":
                    progress["hardware_inputs"] = {camera: running.plan.hardware_input(camera) for camera in received}
                if running.plan.monitor_trigger_wait:
                    progress["hardware_readiness"] = hardware_readiness(
                        running.wait_samples, received, now_ns=now_ns, active=accepting,
                    )
                if running.csv_trigger is not None:
                    progress["csv_trigger"] = running.csv_trigger.status()
            queued = self._frames.qsize()
            self._observe_write_timing(now, queued)
            buffered = self._queue_bytes + self._prebuffer_bytes
            result = {
                "state": self._state,
                "experiment_id": self._store.experiment_id if self._store else None,
                **({"csv_triggers": copy.deepcopy(self._csv_triggers)} if self._csv_triggers else {}),
                **({"event_phases": copy.deepcopy(self._event_phases)} if self._event_phases else {}),
                "phase": self._phase,
                "mode": self._mode,
                "phase_progress": progress,
                "elapsed_s": elapsed,
                "queue_size": queued,
                "queue_capacity": self._frames.maxsize,
                "queue_mode": "auto" if self.config.queue_capacity == "auto" else "fixed",
                "queue_high_water": self._queue_high_water,
                "buffered_raw_bytes": buffered,
                "raw_buffer_high_water_bytes": self._raw_high_water,
                "raw_buffer_limit_bytes": self.config.max_buffer_bytes,
                "scheduled_misses": self._schedule_misses,
                "short_pre_windows": self._short_pre_windows,
                "disk_bytes_per_s": sum(item.saved_bytes for item in self._stats.values()) / elapsed if elapsed else 0,
                "disk_free_bytes": self._store.disk_free_bytes if self._store else None,
                "cameras": per_camera,
                "error": dict(self._error) if self._error else None,
                "write_pressure": self._live_write.status(
                    running=bool(progress and progress["accepting_images"]), failed=bool(self._error),
                    pending=self._queue_bytes, buffered=buffered, queued=queued,
                ),
            }
            if alerts := self._live_write.alerts():
                result["write_alerts"] = alerts
            if self._phase_confirmation is not None:
                result["phase_confirmation"] = self._phase_confirmation.status(self._stop.is_set())
            if pause is not None:
                result["phase_pause"] = pause.status(self._stop.is_set())
            if pulses := self._pulse_groups:
                result["hardware_groups"] = pulses.snapshot(
                    accepting=bool(progress and self._phase == pulses.phase and progress["accepting_images"]),
                )
            if self._write_timing is not None:
                result["write_timing"] = copy.deepcopy(self._write_timing)
            if self._quality_watch is not None:
                result["image_quality_watch"] = self._quality_watch.snapshot()
        return result

    def _observe_write_timing(self, now, queued):
        if (running := self._running_phase) is not None and not self._stop.is_set() and now < running.deadline:
            self._live_write.observe(now, pending=self._queue_bytes, buffered=self._queue_bytes + self._prebuffer_bytes,
                                     queued=queued)

    def _phase_camera_receipts(self, running, received, now_ns):
        result = {}
        for camera, count in received.items():
            last = self._latest[camera][1].received_monotonic_ns if count else None
            first = running.first_receipt_ns.get(camera)
            span = last - first if last is not None and first is not None else 0
            result[camera] = {
                "saved": running.saved_by_camera.get(camera, 0),
                "last_receipt_age_s": None if last is None else max(0, now_ns - last) / 1e9,
                "host_receive_fps": (count - 1) * 1e9 / span if count > 1 and span > 0 else None,
            }
        return result

    def _fail(self, exc: Exception, fallback: str, *, frame: _WriteReceipt | None = None, phase: int | None = None) -> None:
        code = exc.code if isinstance(exc, CaptureError) else fallback
        with self._mutex:
            if self._error is None:
                self._error = {"code": code, "message": str(exc)}
            if phase is None:
                phase = self._phase
        self._stop.set()
        details = {"code": code, "message": str(exc), "phase": phase}
        if frame is not None:
            details.update(phase=frame.phase, camera_id=frame.camera_id, sequence=frame.sequence,
                           frame_received_utc_ns=frame.received_utc_ns)
        try:
            self._store.event("error", details)
        except Exception:
            # The failure may itself be a full disk or unavailable journal.
            # In that case the last committed ledger and in-memory status survive.
            pass

    def _run(self) -> None:
        tiff_workers = None
        if self.config.durability == "buffered":
            tiff_workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tiff-checksum")
        self._tiff_workers = tiff_workers
        writer = threading.Thread(target=self._write_guarded, name="capture-disk")
        writer_started = False
        quality_stop = threading.Event()
        quality_worker = None
        try:
            try:
                writer.start()
                writer_started = True
            except RuntimeError as exc:
                self._writer_crashed = True
                raise CaptureError("WRITER_THREAD_FAILED", f"could not start capture writer: {exc}") from exc
            if self._quality_watch is not None:
                quality_worker = threading.Thread(target=self._sample_image_quality, args=(quality_stop,),
                                                  name="capture-image-quality")
                try:
                    quality_worker.start()
                except RuntimeError:
                    with self._mutex:
                        self._quality_watch.measurement_errors += 1
                    quality_worker = None
            with ExitStack() as stack:
                cameras = []
                for configured in self.config.cameras:
                    if self._stop.is_set():
                        break
                    camera = stack.enter_context(self.registry.camera(configured.id))
                    if camera.device.id != configured.id:
                        raise CaptureError("CAMERA_ID_MISMATCH", f"expected {configured.id}, found {camera.device.id}")
                    camera.set_features(configured.settings)
                    self._store.register_camera(camera.device, {name: asdict(item) for name, item in camera.features().items()})
                    cameras.append(camera)
                if not self._stop.is_set():
                    with self._mutex:
                        self._state = "running"
                    self._store.set_state("running")
                    with ThreadPoolExecutor(max_workers=len(cameras), thread_name_prefix="camera-read") as executor:
                        for index, phase in enumerate(self.config.phases, start=1):
                            if self._stop.is_set():
                                break
                            try:
                                if phase.start_confirmation and not self._wait_for_phase_confirmation(index, phase.start_confirmation):
                                    break
                                self._run_phase(executor, cameras, phase, index)
                            except Exception as exc:
                                self._fail(exc, "CAMERA_ERROR", phase=index)
                                break
        except Exception as exc:
            self._fail(exc, "CAMERA_ERROR")
        finally:
            with self._mutex:
                self._state = "draining"
                self._phase_confirmation = None
            if quality_worker is not None:
                quality_stop.set()
                quality_worker.join()
                with self._mutex:
                    self._quality_watch.deferred = False
            self._producers_done.set()
            self._writer_ready.set()
            if writer_started:
                writer.join()
            if tiff_workers is not None:
                tiff_workers.shutdown(wait=True, cancel_futures=True)
                self._tiff_workers = None
            if self._writer_crashed:
                self._reconcile_crashed_writer()
            with self._mutex:
                self._ended = time.perf_counter()
                if self._trial_timing is not None:
                    self._write_timing = self._trial_timing.finish(self._ended)
                self._state = "failed" if self._error else ("stopped" if self._stop.is_set() else "completed")
                self._phase = self._mode = None
            try:
                self._store.set_state(self._state, self.status())
            except Exception as exc:
                self._fail(exc, "LEDGER_FINALIZE_FAILED")
                with self._mutex:
                    self._state = "failed"
            finally:
                try:
                    self._store.close()
                finally:
                    self._finished.set()

    def _wait_for_phase_confirmation(self, index: int, instruction: str) -> bool:
        gate = _PhaseConfirmation(index, str(uuid4()), instruction)
        with self._phase_changed:
            self._phase_confirmation = gate
            while any(row.pending_write for row in self._stats.values()) and not self._stop.is_set():
                self._phase_changed.wait(.1)
        if not self._stop.is_set():
            utc, monotonic = receipt_times()
            self._confirmation_event("phase_confirmation_waiting", {
                **gate.record(), "waiting_since_utc_ns": utc, "waiting_since_monotonic_ns": monotonic,
            })
            with self._phase_changed:
                gate.waiting_since = (utc, monotonic)
                while gate.request is None and not self._stop.is_set():
                    self._phase_changed.wait(.1)
        request = gate.request
        if request is not None and not self._stop.is_set():
            self._confirmation_event("phase_confirmed", {**gate.record(), **request})
            if not self._stop.is_set():
                return True
        self._confirmation_event("phase_confirmation_cancelled", {
            **gate.record(), "confirmation_requested": request is not None,
        })
        return False

    def _confirmation_event(self, kind, record):
        try:
            self._store.event(kind, record)
        except Exception as exc:
            raise CaptureError("PHASE_CONFIRMATION_LEDGER_FAILED", f"phase {record['phase']}: {exc}") from exc

    def _run_phase(self, executor: ThreadPoolExecutor, cameras: list[Camera], phase: Phase, index: int) -> None:
        trigger = phase.camera_trigger
        sources = []
        with ExitStack() as stack:
            wait_monitors = ({camera.device.id: stack.enter_context(camera.trigger_wait_monitor()) for camera in cameras}
                             if phase.monitor_trigger_wait else {})
            for camera in cameras:
                stack.callback(camera.stop)
                host_sampling = (trigger == "software" and phase.software_trigger_fallback
                                 and camera.trigger_support().software is False)
                camera.start("free" if host_sampling else trigger, phase.hardware_input(camera.device.id))
                if host_sampling:
                    sources.append(HostFrameSampler(
                        camera,
                        lambda image, chosen, key=camera.device.id: self._observe_host_frame(key, index, image, chosen),
                        lambda exc: self._fail(exc, "HOST_FRAME_READ_FAILED"),
                    ))
                else:
                    sources.append(camera)
            camera_methods = {item.device.id: item.trigger_method for item in sources} if trigger == "software" else {}
            confirmation = {}
            if gate := self._phase_confirmation:
                utc, monotonic = receipt_times()
                confirmation = {"confirmation_id": gate.confirmation_id, "camera_start_returned_utc_ns": utc,
                                "camera_start_returned_monotonic_ns": monotonic}
            readbacks = {
                camera.device.id: {name: asdict(item) for name, item in camera.features().items()}
                for camera in cameras
            }
            devices = {camera.device.id: camera.device for camera in cameras}
            comparison = compare_continuation_phase(self.config, index, readbacks, devices)
            trial_check = {}
            if trial := self._trial_stages.get(index):
                observed = trial["readbacks"]
                trial_check = {"trial_setup_check": {"trial_experiment_id": trial["experiment_id"], **compare_phase_readbacks(
                    observed.keys(), observed, trial["device_identities"], readbacks, devices,
                )}}
            self._store.event("phase_started", {
                "phase": index, "plan": phase.to_dict(),
                **confirmation,
                "camera_methods": camera_methods,
                "camera_features": readbacks,
                **({"continuation_check": comparison} if comparison is not None else {}),
                **trial_check,
            })
            csv_watch = None
            if phase.csv_trigger is not None:
                csv_watch = CsvTriggerWatch(phase.csv_trigger, lambda source, payload: self._submit_trigger(
                    source, payload=payload, expected_phase=index, require_prebuffer=True, from_csv=True,
                ), repeat=phase.repeats_events)
            with self._mutex:
                self._phase_confirmation = None
                self._phase, self._mode = index, phase.mode
                self._counter.clear()
                started = time.perf_counter()
                started_utc_ns = time.time_ns()
                if self._trial_timing is not None:
                    self._trial_timing.begin(started, sample_rates=phase.mode in ("continuous", "timed"))
                deadline = started + (phase.duration_s if phase.duration_s is not None else math.inf)
                window = None
                if phase.mode == "event":
                    window = _EventWindow(phase.pre_frames, phase.post_frames)
                running = self._running_phase = _RunningPhase(
                    phase, started, deadline, {key: item.received for key, item in self._stats.items()}, window,
                    tuple(camera for camera, method in camera_methods.items() if method == HOST_NEXT_RECEIPT),
                    quality_sensor_values={key: {name: field["value"] for name, field in readbacks[key].items()}
                                           for key in self.config.quality_regions if self._quality_watch is not None},
                    csv_trigger=csv_watch,
                    overlap=_OverlapEvents() if phase.overlap_events else None,
                )
                if phase.hardware_pairing is not None:
                    self._pulse_groups = LivePulseGroups(index, phase.hardware_pairing)
                self._live_write.begin(index, phase.mode, started)
                self._observe_write_timing(started, self._frames.qsize())
            try:
                if trigger == "software":
                    with ExitStack() as readers:
                        for item in sources:
                            if isinstance(item, HostFrameSampler):
                                readers.enter_context(item)
                        if csv_watch is not None:
                            readers.enter_context(csv_watch)
                        self._software_phase(executor, sources, running, index, started_utc_ns)
                else:
                    futures = [executor.submit(self._stream, camera, phase, index, deadline, wait_monitors.get(camera.device.id))
                               for camera in cameras]
                    for future in futures:
                        future.result()
            except Exception as exc:
                if window is None:
                    raise
                # Publish the failure in the live status before draining writes.
                self._fail(exc, "CAMERA_ERROR", phase=index)
                return
            finally:
                csv_record = None
                if csv_watch is not None:
                    csv_record = {"phase": index, **csv_watch.status()}
                    if phase.repeats_events:
                        csv_record.pop("crossing", None)
                        csv_record.pop("signal_id", None)
                    elif crossing := csv_record.get("crossing"):
                        csv_record["detected_utc_ns"] = crossing["detected_utc_ns"]
                window = running.event
                with self._mutex:
                    if self._trial_timing is not None:
                        self._trial_timing.end(time.perf_counter())
                    self._live_write.end()
                    self._mode = None
                    self._running_phase = None
                    if csv_record is not None:
                        self._csv_triggers.append(csv_record)
                    pause, self._phase_pause = self._phase_pause, None
                    unused_signals = []
                    while not self._signals.empty():
                        unused_signals.append(self._signals.get_nowait())
                if pause is not None:
                    self._close_phase_pause(running, pause)
                for signal in unused_signals:
                    self._store.event("signal_unused", {"phase": index, **signal}, event_id=signal["id"])
                if window is not None and not window.recorded and (window.signal is not None or not phase.repeats_events):
                    self._record_event_windows(running, index, (window,))
                if window is not None:
                    self._event_ledger_drained.wait()
                if csv_record is not None:
                    self._store.event(CSV_TRIGGER_EVENT, csv_record)
                if phase.repeats_events:
                    partial_events = (running.overlap.accepted - running.completed_events if running.overlap is not None
                                      else int(window.signal is not None and window.number > running.completed_events))
                    summary = {"phase": index, "event_count": phase.event_count, "completed_events": running.completed_events,
                               "partial_events": partial_events,
                               "busy_signals": running.busy_event_signals, "early_signals": running.early_event_signals}
                    if running.overlap is not None:
                        summary.update(overlap=True, accepted_events=running.overlap.accepted)
                    self._store.event("event_phase_summary", summary)
                    with self._mutex:
                        self._event_phases.append(summary)
                if phase.monitor_trigger_wait:
                    self._store.event("hardware_wait_observations", {
                        "phase": index, "basis": "camera_frame_trigger_wait_output",
                        "received_by_camera": {key: stats.received - running.received_at_start[key] for key, stats in self._stats.items()},
                        "cameras": {key: {**asdict(sample), "poll_started_monotonic_ns": stamp, "poll_ms": elapsed,
                                          "first_asserted_poll_monotonic_ns": running.first_wait_asserted.get(key)}
                                    for key, (sample, stamp, elapsed) in running.wait_samples.items()},
                    })
        self._store.event("phase_ended", {
            "phase": index, "stop_requested": self._stop.is_set(),
            **({"paused_duration_ns": running.paused_ns} if running.paused_ns else {}),
        })

    def _native_counter_change(self, camera_id, index, image, stats):
        # The caller holds the session lock. Sampled streams call this for
        # every received image so planned selection cannot appear as gaps.
        stats.reported_driver_skips += int(image.metadata.get("pylon_skipped_images", 0))
        counter = image.native_counter
        if counter is None:
            return None
        last = self._counter.get(camera_id)
        self._counter[camera_id] = counter
        if last is None or counter == last + 1:
            return None
        detail = {"camera_id": camera_id, "phase": index, "previous": last, "current": counter,
                  "received_utc_ns": image.received_utc_ns}
        if counter > last + 1:
            stats.native_counter_gaps += counter - last - 1
            return "native_gap_unattributed", detail
        if counter == last:
            stats.native_counter_duplicates += 1
            return "native_duplicate", detail
        stats.native_counter_resets += 1
        return "native_reset_or_wrap", detail

    def _observe_host_frame(self, camera_id, index, image, chosen):
        with self._mutex:
            stats = self._stats[camera_id]
            stats.host_unselected_frames += not chosen
            change = self._native_counter_change(camera_id, index, image, stats)
        if change:
            self._diagnostic_event(*change)

    def _record_image(self, camera: Camera | HostFrameSampler, phase: Phase, index: int,
                      image: Image, trigger: TriggerEvent | None) -> _Pending:
        camera_id = camera.device.id
        discontinuity = None
        with self._mutex:
            stats = self._stats[camera_id]
            stats.received += 1
            sequence = stats.received
            self._latest[camera_id] = (index, image)
            if self._running_phase:
                self._running_phase.first_receipt_ns.setdefault(camera_id, image.received_monotonic_ns)
            if (pulses := self._pulse_groups) is not None and pulses.phase == index:
                pulses.receive(camera_id, trigger.details["external_event_number"] if trigger else None)
            if not isinstance(camera, HostFrameSampler):
                discontinuity = self._native_counter_change(camera_id, index, image, stats)
        if discontinuity:
            self._diagnostic_event(*discontinuity)
        return _Pending(camera_id, sequence, index, phase.role, image, trigger)

    def _diagnostic_event(self, kind, details, *, event_id=None, once=False):
        try:
            self._store.event(kind, details, event_id=event_id, once=once)
        except Exception as exc:
            self._fail(CaptureError("DIAGNOSTIC_LEDGER_WRITE_FAILED", f"{kind}: {exc}"),
                       "DIAGNOSTIC_LEDGER_WRITE_FAILED", phase=details.get("phase"))

    def _enqueue(self, item: _Pending) -> None:
        self._enqueue_batch((item,))

    def _prefetch_tiff(self, item: _Pending) -> _Pending:
        image = item.image
        workers = self._tiff_workers
        # These layouts can be hashed directly from the already-accounted sensor buffer.
        if (workers is None or image.pixels.nbytes < _TIFF_PREFETCH_FRAME_BYTES or not image.pixels.flags.c_contiguous
                or image.color_model in ("bgr", "bgra") or not self._tiff_slots.acquire(blocking=False)):
            return item
        try:
            job = workers.submit(prepare_tiff, image)
        except RuntimeError:
            self._tiff_slots.release()
            self._tiff_workers = None
            return item
        job.add_done_callback(lambda _job: self._tiff_slots.release())
        return replace(item, tiff=job)

    def _enqueue_batch(self, items: tuple[_Pending, ...]) -> None:
        if not items:
            return
        size = sum(item.image.raw_bytes for item in items)
        with self._mutex:
            if self._writer_failed:
                for item in items:
                    self._stats[item.camera_id].app_unsaved += 1
                return
            try:
                buffered = self._queue_bytes + self._prebuffer_bytes
                if buffered + size > self.config.max_buffer_bytes:
                    raise CaptureError("RAW_BUFFER_FULL",
                                       f"{buffered} buffered bytes + {size} incoming bytes exceed {self.config.max_buffer_bytes}")
                if self._tiff_workers is not None and not self._frames.full():
                    # Producers hold the session lock. The disk thread can only make more queue space here.
                    items = tuple(self._prefetch_tiff(item) for item in items)
                self._frames.put_nowait(items)
                self._writer_ready.set()
            except (queue.Full, CaptureError) as exc:
                for item in items:
                    self._stats[item.camera_id].app_unsaved += 1
                first = items[0]
                identity = f"{first.camera_id} sequence {first.sequence}"
                if isinstance(exc, queue.Full):
                    self._queue_high_water = self._frames.maxsize
                    raise CaptureError("WRITE_QUEUE_FULL",
                                       f"write queue reached {self._frames.maxsize} batches at {identity}") from exc
                raise CaptureError(exc.code, f"{identity}: {exc}") from exc
            self._queue_bytes += size
            for item in items:
                self._stats[item.camera_id].pending_write += 1
                if self._quality_watch is not None:
                    admitted = self._quality_admitted
                    admitted[item.camera_id] = max(item.sequence, admitted.get(item.camera_id, 0))
            queued = self._frames.qsize()
            now = time.perf_counter()
            self._live_write.accept(size, len(items))
            self._observe_write_timing(now, queued)
            if self._trial_timing is not None:
                self._trial_timing.accept(now, size, len(items))
            self._queue_high_water = max(self._queue_high_water, queued)
            self._raw_high_water = max(self._raw_high_water, self._queue_bytes + self._prebuffer_bytes)

    def _write_guarded(self) -> None:
        try:
            self._write()
        except Exception as exc:
            self._signal_writer_failure(exc)

    def _signal_writer_failure(self, exc):
        with self._mutex:
            if self._writer_crashed:
                return
            self._writer_crashed = self._writer_failed = self._event_ledger_failed = True
            self._stop.set()
        # Release waiters before trying to log the error: the ledger itself may
        # still be finishing a previous batch while holding the database lock.
        self._event_ledger_drained.set()
        while True:
            try:
                _identity, _record, receipt = self._markers.get_nowait()
            except queue.Empty:
                break
            receipt.set_exception(CaptureError("MARKER_WRITE_FAILED", "capture writer stopped unexpectedly"))
        self._fail(exc, "WRITER_THREAD_FAILED")

    def _reconcile_crashed_writer(self):
        # A ledger task may have committed while the disk thread was unwinding.
        # Read its authoritative totals before classifying outstanding frames.
        try:
            totals = self._store.committed_frame_totals()
        except Exception as exc:
            self._fail(exc, "WRITER_RECONCILIATION_FAILED")
            totals = None
        with self._mutex:
            if totals is not None:
                for camera_id, stats in self._stats.items():
                    count, size = totals.get(camera_id, (0, 0))
                    difference = count - stats.saved
                    if 0 <= difference <= stats.pending_write + stats.app_unsaved:
                        stats.app_unsaved += stats.pending_write - difference
                        stats.pending_write = 0
                        stats.saved, stats.saved_bytes = count, size
            while True:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    break
            self._queue_bytes = 0

    def _write(self) -> None:
        with ExitStack() as stack:
            ledger = (stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="capture-ledger"))
                      if self.config.durability == "buffered" else None)
            try:
                self._write_batches(ledger)
            except Exception as exc:
                # Stop producers before ExitStack waits for any in-flight commit.
                self._signal_writer_failure(exc)
                raise

    def _write_batches(self, ledger) -> None:
        pending: deque[_Pending] = deque()
        buffered = ledger is not None
        batch_wait_s = _BUFFERED_BATCH_WAIT_S if buffered else _WRITE_BATCH_WAIT_S
        ledger_writes: deque[Future[bool]] = deque()
        commit_allowed = True

        def finish_ledger_batch():
            nonlocal commit_allowed
            commit_allowed = ledger_writes.popleft().result() and commit_allowed

        while pending or not (self._producers_done.is_set() and self._frames.empty() and self._markers.empty()
                              and self._event_writes.empty() and not self._quality_notices_pending()):
            self._write_event_records()
            self._write_markers()
            self._write_quality_notices()
            if not pending and not self._take_work(pending, None):
                continue
            start = time.perf_counter()
            prepared, items = [], []
            small_batch = buffered
            while len(items) < WRITE_BATCH_FRAME_LIMIT:
                if not pending:
                    wait_s = _SMALL_BUFFERED_BATCH_WAIT_S if small_batch else batch_wait_s
                    remaining = wait_s - (time.perf_counter() - start)
                    if (not small_batch and remaining <= 0) or not self._take_work(pending, remaining):
                        break
                source = pending.popleft()
                item = _WriteReceipt(source.camera_id, source.sequence, source.phase, source.image.received_utc_ns,
                                     source.image.raw_bytes, source.trigger)
                items.append(item)
                small_batch = small_batch and item.raw_bytes <= _SMALL_BUFFERED_FRAME_BYTES
                try:
                    if row := self._stage_pending_frame(source):
                        prepared.append(row)
                except Exception as exc:
                    self._write_failure(exc, item)
                finally:
                    del source
                    with self._mutex:
                        self._queue_bytes -= item.raw_bytes
            while ledger_writes and ledger_writes[0].done():
                finish_ledger_batch()
            if ledger is not None and not small_batch and prepared and commit_allowed:
                previous = ledger_writes[-1] if ledger_writes else None
                ledger_writes.append(ledger.submit(self._commit_queued_write_batch, previous, items, prepared))
                if len(ledger_writes) >= _BUFFERED_LEDGER_BATCH_LIMIT:
                    finish_ledger_batch()
                if self._store.frame_journal_needs_checkpoint():
                    while ledger_writes:
                        finish_ledger_batch()
            else:
                while ledger_writes:
                    finish_ledger_batch()
                if not commit_allowed:
                    prepared.clear()
                commit_allowed = self._commit_write_batch(items, prepared) and commit_allowed
        while ledger_writes:
            finish_ledger_batch()

    def _stage_pending_frame(self, item):
        if self._writer_failed:
            if item.tiff is not None and not item.tiff.cancel():
                with suppress(Exception):
                    item.tiff.result()
            return None
        options = {} if item.tiff is None else {"payload": item.tiff.result()}
        if item.event_memberships:
            options["event_memberships"] = item.event_memberships
        return self._store.prepare_frame(
            item.camera_id, item.sequence, item.phase, item.role, item.image,
            item.trigger, item.context_id, item.position, self.config.min_free_bytes, **options,
        )

    def _commit_queued_write_batch(self, previous: Future[bool] | None, items, prepared) -> bool:
        if previous is not None and not previous.result():
            # The journal still owns these TIFFs. Leave them for recovery in order.
            self._account_writes(items, ())
            return False
        return self._commit_write_batch(items, prepared)

    def _commit_write_batch(self, items, prepared):
        success = True
        try:
            if prepared:
                self._store.commit_frames(prepared)
        except PartialFrameCommitError as exc:
            self._write_failure(exc, items[exc.committed_count])
            del prepared[exc.committed_count:]
            success = False
        except Exception as exc:
            self._write_failure(exc, items[0])
            prepared.clear()
            success = False
        self._account_writes(items, prepared)
        return success

    def _queue_event_records(self, records, *, links=()):
        if not records and not links:
            return
        batch = _EventWrite(tuple(records), tuple(links))
        with self._mutex:
            if self._event_ledger_failed:
                return
            try:
                self._event_writes.put_nowait(batch)
            except queue.Full as exc:
                raise CaptureError("EVENT_LEDGER_QUEUE_FULL", "the event ledger write queue is full") from exc
            self._event_ledger_drained.clear()
            self._writer_ready.set()

    def _write_event_records(self):
        with self._mutex:
            batches = [self._event_writes.get_nowait() for _ in range(self._event_writes.qsize())]
        if not batches:
            return
        try:
            if not self._event_ledger_failed:
                self._store.record_capture_events(
                    (record for batch in batches for record in batch.records),
                    links=tuple(link for batch in batches for link in batch.links),
                )
        except Exception as exc:
            with self._mutex:
                self._event_ledger_failed = self._writer_failed = True
            self._fail(exc, "EVENT_LEDGER_WRITE_FAILED")
        finally:
            with self._mutex:
                if self._event_writes.empty():
                    self._event_ledger_drained.set()

    def _write_markers(self):
        batch = []
        for _ in range(self._markers.maxsize):
            try:
                batch.append(self._markers.get_nowait())
            except queue.Empty:
                break
        if not batch:
            return
        try:
            self._store.record_markers((identity, record) for identity, record, _receipt in batch)
        except Exception as exc:
            for _identity, _record, receipt in batch:
                receipt.set_exception(CaptureError("MARKER_WRITE_FAILED", str(exc)))
        else:
            for identity, _record, receipt in batch:
                receipt.set_result(identity)

    def _quality_notices_pending(self):
        return self._quality_notices is not None and not self._quality_notices.empty()

    def _write_quality_notices(self):
        if not self._quality_notices_pending():
            return
        from .image_quality_watch import NOTICE_EVENT

        # Keep the time before the next frame batch bounded under load.
        for _ in range(2):
            try:
                record = self._quality_notices.get_nowait()
            except queue.Empty:
                return
            try:
                self._store.event(NOTICE_EVENT, record)
            except Exception:
                with self._mutex:
                    self._quality_watch.ledger_errors += 1

    def _next_quality_sample(self, cameras, examined):
        with self._mutex:
            watch = self._quality_watch
            watch.deferred = False
            running = self._running_phase
            if running is None or self._stop.is_set() or self._phase_pause is not None:
                return None
            if (2 * (self._queue_bytes + self._prebuffer_bytes) >= self.config.max_buffer_bytes
                    or 2 * self._frames.qsize() >= self._frames.maxsize):
                watch.deferred = True
                watch.deferred_intervals += 1
                return None
            for _ in range(len(cameras)):
                camera = cameras[0]
                cameras.rotate(-1)
                recent = self._latest.get(camera)
                sequence = self._stats[camera].received
                if (recent is not None and recent[0] == self._phase
                        and self._quality_admitted.get(camera) == sequence
                        and examined.get(camera) != sequence):
                    examined[camera] = sequence
                    return camera, recent[0], sequence, recent[1], running
        return None

    def _sample_image_quality(self, stop):
        from .image_quality_watch import SAMPLE_INTERVAL_S, measure_primary_image

        cameras = deque(self._stats)
        examined = {}
        while not stop.wait(SAMPLE_INTERVAL_S):
            sample = self._next_quality_sample(cameras, examined)
            if sample is not None:
                try:
                    self._evaluate_quality_sample(sample, measure_primary_image)
                except Exception:
                    with self._mutex:
                        self._quality_watch.measurement_errors += 1
                finally:
                    del sample

    def _evaluate_quality_sample(self, sample, measure_primary_image):
        camera, phase, sequence, image, running = sample
        scope = None
        roi = None
        if region := self.config.quality_regions.get(camera):
            from .image_quality_watch import quality_region_mismatch

            reason = quality_region_mismatch(region, image, running.quality_sensor_values.get(camera, {}))
            if reason is None:
                roi = region.bounds
            scope = {"bounds": list(region.bounds), "fallback": reason}
        measured = measure_primary_image(image, roi=roi) if roi is not None else measure_primary_image(image)
        with self._mutex:
            if self._running_phase is not running or self._stop.is_set() or self._phase_pause is not None:
                return
            notice = self._quality_watch.update(
                camera, phase, sequence, image, measured, compare_sharpness=running.plan.role == "test", region=scope,
            )
            if notice is not None:
                try:
                    self._quality_notices.put_nowait(notice)
                except queue.Full:
                    self._quality_watch.ledger_errors += 1
                else:
                    self._writer_ready.set()

    def _take_work(self, pending: deque[_Pending], timeout: float | None) -> bool:
        deadline = None if timeout is None else time.perf_counter() + timeout
        while True:
            # Clear before checking the queues so work arriving during the checks wakes the writer.
            self._writer_ready.clear()
            with self._mutex:
                # Admission uses the same lock. The writer must commit a signal
                # before staging any TIFF that references it.
                if not self._event_writes.empty():
                    return False
                try:
                    pending.extend(self._frames.get_nowait())
                    return True
                except queue.Empty:
                    if (self._producers_done.is_set() or not self._markers.empty()
                            or (timeout is None and self._quality_notices_pending())):
                        return False
            if deadline is None:
                self._writer_ready.wait()
            else:
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or not self._writer_ready.wait(remaining):
                    return False

    def _write_failure(self, exc, item):
        with self._mutex:
            self._writer_failed = True
        prefix = f"{item.camera_id} sequence {item.sequence}"
        error = CaptureError(exc.code if isinstance(exc, CaptureError) else "WRITE_FAILED", f"{prefix}: {exc}")
        self._fail(error, "WRITE_FAILED", frame=item)

    def _account_writes(self, items, prepared):
        byte_counts = {
            (row.record["camera_id"], row.record["sequence"]): sum(asset["file_bytes"] for asset in frame_image_records(row.record))
            for row in prepared
        }
        with self._mutex:
            committed_bytes = 0
            for item in items:
                stats = self._stats[item.camera_id]
                stats.pending_write -= 1
                if size := byte_counts.get((item.camera_id, item.sequence)):
                    stats.saved += 1
                    stats.saved_bytes += size
                    committed_bytes += item.raw_bytes
                    if self._running_phase and self._phase == item.phase:
                        counts = self._running_phase.saved_by_camera
                        counts[item.camera_id] = counts.get(item.camera_id, 0) + 1
                    if (pulses := self._pulse_groups) is not None and pulses.phase == item.phase and item.trigger:
                        pulses.commit(item.camera_id, item.trigger.details["external_event_number"])
                else:
                    stats.app_unsaved += 1
            if self._trial_timing is not None:
                self._trial_timing.commit(committed_bytes, len(byte_counts))
            self._live_write.commit(committed_bytes, len(byte_counts))
            if self._running_phase is not None:
                self._observe_write_timing(time.perf_counter(), self._frames.qsize())
            if self._phase_confirmation is not None and not any(row.pending_write for row in self._stats.values()):
                self._phase_changed.notify_all()

    def _stream(self, camera: Camera, phase: Phase, index: int, deadline: float, wait_monitor=None) -> None:
        count = 0
        last = time.perf_counter()
        next_poll = last
        try:
            while not self._stop.is_set() and (phase.frame_count is None or count < phase.frame_count):
                now = time.perf_counter()
                if wait_monitor is not None and not count and now >= next_poll and now < deadline:
                    sample_started = time.perf_counter_ns()
                    try:
                        sample = wait_monitor()
                    except Exception:
                        sample = TriggerWaitSample(reason="line_status_unreadable")
                    elapsed_ms = (time.perf_counter_ns() - sample_started) / 1e6
                    now = time.perf_counter()
                    next_poll = now + WAIT_POLL_S
                    with self._mutex:
                        running = self._running_phase
                        running.wait_samples[camera.device.id] = sample, sample_started, elapsed_ms
                        if sample.asserted is True and elapsed_ms <= WAIT_MAX_AGE_MS:
                            running.first_wait_asserted.setdefault(camera.device.id, sample_started)
                remaining = deadline - now
                if remaining <= 0:
                    break
                image = camera.read(min(0.2, phase.frame_timeout_s, remaining))
                if image is None:
                    if phase.mode == "continuous" and time.perf_counter() - last >= phase.frame_timeout_s:
                        raise CaptureError("FRAME_TIMEOUT", f"no frame from {camera.device.id}")
                    continue
                last = time.perf_counter()
                event = self._hardware_event(camera.device.id, phase, index, image)
                self._enqueue(self._record_image(camera, phase, index, image, event))
                count += 1
        except Exception as exc:
            if isinstance(exc, CaptureError) and exc.code in ("GENTL_INCOMPLETE_BUFFER", "PYLON_GRAB_FAILED"):
                with self._mutex:
                    self._stats[camera.device.id].transport_errors += 1
            self._fail(exc, "CAMERA_ERROR")

    def _hardware_event(self, camera_id, phase, index, image):
        pairing = phase.hardware_pairing
        if not pairing:
            return None
        value = image.metadata.get(pairing.counter_feature)
        anchor = pairing.first_counter[camera_id]
        if type(value) is not int or value < anchor:
            with self._mutex:
                stats = self._stats[camera_id]
                stats.hardware_identity_missing += 1
                first = stats.hardware_identity_missing == 1
            if first:
                self._diagnostic_event("hardware_identity_unavailable", {
                    "camera_id": camera_id, "phase": index, "counter_feature": pairing.counter_feature,
                    "observed": value, "first_counter": anchor, "received_utc_ns": image.received_utc_ns,
                })
            return None
        number = pairing.first_event_number + value - anchor
        event_id = f"{self._store.experiment_id}:phase-{index}:ttl-{number}"
        return TriggerEvent(event_id, "hardware_trigger", {
            "phase": index, "expected_cameras": list(self._stats), "external_source_id": pairing.source_id,
            "external_event_number": number, "pairing_basis": "explicit_shared_counter_anchor",
            "synchronization": "hardware_association_clock_unverified",
            "counter_feature": pairing.counter_feature, "first_cam_counter": dict(pairing.first_counter),
        })

    def _fire(self, camera: Camera | HostFrameSampler, phase: Phase, index: int, event: TriggerEvent) -> _Pending | None:
        if self._stop.is_set():
            return None
        dispatched_utc, dispatched_monotonic = receipt_times()
        camera.trigger()
        deadline = time.perf_counter() + phase.frame_timeout_s
        while not self._stop.is_set() and (remaining := deadline - time.perf_counter()) > 0:
            image = camera.read(min(0.2, remaining))
            if image is not None:
                metadata = {**image.metadata, "software_trigger_dispatch_utc_ns": dispatched_utc,
                            "software_trigger_dispatch_monotonic_ns": dispatched_monotonic}
                return self._record_image(camera, phase, index, replace(image, metadata=metadata), event)
        if self._stop.is_set():
            return None
        raise CaptureError("TRIGGER_FRAME_TIMEOUT", f"{camera.device.id} produced no image for {event.id}")

    def _round(self, executor, cameras, phase, index, *, due=None, signal=None, prebuffer=False) -> list[_Pending]:
        event_id = str(uuid4())
        utc, monotonic = receipt_times()
        methods = {camera.device.id: camera.trigger_method for camera in cameras}
        event = TriggerEvent(event_id, "software_trigger", {
            "phase": index, "expected_cameras": [camera.device.id for camera in cameras],
            "pairing_basis": "common_capture_request",
            "synchronization": "host_receipt_only" if HOST_NEXT_RECEIPT in methods.values() else "unverified",
            "camera_methods": methods,
            "retention": "rolling_prebuffer" if prebuffer else "save",
            "issued_utc_ns": utc, "issued_monotonic_ns": monotonic,
            "scheduled_monotonic_ns": round(due * 1e9) if due is not None else None,
            "signal_id": signal,
        })
        futures = [executor.submit(self._fire, camera, phase, index, event) for camera in cameras]
        received, failed = [], []
        for camera, future in zip(cameras, futures):
            try:
                item = future.result()
                if item is None:
                    failed.append({"camera_id": camera.device.id, "message": "capture stopped before frame receipt"})
                else:
                    received.append(item)
            except Exception as exc:
                failed.append({"camera_id": camera.device.id, "message": str(exc)})
                self._fail(exc, "SOFTWARE_TRIGGER_FAILED")
        if failed:
            self._diagnostic_event(event.kind, event.details, event_id=event.id, once=True)
            self._diagnostic_event("trigger_incomplete", {
                "phase": index, "trigger_event_id": event_id, "issued_utc_ns": utc, "failed": failed,
            })
        return received

    def _read_signal(self, index: int, timeout: float, *, defer_write=False) -> dict | None:
        try:
            signal = self._signals.get(timeout=timeout) if timeout else self._signals.get_nowait()
        except queue.Empty:
            return None
        details = {"phase": index, **signal}
        if defer_write:
            self._queue_event_records((("signal_received", details, signal["id"]),))
        else:
            self._store.event("signal_received", details, event_id=signal["id"])
        return signal

    def _queue_round(self, items: list[_Pending]) -> None:
        self._enqueue_batch(tuple(items))

    def _discard(self, items: list[_Pending]) -> None:
        with self._mutex:
            for item in items:
                self._stats[item.camera_id].intentionally_skipped += 1

    def _honor_phase_pause(self, running):
        with self._phase_changed:
            gate = self._phase_pause
            if gate is None or self._stop.is_set() or time.perf_counter() >= running.deadline:
                return
        utc, mono = receipt_times()
        self._pause_event("phase_paused", {**gate.record(), "paused_utc_ns": utc, "paused_monotonic_ns": mono})
        with self._phase_changed:
            gate.paused_at = utc, mono
            self._live_write.end()
            while gate.resume_request is None and not self._stop.is_set():
                # Callers may set the Event passed to start() directly.
                self._phase_changed.wait(.1)
            if self._stop.is_set():
                return
            resume = gate.resume_request
        self._pause_event("phase_resume_requested", {
            "phase": gate.phase, "pause_id": gate.pause_id, **resume,
            "pause_to_request_ns": resume["requested_monotonic_ns"] - mono,
        })
        with self._phase_changed:
            if self._stop.is_set():
                return
            # Shift through the ledger commit, including time waiting for its lock.
            duration = time.perf_counter_ns() - mono
            running.paused_ns += duration
            running.deadline += duration / 1e9
            self._phase_pause = None
            self._live_write.begin(gate.phase, running.plan.mode, running.started + running.paused_ns / 1e9)

    def _close_phase_pause(self, running, gate):
        utc, mono = receipt_times()
        duration = mono - gate.paused_at[1] if gate.paused_at else 0
        running.paused_ns += duration
        self._pause_event("phase_pause_cancelled", {
            **gate.record(), "cancelled_utc_ns": utc, "cancelled_monotonic_ns": mono,
            "reason": "stopped" if self._stop.is_set() else "phase_ended",
            **({"pause_duration_ns": duration} if gate.paused_at else {}),
        })

    def _pause_event(self, kind, record):
        try:
            self._store.event(kind, record)
        except Exception as exc:
            raise CaptureError("PHASE_PAUSE_LEDGER_FAILED", f"phase {record['phase']}: {exc}") from exc

    def _timed_round_due(self, due):
        with self._phase_changed:
            if self._phase_pause is not None or self._stop.is_set():
                return False
            remaining = due - time.perf_counter()
            if remaining <= 0:
                return True
            self._phase_changed.wait(min(.1, remaining))
            return False

    def _software_phase(self, executor, cameras: list[Camera | HostFrameSampler], running: _RunningPhase,
                        index: int, started_utc_ns: int) -> None:
        phase, start, window = running.plan, running.started, running.event
        limit = phase.frame_count or math.inf
        if phase.mode == "timed" and phase.duration_s:
            limit = min(limit, math.ceil(phase.duration_s * phase.fps))
        tick = 0
        signal = None
        ring: deque[list[_Pending]] = deque()
        misses = []
        try:
            while not self._stop.is_set() and tick < limit:
                if self._phase_pause is not None:
                    self._honor_phase_pause(running)
                    if self._stop.is_set():
                        break
                deadline = running.deadline
                if time.perf_counter() >= deadline:
                    if phase.mode == "timed" and tick < limit:
                        self._schedule_miss(misses, tick, int(limit - tick), running.paused_ns)
                    break
                if phase.mode == "manual":
                    click = self._read_signal(index, min(0.1, max(0, deadline - time.perf_counter())))
                    if click is None:
                        continue
                    with self._mutex:
                        running.manual_signal_id = click["id"]
                    try:
                        self._queue_round(self._round(executor, cameras, phase, index, signal=click["id"]))
                    finally:
                        with self._mutex:
                            running.manual_signal_id = None
                elif phase.fps is None:
                    self._queue_round(self._round(executor, cameras, phase, index))
                else:
                    due = start + running.paused_ns / 1e9 + tick / phase.fps
                    if due >= deadline:
                        break
                    if phase.mode == "timed":
                        if not self._timed_round_due(due):
                            continue
                    elif self._stop.wait(max(0, due - time.perf_counter())):
                        break
                    stale = int((time.perf_counter() - due) * phase.fps)
                    if stale >= 1:
                        count = int(min(stale, limit - tick))
                        self._schedule_miss(misses, tick, count, running.paused_ns)
                        tick += count
                        continue
                    elif running.overlap is not None:
                        self._accept_overlapping_events(running, index)
                        if running.completed_events == phase.event_count and phase.event_count:
                            break
                        self._acquire_overlapping_round(executor, cameras, running, index, due)
                        if running.completed_events == phase.event_count and phase.event_count:
                            break
                    elif window is not None:
                        if signal is None:
                            signal = self._read_signal(index, 0, defer_write=True)
                            if signal:
                                self._flush_prebuffer(ring, window, index, signal)
                        if signal:
                            items = self._round(executor, cameras, phase, index, due=due, signal=signal["id"])
                            if len(items) == len(cameras):
                                with self._mutex:
                                    window.post_received_rounds += 1
                            self._queue_round([replace(item, context_id=signal["id"], position="post") for item in items])
                            if window.post_received_rounds >= window.post_required_rounds:
                                self._record_event_windows(running, index, (window,))
                                with self._mutex:
                                    if running.completed_events == phase.event_count or self._stop.is_set():
                                        break
                                    window = running.event = _EventWindow(phase.pre_frames, phase.post_frames, window.number + 1)
                                signal = None
                        else:
                            self._prebuffer(executor, cameras, phase, index, due, ring, window)
                    else:
                        self._queue_round(self._round(executor, cameras, phase, index, due=due))
                tick += 1
        finally:
            if running.overlap is not None:
                self._accept_overlapping_events(running, index, closing=True)
            while ring:
                batch = ring.popleft()
                self._change_prebuffer(batch, window, sign=-1)
                self._discard(batch)
            for first, last, offset in misses:
                self._store.event("schedule_miss", {
                    "phase": index, "first_tick": first, "last_tick": last,
                    "first_due_monotonic_ns": round((start + first / phase.fps) * 1e9) + offset,
                    "first_due_utc_ns": started_utc_ns + round(first / phase.fps * 1e9) + offset,
                    "last_due_utc_ns": started_utc_ns + round(last / phase.fps * 1e9) + offset,
                })
        if window is not None and not self._stop.is_set():
            if phase.event_count and running.completed_events < phase.event_count:
                raise CaptureError("EVENT_WINDOW_INCOMPLETE", f"event phase acquired {running.completed_events} of {phase.event_count} planned windows")
            if ((running.overlap is not None and running.overlap.accepted > running.completed_events)
                    or (window.signal is not None and not window.recorded)):
                raise CaptureError("EVENT_WINDOW_INCOMPLETE", "event phase ended during the post-event window")

    def _accept_overlapping_events(self, running, index, *, closing=False):
        overlap, ready = running.overlap, running.event
        with self._mutex:
            overlap.closing = closing
            signals = [self._signals.get_nowait() for _ in range(self._signals.qsize())]
            recent = tuple(overlap.recent) if signals else ()
            expired = []
            retain = 0 if closing else ready.pre_required_rounds
            # Freeze the qualifying history for accepted signals before making
            # an old round unavailable to callers submitting the next signal.
            while len(overlap.recent) > retain:
                batch = overlap.recent.popleft()
                expired.append(batch)
                if batch.complete:
                    ready.pre_receipts_ns.popleft()
        try:
            assignments = {}
            records = []
            for signal in signals:
                records.append(("signal_received", {"phase": index, **signal}, signal["id"]))
                window = _EventWindow(ready.pre_required_rounds, ready.post_required_rounds,
                                      number=signal["event_number"], signal=signal)
                overlap.collecting.append(window)
                for batch, position in self._select_overlapping_rounds(recent, window):
                    assignments.setdefault(batch.issued_ns, (batch, []))[1].append((signal["id"], position))
                if window.pre_retained_rounds < window.pre_required_rounds:
                    records.append(self._short_pre_record(index, window, window.pre_retained_rounds))
            current = sorted(assignments.values(), key=lambda item: item[0].issued_ns)
            links = [(context, camera, sequence, position) for batch, memberships in current if not batch.pending
                     for context, position in memberships for camera, sequence in batch.identities]
            self._queue_event_records(records, links=links)
            for batch, memberships in current:
                if batch.pending:
                    self._publish_overlapping_round(batch, tuple(memberships))
            self._finish_overlapping_events(running, index, closing=closing)
        finally:
            for batch in expired:
                if batch.pending:
                    with self._mutex:
                        self._change_prebuffer_bytes(sum(item.image.raw_bytes for item in batch.pending), sign=-1)
                        self._discard(batch.pending)

    def _select_overlapping_rounds(self, recent, window):
        stamp = window.signal["received_monotonic_ns"]
        required = window.pre_required_rounds
        pre = [batch for batch in recent if batch.complete and batch.received_ns < stamp][-required:] if required else []
        boundary = [batch for batch in recent if batch.issued_ns < stamp <= batch.received_ns] if required else []
        window.pre_retained_rounds = len(pre)
        window.boundary_rounds = sum(batch.complete for batch in boundary)
        for batch in pre:
            yield batch, "pre"
        for batch in boundary:
            yield batch, "boundary"
        for batch in recent:
            if batch.issued_ns >= stamp:
                window.post_received_rounds += batch.complete
                yield batch, "post"
                if window.post_received_rounds >= window.post_required_rounds:
                    break

    def _acquire_overlapping_round(self, executor, cameras, running, index, due):
        overlap, ready = running.overlap, running.event
        if (self._stop.is_set() or time.perf_counter() >= running.deadline
                or (not ready.pre_required_rounds and not overlap.collecting)):
            return
        items = self._round(executor, cameras, running.plan, index, due=due, prebuffer=bool(ready.pre_required_rounds))
        if not items:
            return
        batch = _EventRound.capture(items, len(cameras))
        try:
            with self._mutex:
                self._change_prebuffer(items, ready, sign=1)
                overlap.recent.append(batch)
        except CaptureError:
            with self._mutex:
                for item in items:
                    self._stats[item.camera_id].app_unsaved += 1
            raise
        if overlap.collecting:
            self._publish_overlapping_round(batch, tuple((window.signal_id, "post") for window in overlap.collecting))
            for window in overlap.collecting:
                window.post_received_rounds += batch.complete
            self._finish_overlapping_events(running, index)

    def _publish_overlapping_round(self, batch, memberships):
        context, position = memberships[0]
        with self._mutex:
            items, batch.pending = batch.pending, []
            self._change_prebuffer_bytes(sum(item.image.raw_bytes for item in items), sign=-1)
            self._queue_round([replace(item, context_id=context, position=position, event_memberships=memberships)
                               for item in items])

    def _finish_overlapping_events(self, running, index, *, closing=False):
        overlap = running.overlap
        done, remaining = [], []
        for window in overlap.collecting:
            if closing or window.post_received_rounds >= window.post_required_rounds:
                done.append(window)
            else:
                remaining.append(window)
        self._record_event_windows(running, index, done)
        overlap.collecting = remaining

    def _record_event_windows(self, running, index, windows):
        if not windows:
            return
        self._queue_event_records([("event_window", self._event_window_record(running, index, window), None)
                                   for window in windows])
        with self._mutex:
            for window in windows:
                window.recorded = True
                if running.overlap is not None:
                    running.overlap.recorded += 1
                if window.post_received_rounds >= window.post_required_rounds:
                    running.completed_events += 1

    @staticmethod
    def _event_window_record(running, index, window):
        signal = window.signal
        record = {"phase": index, **window.counts(), "event_count": running.plan.event_count,
                  "pre_time_basis": "host_receive_monotonic"}
        if running.overlap is not None:
            record["overlap"] = True
        if signal is not None:
            record.update(source=signal["source"], signal_utc_ns=signal["received_utc_ns"],
                          signal_monotonic_ns=signal["received_monotonic_ns"])
            if signal.get("origin") == "instrument_csv":
                record["csv_crossing"] = signal["payload"]
        if signal is None:
            record["result"] = "untriggered"
        elif window.post_received_rounds < window.post_required_rounds:
            record["result"] = "short_post"
        elif window.pre_retained_rounds < window.pre_required_rounds:
            record["result"] = "short_pre"
        else:
            record["result"] = "complete"
        return record

    def _short_pre_record(self, index, window, observed):
        with self._mutex:
            self._short_pre_windows += 1
        return ("pre_window_short", {
            "phase": index, "signal_id": window.signal_id, "required_rounds": window.pre_required_rounds,
            "observed_rounds": observed, "basis": "host_receive_monotonic",
        }, None)

    def _schedule_miss(self, ranges, first, count, offset):
        last = first + count - 1
        with self._mutex:
            self._schedule_misses += count
        if ranges and ranges[-1][1] == first - 1 and ranges[-1][2] == offset:
            ranges[-1] = (ranges[-1][0], last, offset)
        else:
            ranges.append((first, last, offset))

    def _change_prebuffer(self, batch: list[_Pending], window: _EventWindow, *, sign: int) -> None:
        size = sum(item.image.raw_bytes for item in batch)
        with self._mutex:
            self._change_prebuffer_bytes(size, sign=sign)
            if len(batch) == len(self._stats):
                if sign > 0:
                    window.pre_receipts_ns.append(max(item.image.received_monotonic_ns for item in batch))
                else:
                    window.pre_receipts_ns.popleft()

    def _change_prebuffer_bytes(self, size, *, sign):
        if sign == 1 and self._queue_bytes + self._prebuffer_bytes + size > self.config.max_buffer_bytes:
            raise CaptureError("RAW_BUFFER_FULL", "pre-trigger ring exceeds the configured raw byte budget")
        self._prebuffer_bytes += sign * size
        self._raw_high_water = max(self._raw_high_water, self._queue_bytes + self._prebuffer_bytes)

    def _prebuffer(self, executor, cameras, phase, index, due, ring, window: _EventWindow) -> None:
        if not phase.pre_frames:
            return
        with self._mutex:
            # A signal can arrive after the scheduler checked the queue and
            # before it reaches this round. Keep its already-qualified window.
            if window.signal_id is not None:
                return
            if len(ring) == phase.pre_frames + 1:
                expired = ring.popleft()
                self._change_prebuffer(expired, window, sign=-1)
                self._discard(expired)
        items = self._round(executor, cameras, phase, index, due=due, prebuffer=True)
        try:
            self._change_prebuffer(items, window, sign=1)
        except CaptureError:
            with self._mutex:
                for item in items:
                    self._stats[item.camera_id].app_unsaved += 1
            raise
        ring.append(items)

    def _flush_prebuffer(self, ring, window: _EventWindow, index, signal):
        receipt = signal["received_monotonic_ns"]

        def before_signal(batch):
            return all(item.image.received_monotonic_ns < receipt for item in batch)

        complete = [before_signal(batch) for batch in ring if len(batch) == len(self._stats)]
        before_count = sum(complete)
        required = window.pre_required_rounds
        with self._mutex:
            window.pre_retained_rounds = min(before_count, required)
            window.boundary_rounds = len(complete) - before_count
        excess = max(0, before_count - required)
        if before_count < required:
            self._queue_event_records((self._short_pre_record(index, window, before_count),))
        keep = []
        while ring:
            batch = ring.popleft()
            self._change_prebuffer(batch, window, sign=-1)
            before = before_signal(batch)
            if before and excess and len(batch) == len(self._stats):
                excess -= 1
                self._discard(batch)
            else:
                position = "pre" if before else "boundary"
                keep.extend(replace(item, context_id=signal["id"], position=position) for item in batch)
        self._queue_round(keep)
