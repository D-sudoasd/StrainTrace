"""Capture controller with bounded raw storage and a replaceable preview slot."""

from __future__ import annotations

import copy
import math
import queue
import shutil
import threading
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from uuid import uuid4

from .cameras import Camera, CameraRegistry
from .cameras.base import Image, receipt_times
from .config import CaptureConfig, CaptureError, Phase, nonempty_text
from .storage import ExperimentWriter, TriggerEvent, json_text


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


@dataclass
class _CameraStats:
    received: int = 0
    saved: int = 0
    saved_bytes: int = 0
    intentionally_skipped: int = 0
    app_unsaved: int = 0
    native_counter_gaps: int = 0
    native_counter_duplicates: int = 0
    native_counter_resets: int = 0
    hardware_identity_missing: int = 0
    reported_driver_skips: int = 0
    transport_errors: int = 0


class CaptureSession:
    """Start a new experiment, monitor it and request authorized camera actions."""

    def __init__(self, config: CaptureConfig, output: str | Path, *, registry: CameraRegistry | None = None):
        self.config = CaptureConfig.from_dict(config.to_dict())
        self.output = Path(output).expanduser().resolve()
        self.registry = registry if registry is not None else CameraRegistry(cti_files=self.config.cti_files)
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._producers_done = threading.Event()
        self._mutex = threading.RLock()
        self._frames: queue.Queue[tuple[_Pending, ...]] = queue.Queue(maxsize=self.config.queue_capacity)
        self._signals: queue.Queue[dict] = queue.Queue(maxsize=16)
        self._latest: dict[str, Image] = {}
        self._stats = {camera.id: _CameraStats() for camera in self.config.cameras}
        self._counter: dict[str, int] = {}
        self._queue_bytes = 0
        self._queue_high_water = 0
        self._prebuffer_bytes = 0
        self._schedule_misses = 0
        self._short_pre_windows = 0
        self._state = "idle"
        self._phase: int | None = None
        self._mode: str | None = None
        self._started: float | None = None
        self._ended: float | None = None
        self._error: dict | None = None
        self._requested_stop = False
        self._writer_failed = False
        self._accepting_event = False
        self._store: ExperimentWriter | None = None

    def start(self) -> str:
        with self._mutex:
            if self._state != "idle":
                raise CaptureError("ALREADY_STARTED", "a CaptureSession can be started only once")
            self._store = ExperimentWriter(self.output, self.config)
            self._state = "opening"
            self._started = time.perf_counter()
            threading.Thread(target=self._run, name="capture-controller").start()
            return self._store.experiment_id

    def request_stop(self) -> None:
        with self._mutex:
            if self._state in ("opening", "running"):
                self._requested_stop = True
                self._stop.set()

    def wait(self, timeout: float | None = None) -> dict:
        if self._started is None:
            raise CaptureError("NOT_STARTED", "start the capture before waiting")
        if not self._finished.wait(timeout):
            raise TimeoutError("capture is still running")
        return self.status()

    def request_trigger(self, source: str = "operator", *, payload: dict | None = None) -> str:
        """Register an incoming signal. This method never controls external equipment."""
        source = nonempty_text(source, "event source")
        if payload is not None and not isinstance(payload, dict):
            raise CaptureError("SIGNAL_PAYLOAD_INVALID", "trigger metadata must be a JSON object")
        utc, monotonic = receipt_times()
        signal = {
            "id": str(uuid4()), "source": source, "payload": dict(payload or {}),
            "received_utc_ns": utc, "received_monotonic_ns": monotonic,
        }
        signal = copy.deepcopy(signal)
        try:
            json_text(signal)
        except (TypeError, ValueError) as exc:
            raise CaptureError("SIGNAL_PAYLOAD_INVALID", f"trigger metadata must contain finite JSON values: {exc}") from exc
        with self._mutex:
            if self._stop.is_set() or self._state != "running" or self._mode not in ("manual", "event"):
                raise CaptureError("NOT_ARMED", "the active phase does not accept external or manual signals")
            if self._mode == "event" and not self._accepting_event:
                raise CaptureError("EVENT_ALREADY_RECEIVED", "the event phase is recording its post-event frames")
            try:
                self._signals.put_nowait(signal)
            except queue.Full as exc:
                raise CaptureError("SIGNAL_QUEUE_FULL", "the camera trigger queue is full") from exc
            if self._mode == "event":
                self._accepting_event = False
        return signal["id"]

    def latest_image(self, camera_id: str) -> Image | None:
        with self._mutex:
            return self._latest.get(camera_id)

    def status(self) -> dict:
        with self._mutex:
            elapsed = (self._ended or time.perf_counter()) - self._started if self._started is not None else 0
            per_camera = {}
            for camera_id, stats in self._stats.items():
                per_camera[camera_id] = {
                    **asdict(stats),
                    "mean_receive_fps": stats.received / elapsed if elapsed else 0,
                    "mean_save_fps": stats.saved / elapsed if elapsed else 0,
                }
            result = {
                "state": self._state,
                "experiment_id": self._store.experiment_id if self._store else None,
                "phase": self._phase,
                "mode": self._mode,
                "elapsed_s": elapsed,
                "queue_size": self._frames.qsize(),
                "queue_capacity": self.config.queue_capacity,
                "queue_high_water": self._queue_high_water,
                "buffered_raw_bytes": self._queue_bytes + self._prebuffer_bytes,
                "scheduled_misses": self._schedule_misses,
                "short_pre_windows": self._short_pre_windows,
                "disk_bytes_per_s": sum(item.saved_bytes for item in self._stats.values()) / elapsed if elapsed else 0,
                "cameras": per_camera,
                "error": dict(self._error) if self._error else None,
            }
        try:
            result["disk_free_bytes"] = shutil.disk_usage(self.output if self.output.is_dir() else self.output.parent).free
        except OSError:
            result["disk_free_bytes"] = None
        return result

    def _fail(self, exc: Exception, fallback: str) -> None:
        code = exc.code if isinstance(exc, CaptureError) else fallback
        with self._mutex:
            if self._error is None:
                self._error = {"code": code, "message": str(exc)}
        self._stop.set()
        try:
            self._store.event("error", {"code": code, "message": str(exc), "phase": self._phase})
        except Exception:
            # The failure may itself be a full disk or unavailable journal.
            # In that case the last committed ledger and in-memory status survive.
            pass

    def _run(self) -> None:
        writer = threading.Thread(target=self._write, name="capture-disk")
        writer.start()
        try:
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
                            self._run_phase(executor, cameras, phase, index)
        except Exception as exc:
            self._fail(exc, "CAMERA_ERROR")
        finally:
            with self._mutex:
                self._state = "draining"
            self._producers_done.set()
            writer.join()
            with self._mutex:
                self._ended = time.perf_counter()
                self._state = "failed" if self._error else ("stopped" if self._requested_stop else "completed")
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

    def _run_phase(self, executor: ThreadPoolExecutor, cameras: list[Camera], phase: Phase, index: int) -> None:
        trigger = {"continuous": "free", "hardware": "hardware"}.get(phase.mode, "software")
        with ExitStack() as stack:
            for camera in cameras:
                stack.callback(camera.stop)
                source = phase.hardware_source
                camera.start(trigger, source[camera.device.id] if isinstance(source, Mapping) else source)
            self._store.event("phase_started", {
                "phase": index, "plan": phase.to_dict(),
                "camera_features": {
                    camera.device.id: {name: asdict(item) for name, item in camera.features().items()}
                    for camera in cameras
                },
            })
            with self._mutex:
                self._phase, self._mode = index, phase.mode
                self._counter.clear()
                self._accepting_event = phase.mode == "event"
            try:
                if trigger == "software":
                    self._software_phase(executor, cameras, phase, index)
                else:
                    deadline = time.perf_counter() + phase.duration_s if phase.duration_s else float("inf")
                    futures = [executor.submit(self._stream, camera, phase, index, deadline) for camera in cameras]
                    for future in futures:
                        future.result()
            finally:
                with self._mutex:
                    self._mode = None
                    unused_signals = []
                    while not self._signals.empty():
                        unused_signals.append(self._signals.get_nowait())
                for signal in unused_signals:
                    self._store.event("signal_unused", {"phase": index, **signal}, event_id=signal["id"])
        self._store.event("phase_ended", {"phase": index, "stop_requested": self._stop.is_set()})

    def _record_image(self, camera: Camera, phase: Phase, index: int, image: Image, trigger: TriggerEvent | None) -> _Pending:
        camera_id = camera.device.id
        discontinuity = None
        with self._mutex:
            stats = self._stats[camera_id]
            stats.received += 1
            stats.reported_driver_skips += int(image.metadata.get("pylon_skipped_images", 0))
            sequence = stats.received
            self._latest[camera_id] = image
            if image.native_counter is not None:
                last = self._counter.get(camera_id)
                if last is not None and image.native_counter != last + 1:
                    detail = {"camera_id": camera_id, "phase": index,
                              "previous": last, "current": image.native_counter}
                    if image.native_counter > last + 1:
                        stats.native_counter_gaps += image.native_counter - last - 1
                        discontinuity = ("native_gap_unattributed", detail)
                    elif image.native_counter == last:
                        stats.native_counter_duplicates += 1
                        discontinuity = ("native_duplicate", detail)
                    else:
                        stats.native_counter_resets += 1
                        discontinuity = ("native_reset_or_wrap", detail)
                self._counter[camera_id] = image.native_counter
        if discontinuity:
            self._store.event(*discontinuity)
        return _Pending(camera_id, sequence, index, phase.role, image, trigger)

    def _enqueue(self, item: _Pending) -> None:
        self._enqueue_batch((item,))

    def _enqueue_batch(self, items: tuple[_Pending, ...]) -> None:
        if not items:
            return
        size = sum(item.image.pixels.nbytes for item in items)
        with self._mutex:
            if self._writer_failed:
                for item in items:
                    self._stats[item.camera_id].app_unsaved += 1
                return
            try:
                if self._queue_bytes + self._prebuffer_bytes + size > self.config.max_buffer_bytes:
                    raise queue.Full
                self._frames.put_nowait(items)
            except queue.Full as exc:
                for item in items:
                    self._stats[item.camera_id].app_unsaved += 1
                first = items[0]
                raise CaptureError(
                    "RAW_BUFFER_FULL", f"raw queue exhausted at {first.camera_id} sequence {first.sequence}",
                ) from exc
            self._queue_bytes += size
            self._queue_high_water = max(self._queue_high_water, self._frames.qsize())

    def _write(self) -> None:
        pending: deque[_Pending] = deque()
        while pending or not (self._producers_done.is_set() and self._frames.empty()):
            if not pending and not self._take_work(pending, .05):
                continue
            start = time.perf_counter()
            prepared, items = [], []
            while len(items) < 16:
                if not pending:
                    remaining = .075 - (time.perf_counter() - start)
                    if remaining <= 0 or not self._take_work(pending, remaining):
                        break
                item = pending.popleft()
                items.append(item)
                if self._writer_failed:
                    continue
                try:
                    prepared.append(self._store.prepare_frame(
                        item.camera_id, item.sequence, item.phase, item.role, item.image,
                        item.trigger, item.context_id, item.position, self.config.min_free_bytes,
                    ))
                except Exception as exc:
                    self._write_failure(exc, item)
            try:
                if prepared:
                    self._store.commit_frames(prepared)
            except Exception as exc:
                self._write_failure(exc, items[0])
                prepared.clear()
            self._account_writes(items, prepared)

    def _take_work(self, pending, timeout) -> bool:
        try:
            pending.extend(self._frames.get(timeout=0 if self._producers_done.is_set() else max(0, timeout)))
            return True
        except queue.Empty:
            return False

    def _write_failure(self, exc, item):
        with self._mutex:
            self._writer_failed = True
        prefix = f"{item.camera_id} sequence {item.sequence}"
        error = CaptureError(exc.code if isinstance(exc, CaptureError) else "WRITE_FAILED", f"{prefix}: {exc}")
        self._fail(error, "WRITE_FAILED")

    def _account_writes(self, items, prepared):
        byte_counts = {
            (row.record["camera_id"], row.record["sequence"]): row.record["file_bytes"] for row in prepared
        }
        with self._mutex:
            for item in items:
                self._queue_bytes -= item.image.pixels.nbytes
                stats = self._stats[item.camera_id]
                if size := byte_counts.get((item.camera_id, item.sequence)):
                    stats.saved += 1
                    stats.saved_bytes += size
                else:
                    stats.app_unsaved += 1

    def _stream(self, camera: Camera, phase: Phase, index: int, deadline: float) -> None:
        count = 0
        last = time.perf_counter()
        try:
            while not self._stop.is_set() and (phase.frame_count is None or count < phase.frame_count):
                remaining = deadline - time.perf_counter()
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
                self._store.event("hardware_identity_unavailable", {
                    "camera_id": camera_id, "phase": index, "counter_feature": pairing.counter_feature,
                    "observed": value, "first_counter": anchor,
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

    def _fire(self, camera: Camera, phase: Phase, index: int, event: TriggerEvent) -> _Pending:
        dispatched_utc, dispatched_monotonic = receipt_times()
        camera.trigger()
        deadline = time.perf_counter() + phase.frame_timeout_s
        while (remaining := deadline - time.perf_counter()) > 0:
            image = camera.read(min(0.2, remaining))
            if image is not None:
                metadata = {**image.metadata, "software_trigger_dispatch_utc_ns": dispatched_utc,
                            "software_trigger_dispatch_monotonic_ns": dispatched_monotonic}
                return self._record_image(camera, phase, index, replace(image, metadata=metadata), event)
        raise CaptureError("TRIGGER_FRAME_TIMEOUT", f"{camera.device.id} produced no image for {event.id}")

    def _round(self, executor, cameras, phase, index, *, due=None, signal=None, prebuffer=False) -> list[_Pending]:
        event_id = str(uuid4())
        utc, monotonic = receipt_times()
        event = TriggerEvent(event_id, "software_trigger", {
            "phase": index, "expected_cameras": [camera.device.id for camera in cameras],
            "pairing_basis": "common_capture_request", "synchronization": "unverified",
            "camera_methods": {camera.device.id: camera.trigger_method for camera in cameras},
            "retention": "rolling_prebuffer" if prebuffer else "save",
            "issued_utc_ns": utc, "issued_monotonic_ns": monotonic,
            "scheduled_monotonic_ns": round(due * 1e9) if due is not None else None,
            "signal_id": signal,
        })
        futures = [executor.submit(self._fire, camera, phase, index, event) for camera in cameras]
        received, failed = [], []
        for camera, future in zip(cameras, futures):
            try:
                received.append(future.result())
            except Exception as exc:
                failed.append({"camera_id": camera.device.id, "message": str(exc)})
                self._fail(exc, "SOFTWARE_TRIGGER_FAILED")
        if failed:
            self._store.event(event.kind, event.details, event_id=event.id, once=True)
            self._store.event("trigger_incomplete", {"phase": index, "trigger_event_id": event_id, "failed": failed})
        return received

    def _read_signal(self, index: int, timeout: float) -> dict | None:
        try:
            signal = self._signals.get(timeout=timeout) if timeout else self._signals.get_nowait()
        except queue.Empty:
            return None
        self._store.event("signal_received", {"phase": index, **signal}, event_id=signal["id"])
        return signal

    def _queue_round(self, items: list[_Pending]) -> None:
        self._enqueue_batch(tuple(items))

    def _discard(self, items: list[_Pending]) -> None:
        with self._mutex:
            for item in items:
                self._stats[item.camera_id].intentionally_skipped += 1

    def _software_phase(self, executor, cameras: list[Camera], phase: Phase, index: int) -> None:
        start = time.perf_counter()
        deadline = start + phase.duration_s if phase.duration_s else float("inf")
        limit = phase.frame_count or math.inf
        if phase.mode == "timed" and phase.duration_s:
            limit = min(limit, math.ceil(phase.duration_s * phase.fps))
        tick = 0
        post_count = 0
        signal = None
        ring: deque[list[_Pending]] = deque()
        misses = []
        try:
            while not self._stop.is_set() and tick < limit:
                if time.perf_counter() >= deadline:
                    if phase.mode == "timed" and tick < limit:
                        self._schedule_miss(misses, tick, int(limit - tick))
                    break
                if phase.mode == "manual":
                    click = self._read_signal(index, min(0.1, max(0, deadline - time.perf_counter())))
                    if click is None:
                        continue
                    self._queue_round(self._round(executor, cameras, phase, index, signal=click["id"]))
                elif phase.fps is None:
                    self._queue_round(self._round(executor, cameras, phase, index))
                else:
                    due = start + tick / phase.fps
                    if due >= deadline or self._stop.wait(max(0, due - time.perf_counter())):
                        break
                    stale = int((time.perf_counter() - due) * phase.fps)
                    if stale >= 1:
                        count = int(min(stale, limit - tick))
                        self._schedule_miss(misses, tick, count)
                        tick += count
                        continue
                    elif phase.mode == "event":
                        if signal is None:
                            signal = self._read_signal(index, 0)
                            if signal:
                                self._flush_prebuffer(ring, phase, index, signal)
                        if signal:
                            items = self._round(executor, cameras, phase, index, due=due, signal=signal["id"])
                            self._queue_round([replace(item, context_id=signal["id"], position="post") for item in items])
                            post_count += 1
                            if post_count >= phase.post_frames:
                                break
                        else:
                            self._prebuffer(executor, cameras, phase, index, due, ring)
                    else:
                        self._queue_round(self._round(executor, cameras, phase, index, due=due))
                tick += 1
        finally:
            while ring:
                batch = ring.popleft()
                self._change_prebuffer(batch, sign=-1)
                self._discard(batch)
            for first, last in misses:
                self._store.event("schedule_miss", {"phase": index, "first_tick": first, "last_tick": last,
                                                   "first_due_monotonic_ns": round((start + first / phase.fps) * 1e9)})
        if phase.mode == "event" and not self._stop.is_set() and post_count < phase.post_frames:
            raise CaptureError("EVENT_WINDOW_INCOMPLETE", "event phase ended before the requested post-event window")

    def _schedule_miss(self, ranges, first, count):
        last = first + count - 1
        with self._mutex:
            self._schedule_misses += count
        if ranges and ranges[-1][1] == first - 1:
            ranges[-1] = (ranges[-1][0], last)
        else:
            ranges.append((first, last))

    def _change_prebuffer(self, batch: list[_Pending], *, sign: int) -> None:
        size = sum(item.image.pixels.nbytes for item in batch)
        with self._mutex:
            if sign == 1 and self._queue_bytes + self._prebuffer_bytes + size > self.config.max_buffer_bytes:
                raise CaptureError("RAW_BUFFER_FULL", "pre-trigger ring exceeds the configured raw byte budget")
            self._prebuffer_bytes += sign * size

    def _prebuffer(self, executor, cameras, phase, index, due, ring) -> None:
        if not phase.pre_frames:
            return
        if len(ring) == phase.pre_frames + 1:
            expired = ring.popleft()
            self._change_prebuffer(expired, sign=-1)
            self._discard(expired)
        items = self._round(executor, cameras, phase, index, due=due, prebuffer=True)
        try:
            self._change_prebuffer(items, sign=1)
        except CaptureError:
            with self._mutex:
                for item in items:
                    self._stats[item.camera_id].app_unsaved += 1
            raise
        ring.append(items)

    def _flush_prebuffer(self, ring, phase, index, signal):
        before_count = sum(
            bool(batch) and all(item.image.received_monotonic_ns < signal["received_monotonic_ns"] for item in batch)
            for batch in ring
        )
        excess = max(0, before_count - phase.pre_frames)
        if before_count < phase.pre_frames:
            with self._mutex:
                self._short_pre_windows += 1
            self._store.event("pre_window_short", {
                "phase": index, "signal_id": signal["id"], "required_rounds": phase.pre_frames,
                "observed_rounds": before_count, "basis": "host_receive_monotonic",
            })
        keep = []
        while ring:
            batch = ring.popleft()
            self._change_prebuffer(batch, sign=-1)
            before = all(item.image.received_monotonic_ns < signal["received_monotonic_ns"] for item in batch)
            if before and excess:
                excess -= 1
                self._discard(batch)
            else:
                position = "pre" if before else "boundary"
                keep.extend(replace(item, context_id=signal["id"], position=position) for item in batch)
        self._queue_round(keep)
