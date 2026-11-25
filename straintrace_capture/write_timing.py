"""Compare writer progress with the raw pixels admitted during acquisition."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Sample:
    time_s: float
    accepted_bytes: int
    committed_bytes: int
    accepted_frames: int
    committed_frames: int
    queued_batches: int = 0
    buffered_bytes: int = 0

    @property
    def pending_frames(self) -> int:
        return self.accepted_frames - self.committed_frames


def _window_start(samples, end):
    return next((point for point in reversed(samples) if 2.0 <= end.time_s - point.time_s <= 3.0), None)


def _rates(start, end):
    seconds = end.time_s - start.time_s
    accepted = end.accepted_bytes - start.accepted_bytes
    committed = end.committed_bytes - start.committed_bytes
    return {
        "duration_s": seconds,
        "accepted_raw_bytes_per_s": accepted / seconds,
        "committed_raw_bytes_per_s": committed / seconds,
        "pending_raw_bytes_growth": accepted - committed,
    }


def _lags(windows):
    return all(row["committed_raw_bytes_per_s"] < 0.9 * row["accepted_raw_bytes_per_s"] for row in windows)


class TrialWriteTiming:
    """Called under the session lock. Retain only the recent acquisition samples."""

    def __init__(self, minimum_growth_frames: int):
        self._minimum_growth_frames = minimum_growth_frames
        self._accepted_bytes = 0
        self._committed_bytes = 0
        self._accepted_frames = 0
        self._committed_frames = 0
        self._samples: deque[_Sample] = deque(maxlen=16)
        self._end: _Sample | None = None

    def _sample(self, now: float) -> _Sample:
        return _Sample(now, self._accepted_bytes, self._committed_bytes,
                       self._accepted_frames, self._committed_frames)

    def begin(self, now: float, *, sample_rates: bool) -> None:
        if sample_rates:
            self._samples.append(self._sample(now))

    def accept(self, now: float, size: int, frames: int) -> None:
        if self._samples and now - self._samples[-1].time_s >= 0.5:
            self._samples.append(self._sample(now))
        self._accepted_bytes += size
        self._accepted_frames += frames

    def commit(self, size: int, frames: int) -> None:
        self._committed_bytes += size
        self._committed_frames += frames

    def end(self, now: float) -> None:
        self._end = self._sample(now)

    def finish(self, now: float) -> dict | None:
        end = self._end
        if end is None:
            return None
        windows = []
        sustained_lag = False
        middle = _window_start(self._samples, end)
        if middle is not None and (start := _window_start(self._samples, middle)) is not None:
            windows = [_rates(start, middle), _rates(middle, end)]
            # Screen out the variation from one camera round or ledger commit.
            sustained_lag = (
                end.pending_frames - start.pending_frames > self._minimum_growth_frames
                and _lags(windows)
            )
        return {
            "acquisition_windows": windows,
            "sustained_lag": sustained_lag,
            "post_capture_committed_frames": self._committed_frames - end.committed_frames,
            "post_capture_elapsed_s": now - end.time_s,
        }


class LiveWriteTiming:
    """Phase-local comparisons for the shared writer. Call under the session lock."""

    def __init__(self, minimum_growth_frames, byte_limit, batch_limit):
        self._minimum_growth_frames = minimum_growth_frames
        self._limits = {"bytes": byte_limit, "batches": batch_limit}
        self._accepted_bytes = self._committed_bytes = 0
        self._accepted_frames = self._committed_frames = 0
        self._samples = deque(maxlen=16)
        self._alerts = {}
        self.end()

    def begin(self, phase, mode, now):
        self.end()
        self._phase = phase
        self._started = now
        self._continuous = mode in ("continuous", "timed", "hardware")

    def end(self):
        self._phase = None
        self._continuous = False
        self._samples.clear()
        self._windows = ()
        self._growth = {}
        self._sustained = None

    def accept(self, size, frames):
        self._accepted_bytes += size
        self._accepted_frames += frames

    def commit(self, size, frames):
        self._committed_bytes += size
        self._committed_frames += frames

    def _remember(self, kind, now, *, buffered, queued):
        self._alerts.setdefault((self._phase, kind), {
            "phase": self._phase, "kind": kind, "phase_elapsed_s": now - self._started,
            "buffered_raw_bytes": buffered, "queued_batches": queued,
            "windows": [dict(row) for row in self._windows] if kind != "near_limit" else [],
        })

    def _near_limit(self, buffered, queued):
        return buffered >= .85 * self._limits["bytes"] or queued >= .85 * self._limits["batches"]

    def observe(self, now, *, pending, buffered, queued):
        if self._phase is None:
            return
        if pending and self._near_limit(buffered, queued) and (self._phase, "near_limit") not in self._alerts:
            self._remember("near_limit", now, buffered=buffered, queued=queued)
        if not self._continuous or (self._samples and now - self._samples[-1].time_s < .5):
            return
        end = _Sample(now, self._accepted_bytes, self._committed_bytes, self._accepted_frames,
                      self._committed_frames, queued, buffered)
        self._samples.append(end)
        self._growth = {}
        self._sustained = None
        middle = _window_start(self._samples, end)
        if middle is None:
            self._windows = ()
            return
        recent = _rates(middle, end)
        self._windows = (recent,)
        if (first := _window_start(self._samples, middle)) is None:
            return
        earlier = _rates(first, middle)
        if end.pending_frames - first.pending_frames <= self._minimum_growth_frames:
            return
        queue_growing = first.queued_batches < middle.queued_batches < end.queued_batches
        byte_lag = _lags((earlier, recent))
        if not byte_lag and not (queue_growing and all(
                row["accepted_raw_bytes_per_s"] > 0 for row in (earlier, recent))):
            return
        self._sustained = "sustained" if byte_lag else "queue_growth"
        self._windows = earlier, recent
        seconds = end.time_s - first.time_s
        if first.buffered_bytes < middle.buffered_bytes < end.buffered_bytes:
            self._growth["bytes"] = (end.buffered_bytes - first.buffered_bytes) / seconds
        if queue_growing:
            self._growth["batches"] = (end.queued_batches - first.queued_batches) / seconds
        if (self._phase, self._sustained) not in self._alerts:
            self._remember(self._sustained, now, buffered=buffered, queued=queued)

    def status(self, *, running, failed, pending, buffered, queued):
        estimate = None
        windows = self._windows if running else ()
        if failed:
            state = "failed"
        elif not running:
            state = "draining" if pending else "idle"
        elif pending and self._near_limit(buffered, queued):
            state = "near_limit"
        elif pending and self._sustained:
            state = self._sustained
        elif windows:
            latest = windows[-1]
            if latest["accepted_raw_bytes_per_s"] == 0:
                state = "draining" if pending else "idle"
            elif pending and latest["pending_raw_bytes_growth"] > 0 and _lags((latest,)):
                state = "growing"
            elif latest["pending_raw_bytes_growth"] < 0:
                state = "recovering"
            else:
                state = "writing"
        else:
            state = ("sampling" if self._continuous else "writing" if pending
                     else "prebuffer" if buffered else "idle")
        if running and not failed and pending and self._growth:
            used = {"bytes": buffered, "batches": queued}
            seconds, resource = min((max(0, self._limits[key] - used[key]) / growth, key)
                                    for key, growth in self._growth.items())
            estimate = {"seconds": seconds, "resource": resource, "growth_per_s": self._growth[resource]}
        return {
            "state": state, "pending_raw_bytes": pending, "prebuffer_raw_bytes": buffered - pending,
            "windows": [dict(row) for row in windows], "estimate": estimate,
            "accepted_raw_bytes": self._accepted_bytes, "committed_raw_bytes": self._committed_bytes,
        }

    def alerts(self):
        return deepcopy(list(self._alerts.values()))
