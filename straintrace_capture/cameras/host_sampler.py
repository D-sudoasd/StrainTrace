"""Select the first image received after a request from an active free-running camera."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from threading import Condition, Event, Thread
from time import perf_counter_ns

from ..config import CaptureError
from .base import HOST_NEXT_RECEIPT, Camera, Image


def first_host_receipt(image: Image, request_ns: int) -> Image | None:
    if image.received_monotonic_ns <= request_ns:
        return None
    return replace(image, metadata={
        **image.metadata, "host_selection_request_monotonic_ns": request_ns,
        "frame_selection": "first_host_receipt_after_request",
    })


class HostFrameSampler:
    """Drain a free-running camera with at most one pending request and one selected image."""

    trigger_method = HOST_NEXT_RECEIPT

    def __init__(self, camera: Camera, observe: Callable[[Image, bool], None], on_error: Callable[[Exception], None]):
        self.device = camera.device
        self._camera = camera
        self._observe = observe
        self._on_error = on_error
        self._condition = Condition()
        self._stop = Event()
        self._request_ns = None
        self._result = None
        self._failure = None
        self._worker = Thread(target=self._receive, name=f"host-sampling-{self.device.id}")

    def __enter__(self):
        self._worker.start()
        return self

    def __exit__(self, *exc):
        with self._condition:
            self._stop.set()
            self._condition.notify_all()
        # The camera owner can stop or close the device after the pending read
        # returns. The adapter's read timeout bounds each read.
        self._worker.join()

    def _receive(self):
        try:
            while not self._stop.is_set():
                image = self._camera.read(.2)
                if image is None:
                    continue
                selected = None
                with self._condition:
                    request_ns = self._request_ns
                    if not self._stop.is_set() and self._result is None and request_ns is not None:
                        selected = first_host_receipt(image, request_ns)
                    if selected is not None:
                        self._result = selected
                        self._condition.notify_all()
                self._observe(image, selected is not None)
        except Exception as exc:
            with self._condition:
                self._failure = exc
                self._condition.notify_all()
            if not self._stop.is_set():
                self._on_error(exc)

    def trigger(self):
        with self._condition:
            if self._failure is not None:
                raise self._failure
            if self._request_ns is not None or self._stop.is_set():
                raise CaptureError("CAMERA_NOT_READY", f"{self.device.id} is not ready to select a frame")
            self._request_ns = perf_counter_ns()

    def read(self, timeout_s: float) -> Image | None:
        with self._condition:
            self._condition.wait_for(lambda: self._result is not None or self._failure is not None or self._stop.is_set(),
                                     timeout=timeout_s)
            result = self._result
            if result is not None:
                self._result = self._request_ns = None
            elif self._failure is not None:
                raise self._failure
            return result
