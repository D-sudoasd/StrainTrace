"""Replaceable previews and setup diagnostics, outside the recording path."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .cameras import CameraRegistry
from .cameras.base import Image
from .config import CaptureError


class PreviewSession:
    def __init__(self, cameras, registry: CameraRegistry):
        self.cameras = tuple(cameras)
        self.registry = registry
        self._mutex = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        self._latest = {}
        self._states = {camera.id: {"state": "connecting", "error": None} for camera in self.cameras}

    def start(self):
        for config in self.cameras:
            worker = threading.Thread(target=self._preview, args=(config,), name=f"preview-{config.id}", daemon=True)
            self._threads.append(worker)
            worker.start()

    def stop(self):
        self._stop.set()

    @property
    def stopped(self):
        return bool(self._threads) and not any(worker.is_alive() for worker in self._threads)

    def wait(self, timeout: float):
        deadline = time.monotonic() + timeout
        for worker in self._threads:
            worker.join(max(0, deadline - time.monotonic()))
        return self.stopped

    def latest_image(self, camera_id: str) -> Image | None:
        with self._mutex:
            return self._latest.get(camera_id)

    def status(self) -> dict:
        with self._mutex:
            return {key: dict(value) for key, value in self._states.items()}

    def _preview(self, config):
        state = {"state": "disconnected", "error": None}
        try:
            with self.registry.camera(config.id) as camera:
                camera.set_features(config.settings)
                features = {name: feature.value for name, feature in camera.features().items()}
                camera.start()
                try:
                    last = time.monotonic()
                    next_preview = last
                    with self._mutex:
                        self._states[config.id] = {"state": "preview", "error": None, "features": features}
                    while not self._stop.is_set():
                        image = camera.read(0.2)
                        now = time.monotonic()
                        if image is not None:
                            last = now
                            if now >= next_preview:
                                with self._mutex:
                                    self._latest[config.id] = image
                                next_preview = now + 1 / 12
                        elif now - last > 4:
                            raise CaptureError("PREVIEW_TIMEOUT", f"{config.id} stopped delivering preview frames")
                finally:
                    camera.stop()
        except Exception as exc:
            state = {"state": "failed", "error": str(exc)}
        finally:
            with self._mutex:
                self._states[config.id] = state


def gray_pixels(image: Image) -> np.ndarray:
    conversion = {
        "rgb": cv2.COLOR_RGB2GRAY, "bgr": cv2.COLOR_BGR2GRAY,
        "rgba": cv2.COLOR_RGBA2GRAY, "bgra": cv2.COLOR_BGRA2GRAY,
    }.get(image.color_model)
    return cv2.cvtColor(image.pixels, conversion) if conversion else image.pixels


def preview_pixels(image: Image, *, palette: str = "native") -> np.ndarray:
    """Return an 8-bit display-only view, scaled to the declared full code range."""
    maximum = (1 << image.significant_bits) - 1
    if palette == "inferno":
        samples = cv2.convertScaleAbs(gray_pixels(image), alpha=255 / maximum)
        return cv2.cvtColor(cv2.applyColorMap(samples, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    pixels = cv2.convertScaleAbs(image.pixels, alpha=255 / maximum)
    conversion = {"bgr": cv2.COLOR_BGR2RGB, "bgra": cv2.COLOR_BGRA2RGB, "rgba": cv2.COLOR_RGBA2RGB}.get(image.color_model)
    return cv2.cvtColor(pixels, conversion) if conversion else pixels


def image_quality(image: Image, roi: tuple[int, int, int, int] | None = None) -> dict:
    """Display diagnostics drawn from the camera samples in image coordinates."""
    gray = gray_pixels(image)
    maximum = (1 << image.significant_bits) - 1
    stride = max(1, int(np.ceil(np.sqrt(gray.size / 500000))))
    samples = gray[::stride, ::stride]
    bins = np.bincount(np.minimum(samples.astype(np.uint32) * 64 // (maximum + 1), 63).ravel(), minlength=64)
    lo, med, hi = np.percentile(samples, [5, 50, 95])
    focus_samples = cv2.convertScaleAbs(samples, alpha=255 / maximum)
    result = {
        "histogram": bins.tolist(), "code_max": maximum,
        "percentiles_p5_p50_p95": [float(lo), float(med), float(hi)],
        "dark_fraction": float(np.mean(samples <= 0.01 * maximum)),
        "bright_fraction": float(np.mean(samples >= 0.99 * maximum)),
        "focus_laplacian_variance": float(cv2.Laplacian(focus_samples, cv2.CV_32F).var()),
        "histogram_stride_px": stride, "texture": None,
    }
    if roi is not None:
        import ezdic_core as core

        if not core.rect_is_inside_image(roi, gray.shape):
            raise CaptureError("ROI_OUT_OF_BOUNDS", "the inspection ROI is outside the acquired image")
        region = core.extract_patch(gray, roi)
        if region.size > 1_000_000:
            raise CaptureError("INSPECTION_ROI_TOO_LARGE", "select a smaller inspection ROI for live texture metrics")
        region8 = cv2.convertScaleAbs(region, alpha=255 / maximum)
        metrics = core.roi_texture_metrics(region8, (0, 0, region.shape[1], region.shape[0]))
        result["texture"] = {
            **metrics,
            "failure_code": core.texture_failure_code(metrics, 8.0, 25.0, 0.20),
        }
    return result
