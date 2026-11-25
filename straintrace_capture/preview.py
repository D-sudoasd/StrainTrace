"""Replaceable previews and setup diagnostics, outside the recording path."""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, replace
from math import isclose
from statistics import median

import cv2
import numpy as np

from .cameras import CameraRegistry
from .cameras._gentl_multipart import AUTO_PART, IMAGE_PART, SAVE_ALL_2D, ImagePartSelectionError
from .cameras.base import Device, Image, TriggerSupport
from .cameras.settings_probe import SettingsCheck
from .config import SENSOR_WINDOW_FIELDS, CameraConfig, CaptureError, settings_object


PREVIEW_SETTINGS = frozenset({
    "width", "height", "offset_x", "offset_y", "pixel_format", "fps", "fps_enable",
    "exposure_us", "exposure_raw", "exposure_auto", "gain_db", "gain_raw", "gain_auto",
    "driver_exposure", "driver_gain",
    IMAGE_PART, SAVE_ALL_2D,
})
PREVIEW_TIMEOUT_S = 4.
_RETRY_DELAYS_S = (1., 2., 4.)
_STEADY_STREAM_NS = 10_000_000_000
_RETRY_ERRORS = frozenset({
    "CAMERA_DISCONNECTED", "CAMERA_ID_UNRESOLVED", "CAMERA_UNAVAILABLE",
    "PREVIEW_TIMEOUT", "UVC_FRAME_UNAVAILABLE",
})
PART_REQUIRED = "part_required"


def _same_value(actual, expected):
    if type(actual) in (int, float) and type(expected) in (int, float):
        return actual == expected or (type(expected) is float and abs(actual) <= 1e15
                                      and isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8))
    return type(actual) is type(expected) and actual == expected


def _feature_readback(features, name):
    feature = features.get(name)
    return feature.value if feature and feature.readable else None


def _check_readback(features, expected):
    for name, value in expected.items():
        actual = _feature_readback(features, name)
        if not _same_value(actual, value):
            raise CaptureError("FEATURE_READBACK_MISMATCH", f"{name}: requested {value!r}, read back {actual!r}")


def _image_layout(image):
    return tuple((part.pixels.shape, part.pixel_format, part.significant_bits, part.color_model,
                  part.metadata.get(IMAGE_PART, {}).get("selector"))
                 for part in (image, *image.auxiliary_images))


@dataclass(frozen=True)
class _PreviewBinding:
    device: Device
    layout: tuple
    readback: dict
    first_receipt_ns: int


class PreviewSession:
    def __init__(self, cameras, registry: CameraRegistry):
        self._cameras = {camera.id: camera for camera in cameras}
        self.registry = registry
        self._mutex = threading.Lock()
        self._changed = threading.Condition(self._mutex)
        self._stop = threading.Event()
        self._threads = []
        self._latest = {}
        self._parameters = {}
        self._trigger_support = {}
        self._pending = {}
        self._bindings = {}
        self._steady = set()
        self._manual_retry = set()
        self._revision = 0
        self._states = {identity: {"state": "connecting", "error": None} for identity in self._cameras}

    @property
    def cameras(self):
        with self._mutex:
            return tuple(self._cameras.values())

    def start(self):
        with self._mutex:
            if self._threads or self._stop.is_set():
                return
            for identity in self._cameras:
                worker = threading.Thread(target=self._preview, args=(identity,), name=f"preview-{identity}", daemon=True)
                self._threads.append(worker)
                worker.start()

    def stop(self):
        with self._changed:
            self._stop.set()
            self._changed.notify_all()

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
            return deepcopy(self._states)

    @property
    def evidence_revision(self) -> int | None:
        """Return a revision only while every selected camera has live evidence."""
        with self._mutex:
            if not self._stop.is_set() and self._states and all(
                    status["state"] == "preview" for status in self._states.values()):
                return self._revision
            return None

    def retry(self, identity):
        """Wake only the failed camera's owner to open the configured device again."""
        with self._changed:
            status = self._states.get(identity, {})
            if (self._stop.is_set() or not status.get("retryable")
                    or status.get("state") not in ("failed", "retry_wait")):
                raise CaptureError("PREVIEW_RETRY_UNAVAILABLE", f"cannot retry preview for {identity}")
            self._manual_retry.add(identity)
            self._states[identity] = {**status, "state": "reconnecting"}
            self._changed.notify_all()

    def parameters(self, identity):
        """Return the last idle feature readback for an active camera."""
        with self._mutex:
            if self._states.get(identity, {}).get("state") != "preview":
                return None
            device, features = self._parameters[identity]
            editable = {name: replace(feature, writable=feature.writable and feature.readable
                                      and feature.value is not None and name in PREVIEW_SETTINGS)
                        for name, feature in features.items()}
            return device, editable, dict(self._cameras[identity].settings)

    def trigger_support(self, identity) -> TriggerSupport | None:
        with self._mutex:
            if self._states.get(identity, {}).get("state") != "preview":
                return None
            return self._trigger_support.get(identity)

    def settings_check(self, config: CameraConfig, *, include_readback=False) -> SettingsCheck | None:
        """Use the device readback bound to the active preview configuration."""
        with self._mutex:
            if self._cameras.get(config.id) != config or self._states[config.id]["state"] != "preview":
                return None
            device, features = self._parameters[config.id]
            checked = tuple(name for name, value in config.settings.items()
                            if _same_value(_feature_readback(features, name), value))
            deferred = tuple(name for name in config.settings if name not in checked)
            return SettingsCheck("preview_readback", checked, deferred, readback=dict(features) if include_readback else None,
                                 device=device if include_readback else None)

    def camera_binding(self, identity):
        """Return the current connection and settings identity for a ready camera."""
        with self._mutex:
            if self._states.get(identity, {}).get("state") == "preview" and not self._stop.is_set():
                return self._bindings.get(identity)
        return None

    def bound_image(self, identity):
        """Read the latest image and its connection/settings identity together."""
        with self._mutex:
            if self._states.get(identity, {}).get("state") != "preview" or self._stop.is_set():
                return None
            return self._bindings[identity], self._latest[identity]

    def reconfigure(self, identity, settings, *, expected_binding=None, verify=None) -> Future[tuple[CameraConfig, Image]]:
        """Submit a read-back-verified adjustment to the camera's owning thread."""
        config = CameraConfig(identity, settings_object(settings))
        with self._changed:
            status = self._states.get(identity, {})
            if expected_binding is not None and self._bindings.get(identity) is not expected_binding:
                raise CaptureError("EXPOSURE_SCOPE_CHANGED", "the preview connection or camera settings changed")
            if self._stop.is_set() or status.get("state") not in ("preview", PART_REQUIRED):
                raise CaptureError("PREVIEW_NOT_READY", f"preview for {identity} is not ready for adjustment")
            if status["state"] == PART_REQUIRED and not any(
                    option.selector == config.settings.get(IMAGE_PART) for option in status["image_parts"]):
                raise CaptureError("PREVIEW_PART_REQUIRED", "请从此相机当前提供的图像部件中选择。")
            future = Future()
            future.set_running_or_notify_cancel()
            self._pending[identity] = config, future, verify
            self._states[identity] = {**status, "state": "adjusting", "error": None}
            self._latest.pop(identity, None)
            self._changed.notify_all()
            return future

    def _first_image(self, camera, features=None):
        deadline = time.monotonic() + PREVIEW_TIMEOUT_S
        values = {name: feature.value for name, feature in (features or {}).items() if feature.readable}
        while not self._stop.is_set() and time.monotonic() < deadline:
            image = camera.read(.2)
            if image is not None:
                if IMAGE_PART in image.metadata:
                    return image
                height, width = image.pixels.shape[:2]
                geometry = {"width": width, "height": height, "pixel_format": image.pixel_format}
                if all(values.get(name) in (None, value) for name, value in geometry.items()):
                    return image
        if self._stop.is_set():
            raise CaptureError("PREVIEW_STOPPED", "preview stopped before a new image arrived")
        raise CaptureError("PREVIEW_TIMEOUT", f"{camera.device.id} did not deliver a preview frame with the expected geometry")

    def _publish(self, camera, config, features, image, *, error=None):
        identity = config.id
        try:
            support = camera.trigger_support()
        except Exception as exc:
            support = TriggerSupport(error=str(exc))
        values = {name: feature.value for name, feature in features.items()}
        state = {"state": "preview", "error": None, "features": values}
        if SAVE_ALL_2D in values:
            state["archived_parts"] = 1 + len(image.auxiliary_images)
        if part := image.metadata.get(IMAGE_PART):
            height, width = image.pixels.shape[:2]
            values.update(width=width, height=height, pixel_format=image.pixel_format,
                          offset_x=part["offset_x"], offset_y=part["offset_y"])
            state.update(image_parts=getattr(camera, "image_part_options", ()), image_part=part["selector"])
        with self._mutex:
            self._cameras[identity] = config
            self._latest[identity] = image
            self._parameters[identity] = camera.device, features
            self._trigger_support[identity] = support
            stable_features = {name: value for name, value in config.settings.items()
                               if _same_value(_feature_readback(features, name), value)}
            stable_features.update({name: value for name in SENSOR_WINDOW_FIELDS
                                    if (value := _feature_readback(features, name)) is not None})
            self._bindings[identity] = _PreviewBinding(camera.device, _image_layout(image), stable_features,
                                                       image.received_monotonic_ns)
            self._revision += 1
            self._states[identity] = state
            request = self._pending.pop(identity, None)
        if request:
            future = request[1]
            if error:
                future.set_exception(error)
            else:
                future.set_result((config, image))

    def _offer_parts(self, camera, identity, problem, *, error=None):
        if not problem.options:
            raise problem
        camera.stop()
        # Keep descriptors while waiting without holding the SDK buffer through a traceback.
        problem = ImagePartSelectionError(problem.code, str(problem), problem.options)
        with self._mutex:
            self._latest.pop(identity, None)
            self._parameters.pop(identity, None)
            self._trigger_support.pop(identity, None)
            self._states[identity] = {"state": PART_REQUIRED, "error": str(problem), "error_code": problem.code,
                                      "image_parts": problem.options}
            request = self._pending.pop(identity, None)
        if request:
            request[1].set_exception(error or problem)
        return problem

    def _adjust(self, camera, current, requested, previous, verify=None):
        camera.stop()
        baseline = {name: feature.value for name, feature in camera.features().items()
                    if name in PREVIEW_SETTINGS and feature.readable and feature.writable and feature.value is not None}
        changed = {name: value for name, value in requested.settings.items()
                   if current.settings.get(name) != value}
        choosing_part = isinstance(previous, ImagePartSelectionError)
        if choosing_part:
            # Confirming the same selector is intentional if the part geometry changed.
            changed[IMAGE_PART] = requested.settings[IMAGE_PART]
        removed = current.settings.keys() - requested.settings.keys()
        if IMAGE_PART in removed:
            changed[IMAGE_PART] = AUTO_PART
        if SAVE_ALL_2D in removed:
            changed[SAVE_ALL_2D] = False
        try:
            for name in changed.keys() | removed:
                if name not in baseline:
                    raise CaptureError("PREVIEW_SETTING_UNAVAILABLE", f"cannot adjust {name} with readback in preview")
            camera.set_features(changed)
            features = camera.features()
            expected = {name: value for name, value in current.settings.items()
                        if name not in removed and _same_value(baseline.get(name), value)}
            _check_readback(features, {**expected, **changed})
            if self._stop.is_set():
                raise CaptureError("PREVIEW_STOPPED", "preview stopped while updating camera parameters")
            camera.start()
            image = self._first_image(camera, features)
            if verify is not None:
                def read_trial(timeout):
                    if self._stop.is_set():
                        raise CaptureError("PREVIEW_STOPPED", "preview stopped while verifying camera parameters")
                    return camera.read(timeout)

                image = verify(image, read_trial)
                if self._stop.is_set():
                    raise CaptureError("PREVIEW_STOPPED", "preview stopped while verifying camera parameters")
        except Exception as exc:
            try:
                camera.stop()
                observed = camera.features()
                restore = {name: value for name, value in baseline.items()
                           if not _same_value(observed[name].value if name in observed else None, value)}
                camera.set_features(restore)
                features = camera.features()
                _check_readback(features, baseline)
                if self._stop.is_set():
                    raise CaptureError("PREVIEW_STOPPED", "preview released after restoring camera settings")
                part_error = CaptureError("PREVIEW_PART_ADJUST_RESTORED", str(exc))
                if choosing_part:
                    problem = exc if isinstance(exc, ImagePartSelectionError) and exc.options else previous
                    return current, self._offer_parts(camera, current.id, problem, error=part_error)
                camera.start()
                try:
                    image = self._first_image(camera, features)
                except ImagePartSelectionError as problem:
                    return current, self._offer_parts(camera, current.id, problem, error=part_error)
                before, after = previous.metadata.get(IMAGE_PART), image.metadata.get(IMAGE_PART)
                if bool(before) != bool(after) or before and (
                        before["selector"] != after["selector"] or previous.pixels.shape != image.pixels.shape):
                    problem = ImagePartSelectionError("GENTL_PART_MISSING", "恢复后收到的图像来源发生变化。",
                                                      getattr(camera, "image_part_options", ()))
                    return current, self._offer_parts(camera, current.id, problem, error=part_error)
            except Exception as restore_error:
                if self._stop.is_set() and getattr(restore_error, "code", "") == "PREVIEW_STOPPED":
                    raise
                raise CaptureError("PREVIEW_RESTORE_FAILED", f"{exc}\n{restore_error}") from restore_error
            self._publish(camera, current, features, image,
                          error=CaptureError("PREVIEW_ADJUST_RESTORED", str(exc)))
            return current, image
        self._publish(camera, requested, features, image)
        return requested, image

    def _forget_live(self, identity):
        self._parameters.pop(identity, None)
        self._trigger_support.pop(identity, None)
        self._latest.pop(identity, None)

    def _await_retry(self, identity, failure, attempts):
        code = getattr(failure, "code", None)
        retryable = code != "UVC_DRIVER_TIMEOUT"
        with self._changed:
            binding = self._bindings.get(identity)
            if identity in self._steady:
                attempts = 0
            device = binding.device if binding else None
            known_identity = device and (device.identity_scope == "host_device_path" or
                                         (device.identity_scope == "serial_number" and device.serial))
            automatic = (retryable and known_identity
                         and code in _RETRY_ERRORS and attempts < len(_RETRY_DELAYS_S))
            delay = _RETRY_DELAYS_S[attempts] if automatic else None
            status = {"state": "retry_wait" if automatic else "failed", "error": str(failure),
                      "error_code": code, "retryable": retryable}
            if automatic:
                status.update(retry_number=attempts + 1, retry_limit=len(_RETRY_DELAYS_S))
            if binding:
                status["identity_scope"] = binding.device.identity_scope
            self._forget_live(identity)
            self._states[identity] = status
            pending = self._pending.pop(identity, None)
        if pending:
            pending[1].set_exception(CaptureError(failure.code, str(failure)))
        del pending
        with self._changed:
            self._changed.wait_for(lambda: self._stop.is_set() or identity in self._manual_retry, timeout=delay)
            if self._stop.is_set():
                return None
            manual = identity in self._manual_retry
            self._manual_retry.discard(identity)
            self._states[identity] = {**status, "state": "reconnecting"}
            return 0 if manual else attempts + 1

    def _preview(self, identity):
        attempts = 0
        reconnect = False
        try:
            while not self._stop.is_set():
                with self._mutex:
                    config = self._cameras[identity]
                    self._steady.discard(identity)
                try:
                    self._run_camera(config, reconnect=reconnect)
                except Exception as exc:
                    if self._stop.is_set():
                        return
                    # Keep the diagnostic without retaining the camera and raw
                    # preview buffers through an exception traceback.
                    failure = CaptureError(getattr(exc, "code", "PREVIEW_DEVICE_ERROR"), str(exc))
                else:
                    return
                attempts = self._await_retry(identity, failure, attempts)
                if attempts is None:
                    return
                reconnect = True
        finally:
            with self._mutex:
                self._states[identity] = {"state": "disconnected", "error": None}
                self._forget_live(identity)
                self._bindings.pop(identity, None)
                self._steady.discard(identity)
                self._manual_retry.discard(identity)
                request = self._pending.pop(identity, None)
            if request:
                request[1].set_exception(CaptureError("PREVIEW_STOPPED", "preview was stopped"))

    def _run_camera(self, config, *, reconnect):
        identity = config.id
        with self._mutex:
            binding = self._bindings.get(identity) if reconnect else None
        with self.registry.camera(identity, refresh=reconnect) as camera:
            if camera.device.id != identity or (binding and camera.device != binding.device):
                raise CaptureError("PREVIEW_IDENTITY_CHANGED", f"{identity} resolved to a different camera")
            camera.set_features(config.settings)
            features = camera.features()
            if binding:
                _check_readback(features, binding.readback)
            if self._stop.is_set():
                return
            camera.start()
            try:
                try:
                    previous = self._first_image(camera)
                except ImagePartSelectionError as problem:
                    previous = self._offer_parts(camera, identity, problem)
                else:
                    if binding and _image_layout(previous) != binding.layout:
                        raise CaptureError("PREVIEW_GEOMETRY_CHANGED", f"{identity} returned a different image layout")
                    self._publish(camera, config, features, previous)
                last = time.monotonic()
                next_preview = last + 1 / 12
                while not self._stop.is_set():
                    with self._changed:
                        if isinstance(previous, ImagePartSelectionError):
                            self._changed.wait_for(lambda: self._stop.is_set() or identity in self._pending)
                        if self._stop.is_set():
                            break
                        request = self._pending.get(identity)
                    if request:
                        config, previous = self._adjust(camera, config, request[0], previous, request[2])
                        last = time.monotonic()
                        next_preview = last + 1 / 12
                        continue
                    try:
                        image = camera.read(0.2)
                    except ImagePartSelectionError as problem:
                        previous = self._offer_parts(camera, identity, problem)
                        continue
                    now = time.monotonic()
                    if image is not None:
                        last = now
                        if now >= next_preview:
                            with self._mutex:
                                if identity not in self._pending:
                                    self._latest[identity] = image
                                    previous = image
                                    if image.received_monotonic_ns - self._bindings[identity].first_receipt_ns >= _STEADY_STREAM_NS:
                                        self._steady.add(identity)
                                    if IMAGE_PART in image.metadata:
                                        self._states[identity]["image_parts"] = getattr(camera, "image_part_options", ())
                            next_preview = now + 1 / 12
                    elif now - last > PREVIEW_TIMEOUT_S:
                        raise CaptureError("PREVIEW_TIMEOUT", f"{identity} stopped delivering preview frames")
            except Exception as exc:
                with self._mutex:
                    self._forget_live(identity)
                    self._states[identity] = {"state": "releasing", "error": str(exc),
                                              "error_code": getattr(exc, "code", None)}
                raise
            finally:
                camera.stop()


def _gray_samples(pixels: np.ndarray, color_model: str) -> np.ndarray:
    conversion = {
        "rgb": cv2.COLOR_RGB2GRAY, "bgr": cv2.COLOR_BGR2GRAY,
        "rgba": cv2.COLOR_RGBA2GRAY, "bgra": cv2.COLOR_BGRA2GRAY,
    }.get(color_model)
    return cv2.cvtColor(pixels, conversion) if conversion else pixels


def gray_pixels(image: Image) -> np.ndarray:
    return _gray_samples(image.pixels, image.color_model)


def preview_pixels(image: Image, *, palette: str = "native") -> np.ndarray:
    """Return an 8-bit display-only view, scaled to the declared full code range."""
    maximum = (1 << image.significant_bits) - 1
    if palette == "inferno":
        samples = cv2.convertScaleAbs(gray_pixels(image), alpha=255 / maximum)
        return cv2.cvtColor(cv2.applyColorMap(samples, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    pixels = cv2.convertScaleAbs(image.pixels, alpha=255 / maximum)
    conversion = {"bgr": cv2.COLOR_BGR2RGB, "bgra": cv2.COLOR_BGRA2RGB, "rgba": cv2.COLOR_RGBA2RGB}.get(image.color_model)
    return cv2.cvtColor(pixels, conversion) if conversion else pixels


class FocusComparison:
    """Compare medians of three fresh frames from a single preview context."""

    def __init__(self, received_ns: int):
        self._received_ns = received_ns
        self._recent = deque(maxlen=3)
        self.current: float | None = None
        self.highest: float | None = None
        self.has_detail = True

    def update(self, received_ns: int, response: float | None):
        if received_ns <= self._received_ns:
            return
        self._received_ns = received_ns
        self.has_detail = response is not None and response > 0
        if not self.has_detail:
            self._recent.clear()
            self.current = None
            return
        self._recent.append(response)
        if len(self._recent) == self._recent.maxlen:
            self.current = median(self._recent)
            self.highest = max(self.highest or 0., self.current)


def _inspection_stride(samples, *, bayer=False, pixel_limit=500_000):
    height, width = samples.shape[:2]
    stride = max(1, int(np.ceil(np.sqrt(height * width / pixel_limit))))
    # An even stride on the mosaic repeatedly samples the same filter.
    stride += int(bayer and stride % 2 == 0)
    while ((height + stride - 1) // stride) * ((width + stride - 1) // stride) > pixel_limit:
        stride = stride * 2 + int(bayer)
    return stride


def _bayer_green(pixels, pixel_format, x, y):
    pattern = pixel_format[5:7]
    if not pixel_format.startswith("Bayer") or pattern not in ("RG", "BG", "GR", "GB"):
        return None
    green_column = int(pattern in ("RG", "BG"))
    return pixels[y % 2::2, (green_column - x) % 2::2]


def _focus_response(samples, maximum):
    if samples is None or min(samples.shape) < 5:
        return None
    scaled = samples.astype(np.float32) * (255. / maximum)
    smoothed = cv2.GaussianBlur(scaled, (3, 3), .65)
    return float(cv2.Laplacian(smoothed, cv2.CV_32F)[2:-2, 2:-2].var())


def _inspection_pixels(image: Image, roi):
    pixels = image.pixels
    x = y = 0
    if roi is not None:
        import ezdic_core as core

        if not core.rect_is_inside_image(roi, pixels.shape):
            raise CaptureError("ROI_OUT_OF_BOUNDS", "the inspection ROI is outside the acquired image")
        x, y, width, height = core.clamp_rect(roi, pixels.shape)
        pixels = pixels[y:y + height, x:x + width]
    return pixels, x, y


QUALITY_ALERT_FRACTION = .20


def image_quality(image: Image, roi: tuple[int, int, int, int] | None = None, *, pixel_limit=500_000,
                  measure_texture=True) -> dict:
    """Measure a single region of the raw image for preview and review."""
    if pixel_limit < 1:
        raise ValueError("pixel_limit must be positive")
    if roi is not None and measure_texture:
        import ezdic_core as core

    pixels, x, y = _inspection_pixels(image, roi)

    maximum = (1 << image.significant_bits) - 1
    color = image.color_model
    bayer = color == "bayer"
    stride = _inspection_stride(pixels, bayer=bayer, pixel_limit=pixel_limit)
    sampled_pixels = pixels[::stride, ::stride]
    samples = _gray_samples(sampled_pixels, color)
    bins = np.bincount(np.minimum(samples.astype(np.uint32) * 64 // (maximum + 1), 63).ravel(), minlength=64)
    lo, med, hi = np.percentile(samples, [5, 50, 95])
    high = .99 * maximum
    if sampled_pixels.ndim == 3:
        clipped = (sampled_pixels[..., 0] >= high) | (sampled_pixels[..., 1] >= high) | (sampled_pixels[..., 2] >= high)
    else:
        clipped = samples >= high
    region = _bayer_green(pixels, image.pixel_format, x, y) if bayer else None
    focus_samples = samples
    if bayer:
        focus_samples = None
        if region is not None:
            green_stride = _inspection_stride(region, pixel_limit=pixel_limit)
            focus_samples = region[::green_stride, ::green_stride]
    result = {
        "histogram": bins.tolist(), "code_max": maximum,
        "percentiles_p5_p50_p95": [float(lo), float(med), float(hi)],
        "dark_fraction": float(np.mean(samples <= 0.01 * maximum)),
        "bright_fraction": float(np.mean(clipped)),
        "focus_laplacian_variance": _focus_response(focus_samples, maximum),
        "histogram_stride_px": stride, "texture": None, "texture_error": None,
    }
    if roi is not None and measure_texture:
        if pixels.shape[0] * pixels.shape[1] > 1_000_000:
            result["texture_error"] = "INSPECTION_ROI_TOO_LARGE"
        elif not bayer or (region is not None and region.size):
            if not bayer:
                region = samples if stride == 1 else _gray_samples(pixels, color)
            region8 = cv2.convertScaleAbs(region, alpha=255 / maximum)
            metrics = core.roi_texture_metrics(region8, (0, 0, region.shape[1], region.shape[0]))
            result["texture"] = {
                **metrics,
                "failure_code": core.texture_failure_code(metrics, 8.0, 25.0, 0.20),
            }
    return result
