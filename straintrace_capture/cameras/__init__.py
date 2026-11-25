"""Lazy provider registry used by CLI, API and desktop alike."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from threading import Lock

from ..config import CaptureError
from .base import Camera, Device


_PROVIDERS = {
    "sim": ("simulated", "StrainTrace speckle simulator"),
    "pylon": ("pylon", "Basler pypylon"),
    "gentl": ("gentl", "GenICam/GenTL producer"),
    "uvc": ("uvc", "OpenCV video input"),
}


class CameraRegistry:
    def __init__(
        self, *, cti_files: Sequence[str] = (),
        custom: dict[str, tuple[Callable[[], list[Device]], Callable[[str], Camera]]] | None = None,
    ):
        self.cti_files = tuple(str(Path(path).resolve()) for path in cti_files)
        self.custom = dict(custom or {})
        self._gentl_group = None
        self._mutex = Lock()

    def _provider(self, name: str):
        try:
            module, label = _PROVIDERS[name]
        except KeyError as exc:
            raise CaptureError("UNKNOWN_PROVIDER", f"unknown camera provider {name!r}") from exc
        try:
            return import_module(f".{module}", __name__)
        except ImportError as exc:
            raise CaptureError("PROVIDER_UNAVAILABLE", f"{label}: {exc}") from exc

    def discover(self, backends: Sequence[str] | None = None) -> dict:
        devices, providers = [], {}
        names = (*_PROVIDERS, *self.custom) if backends is None else backends
        for name in dict.fromkeys(names):
            try:
                if name in self.custom:
                    found = self.custom[name][0]()
                else:
                    found = self._provider(name).discover(self.cti_files)
                devices.extend(asdict(device) for device in found)
                providers[name] = {"state": "ready" if found else "no_devices", "count": len(found)}
            except CaptureError as exc:
                providers[name] = {"state": "unavailable", **exc.as_dict()}
            except Exception as exc:
                providers[name] = {"state": "error", "code": "DISCOVERY_FAILED", "message": str(exc)}
        return {"devices": devices, "providers": providers}

    def camera(self, identity: str) -> Camera:
        name, _, suffix = identity.partition(":")
        if not suffix:
            raise CaptureError("INVALID_CAMERA_ID", f"invalid camera ID: {identity!r}")
        if name in self.custom:
            return self.custom[name][1](identity)
        provider = self._provider(name)
        if name == "gentl":
            with self._mutex:
                if self._gentl_group is None:
                    self._gentl_group = provider.GenTLGroup(self.cti_files)
                return self._gentl_group.camera(identity)
        return provider.camera(identity, self.cti_files)


__all__ = ["Camera", "CameraRegistry", "Device"]
