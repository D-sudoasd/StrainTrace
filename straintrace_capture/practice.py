"""The built-in simulated exercise and its camera scope."""

from .cameras import CameraRegistry
from .config import CaptureConfig, CaptureError


def practice_config() -> CaptureConfig:
    return CaptureConfig.from_dict({
        "config_version": 1,
        "name": "模拟散斑练习",
        "cameras": [{"id": "sim:0", "settings": {"width": 640, "height": 480, "pixel_format": "Mono12"}}],
        "phases": [
            {"mode": "single", "role": "reference"},
            {"mode": "timed", "role": "test", "fps": 6, "frame_count": 12},
        ],
    })


def require_simulated_cameras(camera_ids) -> None:
    if any(identity.partition(":")[0] != "sim" for identity in camera_ids):
        raise CaptureError("PRACTICE_CAMERA_REQUIRED", "模拟练习只使用内置相机。请在图像采集主窗口使用实验相机。")


class PracticeCameraRegistry(CameraRegistry):
    def discover(self, backends=None):
        return super().discover(("sim",) if backends is None or "sim" in backends else ())

    def camera(self, identity, *, refresh=False):
        require_simulated_cameras((identity,))
        return super().camera(identity, refresh=refresh)
