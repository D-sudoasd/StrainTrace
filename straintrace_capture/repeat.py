"""Prepare a new experiment using the full plan of a recorded acquisition."""

from dataclasses import dataclass
from pathlib import Path

from .config import REPEAT_OF, REPEAT_PARENT, CaptureConfig, CaptureError
from .continuation import CONTINUATION_REFERENCE, verified_archived_calibrations
from .storage import TERMINAL_STATES, ExperimentReader


@dataclass(frozen=True)
class RepeatPlan:
    config: CaptureConfig
    source: Path
    source_name: str
    source_state: str
    cleared_anchors: tuple[int, ...]


def validate_repeat_destination(source: Path, destination: str | Path) -> Path:
    if not str(destination).strip():
        raise CaptureError("OUTPUT_REQUIRED", "请填写本次复拍的输出目录。")
    requested = Path(destination).expanduser()
    output = requested.resolve()
    if output.is_relative_to(source):
        raise CaptureError("OUTPUT_INSIDE_REPEAT_SOURCE", "请选择来源实验目录之外的新输出位置。")
    if requested.is_symlink() or output.exists():
        raise CaptureError("OUTPUT_EXISTS", "请选择一个尚未创建的实验目录。")
    return output


def prepare_repeat(root: str | Path, *, name: str | None = None, expected_experiment_id: str | None = None) -> RepeatPlan:
    """Copy the archived plan without opening cameras or creating a destination."""
    with ExperimentReader(root) as reader:
        record = reader.experiment
        identity = record["experiment_id"]
        if expected_experiment_id is not None and identity != expected_experiment_id:
            raise CaptureError("REPEAT_SCOPE_CHANGED", "来源目录中的实验标识已变化，请重新打开实验。")
        if record["state"] not in TERMINAL_STATES:
            raise CaptureError("REPEAT_ACTIVE", "请先结束采集或恢复中断台账，再准备复拍。")
        data = CaptureConfig.from_dict(record["config"]).to_dict()
        source_name = data["name"]
        metadata = data["metadata"]
        if metadata.get("capture_trial") or metadata.get("capture_checkerboard"):
            raise CaptureError("REPEAT_SOURCE_UNSUPPORTED", "请选择一次正式采集作为复拍的来源。")
        if name is not None:
            if not name.strip():
                raise CaptureError("REPEAT_NAME_REQUIRED", "请填写本次复拍的实验名称。")
            data["name"] = name.strip()
        elif not source_name.endswith(" · 复拍"):
            data["name"] = f"{source_name} · 复拍"
        anchors = tuple(index for index, phase in enumerate(data["phases"], 1) if phase.pop("hardware_pairing", None))
        data["calibration_files"] = verified_archived_calibrations(reader.root, record, "REPEAT_CALIBRATION_INVALID")
        for key in ("continuation_of", "continuation_parent", CONTINUATION_REFERENCE):
            metadata.pop(key, None)
        metadata[REPEAT_OF] = identity
        metadata[REPEAT_PARENT] = {"directory": str(reader.root)}
        return RepeatPlan(CaptureConfig.from_dict(data), reader.root, source_name, record["state"], anchors)
