"""A bounded test of the TIFF writer on the planned output filesystem."""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .cameras._genapi import image_layout
from .cameras.base import Device, Image, receipt_times
from .config import CaptureConfig, CaptureError
from .output import output_ancestor, resolve_output_path
from .session import WRITE_BATCH_FRAME_LIMIT
from .storage import ExperimentReader, ExperimentWriter, frame_image_records
from .trial import volume_identity


MIB = 2**20
DEFAULT_PROBE_MIB = 128
MAX_PROBE_MIB = 512
PROBE_SECONDS = 6
MAX_ROUNDS = 128


@dataclass(frozen=True, slots=True)
class ProbeImage:
    shape: tuple[int, ...]
    dtype: str
    pixel_format: str
    significant_bits: int
    color_model: str

    @property
    def raw_bytes(self):
        return math.prod(self.shape) * np.dtype(self.dtype).itemsize


@dataclass(frozen=True, slots=True)
class ProbeCamera:
    camera_id: str
    basis: str
    parts: tuple[ProbeImage, ...]


@dataclass(frozen=True, slots=True)
class WriteProbePhase:
    number: int
    mode: str
    role: str
    cameras: tuple[ProbeCamera, ...]
    planned_raw_bytes_per_s: float | None
    unresolved: tuple[str, ...]

    @property
    def round_bytes(self):
        return sum(part.raw_bytes for camera in self.cameras for part in camera.parts)


def _probe_parts(row):
    if row["frame_bytes"] is None:
        return ()
    if parts := row.get("saved_parts"):
        images = tuple(ProbeImage(tuple(part["shape"]), part["dtype"], part["pixel_format"],
                                  part["significant_bits"], part["color_model"]) for part in parts)
    else:
        try:
            bits, color = image_layout(row["pixel_format"])
        except CaptureError:
            return ()
        axes = (row["height_px"], row["width_px"])
        shape = axes if color in ("mono", "bayer") else (*axes, len(color))
        images = (ProbeImage(shape, "uint8" if bits <= 8 else "uint16", row["pixel_format"], bits, color),)
    return images if sum(part.raw_bytes for part in images) == row["frame_bytes"] else ()


def write_probe_phases(report):
    phases = []
    for row in report["phases"]:
        cameras, unresolved = [], []
        for camera_id, image in row["camera_geometry"].items():
            if parts := _probe_parts(image):
                cameras.append(ProbeCamera(camera_id, image["basis"], parts))
            else:
                unresolved.append(camera_id)
        demand = row["configured_stream_raw_bytes_per_s"]
        if row["mode"] in ("timed", "event"):
            demand = row["scheduled_raw_bytes_per_s"]
        phases.append(WriteProbePhase(row["phase"], row["mode"], row["role"], tuple(cameras), demand, tuple(unresolved)))
    return tuple(phases)


def preferred_probe_phase(phases):
    usable = (phase for phase in phases if not phase.unresolved and phase.cameras)
    return max(usable, key=lambda phase: (phase.planned_raw_bytes_per_s or 0, phase.role == "test", -phase.number),
               default=None)


def _synthetic_images(phase):
    rng = np.random.default_rng(4409)
    stamp = receipt_times()
    for camera in phase.cameras:
        images = []
        for spec in camera.parts:
            pixels = rng.integers(0, 2**spec.significant_bits, size=spec.shape, dtype=spec.dtype)
            images.append(Image(pixels, spec.pixel_format, spec.significant_bits, spec.color_model, *stamp))
        primary, *parts = images
        yield Image(primary.pixels, primary.pixel_format, primary.significant_bits, primary.color_model,
                    *stamp, auxiliary_images=tuple(parts))


def _probe_config(config, camera_ids):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "Storage probe", "durability": config.durability,
        "segment_frames": config.segment_frames, "min_free_bytes": config.min_free_bytes,
        "cameras": [{"id": camera} for camera in camera_ids], "phases": [{"mode": "manual"}],
    })


def measure_write_path(config, output, phase: WriteProbePhase, *, expected_parent=None, max_mib=DEFAULT_PROBE_MIB,
                       cancelled=None, progress=None):
    """Write synthetic frames with the actual recorder and discard the temporary experiment."""
    if type(max_mib) is not int or not 1 <= max_mib <= MAX_PROBE_MIB:
        raise CaptureError("WRITE_PROBE_LIMIT", f"测试上限须为 1 至 {MAX_PROBE_MIB} MiB。")
    if phase.unresolved or not phase.cameras:
        raise CaptureError("WRITE_PROBE_GEOMETRY", "请先连接预览或补齐图像尺寸和像素格式，再检查计划。")
    formal = resolve_output_path(output)
    parent = output_ancestor(formal)
    if expected_parent is not None and parent != Path(expected_parent).resolve():
        raise CaptureError("WRITE_PROBE_SCOPE_CHANGED", "目标盘位置已变化。请重新检查计划。")
    if not parent.is_dir():
        raise CaptureError("OUTPUT_PARENT_NOT_DIRECTORY", str(parent))
    volume = volume_identity(parent)
    round_bytes = phase.round_bytes
    if not round_bytes:
        raise CaptureError("WRITE_PROBE_GEOMETRY", "所选阶段缺少可写入的像素。")
    if round_bytes > max_mib * MIB:
        raise CaptureError("WRITE_PROBE_LIMIT", f"一轮合成图像超过 {max_mib} MiB 写入上限。")
    available = max(0, shutil.disk_usage(parent).free - config.min_free_bytes - MIB)
    rounds = min(MAX_ROUNDS, min(max_mib * MIB, available) // round_bytes)
    if not rounds:
        raise CaptureError("DISK_RESERVE_UNAVAILABLE", "目标盘剩余空间无法容纳测试图像并保留设置的空间。")

    def check():
        if cancelled is not None and cancelled():
            raise CaptureError("WRITE_PROBE_CANCELLED", "已取消写盘测试。")

    def update(stage, raw=0, count=0):
        if progress is not None:
            progress(stage, raw, count)
        check()

    update("preparing")
    images = tuple(_synthetic_images(phase))
    camera_ids = tuple(f"probe:{index}" for index in range(len(images)))
    sizes, durations, paths = 0, [], []
    with TemporaryDirectory(prefix=".straintrace-write-probe-", dir=parent) as temp:
        root = Path(temp) / "probe"
        writer = ExperimentWriter(root, _probe_config(config, camera_ids))
        try:
            for camera in camera_ids:
                writer.register_camera(Device(camera, "probe", "StrainTrace", "Synthetic image", None, "diagnostic"), {})
            started = perf_counter()
            for sequence in range(1, rounds + 1):
                if sequence > 1 and perf_counter() - started >= PROBE_SECONDS:
                    break
                check()
                tick = perf_counter()
                for first in range(0, len(images), WRITE_BATCH_FRAME_LIMIT):
                    stop = first + WRITE_BATCH_FRAME_LIMIT
                    batch = []
                    stamp = receipt_times()
                    # Reuse the generated pixel arrays and assign an independent host receipt.
                    for camera, image in zip(camera_ids[first:stop], images[first:stop], strict=True):
                        check()
                        parts = tuple(Image(part.pixels, part.pixel_format, part.significant_bits, part.color_model, *stamp)
                                      for part in image.auxiliary_images)
                        frame = Image(image.pixels, image.pixel_format, image.significant_bits, image.color_model, *stamp,
                                      auxiliary_images=parts)
                        batch.append(writer.prepare_frame(camera, sequence, 1, "test", frame, None, None, None, config.min_free_bytes))
                    writer.commit_frames(batch)
                    for prepared in batch:
                        for asset in frame_image_records(prepared.record):
                            paths.append(root / asset["path"])
                            sizes += asset["file_bytes"]
                durations.append(perf_counter() - tick)
                update("writing", sequence * round_bytes, sequence * len(images))
            writer.set_state("completed")
        finally:
            writer.close()
        committed_s = perf_counter() - started
        cycles = len(durations)
        receipts = cycles * len(images)
        raw_bytes = cycles * round_bytes
        update("flushing", raw_bytes, receipts)
        if config.durability == "buffered":
            for path in paths:
                check()
                with path.open("r+b", buffering=0) as stream:
                    os.fsync(stream.fileno())
        finished_s = perf_counter() - started
        update("verifying", raw_bytes, receipts)
        verify_started = perf_counter()
        with ExperimentReader(root) as reader:
            verification = reader.verify()
            if not verification["complete"] or len(reader.frames()) != receipts:
                raise CaptureError("WRITE_PROBE_VERIFY", "测试图像和台账未通过校验。")
        verification_s = perf_counter() - verify_started
        check()
    if volume is not None and volume_identity(parent) != volume:
        raise CaptureError("WRITE_PROBE_SCOPE_CHANGED", "测试期间目标卷发生变化。请重新检查计划。")
    rate = raw_bytes / finished_s
    planned = phase.planned_raw_bytes_per_s
    comparison = ("no_planned_rate" if planned is None else "short_sample" if cycles < 3
                  else "below_plan" if rate < planned else "at_or_above_plan")
    return {
        "schema": "straintrace.capture.write-probe/v1", "volume_path": str(parent), "volume_id": volume,
        "phase": phase.number, "durability": config.durability, "raw_bytes": raw_bytes, "tiff_bytes": sizes,
        "frames": receipts, "tiffs": len(paths), "rounds": cycles, "sample_budget_mib": max_mib,
        "committed_seconds": committed_s, "flush_inclusive_seconds": finished_s,
        "verification_seconds": verification_s,
        "committed_raw_bytes_per_s": raw_bytes / committed_s, "flushed_raw_bytes_per_s": rate,
        "p95_round_ms": float(np.percentile(durations, 95) * 1e3), "planned_raw_bytes_per_s": planned,
        "comparison": comparison, "cameras": [asdict(camera) for camera in phase.cameras],
    }
