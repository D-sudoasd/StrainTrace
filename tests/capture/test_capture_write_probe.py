from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, inspect_plan, prepare_trial, trial_report
from straintrace_capture.cameras.base import Device, Image
from straintrace_capture.storage import ExperimentWriter
from straintrace_capture import write_probe


def _plan(durability="buffered", *, camera="sim:0", settings=None):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "write test", "durability": durability,
        "cameras": [{"id": camera, "settings": settings or {}}],
        "phases": [{"mode": "single", "role": "reference"}, {"mode": "timed", "fps": 20, "frame_count": 50}],
    })


@pytest.mark.parametrize("durability", ["buffered", "write_through"])
def test_real_writer_probe_is_scoped_and_verified(tmp_path, durability):
    config = _plan(durability, settings={"width": 256, "height": 160, "pixel_format": "Mono12"})
    output = tmp_path / "未创建的下级目录" / "计划实验"
    marker = tmp_path / "原文件.txt"
    marker.write_bytes(b"existing data")
    report = inspect_plan(config, output)
    reference, timed = write_probe.write_probe_phases(report)
    assert reference.planned_raw_bytes_per_s is None
    assert timed is write_probe.preferred_probe_phase((reference, timed))
    assert timed.round_bytes == 256 * 160 * 2 and timed.planned_raw_bytes_per_s == 20 * timed.round_bytes
    result = write_probe.measure_write_path(config, output, timed, expected_parent=tmp_path, max_mib=2)
    assert result["phase"] == 2 and result["durability"] == durability
    assert result["raw_bytes"] == result["frames"] * timed.round_bytes <= 2**21
    assert result["rounds"] == result["frames"] == result["tiffs"] >= 3
    assert result["tiff_bytes"] > result["raw_bytes"]
    assert result["flush_inclusive_seconds"] >= result["committed_seconds"] > 0
    assert 0 < result["flushed_raw_bytes_per_s"] <= result["committed_raw_bytes_per_s"]
    assert result["planned_raw_bytes_per_s"] == timed.planned_raw_bytes_per_s
    assert result["comparison"] in ("at_or_above_plan", "below_plan")
    assert list(tmp_path.iterdir()) == [marker] and marker.read_bytes() == b"existing data"


def test_probe_uses_complete_multipart_preview_geometry(tmp_path):
    camera = "gentl:mock:0"
    config = _plan(camera=camera)
    output = tmp_path / "计划"
    assert write_probe.preferred_probe_phase(write_probe.write_probe_phases(inspect_plan(config, output))) is None
    color = Image(np.zeros((24, 32, 3), np.uint8), "BGR8", 8, "bgr", 1, 1)
    mono = Image(np.zeros((60, 82), np.uint16), "Mono12", 12, "mono", 1, 1, auxiliary_images=(color,))
    report = inspect_plan(config, output, preview_images={camera: mono})
    phase = write_probe.preferred_probe_phase(write_probe.write_probe_phases(report))
    assert phase.round_bytes == mono.raw_bytes
    result = write_probe.measure_write_path(config, output, phase, max_mib=1)
    assert result["tiffs"] == result["frames"] * 2
    assert [part["shape"] for part in result["cameras"][0]["parts"]] == [(60, 82), (24, 32, 3)]
    assert not list(tmp_path.iterdir())
    trial_plan, saved = prepare_trial(config, output, phase_number=2)
    writer = ExperimentWriter(saved, trial_plan)
    try:
        writer.register_camera(Device(camera, "gentl", "fixture", "Two parts", "123", "fixture"), {})
        writer.commit_frames([writer.prepare_frame(camera, 1, 1, "test", mono, None, None, None, 0)])
        writer.set_state("completed")
    finally:
        writer.close()
    checked = trial_report(saved, inspect_images=False)
    report = inspect_plan(config, output, trial_reports=[checked])
    offline = write_probe.write_probe_phases(report)[1]
    assert offline.round_bytes == mono.raw_bytes and offline.cameras[0].parts == phase.cameras[0].parts
    assert offline.cameras[0].basis == "trial"


def test_probe_cancellation_scope_and_reserve_leave_no_test_files(tmp_path, monkeypatch):
    config = _plan(settings={"width": 192, "height": 120})
    output = tmp_path / "目标"
    phase = write_probe.preferred_probe_phase(write_probe.write_probe_phases(inspect_plan(config, output)))
    stop = Event()

    def progress(stage, _written, _frames):
        if stage == "writing":
            stop.set()

    with pytest.raises(CaptureError) as cancelled:
        write_probe.measure_write_path(config, output, phase, max_mib=1, cancelled=stop.is_set, progress=progress)
    assert cancelled.value.code == "WRITE_PROBE_CANCELLED" and not list(tmp_path.iterdir())
    with pytest.raises(CaptureError) as scope:
        write_probe.measure_write_path(config, output, phase, expected_parent=tmp_path.parent)
    assert scope.value.code == "WRITE_PROBE_SCOPE_CHANGED"
    monkeypatch.setattr(write_probe.shutil, "disk_usage", lambda _path: SimpleNamespace(free=config.min_free_bytes))
    with pytest.raises(CaptureError) as reserve:
        write_probe.measure_write_path(config, output, phase)
    assert reserve.value.code == "DISK_RESERVE_UNAVAILABLE" and not list(tmp_path.iterdir())
