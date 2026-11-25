import json
import subprocess
import sys

from straintrace_capture import CaptureSession, ExperimentReader
from test_capture_lifecycle import config, wait_until


def test_capture_continues_while_live_analysis_stalls_exits_and_restarts(tmp_path):
    session = CaptureSession(config([
        {"mode": "single", "role": "reference"},
        {"mode": "continuous", "duration_s": 30},
    ], **{"sim:0": {"fps": 20.0, "motion_px": 0.015, "strain_per_frame": 0.00012, "pixel_format": "Mono12"}}),
        tmp_path / "capture")
    marker = tmp_path / "first-analysis-frame.json"
    ready = marker.with_suffix(".ready")
    live_config = {
        "gauges": [{"name": "G01", "roi1": [41, 20, 27, 27], "roi2": [41, 73, 27, 27], "strain_mode": "y"}],
    }
    child = """
import sys
import time
from pathlib import Path
from straintrace_capture.live import LiveAnalysis
from straintrace_capture.storage import json_text
with LiveAnalysis(sys.argv[1], **__import__('json').loads(sys.argv[3])) as analyst:
    marker = Path(sys.argv[2])
    marker.with_suffix('.ready').touch()
    while (row := analyst.poll()) is None:
        time.sleep(.05)
    pending = marker.with_suffix('.tmp')
    pending.write_text(json_text(row), encoding='utf-8')
    pending.rename(marker)
    time.sleep(60)
"""
    process = subprocess.Popen([sys.executable, "-c", child, str(session.output), str(marker), json.dumps(live_config)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started = False
    try:
        wait_until(ready.is_file, timeout=35)
        session.start()
        started = True
        wait_until(marker.is_file, timeout=20)
        first = json.loads(marker.read_text(encoding="utf-8"))
        assert first["provisional"] and first["gauges"][0]["strain_valid"]
        assert process.poll() is None
        before = session.status()["cameras"]["sim:0"]["saved"]
        wait_until(lambda: session.status()["cameras"]["sim:0"]["saved"] > before + 2)
        assert process.poll() is None
        process.kill()
        process.wait(timeout=8)
        after_exit = session.status()["cameras"]["sim:0"]["saved"]
        wait_until(lambda: session.status()["cameras"]["sim:0"]["saved"] > after_exit + 2)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=8)
        if started:
            session.request_stop()
            final = session.wait(12)
    assert final["state"] == "stopped"
    cfg = tmp_path / "follow.json"
    cfg.write_text(json.dumps(live_config), encoding="utf-8")
    restarted = subprocess.run([sys.executable, "-m", "straintrace_capture", "follow", str(session.output),
                                "--config", str(cfg)], text=True, capture_output=True, timeout=30)
    assert restarted.returncode == 0, restarted.stderr + restarted.stdout
    reference, latest = [json.loads(line) for line in restarted.stdout.splitlines()]
    assert reference["reference_frame_id"] == first["reference_frame_id"]
    with ExperimentReader(session.output) as reader:
        frames = reader.frames()
        assert latest["skipped_committed_frames"] == len(frames) - 2
        assert latest["frame_id"] == frames[-1]["frame_id"]
        integrity = reader.verify()
        assert integrity["ok"] and not integrity["reported_camera_losses"]
