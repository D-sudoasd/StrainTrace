"""Measure the real Capture writer on the user's selected local volume."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil


def benchmark(output: Path, duration: int):
    from straintrace_capture import CaptureConfig, CaptureSession, ExperimentReader

    process = psutil.Process()
    table = []
    cases = [
        ("mono8", 1, 512, 384, "Mono8", 30, 8),
        ("mono12-dual", 2, 960, 720, "Mono12", 15, 8),
        ("mono16-large", 1, 1600, 1200, "Mono16", 10, 8),
        ("dual-endurance", 2, 640, 480, "Mono12", 12, duration),
    ]
    for name, cameras, width, height, fmt, fps, duration_s in cases:
        root = output / name
        config = CaptureConfig.from_dict({
            "config_version": 1, "name": name,
            "cameras": [{"id": f"sim:{index}", "settings": {
                "width": width, "height": height, "pixel_format": fmt, "fps": float(fps), "motion_px": .01,
                "strain_per_frame": 0.,
            }} for index in range(cameras)],
            "phases": [{"mode": "continuous", "duration_s": duration_s}],
        })
        capture = CaptureSession(config, root)
        baseline = peak = process.memory_info().rss
        start = time.perf_counter()
        capture.start()
        while True:
            peak = max(peak, process.memory_info().rss)
            try:
                status = capture.wait(.1)
                break
            except TimeoutError:
                pass
        acquisition_s = time.perf_counter() - start
        with ExperimentReader(root) as reader:
            check_started = time.perf_counter()
            verification = reader.verify()
            verify_s = time.perf_counter() - check_started
            frames = reader.frames()
        intervals = {}
        for camera in config.cameras:
            ticks = np.array([row["timestamps"]["received_monotonic"]["value"]
                              for row in frames if row["camera_id"] == camera.id], dtype=np.int64)
            if len(ticks) > 1:
                deltas = np.diff(ticks) / 1e9
                intervals[camera.id] = {"mean_fps": float(1 / deltas.mean()),
                                        "p95_inter_frame_ms": float(1000 * np.quantile(deltas, .95))}
        case = {
            "name": name, "camera_count": cameras, "width": width, "height": height, "pixel_format": fmt,
            "target_camera_fps": fps, "duration_s": duration_s, "elapsed_s": acquisition_s,
            "receipt_cadence": intervals, "peak_rss_bytes": peak, "initial_rss_bytes": baseline,
            "verify_s": verify_s,
            "verification": verification, "status": status,
        }
        table.append(case)
        print(f"{name}: {len(frames)} committed, {status['disk_bytes_per_s'] / 2**20:.2f} MiB/s, "
              f"queue peak {status['queue_high_water']}, verification {verification['complete']}", flush=True)
    return {
        "schema": "straintrace.capture-benchmark/v1",
        "environment": {
            "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": psutil.cpu_count(),
            "python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
            "available_memory_bytes": psutil.virtual_memory().total, "output_volume": output.anchor,
        },
        "cases": table,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new writable benchmark folder")
    parser.add_argument("--duration", default=60, type=int, help="endurance phase seconds")
    args = parser.parse_args()
    if args.duration < 1:
        parser.error("duration must be positive")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = benchmark(output, args.duration)
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if all(case["verification"]["complete"] for case in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
