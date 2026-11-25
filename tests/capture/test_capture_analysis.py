import json
import subprocess
import sys

import pytest

from straintrace_capture import CaptureSession
from straintrace_capture.handoff import analysis_settings, analysis_snapshot, write_snapshot

from test_capture_lifecycle import config


@pytest.mark.parametrize("mode", ["extensometer", "fullfield"])
def test_committed_capture_files_feed_stock_analysis_cli(tmp_path, mode):
    capture = CaptureSession(config([
        {"mode": "single", "role": "calibration"},
        {"mode": "single", "role": "reference"},
        {"mode": "timed", "fps": 8, "frame_count": 2},
    ], **{"sim:0": {"pixel_format": "Mono12", "motion_px": 0.02}}), tmp_path / "raw")
    capture.start()
    assert capture.wait(6)["state"] == "completed"
    binding = analysis_snapshot(capture.output, "sim:0")
    assert [frame["sequence"] for frame in binding["frames"]] == [2, 3, 4]
    template = {
        "schema_version": 1, "analysis_mode": mode,
        "start_frame_1based": 1, "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "results"),
        "export": {"write_manifest": True, "write_qc": True, "write_full_csv": True},
        "transaction": {"enabled": True, "archive_previous": True, "retain_failed_staging": True},
    }
    if mode == "extensometer":
        template["roi_groups"] = [{
            "name": "G01", "roi1": [41, 20, 27, 27], "roi2": [41, 73, 27, 27],
            "strain_mode": "y", "role": "none",
        }]
    else:
        template["field_roi"] = [24, 24, 80, 72]
        template["solver"] = {"name": "IC-GN", "subset_size_px": 19, "step_px": 12, "strain_window_px": 5,
                              "search_radius_px": 9}
    bound = analysis_settings(binding, template)
    path = write_snapshot(bound, tmp_path / "analysis.json")
    completed = subprocess.run([sys.executable, "-m", "ezdic_cli", "run", "--config", str(path)],
                               text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["scientific_ok"] and summary["manifest_verified"]
    manifest = json.loads((tmp_path / "results/run_manifest.json").read_text(encoding="utf-8"))
    assert [row["sha256"] for row in manifest["inputs"]] == [row["sha256"] for row in binding["frames"]]


def test_live_fullfield_carries_quality_and_frame_identity(tmp_path):
    from straintrace_capture.live import LiveAnalysis

    capture = CaptureSession(config([
        {"mode": "single", "role": "reference"}, {"mode": "single"},
    ], **{"sim:0": {"motion_px": 0., "strain_per_frame": 0.}}), tmp_path / "static")
    capture.start()
    assert capture.wait(6)["state"] == "completed"
    with LiveAnalysis(capture.output, field_roi=(24, 24, 80, 72), field_step=12, field_subset=19) as live:
        assert live.poll()["sequence"] == 1
        row = live.poll()
        assert row["provisional"] and row["field_quality"]["scientific_ok"]
        assert row["sequence"] == 2 and row["frame_id"].endswith("/sim:0/2")
        assert row["field_quality"]["correlation_valid_fraction"] == 1.
        assert live.poll() is None
