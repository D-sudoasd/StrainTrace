import copy
import json
import sys

import numpy as np
import pytest

from benchmarks import real_necking_pilot as pilot


def test_comparison_checks_identities_and_uses_common_valid_points(tmp_path):
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    frame = {
        "frame": 50, "sha256": "same-target", "valid_n": 2, "strain_valid_n": 1,
        "effective_search_radius_px": 3, "elapsed_s": 1.0, "invalid_reason_counts": {},
    }
    before = {
        "schema": "straintrace-real-necking-pilot-v1", "reference_sha256": "same-reference",
        "roi_xywh_px": list(pilot.ROI), "normalization_bounds": list(pilot.BOUNDS),
        "parameters": pilot.PARAMETERS, "rng_seed": 0, "benchmark_script_sha256": "same-script",
        "environment": {"python": "3.11.0", "numpy": "2.0.0", "scipy": "1.11.0", "opencv": "4.8.0"},
        "core_sha256": "before", "frames": [frame],
    }
    after = copy.deepcopy(before)
    after["core_sha256"] = "after"
    after["frames"][0]["effective_search_radius_px"] = 20
    arrays = {
        "x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 4.0, 4.0]),
        "eligible": np.array([True, True, True]), "valid": np.array([True, True, False]),
        "strain_valid": np.array([True, False, False]), "u": np.array([0.0, 2.0, np.nan]),
        "v": np.array([0.0, 0.0, np.nan]), "Exx": np.array([0.1, np.nan, np.nan]),
        "Eyy": np.array([0.2, np.nan, np.nan]), "zncc": np.array([0.9, 0.8, np.nan]),
    }
    np.savez_compressed(before_dir / "frame_050.npz", **arrays)
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["u"][0] = 0.5
    changed["u"][1] = 1.5
    changed["Exx"][0] = 0.15
    np.savez_compressed(after_dir / "frame_050.npz", **changed)
    report_path = before_dir / "report.json"
    report_path.write_text(json.dumps(before), encoding="utf-8")

    result = pilot.compare(report_path, after, after_dir)
    row = result["frames"][0]
    assert row["common_valid_n"] == 2
    assert row["common_strain_valid_n"] == 1
    assert row["common_poi_differences"]["u"]["rms"] == pytest.approx(0.5)
    assert row["common_poi_differences"]["Exx"]["max_abs"] == pytest.approx(0.05)

    after["frames"][0]["sha256"] = "different-target"
    with pytest.raises(ValueError, match="Target image hash differs"):
        pilot.compare(report_path, after, after_dir)

    after["frames"][0]["sha256"] = "same-target"
    after["environment"]["numpy"] = "2.1.0"
    with pytest.raises(ValueError, match="differ in environment"):
        pilot.compare(report_path, after, after_dir)


def test_output_is_not_created_inside_image_source_or_overwritten(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    inner = images / "derived"
    monkeypatch.setattr(sys, "argv", ["real_necking_pilot", "--images", str(images), "--output", str(inner)])
    with pytest.raises(SystemExit):
        pilot.main()
    assert not inner.exists()

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("original", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["real_necking_pilot", "--images", str(images), "--output", str(existing)])
    with pytest.raises(FileExistsError):
        pilot.main()
    assert marker.read_text(encoding="utf-8") == "original"
