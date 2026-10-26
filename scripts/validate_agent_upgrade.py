"""Independent synthetic integration checks; all generated evidence stays outside the repo.

The image generator samples a continuous random Fourier texture at inverse affine
coordinates. It deliberately does not call StrainTrace's synthetic-image or warp
helpers. Expected displacement and strain follow directly from the imposed map.
These checks are software regression checks, not experimental uncertainty estimates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def analytic_image(F=None, translation=(0.0, 0.0), *, size=144, seed=8127):
    """Sample the same continuous texture before/after x'=c+F(x-c)+t."""
    F = np.eye(2) if F is None else np.asarray(F, dtype=float)
    yy, xx = np.mgrid[:size, :size]
    center = np.array([(size - 1) / 2.0] * 2)
    coordinates = np.stack((xx, yy), axis=-1)
    reference_xy = (coordinates - center - translation) @ np.linalg.inv(F).T + center
    return texture_at(reference_xy, seed=seed)


def texture_at(coordinates, *, seed=8127):
    """Evaluate a deterministic continuous two-dimensional speckle texture."""
    rng = np.random.default_rng(seed)
    frequencies = rng.uniform(-0.9, 0.9, size=(96, 2))
    phases = rng.uniform(-math.pi, math.pi, size=96)
    texture = np.zeros(coordinates.shape[:-1], dtype=float)
    for frequency, phase in zip(frequencies, phases):
        texture += np.cos(coordinates @ frequency + phase)
    return 128.0 + 32.0 * texture / math.sqrt(len(phases) / 2.0)


def deformation_cases():
    theta = 0.09
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    return {
        "translation_multiscale": (np.eye(2), np.array([1.35, -0.72]),
                                   {"pyramid_levels": 2, "pyramid_scale": 0.63}),
        "rotation_smoothed": (rotation, np.array([0.23, -0.19]), {"smooth_sigma": 1.0}),
        "affine_extension": (np.array([[1.018, 0.004], [-0.002, 0.994]]),
                             np.array([0.31, 0.24]), {}),
    }


def field_errors(field, F, translation, *, size=144):
    """Compare at original POIs against continuum kinematics, not fitted values."""
    x = np.asarray(field["x"]).ravel()
    y = np.asarray(field["y"]).ravel()
    position = np.column_stack((x, y))
    center = np.array([(size - 1) / 2.0] * 2)
    expected_u = (position - center) @ (np.asarray(F) - np.eye(2)).T + translation
    measured_u = np.column_stack((np.asarray(field["u"]).ravel(), np.asarray(field["v"]).ravel()))
    valid = np.asarray(field["valid"], dtype=bool).ravel()
    strain_valid = valid & np.asarray(field["strain_valid"], dtype=bool).ravel()
    assert valid.any() and strain_valid.any(), "No numerical support for the imposed transform"
    expected_E = 0.5 * (np.asarray(F).T @ F - np.eye(2))
    expected_e = 0.5 * (np.asarray(F) + np.asarray(F).T) - np.eye(2)
    result = {
        "displacement_rmse_px": float(np.sqrt(np.mean((measured_u[valid] - expected_u[valid]) ** 2))),
        "displacement_max_error_px": float(np.max(np.abs(measured_u[valid] - expected_u[valid]))),
        "valid_fraction": float(valid.mean()), "strain_valid_fraction": float(strain_valid.mean()),
        "expected_green_lagrange": expected_E.tolist(), "expected_infinitesimal": expected_e.tolist(),
    }
    for component, expected in (("Exx", expected_E[0, 0]), ("Eyy", expected_E[1, 1]),
                                ("Exy", expected_E[0, 1]), ("exx", expected_e[0, 0]),
                                ("eyy", expected_e[1, 1]), ("exy", expected_e[0, 1])):
        values = np.asarray(field[component]).ravel()
        result[f"{component}_rmse"] = float(np.sqrt(np.mean((values[strain_valid] - expected) ** 2)))
        result[f"{component}_max_error"] = float(np.max(np.abs(values[strain_valid] - expected)))
    return result


def validate_core_case(name, *, destination=None):
    import ezdic_core as core
    F, translation, extra = deformation_cases()[name]
    reference = analytic_image()
    deformed = analytic_image(F, translation)
    started = time.perf_counter()
    field = core.run_2d_dic(reference, deformed, [24, 24, 96, 96], subset_size=25,
                           step=10, search_radius=10, max_iter=45, conv_tol=1e-4,
                           strain_window=5, **extra)
    result = {"case": name, "seconds": time.perf_counter() - started,
              "F": F.tolist(), "translation_px": translation.tolist(), **field_errors(field, F, translation)}
    assert result["valid_fraction"] >= 0.95, result
    assert result["strain_valid_fraction"] >= 0.9, result
    assert result["displacement_rmse_px"] < 0.01, result
    for component in ("Exx", "Eyy", "Exy", "exx", "eyy", "exy"):
        assert result[f"{component}_rmse"] < 0.0005, result
        assert result[f"{component}_max_error"] < 0.0015, result
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / f"{name}.npz", **{key: value for key, value in field.items() if isinstance(value, np.ndarray)})
        write_json(destination / f"{name}.json", result)
    return result


def validate_masked_support(destination):
    """A real masked DIC hole remains unsupported through path and CSV export."""
    import ezdic_core as core
    from straintrace_agent.fields import sample_path, write_rows_csv
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    mask = np.ones((144, 144), dtype=bool)
    mask[66:79, 66:79] = False
    field = core.run_2d_dic(analytic_image(), analytic_image(np.eye(2), [0.61, -0.43]),
                           [24, 24, 96, 96], subset_size=25, step=10, search_radius=10,
                           specimen_mask=mask, smooth_sigma=1.0)
    invalid = ~np.asarray(field["valid"], dtype=bool).ravel()
    assert invalid.any() and (~invalid).any()
    for name in ("u", "v", "Exx", "Eyy", "Exy", "exx", "eyy", "exy"):
        assert np.isnan(np.asarray(field[name]).ravel()[invalid]).all(), name
    index = int(np.flatnonzero(invalid)[0])
    point = [float(np.asarray(field["x"]).ravel()[index]), float(np.asarray(field["y"]).ravel()[index])]
    rows = sample_path(field, [point], ["u", "Exx"], max_distance=0.01)
    assert rows[0]["poi_index"] == index and rows[0]["status"] == "invalid_poi", rows
    assert rows[0]["interpolation"] == "none"
    path = destination / "invalid_path.csv"
    write_rows_csv(rows, path)
    persisted = read_csv(path)[0]
    assert persisted["u"] == persisted["Exx"] == "", persisted
    return {"invalid_poi_count": int(invalid.sum()), "path_poi_index": index,
            "path_status": rows[0]["status"], "csv_missing_cells_are_blank": True}


def validate_spatial_contrast(destination):
    """Two spatial regions recover distinct known strains from a nonlinear map."""
    import ezdic_core as core
    from straintrace_agent.regions import summarize_regions
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    beta, center = 0.00045, 71.5
    yy, xx = np.mgrid[:144, :144]
    shifted = xx - center
    # Stable inverse of x' - c = q + beta*q*q, with q = x - c.
    q = 2 * shifted / (1 + np.sqrt(1 + 4 * beta * shifted))
    deformed = texture_at(np.stack((center + q, yy), axis=-1))
    field = core.run_2d_dic(analytic_image(), deformed, [24, 24, 96, 96],
                           subset_size=25, step=10, search_radius=10, max_iter=45,
                           conv_tol=1e-4, strain_window=5)
    regions = [{"name": name, "bounds": bounds, "min_valid_fraction": 0.9, "min_valid_pois": 8}
               for name, bounds in (("left", [34, 64, 34, 114]), ("right", [84, 114, 34, 114]))]
    rows = summarize_regions(field, regions, components=("Exx",), quality_ok=True)
    x, y = np.asarray(field["x"]).ravel(), np.asarray(field["y"]).ravel()
    analytic_E = 0.5 * ((1 + 2 * beta * (x - center)) ** 2 - 1)
    result = []
    for region, row in zip(regions, rows):
        xmin, xmax, ymin, ymax = region["bounds"]
        selected = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        selected &= np.asarray(field["valid"]).ravel() & np.asarray(field["strain_valid"]).ravel()
        expected = float(analytic_E[selected].mean())
        assert abs(row["mean"] - expected) < 0.0015, (row, expected)
        result.append({"name": region["name"], "expected_Exx_mean": expected, "measured_Exx_mean": row["mean"],
                       "absolute_error": abs(row["mean"] - expected), "n_poi": int(selected.sum())})
    assert result[0]["measured_Exx_mean"] < -0.01 and result[1]["measured_Exx_mean"] > 0.01
    write_json(destination / "spatial_contrast.json", result)
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build_experiment(destination, *, failed_frame=True):
    """Create only owned synthetic images and explicit, checkable acquisition metadata."""
    import cv2
    destination = Path(destination)
    images = destination / "images"
    images.mkdir(parents=True)
    cases = deformation_cases()
    frames = {10: (np.eye(2), np.zeros(2)),
              11: cases["translation_multiscale"][:2],
              13: cases["rotation_smoothed"][:2],
              14: cases["affine_extension"][:2],
              17: (np.diag([1.025, 0.992]), np.array([0.3, -0.1]))}
    truth = {}
    for native, (F, translation) in frames.items():
        path = images / f"image_{native:04d}.png"
        image = np.rint(np.clip(analytic_image(F, translation), 0, 255)).astype(np.uint8)
        assert cv2.imwrite(str(path), image)
        truth[native] = {"F": F, "translation": translation}
    if failed_frame:
        (images / "image_0015.png").write_bytes(b"intentionally corrupt synthetic image\n")
    loads = destination / "loads.csv"
    with loads.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Time_ms", "Force_kN", "Crosshead_um"])
        for index in range(8):
            t = 0.2 * index
            writer.writerow([125 + 1000 * t, 0.1 + 0.01 * t, 50 + 100 * t])
    gauges = [
        {"name": "upper_chord", "axis": "distance", "a_roi_px": [34, 34, 20, 20],
         "b_roi_px": [94, 34, 20, 20]},
        {"name": "lower_chord", "axis": "distance", "a_roi_px": [34, 84, 20, 20],
         "b_roi_px": [94, 84, 20, 20]},
    ]
    regions = [
        {"name": "left", "bounds": [34, 74, 34, 114], "coordinate_system": "image",
         "min_valid_fraction": 0.9, "min_valid_pois": 8},
        {"name": "right", "bounds": [84, 114, 34, 114], "coordinate_system": "image",
         "min_valid_fraction": 0.9, "min_valid_pois": 8},
    ]
    options = {
        "image_dir": str(images), "frame_interval_s": 0.2, "reference_native_frame": 10,
        "roi_override_px": [24, 24, 96, 96], "step_hint_px": 10,
        "loads_csv": str(loads), "load_time_column": "Time_ms", "load_time_unit": "ms",
        "load_time_offset_s": -0.125, "clock_confirmed": True, "max_gap_s": 0.21,
        "load_force_column": "Force_kN", "force_unit": "kN", "area_mm2": 2.0,
        "displacement_column": "Crosshead_um", "displacement_unit": "um",
        "displacement_source": "crosshead", "machine_gauge_length_mm": 10.0,
        "calibration": {"mm_per_px": 0.02, "origin_px": [0, 0], "origin_mm": [1, 2], "axis_sign": [1, -1]},
        "gauges": gauges, "regions": regions,
    }
    write_json(destination / "experiment.json", {"schema": "straintrace.experiment/v1", "options": options})
    write_json(destination / "options.json", options)
    write_json(destination / "regions.json", regions)
    write_json(destination / "gauges.json", gauges)
    write_json(destination / "analytic_truth.json", {
        str(native): {"F": item["F"].tolist(), "translation": item["translation"].tolist()}
        for native, item in truth.items()
    })
    return options, truth


def validate_run_tables(run_root, plan, truth, *, failed_frame=True):
    """Reconstruct geometry and all derived numbers independently from saved POIs."""
    run_root = Path(run_root)
    summary = read_csv(run_root / "summary.csv")
    expected_frames = [10, 11, 13, 14, 15, 17] if failed_frame else [10, 11, 13, 14, 17]
    assert [int(row["native_frame"]) for row in summary] == expected_frames
    np.testing.assert_allclose([float(row["time_s"]) for row in summary], [(native - 10) * 0.2 for native in expected_frames], atol=1e-12)
    arrays_by_frame = {}
    errors = []
    for row in summary:
        native = int(row["native_frame"])
        t = (native - 10) * 0.2
        np.testing.assert_allclose(float(row["force_N"]), 100 + 10 * t, atol=1e-10)
        np.testing.assert_allclose(float(row["stress_MPa"]), 50 + 5 * t, atol=1e-10)
        np.testing.assert_allclose(float(row["machine_displacement_mm"]), 0.1 * t, atol=1e-12)
        np.testing.assert_allclose(float(row["machine_engineering_strain"]), 0.01 * t, atol=1e-12)
        assert int(row["reference_native_frame"]) == 10
        if native == 15:
            assert row["status"] == "failed"
            assert row["engineering_strain"] == row["max_abs_local_exx"] == ""
            assert not (run_root / "checkpoints" / f"frame_native_{native:08d}.npz").exists()
            continue
        assert row["status"] == "complete", row
        with np.load(run_root / "checkpoints" / f"frame_native_{native:08d}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays_by_frame[native] = arrays
        item = truth[native]
        error = field_errors(arrays, item["F"], item["translation"])
        assert error["displacement_rmse_px"] < 0.02, (native, error)
        assert max(error[f"{component}_rmse"] for component in ("Exx", "Eyy", "Exy")) < 0.001, (native, error)
        errors.append({"native_frame": native, **error})
    geometry = {item["name"]: item for item in plan["gauge_reference_geometry"]}
    gauge_count = 0
    for row in read_csv(run_root / "gauges.csv"):
        native = int(row["native_frame"])
        if native == 15:
            assert row["engineering_strain"] == ""
            continue
        arrays = arrays_by_frame[native]
        gauge = geometry[row["gauge_name"]]
        a = np.asarray(gauge["a"]["poi_indices"], dtype=int)
        b = np.asarray(gauge["b"]["poi_indices"], dtype=int)
        points = np.column_stack((arrays["x"].ravel(), arrays["y"].ravel()))
        raw_displacement = np.column_stack((arrays.get("u_raw", arrays["u"]).ravel(), arrays.get("v_raw", arrays["v"]).ravel()))
        a_xy, b_xy = points[a].mean(axis=0), points[b].mean(axis=0)
        np.testing.assert_allclose(a_xy, gauge["a"]["reference_xy"], atol=1e-12)
        np.testing.assert_allclose(b_xy, gauge["b"]["reference_xy"], atol=1e-12)
        initial = np.linalg.norm(b_xy - a_xy)
        current = np.linalg.norm(b_xy - a_xy + raw_displacement[b].mean(axis=0) - raw_displacement[a].mean(axis=0))
        measured = current / initial - 1
        analytic = np.linalg.norm(truth[native]["F"] @ (b_xy - a_xy)) / initial - 1
        np.testing.assert_allclose(float(row["engineering_strain"]), measured, atol=1e-12)
        assert abs(measured - analytic) < 0.001, (native, row["gauge_name"], measured, analytic)
        if row.get("current_length_mm"):
            np.testing.assert_allclose(float(row["current_length_mm"]), current * 0.02, atol=1e-12)
        gauge_count += 1
    region_count = 0
    for row in read_csv(run_root / "regions.csv"):
        native = int(row["native_frame"])
        if native == 15:
            assert row["mean"] == ""
            continue
        arrays = arrays_by_frame[native]
        bounds = [34, 74, 34, 114] if row["region_name"] == "left" else [84, 114, 34, 114]
        x, y = arrays["x"].ravel(), arrays["y"].ravel()
        selected = (x >= bounds[0]) & (x <= bounds[1]) & (y >= bounds[2]) & (y <= bounds[3])
        good = selected & arrays["valid"].ravel().astype(bool) & arrays["strain_valid"].ravel().astype(bool)
        values = arrays[row["component"]].ravel()[good]
        assert int(row["n_poi_total"]) == int(selected.sum())
        assert int(row["n_poi_valid"]) == len(values)
        np.testing.assert_allclose(float(row["mean"]), values.mean(), atol=1e-12)
        np.testing.assert_allclose(float(row["sd_sample"]), values.std(ddof=1), atol=1e-12)
        region_count += 1
    assert gauge_count == 2 * len(truth), gauge_count
    assert region_count >= 6 * len(truth), region_count
    return {"field_errors": errors, "gauge_rows_checked": gauge_count, "region_rows_checked": region_count}


class CliRecorder:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.calls = []

    def run(self, label, *args, allowed=(0,)):
        command = [sys.executable, "-m", "ezdic_cli", "agent", *map(str, args)]
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=REPOSITORY, encoding="utf-8", errors="strict", capture_output=True,
                                   env={**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "Agg"})
        elapsed = time.perf_counter() - started
        (self.output / f"{label}.stdout.jsonl").write_text(completed.stdout, encoding="utf-8")
        (self.output / f"{label}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        events = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        self.calls.append({"label": label, "command": command, "exit_code": completed.returncode,
                           "seconds": elapsed, "frame_complete_events": sum(e.get("event") == "frame_complete" for e in events),
                           "frame_reused_events": sum(e.get("event") == "frame_reused" for e in events)})
        assert completed.returncode in allowed, (label, completed.returncode, completed.stdout, completed.stderr)
        assert events, (label, "CLI produced no JSON result")
        return events[-1], events


def validate_report_sources(report_root, run_root):
    """Check exported rows and every plotted point without relying on report code."""
    report_root, run_root = Path(report_root), Path(run_root)
    data = json.loads((report_root / "data" / "report_data.json").read_text(encoding="utf-8"))
    assert data["schema"] == "straintrace.agent-report/v1"
    plan = json.loads((report_root / "provenance" / "plan.json").read_text(encoding="utf-8"))
    config = json.loads((report_root / "provenance" / "project.json").read_text(encoding="utf-8"))
    assert data["geometry"]["reference_native_frame"] == plan["reference_native_frame"]
    drawn_regions = {item["name"]: item for item in data["geometry"]["rectangles"] if item["kind"] == "region"}
    for region in config["regions"]:
        xmin, xmax, ymin, ymax = region["bounds"]
        np.testing.assert_allclose(drawn_regions[region["name"]]["xywh"], [xmin, ymin, xmax - xmin, ymax - ymin], atol=0)
    drawn_gauges = {item["name"]: item for item in data["geometry"]["gauges"]}
    for gauge in plan["gauge_reference_geometry"]:
        for endpoint in ("a", "b"):
            np.testing.assert_allclose(drawn_gauges[gauge["name"]][endpoint], gauge[endpoint]["reference_xy"], atol=0)
    for name in ("summary", "gauges", "regions"):
        table = data["tables"][name]
        copied = report_root / table["path"]
        assert file_sha(copied) == file_sha(run_root / f"{name}.csv"), name
        assert table["rows"] == read_csv(copied), name
    compared = 0
    gaps = 0
    for curve in data["curves"]:
        source = data["tables"][curve["table"]]["rows"]
        for series in curve["series"]:
            for point in series["points"]:
                row_index = point.get("source_row_index")
                value = point["value"]
                if row_index is None:
                    assert value is None, point
                    gaps += 1
                    continue
                row = source[row_index]
                assert int(row["native_frame"]) == int(point["native_frame"]), point
                if value is not None:
                    np.testing.assert_allclose(value, float(row[curve["y_column"]]), atol=0, rtol=0)
                    np.testing.assert_allclose(point["time_s"], float(row["time_s"]), atol=0, rtol=0)
                    compared += 1
                else:
                    gaps += 1
                if curve.get("x_table"):
                    x_index = point.get("x_source_row_index")
                    if x_index is None:
                        assert point["x_value"] is None
                    elif point["x_value"] is not None:
                        x_row = data["tables"][curve["x_table"]]["rows"][x_index]
                        assert int(x_row["native_frame"]) == int(point["native_frame"])
                        np.testing.assert_allclose(point["x_value"], float(x_row[curve["x_column"]]), atol=0, rtol=0)
    assert compared > 0
    return {"curve_values_checked": compared, "unavailable_curve_points": gaps}


def validate_cli_workflow(output):
    """Exercise real command parsing, recovery and portable exports on owned data."""
    output = Path(output)
    options, truth = build_experiment(output / "experiment")
    initial_input_hashes = {str(path.relative_to(output / "experiment")): file_sha(path)
                            for path in (output / "experiment").rglob("*") if path.is_file()}
    project = output / "project"
    cli = CliRecorder(output / "logs")
    discovery, _ = cli.run("discover", "discover", "--folder", output / "experiment", "--objective", "stress_strain")
    first, _ = cli.run("auto_initial", "auto", "--folder", output / "experiment", "--project", project,
                       "--objective", "stress_strain", "--no-plots", "--field-csv", "--progress-json", allowed=(0, 4))
    assert "run" in first and first["run"]["status"] == "partial", first
    run_root = Path(first["run"]["run_dir"])
    plan = first["plan"]
    numerical = validate_run_tables(run_root, plan, truth)
    verified, _ = cli.run("verify_initial", "verify", "--project", project, allowed=(0, 4))
    assert verified["valid"] is True, verified
    assert verified["numerical_quality_passed"] is False
    assert verified["scientific_acceptance"] == "not_assessed"
    hashes_before = {path.name: file_sha(path) for path in (run_root / "checkpoints").glob("*.npz")}
    reused, events = cli.run("auto_reused", "auto", "--folder", output / "experiment", "--project", project,
                            "--no-plots", "--field-csv", "--progress-json", allowed=(0, 4))
    assert reused["run"]["run_id"] == first["run"]["run_id"]
    assert not any(event.get("event") == "frame_complete" for event in events), events
    assert {event["native_frame"] for event in events if event.get("event") == "frame_reused"} >= set(truth)
    assert hashes_before == {path.name: file_sha(path) for path in (run_root / "checkpoints").glob("*.npz")}
    reprocessed, _ = cli.run("reprocess", "reprocess", "--project", project,
                            "--regions-json", output / "experiment" / "regions.json",
                            "--gauges-json", output / "experiment" / "gauges.json", allowed=(4,))
    assert reprocessed["status"] == "attention_required" and reprocessed["solver_recomputed"] is False
    assert hashes_before == {path.name: file_sha(path) for path in (run_root / "checkpoints").glob("*.npz")}
    report_result, _ = cli.run("report", "report", "--project", project, "--include-checkpoints")
    report_root = run_root / "report"
    report_numbers = validate_report_sources(report_root, run_root)
    standalone = output / "relocated-report"
    shutil.copytree(report_root, standalone)
    checked, _ = cli.run("verify_relocated", "verify-report", "--report", standalone)
    assert checked["valid"] is True, checked
    # The copied delivery must remain verifiable with only the standard library,
    # from its new directory and without importing the installed project.
    checked_process = subprocess.run([sys.executable, "-I", str(standalone / "verify_report.py")],
                                     cwd=standalone, encoding="utf-8", capture_output=True)
    (output / "logs" / "standalone_verify.stdout.txt").write_text(checked_process.stdout, encoding="utf-8")
    (output / "logs" / "standalone_verify.stderr.txt").write_text(checked_process.stderr, encoding="utf-8")
    assert checked_process.returncode == 0, (checked_process.stdout, checked_process.stderr)
    copied_table = standalone / "data" / "summary.csv"
    copied_table.write_bytes(copied_table.read_bytes() + b"\nTAMPERED\n")
    tampered_report, _ = cli.run("verify_report_tamper", "verify-report", "--report", standalone, allowed=(4,))
    assert tampered_report["valid"] is False
    target = run_root / "checkpoints" / "frame_native_00000014.npz"
    target.write_bytes(target.read_bytes() + b"owned validation mutation")
    tampered_checkpoint, _ = cli.run("verify_checkpoint_tamper", "verify", "--project", project, allowed=(4,))
    assert tampered_checkpoint["valid"] is False
    repaired, events = cli.run("auto_repair", "auto", "--folder", output / "experiment", "--project", project,
                              "--no-plots", "--field-csv", "--progress-json", allowed=(0, 4))
    assert repaired["run"]["run_id"] == first["run"]["run_id"]
    assert {event["native_frame"] for event in events if event.get("event") == "frame_complete"} == {14}
    assert hashes_before == {path.name: file_sha(path) for path in (run_root / "checkpoints").glob("*.npz")}
    # Removing calibration and clock confirmation must create a new binding and
    # leave the old run intact, without upgrading the old experiment metadata.
    config = json.loads((project / "project.json").read_text(encoding="utf-8"))
    changes = {"calibration": None, "loads": {**config["loads"], "clock_confirmed": False, "area_mm2": None}}
    write_json(output / "changes.json", changes)
    updated, _ = cli.run("update", "update", "--project", project, "--changes-json", output / "changes.json")
    unavailable, _ = cli.run("auto_unknown_calibration", "auto", "--folder", output / "experiment", "--project", project,
                              "--no-plots", "--progress-json", allowed=(0, 4))
    assert unavailable["run"]["run_id"] != first["run"]["run_id"]
    assert run_root.is_dir() and hashes_before == {path.name: file_sha(path) for path in (run_root / "checkpoints").glob("*.npz")}
    unknown_root = Path(unavailable["run"]["run_dir"])
    for row in read_csv(unknown_root / "summary.csv"):
        assert row["force_N"] == row["stress_MPa"] == row["machine_displacement_mm"] == "", row
    for row in read_csv(unknown_root / "gauges.csv"):
        assert row.get("initial_length_mm", "") == row.get("current_length_mm", "") == "", row
    missing_project = output / "missing-stress-metadata"
    missing, _ = cli.run("auto_missing_stress_metadata", "auto", "--folder", options["image_dir"],
                         "--project", missing_project, "--objective", "stress_strain", "--no-plots", allowed=(4,))
    assert missing["status"] == "needs_input", missing
    assert not missing_project.exists(), "Incomplete mechanics metadata must not create a misleading analyzed project"
    index_project = output / "frame-index-project"
    index_result, _ = cli.run("auto_frame_index", "auto", "--folder", options["image_dir"],
                              "--project", index_project, "--objective", "full_field", "--no-plots", allowed=(0, 4))
    assert index_result.get("run", {}).get("status") in {"complete", "partial"}, index_result
    index_run = Path(index_result["run"]["run_dir"])
    index_config = json.loads((index_project / "project.json").read_text(encoding="utf-8"))
    assert index_config["time_basis"] == "frame_index"
    assert all(row["time_s"] == "" for row in read_csv(index_run / "summary.csv"))
    index_report = json.loads((index_run / "report" / "data" / "report_data.json").read_text(encoding="utf-8"))
    assert all(frame["time_s"] is None for frame in index_report["frames"])
    for curve in index_report["curves"]:
        if curve.get("x_table"):
            continue
        assert curve["x_key"] == "x_value" and "not time" in curve["x_label"], curve
        for series in curve["series"]:
            for point in series["points"]:
                assert point["time_s"] is None
                assert float(point["x_value"]) == int(point["native_frame"])
    assert initial_input_hashes == {str(path.relative_to(output / "experiment")): file_sha(path)
                                    for path in (output / "experiment").rglob("*") if path.is_file()}
    result = {"discovery": discovery, "initial_run_id": first["run"]["run_id"],
              "updated_run_id": unavailable["run"]["run_id"], "numerical": numerical,
              "report": report_numbers, "report_result": report_result, "reprocess": reprocessed,
              "update": updated, "frame_index_run_id": index_result["run"]["run_id"],
              "missing_stress_metadata_status": missing["status"], "calls": cli.calls, "input_files_unchanged": True,
              "performance_scope": "one cold run and one identical resume; timings are observations, not a benchmark speedup"}
    write_json(output / "CLI_VALIDATION.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory outside the repository")
    parser.add_argument("--core-only", action="store_true", help="run only independent numerical checks")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == REPOSITORY or REPOSITORY in output.parents:
        parser.error("validation outputs must be outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    source_files = [REPOSITORY / "ezdic_core.py", REPOSITORY / "ezdic_cli.py", Path(__file__),
                    *sorted((REPOSITORY / "straintrace_agent").glob("*.py"))]
    source_sha = {str(path.relative_to(REPOSITORY)): file_sha(path) for path in source_files}
    write_json(output / "SOURCE_SNAPSHOT.json", source_sha)
    results = [validate_core_case(name, destination=output / "core") for name in deformation_cases()]
    support_result = validate_masked_support(output / "masked-support")
    contrast_result = validate_spatial_contrast(output / "spatial-contrast")
    cli_result = None if args.core_only else validate_cli_workflow(output / "cli")
    assert source_sha == {str(path.relative_to(REPOSITORY)): file_sha(path) for path in source_files}, "Source files changed during validation"
    receipt = {"schema": "straintrace.upgrade-validation/v1", "passed": True,
               "python": sys.executable, "core_cases": results, "masked_support": support_result,
               "spatial_contrast": contrast_result, "cli": cli_result, "source_sha256": source_sha,
               "scope": "synthetic software regression; experimental accuracy and uncertainty not assessed"}
    write_json(output / "VALIDATION.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
