#!/usr/bin/env python
"""Create and optionally run a deterministic, synthetic Agent DIC campaign.

The fixture is synthetic and is not experimental evidence. It uses the existing
core's Gaussian-speckle generator and image-warp helpers, then calls the public
Agent CLI for planning, DIC, export, resume, verification, and batch checks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ezdic_core as core  # noqa: E402


HEIGHT = 160
WIDTH = 192
FRAME_INTERVAL_S = 0.5
EPSILONS = {1: 0.0, 2: 0.004, 3: 0.008, 4: 0.012}
TRANSLATIONS = {1: 0.0, 2: 0.30, 3: 0.60, 4: 0.90}
BAD_NATIVE_FRAMES = (11, 12, 14, 15)
BAD_EPSILONS = {11: 0.0, 12: 0.004, 15: 0.012}
BAD_TRANSLATIONS = {11: 0.0, 12: 0.30, 15: 0.90}
AREA_MM2 = 12.0
FORCE_AT_T0_N = 800.0
FORCE_SLOPE_N_PER_S = 200.0
CROSSHEAD_SLOPE_MM_PER_S = 0.075
STRAIN_TOLERANCE = 0.001
DISPLACEMENT_TOLERANCE_PX = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _band_limited_speckle(seed: int) -> np.ndarray:
    """Low-pass the core's deterministic Gaussian-spot speckle at 0.30 cycles/px."""
    image = core.generate_synthetic_speckle(HEIGHT, WIDTH, seed=seed, sigma=1.7)
    fy = np.fft.fftfreq(HEIGHT)[:, None]
    fx = np.fft.fftfreq(WIDTH)[None, :]
    radial_frequency = np.sqrt(fx * fx + fy * fy)
    spectrum = np.fft.fft2(image)
    spectrum[radial_frequency > 0.30] = 0.0
    filtered = np.fft.ifft2(spectrum).real
    low, high = np.percentile(filtered, (1.0, 99.0))
    normalized = np.clip((filtered - low) / max(high - low, np.finfo(float).eps), 0.0, 1.0)
    return (15.0 + 230.0 * normalized).astype(np.float32)


def _write_tiff(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.rint(np.clip(image, 0, 255)).astype(np.uint8)
    if not cv2.imwrite(str(path), encoded):
        raise RuntimeError(f"failed to write synthetic image: {path}")


def _deformed(reference: np.ndarray, epsilon_x: float, shift_x_px: float) -> np.ndarray:
    gradient = np.array([[1.0 + epsilon_x, 0.0], [0.0, 1.0]], dtype=float)
    center = ((WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0)
    image = core.warp_image_deformation_gradient(reference, gradient, center=center)
    return core.warp_image_translation(image, shift_x_px, 0.0)


def _generate_fixture(output: Path) -> dict[str, Any]:
    input_root = output / "input"
    normal_images = input_root / "sample_normal" / "images"
    gap_images = input_root / "sample_gap_bad" / "images"
    projects_root = output / "projects"
    logs_root = output / "logs"
    input_root.mkdir(parents=True, exist_ok=True)
    projects_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    calibration_source = ROOT / "examples" / "agent" / "demo_calibration.json"
    gauges_source = ROOT / "examples" / "agent" / "demo_gauges.json"
    path_source = ROOT / "examples" / "agent" / "demo_path.json"
    calibration = output / "input" / calibration_source.name
    gauges = output / "input" / gauges_source.name
    path = output / "input" / path_source.name
    for source, destination in ((calibration_source, calibration), (gauges_source, gauges), (path_source, path)):
        shutil.copyfile(source, destination)

    reference = _band_limited_speckle(seed=20261019)
    normal_paths: list[Path] = []
    for native, epsilon in EPSILONS.items():
        image = _deformed(reference, epsilon, TRANSLATIONS[native])
        path_out = normal_images / f"frame_{native:04d}.tif"
        _write_tiff(path_out, image)
        normal_paths.append(path_out)

    bad_paths: list[Path] = []
    for native in BAD_NATIVE_FRAMES:
        filename = f"camera_01_run_02_frame_{native:04d}.tif"
        path_out = gap_images / filename
        if native in BAD_EPSILONS:
            image = _deformed(reference, BAD_EPSILONS[native], BAD_TRANSLATIONS[native])
            _write_tiff(path_out, image)
        else:
            path_out.parent.mkdir(parents=True, exist_ok=True)
            path_out.write_bytes(b"synthetic damaged TIFF frame; intentionally unreadable\n")
        bad_paths.append(path_out)

    frame_map = input_root / "frame_map_sample_gap_bad.csv"
    with frame_map.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("filename", "native_frame"))
        writer.writeheader()
        for native, image_path in zip(BAD_NATIVE_FRAMES, bad_paths):
            writer.writerow({"filename": image_path.name, "native_frame": native})

    load_table = input_root / "synthetic_loads.csv"
    with load_table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("time_ms", "force_kN", "crosshead_um"))
        writer.writeheader()
        for sample_index in range(0, 8):
            aligned_s = sample_index * 0.4
            raw_ms = (aligned_s + 0.125) * 1000.0
            force_n = FORCE_AT_T0_N + FORCE_SLOPE_N_PER_S * aligned_s
            crosshead_mm = CROSSHEAD_SLOPE_MM_PER_S * aligned_s
            writer.writerow({
                "time_ms": f"{raw_ms:.6f}",
                "force_kN": f"{force_n / 1000.0:.9f}",
                "crosshead_um": f"{crosshead_mm * 1000.0:.6f}",
            })

    image_paths = normal_paths + bad_paths
    input_paths = image_paths + [frame_map, load_table, calibration, gauges, path]
    return {
        "input_root": input_root,
        "projects_root": projects_root,
        "logs_root": logs_root,
        "normal_images": normal_images,
        "gap_images": gap_images,
        "normal_project": projects_root / "sample_normal",
        "gap_project": projects_root / "sample_gap_bad",
        "frame_map": frame_map,
        "load_table": load_table,
        "calibration": calibration,
        "gauges": gauges,
        "path_json": path,
        "input_paths": input_paths,
        "input_hashes_before": {str(item): _sha256(item) for item in input_paths},
        "sample_counts": {"sample_normal": len(normal_paths), "sample_gap_bad": len(bad_paths)},
    }


def _cli_prefix(spec: str) -> tuple[list[str], Path]:
    if spec == "source":
        cli_path = ROOT / "ezdic_cli.py"
        if not cli_path.is_file():
            raise FileNotFoundError(f"source CLI not found: {cli_path}; pass --cli PATH")
        return [sys.executable, "-B", str(cli_path), "agent"], ROOT
    direct_path = Path(spec.strip('"')).expanduser()
    if direct_path.is_file():
        cli_path = direct_path.resolve()
        if cli_path.suffix.lower() == ".py":
            return [sys.executable, "-B", str(cli_path), "agent"], cli_path.parent
        return [str(cli_path), "agent"], ROOT
    parts = shlex.split(spec, posix=os.name != "nt")
    if not parts:
        raise ValueError("--cli cannot be empty")
    if len(parts) == 1:
        candidate = Path(parts[0]).expanduser()
        if candidate.suffix.lower() == ".py" and candidate.is_file():
            cli_path = candidate.resolve()
            return [sys.executable, "-B", str(cli_path), "agent"], cli_path.parent
        resolved = shutil.which(parts[0])
        if resolved:
            return [resolved, "agent"], ROOT
        if candidate.is_file():
            return [str(candidate.resolve()), "agent"], ROOT
    if len(parts) >= 3 and Path(parts[0]).name.lower().startswith("python") and parts[1:3] == ["-m", "ezdic_cli"]:
        return [*parts, "agent"], ROOT
    raise ValueError("use --cli source, a path to ezdic_cli.py, a straintrace executable, or 'python -m ezdic_cli'")


def _last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise AssertionError(f"CLI produced no JSON object; stdout tail:\n{stdout[-4000:]}")


def _run_cli(
    prefix: Sequence[str],
    cwd: Path,
    logs: Path,
    label: str,
    args: Sequence[str],
    expected_codes: Sequence[int] = (0,),
) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    command = [*prefix, *args]
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    safe_label = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)
    (logs / f"{safe_label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (logs / f"{safe_label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode not in expected_codes:
        raise AssertionError(
            f"{label} returned {result.returncode}, expected {tuple(expected_codes)}\n"
            f"command={command!r}\nstdout tail:\n{result.stdout[-4000:]}\nstderr tail:\n{result.stderr[-2000:]}"
        )
    return _last_json(result.stdout), result


def _common_init_args(fixture: dict[str, Any], project: Path, image_dir: Path) -> list[str]:
    return [
        "init", "--project", str(project), "--images", str(image_dir),
        "--frame-interval-s", str(FRAME_INTERVAL_S),
        "--loads-csv", str(fixture["load_table"]),
        "--time-column", "time_ms", "--load-time-unit", "ms", "--time-offset-s", "-0.125",
        "--clock-confirmed", "--max-gap-s", "0.6",
        "--force-column", "force_kN", "--force-unit", "kN", "--area-mm2", str(AREA_MM2),
        "--displacement-column", "crosshead_um", "--displacement-unit", "um",
        "--displacement-source", "crosshead",
        "--calibration-json", str(fixture["calibration"]),
        "--gauges-json", str(fixture["gauges"]),
        "--roi", "16,16,160,128", "--step-hint-px", "12",
    ]


def _project_run_dir(project: Path) -> Path:
    pointer = json.loads((project / "CURRENT.json").read_text(encoding="utf-8-sig"))
    relative = pointer.get("current_run")
    if not relative:
        raise AssertionError(f"project has no current run: {project}")
    return project / relative


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _assert_near(label: str, value: float | None, expected: float, tolerance: float) -> None:
    if value is None or abs(value - expected) > tolerance:
        raise AssertionError(f"{label}: observed {value!r}; expected {expected:.8g} ± {tolerance:g}")


def _checkpoint_hashes(run_dir: Path) -> dict[str, str]:
    checkpoints = run_dir / "checkpoints"
    return {
        str(path.relative_to(checkpoints)): _sha256(path)
        for path in sorted(checkpoints.glob("*"))
        if path.is_file()
    }


def _verify_fixture(prefix: Sequence[str], cwd: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    logs = fixture["logs_root"]
    normal = fixture["normal_project"]
    gap_bad = fixture["gap_project"]

    init_normal, _ = _run_cli(prefix, cwd, logs, "init_normal", _common_init_args(fixture, normal, fixture["normal_images"]))
    init_bad_args = _common_init_args(fixture, gap_bad, fixture["gap_images"])
    init_bad_args[init_bad_args.index("--frame-interval-s"):init_bad_args.index("--frame-interval-s") + 2] = [
        "--frame-map-csv", str(fixture["frame_map"]), "--frame-interval-s", str(FRAME_INTERVAL_S),
    ]
    init_bad, _ = _run_cli(prefix, cwd, logs, "init_gap_bad", init_bad_args)
    assert init_normal.get("frame_count") == 4
    assert init_bad.get("frame_count") == 4

    inspect_bad, _ = _run_cli(prefix, cwd, logs, "inspect_gap_bad", ["inspect", "--project", str(gap_bad)])
    assert inspect_bad["native_frames"] == list(BAD_NATIVE_FRAMES), inspect_bad
    assert 14 in inspect_bad["unreadable_frames"], inspect_bad
    assert inspect_bad["time_range_s"] == [0.0, 2.0], inspect_bad

    analysis_normal, _ = _run_cli(
        prefix, cwd, logs, "analyze_normal",
        ["analyze", "--project", str(normal), "--field-csv", "--progress-json"],
        expected_codes=(0, 4),
    )
    normal_result = analysis_normal.get("result", {}).get("run", {})
    if normal_result.get("status") != "complete" or normal_result.get("scientific_ok") is not True:
        raise AssertionError(f"normal synthetic sample failed the existing core QC gate: {analysis_normal}")
    normal_run = Path(normal_result["run_dir"])
    normal_summary = _rows(normal_run / "summary.csv")
    assert [int(row["native_frame"]) for row in normal_summary] == [1, 2, 3, 4]
    assert all(row["status"] == "complete" for row in normal_summary)
    assert normal_summary[0]["force_alignment_kind"] == "exact_sample"
    assert normal_summary[0]["displacement_alignment_kind"] == "exact_sample"
    assert normal_summary[0]["displacement_reference_alignment_kind"] == "exact_sample"
    interpolated_frame = normal_summary[1]
    assert interpolated_frame["force_alignment_kind"] == "bounded_linear_interpolation", interpolated_frame
    _assert_near("force left interpolation support", _float_or_none(interpolated_frame["force_bracket_left_time_s"]), 0.4, 1e-12)
    _assert_near("force right interpolation support", _float_or_none(interpolated_frame["force_bracket_right_time_s"]), 0.8, 1e-12)
    assert int(interpolated_frame["force_bracket_left_source_row"]) == 2
    assert int(interpolated_frame["force_bracket_right_source_row"]) == 3
    assert interpolated_frame["displacement_alignment_kind"] == "bounded_linear_interpolation", interpolated_frame
    _assert_near("displacement left interpolation support", _float_or_none(interpolated_frame["displacement_bracket_left_time_s"]), 0.4, 1e-12)
    _assert_near("displacement right interpolation support", _float_or_none(interpolated_frame["displacement_bracket_right_time_s"]), 0.8, 1e-12)
    assert int(interpolated_frame["displacement_bracket_left_source_row"]) == 2
    assert int(interpolated_frame["displacement_bracket_right_source_row"]) == 3
    _assert_near("frame 1 force_N", _float_or_none(normal_summary[0]["force_N"]), FORCE_AT_T0_N, 1e-6)
    _assert_near("frame 1 stress_MPa", _float_or_none(normal_summary[0]["stress_MPa"]), FORCE_AT_T0_N / AREA_MM2, 1e-6)
    expected_crosshead = CROSSHEAD_SLOPE_MM_PER_S * 1.5
    _assert_near("frame 4 independent crosshead displacement_mm",
                 _float_or_none(normal_summary[-1]["machine_displacement_mm"]), expected_crosshead, 1e-6)
    assert _float_or_none(normal_summary[-1]["machine_engineering_strain"]) is None
    _assert_near("fixed-reference virtual gauge engineering strain",
                 _float_or_none(normal_summary[-1]["dic_virtual_gauge_engineering_strain"]),
                 EPSILONS[4], STRAIN_TOLERANCE)

    plot_files = list((normal_run / "plots").glob("*.png"))
    assert plot_files, "analyze should create default sparse plots"
    assert (normal_run / "fields_csv").is_dir(), "field CSV export was requested"
    assert list((normal_run / "fields_csv").glob("*.csv"))

    roi_query = ["extract", "--project", str(normal), "--component", "Exx",
                 "--roi", "40,150,24,136", "--coordinate-system", "image"]
    roi_result, _ = _run_cli(prefix, cwd, logs, "extract_roi", roi_query)
    roi_rows = roi_result.get("rois") or []
    final_roi = next((item for item in roi_rows if int(item["native_frame"]) == 4), None)
    expected_green_lagrange = 0.5 * ((1.0 + EPSILONS[4]) ** 2 - 1.0)
    _assert_near("known affine final GL Exx ROI mean",
                 _float_or_none(final_roi.get("mean") if final_roi else None),
                 expected_green_lagrange, STRAIN_TOLERANCE)

    path_result, _ = _run_cli(
        prefix, cwd, logs, "extract_path",
        ["extract", "--project", str(normal), "--component", "u",
         "--path-json", str(fixture["path_json"]), "--max-distance", "10",
         "--coordinate-system", "image"],
    )
    path_rows = _rows(Path(path_result["paths_csv"]))
    final_center = next((row for row in path_rows
                         if int(row["native_frame"]) == 4 and int(row["path_index"]) == 1), None)
    observed_poi_x = _float_or_none(final_center.get("poi_x") if final_center else None)
    observed_u = _float_or_none(final_center.get("u") if final_center else None)
    if observed_poi_x is None:
        raise AssertionError("path query did not identify a supported original POI")
    expected_u = EPSILONS[4] * (observed_poi_x - (WIDTH - 1) / 2.0) + TRANSLATIONS[4]
    _assert_near("known affine final path u", observed_u, expected_u, DISPLACEMENT_TOLERANCE_PX)

    normal_checkpoint_hashes = _checkpoint_hashes(normal_run)
    resume_result, resume_process = _run_cli(
        prefix, cwd, logs, "resume_normal",
        ["run", "--project", str(normal), "--progress-json"],
    )
    assert resume_result.get("result", {}).get("status") == "complete", resume_result
    assert any(json.loads(line).get("event") == "frame_reused"
               for line in resume_process.stdout.splitlines()
               if line.lstrip().startswith("{")), resume_process.stdout
    assert _checkpoint_hashes(normal_run) == normal_checkpoint_hashes, "resume must preserve matching checkpoint bytes"

    analysis_bad, _ = _run_cli(
        prefix, cwd, logs, "analyze_gap_bad",
        ["analyze", "--project", str(gap_bad), "--progress-json"],
        expected_codes=(0, 4),
    )
    bad_result = analysis_bad.get("result", {}).get("run", {})
    if bad_result.get("status") != "partial" or bad_result.get("scientific_ok") is not False:
        raise AssertionError(f"damaged-frame sample was not marked partial/quality-failed: {analysis_bad}")
    bad_run = Path(bad_result["run_dir"])
    bad_summary = {int(row["native_frame"]): row for row in _rows(bad_run / "summary.csv")}
    _assert_near("native-frame timing across missing index 13",
                 _float_or_none(bad_summary[14]["time_s"]), 1.5, 1e-12)
    _assert_near("native-frame timing before missing index 13",
                 _float_or_none(bad_summary[12]["time_s"]), 0.5, 1e-12)
    assert bad_summary[14]["status"] == "failed", bad_summary[14]
    assert bad_summary[14]["failure_code"], bad_summary[14]
    assert not bad_summary[14]["checkpoint"]
    assert _float_or_none(bad_summary[14]["engineering_strain"]) is None
    assert bad_summary[14]["force_alignment_kind"] == "bounded_linear_interpolation"
    assert _float_or_none(bad_summary[14]["force_N"]) is not None
    assert _float_or_none(bad_summary[14]["machine_displacement_mm"]) is not None
    assert bad_summary[15]["status"] in {"complete", "quality_failed"}
    assert list((bad_run / "plots").glob("*.png")), "bad sample should still plot its supported frames"

    bad_checkpoint_hashes = _checkpoint_hashes(bad_run)
    resume_bad, resume_bad_process = _run_cli(
        prefix, cwd, logs, "resume_gap_bad",
        ["run", "--project", str(gap_bad), "--progress-json"],
        expected_codes=(0, 4),
    )
    assert resume_bad.get("result", {}).get("status") == "partial", resume_bad
    bad_events = [json.loads(line) for line in resume_bad_process.stdout.splitlines()
                  if line.lstrip().startswith("{")]
    assert any(event.get("event") == "frame_reused" and event.get("native_frame") == 14
               for event in bad_events), resume_bad_process.stdout
    assert _checkpoint_hashes(bad_run) == bad_checkpoint_hashes

    verify_normal, _ = _run_cli(prefix, cwd, logs, "verify_normal",
                                ["verify", "--project", str(normal)])
    verify_bad, _ = _run_cli(prefix, cwd, logs, "verify_gap_bad",
                             ["verify", "--project", str(gap_bad)])
    assert verify_normal.get("valid") is True, verify_normal
    assert verify_bad.get("valid") is True, verify_bad
    assert verify_bad.get("scientific_ok") is False, verify_bad

    batch_result, _ = _run_cli(
        prefix, cwd, logs, "batch_two_samples",
        ["batch", "--project", str(normal), "--project", str(gap_bad),
         "--max-workers", "2"],
        expected_codes=(0, 4),
    )
    assert batch_result.get("status") == "complete", batch_result
    assert batch_result.get("project_count") == 2, batch_result
    assert batch_result.get("execution_completed") == 2, batch_result
    assert batch_result.get("scientific_status") == "attention_required", batch_result

    # A dead-lock fixture is created only inside this new demo project.
    dead_pid = 2_000_000_000
    lock_path = normal / ".agent.lock"
    if lock_path.exists():
        raise AssertionError(f"unexpected existing demo lock: {lock_path}")
    lock_path.write_text(json.dumps({
        "pid": dead_pid, "host": socket.gethostname(), "process_start_token": None,
        "purpose": "synthetic_stale_owner_test", "created_unix_s": time.time(),
        "owner_token": "deterministic-demo-lock-token",
    }) + "\n", encoding="utf-8")
    recovery, _ = _run_cli(
        prefix, cwd, logs, "recover_demo_lock",
        ["recover-lock", "--project", str(normal), "--expected-pid", str(dead_pid)],
    )
    assert recovery.get("recovered") is True and not lock_path.exists(), recovery

    for source, expected in fixture["input_hashes_before"].items():
        actual = _sha256(Path(source))
        if actual != expected:
            raise AssertionError(f"input changed during run: {source}")

    return {
        "status": "e2e_passed",
        "normal_project": str(normal),
        "gap_bad_project": str(gap_bad),
        "normal_expected_GL_Exx": expected_green_lagrange,
        "normal_observed_GL_Exx": _float_or_none(final_roi["mean"]),
        "normal_exx_absolute_error": abs(float(final_roi["mean"]) - expected_green_lagrange),
        "normal_expected_path_u_px": expected_u,
        "normal_observed_path_u_px": observed_u,
        "normal_path_u_absolute_error_px": abs(float(observed_u) - expected_u),
        "interpolated_force_alignment_kind": interpolated_frame["force_alignment_kind"],
        "interpolated_force_bracket_s": [float(interpolated_frame["force_bracket_left_time_s"]), float(interpolated_frame["force_bracket_right_time_s"])],
        "interpolated_force_source_rows": [int(interpolated_frame["force_bracket_left_source_row"]), int(interpolated_frame["force_bracket_right_source_row"])],
        "interpolated_displacement_alignment_kind": interpolated_frame["displacement_alignment_kind"],
        "interpolated_displacement_bracket_s": [float(interpolated_frame["displacement_bracket_left_time_s"]), float(interpolated_frame["displacement_bracket_right_time_s"])],
        "interpolated_displacement_source_rows": [int(interpolated_frame["displacement_bracket_left_source_row"]), int(interpolated_frame["displacement_bracket_right_source_row"])],
        "damaged_native_frame": 14,
        "damaged_frame_status": bad_summary[14]["status"],
        "missing_native_frame": 13,
        "resume_reused_normal_checkpoints": True,
        "resume_reused_failed_frame_record": True,
        "verify_integrity_normal": verify_normal["valid"],
        "verify_integrity_gap_bad": verify_bad["valid"],
        "batch_execution_completed": batch_result["execution_completed"],
        "batch_scientific_status": batch_result["scientific_status"],
        "dead_demo_lock_recovered": recovery["recovered"],
        "default_plot_count_before_batch": len(plot_files),
        "source_inputs_unchanged": True,
        "input_sha256": fixture["input_hashes_before"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new or empty output directory for synthetic input and results")
    parser.add_argument("--run", action="store_true", help="execute the real CLI and assert end-to-end results")
    parser.add_argument("--cli", default="source", help="source | path to ezdic_cli.py | installed straintrace | 'python -m ezdic_cli'")
    args = parser.parse_args(argv)

    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"--output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    fixture = _generate_fixture(output)
    receipt: dict[str, Any] = {
        "schema": "straintrace.agent-demo-receipt/v1",
        "status": "prepared",
        "synthetic_only": True,
        "image_shape_yx": [HEIGHT, WIDTH],
        "frame_interval_s": FRAME_INTERVAL_S,
        "native_frames": {"sample_normal": [1, 2, 3, 4], "sample_gap_bad": list(BAD_NATIVE_FRAMES)},
        "missing_native_frame": 13,
        "damaged_native_frame": 14,
        "output": str(output),
        "projects": {"sample_normal": str(fixture["normal_project"]), "sample_gap_bad": str(fixture["gap_project"])},
    }
    if args.run:
        prefix, cli_cwd = _cli_prefix(args.cli)
        receipt.update(_verify_fixture(prefix, cli_cwd, fixture))
    else:
        receipt["next_command"] = "to run an end-to-end validation, use --run on the first invocation with a new empty --output directory"
    receipt_path = output / "DEMO_VALIDATION.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
