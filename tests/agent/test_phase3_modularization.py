"""Phase 3 modularization, performance, entrypoint, and decoupling regressions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import cv2
import numpy as np

import ezdic_core as core
from straintrace_agent import api, reporting


REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_sequence(folder: Path) -> Path:
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)
    reference = core.generate_synthetic_speckle(112, 112, seed=23)
    for native, stretch in ((1, 1.0), (2, 1.005), (4, 1.012)):
        image = reference if native == 1 else core.warp_image_deformation_gradient(
            reference, [[stretch, 0.0], [0.0, 1.0]],
        )
        assert cv2.imwrite(str(images / f"frame_{native:04d}.png"), np.clip(image, 0, 255).astype(np.uint8))
    return images


def test_plot_uses_single_pass_npz_loading_across_all_components(tmp_path, monkeypatch):
    folder = tmp_path / "exp"
    images = _make_sequence(folder)
    project = tmp_path / "project"
    api.init_project(project, images, frame_interval_s=0.5, step_hint_px=8)
    plan = api.plan_project(project)
    assert plan["status"] == "ready"
    run_res = api.run(project, make_plots=False)
    assert run_res["status"] == "complete"

    load_calls: list[str] = []
    real_load_field = api._load_field

    def counting_load(checkpoint, metadata_path=None):
        load_calls.append(Path(checkpoint).name)
        return real_load_field(checkpoint, metadata_path)

    monkeypatch.setattr(api, "_load_field", counting_load)
    plot_res = api.plot(project)

    assert plot_res["plot_count"] == 27  # 3 frames x 9 components
    assert len(plot_res["selected_native_frames"]) == 3
    # Pass 1 loads each of the 3 frames once for all 9 component limits;
    # Pass 2 loads each of the 3 selected frames once to render all 9 PNGs.
    # Total = 6 loads instead of 9 * 3 + 9 * 3 = 54 loads.
    assert len(load_calls) == 6
    assert api.verify(project)["valid"] is True


def test_auto_analyze_and_export_build_portable_report_only_once(tmp_path, monkeypatch):
    folder = tmp_path / "exp"
    _make_sequence(folder)
    project = tmp_path / "project"

    calls: list[dict[str, object]] = []
    real_build_report = reporting.build_report

    def spy_build_report(run_root, **kwargs):
        calls.append({"run_root": str(run_root), "include_checkpoints": kwargs.get("include_checkpoints")})
        return real_build_report(run_root, **kwargs)

    monkeypatch.setattr(reporting, "build_report", spy_build_report)
    auto_res = api.auto_analyze(folder, project, make_plots=False)
    assert auto_res["status"] == "complete", auto_res
    assert len(calls) == 1
    assert calls[0]["include_checkpoints"] is True
    assert Path(auto_res["report"]["report_html"]).is_file()
    assert auto_res["run"]["artifacts"]["report_html"] == auto_res["report"]["report_html"]
    assert auto_res["verification"]["valid"] is True

    calls.clear()
    export_res = api.export(project, field_csv=False)
    assert export_res["status"] == "complete"
    assert len(calls) == 1
    assert calls[0]["include_checkpoints"] is True
    assert api.verify(project)["valid"] is True
    report_dir = Path(export_res["artifacts"]["report_html"]).parent
    verified = reporting.verify_report(report_dir)
    assert verified["valid"] is True
    assert verified["include_checkpoints"] is True


def test_standalone_verify_report_runs_isolated_without_straintrace(tmp_path):
    folder = tmp_path / "exp"
    _make_sequence(folder)
    project = tmp_path / "project"
    res = api.auto_analyze(folder, project, make_plots=False)
    assert res["status"] == "complete"
    report_dir = Path(res["report"]["report_html"]).parent
    verifier = report_dir / "verify_report.py"
    assert verifier.is_file()

    proc = subprocess.run(
        [sys.executable, "-I", str(verifier), str(report_dir)],
        cwd=str(report_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["valid"] is True
    assert payload["include_checkpoints"] is True


def test_python_module_entrypoint_help():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "-m", "straintrace_agent", "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "straintrace agent" in proc.stdout


def test_headless_package_import_stays_lazy():
    code = (
        "import sys; import straintrace_agent; "
        "forbidden = ('tkinter', 'PIL', 'cv2', 'matplotlib', 'ezdic_core'); "
        "loaded = [name for name in forbidden if name in sys.modules]; "
        "assert not loaded, f'eagerly imported: {loaded}'"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_public_helpers_and_internal_decoupling(tmp_path):
    folder = tmp_path / "exp"
    images = _make_sequence(folder)
    assert api.parse_native_frame(images / "frame_0004.png") == 4

    mapping_csv = tmp_path / "map.csv"
    mapping_csv.write_text("filename,native_frame,time_s\nframe_0001.png,1,0.0\nframe_0002.png,2,0.5\nframe_0004.png,4,1.5\n", encoding="utf-8")
    parsed = api.parse_frame_map(mapping_csv, images, sorted(images.glob("*.png")))
    assert len(parsed) == 3
    assert api.resolve_time_basis({"image_dir": str(images), "frame_map_csv": str(mapping_csv)}) == "physical_time"
    assert api.resolve_time_basis({"image_dir": str(images)}) == "frame_index"

    project = tmp_path / "project"
    api.init_project(project, images, time_basis="frame_index")
    config = api.read_project_config(project)
    inventory = json.loads((project / "inventory.json").read_text(encoding="utf-8"))["frames"]
    record, gray = api.load_reference_frame(config, inventory, plan=None)
    assert int(record["native_frame"]) == 1
    assert gray.shape == (112, 112)

    # Confirm discovery.py, cli.py, and gui.py avoid direct private api._* access.
    for rel in ("straintrace_agent/discovery.py", "straintrace_agent/cli.py", "straintrace_agent/gui.py"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert not re.search(r"\bapi\._[a-zA-Z0-9_]+", text), f"found private api._* access in {rel}"


def test_build_report_false_removes_stale_owned_report_and_returns_none(tmp_path):
    folder = tmp_path / "exp"
    images = _make_sequence(folder)
    project = tmp_path / "project"
    api.init_project(project, images, frame_interval_s=0.5, step_hint_px=8)
    assert api.plan_project(project)["status"] == "ready"

    # Cancel after frame 1 with default build_report=True -> writes a 1-frame report/.
    def cancel_after_first(event):
        if event.get("event") == "frame_complete":
            api.cancel(project)

    cancelled = api.run(project, make_plots=False, progress=cancel_after_first)
    assert cancelled["status"] == "cancelled"
    run_root = Path(cancelled["run_dir"])
    assert (run_root / "report" / "index.html").is_file()

    # Tamper the stale report and resume with build_report=False -> owned stale/corrupt
    # report/ must be removed and must never be certified into manifest.json.
    (run_root / "report" / "index.html").write_text("CORRUPTED", encoding="utf-8")
    resumed = api.run(project, make_plots=False, build_report=False)
    assert resumed["status"] == "complete"
    assert resumed["artifacts"]["report_html"] is None
    assert not (run_root / "report").exists()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert not any(key.startswith("report/") for key in manifest["artifacts"])
    assert api.verify(project)["valid"] is True

    # An unrecognized (unowned) report/ dir must neither be deleted nor certified.
    unowned = run_root / "report"
    unowned.mkdir()
    (unowned / "user_notes.txt").write_text("keep me", encoding="utf-8")
    resumed_again = api.run(project, make_plots=False, build_report=False)
    assert resumed_again["artifacts"]["report_html"] is None
    assert (unowned / "user_notes.txt").is_file()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert not any(key.startswith("report/") for key in manifest["artifacts"])
    assert api.verify(project)["valid"] is True


def test_cancelled_run_respects_build_report_false(tmp_path):
    folder = tmp_path / "exp"
    images = _make_sequence(folder)
    project = tmp_path / "project"
    api.init_project(project, images, frame_interval_s=0.5, step_hint_px=8)
    assert api.plan_project(project)["status"] == "ready"

    def cancel_immediately(event):
        if event.get("event") == "run_started":
            api.cancel(project)

    cancelled = api.run(project, make_plots=False, build_report=False, progress=cancel_immediately)
    assert cancelled["status"] == "cancelled"
    run_root = Path(cancelled["run_dir"])
    assert not (run_root / "report").exists()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert not any(key.startswith("report/") for key in manifest["artifacts"])


def test_run_supports_atomic_report_with_checkpoints_and_auto_resume_never_certifies_stale_report(tmp_path, monkeypatch):
    folder = tmp_path / "exp"
    images = _make_sequence(folder)
    project = tmp_path / "project_direct"
    api.init_project(project, images, frame_interval_s=0.5, step_hint_px=8)
    assert api.plan_project(project)["status"] == "ready"

    calls = []
    real_build_report = reporting.build_report

    def spy_build(run_root, **kwargs):
        calls.append(kwargs.get("include_checkpoints"))
        return real_build_report(run_root, **kwargs)

    monkeypatch.setattr(reporting, "build_report", spy_build)
    res = api.run(project, make_plots=False, report_include_checkpoints=True)
    assert res["status"] == "complete"
    assert calls == [True]
    report_dir = Path(res["artifacts"]["report_html"]).parent
    verified = reporting.verify_report(report_dir)
    assert verified["valid"] is True
    assert verified["include_checkpoints"] is True
    assert api.verify(project)["valid"] is True

    # Auto-analyze: cancel -> resume where report() fails -> stale partial report
    # must already be removed and uncertified, never silently valid with 1-frame data.
    auto_proj = tmp_path / "project_auto"

    def cancel_after_first(event):
        if event.get("event") == "frame_complete":
            api.cancel(auto_proj)

    cancelled = api.auto_analyze(folder, auto_proj, make_plots=False, progress=cancel_after_first)
    assert cancelled["status"] == "cancelled"
    run_root = Path(cancelled["run"]["run_dir"])
    assert (run_root / "report" / "index.html").is_file()

    def fail_report(*_args, **_kwargs):
        raise OSError("injected interruption before report commit")

    with monkeypatch.context() as ctx:
        ctx.setattr(api, "report", fail_report)
        try:
            api.auto_analyze(folder, auto_proj, make_plots=False)
            assert False, "expected injected OSError"
        except OSError:
            pass
    assert not (run_root / "report").exists()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert not any(key.startswith("report/") for key in manifest["artifacts"])

    calls.clear()
    completed = api.auto_analyze(folder, auto_proj, make_plots=False)
    assert completed["status"] == "complete"
    assert calls == [True]
    assert completed["verification"]["valid"] is True
    assert reporting.verify_report(Path(completed["report"]["report_html"]).parent)["valid"] is True

