"""Static and source-level contracts for the v0.2.0-dev delivery surface."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

import ezdic_cli


ROOT = Path(__file__).resolve().parents[1]
DOI = "10.5281/zenodo.20222465"


def test_source_frozen_entrypoint_smoke_is_pre_tk_and_side_effect_free(tmp_path):
    marker = tmp_path / "source-smoke.json"
    env = os.environ.copy()
    env["EZDIC_FROZEN_SMOKE_MARKER"] = str(marker)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "ezdic_frozen_entrypoint.py"), "--smoke-test"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["smoke"] == "passed"
    assert payload["mode"] == "source"
    assert payload["tkinter_loaded"] is False
    assert set(payload["support_files"]) == {
        "README.md",
        "README_使用说明.txt",
        "RELEASE_NOTES_v0.2.0-dev.md",
        "VERSION.txt",
        "NOTICE_Attribution_and_Usage.txt",
        "LICENSE.txt",
        "CITATION.cff",
    }
    assert marker.parent == tmp_path


def test_frozen_entrypoint_defers_gui_import_until_normal_launch():
    text = (ROOT / "ezdic_frozen_entrypoint.py").read_text(encoding="utf-8")
    assert '"--smoke-test"' in text
    assert "import ezdic_benchmark as benchmark" in text
    assert "import ezdic_cli as cli" in text
    assert "import ezdic_core as core" in text
    assert "from dic_virtual_extensometer_gui_v7_multi_roi_range import main" in text
    assert text.index('"--smoke-test"') < text.index("from dic_virtual_extensometer_gui_v7_multi_roi_range import main")
    assert "tk.Tk(" not in text


def test_pyinstaller_spec_contains_two_entrypoints_and_runtime_data():
    text = (ROOT / "ezDIC.spec").read_text(encoding="utf-8")
    for token in (
        "ezdic_frozen_entrypoint.py",
        "ezdic_cli_entrypoint.py",
        "source_datas",
        "dic_virtual_extensometer_gui_v7_multi_roi_range.py",
        "name='ezDIC'",
        "name='ezDIC-cli'",
        "console=False",
        "console=True",
        "ezdic_core",
        "ezdic_cli",
        "ezdic_benchmark",
        "schemas/run_config_v1.json",
        "README.md",
        "RELEASE_NOTES_v0.2.0-dev.md",
        "NOTICE_Attribution_and_Usage.txt",
        "LICENSE.txt",
        "CITATION.cff",
    ):
        assert token in text, token


def test_build_script_is_fail_closed_and_contains_real_release_gates():
    text = (ROOT / "build_release.ps1").read_text(encoding="utf-8")
    for token in (
        "Assert-SafeBuildPath",
        "Remove-SafeBuildPath",
        "$BuildRoot",
        "$DistRoot",
        "$ReleaseRoot",
        "$VenvDir",
        "Test-ModernSourceInventory",
        "Test-BundleInventory",
        "py_compile",
        "pytest",
        "locked v6 synthetic benchmark",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "EZDIC_FROZEN_SMOKE_MARKER",
        "--smoke-test",
        "ezDIC-cli.exe",
        "frozen CLI locked v6 benchmark",
        "Compress-Archive",
        "Start-Process -FilePath $Executable -ArgumentList $Arguments -WindowStyle Hidden -Wait -PassThru",
        "Assert-NoReparsePath",
        "FileAttributes]::ReparsePoint",
        "Resolve-Path",
        "repository root",
        "GuardTest",
        "BundleTest",
        "BundlePath",
        "Assert-FrozenGuiProvenance",
        "_internal\\sources\\dic_virtual_extensometer_gui_v7_multi_roi_range.py",
        "Get-Sha256",
        "Assert-JsonProperties",
        "Assert-BenchmarkV6Report",
        "ezdic-benchmark-report-v6",
        "ezdic-benchmark-cases-v4",
        "quality_score_v1",
        "quality_auc",
        "benchmark_report_csv_sha256",
        "quality_false_accept_count",
        "synthetic_cases_source_sha256",
        "cases_json_sha256",
        "numeric_baseline_pass",
        "near_1d_preflight_pass",
        "quality_ranking_pass",
        "quality_threshold_evaluated",
        "quality_threshold_pass",
        "NOT_CALIBRATED",
        "AMBIGUOUS_TEXTURE",
        "solver_calls",
        "successful_export_artifacts",
        "0fb244586def285fe0ca6b816eac588ddae8d4a0de0ce1a313b183bf369db138",
        "0.000242870585615446",
        "0.000471525609103834",
        "0.000541210285865718",
        "0.000142756487216106",
        "0.000199604933633678",
        "0.000649429296399418",
        "0.000144908819464799",
        "0.000356810906530228",
        "0.000639258340868429",
        "0.000012809688012261",
        "0.000012733433353507",
    ):
        assert token in text, token
    # A recursive cleanup must not target a user-provided path or an unresolved
    # wildcard; all cleanup goes through the explicit allow-list guard.
    assert "Remove-Item -LiteralPath $fullPath -Recurse -Force" in text
    assert "Remove-Item -LiteralPath (Join-Path" not in text


def _run_guard_test(script: Path, target: Path, cwd: Path):
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-GuardTest",
            "-GuardPath",
            str(target),
        ],
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )


def _create_junction(path: Path, target: Path) -> subprocess.CompletedProcess:
    command = (
        "$ErrorActionPreference='Stop'; "
        f"New-Item -ItemType Junction -Path '{path}' -Target '{target}' | Out-Null"
    )
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )


def _run_bundle_test(script: Path, bundle: Path, cwd: Path):
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-BundleTest",
            "-BundlePath",
            str(bundle),
        ],
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )


def _make_simulated_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a disposable PyInstaller-6 onedir layout for inventory tests."""

    fixture = tmp_path / "bundle_fixture"
    fixture.mkdir()
    script = fixture / "build_release.ps1"
    shutil.copy2(ROOT / "build_release.ps1", script)
    shutil.copy2(ROOT / "dic_virtual_extensometer_gui_v7_multi_roi_range.py", fixture / "dic_virtual_extensometer_gui_v7_multi_roi_range.py")

    bundle = fixture / "dist" / "ezDIC_Windows_x64"
    internal = bundle / "_internal"
    (internal / "schemas").mkdir(parents=True)
    (internal / "sources" / "benchmarks").mkdir(parents=True)
    (internal / "benchmarks").mkdir(parents=True)
    for name in ("ezDIC.exe", "ezDIC-cli.exe"):
        (bundle / name).write_bytes(b"simulated executable")
    for relative in (
        "_internal/schemas/run_config_v1.json",
        "_internal/benchmarks/cases_v1.json",
        "_internal/sources/ezdic_core.py",
        "_internal/sources/ezdic_cli.py",
        "_internal/sources/ezdic_benchmark.py",
        "_internal/sources/benchmarks/run_benchmark.py",
        "_internal/sources/benchmarks/synthetic_cases.py",
    ):
        target = bundle / Path(relative)
        target.write_bytes(b"simulated source/data")
    shutil.copy2(ROOT / "dic_virtual_extensometer_gui_v7_multi_roi_range.py", internal / "sources" / "dic_virtual_extensometer_gui_v7_multi_roi_range.py")
    for name in (
        "README.md",
        "README_使用说明.txt",
        "RELEASE_NOTES_v0.2.0-dev.md",
        "VERSION.txt",
        "NOTICE_Attribution_and_Usage.txt",
        "LICENSE.txt",
        "CITATION.cff",
    ):
        shutil.copy2(ROOT / name, bundle / name)
    return fixture, script, bundle


@pytest.mark.skipif(os.name != "nt", reason="bundle inventory uses PowerShell")
def test_bundle_inventory_accepts_actual_onedir_layout_and_requires_gui_source(tmp_path):
    fixture, script, bundle = _make_simulated_bundle(tmp_path)
    result = _run_bundle_test(script, bundle, fixture)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "inventory/provenance test accepted" in output

    gui_source = bundle / "_internal" / "sources" / "dic_virtual_extensometer_gui_v7_multi_roi_range.py"
    gui_source.unlink()
    missing = _run_bundle_test(script, bundle, fixture)
    missing_output = missing.stdout + missing.stderr
    assert missing.returncode != 0, missing_output
    assert "dic_virtual_extensometer_gui_v7_multi_roi_range.py" in missing_output


@pytest.mark.skipif(os.name != "nt", reason="reparse-point guard requires Windows filesystem attributes")
def test_build_guard_rejects_temporary_junction_target(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    script = fixture / "build_release.ps1"
    shutil.copy2(ROOT / "build_release.ps1", script)
    build_root = fixture / "build"
    build_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    junction = build_root / "escape"
    created = _create_junction(junction, outside)
    if created.returncode != 0 or not junction.exists():
        pytest.skip("Windows junction creation is unavailable: " + (created.stderr or created.stdout))

    result = _run_guard_test(script, junction, fixture)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "reparse point" in output.lower(), output


@pytest.mark.skipif(os.name != "nt", reason="reparse-point guard requires Windows filesystem attributes")
def test_build_guard_rejects_temporary_directory_symlink_target(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    script = fixture / "build_release.ps1"
    shutil.copy2(ROOT / "build_release.ps1", script)
    build_root = fixture / "build"
    build_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = build_root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Windows directory symlink creation is unavailable: {exc}")
    if not link.exists():
        pytest.skip("Windows directory symlink was not created")

    result = _run_guard_test(script, link, fixture)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "reparse point" in output.lower(), output


@pytest.mark.skipif(os.name != "nt", reason="reparse-point guard requires Windows filesystem attributes")
def test_build_guard_rejects_reparse_repository_root(tmp_path):
    real_root = tmp_path / "real_repo"
    real_root.mkdir()
    script = real_root / "build_release.ps1"
    shutil.copy2(ROOT / "build_release.ps1", script)
    linked_root = tmp_path / "linked_repo"
    created = _create_junction(linked_root, real_root)
    if created.returncode != 0 or not linked_root.exists():
        pytest.skip("Windows junction creation is unavailable: " + (created.stderr or created.stdout))

    linked_script = linked_root / "build_release.ps1"
    result = _run_guard_test(linked_script, linked_root / "build" / "target", linked_root)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "repository root" in output.lower(), output
    assert "reparse point" in output.lower(), output


@pytest.mark.skipif(os.name != "nt", reason="reparse-point guard requires Windows filesystem attributes")
def test_build_rejects_existing_venv_junction_before_python_probe(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    script = fixture / "build_release.ps1"
    shutil.copy2(ROOT / "build_release.ps1", script)
    outside = tmp_path / "external-venv"
    (outside / "Scripts").mkdir(parents=True)
    # The target only needs a leaf so the old vulnerable branch would reach
    # Test-VenvPython; the new tree guard must reject the junction first.
    shutil.copy2(sys.executable, outside / "Scripts" / "python.exe")
    venv_link = fixture / ".venv-build"
    created = _create_junction(venv_link, outside)
    if created.returncode != 0 or not venv_link.exists():
        pytest.skip("Windows junction creation is unavailable: " + (created.stderr or created.stdout))

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-SmokeTest",
        ],
        cwd=fixture,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "reparse point" in output.lower(), output
    assert "python probe" not in output.lower(), output


def test_requirements_and_ci_pin_python_delivery_contract():
    requirements = (ROOT / "requirements-build.txt").read_text(encoding="utf-8").lower()
    assert "-r requirements.txt" in requirements
    assert "pyinstaller==" in requirements
    assert "pytest==" in requirements

    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'python-version: "3.11"' in ci
    assert "requirements-build.txt" in ci
    assert "pytest" in ci
    assert "py_compile" in ci
    assert "ezdic_cli.py benchmark" in ci
    assert "benchmarks/cases_v1.json" in ci
    assert "ezdic-benchmark-report-v6" in ci
    assert "ezdic-benchmark-cases-v4" in ci
    assert "benchmark_report_csv_sha256" in ci
    assert "quality_false_accept_count" in ci
    assert "synthetic_cases_source_sha256" in ci
    assert "cases_json_sha256" in ci
    assert "numeric_baseline_pass" in ci
    assert "near_1d_preflight_pass" in ci
    assert "quality_ranking_pass" in ci
    assert "quality_threshold_evaluated" in ci
    assert "quality_threshold_pass" in ci
    assert "quality_score_v1" in ci
    assert "quality_auc" in ci
    assert "point_count" in ci
    assert "AMBIGUOUS_TEXTURE" in ci
    assert "solver_calls" in ci
    assert "0fb244586def285fe0ca6b816eac588ddae8d4a0de0ce1a313b183bf369db138" in ci
    assert "Benchmark CSV hash mismatch" in ci
    assert "_internal\\sources\\dic_virtual_extensometer_gui_v7_multi_roi_range.py" in ci
    assert "Get-FileHash" in ci
    assert "PyInstaller" in ci
    assert "EZDIC_FROZEN_SMOKE_MARKER" in ci
    assert "ezDIC-cli.exe" in ci


def test_development_docs_state_scope_metrics_limits_and_attribution():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README_使用说明.txt").read_text(encoding="utf-8")
    notes = (ROOT / "RELEASE_NOTES_v0.2.0-dev.md").read_text(encoding="utf-8")
    combined = "\n".join((readme, chinese, notes))

    for token in (
        "v0.2.0",
        "not released",
        "fixed-reference",
        "local-subset",
        "in-plane",
        "multiscale",
        "IC-GN",
        "IC-LM",
        "strain_valid",
        "_previous_runs/<run_id>",
        "_failed_runs/<run_id>",
        "manifest",
        "0.0199391",
        "0.0115297",
        "0.0239809",
        "0.0325828",
        "0.0269902",
        "0.00363440",
        "0.00651265",
        "0.0103737",
        "0.000273879",
        "0.000270908",
        "report_version=ezdic-benchmark-report-v5",
        "cases_version=ezdic-benchmark-cases-v3",
        "quality-score v1",
        "567",
        "563",
        "4",
        "0.25 px",
        "NOT_CALIBRATED",
        "100%",
        "50%",
        "2/2 = 100%",
        "2/4 = 50%",
        "AMBIGUOUS_TEXTURE",
        "false accept",
        "experimental uncertainty",
        "universal",
        "stereo/3D",
        "DVC",
        "GPU/MPI",
        "global",
        "uncertainty",
        DOI,
        "Dr. Delun Gong",
    ):
        assert token in combined, token
    assert "VERSION.txt" in readme and "CITATION.cff" in readme
    assert "0.1.4" in notes
    assert "run --progress-json" in readme
    assert "verify-manifest" in readme
    assert "run --progress-json" in notes
    assert "benchmark_report.csv" in notes
    assert "numeric solver rows" in combined
    assert "not calibrated" in combined.lower()
    assert "benchmark-report-v3" not in combined
    assert "benchmark-cases-v1" not in combined
    assert "NOT_IMPLEMENTED" not in combined
    assert "pending" not in combined.lower()
    assert "ignored local v0.1.4 ZIP may exist" in notes
    assert "not replaced or published" in notes


def test_mit_transition_preserves_historical_license_and_source_version():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    version = (ROOT / "VERSION.txt").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE_Attribution_and_Usage.txt").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE.txt").read_text(encoding="utf-8")
    assert DOI in citation and DOI in version
    assert "Developer:\nDr. Delun Gong" in notice
    assert "Permission is hereby granted" in license_text
    historical = (ROOT / "docs/history/pre-mit/LICENSE.txt").read_text(encoding="utf-8")
    assert "all rights are reserved" in historical.lower()
    assert "0.1.4" in citation and "0.1.4" in version


def test_documented_1d_and_2d_json_blocks_validate_against_cli_schema():
    schema = json.loads((ROOT / "schemas" / "run_config_v1.json").read_text(encoding="utf-8"))
    allowed = set(schema["properties"])
    for filename in ("README.md", "README_使用说明.txt"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        blocks = re.findall(r"```json\r?\n(.*?)\r?\n```", text, flags=re.S)
        assert len(blocks) >= 2, filename
        for block in blocks[:2]:
            config = json.loads(block)
            assert set(config) <= allowed
            normalized = ezdic_cli.normalize_config(config, base_dir=ROOT)
            assert normalized["schema_version"] == 1
            assert normalized["analysis_mode"] in {"extensometer", "fullfield"}
