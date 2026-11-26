"""Windows launchers, release metadata and packaging contracts."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import dic_virtual_extensometer_gui_v7_multi_roi_range as ezdic

ROOT = Path(__file__).resolve().parents[1]
DOI = "10.5281/zenodo.20222465"
DOI_URL = f"https://doi.org/{DOI}"


def copy_release_contract_inputs(project):
    """Create a minimal but complete metadata fixture for build smoke tests."""
    names = [
        "VERSION.txt",
        "dic_virtual_extensometer_gui_v7_multi_roi_range.py",
        ".zenodo.json",
        "CITATION.cff",
        "requirements.txt",
        "requirements-origin.txt",
        "requirements-build.txt",
        "ezDIC.spec",
        "README_使用说明.txt",
        "NOTICE_Attribution_and_Usage.txt",
        "LICENSE.txt",
    ]
    for name in names:
        shutil.copy2(ROOT / name, project / name)


def test_app_metadata_and_usage_notice_are_explicit():
    assert ezdic.APP_NAME == "ezDIC"
    assert ezdic.APP_VERSION == "0.1.4"
    assert ezdic.APP_DEVELOPER == "Dr. Delun Gong"
    assert ezdic.APP_DOI == DOI
    assert ezdic.APP_DOI_URL == DOI_URL
    assert "developed by Dr. Delun Gong" in ezdic.USAGE_NOTICE
    assert DOI in ezdic.USAGE_NOTICE
    assert "Gong, D. (2026)." in ezdic.CITATION_TEXT
    assert DOI_URL in ezdic.CITATION_TEXT
    assert "MIT License" in ezdic.USAGE_NOTICE
    assert "redistribution are permitted under the MIT License" in ezdic.USAGE_NOTICE
    assert "WITHOUT WARRANTY OF ANY KIND" in ezdic.USAGE_NOTICE
    assert "not an additional license restriction" in ezdic.USAGE_NOTICE
    notice_lower = ezdic.USAGE_NOTICE.lower()
    assert "users are not permitted to" not in notice_lower
    assert "redistribute, copy, forward, or share" not in notice_lower
    assert "outside the authorized research" not in notice_lower


@pytest.mark.parametrize(("launcher_name", "label", "entry"), (
    ("start_ezDIC.bat", "EZDIC", "dic_virtual_extensometer_gui_v7_multi_roi_range.py"),
    ("start_Capture.bat", "Capture", "straintrace_capture.gui"),
))
def test_windows_launcher_bat_invokes_source_entry_in_smoke_mode(launcher_name, label, entry):
    launcher = ROOT / launcher_name
    assert launcher.is_file()
    if os.name != "nt":
        pytest.skip("Windows launcher smoke test requires cmd.exe")

    env = os.environ.copy()
    env["EZDIC_LAUNCHER_SMOKE_TEST"] = "1"
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", launcher_name],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
        timeout=15,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"{label} launcher smoke test" in output
    assert entry in output


@pytest.mark.parametrize(("launcher", "entry"), (
    ("start_ezDIC.bat", "dic_virtual_extensometer_gui_v7_multi_roi_range.py"),
    ("start_Capture.bat", "straintrace_capture/gui.py"),
))
def test_windows_launcher_recreates_stale_venv_before_running_entry(tmp_path, launcher, entry):
    if os.name != "nt":
        pytest.skip("Windows launcher stale-venv test requires cmd.exe")

    project = tmp_path / "launcher & source !"
    (project / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / launcher, project / launcher)
    shutil.copy2(ROOT / "scripts/start_windows.ps1", project / "scripts/start_windows.ps1")
    entry_path = project / entry
    entry_path.parent.mkdir(exist_ok=True)
    if launcher == "start_Capture.bat":
        (entry_path.parent / "__init__.py").touch()
    entry_path.write_text(
        "import ctypes, json, sys\n"
        "print('entry-ran', json.dumps(sys.argv[1:]))\n"
        "assert not ctypes.windll.kernel32.GetConsoleWindow()\n",
        encoding="utf-8",
    )

    stale_python = project / ".venv" / "Scripts" / "python.exe"
    stale_python.parent.mkdir(parents=True)
    prior_bytes = b"not a usable python executable\n"
    stale_python.write_bytes(prior_bytes)
    # A present configuration with an invalid executable also has to recover.
    (project / ".venv/pyvenv.cfg").write_text("home = missing\n", encoding="utf-8")
    requirements = b"# Empty dependency list for the launcher fixture.\n"
    (project / "requirements.txt").write_bytes(requirements)
    state = tmp_path / "logs"
    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"], "EZDIC_STATE_DIR": str(state)}
    env.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    env.pop("EZDIC_LAUNCHER_SMOKE_TEST", None)
    env.pop("PYTHONPATH", None)
    selection = str(tmp_path / "test experiment ! with spaces")

    result = subprocess.run(
        ["cmd.exe", "/d", "/c", launcher, "--experiment", selection],
        cwd=project,
        env=env,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
        timeout=60,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"entry-ran {json.dumps(['--experiment', selection])}" in output
    saved, = project.glob(".venv.backup-*")
    assert (saved / "Scripts/python.exe").read_bytes() == prior_bytes
    assert stale_python.read_bytes().startswith(b"MZ")
    assert (project / ".venv/.ezDIC_requirements.txt").read_bytes() == requirements
    log, = state.glob("startup-*.log")
    body = log.read_text(encoding="utf-8-sig")
    assert "entry-ran" in body and "Exit code 0" in body


def test_build_release_smoke_recreates_stale_build_venv(tmp_path):
    if os.name != "nt":
        pytest.skip("PowerShell build smoke test requires Windows")

    project = tmp_path / "build_project"
    project.mkdir()
    shutil.copy2(ROOT / "build_release.ps1", project / "build_release.ps1")
    copy_release_contract_inputs(project)

    stale_python = project / ".venv-build" / "Scripts" / "python.exe"
    stale_python.parent.mkdir(parents=True)
    stale_python.write_text("not a usable python executable\n", encoding="utf-8")

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(project / "build_release.ps1"),
            "-SmokeTest",
        ],
        cwd=project,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=90,
    )

    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output
    assert "ezDIC build smoke test" in output


def test_build_release_smoke_accepts_usable_venv_after_nonzero_create(tmp_path):
    if os.name != "nt":
        pytest.skip("PowerShell build smoke test requires Windows")

    project = tmp_path / "build_project"
    project.mkdir()
    shutil.copy2(ROOT / "build_release.ps1", project / "build_release.ps1")
    copy_release_contract_inputs(project)

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    (fake_bin / "py.cmd").write_text("@echo off\r\nexit /b 1\r\n", encoding="utf-8")
    (fake_bin / "python.cmd").write_text(
        "\r\n".join(
            [
                "@echo off",
                "if \"%1\"==\"--version\" (",
                "  echo Python 3.11.99",
                "  exit /b 0",
                ")",
                f"\"{sys.executable}\" %*",
                "if \"%1\"==\"-m\" if \"%2\"==\"venv\" exit /b 1",
                "exit /b %ERRORLEVEL%",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(project / "build_release.ps1"),
            "-SmokeTest",
        ],
        cwd=project,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=90,
    )

    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output
    assert "ezDIC build smoke test" in output


def test_release_support_files_exist_and_include_usage_limits():
    notice = ROOT / "NOTICE_Attribution_and_Usage.txt"
    readme = ROOT / "README_使用说明.txt"
    github_readme = ROOT / "README.md"
    citation = ROOT / "CITATION.cff"
    version = ROOT / "VERSION.txt"
    zenodo = ROOT / ".zenodo.json"
    release_notes = ROOT / "RELEASE_NOTES_v0.1.4.md"

    assert notice.exists()
    assert readme.exists()
    assert github_readme.exists()
    assert citation.exists()
    assert version.exists()
    assert zenodo.exists()
    assert release_notes.exists()

    notice_text = notice.read_text(encoding="utf-8")
    readme_text = readme.read_text(encoding="utf-8")
    github_readme_text = github_readme.read_text(encoding="utf-8")
    citation_text = citation.read_text(encoding="utf-8")
    version_text = version.read_text(encoding="utf-8")
    zenodo_text = zenodo.read_text(encoding="utf-8")
    release_notes_text = release_notes.read_text(encoding="utf-8")

    assert "Developer:\nDr. Delun Gong" in notice_text
    assert "MIT" in notice_text
    assert "redistribution are permitted" in notice_text
    assert "Windows 10/11 x64" in readme_text
    assert "Do not copy ezDIC.exe alone" in readme_text
    assert "Dr. Delun Gong" in readme_text
    assert DOI in readme_text
    assert DOI_URL in readme_text
    assert "virtual extensometer" in github_readme_text.lower()
    assert "Origin-compatible TXT" in github_readme_text
    assert "OPJU" in github_readme_text
    assert "OriginPro 2021+" in readme_text
    assert "publication_figures" in github_readme_text
    assert "论文图包" in readme_text
    assert "optional/publication_figures" in readme_text
    assert "PNG/TIFF/PDF/SVG/EPS" in readme_text
    assert "full-field DIC" in github_readme_text
    assert "Dr. Delun Gong" in github_readme_text
    assert DOI in github_readme_text
    assert DOI_URL in github_readme_text
    assert DOI in citation_text and "Historical ezDIC archive" in citation_text
    assert "license: MIT" in citation_text
    assert "ezDIC v0.1.4" in version_text
    assert DOI in version_text
    assert '"upload_type": "software"' in zenodo_text
    assert '"access_right": "open"' in zenodo_text
    assert '"title": "StrainTrace: Virtual extensometry and in-plane digital image correlation"' in zenodo_text
    assert "Poisson ratio export and GUI workflow update" in release_notes_text
    assert "How to cite" in release_notes_text
    assert DOI_URL in release_notes_text


def test_pyinstaller_build_files_define_green_folder_release():
    requirements = ROOT / "requirements.txt"
    build_requirements = ROOT / "requirements-build.txt"
    spec = ROOT / "ezDIC.spec"
    build_script = ROOT / "build_release.ps1"

    assert requirements.exists()
    assert build_requirements.exists()
    assert spec.exists()
    assert build_script.exists()

    req_text = requirements.read_text(encoding="utf-8")
    build_req_text = build_requirements.read_text(encoding="utf-8")
    spec_text = spec.read_text(encoding="utf-8")
    script_text = build_script.read_text(encoding="utf-8")

    for package in ["opencv-python", "numpy", "pandas", "matplotlib", "pillow", "scipy"]:
        assert package in req_text
    assert "originpro" not in req_text
    assert "originpro" in (ROOT / "requirements-origin.txt").read_text(encoding="utf-8")

    assert "pyinstaller" in build_req_text.lower()
    assert "name='ezDIC'" in spec_text
    assert "console=False" in spec_text
    assert "release" in script_text
    assert "ezDIC_Windows_x64" in script_text
    assert "Compress-Archive" in script_text
