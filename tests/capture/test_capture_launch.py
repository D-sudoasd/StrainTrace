"""Launch the sibling workbench from a source session outside the checkout."""

import subprocess

import pytest

from straintrace_capture.handoff import _launch_workbench, _workbench_command


@pytest.mark.parametrize(("executable", "module"), (
    ("ezDIC.exe", "dic_virtual_extensometer_gui_v7_multi_roi_range"),
    ("StrainTrace-Capture.exe", "straintrace_capture.gui"),
))
def test_source_workbench_launch_is_independent_of_callers_directory(tmp_path, monkeypatch, capfd, executable, module):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    process = _launch_workbench([*_workbench_command(executable, module), "--help"])
    try:
        returncode = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    output = capfd.readouterr()
    assert returncode == 0, output.out + output.err
    assert "usage:" in output.out
