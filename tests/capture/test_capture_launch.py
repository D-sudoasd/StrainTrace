"""Launch the sibling workbench from a source session outside the checkout."""

import subprocess

import pytest

from straintrace_capture.handoff import _launch_workbench, _workbench_command


@pytest.mark.parametrize(("executable", "module"), (
    ("ezDIC.exe", "dic_virtual_extensometer_gui_v7_multi_roi_range"),
    ("StrainTrace-Capture.exe", "straintrace_capture.gui"),
))
def test_source_workbench_launch_is_independent_of_callers_directory(tmp_path, monkeypatch, executable, module):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    # A CREATE_NO_WINDOW child has no console on Windows, so read its output through pipes.
    process = _launch_workbench([*_workbench_command(executable, module), "--help"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    text = out.decode("utf-8", "replace")
    assert process.returncode == 0, text + err.decode("utf-8", "replace")
    assert "usage:" in text
