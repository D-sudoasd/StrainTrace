"""Bounded retry for transient Windows atomic JSON replacement failures."""

import pytest

from straintrace_agent.api import _atomic_bytes


def test_atomic_bytes_retries_transient_permission_error_then_commits(tmp_path, monkeypatch):
    import straintrace_agent.api as api

    destination = tmp_path / "progress.json"
    actual_replace = api.os.replace
    attempts = 0

    def transient_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("[WinError 5] transient sharing violation")
        return actual_replace(source, target)

    monkeypatch.setattr(api.os, "replace", transient_replace)
    _atomic_bytes(destination, b'{"fraction":0.5}\n')

    assert attempts == 3
    assert destination.read_bytes() == b'{"fraction":0.5}\n'
    assert not list(tmp_path.glob(".progress.json.*.tmp"))


def test_atomic_bytes_stops_after_bounded_permission_retries_and_cleans_temp(tmp_path, monkeypatch):
    import straintrace_agent.api as api

    destination = tmp_path / "progress.json"
    attempts = 0

    def always_denied(_source, _target):
        nonlocal attempts
        attempts += 1
        raise PermissionError("[WinError 5] persistent sharing violation")

    monkeypatch.setattr(api.os, "replace", always_denied)
    with pytest.raises(PermissionError, match="persistent sharing"):
        _atomic_bytes(destination, b"{}\n")

    assert attempts == len(api._REPLACE_RETRY_DELAYS_S) + 1
    assert not destination.exists()
    assert not list(tmp_path.glob(".progress.json.*.tmp"))
