"""A slow operator channel must end a hardware wait with the normal wait timeout."""

from contextlib import contextmanager

import straintrace_capture.control as control
import straintrace_capture.hardware_readiness as readiness
from straintrace_capture import CaptureError


def test_slow_status_poll_reports_wait_timeout_not_control_timeout(monkeypatch):
    class Reader:
        experiment = {
            "experiment_id": "exp-1", "state": "running",
            "config": {"phases": [{"mode": "hardware", "monitor_trigger_wait": True}],
                       "cameras": [{"id": "sim:0"}]},
        }

    @contextmanager
    def fake_reader(_root):
        yield Reader

    polls = []

    def never_answers(_root, command, *, timeout):
        polls.append((command, timeout))
        raise CaptureError("CONTROL_TIMEOUT", "no receipt for command x; check the ledger before resending")

    monkeypatch.setattr(readiness, "ExperimentReader", fake_reader)
    monkeypatch.setattr(control, "send_command", never_answers)

    result = readiness.wait_for_hardware("ignored", 1, .05)

    assert not result["ok"] and result["error"]["code"] == "WAIT_HARDWARE_TIMEOUT"
    assert result["hardware_readiness"] is None
    # Each status poll gets at least a minimum window even when little of the deadline remains.
    assert polls and all(timeout >= readiness.MIN_STATUS_POLL_S for _, timeout in polls)
