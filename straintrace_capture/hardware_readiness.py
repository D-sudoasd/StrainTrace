"""First-frame startup evidence from an already configured camera wait output."""

from __future__ import annotations

import time

from .config import CaptureError, positive_integer, positive_number
from .storage import TERMINAL_STATES, ExperimentReader


WAIT_POLL_S = .2
WAIT_MAX_AGE_MS = 750
_UNAVAILABLE_WITHOUT_CONFIGURATION = frozenset({
    "driver_unavailable", "line_catalog_unavailable", "wait_output_not_configured", "output_polarity_unconfirmed",
})


def hardware_readiness(samples, receipts, *, active: bool, now_ns: int) -> dict:
    cameras = {}
    for camera, count in receipts.items():
        sample, started_ns, elapsed_ms = samples.get(camera, (None, None, None))
        age = (now_ns - started_ns) / 1e6 if started_ns is not None else None
        if count:
            state = "received"
        elif sample is None:
            state = "checking"
        elif sample.asserted is None:
            state = "unavailable"
        elif not 0 <= age <= WAIT_MAX_AGE_MS:
            state = "stale"
        else:
            state = "ready" if sample.asserted else "not_ready"
        cameras[camera] = {"state": state, "line": sample.line if sample else None,
                           "reason": sample.reason if sample else None, "age_ms": age, "poll_ms": elapsed_ms}
    states = {info["state"] for info in cameras.values()}
    if not active:
        phase_state = "ending"
    elif "received" in states:
        phase_state = "receiving"
    elif states == {"ready"}:
        phase_state = "ready"
    elif "unavailable" in states:
        phase_state = "unverified"
    else:
        phase_state = "waiting"
    return {"state": phase_state, "ready": phase_state == "ready", "basis": "camera_frame_trigger_wait_output",
            "max_age_ms": WAIT_MAX_AGE_MS, "cameras": cameras}


def wait_for_hardware(root, phase: int, timeout: float) -> dict:
    """Wait on the local operator channel for fresh, pre-frame startup evidence."""
    from .control import send_command

    number = positive_integer(phase, "phase")
    seconds = positive_number(timeout, "timeout")
    with ExperimentReader(root) as reader:
        experiment = reader.experiment
        phases = experiment["config"]["phases"]
        cameras = {camera["id"] for camera in experiment["config"]["cameras"]}
        if number > len(phases) or phases[number - 1]["mode"] != "hardware":
            raise CaptureError("WAIT_HARDWARE_PHASE_INVALID", f"phase {number} is not a hardware trigger phase")
        if not phases[number - 1].get("monitor_trigger_wait"):
            raise CaptureError("WAIT_HARDWARE_MONITOR_DISABLED", f"enable trigger wait monitoring for phase {number}")
        if experiment["state"] in TERMINAL_STATES:
            raise CaptureError("WAIT_HARDWARE_PHASE_ENDED", f"hardware phase {number} is no longer active")
    deadline = time.perf_counter() + seconds
    latest = None
    while (remaining := deadline - time.perf_counter()) > 0:
        sent = time.perf_counter()
        try:
            reply = send_command(root, "status", timeout=remaining)
        except CaptureError as exc:
            if exc.code == "CONTROL_TIMEOUT" and latest is not None and time.perf_counter() >= deadline:
                break
            raise
        round_trip_ms = (time.perf_counter() - sent) * 1000
        if not reply["ok"]:
            return reply
        status = reply["status"]
        if status["experiment_id"] != experiment["experiment_id"]:
            raise CaptureError("CONTROL_ID_MISMATCH", "capture experiment changed while waiting")
        current = status["phase"]
        progress = status["phase_progress"]
        if status["state"] in TERMINAL_STATES or (current is not None and (
                current > number or (current == number and progress is None))):
            raise CaptureError("WAIT_HARDWARE_PHASE_ENDED", f"hardware phase {number} has already ended")
        if current == number and progress:
            latest = progress.get("hardware_readiness")
            if latest and latest["state"] == "receiving":
                raise CaptureError("WAIT_HARDWARE_FIRST_FRAME_RECEIVED", f"phase {number} already received an external trigger frame")
            if latest and any(row["reason"] in _UNAVAILABLE_WITHOUT_CONFIGURATION for row in latest["cameras"].values()):
                return {"ok": False, "experiment_id": status["experiment_id"], "phase": number, "hardware_readiness": latest,
                        "error": {"code": "WAIT_HARDWARE_SIGNAL_UNAVAILABLE", "message": "a camera has no verifiable wait output"}}
            if (latest and latest["ready"] and time.perf_counter() <= deadline and latest["cameras"].keys() == cameras
                and all(row["state"] == "ready" and 0 <= row["age_ms"] <= WAIT_MAX_AGE_MS - round_trip_ms
                        for row in latest["cameras"].values())):
                return {"ok": True, "experiment_id": status["experiment_id"], "phase": number,
                        "round_trip_ms": round_trip_ms, "hardware_readiness": latest}
        if (remaining := deadline - time.perf_counter()) > 0:
            time.sleep(min(.08, remaining))
    return {"ok": False, "experiment_id": experiment["experiment_id"], "phase": number, "hardware_readiness": latest,
            "error": {"code": "WAIT_HARDWARE_TIMEOUT", "message": f"no fresh wait signal for all cameras in phase {number}"}}
