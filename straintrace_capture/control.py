"""Opt-in operator command channel on the Capture host."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from uuid import uuid4

from .config import CaptureError
from .storage import ExperimentReader, TERMINAL_STATES, json_text


def _publish(path: Path, data: dict) -> None:
    partial = path.with_suffix(".tmp")
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(json_text(data))
    partial.rename(path)


def _clear_status_poll(response: Path) -> None:
    for path in (response.with_suffix(".request"), response.with_suffix(".claimed"), response):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


class CaptureControl:
    """Use the output folder's filesystem permissions as the operator boundary."""

    def __init__(self, session, *, allow_trigger=True):
        self.session = session
        self.allow_trigger = allow_trigger
        self.folder = session.output / "control"
        self._stop = threading.Event()
        self._worker = None

    def __enter__(self):
        self.folder.mkdir()
        self._worker = threading.Thread(target=self._run, name="capture-operator-commands")
        self._worker.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._worker.join()

    def _run(self):
        while not self._stop.is_set():
            paths = []
            for path in self.folder.glob("*.request"):
                try:
                    paths.append((path.stat().st_mtime_ns, path))
                except FileNotFoundError:
                    continue
            for _, path in sorted(paths):
                self._execute(path)
            self._stop.wait(0.05)

    def _execute(self, path: Path) -> None:
        claimed = path.with_suffix(".claimed")
        try:
            path.rename(claimed)
        except OSError:
            return
        try:
            if claimed.stat().st_size > 65536:
                raise CaptureError("CONTROL_PAYLOAD_TOO_LARGE", "commands may be at most 64 KiB")
            data = json.loads(claimed.read_text(encoding="utf-8"))
            allowed = {"command", "source", "payload", "experiment_id", "expected_phase", "require_prebuffer",
                       "expected_window", "confirmation_id", "pause_id"}
            if not isinstance(data, dict) or data.keys() - allowed:
                raise CaptureError("CONTROL_REQUEST_INVALID", "invalid command object")
            status = self.session.status()
            if data.get("experiment_id") != status["experiment_id"]:
                raise CaptureError("CONTROL_ID_MISMATCH", "command does not target this experiment")
            action = data.get("command")
            if action == "trigger":
                if not self.allow_trigger:
                    raise CaptureError("CONTROL_TRIGGER_DISABLED", "enable external trigger commands when starting Capture")
                event_id = self.session.request_trigger(
                    data.get("source", "operator"), payload=data.get("payload"),
                    expected_phase=data.get("expected_phase"), expected_window=data.get("expected_window"),
                    require_prebuffer=data.get("require_prebuffer", False),
                )
                result = {"ok": True, "signal_id": event_id}
            elif action == "confirm-phase":
                identity = self.session.request_phase_confirmation(
                    data.get("expected_phase"), confirmation_id=data.get("confirmation_id"),
                    source=data.get("source", "operator"),
                )
                result = {"ok": True, "state": "confirmation_requested", "confirmation_id": identity}
            elif action in {"pause-phase", "resume-phase"}:
                if not self.allow_trigger:
                    raise CaptureError("CONTROL_PHASE_PAUSE_DISABLED", "enable external commands to pause or resume capture")
                phase, source = data.get("expected_phase"), data.get("source", "operator")
                if action == "pause-phase":
                    identity = self.session.request_phase_pause(phase, source=source)
                    state = "pause_requested"
                else:
                    identity = self.session.request_phase_resume(phase, data.get("pause_id"), source=source)
                    state = "resume_requested"
                result = {"ok": True, "state": state, "pause_id": identity}
            elif action == "mark":
                payload = data.get("payload")
                if payload is None:
                    payload = {}
                if not isinstance(payload, dict) or payload.keys() - {"label"}:
                    raise CaptureError("CONTROL_REQUEST_INVALID", "a marker payload may contain a text label")
                identity, _ = self.session.request_marker(payload.get("label", ""), source=data.get("source", "operator"))
                result = {"ok": True, "state": "marker_queued", "marker_id": identity}
            elif action == "status":
                result = {"ok": True, "status": status}
            elif action == "stop":
                self.session.request_stop()
                result = {"ok": True, "state": "stop_requested"}
            else:
                raise CaptureError("CONTROL_REQUEST_INVALID", f"unknown command {action!r}")
        except (OSError, TypeError, ValueError, CaptureError) as exc:
            result = {"ok": False, "error": exc.as_dict() if isinstance(exc, CaptureError)
                      else {"code": "CONTROL_REQUEST_INVALID", "message": str(exc)}}
        try:
            response = path.with_suffix(".response")
            if claimed.exists():
                _publish(response, result)
                if not claimed.exists():
                    # A requester may cancel its read-only status poll at its deadline.
                    response.unlink(missing_ok=True)
        except OSError:
            # The claimed command stays on disk. It must not be replayed after
            # an I/O failure that may have occurred after starting the action.
            pass


def send_command(root: str | Path, command: str, *, source="operator", payload=None, timeout=5.0,
                 expected_phase: int | None = None, expected_window: int | None = None, require_prebuffer: bool = False,
                 confirmation_id: str | None = None, pause_id: str | None = None) -> dict:
    with ExperimentReader(root) as reader:
        experiment = reader.experiment
        directory = reader.root / "control"
    if experiment["state"] in TERMINAL_STATES or not directory.is_dir():
        raise CaptureError("CONTROL_INACTIVE", "no active operator channel; enable it when starting Capture")
    key = str(uuid4())
    request = {"command": command, "source": source, "payload": payload, "experiment_id": experiment["experiment_id"]}
    if expected_phase is not None:
        request["expected_phase"] = expected_phase
    if expected_window is not None:
        request["expected_window"] = expected_window
    if require_prebuffer is not False:
        request["require_prebuffer"] = require_prebuffer
    if confirmation_id is not None:
        request["confirmation_id"] = confirmation_id
    if pause_id is not None:
        request["pause_id"] = pause_id
    _publish(directory / f"{key}.request", request)
    response = directory / f"{key}.response"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            receipt = response.read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError):
            # Windows may briefly deny a read immediately after atomic publication.
            # The action may already have happened, so keep waiting for this receipt.
            time.sleep(0.05)
            continue
        result = json.loads(receipt)
        if command == "status":
            # Polling uses the command transport but carries no operator action to archive.
            _clear_status_poll(response)
        return result
    if command == "status":
        _clear_status_poll(response)
    raise CaptureError("CONTROL_TIMEOUT", f"no receipt for command {key}; check the ledger before resending")
