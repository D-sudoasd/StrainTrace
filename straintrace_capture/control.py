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


class CaptureControl:
    """Use the output folder's filesystem permissions as the operator boundary."""

    def __init__(self, session):
        self.session = session
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
            if not isinstance(data, dict) or data.keys() - {"command", "source", "payload", "experiment_id"}:
                raise CaptureError("CONTROL_REQUEST_INVALID", "invalid command object")
            if data.get("experiment_id") != self.session.status()["experiment_id"]:
                raise CaptureError("CONTROL_ID_MISMATCH", "command does not target this experiment")
            action = data.get("command")
            if action == "trigger":
                event_id = self.session.request_trigger(data.get("source", "operator"), payload=data.get("payload"))
                result = {"ok": True, "signal_id": event_id}
            elif action == "stop":
                self.session.request_stop()
                result = {"ok": True, "state": "stop_requested"}
            else:
                raise CaptureError("CONTROL_REQUEST_INVALID", f"unknown command {action!r}")
        except (OSError, TypeError, ValueError, CaptureError) as exc:
            result = {"ok": False, "error": exc.as_dict() if isinstance(exc, CaptureError)
                      else {"code": "CONTROL_REQUEST_INVALID", "message": str(exc)}}
        try:
            _publish(path.with_suffix(".response"), result)
        except OSError:
            # The claimed command stays on disk. It must not be replayed after
            # an I/O failure that may have occurred after starting the action.
            pass


def send_command(root: str | Path, command: str, *, source="operator", payload=None, timeout=5.0) -> dict:
    with ExperimentReader(root) as reader:
        experiment = reader.experiment
        directory = reader.root / "control"
    if experiment["state"] in TERMINAL_STATES or not directory.is_dir():
        raise CaptureError("CONTROL_INACTIVE", "no active operator channel; enable it when starting Capture")
    key = str(uuid4())
    _publish(directory / f"{key}.request", {"command": command, "source": source, "payload": payload,
                                           "experiment_id": experiment["experiment_id"]})
    response = directory / f"{key}.response"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if response.is_file():
            return json.loads(response.read_text(encoding="utf-8"))
        time.sleep(0.05)
    raise CaptureError("CONTROL_TIMEOUT", f"no receipt for command {key}; check the ledger before resending")
