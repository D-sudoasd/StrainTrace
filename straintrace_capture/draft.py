"""Persist the last valid desktop plan and reopen its trial evidence."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from ezdic_core import _operation_state_dir

from .config import CaptureConfig, CaptureError
from .recent import recent_experiments
from .storage import json_text
from .trial import compare_trials
from .trial_reference import referenced_trial_report, trial_locator


_SCHEMA = "straintrace.capture.draft/v1"


@dataclass(frozen=True)
class CaptureDraft:
    config: CaptureConfig
    output: str
    selected_phase: int
    trials: tuple[dict, ...]


def capture_draft(config, output, selected_phase, reports):
    destination = str(Path(output).expanduser().absolute()) if output else ""
    trials = []
    if destination:
        for report in reports:
            # Check camera and phase settings without opening the destination.
            # Preflight will compare the actual output volume before reusing evidence.
            _, matching = compare_trials(config, report["volume"], [report])
            if matching:
                trials.append(trial_locator(report, destination))
    return CaptureDraft(config, destination, selected_phase if selected_phase is not None else 0,
                        tuple(sorted(trials, key=lambda row: row["phase"])))


def _draft_path():
    return _operation_state_dir() / "capture-draft.sqlite3"


def load_capture_draft():
    path = _draft_path().absolute()
    if not path.is_file():
        return None
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=1)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='draft'").fetchone():
            return None
        row = db.execute("SELECT payload FROM draft WHERE singleton=1").fetchone()
    if row is None:
        return None
    payload = json.loads(row[0])
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise CaptureError("CAPTURE_DRAFT_INVALID", "unsupported Capture draft schema")
    config = CaptureConfig.from_dict(payload.get("config"))
    output, selected, trials = (payload.get(key) for key in ("output", "selected_phase", "trials"))
    if (not isinstance(output, str) or "\0" in output or (output and not Path(output).is_absolute())
        or type(selected) is not int or not 0 <= selected < len(config.phases) or not isinstance(trials, list) or (trials and not output)
        or not all(isinstance(item, dict) and type(item.get("phase")) is int and 1 <= item["phase"] <= len(config.phases)
                   and isinstance(item.get("experiment_id"), str) and isinstance(item.get("source"), dict)
                   and isinstance(item.get("image_checks"), dict) for item in trials)):
        raise CaptureError("CAPTURE_DRAFT_INVALID", "invalid saved Capture draft")
    return CaptureDraft(config, output, selected, tuple(trials))


def save_capture_draft(draft, edit_ns):
    path = _draft_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema": _SCHEMA, "config": draft.config.to_dict(), "output": draft.output,
            "selected_phase": draft.selected_phase, "trials": draft.trials}
    with closing(sqlite3.connect(path, timeout=1)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS draft (singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                   "edit_ns INTEGER NOT NULL, payload TEXT NOT NULL)")
        result = db.execute("""INSERT INTO draft VALUES (1, ?, ?)
                             ON CONFLICT(singleton) DO UPDATE SET edit_ns=excluded.edit_ns, payload=excluded.payload
                             WHERE excluded.edit_ns >= draft.edit_ns""", (edit_ns, json_text(data)))
        return result.rowcount == 1


def load_draft_trials(draft, *, cancelled=None):
    loaded, unavailable = [], []
    try:
        recent = {item.experiment_id: str(item.directory) for item in recent_experiments()} if draft.trials else {}
    except (OSError, sqlite3.Error):
        recent = {}
    for item in draft.trials:
        sources = [item]
        if (new := recent.get(item["experiment_id"])) and new != item["source"].get("directory"):
            sources.append({**item, "source": {"directory": new}})
        for source in sources:
            try:
                loaded.append(referenced_trial_report(draft.output, source, cancelled=cancelled))
                break
            except (CaptureError, OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
                if getattr(exc, "code", None) == "TRIAL_IMAGE_CHECK_CANCELLED":
                    raise
        else:
            unavailable.append(item["phase"])
    return loaded, unavailable
