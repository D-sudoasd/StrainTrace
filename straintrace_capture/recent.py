"""Local, bounded history of Capture experiments opened by the desktop."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from ezdic_core import _operation_state_dir


LIMIT = 12
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecentExperiment:
    experiment_id: str
    name: str
    directory: Path


def _history_path():
    return _operation_state_dir() / "capture-recent.sqlite3"


def recent_experiments():
    path = _history_path().resolve()
    if not path.is_file():
        return []
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=1)) as db:
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='recent'").fetchone() is None:
            return []
        rows = db.execute("SELECT experiment_id, name, directory FROM recent ORDER BY visit DESC LIMIT ?", (LIMIT,))
        return [RecentExperiment(identity, name, Path(directory)) for identity, name, directory in rows]


def remember_experiment(experiment: RecentExperiment):
    try:
        path = _history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=1)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS recent (experiment_id TEXT PRIMARY KEY, name TEXT NOT NULL, "
                       "directory TEXT NOT NULL, visit INTEGER NOT NULL)")
            db.execute("""INSERT INTO recent VALUES (?, ?, ?, (SELECT COALESCE(MAX(visit), 0) + 1 FROM recent))
                          ON CONFLICT(experiment_id) DO UPDATE SET name=excluded.name, directory=excluded.directory,
                          visit=excluded.visit""", (experiment.experiment_id, experiment.name, str(experiment.directory)))
            db.execute("DELETE FROM recent WHERE experiment_id NOT IN (SELECT experiment_id FROM recent ORDER BY visit DESC LIMIT ?)",
                       (LIMIT,))
    except (OSError, sqlite3.Error) as exc:
        _LOG.warning("Could not save the Capture experiment history: %s", exc)
