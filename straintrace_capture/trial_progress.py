"""Summarize trial evidence and recommend which planned phase to look at next."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .output import output_ancestor, resolve_output_path
from .trial import compare_trials, trial_image_alerts, volume_identity
from .trial_batch import batch_exclusion


@dataclass(frozen=True, slots=True)
class PhaseTrial:
    number: int
    mode: str
    role: str
    status: str
    record: dict | None
    image_alerts: int
    issues: tuple[dict, ...]
    batch_exclusion: str = ""


@dataclass(frozen=True, slots=True)
class TrialProgress:
    phases: tuple[PhaseTrial, ...]
    recommended: int | None
    checked_plan: bool
    blockers: int

    @property
    def batch_phases(self):
        return self.trial_batch_phases()

    def trial_batch_phases(self, *, guided=False):
        return tuple(row.number for row in self.phases if row.role == "test"
                     and (guided or not row.batch_exclusion) and _pending_batch_trial(row, guided))


def _pending_batch_trial(row, guided):
    if row.status in ("missing", "changed", "unbound"):
        return True
    if row.status != "review":
        return False
    if row.record.get("trial_state") == "stopped":
        return True
    return (guided and row.mode in ("manual", "event", "hardware") and row.record.get("trial_state") == "completed"
            and any(item["code"] == "TRIAL_NO_FRAMES" for item in row.record["findings"]))


def summarize_trial_progress(phases, assessments, *, issues=None):
    historical = {row["phase"]: row for row in assessments}
    matched = {row["phase"]: row for row in assessments if row.get("latest_match") and row["status"] == "matched"}
    alerts = defaultdict(list)
    for issue in issues or ():
        if "phase" in issue and (issue["severity"] == "block" or issue["code"] == "TRIAL_CAMERA_READBACK_CHANGED"):
            alerts[issue["phase"]].append(issue)
    rows = []
    for number, phase in enumerate(phases, 1):
        current = matched.get(number)
        record = current or historical.get(number)
        image_alerts = trial_image_alerts(current or {})
        if current is not None:
            needs_review = (not current["observed"] or current["findings"] or current.get("sustained_write_lag")
                            or image_alerts or alerts[number])
            status = "review" if needs_review else "matched"
        else:
            status = record["status"] if record is not None else "missing"
        rows.append(PhaseTrial(number, phase.mode, phase.role, status, record, image_alerts, tuple(alerts[number]),
                               batch_exclusion(phase)))
    priority = {"review": 0, "changed": 1, "unbound": 1, "missing": 2}
    candidate = min((row for row in rows if row.status in priority and (row.status != "missing" or row.role == "test")),
                    key=lambda row: (priority[row.status], row.role != "test", row.number), default=None)
    return TrialProgress(tuple(rows), candidate.number if candidate else None, issues is not None,
                         sum(issue["severity"] == "block" for issue in issues or ()))


def inspect_trial_progress(config, output, reports):
    """Compare the current draft to loaded trials without connecting to cameras."""
    volume = volume_identity(output_ancestor(resolve_output_path(output))) if reports else None
    assessments, matching = compare_trials(config, volume, reports)
    for assessment, _record in matching.values():
        assessment["latest_match"] = True
    return summarize_trial_progress(config.phases, assessments)
