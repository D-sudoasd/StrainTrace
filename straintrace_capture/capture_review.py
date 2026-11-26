"""Summarize the acquisition ledger without opening image files."""

from dataclasses import dataclass
from pathlib import Path

from .config import CaptureError
from .csv_trigger import CSV_TRIGGER_EVENT
from .recovery import frame_write_records_need_review
from .storage import ExperimentReader, reported_camera_losses


@dataclass(frozen=True)
class CaptureReview:
    cameras: dict[str, dict[str, int]]
    camera_losses: dict[str, dict[str, int]]
    scheduled_misses: int
    short_pre_windows: int
    runtime_recorded: bool
    csv_triggers: tuple[tuple[int, str], ...] = ()
    event_phases: tuple[dict, ...] = ()
    write_records_need_review: bool = False
    recovery_recorded: bool = False
    incident_records: bool = False

    @property
    def frames(self) -> int:
        return sum(counts["frames"] for counts in self.cameras.values())

    @property
    def images(self) -> int:
        return sum(counts["images"] for counts in self.cameras.values())

    @property
    def has_incidents(self) -> bool:
        return bool(self.incident_records or self.camera_losses or self.scheduled_misses or self.short_pre_windows
                    or any(state == "fault" for _phase, state in self.csv_triggers)
                    or any(row["partial_events"] or row["busy_signals"] or row["early_signals"] for row in self.event_phases))


def check_review_identity(record: dict, expected_id: str) -> None:
    if record["experiment_id"] != expected_id:
        raise CaptureError("REVIEW_SCOPE_CHANGED", "实验标识已变化。请重新打开要复查的实验。")


def inspect_capture_review(root: str | Path, *, expected_id: str) -> CaptureReview:
    with ExperimentReader(root) as reader:
        experiment, counts = reader.committed_state()
        check_review_identity(experiment, expected_id)
        csv_triggers = tuple((row["phase"], row["state"]) for row in reader.events(kind=CSV_TRIGGER_EVENT))
        event_phases = tuple(reader.events(kind="event_phase_summary"))
        write_records_need_review = frame_write_records_need_review(reader, experiment)
        recovery_recorded = experiment["state"] in ("interrupted", "failed") and bool(reader.events(kind="recovery"))
        incident_records = reader.has_timeline_incidents()
    summary = experiment.get("summary") or {}
    return CaptureReview(
        counts, reported_camera_losses(summary.get("cameras", {})),
        summary.get("scheduled_misses", 0), summary.get("short_pre_windows", 0),
        {"cameras", "scheduled_misses", "short_pre_windows"} <= summary.keys(),
        csv_triggers, event_phases, write_records_need_review, recovery_recorded, incident_records,
    )
