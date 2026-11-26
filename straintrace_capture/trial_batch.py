"""Run independent rehearsals in plan order using the regular Capture preflight."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

from .config import CaptureError, Phase, positive_number
from .control import CaptureControl
from .presentation import READINESS_CONFIRMATION_ISSUES
from .readiness import inspect_plan, probe_preflight_cameras
from .session import CaptureSession
from .trial import prepare_trial, trial_report


def batch_exclusion(phase):
    if phase.role != "test":
        return "optional"
    if phase.start_confirmation:
        return "confirmation"
    if phase.mode not in ("single", "continuous", "timed"):
        return "trigger"
    return ""


@dataclass(frozen=True, slots=True)
class TrialBatchEnd:
    state: str
    phase: int | None = None
    preflight: dict | None = None
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class TrialBatchLive:
    number: int
    plan: Phase
    session: CaptureSession
    status: dict
    external: bool = False

    @property
    def state(self):
        confirmation = self.status.get("phase_confirmation")
        if confirmation and confirmation["state"] == "waiting":
            return "confirmation"
        progress = self.status.get("phase_progress") or {}
        if progress.get("accepting_images"):
            if self.plan.mode == "hardware":
                return "hardware"
            if self.plan.mode == "manual" and self.action:
                return "manual"
            window = progress.get("event_window") or {}
            if window.get("state") == "buffering":
                return "buffering"
            if window.get("state") == "waiting":
                return "csv" if self.plan.csv_trigger else "manual"
        return "capturing"

    @property
    def action(self):
        confirmation = self.status.get("phase_confirmation")
        if confirmation and confirmation["state"] == "waiting":
            return "confirm", confirmation["phase"], confirmation["confirmation_id"]
        progress = self.status.get("phase_progress") or {}
        if not progress.get("trigger_available") or self.plan.csv_trigger is not None:
            return None
        if self.plan.mode == "manual" and not (progress["queued_signals"] or progress["manual_signal_id"]):
            return "trigger", self.status["phase"], None
        window = progress.get("event_window")
        if self.plan.mode == "event" and window and window["state"] == "waiting":
            return "trigger", self.status["phase"], window["event_number"]
        return None

    def refreshed(self):
        return replace(self, status=self.session.status())

    def request(self):
        action = self.action
        if action is None or action != self.refreshed().action:
            raise CaptureError("TRIAL_BATCH_ACTION_CHANGED", "当前等待状态已变化，请查看阶段提示。")
        kind, phase, token = action
        if kind == "confirm":
            return self.session.request_phase_confirmation(phase, confirmation_id=token)
        return self.session.request_trigger(expected_phase=phase, expected_window=token, require_prebuffer=token is not None)


def run_trial_batch(config, output, phases, seconds, *, registry, discovered_ids, stop,
                    watch_quality, progress, opened, completed, guided=False, observe=None, external=False):
    """Check, capture and verify one stage before opening the next camera session."""
    seconds = positive_number(seconds, "trial duration")
    phases = tuple(phases)
    if (not phases or any(type(number) is not int or not 1 <= number <= len(config.phases) for number in phases)
            or phases != tuple(sorted(set(phases)))):
        raise CaptureError("TRIAL_BATCH_PHASES", "连续试采要求按计划顺序选择实验图阶段。")
    plans = [config.phases[number - 1] for number in phases]
    if any(phase.role != "test" or (not guided and batch_exclusion(phase)) for phase in plans):
        raise CaptureError("TRIAL_BATCH_PHASES", "请在连续试采中纳入需确认或触发的实验图阶段。")
    if observe is None and not external and any(phase.start_confirmation for phase in plans):
        raise CaptureError("TRIAL_BATCH_CONFIRMATION_REQUIRED", "请使用带确认操作的连续试采入口。")

    for number, phase in zip(phases, plans, strict=True):
        if stop.is_set():
            return TrialBatchEnd("stopped")
        progress(number, "checking", None)
        try:
            planned, target = prepare_trial(config, output, phase_number=number, seconds=seconds,
                                            preserve_start_confirmation=guided)
            probe = probe_preflight_cameras(planned, registry, triggers=planned.phases[0].camera_trigger != "free")
            preflight = inspect_plan(planned, target, camera_settings=probe.settings, trigger_support=probe.triggers,
                                     camera_connections=probe.connections, discovered_ids=discovered_ids)
            if stop.is_set():
                return TrialBatchEnd("stopped", number)
            if not preflight["ready"] or any(issue["code"] in READINESS_CONFIRMATION_ISSUES for issue in preflight["issues"]):
                return TrialBatchEnd("preflight", number, preflight)
            live = observe if guided and batch_exclusion(phase) else None
            trial = _capture_trial(planned, target, number, registry, stop, watch_quality, progress, opened,
                                   observe=live, external=guided and external)
            state = trial["verification"]["state"]
            if state == "stopped":
                stage = "stopped"
            elif (guided and planned.phases[0].mode in ("manual", "event", "hardware") and state == "completed"
                  and not trial["verification"]["committed_frames"]):
                stage = "no_images"
            elif not trial["observed"]:
                stage = "failed"
            elif trial["findings"] or any(row.get("image_check", {}).get("flagged") for row in trial["cameras"].values()):
                stage = "review"
            else:
                stage = "completed"
            completed(number, stage, trial)
        except Exception as exc:
            return TrialBatchEnd("stopped" if stop.is_set() else "failed", number, error=exc)
        if stage in ("stopped", "failed", "no_images"):
            return TrialBatchEnd(stage, number)

    return TrialBatchEnd("completed")


def _capture_trial(config, target: Path, number, registry, stop, watch_quality, progress, opened, *, observe, external):
    plan = config.phases[0]
    session = CaptureSession(config, target, registry=registry, watch_quality=watch_quality)
    identity = session.start()
    try:
        opened(identity, config.name, session.output)
        progress(number, "capturing", 0)
        control = (CaptureControl(session, allow_trigger=plan.mode in ("manual", "event") and plan.csv_trigger is None)
                   if external else nullcontext())
        with control:
            while not session.finished:
                if stop.is_set():
                    session.request_stop()
                try:
                    session.wait(.18 if observe else .25)
                except TimeoutError:
                    status = session.status()
                    saved = sum(row["saved"] for row in status["cameras"].values())
                    if observe:
                        live = TrialBatchLive(number, plan, session, status, external)
                        progress(number, live.state, saved)
                        observe(live)
                    else:
                        progress(number, "capturing", saved)
    finally:
        if not session.finished:
            session.request_stop()
            session.wait()
        if observe:
            observe(None)
    progress(number, "verifying", None)
    return trial_report(session.output, expected_experiment_id=identity)
