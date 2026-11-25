"""Derive a lower acquisition rate from comparable trial measurements."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable
from decimal import ROUND_FLOOR, Decimal
from itertools import chain
from pathlib import Path

from .cameras.base import Feature
from .cameras.simulated import FEATURES as SIMULATOR_FEATURES
from .config import CaptureConfig, CaptureError
from .related import experiment_hint_paths
from .trial import DEFAULT_TRIAL_SECONDS, compare_trials, trial_report


def _positive(value) -> bool:
    return type(value) in (int, float) and 0 < value <= 1e15


def _rate_observed(camera, target) -> bool:
    measured = camera["host_receive_fps"]
    return (camera["committed_frames"] >= 4 and (camera["host_receive_span_s"] or 0) >= 4
            and _positive(measured) and _positive(target) and .9 * target <= measured <= 1.1 * target)


def _lower_supported_rate(limit, feature: Feature | None = None):
    if feature is None:
        return math.floor(limit * 100) / 100
    if limit < (feature.minimum or 0):
        return None
    if feature.increment and feature.minimum is not None:
        minimum, step = Decimal(str(feature.minimum)), Decimal(str(feature.increment))
        steps = ((Decimal(str(limit)) - minimum) / step).to_integral_value(rounding=ROUND_FLOOR)
        value = float(minimum + steps * step)
    else:
        value = math.floor(limit * 100) / 100
    if type(feature.value) is int:
        value = math.floor(value)
    try:
        feature.validate(value)
    except CaptureError:
        return None
    return value


def suggest_trial_rate(config: CaptureConfig, phase_number: int, report: dict) -> dict | None:
    """Use a verified, matching trial and keep 20% of the lower measured write rate in reserve."""
    if not report.get("sustained_write_lag") or not report["observed"] or not report["verification"]["file_hashes_checked"]:
        return None
    windows = report["write_timing"]["acquisition_windows"]
    cameras = report["cameras"]
    if len(windows) != 2 or any(row["geometry"] is None for row in cameras.values()):
        return None
    frame_sizes = {key: row["geometry"]["frame_bytes"] for key, row in cameras.items()}
    write_budget = .8 * min(window["committed_raw_bytes_per_s"] for window in windows)
    if not _positive(write_budget):
        return None

    phase = config.phases[phase_number - 1]
    request_rate = None
    camera_rates = {}
    if phase.mode == "timed":
        if not all(_rate_observed(row, phase.fps) for row in cameras.values()):
            return None
        round_bytes = sum(frame_sizes.values())
        proposed = _lower_supported_rate(write_budget / round_bytes)
        if not 0 < proposed < phase.fps:
            return None
        request_rate = {"current": phase.fps, "proposed": proposed}
        proposed_load = round_bytes * proposed
    elif phase.mode == "continuous" and all(item.mode in ("single", "continuous") for item in config.phases):
        current, features = {}, {}
        for identity, row in cameras.items():
            readback = row["phase_feature_readback"]
            enabled = readback.get("fps_enable", {}).get("value", row["configured_features"].get("fps_enable"))
            feature = row.get("idle_fps_feature")
            fps = readback.get("fps", {}).get("value")
            if (enabled in (False, "Off") or not feature or not feature["writable"]
                or not _rate_observed(row, fps)):
                return None
            current[identity] = fps
            features[identity] = SIMULATOR_FEATURES["fps"] if row["backend"] == "sim" else Feature(**feature)
        scale = write_budget / sum(frame_sizes[key] * rate for key, rate in current.items())
        if scale >= 1:
            return None
        for identity, fps in current.items():
            feature = features[identity]
            bound = fps * scale
            if feature.maximum is not None:
                bound = min(bound, feature.maximum)
            proposed = _lower_supported_rate(bound, feature)
            if proposed is None or not 0 < proposed < fps:
                return None
            camera_rates[identity] = {"current": fps, "proposed": proposed}
        proposed_load = sum(frame_sizes[key] * rate["proposed"] for key, rate in camera_rates.items())
    else:
        return None

    slowest = request_rate["proposed"] if request_rate else min(row["proposed"] for row in camera_rates.values())
    trial_seconds = max(DEFAULT_TRIAL_SECONDS, min(30, math.ceil(12 / slowest)))
    if phase.duration_s is not None:
        trial_seconds = min(trial_seconds, phase.duration_s)
    return {"phase": phase_number, "source_experiment_id": report["experiment_id"],
            "source_volume": report["volume"], "request_rate": request_rate, "camera_rates": camera_rates,
            "estimated_raw_bytes_per_s": proposed_load, "write_budget_raw_bytes_per_s": write_budget,
            "trial_seconds": trial_seconds}


def with_trial_rate(config: CaptureConfig, suggestion: dict) -> CaptureConfig:
    data = config.to_dict()
    if rate := suggestion["request_rate"]:
        data["phases"][suggestion["phase"] - 1]["fps"] = rate["proposed"]
    else:
        for camera in data["cameras"]:
            camera["settings"]["fps"] = suggestion["camera_rates"][camera["id"]]["proposed"]
    return CaptureConfig.from_dict(data)


def check_rate_trial(suggestion: dict, source: dict, report: dict) -> str:
    """Return the first unresolved measurement needed to adopt the rate."""
    if not report["observed"] or not report["verification"]["file_hashes_checked"]:
        return "file_integrity"
    if report["volume"] != suggestion["source_volume"]:
        return "volume"
    cameras = report["cameras"]
    originals = source["cameras"]
    if cameras.keys() != originals.keys() or any(
        row["geometry"] != originals[key]["geometry"] or row["formats"] != originals[key]["formats"]
        for key, row in cameras.items()
    ):
        return "geometry"
    for key, row in cameras.items():
        rate = suggestion["request_rate"] or suggestion["camera_rates"][key]
        if not _rate_observed(row, rate["proposed"]):
            return "cadence"
    timing = report.get("write_timing") or {}
    windows = timing.get("acquisition_windows", [])
    if len(windows) != 2 or any(item["accepted_raw_bytes_per_s"] <= 0 for item in windows):
        return "write_measurement"
    accepted = sum(item["accepted_raw_bytes_per_s"] * item["duration_s"] for item in windows)
    growth = sum(item["pending_raw_bytes_growth"] for item in windows)
    round_bytes = sum(row["geometry"]["frame_bytes"] for row in cameras.values())
    # A frame round may still be writing at the end of the phase, especially at low rates.
    if report["sustained_write_lag"] or growth > max(.1 * accepted, round_bytes):
        return "write_lag"
    if any(row["code"] in ("TRIAL_HIGH_BUFFER", "TRIAL_HIGH_WRITE_QUEUE") for row in report["findings"]):
        return "buffer_pressure"
    return "verified"


def read_rate_trial_source(candidate: dict, known_directories: Iterable[str | Path] = ()) -> dict | None:
    """Locate the original write-rate measurement by its recorded experiment ID."""
    trial = candidate["trial"]
    identity = trial.get("rate_source_experiment_id")
    if not isinstance(identity, str) or not identity.strip() or identity == candidate["experiment_id"]:
        return None
    root = Path(candidate["output"])
    hint = trial.get("rate_source")
    hinted = experiment_hint_paths(root, hint) if isinstance(hint, dict) else ()
    opened = (Path(path) for path in known_directories)
    attempted = set()
    for directory in chain(hinted, opened):
        if not directory.is_absolute() or directory in attempted:
            continue
        attempted.add(directory)
        try:
            return trial_report(directory, expected_experiment_id=identity, inspect_images=False)
        except (CaptureError, OSError, ValueError, sqlite3.Error):
            continue
    return None


def assess_rate_trial(config: CaptureConfig, volume: str | None, report: dict,
                      source: dict | None) -> tuple[str, CaptureConfig | None]:
    """Rebuild the proposal from both trials and the current formal plan."""
    if not report["observed"] or not report["verification"]["file_hashes_checked"]:
        return "file_integrity", None
    if volume is None or report["volume"] != volume:
        return "volume", None
    phase = report["trial"]["phase_number"]
    _, current = compare_trials(config, volume, [report])
    if phase in current:
        return "current", None
    if source is None or source["experiment_id"] != report["trial"].get("rate_source_experiment_id"):
        return "source_unavailable", None
    if not source["observed"] or not source["verification"]["file_hashes_checked"]:
        return "source_integrity", None
    if source["volume"] != volume:
        return "volume", None
    _, baseline = compare_trials(config, volume, [source])
    if phase not in baseline:
        return "original_plan", None
    suggestion = suggest_trial_rate(config, phase, source)
    if suggestion is None:
        return "source_measurement", None
    revised = with_trial_rate(config, suggestion)
    _, matching = compare_trials(revised, volume, [report])
    if phase not in matching:
        return "setup", None
    status = check_rate_trial(suggestion, source, report)
    return status, revised if status == "verified" else None
