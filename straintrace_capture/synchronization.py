"""Explicit clock conversion, DAQ identity joins and bounded mechanics alignment."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from .config import CaptureError, fields, invalid, nonempty_text, positive_integer, positive_number
from .storage import ExperimentReader, sha256_file


@dataclass(frozen=True)
class ClockAnchor:
    """Measured conversion valid only across the verified source-clock interval."""

    clock_id: str
    unit: str
    value_at_anchor: int
    unix_ns_at_anchor: int
    units_per_second: int
    valid_min: int
    valid_max: int
    uncertainty_ns: int
    evidence: str

    @classmethod
    def from_dict(cls, value: Mapping) -> ClockAnchor:
        data = fields(value, set(cls.__dataclass_fields__), "clock anchor")
        if set(data) != set(cls.__dataclass_fields__):
            raise invalid("clock anchor must specify all conversion, validity and evidence fields")
        for name in ("value_at_anchor", "unix_ns_at_anchor", "units_per_second", "valid_min", "valid_max", "uncertainty_ns"):
            positive_integer(data[name], name, minimum=1 if name in ("units_per_second", "unix_ns_at_anchor") else 0)
        for name in ("clock_id", "unit", "evidence"):
            nonempty_text(data[name], name)
        anchor = cls(**data)
        if not anchor.valid_min <= anchor.value_at_anchor <= anchor.valid_max:
            raise invalid("clock anchor is outside its valid source clock interval")
        if anchor.unit not in ("ns", "ticks") or (anchor.unit == "ns" and anchor.units_per_second != 10**9):
            raise invalid("supported source units are ns at 1 GHz and device ticks with explicit frequency")
        return anchor

    def at(self, stamp: dict) -> int | None:
        if (stamp["clock_id"], stamp["unit"]) != (self.clock_id, self.unit):
            return None
        value = stamp["value"]
        if not self.valid_min <= value <= self.valid_max:
            return None
        frequency = stamp.get("tick_frequency_hz")
        if frequency is not None and frequency != self.units_per_second:
            raise CaptureError("CLOCK_FREQUENCY_CONFLICT", f"{self.clock_id} reports {frequency}, anchor expects {self.units_per_second}")
        return self.unix_ns_at_anchor + round(Fraction((value - self.value_at_anchor) * 10**9, self.units_per_second))


def capture_axis(frame: dict, basis: str = "exposure-midpoint", anchors=()) -> dict:
    """Expose time with provenance; a receipt is available only by explicit choice."""
    fields_by_basis = {"exposure-midpoint": "exposure_midpoint", "host-receive": "received_utc"}
    if basis not in fields_by_basis:
        raise CaptureError("CLOCK_BASIS_INVALID", f"unsupported capture time basis {basis!r}")
    by_clock = {item.clock_id: item for item in anchors}
    if len(by_clock) != len(anchors):
        raise CaptureError("CLOCK_ANCHOR_AMBIGUOUS", "provide at most one anchor per source clock")

    def convert(stamp):
        if stamp is None:
            return None, None, None
        anchor = by_clock.get(stamp["clock_id"])
        if anchor is not None:
            value = anchor.at(stamp)
            uncertainty = stamp.get("uncertainty_ns")
            return value, anchor.uncertainty_ns + uncertainty if uncertainty is not None else None, asdict(anchor)
        if stamp["reference"] == "unix_epoch" and stamp["unit"] == "ns":
            return stamp["value"], stamp["uncertainty_ns"], None
        return None, None, None

    stamps = frame["timestamps"]
    stamp = stamps[fields_by_basis[basis]]
    utc, uncertainty, mapping = convert(stamp)
    start, _, _ = convert(stamps["exposure_start"])
    end, _, _ = convert(stamps["exposure_end"])
    return {
        "frame": frame["sequence"], "frame_id": frame["frame_id"], "camera_id": frame["camera_id"],
        "time_basis": basis, "clock_id": stamp["clock_id"] if stamp else None,
        "timestamp_unix_ns": utc, "timestamp_unix_s": utc / 1e9 if utc is not None else None,
        "exposure_start_unix_ns": start, "exposure_end_unix_ns": end,
        "exposure_start_unix_s": start / 1e9 if start is not None else None,
        "exposure_end_unix_s": end / 1e9 if end is not None else None,
        "host_receive_utc_ns": stamps["received_utc"]["value"], "time_uncertainty_ns": uncertainty,
        "conversion_evidence": mapping,
        "timestamp_source": stamp["source"] if stamp else None,
        "timestamp_status": ("HOST_RECEIPT_PROXY" if basis == "host-receive" else "EXPOSURE_MIDPOINT_FROM_WINDOW") if utc is not None
                            else "OBSERVATION_OR_CLOCK_MAPPING_UNAVAILABLE",
        "clock_sync_status": stamp["sync_status"] if stamp else "missing",
    }


class LoadAlignment:
    """Use campaign time/unit validation and bounded interpolation for Capture."""

    def __init__(self, source, *, time_column, force_column, force_unit=None, max_gap_s=2.5,
                 basis="exposure-midpoint", anchors=(), mode="instantaneous", displacement_column=None,
                 displacement_unit="mm"):
        if mode not in ("instantaneous", "window_mean") or (mode == "window_mean" and basis == "host-receive"):
            raise CaptureError("ALIGNMENT_MODE_INVALID", "window_mean requires exposure start/end timestamps")
        self.source = Path(source).resolve()
        self.time_column, self.force_column, self.force_unit = time_column, force_column, force_unit
        self.max_gap_s = positive_number(max_gap_s, "maximum load sample gap (s)")
        self.basis, self.anchors, self.mode = basis, tuple(anchors), mode
        self.displacement_column, self.displacement_unit = displacement_column, displacement_unit
        if self.displacement_column and self.displacement_unit not in ("mm", "um", "m"):
            raise CaptureError("DISPLACEMENT_UNIT_UNSUPPORTED", "displacement unit must be mm, um or m")
        self._stamp = None
        self._table = None
        self.provenance = None

    def refresh(self):
        from straintrace_campaign._time_io import finite_number, read_load_table

        def stamp():
            identity = self.source.stat()
            return identity.st_size, identity.st_mtime_ns

        current = stamp()
        if self._stamp == current and self._table is not None:
            return
        table = read_load_table(self.source, self.time_column, self.force_column, self.force_unit)
        if self.displacement_column:
            if self.displacement_column not in table["headers"]:
                raise CaptureError("DISPLACEMENT_COLUMN_MISSING", f"missing {self.displacement_column!r} in mechanics CSV")
            factor = {"mm": 1., "um": .001, "m": 1000.}[self.displacement_unit]
            disp = [finite_number(row.get(self.displacement_column)) for row in table["rows"]]
            table["displacement"] = [value * factor if value is not None else None for value in disp]
        digest = sha256_file(self.source)
        if current != stamp():
            raise CaptureError("MECHANICS_SOURCE_UPDATING", "the mechanics CSV was changing during the read")
        self._table, self._stamp = table, current
        self.provenance = {
            "path": str(self.source), "sha256": digest, "time_column": self.time_column,
            "force_column": self.force_column, "force_unit_override": self.force_unit,
            "max_gap_s": self.max_gap_s, "time_basis": self.basis, "mode": self.mode,
            "displacement_column": self.displacement_column, "displacement_unit": self.displacement_unit,
        }

    def at(self, frame: dict, *, refresh=True) -> dict:
        from straintrace_campaign._time_io import bracket_time, clock_pair_compatible, window_mean_support

        if refresh:
            self.refresh()
        axis = capture_axis(frame, self.basis, self.anchors)
        axis_clock = (axis["timestamp_unix_s"], "unix_epoch_recorded", axis["clock_id"], None)
        data = self._table
        clocks = set(zip(data["time_domains"], data["clock_ids"]))
        matches = [clock_pair_compatible(axis_clock, (0., domain, identity, None))[0] for domain, identity in clocks]

        def interpolate(values, valid):
            if not all(matches) or not matches:
                return None, {"status": "NO_COMPATIBLE_LOAD_TIME", "indices": [], "weights": []}
            if self.mode == "window_mean":
                support = window_mean_support(data["times"], values, valid,
                                              axis["exposure_start_unix_s"], axis["exposure_end_unix_s"], self.max_gap_s)
                return support.get("force_N", support.get("value")), support
            support = bracket_time(data["times"], axis_clock[0], self.max_gap_s, valid)
            if support["status"] not in ("EXACT_TIME_SAMPLE", "LINEAR_TIME_INTERPOLATION"):
                return None, support
            return sum(values[index] * weight for index, weight in zip(support["indices"], support["weights"])), support

        force, force_support = interpolate(data["forces"], data["valid"])
        displacement = displacement_support = None
        if self.displacement_column:
            disp = data["displacement"]
            displacement, displacement_support = interpolate(disp, [value is not None for value in disp])
        return {**axis, "force_N": force, "force_support": force_support, "displacement_mm": displacement,
                "displacement_support": displacement_support, "mechanics_source": self.provenance}


def load_alignment_settings(data, *, base_dir: Path) -> LoadAlignment:
    settings = dict(data)
    source = base_dir / nonempty_text(settings.pop("source", None), "mechanics source path")
    anchors = tuple(ClockAnchor.from_dict(row) for row in settings.pop("anchors", ()))
    settings.setdefault("time_column", "timestamp_unix_s")
    settings.setdefault("force_column", "force_N")
    return LoadAlignment(source, anchors=anchors, **settings)


def timing_rows(root, camera_id, *, basis="exposure-midpoint", anchors=()):
    from .handoff import select_camera

    with ExperimentReader(root) as reader:
        camera_id = select_camera(reader, camera_id)
        exp_id = reader.experiment["experiment_id"]
        for frame in reader.frames(camera_id=camera_id):
            axis = capture_axis(frame, basis, anchors)
            utc = axis["timestamp_unix_ns"]
            timestamp = None
            if utc is not None:
                seconds, ns = divmod(utc, 10**9)
                timestamp = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{ns:09d}Z"
            start, end = axis["exposure_start_unix_ns"], axis["exposure_end_unix_ns"]
            t_start_s = exposure_s = None
            if start is not None and end is not None and end >= start:
                start_s, start_ns = divmod(start, 10**9)
                dur_s, dur_ns = divmod(end - start, 10**9)
                t_start_s, exposure_s = f"{start_s}.{start_ns:09d}", f"{dur_s}.{dur_ns:09d}"
            yield {"series_id": f"{exp_id}/{camera_id}", "timestamp": timestamp, "frame_index": 0,
                   "source_file": str(reader.image_path(frame)), "t_start_s": t_start_s, "exposure_s": exposure_s, **axis}


def link_detector(root, source, *, trigger_source, event_column="event_number", frame_column="detector_frame"):
    """Join a DAQ/detector CSV by shared pulse identity, using no positional inference."""
    from straintrace_campaign._time_io import strict_integer

    source = Path(source).resolve()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set((event_column, frame_column)).issubset(reader.fieldnames or []):
            raise CaptureError("DETECTOR_COLUMNS_MISSING", "detector CSV needs an explicit shared event and detector frame column")
        detector = {}
        seen_frames = set()
        for row in reader:
            number, frame = strict_integer(row[event_column]), row[frame_column]
            if number is None or number < 0 or not frame or number in detector or frame in seen_frames:
                raise CaptureError("DETECTOR_ID_AMBIGUOUS", "detector pulse and frame identities must be unique")
            detector[number] = row
            seen_frames.add(frame)
    with ExperimentReader(root) as experiment:
        pairing = experiment.pairing()
        groups = {}
        for entry in pairing["groups"]:
            if entry["external_source_id"] != trigger_source:
                continue
            number = entry["external_event_number"]
            if number in groups:
                raise CaptureError("PULSE_REUSED", "capture phases reuse this pulse identity; use unique source IDs or event ranges")
            groups[number] = entry
    linked = [
        {"external_event_number": identity, "detector": detector.get(identity), "capture": groups.get(identity),
         "association": ("identity_match" if groups[identity]["status"] == "associated" else "partial_capture")
                        if identity in detector and identity in groups
                        else ("detector_only" if identity in detector else "capture_only")}
        for identity in sorted(detector.keys() | groups.keys())
    ]
    return {"schema": "straintrace.capture-detector-link/v1", "trigger_source": trigger_source,
            "source": str(source), "source_sha256": sha256_file(source), "links": linked}


class ThresholdTrigger:
    """Receive observed samples and dispatch a camera event on an explicit threshold crossing."""

    def __init__(self, request_trigger, *, threshold: float, direction: str, quantity: str, unit: str, source: str):
        if direction not in ("rising", "falling") or type(threshold) not in (int, float) or not math.isfinite(threshold):
            raise CaptureError("THRESHOLD_INVALID", "a threshold needs a finite value and rising or falling direction")
        self.threshold, self.direction = threshold, direction
        self.quantity, self.unit, self.source = (nonempty_text(value, label)
                                                   for value, label in ((quantity, "quantity"), (unit, "unit"), (source, "source")))
        self.request_trigger = request_trigger
        self._prior = None

    def observe(self, value: float, *, source_time: dict | None = None) -> str | None:
        if type(value) not in (float, int) or not math.isfinite(value):
            self._prior = None
            return None
        prior = self._prior
        self._prior = value
        if prior is None:
            return None
        matched = (prior < self.threshold <= value if self.direction == "rising" else value <= self.threshold < prior)
        if matched:
            return self.request_trigger(self.source, payload={
                "quantity": self.quantity, "unit": self.unit, "value": value, "previous": prior,
                "threshold": self.threshold, "direction": self.direction, "source_time": source_time,
            })
        return None
