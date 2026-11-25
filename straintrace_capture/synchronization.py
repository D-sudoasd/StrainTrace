"""Explicit clock conversion, DAQ identity joins and bounded mechanics alignment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from functools import partial
from pathlib import Path

from .config import CaptureError, fields, invalid, nonempty_text, positive_integer, positive_number
from .storage import ExperimentReader, sha256_file


LOAD_TIME_UNITS = {"s": 1., "ms": .001, "us": 1e-6, "ns": 1e-9}
_ORIGIN_STAMPS = {"host-receive": "received_monotonic", "exposure-midpoint": "exposure_midpoint"}
_SAMPLE_COLUMNS = ("times", "forces", "valid", "time_domains", "clock_ids")


def _sample_hash(digest, table, start, end):
    columns = [table[key][start:end] for key in _SAMPLE_COLUMNS]
    if "displacement" in table:
        columns.append(table["displacement"][start:end])
    for sample in zip(*columns, strict=True):
        values = tuple(0.0 if type(value) is float and value == 0. else value for value in sample)
        digest.update((json.dumps(values, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8"))


def _matches_sample_prefix(table, previous, count):
    keys = (*_SAMPLE_COLUMNS, "displacement") if "displacement" in previous else _SAMPLE_COLUMNS
    return all(table[key][:count] == previous[key] for key in keys)


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


def _relative_stamp(stamp):
    if not stamp or type(stamp["value"]) is not int or not stamp["clock_id"]:
        return None
    unit = stamp["unit"]
    rate = 10**9 if unit == "ns" else stamp.get("tick_frequency_hz") if unit == "ticks" else None
    if type(rate) is int and rate > 0:
        return stamp["clock_id"], unit, rate, stamp["reference"]
    return None


@dataclass(frozen=True)
class LoadClockOrigin:
    """A recorded camera frame and the load clock reading at the same event."""

    frame_id: str
    capture_value: int
    capture_clock_id: str
    capture_unit: str
    capture_rate_hz: int
    capture_reference: str
    load_time_s: float

    @property
    def camera_id(self):
        return self.frame_id.split("/", 1)[1].rsplit("/", 1)[0]

    @classmethod
    def from_frame(cls, frame, basis, load_time_s):
        stamp = frame["timestamps"].get(_ORIGIN_STAMPS[basis])
        signature = _relative_stamp(stamp)
        if signature is None:
            raise CaptureError("LOAD_SYNC_FRAME_CLOCK", "所选同步帧没有可用于对齐的时钟读数。")
        if not math.isfinite(load_time_s):
            raise CaptureError("LOAD_SYNC_TIME_INVALID", "同步事件在载荷表中的时间必须是有限数值。")
        return cls(frame["frame_id"], stamp["value"], *signature, load_time_s)

    def at(self, frame, field):
        if frame["frame_id"].rsplit("/", 1)[0] != self.frame_id.rsplit("/", 1)[0]:
            return None
        stamp = frame["timestamps"].get(field)
        expected = self.capture_clock_id, self.capture_unit, self.capture_rate_hz, self.capture_reference
        if _relative_stamp(stamp) != expected:
            return None
        return self.load_time_s + (stamp["value"] - self.capture_value) / self.capture_rate_hz


class LoadAlignment:
    """Use campaign time/unit validation and bounded interpolation for Capture."""

    def __init__(self, source, *, time_column, force_column, force_unit=None, max_gap_s=2.5,
                 basis="exposure-midpoint", anchors=(), mode="instantaneous", displacement_column=None,
                 displacement_unit="mm", time_format=None, time_unit="s", encoding="utf-8-sig",
                 load_origin=None):
        if basis not in _ORIGIN_STAMPS:
            raise CaptureError("CLOCK_BASIS_INVALID", f"unsupported capture time basis {basis!r}")
        if mode not in ("instantaneous", "window_mean") or (mode == "window_mean" and basis == "host-receive"):
            raise CaptureError("ALIGNMENT_MODE_INVALID", "window_mean requires exposure start/end timestamps")
        self.source = Path(source).resolve()
        self.time_column, self.force_column, self.force_unit = time_column, force_column, force_unit
        self.max_gap_s = positive_number(max_gap_s, "maximum load sample gap (s)")
        self.basis, self.anchors, self.mode = basis, tuple(anchors), mode
        self.displacement_column, self.displacement_unit = displacement_column, displacement_unit
        if self.displacement_column and self.displacement_unit not in ("mm", "um", "m"):
            raise CaptureError("DISPLACEMENT_UNIT_UNSUPPORTED", "displacement unit must be mm, um or m")
        if time_format not in (None, "absolute", "relative") or time_unit not in LOAD_TIME_UNITS:
            raise CaptureError("LOAD_TIME_FORMAT_INVALID", "请选择绝对时间或相对计时，并核对数字时间的单位。")
        if (time_format == "relative" and not isinstance(load_origin, LoadClockOrigin)
                or time_format != "relative" and load_origin is not None):
            raise CaptureError("LOAD_SYNC_FRAME_REQUIRED", "相对计时需要一张有已知对应载荷时刻的同步帧。")
        self.time_format, self.time_unit, self.encoding = time_format, time_unit, encoding
        self.load_origin = load_origin
        self._stamp = None
        self._table = None
        self._sample_digest = None
        self.provenance = None

    def refresh(self):
        from straintrace_campaign._time_io import finite_number, parse_row_time, read_load_table

        def stamp():
            identity = self.source.stat()
            return identity.st_dev, identity.st_ino, identity.st_size, identity.st_mtime_ns

        current = stamp()
        if self._stamp == current and self._table is not None:
            return
        parse_time = partial(parse_row_time, numeric_domain=self.time_format, numeric_scale=LOAD_TIME_UNITS[self.time_unit],
                             default_clock_id="load_table_relative")
        table = read_load_table(self.source, self.time_column, self.force_column, self.force_unit,
                                encoding=self.encoding, time_parser=parse_time)
        table["clocks"] = set(zip(table["time_domains"], table["clock_ids"]))
        if self.time_format == "relative" and len(table["clocks"]) != 1:
            raise CaptureError("LOAD_CLOCK_CONFLICT", "时间列含多套相对时钟。请使用同一台仪器、同一次计时的记录。")
        if self.time_format == "absolute" and any(domain != "unix_epoch_recorded" for domain, _ in table["clocks"]):
            raise CaptureError("LOAD_ABSOLUTE_TIME_REQUIRED", "绝对时间列需提供实测的 Unix 时间或带时区的日期时间。")
        if self.displacement_column:
            if self.displacement_column not in table["headers"]:
                raise CaptureError("DISPLACEMENT_COLUMN_MISSING", f"missing {self.displacement_column!r} in mechanics CSV")
            factor = {"mm": 1., "um": .001, "m": 1000.}[self.displacement_unit]
            disp = (finite_number(row.get(self.displacement_column)) for row in table["rows"])
            table["displacement"] = [finite_number(value * factor) if value is not None else None for value in disp]
            table["displacement_valid"] = [value is not None for value in table["displacement"]]
        digest = sha256_file(self.source)
        if current != stamp():
            raise CaptureError("MECHANICS_SOURCE_UPDATING", "the mechanics CSV was changing during the read")
        end = len(table["times"])
        count, sample_digest = 0, hashlib.sha256()
        if self._table is not None:
            previous_count = len(self._table["times"])
            if _matches_sample_prefix(table, self._table, previous_count):
                count, sample_digest = previous_count, self._sample_digest.copy()
            elif self.load_origin:
                raise CaptureError("LOAD_RELATIVE_SOURCE_CHANGED", "载荷表已重写。请重新核对同步事件后再匹配。")
        _sample_hash(sample_digest, table, count, end)
        self._table, self._stamp = table, current
        self._sample_digest = sample_digest
        self.provenance = {**self.source_settings(), "sha256": digest,
                           "sample_prefix": {"rows": end, "sha256": sample_digest.hexdigest()}}

    def sample_prefixes(self, counts):
        """Compute the requested prefix checksums against one validated table."""
        digest = hashlib.sha256()
        result, start = {}, 0
        end = len(self._table["times"])
        for count in sorted(counts):
            if not 0 < count <= end:
                break
            if count == end:
                result[count] = self._sample_digest.hexdigest()
                break
            _sample_hash(digest, self._table, start, count)
            result[count] = digest.hexdigest()
            start = count
        return result

    def source_settings(self):
        settings = {
            "path": str(self.source), "time_column": self.time_column,
            "force_column": self.force_column, "force_unit_override": self.force_unit,
            "max_gap_s": self.max_gap_s, "time_basis": self.basis, "mode": self.mode,
            "displacement_column": self.displacement_column, "displacement_unit": self.displacement_unit,
        }
        if self.time_format is not None:
            settings.update(time_format=self.time_format, time_unit=self.time_unit, encoding=self.encoding,
                            load_clock_origin=asdict(self.load_origin) if self.load_origin else None)
        return settings

    def at(self, frame: dict, *, refresh=True) -> dict:
        from straintrace_campaign._time_io import bracket_time, clock_pair_compatible, window_mean_support

        if refresh:
            self.refresh()
        axis = capture_axis(frame, self.basis, self.anchors)
        data = self._table
        query = axis["timestamp_unix_s"]
        start, end = axis["exposure_start_unix_s"], axis["exposure_end_unix_s"]
        if self.load_origin:
            query = self.load_origin.at(frame, _ORIGIN_STAMPS[self.basis])
            if self.mode == "window_mean":
                start, end = (self.load_origin.at(frame, name) for name in ("exposure_start", "exposure_end"))
            compatible = query is not None
        else:
            axis_clock = (query, "unix_epoch_recorded", axis["clock_id"], None)
            compatible = all(clock_pair_compatible(axis_clock, (0., domain, identity, None))[0]
                             for domain, identity in data["clocks"])

        def interpolate(values, valid):
            if not compatible:
                status = "LOAD_SYNC_CLOCK_MISMATCH" if self.load_origin else "NO_COMPATIBLE_LOAD_TIME"
                return None, {"status": status, "indices": [], "weights": []}
            if self.mode == "window_mean":
                support = window_mean_support(data["times"], values, valid, start, end, self.max_gap_s)
                return support.get("force_N", support.get("value")), support
            support = bracket_time(data["times"], query, self.max_gap_s, valid)
            if support["status"] not in ("EXACT_TIME_SAMPLE", "LINEAR_TIME_INTERPOLATION"):
                return None, support
            return sum(values[index] * weight for index, weight in zip(support["indices"], support["weights"])), support

        force, force_support = interpolate(data["forces"], data["valid"])
        displacement = displacement_support = None
        if self.displacement_column:
            displacement, displacement_support = interpolate(data["displacement"], data["displacement_valid"])
        return {**axis, "force_N": force, "force_support": force_support, "displacement_mm": displacement,
                "displacement_support": displacement_support, "mechanics_source": self.provenance,
                "load_time_s": query, "load_time_domain": "relative" if self.load_origin else "unix_epoch"}

    def summary(self):
        """Describe the validated load table used by the next calculation."""
        data = self._table
        return {"rows": len(data["times"]), "valid_force_rows": sum(data["valid"]),
                "first_time_s": data["times"][0], "last_time_s": data["times"][-1]}


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
