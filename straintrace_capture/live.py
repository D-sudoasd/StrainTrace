"""Read-only online analysis of committed Capture frames."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .calibration import calibration_geometry, phase_sensor_geometry
from .config import CaptureError
from .continuation_analysis import (
    continuation_snapshot, live_reference_root, read_live_reference, require_continuation_geometry,
)
from .handoff import select_camera
from .image_inputs import (
    frame_part_selector, read_analysis_image, require_part_geometry, select_frame_image, validate_analysis_part, verified_image_path,
)
from .storage import ExperimentReader, sha256_file


def _roi(rect) -> tuple[int, int, int, int]:
    if not isinstance(rect, (tuple, list)) or len(rect) != 4 or any(type(value) is not int for value in rect):
        raise CaptureError("ROI_INVALID", "regions must contain four integer values x, y, width, height")
    if min(rect[:2]) < 0 or min(rect[2:]) < 1:
        raise CaptureError("ROI_INVALID", "regions must have nonnegative origins and positive dimensions")
    return tuple(rect)


def number_or_none(value):
    return float(value) if value is not None and math.isfinite(value) else None


def _receive_elapsed(row, first):
    sample = row["timestamps"].get("received_monotonic") or {}
    anchor = first["timestamps"].get("received_monotonic") or {}
    actual, origin = sample.get("value"), anchor.get("value")
    if (type(actual) is int and type(origin) is int and sample.get("clock_id") and
            sample["clock_id"] == anchor.get("clock_id") and sample.get("unit") == anchor.get("unit") == "ns"
            and actual >= origin):
        return (actual - origin) / 1e9
    return None


def read_linked_reference_image(reference, *, debayer):
    path = Path(reference["absolute_path"])
    image = read_analysis_image(path, reference, debayer=debayer)
    if sha256_file(path) != reference["sha256"]:
        raise CaptureError("CAPTURE_IMAGE_CHANGED", reference["frame_id"])
    return image


class LiveAnalysis:
    """Independent Analysis consumer. Capture never waits for this process."""

    def __init__(self, root: str | Path, camera_id: str | None = None, *, reference_sequence: int | None = None,
                 image_part: str | None = None, gauges=(), field_roi=None, field_step=12, field_subset=21, latest=True, load_alignment=None,
                 debayer_luminance=False, reference_experiment=None, reference_binding=None, reference_binding_digest=None,
                 refresh_load=True):
        if not gauges and field_roi is None:
            raise CaptureError("ANALYSIS_ROI_REQUIRED", "configure at least one gauge or a field ROI")
        import ezdic_core as core

        self._core = core
        self._root = Path(root).resolve()
        if reference_binding is not None:
            if (reference_experiment is not None or reference_sequence is not None or
                    live_reference_root(reference_binding) != self._root):
                raise CaptureError("CAPTURE_BINDING_INVALID", "请选择本实验的实时参考图绑定。")
        if reference_experiment is not None and (type(reference_sequence) is not int or reference_sequence < 1):
            raise CaptureError("CONTINUATION_REFERENCE_SEQUENCE", "请指定原实验登记的参考图序号。")
        self._reference_binding = reference_binding
        self._reference_digest = reference_binding_digest
        self._reference_experiment = reference_experiment
        self._linked = None
        self._first_test = None
        self._requested_camera = camera_id
        self.camera_id = None
        self.reference_sequence = reference_sequence
        validate_analysis_part(image_part)
        self.image_part = image_part
        self.field_roi = _roi(field_roi) if field_roi is not None else None
        if type(field_step) is not int or type(field_subset) is not int or field_step < 1 or field_subset < 7 or field_subset % 2 != 1:
            raise CaptureError("FIELD_SETTINGS_INVALID", "field step must be positive and subset must be odd and at least 7 px")
        self.field_step, self.field_subset = field_step, field_subset
        self.groups = [{**group, "roi1": _roi(group["roi1"]), "roi2": _roi(group["roi2"])} for group in gauges]
        names = [group.get("name") for group in self.groups]
        if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
            raise CaptureError("GAUGE_NAMES_INVALID", "live gauges need unique names")
        for group in self.groups:
            group["actual_mode"] = core.resolve_strain_mode(
                group["roi1"], group["roi2"], group.get("strain_mode", "auto"),
            )
        self.latest = latest
        self.load_alignment = load_alignment
        self._refresh_load = refresh_load
        if type(debayer_luminance) is not bool:
            raise CaptureError("BAYER_OPTION_INVALID", "debayer_luminance must be a boolean")
        self.debayer_luminance = debayer_luminance
        self.reader = None
        self.reference = None
        self.image = None
        self.field = None
        self._cursor = 0
        self._counted_through = 0
        self._known_test_frames = 0
        self.skipped = 0
        self.processed = 0
        self.analyzed_test_frames = 0

    def _connect(self):
        if self.reader is None:
            if not (self._root / "capture.sqlite3").is_file():
                return False
            self.reader = ExperimentReader(self._root)
        if self.camera_id is None:
            if not self.reader.cameras():
                return False
            requested = self._requested_camera
            if self._reference_binding is not None:
                self._linked, _ = read_live_reference(self._reference_binding, expected_digest=self._reference_digest)
                if (requested is not None and requested != self._linked["camera_id"] or
                        self.image_part is not None and self.image_part != self._linked["image_part"]):
                    raise CaptureError("CAPTURE_BINDING_INVALID", "相机或图像部件与实时参考图绑定不同。")
                requested, self.image_part = self._linked["camera_id"], self._linked["image_part"]
            self.camera_id = select_camera(self.reader, requested)
        return True

    def _initialize(self):
        if self._reference_experiment is not None:
            self._linked = continuation_snapshot(self._root, self._reference_experiment, self.camera_id,
                                                 self.reference_sequence, image_part=self.image_part, live=True)
        if self._linked:
            reference = self._linked["frames"][0]
            self._linked_geometry = calibration_geometry(reference["shape"], reference["analysis_sensor_geometry"],
                                                        reference.get("backend_metadata") or {})
            self._phases = self._linked["phase_sensor_geometry"]
            raw = read_linked_reference_image(reference, debayer=self.debayer_luminance)
        else:
            candidates = self.reader.reference_frames(self.camera_id, sequence=self.reference_sequence)
            if len(candidates) > 1:
                raise CaptureError("CHOOSE_REFERENCE", "select the committed reference sequence for this camera")
            if not candidates:
                return None
            reference = select_frame_image(candidates[0], self.image_part)
            raw = self._read_image(reference)
        norm = self._core.compute_reference_normalization_bounds(raw)
        reference8 = self._core.normalize_with_bounds(raw, norm)
        states = []
        for group in self.groups:
            state = self._core.initialize_extensometer_group_state(reference8, group)
            code = state["texture_code1"] or state["texture_code2"]
            if code:
                raise CaptureError(code, f"{group['name']} failed reference texture preflight: {code}")
            states.append(state)
        if self.field_roi:
            if not self._core.rect_is_inside_image(self.field_roi, raw.shape):
                raise CaptureError("ROI_OUT_OF_BOUNDS", "live field ROI exceeds the reference image")
            self._core.require_texture(reference8, self.field_roi)
        self.reference = reference
        self._norm, self._reference_raw, self._gauge_states = norm, raw, states
        self._counted_through = 0 if self._linked else reference["ordinal"]
        return reference

    def _read_image(self, row):
        return read_analysis_image(verified_image_path(self.reader, row), row, debayer=self.debayer_luminance)

    def _check_live_geometry(self, row):
        phase = str(row["phase"])
        if phase not in self._phases:
            self._phases.update(phase_sensor_geometry(self.reader, self.camera_id))
        require_continuation_geometry(self.reference, self._linked_geometry, row,
                                      self._phases.get(phase, self._linked["camera_features"]))

    def record_context(self):
        """Snapshot the sources and settings used for the initialized reference."""
        experiment_id = self.reader.experiment["experiment_id"]
        external = self._linked["external_reference"] if self._linked else None
        source = {"experiment_id": experiment_id, "camera_id": self.camera_id,
                  "reference_experiment_id": external["experiment_id"] if external else experiment_id,
                  "reference_sha256": self.reference["sha256"]}
        settings = {"straintrace_version": self._core.APP_VERSION, "gauges": self.groups,
                    "field_roi": self.field_roi, "field_step_px": self.field_step, "field_subset_px": self.field_subset,
                    "image_part": frame_part_selector(self.reference), "debayer_luminance": self.debayer_luminance,
                    "external_reference": external, "reference_binding_sha256": self._reference_digest}
        if channel := self.load_alignment:
            settings["mechanics"] = {**(channel.provenance or channel.source_settings()),
                                     "anchors": [asdict(anchor) for anchor in channel.anchors]}
        return source, settings

    def progress(self) -> dict:
        """Measure the committed test backlog incrementally from the chosen reference."""
        pending = None
        if self.reference is not None and self.reader is not None:
            state, added, self._counted_through = self.reader.committed_test_progress(
                self.camera_id, after=self._counted_through,
            )
            self._known_test_frames += added
            pending = self._known_test_frames - self.analyzed_test_frames - self.skipped
        else:
            state = self.experiment_state
        return {"experiment_state": state, "reference_ready": self.reference is not None,
                "analyzed_test_frames": self.analyzed_test_frames, "skipped_committed_frames": self.skipped,
                "pending_committed_frames": pending}

    def poll(self, *, latest: bool | None = None) -> dict | None:
        if not self._connect():
            return None
        if self.reference is None:
            if (row := self._initialize()) is None:
                return None
            return self._measure(row, self._reference_raw, reference_row=True)
        row, omitted = self.reader.next_test_frame(
            self.camera_id, after=self._cursor, latest=self.latest if latest is None else latest,
        )
        return self.analyze_committed_test(row, omitted=omitted) if row is not None else None

    def analyze_committed_test(self, row, *, omitted=0):
        """Measure a test receipt from this analyzer's camera after initializing the reference."""
        row = select_frame_image(row, self.image_part)
        if self._linked:
            self._check_live_geometry(row)
            if self._first_test is None:
                if omitted:
                    self._first_test, _ = self.reader.next_test_frame(self.camera_id, after=0, latest=False)
                else:
                    self._first_test = row
        elif self.image_part is not None or frame_part_selector(self.reference) or frame_part_selector(row):
            require_part_geometry(self.reference, row)
        return self._measure(row, self._read_image(row), omitted=omitted)

    def _measure(self, row, raw, *, reference_row=False, omitted=0):
        if raw.shape != self._reference_raw.shape:
            raise CaptureError("FRAME_GEOMETRY_CHANGED", f"live frame {row['frame_id']} differs from the reference geometry")
        image8 = self._core.normalize_with_bounds(raw, self._norm)
        started = perf_counter()
        if self._linked:
            frame_index = 0 if reference_row else row["sequence"] - self._first_test["sequence"] + 1
            elapsed = None if reference_row else _receive_elapsed(row, self._first_test)
        else:
            frame_index = row["sequence"] - self.reference["sequence"]
            elapsed = (row["timestamps"]["received_monotonic"]["value"] -
                       self.reference["timestamps"]["received_monotonic"]["value"]) / 1e9
        gauges = []
        for state in self._gauge_states:
            tracked, _ = self._core.track_extensometer_group_frame(
                state, image8, frame_index, row["path"],
                {"use_prev_frame_template": False},
            )
            valid = bool(tracked["strain_valid"])
            gauges.append({
                "name": state["group"]["name"],
                "engineering_strain": number_or_none(tracked["engineering_strain"]) if valid else None,
                "quality_status": str(tracked["tracking_status_code"]),
                "strain_valid": valid,
            })
        field_summary = None
        if self.field_roi:
            self.field = self._core.run_2d_dic(
                self._core.normalize_dic_float(self._reference_raw, self._norm),
                self._core.normalize_dic_float(raw, self._norm), self.field_roi,
                step=self.field_step, subset_size=self.field_subset, reject_nonconverged=True,
            )
            field_summary = self.field["quality_summary"]
        self.image = raw
        if not (reference_row and self._linked):
            self._cursor = row["ordinal"]
            self.processed += 1
        if not reference_row:
            self.analyzed_test_frames += 1
        self.skipped += omitted
        conversion = {
            "bayer": "derived_bayer_luminance", "rgba": "unassociated_alpha_excluded",
            "bgra": "unassociated_alpha_excluded",
        }.get(row["color_model"], "straintrace_core_grayscale")
        output = {
            "schema": "straintrace.live-analysis/v1", "provisional": True,
            "frame_id": row["frame_id"], "reference_frame_id": self.reference["frame_id"], "sequence": row["sequence"],
            "elapsed_receive_s": elapsed, "timestamps": row["timestamps"],
            "processed": self.processed, "skipped_committed_frames": self.skipped,
            "analyzed_test_frames": self.analyzed_test_frames,
            "analysis_compute_s": perf_counter() - started,
            "pixel_format": row["pixel_format"], "significant_bits": row["significant_bits"],
            "image_part": frame_part_selector(row), "image_path": row["path"], "image_sha256": row["sha256"],
            "image_conversion": conversion,
            "gauges": gauges, "field_quality": field_summary,
        }
        if self._linked:
            source = self._linked["external_reference"]
            output.update(
                experiment_id=self._linked["experiment_id"],
                frame_experiment_id=source["experiment_id"] if reference_row else self._linked["experiment_id"],
                reference_experiment_id=source["experiment_id"], reference_sha256=self.reference["sha256"],
                is_reference=reference_row, elapsed_receive_basis="first_current_test_frame" if not reference_row else None,
            )
        if self.load_alignment and not (self._linked and reference_row):
            try:
                output["mechanics"] = self.load_alignment.at(row, refresh=self._refresh_load)
            except (CaptureError, OSError, RuntimeError, ValueError, csv.Error) as exc:
                output["mechanics"] = {"force_N": None, "error": str(exc),
                                       "force_support": {"status": getattr(exc, "code", "LOAD_SOURCE_UNAVAILABLE")}}
        return output

    @property
    def experiment_state(self):
        return self.reader.experiment["state"] if self.reader else "awaiting_capture"

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
