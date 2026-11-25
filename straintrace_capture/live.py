"""Read-only online analysis of committed Capture frames."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter

from .config import CaptureError
from .handoff import select_camera
from .image_inputs import read_analysis_image
from .storage import ExperimentReader


def _roi(rect) -> tuple[int, int, int, int]:
    if not isinstance(rect, (tuple, list)) or len(rect) != 4 or any(type(value) is not int for value in rect):
        raise CaptureError("ROI_INVALID", "regions must contain four integer values x, y, width, height")
    if min(rect[:2]) < 0 or min(rect[2:]) < 1:
        raise CaptureError("ROI_INVALID", "regions must have nonnegative origins and positive dimensions")
    return tuple(rect)


def number_or_none(value):
    return float(value) if value is not None and math.isfinite(value) else None


class LiveAnalysis:
    """Independent Analysis consumer. Capture never waits for this process."""

    def __init__(self, root: str | Path, camera_id: str | None = None, *, reference_sequence: int | None = None,
                 gauges=(), field_roi=None, field_step=12, field_subset=21, latest=True, load_alignment=None,
                 debayer_luminance=False):
        if not gauges and field_roi is None:
            raise CaptureError("ANALYSIS_ROI_REQUIRED", "configure at least one gauge or a field ROI")
        import ezdic_core as core

        self._core = core
        self._root = Path(root).resolve()
        self._requested_camera = camera_id
        self.camera_id = None
        self.reference_sequence = reference_sequence
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
        if type(debayer_luminance) is not bool:
            raise CaptureError("BAYER_OPTION_INVALID", "debayer_luminance must be a boolean")
        self.debayer_luminance = debayer_luminance
        self.reader = None
        self.reference = None
        self.image = None
        self.field = None
        self._cursor = 0
        self.skipped = 0
        self.processed = 0

    def _connect(self):
        if self.reader is None:
            if not (self._root / "capture.sqlite3").is_file():
                return False
            self.reader = ExperimentReader(self._root)
        if self.camera_id is None:
            if not self.reader.cameras():
                return False
            self.camera_id = select_camera(self.reader, self._requested_camera)
        return True

    def _initialize(self):
        candidates = self.reader.reference_frames(self.camera_id, sequence=self.reference_sequence)
        if len(candidates) > 1:
            raise CaptureError("CHOOSE_REFERENCE", "select the committed reference sequence for this camera")
        if not candidates:
            return None
        reference = candidates[0]
        raw = read_analysis_image(self.reader.image_path(reference), reference, debayer=self.debayer_luminance)
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
        return reference

    def poll(self) -> dict | None:
        if not self._connect():
            return None
        if self.reference is None:
            if (row := self._initialize()) is None:
                return None
            omitted = 0
            raw = self._reference_raw
        else:
            row, omitted = self.reader.next_test_frame(self.camera_id, after=self._cursor, latest=self.latest)
            if row is None:
                return None
            raw = read_analysis_image(self.reader.image_path(row), row, debayer=self.debayer_luminance)
        if raw.shape != self._reference_raw.shape:
            raise CaptureError("FRAME_GEOMETRY_CHANGED", f"live frame {row['frame_id']} differs from the reference geometry")
        image8 = self._core.normalize_with_bounds(raw, self._norm)
        started = perf_counter()
        gauges = []
        for state in self._gauge_states:
            tracked, _ = self._core.track_extensometer_group_frame(
                state, image8, row["sequence"] - self.reference["sequence"], row["path"],
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
        self._cursor = row["ordinal"]
        self.processed += 1
        self.skipped += omitted
        elapsed = (row["timestamps"]["received_monotonic"]["value"] -
                   self.reference["timestamps"]["received_monotonic"]["value"]) / 1e9
        conversion = {
            "bayer": "derived_bayer_luminance", "rgba": "unassociated_alpha_excluded",
            "bgra": "unassociated_alpha_excluded",
        }.get(row["color_model"], "straintrace_core_grayscale")
        output = {
            "schema": "straintrace.live-analysis/v1", "provisional": True,
            "frame_id": row["frame_id"], "reference_frame_id": self.reference["frame_id"], "sequence": row["sequence"],
            "elapsed_receive_s": elapsed, "timestamps": row["timestamps"],
            "processed": self.processed, "skipped_committed_frames": self.skipped,
            "analysis_compute_s": perf_counter() - started,
            "pixel_format": row["pixel_format"], "significant_bits": row["significant_bits"],
            "image_conversion": conversion,
            "gauges": gauges, "field_quality": field_summary,
        }
        if self.load_alignment:
            try:
                output["mechanics"] = self.load_alignment.at(row)
            except (CaptureError, OSError, RuntimeError) as exc:
                output["mechanics"] = {"force_N": None, "error": str(exc)}
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
