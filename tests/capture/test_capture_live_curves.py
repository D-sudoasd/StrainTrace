from dataclasses import replace

import numpy as np
import pytest

from straintrace_capture import CaptureError, ExperimentReader
from straintrace_capture.live import LiveAnalysis
from straintrace_capture.live_curve import CurvePoint, curve_segments, read_curve_frame

from test_capture_analysis import ANALYSIS_PART, PART_CAMERA, multipart_analysis_experiment


def _point(root, number, strain, force, *, scope="test", strain_valid=True, force_valid=True, elapsed=True):
    return CurvePoint.from_result({
        "frame_id": f"{scope}/sim:0/{number}", "sequence": number, "image_part": None,
        "image_path": f"images/{number}.tiff", "image_sha256": "a" * 64,
        "elapsed_receive_s": float(number) if elapsed else None, "timestamps": {"received_utc": {"value": number}},
        "gauges": [{"strain_valid": strain_valid, "engineering_strain": strain,
                    "quality_status": "ACCEPTED_HARD" if strain_valid else "FB_FAILED"}],
        "mechanics": {"force_N": force, "force_support": {"status": "EXACT_TIME_SAMPLE" if force_valid else "LONG_TIME_GAP"}},
    }, root, "sim:0")


def test_live_curves_keep_unloading_order_and_gaps_in_each_axis_mode(tmp_path):
    points = (
        _point(tmp_path, 1, 0., 0.),
        _point(tmp_path, 2, .004, 300.),
        _point(tmp_path, 3, .002, 200.),
        _point(tmp_path, 5, .006, 400.),
        _point(tmp_path, 6, .007, 450., strain_valid=False),
        _point(tmp_path, 7, .008, 500., force_valid=False),
        _point(tmp_path, 8, .009, 600.),
        _point(tmp_path, 9, .010, 700., scope="other"),
    )

    def sequences(mode):
        return [[point.frame.sequence for point, _, _ in line] for line in curve_segments(points, mode)]

    assert sequences("force-strain") == [[1, 2, 3], [5], [8], [9]]
    assert sequences("force-time") == [[1, 2, 3], [5, 6], [8], [9]]
    assert sequences("strain-time") == [[1, 2, 3], [5], [7, 8], [9]]
    assert [x for _, x, _ in next(curve_segments(points, "force-strain"))] == [0., .4, .2]
    assert points[5].elapsed_s == 7 and points[5].force_N is None
    assert list(curve_segments([_point(tmp_path, 1, .001, 20., elapsed=False)], "force-time")) == []


def test_curve_readback_checks_the_selected_image_part_and_raw_bytes(tmp_path):
    experiment = tmp_path / "run"
    primary, _, deformed = multipart_analysis_experiment(experiment)
    gauges = [{"name": "G01", "roi1": [45, 24, 25, 25], "roi2": [45, 85, 25, 25], "strain_mode": "y"}]
    with LiveAnalysis(experiment, PART_CAMERA, image_part=ANALYSIS_PART, gauges=gauges, latest=False) as follower:
        assert follower.poll()["sequence"] == 1
        row = follower.poll()
        assert row["sequence"] == 2 and row["image_part"] == ANALYSIS_PART
    point = CurvePoint.from_result(row, experiment, PART_CAMERA)
    image = read_curve_frame(point.frame)
    np.testing.assert_array_equal(image.pixels, deformed, strict=True)
    assert not np.array_equal(image.pixels, primary)
    assert image.significant_bits == 12
    with pytest.raises(CaptureError) as error:
        read_curve_frame(replace(point.frame, image_sha256="a" * 64))
    assert error.value.code == "FRAME_REVIEW_CHANGED"

    other_experiment = tmp_path / "other"
    multipart_analysis_experiment(other_experiment)
    with pytest.raises(CaptureError) as error:
        read_curve_frame(replace(point.frame, experiment=other_experiment))
    assert error.value.code == "FRAME_REVIEW_CHANGED"

    with ExperimentReader(experiment) as reader:
        image_path = reader.image_path({"path": point.frame.image_path})
    image_path.write_bytes(b"replaced image from test")
    with pytest.raises(CaptureError) as error:
        read_curve_frame(point.frame)
    assert error.value.code == "CAPTURE_IMAGE_CHANGED"
