import cv2
import numpy as np
import pytest

from straintrace_capture.cameras.base import Image
from straintrace_capture.preview import FocusComparison, image_quality


def test_inspection_region_changes_exposure_histogram_and_response():
    pixels = np.full((100, 128), 4095, dtype=np.uint16)
    patch = np.random.default_rng(92).integers(750, 2600, size=(48, 44), dtype=np.uint16)
    pixels[19:67, 11:55] = patch
    image = Image(pixels, "Mono12", 12, "mono", 1, 1)
    whole = image_quality(image)
    selected = image_quality(image, (11, 19, 44, 48))
    direct = image_quality(Image(patch, "Mono12", 12, "mono", 1, 1))

    assert whole["bright_fraction"] > .8
    assert selected["bright_fraction"] == selected["dark_fraction"] == 0.
    assert sum(selected["histogram"]) == patch.size
    for key in ("histogram", "focus_laplacian_variance", "percentiles_p5_p50_p95"):
        assert selected[key] == direct[key]
    assert selected["texture"]["contrast_p95_p5"] > 0.


def test_color_clipping_detects_a_single_channel_but_ignores_alpha():
    pixels = np.full((32, 48, 4), 400, dtype=np.uint16)
    pixels[..., 3] = 4095
    pixels[:, :12, 0] = 4095
    result = image_quality(Image(pixels, "RGBa12", 12, "rgba", 1, 1))

    assert result["bright_fraction"] == .25
    assert result["histogram"][-1] == 0


@pytest.mark.parametrize("pattern,green_column", [("RG", 1), ("BG", 1), ("GB", 0), ("GR", 0)])
def test_bayer_filter_pattern_does_not_count_as_focus_detail(pattern, green_column):
    pixels = np.full((96, 100), 2800, dtype=np.uint16)
    pixels[::2, green_column::2] = 1200
    pixels[1::2, (1 - green_column)::2] = 1490
    pixels[1::2, green_column::2] = 400
    image = Image(pixels, f"Bayer{pattern}12p", 12, "bayer", 1, 1)

    assert image_quality(image)["focus_laplacian_variance"] == 0.
    selected = image_quality(image, (5, 7, 66, 56))
    assert selected["focus_laplacian_variance"] == 0.
    assert selected["texture"]["contrast_p95_p5"] == 0.
    assert len([count for count in selected["histogram"] if count]) == 4


def test_focus_response_ranks_the_same_target_at_different_blur():
    target = np.random.default_rng(475).integers(80, 180, size=(128, 160), dtype=np.uint8)
    frames = [target, cv2.GaussianBlur(target, (0, 0), 1.2), cv2.GaussianBlur(target, (0, 0), 2.3)]
    responses = [image_quality(Image(frame, "Mono8", 8, "mono", 1, 1))["focus_laplacian_variance"] for frame in frames]
    assert responses[0] > responses[1] > responses[2] > 0.


def test_large_region_keeps_exposure_diagnostics_and_tiny_region_has_no_focus():
    pixels = np.full((1100, 1040), 4030, dtype=np.uint16)
    image = Image(pixels, "Mono12", 12, "mono", 1, 1)
    large = image_quality(image, (0, 0, 1020, 1020))
    assert large["texture"] is None
    assert large["texture_error"] == "INSPECTION_ROI_TOO_LARGE"
    assert large["percentiles_p5_p50_p95"] == [4030., 4030., 4030.]
    assert sum(large["histogram"]) == 510**2

    tiny = image_quality(image, (5, 7, 1, 1))
    assert tiny["percentiles_p5_p50_p95"] == [4030., 4030., 4030.]
    assert tiny["focus_laplacian_variance"] is None


def test_focus_comparison_uses_fresh_samples_and_retains_the_median_peak():
    comparison = FocusComparison(10)
    for receipt, response in ((10, 700.), (11, 8.), (11, 700.), (12, 700.)):
        comparison.update(receipt, response)
    assert comparison.current is None and comparison.highest is None
    comparison.update(13, 9.)
    assert comparison.current == comparison.highest == 9.
    comparison.update(14, 11.)
    assert comparison.current == comparison.highest == 11.
    comparison.update(15, 7.)
    comparison.update(16, 6.)
    comparison.update(15, 700.)
    assert comparison.current == 7. and comparison.highest == 11.
    comparison.update(17, 0.)
    assert not comparison.has_detail and comparison.current is None
    assert comparison.highest == 11.
    comparison.update(18, 4.)
    comparison.update(19, 5.)
    assert comparison.current is None
