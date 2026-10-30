"""Keep the old uint8-wrap corruption as a separate known-truth stress case."""

import numpy as np

import ezdic_core as core


def test_wrapped_png_corruption_remains_a_quality_stress_case():
    reference_float = core.generate_synthetic_speckle(128, 128, seed=17)
    reference = reference_float.astype(np.uint8)
    warped_float = core.warp_image_translation(reference_float, .8, -.25)
    # Intentionally recreate the former broken fixture: direct cast wraps
    # cubic overshoot rather than saturating it to the 8-bit sensor range.
    wrapped = warped_float.astype(np.uint8)
    saturated = np.clip(warped_float, 0, 255).astype(np.uint8)
    assert np.count_nonzero(wrapped != saturated) > 0
    bounds = core.compute_reference_normalization(reference)
    field = core.run_2d_dic(
        core.normalize_dic_float(reference, bounds),
        core.normalize_dic_float(wrapped, bounds),
        (20, 20, 88, 88), subset_size=15, step=12, search_radius=12,
        strain_window=5, zncc_min=.65, pyramid_levels=2, pyramid_scale=.5,
    )
    valid = np.asarray(field["valid"], dtype=bool)
    assert valid.any()
    assert np.isnan(np.asarray(field["u"])[~valid]).all()
    assert np.isnan(np.asarray(field["v"])[~valid]).all()
    error = np.hypot(np.asarray(field["u_raw"])[valid] - .8,
                     np.asarray(field["v_raw"])[valid] + .25)
    assert np.isfinite(error).all()
    quality = field["quality_summary"]
    if quality["correlation_valid_fraction"] < core.DEFAULT_MIN_CORRELATION_VALID_FRACTION:
        assert not quality["scientific_ok"]
        assert "correlation_valid_fraction_below_threshold" in quality["scientific_reasons"]
    elif quality["scientific_ok"]:
        # If a future version passes the ordinary numeric gate, the known
        # displacement prevents the stress test from declaring success solely
        # from field coverage. This 0.5 px bound is an illustrative fixture
        # tolerance, not a calibrated experimental uncertainty.
        assert np.max(error) <= .5
