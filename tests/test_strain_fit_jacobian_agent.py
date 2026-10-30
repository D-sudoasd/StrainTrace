import numpy as np

from straintrace_agent.fields import jacobian_determinant, principal_strains


def test_strain_fit_jacobian_takes_precedence_over_subset_warp():
    x, y = np.meshgrid(np.arange(3.0), np.arange(3.0))
    field = {
        "X": x, "Y": y, "valid": np.ones(9, dtype=bool),
        "strain_valid": np.ones(9, dtype=bool),
        "Exx": np.zeros(9), "Eyy": np.zeros(9), "Exy": np.zeros(9),
        "P": np.zeros((3, 3, 6)),
        "deformation_jacobian": np.full(9, -1.0),
    }
    np.testing.assert_array_equal(jacobian_determinant(field), -np.ones((3, 3)))
    assert not principal_strains(field)["valid"].any()

    # A valid strain-fit mapping does not erase a contradictory folded subset.
    field["deformation_jacobian"] = np.ones(9)
    field["P"][..., 1] = -2.0
    np.testing.assert_array_equal(jacobian_determinant(field), np.ones((3, 3)))
    assert not principal_strains(field)["valid"].any()
