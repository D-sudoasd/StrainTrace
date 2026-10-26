"""Independent cross-module checks with analytically sampled image inputs."""
import pytest

from scripts.validate_agent_upgrade import (
    build_experiment, deformation_cases, validate_core_case, validate_masked_support,
    validate_run_tables, validate_spatial_contrast,
)


@pytest.mark.parametrize("name", tuple(deformation_cases()))
def test_continuous_texture_obeys_known_transform_through_dic(name):
    validate_core_case(name)


def test_real_masked_dic_hole_stays_empty_in_path_csv(tmp_path):
    validate_masked_support(tmp_path)


def test_independent_nonuniform_images_distinguish_two_regions(tmp_path):
    validate_spatial_contrast(tmp_path)


def test_known_image_transforms_bind_gauges_regions_and_mechanics(tmp_path):
    from straintrace_agent import init_project, plan_project, run, verify
    options, truth = build_experiment(tmp_path / "experiment")
    project = tmp_path / "project"
    init_project(project, **options)
    plan = plan_project(project)
    assert plan["status"] == "ready", plan
    result = run(project, make_plots=False, field_csv=True)
    assert result["status"] == "partial" and result["failed_frames"] == 1, result
    validate_run_tables(result["run_dir"], plan, truth)
    verified = verify(project)
    assert verified["valid"] is True, verified
    assert verified["numerical_quality_passed"] is False
    assert verified["scientific_acceptance"] == "not_assessed"
