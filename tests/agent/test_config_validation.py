"""Strictly revalidate persisted project configuration at every API boundary."""

import json

import cv2
import numpy as np
import pytest

from straintrace_agent import AgentError, init_project, inspect_project
from straintrace_agent.api import _read_config


def _project(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image = np.random.default_rng(7).integers(0, 256, (80, 80), dtype=np.uint8)
    assert cv2.imwrite(str(images / "frame_0001.png"), image)
    loads = tmp_path / "loads.csv"
    loads.write_text("time_s,force_N,displacement_mm\n0,10,0\n1,20,0.1\n", encoding="utf-8")
    project = tmp_path / "sample"
    init_project(
        project, images, frame_interval_s=0.5, loads_csv=loads,
        load_time_column="time_s", load_force_column="force_N",
        displacement_column="displacement_mm", displacement_source="crosshead",
        clock_confirmed=True, load_time_offset_s=0, max_gap_s=1,
        area_mm2=2, machine_gauge_length_mm=10,
    )
    return project


@pytest.mark.parametrize(
    ("mutate", "error_code"),
    [
        (lambda cfg: cfg["loads"].__setitem__("area_mm2", -2), "CONFIG_VALUE_ERROR"),
        (lambda cfg: cfg["loads"].__setitem__("clock_confirmed", "false"), "CONFIG_TYPE_ERROR"),
        (lambda cfg: cfg.__setitem__("roi_override_px", [12, 12, -1, 40]), "CONFIG_VALUE_ERROR"),
        (lambda cfg: cfg["loads"].__setitem__("unexpected", 1), "CONFIG_UNKNOWN_FIELD"),
        (lambda cfg: cfg["loads"].pop("clock_confirmed"), "CONFIG_REQUIRED_FIELD"),
        (lambda cfg: cfg.pop("mask_sha256"), "CONFIG_REQUIRED_FIELD"),
    ],
)
def test_persisted_nested_config_is_revalidated(tmp_path, mutate, error_code):
    project = _project(tmp_path)
    path = project / "project.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    mutate(config)
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(AgentError) as caught:
        inspect_project(project)
    assert caught.value.code == error_code


def test_unit_basis_is_rechecked_and_canonical_units_survive_readback(tmp_path):
    project = _project(tmp_path)
    config = _read_config(project)
    assert config["loads"]["time_unit"] == "s"
    assert config["loads"]["time_unit_basis"] == "canonical_column_name"
    assert config["loads"]["force_unit"] == "N"
    assert config["loads"]["force_unit_basis"] == "canonical_column_name"
    assert config["loads"]["displacement_unit"] == "mm"
    assert config["loads"]["displacement_unit_basis"] == "canonical_column_name"

    project_file = project / "project.json"
    stored = json.loads(project_file.read_text(encoding="utf-8"))
    stored["loads"]["time_unit_basis"] = "guessed"
    project_file.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(AgentError) as caught:
        _read_config(project)
    assert caught.value.code == "CONFIG_VALUE_ERROR"

    stored["loads"]["time_unit_basis"] = "canonical_column_name"
    stored["loads"]["time_unit"] = "ms"
    project_file.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(AgentError) as mismatch:
        _read_config(project)
    assert mismatch.value.code == "CONFIG_UNIT_BASIS_MISMATCH"


def test_custom_mechanics_column_names_require_explicit_units(tmp_path):
    existing = _project(tmp_path)
    custom_loads = tmp_path / "custom-loads.csv"
    custom_loads.write_text("Time,Force\n0,10\n1,20\n", encoding="utf-8")
    image_dir = existing.parent / "images"
    common = {
        "loads_csv": custom_loads, "load_time_column": "Time", "load_force_column": "Force",
        "load_time_offset_s": 0.0, "max_gap_s": 1.0, "area_mm2": 2,
    }

    with pytest.raises(AgentError) as time_error:
        init_project(tmp_path / "missing-time-unit", image_dir, frame_interval_s=0.5, **common)
    assert time_error.value.code == "LOAD_TIME_UNIT"

    with pytest.raises(AgentError) as force_error:
        init_project(tmp_path / "missing-force-unit", image_dir, frame_interval_s=0.5, load_time_unit="s", **common)
    assert force_error.value.code == "FORCE_UNIT"
