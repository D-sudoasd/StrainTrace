"""Safe field archives retain measurement diagnostics and export provenance."""

import json

import numpy as np
import pytest

import ezdic_core as core


@pytest.fixture
def masked_field(monkeypatch):
    reference = core.generate_synthetic_speckle(96, 96, seed=19)
    mask = np.ones(reference.shape, dtype=bool)
    mask[:, 48:56] = False
    field = core.run_2d_dic(
        reference, reference, (16, 16, 65, 65), subset_size=15, step=8,
        initialization="local", reject_nonconverged=True, specimen_mask=mask,
    )
    field["provenance"] = {
        "reference_filename": "reference.png", "frame_filename": "target.png", "native_frame": 2,
    }
    field["display_options"] = {"coordinate_frame": "deformed", "percent": True}
    monkeypatch.setattr(core, "plot_dic_field_map", lambda _field, path, **_kwargs: path)
    return field


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def test_npz_roundtrip_retains_rejection_diagnostics_and_provenance(tmp_path, masked_field):
    first = core.export_dic_field_outputs(masked_field, tmp_path / "first")
    loaded = _load(first["arrays"])
    assert core.field_quality_summary(masked_field)["invalid_reason_histogram"] == {"MASK_EXCLUDED": 14}
    assert core.field_quality_summary(loaded) == core.field_quality_summary(masked_field)
    for key in ("invalid_reason", "stop_reason", "strain_invalid_reason", "competing_peak_status"):
        assert loaded[key].dtype.kind == "U"
        np.testing.assert_array_equal(loaded[key], masked_field[key])
    first_metadata = json.loads(str(loaded["metadata_json"]))
    assert first_metadata["parameters"]["solver"] == masked_field["solver"]
    assert first_metadata["parameters"]["reject_nonconverged"] is True

    second = core.export_dic_field_outputs(loaded, tmp_path / "second")
    reloaded = _load(second["arrays"])
    assert json.loads(str(reloaded["metadata_json"])) == first_metadata
    assert first["csv"].read_bytes() == second["csv"].read_bytes()
    assert first["txt"].read_bytes() == second["txt"].read_bytes()
    assert first["parameters"].read_bytes() == second["parameters"].read_bytes()
    for key in ("u", "v", "Eyy", "valid", "strain_valid", "invalid_reason"):
        np.testing.assert_array_equal(reloaded[key], masked_field[key])


def test_reexport_preserves_explicit_fields_and_experimental_flag(tmp_path, masked_field):
    first = core.export_dic_field_outputs(masked_field, tmp_path / "first")
    loaded = _load(first["arrays"])
    loaded["subset_size"] = 31
    loaded["provenance"] = {"native_frame": 9}
    loaded["display_options"] = {}
    # Metadata reflects the earlier default mode; the current flag takes priority.
    loaded["experimental_full_search"] = np.asarray(True)
    loaded["quality_summary"] = core.field_quality_summary(loaded)
    second = core.export_dic_field_outputs(loaded, tmp_path / "second")
    reloaded = _load(second["arrays"])
    metadata = json.loads(str(reloaded["metadata_json"]))
    assert metadata["parameters"]["subset_size"] == 31
    assert metadata["provenance"] == {
        "native_frame": 9, "experimental_full_search": True, "scientific_ok": False,
        "quality_scientific_reasons": ["experimental_search_unvalidated"],
    }
    assert metadata["display"] == {}
    assert metadata["parameters"]["experimental_full_search"] is True
    assert bool(reloaded["experimental_full_search"]) is True
    assert core.field_quality_summary(reloaded)["scientific_ok"] is False


def test_reexport_accepts_older_minimal_metadata(tmp_path, masked_field):
    first = core.export_dic_field_outputs(masked_field, tmp_path / "first")
    loaded = _load(first["arrays"])
    loaded["metadata_json"] = np.asarray(json.dumps({"roi": [16, 16, 65, 65]}))
    second = core.export_dic_field_outputs(loaded, tmp_path / "second")
    metadata = json.loads(str(_load(second["arrays"])["metadata_json"]))
    assert metadata["roi"] == [16, 16, 65, 65]


@pytest.mark.parametrize("top_level_flag", [False, None])
def test_saved_experimental_mode_cannot_be_downgraded_on_reexport(tmp_path, masked_field, top_level_flag):
    masked_field["experimental_full_search"] = True
    masked_field["quality_summary"] = core.field_quality_summary(masked_field)
    first = core.export_dic_field_outputs(masked_field, tmp_path / "first")
    loaded = _load(first["arrays"])
    if top_level_flag is None:
        del loaded["experimental_full_search"]
    else:
        loaded["experimental_full_search"] = np.asarray(top_level_flag)
    # Even an explicit stale summary cannot clear recorded research status.
    loaded["quality_summary"] = {"scientific_ok": True, "scientific_reasons": []}
    second = core.export_dic_field_outputs(loaded, tmp_path / "second")
    reloaded = _load(second["arrays"])
    metadata = json.loads(str(reloaded["metadata_json"]))
    assert bool(reloaded["experimental_full_search"]) is True
    assert metadata["parameters"]["experimental_full_search"] is True
    assert metadata["quality_summary"]["scientific_ok"] is False
    assert "experimental_search_unvalidated" in metadata["quality_summary"]["scientific_reasons"]
    assert core.field_quality_summary(reloaded)["scientific_ok"] is False


def test_experimental_metadata_overrides_stale_provenance_pass_in_parameters(tmp_path, masked_field):
    first = core.export_dic_field_outputs(masked_field, tmp_path / "first")
    loaded = _load(first["arrays"])
    metadata = json.loads(str(loaded["metadata_json"]))
    metadata["parameters"]["experimental_full_search"] = True
    metadata["provenance"].update({"experimental_full_search": False, "scientific_ok": True})
    loaded["metadata_json"] = np.asarray(json.dumps(metadata))
    loaded["experimental_full_search"] = np.asarray(False)
    loaded["quality_summary"] = {"scientific_ok": True, "scientific_reasons": []}
    direct_parameters = core.write_dic_field_parameters(loaded, tmp_path / "direct_parameters.txt")
    second = core.export_dic_field_outputs(loaded, tmp_path / "second")
    reloaded = _load(second["arrays"])
    saved = json.loads(str(reloaded["metadata_json"]))
    assert saved["provenance"]["experimental_full_search"] is True
    assert saved["provenance"]["scientific_ok"] is False
    assert saved["quality_summary"]["scientific_ok"] is False
    assert core.dic_field_to_dataframe(reloaded)["experimental_full_search"].all()
    assert direct_parameters.read_bytes() == second["parameters"].read_bytes()
    text = second["parameters"].read_text(encoding="utf-8")
    assert "experimental_full_search = True\n" in text
    assert "scientific_ok = False\n" in text
    assert "experimental_full_search = False\n" not in text
    assert "scientific_ok = True\n" not in text


@pytest.mark.parametrize("metadata", ["{broken", "[]", '{"parameters": []}'])
def test_corrupt_archive_metadata_is_not_silently_discarded(tmp_path, masked_field, metadata):
    masked_field["metadata_json"] = np.asarray(metadata)
    with pytest.raises(ValueError, match="metadata_json"):
        core.export_dic_field_outputs(masked_field, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
