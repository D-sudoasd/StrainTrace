"""Agent output contract checks independent of filesystem case semantics."""

import json

import ezdic_core

from straintrace_agent.api import _build_outputs, _component_file_slug


def test_component_artifacts_have_unique_casefolded_names():
    components = (*ezdic_core.DIC_FIELD_COMPONENTS, "principal_max", "PRINCIPAL_MAX")
    slugs = [_component_file_slug(component) for component in components]

    # NTFS and common macOS filesystems treat these names as equivalent.
    assert len({slug.casefold() for slug in slugs}) == len(slugs)
    assert _component_file_slug("Exx") == ezdic_core.DIC_COMPONENT_FILE_SUFFIXES["Exx"]
    assert _component_file_slug("exx") == ezdic_core.DIC_COMPONENT_FILE_SUFFIXES["exx"]
    assert _component_file_slug("Exx").casefold() != _component_file_slug("exx").casefold()

    for prefix, extension in (("roi_", ".csv"), ("roi_", "_summary.json"), ("path_", ".csv"), ("position_time_", ".png")):
        filenames = [f"{prefix}{slug}{extension}" for slug in slugs]
        assert len({name.casefold() for name in filenames}) == len(filenames)

    frame_map_names = [
        f"frame_native_{native:08d}_{_component_file_slug(component)}.png"
        for native in (1, 2, 4)
        for component in ezdic_core.DIC_FIELD_COMPONENTS
    ]
    assert len(frame_map_names) == 27
    assert len({name.casefold() for name in frame_map_names}) == 27


def test_quality_gate_is_not_reported_as_scientific_acceptance(tmp_path):
    root = tmp_path / "project"
    run_root = root / "runs" / "run-test"
    (run_root / "checkpoints").mkdir(parents=True)
    records = [{"native_frame": 1, "path": "frame_0001.tif", "sha256": "source-hash", "time_s": 0.0}]
    rows = [{"frame_position_1based": 1, "native_frame": 1, "time_s": 0.0, "status": "complete", "quality_status": "pass", "scientific_ok": True}]

    result = _build_outputs(
        root, run_root, {"project_id": "test", "loads": None, "mask_sha256": None},
        {"reference_native_frame": 1, "analysis_rectangle_candidate_px": [10, 10, 40, 40], "selected_parameters": {}},
        records, rows, {}, "run-fingerprint", "input-fingerprint", {}, field_csv=False, plots=False,
    )

    quality = json.loads((run_root / "quality.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    report = (run_root / "report.md").read_text(encoding="utf-8")
    assert result["quality"]["scientific_ok"] is True  # compatibility alias
    assert quality["numerical_quality_passed"] is True
    assert quality["scientific_acceptance"] == "not_assessed"
    assert quality["experimental_uncertainty_calibrated"] is False
    assert manifest["scientific_acceptance"] == "not_assessed"
    assert manifest["experimental_uncertainty_calibrated"] is False
    assert "Core numerical quality gates: `PASS`" in report
    assert "Scientific acceptance: not assessed" in report
    assert "Whole-run scientific status" not in report
