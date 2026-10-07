"""Scientific and provenance contracts for the canonical benchmark."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest

import ezdic_benchmark as facade
import ezdic_core


RUNNER = importlib.import_module("benchmarks.run_benchmark")
CASES = importlib.import_module("benchmarks.synthetic_cases")


def test_facade_delegates_to_migrated_case_and_report_contract() -> None:
    first = facade.locked_cases_hash()
    assert first == facade.locked_cases_hash()
    document = facade.locked_cases_document()
    assert document["version"] == "ezdic-benchmark-cases-v4"
    assert [case["case_id"] for case in document["cases"]] == [
        "small_translation",
        "large_translation",
        "small_affine_strain",
        "near_1d_periodic",
    ]
    assert document["cases"][0]["translation"] == [2.3, -1.2]
    assert document["cases"][1]["translation"] == [28.0, -18.0]
    assert document["thresholds"]["small_translation"]["rmse_px_max"] == 0.001
    assert document["quality_contract"]["corruption_panel"]["version"] == "image_corruption_panel_v1"


def test_natural_panel_report_has_strict_baseline_gates_and_mixed_labels(tmp_path: Path) -> None:
    report = facade.run_benchmark(tmp_path)
    assert report["report_version"] == "ezdic-benchmark-report-v6"
    assert report["overall_pass"] is True
    assert report["exit_code"] == 0
    quality = report["quality_error"]
    assert quality["point_count"] > 0
    assert quality["good_label_count"] > 0
    assert quality["ranking_bad_label_count"] >= 2
    assert quality["roc_auc"] >= 0.90
    assert quality["threshold_status"] == "NOT_CALIBRATED"
    assert quality["quality_threshold_evaluated"] is False
    assert quality["quality_threshold_pass"] is None
    assert quality["false_accept_count"] == 0
    assert quality["bad_label_count"] == 0
    assert quality["false_accept_rate"] is None
    assert quality["false_reject_count"] == 0
    assert quality["false_reject_rate"] == pytest.approx(0.0)
    assert quality["ranking_bad_label_count"] == 16
    assert quality["ranking_rejected_bad_count"] == 16
    assert quality["corruption_row_count"] == 4
    cases = {case["case_id"]: case for case in report["cases"]}
    assert cases["near_1d_periodic"]["status"] == "REJECTED"
    assert cases["near_1d_periodic"]["failure_code"] == "AMBIGUOUS_TEXTURE"
    assert cases["near_1d_periodic"]["solver_calls"] == 0
    assert cases["near_1d_periodic"]["texture_preflight"]["metrics"]["metrics_version"]
    assert cases["small_translation"]["metrics"]["rmse_px"] <= 0.05
    assert cases["small_translation"]["metrics"]["p95_error_px"] <= 0.10
    assert cases["large_translation"]["metrics"]["rmse_px"] <= 0.05
    assert cases["large_translation"]["metrics"]["p95_error_px"] <= 0.10
    assert cases["small_affine_strain"]["metrics"]["strain_component_abs_error_max"] <= 5e-4
    assert cases["small_affine_strain"]["metrics"]["strain_consistency_abs_error_max"] <= 5e-4
    rows = (tmp_path / "benchmark_report.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == report["quality_error"]["all_row_count"] + 1


def test_image_corruption_never_overwrites_solver_diagnostics() -> None:
    source = (CASES.__file__ or "")
    runner_source = RUNNER.__file__ or ""
    assert "diagnostic_override" not in Path(source).read_text(encoding="utf-8")
    assert "diagnostic_override" not in Path(runner_source).read_text(encoding="utf-8")
    fixture = CASES.make_case(CASES.LOCKED_CASE_DOCUMENT["cases"][0])
    variant = CASES.LOCKED_CASE_DOCUMENT["quality_contract"]["corruption_panel"]["variants"][0]
    corrupted, panel = CASES.apply_image_corruption(fixture, CASES.LOCKED_CASE_DOCUMENT["cases"][0], variant)
    assert panel["applied_to"] == "deformed_image_only"
    assert np.array_equal(fixture["reference"], corrupted["reference"])
    assert np.array_equal(fixture["oracle_u"], corrupted["oracle_u"])
    assert not np.array_equal(fixture["deformed"], corrupted["deformed"])


def test_error_label_uses_raw_displacement_even_when_point_is_invalid() -> None:
    case = CASES.LOCKED_CASE_DOCUMENT["cases"][0]
    fixture = CASES.make_case(case)
    field = dict(
        ezdic_core.run_2d_dic(
            fixture["reference"],
            fixture["deformed"],
            CASES.ROI,
            subset_size=CASES.SUBSET_SIZE,
            step=CASES.STEP,
            search_radius=8,
            pyramid_levels=1,
            zncc_min=0.75,
            strain_window=CASES.STRAIN_WINDOW,
            smooth_sigma=0.0,
        )
    )
    field["valid"] = np.asarray(field["valid"], dtype=bool).copy()
    field["valid"][0] = False
    field["u_raw"] = np.asarray(field["u_raw"], dtype=float).copy()
    field["v_raw"] = np.asarray(field["v_raw"], dtype=float).copy()
    field["u_raw"][0] = fixture["oracle_u"][0] + 0.3
    field["v_raw"][0] = fixture["oracle_v"][0]
    observation, rows = RUNNER._observation(
        case,
        fixture,
        field,
        run_id="raw-invalid-probe",
        panel=None,
        thresholds=CASES.LOCKED_CASE_DOCUMENT["thresholds"][case["case_id"]],
        quality_contract=CASES.LOCKED_CASE_DOCUMENT["quality_contract"],
    )
    assert observation["point_count"] == 81
    assert rows[0]["accepted"] is False
    assert rows[0]["error_px"] == pytest.approx(0.3)
    assert rows[0]["quality_label_good"] is False


def test_peak_ratio_direction_and_ties_are_explicit() -> None:
    best, direction = RUNNER._canonical_peak_ratio(np.asarray([1.0, 2.0]), direction="best_over_second")
    assert direction == "best_over_second"
    assert np.array_equal(best, [1.0, 2.0])
    inverted, direction = RUNNER._canonical_peak_ratio(np.asarray([1.0, 0.5]), direction="second_over_best")
    assert direction == "best_over_second_inverted"
    assert np.allclose(inverted, [1.0, 2.0])
    tied, _ = RUNNER._canonical_peak_ratio(np.asarray([1.0]), direction="second_over_best")
    assert tied[0] == 1.0
    with pytest.raises(RUNNER.BenchmarkGateError):
        RUNNER._canonical_peak_ratio(np.asarray([0.5]), direction="unsupported")


def test_generic_ambiguity_text_is_not_machine_evidence() -> None:
    assert RUNNER._machine_ambiguity_code("checkerboard processing failed") is False
    assert RUNNER._machine_ambiguity_code("AMBIGUOUS_TEXTURE") is True
    assert RUNNER._ambiguity_result_passes({"code": "AMBIGUOUS_TEXTURE", "valid": np.zeros(81, dtype=bool)}) is True
    assert RUNNER._ambiguity_result_passes({"code": "AMBIGUOUS_TEXTURE", "valid": np.ones(81, dtype=bool)}) is False
    assert RUNNER._ambiguity_result_passes({"valid": np.zeros(81, dtype=bool)}) is False


def test_generic_texture_exception_cannot_pass_near_1d_case(tmp_path: Path) -> None:
    class GenericTextureFailureCore:
        __file__ = ezdic_core.__file__
        generate_synthetic_speckle = staticmethod(ezdic_core.generate_synthetic_speckle)
        warp_image_translation = staticmethod(ezdic_core.warp_image_translation)
        warp_image_deformation_gradient = staticmethod(ezdic_core.warp_image_deformation_gradient)

        @staticmethod
        def require_texture(*_args, **_kwargs):
            raise ValueError("checkerboard processing failed")

    report = facade.run_benchmark(tmp_path, core_module=GenericTextureFailureCore())
    near = next(case for case in report["cases"] if case["case_id"] == "near_1d_periodic")
    assert near["failure_code"] == "TEXTURE_PREFLIGHT_ERROR"
    assert near["status"] == "FAIL"
    assert report["overall_pass"] is False


def test_near_1d_calls_actual_core_preflight_before_solver(tmp_path: Path) -> None:
    calls: list[str] = []

    class SpyCore:
        __file__ = ezdic_core.__file__
        generate_synthetic_speckle = staticmethod(ezdic_core.generate_synthetic_speckle)
        warp_image_translation = staticmethod(ezdic_core.warp_image_translation)
        warp_image_deformation_gradient = staticmethod(ezdic_core.warp_image_deformation_gradient)

        @staticmethod
        def require_texture(*args, **kwargs):
            calls.append("preflight")
            return ezdic_core.require_texture(*args, **kwargs)

        @staticmethod
        def run_2d_dic(*args, **kwargs):
            calls.append("solver")
            return ezdic_core.run_2d_dic(*args, **kwargs)

    report = facade.run_benchmark(tmp_path, core_module=SpyCore())
    assert report["overall_pass"] is True
    assert calls.count("preflight") == 4
    assert calls.count("solver") == 7
    assert calls[-1] == "preflight" or "solver" not in calls[calls.index("preflight") :]


def test_degraded_quality_or_auc_cannot_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "_quality_score_at", lambda *_args, **_kwargs: None)
    report = facade.run_benchmark(tmp_path)
    assert report["overall_pass"] is False
    assert report["exit_code"] != 0
    assert report["gate_summary"]["quality_ranking_pass"] is False


def test_missing_target_rows_fail_closed(tmp_path: Path) -> None:
    class ShortCore:
        __file__ = ezdic_core.__file__
        generate_synthetic_speckle = staticmethod(ezdic_core.generate_synthetic_speckle)
        warp_image_translation = staticmethod(ezdic_core.warp_image_translation)
        warp_image_deformation_gradient = staticmethod(ezdic_core.warp_image_deformation_gradient)
        require_texture = staticmethod(ezdic_core.require_texture)
        calls = 0

        @classmethod
        def run_2d_dic(cls, *args, **kwargs):
            result = dict(ezdic_core.run_2d_dic(*args, **kwargs))
            cls.calls += 1
            if cls.calls == 1:
                for key in ("x", "y", "u_raw", "v_raw", "valid", "strain_valid"):
                    result[key] = np.asarray(result[key]).reshape(-1)[:-1]
            return result

    report = facade.run_benchmark(tmp_path, core_module=ShortCore())
    assert report["overall_pass"] is False
    assert report["gate_summary"]["csv_rows"] is False


def test_frozen_hash_resolution_covers_every_benchmark_input(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "_MEI"
    sources = bundle / "sources"
    package_sources = sources / "benchmarks"
    package_sources.mkdir(parents=True)
    (bundle / "benchmarks").mkdir()
    payloads = {
        sources / "ezdic_benchmark.py": b"facade",
        sources / "ezdic_core.py": b"core",
        sources / "ezdic_cli.py": b"cli",
        package_sources / "run_benchmark.py": b"runner",
        package_sources / "synthetic_cases.py": b"synthetic",
        bundle / "benchmarks" / "cases_v1.json": Path("benchmarks/cases_v1.json").read_bytes(),
    }
    for path, content in payloads.items():
        path.write_bytes(content)
    monkeypatch.setattr(RUNNER.sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(RUNNER, "__file__", str(bundle / "archive.pyz" / "run_benchmark.py"))
    monkeypatch.setattr(CASES, "__file__", str(bundle / "archive.pyz" / "synthetic_cases.py"))
    monkeypatch.setattr(facade, "__file__", str(bundle / "archive.pyz" / "ezdic_benchmark.py"))
    monkeypatch.setattr(ezdic_core, "__file__", str(bundle / "archive.pyz" / "ezdic_core.py"))
    import ezdic_cli

    monkeypatch.setattr(ezdic_cli, "__file__", str(bundle / "archive.pyz" / "ezdic_cli.py"))
    provenance = RUNNER._code_provenance(ezdic_core, bundle / "benchmarks" / "cases_v1.json")
    assert provenance["benchmark_source_sha256"] == hashlib.sha256(b"facade").hexdigest()
    assert provenance["core_source_sha256"] == hashlib.sha256(b"core").hexdigest()
    assert provenance["cli_source_sha256"] == hashlib.sha256(b"cli").hexdigest()
    assert provenance["benchmark_runner_source_sha256"] == hashlib.sha256(b"runner").hexdigest()
    assert provenance["synthetic_cases_source_sha256"] == hashlib.sha256(b"synthetic").hexdigest()
    assert provenance["cases_json_sha256"] == hashlib.sha256(payloads[bundle / "benchmarks" / "cases_v1.json"]).hexdigest()
