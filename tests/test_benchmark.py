"""Executable regression coverage for the migrated quality benchmark."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

import ezdic_core


RUNNER = importlib.import_module("benchmarks.run_benchmark")
CASES = importlib.import_module("benchmarks.synthetic_cases")


def test_locked_benchmark_emits_strict_metrics_and_hash_linked_csv(tmp_path: Path) -> None:
    report = RUNNER.run_benchmark(Path("benchmarks/cases_v1.json"), tmp_path)
    assert report["overall_pass"] is True
    assert report["exit_code"] == 0
    assert report["report_version"] == "ezdic-benchmark-report-v6"
    assert report["cases_version"] == "ezdic-benchmark-cases-v4"
    quality = report["quality_error"]
    assert quality["good_label_count"] > 0
    assert quality["ranking_bad_label_count"] >= 2
    assert quality["roc_auc"] >= 0.90
    assert quality["error_tolerance_px"] == pytest.approx(0.25)
    assert quality["quality_threshold_evaluated"] is False
    assert quality["quality_threshold_pass"] is None
    assert quality["threshold_status"] == "NOT_CALIBRATED"
    assert quality["false_accept_rate"] is None
    assert quality["false_accept_count"] == 0
    assert quality["bad_label_count"] == 0
    assert quality["false_reject_rate"] == pytest.approx(0.0)
    assert quality["ranking_bad_label_count"] == 16
    assert quality["ranking_rejected_bad_count"] == 16
    assert quality["corruption_row_count"] == 4
    cases = {case["case_id"]: case for case in report["cases"]}
    assert cases["small_translation"]["metrics"]["valid_fraction"] >= 0.95
    assert cases["small_translation"]["metrics"]["rmse_px"] <= 0.05
    assert cases["small_translation"]["metrics"]["p95_error_px"] <= 0.10
    assert cases["small_translation"]["metrics"]["max_error_px"] <= 0.15
    assert cases["large_translation"]["metrics"]["valid_fraction"] >= 0.95
    assert cases["large_translation"]["metrics"]["rmse_px"] <= 0.05
    assert cases["large_translation"]["metrics"]["p95_error_px"] <= 0.10
    assert cases["large_translation"]["metrics"]["max_error_px"] <= 0.15
    affine = cases["small_affine_strain"]
    assert affine["metrics"]["rmse_px"] <= 0.05
    assert affine["metrics"]["strain_component_abs_error_max"] <= 5e-4
    assert affine["metrics"]["strain_consistency_abs_error_max"] <= 5e-4
    assert affine["metrics"]["strain_valid_fraction"] >= 0.80
    near = cases["near_1d_periodic"]
    assert near["status"] == "REJECTED"
    assert near["failure_code"] == "AMBIGUOUS_TEXTURE"
    assert near["solver_calls"] == 0
    csv_path = tmp_path / "benchmark_report.csv"
    assert csv_path.is_file()
    assert report["artifacts"]["benchmark_report_csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == quality["all_row_count"] + 1


def test_missing_quality_diagnostic_fails_benchmark(tmp_path: Path) -> None:
    class DegradedCore:
        __file__ = ezdic_core.__file__
        generate_synthetic_speckle = staticmethod(ezdic_core.generate_synthetic_speckle)
        warp_image_translation = staticmethod(ezdic_core.warp_image_translation)
        warp_image_deformation_gradient = staticmethod(ezdic_core.warp_image_deformation_gradient)
        require_texture = staticmethod(ezdic_core.require_texture)

        @staticmethod
        def run_2d_dic(*args, **kwargs):
            result = dict(ezdic_core.run_2d_dic(*args, **kwargs))
            result.pop("residual_rms", None)
            return result

    report = RUNNER.run_benchmark(Path("benchmarks/cases_v1.json"), tmp_path, core_module=DegradedCore)
    assert report["overall_pass"] is False
    assert report["exit_code"] != 0
    assert report["gate_summary"]["numeric_baseline_pass"] is False
    assert report["gate_summary"]["quality_ranking_pass"] is False
    assert any(case["status"] == "FAIL" for case in report["cases"])


def test_uniform_quality_score_cannot_manufacture_ranking_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "_quality_score_at", lambda *_args, **_kwargs: 0.75)
    report = RUNNER.run_benchmark(Path("benchmarks/cases_v1.json"), tmp_path)
    assert report["quality_error"]["roc_auc"] == pytest.approx(0.5)
    assert report["gate_summary"]["quality_ranking_pass"] is False
    assert report["overall_pass"] is False


def test_mutated_case_threshold_or_missing_case_is_rejected(tmp_path: Path) -> None:
    document = json.loads(Path("benchmarks/cases_v1.json").read_text(encoding="utf-8"))
    document["thresholds"]["small_translation"]["rmse_px_max"] = 99.0
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RUNNER.BenchmarkConfigError):
        RUNNER.run_benchmark(path, tmp_path / "mutated-out")
    document["thresholds"].pop("large_translation")
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RUNNER.BenchmarkConfigError):
        RUNNER.run_benchmark(path, tmp_path / "missing-case-out")


def test_missing_source_hash_fails_closed_in_frozen_mode(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "_MEI"
    bundle.mkdir()
    monkeypatch.setattr(RUNNER.sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(RUNNER, "__file__", str(bundle / "archive.pyz" / "run_benchmark.py"))
    report = RUNNER.run_benchmark(Path("benchmarks/cases_v1.json"), tmp_path / "frozen-out", core_module=ezdic_core)
    assert report["overall_pass"] is False
    assert report["provenance_failure"] == "CODE_PROVENANCE_UNAVAILABLE"
