"""Regression tests for coverage and error labels in the locked benchmark."""

import csv
import errno
from pathlib import Path

import numpy as np
import pytest

import ezdic_core
from benchmarks import run_benchmark as runner
from benchmarks import synthetic_cases as cases


def _exact_observation_inputs(case_index):
    case = cases.LOCKED_CASE_DOCUMENT["cases"][case_index]
    coordinates = np.asarray(cases.LOCKED_COORDINATES, dtype=float)
    n = len(coordinates)
    oracle_u, oracle_v, oracle_strain = cases._oracle_for_case(case, coordinates)
    field = {
        "x": coordinates[:, 0], "y": coordinates[:, 1],
        "u_raw": oracle_u.copy(), "v_raw": oracle_v.copy(),
        "valid": np.ones(n, dtype=bool), "strain_valid": np.ones(n, dtype=bool),
        "zncc": np.full(n, 0.99), "second_peak_margin": np.full(n, 0.2),
        "best_to_second_peak_ratio": np.full(n, 2.0), "residual_rms": np.zeros(n),
        "hessian_condition_number": np.full(n, 5.0), "iterations": np.full(n, 3),
        "converged": np.ones(n, dtype=bool), "stop_reason": np.full(n, "converged"),
        "strain_type": "green-lagrange", "strain_convention": "tensor Exy",
    }
    for name, value in zip(("Exx", "Eyy", "Exy"), oracle_strain):
        field[name] = np.full(n, value)
    return case, {"oracle_u": oracle_u, "oracle_v": oracle_v}, field


def _observe(case, fixture, field):
    return runner._observation(
        case, fixture, field, run_id="synthetic-probe", panel=None,
        thresholds=cases.LOCKED_CASE_DOCUMENT["thresholds"][case["case_id"]],
        quality_contract=cases.LOCKED_CASE_DOCUMENT["quality_contract"],
    )


def test_missing_accepted_displacements_do_not_inflate_coverage():
    case, fixture, field = _exact_observation_inputs(0)
    field["u_raw"][1:] = np.nan
    field["v_raw"][1:] = np.nan
    observation, rows = _observe(case, fixture, field)
    assert observation["status"] == "FAIL"
    assert observation["observed"]["correlation_flag_fraction"] == 1.0
    assert observation["observed"]["valid_fraction"] == pytest.approx(1 / 81)
    assert observation["observed"]["accepted_without_raw_displacement_count"] == 80
    assert observation["gates"]["displacement_validity_consistent"] is False
    assert rows[1]["accepted"] is True and rows[1]["error_px"] is None


def test_missing_flagged_strains_do_not_inflate_affine_coverage():
    case, fixture, field = _exact_observation_inputs(2)
    for name in ("Exx", "Eyy", "Exy"):
        field[name][1:] = np.nan
    observation, _ = _observe(case, fixture, field)
    assert observation["status"] == "FAIL"
    assert observation["observed"]["strain_flag_fraction"] == 1.0
    assert observation["observed"]["strain_valid_fraction"] == pytest.approx(1 / 81)
    assert observation["observed"]["strain_flag_without_finite_components_count"] == 80
    assert observation["gates"]["strain_validity_consistent"] is False


def test_strain_flag_cannot_bypass_rejected_correlation():
    case, fixture, field = _exact_observation_inputs(2)
    field["valid"][0] = False
    observation, _ = _observe(case, fixture, field)
    assert observation["status"] == "FAIL"
    assert observation["observed"]["strain_flag_without_valid_displacement_count"] == 1
    assert observation["gates"]["strain_validity_consistent"] is False


class _CoreProxy:
    __file__ = ezdic_core.__file__
    require_texture = staticmethod(ezdic_core.require_texture)
    warp_image_translation = staticmethod(ezdic_core.warp_image_translation)


def test_inconsistent_panel_cannot_be_ignored_by_case_gate(tmp_path: Path):
    class PanelCore(_CoreProxy):
        calls = 0

        @classmethod
        def run_2d_dic(cls, *args, **kwargs):
            result = dict(ezdic_core.run_2d_dic(*args, **kwargs))
            cls.calls += 1
            if cls.calls == 2:
                index = np.flatnonzero(result["valid"])[0]
                result["u_raw"] = np.asarray(result["u_raw"]).copy()
                result["u_raw"][index] = np.nan
            return result

    report = runner.run_benchmark(output_dir=tmp_path, core_module=PanelCore)
    case = next(item for item in report["cases"] if item["case_id"] == "small_translation")
    assert case["gates"]["corruption_rows"] is True
    assert case["gates"]["corruption_observations"] is False
    assert case["runs"][1]["gates"]["displacement_validity_consistent"] is False
    assert report["overall_pass"] is False


def test_finite_error_rejection_is_counted_independently_of_ranking_label(tmp_path: Path):
    class RejectGoodCore(_CoreProxy):
        calls = 0

        @classmethod
        def run_2d_dic(cls, *args, **kwargs):
            result = dict(ezdic_core.run_2d_dic(*args, **kwargs))
            cls.calls += 1
            if cls.calls == 1:
                for key in ("valid", "strain_valid"):
                    result[key] = np.asarray(result[key]).copy()
                    result[key][0] = False
            return result

    report = runner.run_benchmark(output_dir=tmp_path, core_module=RejectGoodCore)
    quality = report["quality_error"]
    assert report["overall_pass"] is True
    assert quality["false_reject_count"] == 1
    assert quality["false_reject_rate"] == pytest.approx(1 / quality["good_label_count"])
    assert quality["bad_label_count"] == 0
    assert quality["quality_threshold_evaluated"] is False
    with (tmp_path / "benchmark_report.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["quality_label_good"] == "0"  # locked ranking outcome
    assert row["finite_error_label_good"] == "1"  # independent finite-error label
    assert row["finite_error_quality_false_reject"] == "1"


def test_changing_source_during_run_fails_provenance_gate(tmp_path: Path, monkeypatch):
    original = runner._code_provenance
    samples = []

    def changing_provenance(*args, **kwargs):
        result = original(*args, **kwargs)
        samples.append(result)
        if len(samples) == 2:
            result["core_source_sha256"] = "0" * 64
        return result

    monkeypatch.setattr(runner, "_code_provenance", changing_provenance)
    report = runner.run_benchmark(output_dir=tmp_path)
    assert len(samples) == 2
    assert report["gate_summary"]["numeric_baseline_pass"] is True
    assert report["gate_summary"]["code_provenance"] is False
    assert report["provenance_failure"] == "CODE_PROVENANCE_CHANGED"
    assert report["code_at_start"]["core_source_sha256"] != report["code"]["core_source_sha256"]
    assert report["overall_pass"] is False


@pytest.mark.parametrize("name", ["benchmark_report.csv", "benchmark_report.json"])
def test_existing_benchmark_evidence_is_not_overwritten(tmp_path: Path, name):
    existing = tmp_path / name
    existing.write_bytes(b"existing benchmark evidence\n")
    with pytest.raises(runner.BenchmarkIOError, match="already exists"):
        runner.run_benchmark(output_dir=tmp_path)
    assert existing.read_bytes() == b"existing benchmark evidence\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [name]


@pytest.mark.parametrize("failure", ["json_serialization", "json_publication"])
def test_failed_two_file_write_can_retry_same_directory(tmp_path: Path, monkeypatch, failure):
    original_link = runner.os.link

    def fail_json_write(*args, **kwargs):
        raise TypeError("synthetic JSON serialization failure")

    def fail_json_link(source, target):
        if Path(target).name == "benchmark_report.json":
            raise OSError("synthetic JSON publication failure")
        return original_link(source, target)

    with monkeypatch.context() as patcher:
        if failure == "json_serialization":
            patcher.setattr(runner, "write_json_lf", fail_json_write)
        else:
            patcher.setattr(runner.os, "link", fail_json_link)
        with pytest.raises(runner.BenchmarkIOError):
            runner.run_benchmark(output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
    report = runner.run_benchmark(output_dir=tmp_path)
    assert report["overall_pass"] is True
    assert (tmp_path / "benchmark_report.csv").is_file()
    assert (tmp_path / "benchmark_report.json").is_file()


@pytest.mark.parametrize("change", ["replaced", "modified"])
def test_publication_failure_preserves_concurrent_writer(tmp_path: Path, monkeypatch, change):
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    for name in ("benchmark_report.csv", "benchmark_report.json"):
        (staging / name).write_bytes(b"our generated file")
    original_link = runner.os.link

    def replace_then_fail(source, target):
        if Path(target).name == "benchmark_report.json":
            if change == "replaced":
                (output / "benchmark_report.csv").unlink()
            (output / "benchmark_report.csv").write_bytes(b"concurrent writer")
            raise OSError("synthetic second publication failure")
        return original_link(source, target)

    monkeypatch.setattr(runner.os, "link", replace_then_fail)
    with pytest.raises(runner.BenchmarkIOError, match="publish"):
        runner._publish_reports(staging, output)
    assert (output / "benchmark_report.csv").read_bytes() == b"concurrent writer"
    assert not (output / "benchmark_report.json").exists()


def test_publication_without_hardlink_support_remains_exclusive(tmp_path: Path, monkeypatch):
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    for name in ("benchmark_report.csv", "benchmark_report.json"):
        (staging / name).write_bytes(b"generated file")

    def unsupported_link(source, target):
        if Path(target).name == "benchmark_report.json":
            Path(target).write_bytes(b"concurrent writer")
        raise OSError(errno.ENOTSUP, "hard links unsupported")

    monkeypatch.setattr(runner.os, "link", unsupported_link)
    with pytest.raises(runner.BenchmarkIOError, match="publish"):
        runner._publish_reports(staging, output)
    assert not (output / "benchmark_report.csv").exists()
    assert (output / "benchmark_report.json").read_bytes() == b"concurrent writer"
