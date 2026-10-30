"""Ground-truth labels must be independent of solver acceptance."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from benchmarks import run_benchmark as runner
from benchmarks import synthetic_cases as cases


def test_candidate_truth_classifies_rejections_and_missing_candidates():
    case = cases.LOCKED_CASE_DOCUMENT["cases"][0]
    fixture = cases.make_case(case)
    coordinates = np.asarray(cases.LOCKED_COORDINATES)
    count = len(coordinates)
    # Controlled diagnostics keep this classification test independent of
    # platform-specific solver acceptance on any one generated image.
    field = {
        "x": coordinates[:, 0], "y": coordinates[:, 1],
        "u_raw": fixture["oracle_u"].copy(), "v_raw": fixture["oracle_v"].copy(),
        "valid": np.ones(count, dtype=bool), "strain_valid": np.ones(count, dtype=bool),
        "zncc": np.full(count, 0.99), "second_peak_margin": np.full(count, 0.3),
        "best_to_second_peak_ratio": np.full(count, 2.0),
        "residual_rms": np.zeros(count), "hessian_condition_number": np.full(count, 10.0),
        "iterations": np.ones(count), "converged": np.ones(count, dtype=bool),
        "stop_reason": np.full(count, "converged", dtype=object),
    }

    # Accepted good/bad, rejected good/bad, and rejected/accepted unknown.
    # Labels use a 0.25 px oracle tolerance regardless of acceptance.
    field["u_raw"][:5] = fixture["oracle_u"][:5] + [0.05, 0.30, 0.05, 0.30, 0.0]
    field["v_raw"][:5] = fixture["oracle_v"][:5]
    field["valid"][2:5] = False
    field["u_raw"][4:6] = np.nan
    _, rows = runner._observation(
        case, fixture, field, run_id="candidate-label-test", panel=None,
        thresholds=cases.LOCKED_CASE_DOCUMENT["thresholds"][case["case_id"]],
        quality_contract=cases.LOCKED_CASE_DOCUMENT["quality_contract"],
    )

    assert [row["candidate_truth_label"] for row in rows[:6]] == [
        "good", "bad", "good", "bad", "unknown", "unknown"
    ]
    assert rows[1]["candidate_false_accept"] is True
    assert rows[2]["candidate_false_reject"] is True
    assert rows[3]["candidate_false_reject"] is False
    assert rows[4]["candidate_false_reject"] is None
    assert rows[5]["candidate_false_accept"] is None
    # Historical fields keep their version-6 definition for compatibility.
    assert rows[2]["quality_label_good"] is False
    assert rows[2]["false_reject"] is False

    summary = runner._candidate_accuracy_summary(rows)
    assert summary["solver_point_count"] == 81
    assert summary["known_candidate_count"] == 79
    assert summary["unknown_candidate_count"] == 2
    assert summary["good_candidate_count"] == 77
    assert summary["bad_candidate_count"] == 2
    assert summary["accepted_bad_count"] == 1
    assert summary["rejected_good_count"] == 1
    assert summary["accepted_unknown_count"] == 1
    assert summary["rejected_unknown_count"] == 1
    assert summary["false_accept_rate"] == pytest.approx(0.5)
    assert summary["false_reject_rate"] == pytest.approx(1 / 77)


def test_empty_truth_class_does_not_report_fabricated_rate():
    rows = [
        {"diagnostic_source": "solver", "accepted": True,
         "candidate_truth_label": "good"},
        {"diagnostic_source": "solver", "accepted": False,
         "candidate_truth_label": "unknown"},
        {"diagnostic_source": "preflight_rejected", "accepted": False,
         "candidate_truth_label": "unknown"},
    ]
    summary = runner._candidate_accuracy_summary(rows)
    assert summary["known_candidate_count"] == 1
    assert summary["unknown_candidate_count"] == 1
    assert summary["preflight_unknown_count"] == 1
    assert summary["bad_candidate_count"] == 0
    assert summary["false_accept_rate"] is None
    assert summary["false_reject_rate"] == 0.0

    with pytest.raises(runner.BenchmarkGateError, match="truth label"):
        runner._candidate_accuracy_summary([
            {"diagnostic_source": "solver", "accepted": True,
             "candidate_truth_label": "not-a-label"}
        ])


def test_locked_benchmark_exposes_candidate_limits_without_changing_gate(tmp_path):
    report = runner.run_benchmark(output_dir=tmp_path)
    assert report["overall_pass"] is True
    assert report["gate_summary"]["quality_ranking_pass"] is True
    assert report["report_version"] == "ezdic-benchmark-report-v6"
    candidate = report["candidate_accuracy"]
    document = cases.LOCKED_CASE_DOCUMENT
    variants = document["quality_contract"]["corruption_panel"]["variants"]
    solver_cases = [case for case in document["cases"] if case["kind"] != "near_1d_periodic"]
    solver_runs = sum(1 + sum(case["case_id"] in variant["case_ids"] for variant in variants)
                      for case in solver_cases)
    assert candidate["solver_point_count"] == solver_runs * len(cases.LOCKED_COORDINATES)
    assert candidate["preflight_unknown_count"] == len(cases.LOCKED_COORDINATES)
    assert candidate["solver_point_count"] == candidate["known_candidate_count"] + candidate["unknown_candidate_count"]
    assert candidate["known_candidate_count"] == candidate["good_candidate_count"] + candidate["bad_candidate_count"]
    assert candidate["good_candidate_count"] == candidate["accepted_good_count"] + candidate["rejected_good_count"]
    assert candidate["bad_candidate_count"] == candidate["accepted_bad_count"] + candidate["rejected_bad_count"]
    assert candidate["unknown_candidate_count"] == candidate["accepted_unknown_count"] + candidate["rejected_unknown_count"]
    for rate, errors, denominator in (
        ("false_accept_rate", "accepted_bad_count", "bad_candidate_count"),
        ("false_reject_rate", "rejected_good_count", "good_candidate_count"),
    ):
        if candidate[denominator]:
            assert candidate[rate] == pytest.approx(candidate[errors] / candidate[denominator])
        else:
            assert candidate[rate] is None
    assert "not an error-discrimination AUC" in report["quality_error"]["legacy_metrics_notice"]
    with (tmp_path / "benchmark_report.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    solver_rows = [row for row in rows if row["diagnostic_source"] == "solver"]
    assert len(solver_rows) == candidate["solver_point_count"]
    for row in solver_rows:
        error = float(row["error_px"]) if row["error_px"] else None
        expected = ("unknown" if error is None else
                    "good" if error <= candidate["error_tolerance_px"] else "bad")
        assert row["candidate_truth_label"] == expected
        accepted = row["accepted"] == "1"
        assert row["candidate_false_accept"] == ("" if expected == "unknown" else str(int(accepted and expected == "bad")))
        assert row["candidate_false_reject"] == ("" if expected == "unknown" else str(int(not accepted and expected == "good")))
    for label in ("good", "bad", "unknown"):
        for status, accepted in (("accepted", "1"), ("rejected", "0")):
            assert candidate[f"{status}_{label}_count"] == sum(
                row["candidate_truth_label"] == label and row["accepted"] == accepted
                for row in solver_rows
            )
    preflight = next(row for row in rows if row["diagnostic_source"] == "preflight_rejected")
    assert preflight["candidate_truth_label"] == "unknown"
    assert preflight["candidate_false_reject"] == ""
