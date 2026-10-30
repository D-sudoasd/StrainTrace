"""Alternate isolated baseline/current runs for a small DIC timing comparison.

This is a controlled microbenchmark, not a representative production workload.
For example: python benchmarks/necking_search_timing.py --baseline-ref HEAD
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD", help="Git ref supplying the baseline ezdic_core.py")
    parser.add_argument("--repeats", type=int, default=6, help="Runs per version, alternating execution order")
    parser.add_argument("--output", type=Path, help="New JSON path; print to stdout if omitted")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    baseline = subprocess.check_output(["git", "show", f"{args.baseline_ref}:ezdic_core.py"], cwd=ROOT)
    commit = subprocess.check_output(["git", "rev-parse", f"{args.baseline_ref}^{{commit}}"], cwd=ROOT, text=True).strip()
    current = (ROOT / "ezdic_core.py").read_bytes()
    scripts = ("__init__.py", "synthetic_cases.py", "necking_search_validation.py")
    observations = {"baseline": [], "current": []}
    with tempfile.TemporaryDirectory(prefix="straintrace-necking-timing-") as tmp:
        locations = {}
        for label, source in (("baseline", baseline), ("current", current)):
            location = Path(tmp) / label
            (location / "benchmarks").mkdir(parents=True)
            (location / "ezdic_core.py").write_bytes(source)
            for name in scripts:
                shutil.copy2(ROOT / "benchmarks" / name, location / "benchmarks" / name)
            locations[label] = location
        for index in range(args.repeats):
            order = ("baseline", "current") if index % 2 == 0 else ("current", "baseline")
            for label in order:
                location = locations[label]
                output = subprocess.check_output(
                    [sys.executable, str(location / "benchmarks" / "necking_search_validation.py")],
                    cwd=location,
                )
                observations[label].append(json.loads(output))

    first = observations["baseline"][0]
    expected_inputs = [(case["case"], case["input_sha256"]) for case in first["cases"]]
    for label, runs in observations.items():
        for run in runs:
            if [(case["case"], case["input_sha256"]) for case in run["cases"]] != expected_inputs:
                raise RuntimeError(f"{label} generated different validation inputs")
            if run["provenance"]["script_sha256"] != first["provenance"]["script_sha256"]:
                raise RuntimeError("validation script hash changed during timing")
            expected_core = sha256(baseline if label == "baseline" else current)
            if run["provenance"]["core_sha256"] != expected_core:
                raise RuntimeError(f"{label} core hash changed during timing")

    cases = []
    for case_index, (case_name, input_hash) in enumerate(expected_inputs):
        timings = {
            label: [float(run["cases"][case_index]["elapsed_seconds"]) for run in runs]
            for label, runs in observations.items()
        }
        cases.append({
            "case": case_name,
            "input_sha256": input_hash,
            "baseline_seconds": timings["baseline"],
            "current_seconds": timings["current"],
            "baseline_median_seconds": statistics.median(timings["baseline"]),
            "current_median_seconds": statistics.median(timings["current"]),
            "baseline_valid_count": [run["cases"][case_index]["correlation_valid_count"] for run in observations["baseline"]],
            "current_valid_count": [run["cases"][case_index]["correlation_valid_count"] for run in observations["current"]],
        })
    report = {
        "report_version": "necking-search-timing-v1",
        "baseline_commit": commit,
        "baseline_core_sha256": sha256(baseline),
        "current_core_sha256": sha256(current),
        "validation_script_sha256": first["provenance"]["script_sha256"],
        "timing_script_sha256": sha256(Path(__file__).read_bytes()),
        "environment": first["provenance"],
        "settings": first["settings"],
        "repeats_per_version": args.repeats,
        "order": "baseline/current on even iterations, current/baseline on odd iterations",
        "limitations": [
            "Timers measure run_2d_dic, including automatic initialization, but exclude image generation and Python startup.",
            "These small synthetic inputs are not representative of full-scale experiment processing.",
            "Shared machine load and power state were not controlled; elapsed time is descriptive.",
        ],
        "cases": cases,
    }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
