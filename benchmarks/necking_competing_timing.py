"""Measure the competing-peak probe overhead on fixed synthetic inputs.

The comparison disables only the probe in memory for the control; both arms
use the same current core, search radius, images, and solver settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import cv2
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ezdic_core as core
from benchmarks import necking_search_validation as search
from benchmarks import necking_validation as neck


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    cases = []
    for name in ("localized_shear_band", "uniform_affine"):
        reference, target = search.make_case(name)
        cases.append((name, reference, target, search.ROI, search.SETTINGS))
    reference, target = neck.image_pair(neck.CASES["narrow_localization"])
    cases.append(("narrow_axial_fine", reference, target, neck.ROI,
                  dict(neck.PARAMETERS, subset_size=15, step=4, strain_window=3)))
    original_peak, original_refine = core._competing_correlation_peak, core._refine_competing_peak
    output_cases = []
    try:
        for name, reference, target, roi, settings in cases:
            records = {"without_probe": [], "with_probe": []}
            for iteration in range(args.repeats):
                modes = ("without_probe", "with_probe") if iteration % 2 == 0 else ("with_probe", "without_probe")
                for mode in modes:
                    core._competing_correlation_peak = original_peak if mode == "with_probe" else lambda *_a, **_k: None
                    core._refine_competing_peak = original_refine if mode == "with_probe" else lambda *_a, **_k: {"status": "PROBE_DISABLED_FOR_CONTROL", "result": None}
                    start = time.perf_counter()
                    field = core.run_2d_dic(reference, target, roi, **settings)
                    records[mode].append({
                        "elapsed_seconds": time.perf_counter() - start,
                        "valid_count": int(np.count_nonzero(field["valid"])),
                        "point_count": int(len(field["valid"])),
                        "ambiguous_count": int(np.count_nonzero(field["peak_is_ambiguous"])),
                    })
            medians = {mode: statistics.median(row["elapsed_seconds"] for row in rows) for mode, rows in records.items()}
            output_cases.append({
                "case": name,
                "input_sha256": hashlib.sha256(reference.tobytes() + target.tobytes()).hexdigest(),
                "settings": dict(settings, roi=roi),
                "runs": records,
                "median_seconds": medians,
                "median_ratio_with_over_without": medians["with_probe"] / medians["without_probe"],
            })
    finally:
        core._competing_correlation_peak, core._refine_competing_peak = original_peak, original_refine
    report = {
        "schema": "competing-peak-timing-v1",
        "core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "timing_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "search_case_script_sha256": hashlib.sha256(Path(search.__file__).read_bytes()).hexdigest(),
        "neck_case_script_sha256": hashlib.sha256(Path(neck.__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__},
        "repeats_per_arm": args.repeats,
        "order": "control/with-probe on even iterations, with-probe/control on odd iterations",
        "limitations": [
            "Control is a same-core in-memory diagnostic with competing-peak detection and refinement disabled; do not use it for production analysis.",
            "Shared machine load and power state were not controlled. Results are descriptive for these small synthetic cases only.",
            "Acceptance counts can differ because the two arms use different reliability gates.",
        ],
        "cases": output_cases,
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
