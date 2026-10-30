"""Compare saturated and accidentally wrapped 8-bit synthetic translations.

The two versions share the same input speckle, prescribed shift and solver
settings. A sensor-like saturation and a deliberately corrupted uint8 cast are
both retained so the consequences of an invalid test image are auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import types

import cv2
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ezdic_core as current_core


SETTINGS = {
    "subset_size": 15, "step": 12, "search_radius": 12, "strain_window": 5,
    "zncc_min": .65, "pyramid_levels": 2, "pyramid_scale": .5,
    "min_correlation_valid_fraction": 0., "min_strain_valid_fraction": 0.,
}
ROI = (20, 20, 88, 88)
TRUTH = (.8, -.25)


def evaluate(core, reference, target):
    bounds = core.compute_reference_normalization(reference)
    field = core.run_2d_dic(
        core.normalize_dic_float(reference, bounds), core.normalize_dic_float(target, bounds), ROI, **SETTINGS,
    )
    valid = np.asarray(field["valid"], dtype=bool)
    error = np.hypot(field["u_raw"] - TRUTH[0], field["v_raw"] - TRUTH[1])[valid]
    return {
        "point_count": int(valid.size), "correlation_valid_count": int(valid.sum()),
        "strain_valid_count": int(np.count_nonzero(field["strain_valid"])),
        "valid_displacement_mean_error_px": float(error.mean()) if error.size else None,
        "valid_displacement_rmse_px": float(np.sqrt(np.mean(error**2))) if error.size else None,
        "valid_displacement_max_error_px": float(error.max()) if error.size else None,
        "valid_displacement_error_over_0_5_px_count": int((error > .5).sum()),
        "invalid_reason_histogram": field["quality_summary"]["invalid_reason_histogram"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD", help="Git ref providing baseline ezdic_core.py")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline_source = subprocess.check_output(["git", "show", f"{args.baseline_ref}:ezdic_core.py"], cwd=ROOT)
    baseline_commit = subprocess.check_output(["git", "rev-parse", f"{args.baseline_ref}^{{commit}}"], cwd=ROOT, text=True).strip()
    baseline_core = types.ModuleType("straintrace_png_baseline_core")
    baseline_core.__file__ = "git-baseline-ezdic-core.py"
    sys.modules[baseline_core.__name__] = baseline_core
    exec(compile(baseline_source, baseline_core.__file__, "exec"), baseline_core.__dict__)

    reference_float = current_core.generate_synthetic_speckle(128, 128, seed=17)
    warped_float = current_core.warp_image_translation(reference_float, *TRUTH)
    reference = reference_float.astype(np.uint8)
    wrapped = warped_float.astype(np.uint8)
    saturated = np.clip(warped_float, 0, 255).astype(np.uint8)
    cases = {}
    for name, target in (("wrapped_corruption", wrapped), ("saturated_png", saturated)):
        cases[name] = {
            "input_sha256": hashlib.sha256(reference.tobytes() + target.tobytes()).hexdigest(),
            "baseline": evaluate(baseline_core, reference, target),
            "current": evaluate(current_core, reference, target),
        }
    report = {
        "schema": "png-quantization-validation-v1",
        "baseline_commit": baseline_commit,
        "baseline_core_sha256": hashlib.sha256(baseline_source).hexdigest(),
        "current_core_sha256": hashlib.sha256(Path(current_core.__file__).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__},
        "input": {
            "shape": [128, 128], "texture_seed": 17, "nominal_translation_uv_px": TRUTH,
            "warped_float_min": float(warped_float.min()), "warped_float_max": float(warped_float.max()),
            "samples_above_255": int(np.count_nonzero(warped_float > 255)),
            "samples_differing_between_conversions": int(np.count_nonzero(wrapped != saturated)),
        },
        "settings": dict(SETTINGS, roi=ROI), "cases": cases,
        "limitations": [
            "Nominal translation is an approximate displacement oracle after cubic interpolation, saturation and 8-bit quantization.",
            "The 0.5 px error threshold is illustrative, not an experimental uncertainty estimate.",
            "Wrapped data are intentionally corrupted; lower coverage does not itself prove that every rejected point is wrong.",
            "Different accepted sets mean their error averages are not a paired accuracy comparison.",
        ],
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
