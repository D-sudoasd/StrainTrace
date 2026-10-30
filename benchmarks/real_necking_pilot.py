"""Reproduce a bounded FIO83 full-field diagnostic with identical input hashes.

This is not a benchmark against ground truth or the user's VIC-2D curve. It
uses five sparse frames and one rectangle from a different historical dataset.
Output is written to a new directory, never over the images or an earlier run.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA256 = "828a257db89a280a9b4f6591ce52d34024cebec86f69429da134c09d74a69344"
FRAME_NUMBERS = (50, 90, 130, 185, 193)
ROI = (1550, 350, 1000, 250)
BOUNDS = (80.0, 3756.0)
PARAMETERS = {
    "subset_size": 31,
    "step": 40,
    "search_radius": 20,
    "solver": "IC-GN",
    "max_iter": 40,
    "conv_tol": 0.001,
    "zncc_min": 0.8,
    "strain_window": 5,
    "reject_nonconverged": True,
    "initialization": "auto",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("real_necking_pilot_core", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run(images: Path, output: Path, core_file: Path) -> dict:
    core_file = core_file.resolve(strict=True)
    core_hash = sha256(core_file)
    core = load_core(core_file)
    reference_path = images / "dir_000001.tif"
    if sha256(reference_path) != REFERENCE_SHA256:
        raise ValueError("Reference hash does not match the reviewed historical FIO83 image")
    raw = core.read_gray_image(reference_path)
    if sha256(reference_path) != REFERENCE_SHA256:
        raise RuntimeError("Reference image changed while being read")
    if raw.shape != (1000, 4112):
        raise ValueError("Unexpected reference image dimensions")
    reference = core.normalize_dic_float(raw, BOUNDS)
    # Identical seed on every invocation, including archived core comparisons.
    cv2.setRNGSeed(0)
    records = []
    for number in FRAME_NUMBERS:
        path = images / f"dir_{number:06d}.tif"
        image_hash = sha256(path)
        deformed = core.normalize_dic_float(core.read_gray_image(path), BOUNDS)
        if sha256(path) != image_hash:
            raise RuntimeError(f"Target image changed while being read: {path.name}")
        started = time.perf_counter()
        field = core.run_2d_dic(reference, deformed, ROI, **PARAMETERS)
        elapsed = time.perf_counter() - started
        valid = np.asarray(field["valid"], dtype=bool)
        strain_valid = np.asarray(field["strain_valid"], dtype=bool)
        invalid = np.asarray(field["invalid_reason"], dtype=str)
        strain_invalid = np.asarray(field["strain_invalid_reason"], dtype=str)
        np.savez_compressed(
            output / f"frame_{number:03d}.npz",
            **{key: np.asarray(field[key]) for key in ("x", "y", "u", "v", "Exx", "Eyy", "zncc", "eligible")},
            valid=valid, strain_valid=strain_valid, invalid_reason=invalid, strain_invalid_reason=strain_invalid,
        )
        record = {
            "frame": number, "filename": path.name, "sha256": image_hash,
            "elapsed_s": elapsed, "eligible_n": int(np.count_nonzero(field["eligible"])),
            "valid_n": int(valid.sum()), "strain_valid_n": int(strain_valid.sum()),
            "invalid_reason_counts": dict(collections.Counter(invalid[~valid].tolist())),
            "strain_invalid_reason_counts": dict(collections.Counter(strain_invalid[~strain_valid].tolist())),
            "median_zncc": float(np.nanmedian(field["zncc"])) if valid.any() else None,
            "initial_transform": field["initial_transform"],
            "effective_search_radius_px": field["effective_search_radius_px"],
            "quality": field["quality_summary"],
        }
        records.append(record)
        print(json.dumps(_jsonable(record), ensure_ascii=False), flush=True)
    if sha256(core_file) != core_hash:
        raise RuntimeError("Core file changed during measurement; output cannot be attributed to one source")
    report = {
        "schema": "straintrace-real-necking-pilot-v1", "dataset_key": "CAMERA_FIO:000083",
        "acquisition_id": "83_LDF_Ti15_812WQ_300_40", "images_path": str(images.resolve()),
        "reference_frame": 1, "reference_sha256": REFERENCE_SHA256,
        "reference_shape": list(raw.shape), "reference_dtype": str(raw.dtype),
        "core_file": str(core_file), "core_sha256": core_hash,
        "benchmark_script_sha256": sha256(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "opencv": cv2.__version__},
        "roi_xywh_px": ROI, "normalization_bounds": BOUNDS, "parameters": PARAMETERS,
        "rng_seed": 0, "frames": records,
        "scope": "Sparse fixed-reference FIO83 image diagnostic; no independent truth or VIC equivalence. "
                 "The displayed user curve's specimen identity is unverified. Frame 193 has an open fracture.",
    }
    _write_json(output / "report.json", report)
    return report


def compare(before_path: Path, after: dict, after_dir: Path) -> dict:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    for key in ("schema", "reference_sha256", "roi_xywh_px", "normalization_bounds", "parameters", "rng_seed", "benchmark_script_sha256", "environment"):
        if before[key] != _jsonable(after[key]):
            raise ValueError(f"Before and after differ in {key}; comparison is invalid")
    if [r["frame"] for r in before["frames"]] != [r["frame"] for r in after["frames"]]:
        raise ValueError("Before and after frame lists differ")
    records = []
    for old, new in zip(before["frames"], after["frames"]):
        if old["sha256"] != new["sha256"]:
            raise ValueError(f"Target image hash differs for frame {new['frame']}")
        frame = new["frame"]
        with np.load(before_path.parent / f"frame_{frame:03d}.npz", allow_pickle=False) as a, \
             np.load(after_dir / f"frame_{frame:03d}.npz", allow_pickle=False) as b:
            for key in ("x", "y", "eligible"):
                if not np.array_equal(a[key], b[key]):
                    raise ValueError(f"Different {key} at frame {frame}")
            common = a["valid"] & b["valid"]
            common_strain = a["strain_valid"] & b["strain_valid"]
            differences = {}
            for key, mask in (("u", common), ("v", common), ("Exx", common_strain), ("Eyy", common_strain), ("zncc", common)):
                d = b[key][mask] - a[key][mask]
                d = d[np.isfinite(d)]
                differences[key] = {
                    "n": int(d.size), "max_abs": float(np.max(np.abs(d))) if d.size else None,
                    "rms": float(np.sqrt(np.mean(d * d))) if d.size else None,
                }
            records.append({
                "frame": frame, "sha256": new["sha256"],
                "valid_n_before_after": [old["valid_n"], new["valid_n"]],
                "strain_valid_n_before_after": [old["strain_valid_n"], new["strain_valid_n"]],
                "common_valid_n": int(common.sum()), "common_strain_valid_n": int(common_strain.sum()),
                "gained_valid_n": int(np.count_nonzero(b["valid"] & ~a["valid"])),
                "lost_valid_n": int(np.count_nonzero(a["valid"] & ~b["valid"])),
                "effective_radius_px_before_after": [old["effective_search_radius_px"], new["effective_search_radius_px"]],
                "elapsed_s_before_after": [old["elapsed_s"], new["elapsed_s"]],
                "invalid_reason_counts_before_after": [old["invalid_reason_counts"], new["invalid_reason_counts"]],
                "common_poi_differences": differences,
            })
    result = {
        "schema": "straintrace-real-necking-comparison-v1", "reference_sha256": after["reference_sha256"],
        "core_before_sha256": before["core_sha256"], "core_after_sha256": after["core_sha256"],
        "environment": after["environment"],
        "frames": records,
        "scope": "Differences on common valid points are not accuracy without experimental truth. "
                 "One timing per run is diagnostic, not a performance benchmark.",
    }
    _write_json(after_dir / "comparison.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Folder with original FIO83 dir_*.tif files")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; must not exist")
    parser.add_argument("--core-file", type=Path, default=ROOT / "ezdic_core.py")
    parser.add_argument("--compare-before", type=Path, help="Earlier report.json and adjacent frame NPZ files")
    args = parser.parse_args()
    images = args.images.resolve(strict=True)
    if not images.is_dir():
        parser.error("--images must be a directory")
    if args.output.resolve().is_relative_to(images):
        parser.error("--output must be outside the original image directory")
    args.output.mkdir(parents=True, exist_ok=False)
    report = run(images, args.output, args.core_file)
    if args.compare_before:
        compare(args.compare_before, report, args.output)


if __name__ == "__main__":
    main()
