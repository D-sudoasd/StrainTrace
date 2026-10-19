"""Default plotting batch uses process-isolated sample runs and honest status."""

from concurrent.futures import Future

import cv2
import numpy as np

import ezdic_core as core
import straintrace_agent.api as agent_api
from straintrace_agent import batch, init_project


def _write_sequence(image_dir, *, include_corrupt=False):
    image_dir.mkdir()
    reference = core.generate_synthetic_speckle(112, 112, seed=17)
    for native, stretch in ((1, 1.0), (2, 1.005), (4, 1.015)):
        image = reference if native == 1 else core.warp_image_deformation_gradient(reference, [[stretch, 0], [0, 1]])
        assert cv2.imwrite(str(image_dir / f"frame_{native:04d}.png"), np.clip(image, 0, 255).astype(np.uint8))
    if include_corrupt:
        (image_dir / "frame_0003.png").write_bytes(b"bad image")


def test_batch_keeps_execution_and_numerical_quality_separate_with_default_plots(tmp_path):
    good_images = tmp_path / "good-raw"
    bad_images = tmp_path / "attention-raw"
    _write_sequence(good_images)
    _write_sequence(bad_images, include_corrupt=True)
    good_project, attention_project = tmp_path / "good", tmp_path / "attention"
    init_project(good_project, good_images, frame_interval_s=0.5, step_hint_px=8)
    init_project(attention_project, bad_images, frame_interval_s=0.5, step_hint_px=8)

    result = batch([good_project, attention_project], max_workers=2, make_plots=True)
    assert result["status"] == "complete"
    assert result["execution_completed"] == 2
    assert result["execution_failed"] == 0
    assert result["scientific_passed"] == 1
    assert result["attention_required"] == 1
    assert result["numerical_quality_status"] == "attention_required"
    assert result["scientific_acceptance"] == "not_assessed"

    per_project = {item["project"]: item for item in result["results"]}
    assert per_project[str(good_project.resolve())]["numerical_quality_status"] == "pass"
    assert per_project[str(attention_project.resolve())]["numerical_quality_status"] == "attention_required"
    for project, expected_maps in ((good_project, 27), (attention_project, 27)):
        current = __import__("json").loads((project / "CURRENT.json").read_text(encoding="utf-8"))
        run_root = project / current["current_run"]
        maps = [path.name for path in (run_root / "plots").glob("*.png")]
        assert len(maps) == expected_maps
        assert len({name.casefold() for name in maps}) == len(maps)


def test_batch_converts_worker_crash_to_per_sample_failure(tmp_path, monkeypatch):
    class FailedFutureExecutor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, _function, project_path, *_args):
            future = Future()
            if project_path.endswith("crashed"):
                future.set_exception(RuntimeError("worker process exited"))
            else:
                future.set_result({
                    "project": project_path, "execution_status": "complete",
                    "scientific_status": "pass", "numerical_quality_status": "pass",
                    "scientific_acceptance": "not_assessed",
                })
            return future

    monkeypatch.setattr(agent_api, "ProcessPoolExecutor", FailedFutureExecutor)
    result = batch([tmp_path / "good", tmp_path / "crashed"], max_workers=2)
    assert result["status"] == "partial"
    assert result["execution_completed"] == 1
    assert result["execution_failed"] == 1
    failed = next(row for row in result["results"] if row["execution_status"] == "failed")
    assert failed["error"]["error_code"] == "BATCH_WORKER_FAILED"
