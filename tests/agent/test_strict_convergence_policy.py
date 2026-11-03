"""The agent must bind and enforce its convergence policy across resume."""

import hashlib
import json

import cv2
import numpy as np
import pytest

import ezdic_core as core
from straintrace_agent import AgentError, init_project, plan_project, run, verify
from straintrace_agent.api import _hash_json, _solver_parameters_match


def _project(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    reference = core.generate_synthetic_speckle(112, 112, seed=17)
    for frame, stretch in ((1, 1.0), (2, 1.005), (3, 1.01)):
        image = reference if frame == 1 else core.warp_image_deformation_gradient(
            reference, [[stretch, 0.0], [0.0, 1.0]]
        )
        assert cv2.imwrite(str(images / f"frame_{frame:04d}.png"), np.clip(image, 0, 255).astype(np.uint8))
    project = tmp_path / "project"
    init_project(project, images, frame_interval_s=0.5, step_hint_px=8)
    return project


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_missing_policy_only_matches_when_both_omit_it():
    legacy = {"subset_size_px": 21, "step_px": 8, "solver": "IC-GN",
              "strain_window_poi": 5, "search_radius_px": 10}
    assert _solver_parameters_match(legacy, legacy.copy())
    for policy in (True, False, 1, "true"):
        newer = {**legacy, "reject_nonconverged": policy}
        assert not _solver_parameters_match(legacy, newer)
        assert not _solver_parameters_match(newer, legacy)
    assert _solver_parameters_match({**legacy, "reject_nonconverged": True},
                                    {**legacy, "reject_nonconverged": True})
    assert not _solver_parameters_match({**legacy, "reject_nonconverged": False},
                                        {**legacy, "reject_nonconverged": True})
    assert not _solver_parameters_match({**legacy, "reject_nonconverged": 1},
                                        {**legacy, "reject_nonconverged": True})


def test_pilot_and_run_enforce_policy_and_preserve_known_unconverged_rejection(tmp_path, monkeypatch):
    project = _project(tmp_path)
    original_dic = core.run_2d_dic
    seen = []

    def observed(*args, **kwargs):
        seen.append(kwargs.get("reject_nonconverged"))
        return original_dic(*args, **kwargs)

    monkeypatch.setattr(core, "run_2d_dic", observed)
    plan = plan_project(project)
    assert plan["status"] == "ready"
    assert seen and all(value is True for value in seen)
    assert plan["selected_parameters"]["reject_nonconverged"] is True
    assert all(item["reject_nonconverged"] is True for item in plan["candidate_trials"])
    seen.clear()

    # Force one otherwise legitimate subset result to report nonconvergence;
    # this checks the real core gate, not only argument plumbing.
    original_refiner = core.refine_subset_icgn
    forced = False

    def refiner(*args, **kwargs):
        nonlocal forced
        result = original_refiner(*args, **kwargs)
        if result is not None and not forced:
            result = dict(result)
            result["converged"] = False
            forced = True
        return result

    monkeypatch.setattr(core, "refine_subset_icgn", refiner)
    result = run(project, make_plots=False)
    assert forced and seen and all(value is True for value in seen)
    run_root = project / "runs" / result["run_id"]
    receipt = json.loads((run_root / "checkpoints" / "frame_native_00000001.json").read_text(encoding="utf-8"))
    assert receipt["parameters"]["reject_nonconverged"] is True
    with np.load(run_root / "checkpoints" / "frame_native_00000001.npz", allow_pickle=False) as arrays:
        invalid = np.asarray(arrays["invalid_reason"]) == "NOT_CONVERGED"
        assert invalid.any()
        assert not np.asarray(arrays["valid"], dtype=bool)[invalid].any()


def test_policy_mismatch_recomputes_only_the_affected_checkpoint(tmp_path):
    project = _project(tmp_path)
    plan = plan_project(project)
    first = run(project, make_plots=False)
    run_root = project / "runs" / first["run_id"]
    checkpoints = run_root / "checkpoints"
    unchanged = checkpoints / "frame_native_00000001.npz"
    before = _sha(unchanged)
    receipt_path = checkpoints / "frame_native_00000002.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["parameters"]["reject_nonconverged"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert any(issue["code"] == "CHECKPOINT_PARAMETER_MISMATCH" for issue in verify(project)["issues"])

    events = []
    result = run(project, make_plots=False, progress=events.append)
    assert result["run_id"] == first["run_id"]
    assert {event["native_frame"] for event in events if event.get("event") == "frame_reused"} == {1, 3}
    assert any(event.get("event") == "frame_complete" and event.get("native_frame") == 2 for event in events)
    assert _sha(unchanged) == before
    repaired = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert repaired["parameters"] == plan["selected_parameters"]
    assert verify(project)["valid"] is True


def test_missing_policy_in_resigned_plan_fails_before_creating_run(tmp_path):
    project = _project(tmp_path)
    plan_project(project)
    pointer = json.loads((project / "CURRENT.json").read_text(encoding="utf-8"))
    plan_path = project / pointer["current_plan"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    del plan["selected_parameters"]["reject_nonconverged"]
    plan.pop("plan_sha256")
    plan["plan_sha256"] = _hash_json(plan)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(AgentError) as error:
        run(project, make_plots=False)
    assert error.value.code == "PLAN_SOLVER_POLICY_INVALID"
    assert not list((project / "runs").glob("run-*"))


def test_code_change_blocks_old_plan_before_creating_or_reusing_checkpoints(tmp_path, monkeypatch):
    import straintrace_agent.api as api

    project = _project(tmp_path)
    plan_project(project)
    old = api._code_fingerprint()
    monkeypatch.setattr(api, "_code_fingerprint", lambda: {**old, "straintrace_agent/api.py": "changed"})
    with pytest.raises(AgentError) as error:
        run(project, make_plots=False)
    assert error.value.code == "PLAN_STALE_CODE"
    assert not list((project / "runs").glob("run-*"))
