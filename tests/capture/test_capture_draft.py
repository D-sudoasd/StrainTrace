from concurrent.futures import ThreadPoolExecutor

from straintrace_capture import ExperimentReader, inspect_plan, trial_report
from straintrace_capture.draft import capture_draft, load_capture_draft, load_draft_trials, save_capture_draft
from straintrace_capture.recent import RecentExperiment, remember_experiment
from straintrace_capture.trial_quality import rescan_trial_images

from test_capture_trial_quality import make_trial, trial_plan


def test_newer_draft_wins_when_desktop_windows_finish_saving_out_of_order(tmp_path, monkeypatch):
    monkeypatch.setenv("EZDIC_STATE_DIR", str(tmp_path / "state"))
    assert load_capture_draft() is None
    drafts = [capture_draft(trial_plan(10, 1, {"exposure_us": index + 800}), str(tmp_path / f"计划-{index}"), 0, ())
              for index in range(1, 8)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda row: save_capture_draft(row[1], row[0]), reversed(list(enumerate(drafts, 1)))))
    assert load_capture_draft() == drafts[-1]
    assert not save_capture_draft(drafts[0], 1)
    assert load_capture_draft() == drafts[-1]


def test_draft_reopens_a_moved_trial_by_id_and_rechecks_full_image_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("EZDIC_STATE_DIR", str(tmp_path / "state"))
    original = make_trial(tmp_path, 105, clip=(12,))
    config = trial_plan(105, 1)
    formal = tmp_path / "正式采集"
    report = trial_report(original)
    report["cameras"]["sim:0"]["image_check"] = rescan_trial_images(original, report["experiment_id"], "sim:0", all_frames=True)
    draft = capture_draft(config, str(formal), 0, [report])
    assert len(draft.trials) == 1 and draft.trials[0]["image_checks"]["sim:0"]["mode"] == "all"
    assert save_capture_draft(draft, 100)
    assert capture_draft(trial_plan(105, 1, {"exposure_us": 2000}), str(formal), 0, [report]).trials == ()

    moved = original.rename(tmp_path / "迁移后的试采")
    make_trial(tmp_path, 4).rename(original)
    restored = load_capture_draft()
    assert restored == draft and not formal.exists()
    assert load_draft_trials(restored) == ([], [1])
    remember_experiment(RecentExperiment(report["experiment_id"], "已搬移的试采", moved))
    loaded, missing = load_draft_trials(restored)
    assert not missing and loaded[0]["output"] == str(moved)
    checks = loaded[0]["cameras"]["sim:0"]["image_check"]
    assert checks["checked"] == 105 and [row["sequence"] for row in checks["rows"] if row["flags"]] == [12]
    preflight = inspect_plan(restored.config, restored.output, trial_reports=loaded)
    assert any(issue["code"] == "TRIAL_IMAGES_NEED_REVIEW" for issue in preflight["issues"])
    assert not formal.exists()

    with ExperimentReader(moved, immutable=True) as reader:
        image = reader.image_path(reader.frame_at_sequence("sim:0", 12))
    pixels = image.read_bytes()
    image.write_bytes(pixels[:-1] + bytes([pixels[-1] ^ 1]))
    loaded, missing = load_draft_trials(restored)
    assert not missing and not loaded[0]["observed"]
    preflight = inspect_plan(restored.config, restored.output, trial_reports=loaded)
    assert any(issue["code"] == "TRIAL_NEEDS_REVIEW" for issue in preflight["issues"])
    assert preflight["phases"][0]["trial_geometry_cameras"] == []
