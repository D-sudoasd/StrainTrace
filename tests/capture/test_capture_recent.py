from concurrent.futures import ThreadPoolExecutor

from straintrace_capture.recent import LIMIT, RecentExperiment, recent_experiments, remember_experiment


def test_recent_experiments_preserve_identity_and_replace_a_moved_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("EZDIC_STATE_DIR", str(tmp_path / "state"))
    assert recent_experiments() == []
    entries = [RecentExperiment(f"experiment-{index}", f"实验 {index}", tmp_path / f"output-{index}")
               for index in range(LIMIT + 2)]
    for experiment in entries:
        remember_experiment(experiment)
    assert recent_experiments() == list(reversed(entries[-LIMIT:]))

    # A relocation uses the saved experiment identity even when paths and names may overlap.
    old = entries[2]
    moved = RecentExperiment(old.experiment_id, old.name, tmp_path / "新位置")
    remember_experiment(moved)
    assert recent_experiments() == [moved, *reversed(entries[3:])]
    same_path = RecentExperiment("another-experiment", moved.name, moved.directory)
    remember_experiment(same_path)
    assert recent_experiments()[:2] == [same_path, moved]


def test_concurrent_capture_windows_preserve_both_recent_records(tmp_path, monkeypatch):
    monkeypatch.setenv("EZDIC_STATE_DIR", str(tmp_path / "state"))
    experiments = [RecentExperiment(f"same-name-{index}", "重复名称", tmp_path / f"输出-{index}") for index in range(8)]
    with ThreadPoolExecutor(max_workers=len(experiments)) as workers:
        list(workers.map(remember_experiment, experiments))
    assert set(recent_experiments()) == set(experiments)
