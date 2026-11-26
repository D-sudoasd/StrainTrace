from pathlib import Path
from threading import Event

import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureError, ExperimentReader
from straintrace_capture.control import send_command
from straintrace_capture.csv_trigger import CSV_TRIGGER_EVENT
from straintrace_capture.presentation import format_trial
from straintrace_capture.trial import compare_trials, prepare_trial
from straintrace_capture.trial_batch import run_trial_batch
from straintrace_capture.trial_progress import inspect_trial_progress


def plan(phases, *, exposure=2000):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "Sequential trials",
        "cameras": [
            {"id": f"sim:{index}", "settings": {"width": 64, "height": 48, "fps": 12, "exposure_us": exposure}}
            for index in range(2)
        ],
        "phases": phases,
    })


def run(config, output, phases, *, stop=None, progress=lambda *_args: None, seconds=3, **guidance):
    opened, completed = [], []
    end = run_trial_batch(config, output, phases, seconds, registry=CameraRegistry(), discovered_ids={"sim:0", "sim:1"},
                          stop=stop if stop is not None else Event(), watch_quality=False, progress=progress,
                          opened=lambda identity, _name, target: opened.append((identity, target)),
                          completed=lambda *result: completed.append(result), **guidance)
    return end, opened, completed


def test_sequential_trials_skip_operator_steps_and_keep_regular_evidence(tmp_path):
    config = plan([
        {"mode": "single", "role": "reference"},
        {"mode": "single"},
        {"mode": "manual", "frame_count": 3},
        {"mode": "continuous", "frame_count": 3},
        {"mode": "timed", "fps": 5, "frame_count": 2},
        {"mode": "continuous", "frame_count": 4, "start_confirmation": "Confirm fixture"},
        {"mode": "event", "fps": 8, "pre_frames": 1, "post_frames": 1, "duration_s": 2},
        {"mode": "hardware", "hardware_source": "Line1", "frame_count": 3},
    ])
    output = tmp_path / "正式采集"
    progress = inspect_trial_progress(config, output, ())
    assert progress.batch_phases == (2, 4, 5)
    assert [progress.phases[index - 1].batch_exclusion for index in (1, 3, 6, 7, 8)] == [
        "optional", "trigger", "confirmation", "trigger", "trigger",
    ]

    result, opened, completed = run(config, output, progress.batch_phases)
    assert result.state == "completed", result
    assert [phase for phase, _state, _report in completed] == [2, 4, 5]
    assert [report["verification"]["committed_images"] for _, _, report in completed] == [2, 6, 4]
    assert len({identity for identity, _path in opened}) == 3
    assert not output.exists()
    for (identity, target), (phase, _state, report) in zip(opened, completed, strict=True):
        assert target.parent == output.parent and target.name.startswith("trial-")
        assert report["trial"]["formal_output"] == str(output)
        assert report["trial"]["phase_number"] == phase
        assert report["experiment_id"] == identity
        with ExperimentReader(target) as reader:
            integrity = reader.verify()
        assert integrity["complete"] and integrity["file_hashes_checked"]
    assert inspect_trial_progress(config, output, [report for _, _, report in completed]).batch_phases == ()


def test_stopping_the_current_trial_preserves_the_completed_prefix_and_supports_resuming(tmp_path):
    config = plan([{"mode": "single"}, {"mode": "continuous", "frame_count": 30}, {"mode": "single"}])
    output = tmp_path / "正式采集"
    stop = Event()

    def progress(number, stage, count):
        if number == 2 and stage == "capturing" and count >= 2:
            stop.set()

    end, opened, done = run(config, output, (1, 2, 3), stop=stop, progress=progress)
    assert (end.state, end.phase) == ("stopped", 2), end
    assert len(opened) == len(done) == 2 and not output.exists()
    assert done[0][2]["verification"]["complete"]
    cancelled = done[1][2]
    assert cancelled["verification"]["state"] == "stopped"
    assert cancelled["verification"]["ok"] and 2 <= cancelled["verification"]["committed_images"] < 60
    assert Path(cancelled["output"]).is_dir()

    pending = inspect_trial_progress(config, output, [report for _, _, report in done])
    assert pending.batch_phases == (2, 3)
    same_id = done[0][2]["experiment_id"]
    continued, _opened, finished = run(config, output, pending.batch_phases)
    assert continued.state == "completed", continued
    assert all(report["experiment_id"] != same_id for _, _, report in finished)
    assert inspect_trial_progress(config, output, [done[0][2], *(report for _, _, report in finished)]).batch_phases == ()


def test_preflight_that_needs_operator_review_never_opens_the_affected_stage(tmp_path):
    config = plan([{"mode": "single"}, {"mode": "timed", "fps": 40, "frame_count": 2}, {"mode": "single"}],
                  exposure=60_000)
    result, opened, completed = run(config, tmp_path / "正式采集", (1, 2, 3))
    assert (result.state, result.phase) == ("preflight", 2), result
    assert result.preflight["ready"]
    assert any(item["code"] == "EXPOSURE_EXCEEDS_INTERVAL" for item in result.preflight["issues"])
    assert len(opened) == len(completed) == 1
    assert list(tmp_path.glob("trial-*")) == [opened[0][1]]


def test_acquisition_error_retains_the_failed_trial_and_leaves_later_stages_pending(tmp_path):
    config = plan([{"mode": "single"}, {"mode": "continuous", "frame_count": 12}, {"mode": "single"}]).to_dict()
    config["cameras"][0]["settings"]["disconnect_after"] = 3
    config = CaptureConfig.from_dict(config)
    result, opened, completed = run(config, tmp_path / "正式采集", (1, 2, 3))
    assert (result.state, result.phase) == ("failed", 2), result
    assert len(opened) == len(completed) == 2
    assert completed[0][2]["verification"]["complete"]
    failed = completed[1][2]
    assert failed["verification"]["state"] == "failed" and failed["error"]["code"] == "CAMERA_DISCONNECTED"
    assert len(list(tmp_path.glob("trial-*"))) == 2


def test_guided_confirmation_and_manual_requests_use_the_live_session_and_preserve_receipts(tmp_path):
    prompt = "核对夹具有无遮挡"
    config = plan([{"mode": "single"}, {"mode": "manual", "frame_count": 2, "start_confirmation": prompt},
                   {"mode": "single"}])
    destination = tmp_path / "正式输出"
    progress = inspect_trial_progress(config, destination, ())
    assert progress.batch_phases == (1, 3) and progress.trial_batch_phases(guided=True) == (1, 2, 3)
    actions = []

    def guide(live):
        if live is None or live.action is None:
            return
        action = live.action[0]
        assert live.number == 2
        if action == "confirm":
            with ExperimentReader(live.session.output) as reader:
                assert not reader.frames() and not reader.events(kind="phase_started")
            actions.append(live.request())
            with pytest.raises(CaptureError, match="当前等待状态"):
                live.request()
        else:
            if len(actions) == 1:
                actions.append(live.request())
            else:
                result = send_command(live.session.output, "trigger", source="instrument-test", expected_phase=1)
                assert result["ok"]
                actions.append(result["signal_id"])

    end, opened, done = run(config, destination, (1, 2, 3), guided=True, observe=guide, external=True)
    assert end.state == "completed", end
    assert [phase for phase, _, _ in done] == [1, 2, 3]
    _, _, report = done[1]
    assert report["verification"]["committed_images"] == 4
    with ExperimentReader(opened[1][1]) as reader:
        confirmed, = reader.events(kind="phase_confirmed")
        assert confirmed["confirmation_id"] == actions[0] and confirmed["instruction"] == prompt
        assert confirmed["source"] == "operator"
        signals = reader.events(kind="signal_received")
        assert [signal["event_id"] for signal in signals] == actions[1:]
        assert [signal["source"] for signal in signals] == ["operator", "instrument-test"]
    assert "阶段 2 · 已确认" in format_trial(report)
    assert inspect_trial_progress(config, destination, [report]).phases[1].status == "matched"
    altered = config.to_dict()
    altered["phases"][1]["start_confirmation"] = "换一个确认条件"
    revised = inspect_trial_progress(CaptureConfig.from_dict(altered), destination, [report])
    assert revised.phases[1].record["changes"] == ["start_confirmation"]


def test_guided_wait_can_be_stopped_before_a_camera_arms(tmp_path):
    config = plan([{"mode": "single", "start_confirmation": "核对夹具"}, {"mode": "single"}])
    stop = Event()

    def guide(live):
        if live and live.action and live.action[0] == "confirm":
            stop.set()

    outcome, opened, done = run(config, tmp_path / "正式输出", (1, 2), guided=True, observe=guide, stop=stop)
    assert (outcome.state, outcome.phase) == ("stopped", 1)
    assert len(opened) == len(done) == 1
    with ExperimentReader(opened[0][1]) as reader:
        assert not reader.frames() and not reader.events(kind="phase_started")
        assert len(reader.events(kind="phase_confirmation_cancelled")) == 1
        assert not reader.events(kind="phase_confirmed")


def test_guided_csv_requires_the_configured_source_and_keeps_the_crossing_receipt(tmp_path):
    source = tmp_path / "试验机.csv"
    source.write_text("时间(s),载荷(kN)\n0,9\n", encoding="utf-8")
    config = plan([{"mode": "event", "fps": 6, "pre_frames": 2, "post_frames": 1,
                    "csv_trigger": {"path": str(source), "value_column": "载荷(kN)", "time_column": "时间(s)",
                                    "quantity": "载荷", "unit": "kN", "threshold": 5, "direction": "rising"}},
                   {"mode": "single"}])
    sent = False

    def guide(live):
        nonlocal sent
        if live is None:
            return
        assert live.action is None
        progress = live.status.get("phase_progress") or {}
        if (not sent and (progress.get("event_window") or {}).get("state") == "waiting"
                and (progress.get("csv_trigger") or {}).get("state") == "watching"):
            response = send_command(live.session.output, "trigger", expected_phase=1)
            assert response["error"]["code"] == "CONTROL_TRIGGER_DISABLED"
            with source.open("a", encoding="utf-8") as file:
                file.write("1,4\n2,6\n")
            sent = True

    outcome, opened, done = run(config, tmp_path / "正式输出", (1, 2), guided=True, observe=guide, external=True)
    assert outcome.state == "completed" and sent, outcome
    with ExperimentReader(opened[0][1]) as reader:
        crossing, = reader.events(kind=CSV_TRIGGER_EVENT)
        signal, = reader.events(kind="signal_received")
        assert crossing["signal_id"] == signal["event_id"] and signal["source"] == "instrument_csv_threshold"
        assert signal["payload"]["sample"]["source_time_text"] == "2"
    assert done[0][2]["verification"]["committed_images"] >= 6
    assert "载荷 上穿 5 kN" in format_trial(done[0][2])


def test_guided_trigger_timeout_remains_pending_and_never_moves_to_a_later_stage(tmp_path):
    config = plan([{"mode": "manual", "frame_count": 2}, {"mode": "single"}])
    output = tmp_path / "正式输出"
    outcome, opened, done = run(config, output, (1, 2), guided=True, seconds=.55)
    assert (outcome.state, outcome.phase) == ("no_images", 1) and len(opened) == len(done) == 1
    assert not done[0][2]["observed"]
    assert inspect_trial_progress(config, output, [done[0][2]]).trial_batch_phases(guided=True) == (1, 2)


def test_truncated_repeating_event_has_a_valid_performance_scope(tmp_path):
    config = plan([{"mode": "event", "fps": 4, "pre_frames": 2, "post_frames": 2, "event_count": 5}])
    output = tmp_path / "正式输出"
    captured, _target = prepare_trial(config, output, phase_number=1, seconds=1.3)
    assert captured.phases[0].event_count == 0
    metadata = captured.metadata["capture_trial"]
    report = {"created_utc_ns": 1, "output": str(output), "experiment_id": "trial", "trial": metadata,
              "phase": captured.phases[0].to_dict(), "volume": metadata["scope"]["volume"],
              "observed": False, "findings": [], "cameras": {}, "setup": {
                  "camera_settings": {camera.id: dict(camera.settings) for camera in config.cameras},
                  "cti_files": [], "storage": {key: config.to_dict()[key] for key in (
                      "durability", "queue_capacity", "max_buffer_bytes", "segment_frames", "min_free_bytes",
                  )},
              }}
    assessed, matches = compare_trials(config, report["volume"], [report])
    assert assessed[0]["status"] == "matched" and 1 in matches


@pytest.mark.parametrize("numbers", [(2,), (1, 2), (1, 1), (3, 1), (0,)])
def test_invalid_or_interactive_selection_is_rejected_before_any_capture(tmp_path, numbers):
    config = plan([{"mode": "single"}, {"mode": "manual", "frame_count": 2}, {"mode": "single"}])
    with pytest.raises(CaptureError) as error:
        run(config, tmp_path / "正式采集", numbers)
    assert error.value.code == "TRIAL_BATCH_PHASES"
    assert not list(tmp_path.iterdir())
