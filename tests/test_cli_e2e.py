"""Real source-CLI end-to-end checks for the headless contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import ezdic_cli
import ezdic_core as core


ROOT = Path(__file__).resolve().parents[1]


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", np.asarray(image, dtype=np.uint8))
    assert ok
    encoded.tofile(str(path))


def _sequence(tmp_path: Path, *, count: int = 3, translation_per_frame: tuple[float, float] = (0.8, -0.25)) -> list[Path]:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    reference = core.generate_synthetic_speckle(128, 128, seed=17)
    paths: list[Path] = []
    for index in range(count):
        image = reference if index == 0 else core.warp_image_translation(
            reference,
            translation_per_frame[0] * index,
            translation_per_frame[1] * index,
        )
        path = image_dir / f"frame_{index + 1:03d}.png"
        _write_png(path, image)
        paths.append(path)
    return paths


def _base_config(paths: list[Path], output: Path, mode: str) -> dict:
    config = {
        "schema_version": 1,
        "analysis_mode": mode,
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": len(paths),
        "reference_frame_1based": 1,
        "output_dir": str(output),
        "export": {"write_manifest": True, "write_qc": True, "write_full_csv": True},
        "transaction": {"enabled": True, "archive_previous": True, "retain_failed_staging": True},
    }
    if mode == "extensometer":
        config["roi_groups"] = [
            {"name": "G01", "roi1": [20, 45, 25, 25], "roi2": [85, 45, 25, 25], "strain_mode": "x", "role": "none"}
        ]
    else:
        config["field_roi"] = [20, 20, 88, 88]
        config["solver"] = {"name": "IC-GN", "subset_size_px": 15, "step_px": 12, "strain_window_px": 5, "search_radius_px": 12}
        config["pyramid"] = {"levels": 1, "scale": 0.5}
    return config


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "ezdic_cli", *args], cwd=ROOT, text=True, capture_output=True, check=False)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def test_real_extensometer_cli_run_progress_and_manifest_verify(tmp_path: Path) -> None:
    paths = _sequence(tmp_path)
    config_path = tmp_path / "ext.json"
    output = tmp_path / "out-ext"
    _write_json(config_path, _base_config(paths, output, "extensometer"))

    result = _run_cli("run", "--config", str(config_path))
    assert result.returncode == ezdic_cli.EXIT_SUCCESS, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "completed"
    assert summary["scientific_ok"] is True
    assert summary["manifest_verified"] is True
    manifest = Path(summary["manifest_path"])
    assert manifest == output / "run_manifest.json"
    assert manifest.is_file()

    normalized = ezdic_cli.normalize_config(json.loads(config_path.read_text(encoding="utf-8")), base_dir=config_path.parent)
    api_settings = ezdic_cli.build_core_settings(normalized, core)
    api_settings["output_dir"] = str(tmp_path / "out-ext-api")
    api_result = core.run_extensometer_sequence(api_settings)
    assert api_result["status"] == "completed"
    api_manifest = json.loads(Path(api_result["manifest_path"]).read_text(encoding="utf-8"))
    cli_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert [entry["sha256"] for entry in cli_manifest["inputs"]] == [entry["sha256"] for entry in api_manifest["inputs"]]
    cli_csv = pd.read_csv(output / "optional" / "full_csv" / "strain_results_all_groups.csv")
    api_df = api_result["dataframe"]
    for column in ("group", "frame_global_1based", "engineering_strain", "strain_valid"):
        if column in cli_csv and column in api_df:
            if column in {"group", "strain_valid"}:
                assert cli_csv[column].astype(str).tolist() == api_df[column].astype(str).tolist()
            else:
                np.testing.assert_allclose(
                    pd.to_numeric(cli_csv[column], errors="coerce").to_numpy(),
                    pd.to_numeric(api_df[column], errors="coerce").to_numpy(),
                    equal_nan=True,
                )

    verified = _run_cli("verify-manifest", "--manifest", str(manifest))
    assert verified.returncode == ezdic_cli.EXIT_SUCCESS, verified.stdout + verified.stderr
    assert json.loads(verified.stdout)["ok"] is True

    progress_output = tmp_path / "out-progress"
    progress_config = _base_config(paths, progress_output, "extensometer")
    progress_path = tmp_path / "ext-progress.json"
    _write_json(progress_path, progress_config)
    progress = _run_cli("run", "--config", str(progress_path), "--progress-json")
    assert progress.returncode == ezdic_cli.EXIT_SUCCESS, progress.stdout + progress.stderr
    events = [json.loads(line) for line in progress.stdout.splitlines() if line.strip()]
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_finished"
    assert events[-1]["exit_code"] == 0
    assert all(event["event"] in {"run_started", "progress", "run_finished"} for event in events)


def test_real_fullfield_cli_run_and_config_hash_parity(tmp_path: Path) -> None:
    paths = _sequence(tmp_path, count=2, translation_per_frame=(0.0, 0.0))
    config = _base_config(paths, tmp_path / "out-field", "fullfield")
    config_path = tmp_path / "field.json"
    _write_json(config_path, config)
    result = _run_cli("run", "--config", str(config_path))
    assert result.returncode == ezdic_cli.EXIT_SUCCESS, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["mode"] == "fullfield"
    assert summary["manifest_verified"] is True
    normalized = ezdic_cli.normalize_config(config, base_dir=config_path.parent)
    assert summary["config_hash"] == ezdic_cli.config_hash(normalized)
    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["config_hash"] == ezdic_cli.config_hash(normalized)
    assert manifest["config"]["analysis_mode"] == "fullfield"
    assert manifest["config"]["quality"]["max_residual_rms"] is None
    assert "inf" not in json.dumps(manifest["config"], ensure_ascii=False).lower()
    gate = manifest["scientific_gate"]
    assert gate["scientific_ok"] is True
    assert gate["valid_frame_count"] >= 1
    assert gate["thresholds"]["min_correlation_valid_fraction"] == 0.95
    assert gate["thresholds"]["min_strain_valid_fraction"] == 0.80
    assert manifest["frames"]
    assert all(frame["correlation_valid_fraction"] >= 0.95 for frame in manifest["frames"])
    assert all(frame["strain_valid_fraction"] >= 0.80 for frame in manifest["frames"])


def test_folder_input_uses_core_natural_order_and_wrong_mode_is_rejected(tmp_path: Path) -> None:
    paths = _sequence(tmp_path, count=2)
    folder = tmp_path / "natural"
    folder.mkdir()
    # Deliberately reverse lexical order; collect_images must apply its natural sort.
    (folder / "frame_10.png").write_bytes(paths[1].read_bytes())
    (folder / "frame_2.png").write_bytes(paths[0].read_bytes())
    config = _base_config(paths, tmp_path / "out-folder", "extensometer")
    config.pop("image_paths")
    config["image_folder"] = str(folder)
    config_path = tmp_path / "folder.json"
    _write_json(config_path, config)
    result = _run_cli("run", "--config", str(config_path))
    assert result.returncode == ezdic_cli.EXIT_SUCCESS, result.stdout + result.stderr
    manifest = json.loads(Path(json.loads(result.stdout)["manifest_path"]).read_text(encoding="utf-8"))
    assert [Path(item["path"]).name for item in manifest["inputs"]] == ["frame_2.png", "frame_10.png"]

    wrong = _base_config(paths, tmp_path / "out-wrong", "extensometer")
    wrong["field_roi"] = [10, 10, 30, 30]
    wrong_path = tmp_path / "wrong.json"
    _write_json(wrong_path, wrong)
    rejected = _run_cli("validate-config", "--config", str(wrong_path))
    assert rejected.returncode == ezdic_cli.EXIT_CONFIG_ERROR
    assert json.loads(rejected.stderr)["error_code"] == "CONFIG_MODE_FIELD"


def test_single_frame_is_rejected_before_run(tmp_path: Path) -> None:
    paths = _sequence(tmp_path, count=2)
    single = _base_config(paths[:1], tmp_path / "out-single", "extensometer")
    single_path = tmp_path / "single.json"
    _write_json(single_path, single)
    rejected = _run_cli("validate-config", "--config", str(single_path))
    assert rejected.returncode == ezdic_cli.EXIT_CONFIG_ERROR


def _run_tamper_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    paths = _sequence(tmp_path, count=2)
    config = _base_config(paths, tmp_path / "out-tamper", "extensometer")
    config_path = tmp_path / "tamper.json"
    _write_json(config_path, config)
    result = _run_cli("run", "--config", str(config_path))
    assert result.returncode == ezdic_cli.EXIT_SUCCESS, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    return Path(summary["manifest_path"]), paths[0], summary


def test_input_tamper_is_reported_as_file_identity_mismatch(tmp_path: Path) -> None:
    manifest_path, input_path, _ = _run_tamper_fixture(tmp_path)
    original = input_path.read_bytes()
    try:
        input_path.write_bytes(original + b"input-tamper")
        tampered = _run_cli("verify-manifest", "--manifest", str(manifest_path))
        assert tampered.returncode == ezdic_cli.EXIT_GATE_ERROR
        summary = json.loads(tampered.stdout)
        assert any(error["code"] == "FILE_IDENTITY_MISMATCH" and error["section"] == "inputs" for error in summary["errors"])
    finally:
        input_path.write_bytes(original)


def test_output_tamper_is_reported_as_file_identity_mismatch(tmp_path: Path) -> None:
    manifest_path, _, _ = _run_tamper_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_entry = next(entry for entry in payload["outputs"] if entry["path"] != "run_manifest.json")
    output_path = manifest_path.parent / output_entry["path"]
    original = output_path.read_bytes()
    try:
        output_path.write_bytes(original + b"output-tamper")
        tampered = _run_cli("verify-manifest", "--manifest", str(manifest_path))
        assert tampered.returncode == ezdic_cli.EXIT_GATE_ERROR
        summary = json.loads(tampered.stdout)
        assert any(error["code"] == "FILE_IDENTITY_MISMATCH" and error["section"] == "outputs" for error in summary["errors"])
    finally:
        output_path.write_bytes(original)


def test_config_tamper_isolated_to_config_hash_mismatch(tmp_path: Path) -> None:
    manifest_path, _, _ = _run_tamper_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["config"]["metadata"] = {"tampered": True}
    # Keep the manifest envelope valid so verification must identify the
    # config binding, rather than stopping at MANIFEST_HASH_MISMATCH.
    payload["manifest_hash"] = core.canonical_json_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    tampered = _run_cli("verify-manifest", "--manifest", str(manifest_path))
    assert tampered.returncode == ezdic_cli.EXIT_GATE_ERROR
    summary = json.loads(tampered.stdout)
    assert any(error["code"] == "CONFIG_HASH_MISMATCH" for error in summary["errors"])
    assert not any(error["code"] == "MANIFEST_HASH_MISMATCH" for error in summary["errors"])


def test_code_tamper_isolated_to_code_fingerprint_mismatch(tmp_path: Path) -> None:
    manifest_path, _, _ = _run_tamper_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_path = ROOT / "schemas" / "run_config_v1.json"
    original = schema_path.read_bytes()
    try:
        schema_path.write_bytes(original + b"\n")
        tampered = _run_cli("verify-manifest", "--manifest", str(manifest_path))
        assert tampered.returncode == ezdic_cli.EXIT_GATE_ERROR
        summary = json.loads(tampered.stdout)
        assert any(error["code"] == "CODE_FINGERPRINT_MISMATCH" for error in summary["errors"])
        assert payload["code_files"]
    finally:
        schema_path.write_bytes(original)


def test_masked_fullfield_exports_and_mask_content_verification(tmp_path: Path) -> None:
    paths = _sequence(tmp_path, count=2, translation_per_frame=(0., 0.))
    config = _base_config(paths, tmp_path / "masked-output", "fullfield")
    mask_path = tmp_path / "specimen.png"
    mask = np.full((128, 128), 255, np.uint8)
    mask[50:70, 50:70] = 0
    _write_png(mask_path, mask)
    config["mask"] = {"mode": "file", "path": str(mask_path)}
    config["display"] = {"background": "deformed", "percent": True, "cmap": "viridis"}
    config_path = tmp_path / "masked.json"
    _write_json(config_path, config)
    result = _run_cli("run", "--config", str(config_path))
    assert result.returncode == ezdic_cli.EXIT_SUCCESS, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    output = tmp_path / "masked-output" / "dic"
    assert (output / "specimen_mask.png").is_file()
    for component in core.DIC_FIELD_COMPONENTS:
        assert (output / f"frame_0002_{core.DIC_COMPONENT_FILE_SUFFIXES[component]}.png").is_file()
    with np.load(output / "frame_0002.npz", allow_pickle=False) as arrays:
        assert not arrays['eligible'].all()
        assert np.isnan(arrays['u'][~arrays['eligible']]).all()
        metadata = json.loads(arrays['metadata_json'].item())
        assert metadata['mask']['mode'] == 'file'
        assert metadata['display']['percent'] is True
    manifest = summary['manifest_path']
    assert core.verify_run_manifest(manifest)['ok']
    mask[0, 0] = 0
    _write_png(mask_path, mask)
    changed = core.verify_run_manifest(manifest)
    assert not changed['ok']
    assert any(error.get('section') == 'inputs' and error.get('field') == 'sha256'
               and Path(error['path']) == mask_path for error in changed['errors'])
