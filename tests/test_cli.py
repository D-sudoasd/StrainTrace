"""Contract tests for the headless CLI surface (Task 6)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import ezdic_cli


def _config(mode: str, paths: list[str]) -> dict:
    config = {
        "schema_version": 1,
        "analysis_mode": mode,
        "image_paths": paths,
        "start_frame_1based": 1,
        "end_frame_1based": len(paths),
        "reference_frame_1based": 1,
        "output_dir": "out",
    }
    if mode == "extensometer":
        config["roi_groups"] = [
            {
                "name": "g01",
                "roi1": [10, 10, 20, 20],
                "roi2": [50, 10, 20, 20],
                "strain_mode": "auto",
                "role": "axial",
            },
        ]
    else:
        config["field_roi"] = [10, 10, 80, 80]
    return config


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_schema_valid_for_both_modes_and_expands_defaults(tmp_path: Path) -> None:
    for mode in ("extensometer", "fullfield"):
        normalized = ezdic_cli.validate_config(_config(mode, ["a.png", "b.png"]))
        assert normalized["schema_version"] == 1
        assert normalized["analysis_mode"] == mode
        assert normalized["reference_frame_1based"] == 1
        if mode == "fullfield":
            assert normalized["quality"]["min_correlation_valid_fraction"] == 0.95
        assert normalized["quality"]["min_strain_valid_fraction"] == 0.80
        assert normalized["quality"]["peak_margin_min"] == 0.02
        assert normalized["quality"]["best_to_second_peak_ratio_min"] == 1.02
        if mode == "fullfield":
            assert normalized["quality"]["max_residual_rms"] is None
        assert normalized["normalization"]["clip"] is True
        assert normalized["export"]["write_manifest"] is True
        assert normalized["transaction"]["enabled"] is True
        if mode == "fullfield":
            assert normalized["solver"]["name"] == "IC-GN"
        assert normalized["normalization"]["policy"] == "reference_percentile"
        assert normalized["transaction"]["enabled"] is True


@pytest.mark.parametrize(
    "mutator,code",
    [
        (lambda c: c.pop("reference_frame_1based"), "CONFIG_SCHEMA_ERROR"),
        (lambda c: c.__setitem__("unexpected", 1), "CONFIG_UNKNOWN_FIELD"),
        (lambda c: c.__setitem__("start_frame_1based", "1"), "CONFIG_TYPE_ERROR"),
        (lambda c: c.__setitem__("end_frame_1based", 3), "CONFIG_INPUT_ERROR"),
        (lambda c: c.setdefault("quality", {}).update({"zncc_min": 2}), "CONFIG_VALUE_ERROR"),
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, mutator, code: str) -> None:
    config = _config("fullfield", ["a.png", "b.png"])
    mutator(config)
    with pytest.raises(ezdic_cli.ConfigError) as error:
        ezdic_cli.validate_config(config)
    assert error.value.code == code


def test_nonfinite_and_numeric_string_are_rejected() -> None:
    config = _config("fullfield", ["a.png", "b.png"])
    config["quality"] = {"zncc_min": float("nan")}
    with pytest.raises(ezdic_cli.ConfigError) as error:
        ezdic_cli.validate_config(config)
    assert error.value.code == "CONFIG_NONFINITE_NUMBER"

    config = _config("fullfield", ["a.png", "b.png"])
    config["pyramid"] = {"levels": "2"}
    with pytest.raises(ezdic_cli.ConfigError) as error:
        ezdic_cli.validate_config(config)
    assert error.value.code == "CONFIG_TYPE_ERROR"

    config = _config("fullfield", ["a.png", "b.png"])
    config["quality"] = {"max_condition_number": 10**1000}
    with pytest.raises(ezdic_cli.ConfigError) as error:
        ezdic_cli.validate_config(config)
    assert error.value.code == "CONFIG_NONFINITE_NUMBER"


def test_scientific_units_use_poi_sigma_and_dimensionless_residual() -> None:
    config = _config("fullfield", ["a.png", "b.png"])
    config["solver"] = {"smooth_sigma_poi": 1.25}
    config["quality"] = {"max_residual_rms": 0.15}
    normalized = ezdic_cli.validate_config(config)
    assert normalized["solver"]["smooth_sigma_poi"] == 1.25
    assert normalized["quality"]["max_residual_rms"] == 0.15

    for section, wrong_key in (("solver", "smooth_sigma_px"), ("quality", "max_residual_rms_px")):
        invalid = _config("fullfield", ["a.png", "b.png"])
        invalid[section] = {wrong_key: 1.0}
        with pytest.raises(ezdic_cli.ConfigError) as error:
            ezdic_cli.validate_config(invalid)
        assert error.value.code == "CONFIG_UNKNOWN_FIELD"
        assert wrong_key in error.value.message

    invalid = _config("fullfield", ["a.png", "b.png"])
    invalid["quality"] = {"max_residual_rms": -0.1}
    with pytest.raises(ezdic_cli.ConfigError) as error:
        ezdic_cli.validate_config(invalid)
    assert "normalized grayscale residual RMS (dimensionless)" in error.value.message


@pytest.mark.parametrize(
    "section,mutation",
    [
        ("normalization", lambda value: value.update({"policy": "fixed_bounds"})),
        ("normalization", lambda value: value.update({"clip": False})),
        ("transaction", lambda value: value.update({"enabled": False})),
        ("transaction", lambda value: value.update({"archive_previous": False})),
        ("transaction", lambda value: value.update({"retain_failed_staging": False})),
        ("export", lambda value: value.update({"write_manifest": False})),
        ("export", lambda value: value.update({"write_parameters": False})),
        ("quality", lambda value: value.update({"best_to_second_peak_ratio_min": 0.9})),
    ],
)
def test_strict_runtime_safety_contracts_reject_unsafe_settings(section: str, mutation) -> None:
    config = _config("fullfield", ["a.png", "b.png"])
    config[section] = {}
    mutation(config[section])
    with pytest.raises(ezdic_cli.ConfigError):
        ezdic_cli.validate_config(config)


def test_fixed_bounds_requires_finite_ordered_lo_hi() -> None:
    config = _config("fullfield", ["a.png", "b.png"])
    config["normalization"] = {"policy": "fixed_bounds", "bounds": {"lo": 1.0}}
    with pytest.raises(ezdic_cli.ConfigError) as error:
        ezdic_cli.validate_config(config)
    assert error.value.code == "CONFIG_SCHEMA_ERROR"

    config["normalization"]["bounds"] = {"lo": 2.0, "hi": 1.0}
    with pytest.raises(ezdic_cli.ConfigError) as error:
        ezdic_cli.validate_config(config)
    assert error.value.code == "CONFIG_VALUE_ERROR"


@pytest.mark.parametrize(
    "mode,key",
    [
        ("extensometer", "zncc_min"),
        ("extensometer", "max_condition_number"),
        ("extensometer", "max_residual_rms"),
        ("extensometer", "reject_nonconverged"),
        ("extensometer", "min_correlation_valid_fraction"),
        ("fullfield", "enable_fb_check"),
        ("fullfield", "fb_tolerance_px"),
    ],
)
def test_quality_fields_are_mode_specific(mode: str, key: str) -> None:
    config = _config(mode, ["a.png", "b.png"])
    config["quality"] = {key: False if key == "reject_nonconverged" or key == "enable_fb_check" else 0.1}
    with pytest.raises(ezdic_cli.ConfigError) as error:
        ezdic_cli.normalize_config(config)
    assert error.value.code == "CONFIG_UNKNOWN_FIELD"


def test_runtime_code_paths_do_not_change_canonical_config_hash(tmp_path: Path) -> None:
    normalized = ezdic_cli.validate_config(_config("fullfield", ["a.png", "b.png"]))
    with_runtime_binding = dict(normalized)
    with_runtime_binding["_code_paths"] = [str(tmp_path / "ezdic_core.py"), str(tmp_path / "ezdic_cli.py")]
    assert ezdic_cli.config_hash(normalized) == ezdic_cli.config_hash(with_runtime_binding)


def test_fullfield_quality_maps_ratio_min_in_core_direction(tmp_path: Path) -> None:
    image_paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for image_path in image_paths:
        image_path.write_bytes(b"placeholder")
    config = _config("fullfield", [str(path) for path in image_paths])
    config["quality"] = {
        "best_to_second_peak_ratio_min": 1.5,
        "max_residual_rms": None,
    }
    normalized = ezdic_cli.normalize_config(config)

    class StubCore:
        __file__ = __file__

        @staticmethod
        def collect_images(_folder):
            return []

    settings = ezdic_cli.build_core_settings(normalized, StubCore)
    assert settings["peak_ratio_min"] == 1.5
    assert settings["quality"]["max_residual_rms"] == float("inf")


def test_frozen_layout_code_paths_prefer_existing_sources_tree(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "_MEI" / "sources"
    schema_root = source_root / "schemas"
    schema_root.mkdir(parents=True)
    for filename in ("ezdic_core.py", "ezdic_cli.py", "ezdic_benchmark.py"):
        (source_root / filename).write_text("# frozen source\n", encoding="utf-8")
    (schema_root / "run_config_v1.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(ezdic_cli.sys, "_MEIPASS", str(tmp_path / "_MEI"), raising=False)
    archive_path = tmp_path / "archive.pyz" / "ezdic_core.py"
    resolved = ezdic_cli._resolve_code_paths(archive_path)
    assert resolved == [
        source_root / "ezdic_core.py",
        source_root / "ezdic_cli.py",
        source_root / "ezdic_benchmark.py",
        schema_root / "run_config_v1.json",
    ]


def test_frozen_layout_code_paths_match_spec_root_schema_and_four_code_files(tmp_path: Path, monkeypatch) -> None:
    bundle_root = tmp_path / "_MEI"
    source_root = bundle_root / "sources"
    (bundle_root / "schemas").mkdir(parents=True)
    source_root.mkdir()
    for filename in ("ezdic_core.py", "ezdic_cli.py", "ezdic_benchmark.py"):
        (source_root / filename).write_text("# frozen source\n", encoding="utf-8")
    (bundle_root / "schemas" / "run_config_v1.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(ezdic_cli.sys, "_MEIPASS", str(bundle_root), raising=False)
    resolved = ezdic_cli._resolve_code_paths(bundle_root / "archive.pyz" / "ezdic_core.py")
    assert resolved == [
        source_root / "ezdic_core.py",
        source_root / "ezdic_cli.py",
        source_root / "ezdic_benchmark.py",
        bundle_root / "schemas" / "run_config_v1.json",
    ]


@pytest.mark.parametrize("code", ["LOW_TEXTURE", "SATURATED_TEXTURE", "AMBIGUOUS_TEXTURE"])
def test_texture_rejections_use_scientific_gate_exit_code(code: str) -> None:
    class CoreFailure(RuntimeError):
        pass

    error = CoreFailure(code)
    error.code = code
    assert ezdic_cli._exit_for_core_error(error) == ezdic_cli.EXIT_GATE_ERROR


def test_schema_and_manual_normalizer_agree_on_representative_contracts() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator_cls = getattr(jsonschema, "Draft202012Validator", None)
    if validator_cls is None:
        pytest.skip("jsonschema Draft 2020-12 validator is not installed")
    schema = ezdic_cli.load_schema()
    validator = validator_cls(schema)
    for mode in ("extensometer", "fullfield"):
        config = _config(mode, ["a.png", "b.png"])
        assert not list(validator.iter_errors(config))
        assert ezdic_cli.normalize_config(config)["schema_version"] == 1

    invalid_roi = _config("fullfield", ["a.png", "b.png"])
    invalid_roi["field_roi"] = ["1", 1, 20, 20]
    assert list(validator.iter_errors(invalid_roi))
    with pytest.raises(ezdic_cli.ConfigError):
        ezdic_cli.normalize_config(invalid_roi)

    invalid_transaction = _config("fullfield", ["a.png", "b.png"])
    invalid_transaction["transaction"] = {"archive_previous": False}
    assert list(validator.iter_errors(invalid_transaction))
    with pytest.raises(ezdic_cli.ConfigError):
        ezdic_cli.normalize_config(invalid_transaction)


def test_extensometer_rejects_single_rect_and_requires_roi_pair() -> None:
    config = _config("extensometer", ["a.png", "b.png"])
    config["roi_groups"] = [{"name": "legacy", "rect": [10, 10, 20, 20]}]
    with pytest.raises(ezdic_cli.ConfigError) as error:
        ezdic_cli.validate_config(config)
    assert error.value.code == "CONFIG_UNKNOWN_FIELD"

    config["roi_groups"] = [{"name": "incomplete", "roi1": [10, 10, 20, 20]}]
    with pytest.raises(ezdic_cli.ConfigError) as error:
        ezdic_cli.validate_config(config)
    assert error.value.code == "CONFIG_SCHEMA_ERROR"

    config["roi_groups"] = [
        {
            "name": "g01",
            "roi1": [10, 10, 20, 20],
            "roi2": [50, 10, 20, 20],
            "strain_mode": "bad",
        }
    ]
    with pytest.raises(ezdic_cli.ConfigError) as error:
        ezdic_cli.validate_config(config)
    assert error.value.code == "CONFIG_VALUE_ERROR"


def test_cli_validate_prints_canonical_config_and_help_is_headless(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _config("fullfield", ["a.png", "b.png"]))
    result = subprocess.run(
        [sys.executable, "-m", "ezdic_cli", "validate-config", "--config", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["output_dir"] == str((tmp_path / "out").resolve())
    assert output["field_roi_reference_frame_1based"] == 1

    help_result = subprocess.run(
        [sys.executable, "-m", "ezdic_cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "validate-config" in help_result.stdout
    assert "tkinter" not in help_result.stdout


def test_cli_invalid_json_has_stable_error_and_exit_code_2(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not-json", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "ezdic_cli", "validate-config", "--config", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == ezdic_cli.EXIT_CONFIG_ERROR
    error = json.loads(result.stderr)
    assert error["error_code"] == "CONFIG_JSON_ERROR"
    assert error["exit_code"] == 2


@pytest.mark.parametrize("command", ["run", "verify-manifest"])
def test_run_and_verify_commands_use_their_stable_error_codes(command: str) -> None:
    args = [sys.executable, "-m", "ezdic_cli", command]
    if command == "run":
        args += ["--config", "missing.json"]
    elif command == "verify-manifest":
        args += ["--manifest", "missing.json"]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    expected_exit = ezdic_cli.EXIT_CONFIG_ERROR if command == "run" else ezdic_cli.EXIT_GATE_ERROR
    assert result.returncode == expected_exit
    error = json.loads(result.stderr)
    assert error["exit_code"] == expected_exit


def test_benchmark_without_output_is_structured_config_error() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ezdic_cli", "benchmark"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == ezdic_cli.EXIT_CONFIG_ERROR
    error = json.loads(result.stderr)
    assert error["error_code"] == "BENCHMARK_OUTPUT_REQUIRED"


def test_schema_enforces_normalization_if_then_odd_solver_and_mode_fields() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(ezdic_cli.load_schema())

    missing_bounds = _config("fullfield", ["a.png", "b.png"])
    missing_bounds["normalization"] = {"policy": "fixed_bounds"}
    assert list(validator.iter_errors(missing_bounds))

    null_bounds = _config("fullfield", ["a.png", "b.png"])
    null_bounds["normalization"] = {"policy": "fixed_bounds", "bounds": None}
    assert list(validator.iter_errors(null_bounds))

    valid_fixed = _config("fullfield", ["a.png", "b.png"])
    valid_fixed["normalization"] = {"policy": "fixed_bounds", "bounds": {"lo": 10.0, "hi": 200.0}}
    assert not list(validator.iter_errors(valid_fixed))

    unexpected_bounds = _config("fullfield", ["a.png", "b.png"])
    unexpected_bounds["normalization"] = {"policy": "reference_percentile", "bounds": {"lo": 0.0, "hi": 1.0}}
    assert list(validator.iter_errors(unexpected_bounds))

    ext_with_field_ref = _config("extensometer", ["a.png", "b.png"])
    ext_with_field_ref["field_roi_reference_frame_1based"] = 1
    assert list(validator.iter_errors(ext_with_field_ref))

    even_subset = _config("fullfield", ["a.png", "b.png"])
    even_subset["solver"] = {"subset_size_px": 20}
    assert list(validator.iter_errors(even_subset))

    even_window = _config("fullfield", ["a.png", "b.png"])
    even_window["solver"] = {"strain_window_px": 4}
    assert list(validator.iter_errors(even_window))

    policy_mismatch = _config("extensometer", ["a.png", "b.png"])
    policy_mismatch["tracking"] = {"template_policy": "fixed_reference", "use_prev_frame_template": True}
    assert list(validator.iter_errors(policy_mismatch))

    policy_mismatch_2 = _config("extensometer", ["a.png", "b.png"])
    policy_mismatch_2["tracking"] = {"template_policy": "follow_deformed_experimental", "use_prev_frame_template": False}
    assert list(validator.iter_errors(policy_mismatch_2))


def test_tracking_rejects_soft_corr_above_hard_corr() -> None:
    config = _config("extensometer", ["a.png", "b.png"])
    config["tracking"] = {"hard_corr": 0.50, "soft_corr": 0.70}
    with pytest.raises(ezdic_cli.ConfigError) as exc:
        ezdic_cli.normalize_config(config)
    assert exc.value.code == "CONFIG_VALUE_ERROR"
    assert exc.value.path == "$.tracking.soft_corr"


def test_core_settings_copy_preserves_canonical_snapshot(tmp_path: Path) -> None:
    images = [tmp_path / "a.png", tmp_path / "b.png"]
    for path in images:
        path.write_bytes(b"img")
    normalized = ezdic_cli.normalize_config(_config("fullfield", [str(p) for p in images]))

    class StubCore:
        __file__ = __file__

    settings = ezdic_cli.build_core_settings(normalized, StubCore)
    cloned = settings.copy()
    assert isinstance(cloned, ezdic_cli._CoreSettings)
    assert list(settings) == list(settings.keys()) == [k for k, _ in settings.items()]
    assert len(settings) == len(list(settings.items()))
    assert list(cloned) == list(cloned.keys()) == [k for k, _ in cloned.items()]
    assert dict(cloned.items()) == dict(settings.items())
    assert cloned.canonical_config == normalized
    assert ezdic_cli.config_hash(cloned) == ezdic_cli.config_hash(normalized)


def test_resolve_code_paths_fails_closed_when_frozen_bundle_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "_MEI"
    (bundle / "sources").mkdir(parents=True)
    (bundle / "sources" / "ezdic_core.py").write_text("# core\n", encoding="utf-8")
    monkeypatch.setattr(ezdic_cli.sys, "_MEIPASS", str(bundle), raising=False)
    with pytest.raises(ezdic_cli.ConfigError) as exc:
        ezdic_cli._resolve_code_paths(bundle / "archive.pyz" / "ezdic_core.py")
    assert exc.value.code == "CODE_FILE_SET_INCOMPLETE"


def test_resolve_code_paths_handles_frozen_external_core_path_and_custom_checkout(tmp_path: Path, monkeypatch) -> None:
    import ezdic_core

    bundle = tmp_path / "_MEI_full"
    (bundle / "sources").mkdir(parents=True)
    (bundle / "schemas").mkdir(parents=True)
    for name in ("ezdic_core.py", "ezdic_cli.py", "ezdic_benchmark.py"):
        (bundle / "sources" / name).write_text(f"# {name}\n", encoding="utf-8")
    (bundle / "schemas" / "run_config_v1.json").write_text("{}\n", encoding="utf-8")

    # Frozen bundle with an external real core_path must resolve inside _MEIPASS.
    monkeypatch.setattr(ezdic_cli.sys, "_MEIPASS", str(bundle), raising=False)
    frozen_paths = ezdic_cli._resolve_code_paths(Path(ezdic_core.__file__).resolve())
    assert [p.name for p in frozen_paths] == [
        "ezdic_core.py",
        "ezdic_cli.py",
        "ezdic_benchmark.py",
        "run_config_v1.json",
    ]
    assert all(bundle.resolve() in p.resolve().parents for p in frozen_paths)

    # Non-frozen custom checkout root must resolve all 4 files inside that checkout.
    monkeypatch.delattr(ezdic_cli.sys, "_MEIPASS", raising=False)
    checkout = tmp_path / "alt_checkout"
    (checkout / "schemas").mkdir(parents=True)
    for name in ("ezdic_core.py", "ezdic_cli.py", "ezdic_benchmark.py"):
        (checkout / name).write_text(f"# alt {name}\n", encoding="utf-8")
    (checkout / "schemas" / "run_config_v1.json").write_text("{}\n", encoding="utf-8")
    alt_paths = ezdic_cli._resolve_code_paths(checkout / "ezdic_core.py")
    assert [p.name for p in alt_paths] == [
        "ezdic_core.py",
        "ezdic_cli.py",
        "ezdic_benchmark.py",
        "run_config_v1.json",
    ]
    assert all(checkout.resolve() in p.resolve().parents for p in alt_paths)


def test_cli_json_output_is_safe_on_cp1252_stream_and_version_flag(monkeypatch) -> None:
    import io

    raw_buf = io.BytesIO()
    cp1252_stream = io.TextIOWrapper(raw_buf, encoding="cp1252", errors="strict")
    monkeypatch.setattr(ezdic_cli.sys, "stderr", cp1252_stream)
    ezdic_cli._write_error("TEST_CODE", "事务提交失败：中文路径", exit_code=3)
    cp1252_stream.flush()
    decoded = json.loads(raw_buf.getvalue().decode("cp1252"))
    assert decoded["error_code"] == "TEST_CODE"
    assert "事务提交失败" in decoded["message"]

    version_result = subprocess.run(
        [sys.executable, "-m", "ezdic_cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert version_result.returncode == 0
    assert "ezdic " in version_result.stdout


def test_frozen_entrypoint_marker_uses_lf_line_endings(tmp_path: Path) -> None:
    import ezdic_frozen_entrypoint as frozen

    marker = tmp_path / "marker.json"
    frozen._write_marker(marker, {"ok": True, "step": 1})
    raw = marker.read_bytes()
    assert b"\r\n" not in raw
    assert json.loads(raw.decode("utf-8")) == {"ok": True, "step": 1}
