"""Frozen/source entrypoint for the ezDIC GUI and release smoke test.

The smoke path is deliberately selected before importing the legacy Tk GUI.
It imports the GUI-independent core, CLI contract and locked benchmark modules,
checks the bundled schema and records only an explicitly requested marker.  A
normal invocation delegates to the established GUI ``main`` function.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


def _support_paths(runtime_root: Path) -> dict[str, Path]:
    """Resolve the portable root documents required for a self-describing build."""

    names = (
        "README.md",
        "README_使用说明.txt",
        "RELEASE_NOTES_v0.2.0-dev.md",
        "VERSION.txt",
        "NOTICE_Attribution_and_Usage.txt",
        "LICENSE.txt",
        "CITATION.cff",
    )
    paths = {name: runtime_root / name for name in names}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError("portable support file(s) missing: " + ", ".join(missing))
    return paths


def _write_marker(path: Path, payload: dict[str, Any]) -> None:
    """Write an optional caller-owned marker without creating default output."""

    path = path.expanduser()
    if not path.parent.is_dir():
        raise RuntimeError(f"smoke marker parent does not exist: {path.parent}")
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def smoke_test() -> int:
    """Run a read-only frozen/source contract smoke before any Tk import."""

    tk_was_loaded = "tkinter" in sys.modules

    # These imports are intentionally local: a smoke launch must not import the
    # legacy GUI module or construct a Tk root.
    import ezdic_benchmark as benchmark
    import ezdic_cli as cli
    import ezdic_core as core

    if not tk_was_loaded and "tkinter" in sys.modules:
        raise RuntimeError("smoke path imported tkinter")

    runtime_root = core._runtime_code_root()
    resolved_paths = core.resolve_code_paths(include_cli=True, include_benchmark=True, include_schema=True)
    by_name = {path.name.casefold(): path for path in resolved_paths}
    schema_path = by_name.get("run_config_v1.json")
    if schema_path is None or not schema_path.is_file():
        raise RuntimeError("run_config_v1.json was not found in source or frozen bundle")
    support_paths = _support_paths(runtime_root)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise RuntimeError("bundled schema root is not an object")
    if schema.get("$id", "").endswith("run_config_v1.json") is False:
        raise RuntimeError("bundled schema is not run_config_v1")
    if schema.get("properties", {}).get("schema_version", {}).get("const") != 1:
        raise RuntimeError("bundled schema does not declare schema_version=1")
    if getattr(cli, "SCHEMA_VERSION", None) != 1:
        raise RuntimeError("CLI schema contract version is unavailable")

    cases = getattr(benchmark, "LOCKED_CASES", ())
    if not cases or not getattr(benchmark, "LOCKED_CASES_VERSION", None):
        raise RuntimeError("locked benchmark cases are unavailable")
    locked_hash = benchmark.locked_cases_hash()
    if not isinstance(locked_hash, str) or len(locked_hash) != 64:
        raise RuntimeError("locked benchmark hash is unavailable")
    if getattr(core, "APP_NAME", None) != "ezDIC":
        raise RuntimeError("core identity is unavailable")

    def _resolved_hash(filename: str) -> str:
        target = by_name.get(filename.casefold())
        return core.sha256_file(target) if target is not None and target.is_file() else "unavailable"

    marker = os.environ.get("EZDIC_FROZEN_SMOKE_MARKER", "").strip()
    if marker:
        payload = {
            "smoke": "passed",
            "mode": "frozen" if getattr(sys, "frozen", False) else "source",
            "executable": str(Path(sys.executable).resolve()),
            "runtime_root": str(runtime_root),
            "core_version": str(getattr(core, "APP_VERSION", "unknown")),
            "core_sha256": _resolved_hash("ezdic_core.py"),
            "cli_sha256": _resolved_hash("ezdic_cli.py"),
            "benchmark_sha256": _resolved_hash("ezdic_benchmark.py"),
            "schema": str(schema_path),
            "schema_sha256": core.sha256_file(schema_path),
            "support_files": {
                name: core.sha256_file(path) for name, path in support_paths.items()
            },
            "locked_cases_version": str(benchmark.LOCKED_CASES_VERSION),
            "locked_cases_sha256": locked_hash,
            "tkinter_loaded": "tkinter" in sys.modules,
        }
        _write_marker(Path(marker), payload)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if getattr(sys, "frozen", False) and Path(sys.executable).stem.casefold() == "straintrace-capture":
        from straintrace_capture.gui import main as capture_main

        return capture_main(arguments)
    if arguments and arguments[0] == "--campaign-io-worker":
        from straintrace_campaign.delivery import worker_main
        return int(worker_main(arguments[1:]))
    if "--smoke-test" in arguments:
        return smoke_test()

    # Keep the established application module and GUI lifecycle unchanged for
    # normal launches.  This import is after the smoke branch by design.
    from dic_virtual_extensometer_gui_v7_multi_roi_range import main as gui_main

    gui_main(arguments)
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except Exception as exc:
        # A windowed PyInstaller executable has no useful console.  Preserve a
        # structured diagnostic when an explicit marker was requested, then
        # return non-zero so the build/CI caller can fail closed.
        marker = os.environ.get("EZDIC_FROZEN_SMOKE_MARKER", "").strip()
        if marker:
            try:
                _write_marker(Path(marker), {"smoke": "failed", "error": str(exc)})
            except Exception:
                pass
        raise
