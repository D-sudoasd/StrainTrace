# -*- mode: python ; coding: utf-8 -*-

"""PyInstaller onedir specification for the GUI and headless CLI.

The two entrypoints intentionally share one portable directory.  The GUI is a
windowed executable; ``ezDIC-cli.exe`` is a console executable that exposes the
same headless CLI contract as ``python ezdic_cli.py`` in a source checkout.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata


# PyInstaller executes a spec in a generated namespace without ``__file__``;
# ``SPECPATH`` is the documented absolute directory of this spec.
PROJECT_ROOT = Path(SPECPATH).resolve()


def source_file(relative_path: str) -> str:
    return str(PROJECT_ROOT / relative_path)


# Root-level documents are copied again by build_release.ps1 so the portable
# package remains self-describing even when a user starts it from a shell.
root_datas = [
    (source_file("README.md"), "."),
    (source_file("README_使用说明.txt"), "."),
    (source_file("RELEASE_NOTES_v0.2.0-dev.md"), "."),
    (source_file("VERSION.txt"), "."),
    (source_file("NOTICE_Attribution_and_Usage.txt"), "."),
    (source_file("LICENSE.txt"), "."),
    (source_file("CITATION.cff"), "."),
    (source_file("schemas/run_config_v1.json"), "schemas"),
    (source_file("docs/2D_DIC_VALIDATION.md"), "docs"),
    (source_file("docs/UI_WORKBENCH.md"), "docs"),
    (source_file("docs/validation/dic_research_20261007.json"), "docs/validation"),
    (source_file("assets/validation/plate-hole-Eyy.png"), "assets/validation"),
    (source_file("assets/ui/two-window-control.png"), "assets/ui"),
    (source_file("assets/ui/two-window-results.png"), "assets/ui"),
]
# Keep auditable source copies in the onedir bundle.  PyInstaller executes the
# modules from its archive, so these copies let the frozen smoke record stable
# source hashes even though ``module.__file__`` can point at a non-existent
# archive member rather than a filesystem file.
source_datas = [
    (source_file("dic_virtual_extensometer_gui_v7_multi_roi_range.py"), "sources"),
    (source_file("ezdic_core.py"), "sources"),
    (source_file("ezdic_cli.py"), "sources"),
    (source_file("ezdic_benchmark.py"), "sources"),
]
benchmark_datas = [
    # The JSON is runtime data; source copies are retained for audit hashes
    # when the package modules execute from the PyInstaller archive.
    (source_file("benchmarks/cases_v1.json"), "benchmarks"),
    (source_file("benchmarks/run_benchmark.py"), "sources/benchmarks"),
    (source_file("benchmarks/synthetic_cases.py"), "sources/benchmarks"),
]
common_datas = root_datas + source_datas + benchmark_datas + collect_data_files("matplotlib") + copy_metadata("scipy")
common_hiddenimports = [
    "ezdic_core",
    "ezdic_cli",
    "ezdic_benchmark",
    "benchmarks",
    "benchmarks.run_benchmark",
    "benchmarks.synthetic_cases",
]
# The source workflow uses Tk and Matplotlib's Agg/Tk backends.  A developer
# machine may have multiple optional Qt bindings installed; excluding them
# avoids PyInstaller's hard error about collecting incompatible Qt bindings and
# keeps the portable bundle independent of unrelated environments.
common_excludes = [
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    # Optional notebook/ML/parallel stacks can be installed globally but are
    # not imported by ezDIC; excluding them keeps the onedir build reproducible.
    "IPython",
    "jedi",
    "parso",
    "nbformat",
    "nbclient",
    "zmq",
    "torch",
    "numba",
    "llvmlite",
    "dask",
    "pyarrow",
    "fsspec",
    "win32com",
    "pythoncom",
    "pywintypes",
    "pytest",
]


gui_analysis = Analysis(
    [source_file("ezdic_frozen_entrypoint.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=common_datas,
    hiddenimports=common_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=common_excludes,
    noarchive=False,
    optimize=0,
)
gui_pyz = PYZ(gui_analysis.pure)
gui_exe = EXE(
    gui_pyz,
    gui_analysis.scripts,
    [],
    exclude_binaries=True,
    name='ezDIC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


cli_analysis = Analysis(
    [source_file("ezdic_cli_entrypoint.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=common_datas,
    hiddenimports=common_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=common_excludes,
    noarchive=False,
    optimize=0,
)
cli_pyz = PYZ(cli_analysis.pure)
cli_exe = EXE(
    cli_pyz,
    cli_analysis.scripts,
    [],
    exclude_binaries=True,
    name='ezDIC-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


coll = COLLECT(
    gui_exe,
    cli_exe,
    gui_analysis.binaries,
    gui_analysis.datas,
    cli_analysis.binaries,
    cli_analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ezDIC_Windows_x64",
)
