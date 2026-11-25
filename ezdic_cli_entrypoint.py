"""Console entrypoint used by the portable ezDIC bundle.

This small wrapper keeps the GUI executable windowed while exposing the same
headless command contract as ``python ezdic_cli.py`` in source checkouts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    if getattr(sys, "frozen", False) and Path(sys.executable).stem.casefold() == "straintrace-capture-cli":
        from straintrace_capture.cli import main as cli_main
    else:
        from ezdic_cli import main as cli_main

    return int(cli_main(argv))


if __name__ == "__main__":
    # PyInstaller's Windows spawn path re-enters this executable with private
    # multiprocessing arguments; handle that before the CLI parses argv.
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
