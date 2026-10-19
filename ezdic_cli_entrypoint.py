"""Console entrypoint used by the portable ezDIC bundle.

This small wrapper keeps the GUI executable windowed while exposing the same
headless command contract as ``python ezdic_cli.py`` in source checkouts.
"""

from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    # Import only after the process has selected the console entrypoint.  The
    # CLI module is GUI-independent and must remain usable without Tk.
    from ezdic_cli import main as cli_main

    return int(cli_main(argv))


if __name__ == "__main__":
    # PyInstaller's Windows spawn path re-enters this executable with private
    # multiprocessing arguments; handle that before the CLI parses argv.
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
