"""Module entry point for ``python -m straintrace_campaign``."""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
