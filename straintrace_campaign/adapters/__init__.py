"""Source adapters for discovering acquisition batches and parsing logs.

Adapters return source facts and explicit uncertainties.  They never assign
physical-specimen identity from a directory name or sequence proximity.
"""

from __future__ import annotations

from typing import Any


def discover(
    root: str,
    adapter: str = "generic",
    pattern: Any = None,
    exclude: tuple[str, ...] | list[str] = (),
) -> list[dict[str, Any]]:
    """Discover batches with a named adapter.

    ``pattern`` is adapter-specific.  Generic accepts a glob or a mapping of
    configurable image patterns and counter rules; DESY accepts the same
    mapping with source-kind inference for Delun DIC and camera paths.
    """
    normalized = str(adapter).strip().casefold()
    if normalized in {"generic", "images", "image_sequence"}:
        from .generic import discover as generic_discover

        return generic_discover(root, pattern=pattern, exclude=exclude)
    if normalized in {"desy", "desy_delun", "desy_camera"}:
        from .desy import discover as desy_discover

        return desy_discover(root, pattern=pattern, exclude=exclude)
    raise ValueError(f"Unknown campaign source adapter: {adapter!r}")


__all__ = ["discover"]
