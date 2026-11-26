"""Publish a new Capture text report only after every record has been written."""

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from .output import resolve_output_path
from .storage import _publish_path


@contextmanager
def new_text_output(path, *, encoding="utf-8", newline=None):
    """Yield a sibling staging file and exclusively publish it on clean exit."""
    target = resolve_output_path(path)
    if os.path.lexists(target):
        raise FileExistsError(f"report already exists: {target}")
    partial = None
    try:
        with NamedTemporaryFile(
            mode="w", dir=target.parent, prefix=".capture-report-", suffix=".part",
            encoding=encoding, newline=newline, delete=False,
        ) as stream:
            partial = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        _publish_path(partial, target, write_through=True)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
