"""CLI JSON Lines remain writable on legacy Windows console code pages."""

import io
import json
import sys

from straintrace_agent.cli import _emit


def test_emit_escapes_unicode_for_cp1252_and_roundtrips_values(monkeypatch):
    byte_stream = io.BytesIO()
    cp1252_stdout = io.TextIOWrapper(byte_stream, encoding="cp1252", errors="strict", newline="\n")
    monkeypatch.setattr(sys, "stdout", cp1252_stdout)
    value = {"message": "图像读取失败", "path": r"C:\样品\原始图像.png"}

    _emit(value)

    payload = byte_stream.getvalue().decode("cp1252").strip()
    assert payload.isascii()
    assert json.loads(payload) == value
    cp1252_stdout.detach()
