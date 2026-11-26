"""A failed Capture export must leave the requested report name available for retry."""

import csv

import pytest

from straintrace_capture import cli, synchronization
from straintrace_capture.pulse_audit import write_pulse_csv
from straintrace_capture.report_files import new_text_output


def test_an_existing_winner_survives_concurrent_publication(tmp_path):
    path = tmp_path / "report.txt"
    with pytest.raises(FileExistsError):
        with new_text_output(path) as slower:
            slower.write("partial")
            assert not path.exists()
            with new_text_output(path) as winner:
                winner.write("complete")
    assert path.read_text(encoding="utf-8") == "complete"
    assert list(tmp_path.iterdir()) == [path]


def test_interrupted_pulse_export_leaves_no_partial_table(tmp_path):
    path = tmp_path / "pulses.csv"
    with pytest.raises(KeyError):
        write_pulse_csv({"pulses": [{}]}, path)
    assert list(tmp_path.iterdir()) == []


def test_timing_export_publishes_only_after_last_frame(tmp_path, monkeypatch, capsys):
    target = tmp_path / "timing.csv"
    args = ["export-timing", str(tmp_path), "--output", str(target)]
    row = {"frame": 1, "conversion_evidence": None}

    def interrupted(*args, **kwargs):
        yield row
        raise OSError("second frame unavailable")

    monkeypatch.setattr(synchronization, "timing_rows", interrupted)
    assert cli.main(args) == 3
    assert "second frame unavailable" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(synchronization, "timing_rows", lambda *a, **kw: iter([row]))
    assert cli.main(args) == 0
    with target.open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"frame": "1", "conversion_evidence": ""}]
