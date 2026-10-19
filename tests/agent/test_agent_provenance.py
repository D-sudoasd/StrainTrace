"""Source fingerprinting uses portable package copies in frozen layouts."""

import hashlib
import sys

import ezdic_core

from straintrace_agent.api import _code_fingerprint


def test_frozen_code_fingerprint_hashes_core_cli_and_all_agent_sources(tmp_path, monkeypatch):
    bundle = tmp_path / "_MEI"
    sources = bundle / "sources"
    package = sources / "straintrace_agent"
    package.mkdir(parents=True)
    core_source = sources / "ezdic_core.py"
    cli_source = sources / "ezdic_cli.py"
    core_source.write_bytes(b"auditable frozen core")
    cli_source.write_bytes(b"auditable frozen cli")
    contents = {
        "__init__.py": b"package",
        "api.py": b"api code",
        "cli.py": b"cli code",
        "fields.py": b"field code",
        "mechanics.py": b"mechanics code",
    }
    for name, payload in contents.items():
        (package / name).write_bytes(payload)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)

    fingerprint = _code_fingerprint()

    assert fingerprint["ezdic_core.py"] == hashlib.sha256(core_source.read_bytes()).hexdigest()
    assert fingerprint["ezdic_cli.py"] == hashlib.sha256(cli_source.read_bytes()).hexdigest()
    for name, payload in contents.items():
        key = f"straintrace_agent/{name}"
        assert fingerprint[key] == hashlib.sha256(payload).hexdigest()
    assert ezdic_core.resolve_source_paths(include_cli=True, include_schema=False)
