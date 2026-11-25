"""Check that a Windows portable bundle acquires through both industrial SDKs."""

from __future__ import annotations

import argparse
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess


def verify_bundle(bundle: Path, output: Path, cti: Path) -> dict:
    executable = bundle.resolve() / "StrainTrace-Capture-cli.exe"
    cti = cti.resolve(strict=True)
    output = output.resolve()
    output.mkdir()
    environment = {**os.environ, "PYLON_CAMEMU": "1", "PYTHONUTF8": "1"}
    for key in ("PYTHONPATH", "PYTHONHOME", "GENICAM_GENTL64_PATH"):
        environment.pop(key, None)
    windows = Path(os.environ["SystemRoot"])
    environment["PATH"] = os.pathsep.join((str(windows / "System32"), str(windows)))

    def cli(*arguments, exit_code=0):
        result = subprocess.run(
            [str(executable), *map(str, arguments)], cwd=output, env=environment,
            capture_output=True, encoding="utf-8", errors="replace", timeout=35,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode != exit_code:
            raise RuntimeError(f"Capture {arguments[0]} returned {result.returncode}: {result.stdout}\n{result.stderr}")
        return json.loads(result.stdout)

    no_cti = cli("discover", "--backend", "gentl")
    assert no_cti["providers"]["gentl"]["code"] == "GENTL_NO_CTI", no_cti
    invalid_cti = output / "invalid.cti"
    invalid_cti.touch()
    bad_cti = cli("discover", "--backend", "gentl", "--cti", invalid_cti)
    assert bad_cti["providers"]["gentl"]["code"] == "GENTL_CTI_LOAD_FAILED", bad_cti
    discovered = cli("discover", "--backend", "pylon", "--backend", "gentl", "--cti", cti)
    emulators = {}
    for device in discovered["devices"]:
        if device["support"] == "vendor_emulator":
            emulators.setdefault(device["backend"], device["id"])
    assert {"pylon", "gentl"} == emulators.keys(), discovered["providers"]

    plan = {
        "config_version": 1,
        "name": "Industrial portable smoke",
        "cti_files": [str(cti)],
        "cameras": [{"id": key, "settings": {"width": 320, "height": 240}} for key in emulators.values()],
        "phases": [{"mode": "timed", "fps": 4, "frame_count": 4, "software_trigger_fallback": True}],
    }
    path = output / "plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    invalid_plan = {**plan, "cameras": [
        {"id": emulators["gentl"], "settings": {"width": 96, "height": 240}},
    ]}
    invalid_config = output / "width-out-of-range.json"
    invalid_config.write_text(json.dumps(invalid_plan, indent=2), encoding="utf-8")
    rejected_output = output / "bad-plan"
    for command in ("preflight", "run", "trial"):
        probe = ("--probe-cameras",) if command == "preflight" else ()
        rejected = cli(command, "--config", invalid_config, "--output", rejected_output, *probe, exit_code=3)
        issue, = (row for row in rejected["issues"] if row["code"] == "CAMERA_SETTING_INVALID")
        assert issue["feature"] == "width" and issue["value"] == 96 and issue["minimum"] == 100, rejected
        assert not Path(rejected["output"]).exists()
    experiment = output / "experiment"
    checked = cli("preflight", "--config", path, "--output", experiment, "--probe-triggers")
    assert checked["ready"] and not experiment.exists()
    assert checked["write_queue"]["mode"] == "auto" and checked["write_queue"]["capacity"] == 4096
    assert checked["trigger_support"][emulators["gentl"]]["software"] is False
    final = cli("run", "--config", path, "--output", experiment)
    assert final["state"] == "completed" and final["scheduled_misses"] == 0, final
    assert final["queue_mode"] == "auto" and final["queue_capacity"] == checked["write_queue"]["capacity"]
    verified = cli("verify", experiment)
    assert verified["complete"] and verified["committed_frames"] == 8 and verified["file_hashes_checked"], verified
    groups = cli("pairing", experiment)["groups"]
    methods = {emulators["pylon"]: "software_frame_start", emulators["gentl"]: "host_next_receipt"}
    assert len(groups) == 4 and all(row["camera_methods"] == methods and row["status"] == "associated"
                                    and row["synchronization"] == "host_receipt_only" for row in groups), groups
    proof = {
        "scope": "vendor_emulators", "providers": discovered["providers"], "methods": methods,
        "cti_diagnostics": [no_cti["providers"]["gentl"]["code"], bad_cti["providers"]["gentl"]["code"]],
        "invalid_width_rejected_by": ["preflight", "run", "trial"],
        "automatic_queue_capacity": final["queue_capacity"],
        "experiment": str(experiment), "verified_tiffs": verified["committed_frames"], "request_groups": len(groups),
    }
    (output / "evidence.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path, help="new directory for the smoke experiment")
    parser.add_argument("--cti", type=Path, help="path to the Viky emulator CTI")
    args = parser.parse_args()
    cti = args.cti
    if cti is None:
        package = find_spec("genicam")
        if package is None or package.origin is None:
            parser.error("install genicam or provide the Viky emulator CTI with --cti")
        cti = Path(package.origin).with_name("viky.cti")
    print(json.dumps(verify_bundle(args.bundle, args.output, cti), indent=2))


if __name__ == "__main__":
    main()
