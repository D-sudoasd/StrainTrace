# StrainTrace Capture

Capture runs the cameras and saves images. Analysis opens an existing image folder or reads a Capture experiment in a separate process. The two workbenches can run independently or together.

## Install and start

| Workbench | Installed command | Windows portable | Source checkout |
| --- | --- | --- | --- |
| Capture desktop | `straintrace-capture-gui` | `StrainTrace-Capture.exe` | `start_Capture.bat` or `python -m straintrace_capture.gui` |
| Capture CLI | `straintrace-capture` | `StrainTrace-Capture-cli.exe` | `python -m straintrace_capture` |
| Analysis desktop | `straintrace-gui` | `ezDIC.exe` | `start_ezDIC.bat` |
| Analysis CLI | `straintrace` | `ezDIC-cli.exe` | `python -m ezdic_cli` |

Install the ordinary source package with `python -m pip install .`. Optional extras are `".[capture-uvc]"` for device-path enumeration, `".[capture-pylon]"` for Basler and `".[capture-gentl]"` for the vendor GenTL route. Use the vendor's matching 64-bit installation for physical devices. Choose installed CTIs in Capture or provide `--cti` to discovery and `cti_files` in the plan. The portable app bundles simulation, UVC and Analysis. Use the source installation for industrial SDKs.

## Desktop workflow

1. Choose one or more devices. Use Ctrl-click to extend the selection. Open **Camera settings** to query ranges, units and access, then apply the desired sensor geometry, gain and exposure. Start **Connect & preview** to inspect the scene.
2. Use **Fit**, **1:1**, wheel zoom and drag to inspect the field. Shift-drag marks a local inspection region. The sensor ROI affects recorded pixels. The viewport and inspection rectangle affect display only. Open Analysis to define numerical ROIs in the committed raw image coordinates.
3. Select a new output folder and edit the **Plan**. Use `reference`, `calibration` and `test` roles. A single reference followed by timed test phases is a typical plan. Review the camera and disk readouts while recording.
4. Use **Review** to browse complete images, verify their checksums or open Analysis. **Open Live Analysis** launches another process. Define the gauge or field ROI in its committed reference, then choose **Follow**. Pause, exit or reopen the follower while Capture continues.

## Capture strategies

| Mode | Operation |
| --- | --- |
| `single` | One requested image per camera. Use role `reference` or `calibration` as needed. |
| `timed` | Scheduled software requests at `fps`, through a count or duration limit. The scheduler reports missed intervals. |
| `continuous` | Save the delivered stream. Set camera `fps` where supported. Optional count or duration limits stop the phase. |
| `manual` | Wait for the operator or an allowed local command. Multiple calibration views can be acquired in one manual phase. |
| `event` | Retain a rolling window of `pre_frames` per camera, then save the requested `post_frames` after a signal at the configured cadence. |
| `hardware` | Arm a camera's physical FrameStart line. The camera receives the external pulses. Provide `hardware_pairing` when tested counters are known to refer to a shared pulse stream. |

Capture types share the same versioned configuration in the GUI, Python and CLI. A phase with both limits ends on the first limit. UVC single, timed, manual and event modes select the next driver-delivered image after a request. Some UVC drivers provide decoded color and report no exposure clock or hardware counter. Read-backed camera parameters and phase trigger settings are recorded in the ledger.

## CLI and Python

Use a local, dedicated volume for the experiment directory. The SQLite WAL and raw files remain in the directory during acquisition.

```powershell
python -m straintrace_capture discover
python -m straintrace_capture features sim:0
python -m straintrace_capture validate-config examples/capture/speckle.json
python -m straintrace_capture run --config examples/capture/speckle.json --output D:/runs/demo --progress
python -m straintrace_capture verify D:/runs/demo
python -m straintrace_capture prepare-analysis D:/runs/demo --output D:/runs/demo-binding.json
python -m straintrace_capture follow D:/runs/demo --config examples/capture/live-rois.json
```

For local operator or DAQ input, enable `--control` when starting the session, then run `straintrace-capture trigger D:/runs/demo --source DAQ` or `straintrace-capture stop D:/runs/demo`. Command receipts are in the experiment's `control` directory. Authorized remote operators can run the command through their existing login on the Capture host. Capture does not open a network listener. A camera request and the resulting trigger association are separate entries. For confirmed load/displacement samples, `ThresholdTrigger.observe` in the Python API can issue a request on a threshold crossing.

```python
from straintrace_capture import CaptureConfig, CaptureSession, ExperimentReader

session = CaptureSession(CaptureConfig.load("examples/capture/speckle.json"), "D:/runs/demo")
experiment_id = session.start()
final = session.wait()
with ExperimentReader(session.output) as reader:
    first = reader.frames(camera_id="sim:0", limit=1)[0]
    raw_tiff = reader.image_path(first)
```

## Raw files, integrity and timing

- `capture.sqlite3` contains the versioned experiment, camera, event and committed-frame records. The TIFFs under `images/` retain the delivered unsigned pixel samples. For 10–14-bit samples, the TIFF container uses 16 bits and the record preserves the significant bit count. Bayer arrays retain their mosaic codes and the record gives the pixel format. In offline Capture handoff, `prepare-analysis --derive-color NEW_FOLDER` prepares Bayer luminance or luminance from RGBa without applying alpha. The Analysis desktop offers the same separate conversion. Four-channel storage supports 8-bit samples.
- Each committed row includes its stable experiment/camera/sequence identity, optional device counter, trigger relation, role, dimensions, dtype and hash. A reader follows only committed rows. TIFFs or partial files left between disk publication and ledger commit are reported. `recover` marks an interrupted ledger after the writer lease has released and retains all observed files. `run --continue-from` creates a linked experiment in a new directory.
- Device ticks, host UTC receipt, host monotonic receipt, operator/DAQ command time and exposure markers have distinct meanings. Exposure midpoint is derived from compatible measured endpoints when present. `export-timing` defaults to exposure midpoint and leaves unavailable UTC time blank. `--time-basis host-receive` explicitly selects a time proxy. Supply bounded measured `--clock-anchors` for device-tick conversion. `align-load` uses the existing campaign gap, unit and invalid-sample rules. `link-detector` joins an event-number table to explicit shared pulse identities and shows unmatched rows on both sides. Add a `mechanics` object to the live configuration, or load it from the live window, to include aligned force alongside the provisional strains.
- A shared software request groups camera results but does not measure exposure skew. Common pulse pairing requires a validated chunk counter and its anchor on each camera. PTP node state and raw clock readings remain available when reported by the device. Record measured clock agreement and TTL latency for the rig in use.
- `segment_frames` optionally places consecutive ranges in separate per-camera subdirectories. Sequence numbers and camera identities continue across segments. By default, Capture closes and atomically publishes TIFFs, then durably commits batches of ledger rows. The `write_through` option also flushes the TIFF bytes before publication. It increases disk latency. Verify the chosen setting on the intended volume and use a power-protected system if required by the experiment.
- The pre-event window is relative to receipt of the command or DAQ signal by Capture. A batch crossing that boundary is saved and labeled `boundary`. A short pre-window remains visible in the verification result.

## Calibration

Use the calibration image role for a series of different checkerboard poses. With two cameras, hold the board stationary for each pair and use a board with one odd and one even inner-corner count. `calibrate` fits the pinhole model and, for two selected cameras, the extrinsics. It reports view IDs, checksums and reprojection fit diagnostics. Add the resulting profile to `calibration_files` for a subsequent run to retain a checksum-verified copy with the experiment. Calibration uses image coordinates and the specified checkerboard spacing in millimeters. A low fit residual still requires independent physical validation. The **Lens calibration** menu toggles the corrected viewport. `prepare-analysis --lens-profile PROFILE --derived-folder NEW_FOLDER` saves separate correction TIFFs with the raw and calibration hashes.

See [source research and license inventory](CAPTURE_RESEARCH.md) and [validation and physical acceptance](CAPTURE_VALIDATION.md).
