# Capture validation and handover

The software is developed on `feat/capture-workbench` from `3853196`. The development contract is `STRAINTRACE_CAPTURE_GOAL.md`. See [Capture workflows](CAPTURE.md) and [upstream source and licenses](CAPTURE_RESEARCH.md).

## Capability and evidence inventory

| Capability | Evidence and current scope |
| --- | --- |
| Lossless storage and frame identity | `tests/capture/test_capture_workflow.py` verifies the stored values, 12 significant bits in a 16-bit TIFF, 1:1 frame identity, shape, timestamp source and checksum. The camera's raw files are independent of display, lens and DIC transforms. |
| Strategies and retention | `test_capture_lifecycle.py` exercises reference and calibration roles, scheduled phases, shared software requests and a command-driven pre/post window. Capture records frame arrival, unsaved files, schedule misses and incomplete trigger groups. |
| Disk and camera failures | Lifecycle tests exercise disk reserve, byte pressure, injected loss after temporary file creation, simulated device removal, hard process termination, writer-lease recovery and orphan identification. Recoverable TIFFs retain their paths and byte values. |
| Offline DIC and analysis isolation | `test_capture_analysis.py` passes committed TIFFs to the existing 1D and 2D CLI. `test_capture_live.py` starts an independent analysis process, stalls and terminates it, then restarts the follower and checks the same reference identity and final raw frame. |
| Setup and calibration | `test_capture_calibration.py` fits a pinhole lens and a stereo rig against synthetic projected board observations with a known 52 mm baseline. The test attaches the resulting profile to a follow-on acquisition and checks its copied bytes. |
| PTP, hardware association and mechanics | `test_capture_synchronization.py` uses scripted hardware-counter metadata, independently known clock conversion, dropped pulse IDs and known forces and displacements. Hardware exposure latency and actual clock offsets are separate bench measurements. |
| Basler native adapter | Windows pypylon 26.10 CamEmu ran two devices using software FrameStart, camera ROI and Mono12 TIFFs. This is the vendor emulator. |
| GenTL adapter | Windows Harvesters 1.4.3 with genicam 1.6.0 ran two Viky devices through one producer session, wrote eight RGBa8 TIFFs, and reopened both devices for a second run. This is the GenTL emulator. |
| UVC adapter | A physical Logitech BRIO completed single, continuous and timed phases at 640 × 480 BGR8. Nine frame files passed checksum verification. DirectShow identified the camera through its host device path. |
| Analysis numerical regression | The locked source benchmark passed four cases on Windows. The benchmark retains its existing quality-threshold status and known-field scope. |
| InSituLink | The local InSituLink table reader opened the generated Capture timing CSV, retained all three fixture frame IDs and source files, and read the known UTC exposure starts and 50 ms exposure durations. The fixture uses declared synthetic sensor clocks. |
| Dual desktop workflow | A rehearsal with two simulated 640 × 480 Mono12 cameras used actual desktop controls and a separate live Analysis process. Five images from one camera were committed while the follower was paused. The operator resumed, stopped Capture and verified the hashes of all 164 committed frames. The raw pixel probe was checked against the in-memory sample. |

The [backend receipt](validation/capture_backends_20261125.json) records the three adapter runs. The [desktop and timing receipt](validation/capture_desktop_20261125.json) records the dual-workbench exercise and InSituLink import.

Windows screenshots from actual runs show [dual preview and quality checks](../assets/ui/capture-preview.png), [the phase editor after resizing to 1020 × 665](../assets/ui/capture-compact.png), [provisional live strain](../assets/ui/capture-live-analysis.png) and [committed frame review](../assets/ui/capture-review.png).

## Throughput and memory

The first simultaneous desktop rehearsal exposed schedule misses on the workspace volume. The controller now separates trigger dispatch from ledger writes and groups closed TIFFs in ledger transactions. The following runs used Windows 11, CPython 3.12.10 and an AMD Ryzen 9 9950X3D. Each throughput is total bytes committed to TIFF divided by time from acquisition setup through final drain. Cadence comes from actual host receipt timestamps. Peak RSS is for the Capture process.

| Buffered test on local C: | Target per camera | Observed per camera | Committed | Throughput | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| One 512 × 384 Mono8 | 30 Hz | 29.99 Hz | 241 | 5.62 MiB/s | 49.6 MiB |
| Two 960 × 720 Mono12 | 15 Hz | 15.07 / 15.03 Hz | 242 | 39.17 MiB/s | 67.7 MiB |
| One 1600 × 1200 Mono16 | 10 Hz | 10.01 Hz | 81 | 36.71 MiB/s | 79.3 MiB |
| Two 640 × 480 Mono12 for 60 s | 12 Hz | 12.00 / 11.98 Hz | 1,439 | 13.91 MiB/s | 75.4 MiB |

All four C: runs verified their files and hashes. On the workspace volume E: under concurrent workloads, the two larger dual-camera runs reached the configured queue limit and preserved 56 and 1,226 committed frames. The reported terminal status included `RAW_BUFFER_FULL`. The simpler E: cases completed, and the dedicated E: two-camera timed smoke received and verified all 120 expected frames. These are measured operating points for this system. [Machine-readable performance comparison](validation/capture_performance_20261125.json)

Reproduce with `python -m pip install psutil` followed by `python scripts/benchmark_capture.py --output PATH --duration 60`. The desktop QA script is `python scripts/validate_capture_desktop.py NEW_DIRECTORY` on Windows. Its captures use a synthetic speckle scene, including the screenshots.

## Windows distribution

The wheel ran from a separate Python 3.12 environment without pypylon or Harvesters. The PyInstaller onedir bundle ran from outside the source checkout. Each package saved and hash-verified five TIFFs using its packaged Capture CLI and the installation smoke plan. The portable app enumerated the simulator and two Windows UVC inputs. Its Basler and GenTL status messages identified the missing optional installations.

The portable Capture, offline Analysis and live Analysis executables each opened a native window for the saved smoke experiment and closed cleanly. The locked numerical benchmark passed all four cases on both source and frozen Analysis. The portable code copies and Capture schema match the checkout hashes. The UVC enumeration MIT notice is included alongside the library metadata.

The full Windows pytest run passed 904 tests. Three existing test cases were skipped because this Windows user lacks the privilege to create file or directory symbolic links. The run included the original Analysis, CLI, Agent, campaign and packaging contracts as well as the Capture checks collected at launch.

## Physical acceptance matrix

| Pending bench setup | Acceptance steps |
| --- | --- |
| Physical Basler and FLIR/Teledyne, Hikrobot, Allied Vision and Daheng Imaging industrial cameras | Install the vendor SDK and matching producer. Enumerate serials. Set legal and illegal values at a nonzero sensor ROI and validate ranges, access and readback. Compare a lossless test pattern with the vendor viewer at 8 and 12/16 bits. Record requested and delivered native cadence. Disconnect a camera under capture, verify the ledger and start a linked acquisition after reconnecting. |
| Shared TTL and mixed hardware lines | Wire a DAQ or calibrated generator to the camera FrameStart inputs according to the model-specific electrical requirements. Record pulses in the DAQ ledger. Observe the first shared pulse and each camera's per-frame counter. Fit the explicit `hardware_pairing` anchors. Check a fixed-count sequence against the pulse log, inject one missing exposure and inspect missing and ambiguous associations. |
| PTP, exposure clocks and beamline detector | Use the real camera clock on a PTP-capable link and a recorded reference clock. Measure and export the time conversion interval and uncertainty. Observe the pulses or independent timing reference and measure exposure start, duration and inter-camera skew. Compare `export-timing` and `link-detector` with the logged detector and DAQ identities. Leave unavailable clocks and unmatched detector frames visible. |
| Physical mechanics and optical accuracy | Use the rig's DAQ/test frame and a stationary certified checkerboard. Validate capture against an independently recorded force/displacement standard with a stated maximum gap, clock origin and uncertainty. Check the mono and stereo fit with independent board poses, baseline and sensor ROI. Confirm physical DIC strain accuracy with an independent displacement or strain reference. |

Laboratory-specific disk throughput, on-camera acquisition rates, PTP lock, shared trigger skew and optical accuracy each require measurements on the intended rig.

The Basler adapter accepts unpacked PFNC formats. Packed Basler input and GenTL multipart payloads are outside the current implementation. See the particular camera's reported formats before starting a run.
