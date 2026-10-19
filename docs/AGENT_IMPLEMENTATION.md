# Agent workflow implementation contract

This document is the executable acceptance plan for the headless Agent workflow. The new package orchestrates the existing numerical core; it does not change the legacy 1D workflow, the campaign human approval contract, or DIC quality thresholds.

## Scope and data rules

- Input images remain read-only. Each image is identified by an explicit native frame counter parsed unambiguously from its filename or supplied by an explicit mapping. Duplicate, missing, or ambiguous identity is an error; mtime and list position never become physical frame identity.
- A sample project records a deterministic image inventory, finite JSON configuration, a versioned analysis plan, per-frame atomic checkpoints, quality state, derived exports, and content hashes. Code, configuration, plan, or input changes create a new run identity and preserve prior runs.
- Every deformation frame is correlated independently to one fixed reference using `ezdic_core.run_2d_dic`. ROI selection is an automatically computed **analysis rectangle candidate** based on image texture. It does not identify or confirm the physical specimen geometry.
- Raw displacement and strain remain in px and dimensionless units. Physical lengths require user-supplied `mm_per_px`; stress requires force in N and user-supplied area in mm². No field average is named or exported as engineering strain.
- Load alignment is unavailable unless the user declares a common clock and supplies a time column, time unit, explicit clock offset, and maximum interpolation gap. No extrapolation or wide-gap interpolation is allowed. Stress is unavailable without confirmed area.
- A frame failure is recorded independently and later frames continue. Unreliable frame quality remains visible; failed or unsupported values are null/NaN and never filled. `scientific_ok` is retained as a compatibility key for the existing core correlation/strain numerical quality gates only; it does not represent scientific acceptance, experimental validation, or calibrated uncertainty. Those remain explicitly `not_assessed` / `false`.
- Project locks are exclusive. A stale lock can be recovered only by an explicit command that checks the recorded PID and refuses a live owner. `status` and `verify` are read-only.

## Acceptance matrix

| ID | Workflow | Acceptance evidence |
| --- | --- | --- |
| A1 | Image directory to results | CLI initializes and inspects a project, plans a texture ROI candidate, runs fixed-reference 2D DIC without GUI, and emits inventory, config, plan, progress, checkpoints, summary, quality, plots, report, and manifest. |
| A2 | Numerical benchmark | Deterministic speckle with known subpixel translation and affine stretch is recovered within documented tolerances; calculations use the existing DIC core and thresholds. |
| A3 | Failure and recovery | Missing frame numbers remain time gaps; a corrupt image is recorded and does not stop later frames; interrupted work resumes from matching checkpoints; changed inputs/config/code cannot reuse old results; stale lock recovery requires a dead matching PID. |
| A4 | Timing and physical units | Explicit frame times and load time units/offset/gap gate synchronization; unconfirmed clocks produce unavailable mechanics; force/area produce stress in MPa only when both are declared. |
| A5 | Multiple samples | Bounded concurrent batch execution handles independent projects, reports per-sample failures, and project locks prevent simultaneous writers to one project. |
| A6 | Traceability and packaging | Read-only verification checks source hashes, code/config/run identity, checkpoint checksums, finite JSON, and package entry points; a new environment can follow the documented CLI and Python API. |

## Initial priorities and known boundaries

The first implementation delivers a reliable 2D core workflow, explicit source identity, bounded pilot-based parameter selection, restartable frame computation, timing/mechanics gates, batch orchestration, and reproducible tables/figures. Stereoscopic DIC, automatic physical specimen recognition, crack topology tracking, and experimental uncertainty calibration are outside this first delivery. Region/path extraction and advanced position–time visualization use the separate `fields.py` integration point when available.
