# ezDIC v0.2.0-dev (unreleased development target)

Status: implemented in the current source tree as v0.2.0-dev, not formally
released. `VERSION.txt`, `CITATION.cff`, the Zenodo metadata, license, author
attribution, and DOI remain at v0.1.4. An ignored local v0.1.4 ZIP may exist,
but it was not replaced or published by this iteration; no v0.2.0 tag, release,
or publication claim is made by this note.

## Scope of this iteration

### October 2026 2D DIC reliability update

Cached quintic B-spline sampling and fourth-order reference gradients replace
the quantized subpixel sampler. GN/LM accept actual residual descent and record
undamped convergence increments. A validated SIFT/RANSAC affine initializer
extends recoverable motion while keeping per-subset acceptance checks.
Stalled IC subsets use a current-target normalized Jacobian to finish
refinement; the desktop defaults to rejecting nonconverged subsets.
Reference-fixed floating-point normalization preserves high-bit-depth changes.

Specimen masks and rectangular exclusions prevent background/hole subsets;
robust, connected local polynomial strain fitting records support, order,
condition and residual diagnostics. Original candidate displacement remains
available after spatial rejection. Missing measurements are never filled.
Continuous contours, reference/deformed backgrounds, common explicit color
limits, percent display and all nine component maps accompany TXT/CSV/NPZ
exports. GUI/core/CLI share these implementations; SciPy is included in both
Python installation metadata and the portable Windows bundle.

The current engineering gate is report v6 / cases v4, using independently
evaluated analytic targets and stricter numeric tolerances. Historical v5
evidence below remains a record of the earlier implementation. See
[the complete validation and workflow report](docs/2D_DIC_VALIDATION.md) for
matched-window results, the explicit noise/resolution tradeoff, successful
public Ncorr plate-hole imagery and insufficient-coverage GFRP imagery.
No experimental uncertainty calibration or VIC-2D equivalence is claimed.

The target keeps the existing two workflows and narrows the upgrade to the
research-use path most directly exercised here:

- fixed-reference, local-subset, in-plane 2D DIC on a POI grid;
- two-ROI 1D virtual-extensometer tracking;
- IC-GN and IC-LM with first-order affine subset warps;
- deterministic reference-percentile normalization;
- bounded coarse-to-fine multiscale initialization for large translations;
- solver and quality diagnostics, including peak separation, residual RMS,
  Hessian conditioning, iteration/stop state, and explicit invalid reasons;
- separate point-correlation `valid` and strain-fit `strain_valid` fields;
- GUI-independent source core, strict JSON configuration, and a Windows onedir
  bundle with a GUI smoke wrapper and console CLI entrypoint.

The transaction/provenance implementation has three explicit output-root
lifecycle locations: current successful `core/`, `qc/`, `optional/`, or `dic/`
outputs, recoverable sibling `_previous_runs/<run_id>/` archives, and retained
fatal-run sibling `_failed_runs/<run_id>/` staging. A sealed manifest binds the
canonical configuration, ordered input identity and SHA-256 hashes, image
shape/dtype, normalization policy and bounds, code/environment fingerprints,
output inventory and hashes, status, scientific gate, and failure semantics.
Manifest verification must report changed, missing, unexpected, or tampered
files without modifying them. Existing current outputs are archived only during
commit; a failed run does not replace them. The 1D reference initialization row
is a registration baseline and is not counted as a scientific deformation frame.

## Headless/source and Windows contract

The implemented source contract is exposed by `python ezdic_cli.py`. The strict
`run_config_v1.json` schema rejects unknown keys and non-finite values. The
stable process codes are:

| Code | Meaning |
| ---: | --- |
| 0 | success |
| 2 | configuration or command-usage failure |
| 3 | I/O, solver, export, or other runtime failure |
| 4 | scientific gate or manifest-verification failure |

The current tree implements `validate-config`, `run`, `run --progress-json`,
`verify-manifest`, and `benchmark`. A minimal complete 1D and 2D configuration
is documented in `README.md` and `README_使用说明.txt`; every key is from
`run_config_v1.json`. The source entrypoint and the portable onedir
`ezDIC-cli.exe` expose the same commands. `ezdic_frozen_entrypoint.py
--smoke-test` imports the core, CLI, benchmark, and bundled schema before any Tk
root; it performs no output write unless `EZDIC_FROZEN_SMOKE_MARKER` names a
caller-owned temporary marker.

`run --progress-json` emits line-delimited `run_started`, `progress`, and
`run_finished` events. Exit code `0` means configuration, execution,
scientific gate, and manifest verification all passed; `2` means configuration,
usage, or preflight failure; `3` means I/O, solver, export, or other runtime
failure; and `4` means scientific-gate or manifest-verification failure.

## Historical locked synthetic benchmark v5

The benchmark uses the canonical `cases-v3` definitions, fixed geometry, clean
baselines, and an image-corruption panel. The verified report is
`benchmark_report.json` with a non-empty, SHA-256-linked
`benchmark_report.csv`; its locked case hash is
`3dbe0dae3fdf8f30ec32c9fd8f036f0a53b4a705380626e7860773f62f31cb20`.

Strict clean baseline observations are:

- small translation `[2.3, -1.2]`: RMSE **0.0199391 px**, P95 **0.0292620 px**,
  max **0.0325828 px**;
- large translation `[28, -18]`: RMSE **0.0115297 px**, P95 **0.0239809 px**,
  max **0.0269902 px**, with `pyramid_levels=3`, `search_radius=8`;
- affine displacement: RMSE **0.00363440 px**, P95 **0.00651265 px**, max
  **0.0103737 px**;
- affine strain: max component error **0.000273879**, consistency error
  **0.000270908**;
- near-1D periodic texture: typed `AMBIGUOUS_TEXTURE` before solver/export,
  `solver_calls=0`, and **0** successful artifacts.

Quality-score v1 ranking covers **567 numeric solver rows**: **563 good** and
**4 bad**, including **2 rejected bad** rows. ROC-AUC is
**0.994227353463588** with error tolerance **0.25 px** (gate ≥ 0.90). The
illustrative threshold is explicitly `NOT_CALIBRATED`,
`quality_threshold_evaluated=false`, and `quality_threshold_pass=null`; there is
no calibrated binary acceptance rule. Finite-error false accepts are **2/2 =
100%**; ranking false accepts are **2/4 = 50%**. These are engineering-gate
observations, not experimental uncertainty, arbitrary-texture performance,
natural-error detection, or cross-project accuracy.
In plain terms, this quality threshold is **not calibrated**.
The canonical evidence CSV SHA-256 is
`39d4e52f35cd3161a1e877b6edcd5187568bf275c6c8d552422605b73b4c0bfb`.

## Explicit limits and follow-up work

The default 1D template is fixed reference
(`template_policy=fixed_reference`, `use_prev_frame_template=false`).
`follow_deformed_experimental` is available only when explicitly enabled and is
outside the fixed-reference claim. Full-field positive coverage gates default
to 0.95 correlation-valid fraction and 0.80 strain-valid fraction. The quality
ratio is `best_to_second_peak_ratio_min = best_peak / second_peak`; larger is
better, and it must not be inverted to `second_peak / best_peak`.

This target does not implement or claim stereo/3D DIC, DVC, GPU/MPI
acceleration, global finite-element DIC, arbitrary
experimental-texture robustness, automatic crack-topology tracking, experimental
calibration, or uncertainty quantification. Those require separate baselines,
data, and scientific acceptance gates.

## Attribution

Gong, D. (2026). ezDIC: A lightweight virtual extensometer for extracting
linear strain from image sequences (Version 0.1.4) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.20222465

See `LICENSE.txt`, `NOTICE_Attribution_and_Usage.txt`, `CITATION.cff`, and the
v0.1.4 release notes for the unchanged usage and attribution boundary.

### Independent desktop windows

The analysis console and image/results workspace are separate, independently resizable windows. ROI, mask and field views retain their state when hidden; the console reopens the workspace with Ctrl + I, and successful analysis opens its results. Quality checks and logs use separate scrollable views. Tall fields remain centered with an adjacent full-height colorbar.
