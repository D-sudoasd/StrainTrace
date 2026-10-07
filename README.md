<p align="center">
  <img src="assets/readme/hero.png" width="100%" alt="StrainTrace: Image-based displacement and strain. AI-generated conceptual illustration.">
</p>

# StrainTrace

**从图像序列得到两点引伸计应变，或固定参考帧下的平面二维 DIC 位移与应变场。**

A dual-mode desktop and headless workstation for image-based strain analysis. Choose a two-ROI gauge for strain histories, or a rectangular field ROI and specimen mask for local-subset in-plane digital image correlation.

[Windows / 源码启动](#windows-quick-start) · [桌面工作流](docs/UI_WORKBENCH.md) · [二维验证与参数](docs/2D_DIC_VALIDATION.md) · [输出约定](#full-field-output-contract) · [引用](#cite)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20222465.svg)](https://doi.org/10.5281/zenodo.20222465) [![MIT](https://img.shields.io/badge/License-MIT-455A64)](LICENSE.txt)

![真实双窗口工作台：图像/结果窗口显示公开板孔图像的应变场；数据和验证范围见二维DIC指南](assets/ui/two-window-results.png)

| 你需要的结果 | 分析方式 | 主要输出 |
| --- | --- | --- |
| 选定标距的应变历程 | 两个 ROI 的虚拟引伸计 | `core/`、`qc/` 与可选导出 |
| 平面位移和应变分布 | 固定参考帧、POI 网格、IC-GN / IC-LM | `dic/` 表格、数组、场图与参数 |

相关失败点保持 `NaN`，无支持区域保持空白。二维场是 POI 网格上的平面估计；数值回归与公开图像检查的范围见验证指南。

**版本选择：** 最新发布便携版为 `v0.1.3`（一维引伸计）。当前 `main` 含未发布的二维 DIC 开发实现；使用二维功能请按源码启动步骤运行。现有可执行文件名和兼容命令保留 `ezDIC`。

<details>
<summary>项目名称、开发状态与详细能力记录</summary>

This repository uses **StrainTrace** as its project name. Existing Windows
executables, CLI commands, and Zenodo citation/DOI metadata retain the name
`ezDIC`.

Main/source version: **0.1.4** · release-ready source · 2026-08-30

This source snapshot has no v0.1.4 tag or GitHub release. Any local ZIP is an
unverified working artifact; use the Releases page for a published, frozen-
smoke-verified Windows asset.

### v0.2.0 development target (not released)

The current `main` source tree contains a narrowly scoped research-grade
upgrade. It does **not** bump `VERSION.txt`, `CITATION.cff`, the Zenodo record,
or the existing DOI. The 27 September MIT preparation updates repository citation and license metadata; the archived DOI remains historical. The
implemented development capability is a reproducible fixed-reference,
local-subset, in-plane 2D DIC workflow plus the existing 1D virtual
extensometer; it is not a claim of universal DIC superiority or experimental
accuracy.

The development target adds a GUI-independent core, reference-based
percentile normalization, bounded coarse-to-fine multiscale initialization,
IC-GN/IC-LM solver diagnostics, explicit point `valid` versus strain-fit
`strain_valid`, and hash-linked run provenance. The locked synthetic benchmark
is a regression gate, not an estimate of experimental uncertainty.

The October 2026 2D DIC update adds cached quintic B-spline subpixel sampling,
validated feature initialization, robust local strain fitting, specimen masks,
and continuous contours with matching reference/deformed-image overlays.
See the [workflow, quantitative before/after results, public-data validation,
and tuning limits](docs/2D_DIC_VALIDATION.md). Display interpolation preserves
measured POIs and leaves unsupported regions blank.

StrainTrace has two explicit workflows:

1. **Virtual extensometer:** track two user-defined ROI markers and export engineering strain, true strain, QC, and Origin-compatible TXT (optional OPJU).
2. **Full-field 2D DIC:** correlate a rectangular ROI on a fixed reference frame with IC-GN or IC-LM, sample a POI grid, and export displacement and strain maps.

The full-field DIC workflow is an in-plane, local-subset 2D method. It does not replace stereo/3D DIC, DVC, or a global finite-element DIC system.

Developed by **Dr. Delun Gong** · [DOI 10.5281/zenodo.20222465](https://doi.org/10.5281/zenodo.20222465)

<p align="center">
  <img src="assets/readme/section-01-why.svg" width="100%" alt="01 Modes: virtual extensometer strain histories and full-field 2D maps.">
</p>

| You need | StrainTrace provides |
| --- | --- |
| Fast 1D strain | Two-ROI virtual extensometer |
| In-plane full-field maps | ROI and specimen mask, POI grid, IC-GN / IC-LM, nine displacement/correlation/strain components |
| A fixed comparison frame | Every full-field deformation frame is correlated to the selected reference frame |
| Honest failures | Failed points/frames remain `NaN`; a full-field run with zero valid strain points fails |
| Lab plotting | Origin-compatible TXT; optional OPJU; publication colormaps |
| Multiple gauges | Mean ± SD / SEM; Poisson-ratio roles |
| No Python for users | Portable Windows folder with `ezDIC.exe` |

```text
Virtual extensometer:  Load images → draw ROI 1 + ROI 2 → track → core/ + qc/
Full-field 2D DIC:    Load images → draw field ROI → IC-GN/IC-LM → dic/
```

</details>

## Full-field output contract

<p align="center"><img src="paper/figures/workflow.png" width="100%" alt="Virtual-extensometer and fixed-reference 2D DIC workflows and output records."></p>

Full-field analysis uses the first frame in the selected analysis range as one fixed reference. Each later frame is correlated to that same reference; the workflow does not silently switch to a frame-to-frame reference. The rectangular ROI is sampled at points of interest (POIs), so the result is a POI grid rather than a value at every image pixel. It is not a 3D measurement.

All images in the selected full-field range must have the same dimensions; a mismatch is a validation failure.

Full-field output is transactional. A new run first writes its candidate files
to a staging area. Only after the whole run completes normally are those files
committed to the `dic/` root as the current result. At commit time, old
successful `frame_####*` files are retention-moved to the output-root
`_previous_runs/<run_id>/` (a sibling of `dic/`), never silently overwritten.

The core output is fixed and is written under `dic/` for every analyzed deformation frame with a finite strain field. A failed frame is not exported as a successful result:

```text
dic/
├─ frame_0002.txt       # x, y, u, v, zncc, valid, Exx, Eyy, Exy, exx, eyy, exy, then strain_valid and solver-quality columns
├─ frame_0002.csv
├─ frame_0002_u.png
├─ frame_0002_v.png
├─ frame_0002_Exx.png
├─ frame_0002_Eyy.png
├─ frame_0002_Exy.png
├─ frame_0002_exx_infinitesimal.png    # also eyy_infinitesimal, exy_infinitesimal and zncc maps
├─ frame_0002.npz        # raw/processed POIs, mask and JSON metadata; no pickle
├─ specimen_mask.png    # when a specimen mask or exclusions are used
└─ frame_0002_parameters.txt
```

`x`, `y`, `u`, and `v` are in **px**. All strain components are **dimensionless**. `Exx`, `Eyy`, and `Exy` are Green–Lagrange components; `exx`, `eyy`, and `exy` are infinitesimal components. `Exy`/`exy` are **tensor shear components** (the off-diagonal strain terms), not engineering shear values with an extra factor of two.

Green–Lagrange component maps retain `Exx/Eyy/Exy` filenames. Infinitesimal component maps use `exx_infinitesimal/eyy_infinitesimal/exy_infinitesimal` to remain distinct on case-insensitive filesystems; CSV and NPZ component names are unchanged.

The `valid` column records whether a POI correlation passed the quality threshold. `strain_valid` is a separate strain-fit column and is exported after the 12 core measurement fields, together with solver-quality diagnostics. Failed points are exported as `NaN` for measurement fields (`u`/`v`, strain, and quality fields as applicable); `x`/`y` remain the POI grid coordinates. Failed measurements are not interpolated or filled. A frame with no finite strain field is a **normally skipped failed frame**: it contributes no current-frame files, but other valid frames may still be committed if the run has no fatal error. If no analyzed deformation frame has valid strain points, the full-field run fails and is not reported as completed.

If a fatal I/O, solver, or export exception interrupts the run, the staging
files already produced by that run are retention-moved to the output-root
`_failed_runs/<run_id>/`; they are never left in the `dic/` root as if they were
current successful output. The run reports failure rather than committing a
partial result. This is distinct from a normally skipped no-finite-strain
frame. Old successful outputs remain traceable in output-root
`_previous_runs/<run_id>/`. A first GUI run that still has pre-ledger
`dic/frame_*` files migrates those exact names once into
`dic/_previous_runs/<timestamp>/` before the core ledger takes over.

Each valid frame also gets `frame_####_parameters.txt`, a human-readable provenance record. It records the fixed `reference_frame_1based` and `reference_filename`, the analyzed `frame_global_1based` and `frame_filename`, `field_roi`, image shape, and effective `subset_size_px`, `step_px`, `strain_window`, `solver`, `zncc_min`, and smoothing settings (plus the sequence fingerprint when available).

The full-field UI exposes a visible optional **Exx overlay** switch. The overlay is written as `frame_NNNN_overlay.png` over the analyzed image and is additional visual QC, not a replacement for the core tables/maps. The 1D export controls do not participate in, and cannot gate, the fixed 2D core output. Preflight requires at least a **3 × 3 POI grid**; a field ROI that is too narrow for three rows or three columns is blocked before processing.

<p align="center">
  <img src="assets/readme/section-02-science.svg" width="100%" alt="02 Scientific contracts: px coordinates, dimensionless strain, and honest NaN.">
</p>

## Virtual-extensometer output

```text
core/      strain_*.txt · optional OPJU · strain plots
qc/        qc_summary.txt
optional/  full CSV, correlation plots, overlays, parameters, publication_figures/
```

Primary 1D TXT:

```text
Frame	EngineeringStrain	TrueStrain
```

```text
engineering strain = (L - L0) / L0
true strain        = ln(L / L0)
PoissonRatio       = - ε_transverse / ε_axial   (engineering)
```

Failed tracking frames stay `NaN`. Poisson uses role-averaged groups; tiny axial magnitude also gives `NaN`.

## Desktop workbench

The desktop uses **two independently resizable windows**. The analysis console
contains image inputs, parameters, quality checks, logs and progress. The
image/results window contains ROI drawing, masks, curves and field maps.
Closing the image window preserves its contents and allows analysis to continue;
reopen it with the console button or Ctrl + I. Successful analysis shows its
results automatically. Both windows can be moved or resized independently.

The workbench supports light/dark themes, DPI scaling, centered image previews,
scrollable zoom, and plot navigation. Field maps identify the analyzed and
reference frames, show separate displacement and strain support counts, and
retain blank invalid regions. A tall specimen remains centered with its
colorbar adjacent and at the same height.

See the [desktop workflow guide](docs/UI_WORKBENCH.md) for interactions,
shortcuts, screenshots, and the UI validation scope. The current example uses
[Ncorr's public plate-with-hole images](https://ncorr.com/download/sample12.zip);
its numerical checks are recorded in the [DIC validation guide](docs/2D_DIC_VALIDATION.md).

![Independent image/results window showing a plate-with-hole strain field](assets/ui/two-window-results.png)

## Windows quick start

1. Open the [Releases page](https://github.com/D-sudoasd/StrainTrace/releases). The latest published portable ZIP is currently **v0.1.3** (1D virtual extensometer). This `main` source tree is unreleased **v0.2.0-dev** with **v0.1.4** citation metadata; it is not a GitHub Release asset. An unverified local ZIP is not a release asset.
2. Extract the full downloaded folder; the portable package's top-level directory is `ezDIC_Windows_x64`. Keep `_internal/` next to `ezDIC.exe` and retain the license, citation, version, and notice files.
3. Run `ezDIC.exe` on Windows 10/11 x64.

For source use, either double-click `start_ezDIC.bat` or:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe dic_virtual_extensometer_gui_v7_multi_roi_range.py
```

For tests and the release-build toolchain, install the pinned build/test set:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe -m pytest -q
```

### Headless CLI (v0.2.0-dev source and frozen contract)

The implemented source CLI is GUI-independent and reads strict UTF-8 JSON
configuration against `schemas/run_config_v1.json`. Relative paths resolve next
to the configuration file; non-finite numbers, unknown keys, invalid mode
fields, and unsupported input shapes are rejected. `image_paths` and
`image_folder` are mutually exclusive. A folder is collected with StrainTrace's
natural image ordering and supported image suffixes. Keep analysis outputs and
benchmark reports outside the repository (for example under a caller-owned
temporary directory).

The following are minimal complete configurations. Replace the example image
and output paths with real paths; the field names are the schema field names.
The 1D reference must equal `start_frame_1based`; the 2D field-ROI reference
(`field_roi_reference_frame_1based`; if omitted, it defaults to
`reference_frame_1based`) must equal `reference_frame_1based`. A 2D run
processes the selected reference first for initialization, then correlates
every other selected frame to that same fixed reference.

```json
{
  "schema_version": 1,
  "analysis_mode": "extensometer",
  "image_paths": [
    "C:/data/seq/frame_001.png",
    "C:/data/seq/frame_002.png"
  ],
  "start_frame_1based": 1,
  "end_frame_1based": 2,
  "reference_frame_1based": 1,
  "output_dir": "C:/data/ezdic_1d_output",
  "roi_groups": [
    {
      "name": "axial",
      "roi1": [10, 20, 21, 21],
      "roi2": [100, 20, 21, 21],
      "strain_mode": "x",
      "role": "axial"
    }
  ],
  "tracking": {
    "template_policy": "fixed_reference",
    "use_prev_frame_template": false
  },
  "quality": {
    "min_valid_frames": 1,
    "min_strain_valid_fraction": 0.8,
    "enable_fb_check": true
  },
  "normalization": {
    "policy": "reference_percentile",
    "clip": true
  },
  "export": { "write_manifest": true },
  "transaction": { "enabled": true },
  "metadata": {}
}
```

```json
{
  "schema_version": 1,
  "analysis_mode": "fullfield",
  "image_paths": [
    "C:/data/seq/frame_001.png",
    "C:/data/seq/frame_002.png"
  ],
  "start_frame_1based": 1,
  "end_frame_1based": 2,
  "reference_frame_1based": 1,
  "output_dir": "C:/data/ezdic_2d_output",
  "field_roi": [20, 20, 120, 100],
  "solver": {
    "name": "IC-GN",
    "subset_size_px": 21,
    "step_px": 5,
    "strain_window_px": 5
  },
  "pyramid": { "levels": 1, "scale": 0.5 },
  "quality": {
    "min_correlation_valid_fraction": 0.95,
    "min_strain_valid_fraction": 0.8
  },
  "normalization": {
    "policy": "reference_percentile",
    "clip": true
  },
  "export": { "write_manifest": true },
  "transaction": { "enabled": true },
  "metadata": {}
}
```

```powershell
# Validate and execute the current source contract
py -3.11 -m ezdic_cli validate-config --config .\run_config.json
py -3.11 -m ezdic_cli run --config .\run_config.json
py -3.11 -m ezdic_cli run --config .\run_config.json --progress-json
py -3.11 -m ezdic_cli verify-manifest --manifest <output_dir>\run_manifest.json
py -3.11 -m ezdic_cli benchmark --output $env:TEMP\ezdic-benchmark
```

Exit codes are stable: `0` means configuration, execution, scientific gate,
and manifest verification all passed; `2` means configuration/usage or
preflight failure; `3` means an I/O, solver, export, or other runtime failure;
and `4` means a scientific gate or manifest-verification failure.
`run --progress-json` emits line-delimited JSON `run_started`, `progress`, and
`run_finished` events while retaining the same exit-code contract. The source
entrypoint is `python ezdic_cli.py`; the frozen onedir bundle exposes the same
contract as `ezDIC-cli.exe` alongside the windowed `ezDIC.exe`.

Headless runs use the same fixed reference and numerical core as the GUI. Their
transaction lifecycle is the output root's current `core/`, `qc/`, `optional/`,
or `dic/` files, recoverable sibling `_previous_runs/<run_id>/` archives, and
retained fatal-run sibling `_failed_runs/<run_id>/` staging. Existing current
outputs are archived only during commit; a failed run does not replace them.
The registration row in a 1D result is a reference baseline and is excluded
from the scientific deformation-frame gate. A failed POI may retain its `x`/`y`
grid coordinates, while measurement fields (`u`/`v`, strain, and applicable
quality fields) remain `NaN`.

A run manifest binds canonical configuration, ordered input identities and
hashes, image shape/dtype, normalization bounds, code/environment fingerprints,
output inventory and hashes, status, and the scientific gate.
`verify-manifest` recomputes this contract and detects changed, missing,
unexpected, or tampered manifest-listed files without modifying user data.

The optional Origin OPJU export additionally requires Windows, OriginPro 2021+, a valid local OriginPro license, and `requirements-origin.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-origin.txt
```

Build a Windows portable package with `powershell -File .\build_release.ps1`. The build script validates release metadata, runs its checks fail-closed, and copies the support files into the package.

The development portable build is an onedir package containing `ezDIC.exe`,
`ezDIC-cli.exe`, the GUI-independent core/CLI/benchmark modules, and
`schemas/run_config_v1.json`. The wrapper's `--smoke-test` path imports the
core/CLI/benchmark contract before any Tk root and performs read-only checks.
It writes a marker only when `EZDIC_FROZEN_SMOKE_MARKER` points to a
caller-provided temporary path.

The default 1D tracking policy is a fixed reference template
(`template_policy=fixed_reference`, `use_prev_frame_template=false`).
`follow_deformed_experimental` is available only when explicitly enabled and
is not part of the fixed-reference claim. For full-field quality, the
`best_to_second_peak_ratio_min` value means `best_peak / second_peak`; larger
is better and the default lower bound is 1.02. It is not the inverse
`second_peak / best_peak` ratio. The positive coverage defaults are a minimum
0.95 correlation-valid fraction and 0.80 strain-valid fraction; they can be
made stricter in a caller configuration but are never silently relaxed.

The current locked gate is **report v6 / cases v4**, with independently
evaluated continuous texture and inverse affine targets. Its case hash is
`0fb244586def285fe0ca6b816eac588ddae8d4a0de0ce1a313b183bf369db138`.
Small/large translation and affine displacement RMSE are 0.000243,
0.000143 and 0.000145 px. The displacement RMSE/P95/max gates are
0.001/0.002/0.005 px; affine strain and consistency gates are 0.0001.
CSV hashes are verified against each run's report; numerical cross-environment
baselines use explicit tolerances. Quality ranking includes 551 good and
16 rejected bad outcomes, while the illustrative threshold remains
`NOT_CALIBRATED`. See the [current validation report](docs/2D_DIC_VALIDATION.md)
for interpretation and noise/resolution tradeoffs.

The **historical v5** synthetic engineering gate recorded the following clean
baseline values (`report_version=ezdic-benchmark-report-v5`,
`cases_version=ezdic-benchmark-cases-v3`, locked case hash
`3dbe0dae3fdf8f30ec32c9fd8f036f0a53b4a705380626e7860773f62f31cb20`):

| Case | Measured value | Gate |
| --- | ---: | ---: |
| Small translation `[2.3, -1.2]` | RMSE **0.0199391 px**, P95 **0.0292620 px**, max **0.0325828 px** | strict clean baseline |
| Large translation `[28, -18]`, 3 pyramid levels, search radius 8 | RMSE **0.0115297 px**, P95 **0.0239809 px**, max **0.0269902 px** | strict clean baseline |
| Affine field | displacement RMSE **0.00363440 px**, P95 **0.00651265 px**, max **0.0103737 px** | strict clean baseline |
| Affine strain | max component error **0.000273879**, consistency error **0.000270908** | strict clean baseline |
| Near-1D periodic texture | typed `AMBIGUOUS_TEXTURE` before solver/export; `solver_calls=0`, 0 artifacts | typed rejection; artifacts = 0 |

The quality-score v1 ranking covers **567 numeric solver rows**: **563 good**
and **4 bad** labels, including **2 rejected bad** rows. It records
ROC-AUC **0.994227353463588** with error tolerance **0.25 px** (gate ≥ 0.90).
The illustrative threshold is explicitly `NOT_CALIBRATED`,
`quality_threshold_evaluated=false`, and `quality_threshold_pass=null`; it is
not a calibrated binary acceptance rule. Under finite-error labels the false
accept count is **2/2 = 100%**; under ranking outcomes it is **2/4 = 50%**.
The JSON report is hash-linked to its non-empty per-point CSV. These rates and
the ranking AUC must not be read as experimental accuracy, uncertainty, or
natural-texture performance.
In plain terms, this quality threshold is **not calibrated**.

The canonical evidence CSV SHA-256 is
`39d4e52f35cd3161a1e877b6edcd5187568bf275c6c8d552422605b73b4c0bfb`.

These are deterministic synthetic engineering-gate observations for the locked
geometry, seeds, clean baselines, and image-corruption panel. They are not
experimental uncertainty, calibration validation, or evidence that StrainTrace is
more accurate or robust than every other DIC project.

## Scientific and implementation limits

Full-field DIC is an in-plane local-subset method using IC-GN / IC-LM with a first-order affine subset warp. It reports a POI grid, not per-pixel measurements. Validated SIFT/RANSAC initialization and imported/manual/texture masks are available; automatic crack-topology tracking, stereo / 3D DIC, DVC, GPU/MPI and global finite-element DIC are not implemented. Arbitrary experimental texture is not an accepted capability claim. Pixel coordinates and displacements remain in px unless the user supplies an external calibration; strain values are dimensionless. Experimental calibration and uncertainty quantification are not implemented in this target. Failed subsets stay `NaN`, and finite output or a passing synthetic gate is not by itself experimental validation.

Relative to projects with stereo/3D, DVC, global-FE, GPU/MPI, or broader feature-guided solvers, StrainTrace deliberately does not claim those capabilities. Its intended narrow advantage is an auditable local workflow for the fixed-reference 2D tensile-image scenario: one numerical core, deterministic normalization, explicit quality/strain validity, transactional output, and reproducible synthetic gates.

## Cite

```text
Gong, D. (2026). ezDIC (v0.1.4). Zenodo. https://doi.org/10.5281/zenodo.20222465
```

See `CITATION.cff`, `NOTICE_Attribution_and_Usage.txt`, `LICENSE.txt`, and `RELEASE_NOTES_v0.1.4.md` for attribution, redistribution terms, and the release boundary.

## JOSS preparation

See the [submission guide](docs/joss/README.md) for the manuscript, verified author metadata, research-use evidence and final checks. This repository is being prepared for submission; no JOSS acceptance is claimed.

## Installable Python package

Python 3.11–3.13; desktop use requires Tk. Windows is the maintained desktop platform.
From a source checkout:

```bash
python -m pip install ".[test]"
straintrace --help
straintrace-gui
```

For a reproducible numerical check, run from a different directory:

```bash
straintrace benchmark --output straintrace-synthetic-check
```

This generates its own synthetic images and reports defined numerical gates. It is
not experimental accuracy or quality-threshold calibration. See [API](docs/API.md),
[contributing](CONTRIBUTING.md), and [support](SUPPORT.md).

## Current license and historical archives

The author authorized **MIT** licensing of the current source on 27 September 2026.
See [LICENSE.txt](LICENSE.txt). Historical ezDIC archives retain their original records;
[the historical metadata](docs/history/pre-mit/LICENSE.txt) does not govern this candidate.
The existing DOI is not an archive of the new MIT preparation. Legacy module names,
source version 0.1.4 and executable identity remain for compatibility; use the Git commit
to identify this candidate and choose a distinct final release version after review.
