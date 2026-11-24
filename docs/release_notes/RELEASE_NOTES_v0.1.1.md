# ezDIC v0.1.1

## DOI-ready archival release

This release is archived by Zenodo and has a citable DOI. It adds structured software metadata for Zenodo, improves citation information, and keeps the Windows green-folder package available for researchers who want to test ezDIC without installing Python.

DOI: [10.5281/zenodo.20222465](https://doi.org/10.5281/zenodo.20222465)

## What ezDIC does

ezDIC is a lightweight virtual extensometer for extracting linear strain from image sequences. It is intended for materials researchers who need a practical 1D strain history rather than a full-field digital image correlation analysis.

The software tracks two user-defined ROI markers and computes:

- engineering strain: `(L - L0) / L0`
- true strain: `ln(L / L0)`

where `L0` is the initial ROI-center separation and `L` is the current ROI-center separation.

## Main features

- ROI-pair tracking for virtual-extensometer strain extraction.
- Multi-ROI-group support for repeatability checks.
- Origin-compatible TXT export with `Frame`, `EngineeringStrain`, and `TrueStrain`.
- Engineering-strain PNG plots.
- QC summary with rejected frames, adaptive accepted frames, correlation scores, and QC level.
- Optional full CSV, correlation plot, overlay, and parameter exports.
- Windows 10/11 x64 green-folder executable package.

## Why this release matters

This version is archived by Zenodo so researchers can cite a stable software record. The repository now includes:

- `.zenodo.json` for Zenodo metadata.
- `CITATION.cff` for GitHub citation support.
- `LICENSE.txt` and `NOTICE_Attribution_and_Usage.txt` for attribution and usage restrictions.
- A Windows portable package attached to the GitHub release.

## How to cite

Please cite this release as:

```text
Gong, D. (2026). ezDIC: A lightweight virtual extensometer for extracting linear strain from image sequences (Version 0.1.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20222465
```

DOI: `10.5281/zenodo.20222465`

## Usage and redistribution

Developed by Dr. Delun Gong.

Users are not permitted to:

1. claim that they developed this software;
2. remove or alter the developer attribution;
3. redistribute, copy, forward, or share this software with unauthorized users;
4. use this software outside the authorized research or teaching context.

If you need to share or reuse this software, please obtain permission from Dr. Delun Gong first.

## Files

- `ezDIC_Windows_x64_v0.1.1.zip`: Windows 10/11 x64 portable package.
- Source code is available from the GitHub release archive.

## Validation

Automated checks cover:

- Origin-compatible TXT export.
- true strain recomputation from engineering strain.
- QC summary generation.
- GUI title and developer attribution.
- release metadata and PyInstaller packaging files.
