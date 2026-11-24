# ezDIC v0.1.3

## Poisson ratio export and GUI workflow update

This release adds a direct Poisson-ratio export workflow for users who define one axial ROI group and one transverse ROI group. It also makes the main analysis action more visible for new users.

DOI: [10.5281/zenodo.20222465](https://doi.org/10.5281/zenodo.20222465)

## What changed

- Added ROI group roles: `none`, `axial`, and `transverse`.
- Added Poisson-ratio calculation from engineering strain:
  `PoissonRatio = - TransverseEngineeringStrain / AxialEngineeringStrain`.
- Added `AxialEngineeringStrain`, `TransverseEngineeringStrain`, and `PoissonRatio` columns to `strain_all_groups.txt` when axial/transverse roles are set.
- Added `poisson_ratio.txt` and `poisson_ratio.png` exports when axial/transverse roles are set.
- Kept failed tracking frames, missing strain values, and near-zero axial strain as `NaN` in Poisson-ratio output.
- Improved the right-side GUI workflow area with a compact first-screen layout, a clearer five-step beginner guide, hover tips for key workflow buttons, a prominent start button, and a separate run-status section.
- Disabled the start button during processing to reduce accidental duplicate runs.

## How to cite

Please cite ezDIC as:

```text
Gong, D. (2026). ezDIC: A lightweight virtual extensometer for extracting linear strain from image sequences (Version 0.1.3) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20222465
```

## Files

- `ezDIC_Windows_x64_v0.1.3.zip`: Windows 10/11 x64 portable package.
- Source code is available from the GitHub branch or release archive.

## Validation

Automated checks cover:

- Origin-compatible TXT export.
- true strain recomputation from engineering strain.
- Poisson-ratio calculation and NaN guards.
- GUI initialization, ROI role selection, and emphasized start action.
- release metadata and PyInstaller packaging files.
