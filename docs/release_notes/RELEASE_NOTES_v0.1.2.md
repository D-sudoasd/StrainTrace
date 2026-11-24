# ezDIC v0.1.2

## Citation and DOI display update

This release updates ezDIC after Zenodo DOI assignment so researchers can see and cite the software record directly from the program, release package, and GitHub page.

DOI: [10.5281/zenodo.20222465](https://doi.org/10.5281/zenodo.20222465)

## What changed

- Added DOI metadata to the program constants and window title.
- Added DOI display to the main GUI attribution area.
- Renamed the About button to `About / Citation / Usage Notice`.
- Added the recommended citation text inside the program's About dialog.
- Added DOI and citation text to `README.md`, `CITATION.cff`, `VERSION.txt`, `NOTICE_Attribution_and_Usage.txt`, `LICENSE.txt`, and `README_使用说明.txt`.
- Added a Zenodo DOI badge to the GitHub README.
- Kept the existing attribution and usage restrictions for Dr. Delun Gong.

## How to cite

Please cite ezDIC as:

```text
Gong, D. (2026). ezDIC: A lightweight virtual extensometer for extracting linear strain from image sequences (Version 0.1.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20222465
```

## Files

- `ezDIC_Windows_x64_v0.1.2.zip`: Windows 10/11 x64 portable package.
- Source code is available from the GitHub release archive.
- SHA256: `3F74BF0620560939722A3298A54989440BF577E229B49F109DF30B3ACD1AB653`

## Validation

Automated checks cover:

- Origin-compatible TXT export.
- true strain recomputation from engineering strain.
- QC summary generation.
- GUI title, DOI, citation text, and developer attribution.
- release metadata and PyInstaller packaging files.
