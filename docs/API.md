# Headless interfaces

`straintrace --help` and `python -m ezdic_cli --help` expose the same interface.

- `validate-config --config run.json`: validate and print normalized version-1 JSON.
- `run --config run.json --progress-json`: execute a permitted image sequence.
- `verify-manifest --manifest PATH`: verify the recorded output bundle (see `--help` for syntax).
- `benchmark --output PATH`: run the locked, generated verification cases.

The schema is in `schemas/run_config_v1.json` in source and `straintrace_data` in the
wheel. The README contains complete extensometer and full-field configuration examples.
Unknown fields and non-finite configuration numbers are rejected.

Full-field configurations additionally accept `mask` (none/auto/file plus
reference-image exclusion rectangles) and `display` (contour/points, explicit
color range, percent units, colormap and reference/deformed background).
Solver options include `initialization` (auto/local), `strain_degree` (1/2),
`robust_strain`, and `outlier_threshold_px` (zero disables spatial rejection).
The legacy `strain_window_px` counts POIs, not image pixels. See
[the 2D workflow](2D_DIC_VALIDATION.md) for scientific interpretation.

Python interfaces: `ezdic_cli.normalize_config(mapping, base_dir=path)` returns the
canonical configuration; `ezdic_cli.main(argv)` returns a process exit status;
`ezdic_benchmark.run_benchmark(output_dir=path)` returns a machine-readable report.
`ezdic_core` contains the GUI-independent numerical implementation. Private helpers
are not a stability promise; prefer the versioned JSON interface for integration.

Coordinates and displacement are in pixels. Strain is dimensionless; tensor shear
must not be interpreted as engineering shear. Retain both displacement-valid and
strain-valid indicators. A verified checksum establishes integrity, not measurement
accuracy. A successful synthetic benchmark does not calibrate experimental uncertainty.
