# Agent field analysis API

`straintrace_agent.fields` adds scientific summaries, spatial extraction, and plotting for the field mapping produced by `ezdic_core.run_2d_dic`. It is a stateless layer: it does not edit the mapping, recompute DIC, smooth values, or fill missing POIs. With the source checkout on `PYTHONPATH`, import the functions directly:

```python
from straintrace_agent.fields import (
    calibrate_field, common_color_limits, iter_roi_rows, plot_field_map,
    plot_position_time, principal_strains, sample_path, summarize_roi,
    write_rows_csv,
)
```

The field supplied by `run_2d_dic` contains flattened per-POI arrays such as `x`, `y`, `u`, `v`, `valid`, `strain_valid`, `Exx`, `Eyy`, `Exy`, `exx`, `eyy`, and `exy`, plus grid arrays `X`, `Y`, `U`, `V`, and a parameter array `P` with shape `(ny, nx, 6)`. `P` is ordered `[u, du/dx, du/dy, v, dv/dx, dv/dy]`. The module accepts flattened values or values already shaped like `X`/`Y`.

## Definitions and masks

Green-Lagrange principal values use the eigenvalues of the in-plane tensor with components `Exx`, `Eyy`, `Exy`. Infinitesimal principal values use `exx`, `eyy`, `exy`. These source components, their derived shear, and principal-axis angles remain in the native **image tensor basis**: +x points right and +y points down. This remains true when `coordinate_system='calibrated'` is used to select physical positions. Calibration in `calibrate_field` derives only `x_mm`, `y_mm`, `u_mm`, and `v_mm`; it does not silently transform the strain tensor or angle. If a physical coordinate basis reverses exactly one image axis, the corresponding tensor shear sign changes under basis transformation; the displayed `Exy`/`exy` remains the image-basis value and is labeled accordingly. `Exy` and `exy` are tensor shear components. The corresponding engineering shears are returned separately as `2*Exy` and `2*exy`. The major principal-axis angle is measured from image +x (right); positive angle appears clockwise because image +y points down. When the two principal values are equal, the direction is undefined and the angle is NaN.

`principal_strains(field, measure='green_lagrange')` combines `valid` and `strain_valid`, finite-value checks, and any supported nonpositive-Jacobian mask. It returns `major`, `minor`, `theta_rad`, `theta_deg`, `tensor_shear`, `engineering_shear`, `measure`, and `valid`, with array shape matching the field grid. The same nonpositive-Jacobian mask applies to direct Green-Lagrange and infinitesimal components and their engineering-shear variants, so ROI, path, common-color-limit, and map operations use consistent support. Component retrieval returns masked values without modifying the stored field. Displacement diagnostics use correlation validity and are not excluded solely because the local Jacobian is nonpositive. A Green-Lagrange tensor alone cannot reveal the sign of `det(F)`, because it is formed from `F.T @ F`; the sign check therefore uses displacement-gradient evidence when available.

`jacobian_determinant(field)` prefers direct `dudx`, `dudy`, `dvdx`, and `dvdy` arrays. Next it uses the core `P` affine subset-warp parameters at each POI. That is a subset affine-warp diagnostic and does not equal the core's local polynomial strain fit. If neither derivative representation exists, it estimates `det(F)` at cell centers from the four original POIs in the cell. A missing corner, invalid displacement, or singular reference cell leaves NaN. The cell estimate is bilinear and approximate; it must not be reported as an additional measured DIC strain component. Both negative and zero determinants indicate a folded or degenerate local mapping and mask affected strain points; unsupported NaN determinants remain unavailable rather than being classified as foldover.

The coordinate convention of the native field is image pixels, with x increasing right and y increasing down. Calibration is optional. Physical coordinates and displacement require all four explicit entries; no origin or sign is inferred:

```python
calibrated = calibrate_field(field, {
    "mm_per_px": 0.0125,
    "origin_px": (0.0, 0.0),
    "origin_mm": (0.0, 0.0),
    "axis_sign": (1, -1),
})
field_mm = {**field, **calibrated}
# calibrated includes x_mm, y_mm, u_mm, v_mm and the calibration record.
```

`mm_per_px` must be positive and finite, origins must be explicit finite pairs, and each sign must be +1 or -1. The scalar scale assumes isotropic image calibration; if pixel scales differ by axis, this helper is not appropriate. Stress calculation is outside this API and still requires a measured cross-sectional area.

## ROI summaries and path extraction

ROI bounds are inclusive `(xmin, xmax, ymin, ymax)` in the selected coordinate system. `summarize_roi` aggregates only finite POIs passing `valid` and `strain_valid` as appropriate. For linear field components it reports `n_poi_total`, `n_poi_valid`, mean, median, sample standard deviation (`ddof=1`), min, and max. The sample SD describes point-to-point spatial scatter; it is not a standard error, confidence interval, or experimental measurement uncertainty. For principal-axis angles, which are axial quantities periodic over 180 degrees, it reports an axial circular mean and circular SD instead of linear sample statistics. An empty ROI has null statistics; one valid linear POI has a null sample SD.

Use `iter_roi_rows` when the point-level status ledger is needed. It includes invalid original POIs as rows with blank values, supports direct streaming to CSV, and labels `interpolation='none'`.

`sample_path` chooses the geometrically nearest original POI to each requested coordinate, then checks its mask and distance tolerance. It does not select a different point to get around an invalid nearest POI. Status is `valid`, `partial_invalid`, `invalid_poi`, `outside_tolerance`, or `no_support`; invalid values are `None`. Ties follow the first row-major POI. Query points and `max_distance` use the same units as the selected coordinate system.

```python
rows = sample_path(
    field,
    points=[(120.0, 80.0), (121.0, 80.0)],
    components=["u", "Exx", "small_strain_gamma_xy"],
    max_distance=2.5,
)
write_rows_csv(rows, "path.csv")
write_rows_csv(iter_roi_rows(field, "Exx", (100, 160, 50, 110)), "roi.csv")
```

CSV uses a UTF-8 BOM for convenient spreadsheet/Origin import, writes missing values as blank cells, and does not require pandas.

## Maps and spatiotemporal plots

`plot_field_map` defaults to square scatter markers at valid original POIs. This makes holes visible and does not bridge them. `method='quadmesh'` uses the original grid with masked invalid nodes and Matplotlib `shading='nearest'`; it is a display choice, not smoothing or newly measured support. For consistent frame-to-frame colors, obtain a shared range once with `common_color_limits(fields, component, symmetric=True)` and pass it as `color_limits` to each map. The helper uses finite min/max across its input; symmetric mode centers that range on zero. With no explicit `color_limits`, a map uses its own finite extent and should not be compared quantitatively to another frame's colors. Image maps display image +y down. Calibrated maps require calibration metadata and show +y up when `axis_sign[1]` is -1, and +y down when it is +1. Strain colorbar labels explicitly retain the image tensor basis (+x right, +y down), even when the spatial axes use calibrated coordinates.

`plot_position_time` consumes its `fields` iterable one frame at a time, so callers can load and release each full field rather than holding an entire experiment in memory. Pass `None` for a missing frame, and pass the acquisition times in seconds in the same order. Each supported path location becomes a scatter point at cumulative path distance and its actual time. There are no connecting lines; missing frames, invalid nearest POIs, and large time gaps stay empty. The returned summary counts missing frames, invalid nearest POIs, distance rejections, and plotted samples. For long runs, write path CSV rows incrementally instead of retaining all row dictionaries.

The plot still retains all selected scatter samples until rendering: its memory grows with the number of frames times the number of path queries. It is not a bounded-memory renderer. In particular, `agent extract --path-json` also creates this plot; avoid that combined command for millions of path samples. For a complete large extraction, stream `sample_path` rows directly to `write_rows_csv`, without calling the plotting helper. Plot an explicitly identified smaller set separately and retain the complete CSV. Do not silently interpolate, average or discard the underlying measurements to fit a figure into memory.

```python
def path_rows():
    # Load each available checkpoint lazily; identities come from the inventory.
    for native_frame, time_s, field in fields_with_identity:
        for row in sample_path(field, points, ["Exx"], max_distance=2.5):
            yield {"native_frame": native_frame, "time_s": time_s, **row}

write_rows_csv(path_rows(), "complete_available_path_samples.csv")
# Keep the inventory and explicit failed/missing-frame ledger alongside this CSV.
```

```python
fig, ax, artist, summary = plot_position_time(
    fields=iter(frame_fields),  # each field is loaded before it is yielded
    times_s=frame_times_s,
    path_points=[(120, 80), (121, 80), (122, 80)],
    component="Exx",
    max_distance=2.5,
    max_time_gap_s=1.0,
)
```

The `max_time_gap_s` value is returned as acquisition metadata; scatter markers are never connected regardless of the gap. Caller-provided times are not inferred from frame order or filesystem timestamps.

All strain values remain dimensionless. Displacements remain in pixels unless a calibrated `u_mm`/`v_mm` component is selected. The API extracts and displays the computed field; it does not certify correlation quality or scientific validity by itself. Keep the core quality summary and the input/reference/calibration provenance with any exported analysis.
