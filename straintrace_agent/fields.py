"""Scientific summaries, point extraction, and plotting for 2-D DIC fields.

The functions operate on the field mapping returned by ``ezdic_core.run_2d_dic``.
They preserve its point support and masks: no interpolation, smoothing, or
extrapolation is performed here.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


_STRAIN_COMPONENTS = {
    "Exx", "Eyy", "Exy", "exx", "eyy", "exy",
    "green_lagrange_gamma_xy", "small_strain_gamma_xy",
    "principal_max", "principal_min", "principal_angle_deg",
    "principal_max_GL", "principal_min_GL", "principal_angle_GL_deg",
    "principal_max_infinitesimal", "principal_min_infinitesimal",
    "principal_angle_infinitesimal_deg",
}


def _shape(field: Mapping) -> tuple[int, ...]:
    grid = field.get("X")
    if grid is not None and np.asarray(grid).ndim == 2:
        return np.asarray(grid).shape
    for key in ("x", "y", "u", "Exx"):
        if key in field:
            return np.asarray(field[key]).shape
    raise ValueError("field must contain x/y POI coordinates or an X/Y grid")


def _array(field: Mapping, key: str, *, required: bool = True, dtype=float) -> np.ndarray | None:
    if key not in field or field[key] is None:
        if required:
            raise ValueError(f"field is missing required array {key!r}")
        return None
    values = np.asarray(field[key], dtype=dtype)
    shape = _shape(field)
    if values.shape == shape:
        return values
    if values.size == int(np.prod(shape)):
        return values.reshape(shape)
    raise ValueError(f"field array {key!r} has shape {values.shape}; expected {shape} or {int(np.prod(shape))} values")


def _coordinates(field: Mapping, coordinate_system: str = "image") -> tuple[np.ndarray, np.ndarray]:
    if coordinate_system == "image":
        xkey, ykey = "X", "Y"
        if xkey not in field or ykey not in field:
            xkey, ykey = "x", "y"
    elif coordinate_system == "calibrated":
        xkey, ykey = "x_mm", "y_mm"
    else:
        raise ValueError("coordinate_system must be 'image' or 'calibrated'")
    x = _array(field, xkey).astype(float, copy=False)
    y = _array(field, ykey).astype(float, copy=False)
    if x.shape != y.shape:
        raise ValueError("x and y coordinate arrays must have identical shapes")
    return x, y


def _finite_pair(value, name: str) -> tuple[float, float]:
    raw = np.asarray(value, dtype=object).ravel()
    if any(isinstance(item, (bool, np.bool_)) for item in raw):
        raise ValueError(f"{name} must contain two finite numbers, not booleans")
    try:
        pair = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite numbers") from exc
    if pair.size != 2 or not np.isfinite(pair).all():
        raise ValueError(f"{name} must contain two finite numbers")
    return float(pair[0]), float(pair[1])


def _has_jacobian_support(field: Mapping) -> bool:
    if "X" not in field or "Y" not in field:
        return False
    return (
        "deformation_jacobian" in field
        or all(key in field for key in ("dudx", "dudy", "dvdx", "dvdy"))
        or "P" in field
        or all(key in field for key in ("U", "V"))
        or all(key in field for key in ("u", "v"))
    )


def _field_mask(field: Mapping, component: str, values: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values)
    if component in _STRAIN_COMPONENTS:
        for key in ("valid", "strain_valid"):
            value = _array(field, key, required=False, dtype=bool)
            if value is not None:
                mask &= value
        if _has_jacobian_support(field):
            mask &= ~nonpositive_jacobian_poi_mask(field)
    elif component in {"u", "v", "u_raw", "v_raw", "u_mm", "v_mm"}:
        value = _array(field, "valid", required=False, dtype=bool)
        if value is not None:
            mask &= value
    return mask


def _base_strain(field: Mapping, measure: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if measure in {"green_lagrange", "GL", "gl"}:
        keys = ("Exx", "Eyy", "Exy")
    elif measure in {"infinitesimal", "small", "infinitesimal_strain"}:
        keys = ("exx", "eyy", "exy")
    else:
        raise ValueError("measure must be 'green_lagrange' or 'infinitesimal'")
    return tuple(_array(field, key).astype(float, copy=False) for key in keys)


def jacobian_determinant(field: Mapping) -> np.ndarray:
    """Return a det(F) diagnostic using the most direct available gradients.

    The core's strain-fit determinant takes precedence, followed by direct
    derivative arrays. The core's ``P`` array is next;
    its entries use ``[u, du/dx, du/dy, v, dv/dx, dv/dy]`` and describe the
    subset affine warp, so that diagnostic is distinct from the local
    polynomial strain fit. If neither is supplied, a bilinear estimate is
    made at grid-cell centers from four original POIs. Unsupported results
    remain NaN. The result is not a calibrated uncertainty estimate.
    """
    X, Y = _coordinates(field, "image")
    if X.ndim != 2 or min(X.shape) < 2:
        raise ValueError("Jacobian estimation requires a two-dimensional X/Y POI grid with at least 2 rows and columns")
    if "deformation_jacobian" in field:
        det = _array(field, "deformation_jacobian").astype(float, copy=False)
        valid = _array(field, "valid", required=False, dtype=bool)
        if valid is not None:
            det = np.where(valid, det, np.nan)
        return np.where(np.isfinite(det), det, np.nan)
    direct_keys = ("dudx", "dudy", "dvdx", "dvdy")
    if all(key in field for key in direct_keys):
        dudx, dudy, dvdx, dvdy = (_array(field, key).astype(float, copy=False) for key in direct_keys)
        det = (1.0 + dudx) * (1.0 + dvdy) - dudy * dvdx
        valid = _array(field, "valid", required=False, dtype=bool)
        if valid is not None:
            det = np.where(valid, det, np.nan)
        return np.where(np.isfinite(det), det, np.nan)
    if "P" in field:
        p = np.asarray(field["P"], dtype=float)
        if p.shape == X.shape + (6,):
            pass
        elif p.size == int(np.prod(X.shape)) * 6:
            p = p.reshape(X.shape + (6,))
        else:
            raise ValueError("P must have shape (ny, nx, 6) or contain six parameters per POI")
        det = (1.0 + p[..., 1]) * (1.0 + p[..., 5]) - p[..., 2] * p[..., 4]
        valid = _array(field, "valid", required=False, dtype=bool)
        supported = np.isfinite(p).all(axis=-1)
        if valid is not None:
            supported &= valid
        return np.where(supported, det, np.nan)
    # Raw solver displacements are preferred for a foldover check when the core
    # also exposes its optional-smoothed strain input as U/V.
    ukey = "u_raw" if "u_raw" in field else ("U" if "U" in field else "u")
    vkey = "v_raw" if "v_raw" in field else ("V" if "V" in field else "v")
    U, V = _array(field, ukey), _array(field, vkey)
    valid = _array(field, "valid", required=False, dtype=bool)
    support = np.isfinite(X) & np.isfinite(Y) & np.isfinite(U) & np.isfinite(V)
    if valid is not None:
        support &= valid

    ny, nx = X.shape
    determinant = np.full((ny - 1, nx - 1), np.nan, dtype=float)
    for i in range(ny - 1):
        for j in range(nx - 1):
            corners = ((i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1))
            if not all(support[row, col] for row, col in corners):
                continue
            ref = np.array([
                [X[i, j], X[i, j + 1], X[i + 1, j], X[i + 1, j + 1]],
                [Y[i, j], Y[i, j + 1], Y[i + 1, j], Y[i + 1, j + 1]],
            ])
            disp = np.array([
                [U[i, j], U[i, j + 1], U[i + 1, j], U[i + 1, j + 1]],
                [V[i, j], V[i, j + 1], V[i + 1, j], V[i + 1, j + 1]],
            ])
            # Bilinear derivatives at the cell center, columns are grid-column
            # and grid-row directions. det(F)=det(J_deformed)/det(J_reference).
            dref = np.array([
                [(-ref[0, 0] + ref[0, 1] - ref[0, 2] + ref[0, 3]) / 4,
                 (-ref[0, 0] - ref[0, 1] + ref[0, 2] + ref[0, 3]) / 4],
                [(-ref[1, 0] + ref[1, 1] - ref[1, 2] + ref[1, 3]) / 4,
                 (-ref[1, 0] - ref[1, 1] + ref[1, 2] + ref[1, 3]) / 4],
            ])
            dcurrent = dref + np.array([
                [(-disp[0, 0] + disp[0, 1] - disp[0, 2] + disp[0, 3]) / 4,
                 (-disp[0, 0] - disp[0, 1] + disp[0, 2] + disp[0, 3]) / 4],
                [(-disp[1, 0] + disp[1, 1] - disp[1, 2] + disp[1, 3]) / 4,
                 (-disp[1, 0] - disp[1, 1] + disp[1, 2] + disp[1, 3]) / 4],
            ])
            det_ref = float(np.linalg.det(dref))
            if abs(det_ref) > np.finfo(float).eps:
                determinant[i, j] = float(np.linalg.det(dcurrent) / det_ref)
    return determinant


def nonpositive_jacobian_poi_mask(field: Mapping) -> np.ndarray:
    """Mark POIs where either fitted or subset-warp mapping folds.

    These describe different spatial supports. When the strain-fit
    determinant is available, do not discard a contradictory fold in P.
    """
    cells = jacobian_determinant(field)
    points = np.zeros(_shape(field), dtype=bool)
    if cells.shape == points.shape:
        points |= np.isfinite(cells) & (cells <= 0)
    else:
        for i, j in zip(*np.nonzero(np.isfinite(cells) & (cells <= 0))):
            points[i:i + 2, j:j + 2] = True
    if "deformation_jacobian" in field and "P" in field:
        p = np.asarray(field["P"], dtype=float)
        if p.size != points.size * 6:
            raise ValueError("P must contain six parameters per POI")
        p = p.reshape(points.shape + (6,))
        subset_det = (1.0 + p[..., 1]) * (1.0 + p[..., 5]) - p[..., 2] * p[..., 4]
        valid = _array(field, "valid", required=False, dtype=bool)
        support = np.isfinite(p).all(axis=-1)
        if valid is not None:
            support &= valid
        points |= support & (subset_det <= 0)
    return points


def negative_jacobian_poi_mask(field: Mapping) -> np.ndarray:
    """Compatibility name; includes negative and zero-Jacobian POIs."""
    return nonpositive_jacobian_poi_mask(field)


def principal_strains(
    field: Mapping,
    measure: str = "green_lagrange",
    *,
    angle_tolerance: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Return in-plane principal strains and the major-axis orientation.

    The supplied strain components and returned principal-axis angles remain
    in the native image tensor basis (x right, y down), regardless of whether
    spatial coordinates have also been calibrated. ``Exy`` and ``exy`` are
    tensor shear components. The corresponding engineering shears are
    ``2*Exy`` (Green-Lagrange) and ``2*exy``
    (infinitesimal). The angle is measured from image +x (right); since image
    +y points down, positive angles appear clockwise on an image. Its direction
    is undefined for equal principal values and is returned as NaN there.
    Values at invalid strain POIs, and at POIs with supported nonpositive
    det(F), are masked as NaN.
    """
    if not np.isfinite(angle_tolerance) or angle_tolerance < 0:
        raise ValueError("angle_tolerance must be finite and nonnegative")
    a, d, b = _base_strain(field, measure)
    mean = (a + d) / 2.0
    radius = np.hypot((a - d) / 2.0, b)
    major, minor = mean + radius, mean - radius
    angle = 0.5 * np.arctan2(2.0 * b, a - d)
    valid = _field_mask(field, "Exx" if measure in {"green_lagrange", "GL", "gl"} else "exx", a)
    valid &= np.isfinite(d) & np.isfinite(b)
    angle_defined = np.abs(major - minor) > angle_tolerance * np.maximum(1.0, np.maximum(np.abs(major), np.abs(minor)))
    valid &= np.isfinite(major) & np.isfinite(minor)
    major = np.where(valid, major, np.nan)
    minor = np.where(valid, minor, np.nan)
    angle = np.where(valid & angle_defined, angle, np.nan)
    return {
        "major": major,
        "minor": minor,
        "theta_rad": angle,
        "theta_deg": np.degrees(angle),
        "tensor_shear": np.where(valid, b, np.nan),
        "engineering_shear": np.where(valid, 2.0 * b, np.nan),
        "measure": "green_lagrange" if measure in {"green_lagrange", "GL", "gl"} else "infinitesimal",
        "valid": valid,
    }


def component_values(field: Mapping, component: str) -> np.ndarray:
    """Return a component with scientific validity masks applied to strains."""
    if component in field and isinstance(field[component], (np.ndarray, list, tuple)):
        values = _array(field, component).astype(float, copy=False)
        if component in _STRAIN_COMPONENTS:
            values = np.where(_field_mask(field, component, values), values, np.nan)
        return values
    aliases = {
        "principal_max": "major",
        "principal_min": "minor",
        "principal_angle_deg": "theta_deg",
        "principal_max_GL": "major",
        "principal_min_GL": "minor",
        "principal_angle_GL_deg": "theta_deg",
        "principal_max_infinitesimal": "major",
        "principal_min_infinitesimal": "minor",
        "principal_angle_infinitesimal_deg": "theta_deg",
    }
    if component in {"green_lagrange_gamma_xy", "small_strain_gamma_xy"}:
        source = "Exy" if component.startswith("green") else "exy"
        values = 2.0 * _array(field, source).astype(float, copy=False)
        return np.where(_field_mask(field, source, values), values, np.nan)
    if component in aliases:
        measure = "infinitesimal" if "infinitesimal" in component else "green_lagrange"
        return principal_strains(field, measure)[aliases[component]]
    raise ValueError(f"unknown field component {component!r}")


def component_unit(component: str) -> str:
    if component in {"u", "v", "u_raw", "v_raw"}:
        return "px"
    if component in {"u_mm", "v_mm"}:
        return "mm"
    if component in {"principal_angle_deg", "principal_angle_GL_deg", "principal_angle_infinitesimal_deg"}:
        return "deg"
    if component in _STRAIN_COMPONENTS:
        return "dimensionless"
    return "unknown"


def _component_valid(field: Mapping, component: str, values: np.ndarray) -> np.ndarray:
    if component in {"principal_max", "principal_min", "principal_angle_deg",
                     "principal_max_GL", "principal_min_GL", "principal_angle_GL_deg",
                     "principal_max_infinitesimal", "principal_min_infinitesimal",
                     "principal_angle_infinitesimal_deg"}:
        measure = "infinitesimal" if "infinitesimal" in component else "green_lagrange"
        principal = principal_strains(field, measure)
        return principal["valid"] & np.isfinite(values)
    if component in {"green_lagrange_gamma_xy", "small_strain_gamma_xy"}:
        source = "Exy" if component.startswith("green") else "exy"
        return _field_mask(field, source, values)
    return _field_mask(field, component, values)


def calibrate_field(field: Mapping, calibration: Mapping) -> dict[str, np.ndarray | dict]:
    """Derive calibrated coordinates/displacements using explicit calibration.

    Required calibration keys are ``mm_per_px``, ``origin_px``, ``origin_mm``,
    and ``axis_sign``. The two signs must each be +1 or -1; no origin or axis
    convention is inferred. This derives physical positions and displacement
    only; it does not transform strain components or principal angles out of
    the image tensor basis. The source field mapping is never modified.
    """
    required = ("mm_per_px", "origin_px", "origin_mm", "axis_sign")
    missing = [key for key in required if key not in calibration]
    if missing:
        raise ValueError("calibration is missing explicit field(s): " + ", ".join(missing))
    try:
        scale = float(calibration["mm_per_px"])
    except (TypeError, ValueError) as exc:
        raise ValueError("mm_per_px must be positive and finite") from exc
    if isinstance(calibration["mm_per_px"], (bool, np.bool_)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("mm_per_px must be positive and finite")
    origin_px = _finite_pair(calibration["origin_px"], "origin_px")
    origin_mm = _finite_pair(calibration["origin_mm"], "origin_mm")
    signs = _finite_pair(calibration["axis_sign"], "axis_sign")
    if any(sign not in (-1.0, 1.0) for sign in signs):
        raise ValueError("axis_sign entries must each be +1 or -1")
    x, y = _coordinates(field, "image")
    sx, sy = signs
    calibrated = {
        "x_mm": origin_mm[0] + sx * (x - origin_px[0]) * scale,
        "y_mm": origin_mm[1] + sy * (y - origin_px[1]) * scale,
        "calibration": {
            "mm_per_px": scale,
            "origin_px": origin_px,
            "origin_mm": origin_mm,
            "axis_sign": signs,
        },
    }
    for key, sign in (("u", sx), ("v", sy)):
        source_key = "u_raw" if key == "u" and "u_raw" in field else key
        source_key = "v_raw" if key == "v" and "v_raw" in field else source_key
        raw = _array(field, source_key, required=False)
        if raw is not None:
            calibrated[key + "_mm"] = raw * sign * scale
    return calibrated


def _validate_roi(roi: Sequence[float]) -> tuple[float, float, float, float]:
    try:
        values = np.asarray(roi, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError("roi must be (xmin, xmax, ymin, ymax)") from exc
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError("roi must contain four finite values: (xmin, xmax, ymin, ymax)")
    xmin, xmax, ymin, ymax = map(float, values)
    if xmin > xmax or ymin > ymax:
        raise ValueError("roi bounds must be ordered")
    return xmin, xmax, ymin, ymax


def _inside_roi(x: np.ndarray, y: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    """Inclusive reference coordinates, allowing only unit-conversion roundoff."""
    xmin, xmax, ymin, ymax = bounds
    epsilon = 8 * np.finfo(float).eps
    xtol = epsilon * max(1.0, abs(xmin), abs(xmax))
    ytol = epsilon * max(1.0, abs(ymin), abs(ymax))
    return (np.isfinite(x) & np.isfinite(y)
            & (x >= xmin - xtol) & (x <= xmax + xtol)
            & (y >= ymin - ytol) & (y <= ymax + ytol))


def iter_roi_rows(
    field: Mapping,
    component: str,
    roi: Sequence[float],
    *,
    coordinate_system: str = "image",
) -> Iterable[dict]:
    """Yield point-level CSV rows for an inclusive ROI, including masked POIs."""
    xmin, xmax, ymin, ymax = _validate_roi(roi)
    x, y = _coordinates(field, coordinate_system)
    values = component_values(field, component)
    valid = _component_valid(field, component, values)
    inside = _inside_roi(x, y, (xmin, xmax, ymin, ymax))
    for index in np.flatnonzero(inside.ravel()):
        ok = bool(valid.ravel()[index])
        yield {
            "poi_index": int(index),
            "x": float(x.ravel()[index]),
            "y": float(y.ravel()[index]),
            "coordinate_system": coordinate_system,
            "component": component,
            "value": float(values.ravel()[index]) if ok else None,
            "value_unit": component_unit(component),
            "status": "valid" if ok else "invalid_poi",
            "interpolation": "none",
        }


def summarize_roi(
    field: Mapping,
    component: str,
    roi: Sequence[float],
    *,
    coordinate_system: str = "image",
) -> dict:
    """Summarize only valid original POIs inside inclusive coordinate bounds.

    ``sd_sample`` uses ddof=1 and describes spatial/point scatter. It is not a
    standard error or an experimental uncertainty estimate.
    """
    rows = iter_roi_rows(field, component, roi, coordinate_system=coordinate_system)
    n_total = n_valid = 0
    samples = []
    for row in rows:
        n_total += 1
        if row["status"] == "valid":
            n_valid += 1
            samples.append(row["value"])
    return {
        "component": component,
        "unit": component_unit(component),
        "roi": _validate_roi(roi),
        "coordinate_system": coordinate_system,
        "n_poi_total": n_total,
        "n_poi_valid": n_valid,
        **_summarize_samples(samples, component),
        "source": "original_poi",
        "interpolation": "none",
    }


def _summarize_samples(samples: Sequence[float], component: str) -> dict:
    """Shared spatial statistics for single and multiple original-POI regions."""
    data = np.asarray(samples, dtype=float)
    n_valid = data.size
    is_angle = component in {
        "principal_angle_deg", "principal_angle_GL_deg", "principal_angle_infinitesimal_deg",
    }
    mean = median = sd_sample = minimum = maximum = None
    sd_axial = None
    if n_valid and is_angle:
        radians = np.deg2rad(data)
        cosine, sine = float(np.mean(np.cos(2 * radians))), float(np.mean(np.sin(2 * radians)))
        resultant = math.hypot(cosine, sine)
        if resultant > np.finfo(float).eps:
            mean = math.degrees(0.5 * math.atan2(sine, cosine))
            mean = (mean + 90.0) % 180.0 - 90.0
            sd_axial = math.degrees(0.5 * math.sqrt(max(0.0, -2.0 * math.log(min(resultant, 1.0)))))
    elif n_valid:
        mean = float(np.mean(data))
        median = float(np.median(data))
        sd_sample = float(np.std(data, ddof=1)) if n_valid >= 2 else None
        minimum = float(np.min(data))
        maximum = float(np.max(data))
    return {
        "mean": mean,
        "median": median,
        "sd_sample": sd_sample,
        "min": minimum,
        "max": maximum,
        "mean_axial_deg": mean if is_angle else None,
        "sd_axial_circular_deg": sd_axial,
        "statistic_kind": "axial_circular" if is_angle else "linear_sample",
        "sd_interpretation": (
            "axial_circular_spatial_scatter_not_SE_or_measurement_uncertainty"
            if is_angle else "spatial_point_scatter_not_SE_or_measurement_uncertainty"
        ),
    }


def sample_path(
    field: Mapping,
    points: Sequence[Sequence[float]],
    components: str | Sequence[str],
    max_distance: float,
    *,
    coordinate_system: str = "image",
) -> list[dict]:
    """Sample each requested location from its nearest original POI.

    The nearest POI is selected geometrically before its component mask is
    checked. Invalid POIs are reported in place; the query does not jump across
    a hole to a more distant valid point. No interpolation is performed.
    """
    try:
        tolerance = float(max_distance)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_distance must be finite and nonnegative") from exc
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("max_distance must be finite and nonnegative")
    component_list = [components] if isinstance(components, str) else list(components)
    if not component_list:
        raise ValueError("at least one component is required")
    try:
        targets = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("points must be an N by 2 array of coordinates") from exc
    if targets.ndim != 2 or targets.shape[1] != 2 or not np.isfinite(targets).all():
        raise ValueError("points must be a finite N by 2 array of coordinates")
    x, y = _coordinates(field, coordinate_system)
    px, py = x.ravel(), y.ravel()
    coord_valid = np.isfinite(px) & np.isfinite(py)
    candidate_indices = np.flatnonzero(coord_valid)
    if not len(candidate_indices):
        return [{"path_index": i, "target_x": float(point[0]), "target_y": float(point[1]), "poi_index": None,
                 "poi_x": None, "poi_y": None, "distance": None, "status": "no_support",
                 "coordinate_system": coordinate_system, "interpolation": "none",
                 **{component: None for component in component_list}} for i, point in enumerate(targets)]
    values_by_component = {component: component_values(field, component).ravel() for component in component_list}
    valid_by_component = {component: _component_valid(field, component, component_values(field, component)).ravel()
                          for component in component_list}
    rows = []
    for path_index, target in enumerate(targets):
        distance = np.hypot(px[candidate_indices] - target[0], py[candidate_indices] - target[1])
        nearest = int(np.argmin(distance))
        poi_index = int(candidate_indices[nearest])
        poi_distance = float(distance[nearest])
        row = {
            "path_index": int(path_index),
            "target_x": float(target[0]),
            "target_y": float(target[1]),
            "poi_index": poi_index,
            "poi_x": float(px[poi_index]),
            "poi_y": float(py[poi_index]),
            "distance": poi_distance,
            "coordinate_system": coordinate_system,
            "interpolation": "none",
        }
        distance_roundoff = 16 * np.finfo(float).eps * max(
            1.0, tolerance, abs(float(target[0])), abs(float(target[1])),
            abs(float(px[poi_index])), abs(float(py[poi_index])),
        )
        if poi_distance > tolerance + distance_roundoff:
            row["status"] = "outside_tolerance"
            row.update({component: None for component in component_list})
        else:
            ok_components = []
            for component in component_list:
                ok = bool(valid_by_component[component][poi_index])
                row[component] = float(values_by_component[component][poi_index]) if ok else None
                ok_components.append(ok)
            row["status"] = "valid" if all(ok_components) else ("partial_invalid" if any(ok_components) else "invalid_poi")
        rows.append(row)
    return rows


def write_rows_csv(rows: Iterable[Mapping], path: str | Path, *, fieldnames: Sequence[str] | None = None) -> Path:
    """Write row mappings as UTF-8 CSV with blank cells for missing values."""
    destination = Path(path)
    iterator = iter(rows)
    first = next(iterator, None)
    if fieldnames is None:
        if first is None:
            raise ValueError("fieldnames are required when rows is empty")
        fieldnames = list(first.keys())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        if first is not None:
            writer.writerow(first)
        writer.writerows(iterator)
    return destination


def common_color_limits(
    fields: Iterable[Mapping | None],
    component: str,
    *,
    symmetric: bool = False,
) -> tuple[float, float]:
    """Find one finite min/max range across fields for consistent frame maps."""
    low, high = math.inf, -math.inf
    for field in fields:
        if field is None:
            continue
        values = component_values(field, component)
        valid = _component_valid(field, component, values)
        finite = values[valid]
        if finite.size:
            low = min(low, float(np.min(finite)))
            high = max(high, float(np.max(finite)))
    if not np.isfinite([low, high]).all():
        raise ValueError("no valid finite values are available for color limits")
    if symmetric:
        extent = max(abs(low), abs(high))
        low, high = -extent, extent
    if low == high:
        delta = max(abs(low) * 1e-6, 1e-12)
        low, high = low - delta, high + delta
    return float(low), float(high)


def plot_field_map(
    field: Mapping,
    component: str,
    *,
    ax=None,
    color_limits: tuple[float, float] | None = None,
    cmap: str | None = None,
    method: str = "scatter",
    coordinate_system: str = "image",
    title: str | None = None,
    colorbar: bool = True,
):
    """Plot values at original POIs, with scatter as the no-bridging default.

    ``method='quadmesh'`` uses the original two-dimensional grid with masked
    invalid POIs and ``shading='nearest'``. It does not smooth or fill values.
    Pass the same ``color_limits`` to each frame for a comparable color scale.
    Returns ``(figure, axes, artist)``.
    """
    import matplotlib.pyplot as plt

    if method not in {"scatter", "quadmesh"}:
        raise ValueError("method must be 'scatter' or 'quadmesh'")
    y_axis_sign = None
    if coordinate_system == "calibrated":
        calibration = field.get("calibration")
        if not isinstance(calibration, Mapping) or "axis_sign" not in calibration:
            raise ValueError("calibrated plot coordinates require explicit calibration.axis_sign")
        _, y_axis_sign = _finite_pair(calibration["axis_sign"], "calibration.axis_sign")
        if y_axis_sign not in (-1.0, 1.0):
            raise ValueError("calibration.axis_sign entries must each be +1 or -1")
    x, y = _coordinates(field, coordinate_system)
    values = component_values(field, component)
    valid = _component_valid(field, component, values) & np.isfinite(x) & np.isfinite(y)
    if color_limits is not None:
        limits = np.asarray(color_limits, dtype=float).ravel()
        if limits.size != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError("color_limits must be finite increasing (vmin, vmax)")
        vmin, vmax = map(float, limits)
    elif valid.any():
        vmin, vmax = float(np.min(values[valid])), float(np.max(values[valid]))
        if vmin == vmax:
            delta = max(abs(vmin) * 1e-6, 1e-12)
            vmin, vmax = vmin - delta, vmax + delta
    else:
        raise ValueError("field has no valid values to plot")
    if cmap is None:
        cmap = "RdBu_r" if component in _STRAIN_COMPONENTS else "viridis"
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    if method == "scatter":
        artist = ax.scatter(x[valid], y[valid], c=values[valid], s=14, marker="s", cmap=cmap,
                            vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
    else:
        if x.ndim != 2 or y.ndim != 2 or values.ndim != 2:
            raise ValueError("quadmesh requires a two-dimensional original POI grid")
        mesh_values = np.ma.array(values, mask=~valid)
        artist = ax.pcolormesh(x, y, mesh_values, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title)
    if coordinate_system == "image":
        ax.set_xlabel("Image x (px; + right)")
        ax.set_ylabel("Image y (px; + down)")
        ax.yaxis.set_inverted(True)
    else:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.yaxis.set_inverted(y_axis_sign > 0)
    ax.set_aspect("equal", adjustable="box")
    if colorbar:
        label = f"{component} ({component_unit(component)}"
        if component in _STRAIN_COMPONENTS:
            label += "; image tensor basis: +x right, +y down"
        label += ")"
        figure.colorbar(artist, ax=ax, label=label)
    return figure, ax, artist


def plot_position_time(
    fields: Iterable[Mapping | None],
    times_s: Iterable[float],
    path_points: Sequence[Sequence[float]],
    component: str,
    max_distance: float,
    *,
    ax=None,
    color_limits: tuple[float, float] | None = None,
    cmap: str = "viridis",
    coordinate_system: str = "image",
    position_unit: str | None = None,
    max_time_gap_s: float | None = None,
    axis_label: str | None = None,
    axis_unit: str = "s",
):
    """Scatter a sampled component over path position and explicit frame time.

    ``fields`` is consumed one item at a time, so the function does not retain
    all full-field arrays. ``None`` represents a missing image frame; a None
    or NaN time marks a frame without usable time evidence. Markers are never
    connected; invalid observations and time gaps remain visibly empty. The
    returned summary counts missing and untimed frames, large gaps, invalid
    nearest POIs, distance rejections, and plotted samples.

    Returns ``(figure, axes, artist, summary)``. Coordinates in ``path_points``
    use the selected coordinate system; horizontal position is cumulative
    distance along the path.

    With ``axis_unit='frame'``, the second iterable contains explicit native
    frame numbers and the ordinate is labeled "Native frame". No seconds or
    acquisition duration are inferred; time-gap arguments are unavailable.
    """
    from itertools import zip_longest
    import matplotlib.pyplot as plt

    if axis_unit not in {"s", "frame"}:
        raise ValueError("axis_unit must be 's' or 'frame'")
    if axis_unit == "frame" and max_time_gap_s is not None:
        raise ValueError("max_time_gap_s requires an axis in seconds")
    if max_time_gap_s is not None and (not np.isfinite(max_time_gap_s) or max_time_gap_s <= 0):
        raise ValueError("max_time_gap_s must be positive and finite when supplied")
    points = np.asarray(path_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("path_points must be a nonempty finite N by 2 array")
    tolerance = float(max_distance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("max_distance must be finite and nonnegative")
    distances = np.r_[0.0, np.cumsum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1])))]
    positions, plot_times, plot_values = [], [], []
    n_frames = n_missing = n_untimed = n_invalid = n_outside = n_sampled = 0
    large_time_gaps = 0
    max_observed_time_gap = 0.0
    last_time = None
    sentinel = object()
    for field, timestamp in zip_longest(fields, times_s, fillvalue=sentinel):
        if field is sentinel or timestamp is sentinel:
            raise ValueError("fields and times_s must have the same number of entries")
        n_frames += 1
        if timestamp is None:
            n_untimed += 1
            if field is None:
                n_missing += 1
            continue
        try:
            time_value = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("times_s entries must be finite numbers or None") from exc
        if not np.isfinite(time_value):
            n_untimed += 1
            if field is None:
                n_missing += 1
            continue
        if axis_unit == "frame" and (time_value < 0 or time_value != int(time_value)):
            raise ValueError("native frame numbers must be nonnegative integers")
        if last_time is not None and time_value < last_time:
            raise ValueError("times_s must be nondecreasing in acquisition order")
        if last_time is not None:
            gap = time_value - last_time
            max_observed_time_gap = max(max_observed_time_gap, gap)
            if max_time_gap_s is not None and gap > max_time_gap_s:
                large_time_gaps += 1
        last_time = time_value
        if field is None:
            n_missing += 1
            continue
        rows = sample_path(field, points, component, tolerance, coordinate_system=coordinate_system)
        for i, row in enumerate(rows):
            if row["status"] == "outside_tolerance":
                n_outside += 1
                continue
            if row["status"] != "valid" or row.get(component) is None:
                n_invalid += 1
                continue
            positions.append(float(distances[i]))
            plot_times.append(time_value)
            plot_values.append(float(row[component]))
            n_sampled += 1
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    values = np.asarray(plot_values, dtype=float)
    if color_limits is None:
        if values.size:
            vmin, vmax = float(values.min()), float(values.max())
            if vmin == vmax:
                delta = max(abs(vmin) * 1e-6, 1e-12)
                vmin, vmax = vmin - delta, vmax + delta
        else:
            vmin, vmax = 0.0, 1.0
    else:
        limits = np.asarray(color_limits, dtype=float).ravel()
        if limits.size != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError("color_limits must be finite increasing (vmin, vmax)")
        vmin, vmax = map(float, limits)
    artist = ax.scatter(positions, plot_times, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=12, marker="s", linewidths=0, rasterized=True)
    ax.set_xlabel(f"Position along path ({position_unit or ('px' if coordinate_system == 'image' else 'mm')})")
    label = axis_label or ("Time (s)" if axis_unit == "s" else "Native frame")
    ax.set_ylabel(label)
    if n_sampled:
        figure.colorbar(artist, ax=ax, label=f"{component} ({component_unit(component)})")
    summary = {
        "frames": n_frames,
        "missing_frames": n_missing,
        "untimed_frames": n_untimed if axis_unit == "s" else None,
        "large_time_gaps": large_time_gaps if axis_unit == "s" else None,
        "max_observed_time_gap_s": max_observed_time_gap if axis_unit == "s" else None,
        "axis_unit": axis_unit, "axis_label": label,
        "missing_axis_values": n_untimed, "max_observed_axis_gap": max_observed_time_gap,
        "sampled_values": n_sampled,
        "invalid_nearest_poi": n_invalid,
        "outside_distance_tolerance": n_outside,
        "max_time_gap_s": max_time_gap_s,
        "interpolation": "none",
        "line_connections": False,
    }
    return figure, ax, artist, summary


__all__ = [
    "calibrate_field",
    "common_color_limits",
    "component_unit",
    "component_values",
    "iter_roi_rows",
    "jacobian_determinant",
    "negative_jacobian_poi_mask",
    "nonpositive_jacobian_poi_mask",
    "plot_field_map",
    "plot_position_time",
    "principal_strains",
    "sample_path",
    "summarize_roi",
    "write_rows_csv",
]
