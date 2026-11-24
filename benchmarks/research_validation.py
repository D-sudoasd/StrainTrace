"""Independent, analytic-image checks of full-field displacement and strain.

The optical texture is evaluated at continuous coordinates: no DIC sampler is
used to manufacture the target. Results are engineering validation, not an
experimental calibration or a comparison with a commercial solver.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ezdic_core as core  # noqa: E402
from benchmarks._common import (  # noqa: E402
    load_core_module,
    set_deterministic_rng,
    sha256_file,
    write_json_lf,
)
from benchmarks.synthetic_cases import analytic_texture  # noqa: E402


def analytic_pair(shape=(144, 168), *, u=0.37, v=-0.24, a=0.0, b=0.0, c=0.0, noise=0.0):
    if 1 + a <= 0 or 1 + c <= 0:
        raise ValueError("analytic_pair requires 1 + a > 0 and 1 + c > 0")
    y, x = np.indices(shape, dtype=float)
    cx, cy = shape[1] / 2, shape[0] / 2
    shifted = x - cx - u
    if b == 0:
        rx = shifted / (1 + a)
    else:
        discriminant = (1 + a)**2 + 4 * b * shifted
        if np.any(discriminant <= 0):
            raise ValueError("analytic_pair quadratic warp is not invertible on the image domain")
        rx = 2 * shifted / ((1 + a) + np.sqrt(discriminant))
    ry = (y - cy - v) / (1 + c)
    reference = analytic_texture(x, y)
    target = analytic_texture(rx + cx, ry + cy)
    if noise:
        rng = np.random.default_rng(801)
        reference += rng.normal(0, noise, shape)
        target += rng.normal(0, noise, shape)
    return reference.astype(np.float64), target.astype(np.float64)


def validation(strain_window=5):
    set_deterministic_rng(0)
    started = time.perf_counter()
    records = {}
    errors, failures, iterations = [], 0, []
    for shift in np.linspace(0.02, 0.98, 17):
        reference, target = analytic_pair(u=float(shift), v=-0.27)
        fit = core.refine_subset_icgn(reference, target, 80, 70, 31, p0=np.array([round(shift), 0, 0, 0, 0, 0]), max_iter=50, tol=1e-5)
        if fit is None or not fit.get('converged', True):
            failures += 1
            continue
        errors.append([fit['u'] - shift, fit['v'] + 0.27])
        iterations.append(fit['iterations'])
    if errors:
        err_arr = np.asarray(errors, dtype=float)
        vec_err = np.hypot(err_arr[:, 0], err_arr[:, 1])
        sweep_rmse = float(np.sqrt(np.mean(vec_err**2)))
        sweep_max = float(np.max(vec_err))
        sweep_iters = float(np.mean(iterations))
    else:
        sweep_rmse = sweep_max = sweep_iters = None
    records['subpixel_sweep'] = {
        'rmse_px': sweep_rmse,
        'max_abs_error_px': sweep_max,
        'failed': failures, 'mean_iterations': sweep_iters,
    }
    for name, params in [('translation', {}), ('affine', {'a': 0.008, 'c': -0.003}),
                         ('quadratic', {'a': 0.008, 'b': 0.00009, 'c': -0.003}),
                         ('noisy_translation', {'noise': 1.5})]:
        reference, target = analytic_pair(**params)
        tick = time.perf_counter()
        field = core.run_2d_dic(reference, target, (22, 22, 124, 100), subset_size=25,
                                step=8, search_radius=5, strain_window=strain_window, max_iter=40, conv_tol=1e-4)
        X, Y = field['X'], field['Y']
        a, b, c = params.get('a', 0), params.get('b', 0), params.get('c', 0)
        u_true = .37 + a * (X - reference.shape[1]/2) + b * (X - reference.shape[1]/2)**2
        v_true = -.24 + c * (Y - reference.shape[0]/2)
        derivative = a + 2*b*(X - reference.shape[1]/2)
        exx_true = derivative + .5*derivative**2
        disp_vec = np.hypot(field['U'] - u_true, field['V'] - v_true)
        finite_disp = disp_vec[np.isfinite(disp_vec)]
        strain_error = field['Exx_grid'] - exx_true
        finite_strain = strain_error[np.isfinite(strain_error)]
        records[name] = {
            'displacement_rmse_px': float(np.sqrt(np.mean(finite_disp**2))) if finite_disp.size else None,
            'Exx_rmse': float(np.sqrt(np.mean(finite_strain**2))) if finite_strain.size else None,
            'Exx_bias': float(np.mean(finite_strain)) if finite_strain.size else None,
            'Exx_p99_abs_error': float(np.percentile(np.abs(finite_strain), 99)) if finite_strain.size else None,
            'valid_fraction': float(np.mean(field['valid'])),
            'strain_valid_fraction': float(np.mean(field['strain_valid'])),
            'elapsed_s': time.perf_counter() - tick,
        }
    records['elapsed_s'] = time.perf_counter() - started
    records['code_sha256'] = sha256_file(core.__file__)
    records['texture'] = 'continuous random Fourier texture; independent analytic inverse warp'
    records['parameters'] = {'subset_size_px':25,'step_px':8,'strain_window_poi':strain_window,
                             'strain_gauge_span_px':8*(strain_window-1),'smooth_sigma':0.0}
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--core-file', type=Path, help='Archived core for a matched before/after comparison')
    parser.add_argument('--strain-window',type=int,default=5)
    args = parser.parse_args()
    if args.core_file:
        core = load_core_module(args.core_file, 'archived_dic_core')
    result = validation(args.strain_window)
    write_json_lf(args.output, result, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))
