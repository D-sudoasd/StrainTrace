"""Reproduce the public Ncorr sample12 checks without downloading or changing inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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

ROI = (45, 30, 302, 975)
SETTINGS = dict(subset_size=31, step=8, search_radius=16, strain_window=5,
                smooth_sigma=0., max_iter=40, zncc_min=.85)


def metrics(field):
    valid = np.asarray(field['strain_valid'], dtype=bool)
    summary = field['quality_summary']
    zncc = np.asarray(field['zncc'], dtype=float)
    zncc_finite = zncc[np.isfinite(zncc)]
    eyy_valid = np.asarray(field['Eyy'], dtype=float)[valid]
    eyy_valid = eyy_valid[np.isfinite(eyy_valid)]
    fit_res = np.asarray(field['strain_fit_residual_rms'], dtype=float)[valid]
    fit_res = fit_res[np.isfinite(fit_res)]
    return {'correlation_fraction': summary['correlation_valid_fraction'],
            'strain_fraction': summary['strain_valid_fraction'],
            'scientific_ok': summary['scientific_ok'],
            'converged_valid_count': int(np.count_nonzero(field['converged'] & field['valid'])),
            'median_zncc': float(np.median(zncc_finite)) if zncc_finite.size else None,
            'median_Eyy': float(np.median(eyy_valid)) if eyy_valid.size else None,
            'Eyy_p01_p99': np.percentile(eyy_valid, [1, 99]).tolist() if eyy_valid.size else None,
            'median_fit_residual_px': float(np.median(fit_res)) if fit_res.size else None,
            'invalid_reasons': summary['invalid_reason_histogram']}


def validate(images: Path, output: Path, baseline_core: Path | None = None):
    set_deterministic_rng(0)
    paths = [images/name for name in ('ohtcfrp_00.tif', 'ohtcfrp_04.tif', 'ohtcfrp_08.tif', 'roi.tif')]
    identities = {path.name: sha256_file(path) for path in paths}
    reference = core.read_gray_image(paths[0])
    with Image.open(paths[-1]) as im:
        mask = np.asarray(im.convert('L')) >= 128
    output.mkdir(parents=True, exist_ok=True)
    baseline = load_core_module(baseline_core, 'baseline_core') if baseline_core else None
    records = {}
    for path in paths[1:3]:
        target = core.read_gray_image(path)
        field = core.run_2d_dic(reference, target, ROI, specimen_mask=mask, reject_nonconverged=True, **SETTINGS)
        record = {'after': metrics(field)}
        if baseline:
            old = baseline.run_2d_dic(reference, target, ROI, **SETTINGS)
            record['before_unmasked'] = metrics(old)
            # Compare local fits on the same retained subset domain. The old
            # solver has no mask argument; remove those measurements explicitly.
            retained = field['eligible'].reshape(field['X'].shape)
            old_u = np.where(retained, old['U'], np.nan)
            old_v = np.where(retained, old['V'], np.nan)
            fit = baseline.compute_strain_fields(old['X'], old['Y'], old_u, old_v, window=5)
            common = fit['strain_valid'] & field['strain_valid'].reshape(retained.shape)
            before_common = np.asarray(fit['fit_residual_rms'], dtype=float)[common]
            before_common = before_common[np.isfinite(before_common)]
            after_common = np.asarray(field['strain_fit_residual_rms'], dtype=float).reshape(retained.shape)[common]
            after_common = after_common[np.isfinite(after_common)]
            record['common_domain_fit_residual_px'] = {
                'point_count': int(common.sum()),
                'before': float(np.median(before_common)) if before_common.size else None,
                'after': float(np.median(after_common)) if after_common.size else None}
        field['reference_image'], field['deformed_image'] = reference, target
        field['mask_provenance'] = {'mode': 'file', 'source_sha256': identities['roi.tif']}
        field['display_options'] = {'percent': True, 'background': 'none', 'cmap': 'viridis'}
        core.export_dic_field_outputs(field, output, stem=path.stem)
        field['display_options']['background'] = 'deformed'
        core.plot_dic_field_map(field, output/f'{path.stem}_Eyy_overlay.png', 'Eyy')
        records[path.name] = record
        print(path.name, json.dumps(record, allow_nan=False), flush=True)
    report = {'source': 'https://ncorr.com/download/sample12.zip', 'input_sha256': identities,
              'core_sha256': sha256_file(core.__file__),
              'settings': SETTINGS | {'after_reject_nonconverged': True}, 'roi_xywh': ROI, 'records': records,
              'boundary': 'Public experimental imagery; no independent strain ground truth or VIC-2D comparison.'}
    write_json_lf(output/'validation.json', report, indent=2, allow_nan=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True, help='Extracted Ncorr sample12 directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-core', type=Path)
    args = parser.parse_args()
    validate(args.images, args.output, args.baseline_core)
