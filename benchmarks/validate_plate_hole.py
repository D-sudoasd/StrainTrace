"""Reproduce the public Ncorr sample12 checks without downloading or changing inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ezdic_core as core

ROI = (45, 30, 302, 975)
SETTINGS = dict(subset_size=31, step=8, search_radius=16, strain_window=5,
                smooth_sigma=0., max_iter=40, zncc_min=.85)


def metrics(field):
    valid = np.asarray(field['strain_valid'], dtype=bool)
    summary = field['quality_summary']
    return {'correlation_fraction': summary['correlation_valid_fraction'],
            'strain_fraction': summary['strain_valid_fraction'],
            'scientific_ok': summary['scientific_ok'],
            'converged_valid_count': int(np.count_nonzero(field['converged'] & field['valid'])),
            'median_zncc': float(np.nanmedian(field['zncc'])),
            'median_Eyy': float(np.nanmedian(field['Eyy'][valid])),
            'Eyy_p01_p99': np.nanpercentile(field['Eyy'][valid], [1, 99]).tolist(),
            'median_fit_residual_px': float(np.nanmedian(field['strain_fit_residual_rms'][valid])),
            'invalid_reasons': summary['invalid_reason_histogram']}


def validate(images: Path, output: Path, baseline_core: Path | None = None):
    paths = [images/name for name in ('ohtcfrp_00.tif', 'ohtcfrp_04.tif', 'ohtcfrp_08.tif', 'roi.tif')]
    identities = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    reference = core.read_gray_image(paths[0])
    mask = np.asarray(Image.open(paths[-1]).convert('L')) >= 128
    output.mkdir(parents=True, exist_ok=True)
    baseline = None
    if baseline_core:
        spec = importlib.util.spec_from_file_location('baseline_core', baseline_core)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
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
            record['common_domain_fit_residual_px'] = {
                'point_count': int(common.sum()),
                'before': float(np.nanmedian(fit['fit_residual_rms'][common])),
                'after': float(np.nanmedian(field['strain_fit_residual_rms'].reshape(retained.shape)[common]))}
        field['reference_image'], field['deformed_image'] = reference, target
        field['mask_provenance'] = {'mode': 'file', 'source_sha256': identities['roi.tif']}
        field['display_options'] = {'percent': True, 'background': 'none', 'cmap': 'viridis'}
        core.export_dic_field_outputs(field, output, stem=path.stem)
        field['display_options']['background'] = 'deformed'
        core.plot_dic_field_map(field, output/f'{path.stem}_Eyy_overlay.png', 'Eyy')
        records[path.name] = record
        print(path.name, json.dumps(record), flush=True)
    report = {'source': 'https://ncorr.com/download/sample12.zip', 'input_sha256': identities,
              'core_sha256': hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
              'settings': SETTINGS | {'after_reject_nonconverged': True}, 'roi_xywh': ROI, 'records': records,
              'boundary': 'Public experimental imagery; no independent strain ground truth or VIC-2D comparison.'}
    (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True, help='Extracted Ncorr sample12 directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-core', type=Path)
    args = parser.parse_args()
    validate(args.images, args.output, args.baseline_core)
