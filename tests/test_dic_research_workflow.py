"""Independent numerical and topology acceptance checks for research DIC."""
import numpy as np
import pytest
from scipy.special import erf
from scipy import ndimage

import ezdic_core as core
from benchmarks.research_validation import analytic_pair, analytic_texture


@pytest.mark.parametrize("solver", [core.refine_subset_icgn, core.refine_subset_iclm])
def test_continuous_sampler_recovers_fractional_displacement_and_brightness(solver):
    reference, target = analytic_pair(u=.37, v=-.24)
    target = target*1.08+7
    fit = solver(reference, target, 80, 70, 31, p0=[0,0,0,0,0,0], max_iter=40, tol=1e-5)
    assert fit is not None and fit['converged']
    assert np.linalg.norm([fit['u']-.37,fit['v']+.24]) < .0001
    assert fit['zncc'] > .99999


def test_stalled_noisy_subsets_converge_to_a_local_correlation_minimum():
    reference = core.generate_synthetic_speckle(96, 112, seed=27)
    target = core.warp_image_translation(reference, .37, -.24)
    target += np.random.default_rng(2).normal(0, 3, reference.shape).astype(target.dtype)
    # This fixture tests optimizer stationarity, not subpixel ground truth:
    # its raster target uses a different, quantized interpolation method.
    for x, y in ((45, 45), (55, 50), (65, 55)):
        fit = core.refine_subset_icgn(reference, target, x, y, 25, max_iter=40, tol=1e-5)
        assert fit['converged'] and fit['increment_norm_px'] <= 1e-5
        yy, xx = np.mgrid[-12:13, -12:13]
        template = np.asarray(reference[y-12:y+13, x-12:x+13], dtype=float).ravel()
        template -= template.mean()
        template /= np.linalg.norm(template)
        def objective(p):
            mx = x+p[0]+(1+p[1])*xx+p[2]*yy
            my = y+p[3]+p[4]*xx+(1+p[5])*yy
            sample = ndimage.map_coordinates(np.asarray(target, dtype=float), [my.ravel(), mx.ravel()], order=5, mode='mirror')
            sample -= sample.mean()
            sample /= np.linalg.norm(sample)
            return np.sum((sample-template)**2)
        optimum = objective(fit['p'])
        for axis in (0, 3):
            for offset in (-.001, .001):
                perturbed = fit['p'].copy()
                perturbed[axis] += offset
                assert objective(perturbed) >= optimum-1e-10


def test_robust_strain_retains_quadratic_gradients_at_boundaries_and_large_origin():
    Y, X = np.mgrid[0:11,0:13].astype(float)
    U, V = .002*X**2+.001*X*Y+.01*X, -.003*Y
    field = core.compute_strain_fields(X+1e6,Y+2e6,U,V,window=5)
    assert field['strain_valid'].all()
    np.testing.assert_allclose(field['exx'], .004*X+.001*Y+.01, atol=1e-9)
    np.testing.assert_allclose(field['eyy'], -.003, atol=1e-9)
    assert (field['fit_degree']==2).any()


def test_fit_does_not_cross_missing_crack_or_fabricate_its_center():
    Y, X = np.mgrid[0:11,0:15].astype(float)
    U, V = .01*X, .002*Y
    U[:,8:] += 4
    U[:,7] = V[:,7] = np.nan
    field = core.compute_strain_fields(X,Y,U,V,window=9)
    np.testing.assert_allclose(field['exx'][:,6], .01, atol=1e-10)
    np.testing.assert_allclose(field['exx'][:,8], .01, atol=1e-10)
    assert np.isnan(field['Exx'][:,7]).all()


def test_optional_gaussian_does_not_mix_disconnected_sides_or_fill_gap():
    values = np.zeros((15, 17))
    values[:, 9:] = 4
    values[:, 8] = np.nan
    smoothed = core._nan_gaussian(values, 2)
    assert np.isnan(smoothed[:, 8]).all()
    np.testing.assert_allclose(smoothed[:, :8], 0, atol=1e-12)
    np.testing.assert_allclose(smoothed[:, 9:], 4, atol=1e-12)


def test_robust_fit_suppresses_an_isolated_mismatch_without_flattening_affine_field():
    Y, X = np.mgrid[0:17,0:17].astype(float)
    U, V = .01*X, -.004*Y
    U[8,9] += 3
    robust = core.compute_strain_fields(X,Y,U,V,window=7)
    ordinary = core.compute_strain_fields(X,Y,U,V,window=7,robust=False,degree=1)
    assert abs(robust['exx'][8,8]-.01) < .001
    assert abs(ordinary['exx'][8,8]-.01) > .005


def test_local_outlier_rejection_keeps_original_values_and_smooth_curvature():
    Y, X = np.mgrid[0:17,0:17].astype(float)
    U, V = .004*X**2+.001*X*Y, -.003*Y
    assert not core._displacement_outliers(X,Y,U,V).any()
    U[8,8] += 4
    bad = core._displacement_outliers(X,Y,U,V)
    assert bad[8,8] and np.count_nonzero(bad) == 1


def test_strain_peak_is_not_erased_by_default_local_fit():
    Y, X = np.mgrid[0:41,0:81].astype(float)
    width, peak = 12., .02
    U = peak*width*np.sqrt(np.pi/2)*erf((X-40)/(np.sqrt(2)*width))
    field = core.compute_strain_fields(X,Y,U,np.zeros_like(U),window=7)
    assert np.max(field['exx']) > .98*peak
    assert np.max(field['exx']) <= 1.001*peak


def test_rigid_body_rotation_has_zero_green_lagrange_strain():
    Y, X = np.mgrid[0:11,0:13].astype(float)
    theta = .2
    U = (np.cos(theta)-1)*X-np.sin(theta)*Y
    V = np.sin(theta)*X+(np.cos(theta)-1)*Y
    field = core.compute_strain_fields(X,Y,U,V)
    for key in ('Exx','Eyy','Exy'):
        assert np.nanmax(np.abs(field[key])) < 1e-12


def test_mask_excludes_full_subsets_and_quality_uses_eligible_denominator():
    reference, target = analytic_pair()
    mask = np.ones(reference.shape, dtype=bool)
    mask[55:85,65:95] = False
    field = core.run_2d_dic(reference,target,(22,22,124,100),subset_size=21,step=6,search_radius=5,specimen_mask=mask)
    excluded = ~field['eligible']
    assert excluded.any() and np.isnan(field['u'][excluded]).all()
    assert (field['invalid_reason'][excluded]=='MASK_EXCLUDED').all()
    q = field['quality_summary']
    assert q['eligible_point_count']+q['masked_point_count'] == q['point_count']
    assert q['correlation_valid_fraction'] == pytest.approx(1.)


def test_texture_mask_identifies_uniform_hole_and_blank_background():
    y, x = np.indices((120,150),dtype=float)
    image = np.full((120,150),120.)
    image[10:110,15:135] = analytic_texture(x,y)[10:110,15:135]
    image[45:75,55:85] = 120.
    mask = core.build_specimen_mask(image,(0,0,150,120))
    assert not mask[0:5,:].any()
    assert not mask[52:68,62:78].any()
    assert mask[25:40,30:50].mean() > .95


def test_feature_initialization_recovers_motion_outside_local_search_radius():
    reference,target = analytic_pair(shape=(240,280),u=24.37,v=-16.24)
    field = core.run_2d_dic(reference,target,(50,50,150,130),subset_size=31,step=15,search_radius=5)
    assert field['initial_transform'] is not None
    assert field['quality_summary']['correlation_valid_fraction'] > .98
    assert np.nanmax(np.abs(field['u']-24.37)) < .002
    assert np.nanmax(np.abs(field['v']+16.24)) < .002


def _display_field():
    Y, X = np.mgrid[10:51:10,10:61:10].astype(float)
    shape = X.shape
    values = .001*X
    return {'X':X,'Y':Y,'u':np.full(shape,.0).ravel(),'v':np.full(shape,.0).ravel(),
            'Exx':values.ravel(),'valid':np.ones(X.size,dtype=bool),'strain_valid':np.ones(X.size,dtype=bool),'step':10}


def test_contour_and_overlay_leave_missing_cells_blank_and_do_not_change_values():
    field = _display_field()
    index = 2*field['X'].shape[1]+2
    field['Exx'][index] = np.nan
    field['strain_valid'][index] = False
    before = field['Exx'].copy()
    image = np.full((80,100),120,dtype=np.uint8)
    overlay = core.overlay_dic_field_on_image(image,field,'Exx')
    assert (overlay[25:35,25:35] == 120).all()
    assert np.any(overlay[15,15] != 120)
    np.testing.assert_array_equal(field['Exx'],before)


def test_deformed_overlay_moves_to_current_coordinates_and_uses_full_range():
    field = _display_field()
    field['u'][:] = 25
    image = np.full((80,100),120,dtype=np.uint8)
    overlay = core.overlay_dic_field_on_image(image,field,'Exx')
    assert np.all(overlay[20,20] == 120)
    assert np.any(overlay[20,70] != 120)
    lo, hi, scale = core.dic_color_limits(field,'Exx')
    assert lo == .01 and hi == .06 and scale == 1
    lo, hi, scale = core.dic_color_limits(field,'Exx',{'percent':True,'color_mode':'symmetric'})
    assert lo == -6 and hi == 6 and scale == 100


def test_high_bit_depth_scaling_preserves_changes_smaller_than_display_quantum():
    image = np.array([[10000,10001],[20000,20001]],dtype=np.uint16)
    result = core.normalize_dic_float(image,{'lo':0,'hi':65535})
    assert result.dtype == np.float64
    assert result[0,1]-result[0,0] == pytest.approx(255/65535)


def test_npz_and_all_strain_components_export_without_interpolated_measurements(tmp_path,monkeypatch):
    reference,target = analytic_pair()
    field = core.run_2d_dic(reference,target,(22,22,124,100),subset_size=25,step=10,search_radius=5)
    def cheap_plot(field,path,**kwargs):
        path.write_bytes(kwargs['component'].encode('ascii'))
        return path
    monkeypatch.setattr(core,'plot_dic_field_map',cheap_plot)
    result = core.export_dic_field_outputs(field,tmp_path)
    assert len(result['plots']) == len(core.DIC_FIELD_COMPONENTS)
    assert len({path.name.casefold() for path in result['plots']}) == 9
    assert len(list(tmp_path.glob('*.png'))) == 9
    assert (tmp_path / 'dic_field_Exx.png').read_bytes() == b'Exx'
    assert (tmp_path / 'dic_field_exx_infinitesimal.png').read_bytes() == b'exx'
    with np.load(result['arrays'],allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['u_raw'],field['u_raw'])
        np.testing.assert_array_equal(arrays['strain_valid'],field['strain_valid'])
        assert arrays['metadata_json'].item()
