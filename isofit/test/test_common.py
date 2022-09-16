import numpy as np
from isofit.core.common import eps, combos, get_absorption, expand_path, load_spectrum, load_wavelen, \
    spectral_response_function, recursive_replace, expand_path_to_absolute, svd_inv_sqrt, svd_inv
from io import StringIO
import scipy
import os

def test_eps():
    assert eps == 1e-5

def test_combos():
    inds = np.array([[1, 2], [3, 4, 5]])
    result = np.array([[1, 3], [2, 3], [1, 4], [2, 4], [1, 5], [2, 5]])
    assert np.array_equal(combos(inds), result)

def test_load_wavelen():
    file = StringIO('0 0.37686 0.00557 \n 1 0.38187 0.00558 \n 2 0.38688 0.00558')
    wl_modified, fwhm_modified = load_wavelen(file)
    assert(wl_modified.ndim == 1)
    assert(fwhm_modified.ndim == 1)
    assert(wl_modified[0] > 100)

def test_get_absorption():
    file = StringIO('12e7,2e7,3e7,4e7,3e7\n16e7,3e7,8e7,5e7,12e7')
    wavelengths = np.array([13e7, 15e7])
    w_abscf_new, i_abscf_new = get_absorption(wavelengths, file)
    assert(w_abscf_new[0] == 1.25e7*np.pi)
    assert(i_abscf_new[0] == 1.5e7*np.pi)
    assert(w_abscf_new[1] == 1.75e7*np.pi)
    assert(i_abscf_new[1] == 2.5e7*np.pi)

def test_expand_to_absolute():
    subpath = 'isofit/test/test_common.py'
    assert(os.path.abspath(subpath) == expand_path_to_absolute(subpath))

def test_spectral_response_function():
    response_range = np.array([10, 8])
    mu = 6.0
    sigma = -2.0
    srf = spectral_response_function(response_range, mu, sigma)
    assert(abs(srf[0] - 0.182425524) < 0.0000001)
    assert(abs(srf[1] - 0.817574476) < 0.0000001)

def test_load_spectrum():
    file = StringIO('0.123 0.132 0.426 \n 0.234 0.234 0.132 \n 0.123 0.423 0.435')
    spectrum_new, wavelength_new = load_spectrum(file)
    assert(wavelength_new.ndim == 1)
    assert(spectrum_new.ndim == 1)
    assert(wavelength_new[0] > 100)

def test_svd_inv_sqrt():
    sample_array_1 = np.array([[7, 0], [0, 1]])
    result_matrix_1, result_matrix_sq_1 = svd_inv_sqrt(sample_array_1)
    assert(result_matrix_1.all() == scipy.linalg.inv(sample_array_1).all())
    assert((result_matrix_sq_1 @ result_matrix_sq_1).all() == scipy.linalg.inv(sample_array_1).all())

    sample_array_4 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    result_matrix_4, result_matrix_sq_4 = svd_inv_sqrt(sample_array_4)
    assert((scipy.linalg.inv(sample_array_4)).all() == result_matrix_4.all())
    assert((result_matrix_sq_4 @ result_matrix_sq_4).all() == scipy.linalg.inv(sample_array_4).all())

def test_svd_inv():

    """
    Written to operate on matrices that are at least positive semi-definite. This is achieved by 
    conditioning of the diagonal to make all eigenvalues greater than or equal to 0.
    """
    sample_array_1 = np.array([[7, 0], [0, 1]])
    assert(svd_inv(sample_array_1).all() == svd_inv_sqrt(sample_array_1)[0].all())
    sample_array_4 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    assert(svd_inv(sample_array_4).all() == svd_inv_sqrt(sample_array_4)[0].all())

def test_recursive_replace():
    list1 = ['list_val_1', 'list_val_2', 'list_val_3']
    recursive_replace(list1,2, 'replacement_val')
    unchanged_list1 = ['list_val_1', 'list_val_2', 'list_val_3']
    assert(unchanged_list1 == list1)

    dict1 = {1: 'dict_val_1', 2: 'dict_val_2', 3: 'dict_val_3'}
    modified_dict1 = {1: 'dict_val_1', 2: 'dict_val_2', 3: 'replacement_val'}
    recursive_replace(dict1, 3, 'replacement_val')
    assert(modified_dict1 == dict1)

    dict2 = {1: 'dict_val_1', 2: ['list_val_1', {1: 'dict_val_2', 2: 'dict_val_3'\
        , 3: ['list_val_2', 'list_val_3']}, 'list_val4'], 3:'dict_val_5'}
    recursive_replace(dict2,2,'replacement_val')
    modified_dict2 = {1: 'dict_val_1', 2: 'replacement_val', 3:'dict_val_5'}
    assert(dict2 == modified_dict2)

    dict3 = {1: ['list_val_1', {1: 'dict_val_1' , 2: 'dict_val_2'}, {1:'dict_val_3', 2:'dict_val_4', 3: ('tuple_val_5'\
        , 'tuple_val_4')}], 2: (['list_val_2', 'list_val_3'], {1: 'dict_val_5',2: ['list_val_4', 'list_val_5'] ,3: 'dict_val_5'}),3: 'dict_val_6'}

    modified_dict3 = {1: ['list_val_1', {1: 'dict_val_1' , 2: 'dict_val_2'}, {1:'dict_val_3', 2:'dict_val_4', 3: 'replacement_val'}]\
        , 2: (['list_val_2', 'list_val_3'], {1: 'dict_val_5',2: ['list_val_4', 'list_val_5'] ,3: 'replacement_val'}),3: 'replacement_val'}
    recursive_replace(dict3,3,'replacement_val')
    assert(modified_dict3 == dict3)

def main():
    test_eps()
    test_load_wavelen()
    test_load_spectrum()
    test_get_absorption()
    test_recursive_replace()
    test_expand_to_absolute()
    test_spectral_response_function()
    test_svd_inv_sqrt()
    test_svd_inv()

    print('TESTS COMPLETE')

main()


