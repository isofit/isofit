from importlib.resources import path
from operator import inv
from unittest.case import _AssertRaisesContext
import numpy as np
import os
from io import StringIO
from os.path import expandvars, split, abspath
from pytest import MonkeyPatch
#import xxhash
#import scipy.linalg
from collections import OrderedDict
import unittest


#from isofit.core.common import load_wavelen, spectral_response_function, get_absorption, \
    #resample_spectrum, load_spectrum, expand_path, find_header, recursive_reencode, recursive_replace, \
        #svd_inv_sqrt, svd_inv, expand_all_paths


def load_wavelen(wavelength_file: str):

    """""
    Load a wavelength file, and convert to nanometers if needed.
    Args:
        wavelength_file: file to read wavelengths from
    Returns:
        (np.array, np.array): wavelengths, full-width-half-max
    """""

    q = np.loadtxt(wavelength_file)
    if q.shape[1] > 2:
        q = q[:, 1:3]
    if q[0, 0] < 100:
        q = q * 1000.0
    wl, fwhm = q.T
    return wl, fwhm

print("BEGIN")

#wl = np.random.rand(425,4)
#file = open("wl_sample.txt", "w+")
#np.savetxt("wl_sample.txt", wl)

#wl_modified, fwhm_modified = load_wavelen("C:/Users/vpatro/Desktop/wl_sample.txt")
#file.close()
#assert(wl_modified.ndim == 1)
#assert(fwhm_modified.ndim == 1)
#assert(wl_modified[0] > 100)

file = StringIO('0 0.37686 0.00557 \n 1 0.38187 0.00558 \n 2 0.38688 0.00558')
#print(np.loadtxt(file))
#wl_modified, fwhm_modified = load_wavelen("C:/Users/vpatro/Desktop/wl_sample.txt")
wl_modified, fwhm_modified = load_wavelen(file)
#print(wl_modified)
#print(fwhm_modified)
assert(wl_modified.ndim == 1)
assert(fwhm_modified.ndim == 1)
assert(wl_modified[0] > 100)


#wl_modified, fwhm_modified = load_wavelen("C:/Users/vpatro/Desktop/Test/wl_multicol.txt")
#print(wl_modified.shape)

def emissive_radiance(emissivity: np.array, T: np.array, wl: np.array) -> (np.array, np.array):
    """Calcluate the radiance of a surface due to emission.

    Args:
        emissivity: surface emissivity.
        T: surface temperature [K]
        wl: emmissivity wavelengths [nm]

    Returns:
        np.array: surface upwelling radiance in uW $cm^{-2} sr^{-1} nm^{-nm}$
        np.array: partial derivative of radiance with respect to temperature uW $cm^{-2} sr^{-1} nm^{-1} k^{-1}$

    """

    c_1 = 1.88365e32/np.pi
    c_2 = 14387690
    J_per_eV = 1.60218e-19
    wl_um = wl / 1000.0
    ph_per_sec_cm2_sr_nm = c_1/(wl**4)/(np.exp(c_2/wl/T)-1.0) * emissivity
    print(ph_per_sec_cm2_sr_nm)
    # photon energy in eV
    eV_per_sec_cm2_sr_nm = 1.2398 * ph_per_sec_cm2_sr_nm/wl_um
    W_per_cm2_sr_nm = J_per_eV * eV_per_sec_cm2_sr_nm
    uW_per_cm2_sr_nm = W_per_cm2_sr_nm*1e6
    dRdn_dT = c_1/(wl**4)*(-pow(np.exp(c_2/wl/T)-1.0, -2.0)) *\
        np.exp(c_2/wl/T)*(-pow(T, -2.0)*c_2/wl) *\
        emissivity/wl_um*1.2398*J_per_eV*1e6
    return uW_per_cm2_sr_nm, dRdn_dT

emissivity = np.array([0.8, 0.8])
T = np.array([300, 300])
wavelength = np.array([400, 600])
#print(400**4)
#print((((1.88365e32/np.pi)/(400**4)/(np.exp(14387690/400/300)-1.0))*0.8)*0.4*1.2398*1.60218e-19*1e6)

#uW_per_cm2_sr_nm_modified, dRdn_dT_modified = emissive_radiance(emissivity, T, wavelength)
#print(uW_per_cm2_sr_nm_modified)
#print(dRdn_dT_modified)

#assert(uW_per_cm2_sr_nm_modified == 7.90527265e-44)


def spectral_response_function(response_range: np.array, mu: float, sigma: float):
    """Calculate the spectral response function.

    Args:
        response_range: signal range to calculate over
        mu: mean signal value
        sigma: signal variation

    Returns:
        np.array: spectral response function

    """

    u = (response_range-mu)/abs(sigma)
    y = (1.0/(np.sqrt(2.0*np.pi)*abs(sigma)))*np.exp(-u*u/2.0)
    srf = y/y.sum()
    return srf


#def test_spectral_response_function():
response_range = np.array([10, 8])
mu = 6.0
sigma = -2.0
srf = spectral_response_function(response_range, mu, sigma)
assert(abs(srf[0] - 0.182425524) < 0.0000001)
assert(abs(srf[1] - 0.817574476) < 0.0000001)




def get_absorption(wl: np.array, absfile: str) -> (np.array, np.array):
    """Calculate water and ice absorption coefficients using indices of
    refraction, and interpolate them to new wavelengths (user specifies nm).
    Args:
        wl: wavelengths to interpolate to
        absfile: file containing indices of refraction
    Returns:
        np.array: interpolated, wavelength-specific water absorption coefficients
        np.array: interpolated, wavelength-specific ice absorption coefficients
    """

    # read the indices of refraction
    q = np.loadtxt(absfile, delimiter=',')
    wl_orig_nm = q[:, 0]
    wl_orig_cm = wl_orig_nm/1e9*1e2
    water_imag = q[:, 2]
    ice_imag = q[:, 4]

    # calculate absorption coefficients in cm^-1
    water_abscf = water_imag*np.pi*4.0/wl_orig_cm
    ice_abscf = ice_imag*np.pi*4.0/wl_orig_cm

    # interpolate to new wavelengths (user provides nm)
    water_abscf_intrp = np.interp(wl, wl_orig_nm, water_abscf)
    ice_abscf_intrp = np.interp(wl, wl_orig_nm, ice_abscf)
    return water_abscf_intrp, ice_abscf_intrp

file = StringIO('12e7,2e7,3e7,4e7,3e7\n16e7,3e7,8e7,5e7,12e7')
#print(np.loadtxt(file, delimiter = ','))
wavelengths = np.array([13e7, 15e7])
w_abscf_new, i_abscf_new = get_absorption(wavelengths, file)
assert(w_abscf_new[0] == 1.25e7*np.pi)
assert(i_abscf_new[0] == 1.5e7*np.pi)
assert(w_abscf_new[1] == 1.75e7*np.pi)
assert(i_abscf_new[1] == 2.5e7*np.pi)

def resample_spectrum(x: np.array, wl: np.array, wl2: np.array, fwhm2: np.array, fill: bool = False) -> np.array:
    """Resample a spectrum to a new wavelength / FWHM.
       Assumes Gaussian SRFs.

    Args:
        x: radiance vector
        wl: sample starting wavelengths
        wl2: wavelengths to resample to
        fwhm2: full-width-half-max at resample resolution
        fill: boolean indicating whether to fill in extrapolated regions

    Returns:
        np.array: interpolated radiance vector

    """

    H = np.array([spectral_response_function(wl, wi, fwhmi / 2.355)
                  for wi, fwhmi in zip(wl2, fwhm2)])
    if fill is False:
        return np.dot(H, x[:, np.newaxis]).ravel()
    else:
        xnew = np.dot(H, x[:, np.newaxis]).ravel()
        good = np.isfinite(xnew)
        for i, xi in enumerate(xnew):
            if not good[i]:
                nearest_good_ind = np.argmin(abs(wl2[good]-wl2[i]))
                xnew[i] = xnew[nearest_good_ind]
        return xnew

"""""
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
print(tuple(zip(arr1, arr2)))
print(list(zip(arr1, arr2)))

print(np.dot([1,2], [2,3]))


wl2 = np.array([400, 500])
fwhm2 = np.array([100, 200])
wl = np.array([600, 700])
print(tuple(zip(wl2, fwhm2)))
print(type(zip(wl2,fwhm2)))

print(np.array([spectral_response_function(wl, wi, fwhmi / 2.355)
                  for wi, fwhmi in zip(wl2, fwhm2)]))

x = np.array([[1, 2], [3, 4]])
print(x)
y = np.array([[1], [2]])
print(y)
print(y.ndim)
print(y[:, np.newaxis])
print(y.ndim)
z = np.dot(x,y[:, np.newaxis])
"""

"""
H = np.array([spectral_response_function(wl, wi, fwhmi / 2.355)
    for wi, fwhmi in zip(wl2, fwhm2)])
print(np.dot(H, x[:, np.newaxis]))    
print(np.dot(H, x[:, np.newaxis]).ravel())
"""

def load_spectrum(spectrum_file: str) -> (np.array, np.array):
    """Load a single spectrum from a text file with initial columns giving
       wavelength and magnitude, respectively.

    Args:
        spectrum_file: file to load spectrum from

    Returns:
        np.array: spectrum values
        np.array: wavelengths, if available in the file

    """

    spectrum = np.loadtxt(spectrum_file)
    if spectrum.ndim > 1:
        spectrum = spectrum[:, :2]
        wavelengths, spectrum = spectrum.T
        if wavelengths[0] < 100:
            wavelengths = wavelengths * 1000.0  # convert microns -> nm if needed
        return spectrum, wavelengths
    else:
        return spectrum, None

file = StringIO('0.123 0.132 0.426 \n 0.234 0.234 0.132 \n 0.123 0.423 0.435')
spectrum_new, wavelength_new = load_spectrum(file)
assert(wavelength_new.ndim == 1)
assert(spectrum_new.ndim == 1)
assert(wavelength_new[0] > 100)


#existing_path = r"C:\Users\vpatro\Desktop\isofit\wl_multicol.txt"
#abs_path = os.path.isabs(existing_path)
#print(abs_path)
#abs_path = os.path.abspath('wl_multicol.txt')
#print(abs_path)

#print(os.path.isabs("\hasdfuilhasdfh"))

# new function to check if a path is an absolute path, isabs only checks it if begins with a backslash however
# still have issues with this function

def expand_path(directory: str, subpath: str) -> str:
    """Expand a path variable to an absolute path, if it is not one already.
    Args:
        directory:  absolute location
        subpath: path to expand
    Returns:
        str: expanded path
    """

    if subpath.startswith('/'):
        return subpath
    return os.path.join(directory, subpath)

#def test_expand_path() -- backslash vs forward slash discrepancy
assert(expand_path("NASA", "JPL") == "NASA/JPL")
assert(expand_path("NASA", "/JPL") == "/JPL")

subpath = 'isofit/test/test_common.py'

def expand_path_to_absolute(subpath: str):
    if os.path.exists(subpath):
        return os.path.abspath(subpath)
    print('Invalid')

#absolute_path = r'C:\Users\vpatro\Desktop\isofit\isofit\test\test_common.py'
#assert(absolute_path == os.path.abspath(subpath))

subpath = 'isofit/test/test_common.py'
assert(os.path.abspath(subpath) == expand_path_to_absolute(subpath))


def find_header(inputpath: str) -> str:

    """
    Convert a envi binary/header path to a header, handling extensions
    Args:
        inputpath: path to envi binary file
    Returns:
        str: the header file associated with the input reference.
    """
    if os.path.splitext(inputpath)[-1] == '.img' or os.path.splitext(inputpath)[-1] == '.dat' or os.path.splitext(inputpath)[-1] == '.raw':
        # headers could be at either filename.img.hdr or filename.hdr.  Check both, return the one that exists if it
        # does, if not return the latter (new file creation presumed).
        hdrfile = os.path.splitext(inputpath)[0] + '.hdr'
        if os.path.isfile(hdrfile):
            return hdrfile
        elif os.path.isfile(inputpath + '.hdr'):
            return inputpath + '.hdr'
        return hdrfile
    elif os.path.splitext(inputpath)[-1] == '.hdr':
        return inputpath
    else:
        return inputpath + '.hdr'

print(os.path.isfile('Varun_Patro_Resume.pdf'))




def recursive_reencode(j, shell_replace: bool = True): # to be deleted
    """Recursively re-encode a mutable object (ascii->str).

    Args:
        j: object to reencode
        shell_replace: boolean helper for recursive calls

    Returns:
        Object: expanded, reencoded object

    """

    if isinstance(j, dict): # if a dictionary
        for key, value in j.items(): # iterate through items in the dictionary
            j[key] = recursive_reencode(value) # at each key, run the function with the value
        return j
    elif isinstance(j, list):
        for i, k in enumerate(j):
            j[i] = recursive_reencode(k)
        return j
    elif isinstance(j, tuple):
        return tuple([recursive_reencode(k) for k in j])
    else:
        if shell_replace and isinstance(j, str):
            try:
                j = expandvars(j)
            except IndexError:
                pass
        return j

sampleDict = {1: {1: 'Goddard', 2: ['Ames', 'Edwards']}, 2: 'Marshall', 3: 'Kennedy'}
assert(sampleDict == recursive_reencode(sampleDict))

sampleList = ['Marshall', ['Kennedy', 'Glenn'], ('Goddard', 'Johnson')]
assert(sampleList == recursive_reencode(sampleList))


sampleTuple = ({1: 'Glenn', 2: ('Marshall', 'Ames')}, ['Johnson', 'Kennedy'], 'JPL')
assert(sampleTuple == recursive_reencode(sampleTuple))

#print(sampleDict.items())
#print(sampleDict.values())

def recursive_replace(obj, key, val) -> None: # wondering how you know the key for the value you want to repace \
    # might it not be the same key for every nested dictionary?

    # potential issue: might replace objects with dictionaries in them
    # the value of the specified key will get replaced, taking along anything inside with it
    """Find and replace a vector in a nested (mutable) structure.

    Args:
        obj: object to replace within
        key: key to replace
        val: value to replace with

    """

    if isinstance(obj, dict): # if object a dictinoary
        if key in obj: # if specified key contained
            obj[key] = val # replace with specified value
        for item in obj.values(): # for all values in the dictionary
            recursive_replace(item, key, val) # pass those in as new objects
    elif any(isinstance(obj, t) for t in (list, tuple)): # if object is a list or tuple
        for item in obj: # iterating through
            recursive_replace(item, key, val) # ultimately trying to see if any of the items inside is a dictionary containing \
            # specified key

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

"""
def svd_inv_sqrt(C: np.array, hashtable: OrderedDict = None, max_hash_size: int = None) -> (np.array, np.array):
    Matrix inversion, based on decomposition.  Built to be stable, and positive.

    Args:
        C: matrix to invert
        hashtable: if used, the hashtable to store/retrieve results in/from
        max_hash_size: maximum size of hashtable

    Return:
        (np.array, np.array): inverse of C and square root of the inverse of C

    

    # If we have a hash table, look for the precalculated solution
    h = None
    if hashtable is not None:
        # If arrays are in Fortran ordering, they are not hashable.
        if not C.flags['C_CONTIGUOUS']:
            C = C.copy(order='C')
        h = xxhash.xxh64_digest(C)
        if h in hashtable:
            return hashtable[h]

    D, P = scipy.linalg.eigh(C)
    for count in range(3):
        if np.any(D < 0) or np.any(np.isnan(D)):
            inv_eps = 1e-6 * (count+1)*10
            D, P = scipy.linalg.eigh(
                C + np.diag(np.ones(C.shape[0]) * inv_eps))
        else:
            break

        if count == 2:
            raise ValueError('Matrix inversion contains negative values,' +
                             'even after adding {} to the diagonal.'.format(inv_eps))

    Ds = np.diag(1/np.sqrt(D))
    L = P@Ds
    Cinv_sqrt = L@P.T
    Cinv = L@L.T

    # If there is a hash table, cache our solution.  Bound the total cache
    # size by removing any extra items in FIFO order.
    if (hashtable is not None) and (max_hash_size is not None):
        hashtable[h] = (Cinv, Cinv_sqrt)
        while len(hashtable) > max_hash_size:
            hashtable.popitem(last=False)

    return Cinv, Cinv_sqrt

# POSITIVE SEMI-DEFINITE
sample_array = np.array([[13, -4], [-4, 3]])
sample_matrix = np.asmatrix(sample_array)
result_matrix, result_matrix_sq = svd_inv_sqrt(sample_array)
assert(result_matrix.all() == scipy.linalg.inv(sample_matrix).all())
assert((result_matrix_sq @ result_matrix_sq).all() == result_matrix.all())

sample_array_2 = np.array([[7, 0], [0, 1]])
sample_matrix_2 = np.asmatrix(sample_array_2)
result_matrix_2, result_matrix_sq_2 = svd_inv_sqrt(sample_array_2)
assert(result_matrix_2.all() == scipy.linalg.inv(sample_matrix_2).all())
assert((result_matrix_sq_2 @ result_matrix_sq_2).all() == result_matrix_2.all())


sample_array_3 = np.array([[27, 20], [20, 16]])
sample_matrix_3 = np.asmatrix(sample_array_3)
result_matrix_3, result_matrix_sq_3 = svd_inv_sqrt(sample_array_3)
assert(result_matrix_3.all() == scipy.linalg.inv(sample_matrix_3).all())
assert((result_matrix_sq_3 @ result_matrix_sq_3).all() == result_matrix_3.all())


# POSITIVE DEFINITE
sample_array_4 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
sample_matrix_4 = np.asmatrix(sample_array_4)
result_matrix_4, result_matrix_sq_4 = svd_inv_sqrt(sample_array_4)
assert((scipy.linalg.inv(sample_matrix_4)).all() == result_matrix_4.all())
assert((result_matrix_sq_4 @ result_matrix_sq_4).all() == result_matrix_4.all())


sample_array_5 = np.array([[25, 15, -5], [15, 18, 0], [-5, 0, 11]])
sample_matrix_5 = np.asmatrix(sample_array_5)
result_matrix_5, result_matrix_sq_5 = svd_inv_sqrt(sample_array_5)
assert((scipy.linalg.inv(sample_matrix_5)).all() == result_matrix_5.all())
assert((result_matrix_sq_5 @ result_matrix_sq_5).all() == result_matrix_5.all())

def svd_inv(C: np.array, hashtable: OrderedDict = None, max_hash_size: int = None):
    
    Matrix inversion, based on decomposition.  Built to be stable, and positive.

    Args:
        C: matrix to invert
        hashtable: if used, the hashtable to store/retrieve results in/from
        max_hash_size: maximum size of hashtable

    Return:
        np.array: inverse of C

    

    return svd_inv_sqrt(C, hashtable, max_hash_size)[0]

assert(svd_inv(sample_array_3).all() == svd_inv_sqrt(sample_array_3)[0].all())
assert(svd_inv(sample_array_4).all() == svd_inv_sqrt(sample_array_4)[0].all())

"""


def expand_all_paths(to_expand: dict):
    """Expand any dictionary entry containing the string 'file' into
       an absolute path, if needed.

    Args:
        to_expand: dictionary to expand
        absdir: path to expand with (absolute directory)

    Returns:
        dict: dictionary with expanded paths

    """

    def recursive_expand(j):
        if isinstance(j, dict): # if dictionary
            for key, value in j.items(): # through all pairs
                if isinstance(key, str) and \
                    ('file' in key or 'directory' in key or 'path' in key) and \
                        isinstance(value, str): # if both value and key are strings and contain specified word
                    j[key] = expand_path_to_absolute(value) # replace it with its absolute path
                else:
                    j[key] = recursive_expand(value)
            return j
        elif isinstance(j, list):
            for i, k in enumerate(j):
                j[i] = recursive_expand(k)
            return j
        elif isinstance(j, tuple):
            return tuple([recursive_expand(k) for k in j])
        return j

    return recursive_expand(to_expand)

subpath = 'isofit/test/test_common.py'
sample_dict = {'file': ['string_1', {'directory': subpath}], 'path': subpath}
expanded_dict = {'file': ['string_1', {'directory': os.path.abspath(subpath)}], \
    'path': os.path.abspath(subpath)}
assert(expanded_dict == expand_all_paths(sample_dict))
sample_list = ['file', ('string_1', {'path': subpath, 'random': {'directory': subpath}}\
    , 'random')]
expanded_list = ['file', ('string_1', {'path': os.path.abspath(subpath)\
    , 'random': {'directory': os.path.abspath(subpath)}}\
    , 'random')]
assert(expanded_list == expand_all_paths(sample_list))


print("FINISHED")


