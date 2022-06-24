import numpy as np
import os
from io import StringIO

#from isofit.core.common import get_absorption
#from isofit.core.common import load_wavelen, spectral_response_function, emissive_radiance
# Test for load_wavelen(...)

def load_wavelen(wavelength_file: str):

    """""
    Load a wavelength file, and convert to nanometers if needed.
    Args:
        wavelength_file: file to read wavelengths from
    Returns:
        (np.array, np.array): wavelengths, full-width-half-max
    """""

    q = np.loadtxt(wavelength_file)
    print(q)
    if q.shape[1] > 2:
        q = q[:, 1:3]
    if q[0, 0] < 100:
        q = q * 1000.0
    wl, fwhm = q.T
    return wl, fwhm

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
    # photon energy in eV
    eV_per_sec_cm2_sr_nm = 1.2398 * ph_per_sec_cm2_sr_nm/wl_um
    W_per_cm2_sr_nm = J_per_eV * eV_per_sec_cm2_sr_nm
    uW_per_cm2_sr_nm = W_per_cm2_sr_nm*1e6
    dRdn_dT = c_1/(wl**4)*(-pow(np.exp(c_2/wl/T)-1.0, -2.0)) *\
        np.exp(c_2/wl/T)*(-pow(T, -2)*c_2/wl) *\
        emissivity/wl_um*1.2398*J_per_eV*1e6
    return uW_per_cm2_sr_nm, dRdn_dT


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




print("BEGIN")

myList = ["Ames", "Marshall", "Goddard", "Kennedy"]

for i in myList:
    print(i)




# Case 1: More than 2 columns



#wl = np.random.rand(425,4)
#file = open("wl_sample.txt", "w+")
#np.savetxt("wl_sample.txt", wl)

#wl_modified, fwhm_modified = load_wavelen("C:/Users/vpatro/Desktop/wl_sample.txt")
#file.close()
#assert(wl_modified.ndim == 1)
#assert(fwhm_modified.ndim == 1)
#assert(wl_modified[0] > 100)


#wl_modified, fwhm_modified = load_wavelen("C:/Users/vpatro/Desktop/Test/wl_multicol.txt")
#print(wl_modified.shape)

emissivity = np.array([0.8, 0.8])
T = np.array([300, 300])
wavelength = np.array([400, 600])

#uW_per_cm2_sr_nm_modified, dRdn_dT_modified = emissive_radiance(emissivity, T, wavelength)
#print(uW_per_cm2_sr_nm_modified)
#print(dRdn_dT_modified)

#assert(uW_per_cm2_sr_nm_modified = 7.90527265e-44)

    
#def test_spectral_response_function():
response_range = np.array([10, 8])
mu = 6.0
sigma = -2.0
srf = spectral_response_function(response_range, mu, sigma)
assert(abs(srf[0] - 0.182425524) < 0.0000001)
assert(abs(srf[1] - 0.817574476) < 0.0000001)

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
assert(expand_path("NASA", "JPL") == "NASA\JPL")
assert(expand_path("NASA", "/JPL") == "/JPL")


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




    

print("FINISHED")


