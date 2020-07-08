#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#

import os
import json
import xxhash
import numpy as np
import scipy.linalg
from scipy.interpolate import RegularGridInterpolator
from os.path import expandvars, split, abspath
from typing import List
from collections import OrderedDict


### Variables ###

# Maximum size of our hash tables
max_table_size = 500

# small value used in finite difference derivatives
eps = 1e-5


### Classes ###

class VectorInterpolator:
    """ Linear look up table interpolator.  Support linear interpolation through radial space by expanding the look
        up tables with sin and cos dimensions.

        Args:
            grid_input: list of lists of floats, indicating the gridpoint elements in each grid dimension
            data_input: n dimensional array of radiative transfer engine outputs (each dimension size corresponds to the
                        given grid_input list length, with the last dimensions equal to the number of sensor channels)
            lut_interp_types: a list indicating if each dimension is in radiance (r), degrees (r), or normal (n) units.
        """

    def __init__(self, grid_input: List[List[float]], data_input: np.array, lut_interp_types: List[str]):
        self.lut_interp_types = lut_interp_types

        # Lists and arrays are mutable, so copy first
        grid = grid_input.copy()
        data = data_input.copy()

        # expand grid dimensionality as needed
        [radian_locations] = np.where(self.lut_interp_types == 'd')
        [degree_locations] = np.where(self.lut_interp_types == 'r')
        angle_locations = np.hstack([radian_locations, degree_locations])
        angle_types = np.hstack(
            [self.lut_interp_types[radian_locations],
             self.lut_interp_types[degree_locations]])
        for _angle_loc in range(len(angle_locations)):

            angle_loc = angle_locations[_angle_loc]
            # get original grid at given location
            original_grid_subset = np.array(grid[angle_loc])

            # convert for angular coordinates
            if (angle_types[_angle_loc] == 'r'):
                grid_subset_cosin = np.cos(original_grid_subset)
                grid_subset_sin = np.sin(original_grid_subset)
            elif (angle_types[_angle_loc] == 'd'):
                grid_subset_cosin = np.cos(original_grid_subset / 180. * np.pi)
                grid_subset_sin = np.sin(original_grid_subset / 180. * np.pi)

            # handle the fact that the grid may no longer be in order
            grid_subset_cosin_order = np.argsort(grid_subset_cosin)
            grid_subset_sin_order = np.argsort(grid_subset_sin)

            # convert current grid location, and add a second
            grid[angle_loc] = grid_subset_cosin[grid_subset_cosin_order]
            grid.insert(angle_loc+1, grid_subset_sin[grid_subset_sin_order])

            # now copy the data to be interpolated through the extra dimension,
            # at the specific angle_loc axes.  We'll use broadcast_to to do
            # this, but we need to do it on the last dimension.  So start by
            # temporarily moving the target axes there, then broadcasting
            data = np.swapaxes(data, -1, angle_loc)
            data_dim = list(np.shape(data))
            data_dim.append(data_dim[-1])
            data = data[..., np.newaxis] * np.ones(data_dim)

            # Now we need to actually copy the data between the first two axes,
            # as broadcast_to doesn't do this
            for ind in range(data.shape[-1]):
                data[..., ind] = data[..., :, ind]

            # Now re-order the cosin dimension
            data = data[..., grid_subset_cosin_order, :]
            # Now re-order the sin dimension
            data = data[..., grid_subset_sin_order]

            # now re-arrange the axes so they're in the right order again,
            dst_axes = np.arange(len(data.shape)-2).tolist()
            dst_axes.insert(angle_loc, len(data.shape)-2)
            dst_axes.insert(angle_loc+1, len(data.shape)-1)
            dst_axes.remove(angle_loc)
            dst_axes.append(angle_loc)
            data = np.ascontiguousarray(np.transpose(data, axes=dst_axes))

            # update the rest of the angle locations
            angle_locations += 1

        self.n = data.shape[-1]
        grid_aug = grid + [np.arange(data.shape[-1])]
        self.itp = RegularGridInterpolator(grid_aug, data,
                                           bounds_error=False, fill_value=None)

    def __call__(self, points):

        x = np.zeros((self.n, len(points) + 1 +
                      np.sum(self.lut_interp_types != 'n')))
        offset_count = 0
        for i in range(len(points)):
            if self.lut_interp_types[i] == 'n':
                x[:, i + offset_count] = points[i]
            elif self.lut_interp_types[i] == 'r':
                x[:, i + offset_count] = np.cos(points[i])
                x[:, i + 1 + offset_count] = np.sin(points[i])
                offset_count += 1
            elif self.lut_interp_types[i] == 'd':
                x[:, i + offset_count] = np.cos(points[i] / 180. * np.pi)
                x[:, i + 1 + offset_count] = np.sin(points[i] / 180. * np.pi)
                offset_count += 1

        # This last dimension is always an integer so no
        # interpolation is performed. This is done only
        # for performance reasons.
        x[:, -1] = np.arange(self.n)
        res = self.itp(x)

        return res


def load_wavelen(wavelength_file: str):
    """Load a wavelength file, and convert to nanometers if needed.

    Args:
        wavelength_file: file to read wavelengths from

    Returns:
        (np.array, np.array): wavelengths, full-width-half-max

    """

    q = np.loadtxt(wavelength_file)
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


def svd_inv(C: np.array, hashtable: OrderedDict = None):
    """Matrix inversion, based on decomposition.  Built to be stable, and positive.

    Args:
        C: matrix to invert
        hashtable: if used, the hashtable to store/retrieve results in/from

    Return:
        np.array: inverse of C

    """

    return svd_inv_sqrt(C, hashtable)[0]


def svd_inv_sqrt(C: np.array, hashtable: OrderedDict = None) -> (np.array, np.array):
    """Matrix inversion, based on decomposition.  Built to be stable, and positive.

    Args:
        C: matrix to invert
        hashtable: if used, the hashtable to store/retrieve results in/from

    Return:
        (np.array, np.array): inverse of C and square root of the inverse of C

    """

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
            inv_eps = 1e-6 * (count-1)*10
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
    if hashtable is not None:
        hashtable[h] = (Cinv, Cinv_sqrt)
        while len(hashtable) > max_table_size:
            hashtable.popitem(last=False)

    return Cinv, Cinv_sqrt


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


def recursive_replace(obj, key, val) -> None:
    """Find and replace a vector in a nested (mutable) structure.

    Args:
        obj: object to replace within
        key: key to replace
        val: value to replace with

    """

    if isinstance(obj, dict):
        if key in obj:
            obj[key] = val
        for item in obj.values():
            recursive_replace(item, key, val)
    elif any(isinstance(obj, t) for t in (list, tuple)):
        for item in obj:
            recursive_replace(item, key, val)


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


def recursive_reencode(j, shell_replace: bool = True):
    """Recursively re-encode a mutable object (ascii->str).

    Args:
        j: object to reencode
        shell_replace: boolean helper for recursive calls

    Returns:
        Object: expanded, reencoded object

    """

    if isinstance(j, dict):
        for key, value in j.items():
            j[key] = recursive_reencode(value)
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


def json_load_ascii(filename: str, shell_replace: bool = True) -> dict:
    """Load a hierarchical structure, convert all unicode to ASCII and
    expand environment variables.

    Args:
        filename: json file to load from
        shell_replace: boolean

    Returns:
        dict: encoded dictionary

    """

    with open(filename, 'r') as fin:
        j = json.load(fin)
        return recursive_reencode(j, shell_replace)


def expand_all_paths(to_expand: dict, absdir: str):
    """Expand any dictionary entry containing the string 'file' into
       an absolute path, if needed.

    Args:
        to_expand: dictionary to expand
        absdir: path to expand with (absolute directory)

    Returns:
        dict: dictionary with expanded paths

    """

    def recursive_expand(j):
        if isinstance(j, dict):
            for key, value in j.items():
                if isinstance(key, str) and \
                    ('file' in key or 'directory' in key or 'path' in key) and \
                        isinstance(value, str):
                    j[key] = expand_path(absdir, value)
                else:
                    j[key] = recursive_expand(value)
            return j
        elif isinstance(j, list):
            for i, k in enumerate(j):
                j[i] = recursive_expand(k)
            return j
        elif isinstance(j, tuple):
            return tuple([recursive_reencode(k) for k in j])
        return j

    return recursive_expand(to_expand)


def find_header(imgfile: str) -> str:
    """Safely return the header associated with an image file.

    Args:
        imgfile: file name of base image

    Returns:
        str: header filename if one exists

    """

    if os.path.exists(imgfile+'.hdr'):
        return imgfile+'.hdr'
    ind = imgfile.rfind('.raw')
    if ind >= 0:
        return imgfile[0:ind]+'.hdr'
    ind = imgfile.rfind('.img')
    if ind >= 0:
        return imgfile[0:ind]+'.hdr'
    raise IOError('No header found for file {0}'.format(imgfile))


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




def combos(inds: List[List[float]]) -> np.array:
    """Return all combinations of indices in a list of index sublists.
    For example, the call::
        combos([[1, 2], [3, 4, 5]])
        ...[[1, 3], [2, 3], [1, 4], [2, 4], [1, 5], [2, 5]]

    This is used for interpolation in the high-dimensional LUT.

    Args:
        inds: list of lists of values to expand

    Returns:
        np.array: meshgrid array of combinations

    """

    n = len(inds)
    cases = np.prod([len(i) for i in inds])
    gridded_combinations = np.array(np.meshgrid(*inds)).reshape((n, cases)).T
    return gridded_combinations


def conditional_gaussian(mu: np.array, C: np.array, window: np.array, remain: np.array, x: np.array) -> \
        (np.array, np.array):
    """Define the conditional Gaussian distribution for convenience.

    len(window)+len(remain)=len(x)

    Args:
        mu: mean values
        C: matrix for conditioning
        window: contains all indices not in remain
        remain: contains indices of the observed part x1
        x: values to condition with

    Returns:
        (np.array, np.array): conditional mean, conditional covariance

    """
    w = np.array(window)[:,np.newaxis]
    r = np.array(remain)[:,np.newaxis]
    C11 = C[r, r.T]
    C12 = C[r, w.T]
    C21 = C[w, r.T]
    C22 = C[w, w.T]

    Cinv = svd_inv(C11)
    conditional_mean = mu[window] + C21 @ Cinv @ (x-mu[remain])
    conditional_cov = C22 - C21 @ Cinv @ C12
    return conditional_mean, conditional_cov

