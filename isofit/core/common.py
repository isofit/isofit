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
import scipy as s
from collections import OrderedDict
from scipy.interpolate import RegularGridInterpolator
from os.path import expandvars, split, abspath
from scipy.linalg import cholesky, inv, det, svd, eigh
from numba import jit

from .. import jit_enabled, conditional_decorator


### Variables ###

# Maximum size of our hash tables
max_table_size = 500

binary_table = [s.array([[]]),
                s.array([[0], [1]]),
                s.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                s.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]),
                s.array([[0, 0, 0, 0], [0, 0, 0, 1],
                         [0, 0, 1, 0], [0, 0, 1, 1],
                         [0, 1, 0, 0], [0, 1, 0, 1],
                         [0, 1, 1, 0], [0, 1, 1, 1],
                         [1, 0, 0, 0], [1, 0, 0, 1],
                         [1, 0, 1, 0], [1, 0, 1, 1],
                         [1, 1, 0, 0], [1, 1, 0, 1],
                         [1, 1, 1, 0], [1, 1, 1, 1]]),
                s.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0], [0, 0, 0, 1, 1],
                         [0, 0, 1, 0, 0], [0, 0, 1, 0, 1],
                         [0, 0, 1, 1, 0], [0, 0, 1, 1, 1],
                         [0, 1, 0, 0, 0], [0, 1, 0, 0, 1],
                         [0, 1, 0, 1, 0], [0, 1, 0, 1, 1],
                         [0, 1, 1, 0, 0], [0, 1, 1, 0, 1],
                         [0, 1, 1, 1, 0], [0, 1, 1, 1, 1],
                         [1, 0, 0, 0, 0], [1, 0, 0, 0, 1],
                         [1, 0, 0, 1, 0], [1, 0, 0, 1, 1],
                         [1, 0, 1, 0, 0], [1, 0, 1, 0, 1],
                         [1, 0, 1, 1, 0], [1, 0, 1, 1, 1],
                         [1, 1, 0, 0, 0], [1, 1, 0, 0, 1],
                         [1, 1, 0, 1, 0], [1, 1, 0, 1, 1],
                         [1, 1, 1, 0, 0], [1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
                s.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1],
                         [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1],
                         [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1],
                         [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1],
                         [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1],
                         [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1],
                         [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1],
                         [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1],
                         [0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 1],
                         [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 1],
                         [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1],
                         [0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 1],
                         [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 1],
                         [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1],
                         [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1],
                         [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 1],
                         [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1],
                         [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1],
                         [1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1],
                         [1, 0, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1],
                         [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1],
                         [1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 1],
                         [1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 1],
                         [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1],
                         [1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1],
                         [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])]

eps = 1e-5  # small value used in finite difference derivatives


### Classes ###

class VectorInterpolator:
    """."""

    def __init__(self, grid, data):
        self.n = data.shape[-1]
        grid_aug = grid + [s.arange(data.shape[-1])]
        self.itp = RegularGridInterpolator(grid_aug, data,
                bounds_error=False, fill_value=None)

    def __call__(self, points):

        x = s.zeros((self.n,len(points)+1))
        for i in range(len(points)):
            x[:,i] = points[i]
        # This last dimension is always an integer so no
        # interpolation is performed. This is done only
        # for performance reasons.
        x[:,-1] = s.arange(self.n)
        res = self.itp(x)

        return res


### Functions ###


def load_wavelen(wavelength_file):
    """Load a wavelength file, and convert to nanometers if needed."""

    q = s.loadtxt(wavelength_file)
    if q.shape[1] > 2:
        q = q[:, 1:3]
    if q[0, 0] < 100:
        q = q * 1000.0
    wl, fwhm = q.T
    return wl, fwhm


def emissive_radiance(emissivity, T, wl):
    """Radiance of a surface due to emission."""

    c_1 = 1.88365e32/s.pi
    c_2 = 14387690
    J_per_eV = 1.60218e-19
    wl_um = wl / 1000.0
    ph_per_sec_cm2_sr_nm = c_1/(wl**4)/(s.exp(c_2/wl/T)-1.0) * emissivity
    # photon energy in eV
    eV_per_sec_cm2_sr_nm = 1.2398 * ph_per_sec_cm2_sr_nm/wl_um
    W_per_cm2_sr_nm = J_per_eV * eV_per_sec_cm2_sr_nm
    uW_per_cm2_sr_nm = W_per_cm2_sr_nm*1e6
    dRdn_dT = c_1/(wl**4)*(-pow(s.exp(c_2/wl/T)-1.0, -2.0)) *\
        s.exp(c_2/wl/T)*(-pow(T, -2)*c_2/wl) *\
        emissivity/wl_um*1.2398*J_per_eV*1e6
    return uW_per_cm2_sr_nm, dRdn_dT


@conditional_decorator(jit, jit_enabled, forceobj=True)
def svd_inv(C, mineig=0, hashtable=None):
    """Fast stable inverse using SVD. This can handle near-singular matrices."""

    return svd_inv_sqrt(C, mineig, hashtable)[0]


@conditional_decorator(jit, jit_enabled)
def svd_inv_sqrt(C, mineig=0, hashtable=None):
    """Fast stable inverse using SVD. This can handle near-singular matrices.
    Also return the square root.
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

    # Cholesky decomposition seems to be too unstable for solving this
    # problem, so we use eigendecompostition instead.
    D, P = eigh(C)
    Ds = s.diag(1/s.sqrt(D))
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


def expand_path(directory, subpath):
    """Expand a path variable to an absolute path, if it is not one already."""

    if subpath.startswith('/'):
        return subpath
    return os.path.join(directory, subpath)


def recursive_replace(obj, key, val):
    """Find and replace a vector in a nested structure."""

    if isinstance(obj, dict):
        if key in obj:
            obj[key] = val
        for item in obj.values():
            recursive_replace(item, key, val)
    elif any(isinstance(obj, t) for t in (list, tuple)):
        for item in obj:
            recursive_replace(item, key, val)


def get_absorption(wl, absfile):
    """Calculate water and ice absorption coefficients using indices of
    refraction, and interpolate them to new wavelengths (user specifies nm)."""

    # read the indices of refraction
    q = s.loadtxt(absfile, delimiter=',')
    wl_orig_nm = q[:, 0]
    wl_orig_cm = wl_orig_nm/1e9*1e2
    water_imag = q[:, 2]
    ice_imag = q[:, 4]

    # calculate absorption coefficients in cm^-1
    water_abscf = water_imag*s.pi*4.0/wl_orig_cm
    ice_abscf = ice_imag*s.pi*4.0/wl_orig_cm

    # interpolate to new wavelengths (user provides nm)
    water_abscf_intrp = s.interp(wl, wl_orig_nm, water_abscf)
    ice_abscf_intrp = s.interp(wl, wl_orig_nm, ice_abscf)
    return water_abscf_intrp, ice_abscf_intrp


def recursive_reencode(j, shell_replace=True):
    """Recursively re-encode a dictionary."""

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


def json_load_ascii(filename, shell_replace=True):
    """Load a hierarchical structure, convert all unicode to ASCII and
    expand environment variables."""

    with open(filename, 'r') as fin:
        j = json.load(fin)
        return recursive_reencode(j, shell_replace)


def load_config(config_file):
    """Configuration files are typically .json, with relative paths."""

    with open(config_file, 'r') as f:
        config = json.load(f)

    configdir, f = split(abspath(config_file))
    return expand_all_paths(config, configdir)


def expand_all_paths(config, configdir):
    """Expand any config entry containing the string 'file' into 
       an absolute path, if needed."""

    def recursive_expand(j):
        if isinstance(j, dict):
            for key, value in j.items():
                if isinstance(key, str) and \
                    ('file' in key or 'directory' in key or 'path' in key) and \
                        isinstance(value, str):
                    j[key] = expand_path(configdir, value)
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

    return recursive_expand(config)


def find_header(imgfile):
    """Return the header associated with an image file."""

    if os.path.exists(imgfile+'.hdr'):
        return imgfile+'.hdr'
    ind = imgfile.rfind('.raw')
    if ind >= 0:
        return imgfile[0:ind]+'.hdr'
    ind = imgfile.rfind('.img')
    if ind >= 0:
        return imgfile[0:ind]+'.hdr'
    raise IOError('No header found for file {0}'.format(imgfile))


def expand_path(directory, subpath):
    """Turn a subpath into an absolute path if it is not absolute already."""

    if subpath.startswith('/'):
        return subpath
    return os.path.join(directory, subpath)


def rdn_translate(wvn, rdn_wvn):
    """Translate radiance out of wavenumber space."""

    dwvn = wvn[1:]-wvn[:-1]
    dwl = 10000.0/wvn[1:] - 10000.0/wvn[:-1]
    return rdn_wvn*(dwl/dwvn)


def resample_spectrum(x, wl, wl2, fwhm2, fill=False):
    """Resample a spectrum to a new wavelength / FWHM. 
       I assume Gaussian SRFs."""

    H = s.array([srf(wl, wi, fwhmi/2.355)
                 for wi, fwhmi in zip(wl2, fwhm2)])
    if fill is False:
        return s.dot(H, x[:, s.newaxis]).ravel()
    else:
        xnew = s.dot(H, x[:, s.newaxis]).ravel()
        good = s.isfinite(xnew)
        for i, xi in enumerate(xnew):
            if not good[i]:
                nearest_good_ind = s.argmin(abs(wl2[good]-wl2[i]))
                xnew[i] = xnew[nearest_good_ind]
        return xnew


def load_spectrum(init):
    """Load a single spectrum from a text file with initial columns giving
       wavelength and magnitude, respectively."""

    x = s.loadtxt(init)
    if x.ndim > 1:
        x = x[:, :2]
        wl, x = x.T
        if wl[0] < 100:
            wl = wl*1000.0  # convert microns -> nm if needed
        return x, wl
    else:
        return x, None


def srf(x, mu, sigma):
    """Spectral response function."""

    u = (x-mu)/abs(sigma)
    y = (1.0/(s.sqrt(2.0*s.pi)*abs(sigma)))*s.exp(-u*u/2.0)
    return y/y.sum()


def combos(inds):
    """Return all combinations of indices in a list of index sublists.
    For example, for the input [[1, 2], [3, 4, 5]] it would return:
        [[1, 3], [2, 3], [1, 4], [2, 4], [1, 5], [2, 5]]
    This is used for interpolation in the high-dimensional LUT.
    """

    n = len(inds)
    cases = s.prod([len(i) for i in inds])
    return s.array(s.meshgrid(*inds)).reshape((n, cases)).T
