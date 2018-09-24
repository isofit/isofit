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
from os.path import expandvars
from scipy.linalg import cholesky, inv, det, svd
from numba import jit

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


def emissive_radiance_old(emissivity, T, wl):
    """Radiance of a surface due to emission"""

    h = 6.62607004e-34  # m2 kg s-1
    c = 299792458  # m s-1
    numerator = 2.0*h*(c**2)  # m4 kg s-1
    wl_m = wl*1e-9
    numerator_per_lam5 = numerator * pow(wl_m, -5)  # kg s-1 m-1
    k = 1.380648520-23  # Boltzmann constant, m2 kg s-2 K-1
    denom = s.exp(h*c/(k*wl_m*T))-1.0  # dimensionless
    L = numerator_per_lam5 / denom  # Watts per m3

    cm2_per_m2, nm_per_m, uW_per_W = 10000, 1e9, 1e6
    conversion = cm2_per_m2 * nm_per_m * uW_per_W / s.pi  # -> uW nm-1 cm-2 sr-1
    L = L * conversion

    ddenom_dT = s.exp(h*c/(k*wl_m*T)) * h*c*(-1.0)/(pow(k*wl_m*T, 2)) * k*wl_m
    dL_dT = -numerator_per_lam5 / pow(denom, 2.0) * ddenom_dT * conversion

    L = L * emissivity
    dL_dT = dL_dT * emissivity
    L[s.logical_not(s.isfinite(L))] = 0
    dL_dT[s.logical_not(s.isfinite(dL_dT))] = 0
    return L, dL_dT


def emissive_radiance(emissivity, T, wl):
    """Radiance of a surface due to emission"""

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


@jit
def chol_inv(C):
    """Fast stable inverse for Hermetian positive definite matrices"""

    R = cholesky(C, lower=False)
    S = inv(R)
    return S.dot(S.T)


@jit
def svd_inv(C, mineig=0, hashtable=None):
    """Fast stable inverse using SVD.  This can handle near-singular matrices"""

    return svd_inv_sqrt(C, mineig, hashtable)[0]


@jit
def svd_inv_sqrt(C, mineig=0, hashtable=None):
    """Fast stable inverse using SVD. This can handle near-singular matrices.
       Also return the square root."""

    h = None
    if hashtable is not None:
        h = xxhash.xxh64_digest(C)
        if h in hashtable:
            return hashtable[h]
    U, V, D = svd(C)
    ignore = s.where(V < mineig)[0]
    Vi = 1.0 / V
    Vi[ignore] = 0
    Visqrt = s.sqrt(Vi)
    Cinv = (D.T).dot(s.diag(Vi)).dot(U.T)
    Cinv_sqrt = (D.T).dot(s.diag(Visqrt)).dot(U.T)
    if hashtable is not None:
        hashtable[h] = (Cinv, Cinv_sqrt)
    return Cinv, Cinv_sqrt


def expand_path(directory, subpath):
    """Expand a path variable to an absolute path, if it is not one already"""

    if subpath.startswith('/'):
        return subpath
    return os.path.join(directory, subpath)


def recursive_replace(obj, key, val):
    """Find and replace a vector in a nested structure"""

    if isinstance(obj, dict):
        if key in obj:
            obj[key] = val
        for item in obj.values():
            recursive_replace(item, key, val)
    elif any(isinstance(obj, t) for t in (list, tuple)):
        for item in obj:
            recursive_replace(item, key, val)


def get_absorption(wl, absfile):
    '''Calculate water and ice absorption coefficients using indices of
  refraction, and interpolate them to new wavelengths (user specifies nm)'''

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


def json_load_ascii(filename, shell_replace=True):
    """Load a hierarchical structure, convert all unicode to ASCII and
    expand environment variables"""

    def recursive_reincode(j):
        if isinstance(j, dict):
            for key, value in j.items():
                j[key] = recursive_reincode(value)
            return j
        elif isinstance(j, list):
            for i, k in enumerate(j):
                j[i] = recursive_reincode(k)
            return j
        elif isinstance(j, tuple):
            return tuple([recursive_reincode(k) for k in j])
        else:
            if shell_replace and type(j) is str:
                try:
                    j = expandvars(j)
                except IndexError:
                    pass
            return j

    with open(filename, 'r') as fin:
        j = json.load(fin)
        return recursive_reincode(j)


def expand_all_paths(config, configdir):
    """Expand any config entry containing the string 'file' into 
       an absolute path, if needed"""

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
            return tuple([recursive_reincode(k) for k in j])
        return j

    return recursive_expand(config)


def find_header(imgfile):
    """Return the header associated with an image file"""
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
    """Turn a subpath into an absolute path if it is not absolute already"""
    if subpath.startswith('/'):
        return subpath
    return os.path.join(directory, subpath)


def rdn_translate(wvn, rdn_wvn):
    """Translate radiance out of wavenumber space"""
    dwvn = wvn[1:]-wvn[:-1]
    dwl = 10000.0/wvn[1:] - 10000.0/wvn[:-1]
    return rdn_wvn*(dwl/dwvn)


def spectrumResample(x, wl, wl2, fwhm2, fill=False):
    """Resample a spectrum to a new wavelength / FWHM. 
       I assume Gaussian SRFs"""
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


def spectrumLoad(init):
    """Load a single spectrum from a text file with initial columns giving
       wavelength and magnitude respectively"""
    wl, x = s.loadtxt(init)[:, :2].T
    if wl[0] < 100:
        wl = wl*1000.0  # convert microns -> nm if needed
    return x, wl


def srf(x, mu, sigma):
    """Spectral Response Function """
    u = (x-mu)/abs(sigma)
    y = (1.0/(s.sqrt(2.0*s.pi)*abs(sigma)))*s.exp(-u*u/2.0)
    return y/y.sum()


class VectorInterpolator:

    def __init__(self, grid, data):
        self.n = data.shape[-1]
        grid_aug = grid + [s.arange(data.shape[-1])]
        self.itp = RegularGridInterpolator(grid_aug, data)

    def __call__(self, points):
        res = []
        for v in s.arange(self.n):
            p_aug = s.concatenate((points, s.array([v])), axis=0)
            res.append(self.itp(p_aug))
        return res


class VectorInterpolatorJIT:

    def __init__(self, grid, data):
        """By convention, the final dimensionn of "data" is the wavelength.
           "grid" contains a list of arrays, each representing the input grid 
           points in the ith dimension of the table."""
        self.in_d = len(data.shape)-1
        self.out_d = data.shape[-1]
        self.grid = [i.copy() for i in grid]
        self.data = data.copy()

    @jit
    def __call__(self, point):
        return jitinterp(self.in_d, self.out_d, self.grid, self.data, point)


@jit
def jitinterp(s_in_d, s_out_d, s_grid, s_data, point):

        # we find the bottom index along each input dimension
    lo_inds = s.zeros(s_in_d)
    lo_fracs = s.zeros(s_in_d)
    stride = []
    for i in s.arange(s_in_d):
        stride.append(s.prod(s_data.shape[(i+1):]))

    for d in s.arange(s_in_d):
        n_gridpoints = len(s_grid[d])
        for j in s.arange(n_gridpoints-1):
            if j == 0 and s_grid[d][j] >= point[d]:
                lo_inds[d] = 0
                lo_fracs[d] = 1.0
                break
            if j == n_gridpoints-2 and s_grid[d][-1] <= point[d]:
                lo_inds[d] = n_gridpoints-2
                lo_fracs[d] = 0.0
                break
            if s_grid[d][j] < point[d] and s_grid[d][j+1] >= point[d]:
                lo_inds[d] = j
                denom = (s_grid[d][j+1]-s_grid[d][j])
                lo_fracs[d] = 1.0 - (point[d]-s_grid[d][j])/denom

    # Now we form a list of all points on the hypercube
    # and the associated fractions of each

    hypercube_bin = binary_table[s_in_d].copy()
    n_hypercube = len(hypercube_bin)
    hypercube_weights = s.ones((n_hypercube))
    hypercube_flat_inds = s.zeros((n_hypercube))

    # simple version
    for i in range(n_hypercube):
        for j in range(s_in_d):
            if hypercube_bin[i, j]:
                hypercube_weights[i] = hypercube_weights[i] * lo_fracs[j]
                hypercube_flat_inds[i] = \
                    hypercube_flat_inds[i] + (lo_inds[j]) * stride[j]
            else:
                hypercube_weights[i] = hypercube_weights[i] * (1.0-lo_fracs[j])
                hypercube_flat_inds[i] = \
                    hypercube_flat_inds[i] + (lo_inds[j]+1) * stride[j]

    # once per output datapoint
    res = s.zeros(s_out_d)
    for oi in s.arange(s_out_d):
        val = 0
        for i in s.arange(n_hypercube):
            ind = int(hypercube_flat_inds[i]+oi)
            res[oi] = res[oi] + s_data.flat[ind] * hypercube_weights[i]
    return s.array(res)


def combos(inds):
    '''Return all combinations of indices in a list of index sublists 
    For example, for the input [[1,2],[3,4,5]] it would return:
        [[1,3],[1,4],[1,5],[2,3],[2,4],[2,5]]
    This is used for interpolation in the high-dimensional LUT'''

    n = len(inds)
    cases = s.prod([len(i) for i in inds])
    return s.array(s.meshgrid(*inds)).reshape((n, cases)).T
