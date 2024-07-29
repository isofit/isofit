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

import json
import os
from argparse import ArgumentError
from collections import OrderedDict
from os.path import expandvars
from typing import List

import numpy as np
import scipy.linalg
import xxhash
from scipy.interpolate import RegularGridInterpolator

### Variables ###

# small value used in finite difference derivatives
eps = 1e-5


### Classes ###


class VectorInterpolator:
    """Linear look up table interpolator.  Support linear interpolation through radial space by expanding the look
    up tables with sin and cos dimensions.

    Args:
        grid_input: list of lists of floats, indicating the gridpoint elements in each grid dimension
        data_input: n dimensional array of radiative transfer engine outputs (each dimension size corresponds to the
                    given grid_input list length, with the last dimensions equal to the number of sensor channels)
        version: version to use: 'rg' for scipy RegularGridInterpolator, 'mlg' for multilinear grid interpolator
    """

    def __init__(
        self,
        grid_input: List[List[float]],
        data_input: np.array,
        version="nds-1",
    ):
        # Determine if this a singular unique value, if so just return that directly
        val = data_input[(0,) * data_input.ndim]
        if np.isnan(val) and np.isnan(data_input).all() or np.all(data_input == val):
            self.method = -1
            self.value = val
            return

        self.single_point_data = None

        # Lists and arrays are mutable, so copy first
        grid = grid_input.copy()
        data = data_input.copy()

        # Check if we are using a single grid point. If so, store the grid input.
        if np.prod(list(map(len, grid))) == 1:
            self.single_point_data = data
        self.n = data.shape[-1]

        # RegularGrid
        if version == "rg":
            grid_aug = grid + [np.arange(data.shape[-1])]
            self.itp = RegularGridInterpolator(
                grid_aug, data, bounds_error=False, fill_value=None
            )
            self.method = 1

        # Multilinear Grid
        elif version == "mlg":
            self.method = 2
            self.cache = {
                "points": [np.nan] * len(grid),
                "deltas": [np.nan] * len(grid),
                "diff": [np.nan] * len(grid),
                "idx": [...] * len(grid),
            }

            self.gridtuples = [np.array(t) for t in grid]
            self.gridarrays = data
            self.binwidth = [
                t[1:] - t[:-1] for t in self.gridtuples
            ]  # binwidth arrays for each dimension
            self.maxbaseinds = np.array([len(t) - 1 for t in self.gridtuples])

        else:
            raise ArgumentError(None, f"Unknown interpolator version: {version!r}")

    def _interpolate(self, points):
        """
        Supports styles 'rg' and 'nds-k'
        """
        # If we only have one point, we can't do any interpolation, so just
        # return the original data.
        if self.single_point_data is not None:
            return self.single_point_data

        x = np.zeros((self.n, len(points) + 1))
        x[:, :-1] = points

        # This last dimension is always an integer so no
        # interpolation is performed. This is done only
        # for performance reasons.
        x[:, -1] = np.arange(self.n)
        res = self.itp(x)

        return res

    def _multilinear_grid(self, points):
        """
        Cached version of Jouni's implementation

        Args:
            points: The point being interpolated. If at the limit, the extremal value in
                    the grid is returned.

        Returns:
            cube: np.ndarray
        """
        # Retrieve which indices to update
        cached = np.where(points == self.cache["points"])[0]
        update = set(range(points.size)) - set(cached)

        # Update the cached point
        self.cache["points"] = points

        # Update indices that are different from the last point
        for i in update:
            j = np.searchsorted(self.gridtuples[i][:-1], points[i]) - 1
            self.cache["deltas"][i] = (
                points[i] - self.gridtuples[i][j]
            ) / self.binwidth[i][j]
            self.cache["diff"][i] = 1 - self.cache["deltas"][i]

            # Eliminate indices where it is outside the grid range or on a grid point
            if points[i] >= self.gridtuples[i][-1]:
                self.cache["idx"][i] = max(min(self.maxbaseinds[i] + 2, j + 2), 2) - 1
            elif points[i] <= self.gridtuples[i][0]:
                self.cache["idx"][i] = max(min(self.maxbaseinds[i], j), 0)
            else:
                self.cache["idx"][i] = slice(
                    max(min(self.maxbaseinds[i], j), 0),
                    max(min(self.maxbaseinds[i] + 2, j + 2), 2),
                )

        cube = np.copy(self.gridarrays[tuple(self.cache["idx"])], order="A")

        # Only linear interpolate sliced dimensions
        for i, idx in enumerate(self.cache["idx"]):
            if isinstance(idx, slice):
                cube[0] *= self.cache["diff"][i]
                cube[1] *= self.cache["deltas"][i]
                cube[0] += cube[1]
                cube = cube[0]

        return cube

    def __call__(self, *args, **kwargs):
        """
        Passes args to the appropriate interpolation method defined by the version at
        object init.
        """
        if self.method == -1:
            return self.value
        elif self.method == 1:
            return self._interpolate(*args, **kwargs)
        elif self.method == 2:
            return self._multilinear_grid(*args, **kwargs)


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


def emissive_radiance(
    emissivity: np.array, T: np.array, wl: np.array
) -> (np.array, np.array):
    """Calcluate the radiance of a surface due to emission.

    Args:
        emissivity: surface emissivity.
        T: surface temperature [K]
        wl: emmissivity wavelengths [nm]

    Returns:
        np.array: surface upwelling radiance in uW $cm^{-2} sr^{-1} nm^{-nm}$
        np.array: partial derivative of radiance with respect to temperature uW $cm^{-2} sr^{-1} nm^{-1} k^{-1}$

    """

    c_1 = 1.88365e32 / np.pi
    c_2 = 14387690
    J_per_eV = 1.60218e-19
    wl_um = wl / 1000.0
    ph_per_sec_cm2_sr_nm = c_1 / (wl**4) / (np.exp(c_2 / wl / T) - 1.0) * emissivity
    # photon energy in eV
    eV_per_sec_cm2_sr_nm = 1.2398 * ph_per_sec_cm2_sr_nm / wl_um
    W_per_cm2_sr_nm = J_per_eV * eV_per_sec_cm2_sr_nm
    uW_per_cm2_sr_nm = W_per_cm2_sr_nm * 1e6
    dRdn_dT = (
        c_1
        / (wl**4)
        * (-pow(np.exp(c_2 / wl / T) - 1.0, -2.0))
        * np.exp(c_2 / wl / T)
        * (-pow(T, -2) * c_2 / wl)
        * emissivity
        / wl_um
        * 1.2398
        * J_per_eV
        * 1e6
    )
    return uW_per_cm2_sr_nm, dRdn_dT


def svd_inv(C: np.array, hashtable: OrderedDict = None, max_hash_size: int = None):
    """Matrix inversion, based on decomposition.  Built to be stable, and positive.

    Args:
        C: matrix to invert
        hashtable: if used, the hashtable to store/retrieve results in/from
        max_hash_size: maximum size of hashtable

    Return:
        np.array: inverse of C

    """

    return svd_inv_sqrt(C, hashtable, max_hash_size)[0]


def svd_inv_sqrt(
    C: np.array, hashtable: OrderedDict = None, max_hash_size: int = None
) -> (np.array, np.array):
    """Matrix inversion, based on decomposition.  Built to be stable, and positive.

    Args:
        C: matrix to invert
        hashtable: if used, the hashtable to store/retrieve results in/from
        max_hash_size: maximum size of hashtable

    Return:
        (np.array, np.array): inverse of C and square root of the inverse of C

    """

    # If we have a hash table, look for the precalculated solution
    h = None
    if hashtable is not None:
        # If arrays are in Fortran ordering, they are not hashable.
        if not C.flags["C_CONTIGUOUS"]:
            C = C.copy(order="C")
        h = xxhash.xxh64_digest(C)
        if h in hashtable:
            return hashtable[h]

    D, P = scipy.linalg.eigh(C)
    for count in range(3):
        if np.any(D < 0) or np.any(np.isnan(D)):
            inv_eps = 1e-6 * (count - 1) * 10
            D, P = scipy.linalg.eigh(C + np.diag(np.ones(C.shape[0]) * inv_eps))
        else:
            break

        if count == 2:
            raise ValueError(
                "Matrix inversion contains negative values,"
                + "even after adding {} to the diagonal.".format(inv_eps)
            )

    Ds = np.diag(1 / np.sqrt(D))
    L = P @ Ds
    Cinv_sqrt = L @ P.T
    Cinv = L @ L.T

    # If there is a hash table, cache our solution.  Bound the total cache
    # size by removing any extra items in FIFO order.
    if (hashtable is not None) and (max_hash_size is not None):
        hashtable[h] = (Cinv, Cinv_sqrt)
        while len(hashtable) > max_hash_size:
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

    if subpath.startswith("/"):
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
    q = np.loadtxt(absfile, delimiter=",")
    wl_orig_nm = q[:, 0]
    wl_orig_cm = wl_orig_nm / 1e9 * 1e2
    water_imag = q[:, 2]
    ice_imag = q[:, 4]

    # calculate absorption coefficients in cm^-1
    water_abscf = water_imag * np.pi * 4.0 / wl_orig_cm
    ice_abscf = ice_imag * np.pi * 4.0 / wl_orig_cm

    # interpolate to new wavelengths (user provides nm)
    water_abscf_intrp = np.interp(wl, wl_orig_nm, water_abscf)
    ice_abscf_intrp = np.interp(wl, wl_orig_nm, ice_abscf)
    return water_abscf_intrp, ice_abscf_intrp


def get_refractive_index(k_wi, a, b, col_wvl, col_k):
    """Convert refractive index table entries to numpy array.

    Args:
        k_wi:    variable
        a:       start line
        b:       end line
        col_wvl: wavelength column in pandas table
        col_k:   k column in pandas table

    Returns:
        wvl_arr: array of wavelengths
        k_arr:   array of imaginary parts of refractive index
    """

    wvl_ = []
    k_ = []

    for ii in range(a, b):
        wvl = k_wi.at[ii, col_wvl]
        k = k_wi.at[ii, col_k]
        wvl_.append(wvl)
        k_.append(k)

    wvl_arr = np.asarray(wvl_)
    k_arr = np.asarray(k_)

    return wvl_arr, k_arr


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

    with open(filename, "r") as fin:
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
                if (
                    isinstance(key, str)
                    and ("file" in key or "directory" in key or "path" in key)
                    and isinstance(value, str)
                ):
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

    if os.path.exists(imgfile + ".hdr"):
        return imgfile + ".hdr"
    ind = imgfile.rfind(".raw")
    if ind >= 0:
        return imgfile[0:ind] + ".hdr"
    ind = imgfile.rfind(".img")
    if ind >= 0:
        return imgfile[0:ind] + ".hdr"
    raise IOError("No header found for file {0}".format(imgfile))


def resample_spectrum(
    x: np.array, wl: np.array, wl2: np.array, fwhm2: np.array, fill: bool = False
) -> np.array:
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
    H = np.array(
        [
            spectral_response_function(wl, wi, fwhmi / 2.355)
            for wi, fwhmi in zip(wl2, fwhm2)
        ]
    )
    H[np.isnan(H)] = 0

    dims = len(x.shape)
    if fill:
        if dims > 1:
            raise Exception("resample_spectrum(fill=True) only works with vectors")

        x = x.reshape(-1, 1)
        xnew = np.dot(H, x).ravel()
        good = np.isfinite(xnew)
        for i, xi in enumerate(xnew):
            if not good[i]:
                nearest_good_ind = np.argmin(abs(wl2[good] - wl2[i]))
                xnew[i] = xnew[nearest_good_ind]
        return xnew
    else:
        # Replace NaNs with zeros
        x[np.isnan(x)] = 0

        # Matrix
        if dims > 1:
            return np.dot(H, x.T).T

        # Vector
        else:
            x = x.reshape(-1, 1)
            return np.dot(H, x).ravel()


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

    u = (response_range - mu) / abs(sigma)
    y = (1.0 / (np.sqrt(2.0 * np.pi) * abs(sigma))) * np.exp(-u * u / 2.0)
    srf = y / y.sum()
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


def conditional_gaussian(
    mu: np.array, C: np.array, window: np.array, remain: np.array, x: np.array
) -> (np.array, np.array):
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
    w = np.array(window)[:, np.newaxis]
    r = np.array(remain)[:, np.newaxis]
    C11 = C[r, r.T]
    C12 = C[r, w.T]
    C21 = C[w, r.T]
    C22 = C[w, w.T]

    Cinv = svd_inv(C11)
    conditional_mean = mu[window] + C21 @ Cinv @ (x - mu[remain])
    conditional_cov = C22 - C21 @ Cinv @ C12
    return conditional_mean, conditional_cov


def envi_header(inputpath):
    """
    Convert a envi binary/header path to a header, handling extensions
    Args:
        inputpath: path to envi binary file
    Returns:
        str: the header file associated with the input reference.

    """
    if (
        os.path.splitext(inputpath)[-1] == ".img"
        or os.path.splitext(inputpath)[-1] == ".dat"
        or os.path.splitext(inputpath)[-1] == ".raw"
    ):
        # headers could be at either filename.img.hdr or filename.hdr.  Check both, return the one that exists if it
        # does, if not return the latter (new file creation presumed).
        hdrfile = os.path.splitext(inputpath)[0] + ".hdr"
        if os.path.isfile(hdrfile):
            return hdrfile
        elif os.path.isfile(inputpath + ".hdr"):
            return inputpath + ".hdr"
        return hdrfile
    elif os.path.splitext(inputpath)[-1] == ".hdr":
        return inputpath
    else:
        return inputpath + ".hdr"


def ray_start(num_cores, num_cpus=2, memory_b=-1):
    import subprocess

    base_args = [
        "ray",
        "start",
        "--head",
        "--num-cpus",
        str(int(num_cores / num_cpus)),
        "--include-dashboard",
        "0",
    ]
    if memory_b != -1:
        base_args.append("--memory")
        base_args.append(str(int(memory_b / num_cpus)))

    head_args = base_args.copy()
    head_args.append("--head")
    result = subprocess.run(head_args, capture_output=True)
    stdout = str(result.stdout, encoding="utf-8")

    if num_cpus > 1:
        key = "--address="
        start_loc = stdout.find(key) + len(key) + 1
        end_loc = stdout.find("'", start_loc)

        address = stdout[start_loc:end_loc]
        base_args.append("--address")
        base_args.append(address)

        result = subprocess.run(base_args, capture_output=True)


from datetime import datetime as dtt


class Track:
    """
    Tracks and reports the percentage complete for some arbitrary sized iterable.

    Borrowed from mlky
    """

    def __init__(self, total, step=5, print=print, reverse=False, message="complete"):
        """
        Parameters
        ----------
        total: int, iterable
            Total items in iterable. If iterable, will call len() on it
        step: float, default=0.05
            Percentage step size to use for reporting, eg. 0.05 is every 5%
        print: func, default=print
            Print function to use, eg. logging.info
        reverse: bool, default=False
            Reverse the count such that 0 is 100%
        message: str, default="complete"
            Message to be included in the output
        """
        if hasattr(total, "__iter__"):
            total = len(total)

        self.step = step
        self.total = total
        self.print = print
        self.start = dtt.now()
        self.percent = step
        self.reverse = reverse
        self.message = message

    def __call__(self, count):
        """
        Parameters
        ----------
        count: int, iterable
            The current count of items finished. If iterable, will call len() on it

        Returns
        -------
        bool
            True if a percentage step was just crossed, False otherwise
        """
        if hasattr(count, "__iter__"):
            count = len(count)

        current = count / self.total
        if self.reverse:
            current = 1 - current
        current *= 100

        if current >= self.percent:
            elap = dtt.now() - self.start
            rate = elap / self.total
            esti = 100 / self.percent * elap - elap

            self.print(
                f"{current:6.2f}% {self.message} (elapsed: {elap}, rate: {rate}, eta: {esti})"
            )
            self.percent += self.step

            return True
        return False
