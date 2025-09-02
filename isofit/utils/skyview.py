#! /usr/bin/env python3
#
#  Copyright 2019 California Institute of Technology
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


import logging
from os.path import join

import click
import numpy as np
import ray
from spectral.io import envi

from isofit.core.common import envi_header


def skyview(
    dem_prj_path,
    dem_prj_resolution,
    output_directory,
    obs_or_loc=None,
    method="horizon",
    n_angles=72,
    logging_level="INFO",
    log_file=None,
    n_cores=1,
    ray_address: str = None,
    ray_redis_password: str = None,
    ray_temp_dir=None,
    ray_ip_head=None,
):
    """\
    Applies sky view factor calculation for a given UTM projected DEM or DSM. Much of this code was borrowed from ARS Topo-Calc.
    The key thing here was to create a python-only, rasterio-free port of this that could be used within ISOFIT. We also included 
    improvements that are current in Jeff Dozier's horizon method in Matlab (https://github.com/DozierJeff/Topographic-Horizons).

    Also, following suggestions from Dozier (2021), multiprocessing is leveraged here w.r.t. to n_angles rotating the image. As default,
    sky view is computed with n angles = 72 which in most cases is of sufficient accuracy to resolve but more angles may be used.

    Optionally to this horizon based method, one can pass method="slope" to compute a faster estimate that may be sufficent for regions with lower relief.
    The slope based estimate is simply, svf = cos^2(slope/2).

    \b
    Parameters
    ----------
    dem_prj_path : str
        Path to the projected DEM or DSM file (ENVI format).
    dem_prj_resolution : float
        Spatial resolution of the projected DEM in coordinate units (GSD in meters).
    output_directory : str
        Directory path for temporary files and outputs during processing; similar to apply_oe.
    obs_or_loc : str
        Options here are 'obs', 'loc', or None. Default is None. If 'obs' is selected, it will pick the slope data from index 6 in OBS file.
        If 'loc' is selected it well select the elevation data from index 2. 
    method : str, optional
        Options are either "horizon" or "slope". Passing "horizon" runs the full computation and is recommended for very steep terrain.
        Passing "slope"" runs the simplifed calculation of svf=cos^2(slope/2) and can be useful for more mild slopes. 
    n_angles : int, optional
        Number of angles used in horizon calculations (default is 72).
    logging_level : str, optional
        Logging verbosity level (default is "INFO"); similar to apply_oe.
    log_file : str or None, optional
        File path to write logs; similar to apply_oe.
    n_cores : int, optional
        Number of CPU cores to use for parallel processing (default is 1). Only used for method="horizon".
    """

    # Construct svf output path.
    svf_hdr_path = join(output_directory, "sky_view_factor.hdr")

    # Set up logging
    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=logging_level, filename=log_file
    )

    # Load DEM data (assuming hdr).
    dem = envi.open(envi_header(dem_prj_path))
    dem_data = dem.open_memmap(writeable=False).copy()

    # assign if None, LOC, or OBS based data.
    obs_or_loc = obs_or_loc.lower()
    if obs_or_loc == "obs":
        slope = dem_data[:, :, 6]
        slope[slope > 90.1] = np.nan
        slope[slope < -0.01] = np.nan
    elif obs_or_loc == "loc":
        dem_data = dem_data[:, :, 2]
        dem_data[dem_data > 8900] = np.nan
        dem_data[dem_data < -1360] = np.nan
    elif obs_or_loc == None:
        dem_data[dem_data > 8900] = np.nan
        dem_data[dem_data < -1360] = np.nan
    else:
        err_str = "obs_or_loc must be 'loc', 'obs', or None."
        raise ValueError(err_str)
    if dem_data.ndim == 3 and dem_data.shape[2] == 1:
        dem_data = dem_data[:, :, 0]
    # set metadata
    dem_metadata = dem.metadata.copy()
    dem_metadata.update(
        {
            "description": "Sky View Factor",
            "bands": 1,
            "interleave": "bsq",
            "data type": 4,
            "byte order": 0,
            "band names": {"Sky View Factor"},
        }
    )

    # If only computing slope method
    if method == "slope":
        if obs_or_loc != "obs":
            slope, aspect = gradient_d8(
                dem_data, dx=dem_prj_resolution, dy=dem_prj_resolution, aspect_rad=True
            )
        svf = np.cos(slope / 2) ** 2
        # save.
        envi.save_image(
            svf_hdr_path,
            svf.astype(np.float32),
            dtype=np.float32,
            interleave="bsq",
            metadata=dem_metadata,
            force=True,
        )

    elif method == "horizon":
        # prep the data for correct format for computation
        angles, aspect, cos_slope, sin_slope, tan_slope = viewfdozier_prep(
            dem=dem_data,
            spacing=dem_prj_resolution,
            nangles=n_angles,
            sin_slope=None,
            aspect=None,
        )

        # Start up a ray instance for parallel work
        rayargs = {
            "ignore_reinit_error": True,
            "local_mode": n_cores == 1,
            "address": ray_address,
            "include_dashboard": False,
            "_temp_dir": ray_temp_dir,
            "_redis_password": ray_redis_password,
            "num_cpus": n_cores,
        }
        ray.init(**rayargs)

        # Share ray objects
        dem_ray = ray.put(dem_data)
        aspect_ray = ray.put(aspect)
        cos_ray = ray.put(cos_slope)
        sin_ray = ray.put(sin_slope)
        tan_ray = ray.put(tan_slope)

        # Run n-angles in parallel
        futures = [
            viewfdozier_i.remote(
                a, dem_ray, dem_prj_resolution, aspect_ray, cos_ray, sin_ray, tan_ray
            )
            for a in angles
        ]
        results = ray.get(futures)

        # and so now, we have a list object of 72, 2-d arrays
        # and  can complete integration for svf
        svf = sum(results) / len(angles)

        envi.save_image(
            svf_hdr_path,
            svf.astype(np.float32),
            dtype=np.float32,
            interleave="bsq",
            metadata=dem_metadata,
            force=True,
        )
    else:
        err_str = "method must be either 'horizon' or 'slope'."
        raise ValueError(err_str)


def viewfdozier_prep(dem, spacing, nangles=72, sin_slope=None, aspect=None):
    """
    Preps computations for ray.

    Args:
        dem: numpy array for the DEM
        spacing: grid spacing of the DEM
        nangles: number of angles to estimate the horizon, defaults
                to 72 angles
        sin_slope: optional, will calculate if not provided
                    sin(slope) with range from 0 to 1
        aspect: optional, will calculate if not provided
                Aspect as radians from south (aspect 0 is toward
                the south) with range from -pi to pi, with negative
                values to the west and positive values to the east.

    Returns:
        angles, aspect, cos_slope, sin_slope, tan_slope

    """

    if dem.ndim != 2:
        raise ValueError("viewf input of dem is not a 2D array")

    if nangles < 16:
        raise ValueError("viewf number of angles should be 16 or greater")

    if sin_slope is not None:
        if np.max(sin_slope) > 1:
            raise ValueError("slope must be sin(slope) with range from 0 to 1")

    # calculate the gradient if not provided
    # The slope is returned as radians so convert to sin(S)
    if sin_slope is None:
        slope, aspect = gradient_d8(dem, dx=spacing, dy=spacing, aspect_rad=True)
        sin_slope = np.sin(slope)
        cos_slope = np.cos(slope)
        tan_slope = np.tan(slope)

    # -180 is North
    angles = np.linspace(-180, 180, num=nangles, endpoint=False)

    return angles, aspect, cos_slope, sin_slope, tan_slope


@ray.remote(num_cpus=1)
def viewfdozier_i(angle, dem, spacing, aspect, cos_slope, sin_slope, tan_slope):
    """
    See above, but this is running each horizon angle in parallel (n=72 , or user input)

    This returns the i-th svf array, used in the integral

    """

    # horizon angles
    hcos = horizon(angle, dem, spacing)
    azimuth = np.radians(angle)
    h = np.arccos(hcos)

    # cosines of difference between horizon aspect and slope aspect
    cos_aspect = np.cos(aspect - azimuth)

    # check for slope being obscured
    # EQ 3 in Dozier et al. 2022
    #     H(t) = min(H(t), acos(sqrt(1-1./(1+tand(slopeDegrees)^2*cos(azmRadian(t)-aspectRadian).^2))));
    t = cos_aspect < 0
    h[t] = np.fmin(
        h[t], np.arccos(np.sqrt(1 - 1 / (1 + cos_aspect[t] ** 2 * tan_slope[t] ** 2)))
    )

    # integral in https://github.com/DozierJeff/Topographic-Horizons:
    # qIntegrand = (cosd(slopeDegrees)*sin(H).^2 + sind(slopeDegrees)*cos(aspectRadian-azmRadian).*(H-cos(H).*sin(H)))/2
    svf = cos_slope * np.sin(h) ** 2 + sin_slope * cos_aspect * (
        h - np.sin(h) * np.cos(h)
    )
    svf[svf < 0] = 0

    return svf


"""NOTE:
The rest of the codes below were heavily borrowed from USDA-ARS topo-calc package.

With one exception which is hor1(). This had to be modified to improve speed.
"""


def gradient_d8(dem, dx, dy, aspect_rad=False):
    """
    Calculate the slope and aspect for provided dem,
    using a 3x3 cell around the center

    Given a center cell e and it's neighbors:

    | a | b | c |
    | d | e | f |
    | g | h | i |

    The rate of change in the x direction is

    [dz/dx] = ((c + 2f + i) - (a + 2d + g) / (8 * dx)

    The rate of change in the y direction is

    [dz/dy] = ((g + 2h + i) - (a + 2b + c)) / (8 * dy)

    The slope is calculated

    slope_radians = arctan ( sqrt ([dz/dx]^2 + [dz/dy]^2) )

    Args:
        dem: array of elevation values
        dx: cell size along the x axis
        dy: cell size along the y axis
        aspect_rad: turn the aspect from degrees to IPW radians

    Returns:
        slope in radians
        aspect in degrees or IPW radians
    """

    # Pad the dem
    dem_pad = np.pad(dem, pad_width=1, mode="edge")

    # top
    dem_pad[0, :] = dem_pad[1, :] + (dem_pad[1, :] - dem_pad[2, :])

    # bottom
    dem_pad[-1, :] = dem_pad[-2, :] + (dem_pad[-2, :] - dem_pad[-3, :])

    # left
    dem_pad[:, 0] = dem_pad[:, 1] + (dem_pad[:, 1] - dem_pad[:, 2])

    # right
    dem_pad[:, -1] = dem_pad[:, -2] - (dem_pad[:, -3] - dem_pad[:, -2])

    # finite difference in the y direction
    dz_dy = (
        (dem_pad[2:, :-2] + 2 * dem_pad[2:, 1:-1] + dem_pad[2:, 2:])
        - (dem_pad[:-2, :-2] + 2 * dem_pad[:-2, 1:-1] + dem_pad[:-2, 2:])
    ) / (8 * dy)

    # finite difference in the x direction
    dz_dx = (
        (dem_pad[:-2, 2:] + 2 * dem_pad[1:-1, 2:] + dem_pad[2:, 2:])
        - (dem_pad[:-2, :-2] + 2 * dem_pad[1:-1, :-2] + dem_pad[2:, :-2])
    ) / (8 * dx)

    slope = calc_slope(dz_dx, dz_dy)
    a = aspect(dz_dx, dz_dy)

    if aspect_rad:
        a = aspect_to_ipw_radians(a)

    return slope, a


def calc_slope(dz_dx, dz_dy):
    """Calculate the slope given the finite differences

    Arguments:
        dz_dx: finite difference in the x direction
        dz_dy: finite difference in the y direction

    Returns:
        slope numpy array
    """

    return np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))


def aspect(dz_dx, dz_dy):
    """
    Calculate the aspect from the finite difference.
    Aspect is degrees clockwise from North (0/360 degrees)

    See below for a referance to how ArcGIS calculates slope
    http://help.arcgis.com/en/arcgisdesktop/10.0/help/index.html#/How_Aspect_works/00q900000023000000/

    Args:
        dz_dx: finite difference in the x direction
        dz_dy: finite difference in the y direction

    Returns
        aspect in degrees clockwise from North
    """

    # return in degrees
    a = 180 * np.arctan2(dz_dy, -dz_dx) / np.pi

    aout = 90 - a
    aout[a < 0] = 90 - a[a < 0]
    aout[a > 90] = 360 - a[a > 90] + 90

    # if dz_dy and dz_dx are zero, then handle the
    # special case. Follow the IPW convetion and set
    # the aspect to south or 180 degrees
    idx = (dz_dy == 0) & (dz_dx == 0)
    aout[idx] = 180

    return aout


def aspect_to_ipw_radians(a):
    """
    IPW defines aspect differently than most GIS programs
    so convert an aspect in degrees from due North (0/360)
    to the IPW definition.

    Aspect is radians from south (aspect 0 is toward
    the south) with range from -pi to pi, with negative
    values to the west and positive values to the east

    Args:
        a: aspect in degrees from due North

    Returns
        a: aspect in radians from due South
    """

    arad = np.pi - a * np.pi / 180

    return arad


def horizon(azimuth, dem, spacing):
    """Calculate horizon angles for one direction. Horizon angles
    are based on Dozier and Frew 1990 and are adapted from the
    IPW C code.

    The coordinate system for the azimuth is 0 degrees is South,
    with positive angles through East and negative values
    through West. Azimuth values must be on the -180 -> 0 -> 180
    range.

    Arguments:
        azimuth {float} -- find horizon's along this direction
        dem {np.array2d} -- numpy array of dem elevations
        spacing {float} -- grid spacing

    Returns:
        hcos {np.array} -- cosines of angles to the horizon
    """

    if dem.ndim != 2:
        raise ValueError("horizon input of dem is not a 2D array")

    if azimuth > 180 or azimuth < -180:
        raise ValueError("azimuth must be between -180 and 180 degrees")

    if azimuth == 90:
        # East
        hcos = hor2d(dem, spacing, fwd=True)

    elif azimuth == -90:
        # West
        hcos = hor2d(dem, spacing, fwd=False)

    elif azimuth == 0:
        # South
        hcos = hor2d(dem.transpose(), spacing, fwd=True)
        hcos = hcos.transpose()

    elif np.abs(azimuth) == 180:
        # South
        hcos = hor2d(dem.transpose(), spacing, fwd=False)
        hcos = hcos.transpose()

    elif azimuth >= -45 and azimuth <= 45:
        # South west through south east
        t, spacing = skew_transpose(dem, spacing, azimuth)
        h = hor2d(t, spacing, fwd=True)
        hcos = skew(h.transpose(), azimuth, fwd=False)

    elif azimuth <= -135 and azimuth > -180:
        # North west
        a = azimuth + 180
        t, spacing = skew_transpose(dem, spacing, a)
        h = hor2d(t, spacing, fwd=False)
        hcos = skew(h.transpose(), a, fwd=False)

    elif azimuth >= 135 and azimuth < 180:
        # North East
        a = azimuth - 180
        t, spacing = skew_transpose(dem, spacing, a)
        h = hor2d(t, spacing, fwd=False)
        hcos = skew(h.transpose(), a, fwd=False)

    elif azimuth > 45 and azimuth < 135:
        # South east through north east
        a = 90 - azimuth
        t, spacing = transpose_skew(dem, spacing, a)
        h = hor2d(t, spacing, fwd=True)
        hcos = skew(h.transpose(), a, fwd=False).transpose()

    elif azimuth < -45 and azimuth > -135:
        # South west through north west
        a = -90 - azimuth
        t, spacing = transpose_skew(dem, spacing, a)
        h = hor2d(t, spacing, fwd=False)
        hcos = skew(h.transpose(), a, fwd=False).transpose()

    else:
        ValueError("azimuth not valid")

    # sanity check
    assert hcos.shape == dem.shape

    return hcos


def hor1(z, fwd=True):
    n = len(z)
    h = np.empty(n, dtype=int)

    if fwd:
        stack = []
        for i in reversed(range(n)):
            zi = z[i]
            while stack:
                j = stack[-1]
                dist = j - i
                slope = (z[j] - zi) / dist
                prev_j = h[j]
                if prev_j == j:
                    prev_slope = -np.inf
                else:
                    prev_dist = prev_j - j
                    prev_slope = (z[prev_j] - z[j]) / prev_dist
                if slope <= prev_slope:
                    stack.pop()
                else:
                    break
            if stack:
                h[i] = stack[-1]
            else:
                h[i] = i
            stack.append(i)
    else:
        # backward direction
        stack = []
        for i in range(n):
            zi = z[i]
            while stack:
                j = stack[-1]
                dist = i - j
                slope = (z[j] - zi) / dist
                prev_j = h[j]
                if prev_j == j:
                    prev_slope = -np.inf
                else:
                    prev_dist = j - prev_j
                    prev_slope = (z[prev_j] - z[j]) / prev_dist
                if slope <= prev_slope:
                    stack.pop()
                else:
                    break
            if stack:
                h[i] = stack[-1]
            else:
                h[i] = i
            stack.append(i)

    return h


def horval(z, delta, h):
    j = h
    d = j - np.arange(len(z))
    diff = z[j] - z
    dist = np.abs(d) * delta
    hcos = np.where(d == 0, 0, diff / np.hypot(diff, dist))
    return hcos


def hor2d(z, delta, fwd=True):
    if z.ndim != 2:
        raise ValueError("Input must be 2D array")
    nrows, ncols = z.shape
    hcos = np.empty_like(z)
    for i in range(nrows):
        zbuf = z[i, :]
        hbuf = hor1(zbuf, fwd=fwd)
        obuf = horval(zbuf, delta, hbuf)
        hcos[i, :] = obuf
    return hcos


def adjust_spacing(spacing, skew_angle):
    """Adjust the grid spacing if a skew angle is present

    Arguments:
        spacing {float} -- grid spacing
        skew_angle {float} -- angle to adjust the spacing for [degrees]
    """

    if skew_angle > 45 or skew_angle < 0:
        raise ValueError("skew angle must be between 0 and 45 degrees")

    return spacing / np.cos(skew_angle * np.arctan(1.0) / 45)


def skew(arr, angle, fwd=True, fill_min=True):
    """
    Skew the origin of successive lines by a specified angle
    A skew with angle of 30 degrees causes the following transformation:

        +-----------+       +---------------+
        |           |       |000/          /|
        |   input   |       |00/  output  /0|
        |   image   |       |0/   image  /00|
        |           |       |/          /000|
        +-----------+       +---------------+

    Calling skew with fwd=False will return the output image
    back to the input image.

    Skew angle must be between -45 and 45 degrees

    Args:
        arr: array to skew
        angle: angle between -45 and 45 to skew by
        fwd: add skew to image if True, unskew image if False
        fill_min: While IPW skew says it fills with zeros, the output
            image is filled with the minimum value

    Returns:
        skewed array

    """

    if angle == 0:
        return arr

    if angle > 45 or angle < -45:
        raise ValueError("skew angle must be between -45 and 45 degrees")

    nlines, nsamps = arr.shape

    if angle >= 0.0:
        negflag = False
    else:
        negflag = True
        angle = -angle

    slope = np.tan(angle * np.pi / 180.0)
    max_skew = int((nlines - 1) * slope + 0.5)

    o_nsamps = nsamps
    if fwd:
        o_nsamps += max_skew
    else:
        o_nsamps -= max_skew

    b = np.zeros((nlines, o_nsamps))
    if fill_min:
        b += np.min(arr)

    for line in range(nlines):
        o = line if negflag else nlines - line - 1
        offset = int(o * slope + 0.5)

        if fwd:
            b[line, offset : offset + nsamps] = arr[line, :]
        else:
            b[line, :] = arr[line, offset : offset + o_nsamps]

    return b


def skew_transpose(dem, spacing, angle):
    """Skew and transpose the dem for the given angle.
    Also calculate the new spacing given the skew.

    Arguments:
        dem {array} -- numpy array of dem elevations
        spacing {float} -- grid spacing
        angle {float} -- skew angle

    Returns:
        t -- skew and transpose array
        spacing -- new spacing adjusted for angle
    """

    spacing = adjust_spacing(spacing, np.abs(angle))
    t = skew(dem, angle, fill_min=True).transpose()

    return t, spacing


def transpose_skew(dem, spacing, angle):
    """Transpose, skew then transpose a dem for the
    given angle. Also calculate the new spacing

    Arguments:
        dem {array} -- numpy array of dem elevations
        spacing {float} -- grid spacing
        angle {float} -- skew angle

    Returns:
        t -- skew and transpose array
        spacing -- new spacing adjusted for angle
    """

    t = skew(dem.transpose(), angle, fill_min=True).transpose()
    spacing = adjust_spacing(spacing, np.abs(angle))

    return t, spacing


# Input arguments
@click.command(name="skyview", help=skyview.__doc__, no_args_is_help=True)
@click.argument("dem_prj_path", type=str)
@click.argument("dem_prj_resolution", type=float)
@click.argument("output_directory", type=str)
@click.option("--n_angles", type=int, default=72)
@click.option("--method", type=str, default="horizon")
@click.option("--logging_level", default="INFO")
@click.option("--log_file", type=str, default=None)
@click.option("--n_cores", type=int, default=1)
@click.option("--ray_address", type=str, default=None)
@click.option("--ray_redis_password", type=str, default=None)
@click.option("--ray_temp_dir", type=str, default="/tmp/ray")
@click.option("--ray_ip_head", type=str, default=None)
@click.option(
    "--debug_args",
    help="Print the arguments and exit",
    is_flag=True,
)
def cli(debug_args, **kwargs):
    if debug_args:
        print("Arguments to be passed:")
        for key, value in kwargs.items():
            print(f"  {key} = {value!r}")
    else:
        skyview(**kwargs)

    print("Done")


if __name__ == "__main__":
    raise NotImplementedError(
        "skyview.py can no longer be called this way.  Run as:\n isofit skyview [ARGS]"
    )
