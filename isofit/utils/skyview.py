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

import logging
from os.path import join, isfile, isdir
from os import remove
import time
import warnings

import click
import numpy as np
import ray
import netCDF4 as nc
from spectral.io import envi

from isofit.core.common import envi_header, eps


def skyview(
    input: str,
    output_directory: str,
    resolution: float = np.nan,
    obs_or_loc: str = None,
    method: str = "slope",
    n_angles: int = 72,
    logging_level: str = "INFO",
    log_file: str = None,
    n_cores: int = 1,
    keep_horizon_files: bool = False,
    ray_address: str = None,
    ray_redis_password: str = None,
    ray_temp_dir: str = None,
    ray_ip_head: str = None,
):
    """\
    Applies sky view factor calculation for a given projected DEM or DSM. Much of this code was borrowed from ARS Topo-Calc.
    The key thing here was to create a python-only, rasterio-free port of this that could be used within ISOFIT. We also included 
    improvements that are current in Jeff Dozier's horizon method in Matlab (https://github.com/DozierJeff/Topographic-Horizons).
    Following suggestions from Dozier (2021), multiprocessing is leveraged here w.r.t. to n_angles rotating the image. As default,
    sky view is computed with n angles = 72 which in most cases is of sufficient accuracy to resolve but more angles may be used.
    
    Optionally to this horizon based method, one can pass method="slope" to compute a faster estimate that may be sufficent for regions with lower relief.
    The slope based estimate is simply, svf = cos^2(slope/2). 
    
    Yet another option is to pass an ISOFIT "OBS" or "LOC" file as input and using the obs_or_loc arg. 
    OBS files have slope data and can be used for method='slope' only. LOC files have elevation data and can be used for method='slope'. 
    One can also use the full horizon method on the LOC file although this is not recommended because the edges miss information 
    (a warning will be triggered in this case).

    \b
    Parameters
    ----------
    input : str
        Path to the projected ENVI File. If `obs_or_loc` is None or "loc" input is elevation. If is "obs", then it's a slope product.
    output_directory : str
        Directory path for temporary files and outputs during processing; similar to apply_oe.
    resolution : float, optional
        Spatial resolution of the projected DEM in coordinate units (GSD in meters). Required for elevation input data.
    obs_or_loc : str, optional
        Options here are 'obs', 'loc', or None. Default is None. If 'obs' is selected, it will pick the slope data from index 6 in OBS file.
        If 'loc' is selected it well select the elevation data from index 2. None will assume a single band elevation data is passed.
    method : str, optional
        Options are either "horizon" or "slope". Passing "horizon" runs the full computation and is recommended for very steep terrain.
        Passing "slope"" runs the simplifed calculation of svf=cos^2(slope/2) and can be useful for more mild slopes. 
    n_angles : int, optional
        Number of angles used in horizon calculations (default is 72). Other options could be 32, 64, etc. (see Dozier & Frew).
        As a reference, n=72 computes every 5deg, n=64 every 5.6deg, n=32 every 11.25deg, etc.  
    keep_horizon_files : bool, optional
        Horizon angles are created in output_dir as netcdf files. False deletes files, and True keeps them. These angles are based from zenith.        
    logging_level : str, optional
        Logging verbosity level (default is "INFO"); similar to apply_oe.
    log_file : str or None, optional
        File path to write logs; similar to apply_oe.
    n_cores : int, optional
        Number of CPU cores to use for parallel processing (default is 1). Only used for method="horizon". 
        Note: n_cores should ideally not be larger than n_angles.
    """
    # set up logging for skyview.
    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=logging_level, filename=log_file
    )
    logging.info("Starting skyview utility...")
    start_time = time.time()

    # safeguards on possible incorrect inputs.
    if not isinstance(input, str):
        err_str = "input must be a string."
        raise TypeError(err_str)

    if not isinstance(n_angles, int) or n_angles < 16:
        err_str = "n_angles must be a positive integer greater than 16."
        raise ValueError(err_str)

    if not isinstance(n_cores, int) or n_cores <= 0:
        err_str = "n_cores must be a positive integer."
        raise ValueError(err_str)

    # Construct svf output path.
    if not isdir(output_directory):
        err_str = f"The output directory, {output_directory}, does not exist or was not found."
        raise ValueError(err_str)
    else:
        svf_hdr_path = join(output_directory, "sky_view_factor.hdr")

    # ensuring n_cores does not exceed n_angles
    # See, "Revisiting Topographic Horizons in the Era of Big Data and Parallel Computing"
    # In our case, we split it on rotation and forward/backward. and so max workers = n_angles.
    n_cores_max = n_angles
    if n_cores > n_cores_max:
        logging.info(
            f"n_cores={n_cores}, but max can be {n_cores_max} for n_angles={n_angles}. Setting n_cores={n_cores_max}."
        )
        n_cores = n_cores_max

    # Load input data and clean.
    dem_data, svf_metadata, slope = load_input(
        input=input, resolution=resolution, obs_or_loc=obs_or_loc, method=method
    )

    # If only computing slope method, we do not need to set up Ray.
    if method == "slope":
        if obs_or_loc != "obs":
            slope, aspect = gradient_d8(
                dem_data, dx=resolution, dy=resolution, aspect_rad=True
            )

        # approx. skyview using slope only.
        svf = np.cos(slope / 2) ** 2

        # save file
        save_envi(
            input_data=svf,
            input_hdr_path=svf_hdr_path,
            input_metadata=svf_metadata,
            band_name="Sky View Factor",
            dtype=np.float32,
        )

    # Else if, run the full horizon method.
    elif method == "horizon":

        # raise error if horizon method passed but slope data is input.
        if obs_or_loc == "obs":
            err_str = "A slope product was passed to horizon method, but this method requires the elevation product."
            raise ValueError(err_str)

        # raise warning for running loc file on horizon.
        if obs_or_loc == "loc":
            warn_str = "Running the horizon method with LOC file. Resulting skyview may be incorrect on edges."
            warnings.warn(warn_str, UserWarning)
            logging.info(warn_str)

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

        # prep data for horizon method
        slope, aspect = gradient_d8(
            dem_data, dx=resolution, dy=resolution, aspect_rad=True
        )
        sin_slope = np.sin(slope).astype(np.float32)
        cos_slope = np.cos(slope).astype(np.float32)
        tan_slope = np.tan(slope).astype(np.float32)

        # -180 is North
        angles = np.linspace(-180, 180, num=n_angles, endpoint=False)

        # Share needed ray objects
        dem_ray = ray.put(dem_data)
        aspect_ray = ray.put(aspect)
        tan_slope_ray = ray.put(tan_slope)

        # Run n-angles in parallel
        futures = [
            horizon_worker.remote(
                angle=a,
                dem=dem_ray,
                spacing=resolution,
                aspect=aspect_ray,
                tan_slope=tan_slope_ray,
                output_directory=output_directory,
                logging_level=logging_level,
                log_file=log_file,
            )
            for a in angles
        ]
        ray.get(futures)

        # set up integral for skyview
        qIntegrand = np.zeros_like(dem_data, dtype=np.float32)
        for a in angles:
            file_path = join(output_directory, f"horizon_angle_{np.round(a,5)}.nc")
            h = load_horizon_nc(file_path=file_path)
            azimuth = np.radians(a)
            cos_aspect = np.cos(aspect - azimuth)

            # integral in https://github.com/DozierJeff/Topographic-Horizons:
            # qIntegrand = (  (cosd(slopeDegrees)*sin(H).^2 +
            # sind(slopeDegrees)*cos(aspectRadian-azmRadian).*(H-cos(H).*sin(H)))/2 )
            qIntegrand_i = cos_slope * np.sin(h) ** 2 + sin_slope * cos_aspect * (
                h - np.sin(h) * np.cos(h)
            )
            qIntegrand_i[qIntegrand_i < 0] = 0
            qIntegrand_i[np.isnan(qIntegrand_i)] = 0
            qIntegrand += qIntegrand_i

        # complete integration for svf
        svf = qIntegrand / len(angles)
        svf[(svf <= 0) | (svf > 1)] = -9999  # no such situation svf=0.

        # save file
        save_envi(
            input_data=svf,
            input_hdr_path=svf_hdr_path,
            input_metadata=svf_metadata,
            band_name="Sky View Factor",
            dtype=np.float32,
        )

        # check to remove horizon files.
        if keep_horizon_files is False:
            logging.info("Removing temporary horizon files...")
            for a in angles:
                remove(join(output_directory, f"horizon_angle_{a}.nc"))

    else:
        err_str = "method must be either 'horizon' or 'slope'."
        raise ValueError(err_str)

    logging.info(
        f"Skyview calculation completed in {time.time() - start_time} seconds using {n_cores} cores."
    )


@ray.remote(num_cpus=1)
def horizon_worker(
    angle,
    dem,
    spacing,
    aspect,
    tan_slope,
    output_directory,
    logging_level,
    log_file,
):
    """
    Each worker gets an angle and is sent to this function to compute horizons, and save to a compressed/scaled netcdf file.
    """
    # set up logging for each worker.
    logging.basicConfig(
        format="%(levelname)s:%(asctime)s ||| %(message)s",
        level=logging_level,
        filename=log_file,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    # horizon angles (h)
    hcos = horizon(angle, dem, spacing)
    azimuth = np.radians(angle)
    h = np.arccos(hcos)
    h = update_h_for_local_topo(h, aspect, azimuth, tan_slope)

    # flush current run to disk.
    save_horizon_nc(h=h, angle=angle, output_directory=output_directory)

    del h

    return


def save_envi(input_data, input_hdr_path, input_metadata, band_name, dtype):
    """utility function to house metadata and information for saving output"""

    # update metadata
    input_metadata.update(
        {
            "description": f"{band_name}",
            "bands": 1,
            "data ignore value": -9999,
            "interleave": "bsq",
            "data type": 4,
            "byte order": 0,
            "band names": {f"{band_name}"},
        }
    )

    # save image
    envi.save_image(
        input_hdr_path,
        input_data.astype(dtype),
        dtype=dtype,
        interleave="bsq",
        metadata=input_metadata,
        force=True,
    )

    return


def save_horizon_nc(h, angle, output_directory):
    """utility function to house metadata and information for saving horizon angles"""

    # Scale h to uint16 with nodata=65535
    h_nodata = 65535
    h_sf = np.pi / 65534

    # loss of data, but still accurate to ~0.003 degrees.
    h_scaled = (h / h_sf).astype(np.uint16)
    h_scaled[(np.isnan(h)) | (h == -9999)] = h_nodata  # nodata

    # write h to disk (scaled data reduced ~3-5x file size.)
    angle_to_write = np.round(angle, 5)
    logging.info(f"Flushing horizon angle: {angle_to_write} to disk.")
    filename = join(output_directory, f"horizon_angle_{angle_to_write}.nc")
    with nc.Dataset(filename, "w", format="NETCDF4") as ds:
        ds.createDimension("row", h.shape[0])
        ds.createDimension("col", h.shape[1])
        var = ds.createVariable(
            "horizon",
            "u2",
            ("row", "col"),
            fill_value=h_nodata,
            zlib=True,
            complevel=4,
        )
        var[:] = h_scaled
        var.units = "radians"
        var.long_name = "Horizon angle (radians)"
        var.scale_factor = h_sf
        var.add_offset = 0.0
        var.nodata = h_nodata
        var.note = (
            "Values scaled as uint16 from 0 to pi radians;"
            "convert back with np.pi / 65534 ;"
            "65535 is nodata."
            "NOTE: angle is from zenith."
        )

    return


def load_horizon_nc(file_path):
    """load horizon netcdf in for skyview calc using scale factors defined in save."""
    with nc.Dataset(file_path) as ds:
        ds.set_auto_scale(False)
        h_nodata = ds.variables["horizon"].nodata
        h_sf = ds.variables["horizon"].scale_factor
        h_scaled = ds.variables["horizon"][:].astype(np.float32)
        h_scaled[h_scaled == h_nodata] = np.nan
        h = h_scaled * h_sf
    return h


def create_shadow_mask(
    input: str,
    output_directory: str,
    resolution: float = np.nan,
    sza: float = np.nan,
    saa: float = np.nan,
    logging_level: str = "INFO",
    log_file: str = None,
    n_cores: int = 1,
    ray_address: str = None,
    ray_redis_password: str = None,
    ray_temp_dir: str = None,
    ray_ip_head: str = None,
):
    """
    Computes horizon at a specific geometry to create a binary shadow mask because nearby terrain
    can cast shadows onto pixels as a function of the solar geometry that aren't always captured in cos-i.
    In this case, the input angle is the solar azimuth, and the solar zenith is compared to h at each pixel.

    As of right now, this assumes solar azimuth is constant value (0-360deg). However, solar zenith (0-90deg) can vary by pixel.

    """

    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=logging_level, filename=log_file
    )

    logging.info("Starting shadow mask utility...")
    start_time = time.time()

    # Ensure SAA is within 0-360 degrees.
    if saa > 360 or saa < 0:
        err_str = f"Must us 0-360deg convention for solar azimuth, with 0 at North."
        raise ValueError(err_str)

    # Load DEM data (assuming hdr).
    dem_data, shadow_metadata, slope = load_input(
        input=input, resolution=resolution, obs_or_loc=None, method="horizon"
    )

    # Construct svf output path.
    if not isdir(output_directory):
        err_str = f"The output directory, {output_directory}, does not exist or was not found."
        raise ValueError(err_str)
    else:
        shadow_hdr_path = join(output_directory, "shadow_mask.hdr")

    # create empty array for shadow data
    shadow = np.zeros_like(dem_data)

    # prep data for horizon method
    slope, aspect = gradient_d8(dem_data, dx=resolution, dy=resolution, aspect_rad=True)
    tan_slope = np.tan(slope).astype(np.float32)

    # convert saa into convention h method expects.
    saa = ((180 - saa) + 180) % 360 - 180

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

    # horizon angles (h)
    hcos = horizon(saa, dem_data, resolution, par=(n_cores > 1), n_cores=n_cores)
    azimuth = np.radians(saa)
    h = np.arccos(hcos)
    h = update_h_for_local_topo(h, aspect, azimuth, tan_slope)

    # Check each pixel to identify where sza>H. 1=shadow cast. 0=false.
    shadow[np.radians(sza) >= h] = 1

    # Save data
    save_envi(
        input_data=shadow,
        input_hdr_path=shadow_hdr_path,
        input_metadata=shadow_metadata,
        band_name="Shadow Mask",
        dtype=np.uint8,
    )

    logging.info(
        f"Shadow mask completed in {time.time() - start_time} seconds using {n_cores} cores."
    )

    return


def update_h_for_local_topo(h, aspect, azimuth, tan_slope):
    # update h for within-pixel topography.
    # EQ 3 in Dozier et al. 2022
    #     H(t) = min(H(t), acos(sqrt(1-1./(1+tand(slopeDegrees)^2*cos(azmRadian(t)-aspectRadian).^2))));
    cos_aspect = np.cos(azimuth - aspect)
    t = cos_aspect < 0
    h[t] = np.minimum(
        h[t], np.arccos(np.sqrt(1 - 1 / (1 + tan_slope[t] ** 2 * cos_aspect[t] ** 2)))
    )
    return h


def load_input(input, resolution, obs_or_loc=None, method="slope"):

    if not isfile(input):
        raise FileNotFoundError(f"The DEM file was not found: {input}.")

    dem = envi.open(envi_header(input))
    dem_data = dem.open_memmap(writeable=False).copy().astype(np.float32)

    if method:
        method = method.lower()
    if obs_or_loc:
        obs_or_loc = obs_or_loc.lower()

    if obs_or_loc != "obs":
        if (
            not isinstance(resolution, (float, int))
            or resolution <= 0
            or np.isnan(resolution)
        ):
            raise ValueError(
                "resolution must be positive value when using elevation data."
            )

    max_elev = 10000.0  # slightly higher than Mt.Everest
    min_elev = -2000.0  # slightly lower than Dead Sea

    slope = None
    if obs_or_loc == "obs":
        # Assuming OBS Slope data in degrees
        slope = dem_data[:, :, 6]
        slope[slope > 90.1] = np.nan
        slope[slope < -0.01] = np.nan
        slope = np.radians(slope)
    elif obs_or_loc == "loc":
        dem_data = dem_data[:, :, 2].astype(np.float32)
        dem_data[dem_data > max_elev] = np.nan
        dem_data[dem_data < min_elev] = np.nan
    elif obs_or_loc is None:
        dem_data[dem_data > max_elev] = np.nan
        dem_data[dem_data < min_elev] = np.nan
    else:
        raise ValueError("obs_or_loc must be 'loc', 'obs', or None.")

    # Squeeze 3D data with single band to 2D
    if dem_data.ndim == 3 and dem_data.shape[2] == 1:
        dem_data = dem_data[:, :, 0].astype(np.float32)

    # get metadata
    metadata = dem.metadata.copy()

    return dem_data, metadata, slope


"""NOTE:
The rest of the codes below were heavily borrowed from USDA-ARS topo-calc package.

With one exception which is hor1d(). This had to be modified to improve speed.
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

    return slope.astype(np.float32), a.astype(np.float32)


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


def horizon(azimuth, dem, spacing, par=False, n_cores=1):
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
        hcos = hor2d(dem, spacing, fwd=True, par=par, n_cores=n_cores)

    elif azimuth == -90:
        # West
        hcos = hor2d(dem, spacing, fwd=False, par=par, n_cores=n_cores)

    elif azimuth == 0:
        # South
        hcos = hor2d(dem.transpose(), spacing, fwd=True, par=par, n_cores=n_cores)
        hcos = hcos.transpose()

    elif np.abs(azimuth) == 180:
        # South
        hcos = hor2d(dem.transpose(), spacing, fwd=False, par=par, n_cores=n_cores)
        hcos = hcos.transpose()

    elif azimuth >= -45 and azimuth <= 45:
        # South west through south east
        t, spacing = skew_transpose(dem, spacing, azimuth)
        h = hor2d(t, spacing, fwd=True, par=par, n_cores=n_cores)
        hcos = skew(h.transpose(), azimuth, fwd=False)

    elif azimuth <= -135 and azimuth > -180:
        # North west
        a = azimuth + 180
        t, spacing = skew_transpose(dem, spacing, a)
        h = hor2d(t, spacing, fwd=False, par=par, n_cores=n_cores)
        hcos = skew(h.transpose(), a, fwd=False)

    elif azimuth >= 135 and azimuth < 180:
        # North East
        a = azimuth - 180
        t, spacing = skew_transpose(dem, spacing, a)
        h = hor2d(t, spacing, fwd=False, par=par, n_cores=n_cores)
        hcos = skew(h.transpose(), a, fwd=False)

    elif azimuth > 45 and azimuth < 135:
        # South east through north east
        a = 90 - azimuth
        t, spacing = transpose_skew(dem, spacing, a)
        h = hor2d(t, spacing, fwd=True, par=par, n_cores=n_cores)
        hcos = skew(h.transpose(), a, fwd=False).transpose()

    elif azimuth < -45 and azimuth > -135:
        # South west through north west
        a = -90 - azimuth
        t, spacing = transpose_skew(dem, spacing, a)
        h = hor2d(t, spacing, fwd=False, par=par, n_cores=n_cores)
        hcos = skew(h.transpose(), a, fwd=False).transpose()

    else:
        ValueError("azimuth not valid")

    # sanity check
    assert hcos.shape == dem.shape

    return hcos


def hor1d(z, fwd=True):
    n = len(z)
    h = np.empty(n, dtype=np.int32)

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
                if slope <= prev_slope + eps:
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
                if slope <= prev_slope + eps:
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


def hor2d(z, delta, fwd=True, par=False, n_cores=1):
    if z.ndim != 2:
        raise ValueError("Input must be 2D array")

    nrows, ncols = z.shape

    if not par:
        hcos = np.empty_like(z, dtype=np.float32)
        for i in range(nrows):
            zbuf = z[i, :]
            hbuf = hor1d(zbuf, fwd=fwd)
            obuf = horval(zbuf, delta, hbuf)
            hcos[i, :] = obuf
        return hcos
    else:
        z_ray = ray.put(z)
        chunk_size = int(np.ceil(nrows / n_cores))
        futures = []
        for c in range(n_cores):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, nrows)
            futures.append(_hor2d_chunk.remote(z_ray, delta, fwd, start, end))
        hcos = np.vstack(ray.get(futures)).astype(np.float32)
        return hcos


@ray.remote(num_cpus=1)
def _hor2d_chunk(zbuf, delta, fwd, row_start, row_end):
    results = []
    for i in range(row_start, row_end):
        line = zbuf[i, :]
        hbuf = hor1d(line, fwd=fwd)
        results.append(horval(line, delta, hbuf))
    return np.vstack(results)


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
@click.argument("input", type=str)
@click.argument("output_directory", type=str)
@click.option("--resolution", type=float, default=np.nan)
@click.option("--n_angles", type=int, default=72)
@click.option("--keep_horizon_files", type=bool, default=False)
@click.option("--obs_or_loc", type=str, default=None)
@click.option("--method", type=str, default="slope")
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
