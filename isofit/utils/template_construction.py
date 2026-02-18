#! /usr/bin/env python3
#
# Authors: Philip G. Brodrick and Niklas Bohn
#
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime
from os.path import abspath, dirname, exists, join, split
from shutil import copyfile
from sys import platform
from typing import List

import netCDF4 as nc
import numpy as np
from scipy.io import loadmat
from spectral.io import envi

from isofit import __version__
from isofit.core import isofit, units
from isofit.core.common import (
    envi_header,
    expand_path,
    json_load_ascii,
    resample_spectrum,
)
from isofit.core.multistate import SurfaceMapping
from isofit.data import env
from isofit.radiative_transfer.engines.modtran import ModtranRT
from isofit.utils.surface_model import surface_model


class Pathnames:
    """Class to determine and hold the large number of relative and absolute paths that are needed for isofit and
    MODTRAN configuration files.
    """

    def __init__(
        self,
        input_radiance,
        input_loc,
        input_obs,
        surface_class_file,
        surface_path,
        working_directory,
        ray_temp_dir,
        sensor="NA-*",
        copy_input_files=False,
        modtran_path=None,
        rdn_factors_path=None,
        model_discrepancy_path=None,
        aerosol_climatology_path=None,
        channelized_uncertainty_path=None,
        instrument_noise_path=None,
        interpolate_inplace=False,
        skyview_factor=None,
        subs: bool = False,
        classify_multisurface: bool = False,
        eof_path=None,
    ):
        # Determine FID based on sensor name
        if sensor == "ang":
            self.fid = split(input_radiance)[-1][:18]
        elif sensor == "av3":
            self.fid = split(input_radiance)[-1][:18]
        elif sensor == "av5":
            self.fid = split(input_radiance)[-1][:18]
        elif sensor == "avcl":
            self.fid = split(input_radiance)[-1][:16]
        elif sensor == "emit":
            self.fid = split(input_radiance)[-1][:19]
        elif sensor == "enmap":
            self.fid = split(input_radiance)[-1].split("_")[5]
        elif sensor == "hyp":
            self.fid = split(input_radiance)[-1][:22]
        elif sensor == "neon":
            self.fid = split(input_radiance)[-1][:21]
        elif sensor == "prism":
            self.fid = split(input_radiance)[-1][:18]
        elif sensor == "prisma":
            self.fid = input_radiance.split("/")[-1].split("_")[1]
        elif sensor == "gao":
            self.fid = split(input_radiance)[-1][:23]
        elif sensor == "oci":
            self.fid = split(input_radiance)[-1][:24]
        elif sensor == "tanager":
            self.fid = split(input_radiance)[-1][:23]
        elif sensor[:3] == "NA-":
            self.fid = os.path.splitext(os.path.basename(input_radiance))[0]

        logging.info("Flightline ID: %s" % self.fid)

        # Names from inputs
        self.aerosol_climatology = aerosol_climatology_path
        self.input_radiance_file = input_radiance
        self.input_loc_file = input_loc
        self.input_obs_file = input_obs
        self.working_directory = abspath(working_directory)

        self.full_lut_directory = abspath(join(self.working_directory, "lut_full/"))

        self.surface_path = surface_path

        # set up some sub-directories
        self.lut_h2o_directory = abspath(join(self.working_directory, "lut_h2o/"))
        self.config_directory = abspath(join(self.working_directory, "config/"))
        self.data_directory = abspath(join(self.working_directory, "data/"))
        self.input_data_directory = abspath(join(self.working_directory, "input/"))
        self.output_directory = abspath(join(self.working_directory, "output/"))

        # define all output names
        rdn_fname = self.fid + "_rdn"
        self.rfl_working_path = abspath(
            join(self.output_directory, rdn_fname.replace("_rdn", "_rfl"))
        )
        self.uncert_working_path = abspath(
            join(self.output_directory, rdn_fname.replace("_rdn", "_uncert"))
        )
        self.lbl_working_path = abspath(
            join(self.output_directory, rdn_fname.replace("_rdn", "_lbl"))
        )
        self.state_working_path = abspath(
            join(self.output_directory, rdn_fname.replace("_rdn", "_state"))
        )
        self.surface_template_path = abspath(join(self.data_directory, "surface.mat"))
        self.surface_working_paths = {}

        if copy_input_files is True:
            self.radiance_working_path = abspath(
                join(self.input_data_directory, rdn_fname)
            )
            self.obs_working_path = abspath(
                join(self.input_data_directory, self.fid + "_obs")
            )
            self.loc_working_path = abspath(
                join(self.input_data_directory, self.fid + "_loc")
            )

            if classify_multisurface and not surface_class_file:
                self.surface_class_working_path = abspath(
                    join(self.input_data_directory, self.fid + "_surface_class")
                )
            else:
                self.surface_class_working_path = (
                    abspath(surface_class_file) if surface_class_file else None
                )

            self.surface_class_subs_path = abspath(
                join(self.input_data_directory, self.fid + "_subs_surface_class")
            )

        else:
            self.radiance_working_path = abspath(self.input_radiance_file)
            self.obs_working_path = abspath(self.input_obs_file)
            self.loc_working_path = abspath(self.input_loc_file)

            if classify_multisurface and not surface_class_file:
                self.surface_class_working_path = abspath(
                    join(self.input_data_directory, self.fid + "_surface_class")
                )
            else:
                self.surface_class_working_path = (
                    abspath(surface_class_file) if surface_class_file else None
                )

            self.surface_class_subs_path = abspath(
                join(self.input_data_directory, self.fid + "_subs_surface_class")
            )

        if interpolate_inplace:
            self.radiance_interp_path = self.radiance_working_path
        else:
            self.radiance_interp_path = abspath(
                join(self.input_data_directory, rdn_fname + "_interp")
            )

        if channelized_uncertainty_path:
            self.input_channelized_uncertainty_path = channelized_uncertainty_path
        else:
            self.input_channelized_uncertainty_path = os.getenv(
                "ISOFIT_CHANNELIZED_UNCERTAINTY"
            )

        self.channelized_uncertainty_working_path = abspath(
            join(self.data_directory, "channelized_uncertainty.txt")
        )

        if model_discrepancy_path:
            self.input_model_discrepancy_path = model_discrepancy_path
        else:
            self.input_model_discrepancy_path = None

        self.model_discrepancy_working_path = abspath(
            join(self.data_directory, "model_discrepancy.mat")
        )

        if eof_path:
            self.eof_path = eof_path
        else:
            self.eof_path = None

        self.eof_working_path = abspath(join(self.data_directory, "eof.txt"))

        if skyview_factor:
            self.svf_working_path = abspath(skyview_factor)
        else:
            self.svf_working_path = None

        self.svf_subs_path = abspath(
            join(self.input_data_directory, self.fid + "_subs_svf")
        )

        self.rdn_subs_path = abspath(
            join(self.input_data_directory, self.fid + "_subs_rdn")
        )
        self.obs_subs_path = abspath(
            join(self.input_data_directory, self.fid + "_subs_obs")
        )
        self.loc_subs_path = abspath(
            join(self.input_data_directory, self.fid + "_subs_loc")
        )

        self.rfl_subs_path = abspath(
            # join(self.output_directory, self.fid + f"{subs_str}" + "_rfl")
            join(self.output_directory, self.fid + "_subs_rfl")
        )
        self.atm_coeff_path = abspath(
            join(self.output_directory, self.fid + "_subs_atm")
        )
        self.state_subs_path = abspath(
            join(self.output_directory, self.fid + "_subs_state")
        )
        self.uncert_subs_path = abspath(
            join(self.output_directory, self.fid + "_subs_uncert")
        )
        self.h2o_subs_path = abspath(
            join(self.output_directory, self.fid + "_subs_h2o")
        )

        self.wavelength_path = abspath(join(self.data_directory, "wavelengths.txt"))

        self.modtran_template_path = abspath(
            join(self.config_directory, self.fid + "_modtran_tpl.json")
        )
        self.h2o_template_path = abspath(
            join(self.config_directory, self.fid + "_h2o_tpl.json")
        )

        self.isofit_full_config_path = abspath(
            join(self.config_directory, self.fid + "_isofit.json")
        )
        self.h2o_config_path = abspath(
            join(self.config_directory, self.fid + "_h2o.json")
        )

        if modtran_path:
            self.modtran_path = modtran_path
        else:
            self.modtran_path = os.getenv("MODTRAN_DIR", env.modtran)

        self.sixs_path = os.getenv("SIXS_DIR", env.sixs)

        self.noise_path = None
        if sensor == "avcl":
            self.noise_path = str(env.path("data", "avirisc_noise.txt"))
        elif sensor == "oci":
            self.noise_path = str(env.path("data", "oci", "oci_noise.txt"))
        elif sensor == "emit":
            self.noise_path = str(env.path("data", "emit_noise.txt"))
            if self.input_channelized_uncertainty_path is None:
                self.input_channelized_uncertainty_path = str(
                    env.path("data", "emit_osf_uncertainty.txt")
                )
            if self.input_model_discrepancy_path is None:
                self.input_model_discrepancy_path = str(
                    env.path("data", "emit_model_discrepancy.mat")
                )
            if self.eof_path is None:
                self.eof_path = str(env.path("data", "emit_eofs.txt"))
        elif sensor == "tanager":
            self.noise_path = str(env.path("data", "tanager1_noise_20241016.txt"))

        # Override noise path if provided
        if instrument_noise_path is not None:
            if self.noise_path is not None:
                logging.info(
                    f"Overriding default instrument noise path {self.noise_path} with user-provided path {instrument_noise_path}"
                )
            self.noise_path = instrument_noise_path

        if self.noise_path is None:
            logging.info("no noise path found, proceeding without")

        self.earth_sun_distance_path = str(env.path("data", "earth_sun_distance.txt"))

        irr_path = [
            "examples",
            "20151026_SantaMonica",
            "data",
            "prism_optimized_irr.dat",
        ]
        if sensor == "oci":
            irr_path = ["data", "oci", "tsis_f0_0p1.txt"]

        self.irradiance_file = str(env.path(*irr_path))

        self.aerosol_tpl_path = str(env.path("data", "aerosol_template.json"))
        self.rdn_factors_path = None
        if rdn_factors_path is not None:
            self.rdn_factors_path = abspath(rdn_factors_path)

        self.ray_temp_dir = ray_temp_dir

    def make_directories(self):
        """Build required subdirectories inside working_directory"""
        for dpath in [
            self.working_directory,
            self.lut_h2o_directory,
            self.full_lut_directory,
            self.config_directory,
            self.data_directory,
            self.input_data_directory,
            self.output_directory,
        ]:
            if not exists(dpath):
                os.mkdir(dpath)

    def stage_files(self):
        """Stage data files by copying into working directory"""
        files_to_stage = [
            (self.input_radiance_file, self.radiance_working_path, True),
            (self.input_obs_file, self.obs_working_path, True),
            (self.input_loc_file, self.loc_working_path, True),
            (
                self.input_channelized_uncertainty_path,
                self.channelized_uncertainty_working_path,
                False,
            ),
            (
                self.input_model_discrepancy_path,
                self.model_discrepancy_working_path,
                False,
            ),
            (self.eof_path, self.eof_working_path, False),
        ]

        for src, dst, hasheader in files_to_stage:
            if src is None:
                continue
            if not exists(dst):
                logging.info("Staging %s to %s" % (src, dst))
                copyfile(src, dst)
                if hasheader:
                    copyfile(envi_header(src), envi_header(dst))


class SerialEncoder(json.JSONEncoder):
    """Encoder for json to help ensure json objects can be passed to the workflow manager."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(SerialEncoder, self).default(obj)


class LUTConfig:
    """A look up table class, containing default grid options.  All properties may be overridden with the optional
        input configuration file path

    Args:
        lut_config_file: configuration file to override default values
        emulator: emulator used - will modify required points appropriately
        no_min_lut_spacing: span all LUT dimensions with at least 2 points
    """

    def __init__(
        self,
        lut_config_file: str = None,
        emulator: str = None,
        no_min_lut_spacing: bool = False,
        atmosphere_type="ATM_MIDLAT_SUMMER",
    ):
        if lut_config_file is not None:
            with open(lut_config_file, "r") as f:
                lut_config = json.load(f)

        # For each element, set the look up table spacing (lut_spacing) as the
        # anticipated spacing value, or 0 to use a single point (not LUT).
        # Set the 'lut_spacing_min' as the minimum distance allowed - if separation
        # does not meet this threshold based on the available data, on a single
        # point will be used.

        # Units of kilometers
        self.elevation_spacing = 0.25
        self.elevation_spacing_min = 0.2

        # Units of g / m2
        self.h2o_spacing = 0.25
        self.h2o_spacing_min = 0.03

        # Special parameter to specify the minimum allowable water vapor value in g / m2
        self.h2o_min = 0.2

        # Set defaults, will override based on settings
        # Units of g / m2
        self.h2o_range = [0.2, 5]

        # Units of degrees
        self.to_sensor_zenith_spacing = 10
        self.to_sensor_zenith_spacing_min = 2

        # Units of degrees
        self.to_sun_zenith_spacing = 10
        self.to_sun_zenith_spacing_min = 2

        # Units of degrees
        self.relative_azimuth_spacing = 60
        self.relative_azimuth_spacing_min = 25

        # Units of AOD
        self.aerosol_0_spacing = 0
        self.aerosol_0_spacing_min = 0

        # Units of AOD
        self.aerosol_1_spacing = 0
        self.aerosol_1_spacing_min = 0

        # Units of AOD
        self.aerosol_2_spacing = 0.1
        self.aerosol_2_spacing_min = 0

        # Units of AOD
        self.aerosol_0_range = [
            ModtranRT.modtran_aot_lowerbound_polynomials()[atmosphere_type](0),
            1,
        ]
        self.aerosol_1_range = [
            ModtranRT.modtran_aot_lowerbound_polynomials()[atmosphere_type](0),
            1,
        ]
        self.aerosol_2_range = [
            ModtranRT.modtran_aot_lowerbound_polynomials()[atmosphere_type](0),
            1,
        ]
        self.aot_550_range = [
            ModtranRT.modtran_aot_lowerbound_polynomials()[atmosphere_type](0),
            1,
        ]

        self.aot_550_spacing = 0
        self.aot_550_spacing_min = 0

        # CO2 ppm
        self.co2_range = [380, 440]
        self.co2_spacing = 60
        self.co2_spacing_min = 60

        self.no_min_lut_spacing = no_min_lut_spacing

        # overwrite anything that comes in from the config file
        if lut_config_file is not None:
            for key in lut_config:
                if key in self.__dict__:
                    setattr(self, key, lut_config[key])

        if emulator is not None and os.path.splitext(emulator)[1] != ".jld2":
            self.aot_550_range = self.aerosol_2_range
            self.aot_550_spacing = self.aerosol_2_spacing
            self.aot_550_spacing_min = self.aerosol_2_spacing_min
            self.aerosol_2_spacing = 0

    def get_grid_with_data(
        self, data_input: np.array, spacing: float, min_spacing: float
    ):
        min_val = np.min(data_input)
        max_val = np.max(data_input)
        return self.get_grid(min_val, max_val, spacing, min_spacing)

    def get_grid(
        self, minval: float, maxval: float, spacing: float, min_spacing: float
    ):
        if spacing == 0:
            logging.debug("Grid spacing set at 0, using no grid.")
            return None
        num_gridpoints = int(np.ceil((maxval - minval) / spacing)) + 1

        # if we want to ensure there is no minimum spacing, override the spacing
        # value to set the number of grid points to at least 2
        if (
            self.no_min_lut_spacing
            and num_gridpoints == 1
            and np.isclose(maxval, minval) is False
        ):
            num_gridpoints = 2

        grid = np.linspace(minval, maxval, num_gridpoints)

        if min_spacing > 0.0001:
            grid = np.round(grid, 4)
        if len(grid) == 1:
            logging.debug(
                f"Grid spacing is 0, which is less than {min_spacing}.  No grid used"
            )
            return None
        elif (
            # Need this first conditional to rule out rounding errors
            len(grid) == 2
            and np.abs(grid[1] - grid[0]) < min_spacing
            and self.no_min_lut_spacing is False
        ):
            logging.debug(
                f"Grid spacing is {grid[1]-grid[0]}, which is less than {min_spacing}. "
                " No grid used"
            )
            return None
        else:
            return grid


class SerialEncoder(json.JSONEncoder):
    """Encoder for json to help ensure json objects can be passed to the workflow manager."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(SerialEncoder, self).default(obj)


def check_surface_model(
    surface_path: str,
    output_model_path: str = None,
    wl: np.array = [],
    surface_wavelength_path: str = "",
    surface_category: str = "multicomponent_surface",
    multisurface: bool = False,
) -> str:
    """
    Checks and rebuilds surface model if needed.

    TODO - Could be extended to allow for both dir and file surface_path
           inputs. The dir inputs could accomdate complex surfaces where
           different surface priors might be useful. This extension
           would be relatively easy. Wrap this in a check for surface_path
           vs. surface_path_dir. Each file in the dir can undergo the
           same check coded here.

    Args:
        surface_path: path to surface model or config dict
        wl: instrument center wavelengths
        surface_wavelength_path: path to wavelength file
    """
    if os.path.isfile(surface_path):
        if surface_path.endswith(".mat"):
            # check wavelength grid of surface model if provided
            model_dict = loadmat(surface_path)
            wl_surface = model_dict["wl"][0]

            if not len(wl):
                logging.info("No wl array given. Not checking channel number matchup")

            elif len(wl_surface) != len(wl):
                raise ValueError(
                    "Number of channels provided in surface model file does not match"
                    " wavelengths in radiance cube. Please rebuild your surface model."
                )
            if not np.all(np.isclose(wl_surface, wl, atol=0.01)):
                logging.warning(
                    "Center wavelengths provided in surface model file do not match"
                    " wavelengths in radiance cube. Please consider rebuilding your"
                    " surface model for optimal performance."
                )

            if multisurface:
                # TODO Change the surface model structure for multisurface runs
                # Instead of passing multiple surface.mat files throughout.
                # Carry the surface type keys and dynamically select the matching type within surface.component
                raise ValueError(
                    "Apply OE in multistate-mode can currently only be run from a .json surface file. Please use the path to the .json file as the surface_ath. Must include the 'surface_type' keys."
                )

            return {surface_category: surface_path}

        elif surface_path.endswith(".json"):
            logging.info(
                "No surface model provided. Build new one using given config file."
            )

            if not surface_wavelength_path:
                raise ValueError(
                    "Building surface model requires input surface wavelength path"
                )

            if not output_model_path:
                logging.info(
                    "No output path provided via check_surface, "
                    "using output file within surface config."
                )
                configdir, _ = os.path.split(os.path.abspath(surface_path))
                config = json_load_ascii(surface_path, shell_replace=True)
                output_model_path = expand_path(configdir, config["output_model_file"])

            surface_model(
                config_path=surface_path,
                wavelength_path=surface_wavelength_path,
                output_path=output_model_path,
                multisurface=multisurface,
            )

            # Handle a multisurface run with split
            if multisurface:
                config = json_load_ascii(surface_path, shell_replace=True)
                surface_categories = []
                for source in config["sources"]:
                    surface_category = source.get("surface_category")
                    if not surface_category:
                        raise ValueError(
                            "Multisurface ISOFIT surface configs require "
                            "surface category to be specific in config. "
                            "No 'surface_category' key found. Check config"
                        )
                    surface_categories.append(surface_category)

                surface_paths = {}
                for surface_category in np.unique(surface_categories):
                    name, ext = os.path.splitext(output_model_path)
                    surface_paths[str(surface_category)] = (
                        f"{name}_{str(surface_category)}{ext}"
                    )

            else:
                surface_paths = {surface_category: output_model_path}

            for path in surface_paths.values():
                try:
                    model_dict = loadmat(path)
                except ValueError:
                    raise ValueError(
                        "Surface.mat file failed to load. " "Check configuration."
                    )
            return surface_paths

        else:
            raise FileNotFoundError(
                "Unrecognized format of surface file. Please provide either a .mat model or a .json config dict."
            )
    else:
        raise FileNotFoundError(
            "No surface file found. Please provide either a .mat model or a .json config dict."
        )


def build_config(
    paths: Pathnames,
    h2o_lut_grid: np.array = None,
    elevation_lut_grid: np.array = None,
    to_sensor_zenith_lut_grid: np.array = None,
    to_sun_zenith_lut_grid: np.array = None,
    relative_azimuth_lut_grid: np.array = None,
    co2_lut_grid: np.array = None,
    aerosol_lut_grid: np.array = None,
    aerosol_model_file: str = None,
    aerosol_state_vector: dict = None,
    use_superpixels: bool = True,
    n_cores: int = -1,
    surface_category="multicomponent_surface",
    emulator_base: str = None,
    uncorrelated_radiometric_uncertainty: float = 0.0,
    dn_uncertainty_file: str = None,
    multiple_restarts: bool = False,
    segmentation_size=400,
    pressure_elevation: bool = False,
    debug: bool = False,
    inversion_windows=[[350.0, 1360.0], [1410, 1800.0], [1970.0, 2500.0]],
    prebuilt_lut_path: str = None,
    multipart_transmittance: bool = False,
    surface_mapping: dict = None,
    retrieve_co2: bool = False,
    presolve: bool = False,
    terrain_style: str = "flat",
    cos_i_min: float = 0.3,
) -> None:
    """Write an isofit config file for the main solve, using the specified pathnames and all given info

    Args:
        paths:                                object containing references to all relevant file locations
        h2o_lut_grid:                         the water vapor look up table grid isofit should use for this solve
        elevation_lut_grid:                   the ground elevation look up table grid isofit should use for this solve
        to_sensor_zenith_lut_grid:            the to-sensor zenith angle look up table grid isofit should use for this
                                              solve
        to_sun_zenith_lut_grid:               the to-sun zenith angle look up table grid isofit should use for this
                                              solve
        relative_azimuth_lut_grid:            the relative to-sun azimuth angle look up table grid isofit should use for
                                              this solve
        co2_lut_grid:                         CO2 look up table grid
        aerosol_lut_grid:                      aerosol look up table grid
        aerosol_model_file:                   aerosol model file path
        aerosol_state_vector:                 aerosol state vector info
        use_superpixels:                      flag whether or not to use superpixels for the solution
        n_cores:                              the number of cores to use during processing
        surface_category:                     type of surface to use
        emulator_base:                        the basename of the emulator, if used
        uncorrelated_radiometric_uncertainty: uncorrelated radiometric uncertainty parameter for isofit
        dn_uncertainty_file:                       Path to a linearity .mat file to augment S matrix with linearity uncertainty
        multiple_restarts:                    if true, use multiple restarts
        segmentation_size:                    image segmentation size if empirical line is used
        pressure_elevation:                   if true, retrieve pressure elevation
        debug:                                if true, run ISOFIT in debug mode
        multipart_transmittance:              flag to indicate whether a 4-component transmittance model is to be used
        surface_mapping:                      optional object to pass mapping between surface class and surface model
        retrieve_co2:                         flag to include CO2 in lut and retrieval
        presolve:                             set this up as a presolve configuration
        terrain_style:                        style of terrain to use in the forward model - options are 'flat', 'dem', 'solved'
        cos_i_min:                            minimum cosine of incidence angle to allow isofit to use in the forward model
    """

    avc = np.sum(
        [
            x is not None
            for x in [aerosol_lut_grid, aerosol_model_file, aerosol_state_vector]
        ]
    )
    if avc >= 1 and avc != 3:
        raise ValueError(
            "To use aerosol in LUT, need lut_grid, model_path, and state_vector"
        )

    lut_dir = paths.lut_h2o_directory if presolve else paths.full_lut_directory
    lut_path = (
        join(lut_dir, "lut.nc")
        if prebuilt_lut_path is None
        else abspath(prebuilt_lut_path)
    )

    if emulator_base is None:
        engine_name = "modtran"
    elif emulator_base.endswith(".jld2"):
        engine_name = "KernelFlowsGP"
    else:
        engine_name = "sRTMnet"

    radiative_transfer_config = {
        "radiative_transfer_engines": {
            "vswir": {
                "engine_name": engine_name,
                "multipart_transmittance": multipart_transmittance,
                "sim_path": lut_dir,
                "lut_path": lut_path,
                "aerosol_template_file": paths.aerosol_tpl_path,
                "template_file": (
                    paths.modtran_template_path
                    if not presolve
                    else paths.h2o_template_path
                ),
            }
        },
        "statevector": {},
        "lut_grid": {},
        "unknowns": {"H2O_ABSCO": 0.0},
        "terrain_style": terrain_style,
        "cos_i_min": cos_i_min,
    }

    vswir = {}
    if emulator_base is not None:
        vswir["emulator_file"] = abspath(emulator_base)
        vswir["earth_sun_distance_file"] = paths.earth_sun_distance_path
        vswir["irradiance_file"] = paths.irradiance_file
        vswir["engine_base_dir"] = paths.sixs_path
        if multipart_transmittance:
            vswir["emulator_aux_file"] = abspath(emulator_base)
        else:
            vswir["emulator_aux_file"] = abspath(
                os.path.splitext(emulator_base)[0] + "_aux.npz"
            )
    else:
        vswir["engine_base_dir"] = paths.modtran_path
    radiative_transfer_config["radiative_transfer_engines"]["vswir"].update(vswir)

    if aerosol_model_file is None:
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "aerosol_model_file"
        ] = aerosol_model_file

    # First, build the general lut grid
    lut_grid = {
        "H2OSTR": h2o_lut_grid,
        "surface_elevation_km": elevation_lut_grid,
        "observer_zenith": to_sensor_zenith_lut_grid,
        "solar_zenith": to_sun_zenith_lut_grid,
        "relative_azimuth": relative_azimuth_lut_grid,
        "CO2": co2_lut_grid,
    }
    if aerosol_lut_grid is not None:
        lut_grid.update(aerosol_lut_grid)

    to_remove = []
    for gn, gc in lut_grid.items():
        if gc is None or len(gc) == 1:
            to_remove.append(gn)
        else:
            lut_grid[gn] = np.array(gc).tolist()

    if emulator_base is not None and os.path.splitext(emulator_base)[1] == ".jld2":
        from isofit.radiative_transfer.engines.kernel_flows import bounds_check

        # Should only modify H2OSTR and surface_elevation_km
        bounds_check(lut_grid, emulator_base, modify=True)

    ncds = None
    if prebuilt_lut_path is not None:
        ncds = nc.Dataset(prebuilt_lut_path, "r")
        for gn, gc in lut_grid.items():
            if gn not in ncds.variables:
                logging.warning(
                    f"Key {gn} not found in prebuilt LUT, removing it from LUT."
                )
                to_remove.append(gn)
            else:
                lut_grid[gn] = get_lut_subset(gc)

    for tr in np.unique(to_remove):
        lut_grid.pop(tr)

    radiative_transfer_config["lut_grid"].update(lut_grid)
    radiative_transfer_config["radiative_transfer_engines"]["vswir"]["lut_names"] = {
        key: None for key in lut_grid.keys()
    }

    # Now do statevector
    statekeys = ["H2OSTR"]
    statesigmas = [100.0]
    statescale = [1]
    if pressure_elevation and presolve is False:
        statekeys.append("surface_elevation_km")
        statesigmas.append(1000.0)
        statescale.append(100)
    if retrieve_co2 and presolve is False:
        statekeys.append("CO2")
        statesigmas.append(100.0)
        statescale.append(10)

    for key, sigma, scale in zip(statekeys, statesigmas, statescale):
        if key in lut_grid:
            grid = lut_grid[key]
            radiative_transfer_config["statevector"][key] = {
                "bounds": [grid[0], grid[-1]],
                "scale": scale,
                "init": (grid[0] + grid[-1]) / 2.0,
                "prior_sigma": sigma,
                "prior_mean": (grid[0] + grid[-1]) / 2.0,
            }

    if aerosol_state_vector is not None and presolve is False:
        radiative_transfer_config["statevector"].update(aerosol_state_vector)

    # MODTRAN should know about our whole LUT grid and all of our statevectors, so copy them in
    radiative_transfer_config["radiative_transfer_engines"]["vswir"][
        "statevector_names"
    ] = list(radiative_transfer_config["statevector"].keys())

    # make isofit configuration
    isofit_config_modtran = {
        "input": {},
        "output": {},
        "forward_model": {
            "instrument": {
                "wavelength_file": paths.wavelength_path,
                "integrations": segmentation_size if use_superpixels else 1,
                "unknowns": {
                    "uncorrelated_radiometric_uncertainty": uncorrelated_radiometric_uncertainty,
                    "dn_uncertainty_file": dn_uncertainty_file,
                },
            },
            "surface": make_surface_config(
                paths,
                surface_category,
                pressure_elevation,
                elevation_lut_grid,
                surface_mapping=surface_mapping,
                use_superpixels=use_superpixels,
            ),
            "radiative_transfer": radiative_transfer_config,
        },
        "implementation": {
            "ray_temp_dir": paths.ray_temp_dir,
            "inversion": {"windows": inversion_windows},
            "n_cores": n_cores,
            "debug_mode": debug,
            "isofit_version": __version__,
        },
    }

    input, output = {}, {}
    if use_superpixels:
        input["measured_radiance_file"] = paths.rdn_subs_path
        input["loc_file"] = paths.loc_subs_path
        input["obs_file"] = paths.obs_subs_path
        if paths.svf_working_path:
            input["skyview_factor_file"] = paths.svf_subs_path
        if presolve:
            output["estimated_state_file"] = paths.h2o_subs_path
        else:
            output["estimated_state_file"] = paths.state_subs_path
            output["posterior_uncertainty_file"] = paths.uncert_subs_path
            output["estimated_reflectance_file"] = paths.rfl_subs_path
    else:
        input["measured_radiance_file"] = paths.radiance_working_path
        input["loc_file"] = paths.loc_working_path
        input["obs_file"] = paths.obs_working_path
        if paths.svf_working_path:
            input["skyview_factor_file"] = paths.svf_working_path

        if presolve:
            output["estimated_state_file"] = paths.h2o_working_path
        else:
            output["posterior_uncertainty_file"] = paths.uncert_working_path
            output["estimated_reflectance_file"] = paths.rfl_working_path
            output["estimated_state_file"] = paths.state_working_path
    isofit_config_modtran["output"].update(output)
    isofit_config_modtran["input"].update(input)

    if multiple_restarts:
        grid = {}
        if h2o_lut_grid is not None:
            h2o_delta = float(h2o_lut_grid[-1]) - float(h2o_lut_grid[0])
            grid["H2OSTR"] = [
                round(h2o_lut_grid[0] + h2o_delta * 0.02, 4),
                round(h2o_lut_grid[-1] - h2o_delta * 0.02, 4),
            ]

        # We will initialize using different AODs for the first aerosol in the LUT
        if len(aerosol_lut_grid) > 0:
            key = list(aerosol_lut_grid.keys())[0]
            aer_delta = aerosol_lut_grid[key][-1] - aerosol_lut_grid[key][0]
            grid[key] = [
                round(aerosol_lut_grid[key][0] + aer_delta * 0.02, 4),
                round(aerosol_lut_grid[key][-1] - aer_delta * 0.02, 4),
            ]
        isofit_config_modtran["implementation"]["inversion"]["integration_grid"] = grid
        isofit_config_modtran["implementation"]["inversion"][
            "inversion_grid_as_preseed"
        ] = True

    if paths.input_channelized_uncertainty_path is not None:
        isofit_config_modtran["forward_model"]["instrument"]["unknowns"][
            "channelized_radiometric_uncertainty_file"
        ] = paths.channelized_uncertainty_working_path

    if paths.eof_path is not None:
        isofit_config_modtran["forward_model"]["instrument"][
            "eof_path"
        ] = paths.eof_working_path

        # Add a state vector element for each column in the EOF file
        eof = np.loadtxt(paths.eof_path)
        isofit_config_modtran["forward_model"]["instrument"]["statevector"] = {}
        for idx in range(eof.shape[1]):
            key = "EOF_%i" % (idx + 1)
            isofit_config_modtran["forward_model"]["instrument"]["statevector"][key] = {
                "bounds": [-10, 10],
                "scale": 1,
                "init": 0,
                "prior_sigma": 100.0,
                "prior_mean": 0,
            }

    if paths.input_model_discrepancy_path is not None:
        isofit_config_modtran["forward_model"][
            "model_discrepancy_file"
        ] = paths.model_discrepancy_working_path

    if paths.noise_path is not None:
        isofit_config_modtran["forward_model"]["instrument"][
            "parametric_noise_file"
        ] = paths.noise_path

    else:
        isofit_config_modtran["forward_model"]["instrument"]["SNR"] = 500

    if paths.rdn_factors_path:
        isofit_config_modtran["input"][
            "radiometry_correction_file"
        ] = paths.rdn_factors_path

    # write main config file
    with open(paths.isofit_full_config_path, "w") as fout:
        fout.write(
            json.dumps(
                isofit_config_modtran, cls=SerialEncoder, indent=4, sort_keys=True
            )
        )

    # Create a template version of the config
    if presolve:
        outfile = paths.h2o_config_path
    else:
        outfile = paths.isofit_full_config_path
    with open(outfile, "w") as fout:
        fout.write(
            json.dumps(
                isofit_config_modtran, cls=SerialEncoder, indent=4, sort_keys=True
            )
        )
    env.toTemplate(outfile, working_directory=paths.working_directory)


def get_lut_subset(vals):
    """Populate lut_names for the appropriate style of subsetting

    Args:
        vals: the values to use for subsetting
    """

    if vals is not None and len(vals) == 1:
        return {"interp": vals[0]}
    elif vals is not None and len(vals) > 1:
        return {"gte": vals[0], "lte": vals[-1]}
    else:
        return None


def write_modtran_template(
    atmosphere_type: str,
    fid: str,
    altitude_km: float,
    dayofyear: int,
    to_sensor_azimuth: float,
    to_sensor_zenith: float,
    to_sun_zenith: float,
    relative_azimuth: float,
    gmtime: float,
    elevation_km: float,
    output_file: str,
    ihaze_type: str = "AER_RURAL",
):
    """Write a MODTRAN template file for use by isofit look up tables

    Args:
        atmosphere_type:   label for the type of atmospheric profile to use in modtran
        fid:               flight line id (name)
        altitude_km:       altitude of the sensor in km
        dayofyear:         the current day of the given year
        to_sensor_azimuth: azimuth view angle to the sensor, in degrees
        to_sensor_zenith:  sensor/observer zenith angle, in degrees
        to_sun_zenith:     final altitude solar zenith angle (0→180°)
        relative_azimuth:  final altitude relative solar azimuth (0→360°)
        gmtime:            greenwich mean time
        elevation_km:      elevation of the land surface in km
        output_file:       location to write the modtran template file to
        ihaze_type:        type of extinction and default meteorological range for the boundary-layer aerosol model
    """
    # make modtran configuration
    output_template = {
        "MODTRAN": [
            {
                "MODTRANINPUT": {
                    "NAME": fid,
                    "DESCRIPTION": "",
                    "CASE": 0,
                    "RTOPTIONS": {
                        "MODTRN": "RT_CORRK_FAST",
                        "LYMOLC": False,
                        "T_BEST": False,
                        "IEMSCT": "RT_SOLAR_AND_THERMAL",
                        "IMULT": "RT_DISORT",
                        "DISALB": False,
                        "NSTR": 8,
                        "SOLCON": 0.0,
                    },
                    "ATMOSPHERE": {
                        "MODEL": atmosphere_type,
                        "M1": atmosphere_type,
                        "M2": atmosphere_type,
                        "M3": atmosphere_type,
                        "M4": atmosphere_type,
                        "M5": atmosphere_type,
                        "M6": atmosphere_type,
                        "CO2MX": 420.0,
                        "H2OSTR": 1.0,
                        "H2OUNIT": "g",
                        "O3STR": 0.3,
                        "O3UNIT": "a",
                    },
                    "AEROSOLS": {"IHAZE": ihaze_type},
                    "GEOMETRY": {
                        "ITYPE": 3,
                        "H1ALT": altitude_km,
                        "IDAY": dayofyear,
                        "IPARM": 12,
                        "PARM1": relative_azimuth,
                        "PARM2": to_sun_zenith,
                        "TRUEAZ": to_sensor_azimuth,
                        "OBSZEN": 180 - to_sensor_zenith,  # MODTRAN convention
                        "GMTIME": gmtime,
                    },
                    "SURFACE": {
                        "SURFTYPE": "REFL_LAMBER_MODEL",
                        "GNDALT": elevation_km,
                        "NSURF": 1,
                        "SURFP": {"CSALB": "LAMB_CONST_0_PCT"},
                    },
                    "SPECTRAL": {
                        "V1": 340.0,
                        "V2": 2520.0,
                        "DV": 0.1,
                        "FWHM": 0.1,
                        "YFLAG": "R",
                        "XFLAG": "N",
                        "FLAGS": "NT A   ",
                        "BMNAME": "p1_2013",
                    },
                    "FILEOPTIONS": {"NOPRNT": 2, "CKPRNT": True},
                }
            }
        ]
    }

    # write modtran_template
    with open(output_file, "w") as fout:
        fout.write(
            json.dumps(output_template, cls=SerialEncoder, indent=4, sort_keys=True)
        )


def load_climatology(
    config_path: str,
    latitude: float,
    longitude: float,
    acquisition_datetime: datetime,
    lut_params: LUTConfig,
):
    """Load climatology data, based on location and configuration

    Args:
        config_path: path to the base configuration directory for isofit
        latitude: latitude to set for the segment (mean of acquisition suggested)
        longitude: latitude to set for the segment (mean of acquisition suggested)
        acquisition_datetime: datetime to use for the segment( mean of acquisition suggested)
        lut_params: parameters to use to define lut grid

    :Returns
        tuple containing:
            aerosol_state_vector - A dictionary that defines the aerosol state vectors for isofit
            aerosol_lut_grid - A dictionary of the aerosol lookup table (lut) grid to be explored
            aerosol_model_path - A path to the location of the aerosol model to use with MODTRAN.

    """

    aerosol_model_path = str(env.path("data", "aerosol_model.txt"))
    aerosol_state_vector = {}
    aerosol_lut_grid = {}
    aerosol_lut_ranges = [
        lut_params.aerosol_0_range,
        lut_params.aerosol_1_range,
        lut_params.aerosol_2_range,
    ]
    aerosol_lut_spacing = [
        lut_params.aerosol_0_spacing,
        lut_params.aerosol_1_spacing,
        lut_params.aerosol_2_spacing,
    ]
    aerosol_lut_spacing_mins = [
        lut_params.aerosol_0_spacing_min,
        lut_params.aerosol_1_spacing_min,
        lut_params.aerosol_2_spacing_min,
    ]

    for _a, alr in enumerate(aerosol_lut_ranges):
        aerosol_lut = lut_params.get_grid(
            alr[0], alr[1], aerosol_lut_spacing[_a], aerosol_lut_spacing_mins[_a]
        )

        if aerosol_lut is not None:
            aerosol_state_vector["AERFRAC_{}".format(_a)] = {
                "bounds": [float(alr[0]), float(alr[1])],
                "scale": 1,
                "init": float((alr[1] - alr[0]) / 10.0 + alr[0]),
                "prior_sigma": 10.0,
                "prior_mean": float((alr[1] - alr[0]) / 10.0 + alr[0]),
            }

            aerosol_lut_grid["AERFRAC_{}".format(_a)] = aerosol_lut.tolist()

    aot_550_lut = lut_params.get_grid(
        lut_params.aot_550_range[0],
        lut_params.aot_550_range[1],
        lut_params.aot_550_spacing,
        lut_params.aot_550_spacing_min,
    )

    if aot_550_lut is not None:
        aerosol_lut_grid["AOT550"] = aot_550_lut.tolist()
        alr = [aerosol_lut_grid["AOT550"][0], aerosol_lut_grid["AOT550"][-1]]
        aerosol_state_vector["AOT550"] = {
            "bounds": [float(alr[0]), float(alr[1])],
            "scale": 1,
            "init": float((alr[1] - alr[0]) / 10.0 + alr[0]),
            "prior_sigma": 10.0,
            "prior_mean": float((alr[1] - alr[0]) / 10.0 + alr[0]),
        }

    logging.info("Loading Climatology")
    # If a configuration path has been provided, use it to get relevant info
    if config_path is not None:
        month = acquisition_datetime.timetuple().tm_mon
        year = acquisition_datetime.timetuple().tm_year

        with open(config_path, "r") as fin:
            for case in json.load(fin)["cases"]:
                match = True
                logging.info("matching", latitude, longitude, month, year)

                for criterion, interval in case["criteria"].items():
                    logging.info(criterion, interval, "...")

                    if criterion == "latitude":
                        if latitude < interval[0] or latitude > interval[1]:
                            match = False

                    if criterion == "longitude":
                        if longitude < interval[0] or longitude > interval[1]:
                            match = False

                    if criterion == "month":
                        if month < interval[0] or month > interval[1]:
                            match = False

                    if criterion == "year":
                        if year < interval[0] or year > interval[1]:
                            match = False

                if match:
                    aerosol_state_vector = case["aerosol_state_vector"]
                    aerosol_lut_grid = case["aerosol_lut_grid"]
                    aerosol_model_path = case["aerosol_mdl_path"]
                    break

    logging.info(
        "Climatology Loaded.  Aerosol State Vector:\n{}\nAerosol LUT Grid:\n{}\nAerosol"
        " model path:{}".format(
            aerosol_state_vector, aerosol_lut_grid, aerosol_model_path
        )
    )

    return aerosol_state_vector, aerosol_lut_grid, aerosol_model_path


def calc_modtran_max_water(paths: Pathnames) -> float:
    """MODTRAN may put a ceiling on "legal" H2O concentrations.  This function calculates that ceiling.  The intended
     use is to make sure the LUT does not contain useless gridpoints above it.

    Args:
        paths: object containing references to all relevant file locations

    Returns:
        max_water - maximum MODTRAN H2OSTR value for provided obs conditions
    """

    max_water = None
    # TODO: this is effectively redundant from the radiative_transfer->modtran. Either devise a way
    # to port in from there, or put in utils to reduce redundancy.
    xdir = {"linux": "linux", "darwin": "macos", "windows": "windows"}
    name = "H2O_bound_test"
    filebase = os.path.join(paths.lut_h2o_directory, name)

    with open(paths.h2o_template_path, "r") as f:
        bound_test_config = json.load(f)

    bound_test_config["MODTRAN"][0]["MODTRANINPUT"]["NAME"] = name
    bound_test_config["MODTRAN"][0]["MODTRANINPUT"]["ATMOSPHERE"]["H2OSTR"] = 50

    with open(filebase + ".json", "w") as fout:
        fout.write(
            json.dumps(bound_test_config, cls=SerialEncoder, indent=4, sort_keys=True)
        )

    cmd = os.path.join(
        paths.modtran_path, "bin", xdir[platform], "mod6c_cons " + filebase + ".json"
    )

    try:
        subprocess.call(cmd, shell=True, timeout=10, cwd=paths.lut_h2o_directory)
    except:
        pass

    with open(filebase + ".tp6", errors="ignore") as tp6file:
        for count, line in enumerate(tp6file):
            if "The water column is being set to the maximum" in line:
                max_water = line.split(",")[1].strip()
                max_water = float(max_water.split(" ")[0])
                break

    if max_water is None:
        logging.error(
            "Could not find MODTRAN H2O upper bound in file {}".format(
                filebase + ".tp6"
            )
        )
        raise KeyError("Could not find MODTRAN H2O upper bound")

    return max_water


def define_surface_types(
    tsip: dict,
    rdnfile: str,
    obsfile: str,
    out_class_path: str,
    wl: np.array,
    fwhm: np.array,
):
    if np.all(wl < 10):
        wl = units.micron_to_nm(wl)
        fwhm = unts.micron_to_nm(fwhm)

    irr_file = os.path.join(
        os.path.dirname(isofit.__file__), "..", "..", "data", "kurucz_0.1nm.dat"
    )
    irr_wl, irr = np.loadtxt(irr_file, comments="#").T
    irr = irr / 10  # convert to uW cm-2 sr-1 nm-1
    irr_resamp = resample_spectrum(irr, irr_wl, wl, fwhm)
    irr_resamp = np.array(irr_resamp, dtype=np.float32)
    irr = irr_resamp

    rdn_ds = envi.open(envi_header(rdnfile)).open_memmap(interleave="bip")
    obs_src = envi.open(envi_header(obsfile))
    obs_ds = obs_src.open_memmap(interleave="bip")

    # determine glint bands having negligible water reflectance
    try:
        b1000 = np.argmin(abs(wl - tsip["water"]["toa_threshold_wavelengths"][0]))
        b1380 = np.argmin(abs(wl - tsip["water"]["toa_threshold_wavelengths"][1]))
    except KeyError:
        logging.info(
            "No threshold wavelengths for water classification found in config file. "
            "Setting to 1000 and 1380 nm."
        )
        b1000 = np.argmin(abs(wl - 1000))
        b1380 = np.argmin(abs(wl - 1380))

    # determine cloud bands having high TOA reflectance
    try:
        b450 = np.argmin(abs(wl - tsip["cloud"]["toa_threshold_wavelengths"][0]))
        b1250 = np.argmin(abs(wl - tsip["cloud"]["toa_threshold_wavelengths"][1]))
        b1650 = np.argmin(abs(wl - tsip["cloud"]["toa_threshold_wavelengths"][2]))
    except KeyError:
        logging.info(
            "No threshold wavelengths for cloud classification found in config file. "
            "Setting to 450, 1250, and 1650 nm."
        )
        b450 = np.argmin(abs(wl - 450))
        b1250 = np.argmin(abs(wl - 1250))
        b1650 = np.argmin(abs(wl - 1650))

    classes = np.zeros(rdn_ds.shape[:2])

    for line in range(classes.shape[0]):
        for sample in range(classes.shape[1]):
            zen = np.cos(np.deg2rad(obs_ds[line, sample, 4]))

            rho = (((rdn_ds[line, sample, :] * np.pi) / irr.T).T / np.cos(zen)).T

            rho[rho[0] < -9990, :] = -9999.0

            if rho[0] < -9999:
                classes[line, sample] = -1
                continue

            # Cloud threshold from Sandford et al.
            total = (
                np.array(
                    rho[b450] > tsip["cloud"]["toa_threshold_values"][0], dtype=int
                )
                + np.array(
                    rho[b1250] > tsip["cloud"]["toa_threshold_values"][1], dtype=int
                )
                + np.array(
                    rho[b1650] > tsip["cloud"]["toa_threshold_values"][2], dtype=int
                )
            )

            if rho[b1000] < tsip["water"]["toa_threshold_values"][0]:
                classes[line, sample] = 2

            if total > 2 or rho[b1380] > tsip["water"]["toa_threshold_values"][1]:
                classes[line, sample] = 1

    header = obs_src.metadata.copy()
    header["bands"] = 1

    if "band names" in header.keys():
        header["band names"] = "Class"

    output_ds = envi.create_image(
        envi_header(out_class_path), header, ext="", force=True
    )
    output_mm = output_ds.open_memmap(interleave="bip", writable=True)
    output_mm[:, :, 0] = classes

    return classes


def copy_file_subset(matching_indices: np.array, pathnames: List):
    """Copy over subsets of given files to new locations

    Args:
        matching_indices (np.array): indices to select from (y dimension) from source dataset
        pathnames (List): list of tuples (input_filename, output_filename) to read/write to/from
    """
    for inp, outp in pathnames:
        input_ds = envi.open(envi_header(inp), inp)
        header = input_ds.metadata.copy()
        header["lines"] = np.sum(matching_indices)
        header["samples"] = 1
        output_ds = envi.create_image(envi_header(outp), header, ext="", force=True)
        output_mm = output_ds.open_memmap(interleave="bip", writable=True)
        input_mm = input_ds.open_memmap(interleave="bip", writable=True)
        output_mm[:, 0, :] = input_mm[matching_indices[:, :, 0], ...].copy()


def get_metadata_from_obs(
    obs_file: str,
    lut_params: LUTConfig,
    trim_lines: int = 5,
    max_flight_duration_h: int = 8,
    nodata_value: float = -9999,
) -> (List, bool, float, float, float, np.array, List, List):
    """Get metadata needed for complete runs from the observation file
    (bands: path length, to-sensor azimuth, to-sensor zenith, to-sun azimuth,
    to-sun zenith, phase, slope, aspect, cosine i, UTC time).

    Args:
        obs_file:              file name to pull data from
        lut_params:            parameters to use to define lut grid
        trim_lines:            number of lines to ignore at beginning and end of file (good if lines contain values
                               that are erroneous but not nodata
        max_flight_duration_h: maximum length of the current acquisition, used to check if we've lapped a UTC day
        nodata_value:          value to ignore from location file

    :Returns:
        tuple containing:
            h_m_s - list of the mean-time hour, minute, and second within the line
            increment_day - indicator of whether the UTC day has been changed since the beginning of the line time
            mean_path_km - mean distance between sensor and ground in km for good data
            mean_to_sensor_azimuth - mean to-sensor azimuth for good data
            mean_to_sensor_zenith - mean to-sensor zenith for good data
            mean_to_sun_azimuth - mean to-sun-azimuth for good data
            mean_to_sun_zenith - mean to-sun zenith for good data
            mean_relative_azimuth - mean relative to-sun azimuth for good data
            valid - boolean array indicating which pixels were NOT nodata
            to_sensor_zenith_lut_grid - the to-sensor zenith look up table grid for good data
            to_sun_zenith_lut_grid - the to-sun zenith look up table grid for good data
            relative_azimuth_lut_grid - the relative to-sun azimuth look up table grid for good data
    """
    obs_dataset = envi.open(envi_header(obs_file), obs_file)
    obs = obs_dataset.open_memmap(interleave="bip", writable=False)
    valid = np.logical_not(np.any(np.isclose(obs, nodata_value), axis=2))

    path_km = units.m_to_km(obs[:, :, 0])
    to_sensor_azimuth = obs[:, :, 1]
    to_sensor_zenith = obs[:, :, 2]
    to_sun_azimuth = obs[:, :, 3]
    to_sun_zenith = obs[:, :, 4]
    time = obs[:, :, 9].copy()

    # calculate relative to-sun azimuth
    delta_phi = np.abs(to_sun_azimuth - to_sensor_azimuth)
    relative_azimuth = np.minimum(delta_phi, 360 - delta_phi)

    use_trim = trim_lines != 0 and valid.shape[0] > trim_lines * 2
    if use_trim:
        actual_valid = valid.copy()
        valid[:trim_lines, :] = False
        valid[-trim_lines:, :] = False

    mean_path_km = np.mean(path_km[valid])
    del path_km

    mean_to_sensor_azimuth = np.mean(to_sensor_azimuth[valid]) % 360
    mean_to_sun_azimuth = np.mean(to_sun_azimuth[valid]) % 360
    mean_to_sensor_zenith = np.mean(to_sensor_zenith[valid])
    mean_to_sun_zenith = np.mean(to_sun_zenith[valid])
    mean_relative_azimuth = np.mean(relative_azimuth[valid])

    # geom_margin = EPS * 2.0
    to_sensor_zenith_lut_grid = lut_params.get_grid_with_data(
        to_sensor_zenith[valid],
        lut_params.to_sensor_zenith_spacing,
        lut_params.to_sensor_zenith_spacing_min,
    )

    if to_sensor_zenith_lut_grid is not None:
        to_sensor_zenith_lut_grid = np.sort(to_sensor_zenith_lut_grid)

    to_sun_zenith_lut_grid = lut_params.get_grid_with_data(
        to_sun_zenith[valid],
        lut_params.to_sun_zenith_spacing,
        lut_params.to_sun_zenith_spacing_min,
    )

    if to_sun_zenith_lut_grid is not None:
        to_sun_zenith_lut_grid = np.sort(to_sun_zenith_lut_grid)

    relative_azimuth_lut_grid = lut_params.get_grid_with_data(
        relative_azimuth[valid],
        lut_params.relative_azimuth_spacing,
        lut_params.relative_azimuth_spacing_min,
    )

    if relative_azimuth_lut_grid is not None:
        relative_azimuth_lut_grid = np.sort(
            np.array([x % 360 for x in relative_azimuth_lut_grid])
        )

    del to_sensor_azimuth
    del to_sensor_zenith
    del to_sun_azimuth
    del to_sun_zenith
    del relative_azimuth

    # Make time calculations
    mean_time = np.mean(time[valid])
    min_time = np.min(time[valid])
    max_time = np.max(time[valid])

    increment_day = False
    # UTC day crossover corner case
    if max_time > 24 - max_flight_duration_h and min_time < max_flight_duration_h:
        time[np.logical_and(time < max_flight_duration_h, valid)] += 24
        mean_time = np.mean(time[valid])

        # This means the majority of the line was really in the next UTC day,
        # increment the line accordingly
        if mean_time > 24:
            mean_time -= 24
            increment_day = True

    # Calculate hour, minute, second
    h_m_s = [np.floor(mean_time)]
    h_m_s.append(np.floor((mean_time - h_m_s[-1]) * 60))
    h_m_s.append(np.floor((mean_time - h_m_s[-2] - h_m_s[-1] / 60.0) * 3600))

    if use_trim:
        valid = actual_valid

    return (
        h_m_s,
        increment_day,
        mean_path_km,
        mean_to_sensor_azimuth,
        mean_to_sensor_zenith,
        mean_to_sun_azimuth,
        mean_to_sun_zenith,
        mean_relative_azimuth,
        valid,
        to_sensor_zenith_lut_grid,
        to_sun_zenith_lut_grid,
        relative_azimuth_lut_grid,
    )


def get_metadata_from_loc(
    loc_file: str,
    lut_params: LUTConfig,
    trim_lines: int = 5,
    nodata_value: float = -9999,
    pressure_elevation: bool = False,
) -> (float, float, float, np.array):
    """Get metadata needed for complete runs from the location file (bands long, lat, elev).

    Args:
        loc_file: file name to pull data from
        lut_params: parameters to use to define lut grid
        trim_lines: number of lines to ignore at beginning and end of file (good if lines contain values that are
                    erroneous but not nodata
        nodata_value: value to ignore from location file
        pressure_elevation: retrieve pressure elevation (requires expanded ranges)

    :Returns:
        tuple containing:
            mean_latitude - mean latitude of good values from the location file
            mean_longitude - mean latitude of good values from the location file
            mean_elevation_km - mean ground estimate of good values from the location file
            elevation_lut_grid - the elevation look up table, based on globals and values from location file
    """

    loc_dataset = envi.open(envi_header(loc_file), loc_file)
    loc_data = loc_dataset.open_memmap(interleave="bsq", writable=False)
    valid = np.logical_not(np.any(loc_data == nodata_value, axis=0))

    use_trim = trim_lines != 0 and valid.shape[0] > trim_lines * 2
    if use_trim:
        valid[:trim_lines, :] = False
        valid[-trim_lines:, :] = False

    # Grab zensor position and orientation information
    mean_latitude = np.mean(loc_data[1, valid].flatten())
    mean_longitude = np.mean(-1 * loc_data[0, valid].flatten())
    mean_elevation_km = max(units.m_to_km(np.mean(loc_data[2, valid])), 0)

    # make elevation grid
    min_elev = units.m_to_km(np.min(loc_data[2, valid]))
    max_elev = units.m_to_km(np.max(loc_data[2, valid]))
    if pressure_elevation:
        min_elev = max(min_elev - 2, 0)
        max_elev += 2
    elevation_lut_grid = lut_params.get_grid(
        min_elev,
        max_elev,
        lut_params.elevation_spacing,
        lut_params.elevation_spacing_min,
    )

    return mean_latitude, mean_longitude, mean_elevation_km, elevation_lut_grid


def reassemble_cube(matching_indices: np.array, paths: Pathnames):
    """Copy over subsets of given files to new locations

    Args:
        matching_indices (np.array): indices to select from (y dimension) from source dataset
        paths (Pathnames): output file array set
    """

    logging.info(f"Reassemble {paths.rfl_subs_path}")
    input_ds = envi.open(envi_header(paths.surface_subs_files["base"]["rfl"]))
    header = input_ds.metadata.copy()
    header["lines"] = len(matching_indices)
    output_ds = envi.create_image(
        envi_header(paths.rfl_subs_path), header, ext="", force=True
    )
    output_mm = output_ds.open_memmap(interleave="bip", writable=True)

    for _st, surface_type in enumerate(list(paths.surface_config_paths.keys())):
        if np.sum(matching_indices == _st) > 0:
            input_ds = envi.open(
                envi_header(paths.surface_subs_files[surface_type]["rfl"])
            )
            output_mm[matching_indices == _st, ...] = input_ds.open_memmap(
                interleave="bip"
            ).copy()[:, 0, :]

    # TODO: only records reflectance uncertainties, could grab additional states (consistent between classes)
    logging.info(f"Reassemble {paths.uncert_subs_path}")
    input_ds = envi.open(envi_header(paths.surface_subs_files["base"]["uncert"]))
    rdn_ds = envi.open(envi_header(paths.surface_subs_files["base"]["rdn"]))
    header = input_ds.metadata.copy()
    header["lines"] = len(matching_indices)
    header["bands"] = rdn_ds.metadata["bands"]

    if "band names" in header.keys():
        header["band names"] = [
            input_ds.metadata["band names"][x] for x in range(int(header["bands"]))
        ]

    output_ds = envi.create_image(
        envi_header(paths.uncert_subs_path), header, ext="", force=True
    )
    output_mm = output_ds.open_memmap(interleave="bip", writable=True)

    for _st, surface_type in enumerate(list(paths.surface_config_paths.keys())):
        if np.sum(matching_indices == _st) > 0:
            input_ds = envi.open(
                envi_header(paths.surface_subs_files[surface_type]["uncert"])
            )
            output_mm[matching_indices == _st, ...] = input_ds.open_memmap(
                interleave="bip"
            )[:, :, : int(header["bands"])].copy()[:, 0, :]


def sensor_name_to_dt(sensor: str, fid: str):
    inversion_window_update = None
    if sensor == "ang":
        # parse flightline ID (AVIRIS-NG assumptions)
        dt = datetime.strptime(fid[3:], "%Y%m%dt%H%M%S")
    elif sensor == "av3":
        # parse flightline ID (AVIRIS-3 assumptions)
        dt = datetime.strptime(fid[3:], "%Y%m%dt%H%M%S")
        inversion_window_update = [[380.0, 1350.0], [1435, 1800.0], [1970.0, 2500.0]]
    elif sensor == "av5":
        # parse flightline ID (AVIRIS-5 assumptions)
        dt = datetime.strptime(fid[3:], "%Y%m%dt%H%M%S")
    elif sensor == "avcl":
        # parse flightline ID (AVIRIS-Classic assumptions)
        dt = datetime.strptime("20{}t000000".format(fid[1:7]), "%Y%m%dt%H%M%S")
    elif sensor == "emit":
        # parse flightline ID (EMIT assumptions)
        dt = datetime.strptime(fid[:19], "emit%Y%m%dt%H%M%S")
        INVERSION_WINDOWS = [[380.0, 1325.0], [1435, 1770.0], [1965.0, 2500.0]]
    elif sensor == "enmap":
        # parse flightline ID (EnMAP assumptions)
        dt = datetime.strptime(fid[:15], "%Y%m%dt%H%M%S")
    elif sensor == "hyp":
        # parse flightline ID (Hyperion assumptions)
        dt = datetime.strptime(fid[10:17], "%Y%j")
    elif sensor == "neon":
        # parse flightline ID (NEON assumptions)
        dt = datetime.strptime(fid, "NIS01_%Y%m%d_%H%M%S")
    elif sensor == "prism":
        # parse flightline ID (PRISM assumptions)
        dt = datetime.strptime(fid[3:], "%Y%m%dt%H%M%S")
    elif sensor == "prisma":
        # parse flightline ID (PRISMA assumptions)
        dt = datetime.strptime(fid, "%Y%m%d%H%M%S")
    elif sensor == "gao":
        # parse flightline ID (GAO/CAO assumptions)
        dt = datetime.strptime(fid[3:-5], "%Y%m%dt%H%M%S")
    elif sensor == "oci":
        # parse flightline ID (PACE OCI assumptions)
        dt = datetime.strptime(fid[9:24], "%Y%m%dT%H%M%S")
    elif sensor == "tanager":
        # parse flightline ID (Tanager assumptions)
        dt = datetime.strptime(fid[:15], "%Y%m%d_%H%M%S")
    elif sensor[:3] == "NA-":
        dt = datetime.strptime(sensor[3:], "%Y%m%d")
    else:
        raise ValueError(
            "Datetime object could not be obtained. Please check file name of input"
            " data."
        )
    return dt, inversion_window_update


def get_wavelengths(
    envi_file: str, wavelength_path: str = None
) -> (np.array, np.array):
    """Get wavelengths and FWHM from the header of an ENVI file

    Args:
        envi_file: path to the ENVI file to read wavelengths from
        wavelength_path: optional path to a file containing wavelengths and FWHM

    Returns:
        tuple containing:
            wl - array of wavelengths in nm
            fwhm - array of full width at half maximum in nm
    """

    # get radiance file, wavelengths, fwhm
    radiance_dataset = envi.open(envi_header(envi_file))
    wl_ds = np.array([float(w) for w in radiance_dataset.metadata["wavelength"]])
    if wavelength_path:
        if os.path.isfile(wavelength_path):
            chn, wl, fwhm = np.loadtxt(wavelength_path).T
            if len(chn) != len(wl_ds):
                raise ValueError(
                    "Number of channels in wavelength file do not match"
                    " wavelengths in radiance cube. Please adjust your wavelength file."
                )
        else:
            pass
    else:
        logging.info(
            "No wavelength file provided. Obtaining wavelength grid from ENVI header of radiance cube."
        )
        wl = wl_ds
        if "fwhm" in radiance_dataset.metadata:
            fwhm = np.array([float(f) for f in radiance_dataset.metadata["fwhm"]])
        elif "FWHM" in radiance_dataset.metadata:
            fwhm = np.array([float(f) for f in radiance_dataset.metadata["FWHM"]])
        else:
            fwhm = np.ones(wl.shape) * (wl[1] - wl[0])

    # Close out radiance dataset to avoid potential confusion
    del radiance_dataset

    # Convert to microns if needed
    if wl[0] > 100:
        logging.info("Wavelength units of nm inferred...converting to microns")
        wl = units.nm_to_micron(wl)
        fwhm = units.nm_to_micron(fwhm)

    return wl, fwhm


def write_wavelength_file(filename, wl, fwhm):
    """Write a wavelength file in isofit-expected format
    Units can be either nm or microns, but should be the same

    Args:
        filename: path to the file to write
        wl: array of wavelengths
        fwhm: array of full width at half maximum

    Returns:
        None
    """
    # write wavelength file
    wl_data = np.concatenate(
        [np.arange(len(wl))[:, np.newaxis], wl[:, np.newaxis], fwhm[:, np.newaxis]],
        axis=1,
    )
    np.savetxt(filename, wl_data, delimiter=" ")


def make_surface_config(
    paths: Pathnames,
    surface_category="multicomponent_surface",
    pressure_elevation=None,
    elevation_lut_grid=[],
    surface_mapping: dict = None,
    use_superpixels=False,
):
    """
    Constructs the surface component of the config
    Args:
        paths: Pathnames object with all key values passed from apply_oe
        surface_category: Base surface category
    Returns:
        surface_config_dict: Dictionary with all surface parameters and file
                             locations.
    """

    # Initialize config dict
    surface_config_dict = {
        "multi_surface_flag": False,
    }

    # Check to see if a classification file is being propogated
    # If so, use multisurface
    if paths.surface_class_working_path:
        surface_config_dict["Surfaces"] = {}

        if use_superpixels:
            surface_config_dict["surface_class_file"] = paths.surface_class_subs_path
        else:
            surface_config_dict["surface_class_file"] = paths.surface_class_working_path

        surface_config_dict["base_surface_class_file"] = (
            paths.surface_class_working_path
        )

        surface_config_dict["multi_surface_flag"] = True

        # Get the surface categories present.
        surface_classes_present = np.unique(
            envi.open(envi_header(paths.surface_class_working_path)).open_memmap(
                inteleave="bip"
            )
        )

        # Iterate through all classes present in class image
        for i in surface_classes_present:
            surface_category = SurfaceMapping[int(i)]
            # If surface_path given, use for all surfaces
            surface_path = paths.surface_working_paths[surface_category]

            # Set up "Surfaces" component of surface config
            surface_config_dict["Surfaces"][surface_category] = {
                "surface_int": int(i),
                "surface_file": surface_path,
                "surface_category": surface_category,
            }

    # Single surface run
    else:
        surface_config_dict["surface_file"] = paths.surface_working_paths[
            surface_category
        ]
        surface_config_dict["surface_category"] = surface_category

    return surface_config_dict
