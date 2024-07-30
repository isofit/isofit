#! /usr/bin/env python3
#
# Authors: Philip G. Brodrick and Niklas Bohn
#

import json
import logging
import os
import subprocess
from datetime import datetime
from os.path import abspath, exists, join, split
from shutil import copyfile
from sys import platform
from typing import List

import netCDF4 as nc
import numpy as np
from scipy.io import loadmat
from sklearn import mixture
from spectral.io import envi

from isofit.core import isofit
from isofit.core.common import (
    envi_header,
    expand_path,
    json_load_ascii,
    resample_spectrum,
)
from isofit.utils import surface_model


class Pathnames:
    """Class to determine and hold the large number of relative and absolute paths that are needed for isofit and
    MODTRAN configuration files.

    Args:
        args: an argparse Namespace object with all inputs
    """

    def __init__(self, args):
        # Determine FID based on sensor name
        if args.sensor == "ang":
            self.fid = split(args.input_radiance)[-1][:18]
        elif args.sensor == "av3":
            self.fid = split(args.input_radiance)[-1][:18]
        elif args.sensor == "avcl":
            self.fid = split(args.input_radiance)[-1][:16]
        elif args.sensor == "emit":
            self.fid = split(args.input_radiance)[-1][:19]
        elif args.sensor == "enmap":
            self.fid = split(args.input_radiance)[-1].split("_")[5]
        elif args.sensor == "hyp":
            self.fid = split(args.input_radiance)[-1][:22]
        elif args.sensor == "neon":
            self.fid = split(args.input_radiance)[-1][:21]
        elif args.sensor == "prism":
            self.fid = split(args.input_radiance)[-1][:18]
        elif args.sensor == "prisma":
            self.fid = args.input_radiance.split("/")[-1].split("_")[1]
        elif args.sensor == "gao":
            self.fid = split(args.input_radiance)[-1][:23]
        elif args.sensor[:3] == "NA-":
            self.fid = os.path.splitext(os.path.basename(args.input_radiance))[0]

        logging.info("Flightline ID: %s" % self.fid)

        # Names from inputs
        self.aerosol_climatology = args.aerosol_climatology_path
        self.input_radiance_file = args.input_radiance
        self.input_loc_file = args.input_loc
        self.input_obs_file = args.input_obs
        self.working_directory = abspath(args.working_directory)

        self.full_lut_directory = abspath(join(self.working_directory, "lut_full/"))

        self.surface_path = args.surface_path

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
        self.surface_working_path = abspath(join(self.data_directory, "surface.mat"))

        if args.copy_input_files is True:
            self.radiance_working_path = abspath(
                join(self.input_data_directory, rdn_fname)
            )
            self.obs_working_path = abspath(
                join(self.input_data_directory, self.fid + "_obs")
            )
            self.loc_working_path = abspath(
                join(self.input_data_directory, self.fid + "_loc")
            )
        else:
            self.radiance_working_path = abspath(self.input_radiance_file)
            self.obs_working_path = abspath(self.input_obs_file)
            self.loc_working_path = abspath(self.input_loc_file)

        if args.channelized_uncertainty_path:
            self.input_channelized_uncertainty_path = args.channelized_uncertainty_path
        else:
            self.input_channelized_uncertainty_path = os.getenv(
                "ISOFIT_CHANNELIZED_UNCERTAINTY"
            )

        self.channelized_uncertainty_working_path = abspath(
            join(self.data_directory, "channelized_uncertainty.txt")
        )

        if args.model_discrepancy_path:
            self.input_model_discrepancy_path = args.model_discrepancy_path
        else:
            self.input_model_discrepancy_path = None

        self.model_discrepancy_working_path = abspath(
            join(self.data_directory, "model_discrepancy.mat")
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

        if args.modtran_path:
            self.modtran_path = args.modtran_path
        else:
            self.modtran_path = os.getenv("MODTRAN_DIR")

        self.sixs_path = os.getenv("SIXS_DIR")

        if os.getenv("ISOFIT_DIR"):
            self.isofit_path = os.getenv("ISOFIT_DIR")
        else:
            # isofit file should live at isofit/isofit/core/isofit.py
            self.isofit_path = os.path.dirname(
                os.path.dirname(os.path.dirname(isofit.__file__))
            )

        if args.sensor == "avcl":
            self.noise_path = join(self.isofit_path, "data", "avirisc_noise.txt")
        elif args.sensor == "emit":
            self.noise_path = join(self.isofit_path, "data", "emit_noise.txt")
            if self.input_channelized_uncertainty_path is None:
                self.input_channelized_uncertainty_path = join(
                    self.isofit_path, "data", "emit_osf_uncertainty.txt"
                )
            if self.input_model_discrepancy_path is None:
                self.input_model_discrepancy_path = join(
                    self.isofit_path, "data", "emit_model_discrepancy.mat"
                )
        elif args.sensor == "av3":
            self.noise_path = None
            logging.info("no noise path found, proceeding without")
            if self.input_channelized_uncertainty_path is None:
                self.input_channelized_uncertainty_path = join(
                    self.isofit_path, "data", "av3_osf_uncertainty.txt"
                )
        else:
            self.noise_path = None
            logging.info("no noise path found, proceeding without")
            # quit()

        self.earth_sun_distance_path = abspath(
            join(self.isofit_path, "data", "earth_sun_distance.txt")
        )
        self.irradiance_file = abspath(
            join(
                self.isofit_path,
                "examples",
                "20151026_SantaMonica",
                "data",
                "prism_optimized_irr.dat",
            )
        )

        self.aerosol_tpl_path = join(self.isofit_path, "data", "aerosol_template.json")
        self.rdn_factors_path = None
        if args.rdn_factors_path is not None:
            self.rdn_factors_path = abspath(args.rdn_factors_path)

        self.ray_temp_dir = args.ray_temp_dir

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
            (self.surface_path, self.surface_working_path, False),
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
    """

    def __init__(self, lut_config_file: str = None, emulator: bool = False):
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
        self.h2o_min = 0.05

        # Set defaults, will override based on settings
        # Units of g / m2
        self.h2o_range = [0.05, 5]

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
        self.aerosol_0_range = [0.001, 1]
        self.aerosol_1_range = [0.001, 1]
        self.aerosol_2_range = [0.001, 1]
        self.aot_550_range = [0.001, 1]

        self.aot_550_spacing = 0
        self.aot_550_spacing_min = 0

        # overwrite anything that comes in from the config file
        if lut_config_file is not None:
            for key in lut_config:
                if key in self.__dict__:
                    setattr(self, key, lut_config[key])

        if emulator and os.path.splitext(emulator)[1] != ".jld2":
            self.aot_550_range = self.aerosol_2_range
            self.aot_550_spacing = self.aerosol_2_spacing
            self.aot_550_spacing_min = self.aerosol_2_spacing_min
            self.aerosol_2_spacing = 0

    def get_grid_with_data(
        self, data_input: np.array, spacing: float, min_spacing: float
    ):
        min_val = np.min(data_input)
        max_val = np.max(data_input)
        return get_grid(min_val, max_val, spacing, min_spacing)

    def get_grid(
        self, minval: float, maxval: float, spacing: float, min_spacing: float
    ):
        if spacing == 0:
            logging.debug("Grid spacing set at 0, using no grid.")
            return None
        num_gridpoints = int(np.ceil((maxval - minval) / spacing)) + 1

        grid = np.linspace(minval, maxval, num_gridpoints)

        if min_spacing > 0.0001:
            grid = np.round(grid, 4)
        if len(grid) == 1:
            logging.debug(
                f"Grid spacing is 0, which is less than {min_spacing}.  No grid used"
            )
            return None
        elif np.abs(grid[1] - grid[0]) < min_spacing:
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


def get_grid(minval: float, maxval: float, spacing: float, min_spacing: float):
    if spacing == 0:
        logging.debug("Grid spacing set at 0, using no grid.")
        return None

    num_gridpoints = int(np.ceil((maxval - minval) / spacing)) + 1

    grid = np.linspace(minval, maxval, num_gridpoints)

    if min_spacing > 0.0001:
        grid = np.round(grid, 4)

    if len(grid) == 1:
        logging.debug(
            f"Grid spacing is 0, which is less than {min_spacing}.  No grid used"
        )
        return None
    elif np.abs(grid[1] - grid[0]) < min_spacing:
        logging.debug(
            f"Grid spacing is {grid[1] - grid[0]}, which is less than {min_spacing}. "
            " No grid used"
        )
        return None
    else:
        return grid


def check_surface_model(surface_path: str, wl: np.array, paths: Pathnames) -> str:
    """
    Checks and rebuilds surface model if needed.

    Args:
        surface_path: path to surface model or config dict
        wl: instrument center wavelengths
        paths: object containing references to all relevant file locations
    """
    if os.path.isfile(surface_path):
        if surface_path.endswith(".mat"):
            # check wavelength grid of surface model if provided
            model_dict = loadmat(surface_path)
            wl_surface = model_dict["wl"][0]
            if len(wl_surface) != len(wl):
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
            return surface_path
        elif surface_path.endswith(".json"):
            logging.info(
                "No surface model provided. Build new one using given config file."
            )
            surface_model(config_path=surface_path)
            configdir, _ = os.path.split(os.path.abspath(surface_path))
            config = json_load_ascii(surface_path, shell_replace=True)
            return expand_path(configdir, config["output_model_file"])
        else:
            raise FileNotFoundError(
                "Unrecognized format of surface file. Please provide either a .mat model or a .json config dict."
            )
    else:
        raise FileNotFoundError(
            "No surface file found. Please provide either a .mat model or a .json config dict."
        )


def build_presolve_config(
    paths: Pathnames,
    h2o_lut_grid: np.array,
    n_cores: int = -1,
    use_emp_line: bool = False,
    surface_category="multicomponent_surface",
    emulator_base: str = None,
    uncorrelated_radiometric_uncertainty: float = 0.0,
    segmentation_size: int = 400,
    debug: bool = False,
    inversion_windows=[[350.0, 1360.0], [1410, 1800.0], [1970.0, 2500.0]],
    prebuilt_lut_path: str = None,
) -> None:
    """Write an isofit config file for a presolve, with limited info.

    Args:
        paths: object containing references to all relevant file locations
        h2o_lut_grid: the water vapor look up table grid isofit should use for this solve
        n_cores: number of cores to use in processing
        use_emp_line: flag whether or not to set up for the empirical line estimation
        surface_category: type of surface to use
        emulator_base: the basename of the emulator, if used
        uncorrelated_radiometric_uncertainty: uncorrelated radiometric uncertainty parameter for isofit
        segmentation_size: image segmentation size if empirical line is used
        debug: flag to enable debug_mode in the config.implementation
        lut_path: lut path to use; if none, presolve config will create a new file
    """

    # Determine number of spectra included in each retrieval.  If we are
    # operating on segments, this will average down instrument noise
    if use_emp_line:
        spectra_per_inversion = segmentation_size
    else:
        spectra_per_inversion = 1

    if emulator_base is None:
        engine_name = "modtran"
    elif emulator_base.endswith(".jld2"):
        engine_name = "KernelFlowsGP"
    else:
        engine_name = "sRTMnet"

    if prebuilt_lut_path is None:
        lut_path = join(paths.lut_h2o_directory, "lut.nc")
    else:
        lut_path = prebuilt_lut_path

    # set up specific presolve LUT grid
    lut_grid = {"H2OSTR": [float(x) for x in h2o_lut_grid]}
    if emulator_base is not None and os.path.splitext(emulator_base)[1] == ".jld2":
        from isofit.radiative_transfer.kernel_flows import bounds_check

        bounds_check(lut_grid, emulator_base, modify=True)

    radiative_transfer_config = {
        "radiative_transfer_engines": {
            "vswir": {
                "engine_name": engine_name,
                "lut_path": lut_path,
                "sim_path": paths.lut_h2o_directory,
                "template_file": paths.h2o_template_path,
                "lut_names": {"H2OSTR": get_lut_subset(h2o_lut_grid)},
                "statevector_names": ["H2OSTR"],
            }
        },
        "statevector": {
            "H2OSTR": {
                "bounds": [
                    float(np.min(lut_grid["H2OSTR"])),
                    float(np.max(lut_grid["H2OSTR"])),
                ],
                "scale": 0.01,
                "init": np.percentile(lut_grid["H2OSTR"], 25),
                "prior_sigma": 100.0,
                "prior_mean": 1.5,
            }
        },
        "lut_grid": lut_grid,
        "unknowns": {"H2O_ABSCO": 0.0},
    }

    if emulator_base is not None:
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "emulator_file"
        ] = abspath(emulator_base)
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "emulator_aux_file"
        ] = abspath(os.path.splitext(emulator_base)[0] + "_aux.npz")
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "interpolator_base_path"
        ] = abspath(
            os.path.join(
                paths.lut_h2o_directory, os.path.basename(emulator_base) + "_vi"
            )
        )
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "earth_sun_distance_file"
        ] = paths.earth_sun_distance_path
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "irradiance_file"
        ] = paths.irradiance_file
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "engine_base_dir"
        ] = paths.sixs_path

    else:
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "engine_base_dir"
        ] = paths.modtran_path

    # make isofit configuration
    isofit_config_h2o = {
        "ISOFIT_base": paths.isofit_path,
        "output": {"estimated_state_file": paths.h2o_subs_path},
        "input": {},
        "forward_model": {
            "instrument": {
                "wavelength_file": paths.wavelength_path,
                "integrations": spectra_per_inversion,
                "unknowns": {
                    "uncorrelated_radiometric_uncertainty": uncorrelated_radiometric_uncertainty
                },
            },
            "surface": {
                "surface_category": surface_category,
                "surface_file": paths.surface_working_path,
                "select_on_init": True,
            },
            "radiative_transfer": radiative_transfer_config,
        },
        "implementation": {
            "ray_temp_dir": paths.ray_temp_dir,
            "inversion": {"windows": inversion_windows},
            "n_cores": n_cores,
            "debug_mode": debug,
        },
    }

    if paths.input_channelized_uncertainty_path is not None:
        isofit_config_h2o["forward_model"]["instrument"]["unknowns"][
            "channelized_radiometric_uncertainty_file"
        ] = paths.channelized_uncertainty_working_path

    if paths.input_model_discrepancy_path is not None:
        isofit_config_h2o["forward_model"][
            "model_discrepancy_file"
        ] = paths.model_discrepancy_working_path

    if paths.noise_path is not None:
        isofit_config_h2o["forward_model"]["instrument"][
            "parametric_noise_file"
        ] = paths.noise_path
    else:
        isofit_config_h2o["forward_model"]["instrument"]["SNR"] = 1000

    if paths.rdn_factors_path:
        isofit_config_h2o["input"][
            "radiometry_correction_file"
        ] = paths.rdn_factors_path

    if use_emp_line:
        isofit_config_h2o["input"]["measured_radiance_file"] = paths.rdn_subs_path
        isofit_config_h2o["input"]["loc_file"] = paths.loc_subs_path
        isofit_config_h2o["input"]["obs_file"] = paths.obs_subs_path
    else:
        isofit_config_h2o["input"][
            "measured_radiance_file"
        ] = paths.radiance_working_path
        isofit_config_h2o["input"]["loc_file"] = paths.loc_working_path
        isofit_config_h2o["input"]["obs_file"] = paths.obs_working_path

    # write modtran_template
    with open(paths.h2o_config_path, "w") as fout:
        fout.write(
            json.dumps(isofit_config_h2o, cls=SerialEncoder, indent=4, sort_keys=True)
        )


def build_main_config(
    paths: Pathnames,
    lut_params: LUTConfig,
    h2o_lut_grid: np.array = None,
    elevation_lut_grid: np.array = None,
    to_sensor_zenith_lut_grid: np.array = None,
    to_sun_zenith_lut_grid: np.array = None,
    relative_azimuth_lut_grid: np.array = None,
    mean_latitude: float = None,
    mean_longitude: float = None,
    dt: datetime = None,
    use_emp_line: bool = True,
    n_cores: int = -1,
    surface_category="multicomponent_surface",
    emulator_base: str = None,
    uncorrelated_radiometric_uncertainty: float = 0.0,
    multiple_restarts: bool = False,
    segmentation_size=400,
    pressure_elevation: bool = False,
    debug: bool = False,
    inversion_windows=[[350.0, 1360.0], [1410, 1800.0], [1970.0, 2500.0]],
    prebuilt_lut_path: str = None,
) -> None:
    """Write an isofit config file for the main solve, using the specified pathnames and all given info

    Args:
        paths:                                object containing references to all relevant file locations
        lut_params:                           configuration parameters for the lut grid
        h2o_lut_grid:                         the water vapor look up table grid isofit should use for this solve
        elevation_lut_grid:                   the ground elevation look up table grid isofit should use for this solve
        to_sensor_zenith_lut_grid:            the to-sensor zenith angle look up table grid isofit should use for this
                                              solve
        to_sun_zenith_lut_grid:               the to-sun zenith angle look up table grid isofit should use for this
                                              solve
        relative_azimuth_lut_grid:            the relative to-sun azimuth angle look up table grid isofit should use for
                                              this solve
        mean_latitude:                        the latitude isofit should use for this solve
        mean_longitude:                       the longitude isofit should use for this solve
        dt:                                   the datetime object corresponding to this flightline to use for this solve
        use_emp_line:                         flag whether or not to set up for the empirical line estimation
        n_cores:                              the number of cores to use during processing
        surface_category:                     type of surface to use
        emulator_base:                        the basename of the emulator, if used
        uncorrelated_radiometric_uncertainty: uncorrelated radiometric uncertainty parameter for isofit
        multiple_restarts:                    if true, use multiple restarts
        segmentation_size:                    image segmentation size if empirical line is used
        pressure_elevation:                   if true, retrieve pressure elevation
        debug:                                if true, run ISOFIT in debug mode
    """

    # Determine number of spectra included in each retrieval.  If we are
    # operating on segments, this will average down instrument noise
    if use_emp_line:
        spectra_per_inversion = segmentation_size
    else:
        spectra_per_inversion = 1

    if prebuilt_lut_path is None:
        lut_path = join(paths.full_lut_directory, "lut.nc")
    else:
        lut_path = abspath(prebuilt_lut_path)

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
                "sim_path": paths.full_lut_directory,
                "lut_path": lut_path,
                "aerosol_template_file": paths.aerosol_tpl_path,
                "template_file": paths.modtran_template_path,
                # lut_names - populated below
                # statevector_names - populated below
            }
        },
        "statevector": {},
        "lut_grid": {},
        "unknowns": {"H2O_ABSCO": 0.0},
    }

    if emulator_base is not None:
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "emulator_file"
        ] = abspath(emulator_base)
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "emulator_aux_file"
        ] = abspath(os.path.splitext(emulator_base)[0] + "_aux.npz")
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "interpolator_base_path"
        ] = abspath(
            os.path.join(
                paths.full_lut_directory,
                os.path.basename(os.path.splitext(emulator_base)[0]) + "_vi",
            )
        )
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "earth_sun_distance_file"
        ] = paths.earth_sun_distance_path
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "irradiance_file"
        ] = paths.irradiance_file
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "engine_base_dir"
        ] = paths.sixs_path

    else:
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "engine_base_dir"
        ] = paths.modtran_path

    # add aerosol elements from climatology
    aerosol_state_vector, aerosol_lut_grid, aerosol_model_path = load_climatology(
        paths.aerosol_climatology,
        mean_latitude,
        mean_longitude,
        dt,
        paths.isofit_path,
        lut_params=lut_params,
    )
    radiative_transfer_config["radiative_transfer_engines"]["vswir"][
        "aerosol_model_file"
    ] = aerosol_model_path

    if prebuilt_lut_path is None:
        if h2o_lut_grid is not None and len(h2o_lut_grid) > 1:
            radiative_transfer_config["lut_grid"]["H2OSTR"] = h2o_lut_grid.tolist()
        if elevation_lut_grid is not None and len(elevation_lut_grid) > 1:
            radiative_transfer_config["lut_grid"][
                "surface_elevation_km"
            ] = elevation_lut_grid.tolist()
        if to_sensor_zenith_lut_grid is not None and len(to_sensor_zenith_lut_grid) > 1:
            radiative_transfer_config["lut_grid"][
                "observer_zenith"
            ] = to_sensor_zenith_lut_grid.tolist()
        if to_sun_zenith_lut_grid is not None and len(to_sun_zenith_lut_grid) > 1:
            radiative_transfer_config["lut_grid"][
                "solar_zenith"
            ] = to_sun_zenith_lut_grid.tolist()
        if relative_azimuth_lut_grid is not None and len(relative_azimuth_lut_grid) > 1:
            radiative_transfer_config["lut_grid"][
                "relative_azimuth"
            ] = relative_azimuth_lut_grid.tolist()
        radiative_transfer_config["lut_grid"].update(aerosol_lut_grid)

    rtc_ln = {}
    for key in radiative_transfer_config["lut_grid"].keys():
        rtc_ln[key] = None
    radiative_transfer_config["radiative_transfer_engines"]["vswir"][
        "lut_names"
    ] = rtc_ln

    if emulator_base is not None and os.path.splitext(emulator_base)[1] == ".jld2":
        from isofit.radiative_transfer.kernel_flows import bounds_check

        bounds_check(radiative_transfer_config["lut_grid"], emulator_base, modify=True)
        # modify so we set the statevector appropriately
        if "H2OSTR" in radiative_transfer_config["lut_grid"]:
            h2o_lut_grid = np.array(radiative_transfer_config["lut_grid"]["H2OSTR"])
        if "surface_elevation_km" in radiative_transfer_config["lut_grid"]:
            elevation_lut_grid = np.array(
                radiative_transfer_config["lut_grid"]["surface_elevation_km"]
            )

    if prebuilt_lut_path is not None:
        ncds = nc.Dataset(prebuilt_lut_path, "r")

        radiative_transfer_config["radiative_transfer_engines"]["vswir"]["lut_names"][
            "H2OSTR"
        ] = get_lut_subset(h2o_lut_grid)
        radiative_transfer_config["radiative_transfer_engines"]["vswir"]["lut_names"][
            "surface_elevation_km"
        ] = get_lut_subset(elevation_lut_grid)
        radiative_transfer_config["radiative_transfer_engines"]["vswir"]["lut_names"][
            "observer_zenith"
        ] = get_lut_subset(to_sensor_zenith_lut_grid)
        radiative_transfer_config["radiative_transfer_engines"]["vswir"]["lut_names"][
            "solar_zenith"
        ] = get_lut_subset(to_sun_zenith_lut_grid)
        radiative_transfer_config["radiative_transfer_engines"]["vswir"]["lut_names"][
            "relative_azimuth"
        ] = get_lut_subset(relative_azimuth_lut_grid)
        for key in aerosol_lut_grid.keys():
            radiative_transfer_config["radiative_transfer_engines"]["vswir"][
                "lut_names"
            ][key] = get_lut_subset(aerosol_lut_grid[key])

        rm_keys = []
        for key, item in radiative_transfer_config["radiative_transfer_engines"][
            "vswir"
        ]["lut_names"].items():
            if key not in ncds.variables:
                logging.warning(
                    f"Key {key} not found in prebuilt LUT, removing it from LUT.  Spacing would have been: {item}"
                )
                rm_keys.append(key)
        for key in rm_keys:
            del radiative_transfer_config["radiative_transfer_engines"]["vswir"][
                "lut_names"
            ][key]

    # Now do statevector
    if h2o_lut_grid is not None:
        radiative_transfer_config["statevector"]["H2OSTR"] = {
            "bounds": [h2o_lut_grid[0], h2o_lut_grid[-1]],
            "scale": 1,
            "init": (h2o_lut_grid[1] + h2o_lut_grid[-1]) / 2.0,
            "prior_sigma": 100.0,
            "prior_mean": (h2o_lut_grid[1] + h2o_lut_grid[-1]) / 2.0,
        }

    if pressure_elevation:
        radiative_transfer_config["statevector"]["surface_elevation_km"] = {
            "bounds": [elevation_lut_grid[0], elevation_lut_grid[-1]],
            "scale": 100,
            "init": (elevation_lut_grid[0] + elevation_lut_grid[-1]) / 2.0,
            "prior_sigma": 1000.0,
            "prior_mean": (elevation_lut_grid[0] + elevation_lut_grid[-1]) / 2.0,
        }
    radiative_transfer_config["statevector"].update(aerosol_state_vector)

    # MODTRAN should know about our whole LUT grid and all of our statevectors, so copy them in
    radiative_transfer_config["radiative_transfer_engines"]["vswir"][
        "statevector_names"
    ] = list(radiative_transfer_config["statevector"].keys())

    # make isofit configuration
    isofit_config_modtran = {
        "ISOFIT_base": paths.isofit_path,
        "input": {},
        "output": {},
        "forward_model": {
            "instrument": {
                "wavelength_file": paths.wavelength_path,
                "integrations": spectra_per_inversion,
                "unknowns": {
                    "uncorrelated_radiometric_uncertainty": uncorrelated_radiometric_uncertainty
                },
            },
            "surface": {
                "surface_file": paths.surface_working_path,
                "surface_category": surface_category,
                "select_on_init": True,
            },
            "radiative_transfer": radiative_transfer_config,
        },
        "implementation": {
            "ray_temp_dir": paths.ray_temp_dir,
            "inversion": {"windows": inversion_windows},
            "n_cores": n_cores,
            "debug_mode": debug,
        },
    }

    if use_emp_line:
        isofit_config_modtran["input"]["measured_radiance_file"] = paths.rdn_subs_path
        isofit_config_modtran["input"]["loc_file"] = paths.loc_subs_path
        isofit_config_modtran["input"]["obs_file"] = paths.obs_subs_path
        isofit_config_modtran["output"]["estimated_state_file"] = paths.state_subs_path
        isofit_config_modtran["output"][
            "posterior_uncertainty_file"
        ] = paths.uncert_subs_path
        isofit_config_modtran["output"][
            "estimated_reflectance_file"
        ] = paths.rfl_subs_path
        isofit_config_modtran["output"][
            "atmospheric_coefficients_file"
        ] = paths.atm_coeff_path
    else:
        isofit_config_modtran["input"][
            "measured_radiance_file"
        ] = paths.radiance_working_path
        isofit_config_modtran["input"]["loc_file"] = paths.loc_working_path
        isofit_config_modtran["input"]["obs_file"] = paths.obs_working_path
        isofit_config_modtran["output"][
            "posterior_uncertainty_file"
        ] = paths.uncert_working_path
        isofit_config_modtran["output"][
            "estimated_reflectance_file"
        ] = paths.rfl_working_path
        isofit_config_modtran["output"][
            "estimated_state_file"
        ] = paths.state_working_path

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
                        "CO2MX": 410.0,
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
    isofit_path: str,
    lut_params: LUTConfig,
):
    """Load climatology data, based on location and configuration

    Args:
        config_path: path to the base configuration directory for isofit
        latitude: latitude to set for the segment (mean of acquisition suggested)
        longitude: latitude to set for the segment (mean of acquisition suggested)
        acquisition_datetime: datetime to use for the segment( mean of acquisition suggested)
        isofit_path: base path to isofit installation (needed for data path references)
        lut_params: parameters to use to define lut grid

    :Returns
        tuple containing:
            aerosol_state_vector - A dictionary that defines the aerosol state vectors for isofit
            aerosol_lut_grid - A dictionary of the aerosol lookup table (lut) grid to be explored
            aerosol_model_path - A path to the location of the aerosol model to use with MODTRAN.

    """

    aerosol_model_path = os.path.join(isofit_path, "data", "aerosol_model.txt")
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
        aerosol_lut = get_grid(
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

    aot_550_lut = get_grid(
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
        wl = wl * 1000
        fwhm = fwhm * 1000

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

    path_km = obs[:, :, 0] / 1000.0
    to_sensor_azimuth = obs[:, :, 1]
    to_sensor_zenith = obs[:, :, 2]
    to_sun_azimuth = obs[:, :, 3]
    to_sun_zenith = obs[:, :, 4]
    time = obs[:, :, 9]

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
    mean_elevation_km = np.mean(loc_data[2, valid]) / 1000.0

    # make elevation grid
    min_elev = np.min(loc_data[2, valid]) / 1000.0
    max_elev = np.max(loc_data[2, valid]) / 1000.0
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
