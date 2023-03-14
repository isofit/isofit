#! /usr/bin/env python3
#
# Authors: Philip G. Brodrick and Niklas Bohn
#

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
from datetime import datetime
from shutil import copyfile
from sys import platform
from typing import List

import numpy as np
import utm
from osgeo import gdal
from sklearn import mixture
from spectral.io import envi

from isofit.core import isofit
from isofit.core.common import envi_header, resample_spectrum


class Pathnames:
    """
    Class to determine and hold the large number of relative and absolute paths that are needed for isofit and MODTRAN
    configuration files.

    Args:
        opt: dictionary of general options
        gip: dictionary of general inversion parameters
        args: an argparse Namespace object with all inputs
        fid: string of instrument specific flight identification number
    """

    def __init__(self, opt: dict, gip: dict, args: argparse.Namespace, fid: str):
        # Names from inputs
        self.aerosol_climatology = gip["filepaths"]["aerosol_climatology_path"]
        self.input_radiance_file = args.input_radiance
        self.input_loc_file = args.input_loc
        self.input_obs_file = args.input_obs
        self.working_directory = os.path.abspath(args.working_directory)
        self.fid = fid

        self.lut_modtran_directory = os.path.abspath(
            os.path.join(self.working_directory, "lut_full", "")
        )

        self.surface_lut_paths = {}

        # set up some sub-directories
        self.lut_h2o_directory = os.path.abspath(
            os.path.join(self.working_directory, "lut_h2o", "")
        )
        self.config_directory = os.path.abspath(
            os.path.join(self.working_directory, "config", "")
        )
        self.data_directory = os.path.abspath(
            os.path.join(self.working_directory, "data", "")
        )
        self.input_data_directory = os.path.abspath(
            os.path.join(self.working_directory, "input", "")
        )
        self.output_directory = os.path.abspath(
            os.path.join(self.working_directory, "output", "")
        )

        # get surface model, rebuild if needed
        if gip["filepaths"]["surface_path"]:
            self.surface_path = gip["filepaths"]["surface_path"]
        elif os.getenv("ISOFIT_SURFACE_MODEL"):
            self.surface_path = os.getenv("ISOFIT_SURFACE_MODEL")
        else:
            self.surface_path = os.path.abspath(
                os.path.join(self.data_directory, "surface.mat")
            )

        # define all output names
        rdn_fname = self.fid + "_rdn"
        self.rfl_working_path = os.path.abspath(
            os.path.join(self.output_directory, rdn_fname.replace("_rdn", "_rfl"))
        )
        self.uncert_working_path = os.path.abspath(
            os.path.join(self.output_directory, rdn_fname.replace("_rdn", "_uncert"))
        )
        self.lbl_working_path = os.path.abspath(
            os.path.join(self.output_directory, rdn_fname.replace("_rdn", "_lbl"))
        )
        self.state_working_path = os.path.abspath(
            os.path.join(self.output_directory, rdn_fname.replace("_rdn", "_state"))
        )
        self.surface_working_path = os.path.abspath(
            os.path.join(self.data_directory, "surface.mat")
        )

        if opt["copy_input_files"] is True:
            self.radiance_working_path = os.path.abspath(
                os.path.join(self.input_data_directory, rdn_fname)
            )
            self.obs_working_path = os.path.abspath(
                os.path.join(self.input_data_directory, self.fid + "_obs")
            )
            self.loc_working_path = os.path.abspath(
                os.path.join(self.input_data_directory, self.fid + "_loc")
            )
        else:
            self.radiance_working_path = os.path.abspath(self.input_radiance_file)
            self.obs_working_path = os.path.abspath(self.input_obs_file)
            self.loc_working_path = os.path.abspath(self.input_loc_file)

        if gip["filepaths"]["channelized_uncertainty_path"]:
            self.input_channelized_uncertainty_path = gip["filepaths"][
                "channelized_uncertainty_path"
            ]
        else:
            self.input_channelized_uncertainty_path = os.getenv(
                "ISOFIT_CHANNELIZED_UNCERTAINTY"
            )

        self.channelized_uncertainty_working_path = os.path.abspath(
            os.path.join(self.data_directory, "channelized_uncertainty.txt")
        )

        if gip["filepaths"]["model_discrepancy_path"]:
            self.input_model_discrepancy_path = gip["filepaths"][
                "model_discrepancy_path"
            ]
        else:
            self.input_model_discrepancy_path = None

        self.model_discrepancy_working_path = os.path.abspath(
            os.path.join(self.data_directory, "model_discrepancy.mat")
        )

        self.rdn_subs_path = os.path.abspath(
            os.path.join(self.input_data_directory, self.fid + "_subs_rdn")
        )
        self.obs_subs_path = os.path.abspath(
            os.path.join(self.input_data_directory, self.fid + "_subs_obs")
        )
        self.loc_subs_path = os.path.abspath(
            os.path.join(self.input_data_directory, self.fid + "_subs_loc")
        )
        self.rfl_subs_path = os.path.abspath(
            os.path.join(self.output_directory, self.fid + "_subs_rfl")
        )
        self.atm_coeff_path = os.path.abspath(
            os.path.join(self.output_directory, self.fid + "_subs_atm")
        )
        self.state_subs_path = os.path.abspath(
            os.path.join(self.output_directory, self.fid + "_subs_state")
        )
        self.uncert_subs_path = os.path.abspath(
            os.path.join(self.output_directory, self.fid + "_subs_uncert")
        )
        self.h2o_subs_path = os.path.abspath(
            os.path.join(self.output_directory, self.fid + "_subs_h2o")
        )
        self.class_subs_path = os.path.abspath(
            os.path.join(self.output_directory, self.fid + "_subs_classes")
        )
        self.surface_subs_files = {}

        self.wavelength_path = os.path.abspath(
            os.path.join(self.data_directory, "wavelengths.txt")
        )

        self.modtran_template_path = os.path.abspath(
            os.path.join(self.config_directory, self.fid + "_modtran_tpl.json")
        )
        self.h2o_template_path = os.path.abspath(
            os.path.join(self.config_directory, self.fid + "_h2o_tpl.json")
        )

        self.modtran_config_path = os.path.abspath(
            os.path.join(self.config_directory, self.fid + "_modtran.json")
        )
        self.h2o_config_path = os.path.abspath(
            os.path.join(self.config_directory, self.fid + "_h2o.json")
        )

        self.surface_config_paths = {}

        if gip["filepaths"]["modtran_path"]:
            self.modtran_path = gip["filepaths"]["modtran_path"]
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

        if opt["sensor"] == "ang":
            self.noise_path = os.path.join(
                self.isofit_path, "data", "avirisng_noise.txt"
            )
        elif opt["sensor"] == "avcl":
            self.noise_path = os.path.join(
                self.isofit_path, "data", "avirisc_noise.txt"
            )
        elif opt["sensor"] == "emit":
            self.noise_path = os.path.join(self.isofit_path, "data", "emit_noise.txt")
            if self.input_channelized_uncertainty_path is None:
                self.input_channelized_uncertainty_path = os.path.join(
                    self.isofit_path, "data", "emit_osf_uncertainty.txt"
                )
            if self.input_model_discrepancy_path is None:
                self.input_model_discrepancy_path = os.path.join(
                    self.isofit_path, "data", "emit_model_discrepancy.mat"
                )
        else:
            self.noise_path = None
            logging.info("No noise path found, proceeding without.")

        self.earth_sun_distance_path = os.path.abspath(
            os.path.join(self.isofit_path, "data", "earth_sun_distance.txt")
        )
        self.irradiance_file = os.path.abspath(
            os.path.join(
                self.isofit_path,
                "examples",
                "20151026_SantaMonica",
                "data",
                "prism_optimized_irr.dat",
            )
        )

        self.aerosol_tpl_path = os.path.join(
            self.isofit_path, "data", "aerosol_template.json"
        )
        self.rdn_factors_path = None

        if gip["filepaths"]["rdn_factors_path"] is not None:
            self.rdn_factors_path = os.path.abspath(
                gip["filepaths"]["rdn_factors_path"]
            )

        self.ray_temp_dir = (
            "/tmp/ray" if not opt["ray_temp_dir"] else opt["ray_temp_dir"]
        )

    def make_directories(self, surface_types: list):
        """
        Build required subdirectories inside working_directory.

        Args:
            surface_types: list of optional surface types
        """
        for dpath in [
            self.working_directory,
            self.lut_h2o_directory,
            self.lut_modtran_directory,
            self.config_directory,
            self.data_directory,
            self.input_data_directory,
            self.output_directory,
        ]:
            if not os.path.exists(dpath):
                os.mkdir(dpath)

        # build directories for storing surface-specific LUTs and interpolators
        for surface_type in surface_types:
            if not os.path.exists(self.surface_lut_paths[surface_type]):
                os.mkdir(self.surface_lut_paths[surface_type])

    def stage_files(self):
        """
        Stage data files by copying into working directory.
        """
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

            if not os.path.exists(dst):
                logging.info("Staging %s to %s" % (src, dst))
                copyfile(src, dst)

                if hasheader:
                    copyfile(envi_header(src), envi_header(dst))

    def add_surface_subs_files(self, surface_type):
        self.surface_subs_files[surface_type] = {
            "rdn": self.rdn_subs_path + "_" + surface_type,
            "loc": self.loc_subs_path + "_" + surface_type,
            "obs": self.obs_subs_path + "_" + surface_type,
            "rfl": self.rfl_subs_path + "_" + surface_type,
            "state": self.state_subs_path + "_" + surface_type,
            "uncert": self.uncert_subs_path + "_" + surface_type,
            "h2o": self.h2o_subs_path + "_" + surface_type,
        }

        self.surface_config_paths[surface_type] = os.path.abspath(
            os.path.join(
                self.config_directory, f"{self.fid}_{surface_type}_isofit.json"
            )
        )

        self.surface_lut_paths[surface_type] = os.path.abspath(
            os.path.join(self.lut_modtran_directory, f"{surface_type}")
        )


class LUTConfig:
    """A look up table class, containing default grid options.  All properties may be overridden with the optional
        input configuration file path

    Args:
        lut_config_file: configuration file to override default values
    """

    def __init__(self, gip: dict, tsip: dict, lut_config_file: str = None):
        if lut_config_file is not None:
            with open(lut_config_file, "r") as f:
                lut_config = json.load(f)

        # For each element, set the look up table spacing (lut_spacing) as the
        # anticipated spacing value, or 0 to use a single point (not LUT).
        # Set the "lut_spacing_min" as the minimum distance allowed - if separation
        # does not meet this threshold based on the available data, on a single
        # point will be used.

        # Units of kilometers
        if "GNDALT" in gip["options"]["statevector_elements"] or "cloud" in tsip.keys():
            try:
                self.elevation_spacing = tsip["cloud"]["GNDALT"]["lut_spacing"]
                self.elevation_spacing_min = tsip["cloud"]["GNDALT"]["lut_spacing_min"]
            except KeyError:
                self.elevation_spacing = gip["radiative_transfer_parameters"]["GNDALT"][
                    "lut_spacing"
                ]
                self.elevation_spacing_min = gip["radiative_transfer_parameters"][
                    "GNDALT"
                ]["lut_spacing_min"]
        else:
            logging.info(
                "No spacing information for elevation LUT found in config file. "
                "Setting spacing to 0.25 and minimum spacing to 0.2."
            )
            self.elevation_spacing = 0.25
            self.elevation_spacing_min = 0.2

        # Units of g / m2
        try:
            self.h2o_spacing = gip["radiative_transfer_parameters"]["H2OSTR"][
                "lut_spacing"
            ]
        except KeyError:
            logging.info(
                "No spacing information for H2O LUT found in config file. Setting to"
                " 0.25."
            )
            self.h2o_spacing = 0.25

        try:
            self.h2o_spacing_min = gip["radiative_transfer_parameters"]["H2OSTR"][
                "lut_spacing_min"
            ]
        except KeyError:
            logging.info(
                "No spacing minimum for H2O LUT found in config file. Setting to 0.03."
            )
            self.h2o_spacing_min = 0.03

        # Special parameter to specify the minimum allowable water vapor value in g/m2
        try:
            self.h2o_min = gip["radiative_transfer_parameters"]["H2OSTR"]["min"]
        except KeyError:
            logging.info(
                "No minimum value for H2O LUT found in config file. Setting to 0.05."
            )
            self.h2o_min = 0.05

        # Set defaults, will override based on settings
        # Units of g / m2
        try:
            self.h2o_range = gip["radiative_transfer_parameters"]["H2OSTR"][
                "default_range"
            ]
        except KeyError:
            logging.info(
                "No default range for H2O LUT found in config file. Setting to"
                " [0.05, 5]."
            )
            self.h2o_range = [0.05, 5]

        # Units of degrees
        self.to_sensor_azimuth_spacing = 60
        self.to_sensor_azimuth_spacing_min = 60

        # Units of degrees
        self.to_sensor_zenith_spacing = 10
        self.to_sensor_zenith_spacing_min = 2

        # Units of AOD
        self.aerosol_0_spacing = 0
        self.aerosol_0_spacing_min = 0
        self.aerosol_1_spacing = 0
        self.aerosol_1_spacing_min = 0
        self.aerosol_2_spacing = 0.1
        self.aerosol_2_spacing_min = 0
        self.aerosol_0_range = [0.001, 1]
        self.aerosol_1_range = [0.001, 1]
        self.aerosol_2_range = [0.001, 1]

        try:
            self.aot_550_range = gip["radiative_transfer_parameters"]["AOT550"][
                "default_range"
            ]
        except KeyError:
            logging.info(
                "No default range for AOT LUT found in config file. Setting to"
                " [0.001, 1]."
            )
            self.aot_550_range = [0.001, 1]

        try:
            self.aot_550_spacing = gip["radiative_transfer_parameters"]["AOT550"][
                "lut_spacing"
            ]
        except KeyError:
            logging.info(
                "No spacing information for AOT LUT found in config file. Setting to 0."
            )
            self.aot_550_spacing = 0

        try:
            self.aot_550_spacing_min = gip["radiative_transfer_parameters"]["AOT550"][
                "lut_spacing_min"
            ]
        except KeyError:
            logging.info(
                "No spacing minimum for AOT LUT found in config file. Setting to 0."
            )
            self.aot_550_spacing_min = 0

        # overwrite anything that comes in from the config file
        if lut_config_file is not None:
            for key in lut_config:
                if key in self.__dict__:
                    setattr(self, key, lut_config[key])


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


def get_angular_grid(
    angle_data_input: np.array, spacing: float, min_spacing: float, units: str = "d"
):
    """Find either angular data "center points" (num_points = 1), or a lut set that spans
    angle variation in a systematic fashion.

    Args:
        angle_data_input: set of angle data to use to find center points
        spacing: the desired angular spacing between points, or mean if -1
        min_spacing: the minimum angular spacing between points allowed (if less, no grid)
        units: specifies if data are in degrees (default) or radians

    :Returns:
        angular data center point or lut set spanning space
    """
    if spacing == 0:
        logging.debug("Grid spacing set at 0, using no grid.")
        return None

    # Convert everything to radians so we don"t have to track throughout
    if units == "r":
        angle_data = np.rad2deg(angle_data_input)
    else:
        angle_data = angle_data_input.copy()

    spatial_data = np.hstack(
        [
            np.cos(np.deg2rad(angle_data)).reshape(-1, 1),
            np.sin(np.deg2rad(angle_data)).reshape(-1, 1),
        ]
    )

    # find which quadrants have data
    quadrants = np.zeros((2, 2))

    if np.any(np.logical_and(spatial_data[:, 0] > 0, spatial_data[:, 1] > 0)):
        quadrants[1, 0] = 1

    if np.any(np.logical_and(spatial_data[:, 0] > 0, spatial_data[:, 1] < 0)):
        quadrants[1, 1] += 1

    if np.any(np.logical_and(spatial_data[:, 0] < 0, spatial_data[:, 1] > 0)):
        quadrants[0, 0] += 1

    if np.any(np.logical_and(spatial_data[:, 0] < 0, spatial_data[:, 1] < 0)):
        quadrants[0, 1] += 1

    # Handle the case where angles are < 180 degrees apart
    if np.sum(quadrants) < 3 and spacing != -1:
        if np.sum(quadrants[1, :]) == 2:
            # If angles cross the 0-degree line:
            angle_spread = get_grid(
                np.min(angle_data + 180), np.max(angle_data + 180), spacing, min_spacing
            )

            if angle_spread is None:
                return None
            else:
                return angle_spread - 180
        else:
            # Otherwise, just space things out:
            return get_grid(
                np.min(angle_data), np.max(angle_data), spacing, min_spacing
            )
    else:
        if spacing >= 180:
            logging.warning(
                f"Requested angle spacing is {spacing}, but obs angle divergence is >"
                " 180.  Tighter  spacing recommended"
            )

        # If we"re greater than 180 degree spread, there"s no universal answer. Try GMM.
        if spacing == -1:
            num_points = 1
        else:
            # This very well might overly space the grid, but we don"t / can"t know in general
            num_points = int(np.ceil(360 / spacing))

        # We initialize the GMM with a static seed for repeatability across runs
        gmm = mixture.GaussianMixture(
            n_components=num_points, covariance_type="full", random_state=1
        )

        if spatial_data.shape[0] == 1:
            spatial_data = np.vstack([spatial_data, spatial_data])

        # Protect memory against huge images
        if spatial_data.shape[0] > 1e6:
            use = np.linspace(0, spatial_data.shape[0] - 1, int(1e6), dtype=int)
            spatial_data = spatial_data[use, :]

        gmm.fit(spatial_data)
        central_angles = np.degrees(np.arctan2(gmm.means_[:, 1], gmm.means_[:, 0]))

        if num_points == 1:
            return central_angles[0]

        ca_quadrants = np.zeros((2, 2))

        if np.any(np.logical_and(gmm.means_[:, 0] > 0, gmm.means_[:, 1] > 0)):
            ca_quadrants[1, 0] = 1
        elif np.any(np.logical_and(gmm.means_[:, 0] > 0, gmm.means_[:, 1] < 0)):
            ca_quadrants[1, 1] += 1
        elif np.any(np.logical_and(gmm.means_[:, 0] < 0, gmm.means_[:, 1] > 0)):
            ca_quadrants[0, 0] += 1
        elif np.any(np.logical_and(gmm.means_[:, 0] < 0, gmm.means_[:, 1] < 0)):
            ca_quadrants[0, 1] += 1

        if np.sum(ca_quadrants) < np.sum(quadrants):
            logging.warning(
                f"GMM angles {central_angles} span {np.sum(ca_quadrants)} quadrants, "
                f"while data spans {np.sum(quadrants)} quadrants"
            )

        return central_angles


def build_surface_config(
    macro_config: dict, flight_id: str, output_path: str, wvl_file: str
):
    """Write a surface config file, using the specified pathnames and all given info.

    Args:
        macro_config: dictionary of macro options for surface model
        flight_id: string of instrument specific flight identification number
        output_path: output directory for surface config file
        wvl_file: directory of instrument wavelength file
    """
    if not macro_config["output_model_file"]:
        surface_path = os.path.abspath(os.path.join(output_path, "surface.mat"))
        macro_config["output_model_file"] = surface_path

    if not macro_config["wavelength_file"]:
        macro_config["wavelength_file"] = wvl_file

    surface_config = macro_config

    output_config_name = os.path.join(output_path, flight_id + "_surface.json")

    with open(output_config_name, "w") as fout:
        fout.write(
            json.dumps(surface_config, cls=SerialEncoder, indent=4, sort_keys=True)
        )


def build_presolve_config(
    opt: dict,
    gip: dict,
    paths: Pathnames,
    h2o_lut_grid: np.array,
    use_superpixels: bool = False,
    uncorrelated_radiometric_uncertainty: float = 0.0,
):
    """Write an isofit config file for a presolve, with limited info.

    Args:
        opt: dictionary of general options
        gip: dictionary of general inversion parameters
        paths: object containing references to all relevant file locations
        h2o_lut_grid: the water vapor look up table grid isofit should use for this solve
        use_superpixels: flag whether or not to set up for the empirical or analytical line estimation
        uncorrelated_radiometric_uncertainty: uncorrelated radiometric uncertainty parameter for isofit
    """
    # Determine number of spectra included in each retrieval.  If we are
    # operating on segments, this will average down instrument noise
    if use_superpixels:
        spectra_per_inversion = (
            400 if not opt["segmentation_size"] else opt["segmentation_size"]
        )
    else:
        spectra_per_inversion = 1

    if gip["filepaths"]["emulator_base"] is None:
        engine_name = "modtran"
    else:
        engine_name = "sRTMnet"

    radiative_transfer_config = {
        "radiative_transfer_engines": {
            "vswir": {
                "engine_name": engine_name,
                "lut_path": paths.lut_h2o_directory,
                "template_file": paths.h2o_template_path,
                "lut_names": ["H2OSTR"],
                "statevector_names": ["H2OSTR"],
            }
        },
        "statevector": {
            "H2OSTR": {
                "bounds": [float(np.min(h2o_lut_grid)), float(np.max(h2o_lut_grid))],
                "scale": 0.01,
                "init": np.percentile(h2o_lut_grid, 25),
                "prior_sigma": 100.0,
                "prior_mean": 1.5,
            }
        },
        "lut_grid": {
            "H2OSTR": [float(x) for x in h2o_lut_grid],
        },
        "unknowns": {"H2O_ABSCO": 0.0},
    }

    if gip["filepaths"]["emulator_base"] is not None:
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "emulator_file"
        ] = os.path.abspath(gip["filepaths"]["emulator_base"])
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "emulator_aux_file"
        ] = os.path.abspath(
            os.path.splitext(gip["filepaths"]["emulator_base"])[0] + "_aux.npz"
        )
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "interpolator_base_path"
        ] = os.path.abspath(
            os.path.join(
                paths.lut_h2o_directory,
                os.path.basename(gip["filepaths"]["emulator_base"]) + "_vi",
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
                "surface_category": gip["options"]["surface_category"],
                "surface_file": paths.surface_working_path,
                "select_on_init": True,
            },
            "radiative_transfer": radiative_transfer_config,
        },
        "implementation": {
            "ray_temp_dir": paths.ray_temp_dir,
            "inversion": {"windows": gip["options"]["inversion_windows"]},
            "n_cores": multiprocessing.cpu_count()
            if not opt["n_cores"]
            else opt["n_cores"],
            "debug_mode": opt["debug_mode"],
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

    if use_superpixels:
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
    opt: dict,
    gip: dict,
    tsip: dict,
    paths: Pathnames,
    lut_params: LUTConfig,
    h2o_lut_grid: np.array = None,
    elevation_lut_grid: np.array = None,
    to_sensor_azimuth_lut_grid: np.array = None,
    to_sensor_zenith_lut_grid: np.array = None,
    mean_latitude: float = None,
    mean_longitude: float = None,
    dt: datetime = None,
    use_superpixels: bool = False,
    uncorrelated_radiometric_uncertainty: float = 0.0,
    surface_type: str = None,
):
    """Write an isofit config file for the main solve, using the specified pathnames and all given info

    Args:
        opt: dictionary of general options
        gip: dictionary of general inversion parameters
        tsip: dictionary of type specific inversion parameters
        paths: object containing references to all relevant file locations
        lut_params: configuration parameters for the lut grid
        h2o_lut_grid: the water vapor look up table grid isofit should use for this solve
        elevation_lut_grid: the ground elevation look up table grid isofit should use for this solve
        to_sensor_azimuth_lut_grid: the to-sensor azimuth angle look up table grid isofit should use for this solve
        to_sensor_zenith_lut_grid: the to-sensor zenith angle look up table grid isofit should use for this solve
        mean_latitude: the latitude isofit should use for this solve
        mean_longitude: the longitude isofit should use for this solve
        dt: the datetime object corresponding to this flightline to use for this solve
        use_superpixels: flag whether or not to set up for the empirical or analytical line estimation
        uncorrelated_radiometric_uncertainty: uncorrelated radiometric uncertainty parameter for isofit
        surface_type: surface type to run retrievals over - if None, do generic for all locations
    """
    # Determine number of spectra included in each retrieval.  If we are
    # operating on segments, this will average down instrument noise
    if use_superpixels:
        spectra_per_inversion = (
            400 if not opt["segmentation_size"] else opt["segmentation_size"]
        )
    else:
        spectra_per_inversion = 1

    if gip["filepaths"]["emulator_base"] is None:
        engine_name = "modtran"
    else:
        engine_name = "sRTMnet"

    radiative_transfer_config = {
        "topography_model": gip["options"]["topography_model"],
        "radiative_transfer_engines": {
            "vswir": {
                "engine_name": engine_name,
                "multipart_transmittance": gip["options"]["multipart_transmittance"],
                "lut_path": paths.surface_lut_paths[surface_type],
                "aerosol_template_file": paths.aerosol_tpl_path,
                "template_file": paths.modtran_template_path,
            }
        },
        "statevector": {},
        "lut_grid": {},
        "unknowns": {"H2O_ABSCO": 0.0},
    }

    if h2o_lut_grid is not None:
        radiative_transfer_config["statevector"]["H2OSTR"] = {
            "bounds": [h2o_lut_grid[0], h2o_lut_grid[-1]],
            "scale": 1,
            "init": (h2o_lut_grid[1] + h2o_lut_grid[-1]) / 2.0,
            "prior_sigma": 100.0,
            "prior_mean": (h2o_lut_grid[1] + h2o_lut_grid[-1]) / 2.0,
        }

    if "GNDALT" in gip["options"]["statevector_elements"] or surface_type == "cloud":
        radiative_transfer_config["statevector"]["GNDALT"] = {
            "bounds": [elevation_lut_grid[0], elevation_lut_grid[-1]],
            "scale": 100,
            "init": (elevation_lut_grid[1] + elevation_lut_grid[-1]) / 2.0,
            "prior_sigma": 1000.0,
            "prior_mean": (elevation_lut_grid[1] + elevation_lut_grid[-1]) / 2.0,
        }

    if gip["filepaths"]["emulator_base"] is not None:
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "emulator_file"
        ] = os.path.abspath(gip["filepaths"]["emulator_base"])
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "emulator_aux_file"
        ] = os.path.abspath(
            os.path.splitext(gip["filepaths"]["emulator_base"])[0] + "_aux.npz"
        )
        radiative_transfer_config["radiative_transfer_engines"]["vswir"][
            "interpolator_base_path"
        ] = os.path.abspath(
            os.path.join(
                paths.surface_lut_paths[surface_type],
                os.path.basename(os.path.splitext(gip["filepaths"]["emulator_base"])[0])
                + "_vi",
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

    if h2o_lut_grid is not None:
        radiative_transfer_config["lut_grid"]["H2OSTR"] = h2o_lut_grid.tolist()

    if elevation_lut_grid is not None:
        radiative_transfer_config["lut_grid"]["GNDALT"] = elevation_lut_grid.tolist()

    if to_sensor_azimuth_lut_grid is not None:
        radiative_transfer_config["lut_grid"][
            "TRUEAZ"
        ] = to_sensor_azimuth_lut_grid.tolist()

    if to_sensor_zenith_lut_grid is not None:
        radiative_transfer_config["lut_grid"][
            "OBSZEN"
        ] = to_sensor_zenith_lut_grid.tolist()  # modtran convension

    # add aerosol elements from climatology
    aerosol_state_vector, aerosol_lut_grid, aerosol_model_path = load_climatology(
        config_path=paths.aerosol_climatology,
        latitude=mean_latitude,
        longitude=mean_longitude,
        acquisition_datetime=dt,
        isofit_path=paths.isofit_path,
        lut_params=lut_params,
    )

    radiative_transfer_config["statevector"].update(aerosol_state_vector)
    radiative_transfer_config["lut_grid"].update(aerosol_lut_grid)
    radiative_transfer_config["radiative_transfer_engines"]["vswir"][
        "aerosol_model_file"
    ] = aerosol_model_path

    # MODTRAN should know about our whole LUT grid and all of our statevectors, so copy them in
    radiative_transfer_config["radiative_transfer_engines"]["vswir"][
        "statevector_names"
    ] = list(radiative_transfer_config["statevector"].keys())
    radiative_transfer_config["radiative_transfer_engines"]["vswir"][
        "lut_names"
    ] = list(radiative_transfer_config["lut_grid"].keys())

    if surface_type == "water":
        surface_category = tsip["water"]["surface_category"]
    else:
        surface_category = gip["options"]["surface_category"]

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
            "inversion": {"windows": gip["options"]["inversion_windows"]},
            "n_cores": multiprocessing.cpu_count()
            if not opt["n_cores"]
            else opt["n_cores"],
            "debug_mode": opt["debug_mode"],
        },
    }

    if use_superpixels:
        if surface_type is None:
            isofit_config_modtran["input"][
                "measured_radiance_file"
            ] = paths.rdn_subs_path
            isofit_config_modtran["input"]["loc_file"] = paths.loc_subs_path
            isofit_config_modtran["input"]["obs_file"] = paths.obs_subs_path
            isofit_config_modtran["output"][
                "estimated_state_file"
            ] = paths.state_subs_path
            isofit_config_modtran["output"][
                "posterior_uncertainty_file"
            ] = paths.uncert_subs_path
            isofit_config_modtran["output"][
                "estimated_reflectance_file"
            ] = paths.rfl_subs_path
        else:
            isofit_config_modtran["input"][
                "measured_radiance_file"
            ] = paths.surface_subs_files[surface_type]["rdn"]
            isofit_config_modtran["input"]["loc_file"] = paths.surface_subs_files[
                surface_type
            ]["loc"]
            isofit_config_modtran["input"]["obs_file"] = paths.surface_subs_files[
                surface_type
            ]["obs"]
            isofit_config_modtran["output"][
                "estimated_state_file"
            ] = paths.surface_subs_files[surface_type]["state"]
            isofit_config_modtran["output"][
                "posterior_uncertainty_file"
            ] = paths.surface_subs_files[surface_type]["uncert"]
            isofit_config_modtran["output"][
                "estimated_reflectance_file"
            ] = paths.surface_subs_files[surface_type]["rfl"]
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

    if gip["options"]["multiple_restarts"]:
        eps = 0.02 if not gip["options"]["eps"] else gip["options"]["eps"]
        grid = {}

        if h2o_lut_grid is not None:
            h2o_delta = float(h2o_lut_grid[-1]) - float(h2o_lut_grid[0])
            grid["H2OSTR"] = [
                round(h2o_lut_grid[0] + h2o_delta * eps, 4),
                round(h2o_lut_grid[-1] - h2o_delta * eps, 4),
            ]

        # We will initialize using different AODs for the first aerosol in the LUT
        if len(aerosol_lut_grid) > 0:
            key = list(aerosol_lut_grid.keys())[0]
            aer_delta = aerosol_lut_grid[key][-1] - aerosol_lut_grid[key][0]
            grid[key] = [
                round(aerosol_lut_grid[key][0] + aer_delta * eps, 4),
                round(aerosol_lut_grid[key][-1] - aer_delta * eps, 4),
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
        isofit_config_modtran["forward_model"]["instrument"]["SNR"] = 1000

    if paths.rdn_factors_path:
        isofit_config_modtran["input"][
            "radiometry_correction_file"
        ] = paths.rdn_factors_path

    # write modtran_template
    output_config_name = paths.surface_config_paths[surface_type]

    with open(output_config_name, "w") as fout:
        fout.write(
            json.dumps(
                isofit_config_modtran, cls=SerialEncoder, indent=4, sort_keys=True
            )
        )


def write_modtran_template(
    gip: dict,
    fid: str,
    altitude_km: float,
    dayofyear: int,
    latitude: float,
    longitude: float,
    to_sensor_azimuth: float,
    to_sensor_zenith: float,
    gmtime: float,
    elevation_km: float,
    output_file: str,
    ihaze_type: str = "AER_RURAL",
):
    """Write a MODTRAN template file for use by isofit look up tables

    Args:
        gip: dictionary of general inversion parameters
        fid: flight line id (name)
        altitude_km: altitude of the sensor in km
        dayofyear: the current day of the given year
        latitude: acquisition latitude
        longitude: acquisition longitude
        to_sensor_azimuth: azimuth view angle to the sensor, in degrees (AVIRIS convention)
        to_sensor_zenith: azimuth view angle to the sensor, in degrees (MODTRAN convention: 180 - AVIRIS convention)
        gmtime: greenwich mean time
        elevation_km: elevation of the land surface in km
        output_file: location to write the modtran template file to
        ihaze_type:

    """
    # make modtran configuration
    h2o_template = {
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
                        "MODEL": gip["radiative_transfer_parameters"][
                            "atmosphere_type"
                        ],
                        "M1": gip["radiative_transfer_parameters"]["atmosphere_type"],
                        "M2": gip["radiative_transfer_parameters"]["atmosphere_type"],
                        "M3": gip["radiative_transfer_parameters"]["atmosphere_type"],
                        "M4": gip["radiative_transfer_parameters"]["atmosphere_type"],
                        "M5": gip["radiative_transfer_parameters"]["atmosphere_type"],
                        "M6": gip["radiative_transfer_parameters"]["atmosphere_type"],
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
                        "IPARM": 11,
                        "PARM1": latitude,
                        "PARM2": longitude,
                        "TRUEAZ": to_sensor_azimuth,
                        "OBSZEN": to_sensor_zenith,
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
                        "DV": gip["radiative_transfer_parameters"]["spectral_DV"],
                        "FWHM": gip["radiative_transfer_parameters"]["spectral_FWHM"],
                        "YFLAG": "R",
                        "XFLAG": "N",
                        "FLAGS": "NT A   ",
                        "BMNAME": gip["radiative_transfer_parameters"][
                            "spectral_BMNAME"
                        ],
                    },
                    "FILEOPTIONS": {"NOPRNT": 2, "CKPRNT": True},
                }
            }
        ]
    }

    # write modtran_template
    with open(output_file, "w") as fout:
        fout.write(
            json.dumps(h2o_template, cls=SerialEncoder, indent=4, sort_keys=True)
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
        os.path.dirname(isofit.__file__), "..", "..", "data", "kurudz_0.1nm.dat"
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
):
    """Get metadata needed for complete runs from the observation file
    (bands: path length, to-sensor azimuth, to-sensor zenith, to-sun azimuth,
    to-sun zenith, phase, slope, aspect, cosine i, UTC time).

    Args:
        obs_file: file name to pull data from
        lut_params: parameters to use to define lut grid
        trim_lines: number of lines to ignore at beginning and end of file (good if lines contain values that are
                    erroneous but not nodata
        max_flight_duration_h: maximum length of the current acquisition, used to check if we"ve lapped a UTC day
        nodata_value: value to ignore from location file

    :Returns:
        tuple containing:
            h_m_s - list of the mean-time hour, minute, and second within the line
            increment_day - indicator of whether the UTC day has been changed since the beginning of the line time
            mean_path_km - mean distance between sensor and ground in km for good data
            mean_to_sensor_azimuth - mean to-sensor-azimuth for good data
            mean_to_sensor_zenith_rad - mean to-sensor-zenith in radians for good data
            valid - boolean array indicating which pixels were NOT nodata
            to_sensor_azimuth_lut_grid - the to-sensor azimuth angle look up table for good data
            to_sensor_zenith_lut_grid - the to-sensor zenith look up table for good data
    """
    obs_dataset = gdal.Open(obs_file, gdal.GA_ReadOnly)

    # Initialize values to populate
    valid = np.zeros((obs_dataset.RasterYSize, obs_dataset.RasterXSize), dtype=bool)

    path_km = np.zeros((obs_dataset.RasterYSize, obs_dataset.RasterXSize))
    to_sensor_azimuth = np.zeros((obs_dataset.RasterYSize, obs_dataset.RasterXSize))
    to_sensor_zenith = np.zeros((obs_dataset.RasterYSize, obs_dataset.RasterXSize))
    time = np.zeros((obs_dataset.RasterYSize, obs_dataset.RasterXSize))

    for line in range(obs_dataset.RasterYSize):
        # Read line in
        obs_line = obs_dataset.ReadAsArray(0, line, obs_dataset.RasterXSize, 1)

        # Populate valid
        valid[line, :] = np.logical_not(
            np.any(np.isclose(obs_line, nodata_value), axis=0)
        )

        path_km[line, :] = obs_line[0, ...] / 1000.0
        to_sensor_azimuth[line, :] = obs_line[1, ...]
        to_sensor_zenith[line, :] = obs_line[2, ...]
        time[line, :] = obs_line[9, ...]

    use_trim = trim_lines != 0 and valid.shape[0] > trim_lines * 2

    if use_trim:
        actual_valid = valid.copy()
        valid[:trim_lines, :] = False
        valid[-trim_lines:, :] = False

    mean_path_km = np.mean(path_km[valid])
    del path_km

    mean_to_sensor_azimuth = (
        get_angular_grid(
            angle_data_input=to_sensor_azimuth[valid], spacing=-1, min_spacing=0
        )
        % 360
    )

    mean_to_sensor_zenith = 180 - get_angular_grid(
        angle_data_input=to_sensor_zenith[valid], spacing=-1, min_spacing=0
    )

    to_sensor_zenith_lut_grid = get_angular_grid(
        angle_data_input=to_sensor_zenith[valid],
        spacing=lut_params.to_sensor_zenith_spacing,
        min_spacing=lut_params.to_sensor_zenith_spacing_min,
    )

    if to_sensor_zenith_lut_grid is not None:
        to_sensor_zenith_lut_grid = np.sort(180 - to_sensor_zenith_lut_grid)

    to_sensor_azimuth_lut_grid = get_angular_grid(
        to_sensor_azimuth[valid],
        lut_params.to_sensor_azimuth_spacing,
        lut_params.to_sensor_azimuth_spacing_min,
    )

    if to_sensor_azimuth_lut_grid is not None:
        to_sensor_azimuth_lut_grid = np.sort(
            np.array([x % 360 for x in to_sensor_azimuth_lut_grid])
        )

    del to_sensor_azimuth
    del to_sensor_zenith

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
        valid,
        to_sensor_azimuth_lut_grid,
        to_sensor_zenith_lut_grid,
    )


def get_metadata_from_loc(
    loc_file: str,
    gip: dict,
    tsip: dict,
    lut_params: LUTConfig,
    trim_lines: int = 5,
    nodata_value: float = -9999,
) -> (float, float, float, np.array):
    """Get metadata needed for complete runs from the location file (bands long, lat, elev).

    Args:
        loc_file: file name to pull data from
        gip: dictionary of general inversion parameters
        tsip: dictionary of type specific inversion parameters
        lut_params: parameters to use to define lut grid
        trim_lines: number of lines to ignore at beginning and end of file (good if lines contain values that are
                    erroneous but not nodata
        nodata_value: value to ignore from location file

    :Returns:
        tuple containing:
            mean_latitude - mean latitude of good values from the location file
            mean_longitude - mean latitude of good values from the location file
            mean_elevation_km - mean ground estimate of good values from the location file
            elevation_lut_grid - the elevation look up table, based on globals and values from location file
    """

    loc_dataset = gdal.Open(loc_file, gdal.GA_ReadOnly)

    loc_data = np.zeros(
        (loc_dataset.RasterCount, loc_dataset.RasterYSize, loc_dataset.RasterXSize)
    )

    for line in range(loc_dataset.RasterYSize):
        # Read line in
        loc_data[:, line : line + 1, :] = loc_dataset.ReadAsArray(
            0, line, loc_dataset.RasterXSize, 1
        )

    valid = np.logical_not(np.any(loc_data == nodata_value, axis=0))
    use_trim = trim_lines != 0 and valid.shape[0] > trim_lines * 2

    if use_trim:
        valid[:trim_lines, :] = False
        valid[-trim_lines:, :] = False

    if "Easting (m)" and "Northing (m)" in loc_dataset.GetMetadata().values():
        loc_hdr = envi.open(loc_file + ".hdr")
        zone = [int(s) for s in loc_hdr.metadata["description"].split() if s.isdigit()][
            0
        ]
        hemisphere = [
            s
            for s in loc_hdr.metadata["description"].split()
            if s in ["North", "South"]
        ][0]

        lat, lon = utm.to_latlon(
            easting=loc_data[0, valid].flatten(),
            northing=loc_data[1, valid].flatten(),
            zone_number=zone,
            northern=hemisphere == "North",
        )
    else:
        lat = loc_data[1, valid].flatten()
        lon = loc_data[0, valid].flatten()

    # Grab sensor position and orientation information
    mean_latitude = get_angular_grid(lat, -1, 0)
    mean_longitude = get_angular_grid(-1 * lon, -1, 0)

    mean_elevation_km = np.mean(loc_data[2, valid]) / 1000.0

    # make elevation grid
    min_elev = np.min(loc_data[2, valid]) / 1000.0
    max_elev = np.max(loc_data[2, valid]) / 1000.0

    if "GNDALT" in gip["options"]["statevector_elements"] or "cloud" in tsip.keys():
        try:
            exp_range = tsip["cloud"]["GNDALT"]["expand_range"]
        except KeyError:
            exp_range = gip["radiative_transfer_parameters"]["GNDALT"]["expand_range"]

        min_elev = max(min_elev - exp_range, 0)
        max_elev += exp_range

    elevation_lut_grid = get_grid(
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
