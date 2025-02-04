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
# Authors: David R Thompson, david.r.thompson@jpl.nasa.gov
#          Nimrod Carmon, nimrod.carmon@jpl.nasa.gov
#

import json
import logging
import os
import re
import subprocess
import time
from copy import deepcopy
from sys import platform

import numpy as np
import scipy.interpolate
import scipy.stats

from isofit.core.common import json_load_ascii, recursive_replace
from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine

Logger = logging.getLogger(__file__)

### Variables ###

eps = 1e-5  # used for finite difference derivative calculations
tropopause_altitude_km = 17.0

### Classes ###


class ModtranRT(RadiativeTransferEngine):
    """A model of photon transport including the atmosphere."""

    def __init__(
        self, engine_config, min_samples_per_nm=10, max_samples_per_nm=100, **kwargs
    ):
        self.max_buffer_time = 0.5
        self.resolutions_available = [0.1, 1, 5, 15]
        self.resolution_names = ["p1_2013", "01_2013", "05_2013", "15_2013"]
        self.min_samples_per_nm = min_samples_per_nm
        self.max_samples_per_nm = max_samples_per_nm
        print(f"kwargs: {kwargs}")

        super().__init__(engine_config, **kwargs)

    @staticmethod
    def samples_per_nm(wl, fq_resolution):
        fq_delta = 10**7 / wl - 10**7 / (wl + 1)
        samples_per_nm = fq_delta / fq_resolution

        return samples_per_nm

    @staticmethod
    def calc_band_model(samples_per_nm: int, wavelength: float):
        delta_nm = 1 / float(samples_per_nm)
        delta_freq = 10**7 / wavelength - 10**7 / (
            wavelength + delta_nm
        )  # do unit conversion

        if delta_freq > 15:
            return "15_2013"
        elif delta_freq > 5:
            return "05_2013"
        elif delta_freq > 1:
            return "01_2013"
        elif delta_freq > 0.1:
            return "p1_2013"
        else:
            raise ValueError(f"Unsupported resolution: {delta_freq}")

    @staticmethod
    def parseTokens(tokens: list, coszen: float) -> dict:
        """
        Processes tokens returned by parseLine()

        Parameters
        ----------
        tokens: list
            List of floats returned by parseLine()
        coszen: float
            cos(zenith(filename))

        Returns
        -------
        dict
            Dictionary of calculated values using the tokens list
        """
        irr = tokens[18] * 1e6 * np.pi / tokens[8] / coszen  # uW/nm/cm2

        # fmt: off
        # If classic singlepart transmittance is used,
        # we store total transmittance ((down direct + down diffuse) * (up direct + up diffuse))
        # under the diffuse down transmittance key (transm_down_dif) to ensure consistency
        # Tokens[24] contains only the direct upward transmittance,
        # so we store it under the direct upward transmittance key (transm_up_dir)
        # ToDo: remove in future versions and enforce the use of multipart transmittance
        return {
            'solar_irr'          : irr,       # Solar irradiance
            'wl'                 : tokens[0], # Wavelength
            'rhoatm'             : tokens[4] * 1e6 * np.pi / (irr * coszen), # unitless
            'width'              : tokens[8],
            'thermal_upwelling'  : (tokens[11] + tokens[12]) / tokens[8] * 1e6, # uW/nm/sr/cm2
            'thermal_downwelling': tokens[16] * 1e6 / tokens[8],
            'path_rdn'           : tokens[14] * 1e6 + tokens[15] * 1e6, # The sum of the (1) single scattering and (2) multiple scattering
            'grnd_rflt'          : tokens[16] * 1e6,        # ground reflected radiance (direct+diffuse+multiple scattering)
            'drct_rflt'          : tokens[17] * 1e6,        # same as 16 but only on the sun->surface->sensor path (only direct)
            'transm_down_dif'    : tokens[21] + tokens[22],  # total transmittance (down * up, direct + diffuse)
            'sphalb'             : tokens[23],  # atmospheric spherical albedo
            'transm_up_dir'      : tokens[24],  # upward direct transmittance
        }
        # fmt: on

    @staticmethod
    def parseLine(line: str) -> list:
        """
        Parses a single line of a .chn file into a list of token values

        Parameters
        ----------
        line: str
            Singular data line of a MODTRAN .chn file

        Returns
        -------
        list
            List of floats parsed from the line
        """
        # Fixes issues in large datasets where irrelevant columns touch which breaks parseTokens()
        line = line[:17] + " " + line[18:]

        return [float(match) for match in re.findall(r"(\d\S*)", line)]

    def load_chn(self, file: str, coszen: float, header: int = 5) -> dict:
        """
        Parses a MODTRAN channel file and extracts relevant data

        Parameters
        ----------
        file: str
            Path to a .chn file
        coszen: float
            ...
        header: int, defaults=5
            Number of lines to skip for the header

        Returns
        -------
        chn: dict
            Channel data
        """
        with open(file, "r") as f:
            lines = f.readlines()

        data = [lines[header:]]

        # Determine if this is a multipart transmittance file, break if so
        L = len(lines)
        for N in range(2, 4):  # Currently support 1, 2, and 3 part transmittance files
            # Length of each part
            n = int(L / N)

            # Check if the first line of the next part is the same
            if lines[1] == lines[n + 1]:
                Logger.debug(f"Channel file discovered to be {N} parts: {file}")

                # Parse the lines into N many parts
                data = []
                for i in range(N):
                    j = i + 1
                    data.append(lines[n * i : n * j][header:])

                # No need to check other N sizes
                break
        else:
            Logger.warning(
                "Channel file detected to be a single transmittance, support for this will be dropped in a future version."
                + " Please start using 2 or 3 multipart transmittance files."
            )

        parts = []
        for part, lines in enumerate(data):
            parsed = [self.parseTokens(self.parseLine(line), coszen) for line in lines]

            # Convert from: [{k1: v11, k2: v21}, {k1: v12, k2: v22}]
            #           to: {k1: [v11, v22], k2: [v21, v22]} - as numpy arrays
            combined = {}
            for i, parse in enumerate(parsed):
                for key, value in parse.items():
                    values = combined.setdefault(key, np.full(len(parsed), np.nan))
                    values[i] = value

            parts.append(combined)

        # Single transmittance files will be the first dict in the list, otherwise multiparts use two_albedo_method
        # The below parsing is a stopgap until we update the examples to all use tp7, and the true two albedo method
        chn = parts[0]
        if len(parts) == 2:
            Logger.debug("Using two albedo method from 2 components")
            chn = self.two_albedo_method(parts[0], parts[1], coszen, *self.test_rfls)
        if len(parts) == 3:
            Logger.debug("Using two albedo method from 3 components")
            chn = self.two_albedo_method(parts[1], parts[2], coszen, *self.test_rfls)

        return chn

    @staticmethod
    def load_tp6(file):
        """
        Parses relevant information from a tp6 file. Specifically, seeking a
        table in the unstructured text and extracting a column from it.

        Parameters
        ----------
        tp6: str
            tp6 file path
        """
        with open(file, "r") as tp6:
            lines = tp6.readlines()

        if not lines:
            raise ValueError(f"tp6 file is empty: {file}")

        for i, line in enumerate(lines):
            # Table found
            if "SINGLE SCATTER SOLAR" in line:
                i += 5  # Skip header
                break

        # Start at the table
        solzen = []
        for line in lines[i:]:
            split = line.split()

            # End of table
            if not split:
                break

            # Retrieve solar zenith
            solzen.append(float(split[3]))

        if not solzen:
            raise ValueError(f"No solar zenith found in tp6 file: {file}")

        return np.mean(solzen)

    @staticmethod
    def read_tp7(file_path: str):
        """Read a MODTRAN TP7 file and return the data as a dictionary, one entry
        for each case in the file.  Don't do anything but read the data; all the data.

        Parameters
        ----------
        file_path: str
            Path to the MODTRAN TP7 file

        Returns
        -------
        cases_data: dict
            A dictionary with case numbers as keys and numpy structured arrays as values.
            The structured arrays contain the data for each case in the TP7 file.
        """
        with open(file_path, "r") as file:
            lines = file.readlines()

        case_indices = [i for i, line in enumerate(lines) if "case index" in line]
        end_indices = [i for i, line in enumerate(lines) if line.strip() == "}"]
        cases_data = {}

        # Pair up start and end indices of each case
        end_indices = [i for i, line in enumerate(lines) if line.strip() == "}"]
        case_ranges = zip(case_indices, end_indices)

        for case_num, (start_index, end_index) in enumerate(case_ranges):
            col_names_line_1 = lines[start_index + 4].strip().split(",")
            col_names_line_2 = lines[start_index + 5].strip().split(",")

            column_names = [
                name1.strip() + " " + name2.strip()
                for name1, name2 in zip(col_names_line_1, col_names_line_2)
            ]

            data_lines = lines[start_index + 6 : end_index]
            data = np.genfromtxt(data_lines, delimiter=",", names=column_names)

            cases_data[case_num] = data

        return cases_data

    @staticmethod
    def merge_multiresolution_cases(cases_data: dict, num_albedos, num_models: int):
        """Merge cases data from multiple albedos and models into a single dictionary.

        Args:
            cases_data (dict): dictionary of individual case dictions, straight read from a tp7
            num_albedos (_type_): the number of albedo models used
            num_models (int): the number of wavelength regions used
        """

        def rdn_in_nm(rdn, wvn):
            """
            Convert irradiance wavenumber to wavelength. Steps:

                1. Wavelength (λ) and wavenumber (v) are related by λ = 1/v.
                   To get λ in nanometers, use λ_nm = 10^7 / v.
                2. Differential relation: dλ = -1/v^2 dv. In nanometers: dλ_nm = -10^7 / v^2 dv.
                3. Conversion Formula: I_λ = I_v * |dv/dλ_nm|.
                   Since dλ_nm/dv = -10^7 / v^2, the conversion factor is |dv/dλ_nm| = v^2 / 10^7.
                4. Apply conversion for each wavenumber (v) to obtain irradiance in W cm^-2 / nm.
                   I_λ_nm = I_v_cm^-1 * (v^2 / 10^7)
                Note: This formula considers the squared relationship between v and λ and unit conversion from cm to nm.

            Parameters
            ----------
            rdn : float
                Irradiance in W cm^-2 / cm^-1.
            wvn : float
                Wavenumber in cm^-1.

            Returns
            -------
            float
                Irradiance in W cm^-2 / nm.
            """
            return rdn * (wvn**2 / 10**7)

        product_names = cases_data[0].dtype.names

        # Merge between albedo and case cases
        merged_dicts = [{} for _ in range(num_albedos)]
        for _m in range(len(merged_dicts)):
            for product in product_names:
                merged_dicts[_m][product] = []
        case_count = 0
        for _a in range(num_albedos):
            for _n in range(num_models):
                for product in product_names:
                    merged_dicts[_a][product].append(cases_data[case_count][product])
                case_count += 1

        # Stack
        for _m in range(len(merged_dicts)):
            for product in product_names:
                merged_dicts[_m][product] = np.hstack(merged_dicts[_m][product])

        # Translate units, convert to wavelength, and sort
        for _m in range(len(merged_dicts)):
            merged_dicts[_m]["wl"] = 10**7 / merged_dicts[_m]["Freq_cm1"]

            # lowest frequency should have the highest resolution models
            # sort and de-dup
            order1 = np.argsort(merged_dicts[_m]["Freq_cm1"])
            order2 = np.unique(merged_dicts[_m]["Freq_cm1"][order1], return_index=True)[
                1
            ]
            for product in product_names:
                if product in [
                    "grnd_rflt",
                    "drct_rflt",
                    "total_rad",
                    "path_multiple_scat",
                    "sing_scat",
                    "ToA_irrad",
                ]:
                    merged_dicts[_m][product] = rdn_in_nm(
                        merged_dicts[_m][product], merged_dicts[_m]["Freq_cm1"]
                    )[order1][order2]
                elif product != "Freq_cm1":
                    merged_dicts[_m][product] = merged_dicts[_m][product][order1][
                        order2
                    ]
            merged_dicts[_m]["wl"] = merged_dicts[_m]["wl"][order1][order2]

        # Convert to what the rest of ISOFIT is going to expect
        case_output_dict = {}
        for _i, indict in enumerate(merged_dicts):
            output_dict = {}
            output_dict["solar_irr"] = indict["ToA_irrad"] * 1e6
            output_dict["wl"] = indict["wl"]
            output_dict["transm_up_dir"] = np.exp(-1 * indict["_nat_log_path_trans"])
            output_dict["drct_rflt"] = indict["drct_rflt"] * 1e6
            output_dict["grnd_rflt"] = indict["grnd_rflt"] * 1e6
            if _i == 0:
                output_dict["path_rdn"] = indict["total_rad"] * 1e6
            else:
                output_dict["path_rdn"] = (
                    indict["sing_scat"] + indict["path_multiple_scat"]
                ) * 1e6
            output_dict["width"] = 1  # We're in line reads
            output_dict["rhoatm"] = output_dict["path_rdn"] / output_dict["solar_irr"]

            # The first of these is a guess - not validated.  The second is a placeholder
            output_dict["thermal_upwelling"] = indict["surface_emission"] * 1e6
            output_dict["thermal_downwelling"] = np.zeros_like(
                indict["surface_emission"]
            )

            case_output_dict[_i] = output_dict

        return case_output_dict

    def preSim(self):
        """
        Post-initialized, pre-simulation setup
        """
        self.filtpath = os.path.join(
            self.sim_path,
            f"wavelengths_{self.engine_config.engine_name}_{self.wl[0]}_{self.wl[-1]}.flt",
        )
        self.template = json_load_ascii(self.engine_config.template_file)["MODTRAN"]

        # Regenerate MODTRAN input wavelength file
        if not os.path.exists(self.filtpath):
            self.wl2flt(self.wl, self.fwhm, self.filtpath)

        # Insert aerosol templates, if specified
        if self.engine_config.aerosol_model_file is not None:
            self.template[0]["MODTRANINPUT"]["AEROSOLS"] = json_load_ascii(
                self.engine_config.aerosol_template_file
            )

        # Insert aerosol data, if specified
        if self.engine_config.aerosol_model_file is not None:
            aer_data = np.loadtxt(self.engine_config.aerosol_model_file)
            self.aer_wl = aer_data[:, 0]
            aer_data = np.transpose(aer_data[:, 1:])
            self.naer = int(len(aer_data) / 3)
            aer_absc, aer_extc, aer_asym = [], [], []
            for i in range(self.naer):
                aer_extc.append(aer_data[i * 3])
                aer_absc.append(aer_data[i * 3 + 1])
                aer_asym.append(aer_data[i * 3 + 2])
            self.aer_absc = np.array(aer_absc)
            self.aer_extc = np.array(aer_extc)
            self.aer_asym = np.array(aer_asym)

        # Figure out wavelength grid to run on

        # always run wavelength modeles from fine to coarse spectral resolution,
        # so that for duplicates we take the finer resolution case
        samples_wl_grid = np.arange(
            int(np.floor(np.min(self.wl))), int(np.ceil((np.max(self.wl))))
        )
        samples_per_res = [
            self.samples_per_nm(samples_wl_grid, res)
            for res in self.resolutions_available
        ]

        self.simulation_wavelength_regions = []
        self.wavelength_models = []
        for _s in range(len(samples_per_res)):
            wl_range = samples_wl_grid[
                np.logical_and(
                    samples_per_res[_s] >= self.min_samples_per_nm,
                    samples_per_res[_s] <= self.max_samples_per_nm,
                )
            ]
            if len(wl_range) > 0:
                wl_range = [np.min(wl_range), np.max(wl_range)]
                self.simulation_wavelength_regions.append(wl_range)
                self.wavelength_models.append(self.resolution_names[_s])

        if len(self.simulation_wavelength_regions) == 0:
            raise ValueError(
                "No valid wavelength regions found for simulation. Adjust min or max samples per nm."
            )

        # Don't overlap by more than 1 nm, and prioritize coarser resolution models for comp when we can:
        if len(self.simulation_wavelength_regions) > 2:
            for i in range(len(self.simulation_wavelength_regions) - 2, -1, -1):
                self.simulation_wavelength_regions[i][0] = (
                    self.simulation_wavelength_regions[i + 1][1] - 1
                )

        if self.simulation_wavelength_regions[0][0] > np.min(self.wl):
            self.simulation_wavelength_regions[0][0] = np.min(self.wl)
            logging.info(
                "Adjusted first wavelength region to start at the minimum wavelength."
            )
        if self.simulation_wavelength_regions[-1][-1] < np.max(self.wl):
            self.simulation_wavelength_regions[-1][-1] = np.max(self.wl)
            logging.info(
                "Adjusted last wavelength region to end at the maximum wavelength."
            )

        for _s in range(len(self.simulation_wavelength_regions)):
            logging.info(
                f"Using MODTRAN band model {self.wavelength_models[_s]} in simulation wavelength region: {self.simulation_wavelength_regions[_s]}"
            )

    def readSim(self, point):
        """
        For a given point, parses the tp6 and chn file and returns the data
        """
        file = os.path.join(self.sim_path, self.point_to_filename(point))

        solzen = self.load_tp6(f"{file}.tp6")
        coszen = np.cos(solzen * np.pi / 180.0)
        if os.path.isfile(f"{file}.csv"):
            params = self.read_tp7(f"{file}.csv")
            params = self.merge_multiresolution_cases(
                params, len(self.test_rfls), len(self.wavelength_models)
            )
            # Only need to run two_albedo method if we have multiple cases
            # still at this point.  Note that at this time, merge_multiresolution_cases
            # is set up to NEED to run through two_albedo_method, but this might
            # not always be the case.
            if len(params) == 2:
                params = self.two_albedo_method(
                    *params.values(), coszen, *self.test_rfls
                )
        else:
            params = self.load_chn(f"{file}.chn", coszen)

        # Remove thermal terms in VSWIR runs to avoid incorrect usage
        if self.treat_as_emissive is False:
            for key in ["thermal_upwelling", "thermal_downwelling"]:
                if key in params:
                    Logger.debug(
                        f"Deleting key because treat_as_emissive is False: {key}"
                    )
                    del params[key]

        params["solzen"] = solzen
        params["coszen"] = coszen

        return params

    def makeSim(self, point, file=None, timeout=None):
        """
        Prepares the command to execute MODTRAN
        """
        if self.engine_base_dir is None:
            Logger.error(
                "No MODTRAN installation provided, please set config key `engine_base_dir`"
            )
            return

        filename_base = file or self.point_to_filename(point)

        # Translate ISOFIT generic lut names to MODTRAN-specific names
        translation = {
            "surface_elevation_km": "GNDALT",
            "observer_altitude_km": "H1ALT",
            "observer_azimuth": "TRUEAZ",
            "observer_zenith": "OBSZEN",
        }
        names = [translation.get(key, key) for key in self.lut_names]

        vals = dict([(n, v) for n, v in zip(names, point)])
        vals["DISALB"] = True
        vals["NAME"] = filename_base
        vals["FILTNM"] = os.path.normpath(self.filtpath)
        # import ipdb; ipdb.set_trace()
        # if "CSVPRINT" in vals["FILEOPTIONS"].keys():
        #    vals["FILEOPTIONS"]["CSVPRINT"] = filename_base + ".csv"
        vals["CSVPRNT"] = filename_base + ".csv"

        # Translate to the MODTRAN OBSZEN convention, assumes we are downlooking
        if "OBSZEN" in vals and vals.get("OBSZEN") < 90:
            vals["OBSZEN"] = 180 - abs(vals["OBSZEN"])

        modtran_config_str, modtran_config = self.modtran_driver(dict(vals))

        # Check rebuild conditions: LUT is missing or from a different config
        infilename = "LUT_" + filename_base + ".json"
        infilepath = os.path.join(self.sim_path, "LUT_" + filename_base + ".json")

        if self.required_results_exist(filename_base):
            Logger.warning(f"File already exists, skipping execution: {filename_base}")
            return

        # write_config_file
        with open(infilepath, "w") as f:
            f.write(modtran_config_str)

        if self.engine_config.rte_configure_and_exit:
            return

        # Specify location of the proper MODTRAN 6.0 binary for this OS
        xdir = {"linux": "linux", "darwin": "macos", "windows": "windows"}

        # Generate the CLI path
        cmd = os.path.join(
            self.engine_base_dir, "bin", xdir[platform], "mod6c_cons " + infilename
        )

        call = subprocess.run(
            cmd, shell=True, timeout=timeout, cwd=self.sim_path, capture_output=True
        )
        if call.stdout:
            Logger.error(call.stdout.decode())

    def modtran_driver(self, overrides):
        """Write a MODTRAN 6.0 input file."""

        param = deepcopy(self.template)

        if hasattr(self, "aer_absc"):
            fracs = np.zeros((self.naer))

        if "IPARM" not in param[0]["MODTRANINPUT"]["GEOMETRY"]:
            raise AttributeError("MODTRAN template requires an IPARM specification")

        if param[0]["MODTRANINPUT"]["GEOMETRY"]["ITYPE"] != 3:
            raise AttributeError("Currently unsupported modtran ITYPE specification")

        # Geometry values that depend on IPARM
        if (
            param[0]["MODTRANINPUT"]["GEOMETRY"]["IPARM"] == 12
            and "GMTIME" in overrides.keys()
        ):
            raise AttributeError(
                "GMTIME in MODTRAN driver overrides, but IPARM set to 12.  Check"
                " modtran template."
            )
        elif param[0]["MODTRANINPUT"]["GEOMETRY"]["IPARM"] == 11 and {
            "solar_azimuth",
            "solaz",
            "solar_zenith",
            "solzen",
        }.intersection(set(overrides.keys())):
            raise AttributeError(
                "Solar geometry (solar az/azimuth zen/zenith) is specified, but IPARM"
                " is set to 11.  Check MODTRAN template"
            )

        if {"PARM1", "PARM2"}.intersection(set(overrides.keys())):
            raise AttributeError(
                "PARM1 and PARM2 keys not supported as LUT dimensions.  Please use"
                " either solar_azimuth/solaz or solar_zenith/solzen"
            )

        # Perform overrides
        for key, val in overrides.items():
            recursive_replace(param, key, val)

            if key.startswith("AER"):
                i = int(key.split("_")[-1])
                fracs[i] = val

            elif key in ["EXT550", "AOT550", "AOD550"]:
                # MODTRAN 6.0 convention treats negative visibility as AOT550
                recursive_replace(param, "VIS", -val)

            elif key == "FILTNM":
                param[0]["MODTRANINPUT"]["SPECTRAL"]["FILTNM"] = val

            # Geometry parameters we want to populate even if unassigned
            elif key in ["H1ALT", "IDAY", "TRUEAZ", "OBSZEN", "GMTIME"]:
                param[0]["MODTRANINPUT"]["GEOMETRY"][key] = val

            elif key == "AIRT_DELTA_K":
                # If there is no profile already provided ...
                if (
                    param[0]["MODTRANINPUT"]["ATMOSPHERE"]["MODEL"]
                    != "ATM_USER_ALT_PROFILE"
                ):
                    # MODTRAN cannot accept a ground altitude above 6 km, so keep all layers after that
                    gndalt = param[0]["MODTRANINPUT"]["SURFACE"]["GNDALT"]

                    # E.g.: [1.5, 2, 3, 4, 5]
                    low_altitudes = [gndalt] + list(
                        np.arange(6 - np.ceil(gndalt)) + np.ceil(gndalt)
                    )

                    # MODTRAN cannot accept a ground altitude above 6 km, so keep all layers after that
                    hi_altitudes = [
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                        12.0,
                        13.0,
                        14.0,
                        15.0,
                        16.0,
                        17.0,
                        18.0,
                        19.0,
                        20.0,
                        21.0,
                        22.0,
                        23.0,
                        24.0,
                        25.0,
                        30.0,
                        35.0,
                        40.0,
                        45.0,
                        50.0,
                        55.0,
                        60.0,
                        70.0,
                        80.0,
                        100.0,
                    ]

                    altitudes = (
                        low_altitudes + hi_altitudes
                    )  # Append lists, don't add altitudes!

                    prof_unt_tdelta_kelvin = np.where(
                        np.array(altitudes) <= tropopause_altitude_km, val, 0
                    )

                    altitude_dict = {
                        "TYPE": "PROF_ALTITUDE",
                        "UNITS": "UNT_KILOMETERS",
                        "PROFILE": altitudes,
                    }
                    delta_kelvin_dict = {
                        "TYPE": "PROF_TEMPERATURE",
                        "UNITS": "UNT_TDELTA_KELVIN",
                        "PROFILE": prof_unt_tdelta_kelvin.tolist(),
                    }

                    param[0]["MODTRANINPUT"]["ATMOSPHERE"][
                        "MODEL"
                    ] = "ATM_USER_ALT_PROFILE"
                    param[0]["MODTRANINPUT"]["ATMOSPHERE"]["NPROF"] = 2
                    param[0]["MODTRANINPUT"]["ATMOSPHERE"]["NLAYERS"] = len(altitudes)
                    param[0]["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"] = [
                        altitude_dict,
                        delta_kelvin_dict,
                    ]

                else:  # A profile is already provided, assume that it includes PROF_ALTITUDE
                    nprof = param[0]["MODTRANINPUT"]["ATMOSPHERE"]["NPROF"]
                    profile_types = []
                    for i in range(nprof):
                        profile_types.append(
                            param[0]["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"][i][
                                "TYPE"
                            ]
                        )

                    ind_prof_altitude = profile_types.index("PROF_ALTITUDE")
                    prof_altitude = np.array(
                        param[0]["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"][
                            ind_prof_altitude
                        ]["PROFILE"]
                    )

                    if "PROF_TEMPERATURE" in profile_types:
                        # If a temperature profile already exists, then we must add the temperature delta to that
                        # as MODTRAN apparently does not allow have both an offset and a specified temperature
                        ind_prof_temperature = profile_types.index("PROF_TEMPERATURE")
                        prof_temperature = np.array(
                            param[0]["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"][
                                ind_prof_temperature
                            ]["PROFILE"]
                        )
                        prof_temperature = np.where(
                            prof_altitude <= tropopause_altitude_km,
                            prof_temperature + val,
                            prof_temperature,
                        )
                        param[0]["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"][
                            ind_prof_temperature
                        ]["PROFILE"] = prof_temperature.tolist()

                    else:
                        # If a temperature profile does not exist, then use UNT_TDELTA_KELVIN
                        prof_unt_tdelta_kelvin = np.where(
                            prof_altitude <= tropopause_altitude_km, val, 0.0
                        )
                        prof_unt_tdelta_kelvin_dict = {
                            "TYPE": "PROF_TEMPERATURE",
                            "UNITS": "UNT_TDELTA_KELVIN",
                            "PROFILE": prof_unt_tdelta_kelvin.tolist(),
                        }
                        param[0]["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"].append(
                            prof_unt_tdelta_kelvin_dict
                        )
                        param[0]["MODTRANINPUT"]["ATMOSPHERE"]["NPROF"] = nprof + 1

            # Surface parameters we want to populate even if unassigned
            elif key in ["surface_elevation_km", "GNDALT"]:
                param[0]["MODTRANINPUT"]["SURFACE"]["GNDALT"] = val

            # Make sure that view geometry gets populated if not assigned previously
            elif key in ["observer_azimuth", "trueaz"]:
                param[0]["MODTRANINPUT"]["GEOMETRY"]["TRUEAZ"] = val

            elif key in ["observer_zenith", "obszen"]:
                param[0]["MODTRANINPUT"]["GEOMETRY"]["OBSZEN"] = val

            # Populate solar geometry
            elif key in ["solar_zenith", "solzen", "SOLZEN"]:
                param[0]["MODTRANINPUT"]["GEOMETRY"]["PARM2"] = val

            elif key in ["relative_azimuth", "relaz", "RELAZ"]:
                param[0]["MODTRANINPUT"]["GEOMETRY"]["PARM1"] = val

            elif key in ["DISALB", "NAME"]:
                recursive_replace(param, key, val)
            elif key in param[0]["MODTRANINPUT"]["ATMOSPHERE"].keys():
                recursive_replace(param, key, val)
            elif key in param[0]["MODTRANINPUT"]["FILEOPTIONS"].keys():
                recursive_replace(param, key, val)
            else:
                raise AttributeError(
                    "Unsupported MODTRAN parameter {} specified".format(key)
                )

        # For custom aerosols, specify final extinction and absorption
        # MODTRAN 6.0 convention treats negative visibility as AOT550
        if hasattr(self, "aer_absc"):
            total_aot = fracs.sum()
            recursive_replace(param, "VIS", -total_aot)
            total_extc = self.aer_extc.T.dot(fracs)
            total_absc = self.aer_absc.T.dot(fracs)
            norm_fracs = fracs / (fracs.sum())
            total_asym = self.aer_asym.T.dot(norm_fracs)

            # Normalize to 550 nm
            total_extc550 = scipy.interpolate.interp1d(self.aer_wl, total_extc)(0.55)
            lvl0 = param[0]["MODTRANINPUT"]["AEROSOLS"]["IREGSPC"][0]
            lvl0["NARSPC"] = len(self.aer_wl)
            lvl0["VARSPC"] = [float(v) for v in self.aer_wl]
            lvl0["ASYM"] = [float(v) for v in total_asym]
            lvl0["EXTC"] = [float(v) / total_extc550 for v in total_extc]
            lvl0["ABSC"] = [float(v) / total_extc550 for v in total_absc]

        if self.multipart_transmittance:
            case_count = 0
            for albedo in self.test_rfls:
                for band_name, wvl_set in zip(
                    self.wavelength_models, self.simulation_wavelength_regions
                ):
                    case_param = deepcopy(param[0])
                    case_param["MODTRANINPUT"]["CASE"] = case_count
                    case_param["MODTRANINPUT"]["SURFACE"]["SURREF"] = albedo
                    case_param["MODTRANINPUT"]["SPECTRAL"]["V1"] = wvl_set[0]
                    case_param["MODTRANINPUT"]["SPECTRAL"]["V2"] = wvl_set[1]
                    case_param["MODTRANINPUT"]["SPECTRAL"]["BMNAME"] = band_name

                    # We don't need a .chn file if we're writing a tp7!
                    # Delete it. And set the DV and FWHM parameters to something
                    # arbitrarily high
                    if "FILTNM" in case_param["MODTRANINPUT"]["SPECTRAL"]:
                        del case_param["MODTRANINPUT"]["SPECTRAL"]["FILTNM"]

                    for dp in ["DV", "FWHM"]:
                        if dp in case_param["MODTRANINPUT"]["SPECTRAL"]:
                            case_param["MODTRANINPUT"]["SPECTRAL"][dp] = (
                                1.0 / self.min_samples_per_nm
                            ) * 2.0

                    if case_count == 0:
                        param[0] = case_param
                    else:
                        param.append(case_param)
                    case_count += 1

        return json.dumps({"MODTRAN": param}, cls=SerialEncoder, indent=2), param

    def check_modtran_water_upperbound(self) -> float:
        """Check to see what the max water vapor values is at the first point in the LUT

        Returns:
            float: max water vapor value, or None if test fails
        """
        point = np.array([x[-1] for x in self.lut_grids])

        # Set the H2OSTR value as arbitrarily high - 50 g/cm2 in this case
        point[self.lut_names.index("H2OSTR")] = 50

        filebase = os.path.join(self.sim_path, "H2O_bound_test")
        self.makeSim(point, filebase)

        max_water = None
        with open(
            os.path.join(self.sim_path, filebase + ".tp6"), errors="ignore"
        ) as tp6file:
            for count, line in enumerate(tp6file):
                if "The water column is being set to the maximum" in line:
                    max_water = line.split(",")[1].strip()
                    max_water = float(max_water.split(" ")[0])
                    break

        return max_water

    def required_results_exist(self, filename_base):
        infilename = os.path.join(self.sim_path, "LUT_" + filename_base + ".json")
        outchnname = os.path.join(self.sim_path, filename_base + ".chn")
        outtp6name = os.path.join(self.sim_path, filename_base + ".tp6")
        outtp7name = os.path.join(self.sim_path, filename_base + ".csv")

        if (
            os.path.isfile(infilename)
            and os.path.isfile(outchnname)
            or os.path.isfile(outtp7name)
            and os.path.isfile(outtp6name)
        ):
            return True
        else:
            return False

    def wl2flt(self, wavelengths: np.array, fwhms: np.array, outfile: str) -> None:
        """Helper function to generate Gaussian distributions around the
        center wavelengths.

        Args:
            wavelengths: wavelength centers
            fwhms: full width at half max
            outfile: file to write to

        """

        sigmas = fwhms / 2.355
        span = 2.0 * np.abs(wavelengths[1] - wavelengths[0])  # nm
        steps = 101

        with open(outfile, "w") as fout:
            fout.write("Nanometer data for sensor\n")
            for wl, fwhm, sigma in zip(wavelengths, fwhms, sigmas):
                ws = wl + np.linspace(-span, span, steps)
                vs = scipy.stats.norm.pdf(ws, wl, sigma)
                vs = vs / vs[int(steps / 2)]
                wns = 10000.0 / (ws / 1000.0)

                fout.write("CENTER:  %6.2f NM   FWHM:  %4.2f NM\n" % (wl, fwhm))

                for w, v, wn in zip(ws, vs, wns):
                    fout.write(" %9.4f %9.7f %9.2f\n" % (w, v, wn))


class SerialEncoder(json.JSONEncoder):
    """Encoder for json to help ensure json objects can be passed to the workflow manager."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(SerialEncoder, self).default(obj)
