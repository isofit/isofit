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
from copy import deepcopy
from sys import platform

import numpy as np
import scipy.interpolate
import scipy.stats

from isofit.atmosphere.atmosphere import BaseAtmosphere
from isofit.core import units
from isofit.core.common import json_load_ascii, recursive_replace
from isofit.luts.writer import Writer
from isofit.core.units import modtran_rdn_in_nm

Logger = logging.getLogger(__file__)

TROPOPAUSE_ALTITUDE_KM = 17.0


class ModtranRT(BaseAtmosphere, Writer):

    albedos = [0.0, 0.1, 0.5]

    def __init__(
        self, engine_config, min_samples_per_nm=10, max_samples_per_nm=100, **kwargs
    ):
        self.max_buffer_time = 0.5
        self.resolutions_available = [0.1, 1, 5, 15]
        self.resolution_names = ["p1_2013", "01_2013", "05_2013", "15_2013"]
        self.min_samples_per_nm = min_samples_per_nm
        self.max_samples_per_nm = max_samples_per_nm

        self.use_tp7 = False

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
        irr = units.L_to_E(units.W_to_uW(tokens[18]) / tokens[8], coszen)  # uW/nm/cm2

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
            'rhoatm'             : units.rdn_to_transm(units.W_to_uW(tokens[4]), coszen, irr), # unitless
            'width'              : tokens[8],
            'thermal_upwelling'  : units.W_to_uW((tokens[11] + tokens[12]) / tokens[8]), # uW/nm/sr/cm2
            'thermal_downwelling': units.W_to_uW(tokens[16]) / tokens[8],
            'path_rdn'           : units.W_to_uW(tokens[14]) + units.W_to_uW(tokens[15]), # The sum of the (1) single scattering and (2) multiple scattering
            'grnd_rflt'          : units.W_to_uW(tokens[16]),        # ground reflected radiance (direct+diffuse+multiple scattering)
            'drct_rflt'          : units.W_to_uW(tokens[17]),        # same as 16 but only on the sun->surface->sensor path (only direct)
            'transm_down_dif'    : tokens[21] + tokens[22],  # total transmittance (down * up, direct + diffuse)
            'sphalb'             : tokens[23],  # atmospheric spherical albedo
            'transm_up_dir'      : tokens[24],  # upward direct transmittance
            'albedo'             : np.round(1-tokens[25], 3)
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

        # Read each block of data
        header_pattern = "1ST SPECTRAL"
        header_indices = [i for i, line in enumerate(lines) if header_pattern in line]
        data = []
        for idx, start_idx in enumerate(header_indices):
            end_idx = (
                header_indices[idx + 1] if idx + 1 < len(header_indices) else len(lines)
            )
            block = lines[start_idx + header : end_idx]
            data.append(block)

        # Parse every block into a dictionary of numpy arrays
        parsed_blocks = []
        for lines in data:
            parsed = [
                self.parseTokens(self.parseLine(line), coszen)
                for line in lines
                if line.strip()
            ]

            combined = {}
            for i, parse in enumerate(parsed):
                for key, value in parse.items():
                    values = combined.setdefault(key, np.full(len(parsed), np.nan))
                    values[i] = value

            combined["mean_albedo"] = np.mean(combined["albedo"])
            parsed_blocks.append(combined)

        # Loop over all possible albedos and merge band models if needed
        # NOTE excluding zero case here
        parts = []
        for a in self.albedos[1:]:

            # Matching the albedo to a small tolerance, generally just need to determine if we
            # are a 0.1 or a 0.5 albedo so this atol=0.01 should be sufficent.
            matching_blocks = [
                b for b in parsed_blocks if np.isclose(b["mean_albedo"], a, atol=0.01)
            ]

            # For the case with only zero albedo
            if not matching_blocks:
                continue

            product_names = [k for k in matching_blocks[0].keys() if k != "mean_albedo"]

            # This assumes the lower wavelength region (and highest fidelity model) comes first
            # when sorting the blocks. This is true for the CHN outputs and is how we write the input in json.
            merged = {}
            for prod in product_names:
                merged[prod] = np.hstack([b[prod] for b in matching_blocks])

            order1 = np.argsort(merged["wl"])
            order2 = np.unique(merged["wl"][order1], return_index=True)[1]

            # Then finally, apply the sorting logic to each product
            ref_shape = merged["wl"].shape
            for prod in product_names:
                if (
                    isinstance(merged[prod], np.ndarray)
                    and merged[prod].shape == ref_shape
                ):
                    merged[prod] = merged[prod][order1][order2]

            parts.append(merged)

        # TODO do we still want to support single transmittance MODTRAN runs?
        if not parts:
            chn = parsed_blocks[0]
        else:
            chn = parts[0]
            if len(parts) > 1:
                Logger.debug("Using two albedo method")
                chn = self.two_albedo_method(
                    case_0=parts[0],
                    case_1=parts[0],
                    case_2=parts[1],
                    coszen=coszen,
                    rfl_1=self.albedos[1],
                    rfl_2=self.albedos[2],
                )

        return chn

    def load_tp7(
        self, file_path: str, num_albedos: int, num_models: int, coszen: float
    ):
        """Read a MODTRAN TP7 file and return the data as a dictionary, one entry
        for each case in the file.  Don't do anything but read the data; all the data.

        Parameters
        ----------
        file_path: str
            Path to the MODTRAN TP7 file
        num_albedos: int
            Number of unique albedos
        num_models: int
            Number of models per albedo
        coszen: float
            cosine of the solar zenith angle (TOA)

        Returns
        -------
        dict
            Cases dictionary reordered by model then albedo.
        """
        with open(file_path) as f:
            lines = f.readlines()

        case_indices = [i for i, line in enumerate(lines) if "case index" in line]
        end_indices = [i for i, line in enumerate(lines) if line.strip() == "}"]

        # Go through and grab cases in way saved from MODTRAN
        cases_data_raw = {}
        for case_num, (start, end) in enumerate(zip(case_indices, end_indices)):
            col1 = lines[start + 4].strip().split(",")
            col2 = lines[start + 5].strip().split(",")
            columns = [f"{n1.strip()} {n2.strip()}" for n1, n2 in zip(col1, col2)]

            data_lines = lines[start + 6 : end]
            cases_data_raw[case_num] = np.genfromtxt(
                data_lines, delimiter=",", names=columns
            )

        # Get the albedos based on emiss and sort them
        cases_list = list(cases_data_raw.values())
        all_direct_emiss = np.array(
            [
                (
                    case["direct_emiss"][0]
                    if case["direct_emiss"].ndim > 0
                    else case["direct_emiss"]
                )
                for case in cases_list
            ]
        )
        all_albedos = np.round(1 - all_direct_emiss, 3)
        sorted_indices = np.argsort(all_albedos)
        cases_list = [cases_list[i] for i in sorted_indices]

        # Raise exception of input is wrong or something else mysterious is happening in MODTRAN
        if len(cases_list) != num_albedos * num_models:
            raise ValueError(
                f"Cases length ({len(cases_list)})!= num_albedos ({num_albedos}) * num_models ({num_models})"
            )

        # Albedo groups by band model which can be organized by freq
        albedo_groups = [
            sorted(
                cases_list[i * num_models : (i + 1) * num_models],
                key=lambda c: c["Freq_cm1"].min(),
            )
            for i in range(num_albedos)
        ]

        # Reorder: model-major, then albedo-minor, matching the merge multi band method
        cases_data = {
            m * num_albedos + a: albedo_groups[a][m]
            for m in range(num_models)
            for a in range(num_albedos)
        }
        product_names = cases_data[0].dtype.names

        # Merge between albedo and case cases
        merged_dicts = [{} for _ in range(num_albedos)]
        for _m in range(len(merged_dicts)):
            for product in product_names:
                merged_dicts[_m][product] = []
        case_count = 0
        for _n in range(num_models):
            for _a in range(num_albedos):
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
                    merged_dicts[_m][product] = modtran_rdn_in_nm(
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

            # Only need to run two_albedo method if we have multiple cases
            # still at this point.  Note that at this time, merge_multiresolution_cases
            # is set up to NEED to run through two_albedo_method, but this might
            # not always be the case.
            if len(case_output_dict) == 2:
                params = self.two_albedo_method(
                    case_0=case_output_dict[
                        0
                    ],  # TODO double check we don't actually use the zero case
                    case_1=case_output_dict[0],
                    case_2=case_output_dict[1],
                    coszen=coszen,
                    rfl_1=self.albedos[1],
                    rfl_2=self.albedos[2],
                )

        return params

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

    def preSim(self):
        """
        Post-initialized, pre-simulation setup
        """
        # Track the MODTRAN directory used in the LUT attributes
        self.lut.setAttr("MODTRAN", str(self.engine_base_dir))

        self.filtpath = os.path.join(
            self.sim_path,
            f"wavelengths_{self.config.engine_name}_{self.wl[0]}_{self.wl[-1]}.flt",
        )
        self.template = json_load_ascii(self.config.template_file)["MODTRAN"]

        # Regenerate MODTRAN input wavelength file
        if not os.path.exists(self.filtpath):
            self.wl2flt(self.wl, self.fwhm, self.filtpath)

        # Insert aerosol templates, if specified
        if self.config.aerosol_model_file is not None:
            self.template[0]["MODTRANINPUT"]["AEROSOLS"] = json_load_ascii(
                self.config.aerosol_template_file
            )

        # Insert aerosol data, if specified
        if self.config.aerosol_model_file is not None:
            aer_data = np.loadtxt(self.config.aerosol_model_file)
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

        # If we have a coarser model that fully encapsulates a finer model,
        # discarde the finer model
        if len(self.simulation_wavelength_regions) >= 2:
            for i in range(len(self.simulation_wavelength_regions) - 1, -1, -1):
                if (
                    self.simulation_wavelength_regions[i][0]
                    >= self.simulation_wavelength_regions[i - 1][0]
                    and self.simulation_wavelength_regions[i][1]
                    <= self.simulation_wavelength_regions[i - 1][1]
                ):
                    self.simulation_wavelength_regions.pop(i)

        # Don't overlap by more than 1 nm, and prioritize coarser resolution models for comp when we can:
        if len(self.simulation_wavelength_regions) >= 2:
            for i in range(len(self.simulation_wavelength_regions) - 2, -1, -1):
                self.simulation_wavelength_regions[i][0] = (
                    self.simulation_wavelength_regions[i + 1][1] - 1
                )

        if self.simulation_wavelength_regions[-1][0] > np.min(self.wl) - self.fwhm[0]:
            self.simulation_wavelength_regions[-1][0] = np.min(self.wl) - self.fwhm[0]
            logging.info(
                "Adjusted first wavelength region to start at the minimum wavelength."
            )
        if self.simulation_wavelength_regions[0][-1] < np.max(self.wl) + self.fwhm[-1]:
            self.simulation_wavelength_regions[0][-1] = np.max(self.wl) + self.fwhm[-1]
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
        coszen = np.cos(np.deg2rad(solzen))
        if os.path.isfile(f"{file}.csv"):
            params = self.load_tp7(
                file_path=f"{file}.csv",
                num_albedos=len(self.albedos[1:]),
                num_models=len(self.simulation_wavelength_regions),
                coszen=coszen,
            )
        else:
            params = self.load_chn(file=f"{file}.chn", coszen=coszen, header=5)
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

        if self.config.configure_and_exit:
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
                        np.array(altitudes) <= TROPOPAUSE_ALTITUDE_KM, val, 0
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
                            prof_altitude <= TROPOPAUSE_ALTITUDE_KM,
                            prof_temperature + val,
                            prof_temperature,
                        )
                        param[0]["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"][
                            ind_prof_temperature
                        ]["PROFILE"] = prof_temperature.tolist()

                    else:
                        # If a temperature profile does not exist, then use UNT_TDELTA_KELVIN
                        prof_unt_tdelta_kelvin = np.where(
                            prof_altitude <= TROPOPAUSE_ALTITUDE_KM, val, 0.0
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
            # Ignoring the zero albedo case because it is unused in 2-albedo method
            for albedo in self.albedos[1:]:
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
                    if self.use_tp7:
                        if "FILTNM" in case_param["MODTRANINPUT"]["SPECTRAL"]:
                            del case_param["MODTRANINPUT"]["SPECTRAL"]["FILTNM"]

                        for dp in ["DV", "FWHM"]:
                            if dp in case_param["MODTRANINPUT"]["SPECTRAL"]:
                                case_param["MODTRANINPUT"]["SPECTRAL"][dp] = (
                                    1.0 / self.min_samples_per_nm
                                ) * 2.0

                    # else we need to make sure not to write the csv file
                    else:
                        if "NOFILE" in case_param["MODTRANINPUT"]["FILEOPTIONS"]:
                            del case_param["MODTRANINPUT"]["FILEOPTIONS"]["NOFILE"]

                        if "CSVPRNT" in case_param["MODTRANINPUT"]["FILEOPTIONS"]:
                            del case_param["MODTRANINPUT"]["FILEOPTIONS"]["CSVPRNT"]

                        case_param["MODTRANINPUT"]["FILEOPTIONS"]["CKPRNT"] = True

                    if case_count == 0:
                        param[0] = case_param
                    else:
                        param.append(case_param)
                    case_count += 1

        return json.dumps({"MODTRAN": param}, cls=SerialEncoder, indent=2), param

    def required_results_exist(self, filename_base):
        infilename = os.path.join(self.sim_path, "LUT_" + filename_base + ".json")
        outchnname = os.path.join(self.sim_path, filename_base + ".chn")
        outtp6name = os.path.join(self.sim_path, filename_base + ".tp6")

        if (
            os.path.isfile(infilename)
            and os.path.isfile(outchnname)
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
                wns = units.nm_to_wavenumber(ws)

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
