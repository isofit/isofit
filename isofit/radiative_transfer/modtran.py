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

from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine

from ..core.common import json_load_ascii, recursive_replace
from ..radiative_transfer.look_up_tables import FileExistsError

Logger = logging.getLogger(__file__)

### Variables ###

eps = 1e-5  # used for finite difference derivative calculations
tropopause_altitude_km = 17.0

### Classes ###


class ModtranRT(RadiativeTransferEngine):
    """A model of photon transport including the atmosphere."""

    def __init__(self, engine_config: RadiativeTransferEngineConfig, **kwargs):
        """."""

        super().__init__(engine_config, **kwargs)

        flt_name = "wavelengths_{}_{}_{}.flt".format(
            engine_config.engine_name, self.wl[0], self.wl[-1]
        )
        self.filtpath = os.path.join(self.lut_dir, flt_name)
        self.template = deepcopy(
            json_load_ascii(engine_config.template_file)["MODTRAN"]
        )

        # Insert aerosol templates, if specified
        if engine_config.aerosol_model_file is not None:
            self.template[0]["MODTRANINPUT"]["AEROSOLS"] = deepcopy(
                json_load_ascii(engine_config.aerosol_template_file)
            )

        # Insert aerosol data, if specified
        if engine_config.aerosol_model_file is not None:
            aer_data = np.loadtxt(engine_config.aerosol_model_file)
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

        # Determine whether we are using the three run or single run strategy
        self.multipart_transmittance = engine_config.multipart_transmittance

        self.last_point_looked_up = np.zeros(self.n_point)
        self.last_point_lookup_values = np.zeros(self.n_point)

    def load_tp6(self, tp6_file):
        """Load a '.tp6' file. This contains the solar geometry. We
        Return cosine of mean solar zenith."""

        with open(tp6_file, "r") as f:
            ts, te = -1, -1  # start and end indices
            lines = []
            while len(lines) == 0 or len(lines[-1]) > 0:
                try:
                    lines.append(f.readline())
                except UnicodeDecodeError:
                    pass

            for i, line in enumerate(lines):
                if "SINGLE SCATTER SOLAR" in line:
                    ts = i + 5
                if ts >= 0 and len(line) < 5:
                    te = i
                    break
            if ts < 0:
                logging.error("%s is missing solar geometry" % tp6_file)
                raise ValueError("%s is missing solar geometry" % tp6_file)
        szen = np.array([float(lines[i].split()[3]) for i in range(ts, te)]).mean()
        return szen

    def load_chn(self, chnfile, coszen):
        """Load a '.chn' output file and parse critical coefficient vectors.

           These are:
             * wl      - wavelength vector
             * sol_irr - solar irradiance
             * sphalb  - spherical sky albedo at surface
             * transm  - diffuse and direct irradiance along the
                          sun-ground-sensor path
             * transup - transmission along the ground-sensor path only

           If the "multipart transmittance" option is active, we will use
           a combination of three MODTRAN runs to estimate the following
           additional quantities:
             * t_down_dir - direct downwelling transmittance
             * t_down_dif - diffuse downwelling transmittance
             * t_up_dir   - direct upwelling transmittance
             * t_up_dif   - diffuse upwelling transmittance

        If the "multipart transmittance" option is active, we will use
        a combination of three MODTRAN runs to estimate the following
        additional quantities:
          * t_down_dir - direct downwelling transmittance
          * t_down_dif - diffuse downwelling transmittance
          * t_up_dir   - direct upwelling transmittance
          * t_up_dif   - diffuse upwelling transmittance

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         Be careful with these! They are to be used only by the
         modtran_tir functions because MODTRAN must be run with a
         reflectivity of 1 for them to be used in the RTM defined
         in radiative_transfer.py.

         * thermal_upwelling - atmospheric path radiance
         * thermal_downwelling - sky-integrated thermal path radiance
             reflected off the ground and back into the sensor.

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        We parse them one wavelength at a time."""
        with open(chnfile) as f:
            sols, transms, sphalbs, wls, rhoatms, transups = [], [], [], [], [], []
            t_down_dirs, t_down_difs, t_up_dirs, t_up_difs = [], [], [], []
            grnd_rflts_1, drct_rflts_1, grnd_rflts_2, drct_rflts_2 = [], [], [], []
            transm_dirs, transm_difs, widths = [], [], []
            lp_0, lp_1, lp_2 = [], [], []
            thermal_upwellings, thermal_downwellings = [], []
            lines = f.readlines()
            nheader = 5

            # Mark header and data segments
            nwl = len(self.wl)
            case = -np.ones(nheader * 3 + nwl * 3)
            case[nheader : (nheader + nwl)] = 0
            case[(nheader * 2 + nwl) : (nheader * 2 + nwl * 2)] = 1
            case[(nheader * 3 + nwl * 2) : (nheader * 3 + nwl * 3)] = 2

            for i, line in enumerate(lines):
                # exclude headers
                if case[i] < 0:
                    continue

                # Columns 1 and 2 can touch for large datasets.
                # Since we don't care about the values, we overwrite the
                # character to the left of column 1 with a space so that
                # we can use simple space-separated parsing later and
                # preserve data indices.
                line = line[:17] + " " + line[18:]

                # parse data out of each line in the MODTRAN output
                toks = re.findall(r"[\S]+", line.strip())
                wl, wid = float(toks[0]), float(toks[8])  # nm
                self.solar_irr = (
                    float(toks[18]) * 1e6 * np.pi / wid / coszen
                )  # uW/nm/sr/cm2
                rdnatm = float(toks[4]) * 1e6  # uW/nm/sr/cm2
                rhoatm = rdnatm * np.pi / (self.solar_irr * coszen)
                sphalb = float(toks[23])
                A_coeff = float(toks[21])
                B_coeff = float(toks[22])
                transm = A_coeff + B_coeff
                transup = float(toks[24])

                # Be careful with these! See note in function comments above
                thermal_emission = float(toks[11])
                thermal_scatter = float(toks[12])
                thermal_upwelling = (
                    (thermal_emission + thermal_scatter) / wid * 1e6
                )  # uW/nm/sr/cm2

                # Be careful with these! See note in function comments above
                # grnd_rflt already includes ground-to-sensor transmission
                grnd_rflt = (
                    float(toks[16]) * 1e6
                )  # ground reflected radiance (direct+diffuse+multiple scattering)
                drct_rflt = (
                    float(toks[17]) * 1e6
                )  # same as 16 but only on the sun->surface->sensor path (only direct)
                path_rdn = (
                    float(toks[14]) * 1e6 + float(toks[15]) * 1e6
                )  # The sum of the (1) single scattering and (2) multiple scattering
                thermal_downwelling = grnd_rflt / wid  # uW/nm/sr/cm2

                if case[i] == 0:
                    sols.append(self.solar_irr)  # solar irradiance
                    transms.append(transm)  # total transmittance
                    sphalbs.append(sphalb)  # spherical albedo
                    rhoatms.append(rhoatm)  # atmospheric reflectance
                    transups.append(transup)  # upwelling direct transmittance
                    transm_dirs.append(A_coeff)  # total direct transmittance
                    transm_difs.append(B_coeff)  # total diffuse transmittance
                    widths.append(wid)  # channel width in nm
                    lp_0.append(path_rdn)  # path radiance of zero surface reflectance
                    thermal_upwellings.append(thermal_upwelling)
                    thermal_downwellings.append(thermal_downwelling)
                    wls.append(wl)  # wavelengths in nm

                elif case[i] == 1:
                    grnd_rflts_1.append(grnd_rflt)  # total ground reflected radiance
                    drct_rflts_1.append(
                        drct_rflt
                    )  # direct path ground reflected radiance
                    lp_1.append(
                        path_rdn
                    )  # path radiance (sum of single and multiple scattering)

                elif case[i] == 2:
                    grnd_rflts_2.append(grnd_rflt)  # total ground reflected radiance
                    drct_rflts_2.append(
                        drct_rflt
                    )  # direct path ground reflected radiance
                    lp_2.append(
                        path_rdn
                    )  # path radiance (sum of single and multiple scattering)

        if self.multipart_transmittance:
            (
                transms,
                t_down_dirs,
                t_down_difs,
                t_up_dirs,
                t_up_difs,
                sphalbs,
            ) = self.two_albedo_method(
                transups=transups,
                drct_rflts_1=drct_rflts_1,
                grnd_rflts_1=grnd_rflts_1,
                grnd_rflts_2=grnd_rflts_2,
                lp_1=lp_1,
                lp_2=lp_2,
                coszen=coszen,
                widths=widths,
            )

        params = [
            np.array(i)
            for i in [
                wls,
                sols,
                rhoatms,
                transms,
                sphalbs,
                transups,
                t_down_dirs,
                t_down_difs,
                t_up_dirs,
                t_up_difs,
                thermal_upwellings,
                thermal_downwellings,
            ]
        ]

        return tuple(params)

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
                " is set to 12.  Check MODTRAN template"
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
            elif key in ["GNDALT"]:
                param[0]["MODTRANINPUT"]["SURFACE"][key] = val

            elif key in ["solar_azimuth", "solaz"]:
                if "TRUEAZ" not in param[0]["MODTRANINPUT"]["GEOMETRY"]:
                    raise AttributeError(
                        "Cannot have solar azimuth in LUT without specifying TRUEAZ. "
                        " Use RELAZ instead."
                    )
                param[0]["MODTRANINPUT"]["GEOMETRY"]["PARM1"] = (
                    param[0]["MODTRANINPUT"]["GEOMETRY"]["TRUEAZ"] - val + 180
                )

            elif key in ["solar_zenith", "solzen"]:
                param[0]["MODTRANINPUT"]["GEOMETRY"]["PARM2"] = abs(val)

            # elif key in ['altitude_km']

            # elif key in ['altitude_km']

            elif key in ["DISALB", "NAME"]:
                recursive_replace(param, key, val)
            elif key in param[0]["MODTRANINPUT"]["ATMOSPHERE"].keys():
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
            const_rfl = np.array(np.array(self.test_rfls) * 100, dtype=int)
            # Here we copy the original config and just change the surface reflectance
            param[0]["MODTRANINPUT"]["CASE"] = 0
            param[0]["MODTRANINPUT"]["SURFACE"]["SURFP"][
                "CSALB"
            ] = f"LAMB_CONST_{const_rfl[0]}_PCT"
            param1 = deepcopy(param[0])
            param1["MODTRANINPUT"]["CASE"] = 1
            param1["MODTRANINPUT"]["SURFACE"]["SURFP"][
                "CSALB"
            ] = f"LAMB_CONST_{const_rfl[1]}_PCT"
            param.append(param1)
            param2 = deepcopy(param[0])
            param2["MODTRANINPUT"]["CASE"] = 2
            param2["MODTRANINPUT"]["SURFACE"]["SURFP"][
                "CSALB"
            ] = f"LAMB_CONST_{const_rfl[2]}_PCT"
            param.append(param2)

        return json.dumps({"MODTRAN": param}), param

    def check_modtran_water_upperbound(self) -> float:
        """Check to see what the max water vapor values is at the first point in the LUT

        Returns:
            float: max water vapor value, or None if test fails
        """
        point = np.array([x[-1] for x in self.lut_grids])

        # Set the H2OSTR value as arbitrarily high - 50 g/cm2 in this case
        point[self.lut_names.index("H2OSTR")] = 50

        filebase = os.path.join(self.lut_dir, "H2O_bound_test")
        cmd = self.rebuild_cmd(point, filebase)

        # Run MODTRAN for up to 10 seconds - this should be plenty of time
        if os.path.isdir(self.lut_dir) is False:
            os.mkdir(self.lut_dir)
        try:
            subprocess.call(cmd, shell=True, timeout=10, cwd=self.lut_dir)
        except:
            pass

        max_water = None
        with open(
            os.path.join(self.lut_dir, filebase + ".tp6"), errors="ignore"
        ) as tp6file:
            for count, line in enumerate(tp6file):
                if "The water column is being set to the maximum" in line:
                    max_water = line.split(",")[1].strip()
                    max_water = float(max_water.split(" ")[0])
                    break

        return max_water

    def rebuild_cmd(self, point):
        """."""

        filename_base = self.point_to_filename(point)

        vals = dict([(n, v) for n, v in zip(self.lut_names, point)])
        vals["DISALB"] = True
        vals["NAME"] = filename_base
        vals["FILTNM"] = os.path.normpath(self.filtpath)
        modtran_config_str, modtran_config = self.modtran_driver(dict(vals))

        # Check rebuild conditions: LUT is missing or from a different config
        infilename = "LUT_" + filename_base + ".json"
        infilepath = os.path.join(self.lut_dir, "LUT_" + filename_base + ".json")

        if not self.required_results_exist(filename_base):
            rebuild = True
        else:
            # We compare the two configuration files, ignoring names and
            # wavelength paths which tend to be non-portable
            with open(infilepath, "r") as fin:
                current_config = json.load(fin)["MODTRAN"]
                current_config[0]["MODTRANINPUT"]["NAME"] = ""
                modtran_config[0]["MODTRANINPUT"]["NAME"] = ""
                current_config[0]["MODTRANINPUT"]["SPECTRAL"]["FILTNM"] = ""
                modtran_config[0]["MODTRANINPUT"]["SPECTRAL"]["FILTNM"] = ""
                current_str = json.dumps(current_config)
                modtran_str = json.dumps(modtran_config)
                rebuild = modtran_str.strip() != current_str.strip()

        if not rebuild:
            raise FileExistsError("File exists")

        # write_config_file
        with open(infilepath, "w") as f:
            f.write(modtran_config_str)

        # Specify location of the proper MODTRAN 6.0 binary for this OS
        xdir = {"linux": "linux", "darwin": "macos", "windows": "windows"}

        # Generate the CLI path
        cmd = os.path.join(
            self.engine_base_dir, "bin", xdir[platform], "mod6c_cons " + infilename
        )
        return cmd

    def required_results_exist(self, point):

        filename_base = self.point_to_filename(point)
        infilename = os.path.join(self.lut_dir, "LUT_" + filename_base + ".json")
        outchnname = os.path.join(self.lut_dir, filename_base + ".chn")
        outtp6name = os.path.join(self.lut_dir, filename_base + ".tp6")

        if (
            os.path.isfile(infilename)
            and os.path.isfile(outchnname)
            and os.path.isfile(outtp6name)
        ):
            return True
        else:
            return False

    def read_simulation_results(self, point):
        """."""

        file_basename = self.point_to_filename(point)
        tp6file = os.path.join(self.lut_dir, file_basename + ".tp6")
        solzen = self.load_tp6(tp6file)
        coszen = np.cos(solzen * np.pi / 180.0)

        chnfile = os.path.join(self.lut_dir, file_basename + ".chn")
        params = self.load_chn(chnfile, coszen)

        # Be careful with the two thermal values! They can only be used in
        # the modtran_tir functions as they require the modtran reflectivity
        # be set to 1 in order to use them in the RTM in radiative_transfer.py.
        # Don't add these to the VSWIR functions!
        names = [
            "wl",
            "solar_irr",
            "rhoatm",
            "transm",
            "sphalb",
            "transup",
            "transm_down_dir",
            "transm_down_dif",
            "transm_up_dir",
            "transm_up_dif",
        ]

        # Don't include the thermal terms in VSWIR runs to avoid incorrect usage
        if self.treat_as_emissive:
            names = names + ["thermal_upwelling", "thermal_downwelling"]

        results_dict = {name: param for name, param in zip(names, params)}
        results_dict["solzen"] = solzen
        results_dict["coszen"] = coszen
        return results_dict

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

    def two_albedo_method(
        self,
        transups: list,
        drct_rflts_1: list,
        grnd_rflts_1: list,
        grnd_rflts_2: list,
        lp_1: list,
        lp_2: list,
        coszen: float,
        widths: list,
    ):
        """This implementation follows Guanter et al. (2009) (DOI:10.1080/01431160802438555),
        with modifications by Nimrod Carmon. It is called the "2-albedo" method, referring to running
        MODTRAN with 2 different surface albedos. Alternatively, one could also run the 3-albedo method,
        which is similar to this one with the single difference where the "path_radiance_no_surface"
        variable is taken from a zero-surface-reflectance MODTRAN run instead of being calculated from
        2 MODTRAN outputs.

        There are a few argument as to why the 2- or 3-albedo methods are beneficial:
        (1) for each grid point on the lookup table you sample MODTRAN 2 or 3 times, i.e., you get
        2 or 3 "data points" for the atmospheric parameter of interest. This in theory allows us
        to use a lower band model resolution for the MODTRAN run, which is much faster, while keeping
        high accuracy. (2) we use the decoupled transmittance products to expand
        the forward model and account for more physics, currently topography and glint.

        Args:
            transups:     upwelling direct transmittance
            drct_rflts_1: direct path ground reflected radiance for reflectance case 1
            grnd_rflts_1: total ground reflected radiance for reflectance case 1
            grnd_rflts_2: total ground reflected radiance for reflectance case 2
            lp_1:         path radiance (sum of single and multiple scattering) for reflectance case 1
            lp_2:         path radiance (sum of single and multiple scattering) for reflectance case 2
            coszen:       cosine of solar zenith angle
            widths:       fwhm of radiative transfer simulations

        Returns:
            transms:      total transmittance (downwelling * upwelling)
            t_down_dirs:  downwelling direct transmittance
            t_down_difs:  downwelling diffuse transmittance
            t_up_dirs:    upwelling direct transmittance
            t_up_difs:    upwelling diffuse transmittance
            sphalbs:      atmospheric spherical albedo
        """
        t_up_dirs = np.array(transups)
        direct_ground_reflected_1 = np.array(drct_rflts_1)
        total_ground_reflected_1 = np.array(grnd_rflts_1)
        total_ground_reflected_2 = np.array(grnd_rflts_2)
        path_radiance_1 = np.array(lp_1)
        path_radiance_2 = np.array(lp_2)
        # ToDo: get coszen from LUT and assign as attribute to self
        TOA_Irad = np.array(self.solar_irr) * coszen / np.pi
        rfl_1 = self.test_rfls[0]
        rfl_2 = self.test_rfls[1]

        direct_flux_1 = direct_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs
        global_flux_1 = total_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs

        global_flux_2 = total_ground_reflected_2 * np.pi / rfl_2 / t_up_dirs

        path_radiance_no_surface = (
            rfl_2 * path_radiance_1 * global_flux_2
            - rfl_1 * path_radiance_2 * global_flux_1
        ) / (rfl_2 * global_flux_2 - rfl_1 * global_flux_1)

        # Diffuse upwelling transmittance
        t_up_difs = (
            np.pi
            * (path_radiance_1 - path_radiance_no_surface)
            / (rfl_1 * global_flux_1)
        )

        # Spherical Albedo
        sphalbs = (global_flux_1 - global_flux_2) / (
            rfl_1 * global_flux_1 - rfl_2 * global_flux_2
        )
        direct_flux_radiance = direct_flux_1 / coszen

        global_flux_no_surface = global_flux_1 * (1.0 - rfl_1 * sphalbs)
        diffuse_flux_no_surface = global_flux_no_surface - direct_flux_radiance * coszen

        t_down_dirs = (
            direct_flux_radiance * coszen / np.array(widths) / np.pi
        ) / TOA_Irad
        t_down_difs = (diffuse_flux_no_surface / np.array(widths) / np.pi) / TOA_Irad

        # total transmittance
        transms = (t_down_dirs + t_down_difs) * (t_up_dirs + t_up_difs)

        return transms, t_down_dirs, t_down_difs, t_up_dirs, t_up_difs, sphalbs

    def make_simulation_call(self, point):
        ...


# Version 2.0 of ModtranRT that reimplements the loader functions in an easier to understand format


class ModtranRTv2(ModtranRT):
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
        irr = tokens[18] * 1e6 * np.pi / tokens[8] / coszen  # uW/nm/sr/cm2

        # fmt: off
        return {
            'solar_irr'          : irr,       # Solar irradiance
            'wl'                 : tokens[0], # Wavelength
            'rhoatm'             : tokens[4] * 1e6 * np.pi / (irr * coszen), # uW/nm/sr/cm2
            'width'              : tokens[8],
            'thermal_upwelling'  : (tokens[11] + tokens[12]) / tokens[8] * 1e6, # uW/nm/sr/cm2
            'thermal_downwelling': tokens[16] * 1e6 / tokens[8],
            'path_rdn'           : tokens[14] * 1e6 + tokens[15] * 1e6, # The sum of the (1) single scattering and (2) multiple scattering
            'grnd_rflt'          : tokens[16] * 1e6,        # ground reflected radiance (direct+diffuse+multiple scattering)
            'drct_rflt'          : tokens[17] * 1e6,        # same as 16 but only on the sun->surface->sensor path (only direct)
            'transm'             : tokens[21] + tokens[22], # Total (direct+diffuse) transmittance
            'sphalb'             : tokens[23], #
            'transup'            : tokens[24], #
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

    @staticmethod
    def two_albedo_method(
        p0: dict,
        p1: dict,
        p2: dict,
        coszen: float,
        rfl1: float = 0.1,
        rfl2: float = 0.5,
    ) -> dict:
        """
        Calculates split transmittance values from a multipart file using the
        two-albedo method. See notes for further detail

        Parameters
        ----------
        p0: dict
            Part 0 of the channel file
        p1: dict
            Part 1 of the channel file
        p2: dict
            Part 2 of the channel file
        coszen: float
            ...
        rfl1: float, defaults=0.1
            Reflectance scaler 1
        rfl2: float, defaults=0.5
            Reflectance scaler 2

        Returns
        -------
        data: dict
            Relevant information

        Notes
        -----
        This implementation follows Guanter et al. (2009)
        (DOI:10.1080/01431160802438555), modified by Nimrod Carmon. It is called
        the "2-albedo" method, referring to running MODTRAN with 2 different
        surface albedos. Alternatively, one could also run the 3-albedo method,
        which is similar to this one with the single difference where the
        "path_radiance_no_surface" variable is taken from a
        zero-surface-reflectance MODTRAN run instead of being calculated from
        2 MODTRAN outputs.

        There are a few argument as to why the 2- or 3-albedo methods are
        beneficial:
            (1) For each grid point on the lookup table you sample MODTRAN 2 or
                3 times, i.e., you get 2 or 3 "data points" for the atmospheric
                parameter of interest. This in theory allows us to use a lower
                band model resolution for the MODTRAN run, which is much faster,
                while keeping high accuracy.
            (2) We use the decoupled transmittance products to expand
                the forward model and account for more physics, currently
                topography and glint.
        """
        # Extract relevant columns
        widths = p0["width"]
        t_up_dirs = p0["transup"]

        # REVIEW: two_albedo_method-v1 used a single solar_irr value, but now we have an array of values
        # The last value in the new array is the same as the old v1, so for backwards compatibility setting that here
        toa_irad = p0["solar_irr"][-1] * coszen / np.pi

        # Calculate some fluxes
        directRflt1 = p1["drct_rflt"]
        groundRflt1 = p1["grnd_rflt"]

        directFlux1 = directRflt1 * np.pi / rfl1 / t_up_dirs
        globalFlux1 = groundRflt1 * np.pi / rfl1 / t_up_dirs

        # diffuseFlux = globalFlux1 - directFlux1 # Unused

        globalFlux2 = p2["grnd_rflt"] * np.pi / rfl2 / t_up_dirs

        # Path radiances
        rdn1 = p1["path_rdn"]
        rdn2 = p2["path_rdn"]

        # Path Radiance No Surface = prns
        val1 = rfl1 * globalFlux1  # TODO: Needs a better name
        val2 = rfl2 * globalFlux2  # TODO: Needs a better name
        prns = ((val2 * rdn1) - (val1 * rdn2)) / (val2 - val1)

        # Diffuse upwelling transmittance
        t_up_difs = np.pi * (rdn1 - prns) / (rfl1 * globalFlux1)

        # Spherical Albedo
        sphalbs = (globalFlux1 - globalFlux2) / (val1 - val2)
        dFluxRN = directFlux1 / coszen  # Direct Flux Radiance

        globalFluxNS = globalFlux1 * (1 - rfl1 * sphalbs)  # Global Flux No Surface
        diffusFluxNS = globalFluxNS - dFluxRN * coszen  # Diffused Flux No Surface

        t_down_dirs = dFluxRN * coszen / widths / np.pi / toa_irad
        t_down_difs = diffusFluxNS / widths / np.pi / toa_irad

        transms = (t_down_dirs + t_down_difs) * (t_up_dirs + t_up_difs)

        # Return some keys from the first part plus the new calculated keys
        pass_forward = [
            "wl",
            "rhoatm",
            "solar_irr",
            "thermal_upwelling",
            "thermal_downwelling",
        ]
        data = {
            "sphalb": sphalbs,
            "transm_up_dir": t_up_dirs,
            "transm_up_dif": t_up_difs,
            "transm_down_dir": t_down_dirs,
            "transm_down_dif": t_down_difs,
        }
        for key in pass_forward:
            data[key] = p0[key]

        return data

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

        # Checks if this is a multipart file, separate if so
        n = int(len(lines) / 3)
        if lines[1] == lines[n + 1]:
            if not self.multipart_transmittance:
                Logger.warning(
                    f"This file was detected to be a multipart transmittance but engine_config.multipart_transmittance is set to False: {file}"
                )
            else:
                # Parse the input into three parts
                # fmt: off
                data = [
                    lines[   :n  ][header:],
                    lines[  n:n*2][header:],
                    lines[n*2:   ][header:]
                ]
                # fmt: on
        elif self.multipart_transmittance:
            Logger.warning(
                f"This file was detected to be a single transmittance but engine_config.multipart_transmittance is set to True: {file}"
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
        chn = parts[0]
        if len(parts) > 1:
            Logger.debug("Using two albedo method")
            chn = self.two_albedo_method(*parts, coszen, *self.test_rfls)

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

    def read_simulation_results(self, point):
        """
        For a given point, parses the tp6 and chn file and returns the data
        """
        file = os.path.join(self.sim_path, self.point_to_filename(point))

        solzen = self.load_tp6(f"{file}.tp6")
        coszen = np.cos(solzen * np.pi / 180.0)
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

    def make_simulation_call(self, point):
        ...


ModtranRT = ModtranRTv2
