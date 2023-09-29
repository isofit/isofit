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

from isofit.configs import Config
from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.core.geometry import Geometry
from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine

from ..core.common import VectorInterpolator, json_load_ascii, recursive_replace
from ..radiative_transfer.look_up_tables import FileExistsError, TabularRT

### Variables ###

eps = 1e-5  # used for finite difference derivative calculations
tropopause_altitude_km = 17.0

### Classes ###


class ModtranRT(RadiativeTransferEngine):
    """A model of photon transport including the atmosphere."""

    def __init__(
        self,
        engine_config: RadiativeTransferEngineConfig,
        interpolator_style: str,
        instrument_wavelength_file: str = None,
        overwrite_interpolator: bool = False,
        cache_size: int = 16,
        lut_grid: dict = None,
        build_lut: bool = True,
    ):
        """."""

        super().__init__(
            engine_config,
            interpolator_style,
            instrument_wavelength_file,
            overwrite_interpolator,
            cache_size,
            lut_grid,
        )

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

    def load_tp6(self, point):
        """Load a '.tp6' file. This contains the solar geometry. We
        Return cosine of mean solar zenith."""

        filename_base = self.point_to_filename(point)
        tp6_file = os.path.join(self.lut_dir, filename_base + ".tp6")
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

    def load_chn(self, point):
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

        filename_base = self.point_to_filename(point)
        chn_file = os.path.join(self.lut_dir, filename_base + ".chn")
        coszen = self.get_coszen(point)

        with open(chn_file) as f:
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
                solar_irr = float(toks[18]) * 1e6 * np.pi / wid / coszen  # uW/nm/sr/cm2
                rdnatm = float(toks[4]) * 1e6  # uW/nm/sr/cm2
                rhoatm = rdnatm * np.pi / (solar_irr * coszen)
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
                    sols.append(solar_irr)  # solar irradiance
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
            """
            This implementation is following Gaunter et al. (2009) (DOI:10.1080/01431160802438555),
            and modified by Nimrod Carmon. It is called the "2-albedo" method, referring to running
            modtran with 2 different surface albedos. The 3-albedo method is similar to this one with
            the single difference where the "path_radiance_no_surface" variable is taken from a
            zero-surface-reflectance modtran run instead of being calculated from 2 modtran outputs.
            There are a few argument as to why this approach is beneficial:
            (1) for each grid point on the lookup table you sample modtran 2 or 3 times, i.e. you get
            2 or 3 "data points" for the atmospheric parameter of interest. This in theory allows us
            to use a lower band model resolution modtran run, which is much faster, while keeping
            high accuracy. Currently we have the 5 cm-1 band model resolution configured.
            The second advantage is the possibility to use the decoupled transmittance products to exapnd
            the forward model and account for more physics e.g. shadows \ sky view \ adjacency \ terrain etc.

            """
            t_up_dirs = np.array(transups)
            direct_ground_reflected_1 = np.array(drct_rflts_1)
            total_ground_reflected_1 = np.array(grnd_rflts_1)
            direct_ground_reflected_2 = np.array(drct_rflts_2)
            total_ground_reflected_2 = np.array(grnd_rflts_2)
            path_radiance_1 = np.array(lp_1)
            path_radiance_2 = np.array(lp_2)
            TOA_Irad = np.array(sols) * coszen / np.pi
            rfl_1 = self.test_rfls[1]
            rfl_2 = self.test_rfls[2]
            mus = coszen

            direct_flux_1 = direct_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs
            global_flux_1 = total_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs
            diffuse_flux_1 = global_flux_1 - direct_flux_1  # diffuse flux

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
            direct_flux_radiance = direct_flux_1 / mus

            global_flux_no_surface = global_flux_1 * (1.0 - rfl_1 * sphalbs)
            diffuse_flux_no_surface = (
                global_flux_no_surface - direct_flux_radiance * coszen
            )

            global_flux_no_surface = global_flux_1 * (1.0 - rfl_1 * sphalbs)
            diffuse_flux_no_surface = (
                global_flux_no_surface - direct_flux_radiance * coszen
            )

            t_down_dirs = (direct_flux_radiance * coszen / widths / np.pi) / TOA_Irad
            t_down_difs = (diffuse_flux_no_surface / widths / np.pi) / TOA_Irad

            # total transmittance
            transms = (t_down_dirs + t_down_difs) * (t_up_dirs + t_up_difs)

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

    def ext550_to_vis(self, ext550):
        """."""

        return np.log(50.0) / (ext550 + 0.01159)

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
        elif param[0]["MODTRANINPUT"]["GEOMETRY"]["IPARM"] == 11 and set(
            ["solar_azimuth", "solaz", "solar_zenith", "solzen"]
        ).intersection(set(overrides.keys())):
            raise AttributeError(
                "Solar geometry (solar az/azimuth zen/zenith) is specified, but IPARM"
                " is set to 12.  Check MODTRAN template"
            )

        if set(["PARM1", "PARM2"]).intersection(set(overrides.keys())):
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
            # Here we copy the original config and just change the surface reflectance
            param[0]["MODTRANINPUT"]["CASE"] = 0
            param[0]["MODTRANINPUT"]["SURFACE"]["SURREF"] = self.test_rfls[0]
            param1 = deepcopy(param[0])
            param1["MODTRANINPUT"]["CASE"] = 1
            param1["MODTRANINPUT"]["SURFACE"]["SURREF"] = self.test_rfls[1]
            param.append(param1)
            param2 = deepcopy(param[0])
            param2["MODTRANINPUT"]["CASE"] = 2
            param2["MODTRANINPUT"]["SURFACE"]["SURREF"] = self.test_rfls[2]
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

    def load_rt(self, point):
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
            "sol",
            "rhoatm",
            "transm",
            "sphalb",
            "transup",
            "t_down_dir",
            "t_down_dif",
            "t_up_dir",
            "t_up_dif",
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
