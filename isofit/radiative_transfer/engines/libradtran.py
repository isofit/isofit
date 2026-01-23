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
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import numpy as np

from isofit.core.fileio import IO
from isofit.core import common
from isofit.data import env
from isofit.core.common import (
    units,
    json_load_ascii,
    calculate_resample_matrix,
    resample_spectrum,
)
from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine


Logger = logging.getLogger(__name__)


class LibRadTranRT(RadiativeTransferEngine):

    def __init__(self, engine_config: RadiativeTransferEngineConfig, **kwargs):

        self.albedos = [0.0, 0.1, 0.5]
        self.wl_1 = 340.0
        self.wl_2 = 2510.0
        self.wl_res_out = 0.5

        # aerosol_default = rural type aerosol in the boundary layer,
        # background aerosol above 2km, spring-summer conditions, and a visibility of 50km.
        # NOTE: for radiance quantities libRadtran will always override DISORT to run with at least 16 streams.
        self.libradtran_inp_template = (
            "source solar {path_solar}\n"
            "wavelength {wl_1} {wl_2}\n"
            "spline {wl_1} {wl_2} {wl_res_out}\n"
            "albedo {albedo}\n"
            "atmosphere_file {libradtran_dir}/data/atmmod/{atmos}.dat\n"
            "umu {cos_vza}\n"
            "phi0 {saa_deg}\n"
            "phi {vaa_deg}\n"
            "sza {sza_deg}\n"
            "rte_solver disort\n"
            "number_of_streams {nstr}\n"
            "mol_modify O3 {o3_inp} DU\n"
            "mixing_ratio CO2 {co2_inp}\n"
            "mol_abs_param reptran {band_model}\n"
            "mol_modify H2O {h2o_mm} MM\n"
            "crs_model rayleigh bodhaine\n"
            "aerosol_default\n"
            "zout {zout}\n"
            "altitude {elev}\n"
            "output_quantity transmittance\n"
            "output_user lambda uu eglo edir\n"
        )

        self.sh_template = "#!/bin/bash\n" "cd {lrt_bin_dir}\n" "{uvspecs}\n"

        # Retrieve the path to libRadtran
        if engine_config.engine_base_dir:
            self.libradtran = engine_config.engine_base_dir
            Logger.debug(
                f"Using engine_config.engine_base_dir for libRadtran path: {self.libradtran}"
            )
        else:
            self.libradtran = env.path("libradtran", key="libradtran.version")
            if not self.libradtran.exists():
                self.libradtran = Path(
                    os.environ.get("LIBRADTRAN_DIR", "<LIBRADTRAN_DIR NOT SET>")
                )
                Logger.debug(
                    f"Using environment $LIBRADTRAN_DIR for libradtran path: {self.libradtran}"
                )
            else:
                Logger.debug(f"Using ISOFIT ini for libRadtran path: {self.libradtran}")

        # Validate the path exists
        self.libradtran = Path(self.libradtran)
        if not self.libradtran.exists():
            error = f"""\
LibRadTran directory not found: {self.libradtran}. Please use one of the following to set it correctly:
- Configuration: engine_config.engine_base_dir
- ISOFIT ini: libradtran
- Environment variable: LIBRADTRAN_DIR\
"""
            Logger.error(error)
            raise FileNotFoundError(error)

        self.lrt_bin_dir = self.libradtran / "bin"

        super().__init__(engine_config, **kwargs)

    def preSim(self):

        # define the output wl grid [nm], and stash matrix H
        self.lrt_wl = np.arange(self.wl_1, self.wl_2 + 1e-5, self.wl_res_out)
        self.matrix_H = calculate_resample_matrix(self.lrt_wl, self.wl, self.fwhm)

        # load static information from MODTRAN template
        self.template = json_load_ascii(self.engine_config.template_file)["MODTRAN"]
        self.modtran_geom = self.template[0]["MODTRANINPUT"]["GEOMETRY"]
        self.modtran_surf = self.template[0]["MODTRANINPUT"]["SURFACE"]
        self.modtran_atmos = self.template[0]["MODTRANINPUT"]["ATMOSPHERE"]

        self.atmosphere_type = self.modtran_atmos["M1"]
        self.day_of_year = self.modtran_geom["IDAY"]
        self.nstr = self.template[0]["MODTRANINPUT"]["RTOPTIONS"]["NSTR"]

        atmos = self.atmosphere_type.strip().lower()

        # Possible libRadtran mappings for atmosphere type (either the libradtran or modtran names)
        atm_map = {
            "afglt": "afglt",
            "afglms": "afglms",
            "afglmw": "afglmw",
            "afglss": "afglss",
            "afglsw": "afglsw",
            "afglus": "afglus",
            "atm_tropical": "afglt",
            "atm_midlat_summer": "afglms",
            "atm_midlat_winter": "afglmw",
            "atm_subarc_summer": "afglss",
            "atm_subarc_winter": "afglsw",
            "atm_us_standard_1976": "afglus",
        }

        if atmos not in atm_map:
            valid = ", ".join(sorted(atm_map.keys()))
            raise ValueError(
                f"Unknown atmosphere type '{atmos}'.\n" f"Valid options are:\n  {valid}"
            )

        self.atmosphere_type_lrt = atm_map[atmos]

        # Set up the irradiance spectrum
        self.path_solar = env.path("data", "oci", "tsis_f0_0p1.txt")
        solar_data = np.loadtxt(self.path_solar)
        wl_solar = solar_data[:, 0].T
        solar_irr = solar_data[:, 1].T

        # Decide units of solar irradiance
        # assuming we want to go to uW/nm/cm2 based on any unit input
        if np.nanmax(solar_irr) > 300.0:
            solar_irr = solar_irr / 10.0
        elif np.nanmax(solar_irr) < 30.0 and np.nanmax(solar_irr) > 3.0:
            solar_irr = solar_irr * 10.0
        elif np.nanmax(solar_irr) < 3.0 and np.nanmax(solar_irr) > 0.3:
            solar_irr = solar_irr * 100.0
        elif np.nanmax(solar_irr) < 300.0 and np.nanmax(solar_irr) > 30.0:
            pass  # range is already within ~ 0-250
        else:
            raise ValueError("Verify units of solar irradiance are in uW/nm/cm2.")

        # apply ESD correction
        self.esd = IO.load_esd()
        irr_ref = self.esd[200, 1]
        irr_cur = self.esd[self.day_of_year - 1, 1]
        solar_irr = solar_irr * irr_ref**2 / irr_cur**2

        # stash this for other calcs (in sensor grid)
        self.solar_irr_sensor = common.resample_spectrum(
            solar_irr, wl_solar, self.wl, self.fwhm
        )

        return {
            "coszen": np.cos(np.radians(self.modtran_geom["PARM2"])),
            "solzen": self.modtran_geom["PARM2"],
            "solar_irr": self.solar_irr_sensor,
        }

    def makeSim(self, point: np.array):
        """
        Perform libRadtran simulations

        Parameters
        ----------
        point: np.array
            Point to process
        """
        # Retrieve the files to process
        name = self.point_to_filename(point)

        # always rebuild inp files to be able to gather sza/vza if re-running...
        cmd = self.rebuild_cmd(point, name)

        # Only execute when the .out file is missing (for now jsut checking first out)
        sim1_out = self.sim_path / f"{name}_sim1_alb-0.0.out"
        if sim1_out.exists():
            Logger.warning(f"libRadtran sim files already exist for point {point}")
            return

        if not self.engine_config.rte_configure_and_exit:
            call = subprocess.run(
                cmd, shell=True, capture_output=True, cwd=self.lrt_bin_dir
            )
            if call.stdout:
                Logger.error(call.stdout.decode())

    def readSim(self, point):
        name = self.point_to_filename(point)
        prefix = str(self.sim_path / name)
        a1, a2 = self.albedos[1], self.albedos[2]

        # load all sims, see `output_user` for definitions
        u1 = np.loadtxt(f"{prefix}_sim1_alb-0.0.out", usecols=1)
        e2, d2 = np.loadtxt(f"{prefix}_sim2_alb-0.0.out", usecols=(2, 3)).T
        e3, d3 = np.loadtxt(f"{prefix}_sim3_alb-0.0.out", usecols=(2, 3)).T
        e4 = np.loadtxt(f"{prefix}_sim4_alb-{a1}.out", usecols=2)
        e5 = np.loadtxt(f"{prefix}_sim5_alb-{a2}.out", usecols=2)

        # Read cos_vza and cos_sza from sim1
        inp_path = Path(f"{prefix}_sim1_alb-0.0.inp")
        lines = {
            p[0]: p[1]
            for line in inp_path.read_text().splitlines()
            if (p := line.strip().split()) and len(p) >= 2
        }
        cos_vza = float(lines["umu"])
        cos_sza = np.cos(np.radians(float(lines["sza"])))

        # total downward transmittance
        total_down = e2 / cos_sza
        total_down = np.clip(total_down, 0.0, 1.0)

        # total upward transmittance
        total_up = e3 / cos_vza
        total_up = np.clip(total_up, 0.0, 1.0)

        # path reflectance for zero albedo
        rhoatm = u1 * np.pi / cos_sza
        rhoatm = np.clip(rhoatm, 0.0, 1.0)

        # spherical albedo
        denom = self.albedos[2] * e5 - self.albedos[1] * e4
        sphalb = np.where(np.abs(denom) > 1e-12, (e5 - e4) / denom, 0.0)
        sphalb = np.clip(sphalb, 0.0, 1.0)

        # down direct transmittance
        transm_down_dir = d2 / cos_sza
        transm_down_dir = np.clip(transm_down_dir, 0.0, 1.0)

        # down diffuse transmittance
        # transm_down_dif = (eglo4 * (1 - a1 * sphalb) - edir2) / (cos_sza) #same
        transm_down_dif = np.maximum(total_down - transm_down_dir, 0.0)
        transm_down_dif = np.clip(transm_down_dif, 0.0, 1.0)

        # upward direct transmittance
        transm_up_dir = d3 / cos_vza
        transm_up_dir = np.clip(transm_up_dir, 0.0, 1.0)

        # upward diffuse transmittance
        transm_up_dif = np.maximum(total_up - transm_up_dir, 0.0)
        transm_up_dif = np.clip(transm_up_dif, 0.0, 1.0)

        results = {
            "rhoatm": rhoatm,
            "sphalb": sphalb,
            "transm_down_dif": transm_down_dif,
            "transm_down_dir": transm_down_dir,
            "transm_up_dif": transm_up_dif,
            "transm_up_dir": transm_up_dir,
            "dir-dir": transm_down_dir * transm_up_dir,
            "dif-dir": transm_down_dif * transm_up_dir,
            "dir-dif": transm_down_dir * transm_up_dif,
            "dif-dif": transm_down_dif * transm_up_dif,
        }

        # resample all quantities to sensor wavelengths using H matrix
        results = {
            key: resample_spectrum(
                data, self.lrt_wl, self.wl, self.fwhm, H=self.matrix_H
            )
            for key, data in results.items()
        }

        # set these keys to be appropriate for rdn mode
        for key in ["rhoatm", "dir-dir", "dif-dir", "dir-dif", "dif-dif"]:
            results[key] = units.transm_to_rdn(
                results[key], self.solar_irr_sensor, cos_sza
            )

        return results

    def postSim(self):
        self.rt_mode = "rdn"
        self.lut.setAttr("RT_mode", "rdn")

    def rebuild_cmd(self, point, name):
        # using the MODTRAN template file
        translation = {
            "surface_elevation_km": "GNDALT",
            "observer_altitude_km": "H1ALT",
            "observer_azimuth": "TRUEAZ",
            "observer_zenith": "OBSZEN",
        }

        vals = {translation.get(n, n): v for n, v in zip(self.lut_names, point)}

        vals.update(
            {
                "wl_1": self.wl_1,
                "wl_2": self.wl_2,
                "wl_res_out": self.wl_res_out,
                "path_solar": self.path_solar,
                "band_model": self.engine_config.reptran_band_model,
                "atmos": self.atmosphere_type_lrt,
                "libradtran_dir": self.libradtran,
                "nstr": self.nstr,
                "ssa_scale": self.engine_config.ssa_scale,
                "gg_set": self.engine_config.gg_set,
                "tau_file": self.engine_config.tau_file,
                "ssa_file": self.engine_config.ssa_file,
                "gg_file": self.engine_config.gg_file,
                "moments_file": self.engine_config.moments_file,
            }
        )

        # Sensor altitude above sea level (for libradtran, toa is used for satellite alt)
        alt = vals.get("H1ALT", self.modtran_geom["H1ALT"])
        vals["alt"] = "toa" if alt > 98.0 else min(alt, 99.0)

        # Observer zenith, 0-90 [deg]
        vza = vals.get("OBSZEN", self.modtran_geom["OBSZEN"])
        vza = 180.0 - vza if vza > 90.0 else vza
        vals["vza_deg"] = max(0, vza)
        vals["cos_vza"] = np.cos(np.radians(vals["vza_deg"]))

        # observer azimuth [deg]
        vals["vaa_deg"] = vals.get("TRUEAZ", self.modtran_geom["TRUEAZ"])

        # surface elevation [km]
        vals["elev"] = abs(max(vals.get("GNDALT", self.modtran_surf["GNDALT"]), 0.0))

        # solar zenith [deg], same as PARM2
        vals["sza_deg"] = self.modtran_geom["PARM2"]

        # relative azimuth [deg], same as PARM1
        vals["relative_azimuth"] = self.modtran_geom["PARM1"]

        # solar azimuth, 0-360 [deg]
        vals["saa_deg"] = (vals["vaa_deg"] - vals["relative_azimuth"]) % 360

        # co2 is always populated in MODTRAN template file
        vals["co2_inp"] = self.modtran_atmos["CO2MX"]

        # o3, convert a to DU
        vals["o3_inp"] = self.modtran_atmos["O3STR"] * 1000.0

        # could be empty depending on presolve implementation
        # TODO: what is the default aot for presolve?
        vals["aot"] = 0.10

        # default water vapor
        vals["h2o_mm"] = units.cm_to_mm(self.modtran_atmos["H2OSTR"])

        # Now, override if in LUT
        if "solar_zenith" in vals:
            vals["sza_deg"] = vals["solar_zenith"]

        if "relative_azimuth" in vals:
            vals["saa_deg"] = (vals["vaa_deg"] - vals["relative_azimuth"]) % 360

        if "CO2" in vals:
            vals["co2_inp"] = vals["CO2"]

        if "H2OSTR" in vals:
            vals["h2o_mm"] = units.cm_to_mm(vals["H2OSTR"])

        # assuming only one aerosol value from LUT can be used right now
        if "AOT550" in vals:
            vals["aot"] = vals["AOT550"]
        elif "AERFRAC_2" in vals:
            vals["aot"] = vals["AERFRAC_2"]
        elif "AERFRAC_1" in vals:
            vals["aot"] = vals["AERFRAC_1"]
        else:
            pass

        # Update the libRadtran input file based on the aerosol settings by the user
        # see docs page for more information here, https://www.libradtran.org
        # King-Byrne vs. internal libradtran spline-interp based AOD at 550 and aerosol profile.
        lrt_run_inp = self.libradtran_inp_template
        if self.engine_config.kb_alpha_1 is not None:
            vals["alpha_0"] = (
                np.log(vals["aot"])
                - (self.engine_config.kb_alpha_1 * np.log(0.550))
                - (self.engine_config.kb_alpha_2 * (np.log(0.550) ** 2))
            )
            vals["alpha_1"] = self.engine_config.kb_alpha_1
            vals["alpha_2"] = self.engine_config.kb_alpha_2

            lrt_run_inp += "aerosol_king_byrne {alpha_0} {alpha_1} {alpha_2}\n"

        else:
            lrt_run_inp += "aerosol_set_tau_at_wvl 550 {aot}\n"

        if self.engine_config.ssa_scale is not None:
            lrt_run_inp += "aerosol_modify ssa scale {ssa_scale}\n"

        if self.engine_config.gg_set is not None:
            lrt_run_inp += "aerosol_modify gg set {gg_set}\n"

        if self.engine_config.tau_file is not None:
            lrt_run_inp += "aerosol_file tau {tau_file}\n"

        if self.engine_config.ssa_file is not None:
            lrt_run_inp += "aerosol_file ssa {ssa_file}\n"

        if self.engine_config.gg_file is not None:
            lrt_run_inp += "aerosol_file gg {gg_file}\n"

        if self.engine_config.moments_file is not None:
            lrt_run_inp += "aerosol_file moments {moments_file}\n"

        runs = [
            dict(tag="sim1", albedo=0.0, zout=vals["alt"]),
            dict(tag="sim2", albedo=0.0, zout="sur"),
            dict(tag="sim3", albedo=0.0, zout="sur"),
            dict(tag="sim4", albedo=self.albedos[1], zout="sur"),
            dict(tag="sim5", albedo=self.albedos[2], zout="sur"),
        ]

        files = []

        # create the 5 sim files
        for run in runs:
            run_vals = vals.copy()
            run_vals["albedo"] = run["albedo"]
            run_vals["zout"] = run["zout"]

            # Reciprocal geometry for transm_up_total for sim3
            if run["tag"] == "sim3":
                run_vals["sza_deg"] = run_vals["vza_deg"]

            fname = f"{name}_{run['tag']}_alb-{run['albedo']}"
            files.append(fname)

            inp_file = self.sim_path / f"{fname}.inp"
            inp_file.write_text(lrt_run_inp.format(**run_vals, **env))

        # create shell script to run sims 1-5
        sh_file = self.sim_path / f"{name}.sh"
        uvspecs = "\n".join(
            [
                f"{self.lrt_bin_dir}/uvspec < {self.sim_path / fn}.inp > {self.sim_path / fn}.out"
                for fn in files
            ]
        )

        sh_file.write_text(
            self.sh_template.format(lrt_bin_dir=self.lrt_bin_dir, uvspecs=uvspecs)
        )

        return f"bash {sh_file}"
