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
from netCDF4 import Dataset
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

SCRIPT_TEMPLATE = """\
#!/bin/bash

cd {lrt_bin_dir}

{uvspecs}
"""

# fine or medium models are available for extra downloads from lrt webpage (~500 MB size)
REPTRAN_MODEL = "coarse"

# TODO: these differ slightly from emulator for high AOD, but its likely a function of crs_model rayleigh, as well as other features that can be swapped out

# assumes delta M scaling is very small for these simulations (DISORT with 8 streams). needs to be tested much more to see how large these errors actually are.

# NOTE: quote from aerosol setup for default:
#   "The most simple way to define an aerosol is by the command `aerosol_default``
#    which will set up the aerosol model by Shettle (1989).
#    The default properties are a rural type aerosol in the boundary layer,
#    background aerosol above 2km, spring-summer conditions and a visibility of 50km."

LRT_TEMPLATE = """\
source solar
wavelength {wl_lo} {wl_hi}
albedo {albedo}
atmosphere_file {libradtran_dir}/data/atmmod/{atmos}.dat
umu {cos_vza}
phi0 {saa_deg}
phi {vaa_deg}
sza {sza_deg}
rte_solver disort
number_of_streams {nstr}
mol_modify O3 {o3_inp} DU
mixing_ratio CO2 {co2_inp}
mol_abs_param reptran {band_model}
mol_modify H2O {h2o_mm} MM
crs_model rayleigh bodhaine
zout {zout}
altitude {elev}
aerosol_default
aerosol_set_tau_at_wvl 550 {aot}
output_quantity transmittance
output_user lambda uu eglo edir
"""


class LibRadTranRT(RadiativeTransferEngine):

    def __init__(self, engine_config: RadiativeTransferEngineConfig, **kwargs):

        self.atmosphere_type_lrt = None
        self.reptran_band_model = REPTRAN_MODEL.lower()

        # Load solar irradiance data (similar source to sRTMnet)
        path_solar = env.path("data", "oci", "tsis_f0_0p5.txt")
        self.solar_data = np.loadtxt(path_solar)

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

        self.template = LRT_TEMPLATE

        super().__init__(engine_config, **kwargs)

    def preSim(self):

        # load static geom between sims
        self.modtran_template = json_load_ascii(self.engine_config.template_file)[
            "MODTRAN"
        ]
        self.modtran_geom = self.modtran_template[0]["MODTRANINPUT"]["GEOMETRY"]
        self.modtran_surf = self.modtran_template[0]["MODTRANINPUT"]["SURFACE"]
        self.modtran_atmos = self.modtran_template[0]["MODTRANINPUT"]["ATMOSPHERE"]

        self.atmosphere_type = self.modtran_atmos["M1"]
        self.day_of_year = self.modtran_geom["IDAY"]
        self.nstr = self.modtran_template[0]["MODTRANINPUT"]["RTOPTIONS"]["NSTR"]

        # set approx upper and lower bounds of sims (to be overwritten by actual reptran grid later)
        self.wl_lo = self.wl[0] - 10.0
        self.wl_hi = self.wl[-1] + 10.0

        # Set sim albedos
        self.albedos = [0.0, 0.1, 0.5]

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

        # Set up the wavelength based on the REPTRAN band model, and overwrite hi and lo
        cdf = (
            self.libradtran
            / "data"
            / "correlated_k"
            / "reptran"
            / f"reptran_solar_{self.reptran_band_model}.cdf"
        )
        with Dataset(cdf) as ds:
            wl_min = ds.variables["wvlmin"][:]
            wl_max = ds.variables["wvlmax"][:]

        # snap to the closest reptran band that matches self.wl_lo and hi
        rep_idx = (wl_max >= self.wl_lo) & (wl_min <= self.wl_hi)
        self.wl_lo = wl_min[rep_idx][0]
        self.wl_hi = wl_max[rep_idx][-1]
        self.lrt_wl = 0.5 * (wl_min[rep_idx] + wl_max[rep_idx])

        # Set up the irradiance spectrum
        wl_solar_inp = self.solar_data[:, 0].T
        solar_irr = self.solar_data[:, 1].T / 10.0  # convert from mW/m2/nm to uW/nm/cm2

        # apply ESD correction (in a similar way to sRTMnet)
        self.esd = IO.load_esd()
        self.irr_cur = self.esd[self.day_of_year - 1, 1]
        self.irr_ref = self.esd[200, 1]
        solar_irr = solar_irr * self.irr_ref**2 / self.irr_cur**2

        # stash this for other calcs (in sensor grid)
        self.solar_irr_sensor = common.resample_spectrum(
            solar_irr, wl_solar_inp, self.wl, self.fwhm
        )

        # Now get the H matrix to go from REPTRAN grid to sensor grid
        self.matrix_H = calculate_resample_matrix(self.lrt_wl, self.wl, self.fwhm)

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

        # Only execute when the .out file is missing (for now jsut checking first out)
        sim1_out = self.sim_path / f"{name}_sim1_alb-0.0.out"
        if sim1_out.exists():
            Logger.warning(f"libRadtran sim files already exist for point {point}")
            return

        cmd = self.rebuild_cmd(point, name)

        if not self.engine_config.rte_configure_and_exit:
            call = subprocess.run(
                cmd, shell=True, capture_output=True, cwd=self.lrt_bin_dir
            )
            if call.stdout:
                Logger.error(call.stdout.decode())

    def readSim(self, point):

        name = self.point_to_filename(point)
        base = self.sim_path / name
        a1, a2 = self.albedos[1], self.albedos[2]

        # cols: wvl, uu, eglo, edir
        u1 = np.loadtxt(base.with_name(f"{base.name}_sim1_alb-0.0.out"), usecols=1)
        e2, d2 = np.loadtxt(
            base.with_name(f"{base.name}_sim2_alb-0.0.out"), usecols=(2, 3)
        ).T
        e3, d3 = np.loadtxt(
            base.with_name(f"{base.name}_sim3_alb-0.0.out"), usecols=(2, 3)
        ).T
        e4 = np.loadtxt(base.with_name(f"{base.name}_sim4_alb-{a1}.out"), usecols=2)
        e5 = np.loadtxt(base.with_name(f"{base.name}_sim5_alb-{a2}.out"), usecols=2)

        # Read cos_vza and cos_sza from sim1
        inp_path = base.with_name(f"{base.name}_sim1_alb-0.0.inp")
        lines = {
            p[0]: p[1]
            for line in inp_path.read_text().splitlines()
            if (p := line.strip().split()) and len(p) >= 2
        }
        cos_vza = float(lines["umu"])
        cos_sza = np.cos(np.radians(float(lines["sza"])))

        # total downward transmittance
        total_down = e2 / cos_sza

        # total upward transmittance
        total_up = e3 / cos_vza

        # path reflectance for zero albedo
        rhoatm = u1 * np.pi / cos_sza
        rhoatm = np.where((rhoatm < 1e-12) | (rhoatm > 1), 0.0, rhoatm)

        # spherical albedo
        denom = self.albedos[2] * e5 - self.albedos[1] * e4
        sphalb = np.where(np.abs(denom) > 1e-12, (e5 - e4) / denom, 0.0)
        sphalb = np.where((sphalb < 1e-12) | (sphalb > 1), 0.0, sphalb)

        # down direct transmittance
        transm_down_dir = d2 / cos_sza
        transm_down_dir = np.where(
            (transm_down_dir < 1e-12) | (transm_down_dir > 1), 0.0, transm_down_dir
        )

        # down diffuse transmittance
        transm_down_dif = np.maximum(total_down - transm_down_dir, 0.0)
        transm_down_dif = np.where(
            (transm_down_dif < 1e-12) | (transm_down_dif > 1), 0.0, transm_down_dif
        )

        # upward direct transmittance
        transm_up_dir = d3 / cos_vza
        transm_up_dir = np.where(
            (transm_up_dir < 1e-12) | (transm_up_dir > 1), 0.0, transm_up_dir
        )

        # upward diffuse transmittance
        transm_up_dif = np.maximum(total_up - transm_up_dir, 0.0)
        transm_up_dif = np.where(
            (transm_up_dif < 1e-12) | (transm_up_dif > 1), 0.0, transm_up_dif
        )

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

        # resample all quantities to sensor wl
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
        # uusing the MODTRAN template file
        translation = {
            "surface_elevation_km": "GNDALT",
            "observer_altitude_km": "H1ALT",
            "observer_azimuth": "TRUEAZ",
            "observer_zenith": "OBSZEN",
        }

        vals = {translation.get(n, n): v for n, v in zip(self.lut_names, point)}

        vals.update(
            {
                "wl_lo": self.wl_lo,
                "wl_hi": self.wl_hi,
                "band_model": self.reptran_band_model,
                "atmos": self.atmosphere_type_lrt,
                "libradtran_dir": self.libradtran,
                "nstr": self.nstr,
            }
        )

        # Set defaults from modtran template file
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

        # relative azimuth [deg] = PARM1
        vals["relative_azimuth"] = self.modtran_geom["PARM1"]

        # solar azimuth, 0-360 [deg]
        vals["saa_deg"] = (vals["vaa_deg"] - vals["relative_azimuth"]) % 360

        # co2 is always populated in MODTRAN template file
        vals["co2_inp"] = self.modtran_atmos["CO2MX"]

        # also populated from modtran template, convert a to DU
        vals["o3_inp"] = self.modtran_atmos["O3STR"] * 1000.0

        # could be empty depending on presolve implementation
        # TODO: what is the default aot for sRTMnet for presolve?
        vals["aot"] = 0.10

        # default water vapor from modtran template file
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

        if "AOT550" in vals:
            vals["aot"] = vals["AOT550"]
        elif "AERFRAC_2" in vals:
            vals["aot"] = vals["AERFRAC_2"]
        elif "AERFRAC_1" in vals:
            vals["aot"] = vals["AERFRAC_1"]
        else:
            pass

        # five runs needed to get the quantities needed for 6c model
        runs = [
            dict(tag="sim1", albedo=0.0, zout=vals["alt"]),
            dict(tag="sim2", albedo=0.0, zout="sur"),
            dict(tag="sim3", albedo=0.0, zout="sur"),
            dict(tag="sim4", albedo=self.albedos[1], zout="sur"),
            dict(tag="sim5", albedo=self.albedos[2], zout="sur"),
        ]

        files = []

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
            inp_file.write_text(self.template.format(**run_vals, **env))

        sh_file = self.sim_path / f"{name}.sh"
        uvspecs = "\n".join(
            [
                f"{self.lrt_bin_dir}/uvspec < {self.sim_path / fn}.inp > {self.sim_path / fn}.out"
                for fn in files
            ]
        )

        sh_file.write_text(
            SCRIPT_TEMPLATE.format(lrt_bin_dir=self.lrt_bin_dir, uvspecs=uvspecs)
        )

        return f"bash {sh_file}"
