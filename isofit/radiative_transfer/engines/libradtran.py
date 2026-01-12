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
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from isofit.core.fileio import IO
from isofit.core import common
from isofit.data import env
from isofit.core.common import units
from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine

Logger = logging.getLogger(__name__)

SCRIPT_TEMPLATE = """\
#!/bin/bash
{envir}
{uvspecs}
"""

WL_LO = 340
WL_HI = 2510
REPTRAN_MODEL="coarse"

LRT_TEMPLATE = """\
source solar
wavelength {WL_LO} {WL_HI}
albedo {albedo}
atmosphere_file {libradtran_dir}/data/atmmod/{atmos}.dat
umu {cos_vza}
phi0 {saa_deg}
phi {vaa_deg}
sza {sza_deg}
rte_solver disort
mol_modify O3 300 DU
mixing_ratio CO2 420
mol_abs_param reptran {REPTRAN_MODEL}
mol_modify H2O {h2o_mm} MM
crs_model rayleigh bodhaine
zout {zout}
altitude {elev}
aerosol_default
aerosol_species_file continental_average
aerosol_set_tau_at_wvl 550 {aot}
output_quantity transmittance
output_user lambda uu eglo edir eup
quiet
"""


# See Table 3.1; page 25


# enable a co2 mode??




class LibRadTranRT(RadiativeTransferEngine):
    
    def __init__(self, engine_config: RadiativeTransferEngineConfig, **kwargs):

        self.albedos = [0.0, 0.1, 0.5]

        # Retrieve the path to LibRadTran
        if engine_config.engine_base_dir:
            self.libradtran = engine_config.engine_base_dir
            Logger.debug(
                f"Using engine_config.engine_base_dir for libradtran path: {self.libradtran}"
            )
        else:
            self.libradtran = env.path("libradtran", key="libradtran.version")
            if not self.libradtran.exists():
                self.libradtran = os.getenv(
                    "LIBRADTRAN_DIR", "<LIBRADTRAN_DIR NOT SET>"
                )
                Logger.debug(
                    f"Using environment $LIBRADTRAN_DIR for libradtran path: {self.libradtran}"
                )
            else:
                Logger.debug(f"Using ISOFIT ini for libradtran path: {self.libradtran}")

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

        self.template = LRT_TEMPLATE

        self.environment = engine_config.environment or ""

        super().__init__(engine_config, **kwargs)

        self.atmosphere_type = "Should be able to pull this from config and is set in modtran template file" 
        self.atmosphere_type_lrt = None
        self.wl = None
        self.fwhm = None

    def preSim(self):

        atmos = self.atmosphere_type.strip().lower()

        # Possible libradtran mappings for atmosphere type
        atm_map = {
            "afglt": "afglt",
            "afglms": "afglms",
            "afglmw": "afglmw",
            "afglss": "afglss",
            "afglsw": "afglsw",
            "afglus": "afglus",

            "tropics": "afglt",
            "tropical": "afglt",
            "midlatitude_summer": "afglms",
            "midlat_summer": "afglms",
            "midlatitude_winter": "afglmw",
            "midlat_winter": "afglmw",
            "subarctic_summer": "afglss",
            "subarc_summer": "afglss",
            "subarctic_winter": "afglsw",
            "subarc_winter": "afglsw",
            "us_standard": "afglus",
            "us_standard_1976": "afglus",

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
                f"Unknown atmosphere type '{atmos}'.\n"
                f"Valid options are:\n  {valid}"
            )

        self.atmosphere_type_lrt = atm_map[atmos]

    def makeSim(self, point: np.array):
        """
        Perform LibRadTran simulations

        Parameters
        ----------
        point: np.array
            Point to process
        """
        # Retrieve the files to process
        name = self.point_to_filename(point)

        # Only execute when the .out file is missing
        if (self.sim_path / f"{name}.out").exists():
            Logger.warning(f"LibRadTran sim files already exist for point {point}")
            return

        cmd = self.rebuild_cmd(point, name)

        if not self.engine_config.rte_configure_and_exit:
            call = subprocess.run(
                cmd, shell=True, capture_output=True, cwd=self.sim_path
            )
            if call.stdout:
                Logger.error(call.stdout.decode())

    def readSim(self, point):
        name = self.point_to_filename(point)

        # load outputs: lambda uu eglo edir
        _, uu1, _, _ , _ = np.loadtxt(self.sim_path / f"{name}_sim1_alb-0.0.out").T
        _, _, eglo2, edir2, eup2 = np.loadtxt(self.sim_path / f"{name}_sim2_alb-0.0.out").T
        _, _, eglo3, _,_ = np.loadtxt(self.sim_path / f"{name}_sim3_alb-0.0.out").T
        _, _, eglo4, _,_ = np.loadtxt(self.sim_path / f"{name}_sim4_alb-{[self.albedos[1]]}.out").T
        _, _, eglo5, _,_ = np.loadtxt(self.sim_path / f"{name}_sim5_alb-{[self.albedos[2]]}.out").T

        # get the solzen and coszen from sim1
        with open(self.sim_path / f"{name}_sim1_alb-0.0.inp", "r") as f:
            for line in f:
                if line.lower().startswith("sza"):
                    solzen = float(line.split()[1])
                    break
        # if this doesn't exist should trigger an error, something went wrong with input files
        coszen = np.cos(np.radians(solzen))

        # Gather all of the terms
        transm_up_total = eglo3
        transm_down_total = eglo2

        transm_up_dif = eup2
        transm_up_dir = transm_up_total - transm_up_dif

        transm_down_dir = edir2
        transm_down_dif = transm_down_total - transm_down_dir

        rhoatm = uu1

        # setting very small values to be zero for sphalb
        denom = self.albedos[2] * eglo5 - self.albedos[1] * eglo4
        sphalb = np.where(np.abs(denom) > 1e-12, (eglo5 - eglo4) / denom, 0.0)
        sphalb = np.where((sphalb < 0) | (sphalb > 1), 0.0, sphalb)

        results = {
            "solzen": solzen,
            "coszen": coszen,
            "rhoatm": rhoatm,
            "sphalb": sphalb,
            "transm_down_dir": transm_down_dir,
            "transm_down_dif": transm_down_dif,
            "transm_up_dir": transm_up_dir,
            "transm_up_dif": transm_up_dif,
        }

        return results

    def postSim(self):
        """To run medium and fine, ancillary data must be downloaded from libradtran."""
        # Set up the wavelength based on the REPTRAN band model
        model = REPTRAN_MODEL.lower()
        if model == "fine":
            res_cm = 1.0
        elif model == "medium":
            res_cm = 5.0
        elif model == "coarse":
            res_cm = 15.0
        else:
            raise ValueError(f"REPTRAN band models are: 'fine', 'medium', 'coarse'.")
        
        wl_list = [WL_LO]
        while wl_list[-1] < WL_HI:
            wl = wl_list[-1]
            delta_lambda = wl**2 * res_cm / 1e7  # nm
            wl_next = wl + delta_lambda
            if wl_next > WL_HI:
                break
            wl_list.append(wl_next)

        self.wl = np.array(wl_list)
        self.fwhm = self.wl**2 * res_cm / 1e7

        # Set up the irradiance spectrum
        path_solar = env.path("data", "kurudz_0.1nm.dat")
        data = np.loadtxt(path_solar, comments="#")
        wl_kurucz = data[:, 0]     
        self.solar_irr = data[:, 1] / 10.0 # convert from mW/m2/nm to uW/nm/cm2

        # apply ESD correction
        self.load_esd()
        self.solar_irr = self.solar_irr / self.irr_factor**2  

        # resample to REPTRAN grid
        wl_kurucz = np.arange(250.0, 10000.0001, 0.1)
        self.solar_irr = common.resample_spectrum(self.solar_irr, wl_kurucz, self.wl, self.fwhm)

        return {"solar_irr": self.solar_irr} 

    def rebuild_cmd(self, point, name):
        vals = {"aot": "AOT550"}
        vals.update(zip(self.lut_names, point))
        vals.update({
            "WL_LO": WL_LO,
            "WL_HI": WL_HI,
            "REPTRAN_MODEL": REPTRAN_MODEL,
            "atmos": self.atmosphere_type_lrt,
            "libradtran_dir": self.libradtran,
        })

        if "H2OSTR" in vals:
            vals["h2o_mm"] = units.cm_to_mm(vals["H2OSTR"])

        if "surface_elevation_km" in vals:
            vals["elev"] = abs(max(vals["surface_elevation_km"], 0))

        if "observer_altitude_km" in vals:
            alt = min(vals["observer_altitude_km"], 99)
            vals["alt"] = "toa" if alt > 98 else alt

        if "observer_azimuth" in vals:
            vals["vaa_deg"] = vals["observer_azimuth"]

        if "observer_zenith" in vals:
            vals["vza_deg"] = vals["observer_zenith"]
            vals["cos_vza"] = np.cos(np.radians(vals["observer_zenith"]))

        if "solar_zenith" in vals:
            vals["sza_deg"] = vals["solar_zenith"]

        if "relative_azimuth" in vals:
            vals["saa_deg"] = np.minimum(
                vals["vaa_deg"] + vals["relative_azimuth"],
                vals["vaa_deg"] - vals["relative_azimuth"],
            )

        # five runs needed to get the quantities needed for 6c model
        runs = [
            dict(tag="sim1", albedo=0.0, zout=vals["alt"], swap_geom=False),
            dict(tag="sim2", albedo=0.0, zout="sur",      swap_geom=False),
            dict(tag="sim3", albedo=0.0, zout="sur",      swap_geom=True),
            dict(tag="sim4", albedo=self.albedos[1], zout=vals["alt"], swap_geom=False),
            dict(tag="sim5", albedo=self.albedos[2], zout=vals["alt"], swap_geom=False),
        ]
        files = []

        for run in runs:
            run_vals = vals.copy()
            run_vals["albedo"] = run["albedo"]
            run_vals["zout"] = run["zout"]

            # Reciprocal geometry for downward transmittance
            if run["swap_geom"]:
                run_vals["sza_deg"] = run_vals["vza_deg"]

            fname = f"{name}_{run['tag']}_alb-{run['albedo']}"
            files.append(fname)

            with open(self.sim_path / f"{fname}.inp", "w") as f:
                f.write(self.template.format(**run_vals, **env))

        with open(self.sim_path / f"{name}.sh", "w") as f:
            f.write(
                SCRIPT_TEMPLATE.format(
                    envir=self.environment,
                    uvspecs="\n".join(
                        f"{self.libradtran}/bin/uvspec < {fn}.inp > {fn}.out"
                        for fn in files
                    ),
                )
            )

        return f"bash {name}.sh"
    
    def load_esd(self):
        """
        Loads the earth-sun distance file
        """
        self.esd = IO.load_esd()

        dt = datetime(2000, self.engine_config.month, self.engine_config.day)
        self.day_of_year = dt.timetuple().tm_yday
        self.irr_factor = self.esd[self.day_of_year - 1, 1]