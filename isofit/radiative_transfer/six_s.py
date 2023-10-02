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
#

import logging
import os
import subprocess
from datetime import datetime

import numpy as np

from isofit.configs import Config
from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.core.common import VectorInterpolator, resample_spectrum
from isofit.core.geometry import Geometry
from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine

from .look_up_tables import FileExistsError, TabularRT

eps = 1e-5  # used for finite difference derivative calculations

sixs_template = """0 (User defined)
{solzen} {solaz} {viewzen} {viewaz} {month} {day}
8  (User defined H2O, O3)
{H2OSTR}, {O3}
{aermodel}
0
{AOT550}
{elev:.2f} (target level)
-{alt:.2f} (sensor level)
-{H2OSTR}, -{O3}
{AOT550}
-2 
{wlinf}
{wlsup}
0 Homogeneous surface
0 (no directional effects)
0
0
0
-1 No atm. corrections selected
"""


class SixSRT(RadiativeTransferEngine):
    """A model of photon transport including the atmosphere."""

    def __init__(
        self,
        engine_config: RadiativeTransferEngineConfig,
        interpolator_style: str,
        instrument_wavelength_file: str = None,
        overwrite_interpolator: bool = False,
        cache_size: int = 16,
        lut_grid: dict = None,
        build_lut_only=False,
        wavelength_override=None,
        fwhm_override=None,
        modtran_emulation=False,
    ):
        super().__init__(
            engine_config,
            interpolator_style,
            instrument_wavelength_file,
            overwrite_interpolator,
            cache_size,
            lut_grid,
        )

        if wavelength_override is not None:
            self.wl = wavelength_override
            self.n_chan = len(self.wl)
            self.resample_wavelengths = False
        else:
            self.resample_wavelengths = True

        if fwhm_override is not None:
            self.fwhm = fwhm_override

        self.modtran_emulation = modtran_emulation
        self.engine_config = engine_config

        self.sixs_grid_init = np.arange(self.wl[0], self.wl[-1] + 2.5, 2.5)
        self.sixs_ngrid_init = len(self.sixs_grid_init)

        self.esd = np.loadtxt(engine_config.earth_sun_distance_file)
        dt = datetime(2000, self.params["month"], self.params["day"])
        self.day_of_year = dt.timetuple().tm_yday
        self.irr_factor = self.esd[self.day_of_year - 1, 1]

        irr = np.loadtxt(engine_config.irradiance_file, comments="#")
        iwl, irr = irr.T
        irr = irr / 10.0  # convert, uW/nm/cm2
        irr = irr / self.irr_factor**2  # consider solar distance
        self.solar_irr = resample_spectrum(irr, iwl, self.wl, self.fwhm)

    def rebuild_cmd(self, point) -> str:
        """Build the simulation command file.

        Args:
            point (np.array): conditions to alter in simulation

        Returns:
            str: execution command
        """
        filename_base = self.point_to_filename(point)

        vals = {
            "aermodel": 1,
            "AOT550": 0.01,
            "H2OSTR": 0,
            "O3": 0.30,
            "day": self.engine_config.day,
            "month": self.engine_config.month,
            "elev": self.engine_config.elev,
            "alt": min(self.engine_config.alt, 99),
            "atm_file": None,
            "abscf_data_directory": None,
            "wlinf": self.sixs_grid_init[0] / 1000.0,  # convert to nm
            "wlsup": self.sixs_grid_init[-1] / 1000.0,
        }

        # We have to get geometry from somewhere, so we presume it is
        # in the configuration file.
        vals["solzen"] = self.engine_config.solzen
        vals["viewzen"] = self.engine_config.viewzen
        vals["solaz"] = self.engine_config.solaz
        vals["viewaz"] = self.engine_config.viewaz

        for n, v in zip(self.lut_names, point):
            vals[n] = v

        # Translate a couple of special cases
        if "H2OSTR" in self.lut_names:
            vals["h2o_mm"] = vals["H2OSTR"] * 10.0
        if "GNDALT" in vals:
            vals["elev"] = vals["GNDALT"]
        if "H1ALT" in vals:
            vals["alt"] = min(vals["H1ALT"], 99)
        if "TRUEAZ" in vals:
            vals["viewaz"] = vals["TRUEAZ"]
        if "OBSZEN" in vals:
            vals["viewzen"] = 180 - vals["OBSZEN"]

        if self.modtran_emulation:
            if "AERFRAC_2" in vals:
                vals["AOT550"] = vals["AERFRAC_2"]

        if "elev" in vals:
            vals["elev"] = vals["elev"] * -1

        # Check rebuild conditions: LUT is missing or from a different config
        scriptfilename = "LUT_" + filename_base + ".sh"
        scriptfilepath = os.path.join(self.lut_dir, scriptfilename)
        infilename = "LUT_" + filename_base + ".inp"
        infilepath = os.path.join(self.lut_dir, infilename)
        outfilename = filename_base
        outfilepath = os.path.join(self.lut_dir, outfilename)
        if os.path.exists(outfilepath) and os.path.exists(infilepath):
            raise FileExistsError("Files exist")

        sixspath = self.engine_base_dir + "/sixsV2.1"

        # write config files
        sixs_config_str = sixs_template.format(**vals)
        with open(infilepath, "w") as f:
            f.write(sixs_config_str)

        # Write runscript file
        with open(scriptfilepath, "w") as f:
            f.write("#!/usr/bin/bash\n")
            f.write("%s < %s > %s\n" % (sixspath, infilepath, outfilepath))
            f.write("cd $cwd\n")

        return "bash " + scriptfilepath

    def make_simulation_call(self, point: np.array, template_only: bool = False):
        cmd = self.rebuild_cmd(point)
        if template_only is False:
            subprocess.call(cmd, shell=True)

    def read_simulation_results(self, point):
        """Load the results of a SixS run."""

        filename_base = self.point_to_filename(point)

        with open(os.path.join(self.lut_dir, filename_base), "r") as l:
            lines = l.readlines()

        with open(
            os.path.join(self.lut_dir, "LUT_" + filename_base + ".inp"), "r"
        ) as l:
            inlines = l.readlines()
            solzen = float(inlines[1].strip().split()[0])
        self.coszen = np.cos(solzen / 360 * 2.0 * np.pi)

        # Strip header
        for i, ln in enumerate(lines):
            if ln.startswith("*        trans  down   up"):
                lines = lines[(i + 1) : (i + 1 + self.sixs_ngrid_init)]
                break

        solzens = np.zeros(len(lines))
        sphalbs = np.zeros(len(lines))
        transups = np.zeros(len(lines))
        transms = np.zeros(len(lines))
        rhoatms = np.zeros(len(lines))
        self.grid = np.zeros(len(lines))

        for i, ln in enumerate(lines):
            ln = ln.replace("******", "0.0").strip()
            ln = ln.replace("*", " ").strip()

            w, gt, scad, scau, salb, rhoa, swl, step, sbor, dsol, toar = ln.split()

            self.grid[i] = float(w) * 1000.0  # convert to nm
            solzens[i] = float(solzen)
            sphalbs[i] = float(salb)
            transups[i] = 0.0  # float(scau)
            transms[i] = float(scau) * float(scad) * float(gt)
            rhoatms[i] = float(rhoa)

        results = {
            "solzen": solzens,
            "rhoatm": rhoatms,
            "transm": transms,
            "sphalb": sphalbs,
            "transup": transups,
        }
        return results
