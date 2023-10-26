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
import re
import subprocess
from datetime import datetime

import numpy as np

from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.core.common import resample_spectrum
from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine

Logger = logging.getLogger(__file__)

eps = 1e-5  # used for finite difference derivative calculations

SIXS_TEMPLATE = """\
0 (User defined)
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
        modtran_emulation=False,
        **kwargs,
    ):
        super().__init__(engine_config, **kwargs)

        self.engine_config = engine_config
        self.modtran_emulation = modtran_emulation

        self.sixs_grid_init = np.arange(self.wl[0], self.wl[-1] + 2.5, 2.5)

        self.esd = np.loadtxt(self.earth_sun_distance_path)

        dt = datetime(2000, engine_config.month, engine_config.day)
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
        # Collect files of interest for this point
        name = self.point_to_filename(point)
        file = os.path.join(self.sim_path, name)  # Output path
        luts = os.path.join(self.sim_path, f"LUT_{name}.inp")  # Input path
        bash = os.path.join(self.sim_path, f"LUT_{name}.sh")  # Script path
        sixS = os.path.join(self.engine_base_dir, "sixsV2.1")  # 6S Emulator path

        # REVIEW: Is this necessary?
        # Verify at least one file is missing
        # if os.path.exists(file) and os.path.exists(luts):
        #     raise AttributeError(f"Files already exist: {file}, {luts}")

        ## Prepare template values

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

        # Assume geometry values are provided by the config
        vals |= {
            "solzen": self.engine_config.solzen,
            "viewzen": self.engine_config.viewzen,
            "solaz": self.engine_config.solaz,
            "viewaz": self.engine_config.viewaz,
        }

        # Add the point with its names
        for key, val in zip(self.lut_names, point):
            vals[key] = val

        ## Special cases

        if "H2OSTR" in self.lut_names:
            vals["h2o_mm"] = vals["H2OSTR"] * 10.0

        if "GNDALT" in vals:
            vals["elev"] = vals["GNDALT"]

        if "elev" in vals:
            vals["elev"] = vals["elev"] * -1

        if "H1ALT" in vals:
            vals["alt"] = min(vals["H1ALT"], 99)

        if "TRUEAZ" in vals:
            vals["viewaz"] = vals["TRUEAZ"]

        if "OBSZEN" in vals:
            vals["viewzen"] = 180 - vals["OBSZEN"]

        if self.modtran_emulation:
            if "AERFRAC_2" in vals:
                vals["AOT550"] = vals["AERFRAC_2"]

        # Write sim files
        with open(luts, "w") as f:
            template = SIXS_TEMPLATE.format(**vals)
            f.write(template)

        with open(bash, "w") as f:
            f.write("#!/usr/bin/bash\n")
            f.write(f"{sixS} < {luts} > {file}\n")
            f.write("cd $cwd\n")

        return f"bash {bash}"

    def make_simulation_call(self, point: np.array, template_only: bool = False):
        cmd = self.rebuild_cmd(point)
        if template_only is False:
            subprocess.call(cmd, shell=True)

    def read_simulation_results(self, point):
        """
        Parses a 6S output simulation file for a given point

        Returns
        -------
        data: dict
            Simulated data results. These keys correspond with the expected keys
            of ISOFIT's LUT files
        """
        name = self.point_to_filename(point)
        file = os.path.join(self.sim_path, name)
        luts = os.path.join(self.sim_path, f"LUT_{name}.inp")

        with open(file, "r") as f:
            lines = f.readlines()

        with open(luts, "r") as f:
            solzen = float(f.readlines()[1].strip().split()[0])
        coszen = np.cos(solzen / 360 * 2.0 * np.pi)

        # `data` stores the return values, `append` will append to existing keys and creates them if they don't
        # Easy append to keys whether they exist or not
        data = {}
        append = lambda key, val: data.setdefault(key, []).append(val)

        start = None
        for end, line in enumerate(lines):
            if start is not None:
                # Find all ints/floats for this line
                tokens = re.findall(r"(\d+\.?\d+)", line.replace("******", "0.0"))

                # End of table
                if len(tokens) != 11:
                    break

                (  # Split the tokens
                    w,
                    gt,
                    scad,
                    scau,
                    salb,
                    rhoa,
                    swl,
                    step,
                    sbor,
                    dsol,
                    toar,
                ) = tokens

                # Preprocess the tokens and prepare to save them to LUT
                transm = float(scau) * float(scad) * float(gt)

                append("grid", float(w))
                append("sphalb", float(salb))
                append("rhoatm", float(rhoa))
                append("transm_down_dif", float(scau) * float(scad) * float(gt))
                # REVIEW: How should these be populated?
                # append("transm_down_dir", None)
                # append("transm_up_dir", None)
                # append("transm_up_dif", None)

            # Found beginning of table
            elif line.startswith("*        trans  down   up"):
                start = end

        # Cast to numpy
        data = {k: np.array(v) for k, v in data.items()}

        # Add extras
        data["solzen"] = solzen
        data["coszen"] = coszen

        # Remove before saving to LUT file since this doesn't go in there
        self.grid = data.pop("grid")

        return data
