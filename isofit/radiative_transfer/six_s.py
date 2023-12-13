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
        self.modtran_emulation = modtran_emulation

        super().__init__(engine_config, **kwargs)

        # If the LUT file already exists, still need to calc this post init
        if not hasattr(self, "esd"):
            self.load_esd()

    def preSim(self):
        """
        Check that 6S is installed
        """
        sixS = os.path.join(self.engine_base_dir, "sixsV2.1")  # 6S Emulator path

        if not os.path.exists(sixS):
            Logger.error(
                f"6S path not valid, downstream simulations will be broken: {sixS}"
            )

    def makeSim(self, point: np.array, template_only: bool = False):
        """
        Perform 6S simulations

        Parameters
        ----------
        point: np.array
            Point to process
        template_only: bool, default=False
            Only write the simulation template then exit. If False, subprocess call 6S
        """
        # Retrieve the files to process
        name = self.point_to_filename(point)
        outp = os.path.join(self.sim_path, name)
        inpt = os.path.join(self.sim_path, f"LUT_{name}.inp")

        # Only execute when either the 6S input (ext.inp) or output (no extension) files are missing
        if os.path.exists(outp) and os.path.exists(inpt):
            Logger.warning(f"6S sim files already exist: {outp}, {inpt}")
            return

        cmd = self.rebuild_cmd(point)
        if template_only is False:
            subprocess.call(cmd, shell=True)

    def readSim(self, point: np.array):
        """
        Parses a 6S output simulation file for a given point

        Parameters
        ----------
        point: np.array
            Point to process

        Returns
        -------
        data: dict
            Simulated data results. These keys correspond with the expected keys
            of ISOFIT's LUT files
        """
        name = self.point_to_filename(point)
        file = os.path.join(self.sim_path, name)

        return self.parse_file(file, self.wl.size)

    def postSim(self):
        """
        Update solar_irr after simulations
        """
        self.load_esd()

        irr = np.loadtxt(self.engine_config.irradiance_file, comments="#")
        iwl, irr = irr.T
        irr = irr / 10.0  # convert, uW/nm/cm2
        irr = irr / self.irr_factor**2  # consider solar distance
        solar_irr = resample_spectrum(irr, iwl, self.wl, self.fwhm)

        return {"solar_irr": solar_irr}

    def rebuild_cmd(self, point) -> str:
        """Build the simulation command file.

        Args:
            point (np.array): conditions to alter in simulation

        Returns:
            str: execution command
        """
        # Collect files of interest for this point
        name = self.point_to_filename(point)
        outp = os.path.join(self.sim_path, name)  # Output path
        inpt = os.path.join(self.sim_path, f"LUT_{name}.inp")  # Input path
        bash = os.path.join(self.sim_path, f"LUT_{name}.sh")  # Script path
        sixS = os.path.join(self.engine_base_dir, "sixsV2.1")  # 6S Emulator path

        # Prepare template values
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
            "wlinf": self.wl[0] / 1000.0,  # convert to nm
            "wlsup": self.wl[-1] / 1000.0,
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

        if "H2OSTR" in vals:
            vals["h2o_mm"] = vals["H2OSTR"] * 10.0

        # REVIEW: Should this be surface_elevation_km?
        if "GNDALT" in vals:
            vals["elev"] = vals["GNDALT"]

        if "elev" in vals:
            vals["elev"] = vals["elev"] * -1

        if "H1ALT" in vals:
            vals["alt"] = min(vals["H1ALT"], 99)

        if "TRUEAZ" in vals:
            vals["viewaz"] = vals["TRUEAZ"]

        # REVIEW: Should this be observer_zenith?
        if "OBSZEN" in vals:
            vals["viewzen"] = 180 - vals["OBSZEN"]

        if self.modtran_emulation:
            if "AERFRAC_2" in vals:
                vals["AOT550"] = vals["AERFRAC_2"]

        # Write sim files
        with open(inpt, "w") as f:
            template = SIXS_TEMPLATE.format(**vals)
            f.write(template)

        with open(bash, "w") as f:
            f.write("#!/usr/bin/bash\n")
            f.write(f"{sixS} < {inpt} > {outp}\n")
            f.write("cd $cwd\n")

        return f"bash {bash}"

    def load_esd(self):
        """
        Loads the earth-sun distance file
        """
        self.esd = np.loadtxt(self.earth_sun_distance_path)
        dt = datetime(2000, self.engine_config.month, self.engine_config.day)
        self.day_of_year = dt.timetuple().tm_yday
        self.irr_factor = self.esd[self.day_of_year - 1, 1]

    @staticmethod
    def parse_file(file, wl_size=0) -> dict:
        """
        Parses a 6S sim file

        Parameters
        ----------
        file: str
            Path to simulation file to parse
        wl_size: int, default=0
            Size of the wavelengths dim, will trim data to this size. If zero, does no
            trimming

        Returns
        -------
        data: dict
            Simulated data results. These keys correspond with the expected keys
            of ISOFIT's LUT files

        Examples
        --------
        >>> from isofit.radiative_transfer.six_s import SixSRT
        >>> SixSRT.parse_file('isofit/examples/20151026_SantaMonica/lut/AOT550-0.0000_H2OSTR-0.5000', wl_size=2)
        {'sphalb': array([0.3116, 0.3057, 0.2999]),
         'rhoatm': array([0.2009, 0.1963, 0.1916]),
         'transm_down_dif': array([0.53211358, 0.53993346, 0.54736113]),
         'solzen': 55.21,
         'coszen': 0.5705702414191993}
        """
        path, name = os.path.split(file)
        inpt = os.path.join(path, f"LUT_{name}.inp")

        with open(file, "r") as f:
            lines = f.readlines()

        with open(inpt, "r") as f:
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
                tokens = re.findall(r"NaN|\d+\.?\d+", line.replace("******", "0.0"))

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
                # transm = float(scau) * float(scad) * float(gt)

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

        if start is None:
            Logger.error(f"Failed to parse any data for point: {point}")
            return {}

        total = len(data["grid"])
        if total < wl_size:
            Logger.error(
                f"The following file parsed shorter than expected ({wl_size}), got ({total}): {file}"
            )

        # Cast to numpy and trim excess
        data = {k: np.array(v) for k, v in data.items()}
        if wl_size > 0:
            data = {k: v[:wl_size] for k, v in data.items()}

        # Add extras
        data["solzen"] = solzen
        data["coszen"] = coszen

        # Remove before saving to LUT file since this doesn't go in there
        data.pop("grid")

        return data
