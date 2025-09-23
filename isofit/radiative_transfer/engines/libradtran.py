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
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from isofit.core import common
from isofit.data import env
from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine

Logger = logging.getLogger(__name__)

SCRIPT_TEMPLATE = """\
#!/bin/bash
{env}
{uvspecs}
{libradtran}/bin/zenith -s 0 -q -a {lat} -o {lon} -y {date} > {output}
"""


class LibRadTranRT(RadiativeTransferEngine):
    albedos = [0.0, 0.25, 0.5]

    def __init__(self, engine_config: RadiativeTransferEngineConfig, **kwargs):
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

        self.template = engine_config.template_file
        if not Path(self.template).exists():
            error = f"Template file does not exist: {self.template}"
            Logger.error(error)
            raise FileNotFoundError(error)

        with open(self.template) as f:
            self.template = f.read()

        self.environment = engine_config.environment or ""

        super().__init__(engine_config, **kwargs)

    def preSim(self):
        pass

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

        # Only execute when the .zen file is missing
        if (self.sim_path / f"{name}.zen").exists():
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

        _, rdn0, _ = np.loadtxt(self.sim_path / f"{name}_albedo-0.0.out").T
        _, rdn025, _ = np.loadtxt(self.sim_path / f"{name}_albedo-0.25.out").T
        wl, rdn05, irr = np.loadtxt(self.sim_path / f"{name}_albedo-0.5.out").T

        # Replace a few zeros in the irradiance spectrum via interpolation
        good = irr > 1e-15
        bad = np.logical_not(good)
        irr[bad] = interp1d(wl[good], irr[good])(wl[bad])

        # Translate to Top of Atmosphere (TOA) reflectance
        rhoatm = rdn0 / 10.0 / irr * np.pi  # Translate to uW nm-1 cm-2 sr-1
        rho025 = rdn025 / 10.0 / irr * np.pi
        rho05 = rdn05 / 10.0 / irr * np.pi

        # Resample TOA reflectances to simulate the instrument observation
        rhoatm = common.resample_spectrum(rhoatm, wl, self.wl, self.fwhm)
        rho025 = common.resample_spectrum(rho025, wl, self.wl, self.fwhm)
        rho05 = common.resample_spectrum(rho05, wl, self.wl, self.fwhm)
        irr = common.resample_spectrum(irr, wl, self.wl, self.fwhm)

        # Calculate some atmospheric optical constants NOTE: This calc is not
        # numerically stable for cases where rho025 and rho05 are the same.
        # Anecdotally, in all of these cases, they are also the same as rhoatm,
        # so the equation reduces to 0 / 0. Therefore, we assume that spherical
        # albedo here is zero. Any other non-finite results are (currently)
        # unexpected, so we convert them to errors.
        bad = np.logical_and(rho025 == rhoatm, rho05 == rhoatm)
        sphalb = 2.8 * (2.0 * rho025 - rhoatm - rho05) / (rho025 - rho05)
        if np.sum(bad) > 0:
            logging.debug("Setting sphalb = 0 where rho025 == rho05 == rhoatm.")
            sphalb[bad] = 0

        if not np.all(np.isfinite(sphalb)):
            raise AttributeError("Non-finite values in spherical albedo calculation")

        transm = (rho05 - rhoatm) * (2.0 - sphalb)

        # For now, don't estimate this term!!
        # TODO: Have LibRadTran calculate it directly
        transup = np.zeros(self.wl.shape)

        # Get solar zenith, translate to irradiance at zenith = 0
        # HACK: If a file called `prescribed_geom` exists in the LUT directory,
        # use that instead of the LibRadtran calculated zenith angle. This is
        # not the most elegant or efficient solution, but it seems to work.
        zenfile = self.sim_path / "prescribed_geom"
        if not zenfile.exists():
            zenfile = self.sim_path / f"{name}.zen"

        with open(zenfile, "r") as fin:
            output = fin.read().split()
            solzen, solaz = [float(q) for q in output[1:]]

        coszen = np.cos(solzen / 360.0 * 2.0 * np.pi)
        irr /= coszen

        results = {
            "solzen": solzen,
            "coszen": coszen,
            "solar_irr": irr,
            "rhoatm": rhoatm,
            "transm_down_dif": transm,
            "sphalb": sphalb,
            "transm_up_dir": transup,
        }
        return results

    def postSim(self):
        pass

    def rebuild_cmd(self, point, name):
        vals = {"atmosphere": "midlatitude_summer"}
        vals.update(zip(self.lut_names, point))

        if "AOT550" in vals:
            vals["aerosol_visibility"] = self.ext550_to_vis(vals["AOT550"])

        if "H2OSTR" in vals:
            vals["h2o_mm"] = vals["H2OSTR"] * 10.0

        # Create input files from the template
        files = []
        for albedo in self.albedos:
            files.append(f"{name}_albedo-{albedo}")
            with open(self.sim_path / f"{files[-1]}.inp", "w") as f:
                # **env to insert paths into the template, such as isofit {data}
                f.write(self.template.format(albedo=albedo, **vals, **env))

        # Single regex pattern to capture time, latitude, longitude
        pattern = re.compile(
            r"time\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+).*?"
            r"latitude\s+([NS])\s+([0-9.]+).*?"
            r"longitude\s+([WE])\s+([0-9.]+)",
            re.DOTALL,
        )

        # Extract needed info from that template
        if match := pattern.search(self.template):
            year, month, day, hour, minute, second = map(int, match.groups()[:6])
            lat_dir, lat_val = match.group(7), float(match.group(8))
            lon_dir, lon_val = match.group(9), float(match.group(10))
            lat = lat_val if lat_dir == "N" else -lat_val
            lon = lon_val if lon_dir == "E" else -lon_val

        with open(self.sim_path / f"{name}.sh", "w") as file:
            file.write(
                SCRIPT_TEMPLATE.format(
                    env=self.environment,
                    uvspecs="\n".join(
                        f"{self.libradtran}/bin/uvspec < {file}.inp > {file}.out"
                        for file in files
                    ),
                    libradtran=self.libradtran,
                    lat=lat,
                    lon=lon,
                    date=f"{year} {day} {month} {hour} {minute}",
                    output=f"{name}.zen",
                )
            )

        return f"bash {name}.sh"

    @staticmethod
    def ext550_to_vis(ext550):
        return np.log(50.0) / (ext550 + 0.01159)
