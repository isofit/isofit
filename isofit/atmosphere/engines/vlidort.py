from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ray.util.queue import Queue

from isofit.atmosphere import BaseAtmosphere
from isofit.data import env
from isofit.luts.writer import Writer

Logger = logging.getLogger(__name__)

# {sza} = arg1  = solar zenith angle (degrees)
# {vza} = arg2  = viewing zenith angle (degrees)
# {rza} = arg3  = relative azimuth angle (degrees)
# {pwv} = arg4  = precipitable water vapor (cm)
# {aod} = arg5  = AOD at 550 nm
# {vel} = arg6  = viewing elevation (TOA = 58 km); current set up needs this to be 58.0
# {sel} = arg7  = surface elevation (km)
# {wgs} = arg8  = wavelength grid spacing (nm)
# {co2} = arg9  = xco2 (ppm)
# {ch4} = arg10 = xch4 (ppm)
# {out} = arg11 = output file path
# Example: ./emit_radiance.exe 35 0 0 2.0 0.1 58.0 0.0 0.1 400.0 1.9 emit_outputs_11args.dat
#          {sza} {vza} {rza} {pwv} {aod} {vel} {sel} {wgs} {co2} {ch4}
#            35     0     0   2.0   0.1  58.0   0.0   0.1  400.0  1.9
CMD = """\
./{exe} {sza} {vza} {rza} {pwv} {aod} {vel} {sel} {wgs} {co2} {ch4} {out}\
"""


class VLIDORT(BaseAtmosphere, Writer):
    required = {
        "solar_zenith",
        "observer_zenith",
        "relative_azimuth",
        "H2OSTR",
        "AOT550",
        "surface_elevation_km",
        "CO2",
    }

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def preSim(self):
        if missing := self.required - set(self.lut_names):
            raise AttributeError(f"Missing required LUT dimensions: {missing}")

        spacing = np.unique(np.round(np.diff(self.wl), decimals=6))
        if spacing.size > 1:
            raise ValueError(f"Inconsistent wavelength spacing: {spacing}")

        (self.wl_spacing,) = spacing
        Logger.debug(f"Detected wavelength spacing: {self.wl_spacing}")

        self.sims = Path(self.config.sim_path)
        self.sims.mkdir(parents=True, exist_ok=True)

        self.exe = Path(self.config.engine_base_dir) / "MASTERS"

        return  # TODO: Finish implementing the pre-run

        PRE = """\
        ./emit_hitdump.exe {sza} {vza} {rza} {pwv} {aod} {vel} {sel} {wgs} {co2} {ch4} {out}\
        """

    def makeSim(self, point, **_):
        name = self.point_to_filename(point)
        file = self.sims / name
        if file.exists():
            Logger.debug(
                f"Sim data file for this point already exists, skipping. Point = {name}"
            )
            return

        dims = dict(zip(self.lut_names, point))
        vals = {
            "exe": "emit_radiance.exe",
            "sza": dims["solar_zenith"],
            "vza": dims["observer_zenith"],
            "rza": dims["relative_azimuth"],
            "pwv": dims["H2OSTR"],
            "aod": dims["AOT550"],
            "vel": 58.0,  # Required to be 58.0 per Vijay
            "sel": dims[
                "surface_elevation_km"
            ],  # This doesn't work with any nonzero value?
            "sel": 0.0,
            "wgs": self.wl_spacing,
            "co2": dims["CO2"],
            "ch4": 0.0,  # REVIEW
            "out": file,
        }
        cmd = CMD.format(**vals)

        subprocess.run(
            cmd.split(" "),
            cwd=self.exe,
            check=True,
        )

    def readSim(self, point):
        name = self.point_to_filename(point)
        file = self.sims / name

        lines = file.read_text().splitlines()
        parse = []
        for line in lines:
            data = re.findall(r"(\S+)", line)
            data = np.array(data).astype(float)
            parse.append(data)

        data = np.vstack(parse)

        # wl, PATH, TDIFFDN, TDIRDN, TDIFFUP, TDIRUP, SPHER
        cols = [
            "wl",
            "rhoatm",
            "transm_down_dif",
            "transm_down_dir",
            "transm_up_dif",
            "transm_up_dir",
            "sphalb",
        ]
        data = dict(zip(cols, data.T))

        return data
