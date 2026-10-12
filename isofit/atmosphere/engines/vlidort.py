from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ray.util.queue import Queue

from isofit.atmosphere import BaseAtmosphere
from isofit.core.common import calculate_resample_matrix, combos, resample_spectrum
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

    # VLIDORT simulation wavelength grid spacing (nm), convolved to the LUT wavelengths
    wl_spacing = 0.05

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def preSim(self):
        if missing := self.required - set(self.lut_names):
            raise AttributeError(f"Missing required LUT dimensions: {missing}")

        self.sims = Path(self.config.sim_path)
        self.sims.mkdir(parents=True, exist_ok=True)

        self.exe = Path(self.config.engine_base_dir) / "MASTERS"

        if self.configure_and_exit:
            points = combos(self.lut_grid.values())
            cmds = [self.make_cmd(p)[0] for p in points]

            file = self.sims / "jobs.txt"
            file.write_text("\n".join(cmds))
            Logger.info(f"Wrote configuration to {file}")
            sys.exit(0)

        # Run the first point upfront to retrieve the VLIDORT output wavelength grid
        # and cache the resampling matrix to the LUT wavelengths. This must be done
        # here as readSim is executed in independent ray jobs
        point = combos(self.lut_grid.values())[0]
        self.makeSim(point)

        self.sim_wl = self.parseSim(point)["wl"]
        self.matrix_H = calculate_resample_matrix(self.sim_wl, self.wl, self.fwhm)
        # Logger.debug(f"VLIDORT output grid: {self.sim_wl[0]}-{self.sim_wl[-1]} nm, {self.sim_wl.size} wavelengths")

        return  # TODO: Finish implementing the pre-run

        PRE = """\
        ./emit_hitdump.exe {sza} {vza} {rza} {pwv} {aod} {vel} {sel} {wgs} {co2} {ch4} {out}\
        """

    def makeSim(self, point, **_):
        cmd, file = self.make_cmd(point)

        if file.exists():
            Logger.debug(
                f"Sim data file for this point already exists, skipping. Point = {file.name}"
            )
            return

        Logger.debug(f"Executing {cmd}")
        subprocess.run(
            cmd.split(" "),
            cwd=self.exe,
            check=True,
        )

    def parseSim(self, point):
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

    def readSim(self, point):
        data = self.parseSim(point)

        # Convolve from the VLIDORT wavelength grid to the LUT wavelengths using the matrix cached in preSim
        wl = data.pop("wl")
        H = self.matrix_H
        if not np.array_equal(wl, self.sim_wl):
            Logger.warning(
                f"VLIDORT output wavelengths differ from the cached grid for point {point}, recalculating the resampling matrix"
            )
            H = calculate_resample_matrix(wl, self.wl, self.fwhm)

        data = {
            key: resample_spectrum(vals, wl, self.wl, self.fwhm, H=H)
            for key, vals in data.items()
        }

        return data

    def make_cmd(self, point):
        name = self.point_to_filename(point)
        file = self.sims / name

        dims = dict(zip(self.lut_names, point))
        vals = {
            "exe": "emit_radiance.exe",
            "sza": dims["solar_zenith"],
            "vza": dims["observer_zenith"],
            "rza": dims["relative_azimuth"],
            "pwv": dims["H2OSTR"],
            "aod": dims["AOT550"],
            "vel": 58.0,  # Required to be 58.0 per Vijay
            "sel": round(
                dims["surface_elevation_km"]
            ),  # REVIEW: Temporarily must be an int, eventually remove the rounding once VLIDORT fixes it
            "wgs": self.wl_spacing,
            "co2": dims["CO2"],
            "ch4": 1.9,  # REVIEW: 1.9 ppm is roughly the global average
            "out": file,
        }

        return CMD.format(**vals), file
