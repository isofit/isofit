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
# Example: ./emit.exe 35 0 0 2.0 0.1 58.0 0.0 0.1 400.0 1.9
# {sza} {vza} {rza} {pwv} {aod} {vel} {sel} {wgs} {co2} {ch4}
#   35     0     0   2.0   0.1  58.0   0.0   0.1  400.0  1.9
CMD = """\
./emit.exe {sza} {vza} {rza} {pwv} {aod} {vel} {sel} {wgs} {co2} {ch4}\
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

    def __init__(self, *args, save_sim=True, **kwargs):
        self.save_sim = save_sim
        super().__init__(*args, **kwargs)

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

        self.queue = self.spoof()

    def makeSim(self, point, **_):
        name = self.point_to_filename(point)
        file = self.sims / name
        if file.exists():
            Logger.debug(
                f"Sim data file for this point already exists, skipping. Point = {name}"
            )
            return {"file": file}

        temp = self.queue.get()
        os.chdir(temp)

        dims = dict(zip(self.lut_names, point))
        vals = {
            "sza": dims["solar_zenith"],
            "vza": dims["observer_zenith"],
            "rza": dims["relative_azimuth"],
            "pwv": dims["H2OSTR"],
            "aod": dims["AOT550"],
            "vel": 58.0,  # Required to be 58.0 per Vijay
            # "sel": dims["surface_elevation_km"], # This doesn't work with any nonzero value?
            "sel": 0.0,
            "wgs": self.wl_spacing,
            "co2": dims["CO2"],
            "ch4": 0.0,  # REVIEW
        }
        cmd = CMD.format(**vals)

        subprocess.run(
            cmd.split(" "),
            cwd=temp,
            check=True,
        )

        if self.save_sim:
            data = temp / "fort.40"
            shutil.move(data, file)

            return {"file": file}

        return {"temp": temp}

    def readSim(self, point, temp=None, file=None):
        if not file:
            file = temp / "fort.40"

        lines = file.read_text().splitlines()
        parse = []
        for line in lines:
            data = re.findall(r"(\S+)", line)
            data = np.array(data).astype(float)
            parse.append(data)

        data = np.vstack(parse)

        # REVIEW: Are these correct?
        cols = [
            "wl",
            "rhoatm",
            "sphalb",
            "transm_up_dif",
            "transm_up_dir",
            "transm_down_dif",
            "transm_down_dir",
        ]
        data = dict(zip(cols, data.T))

        if temp:
            self.queue.put(temp)

        return data

    def postSim(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def spoof(self):
        """
        VLIDORT is hardcoded to write to the output file fort.40
        This tricks it into writing to different locations by using softlinks since
        VLIDORT uses relative pathing. By doing so, we can enable parallelism.

        Parameters
        ----------
        n : int
            Number of temp directories to create.
        dir : str
            Directory of VLIDORT to symlink to
        """
        # Root temp directory
        self.root = Path(
            tempfile.mkdtemp(
                prefix="vlidort_",
            )
        )

        tmps = Queue()
        base = Path(self.config.engine_base_dir)
        full = list(base.glob("*"))
        part = list((base / "MASTERS").glob("*"))

        # Create pool dirs
        for i in range(self.n_cores):
            path = self.root / f"tmp_{i}"

            path.mkdir(parents=True, exist_ok=True)
            for obj in full:
                if obj.name == "MASTERS":
                    continue
                (path / obj.name).symlink_to(obj)

            path /= "MASTERS"
            path.mkdir(parents=True, exist_ok=True)
            for obj in part:
                (path / obj.name).symlink_to(obj)

            tmps.put(path)

        return tmps
