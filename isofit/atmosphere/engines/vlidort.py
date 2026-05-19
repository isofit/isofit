from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from isofit import ray
from isofit.atmosphere import BaseAtmosphere
from isofit.data import env
from isofit.luts.writer import Writer

Logger = logging.getLogger(__name__)

# arg1 = solar zenith angle (degrees)
# arg2 = viewing zenith angle (degrees)
# arg3 = relative azimuth angle (degrees)
# arg4 = precipitable water vapor (cm)
# arg5 = AOD at 550 nm
# arg6 = viewing elevation (TOA = 58 km); current set up needs this to be 58.0
# arg7 = surface elevation (km)
# arg8 = wavelength grid spacing (nm)
# arg9 = xco2 (ppm)
# arg10 = xch4 (ppm)
# Example: ./emit.exe 35 0 0 2.0 0.1 58.0 0.0 0.1 400.0 1.9
CMD = """\
./emit.exe {sza} {vza} {rza} {pwv} {aod} {vel} {sel} {wgs} {co2} {ch4}\
"""


@ray.remote
class TempDirPool:
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

    def __init__(self, n: int, dir: str):
        # Root temp directory
        self.root = Path(
            tempfile.mkdtemp(
                prefix="vlidort_",
            )
        )

        self.tmps = []
        data = Path(dir)
        full = list(data.glob("*"))
        part = list((data / "MASTERS").glob("*"))

        # Create pool dirs
        for i in range(n):
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

            self.tmps.append(path)

    def get(self) -> Path:
        """
        Get an available temp directory

        Returns
        -------
        pathlib.Path
            Path to temp dir
        """
        return self.tmps.pop()

    def free(self, path: Path) -> None:
        """
        Release a temp directory back to the pool
        """
        self.tmps.append(path)

    def cleanup(self) -> None:
        """
        Remove entire temp pool
        """
        shutil.rmtree(self.root, ignore_errors=True)
        self.tmps.clear()


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

    def preSim(self):
        if missing := self.required - set(self.lut_names):
            raise AttributeError(f"Missing required LUT dimensions: {missing}")

        self.queue = TempDirPool.remote(n=self.n_cores, dir=self.config.engine_base_dir)

        spacing = np.unique(np.diff(self.wl))
        if spacing.size > 1:
            raise ValueError("")
        (self.wl_spacing,) = spacing

    def makeSim(self, point, **kwargs):
        temp = self.queue.get.remote()
        os.chdir(temp)

        dims = zip(self.lut_names, point)
        vals = {
            "sza": dims["solar_zenith"],
            "vza": dims["observer_zenith"],
            "rza": dims["relative_azimuth"],
            "pwv": dims["H2OSTR"],
            "aod": dims["AOT550"],
            "vel": 58.0,  # Required to be 58.0 per Vijay
            "sel": dims["surface_elevation_km"],
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

        return {"temp": temp}

    def readSim(self, point, temp):
        lines = (temp / "fort.40").read_text().splitlines()
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

        self.queue.free.remote(temp)

        return data

    def postSim(self):
        self.queue.cleanup.remote()
