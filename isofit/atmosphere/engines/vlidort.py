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
#
# Both emit_hitdump.exe and emit_radiance.exe take the same arguments. The HITRAN
# dump must be generated once per surface elevation (and atmospheric profile) as
# the layer structure changes the gas absorption cross sections. emit_radiance.exe
# then reads the dumps from ../data/Hitran_Dumps/Case_*_{sel:.4f}.dat, relative
# to the MASTERS directory. For emit_hitdump.exe, arg11 is unused
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
        self.dumps = self.exe.parent / "data" / "Hitran_Dumps"

        if self.configure_and_exit:
            # HITRAN dumps must be generated before any radiance calculations
            cmds = [cmd for cmd, _ in self.hitdump_cmds()]

            # Group the radiance calculations by surface elevation
            points = combos(self.lut_grid.values())
            i = self.lut_names.index("surface_elevation_km")
            points = sorted(points, key=lambda p: p[i])
            cmds += [self.make_cmd(p)[0] for p in points]

            file = self.sims / "jobs.txt"
            file.write_text("\n".join(cmds))
            Logger.info(f"Wrote configuration to {file}")
            sys.exit(0)

        self.makeHitDumps()

        # Run the first point upfront to retrieve the VLIDORT output wavelength grid
        # and cache the resampling matrix to the LUT wavelengths
        point = combos(self.lut_grid.values())[0]
        self.makeSim(point)

        self.sim_wl = self.parseSim(point)["wl"]
        self.matrix_H = calculate_resample_matrix(self.sim_wl, self.wl, self.fwhm)

    def hitdump_cmds(self):
        """
        Builds the emit_hitdump.exe commands for each unique surface elevation that
        does not yet have HITRAN dump files

        Returns
        -------
        list[tuple[str, float]]
            (command, surface elevation) pairs
        """
        # The dumps only depend on the surface elevation, use the first value of the other dimensions
        point = [vals[0] for vals in self.lut_grid.values()]
        i = self.lut_names.index("surface_elevation_km")

        cmds = []
        elevs = [round(e, 4) for e in self.lut_grid["surface_elevation_km"]]
        for elev in elevs:
            if any(self.dumps.glob(f"Case_*_{elev:.4f}.dat")):
                Logger.debug(
                    f"HITRAN dumps already exist for elevation {elev:.4f}, skipping"
                )
                continue

            point[i] = elev
            out = self.sims / f"hitdump_{elev:.4f}.log"
            cmd, _ = self.make_cmd(point, exe="emit_hitdump.exe", out=out)
            cmds.append((cmd, elev))

        return cmds

    def makeHitDumps(self):
        """
        Generates the HITRAN dumps for each surface elevation in parallel. These must
        exist before emit_radiance.exe is executed for that elevation
        """
        cmds = self.hitdump_cmds()
        if not cmds:
            return

        Logger.info(f"Generating HITRAN dumps for {len(cmds)} surface elevations")
        procs = []
        for cmd, elev in cmds:
            Logger.debug(f"Executing {cmd}")
            procs.append((subprocess.Popen(cmd.split(" "), cwd=self.exe), elev))

        if failed := [elev for proc, elev in procs if proc.wait() != 0]:
            raise RuntimeError(
                f"emit_hitdump.exe failed for surface elevations: {failed}"
            )

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

    def make_cmd(self, point, exe="emit_radiance.exe", out=None):
        file = out or self.sims / self.point_to_filename(point)

        dims = dict(zip(self.lut_names, point))
        vals = {
            "exe": exe,
            "sza": dims["solar_zenith"],
            "vza": dims["observer_zenith"],
            "rza": dims["relative_azimuth"],
            "pwv": dims["H2OSTR"],
            "aod": dims["AOT550"],
            "vel": 58.0,  # Required to be 58.0 per Vijay
            "sel": round(dims["surface_elevation_km"], 4),
            "wgs": self.wl_spacing,
            "co2": dims["CO2"],
            "ch4": 1.9,  # REVIEW: 1.9 ppm is roughly the global average
            "out": file,
        }

        return CMD.format(**vals), file
