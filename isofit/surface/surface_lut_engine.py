import numpy as np
import logging
import subprocess
from os import getenv
from os.path import abspath, exists, join
from pathlib import Path


from isofit.core.common import (
    calculate_resample_matrix,
    json_load_ascii,
    resample_spectrum,
)
from isofit.luts.writer import Reader, Writer

Logger = logging.getLogger(__name__)


class SurfaceLUTEngine(Reader):
    """Used to bridge any BRDF/Surface model into an ISOFIT formatted LUT... very general so Isofit does not need to hold
    many brdf rt codes and optical propertis.

    TO think of better names for the file and class.

    TODO where is this thing being saved? , logic if the user does this but does not provide a path in the surface file?
    Probably should just error out, and ask for them to populate this path.. This is nice because that would also allow
    to run for different Surface LUTs.

    """

    def __init__(self, full_config: Config):

        # TODO
        # So not here, but in surface.py, will need to include (RIGHT BEFORE LOAD-MAT) an iff statement to try and trigger this if needed.
        # and also within surface.py, it will need to try inherit the lut grid

        # NOTE this expects the user to have a full path to an executable shell script on their machine
        # For the user this could be as easy as having say some python code, etc, and
        # making a shell wrapper for it .. that takes a txt file as input.
        self.surface_lut_engine_script = "TODO"

        self.sh_template = "#!/bin/bash\n"

        pass

    def preSim(self):
        """
        TODO
        """

        # Run on first point to gather static variables
        self.makeSim(point=self.points[0])
        self.readSim(point=self.points[0], presim=True)

        # Compute resampling matrix to be used in readSim for storing data in sensor wl
        self.matrix_H = calculate_resample_matrix(self.engine_wl, self.wl, self.fwhm)

        # Resample endmembers if present
        if self.endmembers:
            results = {}
            for i in self.endmembers:
                results[f"endmember_{i}"] = self.endmembers[i]

            # Resample all quantities to sensor wavelengths using H matrix
            results = {
                key: resample_spectrum(
                    data, self.engine_wl, self.wl, self.fwhm, H=self.matrix_H
                )
                for key, data in results.items()
            }

            return results

    def makeSim(self, point: np.array):
        """
        TODO
        """

        # Retrieve the files to process
        name = self.point_to_filename(point)

        # Rebuild command
        cmd = self.rebuild_cmd(point, name)

        # Only execute when the .out file is missing
        sim_path = abspath(join(self.sim_path, f"{name}.OUT"))
        if exists(sim_path):
            Logger.warning(f"libRadtran sim files already exist for point {point}")
            return

        if not self.engine_config.rte_configure_and_exit:
            call = subprocess.run(cmd, shell=True, capture_output=True)
            if call.stdout:
                Logger.error(call.stdout.decode())

    def readSim(self, point: np.array, presim=False):
        """
        TODO
        """

        # Output columns: wl, rho_dif_dir, rho_dir_dir, and then any endmembers
        name = self.point_to_filename(point)
        sim = np.loadtxt(f"{name}.OUT")

        # If we are presim gather wl, endmembers if present, and return
        if presim:
            self.engine_wl = sim[:, 0].T
            self.endmembers = {}
            if sim.shape[1] >= 4:
                for i in range(0, len(sim.shape[1])):
                    if i < 2:
                        continue
                    else:
                        self.endmembers[i] = sim.shape
            return

        # Store and check user provided all fields
        results = {}
        try:
            results["rho_dif_dir"] = sim[:, 1].T
        except:
            raise ValueError(
                f"Error loading data from index 1 for rho_dif_dir for point: {point}"
            )
        try:
            results["rho_dir_dir"] = sim[:, 2].T
        except:
            raise ValueError(
                f"Error loading data from index 2 for rho_dir_dir for point {point}"
            )

        # Resample all quantities to sensor wavelengths using H matrix
        results = {
            key: resample_spectrum(
                data, self.engine_wl, self.wl, self.fwhm, H=self.matrix_H
            )
            for key, data in results.items()
        }
        return results

    def postSim(self):
        """
        Not needed but required to be present. keeping pass.
        """
        pass

    def rebuild_cmd(self, point, name):

        # Add the point with its names to an input
        sim_inputs = ""
        for key, val in zip(self.lut_names, point):
            sim_inputs += f"{key} {val}\n"

        # Write input text file
        inp_file = Path(abspath(join(self.sim_path, f"{name}.INP")))
        inp_file.write_text(sim_inputs)

        # Create shell script to run sim
        sh_file = Path(abspath(join(self.sim_path, f"{name}.sh")))
        sh_inp = self.sh_template
        sh_inp += f"{self.surface_lut_engine_script} {inp_file}"
        sh_file.write_text(sh_inp)

        return f"bash {sh_file}"
