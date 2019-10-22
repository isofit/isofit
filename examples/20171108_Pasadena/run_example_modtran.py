# David R Thompson, Adam Erickson

import subprocess
from isofit.utils import surfmodel

# Build the surface model
surfmodel("configs/ang20171108t184227_surface.json")

# Run retrievals
subprocess.call(["isofit", "--level", "DEBUG", "configs/ang20171108t173546_darklot.json"],
                shell=False)

subprocess.call(["isofit", "--level", "DEBUG", "configs/ang20171108t173546_horse.json"],
                shell=False)

subprocess.call(["isofit", "--level", "DEBUG", "configs/ang20171108t184227_astrored.json"],
                shell=False)

subprocess.call(["isofit", "--level", "DEBUG", "configs/ang20171108t184227_astrogreen.json"],
                shell=False)

subprocess.call(["isofit", "--level", "DEBUG", "configs/ang20171108t184227_beckmanlawn.json"],
                shell=False)

subprocess.call(["isofit", "--level", "DEBUG", "configs/ang20171108t184227_beckmanlawn-oversmoothed.json"],
                shell=False)

subprocess.call(["isofit", "--level", "DEBUG", "configs/ang20171108t184227_beckmanlawn-undersmoothed.json"],
                shell=False)
