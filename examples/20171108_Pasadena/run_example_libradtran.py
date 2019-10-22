# David R Thompson, Adam Erickson

import subprocess
from isofit.utils import surfmodel

# Build the surface model
surfmodel("configs/ang20171108t184227_surface.json")

# Run retrievals
subprocess.call(["isofit", "--level", "DEBUG",
                 "configs/ang20171108t184227_beckmanlawn-libradtran.json"],
                shell=False)
