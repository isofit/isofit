# David R Thompson, Adam Erickson

import subprocess
from isofit.utils import surfmodel

# Build the surface model
surfmodel("configs/prm20151026t173213_surface_coastal.json")

# Run retrievals
subprocess.call(["isofit", "--level", "DEBUG", "configs/prm20151026t173213_D8W_6s.json"],
                shell=False)
subprocess.call(["isofit", "--level", "DEBUG", "configs/prm20151026t173213_D8p5W_6s.json"],
                shell=False)
subprocess.call(["isofit", "--level", "DEBUG", "configs/prm20151026t173213_D9W_6s.json"],
                shell=False)
subprocess.call(["isofit", "--level", "DEBUG", "configs/prm20151026t173213_D9p5W_6s.json"],
                shell=False)
