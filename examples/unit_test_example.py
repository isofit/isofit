from isofit.core.isofit import Isofit
from isofit.utils import surface_model

# Surface model
surface_model("20171108_Pasadena/configs/ang20171108t184227_surface.json")

# Instantiate Isofit
example = Isofit("20171108_Pasadena/configs/ang20171108t184227_beckmanlawn.json")
example.run()
del example