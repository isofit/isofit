# David R Thompson, Adam Erickson

from isofit.core.isofit import Isofit
from isofit.utils import surface_model

# Build the surface model
surface_model("configs/ang20171108t184227_surface.json")

# Run retrievals
model = Isofit("configs/ang20171108t173546_darklot.json")
model.run()
del model

model = Isofit("configs/ang20171108t173546_horse.json")
model.run()
del model

model = Isofit("configs/ang20171108t184227_astrored.json")
model.run()
del model

model = Isofit("configs/ang20171108t184227_astrogreen.json")
model.run()
del model

model = Isofit("configs/ang20171108t184227_beckmanlawn.json")
model.run()
del model

model = Isofit("configs/ang20171108t184227_beckmanlawn-oversmoothed.json")
model.run()
del model

model = Isofit("configs/ang20171108t184227_beckmanlawn-undersmoothed.json")
model.run()
del model
