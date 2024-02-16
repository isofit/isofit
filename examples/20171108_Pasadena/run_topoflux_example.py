from isofit.core.isofit import Isofit
from isofit.utils import surface_model

# Disable Modtran's makeSim for local testing
from isofit.radiative_transfer.modtran import ModtranRT
ModtranRT.makeSim = lambda *_: ...

# Build the surface model
surface_model("configs/ang20171108t184227_surface.json")

# Run retrievals
model = Isofit("configs/ang20171108t184227_beckmanlawn-multimodtran-topoflux.json")
model.run()
del model
