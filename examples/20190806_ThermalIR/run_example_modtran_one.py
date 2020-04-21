# David R Thompson, Adam Erickson, Jay Fahlen
from isofit.core.isofit import Isofit
from isofit.utils import surface_model

# Build the surface model
surface_model("configs/surface.json")

# Run retrievals
model1 = Isofit("configs/joint_isofit_with_prof_WATER_nogrid.json").run(debug = True)


