# David R Thompson, Adam Erickson

from isofit.core.isofit import Isofit
from isofit.utils import surface_model


# Build the surface model
surface_model("configs/prm20151026t173213_surface_coastal.json")

# Run retrievals
model = Isofit("configs/prm20151026t173213_D8W_6s.json")
model.run()
del model

model = Isofit("configs/prm20151026t173213_D8p5W_6s.json")
model.run()
del model

model = Isofit("configs/prm20151026t173213_D9W_6s.json")
model.run()
del model

model = Isofit("configs/prm20151026t173213_D9p5W_6s.json")
model.run()
del model
