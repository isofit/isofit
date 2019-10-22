# David R Thompson, Adam Erickson


from isofit.isofit import Isofit
from isofit.utils import surfmodel


# Build the surface model
surfmodel("configs/prm20151026t173213_surface_coastal.json")

# Run retrievals
model1 = Isofit("configs/prm20151026t173213_D8W_6s.json")
model1.run()

model2 = Isofit("configs/prm20151026t173213_D8p5W_6s.json")
model2.run()

model3 = Isofit("configs/prm20151026t173213_D9W_6s.json")
model3.run()

model4 = Isofit("configs/prm20151026t173213_D9p5W_6s.json")
model4.run()
