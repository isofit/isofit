from isofit.inversion.inverse import error_code

from isofit.configs.base_config import BaseConfigSection
from isofit.configs.configs import Config
from isofit.utils import surface_model
from isofit.core.forward import ForwardModel
from isofit.configs.configs import create_new_config, get_config_differences
from isofit.inversion.inverse import Inversion
from isofit.core.fileio import IO
import numpy as np


def test_error_code():
    assert error_code == -1

print('BUILDING ...')

# Surface model
surface_model("examples/20171108_Pasadena/configs/ang20171108t184227_surface.json")

config = create_new_config("examples/20171108_Pasadena/configs/ang20171108t184227_beckmanlawn.json")
fm = ForwardModel(config)

x = np.random.rand(427)
x[425] = 1.75
x[426] = 0.05
val = 0
for i in range(425):
  val = val + 0.001
  x[i] = val


inv = Inversion(config, fm)
io = IO(config, fm)

io.get_components_at_index(0, 0)
geom = io.current_input_data.geom # alternately, call via geom = Geometry()...this won't have data from the above config file
meas = io.current_input_data.meas  # a numpy  array

assert(inv.full_statevector(x).all() == x.all()) # inv.self_fixed = None

radiance_measurement = fm.calc_rdn(x, geom)
print(type(inv.invert(radiance_measurement, geom)))
print(inv.invert(radiance_measurement, geom).shape)
print(inv.invert(radiance_measurement, geom)[0])

print('TESTS COMPLETE')




