from isofit.configs import configs
from isofit.core import isofit
from isofit.core.forward import ForwardModel
from isofit.inversion.inverse import Inversion

config = configs.create_new_config(config_file="/mnt/data/20230401_ASO/L2A_reflectance/config/asovnir20230401t163434_isofit.json")
fm = ForwardModel(config)
inv = Inversion(config, fm)
workers = isofit.Worker(config, fm, loglevel="INFO", logfile=None)

# i1 = np.random.randint(3000)
# i2 = np.random.randint(598)
input_data = workers.io.get_components_at_index(0, 0)
res = workers.iv.invert(input_data.meas, input_data.geom)

