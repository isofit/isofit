#from isofit.core.isofit import Isofit
from isofit.configs.base_config import BaseConfigSection
from isofit.configs.configs import Config
from isofit.utils import surface_model
from isofit.core.forward import ForwardModel
from isofit.configs.configs import create_new_config, get_config_differences
from isofit.inversion.inverse import Inversion
from isofit.core.fileio import IO


print('BUILDING ...')


# Surface model
surface_model("20171108_Pasadena/configs/ang20171108t184227_surface.json")

# Instantiate Isofit
#example = Isofit("20171108_Pasadena/configs/ang20171108t184227_beckmanlawn.json")
#example.run()
#del example

# created dictionary with first two keys and values
# of beckmanlawn.json
sample_dict_1 = {
  "ISOFIT_BASE": "../../..",

  "input": {
    "measured_radiance_file": "../remote/ang20171108t184227_rdn_v2p11_BeckmanLawn.txt",
    "reference_reflectance_file": "../insitu/BeckmanLawn.txt"
  }}

sample_dict_2 = {
  "ISOFIT_BASE": "../../.."
  }

sample_dict_3 = {

  "input": {
    "measured_radiance_file": "../remote/ang20171108t184227_rdn_v2p11_BeckmanLawn.txt",
    "reference_reflectance_file": "../insitu/BeckmanLawn.txt"
  }}


config1 = Config(sample_dict_1)
config1_duplicate = Config(sample_dict_1)
config2 = Config(sample_dict_2)
config3 = Config(sample_dict_3)
assert(config1._get_nontype_attributes() == ['input', 'output', 'forward_model', 'implementation'])
assert(config1.get_all_element_names() == ['input', 'output', 'forward_model', 'implementation'])
#print(config1.get_all_elements())
assert(config1._get_type_attributes() == ['_input_type', '_output_type', \
    '_forward_model_type', '_implementation_type'])
assert(config2._get_type_attributes() == ['_input_type', '_output_type', \
    '_forward_model_type', '_implementation_type'])
assert(config2._get_hidden_attributes() == [])
assert(config3._get_hidden_attributes() == [])
#print(config1.get_single_element_by_name('input'))


#print(config1.check_config_validity())


# functions in config.py

# returns empty dict since configs have no differences
assert(get_config_differences(config1, config1_duplicate) == {})
assert(get_config_differences(config1, config2) == {'input': {'measured_radiance_file': \
    ('../remote/ang20171108t184227_rdn_v2p11_BeckmanLawn.txt', None), 'reference_reflectance_file': \
         ('../insitu/BeckmanLawn.txt', None)}})

#print(config1.get_config_as_dict())
#print(config1.get_config_errors())

"""
config = create_new_config("20171108_Pasadena/configs/ang20171108t184227_beckmanlawn.json")
fm = ForwardModel(config)
inv = Inversion(config, fm)
io = IO(config, fm)

io.get_components_at_index(0, 0)
geom = io.current_input_data.geom # alternately, call via geom = Geometry()...this won't have data from the above config file
meas = io.current_input_data.meas  # alternately, pass in a num_wavelength numpy array (e.g., 425)

"""

print('END')