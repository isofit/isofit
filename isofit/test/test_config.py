#from isofit.core.isofit import Isofit
from isofit.configs.base_config import BaseConfigSection
from isofit.configs.configs import Config
from isofit.configs.configs import get_config_differences


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

#import pdb; pdb.set_trace()
#print(config1._get_expected_type_for_option_key('input'))
#print(config1._get_expected_type_for_option_key('input') == 'isofit.configs.sections.input_config.InputConfig')
#print(config1.get_config_options_as_dict())

def test_get_non_type_attributes():
  assert(config1._get_nontype_attributes() == ['input', 'output', 'forward_model', 'implementation'])
  assert(config1._get_type_attributes() == ['_input_type', '_output_type', \
      '_forward_model_type', '_implementation_type'])

  # would ideally also include an example config containing a hidden attribute
def test_get_hidden_attributes():
  assert(config1._get_hidden_attributes() == [])
  assert(config2._get_hidden_attributes() == [])

def test_get_all_element_names():
  assert(config1.get_all_element_names() == ['input', 'output', 'forward_model', 'implementation'])
  assert(config1.get_element_names() == ['forward_model', 'implementation', 'input', 'output'])

#print(config1.get_all_elements())


#print(config1.get_single_element_by_name('input'))


#print(config1.check_config_validity())


# functions in config.py

# returns empty dict since configs have no differences

def test_get_config_differences():
  assert(get_config_differences(config1, config1_duplicate) == {})
  assert(get_config_differences(config1, config2) == {'input': {'measured_radiance_file': \
      ('../remote/ang20171108t184227_rdn_v2p11_BeckmanLawn.txt', None), 'reference_reflectance_file': \
          ('../insitu/BeckmanLawn.txt', None)}})

#print(config1.get_config_as_dict())
#print(config1.get_config_errors())



def main():

  print('BEGINNING TESTS')

  test_get_non_type_attributes()
  test_get_hidden_attributes()
  test_get_all_element_names()

  print('COMPLETED')
      


main()


  