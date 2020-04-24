
from typing import Dict, List, Type
from isofit.configs.base_config import BaseConfigSection

class OutputConfig(BaseConfigSection):
    """
    Output file(s) configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._estimated_state_file_type = str
        self.estimated_state_file = None

        self._estimated_reflectance_file_type = str
        self.estimated_reflectance_file = None

        self._estimated_emission_file_type = str
        self.estimated_emission_file = None

        self._modeled_radiance_file_type = str
        self.modeled_radiance_file = None

        self._apparent_reflectance_file_type = str
        self.apparent_reflectance_file = None

        self._path_radiance_file_type = str
        self.path_radiance_file = None

        self._simulated_measurement_file_type = str
        self.simulated_measurement_file = None

        self._algebraic_inverse_file_type = str
        self.algebraic_inverse_file = None

        self._atmospheric_coefficients_file_type = str
        self.atmospheric_coefficients_file = None

        self._radiometry_correction_file_type = str
        self.radiometry_correction_file = None

        self._spectral_calibration_file_type = str
        self.spectral_calibration_file = None

        self._posterior_uncertainty_file_type = str
        self.posterior_uncertainty_file = None

        self._plot_directory_type = str
        self.plot_directory = None

        self._data_dump_file_type = str
        self.data_dump_file = None


        self.set_config_options(sub_configdic)


    def _check_config_validity(self) -> List[str]:
        errors = list()

        #TODO: add flags for rile overright, and make sure files don't exist if not checked?

        return errors
