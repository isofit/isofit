
from typing import Dict, List, Type
from isofit.configs.configs import BaseConfigSection

class OutputConfig(BaseConfigSection):
    """
    Output file(s) configuration.
    """
    _estimated_state_file_type = str
    estimated_state_file = None

    _estimated_reflectance_file_type = str
    estimated_reflectance_file = None

    _estimated_emission_file_type = str
    estimated_emission_file = None

    _modeled_radiance_file_type = str
    modeled_radiance_file = None

    _apparent_reflectance_file_type = str
    apparent_reflectance_file = None

    _path_radiance_file_type = str
    path_radiance_file = None

    _simulated_measurement_file_type = str
    simulated_measurement_file = None

    _algebraic_inverse_file_type = str
    algebraic_inverse_file = None

    _atmospheric_coefficients_file_type = str
    atmospheric_coefficients_file = None

    _radiometry_correction_file_type = str
    radiometry_correction_file = None

    _spectral_calibration_file_type = str
    spectral_calibration_file = None

    _posterior_uncertainty_file_type = str
    posterior_uncertainty_file = None

    _plot_directory_type = str
    plot_directory = None

    _data_dump_file_type = str
    data_dump_file = None


    def _check_config_validity(self) -> List[str]:
        self.get_option_keys()
        errors = list()

        #TODO: add flags for rile overright, and make sure files don't exist if not checked?

        return errors
