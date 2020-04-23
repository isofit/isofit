
from typing import Dict, List, Type
from isofit.configs.configs import BaseConfigSection
from isofit.configs.sections import StateVectorConfig
import os



class InstrumentUnknowns(BaseConfigSection):
    """
    Instrument Unknowns configuration.
    """
    channelized_radiometric_uncertainty_file = None
    _channelized_radiometric_uncertainty_file_type = str

    uncorrelated_radiometric_uncertainty = None
    _uncorrelated_radiometric_uncertainty_type = float

    wavelength_calibration_uncertainty = None
    _wavelength_calibration_uncertainty = float

    stray_srf_uncertainty = None
    _stray_srf_uncertainty_type = float

    def _check_config_validity(self) -> List[str]:
        self.get_option_keys()
        errors = list()

        file_params = [self.channelized_radiometric_uncertainty_file, self.uncorrelated_radiometric_uncertainty]
        for param in file_params:
            if param is not None:
                if os.path.isfile(param) is False:
                    errors.append('Instrument unknown file: {} not found'.format(param))


class InstrumentConfig(BaseConfigSection):
    """
    Instrument configuration.
    """

    _wavelength_file_type = str
    wavelength_file = None

    _integrations_type = int
    integrations = None

    _unknowns_type = InstrumentUnknowns
    unknowns = None

    _fast_resample_type = bool
    fast_resample = True

    _statevector_type = StateVectorConfig
    statevector = None

    _snr_type = float
    snr = None

    _parametric_noise_file_type = str
    parametric_noise_file = None

    _pushbroom_noise_file_type = str
    pushbroom_noise_file = None

    _nedt_noise_file_type = str
    nedt_noise_file = None

    def __init__(self, sub_configdic: dict = None):
        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        self.get_option_keys()
        errors = list()

        noise_options = [self.snr, self.parametric_noise_file, self.pushbroom_noise_file, self.nedt_noise_file]
        used_noise_options = [x for x in noise_options if x is not None]

        if len(used_noise_options) == 0:
            errors.append('Instrument noise not defined.')

        if len(used_noise_options) > 1:
            errors.append('Multiple instrument noise options selected - please choose only 1.')

        file_params = [self.parametric_noise_file, self.pushbroom_noise_file, self.nedt_noise_file]
        for param in file_params:
            if param is not None:
                if os.path.isfile(param) is False:
                    errors.append('Instrument config file: {} not found'.format(param))

        #TODO: figure out submodule checking

        return errors

