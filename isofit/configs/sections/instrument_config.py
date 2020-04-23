
from typing import Dict, List, Type
from isofit.configs.configs import BaseConfigSection
from isofit.configs.sections import StateVectorConfig
import os



class InstrumentUnknowns(BaseConfigSection):
    """
    Instrument Unknowns configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self.channelized_radiometric_uncertainty_file = None
        self._channelized_radiometric_uncertainty_file_type = str

        self.uncorrelated_radiometric_uncertainty = None
        self._uncorrelated_radiometric_uncertainty_type = float

        self.wavelength_calibration_uncertainty = None
        self._wavelength_calibration_uncertainty = float

        self.stray_srf_uncertainty = None
        self._stray_srf_uncertainty_type = float

        self.set_config_options(sub_configdic)

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

    def __init__(self, sub_configdic: dict = None):

        self._wavelength_file_type = str
        self.wavelength_file = None

        self._integrations_type = int
        self.integrations = None

        self._unknowns_type = InstrumentUnknowns
        self.unknowns = None

        self._fast_resample_type = bool
        self.fast_resample = True

        self._statevector_type = StateVectorConfig
        self.statevector = None

        self._snr_type = float
        self.snr = None

        self._parametric_noise_file_type = str
        self.parametric_noise_file = None

        self._pushbroom_noise_file_type = str
        self.pushbroom_noise_file = None

        self._nedt_noise_file_type = str
        self.nedt_noise_file = None

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

