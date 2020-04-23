
from typing import Dict, List, Type
from isofit.configs.configs import BaseConfigSection
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


class SurfaceConfig(BaseConfigSection):
    """
    Instrument configuration.
    """

    def __init__(self, sub_configdic: dict = None):

        self._surface_file_type = str
        self.surface_file = None

        self._surface_category_type = str
        self.surface_category = None

        # Multicomponent Surface
        self._select_on_init_type = bool
        self.select_on_init = False

        self._selection_metric_type = str
        self.selection_metric = 'Mahalanobis'

        self._normalize_type = str
        self.normalize = 'Euclidean'

        # Surface Thermal
        self._emissivity_for_surface_T_init_type = float
        self.emissivity_for_surface_T_init_type = 0.98

        self._surface_T_prior_sigma_degK_type = float
        self.surface_T_prior_sigma_degK = 1.

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        valid_surface_categories = ['surface','multicomponent_surface','glint_surface','thermal_surface']
        if self.surface_category is None:
            errors.append('surface->surface_category must be specified')
        elif self.surface_category not in valid_surface_categories:
            errors.append('surface->surface_category: {} not in valid surface categories: {}'.format(self.surface_category, valid_surface_categories))

        if self.surface_category is None:
            errors.append('surface->surface_category must be specified')

        valid_normalize_categories = ['Euclidean','RMS','None']
        if self.normalize not in valid_normalize_categories:
            errors.append('surface->normalize: {} not in valid surface categories: {}'.format(self.normalize, valid_normalize_categories))


        return errors

