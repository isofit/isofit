


from typing import Dict, List, Type
from isofit.configs.configs import BaseConfigSection
import os

class InputConfig(BaseConfigSection):
    """
    Input file(s) configuration.
    """

    _measured_radiance_file_type = str
    measured_radiance_file = None
    """
    str: Input radiance file.  Can be either a .mat, .txt, or ENVI formatted binary cube. 
    Used for inverse-modeling (radiance -> reflectance).
    """

    _reference_reflectance_file_type = str
    reference_reflectance_file = None
    """
    str: Input reference reflectance file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
    Used for radiometric calibration.
    """

    _reflectance_file_type = str
    reflectance_file = None
    """
    str: Input reflectance file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
    Used for forward-modeling (reflectance -> radiance).
    """

    _obs_file_type = str
    obs_file = None
    """
    str: Input 'obs', or observation, file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
    Provides information about the conditions during observaiton.  Assumed to be in the band-wise format:
    {path length, to-sensor azimuth, to-sensor zenith, to-sun azimuth, to-sun zenith, phase, slope, aspect, cosine i, 
    UTC time}
    """

    _glt_file_type = str
    glt_file = None
    """
    str: Input glt file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
    Provides (x,y) offset information for the spatial location of raw-space input files
    """

    _loc_file_type = str
    loc_file = None
    """
    str: Input 'loc', or location, file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
    Provides per-pixel lat, long, and elevation information.
    """

    _surface_prior_mean_file_type = str
    surface_prior_mean_file = None
    """
    str: Input surface prior mean file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
    """

    _surface_prior_variance_file_type = str
    surface_prior_variance_file = None
    """
    str: Input surface prior variance file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
    """

    _rt_prior_mean_file_type = str
    rt_prior_mean_file = None
    """
    str: Input rt prior mean file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
    """

    _rt_prior_variance_file_type = str
    rt_prior_variance_file = None
    """
    str: Input rt prior variance file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
    """

    _instrument_prior_mean_file_type = str
    instrument_prior_mean_file = None
    """
    str: Input instrument prior mean file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
    """

    _instrument_prior_variance_file_type = str
    instrument_prior_variance_file = None
    """
    str: Input instrument prior variance file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
    """

    _radiometry_correction_file_type = str
    radiometry_correction_file = None
    """
    str: Input radiometric correction file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
    Used to make minor channelized corrections to account for slight systematic errors not captured in calibration.
    """

    def __init__(self, sub_configdic: dict = None):
        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        self.get_option_keys()
        errors = list()

        # Check that all input files exist
        for key in self.get_option_keys():
            value = getattr(self, key)
            if value is not None:
                if os.path.isfile(value) is False:
                    errors.append('Config value Input->{}: {} not found'.format(key, value))
                #TODO: check that files are of valid type

        #TODO: check that the right combination of input files exists

        return errors
