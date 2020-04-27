import logging
import re
from collections import OrderedDict
from typing import Dict, List, Type
import json
import os
import yaml
from isofit.configs.sections.input_config import InputConfig
from isofit.configs.sections.output_config import OutputConfig
from isofit.configs.sections.forward_model_config import ForwardModelConfig
from isofit.configs.base_config import BaseConfigSection
from isofit.core import common


class Config(BaseConfigSection):
    """
    Handles the reading and formatting of configuration files.  Please note - there are many ways to do this, some
    of which require fewer lines of code.  This method was chosen to facilitate more clarity when using / adding /
    modifying code, particularly given the highly flexible nature of Isofit.
    """

    def __init__(self, configdict) -> None:

        self._input_type = InputConfig
        self.input = None
        """InputConfig: Input config. Holds all input file information."""

        self._output_type = OutputConfig
        self.output = None
        """OutputConfig: Output config. Holds all output file information."""

        self._forward_model_type = ForwardModelConfig
        self.forward_model = None
        """ForwardModelConfig: forward_model config. Holds information about surface models, 
        radiative transfer models, and the instrument."""

        self._implementation_type = ImplementationConfig
        self.implementation = None
        """ImplementationConfig: holds information regarding how isofit is to be run, including relevant sub-configs 
        (e.g. inversion information)."""

        # Load sub-classes and attributes
        self.set_config_options(configdict)

        # check for errors
        self.get_config_errors()



    def get_config_as_dict(self) -> dict:
        """Get configuration options as a nested dictionary with delineated sections.

        Returns:
            Configuration options as a nested dictionary with delineated sections.
        """
        config = OrderedDict()
        for config_section in self._get_nontype_attributes():
            populated_section = getattr(self, config_section)
            config[config_section] = populated_section.get_config_options_as_dict()
        return config


    def get_config_errors(self, include_sections: List[str] = None, exclude_sections: List[str] = None):
        """Get configuration option errors by checking the validity of each config section.
        Args:
            include_sections: Config sections that should be included. All config sections are included if None and
              exclude_sections is not specified. Cannot specify both include_sections and exclude_sections.
            exclude_sections: Config sections that should be excluded. All config sections are included if None and
              exclude_sections is not specified. Cannot specify both include_sections and exclude_sections.
        """
        if include_sections and exclude_sections:
            raise AttributeError("Both include_sections and exclude_sections cannot be specified.")

        logging.info("Checking config sections for configuration issues")

        errors = []
        for key in self._get_nontype_attributes():
            value = getattr(self, key)
            try:
                errors.extend(value.check_config_validity())
            except AttributeError:
                logging.debug('Configuration check: {} is not an object, skipping'.format(key))

        ##TODO: do same thing here as with global hidden function used within BaseConfigSection, so that this is
        ## recursive
        #config_sections = get_config_sections()
        #if include_sections:
        #    logging.info("Only checking config sections: {}".format(", ".join(include_sections)))
        #    config_sections = [
        #        section for section in config_sections if section.get_config_name_as_snake_case() in include_sections
        #    ]
        #if exclude_sections:
        #    logging.info("Not checking config sections: {}".format(", ".join(exclude_sections)))
        #    config_sections = [
        #        section
        #        for section in config_sections
        #        if section.get_config_name_as_snake_case() not in exclude_sections
        #    ]
        #for config_section in config_sections:
        #    section_name = config_section.get_config_name_as_snake_case()
        #    populated_section = getattr(self, section_name)
        #    errors.extend(populated_section.check_config_validity())

        #logging.info("{} configuration issues found".format(len(errors)))

        for e in errors:
            logging.error(e)

        if len(errors) > 0:
            raise AttributeError('Configuration error(s) found.  See log for details.')

        logging.info('Configuration file checks complete, no errors found.')




def get_config_differences(config_a: Config, config_b: Config) -> Dict:
    differing_items = dict()
    dict_a = config_a.get_config_as_dict()
    dict_b = config_b.get_config_as_dict()
    all_sections = set(list(dict_a.keys()) + list(dict_b.keys()))
    for section in all_sections:
        section_a = dict_a.get(section, dict())
        section_b = dict_b.get(section, dict())
        all_options = set(list(section_a.keys()) + list(section_b.keys()))
        for option in all_options:
            value_a = section_a.get(option, None)
            value_b = section_b.get(option, None)
            if value_a != value_b:
                logging.debug(
                    "Configs have different values for option {} in section {}:  {} and {}".format(
                        option, section, value_a, value_b
                    )
                )
                differing_items.setdefault(section, dict())[option] = (value_a, value_b)
    return differing_items


def create_new_config(config_file: str) -> Config:
    """Load a config file from disk.
    Args:
        config_file: file to load config from.  Currently accepted formats: JSON

    Returns:
        Config object, having completed all necessary config checks
    """
    #TODO: facilitate YAML read as well
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    configdir, f = os.path.split(os.path.abspath(config_file))

    config_dict = common.expand_all_paths(config_dict, configdir)
    config = Config(config_dict)

    return config


