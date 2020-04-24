import copy
import logging
import os
import re
from collections import OrderedDict
from importlib import import_module
from typing import Dict, List, Type
import isofit.core.common
import json
import yaml


class Config(object):
    """
    Handles the reading and formatting of configuration files.  Please note - there are many ways to do this, some
    of which require fewer lines of code.  This method was chosen to facilitate more clarity when using / adding /
    modifying code, particularly given the highly flexible nature of Isofit.
    """

    def __init__(self, configdict: dict = None) -> None:
        self.input = None
        self.output = None
        self.test = None
        self.forward_model = None
        #self.inversion = None

        # Load sub-classes and attributes
        _set_callable_attributes(self, configdict)

        # check for errors
        self.get_config_errors()



    def get_config_as_dict(self) -> dict:
        """Get configuration options as a nested dictionary with delineated sections.

        Returns:
            Configuration options as a nested dictionary with delineated sections.
        """
        config = OrderedDict()
        for config_section in get_config_sections():
            section_name = config_section.get_config_name_as_snake_case()
            populated_section = getattr(self, section_name)
            config[section_name] = populated_section.get_config_options_as_dict()
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
        for key in self.__dict__.keys():
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


class BaseConfigSection(object):
    """
    Base Configuration Section from which all Configuration Sections inherit. Handles shared functionality like getting,
    setting, and cleaning configuration options.


    How to use sections:  Configs and ConfigSections are the tools we use to handle the numerous parameters
    associated with atmospheric correction. In general, we want to validate that ConfigSections have options with
    the expected values and relationships, and we use two important methods to do this without duplicating too much code.

    To validate that attributes have the correct type, ensure that the attributes are on the config section and have an
    associated hidden attribute with a particular name pattern. Specifically, given an attribute named 'attribute', the
    hidden attribute should be named '_attribute_type' and its value should be the type expected for that attribute.
    Methods on the BaseConfigSection will ensure that this attribute type is checked and errors will be raised to the user
    if it's not appropriate. Example:

    ```
    class GenericConfigSection(BaseConfigSection):
        _attribute_type = list                  <-- Used to validate attributes have correct types
        attribute = DEFAULT_REQUIRED_VALUE
    ```

    To validate that attributes have appropriate relationships or characteristics, use the hidden _check_config_validity
    method to add more detailed validation checks. Simply return a list of string descriptions of errors from the
    method as demonstrated:

    ```
    def _check_config_validity(self) -> List[str]:
        errors = list()
        if self.attribute_min >= self.attribute_max:
            errors.append('attribute_min must be less than attribute_max.')
        return errors
    ```
    """

    def __init__(self) -> None:
        return

    def set_config_options(self, configdict: dict = None) -> None:
        """ Read dictionary and assign to attributes, leaning on _set_callable_attributes
        Args:
            configdict: dictionary-style config for parsing
        """

        #TODO: grab callable based on type
        for key in self.__dict__:
            if callable(key):
                if configdict is None:
                    _set_callable_attributes(self, None)
                elif key in configdict:
                    _set_callable_attributes(self, configdict[key])
            else:
                if configdict is not None:
                    if key in configdict:
                        setattr(self, key, configdict[key])
        return


    def check_config_validity(self) -> List[str]:
        errors = list()
        message_type = (
                "Invalid type for config option {} in config section {}. The provided value {} is a {}, "
                + "but the required value should be a {}."
        )

        # First check typing
        for key in self._get_nontype_attributes():

            # get the actual parameter value
            value = getattr(self, key)

            # check it against expected
            type_expected = self._get_expected_type_for_option_key(key)
            if type(value) is type_expected:
                continue

            # None's are okay too (unassigned)
            if value is None:
                continue

            # At this point, we have a type mismatch, add to error list
            errors.append(message_type.format(key, self.__class__.__name__, value, type(value), type_expected))

        # Now do a full check on each submodule
        errors.extend(self._check_config_validity())

        return errors

    def _get_callable_errors(object: object, configdict: dict) -> None:
        for config_section_name in object.__dict__.keys():
            camelcase_section_name = snake_to_camel(config_section_name)

            subdict = None
            if camelcase_section_name in configdict.keys:
                subdict = configdict[camelcase_section_name]

            setattr(object, config_section_name, getattr('isofit.configs.sections', camelcase_section_name)(subdict))


    @classmethod
    def get_config_name_as_snake_case(cls) -> str:
        snake_case_converter = re.compile("((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))")
        return snake_case_converter.sub(r"_\1", cls.__name__).lower()

    def get_config_options_as_dict(self) -> Dict[str, Dict[str, any]]:
        config_options = OrderedDict()
        for key in self.get_option_keys():
            value = getattr(self, key)
            if type(value) is tuple:
                value = list(value)  # Lists look nicer in config files and seem friendlier
            config_options[key] = value
        return config_options


    def _clean_config_option_value(self, option_key: str, value: any) -> any:
        # None read as string so we need to convert to the None type
        if value in ("None", "none"):
            value = None

        # Some parameters are treated as floats, but ints are acceptable input formats
        # Treating ints as floats is more flexible and requires fewer assumptions / accomodations in the code
        # Example:  users are likely to provide -9999 instead of -9999.0
        type_expected = self._get_expected_type_for_option_key(option_key)
        if type(value) is int and type_expected is float:
            value = float(value)

        return value


    def _check_config_validity(self) -> List[str]:
        return list()

    def _get_expected_type_for_option_key(self, option_key: str) -> type:
        return getattr(self, "_{}_type".format(option_key))

    def _get_nontype_attributes(self) -> List[str]:
        keys = []
        for key in self.__dict__.keys():
            if key[0] == '_' and key[-5:] == '_type':
                continue
            keys.append(key)
        return keys


def create_config_from_file(config_file: str) -> Config:
    """Creates a Config object from a JSON or YAML file.
    Args:
        config_file: Filepath to existing JSON or YAML file.

    Returns:
        Config object with parsed attributes.
    """
    if os.path.isfile(config_file) is False:
        raise FileNotFoundError('No config file found at {}'.format(config_file))

    logging.debug("Loading config file from {}".format(config_file))

    if os.path.splitext(config_file)[-1].lower() == 'json':
        with open(config_file, 'r') as f:
            rawconfig = json.load(f)
    elif os.path.splitext(config_file)[-1].lower() == 'yaml':
        with open(config_file, 'r') as f:
            rawconfig = yaml.safe_load(f)
    else:
        raise ImportError('File of known type - require either json or yaml')

    configdir, f = os.path.split(os.path.abspath(config_file))
    configdict = isofit.core.common.expand_all_paths(rawconfig, configdir)

    config = Config(configdict)
    return config


#def save_config_to_file(config: Config, filepath: str, include_sections: List[str] = None) -> None:
#    """Saves/serializes a Config object to a YAML file.
#
#    Args:
#        config: Config object.
#        filepath: Filepath to which YAML file is saved.
#        include_sections: Config sections that should be included. All config sections are included if None.
#
#    Returns:
#        None
#    """
#
#    def _represent_dictionary_order(self, dict_data):
#        # via https://stackoverflow.com/questions/31605131/dumping-a-dictionary-to-a-yaml-file-while-preserving-order
#        return self.represent_mapping("tag:yaml.org,2002:map", dict_data.items())
#
#    def _represent_list_inline(self, list_data):
#        return self.represent_sequence("tag:yaml.org,2002:seq", list_data, flow_style=True)
#
#    yaml.add_representer(OrderedDict, _represent_dictionary_order)
#    yaml.add_representer(list, _represent_list_inline)
#    config_out = config.get_config_as_dict()
#    logging.debug("Saving config file to {}".format(filepath))
#    if include_sections:
#        logging.debug("Only saving config sections: {}".format(", ".join(include_sections)))
#        config_out = {section: config_out[section] for section in include_sections}
#    with open(filepath, "w") as file_:
#        yaml.dump(config_out, file_, default_flow_style=False)

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


def get_config_sections() -> List[Type[BaseConfigSection]]:
    return [
        sections.input_config.Input,
        sections.output_config.Output,
    ]

def snake_to_camel(word: str) -> None:
    """ Function to convert snake case to camel case, e.g.
    snake_to_camel -> SnakeToCamel
    Args:
        word: snake_case string
    Returns:
        CamelCase string
    """
    return ''.join(x.capitalize() or '_' for x in word.split('_'))


def _set_callable_attributes(object: object, configdict: dict) -> None:
    """ Function to read a dictionary, determine if any of it's elements are config sections defined in
    isofit.configs.sections. and if so, initialize an object and populate it's subdirectory.  Meant to be called
    recursively.  Defined here for use in both Config and BaseConfigSection
    Args:
        object: Object to check for the existence of dictionary keys in
        configdict: dictionary-style config for parsing
    """

    for config_section_name in object.__dict__.keys():
        camelcase_section_name = snake_to_camel(config_section_name)

        subdict = None
        if config_section_name in configdict.keys():
            subdict = configdict[config_section_name]

        try:
            sub_config = getattr(import_module('isofit.configs.sections'), camelcase_section_name + 'Config')(subdict)
            setattr(object, config_section_name, sub_config)
        except AttributeError:
            logging.debug('Cannot set sub-attrubutes for: {}, skipping'.format(config_section_name))








