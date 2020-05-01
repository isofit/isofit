import re
from collections import OrderedDict
from typing import Dict, List, Type
import numpy as np


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
        for key in self._get_nontype_attributes():
            keytype = getattr(self, '_' + key + '_type')
            if key in configdict:
                if callable(keytype):
                    sub_config = keytype(configdict[key])
                    setattr(self, key, sub_config)
                else:
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
            errors.append(message_type.format(key, self.__class__.__name__,
                                              value, type(value), type_expected))

        # Now do a full check on each submodule
        errors.extend(self._check_config_validity())

        return errors

    def _get_callable_errors(object: object, configdict: dict) -> None:
        for config_section_name in object.__dict__.keys():
            camelcase_section_name = snake_to_camel(config_section_name)

            subdict = None
            if camelcase_section_name in configdict.keys:
                subdict = configdict[camelcase_section_name]

            setattr(object, config_section_name, getattr(
                'isofit.configs.sections', camelcase_section_name)(subdict))

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
            if key[0] == '_':
                continue
            keys.append(key)
        return keys

    def _get_type_attributes(self) -> List[str]:
        keys = []
        for key in self.__dict__.keys():
            if key[0] == '_' and key[-5:] == '_type':
                keys.append(key)
        return keys

    def _get_hidden_attributes(self) -> List[str]:
        keys = []
        for key in self.__dict__.keys():
            if key[0] == '_' and key[-5:] != '_type':
                keys.append(key)
        return keys

    def get_all_elements(self):
        return [getattr(self, x) for x in self._get_nontype_attributes()]

    def get_all_element_names(self):
        return self._get_nontype_attributes()

    def get_elements(self):
        elements = self.get_all_elements()
        element_names = self._get_nontype_attributes()
        valid = [x is not None for x in elements]
        elements = [elements[x] for x in range(len(elements)) if valid[x]]
        element_names = [element_names[x] for x in range(len(elements)) if valid[x]]

        order = np.argsort(element_names)
        elements = [elements[idx] for idx in order]
        element_names = [element_names[idx] for idx in order]

        return elements, element_names

    def get_element_names(self):
        elements, element_names = self.get_elements()
        return element_names

    def get_single_element_by_name(self, name):
        elements, element_names = self.get_elements()
        return elements[element_names.index(name)]


def snake_to_camel(word: str) -> None:
    """ Function to convert snake case to camel case, e.g.
    snake_to_camel -> SnakeToCamel
    Args:
        word: snake_case string
    Returns:
        CamelCase string
    """
    return ''.join(x.capitalize() or '_' for x in word.split('_'))
