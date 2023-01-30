#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import logging
from collections import OrderedDict
from typing import Dict, List, Type

import numpy as np


class BaseConfigSection(object):
    """
    Base Configuration Section from which all Configuration Sections inherit. Handles shared functionality like getting,
    setting, and cleaning configuration options.
    """

    def __init__(self) -> None:
        return

    def set_config_options(self, configdict: dict = None) -> None:
        """Read dictionary and assign to attributes, leaning on _set_callable_attributes
        Args:
            configdict: dictionary-style config for parsing
        """
        for key in self._get_nontype_attributes():
            keytype = getattr(self, "_" + key + "_type")
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
            "Invalid type for config option {} in config section {}. The provided value"
            " {} is a {}, " + "but the required value should be a {}."
        )

        # First check typing
        for key in self._get_nontype_attributes():
            # get the actual parameter value
            value = getattr(self, key)

            # None assignments are fine
            if value is None:
                continue

            # check it against expected
            type_expected = self._get_expected_type_for_option_key(key)
            # Lists are complicated, retype
            if isinstance(type_expected, List):
                type_expected = List

            if isinstance(value, type_expected):
                continue

            # At this point, we have a type mismatch, add to error list
            errors.append(
                message_type.format(
                    key, self.__class__.__name__, value, type(value), type_expected
                )
            )

        errors.extend(self._check_config_validity())

        # Now do a full check on each submodule
        for key in self._get_nontype_attributes():
            value = getattr(self, key)
            try:
                logging.debug("Configuration check of: {}".format(key))
                errors.extend(value.check_config_validity())
            except AttributeError:
                logging.debug(
                    "Configuration check: {} is not an object, skipping".format(key)
                )

        return errors

    def get_config_options_as_dict(self) -> Dict[str, Dict[str, any]]:
        config_options = OrderedDict()
        for key in self._get_nontype_attributes():
            value = getattr(self, key)
            if type(value) is tuple:
                value = list(
                    value
                )  # Lists look nicer in config files and seem friendlier
            config_options[key] = value
        return config_options

    def _check_config_validity(self) -> List[str]:
        return list()

    def _get_expected_type_for_option_key(self, option_key: str) -> type:
        return getattr(self, "_{}_type".format(option_key))

    def _get_nontype_attributes(self) -> List[str]:
        keys = []
        for key in self.__dict__.keys():
            if key[0] == "_":
                continue
            keys.append(key)
        return keys

    def _get_type_attributes(self) -> List[str]:
        keys = []
        for key in self.__dict__.keys():
            if key[0] == "_" and key[-5:] == "_type":
                keys.append(key)
        return keys

    def _get_hidden_attributes(self) -> List[str]:
        keys = []
        for key in self.__dict__.keys():
            if key[0] == "_" and key[-5:] != "_type":
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
        element_names = [
            element_names[x] for x in range(len(element_names)) if valid[x]
        ]

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
