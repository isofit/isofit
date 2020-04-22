import copy
import logging
import os
import re
from collections import OrderedDict
from typing import Dict, List
from isofit.configs import sections


class Config(object):
    """
    Handles the reading and formatting of configuration files.  Please note - there are many ways to do this, some
    of which require fewer lines of code.  This method was chosen to facilitate more clarity when using / adding /
    modifying code, particularly given the highly flexible nature of Isofit.
    """

    def __init__(self) -> None:
        self.input = None
        self.output = None
        self.forward_model = None
        self.inversion = None

    def get_config_as_dict(self) -> dict:
        """Get configuration options as a nested dictionary with delineated sections.

        Returns:
            Configuration options as a nested dictionary with delineated sections.
        """
        config = OrderedDict()
        for config_section in []:#sections.get_config_sections():
            section_name = config_section.get_config_name_as_snake_case()
            populated_section = getattr(self, section_name)
            config[section_name] = populated_section.get_config_options_as_dict()
            #if config_section is sections.ModelTraining:
            #    # Given ordered output, architecture options make the most sense after model training options
            #    config["architecture"] = self.architecture.get_config_options_as_dict()
        return config

    def get_config_errors(self, include_sections: List[str] = None, exclude_sections: List[str] = None) -> list:
        """Get configuration option errors by checking the validity of each config section.

        Args:
            include_sections: Config sections that should be included. All config sections are included if None and
              exclude_sections is not specified. Cannot specify both include_sections and exclude_sections.
            exclude_sections: Config sections that should be excluded. All config sections are included if None and
              exclude_sections is not specified. Cannot specify both include_sections and exclude_sections.

        Returns:
            List of errors associated with the current configuration.
        """
        assert not (
                include_sections and exclude_sections
        ), "Both include_sections and exclude_sections cannot be specified."
        logging.debug("Checking config sections for configuration issues")
        errors = list()
        config_sections = []#sections.get_config_sections()
        if include_sections:
            logging.debug("Only checking config sections: {}".format(", ".join(include_sections)))
            config_sections = [
                section for section in config_sections if section.get_config_name_as_snake_case() in include_sections
            ]
        if exclude_sections:
            logging.debug("Not checking config sections: {}".format(", ".join(exclude_sections)))
            config_sections = [
                section
                for section in config_sections
                if section.get_config_name_as_snake_case() not in exclude_sections
            ]
        for config_section in config_sections:
            section_name = config_section.get_config_name_as_snake_case()
            populated_section = getattr(self, section_name)
            errors.extend(populated_section.check_config_validity())
            if config_section is []:#sections.ModelTraining:
                errors.extend(self.architecture.check_config_validity())
        logging.debug("{} configuration issues found".format(len(errors)))
        return errors

    def get_human_readable_config_errors(
            self, include_sections: List[str] = None, exclude_sections: List[str] = None
    ) -> str:
        """Generates a human-readable string of configuration option errors.

        Args:
            include_sections: Config sections that should be included. All config sections are included if None and
              exclude_sections is not specified. Cannot specify both include_sections and exclude_sections.
            exclude_sections: Config sections that should be excluded. All config sections are included if None and
              exclude_sections is not specified. Cannot specify both include_sections and exclude_sections.

        Returns:
            Human-readable string of configuration option errors.
        """
        errors = self.get_config_errors(include_sections=include_sections, exclude_sections=exclude_sections)
        if not errors:
            return ""
        return "List of configuration section and option errors is as follows:\n" + "\n".join(error for error in errors)



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

    def get_option_keys(self) -> List[str]:
        return [
            key
            for key in self.__class__.__dict__.keys()
            if not key.startswith("_") and not callable(getattr(self, key))
        ]

    def set_config_options(self, config_options: dict, highlight_required: bool) -> None:
        logging.debug("Setting config options for section {} from {}".format(self.__class__.__name__, config_options))
        for key in self.get_option_keys():
            if key in config_options:
                value = config_options.pop(key)
                logging.debug('Setting option "{}" to provided value "{}"'.format(key, value))
            #TODO: verify non-use
            #else:
            #    value = getattr(self, key)
            #    # We leave the labels for required and optional values as-is for templates, i.e., when highlights are
            #    # required, otherwise we convert those labels to None
            #    if not highlight_required and value in (DEFAULT_REQUIRED_VALUE, DEFAULT_OPTIONAL_VALUE):
            #        value = None
            #    logging.debug('Setting option "{}" to default value "{}"'.format(key, value))
            #setattr(self, key, self._clean_config_option_value(key, value))
        return

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

    def check_config_validity(self) -> List[str]:
        errors = list()
        message_type = (
                "Invalid type for config option {} in config section {}. The provided value {} is a {}, "
                + "but the required value should be a {}."
        )

        # First check typing
        for key in self.get_option_keys():
            value = getattr(self, key)
            # No further checking is necessary if the provided type matches to expected type
            type_expected = self._get_expected_type_for_option_key(key)
            if type(value) is type_expected:
                continue

            # None's are okay too at this point
            if value is None:
                continue

            # At this point, we have a type mismatch
            errors.append(message_type.format(key, self.__class__.__name__, value, type(value), type_expected))

        # Now check section-specific requirements
        errors.extend(self._check_config_validity())
        return errors

    def _check_config_validity(self) -> List[str]:
        return list()

    def _get_expected_type_for_option_key(self, option_key: str) -> type:
        return getattr(self, "_{}_type".format(option_key))


def create_config_from_file(filepath: str) -> Config:
    """Creates a Config object from a YAML file.

    Args:
        filepath: Filepath to existing YAML file.

    Returns:
        Config object with parsed YAML file attributes.
    """
    assert os.path.exists(filepath), "No config file found at {}".format(filepath)
    logging.debug("Loading config file from {}".format(filepath))
    with open(filepath) as file_:
        raw_config = yaml.safe_load(file_)
    return _create_config(raw_config, is_template=False)


def create_config_template(architecture_name: str, filepath: str = None) -> Config:
    """Creates a template version of a Config for a given architecture, with required and optional parameters
    highlighted, and default values for other parameters. Config is returned but can optionally be written to YAML file.

    Args:
        architecture_name: Name of available architecture.
        filepath: Filepath to which template YAML file is saved, if desired.

    Returns:
        Template version of a Config.
    """
    logging.debug("Creating config template for architecture {} at {}".format(architecture_name, filepath))
    config_options = {"model_training": {"architecture_name": architecture_name}}
    config = _create_config(config_options, is_template=True)
    if filepath is not None:
        save_config_to_file(config, filepath)
    return config


def _create_config(config_options: dict, is_template: bool) -> Config:
    config_copy = copy.deepcopy(config_options)  # Use a copy because config options are popped from the dict
    # Populate config sections with the provided configuration options, tracking errors
    populated_sections = dict()
    for config_section in []:#sections.get_config_sections():
        section_name = config_section.get_config_name_as_snake_case()
        populated_section = config_section()
        populated_section.set_config_options(config_copy.get(section_name, dict()), is_template)
        populated_sections[section_name] = populated_section
    # Populate architecture options given architecture name
    architecture_name = populated_sections["model_training"].architecture_name
    architecture = []#config_sections.get_architecture_config_section(architecture_name)
    architecture.set_config_options(config_copy.get("architecture", dict()), is_template)
    populated_sections["architecture"] = architecture
    return Config(**populated_sections)


def save_config_to_file(config: Config, filepath: str, include_sections: List[str] = None) -> None:
    """Saves/serializes a Config object to a YAML file.

    Args:
        config: Config object.
        filepath: Filepath to which YAML file is saved.
        include_sections: Config sections that should be included. All config sections are included if None.

    Returns:
        None
    """

    def _represent_dictionary_order(self, dict_data):
        # via https://stackoverflow.com/questions/31605131/dumping-a-dictionary-to-a-yaml-file-while-preserving-order
        return self.represent_mapping("tag:yaml.org,2002:map", dict_data.items())

    def _represent_list_inline(self, list_data):
        return self.represent_sequence("tag:yaml.org,2002:seq", list_data, flow_style=True)

    yaml.add_representer(OrderedDict, _represent_dictionary_order)
    yaml.add_representer(list, _represent_list_inline)
    config_out = config.get_config_as_dict()
    logging.debug("Saving config file to {}".format(filepath))
    if include_sections:
        logging.debug("Only saving config sections: {}".format(", ".join(include_sections)))
        config_out = {section: config_out[section] for section in include_sections}
    with open(filepath, "w") as file_:
        yaml.dump(config_out, file_, default_flow_style=False)


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

