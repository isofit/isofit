from isofit.surface.surface import Surface as BaseSurface
from isofit.surface.surface_additive_glint import AdditiveGlintSurface
from isofit.surface.surface_glint_model import GlintModelSurface
from isofit.surface.surface_lut import LUTSurface
from isofit.surface.surface_multicomp import MultiComponentSurface
from isofit.surface.surface_thermal import ThermalSurface


def Surface(config):
    """
    Reads an ISOFIT full config and initializes the desired Surface model

    Parameters
    ----------
    config : isofit.configs.Config
        The full_config to determine the surface category from and to pass along to the
        Surface model's initialization

    Returns
    -------
    Surface Model
    """
    surface_params = full_config.forward_model.surface.surface_params

    # Check if multi-surface config else use single surface config
    if config.forward_model.surface.multi_surface_flag:
        category = fm_config.surface.Surfaces[surface_class_str]["surface_category"]
        surface_file = config.forward_model.surface.Surfaces[surface_class_str][
            "surface_file"
        ]
    else:
        category = config.forward_model.surface.surface_category
        surface_file = config.forward_model.surface.surface_file

    # Handle error if there is no surface file
    if not surface_file:
        raise FileNotFoundError("No surface .mat file exists")
    category = config.forward_model.surface.surface_category

    if category == "surface":
        return BaseSurface(surface_file, surface_params)

    elif category == "multicomponent_surface":
        return MultiComponentSurface(surface_file, surface_params)

    elif category == "additive_glint_surface":
        return AdditiveGlintSurface(surface_file, surface_params)

    elif category == "glint_model_surface":
        return GlintModelSurface(surface_file, surface_params)

    elif category == "thermal_surface":
        return ThermalSurface(surface_file, surface_params)

    elif category == "lut_surface":
        return LUTSurface(surface_file, surface_params)

    else:
        # No need to be more specific - should have been checked in config already
        raise ValueError("Must specify a valid surface model")
