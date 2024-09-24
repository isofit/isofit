from .surface import Surface as BaseSurface
from .surface_additive_glint import AdditiveGlintSurface
from .surface_glint_model import GlintModelSurface
from .surface_lut import LUTSurface
from .surface_multicomp import MultiComponentSurface
from .surface_thermal import ThermalSurface


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
    category = config.forward_model.surface.surface_category

    if category == "surface":
        return BaseSurface(config)

    elif category == "multicomponent_surface":
        return MultiComponentSurface(config)

    elif category == "additive_glint_surface":
        return AdditiveGlintSurface(config)

    elif category == "glint_model_surface":
        return GlintModelSurface(config)

    elif category == "thermal_surface":
        return ThermalSurface(config)

    elif category == "lut_surface":
        return LUTSurface(config)

    else:
        # No need to be more specific - should have been checked in config already
        raise ValueError("Must specify a valid surface model")
