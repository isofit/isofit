from .surface import Surface as BaseSurface
from .surface_additive_glint import AdditiveGlintSurface
from .surface_glint_model import GlintModelSurface
from .surface_lut import LUTSurface
from .surface_multicomp import MultiComponentSurface
from .surface_thermal import ThermalSurface


def Surface(config):
    category = config.forward_model.surface.surface_category

    if category == "surface":
        surface = BaseSurface(full_config)

    elif category == "multicomponent_surface":
        surface = MultiComponentSurface(full_config)

    elif category == "additive_glint_surface":
        surface = AdditiveGlintSurface(full_config)

    elif category == "glint_model_surface":
        surface = GlintModelSurface(full_config)

    elif category == "thermal_surface":
        surface = ThermalSurface(full_config)

    elif category == "lut_surface":
        surface = LUTSurface(full_config)

    else:
        # No need to be more specific - should have been checked in config already
        raise ValueError("Must specify a valid surface model")
