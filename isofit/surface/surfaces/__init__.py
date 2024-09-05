from .surface_additive_glint import AdditiveGlintSurface
from .surface_glint_model import GlintModelSurface
from .surface_lut import LUTSurface
from .surface_multicomp import MultiComponentSurface
from .surface_test import TestSurface
from .surface_thermal import ThermalSurface

Surfaces = {
    "multicomponent_surface": MultiComponentSurface,
    "additive_glint_surface": AdditiveGlintSurface,
    "glint_model_surface": GlintModelSurface,
    "thermal_surface": ThermalSurface,
    "lut_surface": LUTSurface,
    "test_surface": TestSurface,
}
