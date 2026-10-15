"""
Atmosphere state vector and unknowns configuration models.

Unlike the surface and instrument state vectors, whose elements are a fixed set
of named fields, the atmospheric state vector is populated dynamically from the
LUT grid (``H2OSTR``, ``CO2``, ``AOT550``, ``surface_elevation_km``,
``AERFRAC_*``, ...). These models are therefore modeled as typed mappings
(:class:`pydantic.RootModel` over a ``dict``), which validates each entry
automatically, while still exposing the historical
``get_elements``/``get_element_names`` accessor API the runtime relies on.
"""

from pydantic import RootModel

from isofit.configs.sections.statevector import StateVectorElementConfig
from isofit.configs.utils.accessors import StateVectorValueMixin


class AtmosphereStateVectorConfig(
    StateVectorValueMixin,
    RootModel[dict[str, StateVectorElementConfig]],
):
    """
    Atmospheric state vector with dynamic, LUT-driven element names.

    Elements are supplied as a mapping of arbitrary keys (one per retrieved
    atmospheric parameter) to :class:`StateVectorElementConfig` values. Pydantic
    validates (and coerces plain dicts into) the element configs automatically.
    The accessor methods iterate over these dynamic entries rather than a fixed
    set of declared fields.

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere.statevector import (
    ...     AtmosphereStateVectorConfig,
    ... )
    >>> sv = AtmosphereStateVectorConfig(
    ...     {"H2OSTR": {"bounds": [0.5, 2.5], "init": 1.5}}
    ... )
    >>> sv.get_element_names()
    ['H2OSTR']
    """

    root: dict[str, StateVectorElementConfig] = {}

    def get_elements(self):
        """
        Return ``(elements, names)`` for all dynamic elements, sorted by name.
        """
        pairs = sorted(self.root.items())
        names = [name for name, _ in pairs]
        elements = [element for _, element in pairs]
        return elements, names

    def get_element_names(self):
        """Return the names of all dynamic elements, sorted."""
        return self.get_elements()[1]

    def get_all_element_names(self):
        """Return the names of every dynamic element, in insertion order."""
        return list(self.root)

    def get_all_elements(self):
        """Return the value of every dynamic element, in insertion order."""
        return list(self.root.values())

    def get(self, name):
        """Return the element with the given name, or ``[]`` if not set."""
        return self.root.get(name, [])


class AtmosphereUnknownsConfig(RootModel[dict[str, float]]):
    """
    Radiative-transfer unknowns (unmodeled-variable uncertainties).

    Each entry maps an unknown parameter name to a scalar uncertainty value.
    Mirrors the old ``RadiativeTransferUnknownsConfig`` accessor surface used by
    the atmosphere runtime (``get_element_names`` / ``get_elements``).

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere.statevector import (
    ...     AtmosphereUnknownsConfig,
    ... )
    >>> u = AtmosphereUnknownsConfig({"H2O_ABSCO": 0.0})
    >>> u.get_elements()
    ([0.0], ['H2O_ABSCO'])
    """

    root: dict[str, float] = {}

    def get_elements(self):
        """Return ``(values, names)`` for all set unknowns, sorted by name."""
        pairs = sorted(self.root.items())
        names = [name for name, _ in pairs]
        values = [value for _, value in pairs]
        return values, names

    def get_element_names(self):
        """Return the names of all set unknowns, sorted."""
        return self.get_elements()[1]
