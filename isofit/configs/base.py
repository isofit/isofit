"""
Shared base model for config sections.

The section models carry no Python-side defaults; those live in the base
configs and mixins under ``isofit/configs/templates/``. :class:`SectionModel`
lets a section be constructed on its own by seeding it from one of those base
configs via the ``base`` keyword, with mixins layered on via ``mixin``, e.g.
``ImplementationConfig(base="vswir", n_cores=4)`` or
``AtmosphereConfig(base="vswir", mixin="modtran")``.
"""

from typing import Any, ClassVar, Optional

import yaml
from pydantic import BaseModel, model_validator


class SectionModel(BaseModel):
    """
    Base model for config sections that can be seeded from a base config.

    Passing ``base=<name>`` loads the base config ``name`` (see
    :func:`isofit.configs.loader.load`) with the :attr:`mixins` and any given
    by ``mixin=<name or list>`` layered on top, deep-merges the remaining
    keyword arguments onto this model's subsection of it, interpolates
    ``${...}`` references against the full config, and validates the result.
    ``base`` and ``mixin`` themselves are not stored on the model.

    Subclasses opt in by setting :attr:`section` to their dotted path within
    the full config. Without ``base``, validation is unchanged.

    Examples
    --------
    >>> from isofit.configs.sections.implementation import ImplementationConfig
    >>> impl = ImplementationConfig(base="vswir", n_cores=4)
    >>> impl.n_cores
    4
    >>> impl.inversion.windows
    [[380.0, 1325.0], [1435.0, 1770.0], [1965.0, 2500.0]]
    >>> impl = ImplementationConfig(base="vswir", mixin="emit")
    >>> impl.inversion.windows
    [[350.0, 1325.0], [1435.0, 1800.0], [1965.0, 2500.0]]
    """

    section: ClassVar[Optional[str]] = None
    """Dotted path of this section within the full config (``""`` for the root
    :class:`~isofit.configs.Config`). ``None`` disables ``base`` support."""

    defaults: ClassVar[str] = "default"
    """Base config used when ``base`` is not given."""

    mixins: ClassVar[tuple[str, ...]] = ()
    """Mixins always layered on top of the base, before any given with
    ``mixin``. Engine sections set their engine mixin (e.g. ``("sixs",)``),
    since the bases each select their own engine."""

    @model_validator(mode="before")
    @classmethod
    def _apply_base(cls, data: Any) -> Any:
        """
        Seed the input from the base config named by ``data["base"]``, with
        the mixins named by ``data["mixin"]`` layered on top.
        """
        if cls.section is None or not isinstance(data, dict):
            return data

        # Deferred: the loader imports Config, which imports every section
        from isofit.configs.loader import load

        sect = data.pop("base", cls.defaults)
        mixin = data.pop("mixin", None) or []
        if isinstance(mixin, str):
            mixin = [mixin]
        mixin = [*cls.mixins, *mixin]

        return load(sect, section=cls.section, overrides=data, mixin=mixin, interp=True)

    def to_yaml(self):
        return yaml.dump(self.model_dump(mode="json"), sort_keys=False)
