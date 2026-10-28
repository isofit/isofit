"""
Shared base model for config sections.

The section models carry no Python-side defaults; those live in the base
configs under ``isofit/configs/templates/``. :class:`SectionModel` lets a
section be constructed on its own by seeding it from one of those base configs
via the ``base`` keyword, e.g. ``ImplementationConfig(base="vswir", n_cores=4)``.
"""

from typing import Any, ClassVar, Optional

import yaml
from pydantic import BaseModel, model_validator


class SectionModel(BaseModel):
    """
    Base model for config sections that can be seeded from a base config.

    Passing ``base=<name>`` loads the base config ``name`` (see
    :func:`isofit.configs.loader.load`), deep-merges the remaining
    keyword arguments onto this model's subsection of it, interpolates
    ``${...}`` references against the full config, and validates the result.
    ``base`` itself is not stored on the model.

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
    """

    section: ClassVar[Optional[str]] = None
    """Dotted path of this section within the full config (``""`` for the root
    :class:`~isofit.configs.Config`). ``None`` disables ``base`` support."""

    defaults: ClassVar[str] = "default"
    """Base config used when ``base`` is not given. Engine sections override
    this with their own base (e.g. ``"sixs"``), since ``default`` only defines
    the prebuilt engine."""

    @model_validator(mode="before")
    @classmethod
    def _apply_base(cls, data: Any) -> Any:
        """
        Seed the input from the base config named by ``data["base"]``.
        """
        if cls.section is None or not isinstance(data, dict):
            return data

        # Deferred: the loader imports Config, which imports every section
        from isofit.configs.loader import load

        sect = data.pop("base", cls.defaults)
        this = load(sect, section=cls.section, overrides=data, interp=True)

        return this

    def to_yaml(self):
        return yaml.dump(self.model_dump(mode="json"), sort_keys=False)
