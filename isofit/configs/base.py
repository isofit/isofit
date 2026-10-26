"""
Shared base model for config sections.

The section models carry no Python-side defaults; those live in the base
configs under ``isofit/configs/yamls/``. :class:`SectionModel` lets a section be
constructed on its own by seeding it from one of those base configs via the
``base`` keyword, e.g. ``ImplementationConfig(base="vswir", n_cores=4)``.
"""

from typing import Any, ClassVar, Optional

from pydantic import BaseModel, model_validator


class SectionModel(BaseModel):
    """
    Base model for config sections that can be seeded from a base config.

    Passing ``base=<name>`` loads the base config ``name`` (see
    :func:`isofit.configs.loader.load_base_config`), deep-merges the remaining
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

    @model_validator(mode="before")
    @classmethod
    def _apply_base(cls, data: Any) -> Any:
        """
        Seed the input from the base config named by ``data["base"]``.
        """
        if cls.section is None or not isinstance(data, dict):
            return data

        # Deferred: the loader imports Config, which imports every section
        from isofit.configs.loader import load_base_config
        from isofit.configs.utils import interpolate

        data = dict(data)
        name = data.pop("base", "default")

        # Overlay before interpolating so ${...} references follow the overrides
        box = load_base_config(name, interp=False)
        target = box
        if cls.section:
            try:
                target = interpolate.get_dotted(box, cls.section)
            except KeyError:
                target = None
            if not isinstance(target, dict):
                target = {}
                interpolate.set_dotted(box, cls.section, target)

        interpolate.merge(target, data)
        interpolate.interpolate(box)
        box.pop("env", None)

        if cls.section:
            return interpolate.get_dotted(box, cls.section)
        return box
