"""
Element-accessor helpers for config sections.

The pre-Pydantic ``BaseConfigSection`` exposed a handful of helper methods for
iterating over the "element" fields of a config section (state-vector elements,
input files, ...). A large amount of runtime code in ``core/``, ``inversion/``
and ``atmosphere/`` still relies on those helpers, so this mixin restores them
for the Pydantic models.

The semantics intentionally match the old base class:

- ``get_all_element_names`` / ``get_all_elements`` return every field, in
  declaration order, without filtering.
- ``get_elements`` drops fields whose value is ``None`` and returns the
  remaining ``(elements, names)`` sorted alphabetically by name.
- ``get`` looks an element up by name, returning ``[]`` when absent (mirroring
  the historical behaviour that callers depend on).
"""


class ElementAccessorMixin:
    """Restore the ``BaseConfigSection`` element-iteration API on a model."""

    def get_all_elements(self):
        """Return the value of every field, in declaration order."""
        return [getattr(self, name) for name in type(self).model_fields]

    def get_all_element_names(self):
        """Return the name of every field, in declaration order."""
        return list(type(self).model_fields)

    def get_elements(self):
        """
        Return ``(elements, names)`` for all set (non-``None``) fields.

        Fields whose value is ``None`` are dropped, and the surviving elements
        and names are sorted alphabetically by name to give a deterministic
        ordering.
        """
        names = list(type(self).model_fields)
        elements = [getattr(self, name) for name in names]

        pairs = [
            (element, name)
            for element, name in zip(elements, names)
            if element is not None
        ]
        pairs.sort(key=lambda pair: pair[1])

        elements = [element for element, _ in pairs]
        names = [name for _, name in pairs]
        return elements, names

    def get_element_names(self):
        """Return the names of all set (non-``None``) fields, sorted."""
        return self.get_elements()[1]

    def get(self, name):
        """Return the element with the given name, or ``[]`` if not set."""
        elements, names = self.get_elements()
        try:
            return elements[names.index(name)]
        except ValueError:
            return []
