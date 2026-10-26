"""
Configuration interpolation utilities for variable substitution.

This package loads configuration files with inheritance, override, and variable
interpolation support. It allows using ${key} or ${.key} syntax to reference
other values within the configuration, and supports command-line overrides
through Click contexts.

Configs are plain nested ``dict`` / ``list`` trees throughout.

The pipeline, run in order by :func:`load`, is split into one module per stage:

- :mod:`.common`    -- :func:`read` plus the dict helpers :func:`get_dotted`,
                       :func:`set_dotted`, and :func:`merge`
- :mod:`.inherit`   -- ``^^`` inheritance (:func:`patch`, :func:`resolve_file`)
- :mod:`.overrides` -- dotted-key overrides (:func:`parse_overrides`, :func:`override`)
- :mod:`.resolve`   -- ``${...}`` interpolation (:func:`interp`, :func:`interpolate`)
- :mod:`.load`      -- :func:`load`, which chains the stages above
- :mod:`.cli`       -- the ``isofit config`` command group
"""

from .common import Loaders, get_dotted, merge, read, set_dotted
from .inherit import patch, resolve_file
from .load import load
from .overrides import override, parse_overrides
from .resolve import Interp, interp, interpolate
