#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
#
"""Struct-of-arrays geometry for batched retrievals.

:class:`isofit.core.geometry.Geometry` holds one pixel's observation angles as
Python scalars. The batched path needs the same fields across many pixels at
once, so this module stores each field as a tensor indexed by pixel.

The scalar fields are deliberately kept as an explicit, named set rather than
copied reflectively: the LUT dimension names in a config refer to these fields by
name (see ``isofit.atmosphere.atmosphere.Atmosphere.get``), so a silent rename or
omission would surface as a wrong LUT coordinate rather than an error.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

Logger = logging.getLogger(__name__)

#: Per-pixel scalar geometry fields. These are the names a LUT grid dimension or
#: ``indices.geom`` entry may refer to.
SCALAR_FIELDS = (
    "observer_zenith",
    "observer_azimuth",
    "solar_zenith",
    "solar_azimuth",
    "relative_azimuth",
    "surface_elevation_km",
    "path_length_km",
    "coszen",
    "cos_i",
    "skyview_factor",
)


class BatchedGeometry:
    """A batch of per-pixel observation geometries.

    Args:
        fields: Mapping of field name (from :data:`SCALAR_FIELDS`) to a
            ``(B,)`` array or tensor.
        bg_rfl: Optional ``(B, n_wl)`` background reflectance for adjacency
            effects. ``None`` matches the homogeneous case where
            ``Geometry.bg_rfl`` is not an array.
        device: Torch device to hold the batch on.
        dtype: Floating point dtype for the geometry values.

    Attributes:
        surf_cmp_init: Optional ``(B,)`` int64 tensor of frozen multicomponent
            surface indices, mirroring ``geom.surf_cmp_init`` in the scalar path.
        x_surf_init: Optional ``(B, n_surface)`` tensor of initial surface state,
            mirroring ``geom.x_surf_init``.
    """

    def __init__(self, fields: dict, bg_rfl=None, device=None, dtype=torch.float64):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

        unknown = set(fields) - set(SCALAR_FIELDS)
        if unknown:
            raise ValueError(
                f"Unknown geometry field(s): {sorted(unknown)}. "
                f"Known fields are: {list(SCALAR_FIELDS)}"
            )

        self._fields = {}
        size = None
        for name, value in fields.items():
            if value is None:
                continue
            tensor = torch.as_tensor(
                np.asarray(value), dtype=self.dtype, device=self.device
            ).reshape(-1)
            if size is None:
                size = tensor.numel()
            elif tensor.numel() != size:
                raise ValueError(
                    f"Geometry field {name!r} has length {tensor.numel()}, "
                    f"expected {size} to match the rest of the batch"
                )
            self._fields[name] = tensor

        if size is None:
            raise ValueError("BatchedGeometry requires at least one populated field")
        self.size = int(size)

        self.bg_rfl = None
        if bg_rfl is not None:
            self.bg_rfl = torch.as_tensor(
                np.asarray(bg_rfl), dtype=self.dtype, device=self.device
            )
            if self.bg_rfl.ndim != 2 or self.bg_rfl.shape[0] != self.size:
                raise ValueError(
                    f"bg_rfl must have shape ({self.size}, n_wl), "
                    f"got {tuple(self.bg_rfl.shape)}"
                )

        # Retrieval state carried on the geometry in the scalar path.
        self.surf_cmp_init = None
        self.x_surf_init = None
        self.x_atmosphere_init = None
        self.xa = None

    def __len__(self) -> int:
        return self.size

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def get(self, name: str) -> torch.Tensor:
        """Return a geometry field, raising a clear error when it is absent.

        A missing field usually means the LUT expects a grid dimension the input
        obs/loc files did not supply, which is worth failing loudly on rather
        than defaulting to zero.
        """
        if name not in self._fields:
            raise KeyError(
                f"Geometry field {name!r} was not provided for this batch "
                f"(available: {sorted(self._fields)})"
            )
        return self._fields[name]

    def __getattr__(self, name: str):
        # Only consulted for attributes not found normally, so the explicit
        # attributes above (bg_rfl, surf_cmp_init, ...) still take precedence.
        fields = self.__dict__.get("_fields", {})
        if name in fields:
            return fields[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    @classmethod
    def from_geometries(cls, geoms, device=None, dtype=torch.float64):
        """Stack a sequence of scalar :class:`Geometry` objects into a batch.

        Primarily a testing and migration aid: it makes the batched path directly
        comparable against the per-pixel path for the same inputs.
        """
        geoms = list(geoms)
        if not geoms:
            raise ValueError("from_geometries requires at least one Geometry")

        fields = {}
        for name in SCALAR_FIELDS:
            values = [getattr(g, name, None) for g in geoms]
            if any(v is None for v in values):
                continue
            fields[name] = np.asarray(values, dtype=float)

        bg = [getattr(g, "bg_rfl", None) for g in geoms]
        bg_rfl = (
            np.stack(bg)
            if all(isinstance(b, np.ndarray) for b in bg) and len(bg) == len(geoms)
            else None
        )

        return cls(fields, bg_rfl=bg_rfl, device=device, dtype=dtype)

    def index_select(self, index: torch.Tensor) -> "BatchedGeometry":
        """Return a new batch containing only the selected pixels."""
        index = torch.as_tensor(index, device=self.device)
        sub = cls_fields = {k: v.index_select(0, index) for k, v in self._fields.items()}
        out = BatchedGeometry.__new__(BatchedGeometry)
        out.device = self.device
        out.dtype = self.dtype
        out._fields = sub
        out.size = int(index.numel())
        out.bg_rfl = None if self.bg_rfl is None else self.bg_rfl.index_select(0, index)
        out.surf_cmp_init = (
            None
            if self.surf_cmp_init is None
            else self.surf_cmp_init.index_select(0, index)
        )
        out.x_surf_init = (
            None if self.x_surf_init is None else self.x_surf_init.index_select(0, index)
        )
        out.x_atmosphere_init = (
            None
            if self.x_atmosphere_init is None
            else self.x_atmosphere_init.index_select(0, index)
        )
        out.xa = None if self.xa is None else self.xa.index_select(0, index)
        return out

    def __repr__(self) -> str:
        return (
            f"<BatchedGeometry n={self.size} "
            f"fields={sorted(self._fields)} "
            f"bg_rfl={'yes' if self.bg_rfl is not None else 'no'}>"
        )
