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
    "observer_altitude_km",
    "path_length_km",
    "coszen",
    "cos_i",
    "min_cosi",
    "skyview_factor",
    "latitude",
    "longitude",
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
            # Accept tensors directly. build_batched_geometry derives its
            # fields ON the device, and round-tripping those through
            # np.asarray raises for a CUDA tensor -- a failure that only
            # appears on GPU, never in a CPU-default unit test.
            if torch.is_tensor(value):
                tensor = value.to(device=self.device, dtype=self.dtype).reshape(-1)
            else:
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
            # Same reason as the scalar fields above: bg_rfl may already be a
            # device tensor.
            if torch.is_tensor(bg_rfl):
                self.bg_rfl = bg_rfl.to(device=self.device, dtype=self.dtype)
            else:
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
        sub = {k: v.index_select(0, index) for k, v in self._fields.items()}
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

    def __getitem__(self, key) -> "BatchedGeometry":
        """Return a sub-batch. Accepts a slice or an index array.

        Exists so callers can walk a block of pixels in device-sized chunks the
        same way they slice the parallel numpy arrays.
        """
        if isinstance(key, slice):
            key = torch.arange(*key.indices(self.size), device=self.device)
        return self.index_select(key)

    def __repr__(self) -> str:
        return (
            f"<BatchedGeometry n={self.size} "
            f"fields={sorted(self._fields)} "
            f"bg_rfl={'yes' if self.bg_rfl is not None else 'no'}>"
        )


# --- batched construction from obs/loc slabs -------------------------------------


def build_batched_geometry(
    obs=None,
    loc=None,
    bg_rfl=None,
    svf=None,
    coszen=None,
    full_config=None,
    device=None,
    dtype=torch.float64,
) -> BatchedGeometry:
    """Derive a whole batch of geometries from ``(B, n)`` obs/loc slabs.

    Batched counterpart of :meth:`isofit.core.geometry.Geometry.__init__`. Every
    line of that constructor falls into one of two groups:

    * **Config-driven and therefore uniform across the batch** -- ``max_slope``,
      ``terrain_style``, ``lut_grid``, and the four-way branch that decides where
      ``coszen`` comes from. These are resolved once, here, and the resolved
      branch is applied to the whole batch. This is safe precisely because none
      of the branch predicates read obs/loc: they test the config and whether
      ``solar_zenith``/``coszen`` were supplied at all, both batch-wide facts.
    * **Genuinely per-pixel** -- everything derived from the obs/loc columns.
      These are vectorized.

    Args:
        obs: ``(B, >=9)`` observation metadata, AVIRIS-NG column order.
        loc: ``(B, >=3)`` location metadata (easting, northing, elevation).
        bg_rfl: Optional ``(B, n_wl)`` background reflectance.
        svf: Optional ``(B,)`` or ``(B, 1)`` sky view factor. ``None`` means 1.
        coszen: Scalar cosine of the solar zenith angle for the top of
            atmosphere, as passed to the scalar constructor.
        full_config: The ISOFIT config, or ``None``/``{}`` for the
            backwards-compatible no-config behaviour.
        device: Torch device for the resulting batch.
        dtype: Storage dtype of the resulting batch.

    Returns:
        :class:`BatchedGeometry` carrying every field the scalar constructor
        derives, plus ``use_universal_coszen``, ``terrain_style`` and
        ``max_slope`` as batch-wide Python attributes.

    Raises:
        ValueError: ``coszen`` cannot be determined, mirroring the scalar
            constructor; or ``cos_i`` is needed but no obs was supplied.

    Numerics:
        The derivation runs in float64 regardless of the input dtype, then casts
        to ``dtype`` for storage. The scalar constructor instead inherits the
        obs/loc memmap dtype, so for a *float32* obs file it computes
        ``relative_azimuth`` / ``observer_altitude_km`` / ``coszen`` in float32
        (with parts of the ``min_cosi`` expression promoted to float64 by numpy's
        scalar rules). The batched result is therefore not bit-identical to the
        scalar one for float32 inputs -- it is the more accurate of the two, and
        it matches what :meth:`BatchedGeometry.from_geometries` already produced,
        since that upcasts the finished scalar fields to float64. For float64
        obs/loc the two agree to the last bit except where ``numpy`` and
        ``torch`` round a transcendental (``cos``/``sin``/``arccos``)
        differently, which is bounded by one ulp.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    work = torch.float64

    def _slab(array, name, min_cols):
        if array is None:
            return None
        t = torch.as_tensor(np.asarray(array), dtype=work, device=dev)
        if t.ndim != 2:
            raise ValueError(f"{name} must be 2-D (B, n), got {tuple(t.shape)}")
        if t.shape[1] < min_cols:
            raise ValueError(
                f"{name} has {t.shape[1]} columns, need at least {min_cols}"
            )
        return t

    obs_t = _slab(obs, "obs", 9)
    loc_t = _slab(loc, "loc", 3)
    if obs_t is None and loc_t is None:
        raise ValueError("build_batched_geometry requires obs, loc, or both")
    if obs_t is not None and loc_t is not None and obs_t.shape[0] != loc_t.shape[0]:
        raise ValueError(
            f"obs has {obs_t.shape[0]} rows but loc has {loc_t.shape[0]}"
        )

    size = obs_t.shape[0] if obs_t is not None else loc_t.shape[0]
    fields = {}

    # --- obs-derived (geometry.py:69-78) ---------------------------------------
    solar_zenith = None
    cos_i = None
    if obs_t is not None:
        fields["path_length_km"] = obs_t[:, 0] / 1000  # units.m_to_km
        fields["observer_azimuth"] = obs_t[:, 1]
        fields["observer_zenith"] = obs_t[:, 2]
        fields["solar_azimuth"] = obs_t[:, 3]
        solar_zenith = obs_t[:, 4]
        fields["solar_zenith"] = solar_zenith
        cos_i = obs_t[:, 8]

        delta_phi = torch.abs(fields["solar_azimuth"] - fields["observer_azimuth"])
        fields["relative_azimuth"] = torch.minimum(delta_phi, 360 - delta_phi)

    # --- loc-derived (geometry.py:83-88) ---------------------------------------
    if loc_t is not None:
        fields["surface_elevation_km"] = loc_t[:, 2] / 1000  # units.m_to_km
        fields["latitude"] = loc_t[:, 1]
        fields["longitude"] = loc_t[:, 0]

    if obs_t is not None and loc_t is not None:
        fields["observer_altitude_km"] = fields["surface_elevation_km"] + fields[
            "path_length_km"
        ] * torch.cos(torch.deg2rad(fields["observer_zenith"]))

    # --- coszen resolution (geometry.py:96-139) ---------------------------------
    # Uniform across the batch: every predicate below reads the config or asks
    # whether a column exists at all, never a per-pixel value.
    max_slope = 0.0
    terrain_style = "flat"
    use_universal_coszen = True
    coszen_t = None

    def _from_solar_zenith():
        return torch.cos(torch.deg2rad(solar_zenith))

    def _broadcast(value):
        return torch.full((size,), float(value), dtype=work, device=dev)

    if not full_config:
        if solar_zenith is not None:
            coszen_t = _from_solar_zenith()
            use_universal_coszen = False
        elif coszen is not None:
            coszen_t = _broadcast(coszen)
    else:
        max_slope = full_config.forward_model.surface.max_slope
        terrain_style = full_config.forward_model.surface.terrain_style
        lut_grid = full_config.forward_model.atmosphere.lut_grid

        if lut_grid is not None and "solar_zenith" in lut_grid:
            if solar_zenith is None:
                raise ValueError(
                    "The LUT grid contains 'solar_zenith' but no obs slab was "
                    "given, so the solar zenith angle is unknown."
                )
            coszen_t = _from_solar_zenith()
            use_universal_coszen = False
        elif coszen is None and solar_zenith is not None:
            coszen_t = _from_solar_zenith()
            use_universal_coszen = False
            Logger.warning(
                "coszen was not defined and solar zenith was not found in the lut grid. "
                "This will proceed with the OBS data, however, this may cause small "
                "errors in the forward model."
            )
        elif coszen is not None:
            coszen_t = _broadcast(coszen)
        else:
            raise ValueError(
                "coszen is not defined and valid solar zenith not found in OBS data."
            )

    # --- skyview factor ---------------------------------------------------------
    if svf is None:
        svf_t = torch.ones(size, dtype=work, device=dev)
    else:
        svf_t = torch.as_tensor(
            np.asarray(svf), dtype=work, device=dev
        ).reshape(size, -1)
        if svf_t.shape[1] != 1:
            raise ValueError(
                f"svf must be one value per pixel, got {svf_t.shape[1]} per pixel"
            )
        svf_t = svf_t[:, 0]

    # --- terrain clamps (geometry.py:141-165) -----------------------------------
    if coszen_t is not None:
        if terrain_style == "flat":
            # Pretend the surface is flat regardless of the obs cos_i.
            cos_i = coszen_t
        elif cos_i is None:
            raise ValueError(
                f"terrain_style={terrain_style!r} needs the obs cos_i column, "
                "but no obs slab was given."
            )

        # max(0, sin(acos(coszen))*sin(max_slope)*cos(180deg) + coszen*cos(max_slope))
        #
        # The clamps below are written as torch.where rather than
        # clamp/maximum/minimum on purpose. Python's builtin max/min return the
        # *first* argument when the comparison is False, so ``max(0, nan)`` is 0
        # and ``min(hi, nan)`` is hi, whereas torch's reductions propagate NaN.
        # A NaN geometry value would otherwise take a different path here than
        # in the scalar constructor.
        slope = torch.sin(torch.arccos(coszen_t)) * np.sin(
            np.radians(max_slope)
        ) * np.cos(np.radians(180)) + coszen_t * np.cos(np.radians(max_slope))
        zero = torch.zeros((), dtype=work, device=dev)
        min_cosi = torch.where(slope > 0, slope, zero)  # max(0, slope)

        one = torch.ones((), dtype=work, device=dev)

        def _clamp(value):
            # max(min_cosi, min(value, 1.0))
            upper = torch.where(one < value, one, value)
            return torch.where(upper > min_cosi, upper, min_cosi)

        coszen_clamped = _clamp(coszen_t)
        cos_i = _clamp(cos_i)
        coszen_t = coszen_clamped
        svf_t = torch.where((svf_t > 0) & (svf_t <= 1), svf_t, one)

        fields["min_cosi"] = min_cosi
        fields["coszen"] = coszen_t
        fields["cos_i"] = cos_i
    else:
        Logger.warning(
            "Unable to determine coszen. Proceeding without will cause errors "
            "during the inversion."
        )
        if cos_i is not None:
            fields["cos_i"] = cos_i

    fields["skyview_factor"] = svf_t

    out = BatchedGeometry(fields, bg_rfl=bg_rfl, device=dev, dtype=dtype)
    # Batch-wide attributes the scalar object also exposes.
    out.use_universal_coszen = use_universal_coszen
    out.terrain_style = terrain_style
    out.max_slope = max_slope
    return out
