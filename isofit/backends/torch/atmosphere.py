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
"""Batched atmospheric look-up and radiance terms.

Mirrors :class:`isofit.atmosphere.atmosphere.BaseAtmosphere` for a batch of
pixels: assembling LUT coordinates from the retrieval state and the per-pixel
geometry, sampling the table, and converting the sampled quantities into the
radiance terms the forward model consumes.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from isofit.backends.torch.lut import BatchedLUT

Logger = logging.getLogger(__name__)


def transm_to_rdn(transm, coszen, solar_irr):
    """Convert a unitless atmospheric quantity to radiance, batched.

    Batched counterpart of :func:`isofit.core.units.transm_to_rdn` composed with
    :func:`isofit.core.units.E_to_L`: ``transm * solar_irr * coszen / pi``.

    Args:
        transm: ``(B, n_wl)`` unitless atmospheric quantity.
        coszen: ``(B,)`` cosine of the solar zenith angle.
        solar_irr: ``(n_wl,)`` solar irradiance.

    Returns:
        ``(B, n_wl)`` radiance.
    """
    return transm * solar_irr.unsqueeze(0) * coszen.unsqueeze(1) / np.pi


class TorchAtmosphere:
    """Batched LUT sampling and radiance terms for a fixed atmosphere model.

    Args:
        atmosphere: A built :class:`BaseAtmosphere`; its LUT interpolators,
            solar irradiance, mode flags and index bookkeeping are reused.
        device: Torch device.
        dtype: Torch dtype for sampled values.

    Notes:
        Constructed from an already-built atmosphere rather than from config, so
        the LUT contents and derived constants are exactly those the numpy path
        would use.
    """

    def __init__(self, atmosphere, device=None, dtype=torch.float64):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.atmosphere = atmosphere

        # BaseAtmosphere subclasses the LUT Reader, so the interpolators live on
        # the atmosphere object itself; `atmosphere.lut` is the xarray Dataset
        # they were built from.
        interpolators = getattr(atmosphere, "interpolators", None)
        if not interpolators:
            raise ValueError(
                "The atmosphere has no LUT interpolators. It must be constructed "
                "with build_interpolators=True for the torch backend to sample it."
            )

        self.lut = BatchedLUT.from_interpolators(
            interpolators, device=self.device, dtype=self.dtype
        )

        self.rt_mode = atmosphere.rt_mode
        self.multipart_transmittance = bool(atmosphere.multipart_transmittance)
        self.coupling_terms = list(getattr(atmosphere, "coupling_terms", []))
        self.esd_correction = float(atmosphere.esd_correction)

        self.solar_irr = torch.as_tensor(
            np.asarray(atmosphere.solar_irr), dtype=self.dtype, device=self.device
        )
        self.n_wl = int(self.solar_irr.numel())

        # LUT coordinate bookkeeping (isofit/atmosphere/atmosphere.py:376-398).
        self.lut_names = list(atmosphere.lut_names)
        self.n_dims = len(self.lut_names)
        self.idx_x_RT = list(atmosphere.indices.x_RT)
        self.idx_geom = dict(atmosphere.indices.geom)
        self.convert_observer_zenith = atmosphere.indices.convert_observer_zenith

    def build_points(self, x_RT: torch.Tensor, geom) -> torch.Tensor:
        """Assemble LUT coordinates for a batch.

        Args:
            x_RT: ``(B, n_atm)`` radiative-transfer portion of the state vector.
            geom: :class:`BatchedGeometry` supplying the geometry dimensions.

        Returns:
            ``(B, D)`` LUT coordinates.
        """
        if x_RT.ndim == 1:
            x_RT = x_RT.unsqueeze(0)
        B = x_RT.shape[0]

        points = torch.zeros(
            (B, self.n_dims), dtype=self.lut.coord_dtype, device=self.device
        )

        if self.idx_x_RT:
            idx = torch.as_tensor(self.idx_x_RT, dtype=torch.int64, device=self.device)
            points[:, idx] = x_RT.to(self.lut.coord_dtype)

        for i, key in self.idx_geom.items():
            points[:, i] = geom.get(key).to(self.lut.coord_dtype)

        # MODTRAN reports observer zenith from the opposite direction.
        if self.convert_observer_zenith:
            cols = list(self.convert_observer_zenith)
            for col in cols:
                points[:, col] = 180.0 - points[:, col]

        return points

    def get(self, x_RT: torch.Tensor, geom) -> dict:
        """Sample every LUT quantity for a batch (``BaseAtmosphere.get``)."""
        return self.lut.interpolate(self.build_points(x_RT, geom))

    def get_L_atm(self, r: dict, geom) -> torch.Tensor:
        """Atmospheric path radiance (``BaseAtmosphere.get_L_atm``).

        Takes an already-sampled ``r`` rather than re-sampling the LUT: the
        scalar implementation calls ``get()`` again here, which would double the
        interpolation cost in the batched path for an identical result.
        """
        rho_atm = r["rhoatm"]
        if self.rt_mode == "rdn":
            L_atm = rho_atm
        else:
            L_atm = transm_to_rdn(rho_atm, geom.coszen, self.solar_irr)
        return L_atm * self.esd_correction

    def get_upward_transm(self, r: dict, max_transm: float = 1.05) -> torch.Tensor:
        """Total upward transmittance (``BaseAtmosphere.get_upward_transm``).

        Returns zeros in the 1-component case, where the upward transmittance
        keys are absent or scalar, matching the scalar implementation.
        """
        transm_up_dir = r.get("transm_up_dir")
        transm_up_dif = r.get("transm_up_dif")

        if not torch.is_tensor(transm_up_dir) or transm_up_dir.shape[-1] == 1:
            return torch.zeros_like(r["rhoatm"])

        transup = transm_up_dir + transm_up_dif

        peak = float(transup.max())
        if peak > max_transm:
            raise ValueError(
                f"Upward transmittance (max:{peak}) is greater than {max_transm}. "
                "Verify 'transm_up_dir' and 'transm_up_dif' keys are in units of "
                "transmittance."
            )
        return transup

    def get_L_coupled(self, r: dict, geom, rho_dif_dif=0.0, terrain_rereflection=None):
        """Radiances along the four sun-surface-sensor paths.

        Batched counterpart of :meth:`ForwardModel.get_L_coupled`
        (isofit/core/forward.py:421-499). Only applicable in the 6-component
        case, where ``r`` carries the separated transmittances.

        Args:
            r: Sampled LUT quantities.
            geom: :class:`BatchedGeometry`.
            rho_dif_dif: ``(B, n_wl)`` hemispherical-hemispherical reflectance,
                or 0.0.
            terrain_rereflection: Callable applying terrain re-reflection, or
                ``None`` for the homogeneous case (a factor of 1).

        Returns:
            ``(L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif)``.
        """
        coszen = geom.coszen.unsqueeze(1)
        cos_i = geom.cos_i.unsqueeze(1)

        L_coupled = []
        for key in self.coupling_terms:
            term = r[key]
            L_coupled.append(
                transm_to_rdn(term, geom.coszen, self.solar_irr)
                if self.rt_mode == "transm"
                else term
            )

        # Topographic shadow mask (0=shadow, 1=sunlit); always 1.0 for now,
        # matching the scalar path.
        b = 1.0

        # Background topography assumptions.
        cos_i_bg = coszen
        skyview_factor_bg = 1.0

        skyview = geom.skyview_factor.unsqueeze(1)

        L_dir_dir = L_coupled[0] / coszen * cos_i * b
        L_dir_dif = L_coupled[2] / coszen * cos_i_bg

        t_down_dir = r["transm_down_dir"]
        L_dif_dir = L_coupled[1] * (
            (b * t_down_dir * (cos_i / coszen))
            + ((1 - b * t_down_dir) * skyview)
        )
        L_dif_dif = L_coupled[3] * (
            (t_down_dir * (cos_i_bg / coszen))
            + ((1 - t_down_dir) * skyview_factor_bg)
        )

        # Re-reflection from nearby terrain.
        t = (
            1.0
            if terrain_rereflection is None
            else terrain_rereflection(rho_dif_dif=rho_dif_dif, geom=geom)
        )
        L_dir_dir = L_dir_dir * t
        L_dif_dir = L_dif_dir * t
        L_dir_dif = L_dir_dif * t
        L_dif_dif = L_dif_dif * t

        # Guanter et al. (2009) eq. 11. With no rho_dif_dif this reduces to 1.
        eq_11_term = 1 - (r["sphalb"] * rho_dif_dif)

        # Order matters: calc_rdn needs L_tot *without* the eq. 11 adjustment,
        # otherwise the surface-atmosphere coupling is counted twice.
        L_tot = L_dir_dir + L_dif_dir + L_dir_dif + L_dif_dif
        L_dif_dir = L_dif_dir / eq_11_term
        L_dif_dif = L_dif_dif / eq_11_term

        return L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif
