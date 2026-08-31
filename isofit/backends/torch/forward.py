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
"""Batched top-of-atmosphere radiance model.

Batched counterpart of the radiance arithmetic in
:class:`isofit.core.forward.ForwardModel`. Every operation here is elementwise
over wavelength and broadcast over the pixel batch, so the batched form is a
direct transcription of the scalar one with an added leading dimension.
"""

from __future__ import annotations

import logging

import torch

from isofit.backends.torch.atmosphere import TorchAtmosphere

Logger = logging.getLogger(__name__)


def terrain_rereflection_heterogeneous(
    rho_dif_dif, geom, skyview_factor_bg=1.0, cos_slope=1.0, cos_slope_bg=1.0
):
    """Isotropic scattering from nearby terrain (forward.py:597-604)."""
    skyview = geom.skyview_factor.unsqueeze(1)
    v_t = (1 + cos_slope) / 2 - skyview
    v_t_avg = (1 + cos_slope_bg) / 2 - skyview_factor_bg
    return 1 + ((rho_dif_dif * v_t) / (1 - rho_dif_dif * v_t_avg))


def terrain_rereflection_homogeneous(rho_dif_dif, geom, **_):
    """Terrain scattering is ignored when all four reflectances are equal."""
    return 1.0


def calc_rdn_bgrfl_heterogeneous(
    rho_dir_dif, rho_dif_dif, L_dir_dif, L_dif_dif, L_tot, s_alb
):
    """Background-reflectance contribution to TOA radiance (forward.py:581-589)."""
    return (
        (L_dir_dif * rho_dir_dif)
        + (L_dif_dif * rho_dif_dif)
        + (L_tot * (s_alb * rho_dif_dif) * rho_dif_dif) / (1 - s_alb * rho_dif_dif)
    )


def calc_rdn_bgrfl_homogeneous(
    rho_dir_dif, rho_dif_dif, L_dir_dif, L_dif_dif, L_tot, s_alb
):
    """Zero in the homogeneous case (forward.py:591-595)."""
    return torch.zeros_like(L_tot)


class TorchRadiance:
    """Batched forward radiance model.

    Args:
        fm: A built :class:`isofit.core.forward.ForwardModel`, used for its
            atmosphere, mode flags and index layout.
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(self, fm, device=None, dtype=torch.float64):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.fm = fm

        self.atmosphere = TorchAtmosphere(fm.atmosphere, device=self.device, dtype=dtype)
        self.multipart_transmittance = self.atmosphere.multipart_transmittance

        # Mirror the dispatch ForwardModel performs at construction
        # (isofit/core/forward.py:161-166) rather than re-testing per call.
        self.use_background_rfl = bool(fm.surface.use_background_rfl)
        if self.use_background_rfl:
            self.terrain_rereflection = terrain_rereflection_heterogeneous
            self.calc_rdn_bgrfl = calc_rdn_bgrfl_heterogeneous
        else:
            self.terrain_rereflection = terrain_rereflection_homogeneous
            self.calc_rdn_bgrfl = calc_rdn_bgrfl_homogeneous

    def calc_atmosphere_quantities(
        self, x_atm: torch.Tensor, geom, rho_dif_dif=0.0, r: dict = None
    ):
        """Sample the LUT and build the radiance terms (forward.py:356-419).

        Args:
            x_atm: ``(B, n_atm)`` atmospheric state.
            geom: :class:`BatchedGeometry`.
            rho_dif_dif: ``(B, n_wl)`` hemispherical-hemispherical reflectance.
            r: Already-sampled LUT quantities. When given, the LUT is not
                sampled and ``x_atm`` is used only for its shape. This exists so
                a caller that already holds ``r`` -- notably the analytic
                derivative path, which needs to differentiate *through* this
                function with respect to ``r`` -- can drive the same arithmetic
                without a second interpolation.

        Returns:
            ``(r, L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif)`` with each
            radiance term shaped ``(B, n_wl)``.
        """
        if r is None:
            r = self.atmosphere.get(x_atm, geom)

        if self.multipart_transmittance:
            (
                L_tot,
                L_dir_dir,
                L_dif_dir,
                L_dir_dif,
                L_dif_dif,
            ) = self.atmosphere.get_L_coupled(
                r,
                geom,
                rho_dif_dif=rho_dif_dif,
                terrain_rereflection=self.terrain_rereflection,
            )
        else:
            # 1-component: transm_down_dif carries the total transmittance.
            if self.atmosphere.rt_mode == "rdn":
                L_tot = r["transm_down_dif"]
            else:
                from isofit.backends.torch.atmosphere import transm_to_rdn

                L_tot = transm_to_rdn(
                    r["transm_down_dif"], geom.coszen, self.atmosphere.solar_irr
                )
            zero = torch.zeros_like(L_tot)
            L_dir_dir = zero
            L_dif_dir = zero
            L_dir_dif = zero
            L_dif_dif = zero

        # Earth-Sun distance correction between the LUT build date and the scene.
        esd = self.atmosphere.esd_correction
        return (
            r,
            L_tot * esd,
            L_dir_dir * esd,
            L_dif_dir * esd,
            L_dir_dif * esd,
            L_dif_dif * esd,
        )

    def calc_rdn(
        self,
        rho_dir_dir,
        rho_dif_dir,
        rho_dir_dif,
        rho_dif_dif,
        Ls,
        L_tot,
        L_dir_dir,
        L_dif_dir,
        L_dir_dif,
        L_dif_dif,
        r,
        geom,
    ):
        """Top-of-atmosphere radiance (forward.py:501-579).

        The four reflectance terms correspond to the direct/diffuse combinations
        of the downward and upward paths (Guanter 2006; Vermote 1997; Tanre
        1983). Under the Lambertian assumption they are equal and the model
        reduces to ``L_atm + L_tot*rho/(1 - S*rho) + L_up``.

        Returns:
            ``(B, n_wl)`` at-sensor radiance at RT wavelengths.
        """
        L_atm = self.atmosphere.get_L_atm(r, geom)

        s_alb = r["sphalb"]
        atm_surface_scattering = s_alb * rho_dif_dif

        if not self.multipart_transmittance:
            # 1-component: all four reflectances collapse to one, and the
            # spherical-albedo factor drops out of the numerator.
            rho_dif_dif = rho_dir_dir
            atm_surface_scattering = 1.0

        L_up = Ls * self.atmosphere.get_upward_transm(r)

        return (
            L_atm
            + L_dir_dir * rho_dir_dir
            + L_dif_dir * rho_dif_dir
            + L_dir_dif * rho_dir_dif
            + L_dif_dif * rho_dif_dif
            + (L_tot * atm_surface_scattering * rho_dif_dif)
            / (1 - s_alb * rho_dif_dif)
            + L_up
        )
