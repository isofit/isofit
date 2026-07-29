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
"""Batched multicomponent surface prior.

The multicomponent surface picks one of several reflectance components per
pixel by nearest-neighbour distance, then uses that component's mean and
covariance as the retrieval prior
(:class:`isofit.surface.surface_multicomp.MultiComponentSurface`).

Two details drive the design here:

* **The component index is discrete and per-pixel.** Rather than gathering a
  ``(B, n, n)`` stack of covariances -- hundreds of megabytes at production
  batch sizes -- callers add each component's shared matrix to the subset of
  pixels that selected it. There are only a handful of components, so the loop
  is short and allocates nothing per pixel.

* **Selection is normally frozen after initialization.** When ``select_on_init``
  is set (the common configuration) the scalar code caches the component on the
  geometry object and never recomputes it, so the batched path must do the same
  or it would drift as the state vector moves.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

Logger = logging.getLogger(__name__)


class TorchMultiComponentSurface:
    """Batched component selection and prior lookup.

    Args:
        surface: A built :class:`MultiComponentSurface` (or a subclass such as
            the glint model).
        device: Torch device.
        dtype: Torch dtype.

    Notes:
        The per-component ``Sa_inv``/``Sa_inv_sqrt`` matrices are copied from the
        surface object as-is. They were produced on CPU by
        ``isofit.core.common.svd_inv_sqrt``, whose eigenvalue-stabilization
        ladder can add a diagonal offset; recomputing them here could silently
        pick a different rung, so they are inherited rather than rebuilt.
    """

    def __init__(self, surface, device=None, dtype=torch.float64):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.surface = surface

        self.n_wl = int(surface.n_wl)
        self.n_comp = int(surface.n_comp)
        self.normalize = surface.normalize
        self.selection_metric = surface.selection_metric
        self.select_on_init = bool(surface.select_on_init)

        self.idx_lamb = self._long(surface.idx_lamb)
        self.idx_ref = self._long(surface.idx_ref)
        self.n_state = int(surface.n_state)

        self.component_means = self._t(np.asarray(surface.component_means))
        self.mus = self._t(np.asarray(surface.mus))

        # Per-component prior inverses, stacked on the component axis.
        self.Sa_inv_normalized = self._t(np.asarray(surface.Sa_inv_normalized))
        self.Sa_inv_sqrt_normalized = self._t(
            np.asarray(surface.Sa_inv_sqrt_normalized)
        )
        self.component_cov_diag = self._t(
            np.stack([np.diag(c) for c in np.asarray(surface.component_covs)])
        )

        # Extra, non-reflectance surface state (the glint model's SKY_GLINT and
        # SUN_GLINT). GlintModelSurface.Sa block-diags a constant inverse onto
        # the per-component one (surface_glint_model.py:153-156) and
        # ForwardModel.Sa then divides the whole thing by scale_surface**2, so
        # the extra block takes the same scaling as the reflectance block and
        # can simply be appended to each component's matrix once, here.
        self.n_extra = self.n_state - self.n_wl
        self.Sa_inv_extra = None
        self.extra_prior_var = None
        self.extra_prior_mean = None
        if self.n_extra:
            self._load_extra_prior(surface)
            padded = torch.zeros(
                (self.n_comp, self.n_state, self.n_state),
                dtype=self.dtype,
                device=self.device,
            )
            padded[:, : self.n_wl, : self.n_wl] = self.Sa_inv_normalized
            padded[:, self.n_wl :, self.n_wl :] = self.Sa_inv_extra
            self.Sa_inv_normalized = padded

        self.wl_ref = self._t(np.asarray(surface.wl)[np.asarray(surface.idx_ref)])

        if self.selection_metric == "SGA":
            # Gradient of each component mean, precomputed once.
            self.mu_grads = self._t(
                np.stack(
                    [
                        _np_gradient_smoothed(
                            np.asarray(surface.wl)[np.asarray(surface.idx_ref)], mu
                        )
                        for mu in np.asarray(surface.mus)
                    ]
                )
            )

    def _t(self, array):
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    def _long(self, array):
        return torch.as_tensor(
            np.asarray(array), dtype=torch.int64, device=self.device
        )

    # --- normalization -----------------------------------------------------------

    def norm(self, lamb_ref: torch.Tensor) -> torch.Tensor:
        """Per-pixel normalization factor (``MultiComponentSurface.norm``).

        Args:
            lamb_ref: ``(B, n_ref)`` reflectance over the reference windows.

        Returns:
            ``(B,)`` normalization factors.
        """
        if self.normalize == "Euclidean":
            return torch.linalg.norm(lamb_ref, dim=1)
        if self.normalize == "RMS":
            return torch.sqrt(torch.mean(lamb_ref**2, dim=1))
        if self.normalize == "None":
            return torch.ones(lamb_ref.shape[0], dtype=self.dtype, device=self.device)
        raise ValueError(f"Unrecognized Normalization: {self.normalize}")

    # --- component selection -----------------------------------------------------

    def component(self, x_surface: torch.Tensor, geom=None) -> torch.Tensor:
        """Select a surface component per pixel.

        Args:
            x_surface: ``(B, n_state)`` surface state.
            geom: :class:`BatchedGeometry`. When it carries ``surf_cmp_init``,
                that frozen selection is returned unchanged.

        Returns:
            ``(B,)`` int64 component indices.

        Notes:
            Mirrors the freeze chain in ``MultiComponentSurface.component``
            (surface_multicomp.py:140-147): a cached selection wins, otherwise
            selection runs against ``x_surf_init`` when ``select_on_init`` is
            set, otherwise against the current state.
        """
        B = x_surface.shape[0]

        if self.n_comp <= 1:
            return torch.zeros(B, dtype=torch.int64, device=self.device)

        if geom is not None and getattr(geom, "surf_cmp_init", None) is not None:
            return geom.surf_cmp_init

        source = x_surface
        if (
            self.select_on_init
            and geom is not None
            and getattr(geom, "x_surf_init", None) is not None
        ):
            source = geom.x_surf_init

        lamb = self.calc_lamb(source)
        lamb_ref = lamb.index_select(1, self.idx_ref)
        lamb_ref = lamb_ref / self.norm(lamb_ref).unsqueeze(1)

        distances = self.distances(lamb_ref)
        closest = torch.argmin(distances, dim=1)

        # Cache the selection so later iterations reuse it, as the scalar path
        # does via geom.surf_cmp_init.
        if (
            self.select_on_init
            and geom is not None
            and getattr(geom, "x_surf_init", None) is not None
            and getattr(geom, "surf_cmp_init", None) is None
        ):
            geom.surf_cmp_init = closest

        return closest

    def distances(self, lamb_ref: torch.Tensor) -> torch.Tensor:
        """Distance from each pixel to each component mean.

        Args:
            lamb_ref: ``(B, n_ref)`` normalized reference reflectance.

        Returns:
            ``(B, n_comp)`` distances under the configured metric.
        """
        if self.selection_metric == "Euclidean":
            return torch.sum(
                (lamb_ref.unsqueeze(1) - self.mus.unsqueeze(0)) ** 2, dim=2
            )
        if self.selection_metric == "SpecAngle":
            return self.spectral_angle_distance(lamb_ref, self.mus)
        if self.selection_metric == "SGA":
            grads = _torch_gradient_smoothed(self.wl_ref, lamb_ref)
            return self.spectral_angle_distance(grads, self.mu_grads)
        raise ValueError(
            f"Surface component selection metric not valid: {self.selection_metric}"
        )

    @staticmethod
    def spectral_angle_distance(lamb_ref: torch.Tensor, mus: torch.Tensor):
        """Angle between each pixel spectrum and each component mean."""
        num = lamb_ref @ mus.T
        denom = torch.linalg.norm(lamb_ref, dim=1).unsqueeze(1) * torch.linalg.norm(
            mus, dim=1
        ).unsqueeze(0)
        return torch.arccos(torch.clamp(num / denom, -1.0, 1.0))

    # --- prior -------------------------------------------------------------------

    def calc_lamb(self, x_surface: torch.Tensor) -> torch.Tensor:
        """Lambertian reflectance portion of the surface state."""
        return x_surface.index_select(1, self.idx_lamb)

    def xa(self, x_surface: torch.Tensor, geom=None, ci: torch.Tensor = None):
        """Prior mean per pixel (``MultiComponentSurface.xa``).

        Returns:
            ``(B, n_state)`` prior mean, zero outside the reflectance block.
        """
        if ci is None:
            ci = self.component(x_surface, geom)

        lamb = self.calc_lamb(x_surface)
        lamb_ref = lamb.index_select(1, self.idx_ref)
        scale = self.norm(lamb_ref).unsqueeze(1)

        mu = torch.zeros(
            (x_surface.shape[0], self.n_state), dtype=self.dtype, device=self.device
        )
        mu[:, self.idx_lamb] = self.component_means.index_select(0, ci) * scale
        if self.n_extra:
            # Constant config means, matching GlintModelSurface.xa
            # (surface_glint_model.py:134-140). Note update_heuristic_prior_means
            # instead uses the pixel's own SUN_GLINT, which is why
            # per_pixel_heuristic_prior is refused alongside glint.
            mu[:, self.n_wl :] = self.extra_prior_mean.unsqueeze(0)
        return mu

    def _load_extra_prior(self, surface):
        """Pull the non-reflectance prior off a glint-style surface.

        Reads the already-built attributes rather than reconstructing them:
        ``Sa_inv_glint`` came from ``svd_inv_sqrt``, whose eigenvalue ladder can
        add a diagonal offset, and recomputing it could silently pick a
        different rung.
        """
        inv = getattr(surface, "Sa_inv_glint", None)
        if inv is None:
            raise NotImplementedError(
                f"surface has {self.n_extra} non-reflectance state element(s) "
                "but no recognized prior for them (expected Sa_inv_glint). Only "
                "the glint model is supported. Run with backend='numpy'."
            )
        self.Sa_inv_extra = self._t(np.asarray(inv))
        if self.Sa_inv_extra.shape != (self.n_extra, self.n_extra):
            raise ValueError(
                f"Sa_inv_glint is {tuple(self.Sa_inv_extra.shape)}, expected "
                f"({self.n_extra}, {self.n_extra})"
            )

        # Ordering is alphabetical by state name, which is how
        # GlintModelSurface appends them (surface_glint_model.py:57) and how
        # multistate.py sorts non-rfl names. SKY_GLINT precedes SUN_GLINT.
        names = list(surface.statevec_names)[self.n_wl :]
        var = {"SKY_GLINT": surface.sky_glint_sigma, "SUN_GLINT": surface.sun_glint_sigma}
        mean = {"SKY_GLINT": surface.sky_glint_mean, "SUN_GLINT": surface.sun_glint_mean}
        missing = [n for n in names if n not in var]
        if missing:
            raise NotImplementedError(
                f"unrecognized non-reflectance surface state {missing}; only "
                "SKY_GLINT and SUN_GLINT are supported."
            )
        self.extra_prior_var = self._t(np.array([float(var[n]) for n in names]))
        self.extra_prior_mean = self._t(np.array([float(mean[n]) for n in names]))

    def prior_scale(self, x_surface: torch.Tensor, ci: torch.Tensor):
        """Per-pixel divisor applied to the normalized prior precision.

        Returns:
            ``(B,)`` values of ``scale_surface**2`` from ``ForwardModel.Sa``.

        Notes:
            ``MultiComponentSurface.Sa`` returns an *un-normalized* covariance
            (``Cov * norm(lamb_ref)**2``) alongside a *normalized* inverse, and
            ``ForwardModel.Sa`` then reconciles them by dividing the inverse by
            ``scale_surface**2 = mean(diag(Sa_surface))``
            (isofit/core/forward.py:259-270). Expanding that:

                scale_surface**2 == mean(diag(Cov_component)) * norm(lamb_ref)**2

            Both factors matter -- using only the norm leaves the prior
            mis-weighted by the component's mean variance.
        """
        lamb_ref = self.calc_lamb(x_surface).index_select(1, self.idx_ref)
        norm_sq = self.norm(lamb_ref) ** 2
        mean_diag = self.component_cov_diag.index_select(0, ci).mean(dim=1)
        if not self.n_extra:
            return mean_diag * norm_sq
        # The mean runs over the FULL surface diagonal, which for a glint
        # surface is n_wl reflectance variances plus the glint variances
        # (surface_glint_model.py:150-151, forward.py:259-270). Averaging over
        # n_wl alone mis-weights the whole prior, and does so silently -- unlike
        # add_Sa_inv, which raises on a shape mismatch.
        total = mean_diag * norm_sq * self.n_wl + self.extra_prior_var.sum()
        return total / self.n_state

    def add_Sa_inv(
        self, target: torch.Tensor, ci: torch.Tensor, scale: torch.Tensor = None
    ) -> torch.Tensor:
        """Add each pixel's prior precision into a batch of matrices.

        Args:
            target: ``(B, n, n)`` matrices to accumulate into.
            ci: ``(B,)`` component index per pixel.
            scale: Optional ``(B,)`` divisor applied to the precision.

        Returns:
            ``target``, modified in place.

        Notes:
            Loops over components rather than gathering ``(B, n, n)``. With 425
            reflectance states and a batch of 512, a gather is ~740 MiB; this
            touches each component's shared matrix once instead.
        """
        n = target.shape[-1]
        for c in range(self.n_comp):
            mask = ci == c
            if not bool(mask.any()):
                continue
            block = self.Sa_inv_normalized[c, :n, :n]
            if scale is None:
                target[mask] += block
            else:
                target[mask] += block.unsqueeze(0) / scale[mask].view(-1, 1, 1)
        return target

    # --- linearization -----------------------------------------------------------

    def evaluate_theta(
        self,
        s_alb,
        geom,
        L_tot,
        L_dir_dir=None,
        L_dir_dif=None,
        L_dif_dir=None,
        L_dif_dif=None,
        heterogeneous: bool = False,
    ):
        """Surface linearization coefficients (surface_multicomp.py:386-409).

        Returns:
            ``(B, n_wl)`` per-channel coefficients; the diagonal of ``H``.
        """
        if heterogeneous:
            return L_dir_dir + L_dif_dir

        bg = geom.bg_rfl
        if bg is None:
            # Homogeneous with no background reflectance: theta reduces to L_tot.
            return L_tot
        return L_tot + (L_tot * (bg * s_alb) / (1 - (bg * s_alb)))


def _np_gradient_smoothed(wl, values, sigma=2):
    """Smoothed spectral gradient, matching the scalar SGA metric."""
    from scipy.ndimage import gaussian_filter1d

    return np.gradient(gaussian_filter1d(values, sigma=sigma), wl)


def _torch_gradient_smoothed(wl: torch.Tensor, values: torch.Tensor, sigma: float = 2):
    """Batched smoothed spectral gradient.

    Mirrors ``gaussian_filter1d`` followed by ``np.gradient``: a reflect-padded
    Gaussian convolution, then central differences over a non-uniform axis.
    """
    radius = int(4 * sigma + 0.5)  # scipy's truncate=4.0 default
    taps = torch.arange(
        -radius, radius + 1, dtype=values.dtype, device=values.device
    )
    kernel = torch.exp(-(taps**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # scipy's gaussian_filter1d mode='reflect' repeats the edge sample
    # (d c b a | a b c d), which numpy calls 'symmetric'. torch's
    # F.pad(mode='reflect') omits it (d c b | a b c d), so build the padding by
    # hand -- otherwise every spectrum differs near its ends, which is exactly
    # where the gradient metric is most sensitive.
    if radius > values.shape[1]:
        raise ValueError(
            f"smoothing radius {radius} exceeds spectrum length {values.shape[1]}"
        )
    left = torch.flip(values[:, :radius], dims=[1])
    right = torch.flip(values[:, -radius:], dims=[1])
    padded = torch.cat([left, values, right], dim=1).unsqueeze(1)

    smoothed = torch.nn.functional.conv1d(
        padded, kernel.view(1, 1, -1)
    ).squeeze(1)

    return _torch_gradient(smoothed, wl)


def _torch_gradient(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``np.gradient`` for a batch of spectra over a non-uniform axis."""
    out = torch.empty_like(values)
    dx = x[1:] - x[:-1]

    # Interior: the non-uniform central difference numpy uses.
    hs, hd = dx[:-1], dx[1:]
    a = -hd / (hs * (hs + hd))
    b = (hd - hs) / (hs * hd)
    c = hs / (hd * (hs + hd))
    out[:, 1:-1] = (
        a * values[:, :-2] + b * values[:, 1:-1] + c * values[:, 2:]
    )

    # Edges: one-sided differences.
    out[:, 0] = (values[:, 1] - values[:, 0]) / dx[0]
    out[:, -1] = (values[:, -1] - values[:, -2]) / dx[-1]
    return out
