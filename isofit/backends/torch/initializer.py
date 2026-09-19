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
"""Batched algebraic initializer.

Batched counterpart of :func:`isofit.inversion.inverse_simple.invert_algebraic`
followed by ``surface.fit_params`` and ``ForwardModel.clip_bounds`` -- the chain
:meth:`isofit.utils.analytical_line_torch.TorchWorker._initial_state` runs once
per pixel for the ``algebraic`` initializer.

Why this is worth batching
--------------------------
The scalar chain is not expensive per pixel, but it is *per pixel*: one LUT
interpolation, three spectral resamplings and a Python loop over ~425 state
elements, all under the interpreter. On a 100k-pixel scene that scalar work --
together with per-pixel ``Geometry`` construction -- dominates the runtime of
the whole retrieval stage, while the batched MAP solve it feeds accounts for a
few percent.

What is constant, and what is not
---------------------------------
``invert_algebraic`` is called with ``x_surface`` and ``x_instrument`` taken from
``fm.init``, which does not vary across pixels. So for the surface models this
module accepts:

* ``surface.calc_rfl``  -> ``rho_init``     is constant,
* ``surface.calc_Ls``   -> ``Ls``           is constant,
* ``instrument.calibration(x_instrument)`` -> ``wl, fwhm`` are constant,

and every spectral resampling operator derived from them is constant too. All of
that is hoisted into :class:`BatchedAlgebraicInitializer.__init__` and evaluated
once. The per-pixel work that remains is one LUT sample, a handful of elementwise
spectral expressions, and two clamps.

The constancy of ``calc_rfl``/``calc_Ls`` is *not* universal -- see
:func:`_check_surface_is_geometry_independent`, which rejects the surface models
whose reflectance depends on the geometry (the glint and LUT surfaces) rather
than silently evaluating them at one pixel's geometry and reusing the answer.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.interpolate import interp1d

from isofit.core.common import eps
from isofit.surface.surface import Surface
from isofit.surface.surface_glint_model import GlintModelSurface
from isofit.surface.surface_multicomp import MultiComponentSurface
from isofit.surface.surface_thermal import ThermalSurface

Logger = logging.getLogger(__name__)

#: Unbound ``calc_rfl`` implementations that ignore ``geom`` and so return the
#: same spectrum for every pixel in a batch when ``x_surface`` is fixed.
#: ``GlintModelSurface.calc_rfl`` reads ``geom.observer_zenith`` and
#: ``LUTSurface.calc_lamb`` reads ``geom.solar_zenith``/``geom.observer_zenith``;
#: both are deliberately absent.
GEOM_INDEPENDENT_CALC_RFL = (
    Surface.calc_rfl,
    MultiComponentSurface.calc_rfl,
)

#: ``calc_rfl`` implementations that DO read the geometry but for which a
#: batched equivalent exists, so the per-pixel hoist is unnecessary rather than
#: unsound. ``GlintModelSurface.calc_rfl`` reads ``geom.observer_zenith``;
#: :class:`~isofit.backends.torch.surface.TorchGlintSurface` evaluates the same
#: expression across the batch.
BATCHED_CALC_RFL = (GlintModelSurface.calc_rfl,)

#: Surfaces whose ``fit_params`` has a batched override below. The scalar
#: ``GlintModelSurface.fit_params`` writes ``self.init[sun_glint_ind]`` and
#: ``geom.sun_glint_init``; neither is ever read back -- ``xa`` uses the
#: separate ``sun_glint_mean`` constant, and ``sun_glint_init`` has no reader
#: anywhere in the tree. Verified empirically: running it over two pixels in
#: both orders gives identical results, so there is no cross-pixel state for
#: batching to break.
BATCHED_FIT_PARAMS = (GlintModelSurface.fit_params,)

#: Unbound ``calc_Ls`` implementations that ignore ``geom``. ``ThermalSurface``
#: qualifies because it reaches ``calc_rfl`` only through
#: ``MultiComponentSurface.calc_rfl``, which is itself geometry independent.
GEOM_INDEPENDENT_CALC_LS = (
    Surface.calc_Ls,
    MultiComponentSurface.calc_Ls,
    ThermalSurface.calc_Ls,
)


def _check_surface_is_geometry_independent(surface) -> None:
    """Raise unless this surface's reflectance/emission ignore the geometry.

    Hoisting ``calc_rfl`` and ``calc_Ls`` out of the batch is only valid when
    they do not read the geometry. Detecting that by evaluating the function at
    two pixels and comparing would pass by luck on a flat scene, so the check is
    on the implementation itself.
    """
    cls = type(surface)
    if cls.calc_rfl not in GEOM_INDEPENDENT_CALC_RFL + BATCHED_CALC_RFL:
        raise NotImplementedError(
            f"{cls.__name__}.calc_rfl depends on the per-pixel geometry, so the "
            "batched algebraic initializer cannot hoist it out of the batch. "
            "Use the per-pixel initializer (batched_gather=False)."
        )
    if cls.calc_Ls not in GEOM_INDEPENDENT_CALC_LS:
        raise NotImplementedError(
            f"{cls.__name__}.calc_Ls depends on the per-pixel geometry, so the "
            "batched algebraic initializer cannot hoist it out of the batch. "
            "Use the per-pixel initializer (batched_gather=False)."
        )
    if (
        cls.fit_params is not MultiComponentSurface.fit_params
        and cls.fit_params not in BATCHED_FIT_PARAMS
    ):
        raise NotImplementedError(
            f"{cls.__name__}.fit_params is not the multicomponent implementation "
            "the batched initializer reproduces. Use the per-pixel initializer "
            "(batched_gather=False)."
        )


def surface_is_batchable(surface) -> tuple:
    """Whether the batched algebraic initializer can reproduce this surface.

    Returns ``(ok, reason)``. The validator below raises, which is right when a
    caller explicitly asked for the batched gather; callers that merely default
    to it need to fall back quietly instead, so they probe with this first.
    """
    try:
        _check_surface_is_geometry_independent(surface)
    except NotImplementedError as exc:
        return False, str(exc).split(". ")[0]
    return True, ""


class BatchedInterp1d:
    """``scipy.interpolate.interp1d(x, y, fill_value='extrapolate')`` for a batch.

    Reproduces scipy's ``_call_linear`` term for term rather than using a
    tolerance-based shortcut. That matters even when the two grids are equal:
    scipy still evaluates ``slope * (x_new - x_lo) + y_lo``, which is not
    bit-identical to ``y`` at the knots, and the scalar path has no
    equal-grids fast path to match.

    Args:
        x: ``(n,)`` source grid.
        x_new: ``(m,)`` destination grid.
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(self, x, x_new, device=None, dtype=torch.float64):
        x = np.asarray(x, dtype=float)
        x_new = np.asarray(x_new, dtype=float)

        # interp1d(assume_sorted=False) sorts the source grid first.
        order = np.argsort(x, kind="mergesort")
        self.order = order
        x = x[order]

        # scipy: searchsorted then clip into [1, n-1]; no bounds error because
        # fill_value='extrapolate' disables the bounds check entirely.
        idx = np.searchsorted(x, x_new).clip(1, len(x) - 1).astype(int)
        lo = idx - 1
        hi = idx

        self.lo = torch.as_tensor(lo, dtype=torch.int64, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.int64, device=device)
        self.order_t = torch.as_tensor(order, dtype=torch.int64, device=device)
        self.dx = torch.as_tensor(x[hi] - x[lo], dtype=dtype, device=device)
        self.offset = torch.as_tensor(x_new - x[lo], dtype=dtype, device=device)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """Interpolate ``(B, n)`` spectra onto the destination grid."""
        y = y.index_select(1, self.order_t)
        y_lo = y.index_select(1, self.lo)
        y_hi = y.index_select(1, self.hi)
        slope = (y_hi - y_lo) / self.dx
        return slope * self.offset + y_lo


class BatchedResampler:
    """Batched :meth:`isofit.core.instrument.Instrument.sample`.

    ``Instrument.sample`` has two behaviours for a 1-D spectrum: return it
    untouched when the instrument calibration is fixed and already matches the
    source grid, or convolve it with the instrument's spectral response. The
    second is ``np.dot(H, x)`` with ``H`` built by
    :func:`isofit.core.common.calculate_resample_matrix` from grids that do not
    vary across the batch, so the batched form is the same operator applied as
    one matmul.

    Args:
        instrument: A built :class:`Instrument`.
        x_instrument: The (constant) instrument state.
        wl_hi: Source wavelength grid.
        device: Torch device.
        dtype: Torch dtype.

    Notes:
        The matmul path is the same linear operator as the scalar path, but a
        ``(B, n) @ (n, m)`` GEMM does not necessarily accumulate in the same
        order as ``B`` separate ``(m, n) @ (n, 1)`` GEMVs, so agreement there is
        to floating-point summation order rather than bit-for-bit. The identity
        path -- the configuration ISOFIT ships and the one this backend has been
        measured on -- is exact.
    """

    def __init__(self, instrument, x_instrument, wl_hi, device=None, dtype=torch.float64):
        from isofit.core.common import calculate_resample_matrix
        from isofit.core.instrument import wl_tol

        wl_hi = np.asarray(wl_hi, dtype=float)
        wl_init = np.asarray(instrument.wl_init, dtype=float)

        # Mirrors Instrument.sample's identity test exactly, including its
        # one-sided comparison.
        self.identity = bool(
            instrument.calibration_fixed
            and len(wl_init) == len(wl_hi)
            and all((wl_init - wl_hi) < wl_tol)
        )

        self.H = None
        if not self.identity:
            wl, fwhm = instrument.calibration(x_instrument)
            H = calculate_resample_matrix(wl_hi, wl, fwhm)
            self.H = torch.as_tensor(
                np.ascontiguousarray(H), dtype=dtype, device=device
            )
            Logger.warning(
                "The batched initializer is resampling from the RT grid "
                f"({len(wl_hi)} channels) to the instrument grid ({len(wl)} "
                "channels). This path applies the same resample matrix as the "
                "scalar path but as a single matmul, so results agree to "
                "floating-point summation order rather than bit-for-bit."
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Resample ``(B, n_hi)`` spectra to the instrument grid."""
        if self.identity:
            return x
        return x @ self.H.T


class BatchedAlgebraicInitializer:
    """Batched ``invert_algebraic`` + ``fit_params`` + ``clip_bounds``.

    Args:
        fm: A built :class:`isofit.core.forward.ForwardModel`.
        radiance: A :class:`isofit.backends.torch.forward.TorchRadiance` sharing
            this forward model's atmosphere, so the LUT is uploaded once.
        device: Torch device.
        dtype: Torch dtype.

    Raises:
        NotImplementedError: The surface model's reflectance or emission depends
            on the per-pixel geometry, or its ``fit_params`` is not the
            multicomponent implementation reproduced here.
    """

    def __init__(self, fm, radiance, device=None, dtype=torch.float64):
        self.fm = fm
        self.radiance = radiance
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

        surface = fm.surface
        _check_surface_is_geometry_independent(surface)

        # --- constants, evaluated once (inverse_simple.py:159-191) --------------
        x_surface, _, x_instrument = fm.unpack(fm.init.copy())
        self.x_instrument = np.asarray(x_instrument, dtype=float)

        # For the geometry-independent surfaces, geom is genuinely unused and
        # passing None makes any future geometry-reading override fail loudly.
        # A glint surface DOES read it, so its rho_init is per-pixel and is
        # computed inside the batch instead; see _rho_init_for.
        self.x_surface_init = x_surface
        if type(surface).calc_rfl in BATCHED_CALC_RFL:
            rho_init = None
        else:
            _, rho_init = surface.calc_rfl(x_surface, None)
        Ls = surface.calc_Ls(x_surface, None)

        wl_cal, _ = fm.instrument.calibration(x_instrument)

        # Ls is interpolated from the surface grid to the RT grid, then scaled by
        # the per-pixel upward transmittance. Constant, so scipy computes it.
        Ls_rt = interp1d(surface.wl, Ls, fill_value="extrapolate")(fm.atmosphere.wl)

        self.rho_init = None if rho_init is None else self._t(rho_init)
        self.Ls_rt = self._t(Ls_rt)

        self.resample = BatchedResampler(
            fm.instrument,
            x_instrument,
            fm.atmosphere.wl,
            device=self.device,
            dtype=dtype,
        )
        self.to_surface_wl = BatchedInterp1d(
            wl_cal, surface.wl, device=self.device, dtype=dtype
        )

        # --- fit_params bounds (surface_multicomp.py:225-236) -------------------
        idx_lamb = np.asarray(surface.idx_lamb, dtype=int)
        if len(idx_lamb) != len(np.asarray(surface.wl)):
            raise ValueError(
                f"Surface has {len(idx_lamb)} Lambertian state elements but "
                f"{len(surface.wl)} wavelengths; fit_params would raise "
                "'Mismatched reflectances'."
            )
        bounds = np.asarray(surface.bounds, dtype=float)
        self.idx_lamb = torch.as_tensor(idx_lamb, dtype=torch.int64, device=self.device)
        self.n_surface_state = len(surface.statevec_names)
        self.fit_lo = self._t(bounds[idx_lamb, 0] + 0.001)
        self.fit_hi = self._t(bounds[idx_lamb, 1] - 0.001)

        # --- glint constants, hoisted once (surface_glint_model.py:161-218) -----
        self.glint = None
        self.torch_surface = None
        if hasattr(surface, "sun_glint_ind"):
            from isofit.backends.torch.surface import TorchGlintSurface

            self.torch_surface = TorchGlintSurface(
                surface, device=self.device, dtype=self.dtype
            )
            wl = np.asarray(surface.wl)
            # argmin over |target - wl|, exactly as the scalar picks its bands.
            b1, b2 = (int(np.argmin(np.abs(t - wl))) for t in (450, 500))
            g1, g2 = (int(np.argmin(np.abs(t - wl))) for t in (1000, 1020))
            self.glint = {
                "blue": torch.arange(b1, b2, dtype=torch.int64, device=self.device),
                "glint": torch.arange(g1, g2, dtype=torch.int64, device=self.device),
                # bounds_glint_est = [0, 5.0], applied as max(lo+eps, min(hi-eps, x))
                "lo": self._t(np.array(0.0 + eps)),
                "hi": self._t(np.array(5.0 - eps)),
                "ref_band": int(np.argmin(np.abs(wl - 1050))),
                "sky_ind": int(surface.sky_glint_ind),
                "sun_ind": int(surface.sun_glint_ind),
                "sky_init": float(
                    np.asarray(surface.init, dtype=float)[surface.sky_glint_ind]
                ),
            }
            if not len(self.glint["blue"]) or not len(self.glint["glint"]):
                raise NotImplementedError(
                    "the glint estimator's 450-500 nm and 1000-1020 nm band "
                    "ranges are empty for this instrument's wavelengths"
                )

        # --- clip_bounds, with the batch-wide fm.bounds -------------------------
        low, high = self._clip_bounds_edges(fm.bounds, eps)
        self.clip_lo = self._t(low)
        self.clip_hi = self._t(high)

        self.x_instrument_t = self._t(self.x_instrument).reshape(1, -1)
        self.nstate = int(fm.nstate)

        # The scalar path builds x0 by positional concatenation
        # (analytical_line_torch.py:301), so the state vector must be laid out
        # surface | atmosphere | instrument for that to be meaningful.
        expected = np.concatenate(
            [
                np.asarray(fm.idx_surface),
                np.asarray(fm.idx_atmosphere),
                np.asarray(fm.idx_instrument),
            ]
        )
        if not np.array_equal(expected, np.arange(self.nstate)):
            raise NotImplementedError(
                "The batched initializer assumes the state vector is laid out "
                "surface | atmosphere | instrument, matching the positional "
                "concatenation the scalar initializer performs."
            )

    def _t(self, array):
        return torch.as_tensor(
            np.ascontiguousarray(np.asarray(array, dtype=float)),
            dtype=self.dtype,
            device=self.device,
        )

    @staticmethod
    def _clip_bounds_edges(bounds, eps_value):
        """Precompute ``ForwardModel.clip_bounds``' effective low/high vectors."""
        check = bounds[0] != bounds[1]
        low = np.asarray(bounds[0]).astype(float).copy()
        high = np.asarray(bounds[1]).astype(float).copy()
        rel_eps = np.maximum(0, np.minimum(eps_value * np.maximum(bounds[1], 1.0), 1))
        low[check] = low[check] + rel_eps[check]
        high[check] = high[check] - rel_eps[check]
        return low, high

    # --- invert_algebraic -------------------------------------------------------

    def _rho_init_for(self, geom, batch: int) -> torch.Tensor:
        """``calc_rfl(x_surface_init, geom)[1]`` for the batch.

        Constant for the geometry-independent surfaces, so it was hoisted into
        ``__init__`` and is simply broadcast. A glint surface reads the view
        angle, making it genuinely per-pixel; evaluating it at one pixel's
        geometry and reusing the answer is what the hoist guard exists to
        prevent, so it is computed here instead.
        """
        if self.rho_init is not None:
            return self.rho_init.unsqueeze(0)

        x = torch.as_tensor(
            self.x_surface_init, dtype=self.dtype, device=self.device
        ).unsqueeze(0).expand(batch, -1)
        _, rho_dif_dir = self.torch_surface.calc_rfl(x, geom)
        return rho_dif_dir

    def invert_algebraic_batch(self, x_atmosphere: torch.Tensor, meas: torch.Tensor, geom):
        """Algebraic reflectance estimate for a batch of pixels.

        Batched :func:`isofit.inversion.inverse_simple.invert_algebraic`.

        Args:
            x_atmosphere: ``(B, n_atm)`` atmospheric state.
            meas: ``(B, n_chan)`` measured radiance.
            geom: :class:`BatchedGeometry`. Its ``bg_rfl`` plays the same role as
                ``Geometry.bg_rfl``: when present it is the
                hemispherical-hemispherical reflectance, otherwise ``rho_init``
                from ``fm.init`` is used for every pixel.

        Returns:
            ``(rfl_est, coeffs)`` where ``rfl_est`` is ``(B, n_surface_wl)`` and
            ``coeffs`` is ``(L_atm, sphalb, L_tot, transup, L_up)``, matching the
            scalar function's return.
        """
        bg = getattr(geom, "bg_rfl", None)
        rho_dif_dif = self._rho_init_for(geom, meas.shape[0]) if bg is None else bg

        (
            r,
            L_tot,
            _L_dir_dir,
            _L_dif_dir,
            _L_dir_dif,
            _L_dif_dif,
        ) = self.radiance.calc_atmosphere_quantities(
            x_atmosphere, geom, rho_dif_dif=rho_dif_dif
        )

        # The scalar path calls atmosphere.get_L_atm(x_atmosphere, geom), which
        # re-samples the LUT at the point `r` was already taken at. Reusing `r`
        # is the same value for half the interpolation cost.
        L_atm = self.radiance.atmosphere.get_L_atm(r, geom)
        sphalb = r["sphalb"]
        transup = self.radiance.atmosphere.get_upward_transm(r)

        L_up = self.Ls_rt.unsqueeze(0) * transup

        L_atm = self.resample(L_atm)
        L_tot = self.resample(L_tot)
        sphalb = self.resample(sphalb)
        L_up = self.resample(L_up)

        rdn_solrfl = meas - L_up
        denom = rdn_solrfl - L_atm
        rfl = 1.0 / (L_tot / denom + sphalb)

        # The three masks from inverse_simple.py:200-205, in order. Written with
        # torch.where so NaN takes the same branch it does under numpy's boolean
        # mask assignment: `nan > 1.6` is False there too, so NaN survives the
        # 1.6 clamp rather than being clipped.
        zero = torch.zeros((), dtype=rfl.dtype, device=rfl.device)
        rfl = torch.where(denom == 0, zero, rfl)
        rfl = torch.where(L_tot == 0, zero, rfl)
        rfl = torch.where(
            rfl > 1.6, torch.full((), 1.6, dtype=rfl.dtype, device=rfl.device), rfl
        )

        rfl_est = self.to_surface_wl(rfl)
        return rfl_est, (L_atm, sphalb, L_tot, transup, L_up)

    # --- fit_params -------------------------------------------------------------

    def fit_params_batch(self, rfl_meas: torch.Tensor) -> torch.Tensor:
        """Batched ``MultiComponentSurface.fit_params``.

        Returns:
            ``(B, n_surface_state)`` surface state, zero outside ``idx_lamb``.

        Notes:
            The scalar loop is ``max(lo, min(hi, r))`` with Python builtins,
            which return their *first* argument when the comparison is False. So
            ``min(hi, nan)`` is ``hi`` and a NaN reflectance is mapped to the
            upper bound, not propagated. ``torch.clamp``/``torch.minimum`` would
            propagate it instead, so the comparisons are spelled out.
        """
        B = rfl_meas.shape[0]
        # min(hi, r): r if r < hi else hi  ->  NaN maps to hi.
        upper = torch.where(rfl_meas < self.fit_hi, rfl_meas, self.fit_hi)
        # max(lo, upper): upper if upper > lo else lo.
        clipped = torch.where(upper > self.fit_lo, upper, self.fit_lo)

        x_surface = torch.zeros(
            (B, self.n_surface_state), dtype=self.dtype, device=self.device
        )
        x_surface[:, self.idx_lamb] = clipped
        return x_surface

    def fit_params_glint_batch(self, rfl_meas: torch.Tensor, geom) -> torch.Tensor:
        """Batched ``GlintModelSurface.fit_params`` (surface_glint_model.py:161-218).

        The scalar estimates an additive glint magnitude from the smaller of two
        band medians, subtracts it before bounding the reflectance, then converts
        it to a SUN_GLINT magnitude by dividing through the Fresnel factor at a
        reference wavelength. SKY_GLINT is set to its configured init.

        The scalar's two writes -- ``self.init[sun_glint_ind]`` and
        ``geom.sun_glint_init`` -- are deliberately not reproduced. Neither is
        ever read back (``xa`` uses the separate ``sun_glint_mean`` constant),
        and running the scalar over two pixels in both orders gives identical
        results, so there is no cross-pixel state to preserve.
        """
        gl = self.glint
        # torch.median returns the LOWER of the two middle elements for an
        # even-length input; np.median averages them. The scalar uses np.median
        # over band slices whose length depends on the instrument's sampling, so
        # the two disagree on any even-length slice. torch.quantile(0.5)
        # interpolates the same way numpy does.
        est = torch.minimum(
            torch.quantile(rfl_meas[:, gl["blue"]], 0.5, dim=1),
            torch.quantile(rfl_meas[:, gl["glint"]], 0.5, dim=1),
        )
        # max(lo + eps, min(hi - eps, est)), Python-builtin semantics as above.
        upper = torch.where(est < gl["hi"], est, gl["hi"])
        est = torch.where(upper > gl["lo"], upper, gl["lo"])

        x_surface = self.fit_params_batch(rfl_meas - est.unsqueeze(1))

        rho_ls_ref = self.torch_surface.fresnel_rf(geom.observer_zenith)[
            :, gl["ref_band"]
        ]
        x_surface[:, gl["sky_ind"]] = gl["sky_init"]
        x_surface[:, gl["sun_ind"]] = est / rho_ls_ref
        return x_surface

    # --- full initial state -----------------------------------------------------

    def initial_state(self, x_atmosphere: torch.Tensor, meas: torch.Tensor, geom):
        """The batched equivalent of ``TorchWorker._initial_state``.

        Args:
            x_atmosphere: ``(B, n_atm)`` atmospheric state.
            meas: ``(B, n_chan)`` measured radiance.
            geom: :class:`BatchedGeometry`.

        Returns:
            ``(B, n_state)`` clipped initial state.
        """
        rfl_est, _ = self.invert_algebraic_batch(x_atmosphere, meas, geom)
        if self.glint is not None:
            x_surface = self.fit_params_glint_batch(rfl_est, geom)
        else:
            x_surface = self.fit_params_batch(rfl_est)

        # np.concatenate([rfl_est, x_atmosphere, x_instrument])
        x0 = torch.cat(
            [
                x_surface,
                x_atmosphere,
                self.x_instrument_t.expand(meas.shape[0], -1),
            ],
            dim=1,
        )

        # np.clip(x, low, high) == minimum(maximum(x, low), high)
        return torch.minimum(torch.maximum(x0, self.clip_lo), self.clip_hi)
