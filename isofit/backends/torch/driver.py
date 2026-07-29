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
"""Batched driver for the analytical-line retrieval.

Bridges a built :class:`isofit.core.forward.ForwardModel` and a block of pixels
to the batched numerics. This is the piece that decides what runs on the device
and what stays scalar.

What stays scalar, and why
--------------------------
* **Geometry construction.** :class:`isofit.core.geometry.Geometry` derives
  ``coszen``, ``cos_i``, ``relative_azimuth`` and friends from the obs/loc
  arrays with a few dozen flops per pixel. Batching it would duplicate that
  logic for no measurable gain while risking a silent divergence in the LUT
  coordinates, so the scalar objects are built and then stacked.

* **The ``algebraic`` and ``simple`` initializers.** These call back into the
  scalar forward model. They are substantially cheaper than the MAP solve they
  initialize (one LUT sample versus two Cholesky factorizations of a
  ~285x285 matrix), so leaving them scalar caps the achievable speedup without
  dominating it. Batching them is a worthwhile follow-up, not a prerequisite.

Everything downstream of that -- LUT sampling, the radiance terms, ``Seps``,
and the MAP solve itself -- runs batched.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from isofit.backends.torch.analytical import invert_analytical_batch
from isofit.backends.torch.forward import TorchRadiance
from isofit.backends.torch.geometry import BatchedGeometry
from isofit.backends.torch.instrument import TorchInstrument
from isofit.backends.torch.seps import sb_diagonal, seps_batch
from isofit.backends.torch.surface import (
    TorchGlintSurface,
    TorchMultiComponentSurface,
)

Logger = logging.getLogger(__name__)


class AnalyticalBatchSolver:
    """Runs the analytical-line retrieval over batches of pixels.

    Args:
        fm: A built :class:`ForwardModel`.
        winidx: Retrieval-window channel indices.
        device: Torch device.
        dtype: Torch dtype.
        num_iter: MAP iterations per pixel.
        strict_parity: Reproduce the CPU path's diagonal-only whitening of the
            innovation (see
            :func:`isofit.backends.torch.linalg.whiten_innovation`).
        analytic_derivatives: When False (default), build the H2O_ABSCO ``Kb``
            column by finite difference, exactly as
            :meth:`ForwardModel.drdn_datmosphereb` does. When True, build it
            analytically instead (see :meth:`_h2o_absco_column`). The analytic
            column is the *more accurate* of the two, and therefore differs from
            what ISOFIT computes today by the finite difference's O(eps)
            truncation error -- so it is opt-in, like ``strict_parity``.
    """

    def __init__(
        self,
        fm,
        winidx,
        device=None,
        dtype=torch.float64,
        num_iter: int = 1,
        strict_parity: bool = True,
        analytic_derivatives: bool = False,
    ):
        self.fm = fm
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.num_iter = num_iter
        self.strict_parity = strict_parity
        self.analytic_derivatives = analytic_derivatives

        # per_pixel_heuristic_prior swaps ForwardModel.xa for xa_heuristic
        # (forward.py:168-169), which derives the prior mean from each pixel's
        # own state and geometry. The batched solve calls surface.xa directly
        # (analytical.py:163) and never consults fm.xa, so it would quietly use
        # the component-mean prior instead -- a different retrieval, not a
        # slower one. Refuse rather than diverge. Note this is independent of
        # batched_gather: turning that off does not restore the right prior.
        if getattr(
            getattr(fm, "full_config", None), "implementation", None
        ) is not None and getattr(
            fm.full_config.implementation, "per_pixel_heuristic_prior", False
        ):
            raise NotImplementedError(
                "per_pixel_heuristic_prior is not supported by the batched "
                "backend: the solve uses the surface's component-mean prior and "
                "would silently ignore the per-pixel heuristic. Run with "
                "backend='numpy'."
            )

        self.radiance = TorchRadiance(fm, device=self.device, dtype=dtype)
        # A glint surface carries two extra state elements and needs the
        # subclass that knows how to build their linearization columns.
        self.is_glint = hasattr(fm.surface, "sun_glint_ind")
        surface_cls = TorchGlintSurface if self.is_glint else TorchMultiComponentSurface
        self.surface = surface_cls(fm.surface, device=self.device, dtype=dtype)
        self.instrument = TorchInstrument(
            fm.instrument, device=self.device, dtype=dtype
        )

        self.winidx = torch.as_tensor(
            np.asarray(winidx), dtype=torch.int64, device=self.device
        )
        self.idx_surface = torch.as_tensor(
            np.asarray(fm.idx_surface), dtype=torch.int64, device=self.device
        )

        # Surface channels outside the retrieval windows are filled with a
        # constant rather than solved (inverse_simple.py:289-292).
        full_idx = np.concatenate(
            (np.asarray(winidx), np.asarray(fm.idx_surf_nonrfl))
        ).astype(int)
        outside = np.ones(len(fm.idx_surface), dtype=bool)
        outside[full_idx] = False
        self.outside_ret_windows = torch.as_tensor(
            np.where(outside)[0], dtype=torch.int64, device=self.device
        )

        self.model_discrepancy = None
        if getattr(fm, "model_discrepancy", None) is not None:
            self.model_discrepancy = torch.as_tensor(
                np.asarray(fm.model_discrepancy), dtype=dtype, device=self.device
            )

        self.n_chan = int(fm.instrument.n_chan)

    def _t(self, array):
        return torch.as_tensor(
            np.ascontiguousarray(array), dtype=self.dtype, device=self.device
        )

    def solve(self, meas, geoms, x_atmosphere, sub_state, x0):
        """Retrieve surface reflectance for a batch of pixels.

        Args:
            meas: ``(B, n_chan)`` measured radiance.
            geoms: Sequence of ``B`` scalar :class:`Geometry` objects, or an
                already-built :class:`BatchedGeometry` for the same pixels.
            x_atmosphere: ``(B, n_atm)`` interpolated atmospheric state.
            sub_state: ``(B, n_state)`` superpixel state, supplying the
                background reflectance.
            x0: ``(B, n_state)`` initial state.

        Returns:
            ``(state, uncertainty)`` as numpy arrays of shape ``(B, n_state)``.
        """
        fm = self.fm
        meas_t = self._t(meas)
        x0_t = self._t(x0)
        sub_t = self._t(sub_state)
        B = meas_t.shape[0]

        # Take the atmospheric state from the CLIPPED x0, mirroring the scalar
        # path's fm.unpack(x) (inverse_simple.py:250). Using the raw interpolated
        # state would sample the LUT at a different point for any pixel whose
        # atmosphere fell outside the model bounds.
        x_atm_t = x0_t.index_select(
            1,
            torch.as_tensor(
                np.asarray(fm.idx_atmosphere), dtype=torch.int64, device=self.device
            ),
        )

        # Background reflectance comes from the superpixel surface state, or the
        # explicit background image when one is supplied.
        if isinstance(geoms, BatchedGeometry):
            geom = geoms
            if geom.bg_rfl is None:
                geom.bg_rfl = self._default_bg_rfl(fm, sub_t, geom)
        else:
            bg = np.stack(
                [
                    g.bg_rfl
                    if isinstance(getattr(g, "bg_rfl", None), np.ndarray)
                    else sub_state[i][fm.idx_surf_rfl]
                    for i, g in enumerate(geoms)
                ]
            )
            geom = BatchedGeometry.from_geometries(
                geoms, device=self.device, dtype=self.dtype
            )
            geom.bg_rfl = self._t(bg)
            if self.is_glint:
                # The per-geometry fallback above used the raw superpixel
                # reflectance, which under glint is missing the sky term.
                supplied = np.array(
                    [isinstance(getattr(g, "bg_rfl", None), np.ndarray) for g in geoms]
                )
                if not supplied.all():
                    default = self._default_bg_rfl(fm, sub_t, geom)
                    keep = torch.as_tensor(
                        supplied, dtype=torch.bool, device=self.device
                    ).unsqueeze(1)
                    geom.bg_rfl = torch.where(keep, geom.bg_rfl, default)

        rho_dif_dif = geom.bg_rfl

        # Atmosphere and radiance terms.
        (
            r,
            L_tot,
            L_dir_dir,
            L_dif_dir,
            L_dir_dif,
            L_dif_dif,
        ) = self.radiance.calc_atmosphere_quantities(
            x_atm_t, geom, rho_dif_dif=rho_dif_dif
        )
        L_atm = self.radiance.atmosphere.get_L_atm(r, geom)
        s_alb = r["sphalb"]

        L_bg = self.radiance.calc_rdn_bgrfl(
            rho_dir_dif=rho_dif_dif,
            rho_dif_dif=rho_dif_dif,
            L_dir_dif=L_dir_dif,
            L_dif_dif=L_dif_dif,
            L_tot=L_tot,
            s_alb=s_alb,
        )

        # Surface linearization: the diagonal of H.
        theta = self.surface.evaluate_theta(
            s_alb,
            geom,
            L_tot,
            L_dir_dir=L_dir_dir,
            L_dir_dif=L_dir_dif,
            L_dif_dir=L_dif_dir,
            L_dif_dif=L_dif_dif,
            heterogeneous=self.radiance.use_background_rfl,
        )

        # Superpixel EOF shift. Hardcoding zeros here silently drops the term
        # for any instrument configured with EOFs, and it is subtracted straight
        # out of the innovation, so the error would land directly in reflectance.
        eof_offset = self._t(
            np.stack([
                fm.eof_offset(sub_state[i][fm.idx_instrument])
                for i in range(B)
            ])
        )

        # Modeled radiance at the initial state, needed for the Kb columns that
        # describe uncertainty from unretrieved unknowns.
        # ForwardModel.Kb takes BOTH reflectance quantities from calc_rfl
        # (forward.py:765-768). They coincide for a Lambertian surface; under
        # glint the direct path carries SUN_GLINT and the diffuse SKY_GLINT.
        x_surface_t = x0_t.index_select(1, self.idx_surface)
        rho_dir_dir, rho_dif_dir = self.surface.calc_rfl(x_surface_t, geom)
        rho = rho_dir_dir
        Ls = torch.zeros_like(rho)
        rdn = self.radiance.calc_rdn(
            rho, rho_dif_dir, rho_dif_dif, rho_dif_dif, Ls,
            L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif, r, geom,
        )

        def rebuild_seps(x_surface_now):
            """Seps at the current surface state, as the scalar recomputes it.

            ``invert_analytical`` evaluates ``fm.Seps(x, meas, geom)`` INSIDE its
            iteration loop (inverse_simple.py:313), and Seps genuinely depends on
            the surface state: the radiometric ``Kb`` block is
            ``diagflat(rdn(x))`` (instrument.py:354-360), so its diagonal carries
            ``rdn_modeled(x)**2 * sb_radiometric``. Freezing it at x0 makes the
            two paths solve different problems from the second iteration onward
            -- measured at ~5e-02 absolute reflectance on a real scene, against a
            parity budget of 1e-08.

            Only the reflectance quantities and the modeled radiance change: the
            atmosphere state is untouched by this solve, so ``r`` and the L_*
            terms are loop-invariant and stay hoisted.
            """
            rho_dd, rho_fd = self.surface.calc_rfl(x_surface_now, geom)
            rdn_now = self.radiance.calc_rdn(
                rho_dd, rho_fd, rho_dif_dif, rho_dif_dif, Ls,
                L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif, r, geom,
            )
            return self._build_seps(
                meas_t, x_atm_t, geom, rho_dd, rho_fd, rho_dif_dif, Ls, rdn_now
            )

        Seps = self._build_seps(
            meas_t,
            x_atm_t,
            geom,
            rho,
            rho_dif_dir,
            rho_dif_dif,
            Ls,
            rdn,
        )

        extra_columns = None
        if self.is_glint:
            extra_columns = self.surface.extra_columns(geom, L_dir_dir, L_dif_dir)

        traj, unc = invert_analytical_batch(
            self.surface,
            self.winidx,
            meas_t,
            x0_t,
            theta,
            Seps,
            L_atm,
            L_bg,
            eof_offset,
            geom=geom,
            idx_surface=self.idx_surface,
            extra_columns=extra_columns,
            seps_fn=rebuild_seps,
            outside_ret_windows=self.outside_ret_windows,
            num_iter=self.num_iter,
            strict_parity=self.strict_parity,
        )
        return traj[:, -1, :].cpu().numpy(), unc.cpu().numpy()

    def _default_bg_rfl(self, fm, sub_t, geom):
        """Background reflectance when no background image was supplied.

        ``invert_analytical`` defaults it to ``calc_rfl(sub_state, geom)[1]``
        -- the *diffuse* reflectance quantity (inverse_simple.py:256-264), not
        the raw reflectance state. For a Lambertian multicomponent surface
        those are the same spectrum, which is why reading the state directly
        worked; under glint the diffuse quantity carries the sky-glint term and
        they differ.
        """
        sub_surface = sub_t.index_select(1, self.idx_surface)
        _, rho_dif_dir = self.surface.calc_rfl(sub_surface, geom)
        return rho_dif_dir

    def _h2o_absco_column(self, x_atm, geom, rho, rho_dif_dir, rho_dif_dif, Ls, rdn):
        """Kb column for the H2O_ABSCO unknown.

        Dispatches to the finite difference (default, bit-comparable with the
        CPU path) or to the analytic derivative when ``analytic_derivatives`` is
        set.

        Returns:
            ``(B, n_chan)`` derivative of radiance with respect to H2O_ABSCO.
        """
        if self.analytic_derivatives:
            return self._h2o_absco_column_analytic(
                x_atm, geom, rho, rho_dif_dir, rho_dif_dif, Ls
            )
        return self._h2o_absco_column_fd(
            x_atm, geom, rho, rho_dif_dir, rho_dif_dif, Ls, rdn
        )

    def _h2o_absco_column_fd(self, x_atm, geom, rho, rho_dif_dir, rho_dif_dif, Ls, rdn):
        """Finite-difference Kb column for the H2O_ABSCO unknown.

        Mirrors :meth:`ForwardModel.drdn_datmosphereb`: perturb the water-vapour
        state multiplicatively by ``1 + eps`` and difference the resulting
        radiance. One extra batched forward evaluation for the whole batch,
        versus one per pixel in the scalar path.

        Returns:
            ``(B, n_chan)`` derivative of radiance with respect to H2O_ABSCO.
        """
        from isofit.core.common import eps

        names = list(self.fm.atmosphere.statevec_names)
        i = names.index("H2OSTR")

        x_perturb = x_atm.clone()
        x_perturb[:, i] = x_perturb[:, i] * (1.0 + eps)

        (
            r_p,
            L_tot_p,
            L_dir_dir_p,
            L_dif_dir_p,
            L_dir_dif_p,
            L_dif_dif_p,
        ) = self.radiance.calc_atmosphere_quantities(
            x_perturb, geom, rho_dif_dif=rho_dif_dif
        )
        rdne = self.radiance.calc_rdn(
            rho, rho_dif_dir, rho_dif_dif, rho_dif_dif, Ls,
            L_tot_p, L_dir_dir_p, L_dif_dir_p, L_dir_dif_p, L_dif_dif_p, r_p, geom,
        )
        return (rdne - rdn) / eps

    def _h2o_absco_column_analytic(self, x_atm, geom, rho, rho_dif_dir, rho_dif_dif, Ls):
        """Analytic Kb column for the H2O_ABSCO unknown.

        The chain has two links, and they are handled by different machinery
        because they are different kinds of function:

        1. **State to LUT quantities.** ``rdn`` depends on H2OSTR *only* through
           the interpolated LUT quantities ``r``. ``build_points`` writes the
           state element straight into one LUT axis, so
           ``d r_k / d H2OSTR`` is that axis's multilinear derivative, taken
           analytically by
           :meth:`~isofit.backends.torch.lut.BatchedLUT.interpolate_with_gradients`.
           This is the link the finite difference was approximating, and the
           only one that is not already differentiable code.

        2. **LUT quantities to radiance.** ``calc_atmosphere_quantities`` and
           ``calc_rdn`` are ordinary closed-form arithmetic on ``r`` (path
           radiance, the four coupled transmittance terms, the
           ``1 / (1 - sphalb * rho)`` surface-atmosphere coupling, the eq. 11
           divisor, the upward transmittance). Rather than transcribe those
           partials by hand -- roughly sixty lines of algebra that would silently
           go stale the next time ``calc_rdn`` changes -- this pushes the
           tangent through *the same functions the forward model runs*, using
           forward-mode AD. That is not a finite difference and introduces no
           truncation error; it evaluates the exact directional derivative of
           the code that computes ``rdn``.

        Scaling matters as much as the derivative itself.
        :meth:`ForwardModel.drdn_datmosphereb` perturbs *multiplicatively*
        (``x * (1 + eps)``) and divides by ``eps``, so the column it produces is
        ``x_H2O * d rdn / d x_H2O`` -- a relative derivative, not
        ``d rdn / d x_H2O``. Folding ``x_H2O`` into the tangent below reproduces
        that convention, so this column differs from the finite-difference one
        only by the latter's O(eps) truncation error.

        Returns:
            ``(B, n_chan)`` derivative of radiance with respect to H2O_ABSCO.
        """
        atm = self.radiance.atmosphere

        names = list(self.fm.atmosphere.statevec_names)
        i = names.index("H2OSTR")

        # LUT axis carrying H2OSTR. build_points assigns x_RT[:, i] directly to
        # column idx_x_RT[i], so d(coordinate)/d(H2OSTR) is exactly 1. The
        # observer-zenith sign flip cannot apply to this column: it only touches
        # geometry columns, and idx_x_RT is their complement.
        dim = atm.idx_x_RT[i]

        points = atm.build_points(x_atm, geom)
        r, grads = atm.lut.interpolate_with_gradients(points, dims=[dim])

        # Constants have no state dependence, so they are closed over rather than
        # differentiated; interpolate_with_gradients omits them from `grads` for
        # the same reason.
        primals = {key: value for key, value in r.items() if torch.is_tensor(value)}
        scale = x_atm[:, i].unsqueeze(1).to(self.dtype)
        tangents = {key: grads[key][:, 0, :] * scale for key in primals}

        def rdn_of_r(sampled):
            full = dict(r)
            full.update(sampled)
            (
                _,
                L_tot,
                L_dir_dir,
                L_dif_dir,
                L_dir_dif,
                L_dif_dif,
            ) = self.radiance.calc_atmosphere_quantities(
                x_atm, geom, rho_dif_dif=rho_dif_dif, r=full
            )
            return self.radiance.calc_rdn(
                rho, rho_dif_dir, rho_dif_dif, rho_dif_dif, Ls,
                L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif, full, geom,
            )

        _, column = torch.func.jvp(rdn_of_r, (primals,), (tangents,))
        return column

    def _build_seps(self, meas_t, x_atm, geom, rho, rho_dif_dir, rho_dif_dif, Ls, rdn):
        # NOTE: Sy depends on the MEASUREMENT; the radiometric Kb block depends
        # on the MODELED radiance. They are different arrays -- see seps_batch.
        """Assemble the windowed observation-error covariance for the batch.

        The radiometric ``Kb`` block is ``diagflat(meas)`` and so contributes
        only a diagonal. Genuinely dense columns become rank-1 updates; the
        H2O_ABSCO one is built here. Any remaining dense column is rejected
        rather than dropped -- omitting a term of ``Seps`` would understate the
        observation error and quietly bias the retrieval, which is far harder to
        notice than a refusal.
        """
        fm = self.fm

        atm_bvec = list(fm.atmosphere.bvec)
        instr_extra = list(fm.instrument.bvec)[self.n_chan :]

        dense_cols = None
        sb_dense = None

        handled = []
        if atm_bvec:
            if atm_bvec == ["H2O_ABSCO"] and "H2OSTR" in fm.atmosphere.statevec_names:
                sb_h2o = float(np.asarray(fm.atmosphere.bval, dtype=float)[0])
                if sb_h2o == 0.0:
                    # The column enters Seps only as sb * k k^T, so a zero
                    # magnitude discards it entirely. Building it would mean an
                    # extra forward evaluation (or a jvp) per batch whose result
                    # is multiplied by zero. apply_oe emits {"H2O_ABSCO": 0.0}
                    # by default (template_construction.py:1556), so this is the
                    # common case, not an edge case.
                    Logger.debug(
                        "H2O_ABSCO uncertainty is 0.0; skipping its Kb column"
                    )
                    handled = ["H2O_ABSCO (zero magnitude, skipped)"]
                else:
                    col = self._h2o_absco_column(
                        x_atm, geom, rho, rho_dif_dir, rho_dif_dif, Ls, rdn
                    )
                    dense_cols = col.unsqueeze(-1)
                    sb_dense = torch.as_tensor(
                        [sb_h2o**2], dtype=self.dtype, device=self.device
                    )
                    handled = ["H2O_ABSCO"]
            else:
                raise NotImplementedError(
                    f"Atmosphere unknowns {atm_bvec} are not supported by the "
                    "batched Seps (only H2O_ABSCO is). Run with backend='numpy'."
                )

        if instr_extra:
            raise NotImplementedError(
                f"Instrument unknowns {instr_extra} are not supported by the "
                "batched Seps. Run with backend='numpy'."
            )

        if handled:
            Logger.debug(f"Batched Seps includes dense Kb column(s): {handled}")

        sb = sb_diagonal(fm.instrument, meas_t, dtype=self.dtype, device=self.device)

        if self.instrument.sy_is_diagonal:
            sy_diag = self.instrument.Sy_diagonal(meas_t)
            sy_full = None
        else:
            sy_diag = None
            sy_full = self.instrument.Sy_shared

        return seps_batch(
            sy_diag,
            rdn,
            sb[: self.n_chan],
            dense_columns=dense_cols,
            sb_dense=sb_dense,
            model_discrepancy=self.model_discrepancy,
            winidx=self.winidx,
            sy_full=sy_full,
        )
