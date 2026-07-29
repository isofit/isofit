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
"""Batched worker for the analytical-line retrieval.

Mirrors :class:`isofit.utils.analytical_line.Worker` -- same constructor
arguments, same memmap inputs, same BIL chunk writes -- but replaces the
per-pixel loop with a gather into batches, a batched solve, and a scatter back.

The IO boundary is deliberately identical to the scalar worker so the two paths
produce byte-comparable outputs and can be swapped without touching the driver.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import numpy as np
from spectral.io import envi

from isofit import ray
from isofit.core.backend import resolve_batch_size, resolve_device, resolve_dtype
from isofit.core.common import envi_header, eps, load_esd
from isofit.core.fileio import write_bil_chunk
from isofit.core.geometry import Geometry
from isofit.core.multistate import fill_statevector
from isofit.inversion.inverse_simple import invert_algebraic, invert_simple
from isofit.utils.analytical_line import retrieve_winidx

Logger = logging.getLogger(__name__)


def read_output_block(path: str, start_line: int, stop_line: int) -> np.ndarray:
    """Read an existing output block so untouched pixels survive.

    The scalar worker seeds each block from what is already on disk
    (``analytical_line.py:558-585``) rather than from zeros, and that is
    load-bearing for multistate runs: ``analytical_line.py:315`` builds a fresh
    worker pool per surface class, and every pool is handed line breaks spanning
    the whole image (``analytical_line.py:389``). A block therefore covers pixels
    belonging to other surface classes, which this class must leave alone.

    Starting from ``np.zeros`` instead means the last surface class to write a
    block erases every earlier class's pixels in it. Single-surface runs -- the
    ``apply_oe`` default -- have one pool and cannot show the bug, which is why
    it survived the 100,000-pixel validation.
    """
    return (
        envi.open(envi_header(path))
        .open_memmap(interleave="bip", writable=False)[start_line:stop_line, ...]
        .copy()
    )


@ray.remote(num_cpus=1)
class TorchWorker:
    """Runs the analytical-line retrieval over line blocks, batched on a device."""

    def __init__(
        self,
        config,
        fm,
        surface_class_str,
        class_idx_pairs,
        full_statevector,
        full_idx_surface,
        full_idx_surf_rfl,
        full_idx_atmosphere,
        rdn_file,
        loc_file,
        obs_file,
        atm_file,
        subs_state_file,
        lbl_file,
        rfl_output,
        unc_output,
        non_rfl_output,
        non_rfl_unc_output,
        num_iter,
        loglevel,
        logfile,
        initializer,
        skyview_factor_file,
        bgrfl_file,
        torch_device="auto",
        torch_batch_size="auto",
        torch_dtype="auto",
        batched_gather=None,
    ):
        logging.basicConfig(
            format="%(levelname)s:%(asctime)s ||| %(message)s",
            level=loglevel,
            filename=logfile,
            datefmt="%Y-%m-%d,%H:%M:%S",
        )

        self.config = config
        self.fm = fm
        self.surface_class_str = surface_class_str
        self.class_idx_pairs = class_idx_pairs
        self.esd = load_esd()

        self.full_statevector = full_statevector
        self.full_idx_surface = full_idx_surface
        self.full_idx_surf_rfl = full_idx_surf_rfl
        self.full_idx_atmosphere = full_idx_atmosphere
        self.n_rfl_bands = len(full_idx_surf_rfl)
        self.n_non_rfl_bands = len(full_idx_surface) - len(full_idx_surf_rfl)

        self.winidx = retrieve_winidx(self.config)

        self.rdn = envi.open(envi_header(rdn_file)).open_memmap(interleave="bip")
        self.loc = envi.open(envi_header(loc_file)).open_memmap(interleave="bip")
        self.obs = envi.open(envi_header(obs_file)).open_memmap(interleave="bip")
        self.rt_state = envi.open(envi_header(atm_file)).open_memmap(interleave="bip")
        self.subs_state = envi.open(envi_header(subs_state_file)).open_memmap(
            interleave="bip"
        )
        self.lbl = envi.open(envi_header(lbl_file)).open_memmap(interleave="bip")

        self.svf = (
            envi.open(envi_header(skyview_factor_file)).open_memmap(interleave="bip")
            if skyview_factor_file
            else []
        )
        self.bg_rfl = (
            envi.open(envi_header(bgrfl_file)).open_memmap(interleave="bip")
            if bgrfl_file
            else []
        )

        self.n_lines = self.rdn.shape[0]
        self.n_samples = self.rdn.shape[1]

        self.rfl_outpath = rfl_output
        self.unc_outpath = unc_output
        self.non_rfl_outpath = non_rfl_output
        self.non_rfl_unc_outpath = non_rfl_unc_output

        self.completed_spectra = 0
        self.hash_table = OrderedDict()
        self.hash_size = config.implementation.max_hash_table_size
        self.subs_state_file = subs_state_file
        self.lbl_file = lbl_file
        self.atm_bands = []
        self.num_iter = num_iter
        self.coszen = fm.atmosphere.coszen
        self.initializer = initializer
        self.per_pixel_heuristic_prior = (
            config.implementation.per_pixel_heuristic_prior
        )
        self.radiance_correction = None

        # Device setup, once per actor. The LUT and prior tensors are uploaded
        # here and reused for every block this worker handles.
        import torch

        self.device = resolve_device(torch_device, allow_cpu_fallback=True)
        self.dtype = resolve_dtype(torch_dtype, self.device)

        from isofit.backends.torch.driver import AnalyticalBatchSolver

        self.solver = AnalyticalBatchSolver(
            fm,
            self.winidx,
            device=self.device,
            dtype=self.dtype,
            num_iter=num_iter,
        )

        # Peak device memory per pixel, fp64. Two Cholesky stages (nw x nw for
        # Seps, ns x ns for the posterior) plus the carried inverses and the
        # symmetrization buffers.
        #
        # This formula is a deliberate UPPER BOUND, not a derivation. Measured
        # peak via torch.cuda.max_memory_allocated on an A100 at nw=285/ns=425
        # is 6.78 MiB/pixel (stable to 0.2% from B=128 to B=1024); this yields
        # 7.37. Erring high means "auto" under-commits slightly rather than
        # exhausting VRAM mid-batch, and it stays valid if an intermediate
        # stops being computed in place. Re-measure if the solve changes:
        # deriving this from tensor bookkeeping has been wrong in both
        # directions already.
        nw = len(self.winidx)
        ns = len(fm.idx_surface)
        bytes_per_pixel = 8 * (4 * ns * ns + 3 * nw * nw)
        self.batch_size = resolve_batch_size(
            torch_batch_size, bytes_per_pixel, self.device, default=512
        )
        Logger.info(
            f"TorchWorker on {self.device}: batch={self.batch_size}, "
            f"{bytes_per_pixel / 2**20:.1f} MiB/pixel"
        )

        # Batched gather: build every pixel's Geometry and algebraic initial
        # guess for a whole block at once instead of one at a time.
        #
        # ON by default where supported. Validated end-to-end on a
        # 100,000-pixel AVIRIS-NG scene: 3.01x faster than the per-pixel gather
        # (173.9s -> 57.7s) with BIT-IDENTICAL output, and unchanged 5.96e-08
        # agreement against the numpy path. It is the same arithmetic in the
        # same order, just vectorized.
        #
        # Configurations it cannot reproduce exactly fall back to the per-pixel
        # gather rather than approximating: the `simple` initializer calls back
        # into the scalar forward model per pixel, and per_pixel_heuristic_prior
        # derives bounds and prior means that live on the scalar Geometry.
        # Resolution order: explicit argument, then ISOFIT_TORCH_BATCHED_GATHER,
        # then on-where-supported. Configurations it cannot reproduce exactly
        # fall back to the per-pixel gather; an EXPLICIT request for one of them
        # raises instead, so a deliberate choice is never silently downgraded.
        # The surface itself can rule out the batched initializer -- a glint
        # surface's calc_rfl depends on the per-pixel view angle and its
        # fit_params mutates shared state. Probe rather than construct: the
        # constructor raises, which is right for an explicit request but would
        # turn a defaulted-on gather into a crash.
        from isofit.backends.torch.initializer import surface_is_batchable

        surface_ok, surface_reason = (
            surface_is_batchable(fm.surface)
            if self.initializer == "algebraic"
            else (True, "")
        )
        supported = (
            self.initializer in ("algebraic", "superpixel")
            and not self.per_pixel_heuristic_prior
            and surface_ok
        )
        env = os.environ.get("ISOFIT_TORCH_BATCHED_GATHER")
        explicit = batched_gather is not None or env is not None
        if batched_gather is not None:
            requested = bool(batched_gather)
        elif env is not None:
            requested = env == "1"
        else:
            requested = supported

        if requested and not supported:
            if self.initializer not in ("algebraic", "superpixel"):
                reason = (
                    f"initializer={self.initializer!r} is not 'algebraic' or "
                    "'superpixel'"
                )
            elif self.per_pixel_heuristic_prior:
                reason = (
                    "per_pixel_heuristic_prior derives per-pixel bounds and "
                    "prior means that live on the scalar Geometry"
                )
            else:
                reason = surface_reason
            if explicit:
                raise NotImplementedError(
                    f"batched_gather was requested but is unsupported: {reason}."
                )
            Logger.info(f"Batched gather unavailable ({reason}); using per-pixel")
            requested = False

        self.batched_gather = requested
        self.initializer_batch = None
        if self.batched_gather:
            Logger.info(
                "TorchWorker using batched gather: batched Geometry "
                "construction and batched algebraic initializer"
            )
            if self.initializer == "algebraic":
                from isofit.backends.torch.initializer import (
                    BatchedAlgebraicInitializer,
                )

                self.initializer_batch = BatchedAlgebraicInitializer(
                    fm,
                    self.solver.radiance,
                    device=self.device,
                    dtype=self.dtype,
                )
            Logger.info("TorchWorker: batched geometry + initializer enabled")

    def run_chunks(self, line_break) -> None:
        """Retrieve every pixel in a block of lines."""
        start_line, stop_line = line_break

        # Seed from disk, not zeros -- see read_output_block. Other surface
        # classes' pixels live in this block and must survive it.
        output_rfl = read_output_block(self.rfl_outpath, start_line, stop_line)
        output_rfl_unc = read_output_block(self.unc_outpath, start_line, stop_line)

        # Non-reflectance surface state (the glint model's SKY_GLINT and
        # SUN_GLINT). analytical_line only creates these cubes when the surface
        # actually has such states, so the paths are None otherwise.
        output_non_rfl = output_non_rfl_unc = None
        if self.non_rfl_outpath:
            output_non_rfl = read_output_block(
                self.non_rfl_outpath, start_line, stop_line
            )
            output_non_rfl_unc = read_output_block(
                self.non_rfl_unc_outpath, start_line, stop_line
            )

        index_pairs = self.class_idx_pairs[
            np.where(
                (self.class_idx_pairs[:, 0] >= start_line)
                & (self.class_idx_pairs[:, 0] < stop_line)
            )
        ]

        if self.batched_gather:
            gathered = self._gather_batched(index_pairs)
        else:
            gathered = self._gather_per_pixel(index_pairs)

        if gathered is None:
            self._write(
                start_line, stop_line, output_rfl, output_rfl_unc,
                output_non_rfl, output_non_rfl_unc,
            )
            return

        rows, cols, meas_all, geoms, atm_all, sub_all, x0_all = gathered

        # Solve in device-sized sub-batches.
        for lo in range(0, len(rows), self.batch_size):
            hi = min(lo + self.batch_size, len(rows))
            state, unc = self.solver.solve(
                meas_all[lo:hi],
                geoms[lo:hi],
                atm_all[lo:hi],
                sub_all[lo:hi],
                x0_all[lo:hi],
            )

            # fill_statevector allocates and fancy-indexes per call, and
            # full_idx/full_miss are constant across the batch, so do the scatter
            # once for the whole sub-batch instead of twice per pixel.
            n_full = len(self.full_statevector)
            idx, miss = self.fm.full_idx, self.fm.full_miss
            full_state = np.full((hi - lo, n_full), -9999.0)
            full_unc = np.full((hi - lo, n_full), -9999.0)
            full_state[:, idx] = state
            full_unc[:, idx] = unc
            if len(miss):
                full_state[:, miss] = -9999.0
                full_unc[:, miss] = -9999.0

            rfl_block = full_state[:, self.full_idx_surf_rfl]
            unc_block = full_unc[:, self.full_idx_surf_rfl]
            if output_non_rfl is not None:
                sl = slice(
                    self.n_rfl_bands, self.n_rfl_bands + self.n_non_rfl_bands
                )
                non_rfl_block = full_state[:, sl]
                non_rfl_unc_block = full_unc[:, sl]
            for k in range(hi - lo):
                r, c = rows[lo + k], cols[lo + k]
                output_rfl[r - start_line, c, :] = rfl_block[k]
                output_rfl_unc[r - start_line, c, :] = unc_block[k]
                if output_non_rfl is not None:
                    output_non_rfl[r - start_line, c, :] = non_rfl_block[k]
                    output_non_rfl_unc[r - start_line, c, :] = non_rfl_unc_block[k]

        self.completed_spectra += len(rows)
        Logger.info(
            f"Analytical line writing lines: {start_line} to {stop_line}. "
            f"Surface: {self.surface_class_str}"
        )
        self._write(
            start_line, stop_line, output_rfl, output_rfl_unc,
            output_non_rfl, output_non_rfl_unc,
        )

    def _gather_per_pixel(self, index_pairs):
        """Gather a block one pixel at a time (the default).

        Geometry construction and the initializer stay scalar here, calling the
        same reference functions the numpy backend does.

        Returns:
            ``(rows, cols, meas, geoms, x_atmosphere, sub_state, x0)`` or
            ``None`` when no pixel in the block is valid.
        """
        rows, cols, meas_list, geoms, atm_list, sub_list = [], [], [], [], [], []
        for r, c, *_ in index_pairs:
            meas = self.rdn[r, c, :]
            if self.radiance_correction is not None:
                meas = meas.copy() * self.radiance_correction
            if np.all(meas < 0):
                continue

            geom = Geometry(
                obs=self.obs[r, c, :],
                loc=self.loc[r, c, :],
                svf=self.svf[r, c] if len(self.svf) else 1,
                bg_rfl=self.bg_rfl[r, c, :] if len(self.bg_rfl) else None,
                coszen=self.coszen,
                full_config=self.config,
            )

            x_atmosphere = self.rt_state[r, c, :]
            iv_idx = self.fm.surface.analytical_iv_idx

            lbl_idx = int(self.lbl[r, c, 0])
            sub_state = np.zeros(self.fm.nstate)
            sub_state[self.fm.idx_surface] = self.subs_state[lbl_idx, 0, iv_idx]
            sub_state[self.fm.idx_atmosphere] = x_atmosphere
            sub_state[self.fm.idx_instrument] = self.subs_state[
                lbl_idx, 0, self.fm.idx_instrument
            ]
            sub_state[np.isnan(sub_state)] = self.fm.init[np.isnan(sub_state)]

            rows.append(r)
            cols.append(c)
            meas_list.append(meas)
            geoms.append(geom)
            atm_list.append(x_atmosphere)
            sub_list.append(sub_state)

        if not rows:
            return None

        meas_all = np.stack(meas_list)
        atm_all = np.stack(atm_list)
        sub_all = np.stack(sub_list)
        x0_all = np.stack(
            [
                self._initial_state(meas_list[i], geoms[i], atm_list[i], sub_list[i])
                for i in range(len(rows))
            ]
        )
        return rows, cols, meas_all, geoms, atm_all, sub_all, x0_all

    def _gather_batched(self, index_pairs):
        """Gather a whole block at once (opt-in, ``batched_gather=True``).

        Replaces the per-pixel loop's two scalar hot spots -- ``Geometry(...)``
        and the algebraic initializer -- with slab reads and their batched
        equivalents. The memmap reads become one fancy-index per input cube
        instead of one per pixel per cube.

        Returns:
            The same tuple as :meth:`_gather_per_pixel`, except that the geometry
            element is a single :class:`BatchedGeometry` rather than a list.
        """
        from isofit.backends.torch.geometry import build_batched_geometry

        if len(index_pairs) == 0:
            return None

        rows = np.asarray(index_pairs[:, 0], dtype=int)
        cols = np.asarray(index_pairs[:, 1], dtype=int)

        meas_all = np.asarray(self.rdn[rows, cols, :])
        if self.radiance_correction is not None:
            meas_all = meas_all * self.radiance_correction

        # Same rejection rule as the scalar loop: drop pixels whose every
        # channel is negative (the ENVI fill convention).
        keep = ~np.all(meas_all < 0, axis=1)
        if not keep.any():
            return None
        rows, cols, meas_all = rows[keep], cols[keep], meas_all[keep]

        obs_all = np.asarray(self.obs[rows, cols, :])
        loc_all = np.asarray(self.loc[rows, cols, :])
        svf_all = np.asarray(self.svf[rows, cols]) if len(self.svf) else None
        bg_all = np.asarray(self.bg_rfl[rows, cols, :]) if len(self.bg_rfl) else None

        geom = build_batched_geometry(
            obs=obs_all,
            loc=loc_all,
            bg_rfl=bg_all,
            svf=svf_all,
            coszen=self.coszen,
            full_config=self.config,
            device=self.device,
            dtype=self.dtype,
        )

        atm_all = np.asarray(self.rt_state[rows, cols, :])

        iv_idx = self.fm.surface.analytical_iv_idx
        lbl_idx = np.asarray(self.lbl[rows, cols, 0]).astype(int)
        subs_rows = np.asarray(self.subs_state[lbl_idx, 0, :])

        sub_all = np.zeros((len(rows), self.fm.nstate))
        sub_all[:, self.fm.idx_surface] = subs_rows[:, iv_idx]
        sub_all[:, self.fm.idx_atmosphere] = atm_all
        sub_all[:, self.fm.idx_instrument] = subs_rows[:, self.fm.idx_instrument]
        nan = np.isnan(sub_all)
        sub_all = np.where(nan, np.broadcast_to(self.fm.init, sub_all.shape), sub_all)

        if self.initializer == "superpixel":
            x0_all = sub_all.copy()
            x0_all[:, self.fm.idx_atmosphere] = atm_all
            # clip_bounds broadcasts its (n_state,) edges over the batch axis.
            x0_all = self.fm.clip_bounds(x0_all, self.fm.bounds, eps=eps)
        else:
            import torch

            meas_t = torch.as_tensor(
                np.ascontiguousarray(meas_all), dtype=self.dtype, device=self.device
            )
            atm_t = torch.as_tensor(
                np.ascontiguousarray(atm_all), dtype=self.dtype, device=self.device
            )
            x0_all = (
                self.initializer_batch.initial_state(atm_t, meas_t, geom).cpu().numpy()
            )

        return (
            list(rows),
            list(cols),
            meas_all,
            geom,
            atm_all,
            sub_all,
            x0_all,
        )

    def _initial_state(self, meas, geom, x_atmosphere, sub_state):
        """Build the starting state for one pixel.

        Kept scalar: the algebraic and simple initializers call back into the
        scalar forward model, and both are far cheaper than the MAP solve they
        seed. Batching them is a follow-up, not a prerequisite.
        """
        if self.initializer == "superpixel":
            x0 = sub_state.copy()
            x0[self.fm.idx_atmosphere] = x_atmosphere

        elif self.initializer == "algebraic":
            x_surface, _, x_instrument = self.fm.unpack(self.fm.init.copy())
            rfl_est, _ = invert_algebraic(
                self.fm, x_surface, x_atmosphere, x_instrument, meas, geom
            )
            rfl_est = self.fm.surface.fit_params(rfl_est, geom)
            x0 = np.concatenate([rfl_est, x_atmosphere, x_instrument])

        elif self.initializer == "simple":
            x0 = invert_simple(self.fm, meas, geom)
            x0[self.fm.idx_atmosphere] = x_atmosphere

        else:
            raise ValueError("No valid initializer given for AOE algorithm")

        bounds = (
            self.fm.heuristic_bounds(geom)
            if self.per_pixel_heuristic_prior
            else self.fm.bounds
        )
        x0 = self.fm.clip_bounds(x0, bounds, eps=eps)
        if self.per_pixel_heuristic_prior:
            self.fm.update_heuristic_prior_means(x0, geom)
        return x0

    def _write(
        self, start_line, stop_line, output_rfl, output_rfl_unc,
        output_non_rfl=None, output_non_rfl_unc=None,
    ):
        """Write the block, matching the scalar worker's BIL layout.

        ``output_rfl`` is accumulated BIP-shaped ``(lines, samples, bands)`` and
        must be written as BIL ``(lines, bands, samples)``. Use ``swapaxes(1, 2)``,
        NOT ``.T``: transpose reverses *all three* axes, giving
        ``(bands, samples, lines)``. ``write_bil_chunk`` serializes with a raw
        ``tobytes()`` and never reshapes, so the wrong permutation silently
        writes a scrambled cube -- same values, wrong pixels. That failure is
        near-invisible in aggregate statistics (identical histogram, identical
        valid/fill fractions) and cost a full end-to-end debugging cycle here.
        """
        write_bil_chunk(
            np.swapaxes(output_rfl, 1, 2),
            self.rfl_outpath,
            start_line,
            (self.n_lines, self.n_rfl_bands, self.n_samples),
        )
        write_bil_chunk(
            np.swapaxes(output_rfl_unc, 1, 2),
            self.unc_outpath,
            start_line,
            (self.n_lines, self.n_rfl_bands, self.n_samples),
        )

        # Non-reflectance surface state, mirroring analytical_line.py:738-750.
        # Without this the glint terms are solved and then discarded: the
        # cubes are created by analytical_line and left at their fill value.
        if self.non_rfl_outpath and output_non_rfl is not None:
            write_bil_chunk(
                np.swapaxes(output_non_rfl, 1, 2),
                self.non_rfl_outpath,
                start_line,
                (self.n_lines, self.n_non_rfl_bands, self.n_samples),
            )
            write_bil_chunk(
                np.swapaxes(output_non_rfl_unc, 1, 2),
                self.non_rfl_unc_outpath,
                start_line,
                (self.n_lines, self.n_non_rfl_bands, self.n_samples),
            )
