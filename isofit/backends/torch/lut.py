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
"""Batched multilinear look-up table interpolation.

This is the torch counterpart of :func:`isofit.core.common._numba_mlg_kernel`,
which interpolates one point at a time. Every radiance evaluation in ISOFIT
samples the LUT, so evaluating thousands of points in a single call is the
foundation the rest of the torch backend is built on.

All LUT quantities in ISOFIT share a single grid (see
``isofit.luts.reader.Reader.build_interpolators``), so their channel dimensions
are concatenated into one tensor. Interpolation weights and corner indices are
then computed once and reused across every quantity, and the ``2**D`` corner
gathers become one gather per corner instead of one per corner *per quantity*.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

Logger = logging.getLogger(__name__)


class BatchedLUT:
    """Multilinear interpolator over an N-dimensional grid, batched over points.

    Args:
        grids: Per-dimension grid axes, each a 1-D monotonically increasing
            array of length ``g_i``.
        data: Mapping of quantity name to an array of shape
            ``(g_0, ..., g_{D-1}, n_wl)``. All entries must share the grid.
        constants: Mapping of quantity name to a scalar, for quantities that are
            constant across the whole grid (mirrors ``VectorInterpolator``'s
            ``method == -1`` shortcut).
        device: Torch device to hold the table on.
        dtype: Torch dtype for the table and the interpolated output.

    Notes:
        Grid axes and the interpolation weights are held in float64 wherever the
        device supports it, even when ``dtype`` is float32. Cell selection uses
        ``searchsorted`` on the grid, and rounding a coordinate in float32 can
        select a *different cell* near a node -- a discontinuous error, not a
        small one. Only the table values and the accumulation follow ``dtype``.

        MPS is the exception: it has no float64 at all, so coordinates there are
        necessarily float32 and cell selection carries that risk. This is one
        more reason MPS is a development target only.
    """

    def __init__(
        self,
        grids: list,
        data: dict,
        constants: dict = None,
        device=None,
        dtype=torch.float64,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.constants = dict(constants or {})

        if not data:
            raise ValueError("BatchedLUT requires at least one gridded quantity")

        # Coordinate/weight precision: float64 unless the device cannot provide
        # it. Kept separate from `dtype` (the table precision) because cell
        # selection is far more sensitive to rounding than the values are.
        self.coord_dtype = (
            torch.float32 if self.device.type == "mps" else torch.float64
        )

        self.grids = [
            torch.as_tensor(np.asarray(g), dtype=self.coord_dtype, device=self.device)
            for g in grids
        ]
        self.dims = len(self.grids)
        self.shape = tuple(int(g.numel()) for g in self.grids)

        if self.dims > 20:
            # 2**D corner gathers; well past anything ISOFIT builds (3-5 dims).
            raise ValueError(f"BatchedLUT supports at most 20 dimensions, got {self.dims}")

        self.keys = list(data)
        n_cells = int(np.prod(self.shape)) if self.shape else 1

        # Validate and stack. Layout is (n_cells, K * n_wl): a single contiguous
        # row per grid cell holding every quantity's channels back to back.
        n_wl = None
        flats = []
        for key in self.keys:
            arr = np.asarray(data[key])
            if arr.shape[: self.dims] != self.shape:
                raise ValueError(
                    f"LUT quantity {key!r} has grid shape {arr.shape[: self.dims]}, "
                    f"expected {self.shape}"
                )
            if arr.ndim != self.dims + 1:
                raise ValueError(
                    f"LUT quantity {key!r} must have exactly one channel "
                    f"dimension after {self.dims} grid dimensions, got shape {arr.shape}"
                )
            if n_wl is None:
                n_wl = arr.shape[-1]
            elif arr.shape[-1] != n_wl:
                raise ValueError(
                    f"LUT quantity {key!r} has {arr.shape[-1]} channels, "
                    f"expected {n_wl} to match {self.keys[0]!r}"
                )
            flats.append(
                torch.as_tensor(arr.reshape(n_cells, n_wl), dtype=self.dtype)
            )

        self.n_wl = int(n_wl)
        self.n_keys = len(self.keys)
        self.data = torch.cat(flats, dim=1).contiguous().to(self.device)

        # Flat cell strides (row-major over the grid dimensions only; the channel
        # stride is handled by the row layout above).
        strides = np.ones(self.dims, dtype=np.int64)
        for d in range(self.dims - 2, -1, -1):
            strides[d] = strides[d + 1] * self.shape[d + 1]
        self.strides = torch.as_tensor(strides, device=self.device)

        # Per-quantity column slices into the stacked row.
        self.slices = {
            key: (i * self.n_wl, (i + 1) * self.n_wl)
            for i, key in enumerate(self.keys)
        }

        # A degenerate (length-1) axis always yields delta == 0, so every corner
        # that steps *up* along it is weighted by exactly zero. Those corners are
        # dropped here, at construction, rather than by testing the weights at
        # runtime: a runtime test would force a device-to-host sync per corner
        # (up to 2**D per call), and skipping them statically also guarantees the
        # upper index of a degenerate axis is never formed, so no gather can run
        # off the end of the table.
        degenerate = [d for d in range(self.dims) if self.shape[d] == 1]
        self.corners = [
            c
            for c in range(1 << self.dims)
            if not any((c >> d) & 1 for d in degenerate)
        ]

        Logger.debug(
            f"BatchedLUT: {self.n_keys} quantities, grid {self.shape}, "
            f"{self.n_wl} channels, {self.data.numel() * self.data.element_size() / 2**20:.1f} MiB "
            f"on {self.device}"
        )

    @classmethod
    def from_interpolators(cls, interpolators: dict, device=None, dtype=torch.float64):
        """Build from a mapping of name -> ``VectorInterpolator``.

        This is the bridge from :meth:`isofit.luts.reader.Reader.build_interpolators`:
        constant quantities (``method == -1``) are carried as scalars, and the
        rest supply their grid and dense table.
        """
        grids = None
        data = {}
        constants = {}

        for key, itp in interpolators.items():
            if getattr(itp, "method", None) == -1:
                constants[key] = float(itp.value)
                continue

            if grids is None:
                grids = [np.asarray(g) for g in itp.grid_tuples]
            data[key] = np.asarray(itp.gridarrays)

        if grids is None:
            raise ValueError(
                "No gridded LUT quantities found; every interpolator was constant"
            )

        return cls(grids, data, constants, device=device, dtype=dtype)

    def _weights_and_cells(self, points: torch.Tensor):
        """Compute corner weights and flat cell indices for a batch of points.

        Args:
            points: ``(B, D)`` coordinates in grid space.

        Returns:
            (low, delta): ``(B, D)`` int64 lower cell indices and ``(B, D)``
            float64 interpolation fractions.

        Notes:
            Mirrors ``_numba_mlg_kernel`` exactly, including its branch
            precedence: the ``p <= grid[0]`` clamp is applied *after* the
            ``p >= grid[-1]`` clamp so that it wins for a degenerate axis where
            both are true.
        """
        B = points.shape[0]
        low = torch.empty((B, self.dims), dtype=torch.int64, device=points.device)
        delta = torch.empty((B, self.dims), dtype=self.coord_dtype, device=points.device)

        for d, grid in enumerate(self.grids):
            p = points[:, d]
            n = grid.numel()

            if n == 1:
                # Degenerate axis: the single node is the only sample available.
                low[:, d] = 0
                delta[:, d] = 0.0
                continue

            # Interior: searchsorted(side='left') - 1, matching fast_searchsorted.
            # An exact interior node hit therefore lands on the *left* cell with
            # delta == 1.0, which is what the numba kernel does.
            idx = torch.searchsorted(grid, p.contiguous()) - 1
            idx = idx.clamp(0, n - 2)
            g0 = grid[idx]
            g1 = grid[idx + 1]
            frac = (p - g0) / (g1 - g0)

            # Above the top node: last cell, fully weighted to the upper corner.
            above = p >= grid[-1]
            idx = torch.where(above, torch.full_like(idx, n - 2), idx)
            frac = torch.where(above, torch.ones_like(frac), frac)

            # At or below the bottom node: first cell, fully weighted to the
            # lower corner. Applied last so it takes precedence (see docstring).
            below = p <= grid[0]
            idx = torch.where(below, torch.zeros_like(idx), idx)
            frac = torch.where(below, torch.zeros_like(frac), frac)

            low[:, d] = idx
            delta[:, d] = frac

        return low, delta

    def interpolate_stacked(self, points: torch.Tensor) -> torch.Tensor:
        """Interpolate every gridded quantity, returning the stacked tensor.

        Args:
            points: ``(B, D)`` coordinates in grid space.

        Returns:
            ``(B, K * n_wl)`` tensor of interpolated values, quantities
            concatenated in ``self.keys`` order.
        """
        points = torch.as_tensor(points, dtype=self.coord_dtype, device=self.device)
        if points.ndim == 1:
            points = points.unsqueeze(0)
        if points.shape[1] != self.dims:
            raise ValueError(
                f"points has {points.shape[1]} dimensions, expected {self.dims}"
            )

        low, delta = self._weights_and_cells(points)
        B = points.shape[0]

        out = torch.zeros((B, self.data.shape[1]), dtype=self.dtype, device=self.device)

        # Accumulate one corner at a time. The alternative -- gathering all 2**D
        # corners at once -- would materialize a (B, 2**D, K*n_wl) intermediate,
        # which is multiple GB at production batch sizes.
        #
        # self.corners omits corners that step up along a degenerate axis; see
        # the note where it is built. No runtime weight test is performed here,
        # deliberately: it would sync the device on every corner.
        upper = low + 1

        for corner in self.corners:
            weight = torch.ones(B, dtype=self.coord_dtype, device=self.device)
            cell = torch.zeros(B, dtype=torch.int64, device=self.device)

            for d in range(self.dims):
                if (corner >> d) & 1:
                    weight = weight * delta[:, d]
                    cell = cell + upper[:, d] * self.strides[d]
                else:
                    weight = weight * (1.0 - delta[:, d])
                    cell = cell + low[:, d] * self.strides[d]

            out += self.data.index_select(0, cell) * weight.unsqueeze(1).to(self.dtype)

        return out

    def _grad_dims(self, dims) -> list:
        """Normalize a requested gradient-dimension list."""
        if dims is None:
            return list(range(self.dims))

        out = [int(d) for d in dims]
        for d in out:
            if not 0 <= d < self.dims:
                raise ValueError(
                    f"gradient dimension {d} is out of range for a "
                    f"{self.dims}-dimensional LUT"
                )
        return out

    def interpolate_stacked_with_gradients(self, points: torch.Tensor, dims=None):
        """Interpolate every quantity *and* its coordinate derivatives, in one pass.

        Args:
            points: ``(B, D)`` coordinates in grid space.
            dims: Grid dimensions to differentiate with respect to. ``None``
                requests every dimension.

        Returns:
            ``(values, grads)``: ``(B, K * n_wl)`` interpolated values and
            ``(B, len(dims), K * n_wl)`` derivatives, where ``grads[:, j]`` is
            ``d value / d p_{dims[j]}``.

        Notes:
            The multilinear form and its derivative share the same corner sum::

                value(p) = sum_c ( prod_d f_d(c) ) * data[cell_c]
                d value / d p_k
                         = (1 / width_k) * sum_c s_k(c)
                             * ( prod_{d != k} f_d(c) ) * data[cell_c]

            with ``f_d(c) = delta_d`` when bit ``d`` of the corner mask is set
            and ``1 - delta_d`` otherwise, ``s_k(c) = +1 / -1`` under the same
            test, and ``width_k = grid_k[low_k + 1] - grid_k[low_k]``.

            Both are accumulated in a single corner loop, so the table is
            gathered exactly once per corner -- the same number of gathers as
            :meth:`interpolate_stacked`. Avoiding a second pass over the table is
            the entire point of this method: at production batch sizes the
            gathers dominate, and the extra weight arithmetic is ``O(B)`` per
            corner against ``O(B * K * n_wl)`` for the gather.
        """
        points = torch.as_tensor(points, dtype=self.coord_dtype, device=self.device)
        if points.ndim == 1:
            points = points.unsqueeze(0)
        if points.shape[1] != self.dims:
            raise ValueError(
                f"points has {points.shape[1]} dimensions, expected {self.dims}"
            )

        dims = self._grad_dims(dims)
        low, delta = self._weights_and_cells(points)
        B = points.shape[0]
        upper = low + 1

        values = torch.zeros(
            (B, self.data.shape[1]), dtype=self.dtype, device=self.device
        )
        grads = torch.zeros(
            (B, len(dims), self.data.shape[1]), dtype=self.dtype, device=self.device
        )

        # Per-requested-dimension 1/width factor. Dimensions whose derivative is
        # identically zero are dropped from the loop entirely and keep the zeros
        # allocated above.
        live = []
        for j, d in enumerate(dims):
            grid = self.grids[d]

            if grid.numel() == 1:
                # Degenerate axis: the interpolant does not depend on this
                # coordinate at all, so the derivative is exactly zero. It must
                # be skipped rather than computed -- self.corners has already
                # dropped this axis's up-corners, so the corner sum below is no
                # longer antisymmetric in d and would return -value.
                continue

            idx = low[:, d]
            scale = 1.0 / (grid[idx + 1] - grid[idx])

            # Outside the grid, _weights_and_cells clamps delta to 0 or 1 exactly
            # as the numba reference does, which makes the interpolant locally
            # CONSTANT there -- so the derivative is zero, not the slope of the
            # edge cell. A point sitting exactly on the first or last node also
            # trips that clamp; the derivative reported there is the one-sided
            # (outward) one, i.e. zero.
            p = points[:, d]
            clamped = (p <= grid[0]) | (p >= grid[-1])
            scale = torch.where(clamped, torch.zeros_like(scale), scale)

            live.append((j, d, scale))

        for corner in self.corners:
            # Per-dimension weight factors, kept individually so the derivative
            # can form the "all dimensions but one" product below.
            factors = []
            cell = torch.zeros(B, dtype=torch.int64, device=self.device)

            for d in range(self.dims):
                if (corner >> d) & 1:
                    factors.append(delta[:, d])
                    cell = cell + upper[:, d] * self.strides[d]
                else:
                    factors.append(1.0 - delta[:, d])
                    cell = cell + low[:, d] * self.strides[d]

            weight = factors[0]
            for f in factors[1:]:
                weight = weight * f

            # The one gather this corner is allowed.
            row = self.data.index_select(0, cell)
            values += row * weight.unsqueeze(1).to(self.dtype)

            for j, d, scale in live:
                # prod_{e != d} f_e, built explicitly rather than as
                # weight / factors[d]. A factor of exactly 0 or 1 is the normal
                # case here (any clamped or on-node coordinate), so dividing it
                # back out would be 0/0.
                excl = None
                for e in range(self.dims):
                    if e == d:
                        continue
                    excl = factors[e] if excl is None else excl * factors[e]
                if excl is None:
                    # 1-D table: the empty product is 1.
                    excl = torch.ones(B, dtype=self.coord_dtype, device=self.device)

                coef = excl * scale
                if not ((corner >> d) & 1):
                    coef = -coef
                grads[:, j] += row * coef.unsqueeze(1).to(self.dtype)

        return values, grads

    def interpolate_with_gradients(self, points: torch.Tensor, dims=None):
        """Interpolate the LUT and its coordinate derivatives at a batch of points.

        Args:
            points: ``(B, D)`` coordinates in grid space, ordered to match the
                LUT's dimension order.
            dims: Grid dimensions to differentiate with respect to. ``None``
                requests every dimension.

        Returns:
            ``(values, grads)``. ``values`` matches :meth:`interpolate`, mapping
            quantity name to ``(B, n_wl)`` plus the constants. ``grads`` maps
            each *gridded* quantity name to ``(B, len(dims), n_wl)``. Constant
            quantities are omitted from ``grads``: their derivative is
            identically zero by construction.
        """
        dims = self._grad_dims(dims)
        stacked, grad_stacked = self.interpolate_stacked_with_gradients(points, dims)

        values = {
            key: stacked[:, start:stop] for key, (start, stop) in self.slices.items()
        }
        values.update(self.constants)
        grads = {
            key: grad_stacked[:, :, start:stop]
            for key, (start, stop) in self.slices.items()
        }
        return values, grads

    def interpolate(self, points: torch.Tensor) -> dict:
        """Interpolate the LUT at a batch of points.

        Args:
            points: ``(B, D)`` coordinates in grid space, ordered to match the
                LUT's dimension order.

        Returns:
            dict mapping quantity name to a ``(B, n_wl)`` tensor, plus any
            constant quantities as plain floats.
        """
        stacked = self.interpolate_stacked(points)

        result = {
            key: stacked[:, start:stop] for key, (start, stop) in self.slices.items()
        }
        result.update(self.constants)
        return result

    def to(self, device=None, dtype=None) -> "BatchedLUT":
        """Move the table to another device and/or dtype, in place."""
        if device is not None:
            self.device = torch.device(device)
            self.coord_dtype = (
                torch.float32 if self.device.type == "mps" else torch.float64
            )
            self.data = self.data.to(self.device)
            self.strides = self.strides.to(self.device)
            self.grids = [g.to(self.device, self.coord_dtype) for g in self.grids]
        if dtype is not None:
            self.dtype = dtype
            self.data = self.data.to(dtype)
        return self

    @property
    def nbytes(self) -> int:
        """Device bytes held by the stacked table."""
        return self.data.numel() * self.data.element_size()
