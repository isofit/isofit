"""Batched torch implementations of the ISOFIT retrieval numerics.

The modules here mirror the semantics of the default numpy/scipy code path while
operating on a batch of pixels at once, so a whole chunk of an image can be
inverted in a handful of GPU kernel launches instead of one Python loop
iteration per pixel.

Every class is constructed *from* an already-built CPU object (a ``ForwardModel``,
a LUT ``Reader``, ...) rather than re-reading configuration. Precomputed
quantities on those objects -- notably the component covariance inverses, whose
values depend on the eigenvalue-stabilization ladder in
``isofit.core.common.svd_inv_sqrt`` -- are inherited verbatim so the batched path
cannot drift from the reference implementation.

This backend is opt-in via ``implementation.backend = "torch"``.
"""

from isofit.backends.torch.atmosphere import TorchAtmosphere
from isofit.backends.torch.forward import TorchRadiance
from isofit.backends.torch.geometry import BatchedGeometry
from isofit.backends.torch.instrument import TorchInstrument
from isofit.backends.torch.linalg import (
    chol_inv_full,
    svd_inv_sqrt_batch,
    upper_read_sym,
    whiten_innovation,
)
from isofit.backends.torch.analytical import invert_analytical_batch
from isofit.backends.torch.lut import BatchedLUT
from isofit.backends.torch.seps import sb_diagonal, seps_batch
from isofit.backends.torch.surface import TorchMultiComponentSurface

__all__ = [
    "BatchedGeometry",
    "BatchedLUT",
    "TorchAtmosphere",
    "TorchInstrument",
    "TorchMultiComponentSurface",
    "TorchRadiance",
    "chol_inv_full",
    "invert_analytical_batch",
    "sb_diagonal",
    "seps_batch",
    "svd_inv_sqrt_batch",
    "upper_read_sym",
    "whiten_innovation",
]
