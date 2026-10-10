"""Numerical backends for ISOFIT.

The default ``numpy`` backend is the historical per-pixel scipy/numpy code path
and lives in the main package. The optional :mod:`isofit.backends.torch` backend
provides batched implementations of the same math that can run on a GPU.

Backends are selected through ``implementation.backend`` in the ISOFIT config.
"""
