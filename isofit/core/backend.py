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
"""Torch device and dtype resolution, shared by every ISOFIT component that
uses torch (the sRTMnet emulator and the optional batched ``torch`` backend).

This is the single source of truth for two policy questions:

1. Which device do we run on? Preference order is **cuda > mps > cpu**.
2. Which floating-point precision? ISOFIT's retrieval math (Cholesky
   factorizations of measurement covariances, finite-difference Jacobians with
   ``eps=1e-5``) needs float64. MPS does not implement float64 at all, so it is
   a development target only.

The central rule is that a GPU request is never silently downgraded. An
explicit device that is unavailable raises; only ``"auto"`` falls back, and it
says so loudly.
"""

from __future__ import annotations

import logging
import re

Logger = logging.getLogger(__name__)

# "cuda", "cuda:0", "mps", "cpu", "auto"
DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")

VALID_DTYPES = ("auto", "float32", "float64")


def _torch():
    """Import torch, converting the ImportError into an actionable message."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch is required for device resolution but is not installed. "
            "Install ISOFIT with its default dependencies, or for CUDA see the "
            "GPU section of the installation docs."
        ) from e
    return torch


def _cuda_available() -> bool:
    return _torch().cuda.is_available()


def _mps_available() -> bool:
    torch = _torch()
    # torch.backends.mps exists on all modern builds, but guard anyway: some
    # minimal builds omit it.
    backends = getattr(torch.backends, "mps", None)
    return bool(backends is not None and backends.is_available())


def resolve_device(spec: str = "auto", allow_cpu_fallback: bool = True):
    """Resolve a device specification string to a ``torch.device``.

    Args:
        spec: One of ``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``, ``"cuda:N"``.
            ``"auto"`` prefers cuda, then mps, then cpu.
        allow_cpu_fallback: When ``spec`` is ``"auto"`` and no accelerator is
            present, fall back to cpu (with a loud warning). When ``False``,
            an ``"auto"`` resolution that finds no accelerator raises instead.
            Has no effect for explicit specs.

    Returns:
        torch.device

    Raises:
        ValueError: ``spec`` is not a recognized device string.
        RuntimeError: An explicitly requested device is unavailable, or
            ``allow_cpu_fallback=False`` and no accelerator was found.

    Notes:
        An explicitly requested device is never silently downgraded to cpu --
        that would turn a performance request into a slow success, which is
        much harder to notice than a failure.
    """
    torch = _torch()

    if not isinstance(spec, str) or not DEVICE_RE.match(spec):
        raise ValueError(
            f"Invalid device specification {spec!r}. Expected one of: "
            "'auto', 'cpu', 'mps', 'cuda', or 'cuda:N'."
        )

    if spec == "cpu":
        Logger.info("Using torch device: cpu (explicitly requested)")
        return torch.device("cpu")

    if spec.startswith("cuda"):
        if not _cuda_available():
            raise RuntimeError(
                f"Device {spec!r} was explicitly requested but CUDA is not "
                f"available (torch.cuda.is_available() is False, torch "
                f"{torch.__version__}). This usually means a CPU-only torch "
                "build is installed -- ISOFIT pins the CPU wheel index by "
                "default. See the GPU section of the installation docs for "
                "installing a CUDA build. To run on CPU instead, set the "
                "device to 'cpu' or 'auto'."
            )
        index = int(spec.split(":")[1]) if ":" in spec else 0
        count = torch.cuda.device_count()
        if index >= count:
            raise RuntimeError(
                f"Device {spec!r} was requested but only {count} CUDA "
                f"device(s) are visible (valid indices 0..{count - 1}). Check "
                "CUDA_VISIBLE_DEVICES."
            )
        Logger.info(
            f"Using torch device: {spec} ({torch.cuda.get_device_name(index)})"
        )
        return torch.device(spec)

    if spec == "mps":
        if not _mps_available():
            raise RuntimeError(
                "Device 'mps' was explicitly requested but Metal Performance "
                "Shaders are not available. MPS requires macOS on Apple "
                "silicon with an MPS-enabled torch build. To run on CPU "
                "instead, set the device to 'cpu' or 'auto'."
            )
        _warn_mps_precision()
        return torch.device("mps")

    # spec == "auto": cuda > mps > cpu.
    if _cuda_available():
        Logger.info(f"Using torch device: cuda ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")

    if _mps_available():
        Logger.info("Using torch device: mps (no CUDA device found)")
        _warn_mps_precision()
        return torch.device("mps")

    if not allow_cpu_fallback:
        raise RuntimeError(
            "No GPU device is available (neither CUDA nor MPS) and CPU "
            "fallback was disabled."
        )

    Logger.warning(
        """
******************************************************************************************
! A torch device was requested but no GPU was found -- falling back to CPU.
! torch on CPU is expected to be SLOWER than the default numpy backend for retrievals.
! Set the device explicitly to 'cpu' to silence this, or install a CUDA-enabled torch.
******************************************************************************************\
"""
    )
    return torch.device("cpu")


def _warn_mps_precision():
    """MPS is a development target: fp32 only, so parity bands are wider."""
    Logger.warning(
        "Using torch device: mps. MPS does not support float64, so retrievals "
        "run in float32 with reduced numerical fidelity. MPS is a development "
        "and plumbing target; use CUDA for production and for any reported "
        "parity or performance numbers."
    )


def resolve_backend_options(config, backend=None, device=None, batch_size=None):
    """Merge explicit overrides with an ISOFIT config's backend settings.

    Utilities such as ``analytical_line`` accept backend options both on the
    command line and through the ISOFIT config file. The command line wins; a
    ``None`` override means "use whatever the config says".

    Args:
        config: A full ISOFIT ``Config`` (its ``implementation`` section is read).
        backend: Optional override for the backend name.
        device: Optional override for the torch device spec.
        batch_size: Optional override for the batch size.

    Returns:
        dict with keys ``backend``, ``torch_device``, ``torch_batch_size``,
        ``torch_dtype``, and ``torch_num_gpu_workers``.
    """
    impl = getattr(config, "implementation", None)

    def _pick(override, attr, fallback):
        if override is not None:
            return override
        return getattr(impl, attr, fallback) if impl is not None else fallback

    return {
        "backend": _pick(backend, "backend", "numpy"),
        "torch_device": _pick(device, "torch_device", "auto"),
        "torch_batch_size": _pick(batch_size, "torch_batch_size", "auto"),
        "torch_dtype": _pick(None, "torch_dtype", "auto"),
        "torch_num_gpu_workers": _pick(None, "torch_num_gpu_workers", None),
    }


def resolve_batch_size(spec, bytes_per_pixel: int, device, default: int = 512) -> int:
    """Resolve a batch-size specification to a concrete integer.

    Args:
        spec: A positive integer (or its string form), or ``"auto"``.
        bytes_per_pixel: Estimated peak device bytes required per pixel, used to
            size an ``"auto"`` batch against free device memory.
        device: The ``torch.device`` the batch will run on.
        default: Batch size used for ``"auto"`` when free memory cannot be
            queried (non-CUDA devices).

    Returns:
        int: number of spectra per batched call, at least 1.
    """
    torch = _torch()

    if str(spec) != "auto":
        size = int(spec)
        if size <= 0:
            raise ValueError(f"batch size must be positive, got {size}")
        return size

    device_type = getattr(device, "type", device)
    if device_type != "cuda" or bytes_per_pixel <= 0:
        return default

    index = getattr(device, "index", None) or 0
    free, _total = torch.cuda.mem_get_info(index)

    # Leave headroom for allocator fragmentation and the LUT already resident.
    size = int(0.7 * free / bytes_per_pixel)

    # Round down to a multiple of 64 for tidier kernel shapes.
    size = max(64, (size // 64) * 64)
    Logger.info(
        f"Auto batch size: {size} spectra "
        f"({bytes_per_pixel / 2**20:.2f} MiB/pixel, {free / 2**30:.2f} GiB free)"
    )
    return size


def resolve_dtype(spec: str = "auto", device=None):
    """Resolve a dtype specification to a ``torch.dtype`` for a given device.

    Args:
        spec: ``"auto"``, ``"float32"``, or ``"float64"``. ``"auto"`` selects
            float64 on cuda/cpu and float32 on mps.
        device: The ``torch.device`` the dtype will be used on. Required when
            ``spec`` is ``"auto"``.

    Returns:
        torch.dtype

    Raises:
        ValueError: Unrecognized ``spec``, or float64 requested on mps (which
            cannot support it).
    """
    torch = _torch()

    if spec not in VALID_DTYPES:
        raise ValueError(
            f"Invalid dtype specification {spec!r}. Expected one of: "
            f"{', '.join(VALID_DTYPES)}."
        )

    device_type = getattr(device, "type", device)

    if spec == "float64":
        if device_type == "mps":
            raise ValueError(
                "dtype 'float64' was requested on device 'mps', but MPS does "
                "not implement float64. Use 'float32' (development fidelity) "
                "or run on cuda/cpu for float64."
            )
        return torch.float64

    if spec == "float32":
        return torch.float32

    # spec == "auto"
    if device is None:
        raise ValueError("resolve_dtype(spec='auto') requires a device.")

    if device_type == "mps":
        Logger.info("Selected dtype float32 (mps does not support float64)")
        return torch.float32

    return torch.float64
