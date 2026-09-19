"""Tests for ``isofit.core.backend`` device and dtype resolution."""

from unittest import mock

import pytest
import torch

from isofit.core.backend import (
    resolve_backend_options,
    resolve_batch_size,
    resolve_device,
    resolve_dtype,
)

# --- helpers --------------------------------------------------------------------


def _patch_availability(cuda: bool, mps: bool, device_count: int = 1):
    """Patch the availability probes in isofit.core.backend."""
    return mock.patch.multiple(
        "isofit.core.backend",
        _cuda_available=mock.MagicMock(return_value=cuda),
        _mps_available=mock.MagicMock(return_value=mps),
    )


# --- spec validation ------------------------------------------------------------


@pytest.mark.parametrize("spec", ["auto", "cpu", "mps", "cuda", "cuda:0", "cuda:3"])
def test_valid_specs_accepted(spec):
    """Recognized specs must not raise ValueError (may raise RuntimeError)."""
    with _patch_availability(cuda=False, mps=False):
        try:
            resolve_device(spec)
        except RuntimeError:
            pass  # unavailable hardware is fine here; a ValueError is not
        except ValueError:
            pytest.fail(f"{spec!r} should be a valid device specification")


@pytest.mark.parametrize("spec", ["gpu", "cuda:", "CUDA", "xpu", "", "cuda:x", None, 3])
def test_invalid_specs_rejected(spec):
    with pytest.raises(ValueError):
        resolve_device(spec)


# --- auto resolution ordering ---------------------------------------------------


def test_auto_prefers_cuda_over_mps():
    """Regression guard: the old sRTMnet code let mps override an available cuda."""
    with _patch_availability(cuda=True, mps=True):
        with mock.patch("torch.cuda.get_device_name", return_value="Fake A100"):
            assert resolve_device("auto").type == "cuda"


def test_auto_uses_mps_when_no_cuda():
    with _patch_availability(cuda=False, mps=True):
        assert resolve_device("auto").type == "mps"


def test_auto_falls_back_to_cpu_with_warning(caplog):
    with _patch_availability(cuda=False, mps=False):
        with caplog.at_level("WARNING", logger="isofit.core.backend"):
            assert resolve_device("auto").type == "cpu"
    assert "falling back to CPU" in caplog.text


def test_auto_without_fallback_raises():
    with _patch_availability(cuda=False, mps=False):
        with pytest.raises(RuntimeError, match="CPU fallback was disabled"):
            resolve_device("auto", allow_cpu_fallback=False)


# --- explicit devices never silently downgrade ----------------------------------


def test_explicit_cuda_unavailable_raises():
    with _patch_availability(cuda=False, mps=False):
        with pytest.raises(RuntimeError, match="CUDA is not"):
            resolve_device("cuda")


def test_explicit_cuda_does_not_fall_back_even_when_allowed():
    """allow_cpu_fallback must not apply to an explicit device request."""
    with _patch_availability(cuda=False, mps=True):
        with pytest.raises(RuntimeError):
            resolve_device("cuda:0", allow_cpu_fallback=True)


def test_explicit_mps_unavailable_raises():
    with _patch_availability(cuda=False, mps=False):
        with pytest.raises(RuntimeError, match="Metal Performance Shaders"):
            resolve_device("mps")


def test_cuda_index_out_of_range_raises():
    with _patch_availability(cuda=True, mps=False):
        with mock.patch("torch.cuda.device_count", return_value=1):
            with pytest.raises(RuntimeError, match="only 1 CUDA"):
                resolve_device("cuda:2")


def test_explicit_cpu_always_works():
    with _patch_availability(cuda=True, mps=True):
        assert resolve_device("cpu").type == "cpu"


# --- dtype ----------------------------------------------------------------------


def test_dtype_auto_is_float64_on_cpu_and_cuda():
    assert resolve_dtype("auto", torch.device("cpu")) is torch.float64
    assert resolve_dtype("auto", torch.device("cuda")) is torch.float64


def test_dtype_auto_is_float32_on_mps():
    assert resolve_dtype("auto", torch.device("mps")) is torch.float32


def test_dtype_float64_on_mps_raises():
    with pytest.raises(ValueError, match="does not implement float64"):
        resolve_dtype("float64", torch.device("mps"))


def test_dtype_explicit_overrides():
    assert resolve_dtype("float32", torch.device("cpu")) is torch.float32
    assert resolve_dtype("float64", torch.device("cpu")) is torch.float64


def test_dtype_auto_requires_device():
    with pytest.raises(ValueError, match="requires a device"):
        resolve_dtype("auto", None)


@pytest.mark.parametrize("spec", ["float16", "double", "", None])
def test_invalid_dtype_rejected(spec):
    with pytest.raises(ValueError):
        resolve_dtype(spec, torch.device("cpu"))


def test_dtype_accepts_plain_device_type_string():
    """Callers holding a device *type* string should work too."""
    assert resolve_dtype("auto", "mps") is torch.float32
    assert resolve_dtype("auto", "cpu") is torch.float64


# --- option merging (CLI overrides vs config) -----------------------------------


class _Impl:
    backend = "torch"
    torch_device = "cuda"
    torch_batch_size = "1024"
    torch_dtype = "float64"
    torch_num_gpu_workers = 2


class _Config:
    implementation = _Impl()


def test_options_read_from_config_when_no_overrides():
    opts = resolve_backend_options(_Config())
    assert opts["backend"] == "torch"
    assert opts["torch_device"] == "cuda"
    assert opts["torch_batch_size"] == "1024"
    assert opts["torch_dtype"] == "float64"
    assert opts["torch_num_gpu_workers"] == 2


def test_explicit_overrides_win_over_config():
    opts = resolve_backend_options(
        _Config(), backend="numpy", device="cpu", batch_size="64"
    )
    assert opts["backend"] == "numpy"
    assert opts["torch_device"] == "cpu"
    assert opts["torch_batch_size"] == "64"


def test_options_fall_back_to_defaults_without_config():
    class Bare:
        pass

    opts = resolve_backend_options(Bare())
    assert opts["backend"] == "numpy"
    assert opts["torch_device"] == "auto"
    assert opts["torch_batch_size"] == "auto"


# --- batch size resolution ------------------------------------------------------


def test_explicit_batch_size_passes_through():
    assert resolve_batch_size(2048, 1000, torch.device("cpu")) == 2048
    assert resolve_batch_size("512", 1000, torch.device("cpu")) == 512


def test_explicit_batch_size_rejects_nonpositive():
    with pytest.raises(ValueError, match="must be positive"):
        resolve_batch_size(0, 1000, torch.device("cpu"))


def test_auto_batch_size_uses_default_off_cuda():
    assert resolve_batch_size("auto", 1000, torch.device("cpu"), default=333) == 333


def test_auto_batch_size_scales_with_free_memory():
    """8 GiB free at 5 MiB/pixel -> ~0.7*8192/5 pixels, rounded down to a x64."""
    free = 8 * 2**30
    bpp = 5 * 2**20
    with mock.patch("torch.cuda.mem_get_info", return_value=(free, 16 * 2**30)):
        size = resolve_batch_size("auto", bpp, torch.device("cuda"))
    assert size % 64 == 0
    assert size == max(64, (int(0.7 * free / bpp) // 64) * 64)


def test_auto_batch_size_never_returns_zero():
    """A pixel larger than free memory must still yield a runnable batch."""
    with mock.patch("torch.cuda.mem_get_info", return_value=(2**20, 2**30)):
        assert resolve_batch_size("auto", 100 * 2**20, torch.device("cuda")) == 64
