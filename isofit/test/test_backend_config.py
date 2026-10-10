"""Tests for the torch-backend options on ImplementationConfig.

These cover the config *surface* only: option defaults, validation rules, and the
guarantee that the default configuration is unchanged from historical behavior.
Device availability is intentionally not validated at config time (config objects
are built inside Ray workers); that is covered by test_backend.py.
"""

import pytest

from isofit.configs.sections.implementation_config import ImplementationConfig


def _config(**kwargs) -> ImplementationConfig:
    return ImplementationConfig(kwargs or {})


def _validate(**kwargs):
    """Return (errors, warnings) from the backend validator alone."""
    return _config(**kwargs)._check_backend_validity()


# --- defaults -------------------------------------------------------------------


def test_defaults_are_the_legacy_path():
    """The default config must select the untouched numpy code path."""
    c = _config()
    assert c.backend == "numpy"
    assert c.torch_device == "auto"
    assert c.torch_batch_size == "auto"
    assert c.torch_dtype == "auto"
    assert c.torch_num_gpu_workers is None


def test_defaults_produce_no_backend_errors_or_warnings():
    errors, warnings = _validate()
    assert errors == []
    assert warnings == []


# --- backend ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_valid_backends(backend):
    errors, _ = _validate(backend=backend)
    assert errors == []


@pytest.mark.parametrize("backend", ["cupy", "jax", "NUMPY", ""])
def test_invalid_backend_errors(backend):
    errors, _ = _validate(backend=backend)
    assert any("Invalid backend" in e for e in errors)


# --- torch_device ----------------------------------------------------------------


@pytest.mark.parametrize(
    "device", ["auto", "cpu", "mps", "cuda", "cuda:0", "cuda:1", "cuda:15"]
)
def test_valid_devices(device):
    errors, _ = _validate(backend="torch", torch_device=device)
    assert errors == []


@pytest.mark.parametrize("device", ["gpu", "gpu:0", "gpu0", "CUDA", "cuda:", "gpu:x"])
def test_invalid_device_errors(device):
    errors, _ = _validate(backend="torch", torch_device=device)
    assert any("Invalid torch_device" in e for e in errors)


# --- torch_dtype -----------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["auto", "float32", "float64"])
def test_valid_dtypes(dtype):
    errors, _ = _validate(backend="torch", torch_dtype=dtype)
    assert errors == []


@pytest.mark.parametrize("dtype", ["float16", "double", "fp32", ""])
def test_invalid_dtype_errors(dtype):
    errors, _ = _validate(backend="torch", torch_dtype=dtype)
    assert any("Invalid torch_dtype" in e for e in errors)


def test_mps_with_float64_is_an_error():
    """MPS has no float64 implementation; this combination can never be satisfied."""
    errors, _ = _validate(
        backend="torch", torch_device="mps", torch_dtype="float64"
    )
    assert any("not supported on torch_device 'mps'" in e for e in errors)


def test_mps_with_float32_is_fine():
    errors, _ = _validate(backend="torch", torch_device="mps", torch_dtype="float32")
    assert errors == []


def test_mps_with_auto_is_fine():
    errors, _ = _validate(backend="torch", torch_device="mps", torch_dtype="auto")
    assert errors == []


def test_cuda_with_float64_is_fine():
    errors, _ = _validate(backend="torch", torch_device="cuda", torch_dtype="float64")
    assert errors == []


# --- torch_batch_size ------------------------------------------------------------


@pytest.mark.parametrize("size", ["auto", 1, 512, 4096, "2048"])
def test_valid_batch_sizes(size):
    errors, _ = _validate(backend="torch", torch_batch_size=size)
    assert errors == []


@pytest.mark.parametrize("size", [0, -1, -4096, "0"])
def test_nonpositive_batch_size_errors(size):
    errors, _ = _validate(backend="torch", torch_batch_size=size)
    assert any("must be positive" in e for e in errors)


@pytest.mark.parametrize("size", ["big", "1.5.2", "none"])
def test_unparseable_batch_size_errors(size):
    errors, _ = _validate(backend="torch", torch_batch_size=size)
    assert any("Invalid torch_batch_size" in e for e in errors)


def test_integer_batch_size_from_json_is_accepted():
    """A JSON config naturally carries an int; the base config coerces to str."""
    c = _config(backend="torch", torch_batch_size=2048)
    errors, _ = c.check_config_validity()
    assert not [e for e in errors if "batch" in e.lower()]
    assert int(c.torch_batch_size) == 2048


# --- torch_num_gpu_workers -------------------------------------------------------


def test_valid_gpu_worker_counts():
    for n in (None, 1, 8):
        errors, _ = _validate(backend="torch", torch_num_gpu_workers=n)
        assert errors == []


@pytest.mark.parametrize("n", [0, -2])
def test_nonpositive_gpu_workers_errors(n):
    errors, _ = _validate(backend="torch", torch_num_gpu_workers=n)
    assert any("torch_num_gpu_workers must be positive" in e for e in errors)


# --- cross-option warnings -------------------------------------------------------


def test_torch_options_with_numpy_backend_warn():
    """Silently ignoring a user's GPU request would be a footgun."""
    _, warnings = _validate(backend="numpy", torch_device="cuda")
    assert any("ignored because backend is 'numpy'" in w for w in warnings)
    assert any("torch_device" in w for w in warnings)


def test_torch_options_with_numpy_backend_do_not_error():
    errors, _ = _validate(backend="numpy", torch_device="cuda", torch_dtype="float32")
    assert errors == []


def test_n_cores_with_torch_backend_warns():
    """n_cores does not govern GPU worker count; say so rather than surprise."""
    _, warnings = _validate(backend="torch", n_cores=32)
    assert any("does not control the number of GPU workers" in w for w in warnings)


def test_n_cores_with_numpy_backend_does_not_warn():
    _, warnings = _validate(backend="numpy", n_cores=32)
    assert warnings == []


def test_backend_validity_is_wired_into_main_validator():
    """The backend checks must actually run as part of _check_config_validity."""
    errors, _ = _config(backend="bogus")._check_config_validity()
    assert any("Invalid backend" in e for e in errors)
