import numpy as np
import pytest
from click.testing import CliRunner
from scipy.io import loadmat, savemat
from spectral.io import envi

from isofit.core.tilt import TiltBasis
from isofit.utils.build_tilt_basis import build_tilt_basis, cli

Windows = [[400.0, 1300.0], [1500.0, 1800.0], [2000.0, 2450.0]]


def write_basis(path, wl, bases, mean=None):
    data = {"wl": wl, "bases": bases}
    if mean is not None:
        data["mean"] = mean
    savemat(path, data)
    return path


def orthonormal(n_components, n_wl, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n_wl, n_components)))
    return q.T


@pytest.fixture
def wl():
    return np.linspace(400.0, 2500.0, 43)


@pytest.fixture
def basis(tmp_path, wl):
    """A three component basis on the instrument grid."""
    bases = orthonormal(3, len(wl))
    mean = 1.0 + 0.1 * np.sin(wl / 300.0)
    write_basis(tmp_path / "basis.mat", wl, bases, mean)
    return TiltBasis(tmp_path / "basis.mat", wl, Windows)


# TiltBasis loading


def test_basis_identical_grid(tmp_path, wl):
    """On the instrument grid, the basis loads unchanged."""
    bases = orthonormal(3, len(wl))
    mean = np.linspace(0.9, 1.1, len(wl))
    tb = TiltBasis(write_basis(tmp_path / "b.mat", wl, bases, mean), wl)

    assert np.allclose(tb.bases, bases)
    assert np.allclose(tb.mean, mean)
    assert tb.n_components == 3
    assert tb.band_names == ["Tilt PC 1", "Tilt PC 2", "Tilt PC 3"]
    assert np.array_equal(tb.winidx, np.arange(len(wl)))


@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_basis_resampled_to_instrument(tmp_path, scale):
    """Bases on a finer grid, in nm or microns, are linearly interpolated to
    the instrument wavelengths; linear functions are reproduced exactly."""
    wl_basis = np.arange(400.0, 2501.0)
    bases = np.array([np.ones_like(wl_basis), (wl_basis - 1450.0) / 1050.0])
    mean = 1.0 + 1e-4 * wl_basis
    path = write_basis(tmp_path / "b.mat", wl_basis * scale, bases, mean)

    wl = np.arange(405.5, 2495.0, 10.0)
    tb = TiltBasis(path, wl)

    assert np.allclose(tb.bases[0], 1.0)
    assert np.allclose(tb.bases[1], (wl - 1450.0) / 1050.0)
    assert np.allclose(tb.mean, 1.0 + 1e-4 * wl)


def test_basis_transposed(tmp_path, wl):
    """Bases stored one component per column are accepted."""
    bases = orthonormal(2, len(wl))
    tb = TiltBasis(write_basis(tmp_path / "b.mat", wl, bases.T), wl)

    assert np.allclose(tb.bases, bases)
    assert np.allclose(tb.mean, 0.0)


@pytest.mark.parametrize(
    "data, match",
    [
        ({"wl": np.arange(10.0) + 400}, "bases"),
        ({"bases": np.ones((2, 10))}, "wl"),
        ({"wl": np.arange(10.0) + 400, "bases": np.ones((2, 9))}, "shape"),
        (
            {"wl": np.arange(10.0) + 400, "bases": np.ones((2, 10)), "mean": [1, 2]},
            "mean",
        ),
        (
            {"wl": np.arange(10.0) + 400, "bases": np.full((2, 10), np.nan)},
            "Non-finite",
        ),
    ],
)
def test_basis_invalid(tmp_path, data, match):
    savemat(tmp_path / "b.mat", data)
    with pytest.raises(ValueError, match=match):
        TiltBasis(tmp_path / "b.mat", np.arange(10.0) + 400)


def test_basis_too_few_window_channels(tmp_path, wl):
    path = write_basis(tmp_path / "b.mat", wl, orthonormal(3, len(wl)))
    with pytest.raises(ValueError, match="inversion windows"):
        TiltBasis(path, wl, [[1000.0, 1100.0]])


# TiltBasis projection


def test_project_roundtrip(basis):
    coeffs = np.array([0.3, -1.2, 0.05])
    assert np.allclose(basis.project(basis.reconstruct(coeffs)), coeffs)


def test_project_ignores_channels_outside_windows(basis):
    """Only inversion-window channels drive the fit."""
    coeffs = np.array([0.3, -1.2, 0.05])
    ratio = basis.reconstruct(coeffs)

    outside = np.setdiff1d(np.arange(len(ratio)), basis.winidx)
    assert len(outside)
    ratio[outside] = 1e6

    assert np.allclose(basis.project(ratio), coeffs)


def test_project_drops_nan_channels(basis):
    coeffs = np.array([0.3, -1.2, 0.05])
    ratio = basis.reconstruct(coeffs)
    ratio[basis.winidx[:5]] = np.nan

    assert np.allclose(basis.project(ratio), coeffs)


def test_project_fill_when_too_few_channels(basis):
    ratio = np.full(len(basis.wl), np.nan)
    ratio[basis.winidx[:2]] = 1.0

    assert np.array_equal(basis.project(ratio), np.full(3, -9999.0))
    assert np.array_equal(basis.project(ratio, fill_value=-1.0), np.full(3, -1.0))


# Basis builder


def synthetic_ratios(wl, n_samples=400, seed=1):
    rng = np.random.default_rng(seed)
    bases = orthonormal(2, len(wl), seed=seed)
    mean = 1.0 + 0.05 * np.cos(wl / 400.0)
    coeffs = rng.normal(size=(n_samples, 2))
    return mean + coeffs @ bases, bases


def test_build_tilt_basis_recovers_subspace(tmp_path, wl):
    """A rank-2 set of ratios gives a 2 component basis spanning the same
    subspace, which round-trips through TiltBasis."""
    X, true_bases = synthetic_ratios(wl)
    X[0] = -9999.0
    X[1, 3] = np.nan
    np.savetxt(tmp_path / "ratios.txt", np.vstack([wl, X]))

    out = tmp_path / "basis.mat"
    build_tilt_basis([str(tmp_path / "ratios.txt")], str(out))

    D = loadmat(out)
    bases = D["bases"]
    assert bases.shape == (2, len(wl))
    assert int(D["n_samples"].squeeze()) == len(X) - 2
    assert np.allclose(true_bases @ bases.T @ bases, true_bases, atol=1e-8)

    tb = TiltBasis(out, wl, Windows)
    assert np.allclose(tb.reconstruct(tb.project(X[5])), X[5])


def test_build_tilt_basis_envi_and_n_components(tmp_path, wl):
    """ENVI cubes of ratio spectra are read, and n_components is honored."""
    X, _ = synthetic_ratios(wl, n_samples=60)
    envi.save_image(
        str(tmp_path / "ratios.hdr"),
        X.reshape(6, 10, len(wl)).astype(np.float32),
        metadata={"wavelength": list(wl)},
        interleave="bip",
        ext="",
        force=True,
    )

    out = tmp_path / "basis.mat"
    build_tilt_basis([str(tmp_path / "ratios")], str(out), n_components=1)
    assert loadmat(out)["bases"].shape == (1, len(wl))


def test_build_tilt_basis_wavelength_mismatch(tmp_path, wl):
    X, _ = synthetic_ratios(wl, n_samples=10)
    np.savetxt(tmp_path / "a.txt", np.vstack([wl, X]))
    np.savetxt(tmp_path / "b.txt", np.vstack([wl + 1.0, X]))

    with pytest.raises(ValueError, match="Wavelengths"):
        build_tilt_basis(
            [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")],
            str(tmp_path / "basis.mat"),
        )


def test_build_tilt_basis_cli(tmp_path, wl):
    X, _ = synthetic_ratios(wl, n_samples=50)
    np.savetxt(tmp_path / "ratios.txt", np.vstack([wl, X]))
    out = tmp_path / "basis.mat"

    result = CliRunner().invoke(
        cli, [str(tmp_path / "ratios.txt"), str(out), "--n_components", "2"]
    )

    assert result.exit_code == 0, result.output
    assert loadmat(out)["bases"].shape == (2, len(wl))
