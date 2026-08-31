"""Parity tests for the batched torch radiance model.

The reference is :class:`isofit.core.forward.ForwardModel`'s own arithmetic,
invoked unbound against a mock in the same style as ``test_forward.py``. That
keeps the comparison on the math itself without requiring a fully configured
ForwardModel (and therefore a downloaded LUT) to run in CI.

Each test drives the scalar path once per pixel and the batched path once for
the whole batch, then requires them to agree.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from isofit.backends.torch.atmosphere import transm_to_rdn as t_transm_to_rdn
from isofit.backends.torch.forward import (
    TorchRadiance,
    calc_rdn_bgrfl_heterogeneous,
    terrain_rereflection_heterogeneous,
)
from isofit.backends.torch.geometry import BatchedGeometry
from isofit.core.forward import ForwardModel
from isofit.core.units import transm_to_rdn as np_transm_to_rdn

pytestmark = pytest.mark.torch_cpu

RTOL = 1e-12
N_WL = 24
B = 16


# --- helpers --------------------------------------------------------------------


def _rand(shape, seed, lo=0.0, hi=1.0):
    return np.random.default_rng(seed).uniform(lo, hi, shape)


def _batched_radiance(multipart, use_background_rfl=False, rt_mode="transm", esd=1.0):
    """A TorchRadiance whose collaborators are mocks, bypassing __init__."""
    tr = TorchRadiance.__new__(TorchRadiance)
    tr.device = torch.device("cpu")
    tr.dtype = torch.float64
    tr.multipart_transmittance = multipart
    tr.use_background_rfl = use_background_rfl

    atm = MagicMock()
    atm.rt_mode = rt_mode
    atm.esd_correction = esd
    atm.multipart_transmittance = multipart
    tr.atmosphere = atm

    from isofit.backends.torch.forward import (
        calc_rdn_bgrfl_heterogeneous,
        calc_rdn_bgrfl_homogeneous,
        terrain_rereflection_heterogeneous,
        terrain_rereflection_homogeneous,
    )

    if use_background_rfl:
        tr.terrain_rereflection = terrain_rereflection_heterogeneous
        tr.calc_rdn_bgrfl = calc_rdn_bgrfl_heterogeneous
    else:
        tr.terrain_rereflection = terrain_rereflection_homogeneous
        tr.calc_rdn_bgrfl = calc_rdn_bgrfl_homogeneous
    return tr


def _scalar_fm(multipart, L_atm, upward_transm):
    """A ForwardModel mock exposing just what calc_rdn touches."""
    fm = MagicMock()
    fm.atmosphere.multipart_transmittance = multipart
    fm.atmosphere.get_L_atm = MagicMock(return_value=L_atm)
    fm.atmosphere.get_upward_transm = MagicMock(return_value=upward_transm)
    return fm


# --- transm_to_rdn --------------------------------------------------------------


def test_transm_to_rdn_matches_scalar():
    transm = _rand((B, N_WL), 0)
    coszen = _rand((B,), 1, 0.2, 1.0)
    solar_irr = _rand((N_WL,), 2, 0.5, 2.0)

    got = t_transm_to_rdn(
        torch.as_tensor(transm),
        torch.as_tensor(coszen),
        torch.as_tensor(solar_irr),
    ).numpy()

    ref = np.stack(
        [np_transm_to_rdn(transm[i], coszen[i], solar_irr) for i in range(B)]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


# --- calc_rdn -------------------------------------------------------------------


@pytest.mark.parametrize("multipart", [False, True])
def test_calc_rdn_matches_scalar(multipart):
    rho_dir_dir = _rand((B, N_WL), 10, 0.01, 0.6)
    rho_dif_dir = _rand((B, N_WL), 11, 0.01, 0.6)
    rho_dir_dif = _rand((B, N_WL), 12, 0.01, 0.6)
    rho_dif_dif = _rand((B, N_WL), 13, 0.01, 0.6)
    Ls = _rand((B, N_WL), 14, 0.0, 0.2)
    L_tot = _rand((B, N_WL), 15, 1.0, 10.0)
    L_dd = _rand((B, N_WL), 16, 0.5, 5.0)
    L_fd = _rand((B, N_WL), 17, 0.5, 5.0)
    L_df = _rand((B, N_WL), 18, 0.5, 5.0)
    L_ff = _rand((B, N_WL), 19, 0.5, 5.0)
    sphalb = _rand((B, N_WL), 20, 0.0, 0.3)
    L_atm = _rand((B, N_WL), 21, 0.1, 2.0)
    transm_up = _rand((B, N_WL), 22, 0.1, 0.9)

    # batched
    tr = _batched_radiance(multipart)
    tr.atmosphere.get_L_atm = MagicMock(return_value=torch.as_tensor(L_atm))
    tr.atmosphere.get_upward_transm = MagicMock(return_value=torch.as_tensor(transm_up))
    got = tr.calc_rdn(
        *[
            torch.as_tensor(a)
            for a in (
                rho_dir_dir,
                rho_dif_dir,
                rho_dir_dif,
                rho_dif_dif,
                Ls,
                L_tot,
                L_dd,
                L_fd,
                L_df,
                L_ff,
            )
        ],
        r={"sphalb": torch.as_tensor(sphalb)},
        geom=MagicMock(),
    ).numpy()

    # scalar, one pixel at a time
    ref = np.stack(
        [
            ForwardModel.calc_rdn(
                _scalar_fm(multipart, L_atm[i], transm_up[i]),
                x_atmosphere=np.zeros(2),
                rho_dir_dir=rho_dir_dir[i],
                rho_dif_dir=rho_dif_dir[i],
                rho_dir_dif=rho_dir_dif[i],
                rho_dif_dif=rho_dif_dif[i],
                Ls=Ls[i],
                L_tot=L_tot[i],
                L_dir_dir=L_dd[i],
                L_dif_dir=L_fd[i],
                L_dir_dif=L_df[i],
                L_dif_dif=L_ff[i],
                r={"sphalb": sphalb[i]},
                geom=MagicMock(),
            )
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


def test_calc_rdn_lambertian_closed_form():
    """1-component mode must reduce to L_atm + L_tot*rho/(1 - S*rho) + L_up."""
    rho = np.full((B, N_WL), 0.2)
    L_tot = np.full((B, N_WL), 10.0)
    sphalb = np.full((B, N_WL), 0.1)
    L_atm = np.full((B, N_WL), 1.0)
    zeros = np.zeros((B, N_WL))

    tr = _batched_radiance(multipart=False)
    tr.atmosphere.get_L_atm = MagicMock(return_value=torch.as_tensor(L_atm))
    tr.atmosphere.get_upward_transm = MagicMock(return_value=torch.as_tensor(zeros))

    got = tr.calc_rdn(
        *[torch.as_tensor(a) for a in (rho, rho, rho, rho, zeros, L_tot)],
        *[torch.as_tensor(zeros) for _ in range(4)],
        r={"sphalb": torch.as_tensor(sphalb)},
        geom=MagicMock(),
    ).numpy()

    expected = L_atm + L_tot * rho / (1 - sphalb * rho)
    np.testing.assert_allclose(got, expected, rtol=RTOL)


def test_calc_rdn_thermal_emission_adds_upward_transmitted_term():
    rho = np.full((B, N_WL), 0.1)
    zeros = np.zeros((B, N_WL))
    Ls = np.full((B, N_WL), 0.5)
    transm_up = np.full((B, N_WL), 0.8)

    tr = _batched_radiance(multipart=False)
    tr.atmosphere.get_L_atm = MagicMock(return_value=torch.as_tensor(zeros))
    tr.atmosphere.get_upward_transm = MagicMock(return_value=torch.as_tensor(transm_up))

    got = tr.calc_rdn(
        *[torch.as_tensor(a) for a in (rho, rho, rho, rho, Ls, zeros)],
        *[torch.as_tensor(zeros) for _ in range(4)],
        r={"sphalb": torch.as_tensor(zeros)},
        geom=MagicMock(),
    ).numpy()

    np.testing.assert_allclose(got, Ls * transm_up, rtol=RTOL)


# --- get_L_coupled --------------------------------------------------------------


def _coupled_inputs(seed=30):
    keys = ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]
    r = {k: _rand((B, N_WL), seed + i, 0.1, 0.9) for i, k in enumerate(keys)}
    r["transm_down_dir"] = _rand((B, N_WL), seed + 10, 0.1, 0.9)
    r["sphalb"] = _rand((B, N_WL), seed + 11, 0.0, 0.3)
    coszen = _rand((B,), seed + 12, 0.3, 1.0)
    cos_i = _rand((B,), seed + 13, 0.3, 1.0)
    svf = _rand((B,), seed + 14, 0.5, 1.0)
    solar_irr = _rand((N_WL,), seed + 15, 0.5, 2.0)
    return keys, r, coszen, cos_i, svf, solar_irr


@pytest.mark.parametrize("rt_mode", ["transm", "rdn"])
@pytest.mark.parametrize("with_bgrfl", [False, True])
def test_get_L_coupled_matches_scalar(rt_mode, with_bgrfl):
    from isofit.backends.torch.atmosphere import TorchAtmosphere

    keys, r, coszen, cos_i, svf, solar_irr = _coupled_inputs()
    rho_dif_dif = _rand((B, N_WL), 99, 0.01, 0.5) if with_bgrfl else 0.0

    # batched
    ta = TorchAtmosphere.__new__(TorchAtmosphere)
    ta.device = torch.device("cpu")
    ta.dtype = torch.float64
    ta.rt_mode = rt_mode
    ta.coupling_terms = keys
    ta.solar_irr = torch.as_tensor(solar_irr)

    geom_b = BatchedGeometry(
        {"coszen": coszen, "cos_i": cos_i, "skyview_factor": svf}
    )
    got = ta.get_L_coupled(
        {k: torch.as_tensor(v) for k, v in r.items()},
        geom_b,
        rho_dif_dif=torch.as_tensor(rho_dif_dif) if with_bgrfl else 0.0,
        terrain_rereflection=None,
    )

    # scalar
    refs = []
    for i in range(B):
        fm = MagicMock()
        fm.atmosphere.coupling_terms = keys
        fm.atmosphere.rt_mode = rt_mode
        fm.atmosphere.solar_irr = solar_irr
        fm.terrain_rereflection = MagicMock(return_value=1.0)
        geom = MagicMock()
        geom.coszen = coszen[i]
        geom.cos_i = cos_i[i]
        geom.skyview_factor = svf[i]
        refs.append(
            ForwardModel.get_L_coupled(
                fm,
                {k: v[i] for k, v in r.items()},
                geom,
                rho_dif_dif=(rho_dif_dif[i] if with_bgrfl else 0),
            )
        )

    for term in range(5):
        np.testing.assert_allclose(
            got[term].numpy(),
            np.stack([ref[term] for ref in refs]),
            rtol=RTOL,
            err_msg=f"coupled term {term} diverged",
        )


def test_get_L_coupled_with_real_terrain_rereflection():
    """Composition check: the coupled terms and terrain re-reflection together.

    The parametrized test above stubs terrain re-reflection on both sides, so it
    validates the coupled-term algebra but not how the two compose. This uses
    the real heterogeneous function on both paths.
    """
    from isofit.backends.torch.atmosphere import TorchAtmosphere

    keys, r, coszen, cos_i, svf, solar_irr = _coupled_inputs(seed=80)
    rho_dif_dif = _rand((B, N_WL), 98, 0.01, 0.5)

    ta = TorchAtmosphere.__new__(TorchAtmosphere)
    ta.device = torch.device("cpu")
    ta.dtype = torch.float64
    ta.rt_mode = "transm"
    ta.coupling_terms = keys
    ta.solar_irr = torch.as_tensor(solar_irr)

    geom_b = BatchedGeometry({"coszen": coszen, "cos_i": cos_i, "skyview_factor": svf})
    got = ta.get_L_coupled(
        {k: torch.as_tensor(v) for k, v in r.items()},
        geom_b,
        rho_dif_dif=torch.as_tensor(rho_dif_dif),
        terrain_rereflection=terrain_rereflection_heterogeneous,
    )

    refs = []
    for i in range(B):
        fm = MagicMock()
        fm.atmosphere.coupling_terms = keys
        fm.atmosphere.rt_mode = "transm"
        fm.atmosphere.solar_irr = solar_irr
        geom = MagicMock()
        geom.coszen = coszen[i]
        geom.cos_i = cos_i[i]
        geom.skyview_factor = svf[i]
        # Bind the real scalar implementation, not a stub.
        fm.terrain_rereflection = (
            lambda rho_dif_dif, geom, _g=geom: ForwardModel.terrain_rereflection_heterogeneous(
                MagicMock(), rho_dif_dif=rho_dif_dif, geom=_g
            )
        )
        refs.append(
            ForwardModel.get_L_coupled(
                fm, {k: v[i] for k, v in r.items()}, geom, rho_dif_dif=rho_dif_dif[i]
            )
        )

    for term in range(5):
        np.testing.assert_allclose(
            got[term].numpy(),
            np.stack([ref[term] for ref in refs]),
            rtol=RTOL,
            err_msg=f"coupled term {term} diverged with terrain re-reflection",
        )


# --- terrain re-reflection and background radiance -------------------------------


def test_terrain_rereflection_matches_scalar():
    rho = _rand((B, N_WL), 40, 0.01, 0.5)
    svf = _rand((B,), 41, 0.4, 0.95)

    geom_b = BatchedGeometry({"skyview_factor": svf})
    got = terrain_rereflection_heterogeneous(torch.as_tensor(rho), geom_b).numpy()

    ref = np.stack(
        [
            ForwardModel.terrain_rereflection_heterogeneous(
                MagicMock(), rho_dif_dif=rho[i], geom=_geom_with_svf(svf[i])
            )
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


def _geom_with_svf(value):
    geom = MagicMock()
    geom.skyview_factor = value
    return geom


def test_calc_rdn_bgrfl_heterogeneous_matches_scalar():
    rho_dir_dif = _rand((B, N_WL), 50, 0.01, 0.5)
    rho_dif_dif = _rand((B, N_WL), 51, 0.01, 0.5)
    L_dir_dif = _rand((B, N_WL), 52, 0.5, 5.0)
    L_dif_dif = _rand((B, N_WL), 53, 0.5, 5.0)
    L_tot = _rand((B, N_WL), 54, 1.0, 10.0)
    s_alb = _rand((B, N_WL), 55, 0.0, 0.3)

    got = calc_rdn_bgrfl_heterogeneous(
        *[
            torch.as_tensor(a)
            for a in (rho_dir_dif, rho_dif_dif, L_dir_dif, L_dif_dif, L_tot, s_alb)
        ]
    ).numpy()

    ref = np.stack(
        [
            ForwardModel.calc_rdn_bgrfl_heterogeneous(
                MagicMock(),
                rho_dir_dif=rho_dir_dif[i],
                rho_dif_dif=rho_dif_dif[i],
                L_dir_dif=L_dir_dif[i],
                L_dif_dif=L_dif_dif[i],
                L_tot=L_tot[i],
                s_alb=s_alb[i],
            )
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


# --- upward transmittance --------------------------------------------------------


def test_get_upward_transm_sums_components():
    from isofit.backends.torch.atmosphere import TorchAtmosphere

    ta = TorchAtmosphere.__new__(TorchAtmosphere)
    up_dir = _rand((B, N_WL), 60, 0.1, 0.4)
    up_dif = _rand((B, N_WL), 61, 0.1, 0.4)
    r = {
        "rhoatm": torch.zeros((B, N_WL), dtype=torch.float64),
        "transm_up_dir": torch.as_tensor(up_dir),
        "transm_up_dif": torch.as_tensor(up_dif),
    }
    np.testing.assert_allclose(
        ta.get_upward_transm(r).numpy(), up_dir + up_dif, rtol=RTOL
    )


def test_get_upward_transm_zero_for_one_component():
    """1-component LUTs have no upward transmittance keys; Ls must drop out."""
    from isofit.backends.torch.atmosphere import TorchAtmosphere

    ta = TorchAtmosphere.__new__(TorchAtmosphere)
    r = {"rhoatm": torch.ones((B, N_WL), dtype=torch.float64)}
    out = ta.get_upward_transm(r)
    assert out.shape == (B, N_WL)
    assert torch.all(out == 0)


def test_get_upward_transm_rejects_unphysical_values():
    """Guards the same physical bound as the scalar path (max 1.05)."""
    from isofit.backends.torch.atmosphere import TorchAtmosphere

    ta = TorchAtmosphere.__new__(TorchAtmosphere)
    r = {
        "rhoatm": torch.zeros((B, N_WL), dtype=torch.float64),
        "transm_up_dir": torch.full((B, N_WL), 0.9, dtype=torch.float64),
        "transm_up_dif": torch.full((B, N_WL), 0.9, dtype=torch.float64),
    }
    with pytest.raises(ValueError, match="greater than 1.05"):
        ta.get_upward_transm(r)


# --- L_atm ----------------------------------------------------------------------


@pytest.mark.parametrize("rt_mode", ["transm", "rdn"])
def test_get_L_atm_matches_scalar(rt_mode):
    from isofit.backends.torch.atmosphere import TorchAtmosphere

    rhoatm = _rand((B, N_WL), 70, 0.01, 0.4)
    coszen = _rand((B,), 71, 0.3, 1.0)
    solar_irr = _rand((N_WL,), 72, 0.5, 2.0)
    esd = 1.03

    ta = TorchAtmosphere.__new__(TorchAtmosphere)
    ta.rt_mode = rt_mode
    ta.esd_correction = esd
    ta.solar_irr = torch.as_tensor(solar_irr)

    geom_b = BatchedGeometry({"coszen": coszen})
    got = ta.get_L_atm({"rhoatm": torch.as_tensor(rhoatm)}, geom_b).numpy()

    if rt_mode == "rdn":
        ref = rhoatm * esd
    else:
        ref = (
            np.stack([np_transm_to_rdn(rhoatm[i], coszen[i], solar_irr) for i in range(B)])
            * esd
        )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


# --- construction against a Reader-shaped atmosphere -------------------------------


def test_torch_atmosphere_constructs_from_reader_style_object():
    """TorchAtmosphere must find interpolators where BaseAtmosphere keeps them.

    BaseAtmosphere subclasses the LUT Reader, so `interpolators` is an attribute
    of the atmosphere itself while `.lut` is the xarray Dataset it was built
    from. Reaching for `atmosphere.lut.interpolators` raised AttributeError only
    once a real scene was run -- the other tests here bypass __init__, so they
    could not catch it.
    """
    from isofit.backends.torch.atmosphere import TorchAtmosphere
    from isofit.core.common import VectorInterpolator

    grids = [np.linspace(0, 1, 4), np.linspace(0, 1, 3)]
    shape = tuple(len(g) for g in grids)
    rng = np.random.default_rng(0)

    atm = MagicMock()
    atm.interpolators = {
        k: VectorInterpolator(list(grids), rng.normal(size=(*shape, N_WL)), "mlg_numba")
        for k in ("rhoatm", "sphalb")
    }
    atm.rt_mode = "transm"
    atm.multipart_transmittance = False
    atm.coupling_terms = []
    atm.esd_correction = 1.0
    atm.solar_irr = np.linspace(0.5, 2.0, N_WL)
    atm.lut_names = ["H2OSTR", "AOT550"]
    atm.indices.x_RT = [0, 1]
    atm.indices.geom = {}
    atm.indices.convert_observer_zenith = None

    ta = TorchAtmosphere(atm)
    assert ta.lut.dims == 2
    assert set(ta.lut.keys) == {"rhoatm", "sphalb"}

    out = ta.get(torch.as_tensor(np.array([[0.5, 0.5], [0.2, 0.8]])), None)
    assert out["rhoatm"].shape == (2, N_WL)


def test_torch_atmosphere_without_interpolators_raises_clearly():
    """A LUT built with build_interpolators=False must fail with a useful message."""
    from isofit.backends.torch.atmosphere import TorchAtmosphere

    atm = MagicMock()
    atm.interpolators = {}
    with pytest.raises(ValueError, match="no LUT interpolators"):
        TorchAtmosphere(atm)
