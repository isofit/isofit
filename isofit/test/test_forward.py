from unittest.mock import MagicMock

import numpy as np
import pytest

from isofit.core.common import eps
from isofit.core.forward import ForwardModel

# Radiance calculation tests


def _mock_fm_1c(n, L_atm, s_alb, upward_transm):
    """Return a minimal ForwardModel mock configured for 1-component (Lambertian) mode."""
    fm = MagicMock()
    fm.atmosphere.multipart_transmittance = False
    fm.atmosphere.get_L_atm = MagicMock(return_value=np.full(n, L_atm))
    fm.atmosphere.get_upward_transm = MagicMock(return_value=upward_transm)
    return fm


def test_calc_rdn_lambertian_no_emission():
    """1-component Lambertian RT formula: L = L_atm + L_tot*rho/(1-S*rho)

    With zero surface emission (Ls=0) the standard formula must hold exactly.
    """
    n = 10
    rho = np.full(n, 0.2)
    L_tot = np.full(n, 10.0)
    L_atm_val = 1.0
    s_alb_val = 0.1
    r = {"sphalb": np.full(n, s_alb_val)}

    fm = _mock_fm_1c(n, L_atm_val, s_alb_val, upward_transm=1.0)
    geom = MagicMock()

    result = ForwardModel.calc_rdn(
        fm,
        x_atmosphere=np.zeros(2),
        rho_dir_dir=rho,
        rho_dif_dir=rho,
        rho_dir_dif=rho,
        rho_dif_dif=rho,
        Ls=np.zeros(n),
        L_tot=L_tot,
        L_dir_dir=np.zeros(n),
        L_dif_dir=np.zeros(n),
        L_dir_dif=np.zeros(n),
        L_dif_dif=np.zeros(n),
        r=r,
        geom=geom,
    )

    expected = L_atm_val + L_tot * rho / (1 - s_alb_val * rho)
    assert np.allclose(result, expected)


def test_calc_rdn_lambertian_zero_sphalb():
    """With zero spherical sky albedo the formula reduces to L = L_atm + L_tot*rho."""
    n = 8
    rho = np.linspace(0.05, 0.5, n)
    L_tot = np.full(n, 5.0)
    r = {"sphalb": np.zeros(n)}

    fm = _mock_fm_1c(n, L_atm=2.0, s_alb=0.0, upward_transm=1.0)
    geom = MagicMock()

    result = ForwardModel.calc_rdn(
        fm,
        x_atmosphere=np.zeros(2),
        rho_dir_dir=rho,
        rho_dif_dir=rho,
        rho_dir_dif=rho,
        rho_dif_dif=rho,
        Ls=np.zeros(n),
        L_tot=L_tot,
        L_dir_dir=np.zeros(n),
        L_dif_dir=np.zeros(n),
        L_dir_dif=np.zeros(n),
        L_dif_dif=np.zeros(n),
        r=r,
        geom=geom,
    )

    expected = 2.0 + L_tot * rho
    assert np.allclose(result, expected)


def test_calc_rdn_includes_surface_emission():
    """Surface thermal emission (Ls) is attenuated by upward transmittance and
    added to the radiance signal."""
    n = 6
    rho = np.full(n, 0.0)
    r = {"sphalb": np.zeros(n)}
    Ls = np.full(n, 3.0)
    upward_transm = 0.8

    fm = _mock_fm_1c(n, L_atm=0.0, s_alb=0.0, upward_transm=upward_transm)
    fm.atmosphere.get_upward_transm = MagicMock(return_value=upward_transm)
    geom = MagicMock()

    result = ForwardModel.calc_rdn(
        fm,
        x_atmosphere=np.zeros(2),
        rho_dir_dir=rho,
        rho_dif_dir=rho,
        rho_dir_dif=rho,
        rho_dif_dif=rho,
        Ls=Ls,
        L_tot=np.zeros(n),
        L_dir_dir=np.zeros(n),
        L_dif_dir=np.zeros(n),
        L_dir_dif=np.zeros(n),
        L_dif_dif=np.zeros(n),
        r=r,
        geom=geom,
    )

    expected = Ls * upward_transm
    assert np.allclose(result, expected)


# out_of_bounds tests


def _mock_fm_bounds(idx_atm, bounds_lwr, bounds_upr):
    fm = MagicMock()
    fm.idx_atmosphere = np.asarray(idx_atm, dtype=int)
    fm.bounds = (
        np.asarray(bounds_lwr, dtype=float),
        np.asarray(bounds_upr, dtype=float),
    )
    return fm


def test_out_of_bounds_in_bounds():
    """State vector strictly inside bounds returns False."""
    fm = _mock_fm_bounds(
        idx_atm=[2, 3], bounds_lwr=[0, 0, 0, 0], bounds_upr=[1, 1, 1, 1]
    )
    x = np.array([0.5, 0.5, 0.5, 0.5])
    assert ForwardModel.out_of_bounds(fm, x) is False


def test_out_of_bounds_at_upper():
    """State vector at the upper bound (within eps*2 tolerance) returns True."""
    fm = _mock_fm_bounds(
        idx_atm=[2, 3], bounds_lwr=[0, 0, 0, 0], bounds_upr=[1, 1, 1, 1]
    )
    x = np.array([0.5, 0.5, 1.0 - eps, 0.5])  # atm[0] at upper - eps (within tolerance)
    assert ForwardModel.out_of_bounds(fm, x) is True


def test_out_of_bounds_at_lower():
    """State vector at the lower bound (within eps*2 tolerance) returns True."""
    fm = _mock_fm_bounds(
        idx_atm=[2, 3], bounds_lwr=[0, 0, 0, 0], bounds_upr=[1, 1, 1, 1]
    )
    x = np.array([0.5, 0.5, 0.5, eps])  # atm[1] at lower + eps (within tolerance)
    assert ForwardModel.out_of_bounds(fm, x) is True


# get_L_coupled tests


def _mock_fm_6c(coupling_terms):
    fm = MagicMock()
    fm.atmosphere.coupling_terms = coupling_terms
    fm.atmosphere.rt_mode = "rdn"  # values taken directly from r, no transm_to_rdn call
    return fm


def test_get_L_coupled_nadir_sum():
    """With nadir geometry (coszen=cos_i=1) and full transmittance, L_tot equals
    the sum of the four coupling term radiances."""
    keys = ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]
    fm = _mock_fm_6c(keys)

    vals = [2.0, 3.0, 4.0, 5.0]
    r = {k: np.full(5, v) for k, v in zip(keys, vals)}
    r["transm_down_dir"] = np.ones(5)
    r["sphalb"] = np.zeros(5)

    geom = MagicMock()
    geom.coszen = 1.0
    geom.cos_i = 1.0
    geom.skyview_factor = 1.0
    geom.bg_rfl = None
    fm.terrain_rereflection = MagicMock(return_value=1.0)

    L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif = ForwardModel.get_L_coupled(
        fm, r, geom, rho_dif_dif=0
    )

    assert np.allclose(L_tot, L_dir_dir + L_dif_dir + L_dir_dif + L_dif_dif)
    assert np.allclose(L_tot, sum(vals))


def test_get_L_coupled_cos_i_scaling():
    """Direct-illuminated components scale linearly with cos_i / coszen."""
    keys = ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]
    fm = _mock_fm_6c(keys)

    r = {k: np.ones(4) for k in keys}
    r["transm_down_dir"] = np.zeros(
        4
    )  # zero direct transmittance isolates the L_coupled[0] scaling
    r["sphalb"] = np.zeros(4)

    geom = MagicMock()
    geom.coszen = 1.0
    geom.cos_i = 0.5
    geom.skyview_factor = 1.0
    geom.bg_rfl = None
    fm.terrain_rereflection = MagicMock(return_value=1.0)

    L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif = ForwardModel.get_L_coupled(
        fm, r, geom, rho_dif_dif=0
    )

    # L_dir_dir = r["L_dd"] / coszen * cos_i = 1.0 / 1.0 * 0.5 = 0.5
    assert np.allclose(L_dir_dir, 0.5)
