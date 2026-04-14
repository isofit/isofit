import pytest
import numpy as np
import xarray as xr
from scipy.io import savemat
from unittest.mock import MagicMock

from isofit.surface.surface_lut import LUTSurface


@pytest.mark.parametrize("ext", [".nc", ".mat"])
def test_LUTSurface(tmp_path, ext):
    """Try all configuration options of the LUTSurface class"""

    lut_path = tmp_path / f"test_surface{ext}"
    savemat(tmp_path / "surface.mat", {})

    RFL_DATA = 0.8
    RFL_CONIFER = 0.2
    RFL_ROCK = 0.05

    # Define LUT data
    wl = np.arange(350, 2501, 50, dtype=np.float32)
    grids = {
        "grain_size": np.array([30, 1500], dtype=np.float32),
        "algae": np.array([0, 2000], dtype=np.float32),
        "soot": np.array([0, 50], dtype=np.float32),
    }
    rho_lut = np.full((2, 2, 2, len(wl)), RFL_DATA, dtype=np.float32)
    em = {
        "conifer": np.full_like(wl, fill_value=RFL_CONIFER, dtype=np.float32),
        "rock": np.full_like(wl, fill_value=RFL_ROCK, dtype=np.float32),
    }
    names = list(grids.keys())

    # And save for both NetCDF and .MAT data formats
    if ext == ".nc":
        ds = xr.Dataset(
            data_vars={
                "rho_dif_dir": (names + ["wl"], rho_lut),
                "rho_dir_dir": (names + ["wl"], rho_lut),
                "statevec_names": (["n_state"], names),
                **{f"endmember_{k}": (["wl"], v) for k, v in em.items()},
            },
            coords={"wl": wl, **grids},
        )
        ds.to_netcdf(lut_path)
    else:
        savemat(
            lut_path,
            {
                "grids": [[g.reshape(1, -1) for g in grids.values()]],
                "lut_names": names,
                "statevec_names": names,
                "wl": wl.reshape(1, -1),
                "rho_dif_dir": rho_lut,
                "rho_dir_dir": rho_lut,
                **{f"endmember_{k}": v for k, v in em.items()},
            },
        )

    config = MagicMock()
    config.forward_model.surface.surface_lut_file = str(lut_path)
    config.forward_model.surface.surface_file = str(tmp_path / "surface.mat")
    config.forward_model.surface.wavelength_file = None
    config.forward_model.surface.statevector = {}
    config.forward_model.radiative_transfer.terrain_style = "flat"

    surface = LUTSurface(config)
    assert surface.n_state == len(names) + len(em) + 1
    assert surface.bounds[0][0] == 30.0
    assert surface.solve_mixed_pixel == True

    # x_surface: [grain, algae, soot, z-data, z-conifer, z-rock]
    x_surface = np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    geom = MagicMock(cos_i=1.0)

    # Compare to target spectrum, should match exactly
    rho_dir_dir, rho_dif_dir = surface.calc_rfl(x_surface, geom)
    target = np.full_like(wl, fill_value=np.mean([RFL_DATA, RFL_CONIFER, RFL_ROCK]))
    assert np.allclose(rho_dif_dir, target, atol=1e-7)
