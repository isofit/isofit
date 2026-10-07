from os.path import basename

import pytest

from isofit.data import env
from isofit.utils.template_construction import Pathnames


def _pathnames(tmp_path, radiance_name):
    return Pathnames(
        input_radiance=str(tmp_path / radiance_name),
        input_loc=str(tmp_path / "loc"),
        input_obs=str(tmp_path / "obs"),
        surface_class_file=None,
        surface_path=None,
        working_directory=str(tmp_path / "work"),
        ray_temp_dir=str(tmp_path / "ray"),
        sensor="NA-20190626",
        use_background_rfl=True,
    )


@pytest.mark.parametrize("radiance_name", ["tile04", "tile04_rdn"])
def test_na_sensor_output_names(tmp_path, monkeypatch, radiance_name):
    """A radiance name already ending in _rdn should not get a second _rdn,
    which previously produced output names like tile04_lbl_lbl."""
    # Only output names are under test, so skip data file lookups
    monkeypatch.setattr(env, "path", lambda *args, **kwargs: tmp_path)

    paths = _pathnames(tmp_path, radiance_name)
    assert paths.fid == radiance_name

    assert basename(paths.rfl_working_path) == "tile04_rfl"
    assert basename(paths.uncert_working_path) == "tile04_uncert"
    assert basename(paths.lbl_working_path) == "tile04_lbl"
    assert basename(paths.state_working_path) == "tile04_state"
    assert basename(paths.h2o_working_path) == "tile04_h2o"
    assert basename(paths.bgrfl_working_path) == "tile04_bgrfl"
    assert basename(paths.atm_presolve) == "tile04_atm_presolve"
