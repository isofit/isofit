from datetime import datetime
from os.path import basename
from unittest.mock import MagicMock

import pytest

import isofit.utils.template_construction as tmpl
from isofit.data import env
from isofit.utils.template_construction import Pathnames, get_sensor_metadata_from_fid


def _pathnames(tmp_path, radiance_name, sensor):
    return Pathnames(
        input_radiance=str(tmp_path / radiance_name),
        input_loc=str(tmp_path / "loc"),
        input_obs=str(tmp_path / "obs"),
        surface_class_file=None,
        surface_path=None,
        working_directory=str(tmp_path / "work"),
        ray_temp_dir=str(tmp_path / "ray"),
        sensor=sensor,
        use_background_rfl=True,
    )


@pytest.mark.parametrize("sensor", ["custom", "NA-20190626"])
@pytest.mark.parametrize("radiance_name", ["tile04", "tile04_rdn"])
def test_custom_sensor_output_names(tmp_path, monkeypatch, radiance_name, sensor):
    """A trailing _rdn is dropped from the fid, so outputs are not named
    like tile04_lbl_lbl."""
    # Only output names are under test, so skip data file lookups
    monkeypatch.setattr(env, "path", lambda *args, **kwargs: tmp_path)

    paths = _pathnames(tmp_path, radiance_name, sensor)
    assert paths.fid == "tile04"

    assert basename(paths.rfl_working_path) == "tile04_rfl"
    assert basename(paths.uncert_working_path) == "tile04_uncert"
    assert basename(paths.lbl_working_path) == "tile04_lbl"
    assert basename(paths.state_working_path) == "tile04_state"
    assert basename(paths.h2o_working_path) == "tile04_h2o"
    assert basename(paths.bgrfl_working_path) == "tile04_bgrfl"
    assert basename(paths.atm_presolve) == "tile04_atm_presolve"


@pytest.mark.parametrize("sensor", ["custom", "NA-20190626"])
@pytest.mark.parametrize(
    "fid, datestr, expected",
    [
        (
            "emit20220810t034528_o22222_s000",
            "emit%Y%m%dt%H%M%S",
            datetime(2022, 8, 10, 3, 45, 28),
        ),
        (
            "CHEESEHEAD20190830-26_tile08",
            "CHEESEHEAD%Y%m%d",
            datetime(2019, 8, 30),
        ),
        ("20190830", "%Y%m%d", datetime(2019, 8, 30)),
    ],
)
def test_custom_sensor_datestr(sensor, fid, datestr, expected):
    dt, _ = get_sensor_metadata_from_fid(sensor, fid, datestr)
    assert dt == expected


def test_na_sensor_requires_datestr():
    with pytest.raises(ValueError, match="datestr is required"):
        get_sensor_metadata_from_fid("NA-20190626", "anything")


def test_custom_sensor_requires_datestr():
    with pytest.raises(ValueError, match="datestr is required"):
        get_sensor_metadata_from_fid("custom", "tile04")


def test_custom_sensor_datestr_no_match():
    with pytest.raises(ValueError, match="does not start with a date"):
        get_sensor_metadata_from_fid("custom", "tile04", "emit%Y%m%dt%H%M%S")


def test_tilt_pathnames(tmp_path, monkeypatch):
    """The tilt basis is staged into the data directory, and the tilt output
    path is kept as given."""
    monkeypatch.setattr(env, "path", lambda *args, **kwargs: tmp_path)

    paths = _pathnames(tmp_path, "tile04", "custom")
    assert paths.tilt_basis_working_path is None
    assert paths.tilt_coefficients_working_path is None

    paths = Pathnames(
        input_radiance=str(tmp_path / "tile04"),
        input_loc=str(tmp_path / "loc"),
        input_obs=str(tmp_path / "obs"),
        surface_class_file=None,
        surface_path=None,
        working_directory=str(tmp_path / "work"),
        ray_temp_dir=str(tmp_path / "ray"),
        sensor="custom",
        tilt_basis_file=str(tmp_path / "my_basis.mat"),
        output_tilt_coefficients_file=str(tmp_path / "out" / "tile04_tilt"),
    )
    assert paths.tilt_basis_working_path == str(
        tmp_path / "work" / "data" / "tilt_basis.mat"
    )
    assert paths.tilt_coefficients_working_path == str(tmp_path / "out" / "tile04_tilt")


def test_make_output_config_tilt():
    assert "tilt_coefficients_file" not in tmpl.make_output_config("state")
    output = tmpl.make_output_config("state", tilt_output_path="tilt")
    assert output["tilt_coefficients_file"] == "tilt"


@pytest.mark.parametrize(
    "use_superpixels, presolve, tilt_output, tilt_basis",
    [
        (False, False, "tilt", "basis.mat"),
        (True, False, None, "basis.mat"),
        (False, True, None, None),
        (True, True, None, None),
    ],
)
def test_build_config_tilt_paths(
    monkeypatch, use_superpixels, presolve, tilt_output, tilt_basis
):
    """Only the full-resolution main solve writes tilt coefficients; every
    non-presolve config carries the basis for the line algorithms."""
    captured = {}

    def capture(name):
        def make(**kwargs):
            captured[name] = kwargs
            return {}

        return make

    for name in [
        "make_input_config",
        "make_output_config",
        "make_instrument_config",
        "make_atmosphere_config",
        "make_surface_config",
        "make_implementation_config",
    ]:
        monkeypatch.setattr(tmpl, name, capture(name))

    class Validated(Exception):
        pass

    def stop(config):
        raise Validated

    monkeypatch.setattr(tmpl, "_validate_config", stop)

    paths = MagicMock()
    paths.tilt_coefficients_working_path = "tilt"
    paths.tilt_basis_working_path = "basis.mat"
    paths.input_model_discrepancy_path = None

    with pytest.raises(Validated):
        tmpl.build_config(paths, use_superpixels=use_superpixels, presolve=presolve)

    assert captured["make_output_config"]["tilt_output_path"] == tilt_output
    assert captured["make_atmosphere_config"]["tilt_basis_path"] == tilt_basis
