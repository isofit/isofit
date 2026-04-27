import numpy as np
import pytest
from unittest.mock import MagicMock

from isofit.core.geometry import Geometry


@pytest.fixture
def input_loc():
    """
    [0] Longitude
    [1] Latitude
    [2] Elevation
    """
    return np.array([-100.0, 80.0, 500.0])


@pytest.fixture
def input_obs():
    """
    [0] Path length
    [1] To-sensor azimuth
    [2] To-sensor zenith
    [3] To-sun azimuth
    [4] To-sun zenith
    [5] Phase angle
    [6] Slope
    [7] Aspect
    [8] Cosine of local solar illumination (cos_i)
    [9] UTC time
    """
    obs = np.zeros(10)
    obs[0] = 2000.0
    obs[1] = 180.0
    obs[2] = 30.0
    obs[3] = 90.0
    obs[4] = 60.0
    obs[8] = 0.5
    return obs


def test_loc(input_loc):
    """Test unit conversion for surface elevation"""
    geom = Geometry(loc=input_loc)

    assert geom.longitude == -100.0
    assert geom.latitude == 80.0
    assert geom.surface_elevation_km == 0.5


def test_obs(input_obs, coszen=0.7):
    """Test unit conversion for path length, and test cos_i to default from OBS data."""
    geom = Geometry(obs=input_obs, coszen=coszen)

    assert geom.path_length_km == 2.0
    assert geom.observer_azimuth == 180.0
    assert geom.observer_zenith == 30.0
    assert geom.solar_azimuth == 90.0
    assert geom.solar_zenith == 60.0
    assert geom.cos_i == np.cos(np.radians(geom.solar_zenith))
    assert geom.coszen == np.cos(np.radians(geom.solar_zenith))


def test_default_behavior():
    """Ensure all initial conditions of Geometry are returned as expected"""
    geom = Geometry()

    assert geom.max_slope == 0.0
    assert geom.terrain_style == "flat"
    assert geom.skyview_factor == 1.0
    assert geom.observer_zenith == None
    assert geom.observer_azimuth == None
    assert geom.solar_zenith == None
    assert geom.solar_azimuth == None
    assert geom.observer_altitude_km == None
    assert geom.surface_elevation_km == None
    assert geom.earth_sun_distance == None
    assert geom.esd_factor == None
    assert geom.earth_sun_distance_reference.shape == (366, 2)
    assert np.all(geom.earth_sun_distance_reference[:, 1] == 1.0)


def test_geom_clipping(input_obs):
    """To ensure cos_i and skyview factor stay within physical bounds"""
    input_obs_small_cosi = input_obs.copy()
    input_obs_small_cosi[4] = 89.99999

    geom = Geometry(obs=input_obs, svf=-0.3)

    assert geom.cos_i >= 0.0
    assert geom.skyview_factor > 0.0


def test_config_terrain_settings(input_obs):
    """Test that max_slope and terrain_style are pulled from full_config"""

    full_config = MagicMock()
    atmosphere = full_config.forward_model.atmosphere
    surface = full_config.forward_model.surface
    surface.max_slope = 20.0
    surface.terrain_style = "dem"
    atmosphere.lut_grid = None

    geom = Geometry(obs=input_obs, full_config=full_config)

    assert geom.max_slope == 20.0
    assert geom.terrain_style == "dem"


def test_coszen_priority(input_obs):
    """Tests how we expect coszen to be defined based on defined priorities in Geometry"""
    full_config = MagicMock()
    atmosphere = full_config.forward_model.atmosphere
    surface = full_config.forward_model.surface
    atmosphere.lut_grid = {"solar_zenith": [10, 20]}
    surface.max_slope = 20.0
    surface.terrain_style = "flat"

    # If user has a lut grid that contains solar_zenith this takes priority
    user_coszen = 0.9
    geom = Geometry(obs=input_obs, coszen=user_coszen, full_config=full_config)

    assert geom.coszen == np.cos(np.radians(geom.solar_zenith))
    assert geom.use_universal_coszen is False

    # And test the other case, where no lut grid is given, it should fall back to coszen
    atmosphere.lut_grid = None
    geom = Geometry(obs=input_obs, coszen=user_coszen, full_config=full_config)

    assert geom.coszen == user_coszen
    assert geom.use_universal_coszen is True
