from isofit.core.geometry import Geometry


def test_Geometry():
    geom = Geometry()
    assert geom.observer_zenith == 0
    assert geom.observer_azimuth == 0
    assert geom.observer_altitude_km is None
    assert geom.surface_elevation_km is None
    assert geom.latitude is None
    assert geom.longitude is None
    assert geom.earth_sun_distance is None
