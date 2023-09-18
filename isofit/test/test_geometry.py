from isofit.core.geometry import Geometry


def test_Geometry():
    geom = Geometry()
    assert geom.earth_sun_file == None
    assert geom.observer_zenith == 0
    assert geom.observer_azimuth == 0
    assert geom.observer_altitude_km == None
    assert geom.surface_elevation_km == None
    assert geom.latitude == None
    assert geom.longitude == None
    assert geom.earth_sun_distance == None
