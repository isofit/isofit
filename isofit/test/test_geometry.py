from isofit.core.geometry import Geometry


def test_Geometry():
    geom = Geometry()
    assert geom.earth_sun_file == None
    assert geom.observer_zenith == 0
    assert geom.observer_azimuth == 0
    assert geom.observer_altitude_km == None
    assert geom.surface_elevation_km == None
    assert geom.datetime == None
    assert geom.day_of_year == None
    assert geom.latitude == None
    assert geom.longitude == None
    assert geom.longitudeE == None
    assert geom.gmtime == None
    assert geom.earth_sun_distance == None
    assert geom.OBSZEN == 180.0
    assert geom.RELAZ == 0.0
    assert geom.TRUEAZ == 0.0
    assert geom.umu == 1.0
