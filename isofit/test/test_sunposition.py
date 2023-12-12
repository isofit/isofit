"""Tests for ``isofit.core.sunposition``."""


import datetime as dt
from datetime import timezone

import pytest

from isofit.core.sunposition import _sp

# Some tests need a single datetime
UTC_DATETIME = dt.datetime(2023, 12, 12, 15, 15, 55, 453718, tzinfo=timezone.utc)


# Mapping between 'datetime.datetime()' objects and their expected Julian Day.
DATETIME_JULIAN_DAY = {
    # Real timestamp
    dt.datetime(
        2023, 12, 12, 15, 15, 55, 453718, tzinfo=timezone.utc
    ): 2460291.136058492,
    # Start of year
    dt.datetime(2023, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc): 2459945.5,
    # 'julian_day()' makes a decision based on the month
    dt.datetime(2023, 2, 1, 2, 3, 4, 5, tzinfo=timezone.utc): 2459976.585462963,
}


def test__sp():
    sp = _sp()
    assert len(sp._EHL_) == 6
    assert len(sp._EHB_) == 2
    assert len(sp._EHR_) == 5


@pytest.mark.parametrize(
    "timestamp",
    [
        # 'datetime.datetime()'
        *DATETIME_JULIAN_DAY.keys(),
        # Unix 'float' timestamps
        *(d.timestamp() for d in DATETIME_JULIAN_DAY.keys()),
    ],
)
def test__sp_calendar_time(timestamp):
    """Test for ``_sp.calyendar_time()``."""

    # Normalize to a 'datetime.datetime()'.
    if isinstance(timestamp, dt.datetime):
        d = timestamp
    else:
        d = dt.datetime.utcfromtimestamp(timestamp)

    expected = (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond)

    actual = _sp.calendar_time(timestamp)
    assert actual == expected


@pytest.mark.parametrize(
    "timestamp,expected",
    [
        # Test with Python 'datetime.datetime()' objects
        *DATETIME_JULIAN_DAY.items(),
        # Test with Unix 'float' timestamps
        *((d.timestamp(), jd) for d, jd in DATETIME_JULIAN_DAY.items()),
    ],
)
def test__sp_julian_day(timestamp, expected):
    """Test for ``_sp.julian_day()``"""

    actual = _sp.julian_day(timestamp)
    assert actual == expected


@pytest.mark.parametrize(
    "deltat,expected", [(0, 1702394155.453718), (1.2, 1702394155.4537318)]
)
def test__sp_julian_ephemeris_day(deltat, expected):
    """Test for ``_sp.julian_ephemeris_day()``."""

    actual = _sp.julian_ephemeris_day(UTC_DATETIME.timestamp(), deltat)
    assert actual == expected


def test__sp_julian_century():
    """Test for ``_sp.julian_century()``."""

    julian_date = _sp.julian_day(UTC_DATETIME.timestamp())

    actual = _sp.julian_century(julian_date)
    assert actual == 0.23945615492107106


def test__sp_millennium():
    """Test for ``_sp.julian_millennium()``."""

    julian_date = _sp.julian_day(UTC_DATETIME.timestamp())
    julian_century = _sp.julian_century(julian_date)

    actual = _sp.julian_millennium(julian_century)
    assert actual == 0.023945615492107105


def test__sp_heliocentric_longitude():
    """Test for ``_sp.julian_millennium()``."""

    julian_date = _sp.julian_day(UTC_DATETIME.timestamp())
    julian_century = _sp.julian_century(julian_date)
    julian_millennium = _sp.julian_millennium(julian_century)

    actual = _sp.heliocentric_longitude(julian_millennium)
    assert actual == 80.33025439987978


def test__sp_heliocentric_latitude():
    """Test for ``_sp.heliocentric_latitude()``."""

    julian_date = _sp.julian_day(UTC_DATETIME.timestamp())
    julian_century = _sp.julian_century(julian_date)
    julian_millennium = _sp.julian_millennium(julian_century)

    actual = _sp.heliocentric_latitude(julian_millennium)
    assert actual == 9.264457172377275e-05


def test__sp_heliocentric_radius():
    """Test for ``_sp.heliocentric_radius()``."""

    julian_date = _sp.julian_day(UTC_DATETIME.timestamp())
    julian_century = _sp.julian_century(julian_date)
    julian_millennium = _sp.julian_millennium(julian_century)

    actual = _sp.heliocentric_radius(julian_millennium)
    assert actual == 0.9846248845544026


def test__sp_heliocentric_position():
    """Test for ``_sp.heliocentric_position()``."""

    julian_date = _sp.julian_day(UTC_DATETIME.timestamp())
    julian_century = _sp.julian_century(julian_date)
    julian_millennium = _sp.julian_millennium(julian_century)

    actual = _sp.heliocentric_position(julian_millennium)
    expected = (80.33025439987978, 9.264457172377275e-05, 0.9846248845544026)
    assert actual == expected


def test__sp_geocentric_position():
    """Test for ``_sp.geocentric_position()``."""

    julian_date = _sp.julian_day(UTC_DATETIME.timestamp())
    julian_century = _sp.julian_century(julian_date)
    julian_millennium = _sp.julian_millennium(julian_century)
    heliocentric_position = _sp.heliocentric_position(julian_millennium)

    actual = _sp.geocentric_position(heliocentric_position)
    assert actual == (260.3302543998798, -9.264457172377275e-05)


# def test__sp_ecliptic_obliquity():


def test__sp_aberration():
    """Test ``_sp.aberration_correction()``."""

    julian_date = _sp.julian_day(UTC_DATETIME.timestamp())
    julian_century = _sp.julian_century(julian_date)
    julian_millennium = _sp.julian_millennium(julian_century)
    heliocentric_radius = _sp.heliocentric_radius(julian_millennium)

    actual = _sp.aberration_correction(heliocentric_radius)
    assert actual == -0.005780486762414989


# def test__sp_sun_longitude():


# def test__sp_greenwich_sidereal_time():


# def test__sp_sun_ra_decl():


@pytest.mark.parametrize(
    "delta_t,expected",
    [
        (0, (259.3454748782359, -23.196796605532196, -67.5735288887317)),
        (1.23, (259.34549063154344, -23.196797671477352, -67.57354460488149)),
    ],
)
def test__sp_sun_topo_ra_decl_hour(delta_t, expected):
    """Test ``_sp.sun_topo_ra_decl_hour()``."""

    latitude = 34.2017248
    longitude = -118.1740883
    elevation = 532.67

    julian_date = _sp.julian_day(UTC_DATETIME.timestamp())

    actual = _sp.sun_topo_ra_decl_hour(
        latitude, longitude, elevation, julian_date, delta_t
    )
    assert actual == expected


def test__sp_sun_topo_azimuth_zenith():
    """Test for ``_sp.sun_topo_azimuth_zenith()``."""

    latitude = 34.2017248
    longitude = -118.1740883
    elevation = 532.67

    julian_date = _sp.julian_day(UTC_DATETIME.timestamp())
    delta_prime, _, H_prime = _sp.sun_topo_ra_decl_hour(
        latitude,
        longitude,
        elevation,
        julian_date,
    )

    actual = _sp.sun_topo_azimuth_zenith(latitude, delta_prime, H_prime)
    assert actual == (12.464594362220225, 127.66550264908123)


@pytest.mark.skip(
    "_sp.norm_lat_lon() method looks wrong - see: elif lon < 0 or lon > 360:"
)
def test__sp_norm_lat_lon():
    """Test ``_sp.norm_lat_lon()``."""


@pytest.mark.skip("calls 'norm_lat_lon()', which looks wrong")
def test__sp_pos():
    """Test ``_sp.pos()``."""


@pytest.mark.skip(
    "calls 'post()', which in turn calls 'norm_lat_lon()', which looks wrong"
)
def test_sunpos():
    """Test ``sunpos()``."""
