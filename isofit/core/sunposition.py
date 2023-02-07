#! /usr/bin/env python3
#
# ISOFIT redistributes this version of sunposition.py for ease of use and
# and compatibility under the terms of The MIT License (MIT):
#
# The MIT License (MIT)
#
# Copyright (c) 2016 Samuel Bear Powell
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from datetime import datetime

import numpy as np


class _sp:
    """."""

    @staticmethod
    def calendar_time(dt):
        """."""

        try:
            x = dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond
            return x
        except AttributeError:
            try:
                # will raise OSError if dt is not acceptable
                return _sp.calendar_time(datetime.utcfromtimestamp(dt))
            except BaseException:
                raise TypeError("dt must be datetime object or POSIX timestamp")

    @staticmethod
    def julian_day(dt):
        """Calculate the Julian Day from a datetime.datetime object in UTC."""

        # year and month numbers
        yr, mo, dy, hr, mn, sc, us = _sp.calendar_time(dt)
        if mo <= 2:  # From paper: "if M = 1 or 2, then Y = Y - 1 and M = M + 12"
            mo += 12
            yr -= 1
        # day of the month with decimal time
        dy = (
            dy
            + hr / 24.0
            + mn / (24.0 * 60.0)
            + sc / (24.0 * 60.0 * 60.0)
            + us / (24.0 * 60.0 * 60.0 * 1e6)
        )
        # b is equal to 0 for the julian calendar and is equal to (2- A +
        # INT(A/4)), A = INT(Y/100), for the gregorian calendar
        a = int(yr / 100)
        b = 2 - a + int(a / 4)
        jd = int(365.25 * (yr + 4716)) + int(30.6001 * (mo + 1)) + dy + b - 1524.5
        return jd

    @staticmethod
    def julian_ephemeris_day(jd, deltat):
        """Calculate the Julian Ephemeris Day from the Julian Day and delta-time = (terrestrial time - universal time) in seconds."""

        return jd + deltat / 86400.0

    @staticmethod
    def julian_century(jd):
        """Caluclate the Julian Century from Julian Day or Julian Ephemeris Day."""

        return (jd - 2451545.0) / 36525.0

    @staticmethod
    def julian_millennium(jc):
        """Calculate the Julian Millennium from Julian Ephemeris Century."""

        return jc / 10.0

    # Earth Periodic Terms
    # Earth Heliocentric Longitude coefficients (L0, L1, L2, L3, L4, and L5 in paper)
    _EHL_ = [  # L0:
        [
            (175347046, 0.0, 0.0),
            (3341656, 4.6692568, 6283.07585),
            (34894, 4.6261, 12566.1517),
            (3497, 2.7441, 5753.3849),
            (3418, 2.8289, 3.5231),
            (3136, 3.6277, 77713.7715),
            (2676, 4.4181, 7860.4194),
            (2343, 6.1352, 3930.2097),
            (1324, 0.7425, 11506.7698),
            (1273, 2.0371, 529.691),
            (1199, 1.1096, 1577.3435),
            (990, 5.233, 5884.927),
            (902, 2.045, 26.298),
            (857, 3.508, 398.149),
            (780, 1.179, 5223.694),
            (753, 2.533, 5507.553),
            (505, 4.583, 18849.228),
            (492, 4.205, 775.523),
            (357, 2.92, 0.067),
            (317, 5.849, 11790.629),
            (284, 1.899, 796.298),
            (271, 0.315, 10977.079),
            (243, 0.345, 5486.778),
            (206, 4.806, 2544.314),
            (205, 1.869, 5573.143),
            (202, 2.4458, 6069.777),
            (156, 0.833, 213.299),
            (132, 3.411, 2942.463),
            (126, 1.083, 20.775),
            (115, 0.645, 0.98),
            (103, 0.636, 4694.003),
            (102, 0.976, 15720.839),
            (102, 4.267, 7.114),
            (99, 6.21, 2146.17),
            (98, 0.68, 155.42),
            (86, 5.98, 161000.69),
            (85, 1.3, 6275.96),
            (85, 3.67, 71430.7),
            (80, 1.81, 17260.15),
            (79, 3.04, 12036.46),
            (71, 1.76, 5088.63),
            (74, 3.5, 3154.69),
            (74, 4.68, 801.82),
            (70, 0.83, 9437.76),
            (62, 3.98, 8827.39),
            (61, 1.82, 7084.9),
            (57, 2.78, 6286.6),
            (56, 4.39, 14143.5),
            (56, 3.47, 6279.55),
            (52, 0.19, 12139.55),
            (52, 1.33, 1748.02),
            (51, 0.28, 5856.48),
            (49, 0.49, 1194.45),
            (41, 5.37, 8429.24),
            (41, 2.4, 19651.05),
            (39, 6.17, 10447.39),
            (37, 6.04, 10213.29),
            (37, 2.57, 1059.38),
            (36, 1.71, 2352.87),
            (36, 1.78, 6812.77),
            (33, 0.59, 17789.85),
            (30, 0.44, 83996.85),
            (30, 2.74, 1349.87),
            (25, 3.16, 4690.48),
        ],
        # L1:
        [
            (628331966747, 0.0, 0.0),
            (206059, 2.678235, 6283.07585),
            (4303, 2.6351, 12566.1517),
            (425, 1.59, 3.523),
            (119, 5.796, 26.298),
            (109, 2.966, 1577.344),
            (93, 2.59, 18849.23),
            (72, 1.14, 529.69),
            (68, 1.87, 398.15),
            (67, 4.41, 5507.55),
            (59, 2.89, 5223.69),
            (56, 2.17, 155.42),
            (45, 0.4, 796.3),
            (36, 0.47, 775.52),
            (29, 2.65, 7.11),
            (21, 5.34, 0.98),
            (19, 1.85, 5486.78),
            (19, 4.97, 213.3),
            (17, 2.99, 6275.96),
            (16, 0.03, 2544.31),
            (16, 1.43, 2146.17),
            (15, 1.21, 10977.08),
            (12, 2.83, 1748.02),
            (12, 3.26, 5088.63),
            (12, 5.27, 1194.45),
            (12, 2.08, 4694),
            (11, 0.77, 553.57),
            (10, 1.3, 3286.6),
            (10, 4.24, 1349.87),
            (9, 2.7, 242.73),
            (9, 5.64, 951.72),
            (8, 5.3, 2352.87),
            (6, 2.65, 9437.76),
            (6, 4.67, 4690.48),
        ],
        # L2:
        [
            (52919, 0.0, 0.0),
            (8720, 1.0721, 6283.0758),
            (309, 0.867, 12566.152),
            (27, 0.05, 3.52),
            (16, 5.19, 26.3),
            (16, 3.68, 155.42),
            (10, 0.76, 18849.23),
            (9, 2.06, 77713.77),
            (7, 0.83, 775.52),
            (5, 4.66, 1577.34),
            (4, 1.03, 7.11),
            (4, 3.44, 5573.14),
            (3, 5.14, 796.3),
            (3, 6.05, 5507.55),
            (3, 1.19, 242.73),
            (3, 6.12, 529.69),
            (3, 0.31, 398.15),
            (3, 2.28, 553.57),
            (2, 4.38, 5223.69),
            (2, 3.75, 0.98),
        ],
        # L3:
        [
            (289, 5.844, 6283.076),
            (
                35,
                0.0,
                0.0,
            ),
            (17, 5.49, 12566.15),
            (3, 5.2, 155.42),
            (1, 4.72, 3.52),
            (1, 5.3, 18849.23),
            (1, 5.97, 242.73),
        ],
        # L4:
        [(114, 3.142, 0.0), (8, 4.13, 6283.08), (1, 3.84, 12566.15)],
        # L5:
        [(1, 3.14, 0.0)],
    ]

    # Earth Heliocentric Longitude coefficients (B0 and B1 in paper)
    _EHB_ = [  # B0:
        [
            (280, 3.199, 84334.662),
            (102, 5.422, 5507.553),
            (80, 3.88, 5223.69),
            (44, 3.7, 2352.87),
            (32, 4.0, 1577.34),
        ],
        # B1:
        [(9, 3.9, 5507.55), (6, 1.73, 5223.69)],
    ]

    # Earth Heliocentric Radius coefficients (R0, R1, R2, R3, R4)
    _EHR_ = [  # R0:
        [
            (100013989, 0.0, 0.0),
            (1670700, 3.0984635, 6283.07585),
            (13956, 3.05525, 12566.1517),
            (3084, 5.1985, 77713.7715),
            (1628, 1.1739, 5753.3849),
            (1576, 2.8469, 7860.4194),
            (925, 5.453, 11506.77),
            (542, 4.564, 3930.21),
            (472, 3.661, 5884.927),
            (346, 0.964, 5507.553),
            (329, 5.9, 5223.694),
            (307, 0.299, 5573.143),
            (243, 4.273, 11790.629),
            (212, 5.847, 1577.344),
            (186, 5.022, 10977.079),
            (175, 3.012, 18849.228),
            (110, 5.055, 5486.778),
            (98, 0.89, 6069.78),
            (86, 5.69, 15720.84),
            (86, 1.27, 161000.69),
            (85, 0.27, 17260.15),
            (63, 0.92, 529.69),
            (57, 2.01, 83996.85),
            (56, 5.24, 71430.7),
            (49, 3.25, 2544.31),
            (47, 2.58, 775.52),
            (45, 5.54, 9437.76),
            (43, 6.01, 6275.96),
            (39, 5.36, 4694),
            (38, 2.39, 8827.39),
            (37, 0.83, 19651.05),
            (37, 4.9, 12139.55),
            (36, 1.67, 12036.46),
            (35, 1.84, 2942.46),
            (33, 0.24, 7084.9),
            (32, 0.18, 5088.63),
            (32, 1.78, 398.15),
            (28, 1.21, 6286.6),
            (28, 1.9, 6279.55),
            (26, 4.59, 10447.39),
        ],
        # R1:
        [
            (103019, 1.10749, 6283.07585),
            (1721, 1.0644, 12566.1517),
            (702, 3.142, 0.0),
            (32, 1.02, 18849.23),
            (31, 2.84, 5507.55),
            (25, 1.32, 5223.69),
            (18, 1.42, 1577.34),
            (10, 5.91, 10977.08),
            (9, 1.42, 6275.96),
            (9, 0.27, 5486.78),
        ],
        # R2:
        [
            (4359, 5.7846, 6283.0758),
            (124, 5.579, 12566.152),
            (12, 3.14, 0.0),
            (9, 3.63, 77713.77),
            (6, 1.87, 5573.14),
            (3, 5.47, 18849),
        ],
        # R3:
        [(145, 4.273, 6283.076), (7, 3.92, 12566.15)],
        # R4:
        [(4, 2.56, 6283.08)],
    ]

    @staticmethod
    def heliocentric_longitude(jme):
        """Compute the Earth Heliocentric Longitude (L) in degrees given the Julian Ephemeris Millennium."""

        # L5, ..., L0
        Li = [
            sum(a * np.cos(b + c * jme) for a, b, c in abcs)
            for abcs in reversed(_sp._EHL_)
        ]
        L = np.polyval(Li, jme) / 1e8
        L = np.rad2deg(L) % 360
        return L

    @staticmethod
    def heliocentric_latitude(jme):
        """Compute the Earth Heliocentric Latitude (B) in degrees given the Julian Ephemeris Millennium."""

        Bi = [
            sum(a * np.cos(b + c * jme) for a, b, c in abcs)
            for abcs in reversed(_sp._EHB_)
        ]
        B = np.polyval(Bi, jme) / 1e8
        B = np.rad2deg(B) % 360
        return B

    @staticmethod
    def heliocentric_radius(jme):
        """Compute the Earth Heliocentric Radius (R) in astronimical units given the Julian Ephemeris Millennium."""

        Ri = [
            sum(a * np.cos(b + c * jme) for a, b, c in abcs)
            for abcs in reversed(_sp._EHR_)
        ]
        R = np.polyval(Ri, jme) / 1e8
        return R

    @staticmethod
    def heliocentric_position(jme):
        """Compute the Earth Heliocentric Longitude, Latitude, and Radius given the Julian Ephemeris Millennium.

        Returns (L, B, R) where L = longitude in degrees, B = latitude in degrees, and R = radius in astronimical units.
        """

        return (
            _sp.heliocentric_longitude(jme),
            _sp.heliocentric_latitude(jme),
            _sp.heliocentric_radius(jme),
        )

    @staticmethod
    def geocentric_position(helio_pos):
        """Compute the geocentric latitude (Theta) and longitude (beta) (in degrees) of the sun given Earth's heliocentric position (L, B, R)."""

        L, B, R = helio_pos
        th = L + 180
        b = -B
        return (th, b)

    # Nutation Longitude and Obliquity coefficients (Y)
    _NLOY_ = [
        (0, 0, 0, 0, 1),
        (-2, 0, 0, 2, 2),
        (0, 0, 0, 2, 2),
        (0, 0, 0, 0, 2),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (-2, 1, 0, 2, 2),
        (0, 0, 0, 2, 1),
        (0, 0, 1, 2, 2),
        (-2, -1, 0, 2, 2),
        (-2, 0, 1, 0, 0),
        (-2, 0, 0, 2, 1),
        (0, 0, -1, 2, 2),
        (2, 0, 0, 0, 0),
        (0, 0, 1, 0, 1),
        (2, 0, -1, 2, 2),
        (0, 0, -1, 0, 1),
        (0, 0, 1, 2, 1),
        (-2, 0, 2, 0, 0),
        (0, 0, -2, 2, 1),
        (2, 0, 0, 2, 2),
        (0, 0, 2, 2, 2),
        (0, 0, 2, 0, 0),
        (-2, 0, 1, 2, 2),
        (0, 0, 0, 2, 0),
        (-2, 0, 0, 2, 0),
        (0, 0, -1, 2, 1),
        (0, 2, 0, 0, 0),
        (2, 0, -1, 0, 1),
        (-2, 2, 0, 2, 2),
        (0, 1, 0, 0, 1),
        (-2, 0, 1, 0, 1),
        (0, -1, 0, 0, 1),
        (0, 0, 2, -2, 0),
        (2, 0, -1, 2, 1),
        (2, 0, 1, 2, 2),
        (0, 1, 0, 2, 2),
        (-2, 1, 1, 0, 0),
        (0, -1, 0, 2, 2),
        (2, 0, 0, 2, 1),
        (2, 0, 1, 0, 0),
        (-2, 0, 2, 2, 2),
        (-2, 0, 1, 2, 1),
        (2, 0, -2, 0, 1),
        (2, 0, 0, 0, 1),
        (0, -1, 1, 0, 0),
        (-2, -1, 0, 2, 1),
        (-2, 0, 0, 0, 1),
        (0, 0, 2, 2, 1),
        (-2, 0, 2, 0, 1),
        (-2, 1, 0, 2, 1),
        (0, 0, 1, -2, 0),
        (-1, 0, 1, 0, 0),
        (-2, 1, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (0, 0, 1, 2, 0),
        (0, 0, -2, 2, 2),
        (-1, -1, 1, 0, 0),
        (0, 1, 1, 0, 0),
        (0, -1, 1, 2, 2),
        (2, -1, -1, 2, 2),
        (0, 0, 3, 2, 2),
        (2, -1, 0, 2, 2),
    ]
    # Nutation Longitude and Obliquity coefficients (a,b)
    _NLOab_ = [
        (-171996, -174.2),
        (-13187, -1.6),
        (-2274, -0.2),
        (2062, 0.2),
        (1426, -3.4),
        (712, 0.1),
        (-517, 1.2),
        (-386, -0.4),
        (-301, 0),
        (217, -0.5),
        (-158, 0),
        (129, 0.1),
        (123, 0),
        (63, 0),
        (63, 0.1),
        (-59, 0),
        (-58, -0.1),
        (-51, 0),
        (48, 0),
        (46, 0),
        (-38, 0),
        (-31, 0),
        (29, 0),
        (29, 0),
        (26, 0),
        (-22, 0),
        (21, 0),
        (17, -0.1),
        (16, 0),
        (-16, 0.1),
        (-15, 0),
        (-13, 0),
        (-12, 0),
        (11, 0),
        (-10, 0),
        (-8, 0),
        (7, 0),
        (-7, 0),
        (-7, 0),
        (-7, 0),
        (6, 0),
        (6, 0),
        (6, 0),
        (-6, 0),
        (-6, 0),
        (5, 0),
        (-5, 0),
        (-5, 0),
        (-5, 0),
        (4, 0),
        (4, 0),
        (4, 0),
        (-4, 0),
        (-4, 0),
        (-4, 0),
        (3, 0),
        (-3, 0),
        (-3, 0),
        (-3, 0),
        (-3, 0),
        (-3, 0),
        (-3, 0),
        (-3, 0),
    ]
    # Nutation Longitude and Obliquity coefficients (c,d)
    _NLOcd_ = [
        (92025, 8.9),
        (5736, -3.1),
        (977, -0.5),
        (-895, 0.5),
        (54, -0.1),
        (-7, 0),
        (224, -0.6),
        (200, 0),
        (129, -0.1),
        (-95, 0.3),
        (0, 0),
        (-70, 0),
        (-53, 0),
        (0, 0),
        (-33, 0),
        (26, 0),
        (32, 0),
        (27, 0),
        (0, 0),
        (-24, 0),
        (16, 0),
        (13, 0),
        (0, 0),
        (-12, 0),
        (0, 0),
        (0, 0),
        (-10, 0),
        (0, 0),
        (-8, 0),
        (7, 0),
        (9, 0),
        (7, 0),
        (6, 0),
        (0, 0),
        (5, 0),
        (3, 0),
        (-3, 0),
        (0, 0),
        (3, 0),
        (3, 0),
        (0, 0),
        (-3, 0),
        (-3, 0),
        (3, 0),
        (3, 0),
        (0, 0),
        (3, 0),
        (3, 0),
        (3, 0),
    ]

    @staticmethod
    def ecliptic_obliquity(jme, delta_epsilon):
        """Calculate the true obliquity of the ecliptic (epsilon, in degrees) given the Julian Ephemeris Millennium and the obliquity."""

        u = jme / 10
        e0 = np.polyval(
            [
                2.45,
                5.79,
                27.87,
                7.12,
                -39.05,
                -249.67,
                -51.38,
                1999.25,
                -1.55,
                -4680.93,
                84381.448,
            ],
            u,
        )
        e = e0 / 3600.0 + delta_epsilon
        return e

    @staticmethod
    def nutation_obliquity(jce):
        """Compute the nutation in longitude (delta_psi) and the true obliquity (epsilon) given the Julian Ephemeris Century."""

        # mean elongation of the moon from the sun, in radians:
        # x0 = 297.85036 + 445267.111480*jce - 0.0019142*(jce**2) + (jce**3)/189474
        x0 = np.deg2rad(
            np.polyval([1.0 / 189474, -0.0019142, 445267.111480, 297.85036], jce)
        )
        # mean anomaly of the sun (Earth), in radians:
        x1 = np.deg2rad(
            np.polyval([-1 / 3e5, -0.0001603, 35999.050340, 357.52772], jce)
        )
        # mean anomaly of the moon, in radians:
        x2 = np.deg2rad(
            np.polyval([1.0 / 56250, 0.0086972, 477198.867398, 134.96298], jce)
        )
        # moon's argument of latitude, in radians:
        x3 = np.deg2rad(
            np.polyval([1.0 / 327270, -0.0036825, 483202.017538, 93.27191], jce)
        )
        # Longitude of the ascending node of the moon's mean orbit on the ecliptic
        # measured from the mean equinox of the date, in radians
        x4 = np.deg2rad(
            np.polyval([1.0 / 45e4, 0.0020708, -1934.136261, 125.04452], jce)
        )

        x = (x0, x1, x2, x3, x4)

        dp = 0.0
        for y, ab in zip(_sp._NLOY_, _sp._NLOab_):
            a, b = ab
            dp += (a + b * jce) * np.sin(np.dot(x, y))
        dp = np.rad2deg(dp) / 36e6

        de = 0.0
        for y, cd in zip(_sp._NLOY_, _sp._NLOcd_):
            c, d = cd
            de += (c + d * jce) * np.cos(np.dot(x, y))
        de = np.rad2deg(de) / 36e6

        e = _sp.ecliptic_obliquity(_sp.julian_millennium(jce), de)

        return dp, e

    @staticmethod
    def abberation_correction(R):
        """Calculate the abberation correction (delta_tau, in degrees) given the Earth Heliocentric Radius (in AU)."""

        return -20.4898 / (3600 * R)

    @staticmethod
    def sun_longitude(helio_pos, delta_psi):
        """Calculate the apparent sun longitude (lambda, in degrees) and geocentric longitude (beta, in degrees) given the earth heliocentric position and delta_psi."""

        L, B, R = helio_pos
        theta = L + 180  # geocentric latitude
        beta = -B
        ll = theta + delta_psi + _sp.abberation_correction(R)
        return ll, beta

    @staticmethod
    def greenwich_sidereal_time(jd, delta_psi, epsilon):
        """Calculate the apparent Greenwich sidereal time (v, in degrees) given the Julian Day."""

        jc = _sp.julian_century(jd)
        # mean sidereal time at greenwich, in degrees:
        v0 = (
            280.46061837
            + 360.98564736629 * (jd - 2451545)
            + 0.000387933 * (jc**2)
            - (jc**3) / 38710000
        ) % 360
        v = v0 + delta_psi * np.cos(np.deg2rad(epsilon))
        return v

    @staticmethod
    def sun_ra_decl(llambda, epsilon, beta):
        """Calculate the sun's geocentric right ascension (alpha, in degrees) and declination (delta, in degrees)."""

        l, e, b = map(np.deg2rad, (llambda, epsilon, beta))
        alpha = np.arctan2(
            np.sin(l) * np.cos(e) - np.tan(b) * np.sin(e), np.cos(l)
        )  # x1 / x2
        alpha = np.rad2deg(alpha) % 360
        delta = np.arcsin(np.sin(b) * np.cos(e) + np.cos(b) * np.sin(e) * np.sin(l))
        delta = np.rad2deg(delta)
        return alpha, delta

    @staticmethod
    def sun_topo_ra_decl_hour(latitude, longitude, elevation, jd, delta_t=0):
        """Calculate the sun's topocentric right ascension (alpha'), declination (delta'), and hour angle (H')."""

        jde = _sp.julian_ephemeris_day(jd, delta_t)
        jce = _sp.julian_century(jde)
        jme = _sp.julian_millennium(jce)

        helio_pos = _sp.heliocentric_position(jme)
        R = helio_pos[-1]
        phi, sigma, E = latitude, longitude, elevation
        # equatorial horizontal parallax of the sun, in radians
        xi = np.deg2rad(8.794 / (3600 * R))
        # rho = distance from center of earth in units of the equatorial radius
        # phi-prime = geocentric latitude
        # NB: These equations look like their based on WGS-84, but are rounded slightly
        # The WGS-84 reference ellipsoid has major axis a = 6378137 m, and flattening factor 1/f = 298.257223563
        # minor axis b = a*(1-f) = 6356752.3142 = 0.996647189335*a
        u = np.arctan(0.99664719 * np.tan(phi))
        x = np.cos(u) + E * np.cos(phi) / 6378140  # rho sin(phi-prime)
        y = 0.99664719 * np.sin(u) + E * np.sin(phi) / 6378140  # rho cos(phi-prime)

        delta_psi, epsilon = _sp.nutation_obliquity(jce)

        llambda, beta = _sp.sun_longitude(helio_pos, delta_psi)

        alpha, delta = _sp.sun_ra_decl(llambda, epsilon, beta)

        v = _sp.greenwich_sidereal_time(jd, delta_psi, epsilon)

        H = v + longitude - alpha
        Hr, dr = map(np.deg2rad, (H, delta))

        dar = np.arctan2(
            -x * np.sin(xi) * np.sin(Hr), np.cos(dr) - x * np.sin(xi) * np.cos(Hr)
        )
        delta_alpha = np.rad2deg(dar)

        alpha_prime = alpha + delta_alpha
        delta_prime = np.rad2deg(
            np.arctan2(
                (np.sin(dr) - y * np.sin(xi)) * np.cos(dar),
                np.cos(dr) - y * np.sin(xi) * np.cos(Hr),
            )
        )
        H_prime = H - delta_alpha

        return alpha_prime, delta_prime, H_prime

    @staticmethod
    def sun_topo_azimuth_zenith(
        latitude, delta_prime, H_prime, temperature=14.6, pressure=1013
    ):
        """Compute the sun's topocentric azimuth and zenith angles.

        Azimuth is measured eastward from north, zenith from vertical.
        Temperature = average temperature in C (default is 14.6 = global average in 2013).
        Pressure = average pressure in mBar (default 1013 = global average).
        """

        phi = np.deg2rad(latitude)
        dr, Hr = map(np.deg2rad, (delta_prime, H_prime))
        P, T = pressure, temperature
        e0 = np.rad2deg(
            np.arcsin(np.sin(phi) * np.sin(dr) + np.cos(phi) * np.cos(dr) * np.cos(Hr))
        )
        tmp = np.deg2rad(e0 + 10.3 / (e0 + 5.11))
        delta_e = (P / 1010.0) * (283.0 / (273 + T)) * (1.02 / (60 * np.tan(tmp)))
        e = e0 + delta_e
        zenith = 90 - e

        gamma = (
            np.rad2deg(
                np.arctan2(
                    np.sin(Hr), np.cos(Hr) * np.sin(phi) - np.tan(dr) * np.cos(phi)
                )
            )
            % 360
        )
        Phi = (gamma + 180) % 360  # azimuth from north
        return Phi, zenith

    @staticmethod
    def norm_lat_lon(lat, lon):
        """."""

        if lat < -90 or lat > 90:
            # convert to cartesian and back
            x = np.cos(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))
            y = np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))
            z = np.sin(np.deg2rad(lat))
            r = np.sqrt(x**2 + y**2 + z**2)
            lon = np.rad2deg(np.arctan2(y, x)) % 360
            lat = np.rad2deg(np.arcsin(z / r))
        elif lon < 0 or lon > 360:
            lon = lon % 360
        return lat, lon

    @staticmethod
    def topo_pos(t, lat, lon, elev, temp, press, dt):
        """Compute RA,dec,H, all in degrees."""

        lat, lon = _sp.norm_lat_lon(lat, lon)
        jd = _sp.julian_day(t)
        RA, dec, H = _sp.sun_topo_ra_decl_hour(lat, lon, elev, jd, dt)
        return RA, dec, H

    @staticmethod
    def pos(t, lat, lon, elev, temp, press, dt):
        """Compute azimute,zenith,RA,dec,H all in degree."""

        lat, lon = _sp.norm_lat_lon(lat, lon)
        jd = _sp.julian_day(t)
        RA, dec, H = _sp.sun_topo_ra_decl_hour(lat, lon, elev, jd, dt)
        azimuth, zenith = _sp.sun_topo_azimuth_zenith(lat, dec, H, temp, press)
        return azimuth, zenith, RA, dec, H


def julian_day(dt):
    """Convert UTC datetimes or UTC timestamps to Julian days.

    Parameters
    ----------
    dt : array_like
        UTC datetime objects or UTC timestamps (as per datetime.utcfromtimestamp)

    Returns
    -------
    jd : ndarray
        datetimes converted to fractional Julian days
    """

    dts = np.array(dt)
    if len(dts.shape) == 0:
        return _sp.julian_day(dt)

    jds = np.empty(dts.shape)
    for i, d in enumerate(dts.flat):
        jds.flat[i] = _sp.julian_day(d)
    return jds


def arcdist(p0, p1, radians=False):
    """Angular distance between azimuth, zenith pairs.

    Parameters
    ----------
    p0 : array_like, shape (..., 2)
    p1 : array_like, shape (..., 2)
        p[...,0] = azimuth angles, p[...,1] = zenith angles
    radians : boolean (default False)
        If False, angles are in degrees, otherwise in radians

    Returns
    -------
    ad :  array_like, shape is broadcast(p0,p1).shape
        Arcdistances between corresponding pairs in p0,p1
        In degrees by default, in radians if radians=True
    """

    # formula comes from translating points into cartesian coordinates
    # taking the dot product to get the cosine between the two vectors
    # then arccos to return to angle, and simplify everything assuming real inputs
    p0, p1 = np.array(p0), np.array(p1)
    if not radians:
        p0, p1 = np.deg2rad(p0), np.deg2rad(p1)
    a0, z0 = p0[..., 0], p0[..., 1]
    a1, z1 = p1[..., 0], p1[..., 1]
    d = np.arccos(np.cos(z0) * np.cos(z1) + np.cos(a0 - a1) * np.sin(z0) * np.sin(z1))
    if radians:
        return d
    else:
        return np.rad2deg(d)


def observed_sunpos(
    dt,
    latitude,
    longitude,
    elevation,
    temperature=None,
    pressure=None,
    delta_t=0,
    radians=False,
):
    """Compute the observed coordinates of the sun as viewed at the given time and location.

    Parameters
    ----------
    dt : array_like
        UTC datetime objects or UTC timestamps (as per datetime.utcfromtimestamp) representing the times of observations
    latitude, longitude : array_like
        decimal degrees, positive for north of the equator and east of Greenwich
    elevation : array_like
        meters, relative to the WGS-84 ellipsoid
    temperature : array_like or None, optional
        celcius, default is 14.6 (global average in 2013)
    pressure : array_like or None, optional
        millibar, default is 1013 (global average in ??)
    delta_t : array_like, optional
        seconds, default is 0, difference between the earth's rotation time (TT) and universal time (UT)
    radians : {True, False}, optional
        return results in radians if True, degrees if False (default)

    Returns
    -------
    coords : ndarray, (...,2)
        The shape of the array is parameters broadcast together, plus a final dimension for the coordinates.
        coords[...,0] = observed azimuth angle, measured eastward from north
        coords[...,1] = observed zenith angle, measured down from vertical
    """

    if temperature is None:
        temperature = 14.6
    if pressure is None:
        pressure = 1013

    # 6367444 = radius of earth
    # numpy broadcasting
    b = np.broadcast(dt, latitude, longitude, elevation, temperature, pressure, delta_t)
    res = np.empty(b.shape + (2,))
    res_vec = res.reshape((-1, 2))
    for i, x in enumerate(b):
        res_vec[i] = _sp.pos(*x)[:2]
    if radians:
        res = np.deg2rad(res)
    return res


def topocentric_sunpos(
    dt, latitude, longitude, temperature=None, pressure=None, delta_t=0, radians=False
):
    """Compute the topocentric coordinates of the sun as viewed at the given time and location.

    Parameters
    ----------
    dt : array_like
        UTC datetime objects or UTC timestamps (as per datetime.utcfromtimestamp) representing the times of observations
    latitude, longitude : array_like
        decimal degrees, positive for north of the equator and east of Greenwich
    elevation : array_like
        meters, relative to the WGS-84 ellipsoid
    temperature : array_like or None, optional
        celcius, default is 14.6 (global average in 2013)
    pressure : array_like or None, optional
        millibar, default is 1013 (global average in ??)
    delta_t : array_like, optional
        seconds, default is 0, difference between the earth's rotation time (TT) and universal time (UT)
    radians : {True, False}, optional
        return results in radians if True, degrees if False (default)

    Returns
    -------
    coords : ndarray, (...,3)
        The shape of the array is parameters broadcast together, plus a final dimension for the coordinates.
        coords[...,0] = topocentric right ascension
        coords[...,1] = topocentric declination
        coords[...,2] = topocentric hour angle
    """

    if temperature is None:
        temperature = 14.6
    if pressure is None:
        pressure = 1013

    # 6367444 = radius of earth
    # numpy broadcasting
    b = np.broadcast(dt, latitude, longitude, elevation, temperature, pressure, delta_t)
    res = np.empty(b.shape + (2,))
    res_vec = res.reshape((-1, 2))
    for i, x in enumerate(b):
        res_vec[i] = _sp.topo_pos(*x)
    if radians:
        res = np.deg2rad(res)
    return res


def sunpos(
    dt,
    latitude,
    longitude,
    elevation,
    temperature=None,
    pressure=None,
    delta_t=0,
    radians=False,
):
    """Compute the observed and topocentric coordinates of the sun as viewed at the given time and location.

    Parameters
    ----------
    dt : array_like
        UTC datetime objects or UTC timestamps (as per datetime.utcfromtimestamp) representing the times of observations
    latitude, longitude : array_like
        decimal degrees, positive for north of the equator and east of Greenwich
    elevation : array_like
        meters, relative to the WGS-84 ellipsoid
    temperature : array_like or None, optional
        celcius, default is 14.6 (global average in 2013)
    pressure : array_like or None, optional
        millibar, default is 1013 (global average in ??)
    delta_t : array_like, optional
        seconds, default is 0, difference between the earth's rotation time (TT) and universal time (UT)
    radians : {True, False}, optional
        return results in radians if True, degrees if False (default)

    Returns
    -------
    coords : ndarray, (...,5)
        The shape of the array is parameters broadcast together, plus a final dimension for the coordinates.
        coords[...,0] = observed azimuth angle, measured eastward from north
        coords[...,1] = observed zenith angle, measured down from vertical
        coords[...,2] = topocentric right ascension
        coords[...,3] = topocentric declination
        coords[...,4] = topocentric hour angle
    """

    if temperature is None:
        temperature = 14.6
    if pressure is None:
        pressure = 1013

    # 6367444 = radius of earth
    # numpy broadcasting
    b = np.broadcast(dt, latitude, longitude, elevation, temperature, pressure, delta_t)
    res = np.empty(b.shape + (5,))
    res_vec = res.reshape((-1, 5))
    for i, x in enumerate(b):
        res_vec[i] = _sp.pos(*x)
    if radians:
        res = np.deg2rad(res)
    return res


class Sunposition:
    """Compute sun position parameters given the time and location."""

    # Inputs
    t = None
    lat, lon = None, None
    elev = None
    temp = None
    p = None
    dt = None
    rad = None
    # Outputs
    az = None
    zen = None
    ra = None
    dec = None
    h = None

    def __init__(self, t, lat, lon, elev, temp, p, dt, rad, csv=False):
        """Initialize the class and run the model."""

        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.temp = temp
        self.p = p
        self.dt = dt
        self.rad = rad

        if t == "now":
            self.t = datetime.utcnow()
        elif ":" in t and "-" in t:
            try:
                # with microseconds
                self.t = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
            except BaseException:
                try:
                    # without microseconds
                    self.t = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.")
                except BaseException:
                    self.t = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        else:
            self.t = datetime.utcfromtimestamp(int(t))

        # Run the sun position calculation
        self.az, self.zen, self.ra, self.dec, self.h = sunpos(
            self.t, lat, lon, elev, temp, p, dt, rad
        )

        # Format output to CSV?
        if csv:
            print(
                "{t}, {dt}, {lat}, {lon}, {elev}, {temp}, {p}, {az}, {zen}, {ra},"
                " {dec}, {h}".format(
                    t=self.t,
                    dt=dt,
                    lat=lat,
                    lon=lon,
                    elev=elev,
                    temp=temp,
                    p=p,
                    az=self.az,
                    zen=self.zen,
                    ra=self.ra,
                    dec=self.dec,
                    h=self.h,
                )
            )
        else:
            dr = "deg"
            if rad:
                dr = "rad"
            print("Computing sun position at T = {t} + {dt} s".format(t=self.t, dt=dt))
            print(
                "Lat, Lon, Elev = {lat} deg, {lon} deg, {elev} m".format(
                    lat=lat, lon=lon, elev=elev
                )
            )
            print("T, P = {temp} C, {press} mbar".format(temp=temp, press=p))
            print("Results:")
            print(
                "Azimuth, zenith = {az} {dr}, {zen} {dr}".format(
                    az=self.az, zen=self.zen, dr=dr
                )
            )
            print(
                "RA, dec, H = {ra} {dr}, {dec} {dr}, {h} {dr}".format(
                    ra=self.ra, dec=self.dec, h=self.h, dr=dr
                )
            )

    @property
    def citation(self):
        """Print the citation."""

        print("Implementation: Samuel Bear Powell, 2016")
        print("Algorithm:")
        print(
            'Ibrahim Reda, Afshin Andreas, "Solar position algorithm for solar'
            ' radiation applications", SolarEnergy, Volume 76, Issue 5, 2004, Pages'
            " 577-589, ISSN 0038-092X, doi:10.1016/j.solener.2003.12.003"
        )
