#! /usr/bin/env python
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#         Philip G Brodrick, philip.brodrick@jpl.nasa.gov
#


### Variables ###

name = "isofit"

__version__ = "2.9.8"

warnings_enabled = False

import argparse
import logging
import sys


def run_isofit(rawargs=sys.argv):
    """
    Executes isofit.py
    """
    from isofit.core.isofit import Isofit

    parser = argparse.ArgumentParser(
        description="Spectroscopic Surface & Atmosphere Fitting"
    )
    parser.add_argument("config_file")
    parser.add_argument("--level", default="INFO")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    args = parser.parse_args(rawargs)

    config = args.config_file
    level = args.level

    logging.basicConfig(format="%(message)s", level=level)

    isofit = Isofit(config_file=config, level=level)

    return isofit.run()


def run_sunposition(rawargs=sys.argv):
    """
    Solar position algorithm of Reda & Andreas (2003).
    """
    from isofit.core import sunposition

    parser = argparse.ArgumentParser(
        prog="sunposition",
        description="Compute sun position parameters given the time and location",
    )
    parser.add_argument(
        "--citation",
        dest="cite",
        action="store_true",
        help="Print citation information",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="t",
        type=str,
        default="now",
        help='"now" or date and time (UTC) in "YYYY-MM-DD hh:mm:ss.ssssss" format or a (UTC) POSIX timestamp',
    )
    parser.add_argument(
        "-lat",
        "--latitude",
        dest="lat",
        type=float,
        default=51.48,
        help="latitude, in decimal degrees, positive for north",
    )
    parser.add_argument(
        "-lon",
        "--longitude",
        dest="lon",
        type=float,
        default=0.0,
        help="longitude, in decimal degrees, positive for east",
    )
    parser.add_argument(
        "-e",
        "--elevation",
        dest="elev",
        type=float,
        default=0,
        help="elevation, in meters",
    )
    parser.add_argument(
        "-T",
        "--temperature",
        dest="temp",
        type=float,
        default=14.6,
        help="temperature, in degrees celcius",
    )
    parser.add_argument(
        "-p",
        "--pressure",
        dest="p",
        type=float,
        default=1013.0,
        help="atmospheric pressure, in millibar",
    )
    parser.add_argument(
        "-dt",
        type=float,
        default=0.0,
        help="difference between earth's rotation time (TT) and universal time (UT1)",
    )
    parser.add_argument(
        "-r",
        "--radians",
        dest="rad",
        action="store_true",
        help="Output in radians instead of degrees",
    )
    parser.add_argument(
        "--csv",
        dest="csv",
        action="store_true",
        help="Comma separated values (time,dt,lat,lon,elev,temp,pressure,az,zen,RA,dec,H)",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    args = parser.parse_args(rawargs)

    if args.cite:
        print("Implementation: Samuel Bear Powell, 2016")
        print("Algorithm:")
        print(
            'Ibrahim Reda, Afshin Andreas, "Solar position algorithm for solar radiation applications", Solar Energy, Volume 76, Issue 5, 2004, Pages 577-589, ISSN 0038-092X, doi:10.1016/j.solener.2003.12.003'
        )
        sys.exit(0)

    if args.t == "now":
        args.t = datetime.utcnow()
    elif ":" in args.t and "-" in args.t:
        try:
            args.t = datetime.strptime(
                args.t, "%Y-%m-%d %H:%M:%S.%f"
            )  # with microseconds
        except:
            try:
                args.t = datetime.strptime(
                    args.t, "%Y-%m-%d %H:%M:%S."
                )  # without microseconds
            except:
                args.t = datetime.strptime(args.t, "%Y-%m-%d %H:%M:%S")
    else:
        args.t = datetime.utcfromtimestamp(int(args.t))

    # sunpos function
    az, zen, ra, dec, h = sunposition.sunpos(
        args.t, args.lat, args.lon, args.elev, args.temp, args.p, args.dt, args.rad
    )

    # machine readable
    if args.csv:
        print(
            f"{args.t}, {args.dt}, {args.lat}, {args.lon}, {args.elev}, {args.temp}, {args.p}, {az}, {zen}, {ra}, {dec}, {h}"
        )

    else:
        dr = "rad" if args.rad else "deg"
        print(f"Computing sun position at T = {args.t} + {args.dt}")
        print(f"Lat, Lon, Elev = {args.lat} deg, {args.lon} deg, {args.elev} m")
        print(f"T, P = {args.temp}, {args.p}")
        print("Results:")
        print(f"Azimuth, zenith = {az} {dr}, {zen} {dr}")
        print(f"RA, dec, H = {ra} {dr}, {dec} {dr}, {h} {dr}")
