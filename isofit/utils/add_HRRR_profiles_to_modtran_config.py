#! /usr/bin/env python3
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
# Author: Jay E Fahlen, jay.e.fahlen@jpl.nasa.gov
#

import json
import os
import sys
import time
import urllib.request
from copy import deepcopy
from datetime import date, timedelta

import click
import numpy as np

from isofit.core.common import json_load_ascii

try:
    import pygrib
except:
    pygrib = None


class HRRR_to_MODTRAN_profiles:
    """
    This class assumes that the MODTRAN config file has already been
    filled with the correct run data, including time, lat/lon, etc.
    """

    def __init__(self, config_file):
        self.config = deepcopy(json_load_ascii(config_file))

        self.modtran_config_filenames = self.config["modtran_config_json_filenames"]
        self.output_modtran_config_filenames = self.config[
            "output_modtran_config_filenames"
        ]
        self.year_for_HRRR_profiles_in_modtran = self.config[
            "year_for_HRRR_profiles_in_modtran"
        ]
        self.HRRR_data_library_path = self.config["HRRR_data_library_path"]

        for modtran_config_filename, output_modtran_config_filename in zip(
            self.modtran_config_filenames, self.output_modtran_config_filenames
        ):
            template = deepcopy(json_load_ascii(modtran_config_filename)["MODTRAN"])

            (
                prof_altitude_dict,
                prof_pressure_dict,
                prof_temperature_dict,
                prof_H2O_dict,
            ) = self.create_profiles(template)

            template[0]["MODTRANINPUT"]["ATMOSPHERE"]["NLAYERS"] = len(
                prof_altitude_dict["PROFILE"]
            )
            template[0]["MODTRANINPUT"]["ATMOSPHERE"]["NPROF"] = 4
            template[0]["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"] = [
                dict(prof_altitude_dict),
                dict(prof_pressure_dict),
                dict(prof_temperature_dict),
                dict(prof_H2O_dict),
            ]

            template_str = json.dumps({"MODTRAN": template})
            with open(output_modtran_config_filename, "w") as f:
                f.write(template_str)

        return

    def create_profiles(self, template):
        """
        Create MODTRAN profile strings from HRRR data. For example:

        print(self.prof_altitude)

        yields:

        {
        "TYPE": "PROF_ALTITUDE",
        "UNITS": "UNT_KILOMETERS",
        "PROFILE": [1.224, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000, 10.000, 11.000, 12.000, 13.000, 14.000, 15.000, 16.000, 17.000, 18.000, 19.000]
        }
        """

        lat = template[0]["MODTRANINPUT"]["GEOMETRY"]["PARM1"]
        lon = template[0]["MODTRANINPUT"]["GEOMETRY"]["PARM2"]
        gmtime = template[0]["MODTRANINPUT"]["GEOMETRY"]["GMTIME"]
        iday = template[0]["MODTRANINPUT"]["GEOMETRY"]["IDAY"]
        h1alt_km = template[0]["MODTRANINPUT"]["GEOMETRY"]["H1ALT"]
        gndalt_km = template[0]["MODTRANINPUT"]["SURFACE"]["GNDALT"]

        date_to_get = date(self.year_for_HRRR_profiles_in_modtran, 1, 1) + timedelta(
            iday - 1
        )
        grb_filename = download_HRRR(
            date_to_get,
            model="hrrr",
            field="prs",
            hour=[int(gmtime)],
            fxx=[0],
            OUTDIR=self.HRRR_data_library_path,
        )

        # Read the HRRR file
        (
            grb_lat,
            grb_lon,
            grb_geo_pot_height_m,
            grb_temperature_K,
            grb_rh_perc,
            grb_pressure_levels_Pa,
        ) = get_HRRR_data(grb_filename)

        # Find nearest spatial pixel
        r2 = (grb_lat - lat) ** 2 + (grb_lon - (-1 * lon)) ** 2
        indx, indy = np.unravel_index(np.argmin(r2), r2.shape)

        # Grab the profile at the nearest spatial pixel
        geo_pot_height_profile_km = grb_geo_pot_height_m[:, indx, indy] / 1000
        temperature_profile_K = grb_temperature_K[:, indx, indy]
        rh_profile_perc = grb_rh_perc[:, indx, indy]

        # Put them in order from lowest to highest
        sort_inds = np.argsort(geo_pot_height_profile_km)
        geo_pot_height_profile_km = geo_pot_height_profile_km[sort_inds]
        temperature_profile_K = temperature_profile_K[sort_inds]
        rh_profile_perc = rh_profile_perc[sort_inds]
        pressure_profile_atm = grb_pressure_levels_Pa[sort_inds] * 9.868e-6

        # Interpolate to how MODTRAN seems to want them, following example
        # on p97 of MODTRAN 6 User's Manual
        if (
            gndalt_km < geo_pot_height_profile_km[0]
            or h1alt_km > geo_pot_height_profile_km[-1]
        ):
            print("Cannot extrapolate from MODTRAN profiles!")
            raise ValueError
        n = np.floor(geo_pot_height_profile_km[-1]) - np.ceil(gndalt_km)
        mod_height_profile_km = [gndalt_km] + list(np.arange(n) + np.ceil(gndalt_km))
        mod_temperature_profile_K = np.interp(
            mod_height_profile_km, geo_pot_height_profile_km, temperature_profile_K
        )
        mod_rh_profile_perc = np.interp(
            mod_height_profile_km, geo_pot_height_profile_km, rh_profile_perc
        )
        mod_pressure_profile_atm = np.interp(
            mod_height_profile_km, geo_pot_height_profile_km, pressure_profile_atm
        )

        # Get water vapor saturation density (p 95 of MODTRAN 6 User's Manual)
        tr = 273.15 / mod_temperature_profile_K
        rho_sat = tr * np.exp(18.9766 - (14.9595 + 2.43882 * tr) * tr)

        # Get water mixing ratio in ppmV (p 95 of MODTRAN 6 User's Manual)
        mod_mixing_ratio_ppmV = (
            rho_sat
            * 0.01
            * mod_rh_profile_perc
            / 18.01528
            * 22413.83
            / mod_pressure_profile_atm
            / tr
        )

        prof_altitude_dict = {}
        prof_altitude_dict["TYPE"] = "PROF_ALTITUDE"
        prof_altitude_dict["UNITS"] = "UNT_KILOMETERS"
        prof_altitude_dict["PROFILE"] = list(mod_height_profile_km)

        prof_pressure_dict = {}
        prof_pressure_dict["TYPE"] = "PROF_PRESSURE"
        prof_pressure_dict["UNITS"] = "UNT_PMILLIBAR"
        prof_pressure_dict["PROFILE"] = list(
            mod_pressure_profile_atm * 1013.25
        )  # Convert atm millibar

        prof_temperature_dict = {}
        prof_temperature_dict["TYPE"] = "PROF_TEMPERATURE"
        prof_temperature_dict["UNITS"] = "UNT_TKELVIN"
        prof_temperature_dict["PROFILE"] = list(mod_temperature_profile_K)

        prof_H2O_dict = {}
        prof_H2O_dict["TYPE"] = "PROF_H2O"
        prof_H2O_dict["UNITS"] = "UNT_DPPMV"
        prof_H2O_dict["PROFILE"] = list(mod_mixing_ratio_ppmV)

        return (
            prof_altitude_dict,
            prof_pressure_dict,
            prof_temperature_dict,
            prof_H2O_dict,
        )


def reporthook(a, b, c):
    """
    Report download progress in megabytes
    """
    # ',' at the end of the line is important!
    print(
        "\r % 3.1f%% of %.2f MB\r" % (min(100, float(a * b) / c * 100), c / 1000000.0),
        end="",
    )


def download_HRRR(
    DATE, model="hrrr", field="sfc", hour=range(0, 24), fxx=range(0, 1), OUTDIR="./"
):
    """
    # Brian Blaylock
    # February 13, 2018

    # Updated December 10, 2018 for Python 3

    # Modified from original by Jay Fahlen to not download the file if it already exists.
    # March 4, 2020

    Download archived HRRR files from MesoWest Pando S3 archive system.

    Please register before downloading from our HRRR archive:
    http://hrrr.chpc.utah.edu/hrrr_download_register.html

    For info on the University of Utah HRRR archive and to see what dates are
    available, look here:
    http://hrrr.chpc.utah.edu/

    Contact:
    brian.blaylock@utah.edu

    Downloads from the University of Utah MesoWest HRRR archive
    Input:
        DATE   - A date object for the model run you are downloading from.
        model  - The model type you want to download. Default is 'hrrr'
                 Model Options are ['hrrr', 'hrrrX','hrrrak']
        field  - Variable fields you wish to download. Default is sfc, surface.
                 Options are fields ['prs', 'sfc','subh', 'nat']
        hour   - Range of model run hours. Default grabs all hours of day.
        fxx    - Range of forecast hours. Default grabs analysis hour (f00).
        OUTDIR - Directory to save the files.

    Outcome:
        Downloads the desired HRRR file and renames with date info preceeding
        the original file name (i.e. 20170101_hrrr.t00z.wrfsfcf00.grib2)
    """
    # Make OUTDIR if path doesn't exist
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    # Loop through each hour and each forecast and download.
    for h in hour:
        for f in fxx:
            # 1) Build the URL string we want to download.
            #    fname is the file name in the format
            #    [model].t[hh]z.wrf[field]f[xx].grib2
            #    i.e. hrrr.t00z.wrfsfcf00.grib2
            fname = "%s.t%02dz.wrf%sf%02d.grib2" % (model, h, field, f)
            URL = "https://pando-rgw01.chpc.utah.edu/%s/%s/%s/%s" % (
                model,
                field,
                DATE.strftime("%Y%m%d"),
                fname,
            )

            # 2) Rename file with date preceeding original filename
            #    i.e. 20170105_hrrr.t00z.wrfsfcf00.grib2
            rename = "%s_%s" % (DATE.strftime("%Y%m%d"), fname)

            filename = OUTDIR + rename
            if not os.path.exists(filename):
                # 3) Download the file via https
                # Check the file size, make it's big enough to exist.
                check_this = urllib.request.urlopen(URL)
                file_size = int(check_this.info()["content-length"])
                if file_size > 10000:
                    print("Downloading:", URL)
                    urllib.request.urlretrieve(URL, OUTDIR + rename, reporthook)
                    print("\n")
                else:
                    # URL returns an "Key does not exist" message
                    print("ERROR:", URL, "Does Not Exist")

                # 4) Sleep five seconds, as a courtesy for using the archive.
                time.sleep(5)
    return filename


def get_HRRR_data(filename):
    if pygrib is None:
        raise ImportError(
            "Missing dependency pygrib. Please install: https://jswhit.github.io/pygrib/installing.html"
        )

    grbs = pygrib.open(filename)

    msgs = [str(grb) for grb in grbs]

    string = "Geopotential Height:gpm"
    temp = [
        msg for msg in msgs if msg.find(string) > -1 and msg.find("isobaricInhPa") > -1
    ]
    pressure_levels_Pa = s.array([int(s.split(" ")[3]) for s in temp])

    geo_pot_height_grbs = grbs.select(
        name="Geopotential Height", typeOfLevel="isobaricInhPa", level=lambda l: l > 0
    )
    temperature_grbs = grbs.select(
        name="Temperature", typeOfLevel="isobaricInhPa", level=lambda l: l > 0
    )
    rh_grbs = grbs.select(
        name="Relative humidity", typeOfLevel="isobaricInhPa", level=lambda l: l > 0
    )

    lat, lon = geo_pot_height_grbs[0].latlons()

    geo_pot_height = s.stack([grb.values for grb in geo_pot_height_grbs])
    temperature = s.stack([grb.values for grb in temperature_grbs])
    rh = s.stack([grb.values for grb in rh_grbs])

    return lat, lon, geo_pot_height, temperature, rh, pressure_levels_Pa


@click.command(name="HRRR_to_modtran")
@click.argument("config_file")
def cli_HRRR_to_modtran(**kwargs):
    if pygrib is None:
        raise ImportError(
            "Missing dependency pygrib. Please install: https://jswhit.github.io/pygrib/installing.html"
        )

    """Add HRRR profiles to MODTRAN"""

    click.echo("Running adding HRRR profiles to MODTRAN")

    HRRR_to_MODTRAN_profiles(**kwargs)

    click.echo("Done")
