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

import argparse
import scipy as s
import json
from copy import deepcopy
from datetime import date, timedelta

from isofit.core.common import download_HRRR, get_HRRR_data
from isofit.core.common import json_load_ascii

class HRRR_to_MODTRAN_profiles():
    '''
    This class assumes that the MODTRAN config file has already been
    filled with the correct run data, including time, lat/lon, etc.
    '''
    def __init__(self, config_filename):

        self.config = deepcopy(json_load_ascii(config_filename))

        self.template = deepcopy(json_load_ascii(
            self.config['modtran_config_json_filename'])['MODTRAN'])
        
        self.year_for_HRRR_profiles_in_modtran = self.config['year_for_HRRR_profiles_in_modtran']
        self.HRRR_data_library_path = self.config['HRRR_data_library_path']
        self.create_profiles()

        self.template[0]['MODTRANINPUT']['ATMOSPHERE']['NLAYERS'] = len(self.prof_altitude_dict['PROFILE'])
        self.template[0]['MODTRANINPUT']['ATMOSPHERE']['NPROF'] = 4
        self.template[0]['MODTRANINPUT']['ATMOSPHERE']['PROFILES'] = [dict(self.prof_altitude_dict),
                                                              dict(self.prof_pressure_dict),
                                                              dict(self.prof_temperature_dict),
                                                              dict(self.prof_H2O_dict)]
        self.template_str = json.dumps({"MODTRAN": self.template})
        with open(self.config["output_modtran_config_filename"], 'w') as f:
            f.write(self.template_str)
             
        a = 0

    def create_profiles(self):
        '''
        Create MODTRAN profile strings from HRRR data. For example:

        print(self.prof_altitude)

        yields:

        {
        "TYPE": "PROF_ALTITUDE",
        "UNITS": "UNT_KILOMETERS",
        "PROFILE": [1.224, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000, 10.000, 11.000, 12.000, 13.000, 14.000, 15.000, 16.000, 17.000, 18.000, 19.000]
        }
        '''

        lat = self.template[0]['MODTRANINPUT']['GEOMETRY']['PARM1']
        lon = self.template[0]['MODTRANINPUT']['GEOMETRY']['PARM2']
        gmtime = self.template[0]['MODTRANINPUT']['GEOMETRY']['GMTIME']
        iday = self.template[0]['MODTRANINPUT']['GEOMETRY']['IDAY']
        h1alt_km = self.template[0]['MODTRANINPUT']['GEOMETRY']['H1ALT']
        gndalt_km = self.template[0]['MODTRANINPUT']['SURFACE']['GNDALT']

        date_to_get = date(self.year_for_HRRR_profiles_in_modtran, 1, 1) + \
                        timedelta(iday - 1)
        grb_filename = download_HRRR(date_to_get, model = 'hrrr', field = 'prs', \
                        hour = [int(gmtime)], fxx = [0], \
                        OUTDIR = self.HRRR_data_library_path)
        
        # Read the HRRR file
        grb_lat, grb_lon, grb_geo_pot_height_m, grb_temperature_K, grb_rh_perc, grb_pressure_levels_Pa = \
            get_HRRR_data(grb_filename)
        
        # Find nearest spatial pixel
        r2 = (grb_lat - lat)**2 + (grb_lon - (-1*lon))**2
        indx, indy = s.unravel_index(s.argmin(r2), r2.shape)

        # Grab the profile at the nearest spatial pixel
        geo_pot_height_profile_km = grb_geo_pot_height_m[:, indx, indy] / 1000
        temperature_profile_K = grb_temperature_K[:, indx, indy]
        rh_profile_perc = grb_rh_perc[:, indx, indy]

        # Put them in order from lowest to highest
        sort_inds = s.argsort(geo_pot_height_profile_km)
        geo_pot_height_profile_km = geo_pot_height_profile_km[sort_inds]
        temperature_profile_K = temperature_profile_K[sort_inds]
        rh_profile_perc = rh_profile_perc[sort_inds]
        pressure_profile_atm = grb_pressure_levels_Pa[sort_inds] * 9.868e-6

        # Interpolate to how MODTRAN seems to want them, following example
        # on p97 of MODTRAN 6 User's Manual
        if gndalt_km < geo_pot_height_profile_km[0] or h1alt_km > geo_pot_height_profile_km[-1]:
            print("Cannot extrapolate from MODTRAN profiles!")
            raise ValueError
        n = s.floor(geo_pot_height_profile_km[-1]) - s.ceil(gndalt_km)
        mod_height_profile_km = [gndalt_km] + list(s.arange(n) + s.ceil(gndalt_km))
        mod_temperature_profile_K = s.interp(mod_height_profile_km, geo_pot_height_profile_km, temperature_profile_K)
        mod_rh_profile_perc = s.interp(mod_height_profile_km, geo_pot_height_profile_km, rh_profile_perc)
        mod_pressure_profile_atm = s.interp(mod_height_profile_km, geo_pot_height_profile_km, pressure_profile_atm)

        # Get water vapor saturation density (p 95 of MODTRAN 6 User's Manual)
        tr = 273.15 / mod_temperature_profile_K
        rho_sat = tr * s.exp(18.9766 - (14.9595 + 2.43882 * tr) * tr)

        # Get water mixing ratio in ppmV (p 95 of MODTRAN 6 User's Manual)
        mod_mixing_ratio_ppmV = rho_sat * 0.01*mod_rh_profile_perc / 18.01528 * 22413.83 / \
                                mod_pressure_profile_atm / tr

        self.prof_altitude_dict = {}
        self.prof_altitude_dict['TYPE'] = 'PROF_ALTITUDE'
        self.prof_altitude_dict['UNITS'] = 'UNT_KILOMETERS'
        self.prof_altitude_dict['PROFILE'] = list(mod_height_profile_km)

        self.prof_pressure_dict = {}
        self.prof_pressure_dict['TYPE'] = 'PROF_PRESSURE'
        self.prof_pressure_dict['UNITS'] = 'UNT_PMILLIBAR'
        self.prof_pressure_dict['PROFILE'] = list(mod_pressure_profile_atm * 1013.25) # Convert atm millibar

        self.prof_temperature_dict = {}
        self.prof_temperature_dict['TYPE'] = 'PROF_TEMPERATURE'
        self.prof_temperature_dict['UNITS'] = 'UNT_TKELVIN'
        self.prof_temperature_dict['PROFILE'] = list(mod_temperature_profile_K)

        self.prof_H2O_dict = {}
        self.prof_H2O_dict['TYPE'] = 'PROF_H2O'
        self.prof_H2O_dict['UNITS'] = 'UNT_DPPMV'
        self.prof_H2O_dict['PROFILE'] = list(mod_mixing_ratio_ppmV)


def main():

    parser = argparse.ArgumentParser(description="Create a surface model")
    parser.add_argument('config', type=str, metavar='INPUT')
    args = parser.parse_args()
    h = HRRR_to_MODTRAN_profiles(args.config)
    #config = json_load_ascii(args.config, shell_replace=True)
    #configdir, configfile = split(abspath(args.config))

if __name__ == "__main__":
    main()