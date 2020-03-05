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
#         Adam Erickson, adam.m.erickson@nasa.gov
#

from ..utils.path import absjoin


def get(config, base_directory):
    """Get an ISOFIT template with dynamic path setting."""

    if config == "modtran":
        return {
            "MODTRAN": [
                {
                    "MODTRANINPUT": {
                        "NAME": "test",
                        "DESCRIPTION": "",
                        "CASE": 0,
                        "RTOPTIONS": {
                            "MODTRN": "RT_CORRK_FAST",
                            "LYMOLC": False,
                            "T_BEST": False,
                            "IEMSCT": "RT_SOLAR_AND_THERMAL",
                            "IMULT": "RT_DISORT",
                            "DISALB": False,
                            "NSTR": 8,
                            "SOLCON": 0.0
                        },
                        "ATMOSPHERE": {
                            "MODEL": "ATM_TROPICAL",
                            "M1": "ATM_TROPICAL",
                            "M2": "ATM_TROPICAL",
                            "M3": "ATM_TROPICAL",
                            "M4": "ATM_TROPICAL",
                            "M5": "ATM_TROPICAL",
                            "M6": "ATM_TROPICAL",
                            "CO2MX": 420.0,
                            "H2OSTR": 0.64,
                            "H2OUNIT": "g",
                            "O3STR": 0.3,
                            "O3UNIT": "a"
                        },
                        "AEROSOLS": {
                            "CDASTM": "b",
                            "IHAZE": "AER_RURAL",
                            "VIS": 50.0
                        },
                        "GEOMETRY": {
                            "ITYPE": 3,
                            "H1ALT": 500.0,
                            "OBSZEN": 180.0,
                            "IDAY": 312,
                            "IPARM": 12,
                            "PARM1": 90.0,
                            "PARM2": 45.0
                        },
                        "SURFACE": {
                            "SURFTYPE": "REFL_LAMBER_MODEL",
                            "GNDALT": 0.01,
                            "NSURF": 1,
                            "SURFP": {
                                "CSALB": "LAMB_CONST_0_PCT"
                            }
                        },
                        "SPECTRAL": {
                            "V1": 350.0,
                            "V2": 2520.0,
                            "DV": 0.1,
                            "FWHM": 0.1,
                            "YFLAG": "R",
                            "XFLAG": "N",
                            "FLAGS": "NT A   ",
                            "BMNAME": "p1_2013"
                        },
                        "FILEOPTIONS": {
                            "NOPRNT": 2,
                            "CKPRNT": True
                        }
                    }
                }
            ]
        }
    elif config == "forward":
        return {
            "input": {
                "reflectance_file": absjoin(base_directory, "input", "reference_reflectance")
            },
            "output": {
                "simulated_measurement_file": absjoin(base_directory, "input", "reference_radiance")
            },
            "forward_model": {
                "instrument": {
                    "wavelength_file": absjoin(base_directory, "data", "wavelengths.txt"),
                    "integrations": 1,
                    "SNR": 300
                },
                "surface": {
                    "wavelength_file": absjoin(base_directory, "data", "wavelengths.txt"),
                    "reflectance_file": absjoin(base_directory, "data", "stonewall_rfl.txt")
                },
                "modtran_radiative_transfer": {
                    "wavelength_file": absjoin(base_directory, "data", "wavelengths.txt"),
                    "lut_path": absjoin(base_directory, "lut"),
                    "configure_and_exit": False,
                    "modtran_template_file": absjoin(base_directory, "config", "modtran.json"),
                    "domain": {
                        "start": 350,
                        "end": 2520,
                        "step": 0.1
                    },
                    "statevector": {
                        "H2OSTR": {
                            "bounds": [1.0, 2.5],
                            "scale": 0.1,
                            "prior_mean": 1.75,
                            "prior_sigma": 5.0,
                            "init": 1.75
                        },
                        "AOT550": {
                            "bounds": [0.001, 0.5],
                            "scale": 0.1,
                            "prior_mean": 0.1,
                            "prior_sigma": 1.0,
                            "init": 0.10
                        }
                    },
                    "lut_grid": {
                        "H2OSTR": [1.0, 1.5, 2.0, 2.5],
                        "AOT550": [0.001, 0.1, 0.3, 0.5]
                    },
                    "unknowns": {}
                }
            },
            "inversion": {
                "windows": [[400.0, 1300.0], [1450, 1780.0], [1950.0, 2450.0]]
            }
        }
    elif config == "inverse":
        return {
            "input": {
                "measured_radiance_file": absjoin(base_directory, "remote", "ang20140625_v2gx_rdn_snr100")
            },
            "output": {
                "estimated_reflectance_file": absjoin(base_directory, "output", "modeled_reflectance"),
                "posterior_uncertainty_file": absjoin(base_directory, "output", "modeled_uncertainty"),
                "estimated_state_file": absjoin(base_directory, "output", "modeled_state")
            },
            "forward_model": {
                "instrument": {
                    "wavelength_file": absjoin(base_directory, "data", "wavelengths.txt"),
                    "integrations": 1,
                    "SNR": 300
                },
                "multicomponent_surface": {
                    "surface_file": absjoin(base_directory, "data", "simple_surface_model.mat")
                },
                "modtran_radiative_transfer": {
                    "wavelength_file": absjoin(base_directory, "data", "wavelengths.txt"),
                    "lut_path": absjoin(base_directory, "luts"),
                    "modtran_template_file": absjoin(base_directory, "config", "modtran.json"),
                    "domain": {
                        "start": 350,
                        "end": 2520,
                        "step": 0.1
                    },
                    "statevector": {
                        "H2OSTR": {
                            "bounds": [1.0, 2.5],
                            "scale": 0.1,
                            "prior_mean": 1.75,
                            "prior_sigma": 5.0,
                            "init": 1.75
                        },
                        "AOT550": {
                            "bounds": [0.001, 0.5],
                            "scale": 0.5,
                            "prior_mean": 0.1,
                            "prior_sigma": 1.0,
                            "init": 0.10
                        }
                    },
                    "lut_grid": {
                        "H2OSTR": [1.0, 1.5, 2.0, 2.5],
                        "AOT550": [0.001, 0.1, 0.3, 0.5]
                    },
                    "unknowns": {}
                }
            },
            "inversion": {
                "windows": [
                    [400.0, 1300.0],
                    [1450, 1780.0],
                    [1950.0, 2450.0]
                ]
            }
        }
    else:
        raise Exception(
            "Parameter 'model' must be one of ['modtran','forward','inverse'].")
