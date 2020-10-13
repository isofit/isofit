Best Practices
==============

ISOFIT is highly configurable.  This provides a great deal of design flexibility for analysts to build their own custom retrieval algorithms for specific investigations.  Scientists can construct custom instrument models to handle new sensors, or define new observation uncertanties to account for model discrepancy errors.  They can refine or constrain the prior distribution based on background knowledge of the state vector.  This flexibility can be powerful, but it can also be daunting for the beginner.  Consequently, we have developed recommendations for best practices that are likely to provide good results over a wide range of conditions.

Surface Models
--------------

The multicomponent surface model is most universal and forgiving.  We recommend constructing Gaussian PDFs from diverse libraries of terrestrial and aquatic spectra, with correlations only in the key water absorption features at 940 and 1140 nm.  Use reference wavelengths for normalization and distance calculations that exclude the deep water absorption features at 1440 and 1880 nm.  An example configuration file formed from libraries in our distribution, for the wavelengths from 380-2500 nm, might be:

.. code-block:: JSON

  {
  "output_model_file": "surface_model.mat",
  "wavelength_file":   "wavelengths.txt",
  "normalize":"Euclidean",
  "reference_windows":[[400,1300],[1450,1700],[2100,2450]],
  "sources":
    [
      {
        "input_spectrum_files":
          [
            "path_to_isofit/data/reflectance/surface_model_ucsb"
          ],
        "n_components": 8,
        "windows": [
          {
            "interval":[300,890],
            "regularizer":100,
            "correlation":"decorrelated"
          },
          {
            "interval":[890,990],
            "regularizer":1e-6,
            "correlation":"EM"
          },
          {
            "interval":[990,1090],
            "regularizer":100,
            "correlation":"decorrelated"
          },
          {
            "interval":[1090,1190],
            "regularizer":1e-6,
            "correlation":"EM"
          },
          {
            "interval":[1190,2500],
            "regularizer":100,
            "correlation":"decorrelated"
          }
        ]
      },
      {
        "input_spectrum_files":
          [
            "path_to_isofit/data/reflectance/ocean_spectra_rev2"
          ],
        "n_components": 4,
        "windows": [
          {
            "interval":[300,890],
            "regularizer":100,
            "correlation":"decorrelated"
          },
          {
            "interval":[890,990],
            "regularizer":1e-6,
            "correlation":"EM"
          },
          {
            "interval":[990,1090],
            "regularizer":100,
            "correlation":"decorrelated"
          },
          {
            "interval":[1090,1190],
            "regularizer":1e-6,
            "correlation":"EM"
          },
          {
            "interval":[1190,2500],
            "regularizer":100,
            "correlation":"decorrelated"
          }
        ]
      }
  ]
  }



Note that the surface model is normalized with the Euclidean norm.  In the top-level configuration file, the "select_on_init" parameter should be set to True, and the "selection_metric" field to "Euclidean."  An example surface configuration block might be:

.. code-block:: JSON

     "surface": {
      "surface_category": "multicomponent_surface",
      "surface_file": "surface.mat"
      "select_on_init":true,
      "selection_metric":"Euclidean"
    },

    
Instrument Models
----------------

We recommend instrument models based on a three-channel parametric noise description.  These models predict noise-equivalent change in radiance as a function of :math:`L`, the radiance at sensor, with the relation :math:`L_{noisy} = a\sqrt{b+L}+c`.  They are stored as five-column ASCII text files with columns representing: wavelength; the a, b, and c coefficients; and the Root Mean Squared approximation error for the coefficient fitting, respectively.  An example is provided in the data/avirisng_noise.txt file.  We also recommend channelized uncertainty files representing the standard deviation of residuals due to forward model or wavelength calibration and response errors.  Finally, we recommend a 0-1% uncorrelated radiometric uncertainty term, depending on the confidence in the radiometric calibration of the instrument.  Certain extreme cases may require higher values. An example instrument configuration might be:

.. code-block:: JSON

   "instrument": {
     "wavelength_file": "wavelengths.txt",
     "parametric_noise_file": "path_to_isofit/data/avirisng_noise.txt",
     "integrations":1,
     "unknowns": {
       "channelized_radiometric_uncertainty_file": "path_to_isofit/data/avirisng_systematic_error.txt",
       "uncorrelated_radiometric_uncertainty": 0.01
     }
   },

The "integrations" field represents the number of coadded spectra that contribute to the measurement; it should typically be set to unity unless one is analyzing the average spectrum from a large area.

Atmosphere
----------------

We highly recommend the MODTRAN 6.0 radiative transfer model over LibRadTran and 6SV options for full-spectrum (380-2500) imaging spectroscopy.  We recommend retrieving water vapor and aerosol optical depth in the VSWIR range, water vapor and ozone in the thermal IR.  For aerosol optical properties, we recommend the third aerosol type found the aerosol file data/aerosol_model.txt.  This can be selected by including the "AERFRAC_2" element in the state vector and lookup tables.  For a simplified configuration that does not include variable viewing geometry, consider something like:

.. code-block:: JSON
   "radiative_transfer": {
            "lut_grid": {
                "AERFRAC_2": [ 0.001,  0.1673, 0.3336,  0.5 ],
                "H2OSTR": [ 1.0, 1.2, 1.4, 1.6, 1.8 ]
            },
            "radiative_transfer_engines": {
                "vswir": {
                    "aerosol_model_file": "path_to_isofit/data/aerosol_model.txt",
                    "aerosol_template_file": "path_to_isofit/data/aerosol_template.json",
                    "engine_base_dir": "path_to_MODTRAN6.0.0/",
                    "engine_name": "modtran",
                    "lut_names": [  "H2OSTR", "AERFRAC_2" ],
                    "lut_path": "./lut_directory/",
                    "statevector_names": [ "H2OSTR",  "AERFRAC_2"],
                    "template_file": "path_to_modtran_6_template.json"
                }
            },
            "statevector": {
                "AERFRAC_2": {
                    "bounds": [ 0.001,  0.5 ],
                    "init": 0.050,
                    "prior_mean": 0.050,
                    "prior_sigma": 10.0,
                    "scale": 1
                },
                "H2OSTR": {
                    "bounds": [  1.0, 1.8 ],
                    "init": 1.4,
                    "prior_mean": 1.4,
                    "prior_sigma": 100.0,
                    "scale": 0.01
                }
            },
            "unknowns": {
                "H2O_ABSCO": 0.0
            }
        }

Note that all atmospheric parameters have extremely wide and uninformed prior distributions.  More advanced users, or those with very heterogeneous flightlines, may wish to track the unique viewing geometry of every pixel in the image.  They should add the "GNDALT", "OBSZEN", and possibly "TRUEAZ" terms to the lookup tables (but not the state vector).  It is important to pass in an OBS-format metadata file in the input block, so that the program knows the geometry associated with each pixel.


Inversion Methods
----------------

We recommend excluding deep water features at 1440 nm and 1880 nm from the inversion windows.  We recommend a multiple-start inversion with four gridpoints at low and high values of atmospheric aerosol and water vapor.  A typical inversion configuration might be:

.. code-block:: JSON

    "inversion": {
        "integration_grid": {
            "AERFRAC_2": [
                0.03,
                0.14
            ],
            "H2OSTR": [
                0.5496509736776353,
                1.1583518081903457
            ]
        },
        "inversion_grid_as_preseed": true,
        "windows": [
            [
                400.0,
                1300.0
            ],
            [
                1450,
                1780.0
            ],
            [
                2050.0,
                2450.0
            ]
        ]
    },



