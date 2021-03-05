# Hypertrace

Hypertrace is a wrapper around Isofit that links forward and inverse modeling.
Starting from known surface reflectance and atmospheric conditions, simulate top-of-atmosphere radiance and then perform atmospheric correction to estimate surface reflectance from this simulated radiance.

## Lightning introduction

This tutorial will walk you through an example Hypertrace workflow using the (recommended) `sRTMnet` MODTRAN model emulator.
Instructions for other models (e.g., libRadtran) are provided below.
NOTE: All of these instructions assume you are operating from inside this directory (`examples/py-hypertrace`) unless otherwise indicated.

1. Install Isofit, following the standard instructions.

2. Create a folder called `6Sv-2.1`.
Download and compile the 6Sv atmospheric radiative transfer model inside this directory.
The default source code location is here (http://6s.ltdri.org/pages/downloads.html),
but if that link doesn't work, you can download a mirrored version from here (https://github.com/ashiklom/isofit/releases/tag/6sv-mirror).
Once downloaded, compile the code by calling `make` inside the source code directory.
(NOTE: If you have `gfortran/gcc` > v8.0, you may need to append the string ` -std=legacy` to the `FFLAGS` in the `Makefile` to prevent errors during compilation.)
The compiled model executable will be located in that directory.

3. Create a folder called `sRTMnet_v100`.
Download the `sRTMnet` model emulator from here (https://doi.org/10.5281/zenodo.4096627) and extract it inside this directory.
Note that these files are quite large -- ~3.3 GB!

4. Download the remaining Hypertrace support datasets from here (https://github.com/ashiklom/isofit/releases/tag/hypertrace-data) and extract them. 

A script has been provided that will accomplish this for you:
``` sh
./prepare_hypertrace_data.sh
```

5. At this point, confirm that, inside the `examples/py-hypertrace` directory, you have
a `6Sv-2.1` directory containing the `sixv2.1` binary executable;
a `sRTMnet_v100` directory containing a subdirectory also called `sRTMnet_v100` and a file `sRTMnet_v100_aux.npz`;
and
a `hypertrace-data` directory containing subdirectories including `noise`, `priors`, `wavelengths`, and `reflectance`.
Assuming this is the case, run the example Hypertrace workflow with the following command:
    ``` sh
    python3 workflow.py configs/example-srtmnet.json
    ```
    

6. Hypertrace also ships with an experimental script to quickly calculate some basic summary statistics and diagnostics.
Note that these statistics are (1) calculated inefficiently, and (2) are probably simpler than what is warranted by the data.
They are primarily intended for quick diagnostics on relatively small images (ones that fit in memory multiple times).
This script also takes the config file as an input:

    ``` sh
    python summarize.py configs/example-srtmnet.json
    ```


## Configuration file

Like Isofit, the configuration file is a `json` file.
Top level settings are as follows:

- `wavelength_file` -- Path to ASCII space delimited table containing two columns, wavelength and full width half max (FWHM); both in nanometers.
- `reflectance_file` -- Path to input reflectance file. This has to be an ENVI-formatted binary reflectance file, in BIL (band interleave line) or BIP (band interleave pixel) format. If the image name is `data/myimage`, it must have an associated header file called `data/myimage.hdr` that _must_ have, among other things, metadata on the wavelengths in the `wavelength` field.
- `libradtran_template_file` -- Path to the LibRadtran template. Note that this is different from the Isofit template in that the Isofit fields are surrounded by two sets of `{{` while a few additional options related to geometry are surrounded by just `{` (this is because Hypertrace does an initial pass at formatting the files).
- `lutdir` -- Directory where atmospheric look-up tables will be stored. Will be created if missing.
- `outdir` -- Directory where outputs will be stored. Will be created if missing.
- `isofit` -- Isofit configuration options (`forward_model`, `implementation`, etc.). This is included to allow you maximum flexiblity in modifying the behavior of Isofit. See the Isofit documentation for more details. Note that some of these will be overwritten by the Hypertrace workflow.
- `hypertrace` -- Each of these is a list of variables that will be iterated as part of Hypertrace. Specifically, Hypertrace will generate the factorial combination of every one of these lists and perform the workflow for each element of that list. Every keyword argument to the `do_hypertrace` function is supported (indeed, that's how they are passed in, via the `**kwargs` mechanism), and include the following:
    - `surface_file` -- Matlab (`.mat`) file containing a multicomponent surface prior. See Isofit documentation for details.
    - `noisefile` -- Parametric instrument noise file. See Isofit documentation for details. Default = `None`
    - `snr` -- Instrument signal-to-noise ratio. Ignored if `noisefile` is present. Default = 300
    - `aod` -- True aerosol optical depth. Default = 0.1
    - `h2o` -- True water vapor content. Default = 1.0
    - `atmosphere_type` -- LibRadtran atmosphere type. See LibRadtran manual for details. Default = `midlatitude_winter`
    - `atm_aod_h2o` -- A list containing three elements: The atmosphere type, AOD, and H2O. This provides a way to iterate over specific known atmospheres that are combinations of the three previous variables. If this is set, it overrides the three previous arguments. Default = `None`
        - For example, `"atm_aod_h2o": [["midlatitude_winter", 0.1, 2.0], ["midlatitude_summer", 0.08, 1.5]]` means to iterate over _two_ atmospheres. On the other hand, a config like `"atm": ["midlatitude_winter", "midlatitude_summer"], "aod": [0.1, 0.08], "h2o": [2.0, 1.5]` would run 2 x 2 x 2 = 8 atmospheres -- one for each combination of these three fields.
    - `solar_zenith`, `observer_zenith` -- Solar and observer zenith angles, respectively (0 = directly overhead, 90 = horizon). These are in degrees off nadir. Default = 0 for both. (Note that using LibRadtran to generate look up tables for off-nadir angles is ~10x slower than at nadir; however, this step only affects the LUT generation, so it shouldn't introduce additional delay if these LUTs already exist). (Note: For `modtran` and `modtran_simulator`, `solar_zenith` is calculated from the `gmtime` and location, so this parameter is ignored.)
    - `solar_azimuth`, `observer_azimuth` -- Solar and observer azimuth angles, respectively, in degrees. Observer azimuth is the sensor _position_ (so 180 degrees off from view direction) relative to N, rotating counterclockwise; i.e., 0 = Sensor in N, looking S; 90 = Sensor in W, looking E (this follows the LibRadtran convention). Default = 0 for both. Note: For `modtran` and `modtran_simulator`, `observer_azimuth` is used as `to_sensor_azimuth`; i.e., the *relative* azimuth of the sensor. The true solar azimuth is calculated from lat/lon and time, so `solar_azimuth` is ignored.
    - The following parameters are currently only supported for `modtran` and `modtran_simulator`:
        - `observer_altitude_km` -- Sensor altitude in km. Must be less than 100. Default = 99.9.
        - `dayofyear` -- Julian date of observation. Default = 200
        - `latitude, longitude` -- Decimal degree coordinates of observation. Default = 34.15, -118.14 (Pasadena, CA).
        - `localtime` -- Local time, in decimal hours (0-24). Default = 10.0
        - `elevation_km` -- Target elevation above sea level, in km. Default = 0.01
    - `inversion_mode` -- One of three options:
        - `"inversion"` (default) -- Standard optimal estimation algorithm in Isofit.
        - `"mcmc_inversion"` -- MCMC inversion using Metropolis-Hastings algorithm. Note that this probably takes significantly longer than the default.
            - If you use this option, you are advised to set `isofit.implementation.inversion.mcmc.verbose = False` in your JSON config file. Otherwise, this will print to screen at _every_ MCMC iteration, which is 10,000 times per inversion!
        - `"simple"` -- Algebraic inversion. This uses the same underlying code as `"inversion"`, but for the least-squares optimization step, sets the number of function evaluations to 1. This works because Isofit uses the algebraic inversion as its initial condition.
    - `create_lut` -- If `true` (default), use LibRadtran to create look-up tables as necessary. If `false`, use whatever LUT configuration (path, engine, etc.) you provided in the Isofit config (note that this is untested and experimental).
    
### Libradtran

Alternatively, you can generate LUTs using the open source atmospheric RTM LibRadtran.
To do this, you'll need a working installation of LibRadTran.
Follow the instructions in the [Isofit `README`](https://github.com/isofit/isofit#quick-start-with-libradtran-20x) to install.
Note that you must install LibRadTran into the source code directory for it to work properly; i.e.

``` sh
# From the LibRadTran source code directory:
./configure --prefix=$(pwd)
make
```

## Running with SLURM
Example SLURM job submission scripts can be found in the slurm/ directory.  For example, a basic sbatch example is provded in slurm/run_hypertrace_sbatch.sh. The script can be edited to match your specific HPC environment, including module, $PATH, and conda requirements.  Once ready you can run this script from the main py-hypertrace directory using, for example:

``` sh
sbatch -w node03 -c 12 --partition compute --job-name=py-hypertrace --mail-user=sserbin@bnl.gov slurm/run_hypertrace_sbatch.sh configs/libradtran.json
```

For an example that will run a broadcast simulation experiment across multiple nodes, you can modify the example run_hypertrace_sbatch_broadcast.sh. That script also leverages the functions contained in set_ray_params.py. For example:

``` sh
sbatch --partition compute --job-name=py-hypertrace --mail-user=sserbin@bnl.gov slurm/run_hypertrace_sbatch_broadcast.sh configs/libradtran.json
```

This will generate a run across more than one node, e.g. 
```
JOBID   PARTITION   NAME        USER     ST TIME  NODES NODELIST(REASON)
281     compute     py-hyper    sserbin  R  5:24      2 node[01-02]
```
