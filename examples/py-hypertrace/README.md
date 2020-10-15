# Hypertrace

Hypertrace is a wrapper around Isofit that links forward and inverse modeling.
Starting from known surface reflectance and atmospheric conditions, simulate top-of-atmosphere radiance and then perform atmospheric correction to estimate surface reflectance from this simulated radiance.

## Lightning introduction

First, make sure you have Isofit installed.
In addition, you will need to have a working compiled version of the LibRadtran atmospheric radiative transfer model (see Installation section below).

Second, download the support datasets from here: https://github.com/ashiklom/isofit/releases/tag/hypertrace-data (~60 MB compressed, ~135 MB uncompressed).
Extract these into the `examples/py-hypertrace/` directory (so that you have a `examples/py-hypertrace/hypertrace-data` folder).

Finally, make a local copy of the `config-example.json` and modify to fit your system (most important is radiative transfer engine base directly and environment.

``` sh
cp config-example.json myconfig.json
```

Then, run the `workflow.py` script with your config file as an argument:

```sh
python workflow.py myconfig.json
```

Hypertrace also ships with a script to quickly calculate some basic summary statistics and diagnostics.
This script also takes the config file as an input:

``` sh
python summarize.py myconfig.json
```

Note that these statistics are (1) calculated inefficiently, and (2) are probably simpler than what is warranted by the data.
They are primarily intended for quick diagnostics on relatively small images (ones that fit in memory multiple times).


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
    - `lrt_atmosphere_type` -- LibRadtran atmosphere type. See LibRadtran manual for details. Default = `midlatitude_winter`
    - `atm_aod_h2o` -- A list containing three elements: The atmosphere type, AOD, and H2O. This provides a way to iterate over specific known atmospheres that are combinations of the three previous variables. If this is set, it overrides the three previous arguments. Default = `None`
        - For example, `"atm_aod_h2o": [["midlatitude_winter", 0.1, 2.0], ["midlatitude_summer", 0.08, 1.5]]` means to iterate over _two_ atmospheres. On the other hand, a config like `"atm": ["midlatitude_winter", "midlatitude_summer"], "aod": [0.1, 0.08], "h2o": [2.0, 1.5]` would run 2 x 2 x 2 = 8 atmospheres -- one for each combination of these three fields.
    - `solar_zenith`, `observer_zenith` -- Solar and observer zenith angles, respectively (0 = directly overhead, 90 = horizon). These are in degrees off nadir. Default = 0 for both. (Note that using LibRadtran to generate look up tables for off-nadir angles is ~10x slower than at nadir; however, this step only affects the LUT generation, so it shouldn't introduce additional delay if these LUTs already exist).
    - `solar_azimuth`, `observer_azimuth` -- Solar and observer azimuth angles, respectively, in degrees. Observer azimuth is the sensor _position_ (so 180 degrees off from view direction) relative to N, rotating counterclockwise; i.e., 0 = Sensor in N, looking S; 90 = Sensor in W, looking E (this follows the LibRadtran convention). Default = 0 for both.
    - `inversion_mode` -- One of three options:
        - `"inversion"` (default) -- Standard optimal estimation algorithm in Isofit.
        - `"mcmc_inversion"` -- MCMC inversion using Metropolis-Hastings algorithm. Note that this probably takes significantly longer than the default.
            - If you use this option, you are advised to set `isofit.implementation.inversion.mcmc.verbose = False` in your JSON config file. Otherwise, this will print to screen at _every_ MCMC iteration, which is 10,000 times per inversion!
        - `"simple"` -- Algebraic inversion. This uses the same underlying code as `"inversion"`, but for the least-squares optimization step, sets the number of function evaluations to 1. This works because Isofit uses the algebraic inversion as its initial condition.
    - `create_lut` -- If `true` (default), use LibRadtran to create look-up tables as necessary. If `false`, use whatever LUT configuration (path, engine, etc.) you provided in the Isofit config (note that this is untested and experimental).
    
### Libradtran

To generate your own atmospheric look-up tables, you'll need a working installation of LibRadTran.
Follow the instructions in the [Isofit `README`](https://github.com/ashiklom/isofit/tree/r-geom-2#quick-start-with-libradtran-20x) to install.
Note that you must install LibRadTran into the source code directory for it to work properly; i.e.

``` sh
# From the LibRadTran source code directory:
./configure --prefix=$(pwd)
make
```
