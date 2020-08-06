#!/usr/bin/env python3

import copy
import pathlib
import json

import numpy as np
from scipy.io import loadmat

from isofit.core.isofit import Isofit


def do_hypertrace(isofit_config, wavelength_file, reflectance_file,
                  libradtran_template_file,
                  lutdir, outdir,
                  surface_file="./data/prior.mat",
                  noisefile=None, snr=300,
                  aod=0.1, h2o=1.0, lrt_atmosphere_type="midlatitude_winter",
                  atm_aod_h2o=None,
                  solar_zenith=0, observer_zenith=0,
                  solar_azimuth=0, observer_azimuth=0,
                  inversion_mode="inversion",
                  create_lut=True):
    """One iteration of the hypertrace workflow.

    Required arguments:
        isofit_config: dict of isofit configuration options

        `wavelength_file`: Path to ASCII space delimited table containing two
        columns, wavelength and full width half max (FWHM); both in nanometers.

        `reflectance_file`: Path to input reflectance file. Note that this has
        to be an ENVI-formatted binary reflectance file, and this path is to the
        associated header file (`.hdr`), not the image file itself (following
        the convention of the `spectral` Python library, which will be used to
        read this file).

        libradtran_template_file: Path to the LibRadtran template. Note that
        this is slightly different from the Isofit template in that the Isofit
        fields are surrounded by two sets of `{{` while a few additional options
        related to geometry are surrounded by just `{` (this is because
        Hypertrace does an initial pass at formatting the files).

        `lutdir`: Directory where look-up tables will be stored. Will be created
        if missing.

        `outdir`: Directory where outputs will be stored. Will be created if
        missing.

    Keyword arguments:
      surface_file: Matlab (`.mat`) file containing a multicomponent surface
      prior. See Isofit documentation for details.

      noisefile: Parametric instrument noise file. See Isofit documentation for
      details. Default = `None`

      snr: Instrument signal-to-noise ratio. Ignored if `noisefile` is present.
      Default = 300

      aod: True aerosol optical depth. Default = 0.1

      h2o: True water vapor content. Default = 1.0

      lrt_atmosphere_type: LibRadtran atmosphere type. See LibRadtran manual for
      details. Default = `midlatitude_winter`

      atm_aod_h2o: A list containing three elements: The atmosphere type, AOD,
      and H2O. This provides a way to iterate over specific known atmospheres
      that are combinations of the three previous variables. If this is set, it
      overrides the three previous arguments. Default = `None`

      solar_zenith, observer_zenith: Solar and observer zenith angles,
      respectively (0 = directly overhead, 90 = horizon). These are in degrees
      off nadir. Default = 0 for both. (Note that off-nadir angles make
      LibRadtran run _much_ more slowly, so be prepared if you need to generate
      those LUTs).

      solar_azimuth, observer_azimuth: Solar and observer azimuth angles,
      respectively, in degrees. Observer azimuth is the sensor _position_ (so
      180 degrees off from view direction) relative to N, rotating
      counterclockwise; i.e., 0 = Sensor in N, looking S; 90 = Sensor in W,
      looking E (this follows the LibRadtran convention). Default = 0 for both.

      inversion_mode: Inversion algorithm to use. Must be either "inversion"
      (default) for standard optimal estimation, or "mcmc_inversion" for MCMC.
    """

    outdir = mkabs(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    isofit_common = copy.deepcopy(isofit_config)
    # NOTE: All of these settings are *not* copied, but referenced. So these
    # changes propagate to the `forward_settings` object below.
    forward_settings = isofit_common["forward_model"]
    instrument_settings = forward_settings["instrument"]
    # NOTE: This also propagates to the radiative transfer engine
    instrument_settings["wavelength_file"] = str(mkabs(wavelength_file))
    surface_settings = forward_settings["surface"]
    surface_settings["surface_file"] = str(mkabs(surface_file))
    if noisefile is not None:
        noisetag = f"noise_{pathlib.Path(noisefile).stem}"
        if "SNR" in instrument_settings:
            instrument_settings.pop("SNR")
        instrument_settings["parametric_noise_file"] = str(mkabs(noisefile))
        if "integrations" not in instrument_settings:
            instrument_settings["integrations"] = 1
    elif snr is not None:
        noisetag = f"snr_{snr}"
        instrument_settings["SNR"] = snr

    priortag = f"prior_{pathlib.Path(surface_file).stem}__" +\
        f"inversion_{inversion_mode}"

    if atm_aod_h2o is not None:
        lrt_atmosphere_type = atm_aod_h2o[0]
        aod = atm_aod_h2o[1]
        h2o = atm_aod_h2o[2]

    lrttag = f"atm_{lrt_atmosphere_type}__" +\
        f"szen_{solar_zenith}__" +\
        f"ozen_{observer_zenith}__" +\
        f"saz_{solar_azimuth}__" +\
        f"oaz_{observer_azimuth}"
    atmtag = f"aod_{aod}__h2o_{h2o}"

    if create_lut:
        lutdir = mkabs(lutdir)
        lutdir.mkdir(parents=True, exist_ok=True)
        lutdir2 = lutdir / lrttag
        lutdir2.mkdir(parents=True, exist_ok=True)
        lrtfile = lutdir2 / "lrt-template.inp"
        vswir_conf = forward_settings["radiative_transfer"]["radiative_transfer_engines"]["vswir"]
        with open(libradtran_template_file, "r") as f:
            fs = f.read()
            open(lrtfile, "w").write(fs.format(
                atmosphere=lrt_atmosphere_type, solar_azimuth=solar_azimuth,
                solar_zenith=solar_zenith,
                cos_observer_zenith=np.cos(observer_zenith * np.pi / 180.0),
                observer_azimuth=observer_azimuth
            ))
        open(lutdir2 / "prescribed_geom", "w").write(f"99:99:99   {solar_zenith}  {solar_azimuth}")
        vswir_conf["lut_path"] = str(lutdir2)
        vswir_conf["template_file"] = str(lrtfile)

    outdir2 = outdir / lrttag / noisetag / priortag / atmtag
    outdir2.mkdir(parents=True, exist_ok=True)

    # Observation file, which describes the geometry
    obsfile = outdir2 / "obs.txt"
    geomvec = [
        -999,              # path length; not used
        observer_azimuth,  # Degrees 0-360; 0 = Sensor in N, looking S; 90 = Sensor in W, looking E
        observer_zenith,   # Degrees 0-90; 0 = directly overhead, 90 = horizon
        solar_azimuth,     # Degrees 0-360; 0 = N, 90 = W, 180 = S, 270 = E
        solar_zenith,      # Same units as observer zenith
        180.0 - abs(observer_zenith),  # MODTRAN OBSZEN -- t
        observer_azimuth - solar_azimuth + 180.0,  # MODTRAN relative azimuth
        observer_azimuth,   # MODTRAN azimuth
        np.cos(observer_zenith * np.pi / 180.0)  # Libradtran cos obsever zenith
    ]
    np.savetxt(obsfile, np.array([geomvec]))

    isofit_common["input"] = {"obs_file": str(obsfile)}

    isofit_fwd = copy.deepcopy(isofit_common)
    isofit_fwd["input"]["reflectance_file"] = str(mkabs(reflectance_file))
    isofit_fwd["implementation"]["mode"] = "simulation"
    isofit_fwd["implementation"]["inversion"]["simulation_mode"] = True
    fwd_surface = isofit_fwd["forward_model"]["surface"]
    fwd_surface["surface_category"] = "surface"

    # Check that prior and wavelength file have the same dimensions
    prior = loadmat(mkabs(surface_file))
    prior_wl = prior["wl"][0]
    prior_nwl = len(prior_wl)
    file_wl = np.loadtxt(wavelength_file)
    file_nwl = file_wl.shape[0]
    assert prior_nwl == file_nwl, \
        f"Mismatch between wavelength file ({file_nwl}) " +\
        f"and prior ({prior_nwl})."

    fwd_surface["wavelength_file"] = str(wavelength_file)

    radfile = outdir2 / "toa-radiance"
    isofit_fwd["output"] = {"simulated_measurement_file": str(radfile)}
    fwd_state = isofit_fwd["forward_model"]["radiative_transfer"]["statevector"]
    fwd_state["AOT550"]["init"] = aod
    fwd_state["H2OSTR"]["init"] = h2o

    fwdfile = outdir2 / "forward.json"
    json.dump(isofit_fwd, open(fwdfile, "w"), indent=2)

    isofit_inv = copy.deepcopy(isofit_common)
    if inversion_mode == "simple":
        # Special case! Use the optimal estimation code, but set `max_nfev` to 1.
        inversion_mode = "inversion"
        imp_inv = isofit_inv["implementation"]["inversion"]
        if "least_squares_params" not in imp_inv:
            imp_inv["least_squares_params"] = {}
        imp_inv["least_squares_params"]["max_nfev"] = 1
    isofit_inv["implementation"]["mode"] = inversion_mode
    isofit_inv["input"]["measured_radiance_file"] = str(radfile)
    est_refl_file = outdir2 / "estimated-reflectance"
    isofit_inv["output"] = {"estimated_reflectance_file": str(est_refl_file)}

    invfile = outdir2 / "inverse.json"
    json.dump(isofit_inv, open(invfile, "w"), indent=2)

    # Run the workflow
    Isofit(fwdfile).run()
    Isofit(invfile).run()


def mkabs(path):
    """Make a path absolute."""
    path2 = pathlib.Path(path)
    return path2.expanduser().resolve()
