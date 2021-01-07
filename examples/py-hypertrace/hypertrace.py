#!/usr/bin/env python3
#
# Authors: Alexey Shiklomanov

import copy
import pathlib
import json
import shutil
import logging

import numpy as np
import spectral as sp
from scipy.io import loadmat
from scipy.interpolate import interp1d

from isofit.core.isofit import Isofit
from isofit.utils import empirical_line, segment, extractions
from isofit.utils.apply_oe import write_modtran_template

logger = logging.getLogger(__name__)

def do_hypertrace(isofit_config, wavelength_file, reflectance_file,
                  rtm_template_file,
                  lutdir, outdir,
                  surface_file="./data/prior.mat",
                  noisefile=None, snr=300,
                  aod=0.1, h2o=1.0, atmosphere_type="ATM_MIDLAT_WINTER",
                  atm_aod_h2o=None,
                  solar_zenith=0, observer_zenith=0,
                  solar_azimuth=0, observer_azimuth=0,
                  observer_altitude_km=99.9,
                  dayofyear=200,
                  latitude=34.15, longitude=-118.14,
                  localtime=10.0,
                  elevation_km=0.01,
                  inversion_mode="inversion",
                  use_empirical_line=False,
                  calibration_uncertainty_file=None,
                  n_calibration_draws=1,
                  calibration_scale=1,
                  create_lut=True,
                  overwrite=False):
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

        rtm_template_file: Path to the atmospheric RTM template. For LibRadtran,
        note that this is slightly different from the Isofit template in that
        the Isofit fields are surrounded by two sets of `{{` while a few
        additional options related to geometry are surrounded by just `{` (this
        is because Hypertrace does an initial pass at formatting the files).

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

      atmosphere_type: LibRadtran or Modtran atmosphere type. See RTM
      manuals for details. Default = `ATM_MIDLAT_WINTER`

      atm_aod_h2o: A list containing three elements: The atmosphere type, AOD,
      and H2O. This provides a way to iterate over specific known atmospheres
      that are combinations of the three previous variables. If this is set, it
      overrides the three previous arguments. Default = `None`

      solar_zenith, observer_zenith: Solar and observer zenith angles,
      respectively (0 = directly overhead, 90 = horizon). These are in degrees
      off nadir. Default = 0 for both. (Note that off-nadir angles make
      LibRadtran run _much_ more slowly, so be prepared if you need to generate
      those LUTs). (Note: For `modtran` and `modtran_simulator`, `solar_zenith`
      is calculated from the `gmtime` and location, so this parameter is ignored.)

      solar_azimuth, observer_azimuth: Solar and observer azimuth angles,
      respectively, in degrees. Observer azimuth is the sensor _position_ (so
      180 degrees off from view direction) relative to N, rotating
      counterclockwise; i.e., 0 = Sensor in N, looking S; 90 = Sensor in W,
      looking E (this follows the LibRadtran convention). Default = 0 for both.
      Note: For `modtran` and `modtran_simulator`, `observer_azimuth` is used as
      `to_sensor_azimuth`; i.e., the *relative* azimuth of the sensor. The true
      solar azimuth is calculated from lat/lon and time, so `solar_azimuth` is ignored.

      observer_altitude_km: Sensor altitude in km. Must be less than 100. Default = 99.9.
      (`modtran` and `modtran_simulator` only)

      dayofyear: Julian date of observation. Default = 200
      (`modtran` and `modtran_simulator` only)

      latitude, longitude: Decimal degree coordinates of observation. Default =
      34.15, -118.14 (Pasadena, CA).
      (`modtran` and `modtran_simulator` only)

      localtime: Local time, in decimal hours (0-24). Default = 10.0
      (`modtran` and `modtran_simulator` only)

      elevation_km: Target elevation above sea level, in km. Default = 0.01
      (`modtran` and `modtran_simulator` only)

      inversion_mode: Inversion algorithm to use. Must be either "inversion"
      (default) for standard optimal estimation, or "mcmc_inversion" for MCMC.

      use_empirical_line: (boolean, default = `False`) If `True`, perform
      atmospheric correction on a segmented image and then resample using the
      empirical line method. If `False`, run Isofit pixel-by-pixel.

      overwrite: (boolean, default = `False`) If `False` (default), skip steps
      where output files already exist. If `True`, run the full workflow
      regardless of existing files.
    """

    outdir = mkabs(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    assert observer_altitude_km < 100, "Isofit 6S does not support altitude >= 100km"

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
        atmosphere_type = atm_aod_h2o[0]
        aod = atm_aod_h2o[1]
        h2o = atm_aod_h2o[2]

    atmtag = f"aod_{aod:.3f}__h2o_{h2o:.3f}"
    if calibration_uncertainty_file is not None:
        caltag = f"cal_{pathlib.Path(calibration_uncertainty_file).stem}__" +\
                f"draw_{n_calibration_draws}__" +\
                f"scale_{calibration_scale}"
    else:
        caltag = "cal_NONE__draw_0__scale_0"

    if create_lut:
        lutdir = mkabs(lutdir)
        lutdir.mkdir(parents=True, exist_ok=True)
        vswir_conf = forward_settings["radiative_transfer"]["radiative_transfer_engines"]["vswir"]
        atmospheric_rtm = vswir_conf["engine_name"]

        if atmospheric_rtm == "libradtran":
            lrttag = f"atm_{atmosphere_type}__" +\
                f"szen_{solar_zenith:.2f}__" +\
                f"ozen_{observer_zenith:.2f}__" +\
                f"saz_{solar_azimuth:.2f}__" +\
                f"oaz_{observer_azimuth:.2f}"
            lutdir2 = lutdir / lrttag
            lutdir2.mkdir(parents=True, exist_ok=True)
            lrtfile = lutdir2 / "lrt-template.inp"
            with open(rtm_template_file, "r") as f:
                fs = f.read()
                open(lrtfile, "w").write(fs.format(
                    atmosphere=atmosphere_type, solar_azimuth=solar_azimuth,
                    solar_zenith=solar_zenith,
                    cos_observer_zenith=np.cos(observer_zenith * np.pi / 180.0),
                    observer_azimuth=observer_azimuth
                ))
            open(lutdir2 / "prescribed_geom", "w").write(f"99:99:99   {solar_zenith}  {solar_azimuth}")

        elif atmospheric_rtm in ("modtran", "simulated_modtran"):
            loctag = f"atm_{atmosphere_type}__" +\
                f"alt_{observer_altitude_km:.2f}__" +\
                f"doy_{dayofyear:.0f}__" +\
                f"lat_{latitude:.3f}__lon_{longitude:.3f}"
            angtag = f"az_{observer_azimuth:.2f}__" +\
                f"zen_{180 - observer_zenith:.2f}__" +\
                f"time_{localtime:.2f}__" +\
                f"elev_{elevation_km:.2f}"
            lrttag = loctag + "/" + angtag
            lutdir2 = lutdir / lrttag
            lutdir2.mkdir(parents=True, exist_ok=True)
            lrtfile = lutdir2 / "modtran-template-h2o.json"
            mt_params = {
                "atmosphere_type": atmosphere_type,
                "fid": "hypertrace",
                "altitude_km": observer_altitude_km,
                "dayofyear": dayofyear,
                "latitude": latitude,
                "longitude": longitude,
                "to_sensor_azimuth": observer_azimuth,
                "to_sensor_zenith": 180 - observer_zenith,
                "gmtime": localtime,
                "elevation_km": elevation_km,
                "output_file": lrtfile,
                "ihaze_type": "AER_NONE"
            }
            write_modtran_template(**mt_params)
            mt_params["ihaze_type"] = "AER_RURAL"
            mt_params["output_file"] = lutdir2 / "modtran-template.json"
            write_modtran_template(**mt_params)

            vswir_conf["modtran_template_path"] = str(mt_params["output_file"])
            if atmospheric_rtm == "simulated_modtran":
                vswir_conf["interpolator_base_path"] = str(lutdir2 / "sRTMnet_interpolator")
                # These need to be absolute file paths
                for path in ["emulator_aux_file", "emulator_file",
                             "earth_sun_distance_file", "irradiance_file"]:
                    vswir_conf[path] = str(mkabs(vswir_conf[path]))

        else:
            raise ValueError(f"Invalid atmospheric rtm {atmospheric_rtm}")

        vswir_conf["lut_path"] = str(lutdir2)
        vswir_conf["template_file"] = str(lrtfile)

    outdir2 = outdir / lrttag / noisetag / priortag / atmtag / caltag
    outdir2.mkdir(parents=True, exist_ok=True)

    # Observation file, which describes the geometry
    # Angles follow LibRadtran conventions
    obsfile = outdir2 / "obs.txt"
    geomvec = [
        -999,              # path length; not used
        observer_azimuth,  # Degrees 0-360; 0 = Sensor in N, looking S; 90 = Sensor in W, looking E
        observer_zenith,   # Degrees 0-90; 0 = directly overhead, 90 = horizon
        solar_azimuth,     # Degrees 0-360; 0 = Sun in S; 90 = Sun in W.
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

    if radfile.exists() and not overwrite:
        logger.info("Skipping forward simulation because file exists.")
    else:
        fwdfile = outdir2 / "forward.json"
        json.dump(isofit_fwd, open(fwdfile, "w"), indent=2)
        logger.info("Starting forward simulation.")
        Isofit(fwdfile).run()
        logger.info("Forward simulation complete.")

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

    post_unc_path = outdir2 / "posterior-uncertainty"

    # Inverse mode
    est_state_file = outdir2 / "estimated-state"
    atm_coef_file = outdir2 / "atmospheric-coefficients"
    post_unc_file = outdir2 / "posterior-uncertainty"
    isofit_inv["output"] = {"estimated_reflectance_file": str(est_refl_file),
                            "estimated_state_file": str(est_state_file),
                            "atmospheric_coefficients_file": str(atm_coef_file),
                            "posterior_uncertainty_file": str(post_unc_file)}

    # Run the workflow
    if calibration_uncertainty_file is not None:
        # Apply calibration uncertainty here
        calmat = loadmat(calibration_uncertainty_file)
        cov = calmat["Covariance"]
        cov_l = np.linalg.cholesky(cov)
        cov_wl = np.squeeze(calmat["wavelengths"])
        rad_img = sp.open_image(str(radfile) + ".hdr")
        rad_wl = rad_img.bands.centers
        del rad_img
        for ical in range(n_calibration_draws):
            icalp1 = ical + 1
            radfile_cal = f"{str(radfile)}-{icalp1:02d}"
            reflfile_cal = f"{str(est_refl_file)}-{icalp1:02d}"
            statefile_cal = f"{str(est_state_file)}-{icalp1:02d}"
            atmfile_cal = f"{str(atm_coef_file)}-{icalp1:02d}"
            uncfile_cal = f"{str(post_unc_file)}-{icalp1:02d}"
            if pathlib.Path(reflfile_cal).exists() and not overwrite:
                logger.info("Skipping calibration %d/%d because output exists",
                            icalp1, n_calibration_draws)
                next
            logger.info("Applying calibration uncertainty (%d/%d)", icalp1, n_calibration_draws)
            sample_calibration_uncertainty(radfile, radfile_cal, cov_l, cov_wl, rad_wl,
                                           bias_scale=calibration_scale)
            logger.info("Starting inversion (calibration %d/%d)", icalp1, n_calibration_draws)
            do_inverse(
                copy.deepcopy(isofit_inv), radfile_cal, reflfile_cal,
                statefile_cal, atmfile_cal, uncfile_cal,
                overwrite=overwrite, use_empirical_line=use_empirical_line
            )
            logger.info("Inversion complete (calibration %d/%d)", icalp1, n_calibration_draws)

    else:
        if est_refl_file.exists() and not overwrite:
            logger.info("Skipping inversion because output exists.")
        else:
            logger.info("Starting inversion.")
            do_inverse(
                copy.deepcopy(isofit_inv), radfile, est_refl_file,
                est_state_file, atm_coef_file, post_unc_file,
                overwrite=overwrite, use_empirical_line=use_empirical_line
            )
            logger.info("Inversion complete.")
    logger.info("Workflow complete!")


##################################################
def do_inverse(isofit_inv: dict,
               radfile: pathlib.Path,
               est_refl_file: pathlib.Path,
               est_state_file: pathlib.Path,
               atm_coef_file: pathlib.Path,
               post_unc_file: pathlib.Path,
               overwrite: bool,
               use_empirical_line: bool):
    if use_empirical_line:
        # Segment first, then run on segmented file
        SEGMENTATION_SIZE = 40
        CHUNKSIZE = 256
        lbl_working_path = radfile.parent / str(radfile).replace("toa-radiance", "segmentation")
        rdn_subs_path = radfile.with_suffix("-subs")
        rfl_subs_path = est_refl_file.with_suffix("-subs")
        state_subs_path = est_state_file.with_suffix("-subs")
        atm_subs_path = atm_coef_file.with_suffix("-subs")
        unc_subs_path = post_unc_file.with_suffix("-subs")
        isofit_inv["input"]["measured_radiance_file"] = str(rdn_subs_path)
        isofit_inv["output"] = {
            "estimated_reflectance_file":  str(rfl_subs_path),
            "estimated_state_file":  str(state_subs_path),
            "atmospheric_coefficients_file":  str(atm_subs_path),
            "posterior_uncertainty_file": str(unc_subs_path)
        }
        if not overwrite and lbl_working_path.exists() and rdn_subs_path.exists():
            logger.info("Skipping segmentation and extraction because files exist.")
        else:
            logger.info("Fixing any radiance values slightly less than zero...")
            rad_img = sp.open_image(str(radfile) + ".hdr")
            rad_m = rad_img.open_memmap(writable=True)
            nearzero = np.logical_and(rad_m < 0, rad_m > -2)
            rad_m[nearzero] = 0.0001
            del rad_m
            del rad_img
            logger.info("Segmenting...")
            segment(spectra=(str(radfile), str(lbl_working_path)),
                    flag=-9999, npca=5, segsize=SEGMENTATION_SIZE, nchunk=CHUNKSIZE)
            logger.info("Extracting...")
            extractions(inputfile=str(radfile), labels=str(lbl_working_path),
                        output=str(rdn_subs_path), chunksize=CHUNKSIZE, flag=-9999)

    else:
        # Run Isofit directly
        isofit_inv["input"]["measured_radiance_file"] = str(radfile)
        isofit_inv["output"] = {
            "estimated_reflectance_file": str(est_refl_file),
            "estimated_state_file": str(est_state_file),
            "atmospheric_coefficients_file": str(atm_coef_file),
            "posterior_uncertainty_file": str(post_unc_file)
        }

    if not overwrite and pathlib.Path(isofit_inv["output"]["estimated_reflectance_file"]).exists():
        logger.info("Skipping inversion because output file exists.")
    else:
        invfile = radfile.parent / (str(radfile).replace("toa-radiance", "inverse") + ".json")
        json.dump(isofit_inv, open(invfile, "w"), indent=2)
        Isofit(invfile).run()

    if use_empirical_line:
        if not overwrite and est_refl_file.exists():
            logger.info("Skipping empirical line because output exists.")
        else:
            logger.info("Applying empirical line...")
            empirical_line(reference_radiance_file=str(rdn_subs_path),
                           reference_reflectance_file=str(rfl_subs_path),
                           reference_uncertainty_file=str(unc_subs_path),
                           reference_locations_file=None,
                           segmentation_file=str(lbl_working_path),
                           input_radiance_file=str(radfile),
                           input_locations_file=None,
                           output_reflectance_file=str(est_refl_file),
                           output_uncertainty_file=str(post_unc_file),
                           isofit_config=str(invfile))

def mkabs(path):
    """Make a path absolute."""
    path2 = pathlib.Path(path)
    return path2.expanduser().resolve()


def sample_calibration_uncertainty(input_file: pathlib.Path,
                                   output_file: pathlib.Path,
                                   cov_l: np.ndarray,
                                   cov_wl: np.ndarray,
                                   rad_wl: np.ndarray,
                                   bias_scale=1.0):
    input_file_hdr = str(input_file) + ".hdr"
    output_file_hdr = str(output_file) + ".hdr"
    shutil.copy(input_file, output_file)
    shutil.copy(input_file_hdr, output_file_hdr)

    img = sp.open_image(str(output_file_hdr))
    img_m = img.open_memmap(writable=True)

    # Here, we assume that the calibration bias is constant across the entire
    # image (i.e., the same bias is added to all pixels).
    z = np.random.normal(size=cov_l.shape[0], scale=bias_scale)
    Az = 1.0 + cov_l @ z
    # Resample the added noise vector to match the wavelengths of the target
    # image.
    Az_resampled = interp1d(cov_wl, Az, fill_value="extrapolate")(rad_wl)
    img_m *= Az_resampled
    return output_file
