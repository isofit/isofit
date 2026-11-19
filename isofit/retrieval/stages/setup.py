import logging

import numpy as np
from spectral.io import envi

from isofit.core.common import envi_header
from isofit.retrieval.utils import utils

Logger = logging.getLogger(__name__)


@utils.log_elapse
def setup(config) -> None:
    """
    Validate the inputs, create the directories of every configured path, stage
    the input and instrument files, and write the wavelength file.
    """
    inputs, fm, s = config.input, config.forward_model, config.setup
    rdn = inputs.measured_radiance_file

    # Existence and the obs/loc shapes are checked by InputConfig validation
    required = ("measured_radiance_file", "obs_file", "loc_file")
    if missing := [name for name in required if getattr(inputs, name) is None]:
        raise ValueError(f"setup requires input.{', input.'.join(missing)} to be set")

    if svf := inputs.skyview_factor_file:
        shape, expected = utils.envi_shape(svf), utils.envi_shape(rdn)
        if shape != expected:
            raise ValueError(
                f"input.skyview_factor_file size is {shape}, which does not match "
                f"the radiance size: {expected}"
            )

        ds = envi.open(envi_header(str(svf)), str(svf))
        if (top := np.nanmax(ds.open_memmap())) > 1:
            raise ValueError(
                f"input.skyview_factor_file has a max of {top}, must be within [0, 1]"
            )

    # Rather than a fixed layout, create wherever the config writes to
    dirs = {path.parent for path in utils.walk_paths(config)}
    dirs |= {config.output.dir, fm.atmosphere.sim_path}
    Logger.info(f"Creating {len(dirs)} working directories")
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)

    for name, kind in zip(required, ("rdn", "obs", "loc")):
        utils.stage(getattr(inputs, name), getattr(s, kind), envi=True)

    utils.stage(
        s.channelized_uncertainty_file,
        fm.instrument.unknowns.channelized_radiometric_uncertainty_file,
    )
    utils.stage(s.model_discrepancy_file, fm.model_discrepancy_file)
    utils.stage(s.eof_file, fm.instrument.eof_path)

    if not (file := fm.instrument.wavelength_file):
        raise ValueError(
            "forward_model.instrument.wavelength_file must be set as the path to "
            "write the wavelength file to"
        )

    wl, fwhm = utils.read_wavelengths(rdn, s.wavelength_file)
    Logger.info(f"Writing {len(wl)} wavelengths to {file}")
    utils.write_wavelengths(file, wl, fwhm)
