import logging

from isofit.retrieval.utils import utils

Logger = logging.getLogger(__name__)


@utils.log_elapse
def preprocess(config) -> None:
    """
    Spectrally interpolate bad radiance channels and compute the sky view factor.
    """
    p = config.preprocess
    if p.interpolate_radiance and p.rdn == config.setup.rdn:
        raise ValueError(
            "preprocess.interpolate_radiance requires preprocess.rdn to be set to "
            "where the interpolated radiance is written, not setup.rdn"
        )

    if p.skyview and p.svf in (None, config.input.skyview_factor_file):
        raise ValueError(
            "preprocess.skyview requires preprocess.svf to be set to where the "
            "sky view factor is written, not input.skyview_factor_file"
        )

    if not (p.interpolate_radiance or p.skyview):
        Logger.info("Nothing to preprocess")
        return

    # TODO: Port from apply_oe
    # 1. interpolate_radiance: isofit.utils.interpolate_spectra from setup.rdn
    #    to p.rdn, skipped when it exists. apply_oe's --interpolate_inplace is
    #    dropped: overwriting an input breaks restarting the stage
    # 2. skyview: isofit.utils.skyview with method=p.skyview, from setup.obs
    #    (slope) or setup.loc (horizon), to p.svf, skipped when it exists.
    #    isofit.utils.skyview names its output sky_view_factor in a directory,
    #    so it is written to p.svf.parent and renamed. The [0, 1] check in
    #    setup should then move to a shared helper so the computed factor is
    #    checked the same way
    raise NotImplementedError("preprocess")
