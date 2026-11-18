import logging

from isofit.retrieval.utils import utils

Logger = logging.getLogger(__name__)


@utils.log_elapse
def interpolate(config) -> None:
    """
    Extend the superpixel solutions to every pixel with the empirical or
    analytical line.
    """
    i = config.interpolate
    if not config.segment.enabled:
        raise ValueError("The interpolate stage requires segment.enabled")

    # The products of the pipeline config, as the main config writes superpixels
    out = config.output
    rfl, uncert = out.estimated_reflectance_file, out.posterior_uncertainty_file
    if utils.skip(rfl) and utils.skip(uncert):
        return

    # Arbitrary, but backwards compatible and works well with the defaults
    neighbors = i.num_neighbors or [round(3950 / 9 - 35 / 36 * config.segment.size)]

    # TODO: Port from apply_oe
    # empirical:  isofit.utils.empirical_line with segment.rdn/loc, the subs
    #             rfl/uncert outputs of configure.file, segment.labels,
    #             preprocess.rdn, setup.loc, and neighbors[0]
    # analytical: isofit.utils.analytical_line with preprocess.rdn, setup.loc,
    #             setup.obs, segment.labels, preprocess.svf, background.rfl when
    #             enabled, neighbors, and i.atm_sigma
    # Both take the main config configure.file and write rfl and uncert
    raise NotImplementedError(f"interpolate: {i.method}")
