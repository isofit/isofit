import logging

from isofit.retrieval.utils import utils

Logger = logging.getLogger(__name__)


@utils.log_elapse
def background(config) -> None:
    """
    Estimate the background reflectance of each pixel from the presolve, for
    the adjacency effect.
    """
    if not config.presolve.enabled:
        raise ValueError("The background stage requires presolve.enabled")

    b = config.background
    if utils.skip(b.rfl) and utils.skip(b.subs):
        return

    # TODO: Port from isofit.utils.adjacency.background_reflectance, which takes
    # a template_construction.Pathnames; rewrite it against explicit files:
    #   presolve config   presolve.file
    #   presolve state    presolve.state
    #   labels            segment.labels
    #   loc               setup.loc
    #   altitudes         Scene.load(config).altitude_km / .elevation_km
    #   output            b.rfl, and its superpixel extraction b.subs
    raise NotImplementedError("background")
