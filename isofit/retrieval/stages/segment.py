import logging

from isofit.retrieval.utils import utils

Logger = logging.getLogger(__name__)

CHUNKSIZE = 256
NODATA = -9999


def Sources(config) -> list[tuple]:
    """
    The working files extracted per superpixel, as ``(input, output, reducer)``.
    The sky view factor and surface classes are only extracted when set.
    """
    from isofit.utils import reducers

    p, s, seg = config.preprocess, config.setup, config.segment
    sources = [
        (p.rdn, seg.rdn, reducers.band_mean),
        (s.obs, seg.obs, reducers.band_mean),
        (s.loc, seg.loc, reducers.band_mean),
        (p.svf, seg.svf, reducers.band_mean),
        (config.surface.classes, seg.surface_class, reducers.class_priority),
    ]

    return [source for source in sources if source[0] is not None]


@utils.log_elapse
def segment(config) -> None:
    """
    Segment the radiance into SLIC superpixels and extract the mean of each
    working input per superpixel.
    """
    # TODO: Port from apply_oe
    # 1. isofit.utils.segment from preprocess.rdn to segment.labels with
    #    segment.size and segment.npca, skipped when it exists
    # 2. isofit.utils.extractions of each of Sources(config), skipped when its
    #    output exists. The background reflectance is extracted by the
    #    background stage, which runs after this
    raise NotImplementedError("segment")
