import logging

from isofit.retrieval.utils import utils

Logger = logging.getLogger(__name__)


@utils.log_elapse
def surface(config) -> None:
    """
    Build the surface model(s) from a .json surface config or check a .mat model
    against the radiance wavelengths, and classify the scene for a multisurface
    retrieval.
    """
    s = config.surface
    if s.file is None:
        raise ValueError("surface.file must be set to a surface model or config")

    if s.classify and s.file.suffix != ".json":
        raise ValueError("surface.classify requires surface.file to be a .json config")

    if s.classify and s.classes is None:
        raise ValueError("surface.classify requires surface.classes to be set")

    # TODO: Port from template_construction.check_surface_model and apply_oe
    # 1. .mat: check its wavelengths against forward_model.instrument.wavelength_file
    #    and stage it to forward_model.surface.surface_file
    # 2. .json: build it with isofit.utils.surface_model to
    #    forward_model.surface.surface_file, skipped when it exists. With
    #    multisurface, one model per surface_category, named
    #    {stem}_{category}{suffix}
    # 3. classify: isofit.utils.multicomponent_classification from setup.rdn,
    #    .obs, .loc to s.classes, using Scene.load(config).day_of_year
    # 4. The multisurface `Surfaces` mapping is scene-derived (the classes
    #    present in the classification) and is built by solve.surface
    raise NotImplementedError("surface")
