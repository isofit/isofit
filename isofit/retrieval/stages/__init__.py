from .background import background
from .cleanup import cleanup
from .configure import configure
from .interpolate import interpolate
from .invert import invert
from .lut import lut
from .preprocess import preprocess
from .presolve import presolve
from .scene import scene
from .segment import segment
from .setup import setup
from .surface import surface

# Every retrieval stage, in pipeline order
Stages = {
    "setup": setup,
    "preprocess": preprocess,
    "scene": scene,
    "surface": surface,
    "segment": segment,
    "presolve": presolve,
    "background": background,
    "configure": configure,
    "lut": lut,
    "invert": invert,
    "interpolate": interpolate,
    "cleanup": cleanup,
}

# Stages that are only valid when others are enabled, as they read their products
Requires = {
    "background": ("segment", "presolve"),
    "interpolate": ("segment",),
}


def check(config) -> None:
    """
    Check that the enabled stages form a valid retrieval.

    Raises
    ------
    ValueError
        If an enabled stage requires a stage that is disabled.
    """
    errors = [
        f"{stage}.enabled requires {need}.enabled"
        for stage, needs in Requires.items()
        if getattr(config, stage).enabled
        for need in needs
        if not getattr(config, need).enabled
    ]

    if (
        config.background.enabled
        and not config.forward_model.surface.use_background_rfl
    ):
        errors.append(
            "background.enabled requires forward_model.surface.use_background_rfl"
        )

    if (
        config.interpolate.method == "empirical"
        and len(config.interpolate.num_neighbors) > 1
    ):
        errors.append(
            "interpolate.num_neighbors may only have one value for the empirical line"
        )

    if config.lut.exit and not config.lut.enabled:
        errors.append("lut.exit requires lut.enabled")

    if config.lut.enabled and config.lut.prebuilt:
        errors.append("lut.prebuilt is a LUT that is not built, disable lut.enabled")

    if errors:
        raise ValueError("Invalid retrieval stages:\n  " + "\n  ".join(errors))
