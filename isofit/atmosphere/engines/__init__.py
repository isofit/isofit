from .kernel_flows import KernelFlowsRT
from .libradtran import LibRadTranRT
from .modtran import ModtranRT
from .six_s import SixSRT
from .sRTMnet import SimulatedModtranRT

# Eradiate is an optional dependency, only raise if the engine is actually requested
try:
    from .eradiate import EradiateRT
except ImportError as e:
    _eradiate_error = e

    def EradiateRT(*args, **kwargs):
        raise ImportError(
            "The eradiate engine requires the optional eradiate dependencies, install with"
            " isofit download to proceed"
        ) from _eradiate_error


Engines = {
    "kernelflows": KernelFlowsRT,
    "modtran": ModtranRT,
    "sixs": SixSRT,
    "srtmnet": SimulatedModtranRT,
    "libradtran": LibRadTranRT,
    "prebuilt": None,
    "eradiate": EradiateRT,
}
