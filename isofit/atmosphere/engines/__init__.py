# NOTE juliacall must be called before pytorch: https://github.com/pytorch/pytorch/issues/78829
try:
    from .vsmartmom import VSmartMomRT
except ImportError as e:
    _vsmartmom_error = e

    def VSmartMomRT(*args, **kwargs):
        raise ImportError(
            "The vSmartMOM engine requires optional dependencies, install with isofit download to proceed."
        ) from _vsmartmom_error


from .kernel_flows import KernelFlowsRT
from .libradtran import LibRadTranRT
from .modtran import ModtranRT
from .six_s import SixSRT
from .sRTMnet import SimulatedModtranRT

Engines = {
    "kernelflows": KernelFlowsRT,
    "modtran": ModtranRT,
    "sixs": SixSRT,
    "srtmnet": SimulatedModtranRT,
    "libradtran": LibRadTranRT,
    "vsmartmom": VSmartMomRT,
    "prebuilt": None,
}
