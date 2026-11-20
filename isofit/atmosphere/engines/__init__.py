# NOTE juliacall must be called before torch: https://github.com/pytorch/pytorch/issues/78829
from .vsmartmom import VSmartMomRT

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
