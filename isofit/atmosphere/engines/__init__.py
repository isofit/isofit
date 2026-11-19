from .kernel_flows import KernelFlowsRT
from .libradtran import LibRadTranRT
from .modtran import ModtranRT
from .six_s import SixSRT
from .sRTMnet import SimulatedModtranRT
from .vsmartmom import VSmartMomRT

Engines = {
    "kernelflows": KernelFlowsRT,
    "modtran": ModtranRT,
    "sixs": SixSRT,
    "srtmnet": SimulatedModtranRT,
    "libradtran": LibRadTranRT,
    "vsmartmom": VSmartMomRT,
    "prebuilt": None,
}
