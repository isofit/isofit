from .kernel_flows import KernelFlowsRT
from .libradtran import LibRadTranRT
from .modtran import ModtranRT
from .six_s import SixSRT
from .sRTMnet import SimulatedModtranRT
from .vlidort import VLIDORT

Engines = {
    "KernelFlowsGP": KernelFlowsRT,
    "modtran": ModtranRT,
    "6s": SixSRT,
    "sRTMnet": SimulatedModtranRT,
    "LibRadTran": LibRadTranRT,
    "VLIDORT": VLIDORT,
    "Prebuilt": None,
}
