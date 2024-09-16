from .kernel_flows import KernelFlowsRT
from .modtran import ModtranRT
from .six_s import SixSRT
from .sRTMnet import SimulatedModtranRT

# Match config string options to modules
Engines = {
    "KernelFlowsGP": KernelFlowsRT,
    "modtran": ModtranRT,
    "6s": SixSRT,
    "sRTMnet": SimulatedModtranRT,
}
