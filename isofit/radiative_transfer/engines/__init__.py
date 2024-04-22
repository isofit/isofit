from .modtran import ModtranRT
from .six_s import SixSRT
from .sRTMnet import SimulatedModtranRT

# Match config string options to modules
Engines = {
    "modtran": ModtranRT,
    "6s": SixSRT,
    "sRTMnet": SimulatedModtranRT,
}
