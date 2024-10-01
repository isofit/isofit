"""
This is a template radiative transfer engine that can be copied and updated for a new engine.
It contains all of the pieces required to create a custom engine.

Be sure to edit engines/__init__.py to import your custom engine and set its name to be referenced via the config:
    from .minimal import CustomRT
    Engines['CustomRT'] = CustomRT
"""

from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine


class CustomRT(RadiativeTransferEngine):
    """
    This is the minimal version that must be defined. Refer to the full template for
    detailed descriptions.
    """

    def preSim(self):
        pass

    def makeSim(self, point, **kwargs):
        pass

    def readSim(self, point):
        pass

    def postSim(self):
        pass
