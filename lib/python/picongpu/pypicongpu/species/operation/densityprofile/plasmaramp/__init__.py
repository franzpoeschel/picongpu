from .plasmaramp import PlasmaRamp
from .exponential import Exponential
from .none import None_

AllPlasmaRamps = Exponential | None_
__all__ = ["PlasmaRamp", "Exponential", "None_", "AllPlasmaRamps"]
