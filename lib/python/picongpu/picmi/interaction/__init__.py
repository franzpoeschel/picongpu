from . import ionization
from .synchrotron import Synchrotron

Interaction = ionization.IonizationModel | Synchrotron

__all__ = ["Interaction", "ionization", "Synchrotron"]
