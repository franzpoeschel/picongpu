from . import ionization
from .synchrotron import Synchrotron
from .collision import Collision, CollisionalPhysicsSetup, ConstLogCollision, DynamicLogCollision

Interaction = ionization.IonizationModel | Synchrotron | Collision | CollisionalPhysicsSetup

__all__ = [
    "Interaction",
    "ionization",
    "Synchrotron",
    "Collision",
    "ConstLogCollision",
    "DynamicLogCollision",
    "CollisionalPhysicsSetup",
]
