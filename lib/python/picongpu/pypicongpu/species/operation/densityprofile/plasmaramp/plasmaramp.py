"""
This file is part of PIConGPU.
Copyright 2023-2025 PIConGPU contributors
Authors: Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel
from .....rendering import SelfRegisteringRenderedObject


class PlasmaRamp(SelfRegisteringRenderedObject, BaseModel):
    """
    abstract parent class for all plasma ramps

    A plasma ramp describes ramp up of an edge of an initial density
    distribution
    """

    pass
