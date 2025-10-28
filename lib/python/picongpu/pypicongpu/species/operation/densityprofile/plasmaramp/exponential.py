"""
This file is part of PIConGPU.
Copyright 2023-2025 PIConGPU contributors
Authors: Kristin Tippey, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, PrivateAttr, Field
from .plasmaramp import PlasmaRamp


class Exponential(PlasmaRamp, BaseModel):
    """exponential plasma ramp, either up or down"""

    _name: str = PrivateAttr("exponential")
    PlasmaLength: float = Field(gt=0.0)
    PlasmaCutoff: float = Field(ge=0.0)
