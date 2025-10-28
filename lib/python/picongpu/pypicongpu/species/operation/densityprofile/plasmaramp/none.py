"""
This file is part of PIConGPU.
Copyright 2023-2025 PIConGPU contributors
Authors: Kristin Tippey, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, PrivateAttr, model_serializer
from .plasmaramp import PlasmaRamp


class None_(PlasmaRamp, BaseModel):
    """no plasma ramp, either up or down"""

    _name: str = PrivateAttr("none")

    @model_serializer()
    def serialize(self):
        return None
