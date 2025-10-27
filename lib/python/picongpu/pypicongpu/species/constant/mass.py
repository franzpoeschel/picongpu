"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from pydantic import BaseModel, Field
from .constant import Constant


class Mass(Constant, BaseModel):
    """
    mass of a physical particle
    """

    mass_si: float = Field(gt=0.0)
    """mass in kg of an individual particle"""
