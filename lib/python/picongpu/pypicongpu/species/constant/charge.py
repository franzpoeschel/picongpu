"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import typeguard
from pydantic import BaseModel

from .constant import Constant


@typeguard.typechecked
class Charge(Constant, BaseModel):
    """
    charge of a physical particle
    """

    charge_si: float
    """charge in C of an individual particle"""
