"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .operation import Operation
from .momentum import Temperature, Drift
from ..species import Species
from pydantic import BaseModel, PrivateAttr

import typing


class SimpleMomentum(Operation):
    """
    provides momentum to a species

    specified by:

    - temperature
    - drift

    Both are optional. If both are missing, momentum **is still provided**, but
    left at 0 (default).
    """

    species: Species
    """species for which momentum will be set"""

    temperature: typing.Optional[Temperature]
    """temperature of particles (if any)"""

    drift: typing.Optional[Drift]
    """drift of particles (if any)"""

    _name: str = PrivateAttr("simplemomentum")

    def __init__(self, *args, **kwargs):
        return BaseModel.__init__(self, *args, **kwargs)
