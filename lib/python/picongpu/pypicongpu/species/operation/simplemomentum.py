"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .operation import Operation
from .momentum import Temperature, Drift
from ..species import Species
from ... import util

import typeguard
import typing


@typeguard.typechecked
class SimpleMomentum(Operation):
    """
    provides momentum to a species

    specified by:

    - temperature
    - drift

    Both are optional. If both are missing, momentum **is still provided**, but
    left at 0 (default).
    """

    species = util.build_typesafe_property(Species)
    """species for which momentum will be set"""

    temperature = util.build_typesafe_property(typing.Optional[Temperature])
    """temperature of particles (if any)"""

    drift = util.build_typesafe_property(typing.Optional[Drift])
    """drift of particles (if any)"""

    _name = "simplemomentum"

    def __init__(self, /, species, temperature=None, drift=None):
        self.species = species
        self.temperature = temperature
        self.drift = drift

    def _get_serialized(self) -> dict:
        context = {
            "species": self.species.get_rendering_context(),
            "temperature": None,
            "drift": None,
        }

        if self.temperature is not None:
            context["temperature"] = self.temperature.get_rendering_context()

        if self.drift is not None:
            context["drift"] = self.drift.get_rendering_context()

        return context
