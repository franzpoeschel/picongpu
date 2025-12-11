"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Masoud Afshari
License: GPLv3+
"""

from .operation import Operation
from ..species import Species
from ... import util

import typeguard


@typeguard.typechecked
class SetChargeState(Operation):
    """
    assigns boundElectrons attribute and sets it to the initial charge state

    used for ionization of ions
    """

    species = util.build_typesafe_property(Species)
    """species which will have boundElectrons set"""

    charge_state = util.build_typesafe_property(int)
    """initial ion charge state"""

    _name = "setchargestate"

    def __init__(self, species, charge_state):
        self.species = species
        self.charge_state = charge_state

    def check_preconditions(self) -> None:
        if self.charge_state < 0:
            raise ValueError("charge state must be > 0")

    def _get_serialized(self) -> dict:
        self.check_preconditions()
        return {
            "species": self.species.get_rendering_context(),
            "charge_state": self.charge_state,
        }
