"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Masoud Afshari
License: GPLv3+
"""

from pydantic import Field, PrivateAttr, BaseModel

from ..species import Species
from .operation import Operation


class SetChargeState(Operation):
    """
    assigns boundElectrons attribute and sets it to the initial charge state

    used for ionization of ions
    """

    species: Species
    """species which will have boundElectrons set"""

    charge_state: int = Field(ge=0)
    """initial ion charge state"""

    _name: str = PrivateAttr("setchargestate")

    def __init__(self, *args, **kwargs):
        return BaseModel.__init__(self, *args, **kwargs)
