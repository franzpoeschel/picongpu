"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ..constant import Constant
from pydantic import model_validator
from ..ionizationcurrent import IonizationCurrent

import typing


class IonizationModel(Constant):
    """
    base class for an ground state only ionization models of an ion species

    Owned by exactly one species.

    Identified by its PIConGPU name.

    PIConGPU term: "ionizer"
    """

    picongpu_name: str
    """C++ Code type name of ionizer"""

    # no typecheck here -- would require circular imports
    ionization_electron_species: typing.Any
    """species to be used as electrons"""

    ionization_current: typing.Optional[IonizationCurrent] = None
    """ionization current implementation to use"""

    @model_validator(mode="after")
    def check(self):
        """check internal consistency"""

        # import here to avoid circular import
        from ...species import Species
        from ..groundstateionization import GroundStateIonization

        # check ionization electron species is actually pypicongpu species instance
        if not isinstance(self.ionization_electron_species, Species):
            raise TypeError("ionization_electron_species must be of type pypicongpu Species")

        # electron species must not be an ionizable
        if self.ionization_electron_species.has_constant_of_type(GroundStateIonization):
            raise ValueError(
                "used electron species {} must not be ionizable itself".format(self.ionization_electron_species.name)
            )
        return self
