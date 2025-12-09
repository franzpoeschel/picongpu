"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from picongpu.picmi.species import OperationalRequirement, ConstantConstructingRequirement
from picongpu.pypicongpu.species.operation.setchargestate import SetChargeState
from picongpu.pypicongpu.species.constant.groundstateionization import GroundStateIonization
from .... import pypicongpu

from pydantic import BaseModel
import typeguard
import typing


@typeguard.typechecked
class IonizationModel(BaseModel):
    """
    common interface for all ionization models

    @note further configurations may be added by implementations
    """

    MODEL_NAME: str
    """ionization model"""

    ion_species: typing.Any
    """PICMI ion species to apply ionization model for"""

    ionization_electron_species: typing.Any
    """PICMI electron species of which to create macro particle upon ionization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ion_species.register_requirements(
            [
                ConstantConstructingRequirement(
                    species=self.ionization_electron_species,
                    return_type=GroundStateIonization,
                    constructor=lambda species, **kwargs: GroundStateIonization(
                        ionization_model_list=[m.get_as_pypicongpu() for m in kwargs["model"]]
                    ),
                    kwargs={"model": [self]},
                    merge_functor=lambda self, other: isinstance(other, ConstantConstructingRequirement)
                    and hasattr(other, "return_type")
                    and (self.kwargs["model"].extend(other.kwargs["model"]) or True),
                ),
                OperationalRequirement(
                    function=SetChargeState, kwargs=dict(charge_state=self.ion_species.charge_state)
                ),
            ]
        )

    def __hash__(self):
        """custom hash function for indexing in dicts"""
        hash_value = hash(type(self))

        for value in self.__dict__.values():
            try:
                if value is not None:
                    hash_value += hash(value)
            except TypeError:
                print(self)
                print(type(self))
                raise TypeError
        return hash_value

    def check(self):
        # import here to avoid circular import that stems from projecting different species types from PIConGPU onto the same `Species` type in PICMI
        from ... import NEW1_Species as Species

        assert isinstance(self.ion_species, Species), "ion_species must be an instance of the species object"
        assert isinstance(self.ionization_electron_species, Species), (
            "ionization_electron_species must be an instance of the species object"
        )

    def get_constants(self) -> list[pypicongpu.species.constant.Constant]:
        raise NotImplementedError("abstract base class only!")

    def get_as_pypicongpu(self) -> pypicongpu.species.constant.ionizationmodel.IonizationModel:
        raise NotImplementedError("abstract base class only!")
