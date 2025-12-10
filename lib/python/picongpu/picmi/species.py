"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from enum import Enum
import re
from itertools import chain
from typing import Any

from pydantic import BaseModel, PrivateAttr, computed_field, model_validator

from picongpu.picmi.distribution import AnyDistribution
from picongpu.picmi.species_requirements import (
    DelayedConstruction,
    check_for_conflict,
    is_same_as,
    must_be_unique,
    run_construction,
    try_update_with,
)
from picongpu.pypicongpu.species.attribute import Momentum, Position
from picongpu.pypicongpu.species.attribute.attribute import Attribute
from picongpu.pypicongpu.species.attribute.weighting import Weighting
from picongpu.pypicongpu.species.constant.charge import Charge
from picongpu.pypicongpu.species.constant.constant import Constant
from picongpu.pypicongpu.species.constant.densityratio import DensityRatio
from picongpu.pypicongpu.species.constant.mass import Mass
from picongpu.pypicongpu.species.operation.operation import Operation
from picongpu.pypicongpu.species.species import Shape, Pusher, Species as PyPIConGPUSpecies

from .. import pypicongpu
from ..pypicongpu.species.util.element import Element
from .predefinedparticletypeproperties import PredefinedParticleTypeProperties


class ParticleShape(Enum):
    NGP = "NGP"
    CIC = "linear"
    TSC = "quadratic"
    PQS = "cubic"
    PCS = "quartic"
    counter = "counter"


class PusherMethod(Enum):
    # supported by standard and PIConGPU
    Boris = "Boris"
    Vay = "Vay"
    HigueraCary = "Higuera-Cary"
    Free = "free"
    # not supported by standard
    ReducedLandauLifshitz = "ReducedLandauLifshitz"
    Acceleration = "Acceleration"
    Photon = "Photon"
    Probe = "Probe"
    Axel = "Axel"
    # not supported by PIConGPU
    Li = "Li"
    LLRK4 = "LLRK4"


class Species(BaseModel):
    name: str
    particle_type: str
    initial_distribution: AnyDistribution | None
    picongpu_fixed_charge: bool = False
    charge_state: int | None = None
    density_scale: float | None = None
    mass: float | None = None
    charge: float | None = None
    particle_shape: ParticleShape = ParticleShape("quadratic")
    method: PusherMethod = PusherMethod["Boris"]

    # Theoretically, Position(), Momentum() and Weighting() are also requirements imposed from the outside,
    # e.g., by the current deposition, pusher, ..., but these concepts are not separately modelled in PICMI
    # particularly not as being applied to a particular species.
    # For now, we add them to all species. Refinements might be necessary in the future.
    _requirements: list[Any] = PrivateAttr(default_factory=lambda: [Position(), Weighting(), Momentum()])

    @model_validator(mode="after")
    def check(self):
        try:
            is_element = Element.is_element(self.particle_type)
        except ValueError:
            is_element = False
        if self.particle_type is None:
            assert self.charge_state is None, (
                f"Species {self.name} specified initial charge state via charge_state without also specifying particle "
                "type, must either set particle_type explicitly or only use charge instead"
            )
            assert self.picongpu_fixed_charge is False, (
                f"Species {self.name} specified fixed charge without also specifying particle_type"
            )
        elif is_element:
            if self.charge_state is not None:
                assert Element(self.particle_type).get_atomic_number() >= self.charge_state, (
                    f"Species {self.name} intial charge state is unphysical"
                )
        else:
            assert self.charge_state is None, "charge_state may only be set for ions"
            assert self.picongpu_fixed_charge is False, (
                f"Species {self.name} configured with fixed charge state but particle_type indicates non ion"
            )
        return self

    @computed_field
    def picongpu_element(self) -> Element | None:
        try:
            return (
                pypicongpu.species.util.Element(self.particle_type) if Element.is_element(self.particle_type) else None
            )
        except ValueError:
            return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_initial_requirements()

    def _register_initial_requirements(self):
        constants = (
            ([DensityRatio(ratio=self.density_scale)] if self.density_scale is not None else [])
            + ([Mass(mass_si=self.mass)] if self.mass is not None else [])
            + ([Charge(charge_si=self.charge)] if self.charge is not None else [])
        )
        self.register_requirements(particle_type_requirements(self.particle_type) + constants)

    class Config:
        arbitrary_types_allowed = True

    def get_as_pypicongpu(self, *args, **kwargs):
        return PyPIConGPUSpecies(
            name=self.name,
            **self._evaluate_species_requirements(),
            shape=Shape[self.particle_shape.name],
            pusher=Pusher[self.method.name],
        )

    def get_all_as_pypicongpu(self, grid, layout):
        species = self.get_as_pypicongpu()
        return species, self._evaluate_operation_requirements(grid, layout, species)

    def _evaluate_species_requirements(self):
        constants, attributes = evaluate_requirements(self._requirements, [Constant, Attribute])
        constants = [run_construction(c) for c in _make_unique(constants)]
        return {"constants": constants, "attributes": attributes}

    def _evaluate_operation_requirements(self, grid, layout, species):
        return [
            run_construction(op)
            for op in resolve_operations(chain(*evaluate_requirements(self._requirements, [Operation])))
        ]

    def __gt__(self, other):
        # This defines a partial ordering on all species.
        # This is necessary to determine the definition order inside of the C++ header.
        if not isinstance(other, Species):
            raise ValueError(f"Unknown comparison between {self=} and {other=}.")
        return any(isinstance(req, DependsOn) and req.species == other for req in self._requirements)

    def register_requirements(self, requirements):
        self._requirements += requirements


def _make_unique(requirements):
    result = []
    for lhs in requirements:
        append_this = True
        for rhs in result:
            check_for_conflict(lhs, rhs)
            merge_success = try_update_with(rhs, lhs)
            if merge_success:
                append_this = False
                break
            if must_be_unique(lhs):
                append_this = append_this and not is_same_as(lhs, rhs)
        if append_this:
            result.append(lhs)
    return result


def resolve_operations(requirements):
    return _make_unique(requirements)


def evaluate_requirements(requirements, Types):
    return list(
        map(
            _make_unique,
            (
                filter(
                    lambda req: isinstance(req, Type)
                    or (isinstance(req, DelayedConstruction) and issubclass(req.metadata.Type, Type)),
                    requirements,
                )
                for Type in Types
            ),
        )
    )


def particle_type_requirements(particle_type):
    if (particle_type is None) or re.match(r"other:.*", particle_type):
        # no particle or custom particle type set
        return []
    if particle_type in (props := PredefinedParticleTypeProperties()).get_known_particle_types():
        mass, charge = props.get_mass_and_charge_of_non_element(particle_type)
    elif Element.is_element(particle_type):
        element = pypicongpu.species.util.Element(particle_type)
        mass = element.get_mass_si()
        charge = element.get_charge_si()
    else:
        # unknown particle type
        raise ValueError(f"Species has unknown particle type {particle_type}")
    return [Mass(mass_si=mass), Charge(charge_si=charge)]


class DependsOn(BaseModel):
    species: Species
