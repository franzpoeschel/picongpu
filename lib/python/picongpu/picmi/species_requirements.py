"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Any, Callable

from pydantic import BaseModel, Field

from picongpu.pypicongpu.species.constant.groundstateionization import GroundStateIonization
from picongpu.pypicongpu.species.operation.setchargestate import SetChargeState
from picongpu.pypicongpu.species.operation.simpledensity import SimpleDensity


def get_as_pypicongpu(obj, *args, **kwargs):
    if hasattr(obj, "get_as_pypicongpu"):
        return obj.get_as_pypicongpu(*args, **kwargs)
    return obj


def must_be_unique(requirement):
    return hasattr(requirement, "must_be_unique") and requirement.must_be_unique


def is_same_as(lhs, rhs):
    return (hasattr(lhs, "is_same_as") and lhs.is_same_as(rhs)) or (hasattr(rhs, "is_same_as") and rhs.is_same_as(lhs))


def try_update_with(into_instance, from_instance):
    try:
        return into_instance.merge(from_instance)
    except Exception:
        return False


class _Operators(BaseModel):
    # More precisely, this returns self.metadata.Type
    # but it is kinda hard to express this in type hints
    # and I've got more urgent matters to deal with.
    constructor: Callable[[Any], Any] = lambda self: self.metadata.Type(
        *map(get_as_pypicongpu, self.metadata.args),
        **dict(map(lambda kv: (kv[0], get_as_pypicongpu(kv[1])), self.metadata.kwargs)),
    )
    try_update_with: Callable[[Any, Any], bool] = lambda self, other: False
    is_same_as: Callable[[Any, Any], bool] = lambda self, other: isinstance(other, DelayedConstruction) and (
        self.metadata == other.metadata
    )


class _Metadata(BaseModel):
    Type: type
    args: tuple[Any, ...] = tuple()
    kwargs: dict[str, Any] = Field(default_factory=dict)

    # This might be necessary to distinguish
    # processes with identical (kw)args but different operatos.
    misc: dict[str, Any] = Field(default_factory=dict)


class DelayedConstruction(BaseModel):
    """
    This class models the delayed construction of an object.

    While the user composes their simulation,
    the individual components will register requirements with our PICMI species
    in the form of what PyPIConGPU objects it needs the PyPIConGPU species to contain.
    But any individual registration cannot assume that
    the PICMI species has already obtained all knowledge
    necessary to construct its PyPIConGPU counterpart,
    so the registering object can only express its intent
    but not actually perform the construction in most cases.

    This class models such an intent.
    The metadata is supposed to contain
    a faithful and complete representation of what is constructed.
    It should be sufficient to
        - perform the construction and
        - compare intents (Is this other DelayedConstruction encoding the same action?)
    The operators are customisable actions to take under specific circumstances:
        - constructor: How to perform the construction.
        - update_with: How to merge another object into this one (returns if successful or not)
        - is_same_as: How to compare with another object
    The constructor is allowed to assume that at the time of its execution
        - all information is available and all objects can be constructed
        - the handled objects are stateless,
          i.e., the process is repeatable and the order of execution doesn't matter.
    A custom constructor should obviously follow these principles as well.
    The other operators cannot make such assumptions; they operate on mutable snapshots.

    A word of warning:
    You've got arbitrary functions for customisation at your disposal.
    Use them wisely!
    In particular, the construction should perform the obvious and minimal action
    required to fulfill the intent declared in the metadata.
    """

    metadata: _Metadata
    operators: _Operators = _Operators()
    must_be_unique: bool = False

    def run_construction(self):
        # This is a member variable not a member function,
        # so we gotta hand it the `self` argument explicitly.
        return self.operators.constructor(self)

    def try_update_with(self, other):
        return self.operators.try_update_with(self, other)

    def is_same_as(self, other):
        return self.operators.is_same_as(self, other)


class GroundStateIonizationConstruction(DelayedConstruction):
    def __init__(self, /, ionization_model):
        def constructor(self):
            return GroundStateIonization(
                ionization_model_list=[m.get_as_pypicongpu() for m in self.metadata.kwargs["ionization_model_list"]]
            )

        def try_update_with(self, other):
            return isinstance(other, GroundStateIonizationConstruction) and (
                self.metadata.kwargs["ionization_model_list"].extend(other.metadata.kwargs["ionization_model_list"])
                or True
            )

        operators = {"constructor": constructor, "try_update_with": try_update_with}
        metadata = {"Type": GroundStateIonization, "kwargs": {"ionization_model_list": [ionization_model]}}

        return super().__init__(operators=operators, metadata=metadata)


class SetChargeStateOperation(DelayedConstruction):
    def __init__(self, /, species):
        metadata = {"Type": SetChargeState, "kwargs": {"species": species, "charge_state": species.charge_state}}
        return super().__init__(metadata=metadata, must_be_unique=True)


class SimpleDensityOperation(DelayedConstruction):
    def __init__(self, /, species, grid, layout):
        def constructor(self):
            kwargs = self.metadata.kwargs
            return self.metadata.Type(
                species=[s.get_as_pypicongpu() for s in kwargs["species"]],
                profile=kwargs["profile"].get_as_pypicongpu(kwargs["grid"]),
                layout=kwargs["layout"].get_as_pypicongpu(),
            )

        def try_update_with(self, other):
            return (
                isinstance(other, SimpleDensityOperation)
                and other.metadata.kwargs["profile"] == self.metadata.kwargs["profile"]
                and other.metaddata.kwargs["layout"] == self.metadata.kwargs["layout"]
                and (self.metadata.kwargs["species"].extend(other.metadata.kwargs["species"]) or True)
            )

        metadata = {
            "Type": SimpleDensity,
            "kwargs": {
                "species": [species],
                "profile": species.initial_distribution,
                "layout": layout,
                "grid": grid,
            },
        }
        operators = {"constructor": constructor, "try_update_with": try_update_with}

        return super().__init__(metadata=metadata, operators=operators)
