"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, computed_field, field_serializer

from picongpu.pypicongpu.rendering.renderedobject import SelfRegisteringRenderedObject
from picongpu.pypicongpu.species.species import Species
from picongpu.pypicongpu.util import unique


class _CollisionFunctor(SelfRegisteringRenderedObject, BaseModel):
    pass


class ConstLogCollision(_CollisionFunctor):
    _name: str = PrivateAttr("constlog")
    coulomb_log: float


class DynamicLogCollision(_CollisionFunctor):
    _name: str = PrivateAttr("dynamiclog")


CollisionFunctor = ConstLogCollision | DynamicLogCollision


class Collision(BaseModel):
    species_pairs: list[tuple[Species, Species]]
    functor: CollisionFunctor

    @computed_field
    def species(self) -> list[Species]:
        return unique(sum(self.species_pairs, tuple()))

    @field_serializer("species_pairs", mode="plain")
    def _species_pairs_serializer(self, value):
        return [
            {"species_lhs": pair[0].model_dump(mode="json"), "species_rhs": pair[1].model_dump(mode="json")}
            for pair in value
        ]

    @field_serializer("functor")
    def _serialize_functor(self, value):
        return value.get_rendering_context()


class CollisionNumericsConfig(BaseModel):
    precision: Literal[32, 64, "X"] = 64
    cell_list_chunk_size: int | None = None
    debug_screening_length: bool = False


class CollisionalPhysicsSetup(BaseModel):
    collisions: list[Collision] = Field(default_factory=list)
    screening_species: list[Species] = Field(default_factory=list)
    numerics_config: CollisionNumericsConfig = CollisionNumericsConfig()
