"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .. import util
from ..species import Species

from .plugin import Plugin

import typeguard
import typing
from typing import Literal


@typeguard.typechecked
class PhaseSpace(Plugin):
    species = util.build_typesafe_property(Species)
    period = util.build_typesafe_property(int)
    spatial_coordinate = util.build_typesafe_property(Literal["x", "y", "z"])
    momentum_coordinate = util.build_typesafe_property(Literal["px", "py", "pz"])
    min_momentum = util.build_typesafe_property(float)
    max_momentum = util.build_typesafe_property(float)

    def __init__(self):
        "do nothing"

    def _get_serialized(self) -> typing.Dict:
        """Return the serialized representation of the object."""
        return {
            "species": self.species.get_rendering_context(),
            "period": self.period,
            "spatial_coordinate": self.spatial_coordinate,
            "momentum_coordinate": self.momentum_coordinate,
            "min_momentum": self.min_momentum,
            "max_momentum": self.max_momentum,
        }
