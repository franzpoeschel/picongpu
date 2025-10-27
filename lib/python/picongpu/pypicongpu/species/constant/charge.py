"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .constant import Constant
from ... import util
import typeguard


@typeguard.typechecked
class Charge(Constant):
    """
    charge of a physical particle
    """

    charge_si = util.build_typesafe_property(float)
    """charge in C of an individual particle"""

    def __init__(self):
        # overwrite from parent
        pass

    def _get_serialized(self) -> dict:
        # (please resist the temptation of removing the check b/c "its not
        # needed here": checks should *always* be run before serialization,
        # so make it a habit of expecting it everywhere)
        return {"charge_si": self.charge_si}
