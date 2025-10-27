"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import json
from typing import Optional

import typeguard
from pydantic import BaseModel

from .. import util
from ..rendering.renderedobject import RenderedObject
from ..species import Species
from .particle_functor import ParticleFunctor as BinningFunctor
from .plugin import Plugin
from .timestepspec import TimeStepSpec


class BinSpec(RenderedObject, BaseModel):
    kind: str
    start: float | int
    stop: float | int
    nsteps: float | int


class BinningAxis(RenderedObject):
    name: str
    bin_spec: BinSpec
    functor: BinningFunctor

    def __init__(self, name, bin_spec, functor, use_overflow_bins):
        self.name = name
        self.bin_spec = bin_spec
        self.functor = functor
        self.use_overflow_bins = use_overflow_bins

    def _get_serialized(self):
        return {
            "axis_name": self.name,
            "bin_spec": self.bin_spec.get_rendering_context(),
            "axis_functor": self.functor.get_rendering_context(),
            "use_overflow_bins": self.use_overflow_bins,
        }


@typeguard.typechecked
class Binning(Plugin):
    name = util.build_typesafe_property(str)
    species = util.build_typesafe_property(list[Species])
    period = util.build_typesafe_property(TimeStepSpec)
    openPMD = util.build_typesafe_property(Optional[dict])
    openPMDExt = util.build_typesafe_property(Optional[str])
    openPMDInfix = util.build_typesafe_property(Optional[str])
    dumpPeriod = util.build_typesafe_property(int)

    _name = "binning"

    def __init__(
        self,
        name,
        deposition_functor,
        axes,
        species,
        period,
        openPMD,
        openPMDExt,
        openPMDInfix,
        dumpPeriod,
    ):
        self.name = name
        self.deposition_functor = deposition_functor
        self.axes = axes
        self.species = species
        self.period = period
        self.openPMD = openPMD
        self.openPMDExt = openPMDExt
        self.openPMDInfix = openPMDInfix
        self.dumpPeriod = dumpPeriod

    def _get_serialized(self) -> dict:
        return {
            "binner_name": self.name,
            "deposition_functor": self.deposition_functor.get_rendering_context(),
            "axes": list(map(BinningAxis.get_rendering_context, self.axes)),
            "species": list(map(Species.get_rendering_context, self.species)),
            "period": self.period.get_rendering_context(),
            "openPMD": json.dumps(self.openPMD) if self.openPMD else self.openPMD,
            "openPMDExtension": self.openPMDExt,
            "openPMDInfix": self.openPMDInfix,
            "dumpPeriod": self.dumpPeriod,
        }
