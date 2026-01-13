"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, PrivateAttr

from ..species import Species
from .plugin import Plugin
from .timestepspec import TimeStepSpec


class EnergyHistogram(Plugin, BaseModel):
    species: Species
    period: TimeStepSpec
    bin_count: int
    min_energy: float
    max_energy: float

    _name: str = PrivateAttr("energyhistogram")
