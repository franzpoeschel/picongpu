"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from .timestepspec import TimeStepSpec
from pydantic import BaseModel, PrivateAttr
from ..species import Species

from .plugin import Plugin


class MacroParticleCount(Plugin, BaseModel):
    species: Species
    period: TimeStepSpec
    _name: str = PrivateAttr("macroparticlecount")
