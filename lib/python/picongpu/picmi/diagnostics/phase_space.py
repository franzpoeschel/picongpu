"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.copy_attributes import converts_to
from ...pypicongpu.output.phase_space import PhaseSpace as PyPIConGPUPhaseSpace

from ..species import Species as PICMISpecies
from .timestepspec import TimeStepSpec

import typeguard
from typing import Literal


@converts_to(
    PyPIConGPUPhaseSpace,
    conversions={
        "species": lambda self, d, *args: d.get(self.species),
        "period": lambda self, _, *args: self.period.get_as_pypicongpu(*args),
    },
    preamble=lambda self, d, *args: self.check(d),
    ignore=["check"],
)
@typeguard.typechecked
class PhaseSpace:
    """
    Specifies the parameters for the output of Phase Space of species such as electrons.

    This plugin extracts phase-space data from the simulation, allowing
    for detailed analysis of particle distributions in position-momentum space.

    Parameters
    ----------
    species: string
        Name of the particle species to track (e.g., "electron", "proton").

    period: TimeStepSpec
        Specify on which time steps the plugin should run.
        Unit: steps (simulation time steps).

    spatial_coordinate: string
        Spatial coordinate used in phase space (e.g., 'x', 'y', 'z').

    momentum: string
        Momentum coordinate used in phase space (e.g., 'px', 'py', 'pz').

    min_momentum: float
        Minimum value for the phase-space coordinate range.
        Unit: kg*m/s (momentum in SI units).

    max_momentum: float
        Maximum value for the phase-space coordinate range.
        Unit: kg*m/s (momentum in SI units).

    name: string, optional
        Optional name for the phase-space plugin.
    """

    def check(self, dict_species_picmi_to_pypicongpu):
        if self.min_momentum >= self.max_momentum:
            raise ValueError("min_momentum must be less than max_momentum")

        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")

        # checks if PICMISpecies instance exists in the dictionary. If yes, it returns the corresponding PyPIConGPUSpecies instance.
        # self.species refers to the species attribute of the class  PhaseSpace(picmistandard.PICMI_PhaseSpace).
        if dict_species_picmi_to_pypicongpu.get(self.species) is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")

    def __init__(
        self,
        species: PICMISpecies,
        period: TimeStepSpec,
        spatial_coordinate: Literal["x", "y", "z"],
        momentum_coordinate: Literal["px", "py", "pz"],
        min_momentum: float,
        max_momentum: float,
    ):
        self.species = species
        self.period = period
        self.spatial_coordinate = spatial_coordinate
        self.momentum_coordinate = momentum_coordinate
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
