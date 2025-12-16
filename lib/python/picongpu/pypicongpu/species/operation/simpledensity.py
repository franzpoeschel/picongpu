"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from pydantic import PrivateAttr, BaseModel

from ..constant import DensityRatio
from ..species import Species
from .densityoperation import DensityOperation
from .densityprofile import DensityProfile
from .layout import Layout


class SimpleDensity(DensityOperation):
    """
    Place a set of species together, using the same density profile

    These species will have **the same** macroparticle placement.

    For this operation, only the random layout is supported.

    parameters:

    - ppc: particles placed per cell
    - profile: density profile to use
    - species: species to be placed with the given profile
      note that their density ratios will be respected
    """

    profile: DensityProfile
    """density profile to use, describes the actual density"""

    species: set[Species]
    """species to be placed"""

    layout: Layout

    _name: str = PrivateAttr("simpledensity")

    def __init__(self, *args, **kwargs):
        return BaseModel.__init__(self, *args, **kwargs)

    def check_preconditions(self) -> None:
        if 0 == len(self.species):
            raise ValueError("must apply to at least one species")

        # check ratios (if present)
        for species in self.species:
            if species.has_constant_of_type(DensityRatio):
                species.get_constant_by_type(DensityRatio).check()

        if hasattr(self.profile, "check"):
            self.profile.check()

    def _get_serialized(self) -> dict:
        """
        get rendering context for C++ generation

        ppc and species are translated 1-by-1,
        species with lowest ratio is "placed initially", i.e. it is placed
        first.

        All other species are put into a separate list, which will have their
        macroparticle position copied from the first species.

        Rationale:
        PIConGPU works using this pattern, i.e. copying the macroparticle
        position from the first species.

        Also, there is a "minimum weighting" inside of PIConGPU, which prevents
        macroparticle creation if the weighting of a macroparticle is too low.
        (This has physics reasons.)

        However, this minimum weighting is ignored when copying particles.
        To ensure that the minimum weighting is kept the species with the
        **lowest density ratio** is placed first.

        I.e. **IF NO PARTICLES ARE PLACED** check **MINIMUM WEIGHTING**

        After the initial species has been placed all other species' (stored in
        a separate list without particluar order) macroparticle positions are
        copied from the first. Weightings are adjusted to the ratio between the
        first (copy source) and the respective (copy destination) species
        density ratios.

        Notably, this requires the initial placement to *adjust the first
        species' weighting according to its density weighting*.
        This is automatically performed by PIConGPU's `CreateDensity`.
        """
        self.check_preconditions()

        # sort species by ratio
        # (treat "has no ratio" as 1)
        sorted_species_by_ratio = sorted(
            self.species,
            key=lambda species: 1
            if not species.has_constant_of_type(DensityRatio)
            else species.get_constant_by_type(DensityRatio).ratio,
        )

        placed_species = []
        for species in sorted_species_by_ratio:
            placed_species.append(species.get_rendering_context())

        return {
            "layout": self.layout.get_rendering_context(),
            "profile": self.profile.get_rendering_context(),
            "placed_species_initial": placed_species[0],
            "placed_species_copied": placed_species[1:],
        }
