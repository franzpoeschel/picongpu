"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre, Pawel Ordyna
License: GPLv3+
"""

from ...pypicongpu import species
from ...pypicongpu import util

from .Distribution import Distribution

import typeguard
import math


@typeguard.typechecked
class CylindricalDistribution(Distribution):
    """
    Describes a cylindrical density distribution of particles with gaussian up-ramp
    with a constant density region in between. It can have an arbitrary orientation
    and position in space.

    Will create the following profile:
      n = density if r < reduced_radius
      n is 0 or follows the exponential ramp if r > reduced_radius
      n is 0 if r > reduced_radius + prePlasmaCutoff
      the reduced_radius is equal = @f[\\sqrt{R^2 -L^2} -L @f]
      with R - cylinder_radius and L - prePlasmaLength (scale length of the ramp)
      the reduced radius ensures mass conservation
    """

    density: float
    """particle number density, [m^-3]"""

    center_position: tuple[float, float, float]
    """center of the cylinder [x, y, z], [m]"""

    radius: float
    """cylinder radius, [m]"""

    cylinder_axis: tuple[float, float, float]
    """cylinder axis [x, y, z], [unitless]"""

    exponential_pre_plasma_length: float | None
    """scale length of the exponential pre-plasma ramp, [m]"""
    exponential_pre_plasma_cutoff: float | None
    """cutoff of the exponential pre-plasma ramp, [m]"""

    # @details pydantic provides an automatically generated __init__/constructor method which allows initialization off
    #   all attributes as keyword arguments

    # @note user may add additional attributes by hand, these will be available but not type verified

    def get_as_pypicongpu(self) -> species.operation.densityprofile.Cylinder:
        util.unsupported("fill in not active", self.fill_in, True)

        profile = species.operation.densityprofile.Cylinder()

        if self.density <= 0.0:
            raise ValueError("density must be > 0")

        min_radius = (
            math.sqrt(2.0) * self.exponential_pre_plasma_length
            if self.exponential_pre_plasma_length is not None
            else 0.0
        )
        if self.radius < min_radius:
            raise ValueError(
                f"radius must be > sqrt(2)*pre_plasma_length = {min_radius}, so that the reduced radius stays non negative. In case of no preplasma radius must be >= 0.0., {self.exponential_pre_plasma_length}, {self.radius}"
            )

        # create prePlasma ramp if indicated by settings
        prePlasma: bool = (self.exponential_pre_plasma_cutoff is not None) and (
            self.exponential_pre_plasma_length is not None
        )
        explicitlyNoPrePlasma: bool = (self.exponential_pre_plasma_cutoff is None) and (
            self.exponential_pre_plasma_length is None
        )

        if prePlasma:
            profile.pre_plasma_ramp = species.operation.densityprofile.plasmaramp.Exponential(
                self.exponential_pre_plasma_length,  # type: ignore
                self.exponential_pre_plasma_cutoff,  # type: ignore
            )
        elif explicitlyNoPrePlasma:
            profile.pre_plasma_ramp = species.operation.densityprofile.plasmaramp.None_()
        else:
            raise ValueError(
                "either both exponential_pre_plasma_length and"
                " exponential_pre_plasma_cutoff must be set to"
                " none or neither!"
            )

        # @todo change to constructor call once we switched PyPIConGPU to use pydantic, Brian Marre, 2024
        profile.density_si = self.density
        profile.center_position_si = self.center_position
        profile.radius_si = self.radius
        profile.cylinder_axis = self.cylinder_axis

        return profile
