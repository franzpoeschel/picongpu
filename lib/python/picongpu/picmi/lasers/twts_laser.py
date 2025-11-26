"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Alexander Debus
License: GPLv3+
"""

import math
import typing

import typeguard

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser
from .polarization_type import PolarizationType

from .. import constants


@default_converts_to(laser.TWTSLaser)
@typeguard.typechecked
class TWTSLaser(BaseLaser):
    """
    Specifies a TWTS laser

    Parameters
    ----------
    - wavelength: float
        Central wavelength of the laser [m].

    - waist : float
        Spot size (1/e^2 radius) of the laser at focus [m].

    - duration: float
        Duration of the TWTS pulse [s], defined as 1 sigma of std. Gauss for intensity.

    - phi: float
        Laser incident angle [rad]
        denoting the mean laser phase propagation direction
        with respect to the y-axis.

    - polarizationAngle : float
        Linear laser polarization direction
        parameterized as a rotation angle [rad]
        of the x-direction around the mean
        laser phase propagation direction.

    - focal_position : list[float]
        3D coordinates of the laser focus [m]
        in the simulation domain.

    - centroid_position : list[float]
        3D coordinates of the inital laser centroid [m].

    - focus_z_offset_si : float
        Offset from the middle of the simulation domain
        to the laser focus in z-direction [m].

    - a0 : float, optional
        Normalized vector potential at focus [dimensionless].
        Specify either a0 or E0 (E0 takes precedence).

    - E0 : float, optional
        Peak electric field amplitude [V/m].
        Specify either a0 or E0 (E0 takes precedence).

    - beta0 : float, optional [default = 1.0]
        Laser centroid speed in y-direction normalized
        to the vacuum speed of light [dimensionless].
    """

    def __init__(
        self,
        wavelength,
        waist,
        duration,
        phi,
        polarizationAngle,
        focal_position,
        centroid_position,
        focus_z_offset_si,
        a0=None,
        E0=None,
        beta0=1.0,
        # make sure to always place Huygens-surface inside PML-boundaries,
        # default is valid for standard PMLs
        # @todo create check for insufficient dimension
        # @todo create check in simulation for conflict between PMLs and
        # Huygens-surfaces
        picongpu_huygens_surface_positions: typing.List[typing.List[int]] = [
            [16, -16],
            [16, -16],
            [16, -16],
        ],
    ):
        if wavelength <= 0:
            raise ValueError(f"wavelength must be > 0. You gave {wavelength=}.")
        if duration <= 0:
            raise ValueError(f"laser pulse duration must be > 0. You gave {duration=}.")
        if beta0 <= 0:
            raise ValueError(f"beta0 must be >0. You gave {beta0=}.")

        self.wavelength = wavelength
        self.k0 = 2.0 * math.pi / wavelength
        self.duration = duration
        self.propagation_direction = [0.0, math.cos(phi), math.sin(phi)]
        self.focal_position = focal_position
        self.centroid_position = centroid_position
        self.polarization_direction = [
            math.cos(polarizationAngle),
            -math.sin(polarizationAngle) * math.sin(phi),
            math.cos(polarizationAngle) * math.cos(phi),
        ]
        self.a0, self.E0 = self._compute_E0_and_a0(self.k0, E0, a0)
        self.phi = phi
        if phi > 0:
            self.phiPos = True
        else:
            self.phiPos = False
        self.beta0 = beta0
        self.phi0 = 0.0
        self.waist = waist
        self.polarizationAngle = polarizationAngle
        self.time_offset_si = (focal_position[1] - centroid_position[1]) / (beta0 * constants.c)
        self.focus_z_offset_si = focus_z_offset_si
        self.picongpu_polarization_type = PolarizationType.LINEAR
        self.picongpu_huygens_surface_positions = picongpu_huygens_surface_positions
        # self._validate_common_properties()
        self.pulse_init = self._compute_pulse_init()

    """PICMI object for TWTS Laser"""


#   def check(self):
#       self._validate_common_properties()
