"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from ..pypicongpu import laser
from . import constants

import math

import typeguard
import typing


@typeguard.typechecked
class PlaneWaveLaser:
    """
    Specifies a plane wave with a temporal shape

    Parameters
    ----------
    wavelength: float
        Laser wavelength [m], defined as :math:`\\lambda_0` in the above formula

    duration: float
        Duration of the Gaussian pulse [s], defined as :math:`\\tau` in the above formula

    propagation_direction: unit vector of length 3 of floats
        Direction of propagation [1]

    polarization_direction: unit vector of length 3 of floats
        Direction of polarization [1]

    focal_position: vector of length 3 of floats
        Position of the laser focus [m]

    centroid_position: vector of length 3 of floats
        Position of the laser centroid at time 0 [m]

    a0: float
        Normalized vector potential at focus
        Specify either a0 or E0 (E0 takes precedence).

    E0: float
        Maximum amplitude of the laser field [V/m]
        Specify either a0 or E0 (E0 takes precedence).

    phi0: float
        Carrier envelope phase (CEP) [rad]
    """

    def __init__(
        self,
        wavelength,
        duration,
        propagation_direction,
        polarization_direction,
        centroid_position,
        a0=None,
        E0=None,
        phi0=None,
        picongpu_phase: float = 0.0,
        picongpu_polarization_type=(laser.PlaneWaveLaser.PolarizationType.LINEAR),
        picongpu_plateau_duration=0.0,
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
        **kw,
    ):
        assert E0 is not None or a0 is not None, "One of E0 or a0 must be speficied"

        k0 = 2.0 * math.pi / wavelength
        if E0 is None:
            E0 = a0 * constants.m_e * constants.c**2 * k0 / constants.q_e
        if a0 is None:
            a0 = E0 / (constants.m_e * constants.c**2 * k0 / constants.q_e)

        self.wavelength = wavelength
        self.k0 = k0
        self.duration = duration
        self.centroid_position = centroid_position
        self.propagation_direction = propagation_direction
        self.polarization_direction = polarization_direction
        self.a0 = a0
        self.E0 = E0
        self.phi0 = phi0

        self.picongpu_phase = picongpu_phase
        self.picongpu_plateau_duration = picongpu_plateau_duration
        self.picongpu_polarization_type = picongpu_polarization_type
        self.picongpu_huygens_surface_positions = picongpu_huygens_surface_positions

    """PICMI object for Plane Wave Laser"""

    def scalarProduct(self, a: typing.List[float], b: typing.List[float]) -> float:
        assert len(a) == len(b), "the scalar product is only defined for two \
            vector of equal dimension"

        result = 0.0
        for i in range(len(a)):
            result += a[i] * b[i]

        return result

    @staticmethod
    def testRelativeError(trueValue, testValue, relativeErrorLimit):
        return abs((testValue - trueValue) / trueValue) < relativeErrorLimit

    def get_as_pypicongpu(self) -> laser.PlaneWaveLaser:
        assert self.testRelativeError(
            1,
            self.scalarProduct(self.polarization_direction, self.polarization_direction),
            1e-9,
        ), "the polarization direction vector must be normalized"

        # check for excessive phase values to avoid numerical precision errors
        assert abs(self.picongpu_phase) <= 2 * math.pi, "abs(phase) must be < 2*pi"

        # check that initialising from y_min-plane only is sensible
        assert (
            self.scalarProduct(self.propagation_direction, [0.0, 1.0, 0.0]) > 0.0
        ), "laser propagation parallel to the y-plane or pointing outside \
            from the inside of the simulation box is not supported by this \
            laser in PIConGPU"
        assert self.testRelativeError(
            1,
            (
                self.propagation_direction[0] ** 2
                + self.propagation_direction[1] ** 2
                + self.propagation_direction[2] ** 2
            ),
            1e-9,
        ), "propagation vector must be normalized"

        # check centroid outside box
        assert self.centroid_position[1] <= 0, "the laser maximum must be \
            outside of the \
            simulation box, otherwise it is impossible to correctly initialize\
            it using a huygens surface in the box, centroid_y <= 0"
        # @todo implement check that laser field strength sufficiently small
        # at simulation box boundary

        # check polarization vector normalization

        assert self.testRelativeError(
            1,
            (
                self.propagation_direction[0] ** 2
                + self.propagation_direction[1] ** 2
                + self.propagation_direction[2] ** 2
            ),
            1e-9,
        ), "polarization vector must be normalized"

        pypicongpu_laser = laser.PlaneWaveLaser()
        pypicongpu_laser.wavelength = self.wavelength
        pypicongpu_laser.duration = self.duration
        pypicongpu_laser.focus_pos = [0.0, 0.0, 0.0]
        pypicongpu_laser.phase = self.picongpu_phase
        pypicongpu_laser.E0 = self.E0

        pypicongpu_laser.pulse_init = max(
            -2 * self.centroid_position[1] / (self.propagation_direction[1] * constants.c) / self.duration,
            15,
        )
        # unit: duration

        pypicongpu_laser.polarization_type = self.picongpu_polarization_type
        pypicongpu_laser.polarization_direction = self.polarization_direction
        pypicongpu_laser.laser_nofocus_constant_si = self.picongpu_plateau_duration
        pypicongpu_laser.propagation_direction = self.propagation_direction

        pypicongpu_laser.huygens_surface_positions = self.picongpu_huygens_surface_positions

        return pypicongpu_laser
