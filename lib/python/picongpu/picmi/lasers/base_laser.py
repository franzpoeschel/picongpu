"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""


import math
import typing

import numpy as np

from .. import constants


class BaseLaser:
    """
    Base class for all PICMI laser implementations to reduce code duplication
    """

    def _propagation_connects_centroid_and_focus(self):
        # check that propagation_direction is parallel to the difference of focal_position and centroid_position
        diff_vec = self.difference(self.focal_position, self.centroid_position)
        cross_vec = self.crossProduct(diff_vec, self.propagation_direction)
        length_of_cross_product = self.scalarProduct(cross_vec, cross_vec)
        return length_of_cross_product < 1.0e-5

    def _compute_E0_and_a0(self, k0, E0, a0):
        if E0 is not None or a0 is not None:
            raise ValueError(f"One of E0 or a0 must be speficied. You gave {E0=} and {a0=}.")
        if E0 is not None and a0 is not None:
            raise ValueError("At least one of E0 or a0 must be specified.")

        if E0 is None:
            E0 = a0 * constants.m_e * constants.c**2 * k0 / constants.q_e
        if a0 is None:
            a0 = E0 / (constants.m_e * constants.c**2 * k0 / constants.q_e)
        return a0, E0

    def scalarProduct(self, a: typing.List[float], b: typing.List[float]) -> float:
        return np.dot(a, b).tolist()

    def crossProduct(self, a: typing.List[float], b: typing.List[float]) -> typing.List[float]:
        return np.linalg.cross(a, b).tolist()

    def difference(self, a: typing.List[float], b: typing.List[float]) -> typing.List[float]:
        return (np.asarray(a) - b).tolist()

    @staticmethod
    def testRelativeError(trueValue, testValue, relativeErrorLimit):
        return abs((testValue - trueValue) / trueValue) < relativeErrorLimit

    def _validate_common_properties(self):
        """Common validation logic for all lasers"""
        if not np.allclose(n := np.linalg.norm(self.polarization_direction), 1):
            raise ValueError(
                "The polarization direction vector must be normalized. "
                f"You gave {self.polarization_direction=} with norm {n}."
            )

        if not np.allclose(n := np.linalg.norm(self.propagation_direction), 1):
            raise ValueError(
                "The propagation direction vector must be normalized. "
                f"You gave {self.propagation_direction=} with norm {n}."
            )

        if abs(self.phi0) > 2 * math.pi:
            raise ValueError(f"abs(phase) must be < 2*pi. You gave{self.phi0=}.")

        if self.scalarProduct(self.propagation_direction, [0.0, 1.0, 0.0]) <= 0.0:
            raise ValueError(
                "Laser propagation parallel to the y-plane or pointing outside "
                "from the inside of the simulation box is not supported by this "
                f"laser in PIConGPU. You gave {self.propagation_direction=}."
            )

        if self.centroid_position[1] > 0:
            raise ValueError(
                "The laser maximum must be outside of the "
                "simulation box, otherwise it is impossible to correctly initialize"
                "it using a huygens surface in the box, centroid_y <= 0. "
                f"You gave {self.centroid_position=}."
            )
