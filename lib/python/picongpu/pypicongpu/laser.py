"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Julian Lenz
License: GPLv3+
"""

from .rendering import SelfRegisteringRenderedObject
from . import util

import enum
import typing
import typeguard
import logging


class Laser(SelfRegisteringRenderedObject):
    pass


@typeguard.typechecked
class GaussianLaser(Laser):
    """
    PIConGPU Gaussian Laser

    Holds Parameters to specify a gaussian laser
    """

    _name = "gaussian"

    class PolarizationType(enum.Enum):
        """represents a polarization of a laser (for PIConGPU)"""

        LINEAR = 1
        CIRCULAR = 2

        def get_cpp_str(self) -> str:
            """retrieve name as used in c++ param files"""
            cpp_by_ptype = {
                GaussianLaser.PolarizationType.LINEAR: "Linear",
                GaussianLaser.PolarizationType.CIRCULAR: "Circular",
            }
            return cpp_by_ptype[self]

    wavelength = util.build_typesafe_property(float)
    """wave length in m"""
    waist = util.build_typesafe_property(float)
    """beam waist in m"""
    duration = util.build_typesafe_property(float)
    """duration in s (1 sigma)"""
    focus_pos = util.build_typesafe_property(typing.List[float])
    """focus position vector in m"""
    phase = util.build_typesafe_property(float)
    """phase in rad, periodic in 2*pi"""
    E0 = util.build_typesafe_property(float)
    """E0 in V/m"""
    pulse_init = util.build_typesafe_property(float)
    """laser will be initialized pulse_init times of duration (unitless)"""
    propagation_direction = util.build_typesafe_property(typing.List[float])
    """propagation direction(normalized vector)"""
    polarization_type = util.build_typesafe_property(PolarizationType)
    """laser polarization"""
    polarization_direction = util.build_typesafe_property(typing.List[float])
    """direction of polarization(normalized vector)"""
    laguerre_modes = util.build_typesafe_property(typing.List[float])
    """array containing the magnitudes of radial Laguerre-modes"""
    laguerre_phases = util.build_typesafe_property(typing.List[float])
    """array containing the phases of radial Laguerre-modes"""
    huygens_surface_positions = util.build_typesafe_property(typing.List[typing.List[int]])
    """Position in cells of the Huygens surface relative to start/
       edge(negative numbers) of the total domain"""

    def _get_serialized(self) -> dict:
        if [] == self.laguerre_modes:
            raise ValueError("Laguerre modes MUST NOT be empty.")
        if [] == self.laguerre_phases:
            raise ValueError("Laguerre phases MUST NOT be empty.")
        if len(self.laguerre_phases) != len(self.laguerre_modes):
            raise ValueError("Laguerre modes and Laguerre phases MUST BE " "arrays of equal length.")
        if len(list(filter(lambda x: x < 0, self.laguerre_modes))) > 0:
            logging.warning("Laguerre mode magnitudes SHOULD BE positive definite.")
        return {
            "wave_length_si": self.wavelength,
            "waist_si": self.waist,
            "pulse_duration_si": self.duration,
            "focus_pos_si": list(map(lambda x: {"component": x}, self.focus_pos)),
            "phase": self.phase,
            "E0_si": self.E0,
            "pulse_init": self.pulse_init,
            "propagation_direction": list(map(lambda x: {"component": x}, self.propagation_direction)),
            "polarization_type": self.polarization_type.get_cpp_str(),
            "polarization_direction": list(map(lambda x: {"component": x}, self.polarization_direction)),
            "laguerre_modes": list(map(lambda x: {"single_laguerre_mode": x}, self.laguerre_modes)),
            "laguerre_phases": list(map(lambda x: {"single_laguerre_phase": x}, self.laguerre_phases)),
            "modenumber": len(self.laguerre_modes) - 1,
            "huygens_surface_positions": {
                "row_x": {
                    "negative": self.huygens_surface_positions[0][0],
                    "positive": self.huygens_surface_positions[0][1],
                },
                "row_y": {
                    "negative": self.huygens_surface_positions[1][0],
                    "positive": self.huygens_surface_positions[1][1],
                },
                "row_z": {
                    "negative": self.huygens_surface_positions[2][0],
                    "positive": self.huygens_surface_positions[2][1],
                },
            },
        }


@typeguard.typechecked
class PlaneWaveLaser(Laser):
    """
    PIConGPU Plane Wave Laser

    Holds Parameters to specify a plane wave laser
    """

    _name = "planewave"

    class PolarizationType(enum.Enum):
        """represents a polarization of a laser (for PIConGPU)"""

        LINEAR = 1
        CIRCULAR = 2

        def get_cpp_str(self) -> str:
            """retrieve name as used in c++ param files"""
            cpp_by_ptype = {
                PlaneWaveLaser.PolarizationType.LINEAR: "Linear",
                PlaneWaveLaser.PolarizationType.CIRCULAR: "Circular",
            }
            return cpp_by_ptype[self]

    wavelength = util.build_typesafe_property(float)
    """wave length in m"""
    duration = util.build_typesafe_property(float)
    """duration in s (1 sigma)"""
    focus_pos = util.build_typesafe_property(typing.List[float])
    """focus position vector in m"""
    phase = util.build_typesafe_property(float)
    """phase in rad, periodic in 2*pi"""
    E0 = util.build_typesafe_property(float)
    """E0 in V/m"""
    pulse_init = util.build_typesafe_property(float)
    """laser will be initialized pulse_init times of duration (unitless)"""
    propagation_direction = util.build_typesafe_property(typing.List[float])
    """propagation direction(normalized vector)"""
    polarization_type = util.build_typesafe_property(PolarizationType)
    """laser polarization"""
    polarization_direction = util.build_typesafe_property(typing.List[float])
    """direction of polarization(normalized vector)"""
    laser_nofocus_constant_si = util.build_typesafe_property(float)
    """constant for plane wave laser without focus (unitless)"""
    huygens_surface_positions = util.build_typesafe_property(typing.List[typing.List[int]])
    """Position in cells of the Huygens surface relative to start/
       edge(negative numbers) of the total domain"""

    def _get_serialized(self) -> dict:
        return {
            "wave_length_si": self.wavelength,
            "pulse_duration_si": self.duration,
            "focus_pos_si": list(map(lambda x: {"component": x}, self.focus_pos)),
            "phase": self.phase,
            "E0_si": self.E0,
            "pulse_init": self.pulse_init,
            "propagation_direction": list(map(lambda x: {"component": x}, self.propagation_direction)),
            "polarization_type": self.polarization_type.get_cpp_str(),
            "polarization_direction": list(map(lambda x: {"component": x}, self.polarization_direction)),
            "laser_nofocus_constant_si": self.laser_nofocus_constant_si,
            "huygens_surface_positions": {
                "row_x": {
                    "negative": self.huygens_surface_positions[0][0],
                    "positive": self.huygens_surface_positions[0][1],
                },
                "row_y": {
                    "negative": self.huygens_surface_positions[1][0],
                    "positive": self.huygens_surface_positions[1][1],
                },
                "row_z": {
                    "negative": self.huygens_surface_positions[2][0],
                    "positive": self.huygens_surface_positions[2][1],
                },
            },
        }


@typeguard.typechecked
class DispersivePulseLaser(Laser):
    """
    PIConGPU Dispersive Pulse Laser

    Holds Parameters to specify a dispersive Gaussian laser pulse with dispersion parameters
    """

    _name = "dispersive"

    class PolarizationType(enum.Enum):
        """represents a polarization of a laser (for PIConGPU)"""

        LINEAR = 1
        CIRCULAR = 2

        def get_cpp_str(self) -> str:
            """retrieve name as used in c++ param files"""
            cpp_by_ptype = {
                DispersivePulseLaser.PolarizationType.LINEAR: "Linear",
                DispersivePulseLaser.PolarizationType.CIRCULAR: "Circular",
            }
            return cpp_by_ptype[self]

    wavelength = util.build_typesafe_property(float)
    """wave length in m"""
    waist = util.build_typesafe_property(float)
    """beam waist in m"""
    duration = util.build_typesafe_property(float)
    """duration in s (1 sigma)"""
    focus_pos = util.build_typesafe_property(typing.List[float])
    """focus position vector in m"""
    phase = util.build_typesafe_property(float)
    """phase in rad, periodic in 2*pi"""
    E0 = util.build_typesafe_property(float)
    """E0 in V/m"""
    pulse_init = util.build_typesafe_property(float)
    """laser will be initialized pulse_init times of duration (unitless)"""
    propagation_direction = util.build_typesafe_property(typing.List[float])
    """propagation direction(normalized vector)"""
    polarization_type = util.build_typesafe_property(PolarizationType)
    """laser polarization"""
    polarization_direction = util.build_typesafe_property(typing.List[float])
    """direction of polarization(normalized vector)"""
    spectral_support = util.build_typesafe_property(float)
    """width of the spectral support for the discrete Fourier transform [none]"""
    sd_si = util.build_typesafe_property(float)
    """spatial dispersion in focus [m*s]"""
    ad_si = util.build_typesafe_property(float)
    """angular dispersion in focus [rad*s]"""
    gdd_si = util.build_typesafe_property(float)
    """group velocity dispersion in focus [s^2]"""
    tod_si = util.build_typesafe_property(float)
    """third order dispersion in focus [s^3]"""
    huygens_surface_positions = util.build_typesafe_property(typing.List[typing.List[int]])
    """Position in cells of the Huygens surface relative to start/
       edge(negative numbers) of the total domain"""

    def _get_serialized(self) -> dict:
        return {
            "wave_length_si": self.wavelength,
            "waist_si": self.waist,
            "pulse_duration_si": self.duration,
            "focus_pos_si": list(map(lambda x: {"component": x}, self.focus_pos)),
            "phase": self.phase,
            "E0_si": self.E0,
            "pulse_init": self.pulse_init,
            "propagation_direction": list(map(lambda x: {"component": x}, self.propagation_direction)),
            "polarization_type": self.polarization_type.get_cpp_str(),
            "polarization_direction": list(map(lambda x: {"component": x}, self.polarization_direction)),
            "spectral_support": self.spectral_support,
            "sd_si": self.sd_si,
            "ad_si": self.ad_si,
            "gdd_si": self.gdd_si,
            "tod_si": self.tod_si,
            "huygens_surface_positions": {
                "row_x": {
                    "negative": self.huygens_surface_positions[0][0],
                    "positive": self.huygens_surface_positions[0][1],
                },
                "row_y": {
                    "negative": self.huygens_surface_positions[1][0],
                    "positive": self.huygens_surface_positions[1][1],
                },
                "row_z": {
                    "negative": self.huygens_surface_positions[2][0],
                    "positive": self.huygens_surface_positions[2][1],
                },
            },
        }
