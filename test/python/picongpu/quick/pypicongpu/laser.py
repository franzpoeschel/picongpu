"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus
License: GPLv3+
"""

import copy
import unittest

from picongpu.pypicongpu.laser import GaussianLaser, PolarizationType
from pydantic import ValidationError

""" @file we only test for types here, test for values errors is done in the
   custom picmi-objects"""

KWARGS = dict(
    wavelength=1.2,
    waist=3.4,
    duration=5.6,
    focal_position=[0, 7.8, 0],
    phi0=2.9,
    E0=9.0,
    pulse_init=1.3,
    propagation_direction=[0.0, 1.0, 0.0],
    polarization_type=PolarizationType.LINEAR,
    polarization_direction=[0.0, 1.0, 0.0],
    laguerre_modes=[1.0],
    laguerre_phases=[0.0],
    huygens_surface_positions=[[1, -1], [1, -1], [1, -1]],
)


class TestGaussianLaser(unittest.TestCase):
    def test_types(self):
        """invalid types are rejected"""
        for not_float in [None, [], {}]:
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(wavelength=not_float))
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(waist=not_float))
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(duration=not_float))
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(phi0=not_float))
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(E0=not_float))
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(pulse_init=not_float))

        for not_position_vector in [1, 1.0, None, ["string"]]:
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(focal_position=not_position_vector))

        for not_polarization_type in [1.3, None, "", []]:
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(polarization_type=not_polarization_type))

        for not_direction_vector in [1, 1.3, None, "", ["string"]]:
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(polarization_direction=not_direction_vector))
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(propagation_direction=not_direction_vector))

        for invalid_list in [None, 1.2, "1.2", ["string"]]:
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(laguerre_modes=invalid_list))
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(laguerre_phases=invalid_list))
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(polarization_direction=invalid_list))
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(propagation_direction=invalid_list))
            with self.assertRaises(ValidationError):
                GaussianLaser(**KWARGS | dict(huygens_surface_positions=invalid_list))

    def test_polarization_type(self):
        """polarization type enum sanity checks"""
        lin = PolarizationType.LINEAR
        circular = PolarizationType.CIRCULAR

        self.assertNotEqual(lin, circular)

        self.assertNotEqual(lin.get_cpp_str(), circular.get_cpp_str())

        for polarization_type in [lin, circular]:
            self.assertEqual(str, type(polarization_type.get_cpp_str()))

    def test_invalid_huygens_surface_description_types(self):
        """Huygens surfaces must be described as
        [[x_min:int, x_max:int], [y_min:int,y_max:int],
        [z_min:int, z_max:int]]"""
        invalid_elements = [None, [], [1.2, 3.4]]
        valid_rump = [[5, 6], [7, 8]]

        invalid_descriptions = []
        for invalid_element in invalid_elements:
            for pos in range(3):
                base = copy.deepcopy(valid_rump)
                base.insert(pos, invalid_element)
                invalid_descriptions.append(base)

        for invalid_description in invalid_descriptions:
            with self.assertRaises(TypeError):
                GaussianLaser(**KWARGS).huygens_surface_positions(invalid_description)

    def test_invalid_laguerre_modes_empty(self):
        """laguerre modes must be set non-empty"""
        with self.assertRaises(ValidationError):
            GaussianLaser(**KWARGS | dict(laguerre_modes=[]))
        with self.assertRaises(ValidationError):
            GaussianLaser(**KWARGS | dict(laguerre_modes=[1.0], laguerre_phases=[]))

    def test_invalid_laguerre_modes_invalid_length(self):
        """num of laguerre modes/phases must be equal"""
        with self.assertRaises(ValidationError):
            GaussianLaser(**KWARGS | dict(laguerre_modes=[1.0], laguerre_phases=[2, 3]))

    def test_positive_definite_laguerre_modes(self):
        """test whether laguerre modes are positive definite"""
        with self.assertLogs(level="WARNING") as caught_logs:
            GaussianLaser(**KWARGS | dict(laguerre_modes=[-1.0]))
        self.assertEqual(1, len(caught_logs.output))

    def test_translation(self):
        """is translated to context object"""
        # note: implicitly checks against schema
        laser = GaussianLaser(**KWARGS)
        context = laser.get_rendering_context()["data"]
        self.assertEqual(context["wave_length_si"], laser.wave_length_si)
        self.assertEqual(context["waist_si"], laser.waist_si)
        self.assertEqual(context["pulse_duration_si"], laser.pulse_duration_si)
        self.assertEqual(
            context["focus_pos_si"],
            [
                {"component": laser.focus_pos_si[0]},
                {"component": laser.focus_pos_si[1]},
                {"component": laser.focus_pos_si[2]},
            ],
        )
        self.assertEqual(context["phase"], laser.phase)
        self.assertEqual(context["E0_si"], laser.E0_si)
        self.assertEqual(context["pulse_init"], laser.pulse_init)
        self.assertEqual(
            context["propagation_direction"],
            [
                {"component": laser.propagation_direction[0]},
                {"component": laser.propagation_direction[1]},
                {"component": laser.propagation_direction[2]},
            ],
        )
        self.assertEqual(context["polarization_type"], laser.polarization_type.get_cpp_str())
        self.assertEqual(
            context["polarization_direction"],
            [
                {"component": laser.polarization_direction[0]},
                {"component": laser.polarization_direction[1]},
                {"component": laser.polarization_direction[2]},
            ],
        )
        self.assertEqual(context["laguerre_modes"], [{"single_laguerre_mode": 1.0}])
        self.assertEqual(context["laguerre_phases"], [{"single_laguerre_phase": 0.0}])
        self.assertEqual(context["modenumber"], 0)
        self.assertEqual(
            context["huygens_surface_positions"],
            {
                "row_x": {
                    "negative": laser.huygens_surface_positions[0][0],
                    "positive": laser.huygens_surface_positions[0][1],
                },
                "row_y": {
                    "negative": laser.huygens_surface_positions[1][0],
                    "positive": laser.huygens_surface_positions[1][1],
                },
                "row_z": {
                    "negative": laser.huygens_surface_positions[2][0],
                    "positive": laser.huygens_surface_positions[2][1],
                },
            },
        )
