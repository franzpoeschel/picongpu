"""
This file is part of PIConGPU.
Copyright 2024-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

import unittest

from picongpu.pypicongpu.species.operation.densityprofile import Foil
from picongpu.pypicongpu.species.operation.densityprofile.plasmaramp import None_
from pydantic import ValidationError

KWARGS = dict(
    density_si=10e-27,
    y_value_front_foil_si=0.0,
    thickness_foil_si=1.0e-5,
    pre_foil_plasmaRamp=None_(),
    post_foil_plasmaRamp=None_(),
)


class TestFoil(unittest.TestCase):
    def test_value_pass_through(self):
        """values are passed through"""
        f = Foil(**KWARGS)
        for key, val in KWARGS.items():
            self.assertAlmostEqual(val, getattr(f, key))

    def test_typesafety(self):
        """typesafety is ensured"""
        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                Foil(**KWARGS | dict(density_si=invalid))

        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                Foil(**KWARGS | dict(y_value_front_foil_si=invalid))

        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                Foil(**KWARGS | dict(thickness_foil_si=invalid))

        for invalid in [None, []]:
            with self.assertRaises(ValidationError):
                Foil(**KWARGS | dict(pre_foil_plasmaRamp=invalid))

        for invalid in [None, []]:
            with self.assertRaises(ValidationError):
                Foil(**KWARGS | dict(post_foil_plasmaRamp=invalid))

    def test_check_density(self):
        """validity check on self for invalid density"""
        for invalid in [-1, 0, -0.00000003]:
            with self.assertRaises(ValidationError):
                Foil(**KWARGS | dict(density_si=invalid))

    def test_check_y_value_front_foil(self):
        """validity check on self for invalid y_value_front_foil_si"""
        for invalid in [-1, -0.00000003]:
            with self.assertRaises(ValidationError):
                Foil(**KWARGS | dict(y_value_front_foil_si=invalid))

    def test_check_thickness(self):
        """validity check on self for invalid y_value_front_foil_si"""
        for invalid in [-1, -0.00000003]:
            with self.assertRaises(ValidationError):
                Foil(**KWARGS | dict(thickness_foil_si=invalid))

    def test_rendering(self):
        """value passed through from rendering"""
        f = Foil(**KWARGS)
        expectedContextNoRamp = {
            "typeID": {"exponential": False, "none": True},
            "data": None,
        }

        context = f.get_rendering_context()
        self.assertTrue(context["typeID"]["foil"])
        context = context["data"]
        self.assertAlmostEqual(f.density_si, context["density_si"])
        self.assertAlmostEqual(f.y_value_front_foil_si, context["y_value_front_foil_si"])
        self.assertAlmostEqual(f.thickness_foil_si, context["thickness_foil_si"])
        self.assertEqual(expectedContextNoRamp, context["pre_foil_plasmaRamp"])
        self.assertEqual(expectedContextNoRamp, context["post_foil_plasmaRamp"])
