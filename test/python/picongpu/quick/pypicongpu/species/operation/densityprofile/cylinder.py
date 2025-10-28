"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import unittest
from pydantic import ValidationError

from picongpu.pypicongpu.species.operation.densityprofile.cylinder import Cylinder
from picongpu.pypicongpu.species.operation.densityprofile.plasmaramp import None_

KWARGS = dict(
    density_si=1.0e20,
    center_position_si=(0.0, 0.0, 0.0),
    radius_si=1.0e-3,
    cylinder_axis=(0.0, 1.0, 0.0),
    pre_plasma_ramp=None_(),
)


class TestCylinder(unittest.TestCase):
    def test_value_pass_through(self):
        """values are passed through correctly"""
        c = Cylinder(
            density_si=2.5e23,
            center_position_si=(1.0, 2.0, 3.0),
            radius_si=5.0e-4,
            cylinder_axis=(1.0, 0.0, 0.0),
            pre_plasma_ramp=None_(),
        )

        self.assertAlmostEqual(2.5e23, c.density_si)
        self.assertEqual((1.0, 2.0, 3.0), c.center_position_si)
        self.assertAlmostEqual(5.0e-4, c.radius_si)
        self.assertEqual((1.0, 0.0, 0.0), c.cylinder_axis)
        self.assertEqual(None_(), c.pre_plasma_ramp)

    def test_typesafety(self):
        """typesafety is ensured"""
        for invalid in [None, "str", [], {}]:
            with self.assertRaises(ValidationError):
                Cylinder(**KWARGS | dict(density_si=invalid))

        for invalid in [None, "str", [], (1.0), (1.0, 2.0)]:
            with self.assertRaises(ValidationError):
                Cylinder(**KWARGS | dict(center_position_si=invalid))

        for invalid in [None, "str", [], {}]:
            with self.assertRaises(ValidationError):
                Cylinder(**KWARGS | dict(radius_si=invalid))

        for invalid in [None, "str", [], (1.0, 2.0)]:
            with self.assertRaises(ValidationError):
                Cylinder(**KWARGS | dict(cylinder_axis=invalid))

        for invalid in [None, "str", [], 0]:
            with self.assertRaises(ValidationError):
                Cylinder(**KWARGS | dict(pre_plasma_ramp=invalid))

    def test_check_density(self):
        """invalid density"""
        for invalid in [0, -1, -1.0e20]:
            with self.assertRaises(ValidationError):
                Cylinder(**KWARGS | dict(density_si=invalid))

    def test_check_radius(self):
        """invalid radius"""
        for invalid in [-1e-5, -123.0]:
            with self.assertRaises(ValidationError):
                Cylinder(**KWARGS | dict(radius_si=invalid))

    def test_rendering(self):
        """check rendering context"""
        c = Cylinder(**KWARGS)
        context = c.get_rendering_context()
        # optional: verify some structure
        self.assertTrue(context.get("typeID", {}).get("cylinder", False))
        data = context["data"]
        self.assertAlmostEqual(c.density_si, data["density_si"])
        self.assertEqual([{"component": pos} for pos in c.center_position_si], data["center_position_si"])
        self.assertAlmostEqual(c.radius_si, data["radius_si"])
        self.assertEqual([{"component": ax} for ax in c.cylinder_axis], data["cylinder_axis"])

        # expected "no ramp" structure
        expectedContextNoRamp = {
            "typeID": {"exponential": False, "none": True},
            "data": None,
        }
        self.assertEqual(expectedContextNoRamp, data["pre_plasma_ramp"])
