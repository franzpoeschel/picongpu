"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import unittest

from picongpu.pypicongpu.species.operation.densityprofile import Uniform
from pydantic import ValidationError


class TestUniform(unittest.TestCase):
    def test_typesafety(self):
        """typesafety is ensured"""
        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                Uniform(density_si=invalid)

    def test_check(self):
        """validity check on self"""
        for invalid in [-1, 0, -0.00000003]:
            # assignment passes, but check catches the error
            with self.assertRaises(ValidationError):
                Uniform(density_si=invalid)

    def test_rendering(self):
        """value passed through from rendering"""
        u = Uniform(density_si=42.17)

        context = u.get_rendering_context()
        self.assertTrue(context["typeID"]["uniform"])
        context = context["data"]
        self.assertAlmostEqual(u.density_si, context["density_si"])
