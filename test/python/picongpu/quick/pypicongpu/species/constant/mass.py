"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from pydantic import ValidationError
from picongpu.pypicongpu.species.constant import Mass

import unittest


class TestMass(unittest.TestCase):
    def test_basic(self):
        m = Mass(mass_si=17)
        # passes
        self.assertEqual([], m.get_species_dependencies())
        self.assertEqual([], m.get_attribute_dependencies())
        self.assertEqual([], m.get_constant_dependencies())

    def test_type(self):
        """types are checked"""
        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                Mass(mass_si=invalid)

    def test_values(self):
        """invalid values are rejected"""
        for invalid in [-1, 0, -0.0000001]:
            with self.assertRaises(ValidationError):
                Mass(mass_si=invalid)

    def test_rendering(self):
        """passes value through"""
        m = Mass(mass_si=1337)

        context = m.get_rendering_context()
        self.assertEqual(1337, context["mass_si"])
