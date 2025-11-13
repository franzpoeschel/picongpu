"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import unittest

from picongpu.pypicongpu.species.constant import Charge
from pydantic import ValidationError


class TestCharge(unittest.TestCase):
    def test_basic(self):
        c = Charge(charge_si=0)
        self.assertEqual([], c.get_species_dependencies())
        self.assertEqual([], c.get_attribute_dependencies())
        self.assertEqual([], c.get_constant_dependencies())

    def test_types(self):
        """types are checked"""
        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                Charge(charge_si=invalid)

    def test_rendering(self):
        """rendering passes information through"""
        c = Charge(charge_si=-3.2)

        context = c.get_rendering_context()
        self.assertAlmostEqual(-3.2, context["charge_si"])
