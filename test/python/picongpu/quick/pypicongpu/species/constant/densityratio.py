"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import unittest

from pydantic import ValidationError
from picongpu.pypicongpu.species.constant import DensityRatio


class TestDensityRatio(unittest.TestCase):
    def test_basic(self):
        """simple example"""
        dr = DensityRatio(ratio=1.0)
        self.assertEqual([], dr.get_species_dependencies())
        self.assertEqual([], dr.get_attribute_dependencies())
        self.assertEqual([], dr.get_constant_dependencies())

    def test_types(self):
        """type safety ensured"""
        for invalid in [None, "asbd", [], {}]:
            with self.assertRaises(ValidationError):
                DensityRatio(ratio=invalid)

        for valid_type in [1, 171238]:
            DensityRatio(ratio=valid_type)

    def test_value_range(self):
        """negative values prohibited"""
        for invalid in [0, -1, -0.00000001]:
            with self.assertRaises(ValidationError):
                DensityRatio(ratio=invalid)

        for valid in [0.000001, 2, 3.5]:
            DensityRatio(ratio=valid)

    def test_rendering_passthru(self):
        """context passes ratio through"""
        dr = DensityRatio(ratio=13.37)
        context = dr.get_rendering_context()
        self.assertAlmostEqual(13.37, context["ratio"])
