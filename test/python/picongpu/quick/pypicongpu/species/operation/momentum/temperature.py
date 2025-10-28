"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from pydantic import ValidationError
from picongpu.pypicongpu.species.operation.momentum import Temperature

import unittest


class TestTemperature(unittest.TestCase):
    def test_basic(self):
        """expected functions return something (valid)"""
        t = Temperature(temperature_kev=17)

        context = t.get_rendering_context()
        self.assertEqual(17, context["temperature_kev"])

    def test_invalid_values(self):
        """temperature must be >=0"""
        for invalid in [-1, -47.1, -0.0000001]:
            with self.assertRaises(ValidationError):
                Temperature(temperature_kev=invalid)

    def test_types(self):
        """invalid types are rejected"""
        for invalid in [None, "asd", {}, []]:
            with self.assertRaises(ValidationError):
                Temperature(temperature_kev=invalid)
