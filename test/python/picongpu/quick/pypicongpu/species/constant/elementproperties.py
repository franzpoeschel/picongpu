"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import unittest

from pydantic import ValidationError
from picongpu.pypicongpu.species.constant import ElementProperties
from picongpu.pypicongpu.species.util import Element


class TestElementProperties(unittest.TestCase):
    def test_basic(self):
        """basic operation"""
        ep = ElementProperties(element=Element("H"))
        self.assertEqual([], ep.get_species_dependencies())
        self.assertEqual([], ep.get_attribute_dependencies())
        self.assertEqual([], ep.get_constant_dependencies())

    def test_rendering(self):
        """members are exposed"""
        ep = ElementProperties(element=Element("N"))
        context = ep.get_rendering_context()
        self.assertEqual(ep.element.get_rendering_context(), context["element"])

    def test_typesafety(self):
        """typesafety is ensured"""
        for invalid in [None, 1, "H", []]:
            with self.assertRaises(ValidationError):
                ElementProperties(element=invalid)
        for invalid in [{}]:
            with self.assertRaises(TypeError):
                ElementProperties(element=invalid)
