"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

import unittest

from picongpu.pypicongpu.species.operation.densityprofile import Gaussian
from pydantic import ValidationError


class TestGaussian(unittest.TestCase):
    values = {
        "gas_center_front": 1.0,
        "gas_center_rear": 2.0,
        "gas_sigma_front": 3.0,
        "gas_sigma_rear": 4.0,
        "gas_power": 5.0,
        "gas_factor": -6.0,
        "vacuum_cells_front": 50,
        "density": 1.0e25,
    }

    def _getGaussian(self, **kwargs):
        return Gaussian(
            **dict(
                center_front=self.values["gas_center_front"],
                center_rear=self.values["gas_center_rear"],
                sigma_front=self.values["gas_sigma_front"],
                sigma_rear=self.values["gas_sigma_rear"],
                power=self.values["gas_power"],
                factor=self.values["gas_factor"],
                vacuum_cells_front=self.values["vacuum_cells_front"],
                density=self.values["density"],
            )
            | kwargs
        )

    def test_value_pass_through(self):
        """values are passed through"""
        g = self._getGaussian()

        self.assertAlmostEqual(self.values["gas_center_front"], g.gas_center_front)
        self.assertAlmostEqual(self.values["gas_center_rear"], g.gas_center_rear)
        self.assertAlmostEqual(self.values["gas_sigma_front"], g.gas_sigma_front)
        self.assertAlmostEqual(self.values["gas_sigma_rear"], g.gas_sigma_rear)
        self.assertAlmostEqual(self.values["gas_power"], g.gas_power)
        self.assertAlmostEqual(self.values["gas_factor"], g.gas_factor)
        self.assertEqual(self.values["vacuum_cells_front"], g.vacuum_cells_front)
        self.assertAlmostEqual(self.values["density"], g.density)

    def test_typesafety(self):
        """typesafety is ensured"""
        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                self._getGaussian(density=invalid)

        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                self._getGaussian(vacuum_cells_front=invalid)

        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                self._getGaussian(factor=invalid)

        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                self._getGaussian(power=invalid)

        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                self._getGaussian(sigma_front=invalid)

        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                self._getGaussian(sigma_rear=invalid)

        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                self._getGaussian(center_front=invalid)

        for invalid in [None, [], {}]:
            with self.assertRaises(ValidationError):
                self._getGaussian(center_rear=invalid)

    def test_check_density(self):
        """validity check on self for invalid density"""
        for invalid in [-1, 0, -0.00000003]:
            with self.assertRaises(ValidationError):
                self._getGaussian(density=invalid)

    def test_check_vacuum_cells_front(self):
        """validity check on self for invalid vacuum_cells_front"""
        for invalid in [-1, -15]:
            with self.assertRaises(ValidationError):
                self._getGaussian(vacuum_cells_front=invalid)

    def test_check_gas_factor(self):
        """validity check on self for invalid gas_factor"""
        for invalid in [0.0, 1.0]:
            with self.assertRaises(ValidationError):
                self._getGaussian(factor=invalid)

    def test_check_gas_power(self):
        """validity check on self for invalid gas_power"""
        for invalid in [0.0]:
            with self.assertRaises(ValidationError):
                self._getGaussian(power=invalid)

    def test_check_gas_sigma_rear(self):
        """validity check on self for invalid gas_sigma_rear"""
        for invalid in [0.0]:
            with self.assertRaises(ValidationError):
                self._getGaussian(sigma_rear=invalid)

    def test_check_gas_sigma_front(self):
        """validity check on self for invalid gas_sigma_front"""
        for invalid in [0.0]:
            with self.assertRaises(ValidationError):
                self._getGaussian(sigma_front=invalid)

    def test_check_gas_center_rear(self):
        """validity check on self for invalid gas_center_rear"""
        for invalid in [-1.0]:
            with self.assertRaises(ValidationError):
                self._getGaussian(center_rear=invalid)
        with self.assertRaises(ValidationError):
            self._getGaussian(center_rear=0.9 * self.values["gas_center_front"])

    def test_check_gas_center_front(self):
        """validity check on self for invalid gas_center_front"""
        for invalid in [-1.0]:
            with self.assertRaises(ValidationError):
                self._getGaussian(center_front=invalid)
        with self.assertRaises(ValidationError):
            self._getGaussian(center_front=1.1 * self.values["gas_center_rear"])

    def test_rendering(self):
        """value passed through from rendering"""
        g = self._getGaussian()

        context = g.get_rendering_context()
        self.assertTrue(context["typeID"]["gaussian"])
        context = context["data"]
        self.assertAlmostEqual(g.gas_center_front, context["gas_center_front"])
        self.assertAlmostEqual(g.gas_center_rear, context["gas_center_rear"])
        self.assertAlmostEqual(g.gas_sigma_front, context["gas_sigma_front"])
        self.assertAlmostEqual(g.gas_sigma_rear, context["gas_sigma_rear"])
        self.assertAlmostEqual(g.gas_power, context["gas_power"])
        self.assertAlmostEqual(g.gas_factor, context["gas_factor"])
        self.assertEqual(g.vacuum_cells_front, context["vacuum_cells_front"])
        self.assertAlmostEqual(g.density, context["density"])
