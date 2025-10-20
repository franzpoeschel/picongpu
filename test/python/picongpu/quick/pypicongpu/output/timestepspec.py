"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from picongpu.pypicongpu.output import TimeStepSpec
from picongpu.pypicongpu.output.timestepspec import Spec
import unittest

from pydantic import ValidationError


class TestTimeStepSpec(unittest.TestCase):
    def test_init_with_valid_slices(self):
        specs = [slice(0, 10, 1), slice(10, 20, 2)]
        time_step_spec = TimeStepSpec(specs=specs)
        self.assertEqual(time_step_spec.specs, [Spec(start=s.start, stop=s.stop, step=s.step) for s in specs])

    def test_init_with_invalid_slices(self):
        with self.assertRaises(ValidationError):
            TimeStepSpec(specs=[slice(0, 10, 1), "invalid"])

    def test_serialize_with_valid_slices(self):
        specs = [slice(0, 10, 1), slice(10, 20, 2)]
        time_step_spec = TimeStepSpec(specs=specs)
        serialized = time_step_spec.get_rendering_context()
        expected = {
            "specs": [
                {"start": 0, "stop": 10, "step": 1},
                {"start": 10, "stop": 20, "step": 2},
            ]
        }
        self.assertEqual(serialized, expected)

    def test_serialize_with_none_values(self):
        specs = [slice(None, 10, 1), slice(10, None, 2), slice(10, 20, None)]
        time_step_spec = TimeStepSpec(specs=specs)
        serialized = time_step_spec.get_rendering_context()
        expected = {
            "specs": [
                {"start": 0, "stop": 10, "step": 1},
                {"start": 10, "stop": -1, "step": 2},
                {"start": 10, "stop": 20, "step": 1},
            ]
        }
        self.assertEqual(serialized, expected)
