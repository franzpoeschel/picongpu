"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from picongpu.pypicongpu.grid import Grid3D, BoundaryCondition

import unittest

from pydantic import ValidationError


class TestGrid3D(unittest.TestCase):
    def setUp(self):
        """setup default grid"""
        self.kwargs = dict(
            cell_size_si=(1.2, 2.3, 4.5),
            cell_cnt=(6, 7, 8),
            boundary_condition=(
                BoundaryCondition.PERIODIC,
                BoundaryCondition.ABSORBING,
                BoundaryCondition.PERIODIC,
            ),
            n_gpus=(2, 4, 1),
            super_cell_size=(8, 8, 4),
            grid_dist=None,
        )
        self.g = Grid3D(**self.kwargs)

    def test_basic(self):
        """test default setup"""
        g = self.g
        self.assertSequenceEqual((1.2, 2.3, 4.5), g.cell_size)
        self.assertSequenceEqual((6, 7, 8), g.cell_cnt)
        self.assertSequenceEqual(
            (BoundaryCondition.PERIODIC, BoundaryCondition.ABSORBING, BoundaryCondition.PERIODIC), g.boundary_condition
        )

    def test_types(self):
        """test raising errors if types are wrong"""
        with self.assertRaises(ValidationError):
            Grid3D(
                **self.kwargs
                | dict(boundary_condition=("open", BoundaryCondition.ABSORBING, BoundaryCondition.PERIODIC))
            )
        with self.assertRaises(ValidationError):
            Grid3D(
                **self.kwargs | dict(boundary_condition=(BoundaryCondition.PERIODIC, BoundaryCondition.ABSORBING, {}))
            )

    def test_gpu_and_cell_cnt_positive(self):
        """test if n_gpus and cell number s are >0"""
        with self.assertRaisesRegex(ValidationError, ".*cell_cnt.*greater than 0.*"):
            Grid3D(**self.kwargs | dict(cell_cnt=(-1, 7, 8)))

        with self.assertRaisesRegex(ValidationError, ".*cell_cnt.*greater than 0.*"):
            Grid3D(**self.kwargs | dict(cell_cnt=(6, -2, 8)))

        with self.assertRaisesRegex(ValidationError, ".*cell_cnt.*greater than 0.*"):
            Grid3D(**self.kwargs | dict(cell_cnt=(6, 7, 0)))

        for wrong_n_gpus in [tuple([-1, 1, 1]), tuple([1, 1, 0])]:
            with self.assertRaisesRegex(ValidationError, ".*greater than 0.*"):
                Grid3D(**self.kwargs | dict(n_gpus=wrong_n_gpus))

    def test_mandatory(self):
        """test if None as content fails"""
        # check that mandatory arguments can't be none
        with self.assertRaises(ValidationError):
            Grid3D(**self.kwargs | dict(cell_size_si=None))
        with self.assertRaises(ValidationError):
            Grid3D(**self.kwargs | dict(cell_cnt=None))
        with self.assertRaises(ValidationError):
            Grid3D(**self.kwargs | dict(boundary_condition=None))
        with self.assertRaises(ValidationError):
            Grid3D(**self.kwargs | dict(n_gpus=None))

    def test_get_rendering_context(self):
        """object is correctly serialized"""
        # automatically checks against schema
        context = self.g.get_rendering_context()
        self.assertEqual(1.2, context["cell_size"]["x"])
        self.assertEqual(2.3, context["cell_size"]["y"])
        self.assertEqual(4.5, context["cell_size"]["z"])
        self.assertEqual(6, context["cell_cnt"]["x"])
        self.assertEqual(7, context["cell_cnt"]["y"])
        self.assertEqual(8, context["cell_cnt"]["z"])

        # boundary condition translated to numbers for cfgfiles
        self.assertEqual("1", context["boundary_condition"]["x"])
        self.assertEqual("0", context["boundary_condition"]["y"])
        self.assertEqual("1", context["boundary_condition"]["z"])

        # n_gpus ouput
        self.assertEqual(2, context["gpu_cnt"]["x"])
        self.assertEqual(4, context["gpu_cnt"]["y"])
        self.assertEqual(1, context["gpu_cnt"]["z"])


class TestBoundaryCondition(unittest.TestCase):
    def test_cfg_translation(self):
        """test boundary condition strings"""
        p = BoundaryCondition.PERIODIC
        a = BoundaryCondition.ABSORBING
        self.assertEqual("0", a.get_cfg_str())
        self.assertEqual("1", p.get_cfg_str())
