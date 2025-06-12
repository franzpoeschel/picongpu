"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import sympy
from sympy.vector import CoordSys3D, Vector
from .compare_particles import compare_particles, compute_densities_from_particles
from picongpu import picmi
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec

NUMBER_OF_CELLS = [64, 64, 64]
UPPER_BOUNDARY = np.array([64.0, 66.0, 74.0])
UPPER_BOUNDARY = np.array([64.0, 64.0, 64.0])
CELL_SIZE = UPPER_BOUNDARY / NUMBER_OF_CELLS


# This is just meant as a form of namespace to bundle the setups together.
class Gaussian:
    def __init__(self):
        self.parameters = {
            "density": 1.0e25,
            # cell size is 1, so we don't need to distinguish
            # between number of cells and value in SI
            "vacuum_front": 14.0,
            "center_front": 29,
            "center_rear": 54,
            "sigma_front": 10,
            "sigma_rear": 3,
            "power": 2.0,
            "factor": -2.0,
        }
        self.distributions = {
            "predefined": picmi.GaussianDistribution(**self.parameters),
            "free_form": picmi.AnalyticDistribution(
                lambda x, y, z: self.free_form(y, cell_size_y=1.0, **self.parameters)
            ),
        }

    @staticmethod
    def free_form(
        y,
        density,
        cell_size_y,
        vacuum_front,
        center_front,
        center_rear,
        sigma_front,
        sigma_rear,
        power,
        factor,
    ):
        # apparently, our SI position is the centre of the cell
        y += -0.5 * CELL_SIZE[1]
        # The last term undoes the shift to the cell origin.
        vacuum_y = vacuum_front - 0.5 * CELL_SIZE[1]

        exponent = sympy.Piecewise(
            (sympy.Abs((y - center_front) / sigma_front), y < center_front),
            (sympy.Abs((y - center_rear) / sigma_rear), y >= center_rear),
            (0.0, True),
        )
        return sympy.Piecewise((0.0, y < vacuum_y), (density * sympy.exp(factor * exponent**power), True))


# This is just meant as a form of namespace to bundle the setups together.
class Uniform:
    def __init__(self):
        self.arbitrary_value = 8.0e24
        self.distributions = {
            "predefined": picmi.UniformDistribution(density=self.arbitrary_value),
            "free_form": picmi.AnalyticDistribution(lambda x, y, z: self.arbitrary_value),
        }


class Foil:
    def __init__(self):
        self.parameters = dict(
            density=8.0e24,
            front=10,
            thickness=17,
            exponential_pre_plasma_length=12,
            exponential_pre_plasma_cutoff=14,
            exponential_post_plasma_length=5,
            exponential_post_plasma_cutoff=16,
        )
        self.distributions = {
            "predefined": picmi.FoilDistribution(**self.parameters),
            "free_form": picmi.AnalyticDistribution(lambda x, y, z: self.free_form(y, **self.parameters)),
        }

    @staticmethod
    def free_form(
        y,
        density,
        front,
        thickness,
        exponential_pre_plasma_length,
        exponential_pre_plasma_cutoff,
        exponential_post_plasma_length,
        exponential_post_plasma_cutoff,
    ):
        pre_plasma_ramp = (
            # expression
            sympy.exp((y - front) / exponential_pre_plasma_length)
            if exponential_pre_plasma_length is not None
            else 0.0,
            # condition
            sympy.And(y < front, y > front - exponential_pre_plasma_cutoff),
        )

        post_plasma_ramp = (
            # expression
            sympy.exp((front + thickness - y) / exponential_post_plasma_length)
            if exponential_post_plasma_length is not None
            else 0.0,
            # condition
            sympy.And(
                y > front + thickness,
                y < front + thickness + exponential_post_plasma_cutoff,
            ),
        )

        foil = (1.0, sympy.And(y >= front, y <= front + thickness))
        vacuum = (0.0, True)

        return density * sympy.Piecewise(pre_plasma_ramp, foil, post_plasma_ramp, vacuum)


def _make_vector(coefficients, basis_vectors):
    # In sympy, vectors are represented as linear combinations of basis vectors.
    # The last argument is important.
    # Otherwise Python tries to start from an integer (scalar) 0 which is not well-defined.
    return sum((coeff * vec for coeff, vec in zip(coefficients, basis_vectors)), Vector.zero)


class Cylinder:
    def __init__(self):
        self.parameters = dict(
            density=8.0e24,
            center_position=(17.0, 23.0, 45.0),
            radius=10,
            cylinder_axis=(1.0, 2.0, 3.0),
            exponential_pre_plasma_length=5.0,
            exponential_pre_plasma_cutoff=3.0,
        )
        self.distributions = {
            "predefined": picmi.CylindricalDistribution(**self.parameters),
            "free_form": picmi.AnalyticDistribution(lambda x, y, z: self.free_form(x, y, z, **self.parameters)),
        }

    @staticmethod
    def free_form(
        x,
        y,
        z,
        density,
        center_position,
        radius,
        cylinder_axis,
        exponential_pre_plasma_length,
        exponential_pre_plasma_cutoff,
    ):
        # The definition of this density uses the origin of the cell
        # while the call operator uses the center.
        x += -0.5 * CELL_SIZE[0]
        y += -0.5 * CELL_SIZE[1]
        z += -0.5 * CELL_SIZE[2]

        # Handling vectors in sympy starts from a coordinate system.
        e = CoordSys3D("e")

        # Every vector is expressed as a linear combination of basis vectors.
        # This is abstracted away in `_make_vector`.
        cylinder_axis = _make_vector(cylinder_axis, e).normalize()
        r = (_make_vector([x, y, z], e) - _make_vector(center_position, e)).cross(cylinder_axis).magnitude()
        radius = (
            sympy.sqrt(radius**2 - exponential_pre_plasma_length**2) - exponential_pre_plasma_length
            if exponential_pre_plasma_cutoff > 0.0 and exponential_pre_plasma_length > 0.0
            else radius
        )

        cylinder = (1.0, r < radius)

        pre_plasma_ramp = (
            # expression
            sympy.exp((radius - r) / exponential_pre_plasma_length)
            if exponential_pre_plasma_cutoff > 0.0 and exponential_pre_plasma_length > 0.0
            else 0.0,
            # condition
            sympy.And(r > radius, r < radius + exponential_pre_plasma_cutoff),
        )
        vacuum = (0.0, True)

        return density * sympy.Piecewise(cylinder, pre_plasma_ramp, vacuum)


# This is a predefined setup within PIConGPU but not PICMI.
class LinearExponential:
    def __init__(self):
        self.parameters = dict(
            density=8.0e24,
            vacuum_y=14.0,
            gas_a=10.0,
            gas_b=12.0,
            gas_d=-0.1,
            gas_y_max=25.0,
        )
        self.distributions = {
            "free_form": picmi.AnalyticDistribution(lambda x, y, z: self.free_form(y, **self.parameters)),
        }

    @staticmethod
    def free_form(y, density, vacuum_y, gas_a, gas_b, gas_d, gas_y_max):
        # move to the origin of the cell
        y += -0.5 * CELL_SIZE[1]

        vacuum = (0.0, y < vacuum_y)
        linear_slope = (
            sympy.Max(0.0, gas_a * y + gas_b),
            sympy.And(y >= vacuum_y, y < gas_y_max),
        )
        exponential_slope = (sympy.exp(gas_d * (y - gas_y_max)), True)

        return density * sympy.Piecewise(vacuum, linear_slope, exponential_slope)


# This is a predefined setup within PIConGPU but not PICMI.
class SphereFlanks:
    def __init__(self):
        self.parameters = dict(
            density=8.0e24,
            vacuum_y=14.0,
            center=[10.0, 40.0, 35.0],
            r=20.0,
            ri=10.0,
            exponent=1.0,
        )
        self.distributions = {
            "free_form": picmi.AnalyticDistribution(lambda x, y, z: self.free_form(x, y, z, **self.parameters)),
        }

    @staticmethod
    def free_form(x, y, z, density, vacuum_y, center, r, ri, exponent):
        # move to the origin of the cell
        y += -0.5 * CELL_SIZE[1]

        distance = sympy.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
        front_vacuum = (0.0, y < vacuum_y)
        inner_vacuum = (0.0, distance < ri)
        sphere = (1.0, distance <= r)
        flanks = (sympy.exp(exponent * (r - distance)), True)

        return density * sympy.Piecewise(front_vacuum, inner_vacuum, sphere, flanks)


DISTRIBUTIONS = {
    "Uniform": Uniform().distributions,
    "Gaussian": Gaussian().distributions,
    "Foil": Foil().distributions,
    "LinearExponential": LinearExponential().distributions,
    "SphereFlanks": SphereFlanks().distributions,
    "Cylinder": Cylinder().distributions,
}


def generate_name(name, suffix):
    return name + "_" + suffix


def add(sim, name, **distributions):
    random_layout = picmi.GriddedLayout(n_macroparticles_per_cell=2)
    for suffix, distribution in distributions.items():
        species_hydrogen = picmi.Species(
            name=generate_name(name, suffix),
            particle_type="H",
            initial_distribution=distribution,
            picongpu_fixed_charge=True,
        )
        sim.add_species(species_hydrogen, random_layout)


def setup_sim():
    grid = picmi.Cartesian3DGrid(
        number_of_cells=NUMBER_OF_CELLS,
        lower_bound=[0, 0, 0],
        # cell size is slightly different from 1
        upper_bound=UPPER_BOUNDARY,
        lower_boundary_conditions=["open", "open", "open"],
        upper_boundary_conditions=["open", "open", "open"],
    )
    solver = picmi.ElectromagneticSolver(method="Yee", grid=grid, cfl=1.0)
    sim = picmi.Simulation(max_steps=0, solver=solver)
    sim.diagnostics = [picmi.diagnostics.Checkpoint(TimeStepSpec[:])]

    for name, distributions in DISTRIBUTIONS.items():
        add(sim, name, **distributions)

    sim.step(0)
    return sim


# only run this once, so we don't compile each and every time
SIM = None


class TestFreeFormulaDensity(unittest.TestCase):
    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
        self.sim = SIM

    @property
    def result_path(self):
        return Path(self.sim._Simulation__runner.run_dir) / "simOutput" / "checkpoints" / "checkpoint_000000.bp5"

    def test_compare_particles_pairwise(self):
        self.assertTrue(compare_particles(self.result_path))

    def test_compare_particles_against_call_operator(self):
        densities = compute_densities_from_particles(self.result_path).to_frame().rename({"weighting": "found"}, axis=1)
        densities["expected"] = (
            densities.reset_index(drop=False)
            .groupby(["setup", "impl"])
            .apply(
                lambda df: pd.Series(
                    # We add 0.5 because the density is evaluated at the centre of the cell.
                    DISTRIBUTIONS[df.name[0]][df.name[1]](
                        df.positionOffset_x.to_numpy() + 0.5 * CELL_SIZE[0],
                        df.positionOffset_y.to_numpy() + 0.5 * CELL_SIZE[1],
                        df.positionOffset_z.to_numpy() + 0.5 * CELL_SIZE[2],
                    ),
                    index=df.set_index(
                        ["positionOffset_x", "positionOffset_y", "positionOffset_z"],
                        drop=True,
                    ).index,
                ).astype(float),
                include_groups=False,
            )
        )

        # The Gaussian rear tail somehow has a bad accuracy.
        np.testing.assert_allclose(densities.found, densities.expected, rtol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
