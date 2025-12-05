"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
import unittest
from difflib import unified_diff
from pathlib import Path
from tempfile import TemporaryDirectory

from picongpu import picmi

from .arbitrary_parameters import (
    CELL_SIZE,
    NUMBER_OF_CELLS,
    UPPER_BOUNDARY,
)
from .distributions import DISTRIBUTIONS

logging.basicConfig(level=logging.INFO)

LAYOUT = picmi.GriddedLayout(n_macroparticles_per_cell=2)


def basic_simulation():
    return picmi.Simulation(
        max_steps=0,
        solver=picmi.ElectromagneticSolver(
            method="Yee",
            cfl=1.0,
            grid=picmi.Cartesian3DGrid(
                number_of_cells=NUMBER_OF_CELLS,
                lower_bound=[0, 0, 0],
                # cell size is slightly different from 1
                upper_bound=UPPER_BOUNDARY,
                lower_boundary_conditions=["open", "open", "open"],
                upper_boundary_conditions=["open", "open", "open"],
            ),
        ),
    )


def generate_name(name, suffix):
    return name + "_" + suffix


def generate_species(name, distribution):
    return [
        picmi.NEW1_Species(
            name=name,
            particle_type="H",
            initial_distribution=distribution,
            picongpu_fixed_charge=True,
        )
    ]


def setup_sim():
    sim = basic_simulation()

    species = sum(
        (
            generate_species(generate_name(name, suffix), dist)
            for name, distributions in DISTRIBUTIONS.items()
            for suffix, dist in distributions.items()
        ),
        [],
    )

    for s in species:
        sim.add_species(s, LAYOUT)
    # sim.diagnostics = [picmi.diagnostics.Checkpoint(TimeStepSpec[:])] + binning_diagnostics(species, sim.time_step_size)

    # sim.step(0)
    sim.write_input_file(TemporaryDirectory(delete=True).name)
    return sim


# only run this once, so we don't compile each and every time
SIM = None
PARAM_PATH = Path("include/picongpu/param/")
SPECIES_DEFINITION_HEADER = PARAM_PATH / "speciesDefinition.param"
SPECIES_INITIALIZATION_HEADER = PARAM_PATH / "speciesInitialization.param"
EXPECTED_RESULT_PATH = Path("expected")


class TestNEW1_Species(unittest.TestCase):
    _setup_path = None

    def setUp(self):
        if self._setup_path is None:
            global SIM
            if SIM is None:
                SIM = setup_sim()
            self.sim = SIM
        else:
            for d in DISTRIBUTIONS:
                for f in DISTRIBUTIONS[d].values():
                    f.cell_size = CELL_SIZE

    @property
    def setup_path(self):
        if self._setup_path is None:
            self._setup_path = Path(self.sim.picongpu_get_runner().setup_dir)
        return self._setup_path

    def _compare_headers(self, header_path):
        with (self.setup_path / header_path).open() as file:
            result = file.read()
        with (EXPECTED_RESULT_PATH / header_path).open() as file:
            expected = file.read()
        try:
            assert result == expected
        except:
            for d in unified_diff(result.split("\n"), expected.split("\n")):
                print(d)
            raise

    def test_compare_species_definition_headers(self):
        self._compare_headers(SPECIES_DEFINITION_HEADER)

    def test_compare_species_initialization_headers(self):
        self._compare_headers(SPECIES_INITIALIZATION_HEADER)
