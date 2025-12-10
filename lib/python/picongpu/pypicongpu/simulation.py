"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

import datetime
import logging
import typing
from pathlib import Path

import typeguard

from picongpu.pypicongpu.species.operation.operation import Operation
from picongpu.pypicongpu.species.species import Species

from . import species, util
from .customuserinput import InterfaceCustomUserInput
from .field_solver.DefaultSolver import Solver
from .grid import Grid3D
from .laser import (
    DispersivePulseLaser,
    FromOpenPMDPulseLaser,
    GaussianLaser,
    PlaneWaveLaser,
)
from .movingwindow import MovingWindow
from .output import Plugin, OpenPMDPlugin
from .rendering import RenderedObject
from .walltime import Walltime


AnyLaser = DispersivePulseLaser | FromOpenPMDPulseLaser | GaussianLaser | PlaneWaveLaser


@typeguard.typechecked
class Simulation(RenderedObject):
    """
    Represents all parameters required to build & run a PIConGPU simulation.

    Most of the individual parameters are delegated to other objects held as
    attributes.

    To run a Simulation object pass it to the Runner (for details see there).
    """

    base_density = util.build_typesafe_property(float)
    """value to normalise densities"""

    delta_t_si = util.build_typesafe_property(float)
    """Width of a single timestep, given in seconds."""

    time_steps = util.build_typesafe_property(int)
    """Total number of time steps to be executed."""

    grid = util.build_typesafe_property(typing.Union[Grid3D])
    """Used grid Object"""

    laser = util.build_typesafe_property(typing.Optional[list[AnyLaser]])
    """List of laser objects to use in the simulation, or None to disable lasers"""

    solver = util.build_typesafe_property(Solver)
    """Used Solver"""

    init_manager = util.build_typesafe_property(species.InitManager)
    """init manager holding all species & their information"""

    typical_ppc = util.build_typesafe_property(int)
    """
    typical number of macro particles spawned per cell, >=1

    used for normalization of units
    """

    custom_user_input = util.build_typesafe_property(typing.Optional[list[InterfaceCustomUserInput]])
    """
    object that contains additional user specified input parameters to be used in custom templates

    @attention custom user input is global to the simulation
    """

    moving_window = util.build_typesafe_property(typing.Optional[MovingWindow])
    """used moving Window, set to None to disable"""

    walltime = util.build_typesafe_property(typing.Optional[Walltime])
    """time limit of the simulation run"""

    binomial_current_interpolation = util.build_typesafe_property(bool)
    """switch on a binomial current interpolation"""

    plugins = util.build_typesafe_property(typing.Optional[list[Plugin]])

    species = util.build_typesafe_property(list[Species])
    init_operations = util.build_typesafe_property(list[Operation])

    def __init__(
        self,
        /,
        typical_ppc,
        delta_t_si,
        custom_user_input,
        solver,
        grid,
        binomial_current_interpolation,
        moving_window,
        walltime,
        species,
        init_operations,
        time_steps,
        laser,
    ):
        self.laser = laser
        self.time_steps = time_steps
        self.moving_window = moving_window
        self.walltime = walltime
        self.custom_user_input = custom_user_input
        self.binomial_current_interpolation = binomial_current_interpolation
        self.grid = grid
        self.solver = solver
        self.delta_t_si = delta_t_si
        self.typical_ppc = typical_ppc
        self.species = list(species)
        self.init_operations = list(init_operations)

    def __render_custom_user_input_list(self) -> dict:
        custom_rendering_context = {"tags": []}

        for entry in self.custom_user_input:
            add_context = entry.get_generic_rendering_context()
            tags = entry.get_tags()

            entry.check_does_not_change_existing_key_values(custom_rendering_context, add_context)
            entry.check_tags(custom_rendering_context["tags"], tags)

            custom_rendering_context.update(add_context)
            custom_rendering_context["tags"].extend(tags)

        return custom_rendering_context

    def __found_custom_input(self, serialized: dict):
        logging.info(
            "found custom user input with tags: "
            + str(serialized["customuserinput"]["tags"])
            + "\n"
            + "\t WARNING: custom input is not checked, it is the user's responsibility to check inputs and generated input.\n"
            + "\t WARNING: custom templates are required if using custom user input.\n"
        )

    def spread_directory_information(self, setup_dir):
        for plugin in self.plugins:
            if isinstance(plugin, OpenPMDPlugin):
                plugin.setup_dir = Path(setup_dir)

    def _get_serialized(self) -> dict:
        serialized = {
            "delta_t_si": self.delta_t_si,
            "base_density": float(self.base_density),
            "time_steps": self.time_steps,
            "typical_ppc": self.typical_ppc,
            "solver": self.solver.get_rendering_context(),
            "grid": self.grid.get_rendering_context(),
            "species_initmanager": self.init_manager.get_rendering_context(),
            "output": [entry.get_rendering_context() for entry in (self.plugins or [])],
            "species": [s.get_rendering_context() for s in self.species],
            "init_operations": [o.get_rendering_context() for o in self.init_operations],
        }

        if self.laser is not None:
            serialized["laser"] = [ll.get_rendering_context() for ll in self.laser]
        else:
            serialized["laser"] = None

        serialized["moving_window"] = None if self.moving_window is None else self.moving_window.get_rendering_context()
        serialized["walltime"] = (
            self.walltime or Walltime(walltime=datetime.timedelta(hours=1))
        ).get_rendering_context()

        serialized["binomial_current_interpolation"] = self.binomial_current_interpolation

        if self.custom_user_input is not None:
            serialized["customuserinput"] = self.__render_custom_user_input_list()
            self.__found_custom_input(serialized)
        else:
            serialized["customuserinput"] = None

        return serialized
