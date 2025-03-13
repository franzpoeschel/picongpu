"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Pawel Ordyna
License: GPLv3+
"""

from ...pypicongpu.output.auto import Auto as PyPIConGPUAuto
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies


from ..species import Species as PICMISpecies


import typeguard
import pydantic


@typeguard.typechecked
class Auto(pydantic.BaseModel):
    """
    Specifies the parameters for the Auto output.

    Parameters
    ----------
    period: int
        Number of simulation steps between consecutive outputs.
        Unit: steps (simulation time steps).
    """

    period: int
    """Number of simulation steps between consecutive outputs. Unit: steps (simulation time steps)."""

    def check(self):
        if self.period <= 0:
            raise ValueError("Period must be > 0")

    def get_as_pypicongpu(
        self,
        # not used here, but needed for the interface
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies],
    ) -> PyPIConGPUAuto:
        self.check()

        pypicongpu_auto = PyPIConGPUAuto()
        pypicongpu_auto.period = self.period
        return pypicongpu_auto
