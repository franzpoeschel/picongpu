"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from pydantic import BaseModel, Field
from ....rendering import RenderedObject

# Note to the future maintainer:
# If you want to add another way to specify the temperature, please turn
# Temperature() into an (abstract) parent class, and add one child class per
# method. (Currently only initialization by giving a temperature in keV is
# supported, so such a structure would be overkill.)


class Temperature(RenderedObject, BaseModel):
    """
    Initialize momentum from temperature
    """

    temperature_kev: float = Field(gt=0.0)
    """temperature to use in keV"""
