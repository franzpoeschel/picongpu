"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, PrivateAttr, computed_field
from .timestepspec import TimeStepSpec
from .plugin import Plugin


class Auto(Plugin, BaseModel):
    """
    Class to provide output **without further configuration**.

    This class requires a period (in time steps) and will enable as many output
    plugins as feasable for all species.
    Note: The list of species from the initmanager is used during rendering.

    No further configuration is possible!
    If you want to provide additional configuration for plugins,
    create a separate class.
    """

    period: TimeStepSpec
    """period to print data at"""
    _name: str = PrivateAttr("auto")

    @computed_field
    def png_axis(self) -> list[dict[str, str]]:
        return [
            {"axis": "yx"},
            {"axis": "yz"},
        ]
