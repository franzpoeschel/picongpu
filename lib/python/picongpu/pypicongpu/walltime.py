"""
This file is part of the PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

import pydantic
import datetime

from .rendering import RenderedObject


class Walltime(RenderedObject, pydantic.BaseModel):
    walltime: datetime.timedelta
    """time after which the cluster scheduler will stop the simulation"""

    def check(self) -> None:
        if self.walltime.total_seconds() <= 0.0:
            raise ValueError("walltime must be > 0.")

    def _get_serialized(self) -> dict:
        return {
            "walltime": "{:d}:{:02d}:{:02d}".format(
                int(self.walltime.total_seconds() // 3600),  # hours
                int(self.walltime.total_seconds() % 3600 // 60),  # minutes
                int(self.walltime.total_seconds() % 3600),  # seconds
            )
        }  # @todo: might be cluster specific
        # @todo might be better to use a version of __str__()
