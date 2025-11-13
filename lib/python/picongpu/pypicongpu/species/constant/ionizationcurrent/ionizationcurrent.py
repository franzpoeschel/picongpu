"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

import pydantic
import typeguard

from ..constant import Constant


@typeguard.typechecked
class IonizationCurrent(Constant, pydantic.BaseModel):
    """base class for all ionization currents models"""

    PICONGPU_NAME: str
    """C++ Code type name of ionizer"""

    def _get_serialized(self) -> dict:
        return {"picongpu_name": self.PICONGPU_NAME}

    def get_generic_rendering_context(self) -> dict:
        return IonizationCurrent(PICONGPU_NAME=self.PICONGPU_NAME).get_rendering_context()
