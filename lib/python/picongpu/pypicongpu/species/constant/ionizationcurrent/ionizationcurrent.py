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

    picongpu_name: str
    """C++ Code type name of ionizer"""

    def _get_serialized(self) -> dict:
        return {"picongpu_name": self.picongpu_name}

    def get_generic_rendering_context(self) -> dict:
        return IonizationCurrent(picongpu_name=self.picongpu_name).get_rendering_context()
