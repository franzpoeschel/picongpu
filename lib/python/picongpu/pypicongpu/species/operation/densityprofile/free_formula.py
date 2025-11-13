"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, PrivateAttr

from ....rendering.pmaccprinter import PMAccPrinter
from .densityprofile import DensityProfile


class FreeFormula(DensityProfile, BaseModel):
    _name: str = PrivateAttr("freeformula")
    function_body: Annotated[str, BeforeValidator(PMAccPrinter().doprint)] = Field(alias="density_expression")
