"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, Field, PrivateAttr

from .layout import Layout


class OnePosition(Layout, BaseModel):
    _name: str = PrivateAttr("one_position")
    ppc: int = Field(gt=0)
    """particles per cell, >0"""
