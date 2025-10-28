"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, PrivateAttr

from .layout import Layout


class Random(Layout, BaseModel):
    _name: str = PrivateAttr("random")
    ppc: int
    """particles per cell (random layout), >0"""
