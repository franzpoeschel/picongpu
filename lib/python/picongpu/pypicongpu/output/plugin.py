"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Brian Edward Marre, Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel

from ..rendering import SelfRegisteringRenderedObject


class Plugin(SelfRegisteringRenderedObject, BaseModel):
    """general interface for all plugins"""

    pass
