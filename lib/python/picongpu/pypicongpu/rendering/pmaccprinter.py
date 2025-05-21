"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from sympy.printing.cxx import cxx_code_printers

PMACC_MATH_FUNCTIONS = {
    "Abs": "abs",
}


class PMAccPrinter(cxx_code_printers["c++17"]):
    # Originally, the C++ printers use `_ns = "std::"`.
    _ns = "pmacc::math::"
    _kf = cxx_code_printers["c++17"]._kf | PMACC_MATH_FUNCTIONS
