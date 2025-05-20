"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from numpy import vectorize
from ...pypicongpu import species


import logging
import typeguard
import typing
import sympy
import traceback

"""
note on rms_velocity:
---------------------
The rms_velocity is converted to a temperature in keV. This conversion requires the mass of the species to be known,
which is not the case inside the picmi density distribution.

As an abstraction, **every** PICMI density distribution implements `picongpu_get_rms_velocity_si()` which returns a
tuple (float, float, float) with the rms_velocity per axis in SI units (m/s).

In case the density profile does not have an rms_velocity, this method **MUST** return (0, 0, 0), which is translated to
"no temperature initialization" by the owning species.

note on drift:
--------------
The drift ("velocity") is represented using either directed_velocity or centroid_velocity (v, gamma*v respectively) and
for the pypicongpu representation stored in a separate object (Drift).

To accommodate that, this separate Drift object can be requested by the method get_picongpu_drift(). In case of no drift,
this method returns None.
"""


@typeguard.typechecked
class AnalyticDistribution:
    """Analytic Particle Distribution as defined by PICMI @todo"""

    def __init__(self, density_expression, directed_velocity=(0.0, 0.0, 0.0)):
        self.density_function = density_expression
        self.rms_velocity = (0.0, 0.0, 0.0)
        self.directed_velocity = tuple(float(v) for v in directed_velocity)
        x, y, z = sympy.symbols("x,y,z")
        self.density_expression = (
            # We add the simplify because of the following:
            # Translating to C++ requires ALL cases of a Piecewise to be defined
            # such that there's a fallback for if-conditions.
            # The user might have written their formula piecing together
            # multiple partial Piecewise instances.
            # Without simplification, sympy tries to translate them individually.
            # and fails to do so even if they supplement each other
            # into a function defined everywhere.
            sympy.simplify(self.density_function(x, y, z))
            # density_expression might be independent of any or all of the three variables.
            # (This might even happen due to the simplification.)
            # In order to be sure to arrive at a function of these three variables,
            # we add this trivial additional term.
            + (0 * x * y * z)
        )
        self.warned_about_lambdify_failure = False

    def get_as_pypicongpu(self) -> species.operation.densityprofile.DensityProfile:
        x, y, z = sympy.symbols("x,y,z")
        return species.operation.densityprofile.FreeFormula(
            density_expression=(
                # We add the simplify because of the following:
                # Translating to C++ requires ALL cases of a Piecewise to be defined
                # such that there's a fallback for if-conditions.
                # The user might have written their formula piecing together
                # multiple partial Piecewise instances.
                # Without simplification, sympy tries to translate them individually.
                # and fails to do so even if they supplement each other
                # into a function defined everywhere.
                sympy.simplify(self.density_expression(x, y, z))
                # density_expression might be independent of any or all of the three variables.
                # (This might even happen due to the simplification.)
                # In order to be sure to arrive at a function of these three variables,
                # we add this trivial additional term.
                + (0 * x * y * z)
            )
        )

    def picongpu_get_rms_velocity_si(self) -> typing.Tuple[float, float, float]:
        return self.rms_velocity

    def get_picongpu_drift(self) -> typing.Optional[species.operation.momentum.Drift]:
        """
        Get drift for pypicongpu
        :return: pypicongpu drift object or None
        """
        if all(v == 0 for v in self.directed_velocity):
            return None

        drift = species.operation.momentum.Drift()
        drift.fill_from_velocity(self.directed_velocity)
        return drift

    def __call__(self, *args, **kwargs):
        try:
            # This produces faster code but the code generation is not perfect.
            # There are cases where the generated code can't handle broadcasting properly.
            return sympy.lambdify(
                sympy.symbols("x,y,z"), self.density_expression, "numpy"
            )(*args, **kwargs)
        except ValueError:
            if not self.warned_about_lambdify_failure:
                message = (
                    "Sympy did not manage to produce proper numpy code for your AnalyticDistribution. "
                    "If you run into performance problems, try to rewrite your function. "
                    "Here's the original error message:"
                )
                logging.warning(message)
                logging.warning(traceback.format_exc())
                logging.warning(
                    "Continuing operation using a slower serialised version now."
                )
                self.warned_about_lambdify_failure = True
        # This basically calls the original function in a big loop.
        # Slower but more reliable in some cases of difficult broadcasting.
        return vectorize(self.density_function)(*args, **kwargs)
