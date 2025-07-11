"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import sys
import openpmd_api as opmd
import numpy as np
import pandas as pd


def read_particles(series_name):
    series = opmd.Series(str(series_name), opmd.Access.read_only)
    names, particles = zip(*series.iterations[0].particles.items())

    data = pd.concat((particle.to_df() for particle in particles), keys=names)
    return data.assign(
        setup=data.index.to_frame()[0].apply(lambda x: x.split("_")[0]),
        impl=data.index.to_frame()[0].apply(lambda x: "_".join(x.split("_")[1:])),
    ).set_index(["setup", "impl"], drop=True)


def sort_particles(data):
    return data.groupby(["setup", "impl"]).apply(lambda df: df.sort_values(list(df.columns), axis=0))


def compare_particles_per_setup(data):
    result = True
    for setup_name, df in sort_particles(data).groupby("setup"):
        # We're doing (more than) twice the work here but that's fine:
        for lhs_name, lhs_df in df.groupby("impl"):
            for rhs_name, rhs_df in df.groupby("impl"):
                matched = np.allclose(
                    lhs_df.reset_index(drop=True),
                    rhs_df.reset_index(drop=True),
                    atol=0,
                    rtol=1.0e-4,
                )
                if not matched:
                    print(f"Mismatch found in {setup_name}: {lhs_name} != {rhs_name}")
                result *= matched
    return result


def compute_densities_per_setup_and_impl(data):
    return data.groupby(["setup", "impl", "positionOffset_x", "positionOffset_y", "positionOffset_z"])[
        "weighting"
    ].sum()


def compute_densities_from_particles(series_name):
    """
    This density is not normalised by volume yet.
    """
    return compute_densities_per_setup_and_impl(read_particles(series_name))


def compare_particles(series_name):
    """
    Compare particles from the given series and return True if they all compare equal.
    """
    return compare_particles_per_setup(read_particles(series_name))


def main(series_name):
    compare_particles(series_name)


if __name__ == "__main__":
    main(sys.argv[1])
