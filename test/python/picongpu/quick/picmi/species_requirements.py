"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from picongpu.picmi.species_requirements import (
    resolve_requirements,
    RequirementConflict,
    SetChargeStateOperation,
    GroundStateIonizationConstruction,
)
from picongpu.picmi.species import Species
from picongpu.pypicongpu.species.constant.mass import Mass
from picongpu.pypicongpu.species.attribute.weighting import Weighting
from unittest import TestCase, main

DUMMY_SPECIES = Species(name="dummy", particle_type="H", charge_state=1)


class TestResolveRequirements(TestCase):
    def test_empty(self):
        assert resolve_requirements([]) == []

    def test_duplicates(self):
        # Constants, Attributes and some DelayedConstructions must be unique and will get deduplicated.
        for requirements in [[Mass(mass_si=1.0)], [Weighting()], [SetChargeStateOperation(DUMMY_SPECIES)]]:
            assert resolve_requirements(2 * requirements) == requirements

    def test_conflicting_constants(self):
        req_with_conflicting_constants = [Mass(mass_si=1.0), Mass(mass_si=2.0)]
        with self.assertRaises(RequirementConflict):
            resolve_requirements(req_with_conflicting_constants)

    def test_custom_merge(self):
        dummy_models = ["a", "b", "c"]
        individuals = [GroundStateIonizationConstruction(m) for m in dummy_models]
        # This is hacky but okay...
        combined = GroundStateIonizationConstruction(None)
        combined.metadata.kwargs["ionization_model_list"] = dummy_models

        self.assertListEqual(resolve_requirements(individuals), [combined])


if __name__ == "__main__":
    main()
