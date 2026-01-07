"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase, main

from picongpu.picmi.species import Species
from picongpu.picmi.species_requirements import (
    run_construction,
    RequirementConflict,
    SetChargeStateOperation,
)
from picongpu.pypicongpu.species.operation.setchargestate import SetChargeState
from picongpu.pypicongpu.species.constant.groundstateionization import GroundStateIonization
from picongpu.picmi.interaction.ionization.fieldionization import ADK, BSI
from picongpu.pypicongpu.species.attribute.weighting import Weighting
from picongpu.pypicongpu.species.constant.mass import Mass


def subset_of(subset, superset):
    superset = list(superset)
    return all(any(e1 == e2 for e2 in superset) for e1 in subset)


def count_all_one(subset, superset):
    superset = list(superset)
    return all(superset.count(e) == 1 for e in subset)


class TestSpeciesRequirementResolution(TestCase):
    def test_deduplicate_attributes(self):
        species = Species(name="dummy")
        requirements = [Weighting()]
        species.register_requirements(2 * requirements)
        assert count_all_one(requirements, species.get_as_pypicongpu().attributes)

    def test_deduplicate_constants(self):
        species = Species(name="dummy")
        requirements = [Mass(mass_si=1.0)]
        species.register_requirements(2 * requirements)
        assert count_all_one(requirements, species.get_as_pypicongpu().constants)

    def test_deduplicate_delayed_construction(self):
        species = Species(name="dummy", particle_type="H", charge_state=1)
        requirements = [SetChargeStateOperation(species)]
        species.register_requirements(2 * requirements)
        assert count_all_one(requirements, species.get_operation_requirements())

    def test_conflicting_constants(self):
        species = Species(name="dummy")
        requirements = [Mass(mass_si=1.0), Mass(mass_si=2.0)]
        with self.assertRaises(RequirementConflict):
            # Not yet decided which one should raise, but one of them definitely will.
            species.register_requirements(requirements)
            species.get_as_pypicongpu()

    def test_ionization(self):
        ion = Species(name="ion", particle_type="H", charge_state=1)
        electron = Species(name="electron", particle_type="electron")
        # These all register requirements:
        ionizations = [
            # Not great: Production code would use the enums not their integer represenation.
            ADK(ion_species=ion, ionization_electron_species=electron, ADK_variant=0, ionization_current=None),
            ADK(ion_species=ion, ionization_electron_species=electron, ADK_variant=1, ionization_current=None),
            BSI(ion_species=ion, ionization_electron_species=electron, BSI_extensions=[0], ionization_current=None),
            BSI(ion_species=ion, ionization_electron_species=electron, BSI_extensions=[1], ionization_current=None),
        ]

        # Ionization makes the ion depend on the electron species.
        # This is important for rendering the corresponding C++ header,
        # so the electron species gets defined before the ion species.
        assert electron < ion

        set_charge_state_op = [
            run_construction(op) for op in ion.get_operation_requirements() if op.metadata.Type == SetChargeState
        ][0]
        assert set_charge_state_op.charge_state == ion.charge_state

        ground_state_ionizations = [
            x for x in ion.get_as_pypicongpu().constants if isinstance(x, GroundStateIonization)
        ]
        # They have been merged:
        assert len(ground_state_ionizations) == 1
        assert len(ground_state_ionizations[0].ionization_model_list) == len(ionizations)


if __name__ == "__main__":
    main()
