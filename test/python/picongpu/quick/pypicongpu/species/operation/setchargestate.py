"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation import SetChargeState

import unittest

from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.constant import GroundStateIonization
from picongpu.pypicongpu.species.constant.ionizationmodel import BSI
from picongpu.pypicongpu.species.constant.ionizationcurrent import None_
from picongpu.pypicongpu.species.attribute import BoundElectrons, Position, Momentum


class TestSetChargeState(unittest.TestCase):
    def setUp(self):
        self.electron = Species(name="e")
        self.species1 = Species(
            name="ion",
            constants=[
                GroundStateIonization(
                    ionization_model_list=[BSI(ionization_electron_species=self.electron, ionization_current=None_())]
                )
            ],
        )

    def test_basic(self):
        """basic operation"""
        scs = SetChargeState(species=self.species1, charge_state=2)

        # checks pass
        scs.check_preconditions()

    def test_attribute_generated(self):
        """creates bound electrons attribute"""
        scs = SetChargeState(species=self.species1, charge_state=1)

        # emulate initmanager
        scs.check_preconditions()
        self.species1.attributes = []
        scs.prebook_species_attributes()

        self.assertEqual(1, len(scs.attributes_by_species))
        self.assertTrue(self.species1 in scs.attributes_by_species)
        self.assertEqual(1, len(scs.attributes_by_species[self.species1]))
        self.assertTrue(isinstance(scs.attributes_by_species[self.species1][0], BoundElectrons))

    def test_ionizers_required(self):
        """ionizers constant must be present"""
        scs = SetChargeState(species=self.species1, charge_state=1)

        # passes:
        self.assertTrue(scs.species.has_constant_of_type(GroundStateIonization))
        scs.check_preconditions()

        # without constants does not pass:
        scs.species.constants = []
        with self.assertRaisesRegex(AssertionError, ".*BoundElectrons requires GroundStateIonization.*"):
            scs.check_preconditions()

    def test_rendering(self):
        """rendering works"""
        # create full electron species
        electron = Species(name="e", constants=[], attributes=[Position(), Momentum()])

        # can be rendered:
        self.assertNotEqual({}, electron.get_rendering_context())

        ion = Species(
            name="ion",
            constants=[
                GroundStateIonization(
                    ionization_model_list=[BSI(ionization_electron_species=electron, ionization_current=None_())]
                ),
            ],
            attributes=[Position(), Momentum(), BoundElectrons()],
        )

        # can be rendered
        self.assertNotEqual({}, ion.get_rendering_context())

        scs = SetChargeState(species=ion, charge_state=1)

        context = scs.get_rendering_context()
        self.assertEqual(1, context["charge_state"])
        self.assertEqual(ion.get_rendering_context(), context["species"])
