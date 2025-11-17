/* Copyright 2024-2024 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

//! @file implements Stewart-Pyatt ionization potential depression(IPD) model

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/IPDModel.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/LocalIPDInputFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/SumFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/ApplyIPDIonization.def"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/CalculateIPDInput.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/FillIPDSumFields_Electron.def"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/FillIPDSumFields_Ion.def"
#include "picongpu/particles/atomicPhysics/localHelperFields/FoundUnboundIonField.hpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    namespace detail
    {
        struct StewartPyattSuperCellConstantInput
        {
            // eV
            float_X temperatureTimesk_Boltzman;
            // UNIT_LENGTH
            float_X debyeLength;
            // unitless
            float_X zStar;

            constexpr StewartPyattSuperCellConstantInput(
                float_X temperatureTimesk_Boltzman_i,
                float_X debyeLength_i,
                float_X zStar_i)
                : temperatureTimesk_Boltzman(temperatureTimesk_Boltzman_i)
                , debyeLength(debyeLength_i)
                , zStar(zStar_i)
            {
            }
        };
    } // namespace detail

    //! short hand for IPD namespace
    namespace s_IPD = picongpu::particles::atomicPhysics::ionizationPotentialDepression;

    /** implementation of Stewart-Pyatt ionization potential depression(IPD) model
     *
     * @tparam T_TemperatureFunctor term A to average over for all macro particles according to equi-partition theorem,
     * average(A) = k_B * T, must follow
     */
    template<typename T_TemperatureFunctor>
    struct StewartPyattIPD : IPDModel
    {
        using SuperCellConstantInput = detail::StewartPyattSuperCellConstantInput;

    private:
        //! reset IPD support infrastructure before we accumulate over particles to calculate new IPD Inputs
        HINLINE static void resetSumFields()
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localSumWeightAllField
                = *dc.get<s_IPD::localHelperFields::SumWeightAllField<picongpu::MappingDesc>>("SumWeightAllField");
            localSumWeightAllField.getDeviceBuffer().setValue(0._X);

            auto& localSumTemperatureFunctionalField
                = *dc.get<s_IPD::localHelperFields::SumTemperatureFunctionalField<picongpu::MappingDesc>>(
                    "SumTemperatureFunctionalField");
            localSumTemperatureFunctionalField.getDeviceBuffer().setValue(0._X);

            auto& localSumWeightElectronField
                = *dc.get<s_IPD::localHelperFields::SumWeightElectronsField<picongpu::MappingDesc>>(
                    "SumWeightElectronsField");
            localSumWeightElectronField.getDeviceBuffer().setValue(0._X);

            auto& localSumChargeNumberIonsField
                = *dc.get<s_IPD::localHelperFields::SumChargeNumberIonsField<picongpu::MappingDesc>>(
                    "SumChargeNumberIonsField");
            localSumChargeNumberIonsField.getDeviceBuffer().setValue(0._X);

            auto& localSumChargeNumberSquaredIonsField
                = *dc.get<s_IPD::localHelperFields::SumChargeNumberSquaredIonsField<picongpu::MappingDesc>>(
                    "SumChargeNumberSquaredIonsField");
            localSumChargeNumberSquaredIonsField.getDeviceBuffer().setValue(0._X);
        }

    public:
        //! create the sum- and IPD-Input superCell fields required by Stewart-Pyatt
        HINLINE static void createHelperFields(
            picongpu::DataConnector& dataConnector,
            picongpu::MappingDesc const mappingDesc)
        {
            // create sumFields
            //@{
            auto sumWeightAllField
                = std::make_unique<s_IPD::localHelperFields::SumWeightAllField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(sumWeightAllField));
            auto sumTemperatureFunctionalField
                = std::make_unique<s_IPD::localHelperFields::SumTemperatureFunctionalField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(sumTemperatureFunctionalField));

            auto sumWeightElectronsField
                = std::make_unique<s_IPD::localHelperFields::SumWeightElectronsField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(sumWeightElectronsField));

            auto sumChargeNumberIonsField
                = std::make_unique<s_IPD::localHelperFields::SumChargeNumberIonsField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(sumChargeNumberIonsField));
            auto sumChargeNumberSquaredIonsField
                = std::make_unique<s_IPD::localHelperFields::SumChargeNumberSquaredIonsField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(sumChargeNumberSquaredIonsField));
            //@}

            // create IPD input Fields
            //@{
            // in sim.unit.length(), not weighted
            auto debyeLengthField
                = std::make_unique<s_IPD::localHelperFields::DebyeLengthField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(debyeLengthField));

            // z^star IPD input field, z^star = = average(q^2) / average(q) ;for q charge number of ion, unitless,
            //  not weighted
            auto zStarField
                = std::make_unique<s_IPD::localHelperFields::ZStarField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(zStarField));

            // local k_Boltzman * Temperature field, in eV
            auto temperatureEnergyField
                = std::make_unique<s_IPD::localHelperFields::TemperatureEnergyField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(temperatureEnergyField));
            //@}
        }

        /** calculate all inputs for the ionization potential depression
         *
         * @tparam T_IPDIonSpeciesList list of all species partaking as ions in IPD input
         * @tparam T_IPDElectronSpeciesList list of all species partaking as electrons in IPD input
         *
         * @attention collective over all IPD species
         */
        template<
            uint32_t T_numberAtomicPhysicsIonSpecies,
            typename T_IPDIonSpeciesList,
            typename T_IPDElectronSpeciesList>
        HINLINE static void calculateIPDInput(picongpu::MappingDesc const mappingDesc, uint32_t const)
        {
            using ForEachElectronSpeciesFillSumFields = pmacc::meta::ForEach<
                T_IPDElectronSpeciesList,
                s_IPD::stage::FillIPDSumFields_Electron<boost::mpl::_1, T_TemperatureFunctor>>;
            using ForEachIonSpeciesFillSumFields = pmacc::meta::
                ForEach<T_IPDIonSpeciesList, s_IPD::stage::FillIPDSumFields_Ion<boost::mpl::_1, T_TemperatureFunctor>>;

            // reset IPD SumFields
            resetSumFields();

            ForEachElectronSpeciesFillSumFields{}(mappingDesc);
            ForEachIonSpeciesFillSumFields{}(mappingDesc);

            s_IPD::stage::CalculateIPDInput<T_numberAtomicPhysicsIonSpecies>()(mappingDesc);
        }

        /** check for and apply a single step of the IPD-ionization cascade
         *
         * @attention assumes that ipd-input fields are up to date
         * @attention invalidates ipd-input fields if at least one ionization electron has been spawned
         *
         * @attention must be called once for each step in a pressure ionization cascade
         *
         * @tparam T_AtomicPhysicsIonSpeciesList list of all species partaking as ion in atomicPhysics
         * @tparam T_IPDIonSpeciesList list of all species partaking as ions in IPD input
         * @tparam T_IPDElectronSpeciesList list of all species partaking as electrons in IPD input
         *
         * @attention collective over all ion species
         */
        template<typename T_AtomicPhysicsIonSpeciesList>
        HINLINE static void applyIPDIonization(picongpu::MappingDesc const mappingDesc, uint32_t const)
        {
            using ForEachIonSpeciesApplyIPDIonization = pmacc::meta::ForEach<
                T_AtomicPhysicsIonSpeciesList,
                s_IPD::stage::ApplyIPDIonization<boost::mpl::_1, StewartPyattIPD<T_TemperatureFunctor>>>;

            ForEachIonSpeciesApplyIPDIonization{}(mappingDesc);
        };

        /** get the super cell constant input
         *
         * @param superCellFieldIdx index of superCell in superCellField(without guards)
         * @param debyeLengthBox deviceDataBox giving access to the local debye length for all local superCells,
         *  sim.unit.length(), not weighted
         * @param temperatureEnergyBox deviceDataBox giving access to the local temperature * k_Boltzman for all
         *  local superCells, in UNIT_ENERGY, not weighted
         * @param zStarBox deviceDataBox giving access to the local z^Star value, = average(q^2) / average(q),
         *  for all local superCells, unitless, not weighted
         */
        template<typename T_DebyeLengthBox, typename T_TemperatureEnergyBox, typename T_ZStarBox>
        HDINLINE static SuperCellConstantInput getSuperCellConstantInput(
            pmacc::DataSpace<simDim> const superCellFieldIdx,
            T_DebyeLengthBox const debyeLengthBox,
            T_TemperatureEnergyBox const temperatureEnergyBox,
            T_ZStarBox const zStarBox)
        {
            // eV, not weighted
            float_X temperatureTimesk_Boltzman = temperatureEnergyBox(superCellFieldIdx);
            // UNIT_LENGTH, not weighted
            float_X debyeLength = debyeLengthBox(superCellFieldIdx);
            // unitless, not weighted
            float_X zStar = zStarBox(superCellFieldIdx);

            return SuperCellConstantInput(temperatureTimesk_Boltzman, debyeLength, zStar);
        }

        /** calculate ionization potential depression
         *
         * @param superCellConstantInput struct storing the values of all super cell constant IPD input parameters
         * @param chargeState charge state of the ion before ionization
         *
         * @return unit: eV, not weighted
         */
        HDINLINE static float_X ipd(SuperCellConstantInput const superCellConstantInput, uint8_t const chargeState)
        {
            // UNIT_CHARGE^2 / ( unitless * UNIT_CHARGE^2 * UNIT_TIME^2 / (UNIT_LENGTH^3 * UNIT_MASS))
            // = UNIT_MASS * UNIT_LENGTH^3 * UNIT_TIME^(-2)
            constexpr float_X constFactor
                = pmacc::math::cPow(picongpu::sim.pic.getElectronCharge(), 2u)
                  / (4._X * static_cast<float_X>(picongpu::PI) * picongpu::sim.pic.getEps0());

            // UNIT_ENERGY/eV
            constexpr float_X eV = sim.pic.get_eV();

            // UNIT_MASS * UNIT_LENGTH^3 / UNIT_TIME^2 * 1/(eV * UNIT_ENERGY/eV * UNIT_LENGTH)
            // = unitless, not weighted
            float_X const K
                = (pmacc::math::isApproxZero(
                      superCellConstantInput.temperatureTimesk_Boltzman * superCellConstantInput.debyeLength))
                      ? 0._X
                      : constFactor * (chargeState + 1)
                            / (superCellConstantInput.temperatureTimesk_Boltzman * eV
                               * superCellConstantInput.debyeLength);

            // eV, not weighted
            return superCellConstantInput.temperatureTimesk_Boltzman
                   * (math::pow((3 * (superCellConstantInput.zStar + 1) * K + 1), 2._X / 3._X) - 1._X)
                   / (2._X * (superCellConstantInput.zStar + 1._X));
        }

        /** call a kernel, appending the IPD-input to the kernel's input
         *
         * @tparam T_Kernel kernel to call
         * @tparam T_chunkSize chunk size to use in call
         * @param dc picongpu data connector
         * @param kernelInput input to pass to kernel in addition to IPD-input
         */
        template<typename T_Kernel, uint32_t T_chunkSize, typename... T_KernelInput>
        HINLINE static void callKernelWithIPDInput(
            pmacc::DataConnector& dc,
            pmacc::AreaMapping<CORE + BORDER, picongpu::MappingDesc>& mapper,
            T_KernelInput... kernelInput)
        {
            auto& debyeLengthField
                = *dc.get<s_IPD::localHelperFields::DebyeLengthField<picongpu::MappingDesc>>("DebyeLengthField");
            auto& temperatureEnergyField
                = *dc.get<s_IPD::localHelperFields::TemperatureEnergyField<picongpu::MappingDesc>>(
                    "TemperatureEnergyField");
            auto& zStarField = *dc.get<s_IPD::localHelperFields::ZStarField<picongpu::MappingDesc>>("ZStarField");

            PMACC_LOCKSTEP_KERNEL(T_Kernel())
                .template config<T_chunkSize>(mapper.getGridDim())(
                    mapper,
                    kernelInput...,
                    debyeLengthField.getDeviceDataBox(),
                    temperatureEnergyField.getDeviceDataBox(),
                    zStarField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
