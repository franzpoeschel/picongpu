/* Copyright 2023-2024 Tapish Narwal
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

#pragma once

#include "picongpu/plugins/binning/DomainInfo.hpp"
#include "picongpu/plugins/binning/utility.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>

#include <cstdint>

namespace picongpu
{
    namespace plugins::binning
    {
        struct FunctorParticle
        {
            using result_type = void;

            DINLINE FunctorParticle() = default;

            template<typename T_HistBox, typename T_DepositionFunctor, uint32_t T_nAxes>
            DINLINE void operator()(
                auto const& worker,
                T_HistBox histBox,
                T_DepositionFunctor const& quantityFunctor,
                auto const& axes,
                DomainInfo<BinningType::Particle> const& domainInfo,
                auto const& userFunctorData,
                DataSpace<T_nAxes> const& extents,
                auto const& particle) const
            {
                using DepositionType = typename T_HistBox::ValueType;

                auto binsDataspace = pmacc::DataSpace<T_nAxes>{};
                bool validIdx = true;
                binning::applyEnumerate(
                    [&](auto const&... tupleArgs)
                    {
                        // This assumes n_bins and getBinIdx exist
                        validIdx
                            = (
                                   [&](auto const& tupleArg)
                                   {
                                        auto [isValid, binIdx] = binning::apply(
                                            [&](auto const&... extraUserData)
                                            { return pmacc::memory::tuple::get<1>(tupleArg).getBinIdx(worker, domainInfo, particle, extraUserData...); },
                                            userFunctorData);

                                        binsDataspace[pmacc::memory::tuple::get<0>(tupleArg)] = binIdx;
                                        return isValid;
                                   }(tupleArgs)
                               && ...);
                    },
                    axes);

                if(validIdx)
                {
                    auto const idxOneD = pmacc::math::linearize(extents, binsDataspace);
                    DepositionType depositVal = binning::apply(
                        [&](auto const&... extraUserData)
                        { return quantityFunctor(worker, domainInfo, particle, extraUserData...); },
                        userFunctorData);
                    alpaka::atomicAdd(worker.getAcc(), &(histBox[idxOneD]), depositVal, ::alpaka::hierarchy::Blocks{});
                }
            }
        };

        /** Creates a histogram based on axis and quantity description
         */
        struct BinningFunctor
        {
            using result_type = void;

            HINLINE BinningFunctor() = default;

            template<typename T_HistBox, typename T_DepositionFunctor, typename T_Mapping, uint32_t T_nAxes>
            DINLINE void operator()(
                auto const& worker,
                T_HistBox binningBox,
                auto particlesBox,
                pmacc::DataSpace<simDim> const& localOffset,
                pmacc::DataSpace<simDim> const& globalOffset,
                auto const& axisTuple,
                T_DepositionFunctor const& quantityFunctor,
                DataSpace<T_nAxes> const& extents,
                auto const& userFunctorData,
                auto const& filter,
                uint32_t const currentStep,
                T_Mapping const& mapper) const
            {
                DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));
                // supercell index relative to the border origin
                auto const physicalSuperCellIdx = superCellIdx - mapper.getGuardingSuperCells();
                /**
                 * Init the Domain info, here because of the possibility of a moving window
                 */
                auto const domainInfo
                    = DomainInfo<BinningType::Particle>{currentStep, globalOffset, localOffset, physicalSuperCellIdx};

                auto const functorParticle = FunctorParticle{};

                auto forEachParticle
                    = pmacc::particles::algorithm::acc::makeForEach(worker, particlesBox, superCellIdx);

                // stop kernel if we have no particles
                if(!forEachParticle.hasParticles())
                    return;

                forEachParticle(
                    [&](auto const& lockstepWorker, auto& particle)
                    {
                        if(filter(worker, domainInfo, particle))
                        {
                            functorParticle(
                                lockstepWorker,
                                binningBox,
                                quantityFunctor,
                                axisTuple,
                                domainInfo,
                                userFunctorData,
                                extents,
                                particle);
                        }
                    });
            }
        };

        namespace detail
        {
            template<typename T1, typename T2, typename T3, std::size_t... Is>
            DINLINE bool insideBoundsImpl(
                T1 const& localCell,
                T2 const& beginCellIdxLocal,
                T3 const& endCellIdxLocal,
                std::index_sequence<Is...>)
            {
                return ((localCell[Is] >= beginCellIdxLocal[Is] && localCell[Is] < endCellIdxLocal[Is]) && ...);
            }

            template<typename T1, typename T2, typename T3>
            DINLINE bool insideBounds(T1 const& localCell, T2 const& beginCellIdxLocal, T3 const& endCellIdxLocal)
            {
                return insideBoundsImpl(
                    localCell,
                    beginCellIdxLocal,
                    endCellIdxLocal,
                    std::make_index_sequence<simDim>{});
            }
        } // namespace detail

        struct BinningFunctorLeaving
        {
            using result_type = void;

            HINLINE BinningFunctorLeaving() = default;

            template<
                typename T_Worker,
                typename TParticlesBox,
                typename T_HistBox,
                typename T_DepositionFunctor,
                typename T_AxisTuple,
                typename T_Mapping,
                uint32_t T_nAxes>
            DINLINE void operator()(
                T_Worker const& worker,
                T_HistBox binningBox,
                TParticlesBox particlesBox,
                pmacc::DataSpace<simDim> const& localOffset,
                pmacc::DataSpace<simDim> const& globalOffset,
                T_AxisTuple const& axisTuple,
                T_DepositionFunctor const& quantityFunctor,
                DataSpace<T_nAxes> const& extents,
                auto const& userFunctorData,
                auto const& filter,
                uint32_t const currentStep,
                pmacc::DataSpace<simDim> const& beginCellIdxLocal,
                pmacc::DataSpace<simDim> const& endCellIdxLocal,
                T_Mapping const& mapper) const
            {
                /* multi-dimensional offset vector from local domain origin on GPU in units of super cells */
                pmacc::DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));
                // supercell index relative to the border origin
                auto const physicalSuperCellIdx = superCellIdx - mapper.getGuardingSuperCells();

                auto const domainInfo
                    = DomainInfo<BinningType::Particle>{currentStep, globalOffset, localOffset, physicalSuperCellIdx};
                auto const functorParticle = FunctorParticle{};

                auto forEachParticle
                    = pmacc::particles::algorithm::acc::makeForEach(worker, particlesBox, superCellIdx);

                // stop kernel if we have no particles
                if(!forEachParticle.hasParticles())
                    return;

                forEachParticle(
                    [&](auto const& lockstepWorker, auto& particle)
                    {
                        if(filter(worker, domainInfo, particle))
                        {
                            // Check if it fits the internal cells range
                            auto const cellInSuperCell = pmacc::math::mapToND(
                                SuperCellSize::toRT(),
                                static_cast<int>(particle[localCellIdx_]));
                            auto const localCell = domainInfo.blockCellOffset + cellInSuperCell;

                            if(detail::insideBounds(localCell, beginCellIdxLocal, endCellIdxLocal))
                            {
                                functorParticle(
                                    lockstepWorker,
                                    binningBox,
                                    quantityFunctor,
                                    axisTuple,
                                    domainInfo,
                                    userFunctorData,
                                    extents,
                                    particle);
                            }
                        }
                    });
            }
        };


    } // namespace plugins::binning
} // namespace picongpu
