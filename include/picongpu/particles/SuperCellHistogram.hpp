/* Copyright 2026 Franz Poeschel
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

#include "picongpu/defines.hpp"

#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>

namespace picongpu
{
    /** Count the particles per supercell that pass the given filters.
     *
     * Writes the result to @p particlesPerSuperCell, indexed like the mapping's supercells.
     */
    struct KernelSuperCellHistogram
    {
        template<
            typename T_Mapping,
            typename T_Worker,
            typename T_ParBox,
            typename T_Filter,
            typename T_PositionFilter>
        DINLINE void operator()(
            T_Worker const& worker,
            T_ParBox pb,
            uint32_t* particlesPerSuperCell,
            T_Filter filter,
            T_PositionFilter posFilter,
            T_Mapping mapper) const
        {
            DataSpace<simDim> const superCellIdxND(mapper.getSuperCellIndex(worker.blockDomIdxND()));

            PMACC_SMEM(worker, numParticlesInBlock, uint32_t);

            auto onlyMaster = lockstep::makeMaster(worker);
            onlyMaster([&]() { numParticlesInBlock = 0u; });

            worker.sync();

            auto forEachParticle = pmacc::particles::algorithm::acc::makeForEach(worker, pb, superCellIdxND);
            if(forEachParticle.hasParticles())
            {
                uint32_t parCount = 0u;
                auto accFilter = filter(worker, superCellIdxND - mapper.getGuardingSuperCells());
                posFilter.setSuperCellPosition(
                    (superCellIdxND - mapper.getGuardingSuperCells()) * mapper.getSuperCellSize());
                forEachParticle(
                    [&accFilter, &posFilter, &worker, &parCount](auto const& lockstepWorker, auto& particle)
                    {
                        if(accFilter(lockstepWorker, particle) && posFilter(particle))
                        {
                            parCount++;
                        }
                    });

                worker.sync();
                alpaka::atomicAdd(worker.getAcc(), &numParticlesInBlock, parCount, alpaka::hierarchy::Threads{});
            }

            worker.sync();

            auto superCellIdx = mapper.superCellIdx(worker.blockDomIdxND());
            onlyMaster([&]() { particlesPerSuperCell[superCellIdx] = numParticlesInBlock; });
        }
    };
} // namespace picongpu
