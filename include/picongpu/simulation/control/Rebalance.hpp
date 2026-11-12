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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace picongpu
{
    namespace rebalance
    {
        /** Validate that a 1D split is possible and return a fallback error message helper. */
        inline void verifySplit(size_t extent, uint32_t blocks, uint32_t minBlock)
        {
            if(blocks == 0u)
                throw std::runtime_error("Rebalance: number of blocks must be positive.");
            if(extent < static_cast<size_t>(blocks) * static_cast<size_t>(minBlock))
                throw std::runtime_error(
                    "Rebalance: domain is too small to be split into the requested number of blocks "
                    "while respecting the minimum block size.");
        }

        /** Split an extent into equally sized blocks. */
        inline std::vector<size_t> equalCuts(size_t extent, uint32_t blocks, uint32_t minBlock)
        {
            verifySplit(extent, blocks, minBlock);
            std::vector<size_t> cuts(static_cast<size_t>(blocks) + 1u);
            cuts[0] = 0u;
            for(uint32_t block = 1u; block <= blocks; ++block)
                cuts[block] = static_cast<size_t>(block) * extent / blocks;
            cuts[blocks] = extent;
            return cuts;
        }

        /** Recursively split a 1D weight profile into blocks of roughly equal weight.
         *
         * Returns block+1 cut positions in units of the profile index.
         */
        inline std::vector<size_t> recursiveWeightedCuts(
            std::vector<uint64_t> const& profile,
            uint32_t blocks,
            uint32_t minBlock)
        {
            size_t const extent = profile.size();
            verifySplit(extent, blocks, minBlock);

            std::vector<uint64_t> prefix(extent + 1u, 0u);
            for(size_t i = 0u; i < extent; ++i)
                prefix[i + 1u] = prefix[i] + profile[i];

            std::vector<size_t> cuts(static_cast<size_t>(blocks) + 1u, 0u);
            cuts[blocks] = extent;

            std::function<void(size_t, size_t, uint32_t, uint32_t)> split
                = [&](size_t lo, size_t hi, uint32_t blockOffset, uint32_t numBlocks)
            {
                if(numBlocks <= 1u)
                    return;

                uint32_t const leftBlocks = numBlocks / 2u;
                uint32_t const rightBlocks = numBlocks - leftBlocks;

                size_t const leftMin = lo + static_cast<size_t>(leftBlocks) * minBlock;
                size_t const leftMax = hi - static_cast<size_t>(rightBlocks) * minBlock;
                if(leftMin > leftMax)
                    throw std::runtime_error(
                        "Rebalance: cannot split domain while respecting the minimum block size.");

                uint64_t const totalWeight = prefix[hi] - prefix[lo];
                uint64_t const target = prefix[lo] + totalWeight / 2u;

                auto begin = prefix.begin() + static_cast<std::ptrdiff_t>(leftMin);
                auto end = prefix.begin() + static_cast<std::ptrdiff_t>(leftMax + 1u);
                auto it = std::lower_bound(begin, end, target);

                size_t cut = leftMax;
                if(it != end)
                    cut = static_cast<size_t>(std::distance(prefix.begin(), it));
                cut = std::max(cut, leftMin);
                cut = std::min(cut, leftMax);

                // Pick the closer of the two neighbours around the target.
                if(cut > leftMin)
                {
                    uint64_t const currentDist = prefix[cut] > target ? prefix[cut] - target : target - prefix[cut];
                    uint64_t const previousDist
                        = prefix[cut - 1u] > target ? prefix[cut - 1u] - target : target - prefix[cut - 1u];
                    if(previousDist < currentDist)
                        --cut;
                }

                cuts[blockOffset + leftBlocks] = cut;
                split(lo, cut, blockOffset, leftBlocks);
                split(cut, hi, blockOffset + leftBlocks, rightBlocks);
            };

            split(0u, extent, 0u, blocks);
            return cuts;
        }
    } // namespace rebalance
} // namespace picongpu
