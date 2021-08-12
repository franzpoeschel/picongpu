/* Copyright 2021 Franz Poeschel
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

#include <map>
#include <set>
#include <vector>

namespace picongpu
{
    namespace toml
    {
        class DataSources
        {
            using Period_t = std::map<uint32_t, std::set<std::string>>;
            Period_t m_period;

            /*
             * For each period p, let s be the upcoming step divisible by p.
             * Then m_nextActiveAt[s] contains p.
             */
            std::map<uint32_t, std::vector<uint32_t>> m_nextActiveAt;

        public:
            DataSources(std::string tomlFiles)
            {
                // todo: read from toml files
            }

            std::vector<std::string> currentDataSources() const
            {
                // todo
                return {std::string("fields_all"), "species_all"};
            }

            uint32_t currentStep() const
            {
                return m_period.begin()->first;
            }

            DataSources& operator++()
            {
                // todo
                return *this;
            }
        };
    } // namespace toml
} // namespace picongpu
