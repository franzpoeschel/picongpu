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

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <mpi.h>

namespace picongpu
{
    namespace toml
    {
        class DataSources
        {
        public:
            using SimulationStep_t = uint32_t;
            using Period_t = std::map<SimulationStep_t, std::set<std::string>>;

        private:
            Period_t m_period;

            /*
             * For each period p, let s be the upcoming step divisible by p.
             * Then m_nextActiveAt[s] contains p.
             */
            std::map<SimulationStep_t, std::vector<SimulationStep_t>> m_upcomingSteps;

        public:
            DataSources(std::string const& tomlFiles, MPI_Comm);

            std::vector<std::string> currentDataSources() const;

            SimulationStep_t currentStep() const;

            DataSources& operator++();
        };

        // Definition of this needs to go in NVCC-compiled files
        // (openPMDWriter.hpp) due to include structure of PIConGPU
        void writeLog(char const* message, size_t argsn = 0, char const** argsv = nullptr);
    } // namespace toml
} // namespace picongpu
