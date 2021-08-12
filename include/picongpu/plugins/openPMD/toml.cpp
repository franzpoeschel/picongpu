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

#if ENABLE_OPENPMD == 1

#    include "picongpu/plugins/openPMD/toml.hpp"

#    include <toml.hpp>


namespace picongpu
{
    namespace toml
    {
        DataSources::DataSources(std::string tomlFiles) : m_period{{100, {std::string("fields_all"), "species_all"}}}
        {
            // todo: read from toml files
            // verify that a step will always be available
            m_upcomingSteps[0].reserve(m_period.size());
            for(auto pair : m_period)
            {
                m_upcomingSteps[0].push_back(pair.first);
            }
        }

        std::vector<std::string> DataSources::currentDataSources() const
        {
            std::set<std::string> result;
            for(SimulationStep_t period : m_upcomingSteps.begin()->second)
            {
                for(std::string const& source : m_period.at(period))
                {
                    result.insert(source);
                }
            }
            return {result.begin(), result.end()};
        }

        auto DataSources::currentStep() const -> SimulationStep_t
        {
            return m_upcomingSteps.begin()->first;
        }

        DataSources& DataSources::operator++()
        {
            SimulationStep_t current = currentStep();
            for(auto period : m_upcomingSteps.begin()->second)
            {
                m_upcomingSteps[current + period].push_back(period);
            }
            m_upcomingSteps.erase(m_upcomingSteps.begin());
            return *this;
        }
    } // namespace toml
} // namespace picongpu

#endif // ENABLE_OPENPMD
