/* Copyright 2021 Franz Poeschel and Fabian Koller
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

#    include "picongpu/plugins/misc/removeSpaces.hpp"
#    include "picongpu/plugins/misc/splitString.hpp"

#    include <chrono>
#    include <thread> // std::this_thread::sleep_for
#    include <utility> // std::forward

#    ifdef _WIN32
#        include <windows.h>
#    else
#        include <sys/stat.h>
#    endif

#    include <toml.hpp>

namespace
{
    bool file_exists(std::string const& path)
    {
#    ifdef _WIN32
        DWORD attributes = GetFileAttributes(path.c_str());
        return (attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY));
#    else
        struct stat s;
        return (0 == stat(path.c_str(), &s)) && S_ISREG(s.st_mode);
#    endif
    }

    using Period_t = picongpu::toml::DataSources::Period_t;
    using SimulationStep_t = picongpu::toml::DataSources::SimulationStep_t;

    void mergePeriodTable(Period_t& into, Period_t const& from)
    {
        for(auto& pair : from)
        {
            for(std::string const& dataSource : pair.second)
            {
                // C++ does not have destructive iterators over std::set,
                // so a copy will have to do here
                into[pair.first].insert(dataSource);
            }
        }
    }

    Period_t parseSingleTomlFile(std::string const& path)
    {
        Period_t res;
        using toml_t = toml::basic_value<toml::discard_comments>;
        auto data = toml::parse(path);
        if(not data.contains("period"))
        {
            return {};
        }
        auto& periodTable = [&data]() -> decltype(data)::table_type& {
            try
            {
                return toml::find(data, "period").as_table();
            }
            catch(toml::type_error const& e)
            {
                throw std::runtime_error(
                    "[openPMD plugin] Key 'period' in TOML file must be a table (" + std::string(e.what()) + ")");
            }
        }();
        for(auto& pair : periodTable)
        {
            SimulationStep_t period{};
            try
            {
                period = std::stoul(pair.first);
            }
            catch(std::invalid_argument const&)
            {
                throw std::runtime_error(
                    "[openPMD plugin] TOML file keys must be unsigned integers, got: '" + pair.first + "'.");
            }
            auto& dataSources = res[period];
            using maybe_array_t = toml::result<decltype(data)::array_type*, toml::type_error>;
            auto dataSourcesInToml = [&pair]() -> maybe_array_t {
                try
                {
                    return toml::ok(&pair.second.as_array());
                }
                catch(toml::type_error const& e)
                {
                    return toml::err(e);
                }
            }();
            if(dataSourcesInToml.is_ok())
            {
                // 1. option: dataSources is an array:
                for(auto& value : *dataSourcesInToml.as_ok())
                {
                    auto dataSource
                        = toml::expect<std::string>(value)
                              .or_else([](auto const&) -> toml::success<std::string> {
                                  throw std::runtime_error("[openPMD plugin] Data sources in TOML "
                                                           "file must be a string or a vector of strings.");
                              })
                              .value;
                    dataSources.insert(std::move(dataSource));
                }
            }
            else
            {
                // 2. option: dataSources is no array, check if it is a simple string
                auto dataSource = toml::expect<std::string>(pair.second)
                                      .or_else([](auto const&) -> toml::success<std::string> {
                                          throw std::runtime_error("[openPMD plugin] Data sources in TOML "
                                                                   "file must be a string or a vector of strings.");
                                      })
                                      .value;
                dataSources.insert(std::move(dataSource));
            }
        }
        return res;
    }

    template<typename ChronoDuration>
    Period_t waitForParseAndMergeTomlFiles(std::vector<std::string> paths, ChronoDuration&& sleepInterval)
    {
        Period_t res;
        std::vector<decltype(paths)::iterator> toRemove;
        toRemove.reserve(paths.size());
        while(true)
        {
            for(auto it = paths.begin(); it != paths.end(); ++it)
            {
                auto const& path = *it;
                if(file_exists(path))
                {
                    mergePeriodTable(res, parseSingleTomlFile(path));
                }
                toRemove.push_back(it);
            }
            // std::vector<>::erase
            // "Invalidates iterators and references at or after the point of the erase, including the end() iterator."
            // -> erase iterators from back to begin
            for(auto it = toRemove.rbegin(); it != toRemove.rend(); ++it)
            {
                paths.erase(*it);
            }
            toRemove.clear();
            // Put the exit condition here, so we can skip the last sleep
            if(paths.empty())
            {
                break;
            }
            std::this_thread::sleep_for(sleepInterval);
        }
        return res;
    }

    template<typename ChronoDuration>
    Period_t waitForParseAndMergeTomlFiles(std::string const& paths, ChronoDuration&& sleepInterval)
    {
        using namespace picongpu::plugins::misc;
        return waitForParseAndMergeTomlFiles(
            splitString(removeSpaces(paths)),
            std::forward<ChronoDuration>(sleepInterval));
    }
} // namespace


namespace picongpu
{
    namespace toml
    {
        using namespace std::literals::chrono_literals;
        DataSources::DataSources(std::string const& tomlFiles) : m_period{waitForParseAndMergeTomlFiles(tomlFiles, 5s)}
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
