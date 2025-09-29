/* Copyright 2025 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/filesystem.hpp"
#include "pmacc/mappings/simulation/Filesystem.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/pluginSystem/Slice.hpp"
#include "pmacc/pluginSystem/containsStep.hpp"
#include "pmacc/pluginSystem/toSlice.hpp"
#include "pmacc/simulationControl/signal.hpp"
#include "pmacc/types.hpp"

#include <charconv>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pmacc::simulationControl
{
    // State of restart.
    // DISABLED means no restart is attempted.
    // TRY means a restart is attempted, but the simulation will continue even if restart fails.
    // FORCE means a restart is attempted, and the simulation will fail if restart fails.
    // SUCCESS means a restart was successful.
    // FAILED means a restart failed.
    enum class RestartState : uint8_t
    {
        DISABLED,
        TRY,
        FORCE,
        SUCCESS,
        FAILED
    };

    /**
     * @brief Stream insertion operator for RestartState, converting enum to string.
     * @param out The output stream.
     * @param state The RestartState value to write.
     * @return The output stream.
     */
    inline std::ostream& operator<<(std::ostream& out, RestartState const& state)
    {
        static std::unordered_map<RestartState, std::string_view> const stateToString
            = {{RestartState::DISABLED, "disabled"},
               {RestartState::TRY, "try"},
               {RestartState::FORCE, "force"},
               {RestartState::SUCCESS, "success"},
               {RestartState::FAILED, "failed"}};

        if(auto it = stateToString.find(state); it != stateToString.end())
        {
            out << it->second;
        }
        else
        {
            // Handle unknown enum values gracefully.
            out << "UNKNOWN_RESTART_STATE(" << static_cast<int>(state) << ")";
        }
        return out;
    }

    /**
     * @brief Stream extraction operator for RestartState, converting string to enum.
     * @param in The input stream.
     * @param state The RestartState variable to populate.
     * @return The input stream.
     */
    inline std::istream& operator>>(std::istream& in, RestartState& state)
    {
        static std::unordered_map<std::string, RestartState> const stringToState
            = {{"disabled", RestartState::DISABLED},
               {"try", RestartState::TRY},
               {"force", RestartState::FORCE},
               {"success", RestartState::SUCCESS},
               {"failed", RestartState::FAILED}};

        std::string token;
        in >> token;

        if(auto it = stringToState.find(token); it != stringToState.end())
        {
            state = it->second;
        }
        else
        {
            // If the token is not a valid state, set the stream's failbit.
            // boost::program_options will catch this and report an error to the user.
            in.setstate(std::ios_base::failbit);
        }
        return in;
    }

    /**
     * @brief Manages simulation checkpointing and restarting.
     *
     * This class provides functionalities to periodically save the simulation state
     * (checkpointing) and to resume a simulation from a saved state (restarting).
     * The checkpointing can be triggered based on simulation steps or wall-clock time.
     * The entire functionality can be enabled or disabled at compile time via the
     * `checkpointingEnabled` template parameter.
     *
     * @tparam checkpointingEnabled A boolean to enable/disable checkpointing features.
     */
    template<bool checkpointingEnabled>
    struct Checkpointing
    {
        using SeqOfTimeSlices = std::vector<pluginSystem::Slice>;

        void registerHelp(po::options_description& desc)
        {
            if constexpr(checkpointingEnabled)
            {
                // clang-format off
                desc.add_options()
                    ("checkpoint.restart.loop", po::value<uint32_t>(&softRestarts)->default_value(0),
                    "Number of times to restart the simulation after simulation has finished (for presentations). "
                    "Note: does not yet work with all plugins, see issue #1305")
                    ("checkpoint.restart", po::value<RestartState>(&restartState)->zero_tokens()->implicit_value(RestartState::FORCE),
                     "Restart simulation from a checkpoint. Requires a valid checkpoint.")
                    ("checkpoint.tryRestart", po::value<RestartState>(&restartState)->zero_tokens()->implicit_value(RestartState::TRY),
                     "Try to restart if a checkpoint is available else start the simulation from scratch.")
                    ("checkpoint.restart.directory", po::value<std::string>(&restartDirectory)->default_value(restartDirectory),
                     "Directory containing checkpoints for a restart")
                    ("checkpoint.restart.step", po::value<int32_t>(&restartStep),
                     "Checkpoint step to restart from")
                    ("checkpoint.period", po::value<std::string>(&checkpointPeriod),
                     "Period for checkpoint creation [interval(s) based on steps]")
                    ("checkpoint.timePeriod", po::value<std::uint64_t>(&checkpointPeriodMinutes),
                     "Time periodic checkpoint creation [period in minutes]")
                    ("checkpoint.directory", po::value<std::string>(&checkpointDirectory)->default_value(checkpointDirectory),
                     "Directory for checkpoints");
                // clang-format on
                // translate checkpointPeriod string into checkpoint intervals
                seqCheckpointPeriod = pluginSystem::toTimeSlice(checkpointPeriod);
            }
        }

        void addCheckpoint(uint32_t signalMaxTimestep)
        {
            if constexpr(checkpointingEnabled)
            {
                seqCheckpointPeriod.push_back(pluginSystem::Slice(signalMaxTimestep, signalMaxTimestep));
            }
            else
            {
                std::cout << "Checkpointing is disabled, no checkpoint will be created." << std::endl;
            }
        }

        template<unsigned DIM>
        void dump(uint32_t currentStep)
        {
            if constexpr(checkpointingEnabled)
            {
                /* trigger checkpoint notification */
                if(pluginSystem::containsStep(seqCheckpointPeriod, currentStep))
                {
                    /* first synchronize: if something failed, we can spare the time
                     * for the checkpoint writing */
                    alpaka::wait(manager::Device<ComputeDevice>::get().current());

                    // avoid deadlock between not finished PMacc tasks and MPI_Barrier
                    eventSystem::getTransactionEvent().waitForFinished();

                    GridController<DIM>& gc = Environment<DIM>::get().GridController();
                    /* can be spared for better scalings, but allows to spare the
                     * time for checkpointing if some ranks died */
                    MPI_CHECK(MPI_Barrier(gc.getCommunicator().getMPIComm()));

                    /* create directory containing checkpoints  */
                    if(numCheckpoints == 0 && gc.getGlobalRank() == 0)
                    {
                        pmacc::Filesystem::get().createDirectoryWithPermissions(checkpointDirectory);
                    }

                    Environment<DIM>::get().PluginConnector().checkpointPlugins(currentStep, checkpointDirectory);

                    /* important synchronize: only if no errors occured until this
                     * point guarantees that a checkpoint is usable */
                    alpaka::wait(manager::Device<ComputeDevice>::get().current());

                    /* avoid deadlock between not finished PMacc tasks and MPI_Barrier */
                    eventSystem::getTransactionEvent().waitForFinished();

                    /* \todo in an ideal world with MPI-3, this would be an
                     * MPI_Ibarrier call and this function would return a MPI_Request
                     * that could be checked */
                    MPI_CHECK(MPI_Barrier(gc.getCommunicator().getMPIComm()));

                    if(gc.getGlobalRank() == 0)
                    {
                        writeCheckpointStep(currentStep);
                    }
                    numCheckpoints++;
                }
            }
        }

        bool doSoftRestart()
        {
            static uint32_t nthSoftRestart = 0;
            if(nthSoftRestart <= softRestarts)
            {
                nthSoftRestart++;
                return true;
            }
            return false;
        }

        void doTimeBasedCheckpointing()
        {
            if constexpr(checkpointingEnabled)
            {
                // register concurrent thread to perform checkpointing periodically after a user defined time
                if(checkpointPeriodMinutes != 0)
                    checkpointTimeThread = std::thread(
                        [&, this]()
                        {
                            std::unique_lock<std::mutex> lk(this->concurrentThreadMutex);
                            while(exitConcurrentThreads.wait_until(
                                      lk,
                                      std::chrono::system_clock::now() + std::chrono::minutes(checkpointPeriodMinutes))
                                  == std::cv_status::timeout)
                            {
                                signal::detail::setCreateCheckpoint(1);
                            }
                        });
            }
        }

        void endTimeBasedCheckpointing()
        {
            if constexpr(checkpointingEnabled)
            {
                {
                    // notify all concurrent threads to exit
                    std::unique_lock<std::mutex> lk(this->concurrentThreadMutex);
                    exitConcurrentThreads.notify_all();
                }
                // wait for time triggered checkpoint thread
                if(checkpointTimeThread.joinable())
                    checkpointTimeThread.join();
            }
        }

        /* Returns whether a restart needs to be performed */
        bool checkRestart(uint32_t& step)
        {
            if constexpr(checkpointingEnabled)
            {
                if(restartState != RestartState::TRY && restartState != RestartState::FORCE)
                {
                    return false;
                }
                std::vector<uint32_t> checkpoints = readCheckpointMasterFile();

                // If no specific restart step is given, default to the latest available checkpoint.
                if(restartStep < 0)
                {
                    if(checkpoints.empty())
                    {
                        if(restartState == RestartState::FORCE)
                        {
                            throw std::runtime_error(
                                "Restart failed. No checkpoints found and no '--checkpoint.restart.step' provided.");
                        }
                        restartState = RestartState::FAILED;
                        return false;
                    }
                    restartStep = checkpoints.back();
                }

                // Checkpoints are expected to be sorted chronologically.
                bool const stepFound = std::binary_search(checkpoints.cbegin(), checkpoints.cend(), restartStep);

                if(!stepFound)
                {
                    if(restartState == RestartState::FORCE)
                    {
                        throw std::runtime_error(
                            "Restart failed. Checkpoint for step " + std::to_string(restartStep) + " not found.");
                    }
                    restartState = RestartState::FAILED;
                    return false;
                }

                // At this point, restart is possible.
                restartState = RestartState::SUCCESS;
                return true;
            }
            return false;
        }

        [[nodiscard]] RestartState getRestartState() const
        {
            return restartState;
        }

        [[nodiscard]] int32_t getRestartStep() const
        {
            return restartStep;
        }

        [[nodiscard]] std::string const& getRestartDir() const
        {
            return restartDirectory;
        }

    private:
        /** Presentations: loop the whole simulation `softRestarts` times from
         *                 initial step to runSteps */
        uint32_t softRestarts{0};

        /* period for checkpoint creation [interval(s) based on steps]*/
        std::string checkpointPeriod;

        /* checkpoint intervals */
        SeqOfTimeSlices seqCheckpointPeriod;

        /* period for checkpoint creation [period in minutes]
         * Zero is disabling time depended checkpointing.
         */
        std::uint64_t checkpointPeriodMinutes = 0u;
        std::thread checkpointTimeThread;

        // conditional variable to notify all concurrent threads and signal exit of the simulation
        std::condition_variable exitConcurrentThreads;
        std::mutex concurrentThreadMutex;

        /* common directory for checkpoints */
        std::string checkpointDirectory{"checkpoints"};

        /* number of checkpoints written */
        uint32_t numCheckpoints{0};

        /* checkpoint step to restart from */
        int32_t restartStep{-1};

        /* common directory for restarts */
        std::string restartDirectory{"checkpoints"};

        /* filename for checkpoint master file with all checkpoint timesteps */
        static constexpr std::string_view CHECKPOINT_MASTER_FILE{"checkpoints.txt"};

        RestartState restartState{RestartState::DISABLED};

        /**
         * Append \p checkpointStep to the master checkpoint file
         *
         * @param checkpointStep current checkpoint step
         */
        void writeCheckpointStep(uint32_t const checkpointStep)
        {
            stdfs::path const checkpointMasterFile = stdfs::path(checkpointDirectory) / CHECKPOINT_MASTER_FILE;

            std::ofstream file(checkpointMasterFile, std::ofstream::app);

            if(!file)
            {
                throw std::runtime_error("Failed to write checkpoint master file: " + checkpointMasterFile.string());
            }

            file << checkpointStep << '\n';
        }

        /**
         * Reads the checkpoint master file if any and returns all found checkpoint steps
         *
         * @return vector of found checkpoints steps in order they appear in the file
         */
        std::vector<uint32_t> readCheckpointMasterFile()
        {
            std::vector<uint32_t> checkpoints;

            stdfs::path const checkpointMasterFile = stdfs::path(restartDirectory) / CHECKPOINT_MASTER_FILE;

            if(!stdfs::exists(checkpointMasterFile))
            {
                return checkpoints;
            }

            std::ifstream file(checkpointMasterFile);
            if(!file)
            {
                std::cerr << "Warning: Could not open checkpoint master file: " << checkpointMasterFile << std::endl;
                return checkpoints;
            }

            /* read each line */
            std::string line;
            while(std::getline(file, line))
            {
                if(line.empty())
                {
                    continue;
                }

                uint32_t step;
                auto const [ptr, ec] = std::from_chars(line.data(), line.data() + line.size(), step);
                if(ec == std::errc{} && ptr == line.data() + line.size())
                {
                    checkpoints.push_back(step);
                }
                else
                {
                    std::cerr << "Warning: checkpoint master file contains invalid data (" << line << ")" << std::endl;
                }
            }

            return checkpoints;
        }
    };

} // namespace pmacc::simulationControl
