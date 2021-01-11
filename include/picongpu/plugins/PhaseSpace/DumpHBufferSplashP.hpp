/* Copyright 2013-2020 Axel Huebl, Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/traits/SplashToPIC.hpp"
#include "picongpu/traits/PICToSplash.hpp"

#include "picongpu/plugins/PhaseSpace/AxisDescription.hpp"
#include <pmacc/communication/manager_common.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/cuSTL/container/HostBuffer.hpp>
#include <pmacc/math/vector/Int.hpp>
#include <pmacc/verify.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <utility>
#include <mpi.h>
#include <splash/splash.h>
#include <openPMD/openPMD.hpp>
#include <vector>

namespace picongpu
{
    class DumpHBuffer
    {
    private:
        using SuperCellSize = typename MappingDesc::SuperCellSize;

    public:
        /** Dump the PhaseSpace host Buffer
         *
         * \tparam Type the HBuffers element type
         * \tparam int the HBuffers dimension
         * \param hBuffer const reference to the hBuffer, including guard cells in spatial dimension
         * \param axis_element plot to create: e.g. py, x from momentum/spatial-coordinate
         * \param unit sim unit of the buffer
         * \param strSpecies unique short hand name of the species
         * \param filenameSuffix infix + extension part of openPMD filename
         * \param currentStep current time step
         * \param mpiComm communicator of the participating ranks
         */
        template<typename T_Type, int T_bufDim>
        void operator()(
            const pmacc::container::HostBuffer<T_Type, T_bufDim>& hBuffer,
            const AxisDescription axis_element,
            const std::pair<float_X, float_X> axis_p_range,
            const float_64 pRange_unit,
            const float_64 unit,
            const std::string strSpecies,
            const std::string filenameExtension,
            const std::string jsonConfig,
            const uint32_t currentStep,
            MPI_Comm mpiComm) const
        {
            using Type = T_Type;
            const int bufDim = T_bufDim;

            /** file name *****************************************************
             *    phaseSpace/PhaseSpace_xpy_timestep.h5                       */
            std::string fCoords("xyz");
            std::ostringstream openPMDFilename;
            openPMDFilename << "phaseSpace/PhaseSpace_" << strSpecies << "_" << fCoords.at(axis_element.space) << "p"
                            << fCoords.at(axis_element.momentum) << "_%T." << filenameExtension;
            std::ostringstream filename;
            filename << "phaseSpace/OldPhaseSpace_" << strSpecies << "_" << fCoords.at(axis_element.space) << "p"
                     << fCoords.at(axis_element.momentum);


            /** get size of the fileWriter communicator ***********************/
            int size;
            MPI_CHECK(MPI_Comm_size(mpiComm, &size));

            /** create parallel domain collector ******************************/
            ::openPMD::Series series(openPMDFilename.str(), ::openPMD::Access::CREATE, jsonConfig);
            ::openPMD::Iteration iteration = series.iterations[currentStep];
            ParallelDomainCollector pdc(mpiComm, MPI_INFO_NULL, Dimensions(size, 1, 1), 10);

            const std::string software("PIConGPU");

            std::stringstream softwareVersion;
            softwareVersion << PICONGPU_VERSION_MAJOR << "." << PICONGPU_VERSION_MINOR << "."
                            << PICONGPU_VERSION_PATCH;
            if(!std::string(PICONGPU_VERSION_LABEL).empty())
                softwareVersion << "-" << PICONGPU_VERSION_LABEL;
            series.setSoftware(software, softwareVersion.str());

            pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();
            DataCollector::FileCreationAttr fAttr;
            Dimensions mpiPosition(gc.getPosition()[axis_element.space], 0, 0);
            fAttr.mpiPosition.set(mpiPosition);

            DataCollector::initFileCreationAttr(fAttr);

            pdc.open(filename.str().c_str(), fAttr);

            /** calculate GUARD offset in the source hBuffer *****************/
            const uint32_t rGuardCells
                = SuperCellSize().toRT()[axis_element.space] * GuardSize::toRT()[axis_element.space];

            /** calculate local and global size of the phase space ***********/
            const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            const std::uint64_t rLocalOffset = subGrid.getLocalDomain().offset[axis_element.space];
            const std::uint64_t rLocalSize = int(hBuffer.size().y() - 2 * rGuardCells);
            const std::uint64_t rGlobalSize = subGrid.getGlobalDomain().size[axis_element.space];
            PMACC_VERIFY(int(rLocalSize) == subGrid.getLocalDomain().size[axis_element.space]);

            /* globalDomain of the phase space */
            splash::Dimensions globalPhaseSpace_size(hBuffer.size().x(), rGlobalSize, 1);
            ::openPMD::Extent globalPhaseSpace_extent{rGlobalSize, hBuffer.size().x()};

            /* global moving window meta information */
            splash::Dimensions globalPhaseSpace_offset(0, 0, 0);
            ::openPMD::Offset _globalPhaseSpace_offset{0, 0};
            std::uint64_t globalMovingWindowOffset = 0;
            std::uint64_t globalMovingWindowSize = rGlobalSize;
            if(axis_element.space == AxisDescription::y) /* spatial axis == y */
            {
                globalPhaseSpace_offset.set(0, numSlides * rLocalSize, 0);
                _globalPhaseSpace_offset[0] = numSlides * rLocalSize;
                Window window = MovingWindow::getInstance().getWindow(currentStep);
                globalMovingWindowOffset = window.globalDimensions.offset[axis_element.space];
                globalMovingWindowSize = window.globalDimensions.size[axis_element.space];
            }

            /* localDomain: offset of it in the globalDomain and size */
            splash::Dimensions localPhaseSpace_offset(0, rLocalOffset, 0);
            splash::Dimensions localPhaseSpace_size(hBuffer.size().x(), rLocalSize, 1);
            ::openPMD::Offset _localPhaseSpace_offset{rLocalOffset, 0};
            ::openPMD::Extent _localPhaseSpace_extent{rLocalSize, hBuffer.size().x()};

            /** Dataset Name **************************************************/
            std::ostringstream dataSetName;
            /* xpx or ypz or ... */
            dataSetName << fCoords.at(axis_element.space) << "p" << fCoords.at(axis_element.momentum);
            std::ostringstream _dataSetName;
            _dataSetName << strSpecies << "_" << fCoords.at(axis_element.space) << "p"
                         << fCoords.at(axis_element.momentum);

            /** debug log *****************************************************/
            int rank;
            MPI_CHECK(MPI_Comm_rank(mpiComm, &rank));
            log<picLog::INPUT_OUTPUT>(
                "Dump buffer %1% to %2% at offset %3% with size %4% for total size %5% for rank %6% / %7%")
                % (*(hBuffer.origin()(0, rGuardCells))) % dataSetName.str() % localPhaseSpace_offset.toString()
                % localPhaseSpace_size.toString() % globalPhaseSpace_size.toString() % rank % size;

            /** write local domain ********************************************/
            typename PICToSplash<Type>::type ctPhaseSpace;

            // avoid deadlock between not finished pmacc tasks and mpi calls in HDF5
            __getTransactionEvent().waitForFinished();

            ::openPMD::Mesh mesh = iteration.meshes[_dataSetName.str()];
            ::openPMD::MeshRecordComponent dataset = mesh[::openPMD::RecordComponent::SCALAR];
            dataset.setPosition(std::vector<float_X>{0., 0.}); // @todo always correct?

            dataset.resetDataset({::openPMD::determineDatatype<Type>(), globalPhaseSpace_extent});
            std::shared_ptr<Type> data(&(*hBuffer.origin()(0, rGuardCells)), [](auto const&) {});
            dataset.storeChunk<Type>(data, _localPhaseSpace_offset, _localPhaseSpace_extent);

            pdc.writeDomain(
                currentStep,
                /* global domain and my local offset within it */
                globalPhaseSpace_size,
                localPhaseSpace_offset,
                /* */
                ctPhaseSpace,
                bufDim,
                /* local data set dimensions */
                splash::Selection(localPhaseSpace_size),
                /* data set name */
                dataSetName.str().c_str(),
                /* global domain */
                splash::Domain(globalPhaseSpace_offset, globalPhaseSpace_size),
                /* dataClass, buffer */
                DomainCollector::GridType,
                &(*hBuffer.origin()(0, rGuardCells)));

            /** meta attributes for the data set: unit, range, moving window **/
            typedef PICToSplash<float_X>::type SplashFloatXType;
            typedef PICToSplash<float_64>::type SplashFloat64Type;
            ColTypeInt ctInt;
            SplashFloat64Type ctFloat64;
            SplashFloatXType ctFloatX;

            pmacc::Selection<simDim> globalDomain = subGrid.getGlobalDomain();
            pmacc::Selection<simDim> totalDomain = subGrid.getTotalDomain();
            // convert things to std::vector<> for the openPMD API to enjoy
            std::vector<int> globalDomainSize{&globalDomain.size[0], &globalDomain.size[0] + simDim};
            std::vector<int> globalDomainOffset{&globalDomain.offset[0], &globalDomain.offset[0] + simDim};
            std::vector<int> totalDomainSize{&totalDomain.size[0], &totalDomain.size[0] + simDim};
            std::vector<int> totalDomainOffset{&totalDomain.offset[0], &totalDomain.offset[0] + simDim};
            std::vector<std::string> globalDomainAxisLabels;
            if(simDim == DIM2)
            {
                globalDomainAxisLabels = {"y", "x"}; // 2D: F[y][x]
            }
            if(simDim == DIM3)
            {
                globalDomainAxisLabels = {"z", "y", "x"}; // 3D: F[z][y][x]
            }

            float_X const dr = cellSize[axis_element.space];

            mesh.setAttribute("globalDomainSize", globalDomainSize);
            mesh.setAttribute("globalDomainOffset", globalDomainOffset);
            mesh.setAttribute("totalDomainSize", totalDomainSize);
            mesh.setAttribute("totalDomainOffset", totalDomainOffset);
            mesh.setAttribute("globalDomainAxisLabels", globalDomainAxisLabels);
            mesh.setAttribute("totalDomainAxisLabels", globalDomainAxisLabels);
            mesh.setAttribute("_global_start", _globalPhaseSpace_offset);
            mesh.setAttribute("_global_size", globalPhaseSpace_extent);
            mesh.setAxisLabels({axis_element.spaceAsString(), axis_element.momentumAsString()});
            mesh.setAttribute("sim_unit", unit);
            dataset.setUnitSI(unit);
            {
                using UD = ::openPMD::UnitDimension;
                mesh.setUnitDimension({{UD::I, 1.0}, {UD::T, 1.0}, {UD::L, -1.0}}); // charge density
            }
            pdc.writeAttribute(currentStep, ctFloat64, dataSetName.str().c_str(), "sim_unit", &unit);
            mesh.setAttribute("p_unit", pRange_unit);
            pdc.writeAttribute(currentStep, ctFloat64, dataSetName.str().c_str(), "p_unit", &pRange_unit);
            mesh.setAttribute("p_min", axis_p_range.first);
            pdc.writeAttribute(currentStep, ctFloatX, dataSetName.str().c_str(), "p_min", &(axis_p_range.first));
            mesh.setAttribute("p_max", axis_p_range.second);
            pdc.writeAttribute(currentStep, ctFloatX, dataSetName.str().c_str(), "p_max", &(axis_p_range.second));
            mesh.setGridGlobalOffset({globalMovingWindowOffset * dr, axis_p_range.first});
            mesh.setAttribute("movingWindowOffset", globalMovingWindowOffset);
            pdc.writeAttribute(
                currentStep,
                ctInt,
                dataSetName.str().c_str(),
                "movingWindowOffset",
                &globalMovingWindowOffset);
            mesh.setAttribute("movingWindowSize", globalMovingWindowSize);
            pdc.writeAttribute(
                currentStep,
                ctInt,
                dataSetName.str().c_str(),
                "movingWindowSize",
                &globalMovingWindowSize);
            mesh.setAttribute("dr", dr);
            pdc.writeAttribute(currentStep, ctFloatX, dataSetName.str().c_str(), "dr", &dr);
            mesh.setAttribute("dV", CELL_VOLUME);
            pdc.writeAttribute(currentStep, ctFloatX, dataSetName.str().c_str(), "dV", &CELL_VOLUME);
            mesh.setGridSpacing(std::vector<float_X>{dr, CELL_VOLUME / dr});
            mesh.setAttribute("dr_unit", UNIT_LENGTH);
            pdc.writeAttribute(currentStep, ctFloat64, dataSetName.str().c_str(), "dr_unit", &UNIT_LENGTH);
            iteration.setDt(DELTA_T);
            pdc.writeAttribute(currentStep, ctFloatX, dataSetName.str().c_str(), "dt", &DELTA_T);
            iteration.setTimeUnitSI(UNIT_TIME);
            pdc.writeAttribute(currentStep, ctFloat64, dataSetName.str().c_str(), "dt_unit", &UNIT_TIME);
            /*
             * The value represents an aggregation over one cell, so any value is correct for the mesh position.
             * Just use the center.
             */
            dataset.setPosition(std::vector<float>{0.5, 0.5});

            /** close file ****************************************************/
            pdc.finalize();
            pdc.close();
            iteration.close();
        }
    };

} /* namespace picongpu */
