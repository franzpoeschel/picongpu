/* Copyright 2014-2019 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Franz Poeschel
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

#include "picongpu/plugins/openPMD/openPMDWriter.def"
#include "picongpu/simulation_defines.hpp"

#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/GetNComponents.hpp>
#include <pmacc/traits/Resolve.hpp>

namespace picongpu
{
namespace openPMD
{
    using namespace pmacc;

    /** write attribute of a particle to openPMD series
     *
     * @tparam T_Identifier identifier of a particle attribute
     */
    template< typename T_Identifier >
    struct ParticleAttribute
    {
        /** write attribute to openPMD series
         *
         * @param params wrapped params
         * @param elements elements of this attribute
         */
        template< typename FrameType >
        HINLINE void
        operator()( ThreadParams * params,
            FrameType & frame,
            const size_t elements )
        {
            typedef T_Identifier Identifier;
            typedef typename pmacc::traits::Resolve< Identifier >::type::type
                ValueType;
            const uint32_t components = GetNComponents< ValueType >::value;
            typedef typename GetComponentsType< ValueType >::type ComponentType;

            log< picLog::INPUT_OUTPUT >(
                "openPMD:  (begin) write species attribute: %1%" ) %
                Identifier::getName();

            std::shared_ptr< ComponentType > storeBfr{
                new ComponentType[ elements ],
                []( ComponentType * ptr ) { delete[] ptr; }
            };

            for( uint32_t d = 0; d < components; d++ )
            {
                ValueType * dataPtr = frame.getIdentifier( Identifier() )
                                          .getPointer(); // can be moved up?
                auto storePtr = storeBfr.get();

/* copy strided data from source to temporary buffer */
#pragma omp parallel for
                for( size_t i = 0; i < elements; ++i )
                {
                    // TODO wtf?
                    storePtr[ i ] = reinterpret_cast< ComponentType * >(
                        dataPtr )[ d + i * components ];
                }

                auto & window = params->particleAttributes.front();
                window.m_data.storeChunk(
                    storeBfr, window.m_offset, window.m_extent );
                params->openPMDSeries->flush();
                params->particleAttributes.pop_front();
            }

            log< picLog::INPUT_OUTPUT >(
                "openPMD:  ( end ) write species attribute: %1%" ) %
                Identifier::getName();
        }
    };

} // namespace openPMD

} // namespace picongpu
