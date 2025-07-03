#pragma once

#include "picongpu/plugins/binning/utility.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>

#include <string>
#include <string_view>

namespace picongpu::plugins::binning
{
    /** @brief Struct to hold information about a field
     *  @tparam Field The type of the field
     *  The constructor takes a id which describes how to get the field from the DataConnector
     */
    template<typename Field>
    struct FieldInfo
    {
        using FieldType = Field;
        std::string id;

        FieldInfo(std::string_view id) : id(id)
        {
        }

        std::string getId() const
        {
            return id;
        }
    };

    auto transformFieldInfo(auto&& arg) -> decltype(auto)
    {
        if constexpr(IsSpecializationOf<std::decay_t<decltype(arg)>, FieldInfo>)
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            return dc
                .get<typename std::remove_cvref_t<decltype(arg)>::FieldType>(std::forward<decltype(arg)>(arg).getId())
                ->getDeviceDataBox();
        }
        else
        {
            return std::forward<decltype(arg)>(arg);
        }
    };

} // namespace picongpu::plugins::binning
