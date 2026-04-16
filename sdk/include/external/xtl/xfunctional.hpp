/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_FUNCTIONAL_HPP
#define XTL_FUNCTIONAL_HPP

#include <utility>

#include "xtl_config.hpp"
#include "xtype_traits.hpp"

namespace xtl
{
    /***************************
     * identity implementation *
     ***************************/

    struct identity
    {
        template <class T>
        T&& operator()(T&& x) const
        {
            return std::forward<T>(x);
        }
    };

    /*************************
     * select implementation *
     *************************/

    template <class B, class T1, class T2, XTL_REQUIRES(all_scalar<B, T1, T2>)>
    inline std::common_type_t<T1, T2> select(const B& cond, const T1& v1, const T2& v2) noexcept
    {
        return cond ? v1 : v2;
    }
}

#endif
