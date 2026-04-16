/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_XMASKED_VALUE_META_HPP
#define XTL_XMASKED_VALUE_META_HPP

#include <type_traits>

namespace xtl
{
    template <class T, class B = bool>
    class xmasked_value;

    namespace detail
    {
        template <class E>
        struct is_xmasked_value_impl : std::false_type
        {
        };

        template <class T, class B>
        struct is_xmasked_value_impl<xmasked_value<T, B>> : std::true_type
        {
        };
    }

    template <class E>
    using is_xmasked_value = detail::is_xmasked_value_impl<E>;

    template <class E, class R>
    using disable_xmasked_value = std::enable_if_t<!is_xmasked_value<E>::value, R>;
}

#endif
